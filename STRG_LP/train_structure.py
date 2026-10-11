"""Streaming spatial Structure model and score generation."""

import argparse
import hashlib
import heapq
import json
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from numba import njit, prange, get_num_threads, set_num_threads

from utils import (
    RunMetrics,
    ScoreWriter,
    TGB_DATASETS,
    THG_DATASETS,
    add_metric_sums,
    add_spatial_args,
    apply_best_params,
    collect_eval_batch,
    compute_ranking_metric_sums,
    cuda_synchronize,
    default_ns_q,
    efficiency_metrics,
    finalize_metric_sums,
    fit_relation_space,
    inverse_aug,
    load_datasets,
    make_dir_name,
    make_locations,
    prediction_weights,
    print_efficiency,
    print_run_metrics,
    resolve_spatial_args,
    save_config,
    select_torch_device,
    set_compute_threads,
    set_random_seed,
    spatial_signature,
    spatial_weight,
)


DSH_GLOBAL_LAZY_SAFE_EXP = 650.0
DSH_BUCKET_CLEAR_EXP = 700.0
DSH_BUCKET_MATERIALIZE_SCALE = 1e-100
EPS = 1e-12
COMPONENT_SCORE_NAMES = ("dsh", "dmh", "shared", "direct", "structure_raw")


class LogicMatrixUpdater:
    """Logic Matrix Updater for M_trans"""
    def __init__(self, num_nodes, num_rels, window_size=10.0, decay_factor=0.1, device='cuda:0'):
        self.num_rels = num_rels
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.device = device

        self.M_trans_counts = torch.zeros((num_rels, num_rels), dtype=torch.float32, device=device)

        self.history_r = [[] for _ in range(num_nodes)]
        self.history_t = [[] for _ in range(num_nodes)]

    def update_M_trans_step(self, events, current_t):
        """
        Update M_trans counts based on new events-match-history
        events: (B, 3) numpy array
        """
        num_events = len(events)
        threshold = current_t - self.window_size

        for i in range(num_events):
            r = int(events[i, 1])
            o = int(events[i, 2])

            hist_r = self.history_r[o]
            hist_t = self.history_t[o]

            if len(hist_t) > 0 and hist_t[0] <= threshold:
                k = 0
                while k < len(hist_t) and hist_t[k] <= threshold:
                    k += 1

                del hist_t[:k]
                del hist_r[:k]

            n_hist = len(hist_r)
            if n_hist > 0:
                r_old_tensor = torch.tensor(hist_r, dtype=torch.long, device=self.device)
                t_prev_tensor = torch.tensor(hist_t, dtype=torch.float32, device=self.device)

                delta = current_t - t_prev_tensor
                weights = torch.exp(-self.decay_factor * delta)

                self.M_trans_counts[r].scatter_add_(0, r_old_tensor, weights)

    def update_history_step(self, events, current_t):
        """
        Add current events to node history
        events: (B, 3) numpy array
        """
        num_events = len(events)
        for i in range(num_events):
            s = int(events[i, 0])
            r = int(events[i, 1])

            self.history_r[s].append(r)
            self.history_t[s].append(current_t)

    def get_normalized_M_trans(self):
        """
        Return M_trans scaled to range [0.2, 1.0] per row.
        """
        target_min = 0.2
        target_max = 1.0

        M_trans_out = torch.empty_like(self.M_trans_counts)
        target_range = target_max - target_min

        r_min = self.M_trans_counts.min(dim=1, keepdim=True)[0]
        r_max = self.M_trans_counts.max(dim=1, keepdim=True)[0]

        diff = r_max - r_min
        mask_zero_diff = (diff < 1e-9)

        diff = torch.where(mask_zero_diff, torch.ones_like(diff), diff)

        norm_01 = (self.M_trans_counts - r_min) / diff

        M_trans_out = target_min + norm_01 * target_range

        M_trans_out = torch.where(mask_zero_diff.expand_as(M_trans_out),
                                   torch.full_like(M_trans_out, target_min),
                                   M_trans_out)

        return M_trans_out


class SemanticMatrixUpdater:
    """Semantic Matrix Updater for M_sim"""
    def __init__(self, num_nodes, num_rels, window_size, device='cuda:0'):
        self.num_rels = num_rels
        self.window_size = window_size
        self.device = device

        self.node_shift = 10 ** int(math.ceil(math.log10(num_nodes + 1)))

        self.M_sim_counts = torch.zeros((num_rels, num_rels), dtype=torch.float32, device=device)

        self.pair_history_t = {}
        self.pair_history_r = {}

    def _get_key(self, s, o):
        return s * self.node_shift + o

    def update_M_sim_step(self, events, current_t):
        num_events = len(events)
        threshold = current_t - self.window_size

        for i in range(num_events):
            s = int(events[i, 0])
            r_new = int(events[i, 1])
            o = int(events[i, 2])

            key = self._get_key(s, o)

            if key not in self.pair_history_t:
                self.pair_history_t[key] = []
                self.pair_history_r[key] = []

            hist_t = self.pair_history_t[key]
            hist_r = self.pair_history_r[key]

            if len(hist_t) > 0 and hist_t[0] <= threshold:
                k = 0
                while k < len(hist_t) and hist_t[k] <= threshold:
                    k += 1

                del hist_t[:k]
                del hist_r[:k]

            n_hist = len(hist_r)
            if n_hist > 0:
                r_old_tensor = torch.tensor(hist_r, dtype=torch.long, device=self.device)
                ones_tensor = torch.ones(n_hist, dtype=torch.float32, device=self.device)

                self.M_sim_counts[:, r_new].scatter_add_(0, r_old_tensor, ones_tensor)
                self.M_sim_counts[r_new, :].scatter_add_(0, r_old_tensor, ones_tensor)

            hist_t.append(current_t)
            hist_r.append(r_new)

    def get_probability_M_sim(self):
        """
        PPMI + Sigmoid -> Mapped to [0.5, 1.0]
        """
        target_min = 0.5
        target_max = 1.0

        epsilon = 1e-8
        alpha = 1.0

        total_count = self.M_sim_counts.sum() + (alpha * self.num_rels * self.num_rels)
        row_sums = self.M_sim_counts.sum(dim=1) + (alpha * self.num_rels)
        col_sums = self.M_sim_counts.sum(dim=0) + (alpha * self.num_rels)

        target_range = target_max - target_min

        c_ij = self.M_sim_counts + alpha

        p_ij = c_ij / total_count
        p_i = row_sums.unsqueeze(1) / total_count
        p_j = col_sums.unsqueeze(0) / total_count

        lift = p_ij / (p_i * p_j + epsilon)
        pmi = torch.log(lift + epsilon)

        sigmoid_val = torch.sigmoid(pmi)

        M_sim_out = target_min + sigmoid_val * target_range

        return M_sim_out


def _top_direct_masks(pos_direct, neg_direct, neg_samples, top_direct):
    top_direct = int(top_direct)
    batch_size = int(pos_direct.shape[0])
    width = int(neg_direct.shape[1])
    pos_mask = np.zeros((batch_size, 1), dtype=np.bool_)
    neg_mask = np.zeros((batch_size, width), dtype=np.bool_)
    if top_direct < 0:
        pos_mask[:, 0] = True
        neg_mask[:, :] = neg_samples >= 0
        return pos_mask, neg_mask
    if top_direct == 0:
        raise ValueError('top_direct must be -1 or a positive integer')
    if batch_size == 0:
        return pos_mask, neg_mask

    valid = np.concatenate(
        (
            np.ones((batch_size, 1), dtype=np.bool_),
            np.asarray(neg_samples >= 0, dtype=np.bool_),
        ),
        axis=1,
    )
    scores = np.concatenate((pos_direct, neg_direct), axis=1).astype(np.float32, copy=False)
    scores = np.where(valid, scores, -np.inf)
    total_width = int(scores.shape[1])
    k = min(top_direct, total_width)
    if k >= total_width:
        selected = valid
    else:
        cols = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        selected = np.zeros(scores.shape, dtype=np.bool_)
        rows = np.arange(batch_size)[:, None]
        selected[rows, cols] = True
        selected &= valid
    return selected[:, :1], selected[:, 1:]


@njit(cache=True)
def _score_dsh_candidates(keys, values, rows, pos_obj, neg_samples, pos_out, neg_out, inv_total):
    n_keys = keys.shape[0]
    for rr in range(rows.shape[0]):
        row = rows[rr]
        target = pos_obj[row]
        score = 0.0
        lo = 0
        hi = n_keys
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo < n_keys and keys[lo] == target:
            score = values[lo] * inv_total
        pos_out[row, 0] = score

        for j in range(neg_samples.shape[1]):
            target = neg_samples[row, j]
            if target < 0:
                neg_out[row, j] = 0.0
                continue
            score = 0.0
            lo = 0
            hi = n_keys
            while lo < hi:
                mid = (lo + hi) // 2
                if keys[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < n_keys and keys[lo] == target:
                score = values[lo] * inv_total
            neg_out[row, j] = score


class DirectSingleHopScorer:
    def __init__(self, num_rels, decay_direct=1.0, max_time_span=None, log_bucket_stats=False):
        self.num_rels = int(num_rels)
        self.decay_direct = float(decay_direct)
        self.log_bucket_stats = bool(log_bucket_stats)
        self.rel_shift = 10 ** int(math.ceil(math.log10(self.num_rels + 1)))
        self.V_sr = {}
        self.V_sr_sum = {}
        self.time_shift = 0.0
        self.bucket_last_ts = {}
        self.bucket_scale = {}
        self._sr_version = {}
        self._array_cache = {}
        self.max_time_span = None if max_time_span is None else float(max_time_span)
        max_exp = 0.0 if self.max_time_span is None else self.decay_direct * max(0.0, self.max_time_span)
        self.decay_mode = 'bucket_lazy' if max_exp > DSH_GLOBAL_LAZY_SAFE_EXP else 'global_lazy'
        print(
            f'[DSH] mode={self.decay_mode} decay_direct={self.decay_direct:g} '
            f'max_time_span={"unknown" if self.max_time_span is None else f"{self.max_time_span:g}"} '
            f'max_exp={"unknown" if self.max_time_span is None else f"{max_exp:g}"}',
            flush=True,
        )

    def _warn(self, where, message):
        print(f'[DSH-warning] {where}: {message}', flush=True)

    def _sr_key(self, s, r):
        return int(s) * self.rel_shift + int(r)

    def _clear_sr(self, sr, where='', reason='', warn=True):
        if warn and (where or reason):
            size = len(self.V_sr.get(sr, {}))
            total = self.V_sr_sum.get(sr, None)
            self._warn(
                where or 'clear_sr',
                f'clearing (source,relation) bucket sr={sr}; reason={reason or "invalid state"}; '
                f'entries={size}; total={total}',
            )
        self.V_sr.pop(sr, None)
        self.V_sr_sum.pop(sr, None)
        self.bucket_last_ts.pop(sr, None)
        self.bucket_scale.pop(sr, None)
        self._mark_dirty(sr)

    def _clear_all(self, where='', reason=''):
        if where or reason:
            self._warn(
                where or 'clear_all',
                f'clearing all DSH state; reason={reason or "invalid state"}; '
                f'buckets={len(self.V_sr)}',
            )
        self.V_sr.clear()
        self.V_sr_sum.clear()
        self.bucket_last_ts.clear()
        self.bucket_scale.clear()
        self._array_cache.clear()
        self._sr_version.clear()

    def _mark_dirty(self, sr):
        self._sr_version[sr] = self._sr_version.get(sr, 0) + 1
        self._array_cache.pop(sr, None)

    def _bucket_arrays(self, sr):
        cached = self._array_cache.get(sr)
        version = self._sr_version.get(sr, 0)
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        bucket = self.V_sr.get(sr)
        if not bucket:
            keys = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.float32)
        else:
            total = float(self.V_sr_sum.get(sr, 0.0))
            if total <= 0.0 or not math.isfinite(total):
                self._clear_sr(sr, where='_bucket_arrays', reason=f'non-finite or non-positive total={total}')
                keys = np.empty(0, dtype=np.int64)
                values = np.empty(0, dtype=np.float32)
            else:
                keys = np.fromiter(bucket.keys(), dtype=np.int64, count=len(bucket))
                raw_values = np.fromiter((float(v) for v in bucket.values()), dtype=np.float64, count=len(bucket))
                if np.any(raw_values < 0.0) or not np.all(np.isfinite(raw_values)):
                    self._clear_sr(sr, where='_bucket_arrays', reason='bucket contains negative or non-finite values')
                    keys = np.empty(0, dtype=np.int64)
                    values = np.empty(0, dtype=np.float32)
                else:
                    values = (raw_values / total).astype(np.float32, copy=False)
                    if not np.all(np.isfinite(values)):
                        self._clear_sr(sr, where='_bucket_arrays', reason=f'normalization produced non-finite values; total={total}')
                        keys = np.empty(0, dtype=np.int64)
                        values = np.empty(0, dtype=np.float32)
            order = np.argsort(keys, kind='stable')
            keys = np.ascontiguousarray(keys[order])
            values = np.ascontiguousarray(values[order])
        self._array_cache[sr] = (version, keys, values)
        return keys, values

    def _update_state_global_lazy(self, events_i64, ts):
        rel_exp = self.decay_direct * (float(ts) - self.time_shift)
        if not math.isfinite(rel_exp):
            self._clear_all(where='update_state', reason=f'non-finite decay exponent rel_exp={rel_exp}, ts={ts}, time_shift={self.time_shift}')
            self.time_shift = float(ts)
            weight = 1.0
        elif rel_exp > 700.0:
            scale = 2.0 ** (-rel_exp)
            if scale == 0.0:
                self._clear_all(where='update_state', reason=f'decay scale underflowed to zero; rel_exp={rel_exp}, ts={ts}')
            else:
                invalid = []
                for key in list(self.V_sr_sum.keys()):
                    old_sum = float(self.V_sr_sum.get(key, 0.0))
                    new_sum = old_sum * scale if math.isfinite(old_sum) else 0.0
                    if new_sum <= 0.0 or not math.isfinite(new_sum):
                        invalid.append(key)
                    else:
                        self.V_sr_sum[key] = new_sum
                for key, inner in list(self.V_sr.items()):
                    if key in invalid:
                        continue
                    bad = False
                    for obj in list(inner.keys()):
                        old_value = float(inner[obj])
                        new_value = old_value * scale if math.isfinite(old_value) else 0.0
                        if new_value < 0.0 or not math.isfinite(new_value):
                            bad = True
                            break
                        if new_value == 0.0:
                            del inner[obj]
                        else:
                            inner[obj] = new_value
                    if bad or not inner:
                        invalid.append(key)
                for key in set(invalid):
                    self._clear_sr(key, where='update_state', reason=f'scaling produced invalid bucket state; rel_exp={rel_exp}, scale={scale}')
            self.time_shift = float(ts)
            weight = 1.0
            self._array_cache.clear()
            for key in list(self.V_sr.keys()):
                self._sr_version[key] = self._sr_version.get(key, 0) + 1
        else:
            weight = 2.0 ** rel_exp
        if weight <= 0.0 or not math.isfinite(weight):
            raise RuntimeError(f'DSH produced invalid update weight: ts={ts}, rel_exp={rel_exp}, weight={weight}')

        dirty = set()
        for s, r, o in events_i64:
            sr = self._sr_key(s, r)
            existing_sum = float(self.V_sr_sum.get(sr, 0.0))
            bucket = self.V_sr.get(sr)
            if bucket is not None and (existing_sum <= 0.0 or not math.isfinite(existing_sum)):
                self._clear_sr(sr, where='update_state', reason=f'existing bucket total invalid before update: total={existing_sum}')
            bucket = self.V_sr.setdefault(sr, {})
            old_value = float(bucket.get(int(o), 0.0))
            if old_value < 0.0 or not math.isfinite(old_value):
                self._clear_sr(sr, where='update_state', reason=f'existing bucket value invalid before update: object={int(o)}, value={old_value}')
                bucket = self.V_sr.setdefault(sr, {})
                old_value = 0.0
                existing_sum = 0.0
            new_value = old_value + weight
            new_sum = float(self.V_sr_sum.get(sr, 0.0)) + weight
            if new_value < 0.0 or new_sum <= 0.0 or not math.isfinite(new_value) or not math.isfinite(new_sum):
                self._clear_sr(sr, where='update_state', reason=f'update would create invalid state: object={int(o)}, new_value={new_value}, new_sum={new_sum}, weight={weight}')
                bucket = self.V_sr.setdefault(sr, {})
                bucket[int(o)] = weight
                self.V_sr_sum[sr] = weight
            else:
                bucket[int(o)] = new_value
                self.V_sr_sum[sr] = new_sum
            dirty.add(sr)
        for sr in dirty:
            self._mark_dirty(sr)

    def _materialize_bucket_scale(self, sr, scale, ts, stats, reason):
        bucket = self.V_sr.get(sr)
        if not bucket:
            self.bucket_scale[sr] = 1.0
            self.bucket_last_ts[sr] = float(ts)
            return 1.0
        old_sum = float(self.V_sr_sum.get(sr, 0.0))
        new_sum = old_sum * scale if math.isfinite(old_sum) else 0.0
        if new_sum <= 0.0 or not math.isfinite(new_sum):
            size = len(bucket)
            self._clear_sr(sr, warn=False)
            self.bucket_last_ts[sr] = float(ts)
            self.bucket_scale[sr] = 1.0
            stats['cleared_buckets'] += 1
            stats['cleared_entries'] += size
            stats['reason'] = f'{reason}: materialized total invalid'
            return 1.0

        empty = []
        for obj in list(bucket.keys()):
            old_value = float(bucket[obj])
            new_value = old_value * scale if math.isfinite(old_value) else 0.0
            if new_value < 0.0 or not math.isfinite(new_value):
                size = len(bucket)
                self._clear_sr(sr, warn=False)
                self.bucket_last_ts[sr] = float(ts)
                self.bucket_scale[sr] = 1.0
                stats['cleared_buckets'] += 1
                stats['cleared_entries'] += size
                stats['reason'] = f'{reason}: materialized value invalid'
                return 1.0
            if new_value == 0.0:
                empty.append(obj)
            else:
                bucket[obj] = new_value
        for obj in empty:
            del bucket[obj]
        if not bucket:
            self._clear_sr(sr, warn=False)
            self.bucket_last_ts[sr] = float(ts)
            self.bucket_scale[sr] = 1.0
            stats['cleared_buckets'] += 1
            stats['reason'] = f'{reason}: materialized bucket emptied'
            return 1.0

        self.V_sr_sum[sr] = new_sum
        self.bucket_scale[sr] = 1.0
        self.bucket_last_ts[sr] = float(ts)
        return 1.0

    def _advance_bucket_scale_to(self, sr, ts, stats):
        ts = float(ts)
        last_ts = self.bucket_last_ts.get(sr)
        if last_ts is None:
            self.bucket_last_ts[sr] = ts
            self.bucket_scale[sr] = 1.0
            return 1.0
        dt = ts - float(last_ts)
        if dt == 0.0:
            return float(self.bucket_scale.get(sr, 1.0))
        if dt < 0.0:
            raise RuntimeError(f'DSH bucket received decreasing timestamp: sr={sr}, last_ts={last_ts}, ts={ts}')
        rel_exp = self.decay_direct * dt
        if rel_exp <= 0.0:
            self.bucket_last_ts[sr] = ts
            return float(self.bucket_scale.get(sr, 1.0))
        if not math.isfinite(rel_exp):
            size = len(self.V_sr.get(sr, {}))
            self._clear_sr(sr, warn=False)
            self.bucket_last_ts[sr] = ts
            self.bucket_scale[sr] = 1.0
            stats['cleared_buckets'] += 1
            stats['cleared_entries'] += size
            stats['reason'] = 'non-finite bucket decay exponent'
            stats['max_rel_exp'] = float('inf')
            return 1.0
        if rel_exp >= DSH_BUCKET_CLEAR_EXP:
            size = len(self.V_sr.get(sr, {}))
            self._clear_sr(sr, warn=False)
            self.bucket_last_ts[sr] = ts
            self.bucket_scale[sr] = 1.0
            stats['cleared_buckets'] += 1
            stats['cleared_entries'] += size
            stats['reason'] = 'bucket stale beyond clear threshold'
            stats['max_rel_exp'] = max(stats['max_rel_exp'], float(rel_exp))
            return 1.0

        old_scale = float(self.bucket_scale.get(sr, 1.0))
        step_scale = 2.0 ** (-rel_exp)
        scale = old_scale * step_scale if math.isfinite(old_scale) else 0.0
        if scale <= 0.0 or not math.isfinite(scale):
            size = len(self.V_sr.get(sr, {}))
            self._clear_sr(sr, warn=False)
            self.bucket_last_ts[sr] = ts
            self.bucket_scale[sr] = 1.0
            stats['cleared_buckets'] += 1
            stats['cleared_entries'] += size
            stats['reason'] = 'bucket lazy scale became invalid'
            stats['max_rel_exp'] = max(stats['max_rel_exp'], float(rel_exp))
            return 1.0
        stats['max_rel_exp'] = max(stats['max_rel_exp'], float(rel_exp))
        if scale < DSH_BUCKET_MATERIALIZE_SCALE:
            return self._materialize_bucket_scale(sr, scale, ts, stats, 'bucket lazy scale too small')
        self.bucket_scale[sr] = scale
        self.bucket_last_ts[sr] = ts
        return scale

    def _update_state_bucket_lazy(self, events_i64, ts):
        grouped = defaultdict(dict)
        for s, r, o in events_i64:
            sr = self._sr_key(s, r)
            obj = int(o)
            counts = grouped[sr]
            counts[obj] = counts.get(obj, 0) + 1

        dirty = set()
        stats = {'cleared_buckets': 0, 'cleared_entries': 0, 'max_rel_exp': 0.0, 'reason': ''}
        for sr, counts in grouped.items():
            scale = self._advance_bucket_scale_to(sr, ts, stats)
            existing_sum = float(self.V_sr_sum.get(sr, 0.0))
            bucket = self.V_sr.get(sr)
            if bucket is not None and (existing_sum <= 0.0 or not math.isfinite(existing_sum)):
                self._clear_sr(sr, where='update_state', reason=f'existing bucket total invalid before update: total={existing_sum}')
                scale = 1.0
            bucket = self.V_sr.setdefault(sr, {})
            self.bucket_last_ts[sr] = float(ts)
            self.bucket_scale[sr] = float(scale)

            add_sum = 0.0
            for obj, count in counts.items():
                old_value = float(bucket.get(obj, 0.0))
                if old_value < 0.0 or not math.isfinite(old_value):
                    self._clear_sr(sr, where='update_state', reason=f'existing bucket value invalid before update: object={obj}, value={old_value}')
                    bucket = self.V_sr.setdefault(sr, {})
                    self.bucket_last_ts[sr] = float(ts)
                    self.bucket_scale[sr] = 1.0
                    scale = 1.0
                    old_value = 0.0
                inc = float(count) / float(scale)
                new_value = old_value + inc
                if new_value < 0.0 or not math.isfinite(new_value):
                    self._clear_sr(sr, where='update_state', reason=f'update would create invalid value: object={obj}, new_value={new_value}, inc={inc}')
                    bucket = self.V_sr.setdefault(sr, {})
                    self.bucket_last_ts[sr] = float(ts)
                    self.bucket_scale[sr] = 1.0
                    scale = 1.0
                    inc = float(count)
                    new_value = inc
                bucket[obj] = new_value
                add_sum += inc

            new_sum = float(self.V_sr_sum.get(sr, 0.0)) + add_sum
            if new_sum <= 0.0 or not math.isfinite(new_sum):
                self._clear_sr(sr, where='update_state', reason=f'update would create invalid total: new_sum={new_sum}, add_sum={add_sum}')
                bucket = self.V_sr.setdefault(sr, {})
                bucket.clear()
                for obj, count in counts.items():
                    bucket[obj] = float(count)
                self.V_sr_sum[sr] = float(sum(counts.values()))
                self.bucket_last_ts[sr] = float(ts)
                self.bucket_scale[sr] = 1.0
            else:
                self.V_sr_sum[sr] = new_sum
            dirty.add(sr)

        for sr in dirty:
            self._mark_dirty(sr)
        if stats['cleared_buckets'] and self.log_bucket_stats:
            print(
                f'[DSH-info] bucket_lazy update_state: cleared stale/invalid buckets={stats["cleared_buckets"]} '
                f'entries={stats["cleared_entries"]} max_rel_exp={stats["max_rel_exp"]:.3g} '
                f'reason={stats["reason"]} ts={float(ts):g}',
                flush=True,
            )

    def update_state(self, events, ts):
        events_i64 = events[:, :3].astype(np.int64, copy=False)
        if self.decay_mode == 'bucket_lazy':
            self._update_state_bucket_lazy(events_i64, ts)
        else:
            self._update_state_global_lazy(events_i64, ts)

    def predict_batch(self, batch_data, neg_samples):
        batch_size, width = neg_samples.shape
        pos = np.zeros((batch_size, 1), dtype=np.float32)
        neg = np.zeros((batch_size, width), dtype=np.float32)
        if batch_size == 0:
            return pos, neg

        batch_i64 = batch_data[:, :3].astype(np.int64, copy=False)
        sources = batch_i64[:, 0]
        rels = batch_i64[:, 1]
        pos_obj = batch_i64[:, 2]
        sr_keys = sources * self.rel_shift + rels
        order = np.argsort(sr_keys, kind='stable')

        start = 0
        while start < batch_size:
            sr = int(sr_keys[order[start]])
            end = start + 1
            while end < batch_size and int(sr_keys[order[end]]) == sr:
                end += 1
            bucket = self.V_sr.get(sr)
            total = self.V_sr_sum.get(sr, 0.0)
            if not bucket:
                start = end
                continue
            if total <= 0.0 or not math.isfinite(float(total)):
                self._clear_sr(sr, where='predict_batch', reason=f'invalid total before prediction: total={total}')
                start = end
                continue

            keys, values = self._bucket_arrays(sr)
            if len(keys):
                rows = np.ascontiguousarray(order[start:end].astype(np.int64, copy=False))
                _score_dsh_candidates(
                    keys,
                    values,
                    rows,
                    pos_obj,
                    neg_samples,
                    pos,
                    neg,
                    np.float32(1.0),
                )
            start = end

        return pos, neg


def _combine_direct_scores(dsh_pos, dsh_neg, dmh_pos, dmh_neg, direct_single_hop):
    direct_single_hop = float(direct_single_hop)
    dmh_weight = 1.0 - direct_single_hop
    if direct_single_hop == 1.0:
        return dsh_pos, dsh_neg
    if direct_single_hop == 0.0:
        return dmh_pos, dmh_neg
    return (
        direct_single_hop * dsh_pos + dmh_weight * dmh_pos,
        direct_single_hop * dsh_neg + dmh_weight * dmh_neg,
    )


@njit(cache=True)
def _fast_tagmax_update_one(keys, scores, tags, lens, norms, m_trans,
                            source, targets, rels, ppr_k, ppr_alpha, ppr_beta,
                            temp_keys, temp_scores, temp_tags,
                            stamp, pos, epoch):
    if lens[source] == 0:
        keys[source, 0] = source
        scores[source, 0] = 1.0
        tags[source, 0] = -1
        lens[source] = 1
        norms[source] = 0.0

    for i in range(targets.shape[0]):
        target = targets[i]
        if lens[target] == 0:
            keys[target, 0] = target
            scores[target, 0] = 1.0
            tags[target, 0] = -1
            lens[target] = 1
            norms[target] = 0.0

    last_norm = norms[source]
    new_norm = last_norm * ppr_beta + ppr_beta
    s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
    s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
    s2_per_event = s2_total / targets.shape[0]

    count = 0
    slen = lens[source]
    for j in range(slen):
        k = keys[source, j]
        temp_keys[count] = k
        temp_scores[count] = scores[source, j] * s1
        temp_tags[count] = tags[source, j]
        stamp[k] = epoch
        pos[k] = count
        count += 1

    for i in range(targets.shape[0]):
        target = targets[i]
        r_update = rels[i]
        tlen = lens[target]
        sum_logic_weighted = 0.0
        for j in range(tlen):
            r_k = tags[target, j]
            factor = 1.0
            if r_k != -1:
                factor = m_trans[r_update, r_k]
            c = scores[target, j] * factor
            if c > 0.0:
                sum_logic_weighted += c
        if sum_logic_weighted <= 0.0:
            continue

        scale = s2_per_event / sum_logic_weighted
        for j in range(tlen):
            k = keys[target, j]
            r_k = tags[target, j]
            factor = 1.0
            if r_k != -1:
                factor = m_trans[r_update, r_k]
            c = scores[target, j] * factor
            if c <= 0.0:
                continue
            contrib = c * scale
            if stamp[k] == epoch:
                p = pos[k]
                if contrib > temp_scores[p]:
                    temp_scores[p] = contrib
                    temp_tags[p] = r_update
            else:
                temp_keys[count] = k
                temp_scores[count] = contrib
                temp_tags[count] = r_update
                stamp[k] = epoch
                pos[k] = count
                count += 1

    if stamp[source] == epoch:
        p = pos[source]
        temp_scores[p] += ppr_alpha
        temp_tags[p] = -1
    else:
        temp_keys[count] = source
        temp_scores[count] = ppr_alpha
        temp_tags[count] = -1
        stamp[source] = epoch
        pos[source] = count
        count += 1

    if count > ppr_k:
        order = np.argsort(temp_scores[:count])
        sum_top = 0.0
        for out_i in range(ppr_k):
            idx = order[count - 1 - out_i]
            sum_top += temp_scores[idx]
        if sum_top > 0.0:
            inv = 1.0 / sum_top
            for out_i in range(ppr_k):
                idx = order[count - 1 - out_i]
                keys[source, out_i] = temp_keys[idx]
                scores[source, out_i] = temp_scores[idx] * inv
                tags[source, out_i] = temp_tags[idx]
            lens[source] = ppr_k
        else:
            keys[source, 0] = source
            scores[source, 0] = 1.0
            tags[source, 0] = -1
            lens[source] = 1
    else:
        order = np.argsort(temp_scores[:count])
        for out_i in range(count):
            idx = order[count - 1 - out_i]
            keys[source, out_i] = temp_keys[idx]
            scores[source, out_i] = temp_scores[idx]
            tags[source, out_i] = temp_tags[idx]
        lens[source] = count

    norms[source] = new_norm


@njit(parallel=True, cache=True)
def _fast_tagmax_update_source_join_batch(keys, scores, tags, lens, norms, m_trans,
                                          source_ids, group_offsets, flat_targets, flat_rels,
                                          ppr_k, ppr_alpha, ppr_beta,
                                          out_keys, out_scores, out_tags, out_lens, out_norms):
    num_groups = source_ids.shape[0]
    for g in prange(num_groups):
        source = source_ids[g]
        start = group_offsets[g]
        end = group_offsets[g + 1]
        num_events = end - start

        last_norm = norms[source]
        new_norm = last_norm * ppr_beta + ppr_beta
        s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
        s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
        s2_per_event = s2_total / num_events

        capacity = ppr_k * (num_events + 1) + 2
        table_size = 1
        while table_size < capacity * 2:
            table_size *= 2
        table_mask = table_size - 1
        table_keys = np.empty(table_size, dtype=np.int32)
        table_pos = np.empty(table_size, dtype=np.int32)
        for h0 in range(table_size):
            table_keys[h0] = -1

        uniq_keys = np.empty(capacity, dtype=np.int32)
        uniq_scores = np.empty(capacity, dtype=np.float32)
        uniq_tags = np.empty(capacity, dtype=np.int32)
        uniq_count = 0

        slen = lens[source]
        for j in range(slen):
            k = keys[source, j]
            sc = scores[source, j] * s1
            tg = tags[source, j]
            h = (np.int64(k) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = k
                    table_pos[h] = uniq_count
                    uniq_keys[uniq_count] = k
                    uniq_scores[uniq_count] = sc
                    uniq_tags[uniq_count] = tg
                    uniq_count += 1
                    break
                if old_key == k:
                    p = table_pos[h]
                    if sc > uniq_scores[p]:
                        uniq_scores[p] = sc
                        uniq_tags[p] = tg
                    break
                h = (h + 1) & table_mask

        for i in range(start, end):
            target = flat_targets[i]
            r_update = flat_rels[i]
            tlen = lens[target]
            sum_logic_weighted = 0.0
            for j in range(tlen):
                r_k = tags[target, j]
                factor = 1.0
                if r_k != -1:
                    factor = m_trans[r_update, r_k]
                c = scores[target, j] * factor
                if c > 0.0:
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0.0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for j in range(tlen):
                k = keys[target, j]
                r_k = tags[target, j]
                factor = 1.0
                if r_k != -1:
                    factor = m_trans[r_update, r_k]
                c = scores[target, j] * factor
                if c <= 0.0:
                    continue
                sc = c * scale
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = uniq_count
                        uniq_keys[uniq_count] = k
                        uniq_scores[uniq_count] = sc
                        uniq_tags[uniq_count] = r_update
                        uniq_count += 1
                        break
                    if old_key == k:
                        p = table_pos[h]
                        if sc > uniq_scores[p]:
                            uniq_scores[p] = sc
                            uniq_tags[p] = r_update
                        break
                    h = (h + 1) & table_mask

        h = (np.int64(source) * 2654435761) & table_mask
        while True:
            old_key = table_keys[h]
            if old_key == -1:
                table_keys[h] = source
                table_pos[h] = uniq_count
                uniq_keys[uniq_count] = source
                uniq_scores[uniq_count] = ppr_alpha
                uniq_tags[uniq_count] = -1
                uniq_count += 1
                break
            if old_key == source:
                p = table_pos[h]
                uniq_scores[p] += ppr_alpha
                uniq_tags[p] = -1
                break
            h = (h + 1) & table_mask

        if uniq_count <= 0:
            uniq_keys[uniq_count] = source
            uniq_scores[uniq_count] = 1.0
            uniq_tags[uniq_count] = -1
            uniq_count += 1

        if uniq_count > ppr_k:
            score_order = np.argsort(uniq_scores[:uniq_count])
            sum_top = 0.0
            for out_i in range(ppr_k):
                idx = score_order[uniq_count - 1 - out_i]
                sum_top += uniq_scores[idx]
            if sum_top > 0.0:
                inv = 1.0 / sum_top
                for out_i in range(ppr_k):
                    idx = score_order[uniq_count - 1 - out_i]
                    out_keys[g, out_i] = uniq_keys[idx]
                    out_scores[g, out_i] = uniq_scores[idx] * inv
                    out_tags[g, out_i] = uniq_tags[idx]
                out_lens[g] = ppr_k
            else:
                out_keys[g, 0] = source
                out_scores[g, 0] = 1.0
                out_tags[g, 0] = -1
                out_lens[g] = 1
        else:
            score_order = np.argsort(uniq_scores[:uniq_count])
            for out_i in range(uniq_count):
                idx = score_order[uniq_count - 1 - out_i]
                out_keys[g, out_i] = uniq_keys[idx]
                out_scores[g, out_i] = uniq_scores[idx]
                out_tags[g, out_i] = uniq_tags[idx]
            out_lens[g] = uniq_count

        out_norms[g] = new_norm


@njit(parallel=True, cache=True)
def _fast_tagmax_commit_source_join(keys, scores, tags, lens, norms,
                                    source_ids, out_keys, out_scores, out_tags, out_lens, out_norms):
    num_groups = source_ids.shape[0]
    for g in prange(num_groups):
        source = source_ids[g]
        n = out_lens[g]
        for j in range(n):
            keys[source, j] = out_keys[g, j]
            scores[source, j] = out_scores[g, j]
            tags[source, j] = out_tags[g, j]
        lens[source] = n
        norms[source] = out_norms[g]


@njit(parallel=True, cache=True)
def _fast_tagsum_update_source_join_batch(keys, scores, tags, tag_scores, lens, norms, m_trans,
                                          source_ids, group_offsets, flat_targets, flat_rels,
                                          ppr_k, ppr_alpha, ppr_beta,
                                          out_keys, out_scores, out_tags, out_tag_scores,
                                          out_lens, out_norms, xy=None, known=None,
                                          eta=1., sigma=2000., power=1, source_scope=False):
    num_groups = source_ids.shape[0]
    for g in prange(num_groups):
        source = source_ids[g]
        start = group_offsets[g]
        end = group_offsets[g + 1]
        num_events = end - start

        last_norm = norms[source]
        new_norm = last_norm * ppr_beta + ppr_beta
        s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
        s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
        s2_per_event = s2_total / num_events

        capacity = ppr_k * (num_events + 1) + 2
        table_size = 1
        while table_size < capacity * 2:
            table_size *= 2
        table_mask = table_size - 1
        table_keys = np.empty(table_size, dtype=np.int32)
        table_pos = np.empty(table_size, dtype=np.int32)
        for h0 in range(table_size):
            table_keys[h0] = -1

        uniq_keys = np.empty(capacity, dtype=np.int32)
        uniq_scores = np.empty(capacity, dtype=np.float32)
        uniq_tags = np.empty(capacity, dtype=np.int32)
        uniq_tag_scores = np.empty(capacity, dtype=np.float32)
        uniq_count = 0

        slen = lens[source]
        for j in range(slen):
            k = keys[source, j]
            sc = scores[source, j] * s1
            tg = tags[source, j]
            tsc = tag_scores[source, j] * s1
            h = (np.int64(k) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = k
                    table_pos[h] = uniq_count
                    uniq_keys[uniq_count] = k
                    uniq_scores[uniq_count] = sc
                    uniq_tags[uniq_count] = tg
                    uniq_tag_scores[uniq_count] = tsc
                    uniq_count += 1
                    break
                if old_key == k:
                    p = table_pos[h]
                    uniq_scores[p] += sc
                    if tsc > uniq_tag_scores[p]:
                        uniq_tag_scores[p] = tsc
                        uniq_tags[p] = tg
                    break
                h = (h + 1) & table_mask

        for i in range(start, end):
            target = flat_targets[i]
            r_update = flat_rels[i]
            tlen = lens[target]
            sum_logic_weighted = 0.0
            weighted = np.empty(tlen, np.float64)
            for j in range(tlen):
                r_k = tags[target, j]
                factor = 1.0
                if r_k != -1:
                    factor = m_trans[r_update, r_k]
                if xy is not None and known is not None:
                    origin = source if source_scope else target
                    factor *= spatial_weight(origin, keys[target, j], xy, known, eta, sigma, power)
                c = scores[target, j] * factor
                weighted[j] = c
                if c > 0.0:
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0.0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for j in range(tlen):
                k = keys[target, j]
                c = weighted[j]
                if c <= 0.0:
                    continue
                contrib = c * scale
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        table_keys[h] = k
                        table_pos[h] = uniq_count
                        uniq_keys[uniq_count] = k
                        uniq_scores[uniq_count] = contrib
                        if k == source:
                            uniq_tags[uniq_count] = -1
                            uniq_tag_scores[uniq_count] = 0.0
                        else:
                            uniq_tags[uniq_count] = r_update
                            uniq_tag_scores[uniq_count] = contrib
                        uniq_count += 1
                        break
                    if old_key == k:
                        p = table_pos[h]
                        uniq_scores[p] += contrib
                        if k != source and contrib > uniq_tag_scores[p]:
                            uniq_tag_scores[p] = contrib
                            uniq_tags[p] = r_update
                        break
                    h = (h + 1) & table_mask

        h = (np.int64(source) * 2654435761) & table_mask
        while True:
            old_key = table_keys[h]
            if old_key == -1:
                table_keys[h] = source
                table_pos[h] = uniq_count
                uniq_keys[uniq_count] = source
                uniq_scores[uniq_count] = ppr_alpha
                uniq_tags[uniq_count] = -1
                uniq_tag_scores[uniq_count] = ppr_alpha
                uniq_count += 1
                break
            if old_key == source:
                p = table_pos[h]
                uniq_scores[p] += ppr_alpha
                uniq_tags[p] = -1
                uniq_tag_scores[p] += ppr_alpha
                break
            h = (h + 1) & table_mask

        if uniq_count > ppr_k:
            score_order = np.argsort(uniq_scores[:uniq_count])
            sum_top = 0.0
            for out_i in range(ppr_k):
                idx = score_order[uniq_count - 1 - out_i]
                sum_top += uniq_scores[idx]
            if sum_top > 0.0:
                inv = 1.0 / sum_top
                for out_i in range(ppr_k):
                    idx = score_order[uniq_count - 1 - out_i]
                    out_keys[g, out_i] = uniq_keys[idx]
                    out_scores[g, out_i] = uniq_scores[idx] * inv
                    out_tags[g, out_i] = uniq_tags[idx]
                    out_tag_scores[g, out_i] = uniq_tag_scores[idx] * inv
                out_lens[g] = ppr_k
            else:
                out_keys[g, 0] = source
                out_scores[g, 0] = 1.0
                out_tags[g, 0] = -1
                out_tag_scores[g, 0] = 1.0
                out_lens[g] = 1
        else:
            score_order = np.argsort(uniq_scores[:uniq_count])
            for out_i in range(uniq_count):
                idx = score_order[uniq_count - 1 - out_i]
                out_keys[g, out_i] = uniq_keys[idx]
                out_scores[g, out_i] = uniq_scores[idx]
                out_tags[g, out_i] = uniq_tags[idx]
                out_tag_scores[g, out_i] = uniq_tag_scores[idx]
            out_lens[g] = uniq_count

        out_norms[g] = new_norm


@njit(parallel=True, cache=True)
def _fast_tagsum_commit_source_join(keys, scores, tags, tag_scores, lens, norms,
                                    source_ids, out_keys, out_scores, out_tags,
                                    out_tag_scores, out_lens, out_norms):
    num_groups = source_ids.shape[0]
    for g in prange(num_groups):
        source = source_ids[g]
        n = out_lens[g]
        for j in range(n):
            keys[source, j] = out_keys[g, j]
            scores[source, j] = out_scores[g, j]
            tags[source, j] = out_tags[g, j]
            tag_scores[source, j] = out_tag_scores[g, j]
        lens[source] = n
        norms[source] = out_norms[g]


@njit(parallel=True, cache=True)
def _fast_perrel_update_source_join_batch(entry_keys, rel_keys, rel_scores, entry_lens, rel_lens,
                                          norms, m_trans, source_ids, group_offsets,
                                          flat_targets, flat_rels, ppr_k, top_k_relation,
                                          num_rels, ppr_alpha, ppr_beta, use_mtrans,
                                          out_entry_keys, out_rel_keys, out_rel_scores,
                                          out_entry_lens, out_rel_lens, out_norms):
    num_groups = source_ids.shape[0]
    rel_base = num_rels + 1
    self_code = num_rels
    for g in prange(num_groups):
        source = source_ids[g]
        start = group_offsets[g]
        end = group_offsets[g + 1]
        num_events = end - start

        last_norm = norms[source]
        new_norm = last_norm * ppr_beta + ppr_beta
        s1 = (1.0 - ppr_alpha) * (last_norm * ppr_beta) / new_norm
        s2_total = (1.0 - ppr_alpha) * ppr_beta / new_norm
        s2_per_event = s2_total / num_events

        cand_cap = ppr_k * top_k_relation + ppr_k * num_events + 2
        cand_combo = np.empty(cand_cap, dtype=np.int64)
        cand_scores = np.empty(cand_cap, dtype=np.float32)
        count = 0

        slen = entry_lens[source]
        for eidx in range(slen):
            entry = entry_keys[source, eidx]
            rlen = rel_lens[source, eidx]
            for ridx in range(rlen):
                r = rel_keys[source, eidx, ridx]
                rel_code = self_code if r == -1 else r
                cand_combo[count] = np.int64(entry) * rel_base + rel_code
                cand_scores[count] = rel_scores[source, eidx, ridx] * s1
                count += 1

        for i in range(start, end):
            target = flat_targets[i]
            r_update = flat_rels[i]
            tlen = entry_lens[target]
            total_w = 0.0
            for eidx in range(tlen):
                rlen = rel_lens[target, eidx]
                m = 0.0
                for ridx in range(rlen):
                    r = rel_keys[target, eidx, ridx]
                    w = rel_scores[target, eidx, ridx]
                    if use_mtrans:
                        factor = 1.0
                        if r != -1:
                            factor = m_trans[r_update, r]
                        m += w * factor
                    else:
                        m += w
                if m > 0.0:
                    total_w += m
            if total_w <= 0.0:
                continue
            scale = s2_per_event / total_w
            for eidx in range(tlen):
                entry = entry_keys[target, eidx]
                rlen = rel_lens[target, eidx]
                m = 0.0
                for ridx in range(rlen):
                    r = rel_keys[target, eidx, ridx]
                    w = rel_scores[target, eidx, ridx]
                    if use_mtrans:
                        factor = 1.0
                        if r != -1:
                            factor = m_trans[r_update, r]
                        m += w * factor
                    else:
                        m += w
                if m <= 0.0:
                    continue
                cand_combo[count] = np.int64(entry) * rel_base + r_update
                cand_scores[count] = m * scale
                count += 1

        cand_combo[count] = np.int64(source) * rel_base + self_code
        cand_scores[count] = ppr_alpha
        count += 1

        if count <= 0:
            out_entry_keys[g, 0] = source
            out_rel_keys[g, 0, 0] = -1
            out_rel_scores[g, 0, 0] = 1.0
            out_entry_lens[g] = 1
            out_rel_lens[g, 0] = 1
            out_norms[g] = new_norm
            continue

        order = np.argsort(cand_combo[:count])
        tmp_entry_keys = np.empty(count, dtype=np.int32)
        tmp_totals = np.zeros(count, dtype=np.float32)
        tmp_rel_keys = np.empty((count, top_k_relation), dtype=np.int32)
        tmp_rel_scores = np.zeros((count, top_k_relation), dtype=np.float32)
        tmp_rel_lens = np.zeros(count, dtype=np.int32)
        entry_count = 0

        first = order[0]
        cur_combo = cand_combo[first]
        cur_score = cand_scores[first]

        cur_entry = np.int32(cur_combo // rel_base)
        cur_rel_code = np.int32(cur_combo - np.int64(cur_entry) * rel_base)
        cur_rel = -1 if cur_rel_code == self_code else cur_rel_code
        cur_rel_count = 0
        cur_rel_keys = np.empty(top_k_relation, dtype=np.int32)
        cur_rel_scores = np.zeros(top_k_relation, dtype=np.float32)

        for oi in range(1, count + 1):
            flush_pair = False
            if oi == count:
                flush_pair = True
            else:
                p = order[oi]
                combo = cand_combo[p]
                if combo == cur_combo:
                    cur_score += cand_scores[p]
                else:
                    flush_pair = True

            if flush_pair:
                if cur_rel_count < top_k_relation:
                    cur_rel_keys[cur_rel_count] = cur_rel
                    cur_rel_scores[cur_rel_count] = cur_score
                    cur_rel_count += 1
                else:
                    min_idx = 0
                    min_score = cur_rel_scores[0]
                    for ridx in range(1, top_k_relation):
                        if cur_rel_scores[ridx] < min_score:
                            min_score = cur_rel_scores[ridx]
                            min_idx = ridx
                    if cur_score > min_score:
                        cur_rel_keys[min_idx] = cur_rel
                        cur_rel_scores[min_idx] = cur_score

                if oi < count:
                    p = order[oi]
                    next_combo = cand_combo[p]
                    next_entry = np.int32(next_combo // rel_base)
                    next_rel_code = np.int32(next_combo - np.int64(next_entry) * rel_base)
                    if next_entry != cur_entry:
                        tmp_entry_keys[entry_count] = cur_entry
                        tmp_rel_lens[entry_count] = cur_rel_count
                        total = 0.0
                        for ridx in range(cur_rel_count):
                            tmp_rel_keys[entry_count, ridx] = cur_rel_keys[ridx]
                            tmp_rel_scores[entry_count, ridx] = cur_rel_scores[ridx]
                            total += cur_rel_scores[ridx]
                        tmp_totals[entry_count] = total
                        entry_count += 1

                        cur_entry = next_entry
                        cur_rel_count = 0
                        for ridx in range(top_k_relation):
                            cur_rel_scores[ridx] = 0.0

                    cur_combo = next_combo
                    cur_score = cand_scores[p]
                    cur_rel = -1 if next_rel_code == self_code else next_rel_code

        tmp_entry_keys[entry_count] = cur_entry
        tmp_rel_lens[entry_count] = cur_rel_count
        total = 0.0
        for ridx in range(cur_rel_count):
            tmp_rel_keys[entry_count, ridx] = cur_rel_keys[ridx]
            tmp_rel_scores[entry_count, ridx] = cur_rel_scores[ridx]
            total += cur_rel_scores[ridx]
        tmp_totals[entry_count] = total
        entry_count += 1

        keep_count = entry_count
        if keep_count > ppr_k:
            keep_count = ppr_k
        entry_order = np.argsort(tmp_totals[:entry_count])
        total_keep = 0.0
        for out_i in range(keep_count):
            idx = entry_order[entry_count - 1 - out_i]
            total_keep += tmp_totals[idx]

        if total_keep > 0.0:
            inv = 1.0 / total_keep
            for out_i in range(keep_count):
                idx = entry_order[entry_count - 1 - out_i]
                out_entry_keys[g, out_i] = tmp_entry_keys[idx]
                out_rel_lens[g, out_i] = tmp_rel_lens[idx]
                for ridx in range(tmp_rel_lens[idx]):
                    out_rel_keys[g, out_i, ridx] = tmp_rel_keys[idx, ridx]
                    out_rel_scores[g, out_i, ridx] = tmp_rel_scores[idx, ridx] * inv
            out_entry_lens[g] = keep_count
        else:
            out_entry_keys[g, 0] = source
            out_rel_keys[g, 0, 0] = -1
            out_rel_scores[g, 0, 0] = 1.0
            out_entry_lens[g] = 1
            out_rel_lens[g, 0] = 1

        out_norms[g] = new_norm


@njit(parallel=True, cache=True)
def _fast_perrel_commit_source_join(entry_keys, rel_keys, rel_scores, entry_lens,
                                    rel_lens, norms, source_ids, out_entry_keys,
                                    out_rel_keys, out_rel_scores, out_entry_lens,
                                    out_rel_lens, out_norms, top_k_relation):
    num_groups = source_ids.shape[0]
    for g in prange(num_groups):
        source = source_ids[g]
        n = out_entry_lens[g]
        for eidx in range(n):
            entry_keys[source, eidx] = out_entry_keys[g, eidx]
            rlen = out_rel_lens[g, eidx]
            rel_lens[source, eidx] = rlen
            for ridx in range(rlen):
                rel_keys[source, eidx, ridx] = out_rel_keys[g, eidx, ridx]
                rel_scores[source, eidx, ridx] = out_rel_scores[g, eidx, ridx]
            for ridx in range(rlen, top_k_relation):
                rel_scores[source, eidx, ridx] = 0.0
        entry_lens[source] = n
        norms[source] = out_norms[g]


@njit(cache=True)
def _fast_perrel_predict_batch(entry_keys, rel_keys, rel_scores, entry_lens,
                               rel_lens, m_sim, batch_data, neg_samples,
                               pos_out, neg_out, stamp, pos, epoch):
    num_queries = batch_data.shape[0]
    num_negs = neg_samples.shape[1]
    num_rels = m_sim.shape[0]

    for i in range(num_queries):
        s = batch_data[i, 0]
        r = batch_data[i, 1]
        slen = entry_lens[s]

        for j in range(slen):
            k = entry_keys[s, j]
            stamp[k] = epoch
            pos[k] = j

        o = batch_data[i, 2]
        if stamp[o] == epoch:
            eidx = pos[o]
            total = 0.0
            rlen = rel_lens[s, eidx]
            for ridx in range(rlen):
                rel = rel_keys[s, eidx, ridx]
                w = 1.0
                if rel != -1:
                    w = m_sim[rel, r]
                total += rel_scores[s, eidx, ridx] * w
            pos_out[i, 0] = total

        for j in range(num_negs):
            ng = neg_samples[i, j]
            if ng < 0:
                continue
            if stamp[ng] == epoch:
                eidx = pos[ng]
                total = 0.0
                rlen = rel_lens[s, eidx]
                for ridx in range(rlen):
                    rel = rel_keys[s, eidx, ridx]
                    w = 1.0
                    if rel != -1:
                        w = m_sim[rel, r]
                    total += rel_scores[s, eidx, ridx] * w
                neg_out[i, j] = total

        epoch += 1
        if epoch > 9000000000000000000:
            for z in range(stamp.shape[0]):
                stamp[z] = 0
            epoch = 1
    return epoch


@njit(cache=True)
def _fast_tagmax_predict_batch(keys, scores, tags, lens, m_sim,
                               batch_data, neg_samples,
                               pos_out, neg_out,
                               stamp, pos, epoch):
    num_queries = batch_data.shape[0]
    num_negs = neg_samples.shape[1]
    num_rels = m_sim.shape[0]
    orig_rels = num_rels // 2

    for i in range(num_queries):
        s = batch_data[i, 0]
        r = batch_data[i, 1]
        o = batch_data[i, 2]
        slen = lens[s]

        for j in range(slen):
            k = keys[s, j]
            stamp[k] = epoch
            pos[k] = j

        if stamp[o] == epoch:
            p = pos[o]
            tag = tags[s, p]
            w = 1.0
            if tag != -1:
                w = m_sim[tag, r]
            pos_out[i, 0] = scores[s, p] * w

        for j in range(num_negs):
            ng = neg_samples[i, j]
            if ng == -1:
                continue
            if stamp[ng] == epoch:
                p = pos[ng]
                tag = tags[s, p]
                w = 1.0
                if tag != -1:
                    w = m_sim[tag, r]
                neg_out[i, j] = scores[s, p] * w

        epoch += 1
        if epoch > 9000000000000000000:
            stamp[:] = 0
            epoch = 1

    return epoch


@njit(parallel=True, cache=True)
def _fast_tag_predict_parts_batch(keys, scores, tags, lens, m_sim,
                                  batch_data, neg_samples, shared_w,
                                  top_share, pos_direct, neg_direct,
                                  pos_shared, neg_shared,
                                  use_top_direct, pos_shared_mask, neg_shared_mask):
    num_queries = batch_data.shape[0]
    num_negs = neg_samples.shape[1]
    num_rels = m_sim.shape[0]
    orig_rels = num_rels // 2

    for i in prange(num_queries):
        s = batch_data[i, 0]
        r = batch_data[i, 1]
        o = batch_data[i, 2]
        r_inv = (r + orig_rels) % num_rels
        slen = lens[s]

        src_share_len = slen
        if top_share > 0 and top_share < src_share_len:
            src_share_len = top_share

        table_size = 1
        while table_size < slen * 2 + 1:
            table_size *= 2
        table_mask = table_size - 1
        table_keys = np.empty(table_size, dtype=np.int32)
        table_pos = np.empty(table_size, dtype=np.int32)
        for h0 in range(table_size):
            table_keys[h0] = -1

        for j in range(slen):
            k = keys[s, j]
            h = (np.int64(k) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = k
                    table_pos[h] = j
                    break
                if old_key == k:
                    table_pos[h] = j
                    break
                h = (h + 1) & table_mask

        h = (np.int64(o) * 2654435761) & table_mask
        while True:
            old_key = table_keys[h]
            if old_key == -1:
                break
            if old_key == o:
                p = table_pos[h]
                tag = tags[s, p]
                w = 1.0
                if tag != -1:
                    w = m_sim[tag, r]
                pos_direct[i, 0] = scores[s, p] * w
                break
            h = (h + 1) & table_mask

        if (not use_top_direct) or pos_shared_mask[i, 0]:
            tlen = lens[o]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total = 0.0
            for tj in range(target_share_len):
                k = keys[o, tj]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == k:
                        sp = table_pos[h]
                        if sp < src_share_len:
                            if shared_w == 'unweighted':
                                total += scores[s, sp] * scores[o, tj]
                            elif shared_w == 'dual_msim':
                                st = tags[s, sp]
                                tt = tags[o, tj]
                                ws = 1.0
                                wo = 1.0
                                if st != -1:
                                    ws = m_sim[st, r]
                                if tt != -1:
                                    wo = m_sim[tt, r_inv]
                                total += scores[s, sp] * scores[o, tj] * ws * wo
                            else:
                                st = tags[s, sp]
                                tt = tags[o, tj]
                                wm = 1.0
                                if st != -1 and tt != -1:
                                    wm = m_sim[st, tt]
                                total += scores[s, sp] * scores[o, tj] * wm
                        break
                    h = (h + 1) & table_mask
            pos_shared[i, 0] = total

        for j in range(num_negs):
            ng = neg_samples[i, j]
            if ng < 0:
                continue

            h = (np.int64(ng) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    break
                if old_key == ng:
                    p = table_pos[h]
                    tag = tags[s, p]
                    w = 1.0
                    if tag != -1:
                        w = m_sim[tag, r]
                    neg_direct[i, j] = scores[s, p] * w
                    break
                h = (h + 1) & table_mask

            if use_top_direct and not neg_shared_mask[i, j]:
                continue

            tlen = lens[ng]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total = 0.0
            for tj in range(target_share_len):
                k = keys[ng, tj]
                h = (np.int64(k) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == k:
                        sp = table_pos[h]
                        if sp < src_share_len:
                            if shared_w == 'unweighted':
                                total += scores[s, sp] * scores[ng, tj]
                            elif shared_w == 'dual_msim':
                                st = tags[s, sp]
                                tt = tags[ng, tj]
                                ws = 1.0
                                wo = 1.0
                                if st != -1:
                                    ws = m_sim[st, r]
                                if tt != -1:
                                    wo = m_sim[tt, r_inv]
                                total += scores[s, sp] * scores[ng, tj] * ws * wo
                            else:
                                st = tags[s, sp]
                                tt = tags[ng, tj]
                                wm = 1.0
                                if st != -1 and tt != -1:
                                    wm = m_sim[st, tt]
                                total += scores[s, sp] * scores[ng, tj] * wm
                        break
                    h = (h + 1) & table_mask
            neg_shared[i, j] = total


@njit(parallel=True, cache=True)
def _fast_perrel_predict_parts_batch(entry_keys, rel_keys, rel_scores,
                                     entry_lens, rel_lens, m_sim,
                                     batch_data, neg_samples, shared_w,
                                     top_share, pos_direct, neg_direct,
                                     pos_shared, neg_shared,
                                     use_top_direct, pos_shared_mask, neg_shared_mask):
    num_queries = batch_data.shape[0]
    num_negs = neg_samples.shape[1]
    num_rels = m_sim.shape[0]
    orig_rels = num_rels // 2

    for i in prange(num_queries):
        s = batch_data[i, 0]
        r = batch_data[i, 1]
        o = batch_data[i, 2]
        r_inv = (r + orig_rels) % num_rels
        slen = entry_lens[s]

        src_share_len = slen
        if top_share > 0 and top_share < src_share_len:
            src_share_len = top_share

        table_size = 1
        while table_size < slen * 2 + 1:
            table_size *= 2
        table_mask = table_size - 1
        table_keys = np.empty(table_size, dtype=np.int32)
        table_pos = np.empty(table_size, dtype=np.int32)
        for h0 in range(table_size):
            table_keys[h0] = -1

        for eidx in range(slen):
            k = entry_keys[s, eidx]
            h = (np.int64(k) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    table_keys[h] = k
                    table_pos[h] = eidx
                    break
                if old_key == k:
                    table_pos[h] = eidx
                    break
                h = (h + 1) & table_mask

        h = (np.int64(o) * 2654435761) & table_mask
        while True:
            old_key = table_keys[h]
            if old_key == -1:
                break
            if old_key == o:
                eidx = table_pos[h]
                total = 0.0
                rlen = rel_lens[s, eidx]
                for ridx in range(rlen):
                    rel = rel_keys[s, eidx, ridx]
                    w = 1.0
                    if rel != -1:
                        w = m_sim[rel, r]
                    total += rel_scores[s, eidx, ridx] * w
                pos_direct[i, 0] = total
                break
            h = (h + 1) & table_mask

        if (not use_top_direct) or pos_shared_mask[i, 0]:
            tlen = entry_lens[o]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total_shared = 0.0
            for teidx in range(target_share_len):
                entry = entry_keys[o, teidx]
                h = (np.int64(entry) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == entry:
                        seidx = table_pos[h]
                        if seidx < src_share_len:
                            srlen = rel_lens[s, seidx]
                            trlen = rel_lens[o, teidx]
                            if shared_w == 'unweighted':
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    sw += rel_scores[s, seidx, ridx]
                                for ridx in range(trlen):
                                    tw += rel_scores[o, teidx, ridx]
                                total_shared += sw * tw
                            elif shared_w == 'dual_msim':
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    rel = rel_keys[s, seidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r]
                                    sw += rel_scores[s, seidx, ridx] * w
                                for ridx in range(trlen):
                                    rel = rel_keys[o, teidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r_inv]
                                    tw += rel_scores[o, teidx, ridx] * w
                                total_shared += sw * tw
                            else:
                                for sridx in range(srlen):
                                    rs = rel_keys[s, seidx, sridx]
                                    ws0 = rel_scores[s, seidx, sridx]
                                    for tridx in range(trlen):
                                        rt = rel_keys[o, teidx, tridx]
                                        wm = 1.0
                                        if rs != -1 and rt != -1:
                                            wm = m_sim[rs, rt]
                                        total_shared += ws0 * rel_scores[o, teidx, tridx] * wm
                        break
                    h = (h + 1) & table_mask
            pos_shared[i, 0] = total_shared

        for j in range(num_negs):
            ng = neg_samples[i, j]
            if ng < 0:
                continue

            h = (np.int64(ng) * 2654435761) & table_mask
            while True:
                old_key = table_keys[h]
                if old_key == -1:
                    break
                if old_key == ng:
                    eidx = table_pos[h]
                    total = 0.0
                    rlen = rel_lens[s, eidx]
                    for ridx in range(rlen):
                        rel = rel_keys[s, eidx, ridx]
                        w = 1.0
                        if rel != -1:
                            w = m_sim[rel, r]
                        total += rel_scores[s, eidx, ridx] * w
                    neg_direct[i, j] = total
                    break
                h = (h + 1) & table_mask

            if use_top_direct and not neg_shared_mask[i, j]:
                continue

            tlen = entry_lens[ng]
            target_share_len = tlen
            if top_share > 0 and top_share < target_share_len:
                target_share_len = top_share
            total_shared = 0.0
            for teidx in range(target_share_len):
                entry = entry_keys[ng, teidx]
                h = (np.int64(entry) * 2654435761) & table_mask
                while True:
                    old_key = table_keys[h]
                    if old_key == -1:
                        break
                    if old_key == entry:
                        seidx = table_pos[h]
                        if seidx < src_share_len:
                            srlen = rel_lens[s, seidx]
                            trlen = rel_lens[ng, teidx]
                            if shared_w == 'unweighted':
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    sw += rel_scores[s, seidx, ridx]
                                for ridx in range(trlen):
                                    tw += rel_scores[ng, teidx, ridx]
                                total_shared += sw * tw
                            elif shared_w == 'dual_msim':
                                sw = 0.0
                                tw = 0.0
                                for ridx in range(srlen):
                                    rel = rel_keys[s, seidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r]
                                    sw += rel_scores[s, seidx, ridx] * w
                                for ridx in range(trlen):
                                    rel = rel_keys[ng, teidx, ridx]
                                    w = 1.0
                                    if rel != -1:
                                        w = m_sim[rel, r_inv]
                                    tw += rel_scores[ng, teidx, ridx] * w
                                total_shared += sw * tw
                            else:
                                for sridx in range(srlen):
                                    rs = rel_keys[s, seidx, sridx]
                                    ws0 = rel_scores[s, seidx, sridx]
                                    for tridx in range(trlen):
                                        rt = rel_keys[ng, teidx, tridx]
                                        wm = 1.0
                                        if rs != -1 and rt != -1:
                                            wm = m_sim[rs, rt]
                                        total_shared += ws0 * rel_scores[ng, teidx, tridx] * wm
                        break
                    h = (h + 1) & table_mask
            neg_shared[i, j] = total_shared


class FastPerRelSourceJoinPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 dict_mode='per_rel', shared_w='dual_msim',
                 per_rel_use_mtrans=False,
                 ppr_k=1000, top_k_relation=4,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1, top_direct=-1,
                 max_events_in_single_batch=20000,
                 source_join_log_batches=True):
        if dict_mode != 'per_rel':
            raise ValueError('FastPerRelSourceJoinPredictor requires dict_mode=per_rel')
        if int(top_k_relation) <= 0:
            raise ValueError('FastPerRelSourceJoinPredictor requires top_k_relation > 0')

        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.device = device
        self.dict_mode = dict_mode
        self.shared_w = shared_w
        self.top_share = int(top_share)
        self.top_direct = int(top_direct)
        self.per_rel_use_mtrans = bool(per_rel_use_mtrans)
        self.ppr_k = int(ppr_k)
        self.top_k_relation = int(top_k_relation)
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_beta = float(ppr_beta)
        self.max_events_in_single_batch = max(1, int(max_events_in_single_batch))
        self.source_join_log_batches = bool(source_join_log_batches)

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.entry_keys = np.empty((self.num_nodes, self.ppr_k), dtype=np.int32)
        self.rel_keys = np.empty((self.num_nodes, self.ppr_k, self.top_k_relation), dtype=np.int32)
        self.rel_scores = np.zeros((self.num_nodes, self.ppr_k, self.top_k_relation), dtype=np.float32)
        self.entry_lens = np.zeros(self.num_nodes, dtype=np.int32)
        self.rel_lens = np.zeros((self.num_nodes, self.ppr_k), dtype=np.int32)
        self.ppr_norms = np.zeros(self.num_nodes, dtype=np.float32)

        self._pred_stamp = np.zeros(self.num_nodes, dtype=np.int64)
        self._pred_pos = np.zeros(self.num_nodes, dtype=np.int32)
        self._pred_epoch = np.int64(1)
        self.source_join_stats = {
            'batches': 0,
            'events': 0,
            'sources': 0,
            'kernel_time_s': 0.0,
            'commit_time_s': 0.0,
            'max_batch_events': 0,
            'max_batch_sources': 0,
            'max_group_events': 0,
            'max_output_mem_mb': 0.0,
            'max_scratch_est_mb': 0.0,
        }

    @property
    def needs_m_trans(self):
        return self.per_rel_use_mtrans

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_sim_np = np.ascontiguousarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_trans_np = np.ascontiguousarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    @staticmethod
    def _max_rss_mb():
        return FastTagMaxSourceJoinPredictor._max_rss_mb()

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.entry_lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.entry_keys[missing, 0] = missing
        self.rel_keys[missing, 0, 0] = self.R_SELF
        self.rel_scores[missing, 0, 0] = 1.0
        self.entry_lens[missing] = 1
        self.rel_lens[missing, 0] = 1
        self.ppr_norms[missing] = 0.0

    def _make_source_batches(self, order, groups):
        return FastTagMaxSourceJoinPredictor._make_source_batches(self, order, groups)

    def _materialize_batch(self, batch_chunks, groups):
        return FastTagMaxSourceJoinPredictor._materialize_batch(self, batch_chunks, groups)

    def update_state(self, events, t_norm=None):
        if self.per_rel_use_mtrans and self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastPerRelSourceJoinPredictor.update_state')
        if self.M_trans_np is None:
            self.M_trans_np = np.ones((self.num_rels, self.num_rels), dtype=np.float32)

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_chunks in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_chunks, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            n_groups = len(batch_chunks)
            out_entry_keys = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_rel_keys = np.empty((n_groups, self.ppr_k, self.top_k_relation), dtype=np.int32)
            out_rel_scores = np.zeros((n_groups, self.ppr_k, self.top_k_relation), dtype=np.float32)
            out_entry_lens = np.empty(n_groups, dtype=np.int32)
            out_rel_lens = np.zeros((n_groups, self.ppr_k), dtype=np.int32)
            out_norms = np.empty(n_groups, dtype=np.float32)

            active_workers = max(1, min(int(n_groups), int(threads)))
            cand_cap = self.ppr_k * self.top_k_relation + self.ppr_k * int(max_group_events) + 2
            output_mem_mb = (
                out_entry_keys.nbytes + out_rel_keys.nbytes + out_rel_scores.nbytes +
                out_entry_lens.nbytes + out_rel_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = cand_cap * 32.0 * active_workers / (1024.0 ** 2)

            t0 = time.perf_counter()
            _fast_perrel_update_source_join_batch(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.ppr_norms, self.M_trans_np, source_ids,
                group_offsets, flat_targets, flat_rels, self.ppr_k,
                self.top_k_relation, self.num_rels, self.ppr_alpha, self.ppr_beta,
                self.per_rel_use_mtrans, out_entry_keys, out_rel_keys, out_rel_scores,
                out_entry_lens, out_rel_lens, out_norms,
            )
            kernel_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            _fast_perrel_commit_source_join(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.ppr_norms, source_ids, out_entry_keys,
                out_rel_keys, out_rel_scores, out_entry_lens, out_rel_lens,
                out_norms, self.top_k_relation,
            )
            commit_s = time.perf_counter() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(n_groups)
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(n_groups))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[Shared-source-join] mode=per_rel t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={n_groups} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )

    def predict_batch(self, batch_data, neg_samples_arr, gamma, direct_scorer, direct_single_hop):
        if self.M_sim_np is None:
            raise RuntimeError('M_sim must be synced before prediction')
        batch_i64 = np.ascontiguousarray(batch_data.astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_samples_arr.astype(np.int64, copy=False))
        gamma = float(gamma)
        need_shared = abs(gamma) > 1e-12
        pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)
        dsh_pos = np.zeros_like(pos_direct)
        dsh_neg = np.zeros_like(neg_direct)

        if not need_shared:
            self._pred_epoch = _fast_perrel_predict_batch(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.M_sim_np, batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
            return _combine_direct_scores(dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop)

        pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
        use_top_direct = self.top_direct >= 0
        if use_top_direct:
            self._pred_epoch = _fast_perrel_predict_batch(
                self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
                self.rel_lens, self.M_sim_np, batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
            mask_pos_direct, mask_neg_direct = _combine_direct_scores(
                dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop
            )
            pos_shared_mask, neg_shared_mask = _top_direct_masks(
                mask_pos_direct, mask_neg_direct, neg_i64, self.top_direct
            )
        else:
            pos_shared_mask = np.empty((0, 0), dtype=np.bool_)
            neg_shared_mask = np.empty((0, 0), dtype=np.bool_)
        _fast_perrel_predict_parts_batch(
            self.entry_keys, self.rel_keys, self.rel_scores, self.entry_lens,
            self.rel_lens, self.M_sim_np, batch_i64, neg_i64, self.shared_w,
            self.top_share, pos_direct, neg_direct, pos_shared, neg_shared,
            use_top_direct, pos_shared_mask, neg_shared_mask,
        )
        if not use_top_direct:
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
        new_pos_direct, new_neg_direct = _combine_direct_scores(
            dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop
        )
        return new_pos_direct + gamma * pos_shared, new_neg_direct + gamma * neg_shared

    def get_source_join_stats(self):
        return dict(self.source_join_stats)


class SharedPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 dict_mode='per_rel', shared_w='dual_msim',
                 per_rel_use_mtrans=False,
                 ppr_k=500, top_k_relation=8,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1, top_direct=-1):
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.device = device

        self.dict_mode = dict_mode
        self.shared_w = shared_w
        self.top_share = int(top_share)
        self.top_direct = int(top_direct)

        self.per_rel_use_mtrans = per_rel_use_mtrans

        self.ppr_k = ppr_k
        self.top_k_relation = top_k_relation
        self.ppr_alpha = ppr_alpha
        self.ppr_beta = ppr_beta

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.ppr_norms = np.zeros(num_nodes, dtype=np.float32)
        if dict_mode in ('tag_max', 'tag_sum'):
            self.ppr_scores = [{} for _ in range(num_nodes)]
            self.ppr_tags = [{} for _ in range(num_nodes)]
            if dict_mode == 'tag_sum':
                self.ppr_tag_scores = [{} for _ in range(num_nodes)]
        elif dict_mode == 'per_rel':
            self.ppr_breakdown = [{} for _ in range(num_nodes)]
        else:
            raise ValueError(dict_mode)

    @property
    def needs_m_trans(self):
        """Whether M_trans is needed."""
        return self.dict_mode in ('tag_max', 'tag_sum') or self.per_rel_use_mtrans

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = M.detach().cpu().numpy()
        else:
            self.M_sim_np = np.asarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = M.detach().cpu().numpy()
        else:
            self.M_trans_np = np.asarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    def _tag_share_keys(self, score_dict):
        if self.top_share <= 0 or len(score_dict) <= self.top_share:
            return score_dict.keys()
        return set(k for k, _ in heapq.nlargest(self.top_share, score_dict.items(), key=lambda x: x[1]))

    def _perrel_share_keys(self, breakdown):
        if self.top_share <= 0 or len(breakdown) <= self.top_share:
            return breakdown.keys()
        return set(k for k, _ in heapq.nlargest(
            self.top_share, breakdown.items(), key=lambda x: sum(x[1].values())
        ))


    def update_state(self, events):
        by_source = defaultdict(list)
        for i in range(len(events)):
            by_source[int(events[i, 0])].append(
                (int(events[i, 2]), int(events[i, 1]))
            )
        for source, src_events in by_source.items():
            self._update_batched(source, src_events)

    def _update_batched(self, source, src_events):
        """Apply one timestamp-level update per source."""
        if self.dict_mode == 'tag_max':
            self._update_tag_max(source, src_events)
        elif self.dict_mode == 'tag_sum':
            self._update_tag_sum(source, src_events)
        else:
            self._update_per_rel(source, src_events)

    def _ppr_coeffs(self, source):
        """Return PPR update coefficients and normalization."""
        last_norm = float(self.ppr_norms[source])
        new_norm = last_norm * self.ppr_beta + self.ppr_beta
        s1 = (1.0 - self.ppr_alpha) * (last_norm * self.ppr_beta) / new_norm
        s2 = (1.0 - self.ppr_alpha) * self.ppr_beta / new_norm
        return s1, s2, self.ppr_alpha, new_norm


    def _ensure_tag_max_init(self, node):
        if not self.ppr_scores[node]:
            self.ppr_scores[node][node] = 1.0
            self.ppr_tags[node][node] = self.R_SELF
            self.ppr_norms[node] = 0.0

    def _update_tag_max(self, source, src_events):
        K = len(src_events)
        self._ensure_tag_max_init(source)
        for target, _ in src_events:
            self._ensure_tag_max_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_scores = self.ppr_scores[source]
        s_tags = self.ppr_tags[source]

        temp_scores = {k: w * s1 for k, w in s_scores.items()}
        temp_tags = dict(s_tags)

        for target, r_update in src_events:
            o_scores = self.ppr_scores[target]
            o_tags = self.ppr_tags[target]

            candidates = []
            sum_logic_weighted = 0.0
            for k, w_raw in o_scores.items():
                r_k = o_tags[k]
                factor = 1.0 if r_k == self.R_SELF else float(M_trans[r_update, r_k])
                c = w_raw * factor
                if c > 0:
                    candidates.append((k, c))
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for k, c in candidates:
                contrib = c * scale
                if k not in temp_scores or contrib > temp_scores[k]:
                    temp_scores[k] = contrib
                    temp_tags[k] = r_update

        temp_scores[source] = temp_scores.get(source, 0.0) + restart
        temp_tags[source] = self.R_SELF

        if len(temp_scores) > self.ppr_k:
            top = heapq.nlargest(self.ppr_k, temp_scores.items(), key=lambda x: x[1])
            sum_top = sum(v for _, v in top) or 1.0
            self.ppr_scores[source] = {k: v / sum_top for k, v in top}
            self.ppr_tags[source] = {k: temp_tags[k] for k, _ in top}
        else:
            self.ppr_scores[source] = temp_scores
            self.ppr_tags[source] = temp_tags

        self.ppr_norms[source] = new_norm


    def _ensure_tag_sum_init(self, node):
        if not self.ppr_scores[node]:
            self.ppr_scores[node][node] = 1.0
            self.ppr_tags[node][node] = self.R_SELF
            self.ppr_tag_scores[node][node] = 1.0
            self.ppr_norms[node] = 0.0

    def _update_tag_sum(self, source, src_events):
        K = len(src_events)
        self._ensure_tag_sum_init(source)
        for target, _ in src_events:
            self._ensure_tag_sum_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_scores = self.ppr_scores[source]            # sum
        s_tags = self.ppr_tags[source]
        s_tag_scores = self.ppr_tag_scores[source]

        temp_scores = {k: w * s1 for k, w in s_scores.items()}
        temp_tag_scores = {k: w * s1 for k, w in s_tag_scores.items()}
        temp_tags = dict(s_tags)

        for target, r_update in src_events:
            o_scores = self.ppr_scores[target]
            o_tags = self.ppr_tags[target]

            candidates = []
            sum_logic_weighted = 0.0
            for k, w_raw in o_scores.items():
                r_k = o_tags[k]
                factor = 1.0 if r_k == self.R_SELF else float(M_trans[r_update, r_k])
                c = w_raw * factor
                if c > 0:
                    candidates.append((k, c))
                    sum_logic_weighted += c
            if sum_logic_weighted <= 0:
                continue

            scale = s2_per_event / sum_logic_weighted
            for k, c in candidates:
                contrib = c * scale
                temp_scores[k] = temp_scores.get(k, 0.0) + contrib   # sum
                if k == source:
                    continue
                if k not in temp_tag_scores or contrib > temp_tag_scores[k]:
                    temp_tags[k] = r_update
                    temp_tag_scores[k] = contrib

        temp_scores[source] = temp_scores.get(source, 0.0) + restart
        temp_tags[source] = self.R_SELF
        temp_tag_scores[source] = temp_tag_scores.get(source, 0.0) + restart

        if len(temp_scores) > self.ppr_k:
            top = heapq.nlargest(self.ppr_k, temp_scores.items(), key=lambda x: x[1])
            sum_top = sum(v for _, v in top) or 1.0
            inv = 1.0 / sum_top
            kept = [k for k, _ in top]
            self.ppr_scores[source] = {k: temp_scores[k] * inv for k in kept}
            self.ppr_tags[source] = {k: temp_tags[k] for k in kept}
            self.ppr_tag_scores[source] = {k: temp_tag_scores[k] * inv for k in kept}
        else:
            self.ppr_scores[source] = temp_scores
            self.ppr_tags[source] = temp_tags
            self.ppr_tag_scores[source] = temp_tag_scores

        self.ppr_norms[source] = new_norm


    def _ensure_per_rel_init(self, node):
        if not self.ppr_breakdown[node]:
            self.ppr_breakdown[node] = {node: {self.R_SELF: 1.0}}
            self.ppr_norms[node] = 0.0

    def _update_per_rel(self, source, src_events):
        K = len(src_events)
        self._ensure_per_rel_init(source)
        for target, _ in src_events:
            self._ensure_per_rel_init(target)

        s1, s2_total, restart, new_norm = self._ppr_coeffs(source)
        s2_per_event = s2_total / K
        M_trans = self.M_trans_np if self.M_trans_np is not None else self.M_trans

        s_bd = self.ppr_breakdown[source]

        new_bd = {entry: {r: w * s1 for r, w in rd.items()}
                  for entry, rd in s_bd.items()}

        for target, r_update in src_events:
            t_bd = self.ppr_breakdown[target]

            if self.per_rel_use_mtrans:
                weighted = {}
                total_w = 0.0
                for entry, t_rd in t_bd.items():
                    m = sum(
                        w * (1.0 if r == self.R_SELF else float(M_trans[r_update, r]))
                        for r, w in t_rd.items()
                    )
                    if m > 0:
                        weighted[entry] = m
                        total_w += m
            else:
                weighted = {entry: sum(t_rd.values()) for entry, t_rd in t_bd.items()}
                total_w = sum(weighted.values())

            if total_w <= 0:
                continue
            scale = s2_per_event / total_w
            for entry, m in weighted.items():
                contrib = m * scale
                if entry not in new_bd:
                    new_bd[entry] = {}
                new_bd[entry][r_update] = new_bd[entry].get(r_update, 0.0) + contrib

        if source not in new_bd:
            new_bd[source] = {}
        new_bd[source][self.R_SELF] = new_bd[source].get(self.R_SELF, 0.0) + restart

        if self.top_k_relation > 0:
            for entry in list(new_bd.keys()):
                rd = new_bd[entry]
                if len(rd) > self.top_k_relation:
                    top_rel = heapq.nlargest(self.top_k_relation, rd.items(), key=lambda x: x[1])
                    new_bd[entry] = dict(top_rel)

        if len(new_bd) > self.ppr_k:
            totals = [(e, sum(rd.values())) for e, rd in new_bd.items()]
            top = heapq.nlargest(self.ppr_k, totals, key=lambda x: x[1])
            kept = set(e for e, _ in top)
            new_bd = {e: new_bd[e] for e in kept}

        total = sum(sum(rd.values()) for rd in new_bd.values())
        if total > 0:
            inv = 1.0 / total
            for entry in new_bd:
                new_bd[entry] = {r: w * inv for r, w in new_bd[entry].items()}
        else:
            new_bd = {source: {self.R_SELF: 1.0}}

        self.ppr_breakdown[source] = new_bd
        self.ppr_norms[source] = new_norm


    def predict_batch(self, batch_data, neg_samples_arr, gamma, direct_scorer, direct_single_hop):
        gamma = float(gamma)
        need_shared = abs(gamma) > 1e-12
        d_pos, d_neg, s_pos, s_neg = self._predict_parts(batch_data, neg_samples_arr, need_shared)
        dsh_pos, dsh_neg = direct_scorer.predict_batch(
            np.ascontiguousarray(batch_data.astype(np.int64, copy=False)),
            np.ascontiguousarray(neg_samples_arr.astype(np.int64, copy=False)),
        )
        new_d_pos, new_d_neg = _combine_direct_scores(dsh_pos, dsh_neg, d_pos, d_neg, direct_single_hop)
        if need_shared and self.top_direct >= 0:
            pos_mask, neg_mask = _top_direct_masks(
                new_d_pos,
                new_d_neg,
                np.asarray(neg_samples_arr, dtype=np.int64),
                self.top_direct,
            )
            s_pos = np.where(pos_mask, s_pos, 0.0).astype(np.float32, copy=False)
            s_neg = np.where(neg_mask, s_neg, 0.0).astype(np.float32, copy=False)
        return new_d_pos + gamma * s_pos, new_d_neg + gamma * s_neg

    def _predict_parts(self, batch_data, neg_samples_arr, need_shared):
        if self.dict_mode in ('tag_max', 'tag_sum'):
            return self._predict_tag_parts(batch_data, neg_samples_arr, need_shared)
        return self._predict_per_rel_parts(batch_data, neg_samples_arr, need_shared)


    def _predict_tag_parts(self, batch_data, neg_samples_arr, need_shared):
        M_sim = self.M_sim_np if self.M_sim_np is not None else self.M_sim.cpu().numpy()
        orig_rels = self.num_rels // 2

        B = len(batch_data)
        N = neg_samples_arr.shape[1]
        d_pos = np.zeros((B, 1), dtype=np.float32)
        d_neg = np.zeros((B, N), dtype=np.float32)
        s_pos = np.zeros((B, 1), dtype=np.float32)
        s_neg = np.zeros((B, N), dtype=np.float32)

        for i in range(B):
            s, r, o = int(batch_data[i, 0]), int(batch_data[i, 1]), int(batch_data[i, 2])
            r_inv = (r + orig_rels) % self.num_rels
            s_scores = self.ppr_scores[s]
            s_tags = self.ppr_tags[s]
            s_share_keys = self._tag_share_keys(s_scores) if need_shared else None

            def direct(target):
                if target not in s_scores:
                    return 0.0
                tag = s_tags[target]
                w = 1.0 if tag == self.R_SELF else float(M_sim[tag, r])
                return s_scores[target] * w

            def shared(target):
                t_scores = self.ppr_scores[target]
                t_tags = self.ppr_tags[target]
                common = s_share_keys & self._tag_share_keys(t_scores)
                if not common:
                    return 0.0
                tot = 0.0
                if self.shared_w == 'unweighted':
                    for z in common:
                        tot += s_scores[z] * t_scores[z]
                elif self.shared_w == 'dual_msim':
                    for z in common:
                        ws = 1.0 if s_tags[z] == self.R_SELF else float(M_sim[s_tags[z], r])
                        wo = 1.0 if t_tags[z] == self.R_SELF else float(M_sim[t_tags[z], r_inv])
                        tot += s_scores[z] * t_scores[z] * ws * wo
                elif self.shared_w == 'cross_msim':
                    for z in common:
                        if s_tags[z] == self.R_SELF or t_tags[z] == self.R_SELF:
                            wm = 1.0
                        else:
                            wm = float(M_sim[s_tags[z], t_tags[z]])
                        tot += s_scores[z] * t_scores[z] * wm
                return tot

            d_pos[i, 0] = direct(o)
            if need_shared:
                s_pos[i, 0] = shared(o)
            for j in np.flatnonzero(neg_samples_arr[i] != -1):
                ng = int(neg_samples_arr[i, j])
                d_neg[i, j] = direct(ng)
                if need_shared:
                    s_neg[i, j] = shared(ng)
        return d_pos, d_neg, s_pos, s_neg


    def _predict_per_rel_parts(self, batch_data, neg_samples_arr, need_shared):
        M_sim = self.M_sim_np if self.M_sim_np is not None else self.M_sim.cpu().numpy()
        orig_rels = self.num_rels // 2

        B = len(batch_data)
        N = neg_samples_arr.shape[1]
        d_pos = np.zeros((B, 1), dtype=np.float32)
        d_neg = np.zeros((B, N), dtype=np.float32)
        s_pos = np.zeros((B, 1), dtype=np.float32)
        s_neg = np.zeros((B, N), dtype=np.float32)

        def entry_w(rd, q_r):
            """sum_r rd[r] * M_sim[r, q_r]."""
            tot = 0.0
            for r, w in rd.items():
                tot += w * (1.0 if r == self.R_SELF else float(M_sim[r, q_r]))
            return tot

        for i in range(B):
            s, r, o = int(batch_data[i, 0]), int(batch_data[i, 1]), int(batch_data[i, 2])
            r_inv = (r + orig_rels) % self.num_rels
            s_bd = self.ppr_breakdown[s]
            if not s_bd:
                continue
            s_share_keys = self._perrel_share_keys(s_bd) if need_shared else None

            s_direct_w = {z: entry_w(rd, r) for z, rd in s_bd.items()}
            s_total_w = None
            if need_shared and self.shared_w == 'unweighted':
                s_total_w = {z: sum(rd.values()) for z, rd in s_bd.items()}

            def shared(target):
                t_bd = self.ppr_breakdown[target]
                if not t_bd:
                    return 0.0
                common = s_share_keys & self._perrel_share_keys(t_bd)
                if not common:
                    return 0.0
                tot = 0.0
                if self.shared_w == 'unweighted':
                    for z in common:
                        tot += s_total_w[z] * sum(t_bd[z].values())
                elif self.shared_w == 'dual_msim':
                    for z in common:
                        tot += s_direct_w[z] * entry_w(t_bd[z], r_inv)
                elif self.shared_w == 'cross_msim':
                    for z in common:
                        for r_s, w_s in s_bd[z].items():
                            for r_o, w_o in t_bd[z].items():
                                if r_s == self.R_SELF or r_o == self.R_SELF:
                                    wm = 1.0
                                else:
                                    wm = float(M_sim[r_s, r_o])
                                tot += w_s * w_o * wm
                return tot

            d_pos[i, 0] = s_direct_w.get(o, 0.0)
            if need_shared:
                s_pos[i, 0] = shared(o)
            for j in np.flatnonzero(neg_samples_arr[i] != -1):
                ng = int(neg_samples_arr[i, j])
                d_neg[i, j] = s_direct_w.get(ng, 0.0)
                if need_shared:
                    s_neg[i, j] = shared(ng)
        return d_pos, d_neg, s_pos, s_neg


class FastTagMaxArrayPredictor:
    R_SELF = -1

    def __init__(self, num_nodes, num_rels, *,
                 dict_mode='tag_max', shared_w='dual_msim',
                 per_rel_use_mtrans=False,
                 ppr_k=1000, top_k_relation=8,
                 ppr_alpha=0.1, ppr_beta=0.8, device='cuda:0',
                 top_share=-1, top_direct=-1):
        if dict_mode not in ('tag_max', 'tag_sum'):
            raise ValueError('FastTagMaxArrayPredictor only supports dict_mode=tag_max/tag_sum')
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.device = device
        self.dict_mode = dict_mode
        self.shared_w = shared_w
        self.top_share = int(top_share)
        self.top_direct = int(top_direct)
        self.per_rel_use_mtrans = per_rel_use_mtrans
        self.ppr_k = int(ppr_k)
        self.top_k_relation = top_k_relation
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_beta = float(ppr_beta)

        self.M_trans = None
        self.M_sim = None
        self.M_trans_np = None
        self.M_sim_np = None
        self.M_sim_dirty = True

        self.keys = np.empty((num_nodes, self.ppr_k), dtype=np.int32)
        self.scores = np.empty((num_nodes, self.ppr_k), dtype=np.float32)
        self.tags = np.empty((num_nodes, self.ppr_k), dtype=np.int32)
        self.lens = np.zeros(num_nodes, dtype=np.int32)
        self.ppr_norms = np.zeros(num_nodes, dtype=np.float32)

        self._stamp = np.zeros(num_nodes, dtype=np.int64)
        self._pos = np.zeros(num_nodes, dtype=np.int32)
        self._epoch = np.int64(1)
        self._pred_stamp = np.zeros(num_nodes, dtype=np.int64)
        self._pred_pos = np.zeros(num_nodes, dtype=np.int32)
        self._pred_epoch = np.int64(1)

        self._temp_capacity = 0
        self._temp_keys = np.empty(1, dtype=np.int32)
        self._temp_scores = np.empty(1, dtype=np.float32)
        self._temp_tags = np.empty(1, dtype=np.int32)

    @property
    def needs_m_trans(self):
        return True

    def sync_M_sim(self, M):
        self.M_sim = M
        if isinstance(M, torch.Tensor):
            self.M_sim_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_sim_np = np.ascontiguousarray(M, dtype=np.float32)
        self.M_sim_dirty = False

    def sync_M_trans(self, M):
        self.M_trans = M
        if isinstance(M, torch.Tensor):
            self.M_trans_np = np.ascontiguousarray(M.detach().cpu().numpy(), dtype=np.float32)
        else:
            self.M_trans_np = np.ascontiguousarray(M, dtype=np.float32)

    def mark_M_sim_dirty(self):
        self.M_sim_dirty = True

    def ensure_M_sim(self, semantic_updater):
        if self.M_sim_dirty or self.M_sim_np is None:
            self.sync_M_sim(semantic_updater.get_probability_M_sim())

    def _ensure_temp_capacity(self, num_events):
        capacity = self.ppr_k * (int(num_events) + 1) + 1
        if capacity <= self._temp_capacity:
            return
        self._temp_capacity = capacity
        self._temp_keys = np.empty(capacity, dtype=np.int32)
        self._temp_scores = np.empty(capacity, dtype=np.float32)
        self._temp_tags = np.empty(capacity, dtype=np.int32)

    def update_state(self, events):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagMaxArrayPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))

        max_events = 1
        for s in order:
            n = len(groups[s][0])
            if n > max_events:
                max_events = n
        self._ensure_temp_capacity(max_events)

        for source in order:
            target_list, rel_list = groups[source]
            targets = np.asarray(target_list, dtype=np.int32)
            rels = np.asarray(rel_list, dtype=np.int32)
            if self._epoch > 9000000000000000000:
                self._stamp.fill(0)
                self._epoch = np.int64(1)
            _fast_tagmax_update_one(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                self.M_trans_np, int(source), targets, rels,
                self.ppr_k, self.ppr_alpha, self.ppr_beta,
                self._temp_keys, self._temp_scores, self._temp_tags,
                self._stamp, self._pos, self._epoch,
            )
            self._epoch += 1

    def predict_batch(self, batch_data, neg_samples_arr, gamma, direct_scorer, direct_single_hop):
        if self.M_sim_np is None:
            raise RuntimeError('M_sim must be synced before prediction')

        batch_i64 = np.ascontiguousarray(batch_data.astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_samples_arr.astype(np.int64, copy=False))
        gamma = float(gamma)
        need_shared = abs(gamma) > 1e-12
        pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)

        if not need_shared:
            self._pred_epoch = _fast_tagmax_predict_batch(
                self.keys, self.scores, self.tags, self.lens, self.M_sim_np,
                batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            if abs(float(direct_single_hop)) <= EPS:
                return pos_direct, neg_direct
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
            return _combine_direct_scores(dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop)

        dsh_pos = np.zeros_like(pos_direct)
        dsh_neg = np.zeros_like(neg_direct)
        pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
        neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
        use_top_direct = self.top_direct >= 0
        if use_top_direct:
            self._pred_epoch = _fast_tagmax_predict_batch(
                self.keys, self.scores, self.tags, self.lens, self.M_sim_np,
                batch_i64, neg_i64, pos_direct, neg_direct,
                self._pred_stamp, self._pred_pos, self._pred_epoch,
            )
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
            mask_pos_direct, mask_neg_direct = _combine_direct_scores(
                dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop
            )
            pos_shared_mask, neg_shared_mask = _top_direct_masks(
                mask_pos_direct, mask_neg_direct, neg_i64, self.top_direct
            )
        else:
            pos_shared_mask = np.empty((0, 0), dtype=np.bool_)
            neg_shared_mask = np.empty((0, 0), dtype=np.bool_)
        _fast_tag_predict_parts_batch(
            self.keys, self.scores, self.tags, self.lens, self.M_sim_np,
            batch_i64, neg_i64, self.shared_w, self.top_share,
            pos_direct, neg_direct, pos_shared, neg_shared,
            use_top_direct, pos_shared_mask, neg_shared_mask,
        )
        if not use_top_direct:
            dsh_pos, dsh_neg = direct_scorer.predict_batch(batch_i64, neg_i64)
        new_pos_direct, new_neg_direct = _combine_direct_scores(
            dsh_pos, dsh_neg, pos_direct, neg_direct, direct_single_hop
        )
        return new_pos_direct + gamma * pos_shared, new_neg_direct + gamma * neg_shared


class FastTagMaxSourceJoinPredictor(FastTagMaxArrayPredictor):
    def __init__(self, *args, max_events_in_single_batch=20000,
                 source_join_log_batches=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_events_in_single_batch = max(1, int(max_events_in_single_batch))
        self.source_join_log_batches = bool(source_join_log_batches)
        self.source_join_stats = {
            'batches': 0,
            'events': 0,
            'sources': 0,
            'kernel_time_s': 0.0,
            'commit_time_s': 0.0,
            'max_batch_events': 0,
            'max_batch_sources': 0,
            'max_group_events': 0,
            'max_output_mem_mb': 0.0,
            'max_scratch_est_mb': 0.0,
        }

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.keys[missing, 0] = missing
        self.scores[missing, 0] = 1.0
        self.tags[missing, 0] = -1
        self.lens[missing] = 1
        self.ppr_norms[missing] = 0.0

    @staticmethod
    def _max_rss_mb():
        try:
            import resource
        except Exception:
            return None
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == 'darwin':
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0

    def _make_source_batches(self, order, groups):
        batches = []
        current = []
        cur_events = 0
        limit = self.max_events_in_single_batch
        for source in order:
            n_total = len(groups[source][0])
            start = 0
            while start < n_total:
                end = min(start + limit, n_total)
                n = end - start
                if cur_events > 0 and cur_events + n > limit:
                    batches.append(current)
                    current = []
                    cur_events = 0
                current.append((source, start, end))
                cur_events += n
                start = end
                if start < n_total:
                    batches.append(current)
                    current = []
                    cur_events = 0
        if current:
            batches.append(current)
        return batches

    def _materialize_batch(self, batch_chunks, groups):
        source_ids = np.asarray([chunk[0] for chunk in batch_chunks], dtype=np.int32)
        group_offsets = np.empty(len(batch_chunks) + 1, dtype=np.int32)
        group_offsets[0] = 0
        total_events = 0
        max_group_events = 0
        for i, (_, start, end) in enumerate(batch_chunks):
            n = end - start
            total_events += n
            if n > max_group_events:
                max_group_events = n
            group_offsets[i + 1] = total_events

        flat_targets = np.empty(total_events, dtype=np.int32)
        flat_rels = np.empty(total_events, dtype=np.int32)
        pos = 0
        for source, start, end in batch_chunks:
            target_list, rel_list = groups[source]
            n = end - start
            flat_targets[pos:pos + n] = np.asarray(target_list[start:end], dtype=np.int32)
            flat_rels[pos:pos + n] = np.asarray(rel_list[start:end], dtype=np.int32)
            pos += n
        return source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events

    def update_state(self, events, t_norm=None):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagMaxSourceJoinPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_sources in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_sources, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            out_keys = np.empty((len(batch_sources), self.ppr_k), dtype=np.int32)
            out_scores = np.empty((len(batch_sources), self.ppr_k), dtype=np.float32)
            out_tags = np.empty((len(batch_sources), self.ppr_k), dtype=np.int32)
            out_lens = np.empty(len(batch_sources), dtype=np.int32)
            out_norms = np.empty(len(batch_sources), dtype=np.float32)

            active_workers = max(1, min(int(len(batch_sources)), int(threads)))
            candidate_capacity = self.ppr_k * (int(max_group_events) + 1) + 1
            output_mem_mb = (
                out_keys.nbytes + out_scores.nbytes + out_tags.nbytes +
                out_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = candidate_capacity * 40.0 * active_workers / (1024.0 ** 2)

            t0 = time.perf_counter()
            _fast_tagmax_update_source_join_batch(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                self.M_trans_np, source_ids, group_offsets, flat_targets, flat_rels,
                self.ppr_k, self.ppr_alpha, self.ppr_beta,
                out_keys, out_scores, out_tags, out_lens, out_norms,
            )
            kernel_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            _fast_tagmax_commit_source_join(
                self.keys, self.scores, self.tags, self.lens, self.ppr_norms,
                source_ids, out_keys, out_scores, out_tags, out_lens, out_norms,
            )
            commit_s = time.perf_counter() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(len(batch_sources))
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(len(batch_sources)))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[Shared-source-join] t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={len(batch_sources)} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )

    def get_source_join_stats(self):
        return dict(self.source_join_stats)


class FastTagSumSourceJoinPredictor(FastTagMaxSourceJoinPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dict_mode != 'tag_sum':
            raise ValueError('FastTagSumSourceJoinPredictor requires dict_mode=tag_sum')
        self.tag_scores = np.empty((self.num_nodes, self.ppr_k), dtype=np.float32)

    def _ensure_nodes_initialized(self, node_ids):
        if node_ids.size == 0:
            return
        unique_nodes = np.unique(np.ascontiguousarray(node_ids, dtype=np.int32))
        missing = unique_nodes[self.lens[unique_nodes] == 0]
        if missing.size == 0:
            return
        self.keys[missing, 0] = missing
        self.scores[missing, 0] = 1.0
        self.tags[missing, 0] = -1
        self.tag_scores[missing, 0] = 1.0
        self.lens[missing] = 1
        self.ppr_norms[missing] = 0.0

    def update_state(self, events, t_norm=None):
        if self.M_trans_np is None:
            raise RuntimeError('M_trans must be synced before FastTagSumSourceJoinPredictor.update_state')

        groups = {}
        order = []
        for i in range(len(events)):
            s = int(events[i, 0])
            if s not in groups:
                groups[s] = ([], [])
                order.append(s)
            groups[s][0].append(int(events[i, 2]))
            groups[s][1].append(int(events[i, 1]))
        if not order:
            return

        batches = self._make_source_batches(order, groups)
        threads = get_num_threads() if get_num_threads is not None else 1
        t_label = 'n/a' if t_norm is None else str(int(t_norm))

        for batch_idx, batch_chunks in enumerate(batches, start=1):
            source_ids, group_offsets, flat_targets, flat_rels, total_events, max_group_events = (
                self._materialize_batch(batch_chunks, groups)
            )
            self._ensure_nodes_initialized(np.concatenate((source_ids, flat_targets)))

            n_groups = len(batch_chunks)
            out_keys = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_scores = np.empty((n_groups, self.ppr_k), dtype=np.float32)
            out_tags = np.empty((n_groups, self.ppr_k), dtype=np.int32)
            out_tag_scores = np.empty((n_groups, self.ppr_k), dtype=np.float32)
            out_lens = np.empty(n_groups, dtype=np.int32)
            out_norms = np.empty(n_groups, dtype=np.float32)

            active_workers = max(1, min(int(n_groups), int(threads)))
            candidate_capacity = self.ppr_k * (int(max_group_events) + 1) + 2
            output_mem_mb = (
                out_keys.nbytes + out_scores.nbytes + out_tags.nbytes +
                out_tag_scores.nbytes + out_lens.nbytes + out_norms.nbytes
            ) / (1024.0 ** 2)
            scratch_est_mb = candidate_capacity * 48.0 * active_workers / (1024.0 ** 2)

            t0 = time.perf_counter()
            _fast_tagsum_update_source_join_batch(
                self.keys, self.scores, self.tags, self.tag_scores, self.lens,
                self.ppr_norms, self.M_trans_np, source_ids, group_offsets,
                flat_targets, flat_rels, self.ppr_k, self.ppr_alpha, self.ppr_beta,
                out_keys, out_scores, out_tags, out_tag_scores, out_lens, out_norms,
                *getattr(self, 'spatial_state', (None, None, 1., 2000., 1, False)),
            )
            kernel_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            _fast_tagsum_commit_source_join(
                self.keys, self.scores, self.tags, self.tag_scores, self.lens,
                self.ppr_norms, source_ids, out_keys, out_scores, out_tags,
                out_tag_scores, out_lens, out_norms,
            )
            commit_s = time.perf_counter() - t1

            st = self.source_join_stats
            st['batches'] += 1
            st['events'] += int(total_events)
            st['sources'] += int(n_groups)
            st['kernel_time_s'] += float(kernel_s)
            st['commit_time_s'] += float(commit_s)
            st['max_batch_events'] = max(st['max_batch_events'], int(total_events))
            st['max_batch_sources'] = max(st['max_batch_sources'], int(n_groups))
            st['max_group_events'] = max(st['max_group_events'], int(max_group_events))
            st['max_output_mem_mb'] = max(st['max_output_mem_mb'], float(output_mem_mb))
            st['max_scratch_est_mb'] = max(st['max_scratch_est_mb'], float(scratch_est_mb))

            if self.source_join_log_batches:
                rss_mb = self._max_rss_mb()
                rss_text = 'n/a' if rss_mb is None else f'{rss_mb:.1f}MB'
                print(
                    f'[Shared-source-join] mode=tag_sum t_norm={t_label} '
                    f'batch={batch_idx}/{len(batches)} events={int(total_events)} '
                    f'sources={n_groups} max_group_events={int(max_group_events)} '
                    f'threads={int(threads)} out_mem={output_mem_mb:.1f}MB '
                    f'scratch_est={scratch_est_mb:.1f}MB max_rss={rss_text} '
                    f'kernel={kernel_s:.4f}s commit={commit_s:.4f}s',
                    flush=True,
                )


def update_runtime(predictor, direct_scorer, events_f64, t_norm, semantic_updater, logic_updater, step_times=None):
    """Advance matrices and predictor state."""
    def step(name, operation):
        if step_times is None:
            operation()
            return
        started = time.perf_counter()
        operation()
        cuda_synchronize(predictor.device)
        step_times[name] = step_times.get(name, 0.0) + time.perf_counter() - started

    step('msim_counts', lambda: semantic_updater.update_M_sim_step(events_f64, t_norm))
    predictor.mark_M_sim_dirty()
    if logic_updater is not None:
        step('mtrans_counts', lambda: logic_updater.update_M_trans_step(events_f64, t_norm))
        step('history_append', lambda: logic_updater.update_history_step(events_f64, t_norm))
        step('mtrans_normalize_copy', lambda: predictor.sync_M_trans(logic_updater.get_normalized_M_trans()))
    if hasattr(predictor, 'get_source_join_stats'):
        step('ppr_update', lambda: predictor.update_state(events_f64, t_norm=t_norm))
    else:
        step('ppr_update', lambda: predictor.update_state(events_f64))
    if direct_scorer is not None:
        step('dsh_update', lambda: direct_scorer.update_state(events_f64, t_norm))


def make_result_dir(args):
    spatial = spatial_signature(args, 'structure')
    common = dict(
        spatial=hashlib.sha256(json.dumps(spatial, sort_keys=True).encode()).hexdigest()[:12],
        dict_mode=args.dict_mode,
        shared_w=args.shared_w,
        ppr_k=args.ppr_k,
        batch_size=args.batch_size,
        ppr_a=args.ppr_alpha,
        ppr_b=args.ppr_beta,
        gamma=float(args.gamma),
        direct_single_hop=float(args.direct_single_hop),
        decay_direct=float(args.decay_direct),
        ns_q=args.ns_q,
        ns_seed=args.ns_seed,
        train_predict_ratio=args.train_predict_ratio,
        ws=args.window_semantic_sim,
        meb=args.max_events_in_single_batch,
        ts=args.top_share,
    )
    if bool(getattr(args, 'skip_val_eval', False)):
        common['skip_val_eval'] = 1
    if int(getattr(args, 'top_direct', -1)) >= 0:
        common['td'] = int(args.top_direct)
    if args.dataset in TGB_DATASETS or args.dataset in THG_DATASETS:
        common['close_update_backward'] = bool(getattr(args, 'close_update_backward', False))
    if args.dict_mode in ('tag_max', 'tag_sum') or args.per_rel_use_mtrans:
        common['decay_rt'] = args.decay_rel_trans
        common['wt'] = args.window_trans
    if args.dict_mode == 'per_rel':
        common['top_kr'] = args.top_k_relation
        common['prtm'] = int(args.per_rel_use_mtrans)
    return make_dir_name(getattr(args, 'output_root', 'results_structure'), args.dataset, args.seed, **common)


def build_runtime(args, num_nodes, num_rels, device):
    if args.dict_mode == "per_rel" and args.top_k_relation <= 0:
        raise ValueError("per_rel requires top_k_relation > 0")
    if args.source_join_threads > 0:
        set_num_threads(args.source_join_threads)
    torch.set_num_threads(args.num_threads)
    predictor_type = {"tag_max": FastTagMaxSourceJoinPredictor,
                      "tag_sum": FastTagSumSourceJoinPredictor,
                      "per_rel": FastPerRelSourceJoinPredictor}[args.dict_mode]
    predictor = predictor_type(
        num_nodes=num_nodes, num_rels=num_rels, dict_mode=args.dict_mode,
        shared_w=args.shared_w, per_rel_use_mtrans=args.per_rel_use_mtrans,
        ppr_k=args.ppr_k, top_k_relation=args.top_k_relation,
        ppr_alpha=args.ppr_alpha, ppr_beta=args.ppr_beta, device=device,
        top_share=args.top_share, top_direct=args.top_direct,
        max_events_in_single_batch=args.max_events_in_single_batch,
        source_join_log_batches=args.source_join_log_batches)
    semantic_updater = SemanticMatrixUpdater(
        num_nodes=num_nodes, num_rels=num_rels,
        window_size=args.window_semantic_sim, device=device)
    predictor.sync_M_sim(semantic_updater.get_probability_M_sim())
    logic_updater = None
    if predictor.needs_m_trans:
        logic_updater = LogicMatrixUpdater(
            num_nodes=num_nodes, num_rels=num_rels, window_size=args.window_trans,
            decay_factor=args.decay_rel_trans, device=device)
        predictor.sync_M_trans(logic_updater.get_normalized_M_trans())
    return predictor, semantic_updater, logic_updater


def load_structure_data(args):
    return load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
        data_root=args.data_root,
        lazy_negatives=args.dataset in TGB_DATASETS,
    )


def structure_time_value(data, t_norm, t_orig):
    return float(t_orig) / 86400.0 if data.get('is_thg', False) else t_norm


def runtime_num_rels(data):
    return int(data['num_rels_raw']) * 2 if data.get('is_thg', False) else data['num_rels']


def estimate_max_model_time(data):
    max_t = 0.0
    for split_name in ('train_list', 'val_list', 'test_list'):
        for _, t_norm, t_orig in data.get(split_name, []):
            max_t = max(max_t, float(structure_time_value(data, t_norm, t_orig)))
    return max_t


@dataclass(frozen=True)
class BConfig:
    mode: str = "continuous"
    binary_unseen: float = 0.0
    continuous_alpha: float = 0.0001


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    if split == "val":
        return data["val_list"]
    if split == "test":
        return data["test_list"]
    raise ValueError(split)


def events_for_update(events, data, args, is_train=False):
    events_f64 = np.ascontiguousarray(events, dtype=np.float64)
    reverse_update = data.get('is_thg', False) or (not is_train and data['is_tgb'])
    if reverse_update and not bool(getattr(args, 'close_update_backward', False)):
        return inverse_aug(events_f64, data['num_rels_raw'], runtime_num_rels(data))
    return events_f64


def max_norm(scores, valid):
    max_score = np.max(np.where(valid, scores, 0.0), axis=1, keepdims=True)
    denom = np.where(max_score > 0.0, max_score, 1.0)
    return np.where(valid, scores / denom, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    std = np.where(var <= EPS, 1.0, np.sqrt(var))
    return np.where(valid, (scores - mean) / std, 0.0).astype(np.float32, copy=False)


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def check_finite_scores(label, scores, valid):
    selected = scores if np.all(valid) else scores[valid]
    bad = int(selected.size - np.sum(np.isfinite(selected)))
    if bad:
        raise RuntimeError(f"{label} has {bad} non-finite valid scores")


def event_delta(events, num_nodes):
    if len(events) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    rels = events[:, 1].astype(np.int64, copy=False)
    objs = events[:, 2].astype(np.int64, copy=False)
    flat = rels * int(num_nodes) + objs
    unique, counts = np.unique(flat, return_counts=True)
    return (
        (unique // int(num_nodes)).astype(np.int64),
        (unique % int(num_nodes)).astype(np.int64),
        counts.astype(np.float32),
    )


class BTimeline:
    def __init__(self, num_rels, num_nodes, dense=False):
        self.num_rels = int(num_rels)
        self.num_nodes = int(num_nodes)
        self.counts = {}
        self.dense_counts = np.zeros((self.num_rels, self.num_nodes), dtype=np.float64) if dense else None
        self.counts_r = np.zeros(self.num_rels, dtype=np.float32)

    def update(self, events):
        rels, objs, vals = event_delta(events, self.num_nodes)
        if self.dense_counts is not None:
            self.dense_counts[rels, objs] += vals
        else:
            for r, o, v in zip(rels, objs, vals):
                key = int(r) * self.num_nodes + int(o)
                self.counts[key] = self.counts.get(key, 0.0) + float(v)
        if len(rels):
            self.counts_r += np.bincount(rels, weights=vals, minlength=self.num_rels).astype(np.float32)

    def count_batch(self, batch_data, neg_arr, width):
        rels = batch_data[:, 1].astype(np.int64)
        pos_obj = batch_data[:, 2].astype(np.int64)
        neg_obj = np.where(neg_arr[:, :width] < 0, 0, neg_arr[:, :width]).astype(np.int64)
        if self.dense_counts is not None:
            pos = self.dense_counts[rels, pos_obj].astype(np.float32)
            neg = self.dense_counts[rels[:, None], neg_obj].astype(np.float32)
            return rels, pos, np.where(neg_arr[:, :width] >= 0, neg, 0.0)
        pos_counts = np.zeros(len(batch_data), dtype=np.float32)
        neg_counts = np.zeros((len(batch_data), width), dtype=np.float32)
        for i, r in enumerate(rels):
            pos_counts[i] = self.counts.get(int(r) * self.num_nodes + int(pos_obj[i]), 0.0)
            for j in range(width):
                if neg_arr[i, j] >= 0:
                    neg_counts[i, j] = self.counts.get(int(r) * self.num_nodes + int(neg_obj[i, j]), 0.0)
        return rels, pos_counts, neg_counts

    def score_batch(self, rels, pos_counts, neg_counts, b_cfg):
        if b_cfg.mode == "binary":
            pos = np.where(pos_counts > 0, 1.0, b_cfg.binary_unseen).astype(np.float32).reshape(-1, 1)
            neg = np.where(neg_counts > 0, 1.0, b_cfg.binary_unseen).astype(np.float32)
            return pos, neg
        alpha = float(b_cfg.continuous_alpha)
        denom = np.maximum(self.counts_r[rels] + alpha * self.num_nodes, EPS)
        pos = ((pos_counts + alpha) / denom).astype(np.float32).reshape(-1, 1)
        neg = ((neg_counts + alpha) / denom.reshape(-1, 1)).astype(np.float32)
        return pos, neg


class CausalHistory:
    def __init__(self, num_nodes, num_rels):
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.source_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.tail_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.incident_count = np.zeros(self.num_nodes, dtype=np.float32)
        self.rel_count = np.zeros(self.num_rels, dtype=np.float32)
        self.source_rel = {}
        self.sr_tail = {}
        self.source_tail = {}

    def update(self, events):
        for s, r, o in events[:, :3].astype(np.int64, copy=False):
            s = int(s)
            r = int(r)
            o = int(o)
            if 0 <= s < self.num_nodes:
                self.source_count[s] += 1.0
                self.incident_count[s] += 1.0
            if 0 <= o < self.num_nodes:
                self.tail_count[o] += 1.0
                self.incident_count[o] += 1.0
            if 0 <= r < self.num_rels:
                self.rel_count[r] += 1.0
            sr = (s, r)
            self.source_rel[sr] = self.source_rel.get(sr, 0.0) + 1.0
            self.sr_tail.setdefault(sr, {})[o] = self.sr_tail.setdefault(sr, {}).get(o, 0) + 1
            self.source_tail.setdefault(s, {})[o] = self.source_tail.setdefault(s, {}).get(o, 0) + 1

    def features(self, sources, rels, cand_ids, rel_tail_counts):
        bsz, width = cand_ids.shape
        safe_o = np.where((cand_ids >= 0) & (cand_ids < self.num_nodes), cand_ids, 0)
        safe_s = np.where((sources >= 0) & (sources < self.num_nodes), sources, 0)
        safe_r = np.where((rels >= 0) & (rels < self.num_rels), rels, 0)
        source_count = np.repeat(self.source_count[safe_s].reshape(-1, 1), width, axis=1)
        rel_count = np.repeat(self.rel_count[safe_r].reshape(-1, 1), width, axis=1)
        source_rel = np.zeros((bsz, width), dtype=np.float32)
        exact_sro = np.zeros((bsz, width), dtype=np.float32)
        source_tail = np.zeros((bsz, width), dtype=np.float32)
        for i in range(bsz):
            source_rel[i, :] = self.source_rel.get((int(sources[i]), int(rels[i])), 0.0)
            sr_counter = self.sr_tail.get((int(sources[i]), int(rels[i])), {})
            st_counter = self.source_tail.get(int(sources[i]), {})
            exact_sro[i, :] = [sr_counter.get(int(o), 0) for o in cand_ids[i]]
            source_tail[i, :] = [st_counter.get(int(o), 0) for o in cand_ids[i]]
        invalid = (cand_ids < 0) | (cand_ids >= self.num_nodes)
        arrays = [
            self.tail_count[safe_o],
            self.incident_count[safe_o],
            rel_tail_counts,
            source_count,
            rel_count,
            source_rel,
            exact_sro,
            source_tail,
        ]
        return [np.where(invalid, 0.0, x).astype(np.float32, copy=False) for x in arrays]


class StructureComponentRuntime:
    def __init__(self, data, args, device, enable_pure_dmh_fast_path=False):
        num_rels = runtime_num_rels(data)
        self.predictor, self.semantic_updater, self.logic_updater = build_runtime(
            args,
            data["num_nodes"],
            num_rels,
            device,
        )
        self.pure_dmh_fast_path = bool(
            enable_pure_dmh_fast_path
            and isinstance(self.predictor, FastTagMaxArrayPredictor)
        )
        self.direct = None
        if not self.pure_dmh_fast_path:
            self.direct = DirectSingleHopScorer(
                num_rels,
                decay_direct=float(args.decay_direct),
                max_time_span=estimate_max_model_time(data),
                log_bucket_stats=bool(getattr(args, "dsh_log_bucket_stats", False)),
            )
        self.args = args
        resolve_spatial_args(args, 'structure')
        self.locations = make_locations(data, args) if args.use_spatial else None
        self.data = data
        self.last_t = -float('inf')
        self.relation_phase = None
        self.full_train_locations = None
        self.relation_eta = np.ones(num_rels)
        if self.locations is not None:
            if not isinstance(self.predictor, FastTagSumSourceJoinPredictor):
                raise ValueError('Spatial Structure requires dict_mode=tag_sum and Numba')
            self.predictor.spatial_state = (
                self.locations.xy, self.locations.known, args.spatial_eta,
                args.spatial_sigma_km, args.spatial_power, args.spatial_scope == 'source',
            )
        if self.pure_dmh_fast_path:
            print("[Structure] pure-DMH fast path: DSH and Shared disabled", flush=True)

    def ensure_ready_for_prediction(self):
        self.predictor.ensure_M_sim(self.semantic_updater)
        if self.locations is None or self.data.get('is_thg', False) or self.args.predict_relation_power == 0:
            return
        final = self.data['train_list'][-1]
        full = self.last_t >= structure_time_value(self.data, final[1], final[2])
        phase = 'full' if full else 'prefix'
        if phase != self.relation_phase:
            snapshots = self.data['train_list'] if full else self.data['train_list'][:self.data['train_predict_start_idx']]
            xy, known = self.full_train_locations if full else (self.locations.xy, self.locations.known)
            self.relation_eta = fit_relation_space(
                snapshots, xy, known, runtime_num_rels(self.data),
                self.args.predict_spatial_sigma_km, self.args.predict_spatial_power,
                self.args.predict_relation_power, self.args.predict_relation_prior,
            )
            self.relation_phase = phase

    def spatial_scores(self, batch_data, neg_arr, pos, neg):
        if self.locations is None:
            return pos, neg
        sources = batch_data[:, 0].astype(np.int64)
        candidates = np.concatenate((batch_data[:, 2:3].astype(np.int64), neg_arr), axis=1)
        weights = prediction_weights(sources, candidates, self.locations.xy, self.locations.known,
            self.args.predict_spatial_eta, self.args.predict_spatial_sigma_km, self.args.predict_spatial_power)
        if not self.data.get('is_thg', False) and self.args.predict_relation_power > 0:
            strength = self.relation_eta[batch_data[:, 1].astype(np.int64)]
            weights = 1. - strength[:, None] * (1.-weights)
        return pos*weights[:, :1], neg*weights[:, 1:]

    def predict_raw(self, batch_data, neg_arr):
        pos, neg = self.predictor.predict_batch(
            batch_data, neg_arr, self.args.gamma, self.direct,
            self.args.direct_single_hop,
        )
        return self.spatial_scores(batch_data, neg_arr, pos, neg)

    def predict_parts(self, batch_data, neg_arr, ensure_ready=True):
        if ensure_ready:
            self.ensure_ready_for_prediction()
        batch_i64 = np.ascontiguousarray(batch_data[:, :3].astype(np.int64, copy=False))
        neg_i64 = np.ascontiguousarray(neg_arr.astype(np.int64, copy=False))
        dsh_pos, dsh_neg = self.direct.predict_batch(batch_i64, neg_i64)
        dmh_pos, dmh_neg, shared_pos, shared_neg = predict_dmh_shared_parts(self.predictor, batch_i64, neg_i64)
        final_direct_pos, final_direct_neg = _combine_direct_scores(
            dsh_pos, dsh_neg, dmh_pos, dmh_neg, float(self.args.direct_single_hop)
        )
        if float(self.args.gamma) > 0.0 and int(getattr(self.args, "top_direct", -1)) >= 0:
            pos_mask, neg_mask = _top_direct_masks(
                final_direct_pos,
                final_direct_neg,
                neg_i64,
                int(self.args.top_direct),
            )
            shared_pos = np.where(pos_mask, shared_pos, 0.0).astype(np.float32, copy=False)
            shared_neg = np.where(neg_mask, shared_neg, 0.0).astype(np.float32, copy=False)
        final_pos = final_direct_pos + float(self.args.gamma) * shared_pos
        final_neg = final_direct_neg + float(self.args.gamma) * shared_neg
        final_pos, final_neg = self.spatial_scores(batch_data, neg_arr, final_pos, final_neg)
        return {
            "dsh": (dsh_pos, dsh_neg),
            "dmh": (dmh_pos, dmh_neg),
            "shared": (shared_pos, shared_neg),
            "direct": (final_direct_pos, final_direct_neg),
            "structure_raw": (final_pos, final_neg),
        }

    def update(self, events, t_norm, profile=None):
        events_f64 = np.asarray(events, dtype=np.float64)
        if self.locations is not None:
            started = time.perf_counter()
            self.locations.update(events_f64, t_norm)
            if profile is not None:
                steps = profile['update_steps']
                steps['spatial_locations'] = steps.get('spatial_locations', 0.) + time.perf_counter()-started
        self.last_t = t_norm
        final = self.data['train_list'][-1]
        if (self.locations is not None and not self.data.get('is_thg', False)
                and self.args.predict_relation_power > 0 and self.full_train_locations is None
                and t_norm >= structure_time_value(self.data, final[1], final[2])):
            self.full_train_locations = (self.locations.xy.copy(), self.locations.known.copy())
        update_runtime(
            self.predictor,
            self.direct,
            events_f64,
            t_norm,
            self.semantic_updater,
            self.logic_updater,
            profile['update_steps'] if profile is not None and getattr(self.args, 'profile_updates', True) else None,
        )


def predict_dmh_shared_parts(predictor, batch_i64, neg_i64):
    if hasattr(predictor, "_predict_parts"):
        return predictor._predict_parts(batch_i64, neg_i64, True)
    if getattr(predictor, "M_sim_np", None) is None:
        raise RuntimeError("M_sim must be synced before prediction")
    pos_direct = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
    neg_direct = np.zeros(neg_i64.shape, dtype=np.float32)
    pos_shared = np.zeros((batch_i64.shape[0], 1), dtype=np.float32)
    neg_shared = np.zeros(neg_i64.shape, dtype=np.float32)
    use_top_direct = False
    pos_mask = np.empty((0, 0), dtype=np.bool_)
    neg_mask = np.empty((0, 0), dtype=np.bool_)
    if isinstance(predictor, FastPerRelSourceJoinPredictor):
        _fast_perrel_predict_parts_batch(
            predictor.entry_keys,
            predictor.rel_keys,
            predictor.rel_scores,
            predictor.entry_lens,
            predictor.rel_lens,
            predictor.M_sim_np,
            batch_i64,
            neg_i64,
            predictor.shared_w,
            predictor.top_share,
            pos_direct,
            neg_direct,
            pos_shared,
            neg_shared,
            use_top_direct,
            pos_mask,
            neg_mask,
        )
    else:
        _fast_tag_predict_parts_batch(
            predictor.keys,
            predictor.scores,
            predictor.tags,
            predictor.lens,
            predictor.M_sim_np,
            batch_i64,
            neg_i64,
            predictor.shared_w,
            predictor.top_share,
            pos_direct,
            neg_direct,
            pos_shared,
            neg_shared,
            use_top_direct,
            pos_mask,
            neg_mask,
        )
    return pos_direct, neg_direct, pos_shared, neg_shared


def warmup_numba_updates(predictor):
    if not isinstance(predictor, (FastTagMaxSourceJoinPredictor, FastPerRelSourceJoinPredictor)):
        return 0.0
    started = time.perf_counter()
    temporary = copy(predictor)
    temporary.num_nodes = 2
    for name in (
        'keys', 'scores', 'tags', 'tag_scores', 'lens', 'ppr_norms',
        'entry_keys', 'rel_keys', 'rel_scores', 'entry_lens', 'rel_lens',
    ):
        array = getattr(predictor, name, None)
        if array is not None:
            setattr(temporary, name, array[:2].copy())
    temporary.source_join_stats = predictor.source_join_stats.copy()
    temporary.source_join_log_batches = False
    temporary.update_state(np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float64), t_norm=0)
    return time.perf_counter() - started


class StructureProgress:
    def __init__(self, stage, snapshots, args, profile=None, update_key=None):
        self.stage = stage
        self.total = len(snapshots)
        self.interval = args.log_interval_seconds
        self.last_log = time.perf_counter()
        print(f"[Structure] {stage}: {self.total} timestamps", flush=True)

    def report(self, completed, t_norm):
        now = time.perf_counter()
        if completed == self.total or now - self.last_log >= self.interval:
            print(f"[Structure] {self.stage}: {completed}/{self.total}", flush=True)
            self.last_log = now


def init_stream_state(data, args, device, profile=None, enable_pure_dmh_fast_path=False):
    prefix_started = time.perf_counter()
    runtime = StructureComponentRuntime(
        data, args, device, enable_pure_dmh_fast_path=enable_pure_dmh_fast_path
    )
    if profile is not None:
        profile['prefix_setup_seconds'] = time.perf_counter() - prefix_started
    numba_warmup_seconds = warmup_numba_updates(runtime.predictor)
    if runtime.locations is not None:
        started = time.perf_counter()
        prediction_weights(np.zeros(1, np.int64), np.zeros((1, 1), np.int64),
            runtime.locations.xy, runtime.locations.known, args.predict_spatial_eta,
            args.predict_spatial_sigma_km, args.predict_spatial_power)
        spatial_warmup = time.perf_counter() - started
        if profile is not None:
            profile['spatial_predict_warmup_seconds'] = spatial_warmup
    print(f'[Structure] effective_threads torch={torch.get_num_threads()} numba={get_num_threads()}', flush=True)
    if profile is not None:
        profile['numba_update_warmup_seconds'] = numba_warmup_seconds
    warmup = data["train_list"][: data["train_predict_start_idx"]]
    progress = StructureProgress('train_prefix', warmup, args, profile, 'prefix_update_seconds')
    for index, (events, t_norm, t_orig) in enumerate(warmup, 1):
        update_events = events_for_update(events, data, args, is_train=True)
        cuda_synchronize(device)
        started = time.perf_counter()
        with args._monitor.phase("train_update") if getattr(args, "_monitor", None) else nullcontext():
            runtime.update(update_events, structure_time_value(data, t_norm, t_orig), profile)
        cuda_synchronize(device)
        if profile is not None:
            profile['prefix_update_seconds'] += time.perf_counter() - started
            profile['prefix_events'] += len(update_events)
            profile['prefix_blocks'] += 1
        progress.report(index, t_norm)
    if profile is not None:
        profile['prefix_wall_seconds'] = time.perf_counter() - prefix_started
        profile['prefix_replay_seconds'] = (
            profile['prefix_wall_seconds'] - profile['prefix_setup_seconds'] - numba_warmup_seconds
        )
        profile['prefix_update_steps'] = profile['update_steps'].copy()
    return runtime


def init_timeline_history_state(data, split, args):
    timeline = BTimeline(runtime_num_rels(data), data["num_nodes"], dense=getattr(args, 'b_dense_counts', False))
    history = CausalHistory(data["num_nodes"], runtime_num_rels(data))
    warmup = []
    if split == "train":
        warmup = [(snap, True) for snap in data["train_list"][: data["train_predict_start_idx"]]]
    elif split == "val":
        warmup = [(snap, True) for snap in data["train_list"]]
    elif split == "test":
        warmup = [(snap, True) for snap in data["train_list"]]
        warmup.extend((snap, False) for snap in data["val_list"])
    for (events, _, _), is_train_update in warmup:
        update_events = events_for_update(events, data, args, is_train=is_train_update)
        timeline.update(update_events)
        history.update(update_events)
    return timeline, history


def component_score_dict(parts, valid):
    scores = {}
    for name, (pos, neg) in parts.items():
        scores[name] = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
    for key, val in scores.items():
        check_finite_scores(f"component {key}", val, valid)
    return scores


def structure_base_scores(scores, valid):
    keys = ("dsh", "dmh", "shared", "b")
    total = np.zeros_like(scores["dsh"], dtype=np.float32)
    for key in keys:
        total += max_norm(scores[key], valid)
    return (total / float(len(keys))).astype(np.float32, copy=False)


def timed_eval_batches(events, t_orig, neg_sampler, split, batch_size, profile):
    batches = iter(collect_eval_batch(events, t_orig, neg_sampler, split, batch_size))
    while True:
        started = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            return
        if profile is not None:
            profile['candidate_prepare_seconds'] += time.perf_counter() - started
        yield batch


def iter_component_blocks(data, split, args, device, runtime, component_names, profile=None):
    neg_sampler = data["negative_sampler"]
    snapshots = split_snapshots(data, split)
    total_events = sum(len(events) for events, _, _ in snapshots)
    processed_events = 0
    next_trace = 0.05
    stream_started = time.perf_counter()
    progress = StructureProgress(f'{split}_stream', snapshots, args, profile)
    source_join_before = (
        runtime.predictor.get_source_join_stats().copy()
        if hasattr(runtime.predictor, 'get_source_join_stats') else {}
    )
    for snapshot_index, (events, t_norm, t_orig) in enumerate(snapshots):
        for batch_data, neg_arr, neg_mask in timed_eval_batches(
            events, t_orig, neg_sampler, split, args.query_batch_size, profile
        ):
            cuda_synchronize(device)
            score_started = time.perf_counter()
            width = int(neg_arr.shape[1])
            started = time.perf_counter()
            runtime.ensure_ready_for_prediction()
            cuda_synchronize(device)
            if profile is not None:
                profile['ensure_msim_seconds'] += time.perf_counter() - started
            valid = np.concatenate(
                (np.ones((len(batch_data), 1), dtype=bool), neg_mask[:, :width]),
                axis=1,
            )
            started = time.perf_counter()
            with args._monitor.phase("test_forward") if split == "test" and getattr(args, "_monitor", None) else nullcontext():
                if component_names == ("structure_raw",):
                    pos, neg = runtime.predict_raw(batch_data, neg_arr)
                else:
                    parts = runtime.predict_parts(batch_data, neg_arr, ensure_ready=False)
            cuda_synchronize(device)
            if profile is not None:
                profile['structure_forward_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            if component_names == ("structure_raw",):
                scores = {
                    "structure_raw": np.concatenate((pos, neg), axis=1).astype(
                        np.float32, copy=False
                    )
                }
                check_finite_scores("structure_raw", scores["structure_raw"], valid)
            else:
                scores = component_score_dict(parts, valid)
            if profile is not None:
                profile['component_assembly_seconds'] += time.perf_counter() - started
            cuda_synchronize(device)
            if profile is not None:
                elapsed = time.perf_counter() - score_started
                profile['score_seconds'] += elapsed
                profile['queries'] += len(batch_data)
                profile['candidate_scores'] += int(np.sum(valid))
            yield SimpleNamespace(
                batch_data=batch_data,
                valid=valid,
                scores=scores,
                t_norm=t_norm,
                t_orig=t_orig,
            )
        skip_update = split == "test" and snapshot_index == len(snapshots) - 1
        if not skip_update:
            update_events = events_for_update(events, data, args, is_train=(split == "train"))
            cuda_synchronize(device)
            update_started = time.perf_counter()
            with args._monitor.phase("train_update") if split == "train" and getattr(args, "_monitor", None) else nullcontext():
                runtime.update(update_events, structure_time_value(data, t_norm, t_orig), profile)
            cuda_synchronize(device)
        if profile is not None:
            if skip_update:
                profile['final_update_skipped'] = True
            else:
                elapsed = time.perf_counter() - update_started
                profile['update_seconds'] += elapsed
                profile['update_blocks'] += 1
                profile['updated_events'] += len(update_events)
                profile['update_latency_seconds'].append(elapsed)
        progress.report(snapshot_index + 1, t_norm)
    if profile is not None:
        profile['stream_wall_seconds'] = time.perf_counter() - stream_started
        profile['all_update_seconds'] = (
            profile['prefix_update_seconds'] + profile['update_seconds']
        )
        profile['stream_update_steps'] = {
            name: seconds - profile.get('prefix_update_steps', {}).get(name, 0.0)
            for name, seconds in profile['update_steps'].items()
        }


def component_score_root(out_dir):
    return osp.join(out_dir, "component_scores")


def new_structure_profile():
    return {
        'update_steps': {},
        'negative_prepare_seconds': 0.0,
        'negative_release_seconds': 0.0,
        'numba_update_warmup_seconds': 0.0,
        'prefix_setup_seconds': 0.0,
        'prefix_wall_seconds': 0.0,
        'prefix_replay_seconds': 0.0,
        'prefix_update_seconds': 0.0,
        'prefix_events': 0,
        'prefix_blocks': 0,
        'candidate_prepare_seconds': 0.0,
        'ensure_msim_seconds': 0.0,
        'structure_forward_seconds': 0.0,
        'component_assembly_seconds': 0.0,
        'write_batch_seconds': 0.0,
        'ranking_metrics_seconds': 0.0,
        'update_seconds': 0.0,
        'update_blocks': 0,
        'updated_events': 0,
        'final_update_skipped': False,
        'update_latency_seconds': [],
        'score_seconds': 0.0,
        'stream_wall_seconds': 0.0,
        'queries': 0,
        'candidate_scores': 0,
        'query_latency_seconds': [],
        'stream_trace': [],
    }


def advance_structure_state(data, split, args, device, runtime, profile=None):
    if profile is None:
        profile = new_structure_profile()
    snapshots = split_snapshots(data, split)
    progress = StructureProgress(f'{split}_update_only', snapshots, args, profile)
    for index, (events, t_norm, t_orig) in enumerate(snapshots, 1):
        update_events = events_for_update(events, data, args, is_train=(split == "train"))
        cuda_synchronize(device)
        started = time.perf_counter()
        with args._monitor.phase("train_update") if split == "train" and getattr(args, "_monitor", None) else nullcontext():
            runtime.update(update_events, structure_time_value(data, t_norm, t_orig), profile)
        cuda_synchronize(device)
        profile['update_seconds'] += time.perf_counter() - started
        profile['updated_events'] += len(update_events)
        profile['update_blocks'] += 1
        progress.report(index, t_norm)
    return profile['update_seconds']


def save_component_score_stores(
    data, args, device, out_dir, splits=("train", "val", "test"), compute_metrics=True,
    monitor=None, component_names=("structure_raw",),
):
    root = ensure_dir(component_score_root(out_dir))
    context = {"dataset": args.dataset, "seed": args.seed, "ns_q": args.ns_q,
               "ns_seed": args.ns_seed, "train_predict_ratio": args.train_predict_ratio,
               "component_names": list(component_names),
               "spatial": spatial_signature(args, "structure")}
    metrics_path = osp.join(root, "metrics.json")
    expected = [osp.join(root, split, name, f"{split}_{suffix}")
                for split in splits for name in component_names
                for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json")]
    if not args.force_recompute and osp.isfile(metrics_path) and all(osp.isfile(p) for p in expected):
        with open(metrics_path, encoding="utf-8") as file:
            cached = json.load(file)
        if cached["run_context"] == context:
            print(f"[Structure] reuse scores: {root}", flush=True)
            return root, cached
    initial = new_structure_profile()
    raw_only_dmh = (component_names == ("structure_raw",) and args.dict_mode == "tag_sum"
                    and abs(args.gamma) <= EPS and abs(args.direct_single_hop) <= EPS)
    runtime = init_stream_state(data, args, device, initial, raw_only_dmh)
    metrics = {"run_context": context, "splits": {}}
    suffix_updates = 0.0
    for split in ("train", "val", "test"):
        if split not in splits:
            if any(s in splits for s in ("val", "test")) and split != "test":
                elapsed = advance_structure_state(data, split, args, device, runtime)
                if split == "train":
                    suffix_updates = elapsed
            continue
        writers = {name: ScoreWriter(ensure_dir(osp.join(root, split, name)), split)
                   for name in component_names}
        sums, rows = {}, 0
        profile = initial if split == "train" else new_structure_profile()
        for block in iter_component_blocks(data, split, args, device, runtime, component_names, profile):
            mask = block.valid[:, 1:]
            for name, scores in block.scores.items():
                writers[name].write_batch(scores[:, :1], scores[:, 1:], mask)
            if compute_metrics:
                scores = block.scores["structure_raw"]
                add_metric_sums(sums, compute_ranking_metric_sums(scores[:, :1], scores[:, 1:], mask))
            rows += len(block.valid)
        data["negative_sampler"].release_mode(split)
        print(f"[Structure] saving {split}: {rows} queries", flush=True)
        for writer in writers.values():
            writer.close()
        values = finalize_metric_sums(sums) if compute_metrics else {}
        metrics["splits"][split] = {"rows": rows, "metrics": {"structure_raw": values}}
        if values:
            print_run_metrics(f"Structure/{split}", values)
        if split == "train":
            suffix_updates = profile["update_seconds"]
        elif split == "test":
            test_seconds = profile["structure_forward_seconds"]
    metrics["efficiency"] = efficiency_metrics(
        initial["prefix_update_seconds"] + suffix_updates,
        monitor.peak_rss("train_update") if monitor else 0,
        test_seconds if "test" in splits else 0,
        monitor.peak_rss("test_forward") if monitor else 0, "cpu_rss")
    print_efficiency("Structure", metrics["efficiency"])
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return root, metrics


def main(args):
    splits = ('train', 'test') if args.skip_val_eval else ('train', 'val', 'test')
    return run_component_scores(
        args, splits=splits,
        include_hybrid_features=not args.save_structure_only,
    )


def run_component_scores(args, data=None, splits=("train", "test"), include_hybrid_features=False):
    resolve_spatial_args(args, "structure")
    args.ns_q = default_ns_q(args.dataset) if args.ns_q is None else args.ns_q
    args.query_batch_size = getattr(args, "query_batch_size", None) or args.batch_size
    set_compute_threads(args.num_threads)
    set_random_seed(args.seed)
    device = select_torch_device(args.gpu)
    out_dir = make_result_dir(args)
    monitor = RunMetrics("structure", args)
    args._monitor = monitor
    try:
        if data is None:
            data = load_structure_data(args)
        component_dir, metrics = save_component_score_stores(
            data, args, device, out_dir, splits=splits, monitor=monitor,
            component_names=COMPONENT_SCORE_NAMES if include_hybrid_features else ("structure_raw",))
    finally:
        report = monitor.finish()
        del args._monitor
    save_config(out_dir, report["parameters"])
    print(f"[Structure] output={out_dir}", flush=True)
    return component_dir, metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser('Score Structure components.')
    parser.add_argument('--dataset', default='ICEWS14')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ns_q', type=int)
    parser.add_argument('--ns_seed', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=80)
    parser.add_argument('--save_structure_only', action='store_true',
                        help='Save only final Structure scores; by default also save Hybrid component scores.')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--output_root', default='results_structure')
    parser.add_argument('--use_best_hyper_params', action='store_true')
    parser.add_argument('--best_hyper_params_file')
    parser.add_argument('--train_predict_ratio', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_events_in_single_batch', type=int, default=20000)
    parser.add_argument('--source_join_threads', type=int, default=0)
    parser.add_argument('--source_join_log_batches', type=int, default=0)
    parser.add_argument('--profile_updates', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--log_interval_seconds', type=float, default=60.0)
    parser.add_argument('--dsh_log_bucket_stats', action='store_true', default=False)
    parser.add_argument('--close_update_backward', action='store_true', default=False)
    parser.add_argument('--dict_mode', choices=('tag_sum', 'tag_max', 'per_rel'), default='tag_sum')
    parser.add_argument('--shared_w', choices=('dual_msim', 'cross_msim', 'unweighted'), default='dual_msim')
    parser.add_argument('--per_rel_use_mtrans', action='store_true', default=False)
    parser.add_argument('--ppr_k', type=int, default=1000)
    parser.add_argument('--top_k_relation', type=int, default=0)
    parser.add_argument('--ppr_alpha', type=float, default=0.01)
    parser.add_argument('--ppr_beta', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--direct_single_hop', type=float, default=1.0)
    parser.add_argument('--decay_direct', type=float, default=1.0)
    parser.add_argument('--top_share', type=int, default=100)
    parser.add_argument('--top_direct', type=int, default=-1)
    parser.add_argument('--decay_rel_trans', type=float, default=0.05)
    parser.add_argument('--window_semantic_sim', type=float, default=5.0)
    parser.add_argument('--window_trans', type=float, default=5.0)
    parser.add_argument('--skip_val_eval', action='store_true', default=False)
    add_spatial_args(parser, 'structure')
    args = parser.parse_args(argv)
    if args.use_best_hyper_params or args.best_hyper_params_file:
        tokens = sys.argv[1:] if argv is None else argv
        provided = {token.split("=", 1)[0][2:].removeprefix("no-").replace("-", "_")
                    for token in tokens if token.startswith("--")}
        overrides = {key: getattr(args, key) for key in provided if hasattr(args, key)}
        apply_best_params(args, "structure", args.best_hyper_params_file)
        for key, value in overrides.items():
            setattr(args, key, value)
    args.ns_q = default_ns_q(args.dataset) if args.ns_q is None else args.ns_q
    return args


def cli():
    main(parse_args())


if __name__ == '__main__':
    cli()
