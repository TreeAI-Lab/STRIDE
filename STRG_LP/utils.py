import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import pickle
import random
import re
import struct
import threading
import time
import uuid
import zipfile
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch
from numba import njit, prange
from scipy import sparse


TGB_DATASETS = ["tkgl-polecat", "tkgl-icews"]
THG_DATASETS = {
    "Yelp-NOLA": "New_Orleans_LA",
    "Yelp-PHL": "Philadelphia_PA",
    "Yelp-TPA": "Tampa_FL",
    "Yelp-BOI": "Boise_ID",
}
NOT_TGB_DATASETS = [
    "ICEWS14",
    "GDELT",
]
SUPPORTED_DATASETS = TGB_DATASETS + NOT_TGB_DATASETS + list(THG_DATASETS)
HIT_KS = [1, 10]
PROJECT_DIR = Path(__file__).resolve().parent


def data_directory(dataset, root=None):
    """Resolve a dataset folder relative to the training code, not the working directory."""
    root = Path(root or "data").expanduser()
    return (PROJECT_DIR / root / dataset).resolve()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda_device(device):
    device = torch.device(device)
    return device if device.type == "cuda" and torch.cuda.is_available() else None


def select_torch_device(gpu=0):
    return f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"


def cuda_synchronize(device):
    dev = cuda_device(device)
    if dev is not None:
        torch.cuda.synchronize(dev)


def inverse_aug(events, num_rels_raw, num_rels):
    inverse = events.copy()
    inverse[:, 0] = events[:, 2]
    inverse[:, 2] = events[:, 0]
    inverse[:, 1] = (events[:, 1] + int(num_rels_raw)) % int(num_rels)
    return np.vstack((events, inverse))


def _reformat_ts(timestamps, dataset_name='tkgl'):
    unique = np.array(sorted(np.unique(timestamps)), dtype=np.float64)
    if len(unique) <= 1:
        return np.zeros_like(timestamps, dtype=np.int64)

    diffs = np.diff(unique)
    step = float(np.min(diffs))
    if dataset_name in TGB_DATASETS and float(np.mean(diffs)) != step:
        if "icews" in dataset_name or "polecat" in dataset_name:
            step = 86400.0
        elif "wiki" in dataset_name or "yago" in dataset_name:
            step = 31536000.0
        else:
            step = float(np.mean(diffs))

    return np.ceil((timestamps - unique[0]) / step).astype(np.int64)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    # Multiple GPU servers may request the same negative cache concurrently.
    # Publish only a fully-written pickle; duplicate deterministic writers are
    # harmless and the final os.replace is atomic on the target filesystem.
    temp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if osp.exists(temp_path):
            os.remove(temp_path)


def _sparse_tools():
    from scipy.sparse import csr_matrix, save_npz, load_npz, vstack as sparse_vstack

    return csr_matrix, save_npz, load_npz, sparse_vstack


def _split_by_time(triples, ts_norm, ts_orig):
    if len(ts_norm) == 0:
        return []
    order = np.argsort(ts_norm, kind="stable")
    times = ts_norm[order]
    rows = np.ascontiguousarray(triples[order], dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1, len(times)]
    return [
        (rows[start:end], int(times[start]), int(ts_orig[order[start]]))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _split_events4_by_time(events4, time_to_norm=None):
    if len(events4) == 0:
        return []
    if len(events4) > 1 and np.any(events4[1:, 3] < events4[:-1, 3]):
        events4 = events4[np.argsort(events4[:, 3], kind="stable")]
    snapshots = []
    raw_times = events4[:, 3].astype(np.int64, copy=False)
    start = 0
    while start < len(events4):
        raw_ts = int(raw_times[start])
        end = start + 1
        while end < len(events4) and int(raw_times[end]) == raw_ts:
            end += 1
        t_norm = int(time_to_norm[int(raw_ts)] if time_to_norm is not None else raw_ts)
        snapshots.append(
            (
                np.ascontiguousarray(events4[start:end, :3], dtype=np.int64),
                t_norm,
                int(raw_ts),
            )
        )
        start = end
    return snapshots


def _selected_train_suffix(train_list, ratio):
    if ratio <= 0:
        return len(train_list), []
    count = int(len(train_list) * float(ratio))
    start = len(train_list) - count
    return start, train_list[start:]


def _events4_from_snapshots(snapshot_list):
    chunks = []
    for events, _, t_orig in snapshot_list:
        if len(events) == 0:
            continue
        t_col = np.full((len(events), 1), int(t_orig), dtype=np.int64)
        chunks.append(np.hstack((events.astype(np.int64, copy=False), t_col)))
    if not chunks:
        return np.empty((0, 4), dtype=np.int64)
    return np.vstack(chunks)


def _positive_maps(events4):
    by_query = {}
    by_time_rel = {}
    for s, r, o, t in events4.astype(np.int64, copy=False):
        by_query.setdefault((int(t), int(s), int(r)), set()).add(int(o))
        by_time_rel.setdefault((int(t), int(r)), set()).add(int(o))
    by_query = {k: np.array(sorted(v), dtype=np.int64) for k, v in by_query.items()}
    by_time_rel = {k: np.array(sorted(v), dtype=np.int64) for k, v in by_time_rel.items()}
    return by_query, by_time_rel


def _cache_path(root, dataset_name, mode, q, seed, load_train_ratio):
    ratio = ratio_path_token(load_train_ratio)
    name = f"{dataset_name}_{mode}_nsq{int(q)}_nsseed{int(seed)}_tpr{ratio}_same_t_rel.pkl"
    return osp.join(root, name)


def _thg_cache_path(root, dataset_name, mode, q, seed, load_train_ratio):
    ratio = f"{float(load_train_ratio):.8g}"
    name = f"{dataset_name}_{mode}_nsq{int(q)}_seed{int(seed)}_trainratio{ratio}_business_reject.pkl"
    return osp.join(root, name)


def _load_or_create_full_negatives(snapshot_list, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading full negatives: {cache_path}", flush=True)
        payload = _load_pkl(cache_path)
        if isinstance(payload, dict) and payload.get("__format__") == "thg_full_negatives":
            return payload["query_pos"]
        return payload

    query_pos, _ = _positive_maps(_events4_from_snapshots(snapshot_list))
    _save_pkl(
        cache_path,
        {
            "__format__": "thg_full_negatives",
            "description": "Compact cache for ns_q=-1: negatives are all business ids except positives for the same (ts, head, rel).",
            "query_pos": query_pos,
        },
    )
    print(f"[data] saved full negatives: {cache_path}", flush=True)
    return query_pos


def _available_negative_stats(snapshot_list, all_dst):
    query_pos, _ = _positive_maps(_events4_from_snapshots(snapshot_list))
    total_dst = int(len(all_dst))
    available = []
    for positives in query_pos.values():
        positive_count = int(len(positives))
        available.append(max(total_dst - positive_count, 0))
    if not available:
        return query_pos, {"queries": 0, "min_available": total_dst, "available": np.empty(0, dtype=np.int64)}
    arr = np.asarray(available, dtype=np.int64)
    return query_pos, {"queries": int(len(arr)), "min_available": int(arr.min()), "available": arr}


def _sample_without_replace(rng, pool, count):
    if count <= 0 or len(pool) == 0:
        return np.empty(0, dtype=np.int64)
    if len(pool) <= count:
        return np.asarray(pool, dtype=np.int64).copy()
    return rng.choice(pool, count, replace=False)


def _sample_negatives(snapshot_list, all_dst, q, seed, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading negatives: {cache_path}", flush=True)
        return _load_pkl(cache_path)

    rng = np.random.RandomState(seed)
    query_pos, time_rel_pos = _positive_maps(_events4_from_snapshots(snapshot_list))
    sampled = {}
    pools = {}
    pool_time = None
    started = last_log = time.perf_counter()
    for index, (key, positives) in enumerate(query_pos.items(), 1):
        t, _, r = key
        same_time_rel = time_rel_pos.get((t, r), positives)
        if t != pool_time:
            pools.clear()
            pool_time = t
        if r not in pools:
            pools[r] = np.setdiff1d(all_dst, same_time_rel, assume_unique=False)
        primary_pool = pools[r]
        parts = [_sample_without_replace(rng, primary_pool, int(q))]
        current = sum(len(part) for part in parts)

        if current < int(q):
            fill_pool = np.setdiff1d(same_time_rel, positives, assume_unique=False)
            parts.append(_sample_without_replace(rng, fill_pool, int(q) - current))
            current = sum(len(part) for part in parts)

        if current < int(q):
            fallback_pool = np.setdiff1d(all_dst, positives, assume_unique=False)
            need = int(q) - current
            if len(fallback_pool) == 0:
                fallback_pool = all_dst
            parts.append(rng.choice(fallback_pool, need, replace=len(fallback_pool) < need))

        sampled[key] = np.concatenate(parts)[: int(q)].astype(np.int32, copy=False)
        now = time.perf_counter()
        if now - last_log >= 60.0 or index == len(query_pos):
            print(
                f"[data] negatives {index}/{len(query_pos)} queries "
                f"elapsed={now - started:.1f}s -> {cache_path}", flush=True,
            )
            last_log = now

    print(f"[data] saving negatives: {cache_path}", flush=True)
    _save_pkl(cache_path, sampled)
    print(f"[data] saved negatives: {cache_path}", flush=True)
    return sampled


def _sample_negatives_rejection(snapshot_list, all_dst, q, seed, cache_path):
    if osp.exists(cache_path):
        print(f"[data] loading negatives: {cache_path}", flush=True)
        return _load_pkl(cache_path)

    rng = np.random.RandomState(seed)
    query_pos, time_rel_pos = _positive_maps(_events4_from_snapshots(snapshot_list))
    all_dst = np.asarray(all_dst, dtype=np.int64)
    sampled = {}
    for key, positives in query_pos.items():
        t, _, r = key
        positives_set = set(int(x) for x in positives)
        same_time_rel = time_rel_pos.get((t, r), positives)
        primary_exclude = set(int(x) for x in same_time_rel)
        selected = []
        seen = set()
        attempts = 0
        while len(selected) < int(q) and attempts < max(64, int(q) * 8):
            need = int(q) - len(selected)
            draw = all_dst[rng.randint(0, len(all_dst), size=max(need * 2, 16))]
            for value in draw:
                value = int(value)
                if value in primary_exclude or value in seen:
                    continue
                selected.append(value)
                seen.add(value)
                if len(selected) == int(q):
                    break
            attempts += len(draw)
        if len(selected) < int(q):
            for value in all_dst:
                value = int(value)
                if value in primary_exclude or value in seen:
                    continue
                selected.append(value)
                seen.add(value)
                if len(selected) == int(q):
                    break
        if len(selected) < int(q):
            fill = [int(x) for x in same_time_rel if int(x) not in positives_set and int(x) not in seen]
            if fill:
                take = min(int(q) - len(selected), len(fill))
                extra = _sample_without_replace(rng, np.asarray(fill, dtype=np.int64), take)
                selected.extend(int(x) for x in extra)
                seen.update(int(x) for x in extra)
        if len(selected) < int(q):
            fallback = [int(x) for x in all_dst if int(x) not in positives_set and int(x) not in seen]
            if fallback:
                take = min(int(q) - len(selected), len(fallback))
                extra = _sample_without_replace(rng, np.asarray(fallback, dtype=np.int64), take)
                selected.extend(int(x) for x in extra)
        sampled[key] = np.asarray(selected[: int(q)], dtype=np.int32)

    _save_pkl(cache_path, sampled)
    print(f"[data] saved negatives: {cache_path}", flush=True)
    return sampled


class NegativeSampler:
    def __init__(self, num_nodes, first_dst_id=0, last_dst_id=None, base_sampler=None):
        self.base_sampler = base_sampler
        self.eval_set = {}
        self.full_modes = {}
        self.lazy_modes = {}
        self.strategy = "dst-time-filtered"
        self.first_dst_id = int(first_dst_id)
        self.last_dst_id = int(num_nodes - 1 if last_dst_id is None else last_dst_id)
        self.all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1, dtype=np.int64)

    def add_full_mode(self, mode, snapshot_list=None, positives=None):
        if positives is None:
            positives = _positive_maps(_events4_from_snapshots(snapshot_list))[0]
        self.full_modes[mode] = positives

    def add_sampled_mode(self, mode, sampled):
        self.eval_set[mode] = sampled

    def prepare_mode(self, mode):
        if mode in self.lazy_modes and mode not in self.eval_set:
            for previous in self.lazy_modes:
                self.release_mode(previous)
            self.lazy_modes[mode]()

    def release_mode(self, mode):
        if mode in self.lazy_modes:
            self.eval_set.pop(mode, None)

    def query_batch(self, sources, destinations, timestamps, relations, mode):
        self.prepare_mode(mode)
        if mode in self.full_modes:
            positives = self.full_modes[mode]
            return [
                np.setdiff1d(
                    self.all_dst,
                    positives.get((int(t), int(s), int(r)), np.array([int(o)], dtype=np.int64)),
                    assume_unique=False,
                ).astype(np.int32, copy=False)
                for s, o, t, r in zip(sources, destinations, timestamps, relations)
            ]

        if mode in self.eval_set:
            data = self.eval_set[mode]
            empty = np.empty(0, dtype=np.int32)
            return [data.get((int(t), int(s), int(r)), empty) for s, t, r in zip(sources, timestamps, relations)]

        if self.base_sampler is not None:
            return self.base_sampler.query_batch(sources, destinations, timestamps, relations, mode)

        return [np.empty(0, dtype=np.int32) for _ in sources]


def _attach_negatives(
    sampler,
    root,
    dataset_name,
    mode,
    snapshot_list,
    q,
    seed,
    load_train_ratio,
    lazy=False,
):
    if not snapshot_list:
        return
    if q == -1:
        sampler.add_full_mode(mode, snapshot_list)
        return
    path = _cache_path(root, dataset_name, mode, q, seed, load_train_ratio)
    if lazy:
        sampler.lazy_modes[mode] = lambda: _attach_negatives(
            sampler, root, dataset_name, mode, snapshot_list, q, seed, load_train_ratio
        )
        print(f"[data] deferred {mode} negatives: {path}", flush=True)
        return
    sampled = _sample_negatives(snapshot_list, sampler.all_dst, q, seed, path)
    sampler.add_sampled_mode(mode, sampled)


def _attach_thg_negatives(
    sampler,
    root,
    dataset_name,
    mode,
    snapshot_list,
    q,
    seed,
    load_train_ratio,
):
    if not snapshot_list:
        return
    full_path = _thg_cache_path(root, dataset_name, mode, -1, seed, load_train_ratio)
    if q == -1:
        positives = _load_or_create_full_negatives(snapshot_list, full_path)
        sampler.add_full_mode(mode, positives=positives)
        return
    _, stats = _available_negative_stats(snapshot_list, sampler.all_dst)
    if int(stats["min_available"]) < int(q):
        below = int(np.sum(stats["available"] < int(q)))
        print(
            f"[data] {dataset_name} {mode}: {below}/{stats['queries']} queries have fewer than "
            f"ns_q={int(q)} available business negatives (min_available={stats['min_available']}); "
            "falling back to ns_q=-1/full business negatives for this split.",
            flush=True,
        )
        positives = _load_or_create_full_negatives(snapshot_list, full_path)
        sampler.add_full_mode(mode, positives=positives)
        return
    path = _thg_cache_path(root, dataset_name, mode, q, seed, load_train_ratio)
    sampled = _sample_negatives_rejection(snapshot_list, sampler.all_dst, q, seed, path)
    sampler.add_sampled_mode(mode, sampled)


def default_ns_q(dataset_name):
    return 1000 if dataset_name == "ICEWS14" or dataset_name in THG_DATASETS else 5000


def load_datasets(
    dataset_name, q=None, load_train_ratio=0.0, load_eval_neg=True,
    ns_seed=1, data_root=None, lazy_negatives=False,
):
    q = default_ns_q(dataset_name) if q is None else q
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    if q == 0 or q < -1:
        raise ValueError("q must be -1 or a positive integer")
    if not 0.0 <= float(load_train_ratio) <= 1.0:
        raise ValueError("load_train_ratio must be in [0, 1]")
    if dataset_name in THG_DATASETS:
        return load_dataset_thg(dataset_name, q, load_train_ratio, load_eval_neg, ns_seed, data_root)
    if dataset_name in TGB_DATASETS:
        return load_dataset_tgb(
            dataset_name, q, load_train_ratio, load_eval_neg, ns_seed, data_root, lazy_negatives
        )
    return load_dataset_tkg(dataset_name, q, load_train_ratio, load_eval_neg, ns_seed, data_root)


def load_dataset_tgb(
    dataset_name, q=None, load_train_ratio=0.0, load_eval_neg=True,
    ns_seed=1, data_root=None, lazy_negatives=False,
):
    q = default_ns_q(dataset_name) if q is None else q
    from tgb.linkproppred.dataset import LinkPropPredDataset

    # TGB joins dir_name to its package root; an absolute directory keeps data here.
    dataset = LinkPropPredDataset(
        name=dataset_name, root="",
        meta_dict={"dir_name": str(data_directory(dataset_name, data_root))}, preprocess=True,
    )
    num_nodes = int(dataset.num_nodes)
    num_rels = int(dataset.num_rels)
    num_rels_raw = num_rels // 2

    ts_orig = dataset.full_data["timestamps"]
    ts_norm = _reformat_ts(ts_orig, dataset_name)
    triples = np.stack((dataset.full_data["sources"], dataset.edge_type, dataset.full_data["destinations"]), axis=1)

    train_list = _split_by_time(triples[dataset.train_mask], ts_norm[dataset.train_mask], ts_orig[dataset.train_mask])
    val_list = _split_by_time(triples[dataset.val_mask], ts_norm[dataset.val_mask], ts_orig[dataset.val_mask])
    test_list = _split_by_time(triples[dataset.test_mask], ts_norm[dataset.test_mask], ts_orig[dataset.test_mask])

    base_sampler = dataset.negative_sampler
    if load_eval_neg and q == -1:
        dataset.load_val_ns()
        dataset.load_test_ns()
        base_sampler = dataset.negative_sampler

    first = getattr(base_sampler, "first_dst_id", 0)
    last = getattr(base_sampler, "last_dst_id", num_nodes - 1)
    sampler = NegativeSampler(num_nodes, first, last, base_sampler=base_sampler if q == -1 else None)

    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_negatives(sampler, dataset.root, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio, lazy=lazy_negatives)

    if load_eval_neg and q > 0:
        _attach_negatives(sampler, dataset.root, dataset_name, "val", val_list, q, ns_seed, load_train_ratio, lazy=lazy_negatives)
        _attach_negatives(sampler, dataset.root, dataset_name, "test", test_list, q, ns_seed, load_train_ratio, lazy=lazy_negatives)

    # TGB exposes dataset.negative_sampler as a read-only property in some versions.
    # Keep EAGLE's train/val/test sampler in the returned data dict instead.
    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": True,
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_rels_raw": num_rels_raw,
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(np.max(ts_norm)),
    }


def _read_thg_events(dataset_dir, split):
    path = osp.join(dataset_dir, f"{split}_events.csv")
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["user_id"]),
                    int(row["rel_id"]),
                    int(row["business_id"]),
                    int(row["ts"]),
                )
            )
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    arr = np.asarray(rows, dtype=np.int64)
    if len(arr) > 1 and np.any(arr[1:, 3] < arr[:-1, 3]):
        arr = arr[np.argsort(arr[:, 3], kind="stable")]
    return np.ascontiguousarray(arr, dtype=np.int64)


def _read_thg_static_edges(dataset_dir):
    path = osp.join(dataset_dir, "static_user_friend_edges.csv")
    if not osp.isfile(path):
        return np.empty((0, 2), dtype=np.int64)
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["user_id"]), int(row["friend_user_id"])))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _read_thg_business_geo(dataset_dir):
    path = osp.join(dataset_dir, "business_id_map.csv")
    ids, lats, lons = [], [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["business_id"]))
            lats.append(float(row["latitude"]))
            lons.append(float(row["longitude"]))
    if not ids:
        return {
            "business_ids": np.empty(0, dtype=np.int64),
            "latitude": np.empty(0, dtype=np.float32),
            "longitude": np.empty(0, dtype=np.float32),
        }
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    return {
        "business_ids": np.asarray(ids, dtype=np.int64)[order],
        "latitude": np.asarray(lats, dtype=np.float32)[order],
        "longitude": np.asarray(lons, dtype=np.float32)[order],
    }


def _read_thg_stats(dataset_dir):
    with open(osp.join(dataset_dir, "model_input_stats.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_thg(dataset_name, q=None, load_train_ratio=0.0, load_eval_neg=True, ns_seed=1, data_root=None):
    q = default_ns_q(dataset_name) if q is None else q
    if dataset_name not in THG_DATASETS:
        raise ValueError(f"unknown THG dataset: {dataset_name}")

    dataset_dir = str(data_directory(dataset_name, data_root))
    stats = _read_thg_stats(dataset_dir)
    num_users = int(stats["num_users"])
    num_businesses = int(stats["num_businesses"])
    num_nodes = int(stats["num_entities_user_business_disjoint"])
    num_rels = int(stats["num_relations"])
    business_first = int(stats["business_id_range"][0])
    business_last = int(stats["business_id_range"][1])

    train_raw = _read_thg_events(dataset_dir, "train")
    val_raw = _read_thg_events(dataset_dir, "valid")
    test_raw = _read_thg_events(dataset_dir, "test")
    all_times = np.array(
        sorted(np.unique(np.concatenate((train_raw[:, 3], val_raw[:, 3], test_raw[:, 3])))),
        dtype=np.int64,
    )
    time_to_norm = {int(t): i for i, t in enumerate(all_times)}

    train_list = _split_events4_by_time(train_raw, time_to_norm)
    val_list = _split_events4_by_time(val_raw, time_to_norm)
    test_list = _split_events4_by_time(test_raw, time_to_norm)

    sampler = NegativeSampler(num_nodes, first_dst_id=business_first, last_dst_id=business_last)
    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio)
    if load_eval_neg:
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "val", val_list, q, ns_seed, load_train_ratio)
        _attach_thg_negatives(sampler, dataset_dir, dataset_name, "test", test_list, q, ns_seed, load_train_ratio)

    all_events = _events4_from_snapshots(train_list + val_list + test_list)
    static_friend_edges = _read_thg_static_edges(dataset_dir)
    business_geo = _read_thg_business_geo(dataset_dir)
    dataset = SimpleNamespace(
        name=dataset_name,
        root=dataset_dir,
        num_nodes=num_nodes,
        num_rels=num_rels,
        num_rels_raw=num_rels,
        num_users=num_users,
        num_businesses=num_businesses,
        business_first_id=business_first,
        business_last_id=business_last,
        negative_sampler=sampler,
        full_data={
            "sources": all_events[:, 0],
            "edge_type": all_events[:, 1],
            "destinations": all_events[:, 2],
            "timestamps": all_events[:, 3],
            "edge_idxs": np.arange(len(all_events), dtype=np.int64),
        },
    )

    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": False,
        "is_thg": True,
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "num_rels_raw": num_rels,
        "num_users": num_users,
        "num_businesses": num_businesses,
        "business_first_id": business_first,
        "business_last_id": business_last,
        "business_ids": business_geo["business_ids"],
        "business_latitude": business_geo["latitude"],
        "business_longitude": business_geo["longitude"],
        "static_user_friend_edges": static_friend_edges,
        "relation2id": dict(stats.get("relation2id", {})),
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(len(all_times) - 1),
        "raw_timestamps": all_times,
        "raw_ts_to_cont": time_to_norm,
        "root": dataset_dir,
    }


def _read_tkg_stat(dataset_dir):
    with open(osp.join(dataset_dir, "stat.txt"), "r") as f:
        cols = f.readline().split()
    return int(cols[0]), int(cols[1])


def _read_tkg_split(dataset_dir, split):
    path = osp.join(dataset_dir, f"{split}.txt")
    arr = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2, 3), ndmin=2)
    if len(arr) > 1 and np.any(arr[1:, 3] < arr[:-1, 3]):
        arr = arr[np.argsort(arr[:, 3], kind="stable")]
    return np.ascontiguousarray(arr, dtype=np.int64)


def _augment_tkg_snapshots(events4, time_to_norm, num_rels_raw):
    snapshots = []
    for t in sorted(np.unique(events4[:, 3])):
        chunk = events4[events4[:, 3] == t]
        forward = chunk[:, [0, 1, 2]]
        inverse = np.stack((chunk[:, 2], chunk[:, 1] + num_rels_raw, chunk[:, 0]), axis=1)
        snapshots.append((np.vstack((forward, inverse)).astype(np.int64, copy=False), int(time_to_norm[int(t)]), int(t)))
    return snapshots


def load_dataset_tkg(dataset_name, q=None, load_train_ratio=0.0, load_eval_neg=True, ns_seed=1, data_root=None):
    q = default_ns_q(dataset_name) if q is None else q
    if dataset_name not in NOT_TGB_DATASETS:
        raise ValueError(f"unknown non-TGB dataset: {dataset_name}")

    dataset_dir = str(data_directory(dataset_name, data_root))
    num_nodes, num_rels_raw = _read_tkg_stat(dataset_dir)

    train_raw = _read_tkg_split(dataset_dir, "train")
    val_raw = _read_tkg_split(dataset_dir, "valid")
    test_raw = _read_tkg_split(dataset_dir, "test")

    all_times = np.array(sorted(np.unique(np.concatenate((train_raw[:, 3], val_raw[:, 3], test_raw[:, 3])))), dtype=np.int64)
    time_to_norm = {int(t): i for i, t in enumerate(all_times)}

    train_list = _augment_tkg_snapshots(train_raw, time_to_norm, num_rels_raw)
    val_list = _augment_tkg_snapshots(val_raw, time_to_norm, num_rels_raw)
    test_list = _augment_tkg_snapshots(test_raw, time_to_norm, num_rels_raw)

    sampler = NegativeSampler(num_nodes)
    train_predict_start, train_predict_list = _selected_train_suffix(train_list, load_train_ratio)
    if train_predict_list:
        _attach_negatives(sampler, dataset_dir, dataset_name, "train", train_predict_list, q, ns_seed, load_train_ratio)
    if load_eval_neg:
        _attach_negatives(sampler, dataset_dir, dataset_name, "val", val_list, q, ns_seed, load_train_ratio)
        _attach_negatives(sampler, dataset_dir, dataset_name, "test", test_list, q, ns_seed, load_train_ratio)

    all_aug = _events4_from_snapshots(train_list + val_list + test_list)
    dataset = SimpleNamespace(
        name=dataset_name,
        root=dataset_dir,
        num_nodes=int(num_nodes),
        num_rels=int(num_rels_raw * 2),
        num_rels_raw=int(num_rels_raw),
        edge_feat=None,
        node_feat=None,
        negative_sampler=sampler,
        full_data={
            "sources": all_aug[:, 0],
            "edge_type": all_aug[:, 1],
            "destinations": all_aug[:, 2],
            "timestamps": all_aug[:, 3],
            "edge_idxs": np.arange(len(all_aug), dtype=np.int64),
        },
    )

    return {
        "dataset": dataset,
        "negative_sampler": sampler,
        "name": dataset_name,
        "is_tgb": False,
        "num_nodes": int(num_nodes),
        "num_rels": int(num_rels_raw * 2),
        "num_rels_raw": int(num_rels_raw),
        "train_list": train_list,
        "val_list": val_list,
        "test_list": test_list,
        "train_predict_start_idx": train_predict_start,
        "train_predict_count": len(train_predict_list),
        "timestamps_norm_max": int(len(all_times) - 1),
    }


def collect_eval_batch(events, ts_orig, neg_sampler, mode, batch_size):
    for start in range(0, len(events), batch_size):
        batch = np.ascontiguousarray(events[start : start + batch_size], dtype=np.int64)
        ts = np.full(len(batch), int(ts_orig), dtype=np.int64)
        negs = neg_sampler.query_batch(batch[:, 0], batch[:, 2], ts, batch[:, 1], mode)
        lengths = np.fromiter((len(row) for row in negs), dtype=np.int64, count=len(negs))
        width = int(lengths.max()) if len(lengths) else 0
        if width and np.all(lengths == width):
            neg_arr = np.ascontiguousarray(np.stack(negs), dtype=np.int64)
            neg_mask = np.broadcast_to(np.bool_(True), neg_arr.shape)
        else:
            neg_arr = np.full((len(batch), width or 1), -1, dtype=np.int64)
            for i, row in enumerate(negs):
                if len(row):
                    neg_arr[i, : len(row)] = row
            neg_mask = neg_arr != -1
        yield batch, neg_arr, neg_mask


def compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask, hit_ks=HIT_KS):
    ranks = 1 + np.sum((np.asarray(neg_scores) >= np.asarray(pos_scores).reshape(-1, 1)) & neg_mask, axis=1)
    return {"count": len(ranks), "mrr_strict": float(np.sum(1.0 / ranks)),
            **{f"hit@{k}_strict": float(np.sum(ranks <= k)) for k in hit_ks}}


def add_metric_sums(total, batch):
    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric_sums(sums, hit_ks=HIT_KS):
    count = sums.get("count", 0)
    keys = ["mrr_strict", *(f"hit@{k}_strict" for k in hit_ks)]
    return {key: float(sums.get(key, 0.0)) / count if count else 0.0 for key in keys}


def dense_rank(scores, valid):
    """Descending strict/max rank with equal scores assigned equal ranks."""

    masked = np.where(valid, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(masked, order, axis=1)
    width = int(order.shape[1])
    positions = np.arange(1, width + 1, dtype=np.int32)[None, :]
    group_end = np.ones(sorted_scores.shape, dtype=bool)
    if width > 1:
        group_end[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    markers = np.where(group_end, positions, width + 1)
    sorted_ranks = np.minimum.accumulate(markers[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty(order.shape, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = sorted_ranks
    return np.where(valid, ranks, width + 1).astype(np.int32, copy=False)


class ScoreWriter:
    def __init__(self, out_dir, mode):
        self.out_dir = out_dir
        self.mode = mode
        self.pos_chunks = []
        self.neg_chunks = []
        self.valid_lens = []
        self.max_negs = 0
        self.num_rows = 0
        os.makedirs(out_dir, exist_ok=True)

    def write_batch(self, pos_scores, neg_scores, neg_mask):
        csr_matrix, _, _, _ = _sparse_tools()
        pos = np.asarray(pos_scores, dtype=np.float32).reshape(-1, 1)
        mask = np.asarray(neg_mask, dtype=bool)
        all_valid = bool(np.all(mask))
        if all_valid:
            neg = np.asarray(neg_scores, dtype=np.float32)
        else:
            neg = np.where(mask, neg_scores, 0.0).astype(np.float32, copy=False)
        # Preserve equal scores independently of whether a candidate is positive.
        # Sparse storage already omits exact zeros; thresholding only negatives
        # can manufacture a positive's rank advantage for tiny equal scores.
        valid_neg = neg if all_valid else neg[mask]
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(valid_neg)):
            raise ValueError("cannot save non-finite valid candidate scores")
        self.pos_chunks.append(pos)
        self.neg_chunks.append(csr_matrix(neg))
        if all_valid:
            self.valid_lens.append(np.full(pos.shape[0], neg.shape[1], dtype=np.int32))
        else:
            self.valid_lens.append(np.sum(mask, axis=1).astype(np.int32))
        self.max_negs = max(self.max_negs, neg.shape[1])
        self.num_rows += pos.shape[0]

    def close(self):
        if self.num_rows == 0:
            return
        csr_matrix, save_npz, _, sparse_vstack = _sparse_tools()
        chunks = []
        for chunk in self.neg_chunks:
            if chunk.shape[1] == self.max_negs:
                chunks.append(chunk)
            else:
                chunks.append(csr_matrix((chunk.data, chunk.indices, chunk.indptr), shape=(chunk.shape[0], self.max_negs)))
        np.save(osp.join(self.out_dir, f"{self.mode}_pos.npy"), np.vstack(self.pos_chunks))
        np.save(osp.join(self.out_dir, f"{self.mode}_valid_lens.npy"), np.concatenate(self.valid_lens))
        save_npz(osp.join(self.out_dir, f"{self.mode}_neg.npz"), sparse_vstack(chunks, format="csr"), compressed=False)
        meta = {
            "format": "stream_sparse",
            "score_transform": "float32_no_threshold",
            "num_rows": int(self.num_rows),
            "max_negs": int(self.max_negs),
            "neg_default_kind": "zero",
            "neg_default_scalar": 0.0,
        }
        with open(osp.join(self.out_dir, f"{self.mode}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def _mmap_sparse_npz(path):
    from scipy.sparse import csr_matrix

    arrays = {}
    with zipfile.ZipFile(path) as archive, open(path, 'rb') as source:
        if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
            return None
        for name in ('data', 'indices', 'indptr', 'shape'):
            info = archive.getinfo(name + '.npy')
            source.seek(info.header_offset + 26)
            name_len, extra_len = struct.unpack('<HH', source.read(4))
            source.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(source)
            shape, fortran, dtype = np.lib.format._read_array_header(source, version)
            arrays[name] = np.memmap(path, mode='r', dtype=dtype, shape=shape,
                                     offset=source.tell(), order='F' if fortran else 'C')
    # Keep stored indices; the CSR constructor otherwise scans the full archive.
    matrix = csr_matrix((0, 0), dtype=arrays['data'].dtype)
    matrix.data, matrix.indices, matrix.indptr = arrays['data'], arrays['indices'], arrays['indptr']
    matrix._shape = tuple(arrays['shape'])
    matrix.check_format(full_check=False)
    return matrix


class ScoreStore:
    def __init__(self, out_dir, mode, mmap=False):
        _, _, load_npz, _ = _sparse_tools()
        self.out_dir = out_dir
        self.mode = mode
        self.pos = np.load(osp.join(out_dir, f"{mode}_pos.npy"))
        if self.pos.ndim == 1:
            self.pos = self.pos.reshape(-1, 1)
        path = osp.join(out_dir, f"{mode}_neg.npz")
        self.neg = _mmap_sparse_npz(path) if mmap else None
        if self.neg is None:
            self.neg = load_npz(path).tocsr()
        self.valid_lens = np.load(osp.join(out_dir, f"{mode}_valid_lens.npy")).astype(np.int32)
        self.num_rows = int(self.pos.shape[0])
        self.max_negs = int(self.neg.shape[1])

    def get_block(self, start, end, width):
        if end > self.num_rows:
            raise ValueError(f"score row overflow: {self.out_dir}/{self.mode}, end={end}, rows={self.num_rows}")
        width = min(int(width), self.max_negs)
        neg = self.neg[start:end, :width]
        if neg.dtype != np.float32:
            neg = neg.astype(np.float32)
        neg = neg.toarray().astype(np.float32, copy=False)
        lens = np.minimum(self.valid_lens[start:end], width)
        mask = np.arange(width)[None, :] < lens[:, None]
        return (
            self.pos[start:end].astype(np.float32, copy=False),
            np.where(mask, neg, 0.0).astype(np.float32, copy=False),
            mask,
        )


def describe_loaded_data(data, prefix="[data]"):
    def stats(name):
        snaps = data[name]
        return len(snaps), sum(len(events) for events, _, _ in snaps)

    train = stats("train_list")
    val = stats("val_list")
    test = stats("test_list")
    print(
        f"{prefix} {data['name']}: train={train[0]} ts/{train[1]} events, "
        f"val={val[0]} ts/{val[1]} events, test={test[0]} ts/{test[1]} events, "
        f"nodes={data['num_nodes']}, rels={data['num_rels']} raw_rels={data['num_rels_raw']}",
        flush=True,
    )


def get_negative_sampler(data):
    if "negative_sampler" in data:
        return data["negative_sampler"]
    return data["dataset"].negative_sampler


def get_destination_pool(data, num_nodes):
    sampler = get_negative_sampler(data)
    if hasattr(sampler, "first_dst_id") and hasattr(sampler, "last_dst_id"):
        first_dst = int(sampler.first_dst_id)
        last_dst = int(sampler.last_dst_id)
        if first_dst <= last_dst:
            return np.arange(first_dst, last_dst + 1, dtype=np.int64)
    return np.arange(int(num_nodes), dtype=np.int64)


def ratio_path_token(value):
    """Return a stable, shell-friendly token for a split ratio."""

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"ratio must be finite, got {value!r}")
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_scope_name(ns_seed, train_predict_ratio):
    """Canonical directory segment for negative sampling and train split."""

    return f"nsseed{int(ns_seed)}_tpr{ratio_path_token(train_predict_ratio)}"


def scoped_run_dir(prefix, dataset, seed, ns_seed, train_predict_ratio):
    return osp.join(
        prefix,
        dataset,
        f"seed{int(seed)}",
        run_scope_name(ns_seed, train_predict_ratio),
    )


def make_dir_name(prefix, dataset, seed, **kwargs):
    ns_seed = kwargs.pop("ns_seed", None)
    train_predict_ratio = kwargs.pop("train_predict_ratio", None)
    parts = []
    for key in sorted(kwargs):
        value = kwargs[key]
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}={value}")
    digest = hashlib.sha1("_".join(parts).encode("utf-8")).hexdigest()[:12]
    name = (
        f"ppr{kwargs.get('ppr_k', 'na')}"
        f"_a{kwargs.get('ppr_a', 'na')}"
        f"_b{kwargs.get('ppr_b', 'na')}"
        f"_{digest}"
    )
    if (ns_seed is None) != (train_predict_ratio is None):
        raise ValueError("ns_seed and train_predict_ratio must be scoped together")
    if ns_seed is None:
        root = osp.join(prefix, dataset, f"seed{seed}")
    else:
        root = scoped_run_dir(prefix, dataset, seed, ns_seed, train_predict_ratio)
    return osp.join(root, name)


def save_config(out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_metrics(out_dir, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(out_dir):
    with open(osp.join(out_dir, "metrics.json"), "r") as f:
        return json.load(f)


def score_store_complete(out_dir, split, expected_rows=None):
    paths = {suffix: osp.join(out_dir, f"{split}_{suffix}")
             for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json")}
    if not all(osp.isfile(path) for path in paths.values()):
        return False
    try:
        with open(paths["meta.json"], encoding="utf-8") as file:
            meta = json.load(file)
        pos = np.load(paths["pos.npy"], mmap_mode="r")
        lengths = np.load(paths["valid_lens.npy"], mmap_mode="r")
        with np.load(paths["neg.npz"]) as archive:
            shape = tuple(archive["shape"])
        return (meta["score_transform"] == "float32_no_threshold"
                and shape == (meta["num_rows"], meta["max_negs"])
                and pos.shape in ((shape[0],), (shape[0], 1))
                and lengths.shape == (shape[0],)
                and (expected_rows is None or shape[0] == expected_rows)
                and np.all((lengths >= 0) & (lengths <= shape[1])))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def is_run_complete(out_dir, modes=("val", "test")):
    return (osp.isfile(osp.join(out_dir, "config.json"))
            and osp.isfile(osp.join(out_dir, "metrics.json"))
            and all(score_store_complete(out_dir, split) for split in modes))


def set_compute_threads(count):
    os.environ["OMP_NUM_THREADS"] = str(count)
    os.environ["MKL_NUM_THREADS"] = str(count)
    torch.set_num_threads(int(count))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class RankingAccumulator:
    def __init__(self):
        self.sums = {}

    def add(self, pos, neg, mask):
        add_metric_sums(self.sums, compute_ranking_metric_sums(pos, neg, mask))

    def finish(self):
        return finalize_metric_sums(self.sums)


def score_store_metrics(out_dir, split, batch_rows=2048):
    store = ScoreStore(out_dir, split)
    recorder = RankingAccumulator()
    for start in range(0, store.num_rows, batch_rows):
        recorder.add(*store.get_block(start, min(start + batch_rows, store.num_rows), store.max_negs))
    return recorder.finish()


class RunMetrics:
    """Phase wall time and peak process RSS."""

    def __init__(self, method, args, interval=0.1):
        self.method = method
        self.args = args
        self.process = psutil.Process()
        self.phases = {}
        self.active = []
        self.stop_event = threading.Event()
        self.interval = interval
        self.worker = threading.Thread(target=self._sample_loop, daemon=True)
        self.worker.start()

    def _sample(self):
        rss = self.process.memory_info().rss
        for record in self.active:
            record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)

    def _sample_loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    @contextmanager
    def phase(self, name, device=None):
        record = {"peak_rss_bytes": self.process.memory_info().rss}
        self.active.append(record)
        started = time.perf_counter()
        try:
            yield record
        finally:
            self._sample()
            elapsed = time.perf_counter() - started
            self.active = [item for item in self.active if item is not record]
            total = self.phases.setdefault(name, {"wall_seconds": 0.0, "peak_rss_bytes": 0})
            total["wall_seconds"] += elapsed
            total["peak_rss_bytes"] = max(total["peak_rss_bytes"], record["peak_rss_bytes"])

    def finish(self, status="OK"):
        self.stop_event.set()
        self.worker.join()
        return {"parameters": json_safe({k: v for k, v in vars(self.args).items()
                                        if not k.startswith("_")}),
                "phases": self.phases}

    def peak_rss(self, name):
        return self.phases.get(name, {}).get("peak_rss_bytes", 0)


def print_run_metrics(method, metrics):
    metrics = metrics.get("test", metrics)
    print(f"[{method}] MRR={metrics['mrr_strict']:.6f} "
          f"H1={metrics['hit@1_strict']:.6f} H10={metrics['hit@10_strict']:.6f}", flush=True)


def efficiency_metrics(train_seconds, train_bytes, test_seconds, test_bytes, memory_kind):
    return {"train_time_seconds": float(train_seconds),
            "train_memory_bytes": int(train_bytes),
            "test_infer_time_seconds": float(test_seconds),
            "test_infer_memory_bytes": int(test_bytes),
            "memory_kind": memory_kind}


def print_efficiency(method, metrics):
    print(f"[{method}] train_time={metrics['train_time_seconds']:.3f}s "
          f"train_memory={metrics['train_memory_bytes'] / 2**20:.1f}MiB "
          f"test_infer_time={metrics['test_infer_time_seconds']:.3f}s "
          f"test_infer_memory={metrics['test_infer_memory_bytes'] / 2**20:.1f}MiB "
          f"({metrics['memory_kind']})", flush=True)


def load_best_params(dataset, path=None):
    path = path or PROJECT_DIR / "configs" / "best-hyper-params" / f"{dataset}.yaml"
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def apply_best_params(args, section, path=None):
    for key, value in load_best_params(args.dataset, path)[section].items():
        setattr(args, key, value)
    return args


COMMON_PARAMS = dict(use_spatial=True, user_center_half_life_days=730.0)
TIME_PARAMS = dict(spatial_features="country", spatial_gain=0.35, spatial_bottleneck=32,
                   spatial_pair_gain=0.7, spatial_pair_hidden=32, spatial_cache_projection=True)
STRUCTURE_PARAMS = dict(spatial_eta=0.9, spatial_sigma_km=2000.0, spatial_power=1,
                        spatial_scope="hop", predict_spatial_eta=0.9,
                        predict_spatial_sigma_km=1000.0, predict_spatial_power=2,
                        predict_relation_power=1.25, predict_relation_prior=32.0)


def spatial_defaults(dataset, section):
    values = dict(COMMON_PARAMS, **(TIME_PARAMS if section == "time" else STRUCTURE_PARAMS))
    if dataset.startswith("Yelp-"):
        if section == "time":
            values.update(spatial_features="geo")
        if section == "structure":
            values.update(spatial_sigma_km=5.0, predict_spatial_sigma_km=5.0,
                          predict_spatial_power=1, predict_relation_power=0.0)
    return values


def add_spatial_args(parser, section, prefix=""):
    for key, value in spatial_defaults("ICEWS14", section).items():
        kwargs = dict(default=None)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value)
        if key == "spatial_features":
            kwargs["choices"] = ("geo", "country")
        if key == "spatial_scope":
            kwargs["choices"] = ("hop", "source")
        parser.add_argument(f"--{prefix}{key}", **kwargs)


def resolve_spatial_args(args, section):
    for key, value in spatial_defaults(args.dataset, section).items():
        supplied = getattr(args, key, None)
        setattr(args, key, value if supplied is None else type(value)(supplied))
    return args


@lru_cache(maxsize=32)
def _file_hash(path, size, modified):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spatial_signature(args, section):
    resolve_spatial_args(args, section)
    values = {key: getattr(args, key) for key in spatial_defaults(args.dataset, section)}
    if args.use_spatial:
        directory = data_directory(args.dataset, getattr(args, "data_root", None))
        if args.dataset.startswith("Yelp-"):
            paths = [directory / "business_id_map.csv"]
        else:
            paths = [directory / f"{args.dataset}_node_spatial.csv",
                     directory / "country_reference.csv"]
        values["source_sha256"] = []
        for path in paths:
            stat = path.stat()
            values["source_sha256"].append(_file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    values["policy"] = "causal_median" if args.dataset.startswith("Yelp-") else "static_anchors_train_or_past_inference"
    if section == "time" and args.use_spatial and not args.dataset.startswith("Yelp-"):
        values["feature_schema"] = "position_country_distance_actor_pair"
    return values


class SpatialMetadata:
    def __init__(self, root, dataset, num_nodes):
        directory = data_directory(dataset, root)
        with (directory / f"{dataset}_node_spatial.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (directory / "country_reference.csv").open(encoding="utf-8-sig", newline="") as stream:
            countries = list(csv.DictReader(stream))
        if len(rows) != num_nodes or any(int(row["node_id"]) != i for i, row in enumerate(rows)):
            raise ValueError(f"{dataset}: spatial node IDs do not match the dataset")
        self.n = num_nodes
        index = {row["iso3"]: i for i, row in enumerate(countries)}
        lat = np.deg2rad([float(row["latitude"]) for row in countries])
        lon = np.deg2rad([float(row["longitude"]) for row in countries])
        self.country_xyz = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        self.fixed = np.zeros((num_nodes, len(countries)), np.float32)
        self.xyz = np.zeros((num_nodes, 3), np.float64)
        self.known = np.zeros(num_nodes, bool)
        for i, row in enumerate(rows):
            method = row["location_method"]
            # Train-derived tables are not static anchors for prefix/online updates.
            static = method.startswith("entity_label_") or method.startswith("wikidata_")
            if not static or row["has_spatial"] != "1" or "audit" in method:
                continue
            xyz = np.array([float(row[key]) for key in ("unit_x", "unit_y", "unit_z")])
            if not np.isfinite(xyz).all() or np.linalg.norm(xyz) < 1e-8:
                continue
            self.known[i], self.xyz[i] = True, xyz
            codes = [index[iso] for iso in row["country_iso3"].split("|") if iso in index]
            if codes:
                self.fixed[i, codes] = 1. / len(codes)
        labels = [re.match(r"^(.*) \(([^()]+)\)$", row["entity_name"]) for row in rows]
        counts = Counter(match[1] for match in labels if match)
        roles = {role: i + 2 for i, role in enumerate(sorted(role for role, count in counts.items() if count >= 3))}
        self.roles = np.array([roles.get(match[1], 0) if match else
                               (1 if row["location_method"] == "entity_label_country" else 0)
                               for match, row in zip(labels, rows)], np.int64)
        print(f'[Spatial] {dataset} static_coordinates={self.known.sum()}/{num_nodes} '
              f'unknown=infer_from_allowed_events_or_neutral', flush=True)

    def features(self, snapshots, mode):
        s = np.concatenate([events[:, 0] for events, _, _ in snapshots]).astype(np.int32)
        o = np.concatenate([events[:, 2] for events, _, _ in snapshots]).astype(np.int32)
        adjacency = sparse.csr_matrix((np.ones(len(s), np.float32), (s, o)), shape=(self.n, self.n))
        adjacency.data[:] = 1.
        degree = np.asarray(adjacency.sum(1)).ravel().clip(1.)
        adjacency = sparse.diags(1/np.sqrt(degree)) @ adjacency @ sparse.diags(1/np.sqrt(degree))
        adjacency = sparse.diags(1/np.asarray(adjacency.sum(1)).ravel().clip(1e-8)) @ adjacency
        distribution = self.fixed.copy()
        anchored = distribution.sum(1) > 0
        for _ in range(3):
            distribution = adjacency @ distribution
            distribution[anchored] = self.fixed[anchored]
            distribution /= distribution.sum(1, keepdims=True).clip(1e-8)
        xyz = distribution @ self.country_xyz
        norm = np.linalg.norm(xyz, axis=1)
        xyz /= norm[:, None].clip(1e-8)
        xyz[self.known] = self.xyz[self.known]
        known = (norm > 1e-8) | self.known
        confidence = distribution.max(1)
        confidence[self.known] = 1.
        pieces = [xyz, confidence[:, None], known[:, None]]
        if mode == "country":
            pieces.append(distribution)
        features = np.concatenate(pieces, axis=1).astype(np.float32)
        return np.vstack((features, np.zeros((1, features.shape[1]), np.float32))), distribution


class CausalLocations:
    def __init__(self, metadata, raw_relations):
        self.metadata, self.raw_relations = metadata, raw_relations
        self.counts = np.zeros_like(metadata.fixed, dtype=np.float64)
        self.xy = np.column_stack((metadata.xyz.copy(), metadata.known.astype(float)))
        self.known = metadata.known.copy()

    def update(self, events, current_t):
        events = np.asarray(events)
        events = events[events[:, 1] < self.raw_relations]
        s, o = events[:, 0].astype(int), events[:, 2].astype(int)
        np.add.at(self.counts, s, self.metadata.fixed[o])
        np.add.at(self.counts, o, self.metadata.fixed[s])
        ids = np.unique(np.concatenate((s, o)))
        ids = ids[~self.metadata.known[ids]]
        counts = self.counts[ids]
        support = counts.sum(1)
        xyz = counts @ self.metadata.country_xyz
        norm = np.linalg.norm(xyz, axis=1)
        valid = (support > 0) & (norm > 1e-8)
        ids = ids[valid]
        self.xy[ids, :3] = xyz[valid] / norm[valid, None]
        self.xy[ids, 3] = counts[valid].max(1) / support[valid]
        self.known[ids] = True


def weighted_median(values, weights, order):
    positive = order[weights[order] > 0]
    cumulative = np.cumsum(weights[positive])
    half = cumulative[-1] / 2
    i = int(np.searchsorted(cumulative, half))
    value = values[positive[i]]
    if i + 1 < len(positive) and abs(cumulative[i] - half) <= cumulative[-1] * 1e-12:
        value = (value + values[positive[i + 1]]) / 2
    return value


class UserVisits:
    def __init__(self, coordinates, half_life):
        self.coordinates, self.half_life = coordinates, half_life
        self.businesses, self.index = [], {}
        self.weights = np.empty(0, np.float64)
        self.last_t = None

    def add(self, business, current_t):
        if self.last_t is not None and self.half_life > 0:
            self.weights *= np.exp2(-max(0., current_t - self.last_t) / self.half_life)
        self.last_t = current_t
        if business not in self.index:
            self.index[business] = len(self.businesses)
            self.businesses.append(business)
            self.weights = np.append(self.weights, 0.)
            self.values = self.coordinates[self.businesses, :2]
            self.orders = np.argsort(self.values, axis=0, kind="stable")
        self.weights[self.index[business]] += 1.
        return np.array([weighted_median(self.values[:, axis], self.weights, self.orders[:, axis])
                         for axis in range(2)])


class YelpLocations:
    def __init__(self, data, half_life):
        self.data, self.half_life = data, half_life
        self.business_first = int(data.get("business_first_id", np.min(data["business_ids"])))
        coordinates = np.column_stack((data["business_latitude"], data["business_longitude"]))
        valid = np.isfinite(coordinates).all(1)
        if not valid.any():
            raise ValueError('Yelp business coordinates are missing')
        self.center = coordinates[valid].mean(0)
        self.scale = coordinates[valid].std(0).clip(1e-6)
        lat0, lon0 = np.median(coordinates[valid], axis=0)
        self.projection = np.array([np.pi/180*6371.0088, np.pi/180*6371.0088*np.cos(np.deg2rad(lat0))])
        self.origin = np.array([lat0, lon0])
        self.coordinates = np.zeros((data["num_nodes"] + 1, 2), np.float64)
        ids = np.asarray(data["business_ids"])[valid]
        self.coordinates[ids] = coordinates[valid]
        self.known = np.zeros(data["num_nodes"] + 1, bool)
        self.known[ids] = True
        self.xy = np.zeros_like(self.coordinates)
        self.xy[ids] = (coordinates[valid] - self.origin) * self.projection
        self.visits = {}

    def fresh(self):
        return YelpLocations(self.data, self.half_life)

    def update(self, events, current_t):
        for s, _, o in np.asarray(events)[:, :3].astype(int):
            if s >= self.business_first or o < self.business_first or not self.known[o]:
                continue
            if s not in self.visits:
                self.visits[s] = UserVisits(self.coordinates, self.half_life)
            self.coordinates[s] = self.visits[s].add(int(o), current_t)
            self.xy[s] = (self.coordinates[s] - self.origin) * self.projection
            self.known[s] = True

    def features(self, nodes, current_t):
        nodes = np.asarray(nodes, dtype=int)
        features = np.zeros((*nodes.shape, 3), np.float32)
        valid = self.known[nodes]
        features[..., :2] = (self.coordinates[nodes] - self.center) / self.scale * valid[..., None]
        confidence = valid.astype(np.float32)
        for user in np.unique(nodes[nodes < self.business_first]):
            visit = self.visits.get(int(user))
            if visit is None:
                continue
            decay = np.exp2(-max(0., current_t-visit.last_t)/self.half_life) if self.half_life > 0 else 1.
            confidence[nodes == user] = np.log1p(visit.weights.sum() * decay) / 8.
        features[..., 2] = confidence
        return features


def make_locations(data, args):
    if data.get("is_thg", False):
        return YelpLocations(data, args.user_center_half_life_days)
    return CausalLocations(SpatialMetadata(args.data_root, args.dataset, data["num_nodes"]), data["num_rels_raw"])


@njit(cache=True)
def spatial_weight(a, b, xy, known, eta, sigma, power):
    if eta == 1. or not known[a] or not known[b]:
        return 1.
    if xy.shape[1] == 4:
        dx, dy, dz = xy[a, 0]-xy[b, 0], xy[a, 1]-xy[b, 1], xy[a, 2]-xy[b, 2]
        distance2 = (dx*dx+dy*dy+dz*dz) * (6371.0088/sigma) ** 2
        confidence = xy[a, 3] * xy[b, 3]
    else:
        dx, dy = xy[a, 0]-xy[b, 0], xy[a, 1]-xy[b, 1]
        distance2 = (dx*dx+dy*dy) / sigma ** 2
        confidence = 1.
    f = distance2 if power == 2 else np.sqrt(distance2)
    return 1. - confidence * (1.-eta) * (1.-np.exp(-f))


@njit(cache=True, parallel=True)
def prediction_weights(sources, candidates, xy, known, eta, sigma, power):
    weights = np.ones(candidates.shape, np.float32)
    for i in prange(len(sources)):
        for j in range(candidates.shape[1]):
            target = candidates[i, j]
            if target >= 0:
                weights[i, j] = spatial_weight(sources[i], target, xy, known, eta, sigma, power)
    return weights


def fit_relation_space(snapshots, xy, known, num_relations, sigma, power, exponent, prior):
    near, support = np.zeros(num_relations), np.zeros(num_relations)
    for events, _, _ in snapshots:
        s, r, o = events[:, :3].astype(int).T
        confidence = known[s] * known[o] * xy[s, 3] * xy[o, 3]
        distance = np.linalg.norm(xy[s, :3]-xy[o, :3], axis=1) * 6371.0088 / sigma
        near += np.bincount(r, confidence*np.exp(-distance**power), minlength=num_relations)
        support += np.bincount(r, confidence, minlength=num_relations)
    global_near = near.sum() / max(support.sum(), 1e-8)
    rate = np.divide(near+prior*global_near, support+prior,
                     out=np.full(num_relations, global_near), where=support+prior > 0)
    return np.clip(rate, 0., 1.) ** exponent
