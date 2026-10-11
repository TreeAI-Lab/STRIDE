"""Suffix-trained LightGBM reranking of Structure and Time scores."""

import argparse
import gc
import hashlib
import json
import os.path as osp
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import train_structure as structure
import train_time
from train_structure import (
    BConfig,
    COMPONENT_SCORE_NAMES,
    ensure_dir,
    events_for_update,
    init_timeline_history_state,
    minmax_by_query,
    split_snapshots,
    structure_base_scores,
    zscore_by_query,
)
from utils import (
    RankingAccumulator,
    RunMetrics,
    ScoreStore,
    add_spatial_args,
    apply_best_params,
    collect_eval_batch,
    default_ns_q,
    dense_rank,
    efficiency_metrics,
    is_run_complete,
    load_best_params,
    load_datasets,
    print_efficiency,
    print_run_metrics,
    resolve_spatial_args,
    save_config,
    scoped_run_dir,
    score_store_complete,
    select_torch_device,
    set_compute_threads,
    set_random_seed,
    spatial_defaults,
    spatial_signature,
)


def predict_lgbm(model, X, profile=None, monitor=None, split="test"):
    with monitor.phase("test_forward") if monitor is not None and split == "test" else nullcontext():
        started = time.perf_counter()
        pred = model.predict(X)
        elapsed = time.perf_counter() - started
    if profile is not None:
        profile["model_forward_seconds"] += elapsed
        profile["prediction_calls"] += 1
        profile["predicted_candidates"] += len(X)
    return np.asarray(pred, dtype=np.float32)


class RescueHybridFeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self._init_names()

    def _add(self, name):
        self.feature_names.append(name)

    def _init_names(self):
        for prefix in ("structure_raw", "time", "dsh", "dmh", "direct", "shared", "b", "base", "time_structure_base"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_minmax")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
            self._add(f"{prefix}_top10")
            self._add(f"{prefix}_top50")
            self._add(f"{prefix}_top100")
        for name in (
            "structure_minus_time",
            "direct_minus_time",
            "dsh_minus_time",
            "dmh_minus_time",
            "shared_minus_time",
            "b_minus_time",
            "abs_structure_minus_time",
            "structure_times_time",
            "direct_times_time",
            "component_mean",
            "component_max",
            "component_min",
            "component_std",
            "rank_min_structure_time",
            "rank_gap_structure_time",
            "rank_gap_direct_time",
            "structure_and_time_top10",
            "structure_or_time_top10",
            "structure_and_time_top50",
            "structure_or_time_top50",
            "b_count_log1p",
            "b_seen",
            "tail_count_log1p",
            "incident_count_log1p",
            "rel_tail_count_log1p",
            "source_count_log1p",
            "rel_count_log1p",
            "source_rel_log1p",
            "exact_sro_log1p",
            "source_tail_log1p",
            "relation_is_inverse",
            "candidate_is_source",
        ):
            self._add(name)

    def make(self, scores, time_scores, valid, batch_data, cand_ids, b_counts=None, history_values=None, selected=None):
        score_map = {
            "structure_raw": scores["structure_raw"],
            "time": time_scores,
            "dsh": scores["dsh"],
            "dmh": scores["dmh"],
            "direct": scores["direct"],
            "shared": scores["shared"],
            "b": scores["b"],
            "base": scores["base"],
        }
        score_map["time_structure_base"] = (
            (minmax_by_query(score_map["structure_raw"], valid) + minmax_by_query(time_scores, valid)) * 0.5
        ).astype(np.float32, copy=False)
        ranks = {name: dense_rank(value, valid).astype(np.float32, copy=False) for name, value in score_map.items()}

        features = []
        for prefix in ("structure_raw", "time", "dsh", "dmh", "direct", "shared", "b", "base", "time_structure_base"):
            score = np.where(valid, score_map[prefix], 0.0).astype(np.float32, copy=False)
            rank = ranks[prefix]
            features.extend(
                [
                    score,
                    zscore_by_query(score, valid),
                    minmax_by_query(score, valid),
                    np.log1p(rank).astype(np.float32, copy=False),
                    (1.0 / np.maximum(rank, 1.0)).astype(np.float32, copy=False),
                    ((rank <= 10) & valid).astype(np.float32),
                    ((rank <= 50) & valid).astype(np.float32),
                    ((rank <= 100) & valid).astype(np.float32),
                ]
            )

        structure = score_map["structure_raw"]
        time_score = score_map["time"]
        direct = score_map["direct"]
        dsh = score_map["dsh"]
        dmh = score_map["dmh"]
        shared = score_map["shared"]
        b = score_map["b"]
        stack = np.stack([structure, time_score, dsh, dmh, direct, shared, b, score_map["base"]], axis=2)
        sr = ranks["structure_raw"]
        tr = ranks["time"]
        dr = ranks["direct"]
        features.extend(
            [
                (structure - time_score).astype(np.float32, copy=False),
                (direct - time_score).astype(np.float32, copy=False),
                (dsh - time_score).astype(np.float32, copy=False),
                (dmh - time_score).astype(np.float32, copy=False),
                (shared - time_score).astype(np.float32, copy=False),
                (b - time_score).astype(np.float32, copy=False),
                np.abs(structure - time_score).astype(np.float32, copy=False),
                (structure * time_score).astype(np.float32, copy=False),
                (direct * time_score).astype(np.float32, copy=False),
                np.mean(stack, axis=2).astype(np.float32, copy=False),
                np.max(stack, axis=2).astype(np.float32, copy=False),
                np.min(stack, axis=2).astype(np.float32, copy=False),
                np.std(stack, axis=2).astype(np.float32, copy=False),
                np.minimum(sr, tr).astype(np.float32, copy=False),
                np.abs(sr - tr).astype(np.float32, copy=False),
                np.abs(dr - tr).astype(np.float32, copy=False),
                ((sr <= 10) & (tr <= 10) & valid).astype(np.float32),
                (((sr <= 10) | (tr <= 10)) & valid).astype(np.float32),
                ((sr <= 50) & (tr <= 50) & valid).astype(np.float32),
                (((sr <= 50) | (tr <= 50)) & valid).astype(np.float32),
            ]
        )
        if b_counts is None:
            b_counts = np.zeros_like(structure, dtype=np.float32)
        features.append(np.log1p(np.maximum(b_counts, 0.0)).astype(np.float32, copy=False))
        features.append((b_counts > 0).astype(np.float32, copy=False))
        if history_values is None:
            history_values = [np.zeros_like(structure, dtype=np.float32) for _ in range(8)]
        for arr in history_values:
            features.append(np.log1p(np.maximum(arr, 0.0)).astype(np.float32, copy=False))
        rels = batch_data[:, 1].astype(np.int64, copy=False)
        sources = batch_data[:, 0].astype(np.int64, copy=False)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
        features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        if selected is not None:
            features = [arr[selected] if arr.ndim == 2 else arr for arr in features]
        cube = np.stack(features, axis=-1).astype(np.float32, copy=False)
        bad = int(np.size(cube) - np.sum(np.isfinite(cube)))
        if bad:
            raise RuntimeError(f"rescue hybrid feature cube has {bad} non-finite values")
        return cube


def selected_history_features(history, batch_data, cand_ids, b_counts, selected):
    counts = selected.sum(axis=1)
    width = int(counts.max(initial=0))
    columns = np.zeros((len(counts), width), dtype=np.int64)
    mask = np.arange(width)[None, :] < counts[:, None]
    columns[mask] = np.nonzero(selected)[1]
    candidates = np.where(mask, np.take_along_axis(cand_ids, columns, axis=1), -1)
    values = history.features(batch_data[:, 0], batch_data[:, 1], candidates,
                              np.take_along_axis(b_counts, columns, axis=1))
    return [value[mask] for value in values]


def iter_saved_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, component_root, time_dir):
    started = time.perf_counter()
    time_store = ScoreStore(time_dir, split, mmap=True)
    streamed = getattr(args, 'stream_structure_features', False)
    names = ('structure_raw',) if streamed else COMPONENT_SCORE_NAMES
    component_stores = {
        name: ScoreStore(osp.join(component_root, split, name), split, mmap=True)
        for name in names
    }
    profile = {'score_load_seconds': time.perf_counter() - started,
               'structure_subfeatures_seconds': 0.0, 'frequency_seconds': 0.0,
               'feature_build_seconds': 0.0, 'update_seconds': 0.0, 'rows': 0}
    if not hasattr(args, 'hybrid_profiles'):
        args.hybrid_profiles = {}
    args.hybrid_profiles[split] = profile
    expected_rows = next(iter(component_stores.values())).num_rows
    for name, store in component_stores.items():
        if store.num_rows != expected_rows:
            raise RuntimeError(f"component row mismatch for {split}/{name}: {store.num_rows} != {expected_rows}")
    if time_store.num_rows != expected_rows:
        raise RuntimeError(f"time/component row mismatch for {split}: time={time_store.num_rows}, component={expected_rows}")

    runtime = None
    if streamed:
        set_compute_threads(1)
        print(f'[Hybrid/{split}] replaying Structure for missing subfeatures (no score files written)', flush=True)
        runtime = structure.init_stream_state(data, args, args.hybrid_device)
        if split != 'train':
            structure.advance_structure_state(data, 'train', args, args.hybrid_device, runtime)
        if split == 'test':
            structure.advance_structure_state(data, 'val', args, args.hybrid_device, runtime)
    profile['structure_replay_seconds'] = time.perf_counter() - started - profile['score_load_seconds']
    part_started = time.perf_counter()
    timeline, history = init_timeline_history_state(data, split, args)
    profile['history_replay_seconds'] = time.perf_counter() - part_started
    neg_sampler = data["negative_sampler"]
    part_started = time.perf_counter()
    if hasattr(neg_sampler, 'prepare_mode'):
        neg_sampler.prepare_mode(split)
    profile['negative_prepare_seconds'] = time.perf_counter() - part_started
    print(f'[Hybrid/{split}] scores and history ready in {time.perf_counter() - started:.1f}s', flush=True)
    last_log = time.perf_counter()
    row_offset = 0
    snapshots = split_snapshots(data, split)
    for timestamp_index, (events, t_norm, t_orig) in enumerate(snapshots):
        for batch_data, neg_arr, neg_mask in collect_eval_batch(
            events, t_orig, neg_sampler, split, args.hybrid_query_batch_size
        ):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            scores = {}
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_mask[:, :width]), axis=1)
            for name, store in component_stores.items():
                pos, neg, mask = store.get_block(row_offset, end, width)
                valid[:, 1:] &= mask
                scores[name] = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
            if runtime is not None:
                part_started = time.perf_counter()
                parts = structure.component_score_dict(runtime.predict_parts(batch_data, neg_arr), valid)
                if not np.allclose(parts['structure_raw'][valid], scores['structure_raw'][valid], rtol=1e-5, atol=1e-7):
                    raise ValueError(f'Structure replay disagrees with cached scores at {split} row {row_offset}')
                scores.update({name: parts[name] for name in COMPONENT_SCORE_NAMES if name != 'structure_raw'})
                profile['structure_subfeatures_seconds'] += time.perf_counter() - part_started
            time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
            valid[:, 1:] &= time_mask
            time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
            time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)

            part_started = time.perf_counter()
            rels, pos_counts, neg_counts = timeline.count_batch(batch_data, neg_arr, width)
            b_pos, b_neg = timeline.score_batch(rels, pos_counts, neg_counts, args.b_cfg)
            scores["b"] = np.concatenate((b_pos, b_neg), axis=1).astype(np.float32, copy=False)
            scores["base"] = structure_base_scores(scores, valid)
            profile['frequency_seconds'] += time.perf_counter() - part_started
            part_started = time.perf_counter()
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr[:, :width]), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            b_counts = np.concatenate((pos_counts.reshape(-1, 1), neg_counts), axis=1).astype(np.float32, copy=False)
            selected, component_ranks = rescue_candidate_mask(
                scores, time_scores, valid, args.hybrid_topk,
                time_topk=args.hybrid_time_topk, base_topk=args.hybrid_base_topk,
            )
            history_values = selected_history_features(history, batch_data, cand_ids, b_counts, selected)
            features = rescue_feature_builder.make(
                scores,
                time_scores,
                valid,
                batch_data,
                cand_ids,
                b_counts=b_counts,
                history_values=history_values,
                selected=selected,
            )
            profile['feature_build_seconds'] += time.perf_counter() - part_started
            profile['rows'] = end
            profile['elapsed_seconds'] = time.perf_counter() - started
            yield SimpleNamespace(
                batch_data=batch_data,
                valid=valid,
                scores=scores,
                time_scores=time_scores,
                cand_ids=cand_ids,
                features=features,
                selected=selected,
                component_ranks=component_ranks,
                feature_offsets=np.concatenate(([0], np.cumsum(selected.sum(axis=1)))),
                t_norm=t_norm,
                t_orig=t_orig,
            )
            row_offset = end
            now = time.perf_counter()
            if now - last_log >= 60:
                rate = row_offset / max(now - started, 1e-9)
                print(f'[Hybrid/{split}] rows={row_offset}/{expected_rows} elapsed={now - started:.1f}s '
                      f'subfeatures={profile["structure_subfeatures_seconds"]:.1f}s '
                      f'features={profile["feature_build_seconds"]:.1f}s rows_per_second={rate:.1f}', flush=True)
                last_log = now

        update_events = events_for_update(events, data, args, is_train=(split == "train"))
        if split != 'test' or timestamp_index + 1 < len(snapshots):
            part_started = time.perf_counter()
            timeline.update(update_events)
            history.update(update_events)
            if runtime is not None:
                runtime.update(update_events, t_norm)
            profile['update_seconds'] += time.perf_counter() - part_started

    if row_offset != expected_rows:
        raise RuntimeError(f"component row count mismatch for split={split}: stream={row_offset}, stores={expected_rows}")


def iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root=None):
    yield from iter_saved_rescue_hybrid_blocks(data, split, args, rescue_feature_builder, component_root, time_dir)


def rescue_candidate_mask(
    scores,
    time_scores,
    valid,
    raw_topk,
    time_topk=0,
    base_topk=0,
):
    """Build a label-free union of strict-rank component candidate sets.

    A zero top-k disables that component. ``dense_rank`` is replaced by the
    tie-safe wrapper during production Hybrid runs, so a boundary tie is
    either included or excluded as one group and never depends on column 0.
    """

    raw_topk = int(raw_topk)
    time_topk = int(time_topk or 0)
    base_topk = int(base_topk or 0)
    if raw_topk <= 0:
        raise ValueError("raw_topk must be > 0")
    if time_topk < 0 or base_topk < 0:
        raise ValueError("time_topk/base_topk must be >= 0")

    ranks = {"raw": dense_rank(scores["structure_raw"], valid)}
    selected = ranks["raw"] <= raw_topk
    if time_topk:
        ranks["time"] = dense_rank(time_scores, valid)
        selected |= ranks["time"] <= time_topk
    if base_topk:
        ranks["base"] = dense_rank(scores["base"], valid)
        selected |= ranks["base"] <= base_topk
    return selected & valid, ranks


def build_rescue_hybrid_matrix(
    data,
    split,
    args,
    rescue_feature_builder,
    time_dir,
    device,
    topk,
    component_root=None,
    min_pos_rank=1,
    max_pos_rank=100,
    include_top10=True,
    max_queries=0,
    query_stride=1,
    time_topk=0,
    base_topk=0,
):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    skipped_pos_after_topk = 0
    preserve_queries = 0
    rescue_queries = 0
    eligible_queries = 0
    sampled_out_queries = 0
    max_queries = int(max_queries or 0)
    query_stride = max(1, int(query_stride or 1))
    for block in iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root):
        selected, component_ranks = (block.selected, block.component_ranks) if hasattr(block, 'selected') else rescue_candidate_mask(
            block.scores,
            block.time_scores,
            block.valid,
            topk,
            time_topk=time_topk,
            base_topk=base_topk,
        )
        raw_pos_ranks = component_ranks["raw"][:, 0]
        eligible_pos_ranks = np.minimum.reduce(
            [ranks[:, 0] for ranks in component_ranks.values()]
        )
        for row in range(selected.shape[0]):
            if max_queries > 0 and queries >= max_queries:
                break
            pos_rank = int(eligible_pos_ranks[row])
            raw_pos_rank = int(raw_pos_ranks[row])
            if pos_rank > int(max_pos_rank):
                skipped_pos_after_topk += 1
                continue
            if pos_rank < int(min_pos_rank) and not include_top10:
                continue
            cols = np.flatnonzero(selected[row])
            if len(cols) <= 1 or 0 not in cols:
                skipped_pos_after_topk += 1
                continue
            eligible_queries += 1
            if (eligible_queries - 1) % query_stride != 0:
                sampled_out_queries += 1
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[int(np.flatnonzero(cols == 0)[0])] = 1.0
            X_parts.append(block.features[block.feature_offsets[row]:block.feature_offsets[row + 1]].copy()
                           if block.features.ndim == 2 else block.features[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)
            if raw_pos_rank <= 10:
                preserve_queries += 1
            else:
                rescue_queries += 1
        if max_queries > 0 and queries >= max_queries:
            break
    if not X_parts:
        raise RuntimeError(f"no rescue hybrid rows built for split={split}")
    positives = int(np.sum(np.concatenate(y_parts) > 0.0))
    if positives != queries:
        raise RuntimeError(f"rescue hybrid matrix expected one positive per query, got positives={positives}, queries={queries}")
    nonfinite = int(sum(np.size(x) - np.sum(np.isfinite(x)) for x in X_parts))
    if nonfinite:
        raise RuntimeError(f"rescue hybrid matrix has {nonfinite} non-finite feature values")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "split": split,
            "queries": int(queries),
            "rows": int(rows),
            "topk": int(topk),
            "raw_topk": int(topk),
            "time_topk": int(time_topk or 0),
            "base_topk": int(base_topk or 0),
            "candidate_policy": (
                f"raw{int(topk)}_time{int(time_topk or 0)}_base{int(base_topk or 0)}_union"
            ),
            "min_pos_rank": int(min_pos_rank),
            "max_pos_rank": int(max_pos_rank),
            "include_top10": bool(include_top10),
            "preserve_queries": int(preserve_queries),
            "rescue_queries": int(rescue_queries),
            "skipped_pos_after_topk": int(skipped_pos_after_topk),
            "eligible_queries": int(eligible_queries),
            "sampled_out_queries": int(sampled_out_queries),
            "max_queries": int(max_queries),
            "query_stride": int(query_stride),
        },
    )


def _score_rescue_selected(raw_scores, selected, pred):
    final = raw_scores.astype(np.float32, copy=True)
    if not np.any(selected):
        return final
    outside = (~selected) & np.isfinite(final)
    floor = float(np.max(final[outside])) if np.any(outside) else 0.0
    pred = np.asarray(pred, dtype=np.float32)
    if pred.size == 0:
        return final
    selected_cols = np.flatnonzero(selected)
    raw_sel = raw_scores[selected_cols].astype(np.float32, copy=False)
    order = np.lexsort((-raw_sel, -pred))
    pred_sorted = pred[order]
    raw_sorted = raw_sel[order]
    group_start = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        group_start[1:] = (pred_sorted[1:] != pred_sorted[:-1]) | (
            raw_sorted[1:] != raw_sorted[:-1]
        )
    group_index = np.cumsum(group_start, dtype=np.int32) - 1
    num_groups = int(group_index[-1]) + 1
    final[selected_cols[order]] = floor + 1.0 + (
        (num_groups - group_index).astype(np.float32) / max(num_groups, 1)
    )
    return final


def evaluate_rescue_hybrid_model(
    data, split, args, rescue_feature_builder, hybrid_model, time_dir, device, topk,
    component_root=None, time_topk=0, base_topk=0,
):
    recorder = RankingAccumulator()
    profile = {"model_forward_seconds": 0.0, "prediction_calls": 0, "predicted_candidates": 0}
    for block in iter_rescue_blocks(data, split, args, rescue_feature_builder, time_dir, device, component_root):
        selected, _ = (block.selected, block.component_ranks) if hasattr(block, "selected") else rescue_candidate_mask(
            block.scores, block.time_scores, block.valid, topk, time_topk, base_topk)
        features = block.features if block.features.ndim == 2 else block.features[selected]
        predictions = predict_lgbm(hybrid_model, features, profile,
                                  getattr(args, "_monitor", None), split) if len(features) else np.empty(0, np.float32)
        offset = 0
        final = np.where(block.valid, block.scores["structure_raw"], -np.inf).astype(np.float32)
        for row in range(len(block.valid)):
            count = int(selected[row].sum())
            if count:
                final[row] = _score_rescue_selected(final[row], selected[row], predictions[offset:offset + count])
                offset += count
        recorder.add(final[:, :1], final[:, 1:], block.valid[:, 1:])
    return {**recorder.finish(), "profile": profile}


def fit_lgbm_ranker(X, y, group, feature_names, categorical_indices, args, params):
    import lightgbm as lgb

    parameters = dict(params)
    parameters.setdefault("subsample_freq", int(parameters.get("subsample", 1.0) < 1.0))
    model = lgb.LGBMRanker(**parameters, objective="lambdarank", metric="ndcg",
                          random_state=args.seed, n_jobs=args.num_threads,
                          deterministic=True, force_col_wise=True, verbosity=-1)
    fit_kwargs = {"group": group.tolist(), "feature_name": feature_names}
    if categorical_indices:
        fit_kwargs["categorical_feature"] = categorical_indices
    started = time.perf_counter()
    model.fit(X, y, **fit_kwargs)
    model.fit_seconds_ = time.perf_counter() - started
    return model


def save_lgbm_model(model, path):
    ensure_dir(osp.dirname(path))
    model.booster_.save_model(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train the original Structure + Time hybrid.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_seed", type=int)
    parser.add_argument("--ns_q", type=int)
    parser.add_argument("--train_predict_ratio", type=float)
    parser.add_argument("--query_batch_size", type=int, default=256)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--num_threads", type=int, default=80)
    parser.add_argument("--max_train_queries", type=int, default=0)
    parser.add_argument("--train_query_stride", type=int, default=1)
    parser.add_argument("--rescue_topk", type=int)
    parser.add_argument("--candidate_time_topk", type=int)
    parser.add_argument("--candidate_base_topk", type=int)
    parser.add_argument("--min_pos_rank", type=int)
    parser.add_argument("--max_pos_rank", type=int)
    parser.add_argument("--include_top10", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output_root", default="results_original_hybrid")
    parser.add_argument("--lgbm_params", help="JSON file with LightGBM parameters.")
    parser.add_argument("--time_config", help="JSON-compatible YAML file for Time parameters.")
    parser.add_argument("--time_dir", help="Reuse complete prefix-OOF train/val/test scores.")
    parser.add_argument("--structure_output_root", default="results_structure")
    parser.add_argument('--structure_dir', help='Existing component_scores directory.')
    parser.add_argument('--stream_structure_features', action='store_true', help='Replay missing subfeatures without saving scores.')
    parser.add_argument('--structure_config', help='Structure hyperparameter YAML file.')
    parser.add_argument('--skip_val_eval', action='store_true')
    parser.add_argument('--b_dense_counts', action='store_true', help='Vectorized frequency lookup; uses 8*num_rels*num_nodes bytes.')
    parser.add_argument("--use_best_hyper_params", action="store_true")
    best = load_best_params("ICEWS14")
    for key, value in best["hybrid"]["lgbm"].items():
        parser.add_argument(f"--lgbm_{key}", type=type(value))
    for section in ("time", "structure"):
        for key, value in best[section].items():
            if key in ("reuse_full_dir", "require_reuse_full", "reuse_no_retrain_full"):
                continue
            if key in spatial_defaults('ICEWS14', section):
                continue
            if isinstance(value, bool):
                parser.add_argument(f"--{section}_{key}", action=argparse.BooleanOptionalAction, default=None)
            else:
                value_type = float if key in ("window_semantic_sim", "window_trans") else int if value is None else type(value)
                parser.add_argument(f"--{section}_{key}", type=value_type)
        add_spatial_args(parser, section, prefix=f'{section}_')
    parser.add_argument('--time_stream_train_batch_events', type=int)
    parser.add_argument('--time_stream_eval_batch_events', type=int)
    parser.add_argument('--time_use_amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--data_root', default='data')
    return parser.parse_args(argv)


def _component_args(args, section):
    best = load_best_params(args.dataset) if args.use_best_hyper_params else {}
    for key, default in (("ns_seed", 1), ("ns_q", default_ns_q(args.dataset)), ("train_predict_ratio", .5)):
        if getattr(args, key) is None:
            setattr(args, key, best.get("hybrid", best.get("time", {})).get(key, default))
    if section == "time":
        component = train_time.parse_args([])
        component.use_best_hyper_params = args.use_best_hyper_params
        component.config = args.time_config
        if args.time_config:
            with open(args.time_config, encoding="utf-8") as file:
                config = json.load(file)
            component.seed = config.get("time", config).get("seed")
        elif args.use_best_hyper_params:
            component.seed = best["time"]["seed"]
    else:
        component = structure.parse_args([])
        component.dataset = args.dataset
        if args.structure_config or args.use_best_hyper_params:
            apply_best_params(component, "structure", args.structure_config)
        component.output_root = args.structure_output_root
        component.num_threads = args.num_threads
        component.source_join_threads = args.source_join_threads
    component.dataset, component.gpu = args.dataset, args.gpu
    component.data_root = args.data_root
    component.ns_q, component.ns_seed = args.ns_q, args.ns_seed
    component.train_predict_ratio = args.train_predict_ratio
    keys = set(load_best_params("ICEWS14")[section]) | set(spatial_defaults(args.dataset, section))
    if section == "time":
        keys.update(("stream_train_batch_events", "stream_eval_batch_events", "use_amp"))
    for key in keys:
        value = getattr(args, f"{section}_{key}", None)
        if value is not None:
            setattr(component, key, value)
    for key in ("ns_q", "ns_seed", "train_predict_ratio"):
        if getattr(component, key) != getattr(args, key):
            raise ValueError(f"{section}.{key} must match the Hybrid evaluation protocol")
    if section == "structure":
        resolve_spatial_args(component, "structure")
        component.query_batch_size = component.batch_size
        component.hybrid_query_batch_size = args.query_batch_size
        component.b_cfg = BConfig(**best.get("hybrid", {}).get("b", {}))
        component.stream_structure_features = args.stream_structure_features
        component.hybrid_device = select_torch_device(args.gpu)
        component.b_dense_counts = args.b_dense_counts
    return component


def reuse_time_scores(time_dir, time_args, splits=('train', 'val', 'test')):
    with open(osp.join(time_dir, "config.json"), encoding="utf-8") as file:
        config = json.load(file)
    with open(osp.join(time_dir, "metrics.json"), encoding="utf-8") as file:
        metrics = json.load(file)
    expected = train_time.build_time_args(time_args)
    expected_params = train_time.get_run_params(expected)
    execution_keys = {
        "ebs", "eval_relation_group_size", "eval_neg_chunk",
        "eval_node_preload_chunk", "max_eval_pairs", "max_eval_node_cache_mb",
        "preload_eval_nodes", "dense_eval_node_cache", "cache_eval_source",
    }
    mismatch = [key for key, value in expected_params.items()
                if key not in execution_keys and config["run_params"].get(key) != value]
    if config["dataset"] != expected.dataset or config["seed"] != expected.seed:
        mismatch.append("dataset/seed")
    if mismatch:
        raise ValueError(f"Time cache parameters differ: {mismatch}")
    if metrics["score_protocol"] != "train_oof_prefix_valtest_full_train":
        raise ValueError("Hybrid requires prefix-OOF train scores, not evaluation-only scores.")
    if not is_run_complete(time_dir, splits):
        raise ValueError(f"Incomplete Time scores: {time_dir}")
    print(f"[Hybrid] reuse Time scores: {time_dir} best_epoch={metrics['best_epoch']}", flush=True)
    return time_dir


def reuse_structure_scores(root, args, splits):
    with open(osp.join(osp.dirname(root), "config.json"), encoding="utf-8") as file:
        cached = json.load(file)
    keys = ("dataset", "seed", "ns_q", "ns_seed", "train_predict_ratio", "dict_mode",
            "shared_w", "per_rel_use_mtrans", "ppr_k", "top_k_relation", "ppr_alpha",
            "ppr_beta", "gamma", "direct_single_hop", "decay_direct", "top_share",
            "top_direct", "decay_rel_trans", "window_semantic_sim", "window_trans",
            "close_update_backward", "max_events_in_single_batch")
    mismatch = [key for key in keys if cached.get(key) != getattr(args, key)]
    with open(osp.join(root, "metrics.json"), encoding="utf-8") as file:
        context = json.load(file)["run_context"]
    if context["spatial"] != spatial_signature(args, "structure"):
        mismatch.append("spatial")
    if mismatch:
        raise ValueError(f"Structure cache parameters differ: {mismatch}")
    names = ("structure_raw",) if args.stream_structure_features else COMPONENT_SCORE_NAMES
    for split in splits:
        for name in names:
            if not score_store_complete(osp.join(root, split, name), split):
                raise ValueError(f"Incomplete Structure scores: {root}/{split}/{name}")
    print(f"[Hybrid] reuse Structure scores: {root}", flush=True)
    return root


def cached_component(loader, path, args, splits, data=None):
    # Validate cache protocol and dimensions before starting Hybrid.
    try:
        root = loader(path, args, splits)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if data is not None:
        names = ("structure_raw",) if getattr(args, "stream_structure_features", False) else COMPONENT_SCORE_NAMES
        for split in splits:
            rows = sum(len(events) for events, _, _ in split_snapshots(data, split))
            folders = [root] if loader is reuse_time_scores else [osp.join(root, split, name) for name in names]
            if not all(score_store_complete(folder, split, rows) for folder in folders):
                return None
    return root


def main(args):
    component_defaults = load_best_params(args.dataset)['time'] if args.use_best_hyper_params else {}
    if args.ns_seed is None:
        args.ns_seed = component_defaults.get('ns_seed', 1)
    if args.train_predict_ratio is None:
        args.train_predict_ratio = component_defaults.get('train_predict_ratio', .5)
    args.ns_q = default_ns_q(args.dataset) if args.ns_q is None else args.ns_q
    best = load_best_params(args.dataset)["hybrid"] if args.use_best_hyper_params else None
    if args.lgbm_params:
        with open(args.lgbm_params, "r", encoding="utf-8") as file:
            lgbm_params = json.load(file)
    elif best:
        lgbm_params = best["lgbm"]
    else:
        raise ValueError("Pass --use_best_hyper_params or --lgbm_params.")

    lgbm_params = dict(lgbm_params)
    for key in load_best_params("ICEWS14")["hybrid"]["lgbm"]:
        value = getattr(args, f"lgbm_{key}", None)
        if value is not None:
            lgbm_params[key] = value
    set_compute_threads(args.num_threads)
    set_random_seed(args.seed)
    device = select_torch_device(args.gpu)
    time_args = _component_args(args, "time")
    structure_args = _component_args(args, "structure")
    splits = ('train', 'test') if args.skip_val_eval else ('train', 'val', 'test')
    if args.stream_structure_features and not args.structure_dir:
        raise ValueError('--stream_structure_features requires --structure_dir')
    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio,
                         load_eval_neg=True, ns_seed=args.ns_seed, data_root=args.data_root,
                         lazy_negatives=True)
    expected_time_args = train_time.build_time_args(time_args)
    bounds = train_time.infer_time_bounds(data, split_names=("train_list",))
    raw_bounds = train_time.infer_time_bounds(data, split_names=("train_list",), use_raw=True)
    train_time.resolve_abs_time_periods(expected_time_args, bounds, raw_bounds)
    time_args.abs_time_periods = expected_time_args.abs_time_periods
    expected_time_dir = args.time_dir or train_time.get_out_dir(expected_time_args)
    time_dir = cached_component(reuse_time_scores, expected_time_dir, time_args, splits, data)
    if time_dir is None:
        time_args.force = True
        time_dir = train_time.main(time_args, data=data)
        reuse_time_scores(time_dir, time_args, splits)
    expected_structure_dir = args.structure_dir or structure.component_score_root(structure.make_result_dir(structure_args))
    component_dir = cached_component(reuse_structure_scores, expected_structure_dir, structure_args, splits, data)
    if component_dir is None:
        structure_args.force_recompute = True
        component_dir, _ = structure.run_component_scores(
            structure_args, data, splits=splits, include_hybrid_features=True)
    set_compute_threads(args.num_threads)
    builder = RescueHybridFeatureBuilder(structure.runtime_num_rels(data))
    topk = int(args.rescue_topk if args.rescue_topk is not None else best["rescue_topk"] if best else 100)
    time_topk = int(args.candidate_time_topk if args.candidate_time_topk is not None else best["time_topk"] if best else 100)
    base_topk = int(args.candidate_base_topk if args.candidate_base_topk is not None else best["base_topk"] if best else 100)
    min_pos_rank = int(args.min_pos_rank if args.min_pos_rank is not None else best["min_pos_rank"] if best else 1)
    max_pos_rank = int(args.max_pos_rank if args.max_pos_rank is not None else best["max_pos_rank"] if best else 100)
    include_top10 = bool(args.include_top10 if args.include_top10 is not None else best["include_top10"] if best else True)
    monitor = RunMetrics("hybrid", args)
    try:
        structure_args._monitor = monitor
        structure_args.hybrid_topk = topk
        structure_args.hybrid_time_topk = time_topk
        structure_args.hybrid_base_topk = base_topk
        print("[Hybrid] building LightGBM train-suffix features from prefix-OOF Time scores", flush=True)
        X, y, group, train_info = build_rescue_hybrid_matrix(
            data, "train", structure_args, builder, time_dir, device, topk,
            component_root=component_dir,
            min_pos_rank=min_pos_rank,
            max_pos_rank=max_pos_rank,
            include_top10=include_top10,
            max_queries=args.max_train_queries,
            query_stride=args.train_query_stride,
            time_topk=time_topk,
            base_topk=base_topk,
        )
        train_info["feature_matrix_bytes"] = int(X.nbytes + y.nbytes + group.nbytes)
        data['negative_sampler'].release_mode('train')
        gc.collect()
        print(f"[Hybrid] train queries={len(group)} candidates={len(y)} features={X.shape[1]} matrix_MiB={train_info['feature_matrix_bytes'] / 2**20:.1f}", flush=True)
        with monitor.phase("lgbm_train", device):
            set_compute_threads(args.num_threads)
            print(f"[Hybrid] fitting LightGBM trees={lgbm_params['n_estimators']} threads={args.num_threads}", flush=True)
            model = fit_lgbm_ranker(
                X, y, group, builder.feature_names, [], args, lgbm_params,
            )
        del X, y, group
        run_key = {
            "time_dir": time_dir,
            "structure_dir": component_dir,
            "lgbm": lgbm_params,
            "b": vars(structure_args.b_cfg),
            "topk": (topk, time_topk, base_topk),
            "rank_range": (min_pos_rank, max_pos_rank, include_top10),
            "train_queries": (args.max_train_queries, args.train_query_stride),
            "num_threads": args.num_threads,
            "skip_val_eval": args.skip_val_eval,
            "stream_structure_features": args.stream_structure_features,
        }
        run_hash = hashlib.sha256(json.dumps(run_key, sort_keys=True).encode()).hexdigest()[:12]
        out_dir = osp.join(
            scoped_run_dir(args.output_root, args.dataset, args.seed, args.ns_seed, args.train_predict_ratio),
            f"hybrid_{run_hash}",
        )
        ensure_dir(out_dir)
        save_lgbm_model(model, osp.join(out_dir, "hybrid_model.txt"))
        print(f'[Hybrid] model saved: {out_dir}', flush=True)
        results = {}
        for split in (("test",) if args.skip_val_eval else ("val", "test")):
            values = evaluate_rescue_hybrid_model(
                data, split, structure_args, builder, model, time_dir, device, topk,
                component_root=component_dir, time_topk=time_topk, base_topk=base_topk)
            results[split] = {key: values[key] for key in ("mrr_strict", "hit@1_strict", "hit@10_strict")}
            if split == "test":
                test_seconds = values["profile"]["model_forward_seconds"]
            print_run_metrics(f"Hybrid/{split}", results[split])
        efficiency = efficiency_metrics(model.fit_seconds_, monitor.peak_rss("lgbm_train"),
                                        test_seconds, monitor.peak_rss("test_forward"), "cpu_rss")
        save_config(out_dir, run_key)
        summary = {**results, "efficiency": efficiency}
        with open(osp.join(out_dir, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print_efficiency("Hybrid", efficiency)
        print(f"[Hybrid] output={out_dir}", flush=True)
        return summary
    finally:
        monitor.finish()
        del structure_args._monitor


if __name__ == "__main__":
    main(parse_args())
