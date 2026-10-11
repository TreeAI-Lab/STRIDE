"""Two-score, 50-feature LightGBM fusion on chronological validation events."""
import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score

import train_time
import train_structure
from utils import (PROJECT_DIR, SUPPORTED_DATASETS, configure_args, component_signature, get_data_transductive,
                   score_paths, load_score_metadata, PhaseMonitor, write_result)

def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def strict_tie_rank(scores):
    """Descending strict/max ranks: [10, 9, 9, 8] -> [1, 3, 3, 4]."""

    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"strict_tie_rank expects a 2-D array, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("cannot rank non-finite scores")
    width = int(scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    positions = np.arange(1, width + 1, dtype=np.int32)[None, :]
    group_end = np.ones(sorted_scores.shape, dtype=np.bool_)
    if width > 1:
        group_end[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    markers = np.where(group_end, positions, width + 1)
    sorted_ranks = np.minimum.accumulate(markers[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty(order.shape, dtype=np.int32)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = sorted_ranks
    return ranks


def row_stats(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    scale = np.maximum(maximum - minimum, 1e-12)
    z = (scores - mean) / np.maximum(std, 1e-12)
    minmax = (scores - minimum) / scale
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "z": z,
        "minmax": minmax,
        "centered": scores - mean,
        "gap_to_max": maximum - scores,
    }


def repeat_query_value(value, width):
    return np.repeat(np.asarray(value, dtype=np.float32), width, axis=1)


def build_feature_cube(structure_scores, time_scores):
    structure_scores = np.asarray(structure_scores, dtype=np.float32)
    time_scores = np.asarray(time_scores, dtype=np.float32)
    if structure_scores.shape != time_scores.shape or structure_scores.ndim != 2:
        raise ValueError(
            f"component shape mismatch: structure={structure_scores.shape} time={time_scores.shape}"
        )
    if not np.all(np.isfinite(structure_scores)) or not np.all(np.isfinite(time_scores)):
        raise ValueError("component scores contain non-finite values")

    width = int(structure_scores.shape[1])
    s_stats = row_stats(structure_scores)
    t_stats = row_stats(time_scores)
    s_rank = strict_tie_rank(structure_scores).astype(np.float32)
    t_rank = strict_tie_rank(time_scores).astype(np.float32)
    denom = float(max(width, 1))

    clipped_time = np.clip(time_scores, 1e-7, 1.0 - 1e-7)
    time_logit = np.log(clipped_time / (1.0 - clipped_time)).astype(np.float32)

    features = []
    names = []

    def add(name, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != structure_scores.shape:
            raise ValueError(f"feature {name} has shape {value.shape}, expected {structure_scores.shape}")
        names.append(name)
        features.append(value)

    for prefix, score, rank, stats in (
        ("structure", structure_scores, s_rank, s_stats),
        ("time", time_scores, t_rank, t_stats),
    ):
        add(f"{prefix}_score", score)
        add(f"{prefix}_log1p_nonnegative", np.log1p(np.maximum(score, 0.0)))
        add(f"{prefix}_z", stats["z"])
        add(f"{prefix}_minmax", stats["minmax"])
        add(f"{prefix}_centered", stats["centered"])
        add(f"{prefix}_gap_to_max", stats["gap_to_max"])
        add(f"{prefix}_rank_log", np.log1p(rank))
        add(f"{prefix}_rank_recip", 1.0 / np.maximum(rank, 1.0))
        add(f"{prefix}_rank_fraction", rank / denom)
    add("time_logit", time_logit)

    add("structure_minus_time", structure_scores - time_scores)
    add("abs_structure_minus_time", np.abs(structure_scores - time_scores))
    add("structure_times_time", structure_scores * time_scores)
    add("score_mean", (structure_scores + time_scores) * 0.5)
    add("score_max", np.maximum(structure_scores, time_scores))
    add("score_min", np.minimum(structure_scores, time_scores))
    add("z_difference", s_stats["z"] - t_stats["z"])
    add("abs_z_difference", np.abs(s_stats["z"] - t_stats["z"]))
    add("z_product", s_stats["z"] * t_stats["z"])
    add("minmax_difference", s_stats["minmax"] - t_stats["minmax"])
    add("abs_minmax_difference", np.abs(s_stats["minmax"] - t_stats["minmax"]))
    add("minmax_product", s_stats["minmax"] * t_stats["minmax"])
    add("rank_min", np.minimum(s_rank, t_rank))
    add("rank_max", np.maximum(s_rank, t_rank))
    add("rank_gap", np.abs(s_rank - t_rank))
    add("rank_fraction_min", np.minimum(s_rank, t_rank) / denom)
    add("rank_fraction_max", np.maximum(s_rank, t_rank) / denom)
    add("rank_fraction_gap", np.abs(s_rank - t_rank) / denom)
    add("rank_recip_product", 1.0 / np.maximum(s_rank * t_rank, 1.0))

    for prefix, stats in (("structure", s_stats), ("time", t_stats)):
        for stat_name in ("mean", "std", "min", "max", "range"):
            add(f"query_{prefix}_{stat_name}", repeat_query_value(stats[stat_name], width))

    covariance = np.mean(s_stats["centered"] * t_stats["centered"], axis=1, keepdims=True)
    correlation = covariance / np.maximum(s_stats["std"] * t_stats["std"], 1e-12)
    add("query_structure_time_covariance", repeat_query_value(covariance, width))
    add("query_structure_time_correlation", repeat_query_value(correlation, width))

    cube = np.stack(features, axis=2).astype(np.float32, copy=False)
    if not np.all(np.isfinite(cube)):
        raise ValueError("hybrid feature cube contains non-finite values")
    return cube, names


def load_structure_scores(path, expected_rows, expected_num_neg, data):
    payload = load_pickle(path)
    if not isinstance(payload, tuple) or len(payload) < 3:
        raise ValueError(f"unexpected structure score format: {path}")
    nodes = to_numpy(payload[0]).reshape(-1).astype(np.int64, copy=False)
    scores = to_numpy(payload[2]).reshape(-1).astype(np.float32, copy=False)
    width = int(expected_num_neg) + 1
    expected_scores = int(expected_rows) * width
    expected_nodes = int(expected_rows) * (int(expected_num_neg) + 2)
    if len(scores) != expected_scores or len(nodes) != expected_nodes:
        raise ValueError(
            f"structure score size mismatch in {path}: scores={len(scores)}/{expected_scores} "
            f"nodes={len(nodes)}/{expected_nodes}"
        )
    matrix = np.concatenate(
        (
            scores[:expected_rows].reshape(expected_rows, 1),
            scores[expected_rows:].reshape(expected_num_neg, expected_rows).T,
        ),
        axis=1,
    )
    if not np.array_equal(nodes[:expected_rows], np.asarray(data.sources, dtype=np.int64)):
        raise ValueError(f"structure source order does not match dataset: {path}")
    if not np.array_equal(
        nodes[expected_rows : 2 * expected_rows], np.asarray(data.destinations, dtype=np.int64)
    ):
        raise ValueError(f"structure positive destination order does not match dataset: {path}")
    return matrix, nodes


def load_time_scores(path, expected_rows, expected_num_neg):
    payload = load_pickle(path)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"unexpected time score format: {path}")
    pos_batches, neg_batches = payload
    if len(pos_batches) != len(neg_batches):
        raise ValueError(f"time score batch count mismatch: {path}")
    pos_parts = []
    neg_parts = []
    for batch_idx, (pos_raw, neg_raw) in enumerate(zip(pos_batches, neg_batches)):
        pos = to_numpy(pos_raw).reshape(-1).astype(np.float32, copy=False)
        neg = to_numpy(neg_raw).reshape(-1).astype(np.float32, copy=False)
        if len(neg) != len(pos) * int(expected_num_neg):
            raise ValueError(
                f"time negative size mismatch in {path}, batch={batch_idx}: "
                f"pos={len(pos)} neg={len(neg)} q={expected_num_neg}"
            )
        pos_parts.append(pos)
        neg_parts.append(neg.reshape(expected_num_neg, len(pos)).T)
    pos = np.concatenate(pos_parts) if pos_parts else np.empty(0, dtype=np.float32)
    neg = np.concatenate(neg_parts, axis=0) if neg_parts else np.empty((0, expected_num_neg), dtype=np.float32)
    if len(pos) != int(expected_rows):
        raise ValueError(f"time score row mismatch in {path}: got={len(pos)} expected={expected_rows}")
    return np.concatenate((pos.reshape(-1, 1), neg), axis=1)


def make_lgbm_params(args, n_estimators, overrides=None):
    overrides = overrides or {}

    def value(name):
        return overrides.get(name, getattr(args, name))

    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "n_estimators": int(n_estimators),
        "learning_rate": float(value("learning_rate")),
        "num_leaves": int(value("num_leaves")),
        "max_depth": int(value("max_depth")),
        "min_child_samples": int(value("min_child_samples")),
        "reg_lambda": float(value("reg_lambda")),
        "reg_alpha": float(value("reg_alpha")),
        "min_split_gain": float(value("min_split_gain")),
        "subsample": float(value("subsample")),
        "subsample_freq": 1,
        "colsample_bytree": float(value("colsample_bytree")),
        "random_state": int(args.seed),
        "n_jobs": int(args.num_threads),
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }


def flatten_queries(cube):
    rows, width, num_features = cube.shape
    X = cube.reshape(rows * width, num_features).astype(np.float32, copy=False)
    y = np.zeros(rows * width, dtype=np.float32)
    y[::width] = 1.0
    group = np.full(rows, width, dtype=np.int32)
    return X, y, group


def chronological_meta_split(timestamps, meta_val_ratio):
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("at least two validation queries are required for a meta split")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("validation queries are not ordered by timestamp")
    target = min(max(int(len(timestamps) * (1.0 - float(meta_val_ratio))), 1), len(timestamps) - 1)
    boundary = timestamps[target]
    split = int(np.searchsorted(timestamps, boundary, side="left"))
    if split <= 0:
        split = int(np.searchsorted(timestamps, boundary, side="right"))
    if split <= 0 or split >= len(timestamps):
        raise ValueError(
            "cannot create a timestamp-disjoint validation prefix/tail split; "
            "adjust --meta_val_ratio"
        )
    return split


def evaluate_scores(scores, ks=(1, 10)):
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite prediction scores')
    pos = scores[:, :1]
    neg = scores[:, 1:]
    loose = 1 + np.sum(neg > pos, axis=1)
    strict = 1 + np.sum(neg >= pos, axis=1)
    avg = (loose + strict) * 0.5
    result = {
        "num_queries": int(len(scores)),
        "ap": float(
            average_precision_score(
                np.concatenate(
                    [np.ones(len(pos), dtype=np.int8), np.zeros(neg.size, dtype=np.int8)]
                ),
                np.concatenate([pos.reshape(-1), neg.reshape(-1)]),
            )
        ),
        "mrr_loose": float(np.mean(1.0 / loose)),
        "mrr_strict": float(np.mean(1.0 / strict)),
        "mrr_avg": float(np.mean(1.0 / avg)),
    }
    for k in ks:
        result[f"hit@{k}_loose"] = float(np.mean(loose <= k))
        result[f"hit@{k}_strict"] = float(np.mean(strict <= k))
        result[f"hit@{k}_avg"] = float(np.mean(avg <= k))
    return result


def select_boosting_rounds(
    lgb,
    args,
    feature_names,
    X_train,
    y_train,
    group_train,
    X_meta,
    y_meta,
    group_meta,
    params,
    fixed_rounds=False,
):
    max_estimators = int(params.get("n_estimators", args.n_estimators))
    selector = lgb.LGBMRanker(
        **make_lgbm_params(args, max_estimators, overrides=params)
    )
    if fixed_rounds:
        selector.fit(
            X_train,
            y_train,
            group=group_train.tolist(),
            feature_name=feature_names,
        )
        best_iteration = max_estimators
    else:
        selector.fit(
            X_train,
            y_train,
            group=group_train.tolist(),
            feature_name=feature_names,
            eval_set=[(X_meta, y_meta)],
            eval_group=[group_meta.tolist()],
            eval_at=[1, 10],
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(getattr(selector, "best_iteration_", 0) or max_estimators)
    meta_pred = selector.predict(X_meta, num_iteration=best_iteration).reshape(
        len(group_meta), int(group_meta[0])
    )
    return best_iteration, evaluate_scores(meta_pred)


def fit_full_validation_ranker(
    lgb, args, feature_names, X_val, y_val, group_val, params, best_iteration
):
    model = lgb.LGBMRanker(
        **make_lgbm_params(args, best_iteration, overrides=params)
    )
    model.fit(
        X_val,
        y_val,
        group=group_val.tolist(),
        feature_name=feature_names,
    )
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '--dataset_name', dest='dataset_name', choices=SUPPORTED_DATASETS, default='wikipedia')
    parser.add_argument('--gpu', type=int, default=0, help='Device used if Time/Structure scores need generating.')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_threads', type=int, default=16)
    parser.add_argument('--meta_val_ratio', type=float, default=0.2)
    parser.add_argument('--structure_noise_floor', type=float, default=1e-8)
    parser.add_argument('--early_stopping_rounds', type=int, default=30)
    parser.add_argument('--fixed_rounds', action=argparse.BooleanOptionalAction, default=False, help='Fit saved historical boosting rounds, without a new search.')
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--min_child_samples', type=int, default=80)
    parser.add_argument('--reg_lambda', type=float, default=1.0)
    parser.add_argument('--reg_alpha', type=float, default=0.05)
    parser.add_argument('--min_split_gain', type=float, default=0.001)
    parser.add_argument('--subsample', type=float, default=0.9)
    parser.add_argument('--colsample_bytree', type=float, default=0.85)
    parser.add_argument('--time_config', default='', help='Optional full preset overriding the paired Time configuration.')
    parser.add_argument('--structure_config', default='', help='Optional full preset overriding the paired Structure configuration.')
    return configure_args(parser, 'hybrid', argv)


def prepare_components(args):
    paths = {}
    for component, module in (('structure', train_structure), ('time', train_time)):
        cli = ['--dataset', args.dataset_name, '--gpu', str(args.gpu), '--data_root', args.data_root]
        config = getattr(args, f'{component}_config') or args.best_hyper_params_file
        if config:
            cli += ['--best_hyper_params_file', config]
        if config or args.use_best_hyper_params:
            cli += ['--use_best_hyper_params_for_hybrid']
        component_args = module.parse_args(cli)
        component_paths = score_paths(component, component_args)
        if not all(Path(component_paths[key]).is_file() and Path(component_paths[key] + '.meta.json').is_file()
                   for key in ('hybrid_val', 'test')):
            module.main(component_args)
        paths[component] = component_paths
    return paths


def load_aligned_scores(paths, split, data, dataset):
    time_path = paths['time'][split]
    structure_path = paths['structure'][split]
    time_meta = load_score_metadata(time_path)
    structure_meta = load_score_metadata(structure_path)
    expected = {'dataset': dataset, 'split': 'Test' if split == 'test' else 'HybridVal',
                'num_queries': data.n_interactions, 'num_negatives': 99}
    for metadata in (time_meta, structure_meta):
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError(f'{split} score context does not match the dataset')
    if time_meta['candidate_fingerprint'] != structure_meta['candidate_fingerprint']:
        raise ValueError(f'{split} Time/Structure candidates differ; regenerate both stores together')
    structure, _ = load_structure_scores(structure_path, data.n_interactions, 99, data)
    temporal = load_time_scores(time_path, data.n_interactions, 99)
    return structure, temporal


def main(args):
    os.chdir(PROJECT_DIR)
    np.random.seed(args.seed)
    paths = prepare_components(args)
    _, _, val, test, _, _ = get_data_transductive(args.dataset_name, True, args.data_root)
    val_structure, val_time = load_aligned_scores(paths, 'hybrid_val', val, args.dataset_name)
    test_structure, test_time = load_aligned_scores(paths, 'test', test, args.dataset_name)
    for scores in (val_structure, test_structure):
        scores[np.abs(scores) <= args.structure_noise_floor] = 0
    val_cube, feature_names = build_feature_cube(val_structure, val_time)
    test_cube, _ = build_feature_cube(test_structure, test_time)
    X_val, y_val, groups = flatten_queries(val_cube)
    X_test, _, _ = flatten_queries(test_cube)
    params = make_lgbm_params(args, args.n_estimators)
    rounds = args.n_estimators
    if not args.fixed_rounds:
        split = chronological_meta_split(val.timestamps, args.meta_val_ratio)
        X_train, y_train, group_train = flatten_queries(val_cube[:split])
        X_meta, y_meta, group_meta = flatten_queries(val_cube[split:])
        rounds, _ = select_boosting_rounds(lgb, args, feature_names, X_train, y_train,
                                          group_train, X_meta, y_meta, group_meta, params)
    print(f'Fitting LightGBM: {len(groups)} validation queries, {rounds} rounds', flush=True)
    with PhaseMonitor() as train_monitor:
        model = fit_full_validation_ranker(lgb, args, feature_names, X_val, y_val, groups, params, rounds)
    with PhaseMonitor() as test_monitor:
        predictions = model.predict(X_test).reshape(test.n_interactions, 100)
    metrics = evaluate_scores(predictions)
    result = {
        'test': {'AP': metrics['ap'], 'MRR': metrics['mrr_strict'],
                 'H1': metrics['hit@1_strict'], 'H10': metrics['hit@10_strict']},
        'best_iteration': rounds, 'train_time_s': train_monitor.seconds,
        'train_peak_rss_bytes': train_monitor.peak_rss_bytes,
        'test_infer_time_s': test_monitor.seconds, 'test_peak_rss_bytes': test_monitor.peak_rss_bytes,
        'score_paths': paths,
    }
    result = write_result('hybrid', args, result)
    model.booster_.save_model(str(Path('results') / args.dataset_name / 'hybrid' / component_signature(args) / 'model.txt'))
    return result


if __name__ == '__main__':
    main(parse_args())
