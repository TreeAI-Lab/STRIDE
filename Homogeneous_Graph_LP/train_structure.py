"""Relation-free Structure: causal event-wise TPPR updates and shared-neighbor scores."""
import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from utils import (PROJECT_DIR, SUPPORTED_DATASETS, StructureState, NegEdgeSampler,
                   configure_args, data_directory, get_data_transductive,
                   set_random_seed, score_paths, atomic_pickle_dump,
                   canonical_negative_matrix, make_score_metadata, write_score_metadata,
                   compute_metrics, PhaseMonitor, write_result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '--dataset_name', dest='dataset_name', choices=SUPPORTED_DATASETS, default='wikipedia')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--eval_batch_size', type=int, help='Negative-sampling batch size; defaults to batch_size.')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--sim', choices=('mul_wo_norm',), default='mul_wo_norm')
    parser.add_argument('--gpu', type=int, default=0, help='Negative-sampling device; structural updates use CPU. -1 selects CPU.')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--negative_cache_tag', default='')
    return configure_args(parser, 'structure', argv)


def sample_negatives(args, split, data, full_destinations, q, seed, device):
    sampler = NegEdgeSampler(data.destinations, full_destinations, q, device, seed)
    root = data_directory(args.dataset_name, args.data_root) / 'cache/batchneg'
    suffix = f'_tag_{args.negative_cache_tag}' if args.negative_cache_tag else ''
    parts = []
    batch_size = args.eval_batch_size or args.batch_size
    for batch, start in enumerate(tqdm(range(0, data.n_interactions, batch_size), desc=f'{split} negatives')):
        destinations = data.destinations[start:start + batch_size]
        path = root / f'{split}_neg{q}_bs{batch_size}{suffix}_batch{batch}.pkl'
        if path.exists():
            with path.open('rb') as stream:
                values = pickle.load(stream)
        else:
            values = sampler.sample(destinations).cpu().numpy()
            atomic_pickle_dump(values, str(path))
        parts.append(canonical_negative_matrix(values, len(destinations), q, str(path)))
    return np.concatenate(parts)


def update(state, data):
    nodes = np.concatenate([data.sources, data.destinations], dtype=np.int32)
    state.precompute_link_prediction(nodes, 0, False)


def predict_and_update(state, data, negatives):
    with PhaseMonitor() as monitor:
        nodes = np.concatenate([data.sources, data.destinations, negatives.T.reshape(-1)], dtype=np.int32)
        scores = state.precompute_link_prediction(nodes, negatives.shape[1])
    return (nodes, data.timestamps, scores, monitor.seconds, monitor.peak_rss_bytes), monitor


def save_scores(args, split, path, data, negatives, payload):
    atomic_pickle_dump(payload, path)
    metadata = make_score_metadata(args.dataset_name, split, data.sources,
                                   data.destinations, data.timestamps, negatives,
                                   'Simple-LP-Structure', vars(args))
    write_score_metadata(path, metadata)


def main(args):
    os.chdir(PROJECT_DIR)
    set_random_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    full, train, val, test, n_nodes, _ = get_data_transductive(args.dataset_name, True, args.data_root)
    # Compile on a disposable two-node state before measuring real updates.
    dummy = StructureState(2, args.topk, args.alpha, args.beta, args.sim)
    dummy.precompute_link_prediction(np.array([0, 1], dtype=np.int32), 0, False)
    dummy.precompute_link_prediction(np.array([0, 1, 0], dtype=np.int32), 1)
    state = StructureState(n_nodes + 1, args.topk, args.alpha, args.beta, args.sim)
    print('Updating training events...', flush=True)
    with PhaseMonitor() as train_monitor:
        update(state, train)
    paths = score_paths('structure', args)
    val_negatives = sample_negatives(args, 'HybridVal', val, full.destinations, 99, 2025, device)
    val_payload, _ = predict_and_update(state, val, val_negatives)
    save_scores(args, 'HybridVal', paths['hybrid_val'], val, val_negatives, val_payload)
    test_negatives = sample_negatives(args, 'Test', test, full.destinations, 99, 2026, device)
    print('Predicting test events...', flush=True)
    test_payload, test_monitor = predict_and_update(state, test, test_negatives)
    save_scores(args, 'Test', paths['test'], test, test_negatives, test_payload)
    scores = test_payload[2].astype(np.float32)
    scores[np.abs(scores) <= 1e-8] = 0
    n = test.n_interactions
    ap, mrr, hits = compute_metrics(torch.from_numpy(scores[:n]), torch.from_numpy(scores[n:]), 'cpu')
    result = {
        'test': {'AP': ap, 'MRR': mrr, 'H1': hits[0], 'H10': hits[1]},
        'train_time_s': train_monitor.seconds, 'train_peak_rss_bytes': train_monitor.peak_rss_bytes,
        'test_infer_time_s': test_monitor.seconds, 'test_peak_rss_bytes': test_monitor.peak_rss_bytes,
        'score_paths': {'hybrid_val': paths['hybrid_val'], 'test': paths['test']},
    }
    return write_result('structure', args, result)


if __name__ == '__main__':
    main(parse_args())
