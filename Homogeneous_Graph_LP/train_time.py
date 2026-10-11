"""Relation-free Time: temporal history encoding, validation selection, score export."""
import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from utils import (PROJECT_DIR, configure_args, component_signature, score_paths,
                   data_directory, get_data_transductive, set_random_seed,
                   compute_metrics, EarlyStopMonitor, NegEdgeSampler, TimeModel,
                   get_neighbor_finder, move_to_cpu, atomic_pickle_dump,
                   canonical_negative_matrix, make_score_metadata, write_score_metadata,
                   synchronize, reset_peak_memory, peak_memory_mb, PhaseMonitor, write_result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Training Simple LP Time.")
    parser.add_argument(
        "--dataset", "--dataset_name", dest="dataset_name",
        type=str,
        default="wikipedia",
        choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--topk_sample_flag", type=str, default="last")

    parser.add_argument("--max_epoch", "--num_epochs", dest="num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)

    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", "--hidden_dims", dest="hidden_dims", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--shuffle_batches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--train_num_neg",
        type=int,
        default=1,
        help="Training negatives per positive. The V1-compatible default is 1.",
    )
    parser.add_argument(
        "--loss_type",
        choices=["bce", "balanced_bce", "bpr", "hybrid", "sampled_softmax"],
        default="bce",
        help="Opt-in ranking-aligned loss; bce preserves the previous behavior.",
    )
    parser.add_argument("--ranking_loss_weight", type=float, default=1.0)
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument(
        "--masked_pooling",
        action=argparse.BooleanOptionalAction, default=False,
        help="Ignore padded history slots during pooling (off by default).",
    )
    parser.add_argument(
        "--time_summary",
        action=argparse.BooleanOptionalAction, default=False,
        help="Add count/recency/dispersion summaries of temporal history.",
    )
    parser.add_argument(
        "--log_time_input",
        action=argparse.BooleanOptionalAction, default=False,
        help="Feed log1p(delta-time) to the Fourier time encoder.",
    )
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--disable_scheduler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--selection_metric",
        choices=["global_ap", "batch_ap", "strict_mrr"],
        default="global_ap",
        help=(
            "Validation checkpoint metric. MRR choices use a fixed sampled "
            "multi-negative validation set. batch_ap reproduces the V1 policy."
        ),
    )
    parser.add_argument("--ranking_val_num_neg", type=int, default=20)
    parser.add_argument(
        "--ranking_val_max_queries",
        type=int,
        default=50000,
        help="Evenly sampled validation queries for MRR selection; <=0 means all.",
    )
    parser.add_argument(
        "--legacy_v1_rng",
        action=argparse.BooleanOptionalAction, default=False,
        help="Reproduce V1's shared/global negative-sampler RNG side effects.",
    )
    parser.add_argument(
        "--negative_cache_tag",
        type=str,
        default="",
        help="Optional namespace for fixed negative caches.",
    )
    parser.add_argument(
        "--lazy_eval_preprocess",
        action=argparse.BooleanOptionalAction, default=False,
        help="Build full HybridVal/Test tensors only after training finishes.",
    )
    parser.add_argument(
        "--cpu_preprocessed_data",
        action=argparse.BooleanOptionalAction, default=False,
        help="Keep preprocessed batches on CPU and transfer one batch at a time.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional cache/model suffix; use a unique value for parameter sweeps.",
    )

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)

    return configure_args(parser, "time", argv)


def main(args):
    os.chdir(PROJECT_DIR)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    set_random_seed(args.seed)
    DATA = args.dataset_name

    filename = component_signature(args)
    paths = score_paths("time", args)

    safe_negative_cache_tag = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in args.negative_cache_tag
    )
    negative_cache_suffix = (
        f"_tag_{safe_negative_cache_tag}" if safe_negative_cache_tag else ""
    )


    Path(f"log_learn_time/{DATA}").mkdir(parents=True, exist_ok=True)
    Path(f"time_score_cache/{DATA}").mkdir(parents=True, exist_ok=True)
    print(
        f"Time training result will be written at log_learn_time/{DATA}/{filename}.\n"
    )

    best_checkpoint_path = f"saved_checkpoints/{time.time()}.pth"
    Path(f"saved_checkpoints").mkdir(parents=True, exist_ok=True)
    Path(f"saved_time_models/learn_time/{DATA}").mkdir(parents=True, exist_ok=True)
    Path(f"time_processed_data/{DATA}").mkdir(parents=True, exist_ok=True)
    best_model_name = filename
    best_model_path = f"saved_time_models/learn_time/{DATA}/{best_model_name}.pth"


    full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(
        DATA, use_validation=True, data_root=args.data_root
    )
    n_train = train_data.n_interactions
    n_val = val_data.n_interactions
    n_test = test_data.n_interactions
    print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")

    independent_sampler_rng = not args.legacy_v1_rng
    train_neg_edge_sampler = NegEdgeSampler(
        destinations=train_data.destinations,
        full_destinations=train_data.destinations,
        num_neg=args.train_num_neg,
        device=device,
        seed=2024,
        independent_rng=independent_sampler_rng,
    )
    val_neg_edge_sampler = NegEdgeSampler(
        destinations=val_data.destinations,
        full_destinations=full_data.destinations,
        num_neg=1,
        device=device,
        seed=2025,
        independent_rng=independent_sampler_rng,
    )
    test_neg_edge_sampler = NegEdgeSampler(
        destinations=test_data.destinations,
        full_destinations=full_data.destinations,
        num_neg=99,
        device=device,
        seed=2026,
        independent_rng=independent_sampler_rng,
    )
    # These post-V1 samplers always own their RNG so that enabling V1 compatibility
    # does not introduce additional global reseeding that V1 never performed.
    hybrid_val_neg_edge_sampler = NegEdgeSampler(
        destinations=val_data.destinations,
        full_destinations=full_data.destinations,
        num_neg=99,
        device=device,
        seed=2025,
        independent_rng=True,
    )


    finder = get_neighbor_finder(full_data)


    edge_predictor_configs = {
        "dim": args.hidden_dims,
    }

    encoder_config = {
        "per_graph_size": args.topk,
        "time_channels": args.hidden_dims,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_single_layer": False,
        "device": device,
        "masked_pooling": args.masked_pooling,
        "use_time_summary": args.time_summary,
        "log_time_input": args.log_time_input,
    }

    # Samplers have independent generators, but reseeding here also makes the
    # intended model/training seed explicit and robust to future setup changes.
    if not args.legacy_v1_rng:
        set_random_seed(args.seed)
    model = TimeModel(encoder_config, edge_predictor_configs).to(device)

    print(f"Time parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if not args.disable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.min_lr,
        )
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    phase_rss = {}


    def load_or_sample_negative_matrix(
        path, destinations_batch, num_neg, neg_edge_sampler
    ):
        if os.path.exists(path):
            with open(path, "rb") as file_handle:
                return canonical_negative_matrix(pickle.load(file_handle), len(destinations_batch), num_neg, path)

        negative_matrix = canonical_negative_matrix(
            move_to_cpu(neg_edge_sampler.sample(destinations_batch)),
            len(destinations_batch),
            num_neg,
            path,
        )
        atomic_pickle_dump(negative_matrix, path)
        return negative_matrix


    def process_data(mode, finder, data, bs, num_neg=1, filepath=None, neg_edge_sampler=None):
        print(f"Processing {mode} data...")
        num_instance = data.n_interactions
        BATCH_SIZE = bs if bs != -1 else num_instance
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        delta_times_list = []
        all_inds_list = []
        batch_size_list = []

        for batch_idx in tqdm(range(0, num_batch)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sample_inds = np.array(list(range(start_idx, end_idx)))

            sources_batch = data.sources[sample_inds]
            destinations_batch = data.destinations[sample_inds]
            timestamps_batch = data.timestamps[sample_inds]

            if neg_edge_sampler is None:
                raise ValueError(f"negative sampler is required for mode={mode}")
            neg_filepath = (
                f"{data_directory(DATA, args.data_root)}/cache/batchneg/{mode}_neg{num_neg}_bs{bs}"
                f"{negative_cache_suffix}_batch{batch_idx}.pkl"
            )
            negative_matrix = load_or_sample_negative_matrix(
                neg_filepath, destinations_batch, num_neg, neg_edge_sampler
            )
            negatives_batch = negative_matrix.T.reshape(-1)

            source_nodes = np.concatenate(
                [sources_batch, destinations_batch, negatives_batch], dtype=np.int32
            )
            timestamps = np.tile(timestamps_batch, num_neg + 2)

            delta_times, n_neighbors, total_edges = finder.get_clean_delta_times(
                source_nodes, timestamps, args.topk, args.topk_sample_flag
            )
            delta_times = delta_times[:total_edges]
            delta_times = torch.from_numpy(delta_times).to(device).unsqueeze(-1)

            all_inds = []
            for i, n_ngh in enumerate(n_neighbors):
                all_inds.extend([(args.topk * i + j) for j in range(n_ngh)])

            all_inds = torch.tensor(all_inds, dtype=torch.long, device=device)
            batch_size = len(n_neighbors)

            delta_times = move_to_cpu(delta_times)
            all_inds = move_to_cpu(all_inds)

            delta_times_list.append(delta_times)
            all_inds_list.append(all_inds)
            batch_size_list.append(batch_size)

        if filepath:
            atomic_pickle_dump(
                (num_batch, delta_times_list, all_inds_list, batch_size_list), filepath
            )
            print(f"Processed data has been saved at {filepath}.")

        return num_batch, delta_times_list, all_inds_list, batch_size_list


    def process_time_data(
        mode,
        finder,
        data,
        batch_size,
        num_neg=1,
        filepath=None,
        neg_edge_sampler=None,
    ):
        if filepath and os.path.exists(filepath):
            print(f"Loading cached {mode} data from {filepath}.")
            with open(filepath, "rb") as f:
                num_batch, delta_times_list, all_inds_list, batch_size_list = pickle.load(f)

            if not args.cpu_preprocessed_data:
                all_inds_list = [tensor.to(device) for tensor in all_inds_list]
                delta_times_list = [tensor.to(device) for tensor in delta_times_list]

            return num_batch, delta_times_list, all_inds_list, batch_size_list

        else:
            num_batch, delta_times_list, all_inds_list, batch_size_list = process_data(
                mode, finder, data, batch_size, num_neg, filepath, neg_edge_sampler
            )

            if not args.cpu_preprocessed_data:
                all_inds_list = [tensor.to(device) for tensor in all_inds_list]
                delta_times_list = [tensor.to(device) for tensor in delta_times_list]

            return num_batch, delta_times_list, all_inds_list, batch_size_list


    def combine_score_batches(pos_batches, neg_batches, num_neg):
        pos_all = torch.cat(pos_batches, dim=0)
        neg_matrix = np.concatenate(
            [
                neg.detach().cpu().numpy().reshape(num_neg, len(pos)).T
                for pos, neg in zip(pos_batches, neg_batches)
            ],
            axis=0,
        )
        neg_all = torch.from_numpy(neg_matrix.T.reshape(-1, 1))
        return pos_all, neg_all


    def compute_training_loss(pos_score, neg_score, num_neg):
        """Return the requested loss while retaining V1 BCE as the default."""
        if args.loss_type == "bce":
            predicts = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat(
                [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
            )
            return criterion(predicts, labels)

        positive_bce = criterion(pos_score, torch.ones_like(pos_score))
        negative_bce = criterion(neg_score, torch.zeros_like(neg_score))
        balanced_bce = 0.5 * (positive_bce + negative_bce)
        if args.loss_type == "balanced_bce":
            return balanced_bce

        negative_matrix = neg_score.reshape(num_neg, pos_score.shape[0]).T
        if args.loss_type in {"bpr", "hybrid"}:
            ranking_loss = torch.nn.functional.softplus(
                negative_matrix - pos_score
            ).mean()
            if args.loss_type == "bpr":
                return ranking_loss
            return balanced_bce + args.ranking_loss_weight * ranking_loss

        logits = torch.cat([pos_score, negative_matrix], dim=1)
        logits = logits / args.softmax_temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, targets)


    def run(
        model,
        mode,
        epoch,
        optimizer,
        criterion,
        f,
        num_neg,
        num_batch,
        delta_times_list=None,
        all_inds_list=None,
        batch_size_list=None,
        k_list = [10],
    ):
        if mode == "Train":
            model = model.train()
        else:
            model = model.eval()

        ap_list = []
        t_epo = 0.0
        allocated_memory = 0.0

        all_pos_score = []
        all_neg_score = []

        batch_indices = np.arange(num_batch)
        if mode == "Train" and args.shuffle_batches:
            batch_rng = np.random.RandomState(args.seed + epoch)
            batch_rng.shuffle(batch_indices)

        with PhaseMonitor() as phase:
            for batch_idx in tqdm(batch_indices):
                t1 = time.time()
                synchronize(device)
                reset_peak_memory(device)
                no_neighbor_flag = False
                batch_delta_times = delta_times_list[batch_idx].to(
                    device, non_blocking=True
                )
                batch_all_inds = all_inds_list[batch_idx].to(device, non_blocking=True)

                if batch_delta_times.numel() == 0:
                    no_neighbor_flag = True

                    num_pos_sc = batch_size_list[batch_idx] // (num_neg + 2)
                    num_neg_sc = num_pos_sc * num_neg
                    pos_score = torch.zeros(num_pos_sc, 1)
                    neg_score = torch.zeros(num_neg_sc, 1)
                else:
                    pos_score, neg_score = model(
                        batch_delta_times,
                        batch_all_inds,
                        batch_size_list[batch_idx],
                        num_neg,
                    )

                mem = peak_memory_mb(device) # /MB
                allocated_memory = max(allocated_memory, mem)

                synchronize(device)
                t_epo += time.time() - t1

                if mode == "Train" and no_neighbor_flag == False:
                    t2 = time.time()
                    optimizer.zero_grad()
                    loss = compute_training_loss(pos_score, neg_score, num_neg)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    synchronize(device)
                    allocated_memory = max(allocated_memory, peak_memory_mb(device))
                    t_epo += time.time() - t2
                elif mode == "Val":
                    with torch.no_grad():
                        all_pos_score.append(pos_score.sigmoid().cpu())
                        all_neg_score.append(neg_score.sigmoid().cpu())
                        batch_pred = torch.cat([pos_score, neg_score], dim=0).sigmoid().cpu()
                        batch_true = torch.cat(
                            [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
                        ).cpu()
                        ap_list.append(average_precision_score(batch_true, batch_pred))
                elif mode in {"Test", "HybridVal", "RankingVal"}:
                    with torch.no_grad():
                        all_pos_score.append(pos_score.sigmoid().cpu())
                        all_neg_score.append(neg_score.sigmoid().cpu())
        phase_rss[mode] = max(phase_rss.get(mode, 0), phase.peak_rss_bytes)

        if mode == "Train":
            print(
                f"Epoch{epoch}-{mode}: loss: {loss.item():.5f}, time: {t_epo}, memory used: {allocated_memory}"
            )
            return t_epo, allocated_memory

        elif mode == "Val":
            pos_all, neg_all = combine_score_batches(all_pos_score, all_neg_score, num_neg)
            y_pred = torch.cat([pos_all, neg_all], dim=0).numpy().reshape(-1)
            y_true = np.concatenate(
                [
                    np.ones(pos_all.numel(), dtype=np.int8),
                    np.zeros(neg_all.numel(), dtype=np.int8),
                ]
            )
            global_ap = float(average_precision_score(y_true, y_pred))
            batch_ap = float(np.mean(ap_list))
            ap = global_ap if args.selection_metric == "global_ap" else batch_ap
            print(
                f"Epoch{epoch}-{mode}: selection_ap: {ap:.4f}, "
                f"global_ap: {global_ap:.4f}, batch_ap: {batch_ap:.4f}, "
                f"time: {t_epo}, memory used: {allocated_memory}"
            )

        elif mode in {"Test", "HybridVal", "RankingVal"}:
            pos_all, neg_all = combine_score_batches(all_pos_score, all_neg_score, num_neg)
            ap, mrr, mean_hr = compute_metrics(
                pos_all, neg_all, device, k_list=k_list
            )
            print(
                f"Epoch{epoch}-{mode}-Neg_sam{num_neg}: time: {t_epo}, ap: {ap:.4f}, mrr: {mrr:.4f}, "
                + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                + f", memory used: {allocated_memory}"
            )

            if mode in {"Test", "HybridVal"}:
                with open(f, "a") as log_file:
                    log_file.write(
                        f"Final {mode} with Neg_sam {num_neg}: ap: {ap:.4f}, mrr: {mrr:.4f}, "
                        + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                        + "\n"
                    )

            if mode == "RankingVal":
                return mrr, t_epo, all_pos_score, all_neg_score, allocated_memory

        return ap, t_epo, all_pos_score, all_neg_score, allocated_memory


    train_bs = args.batch_size
    val_bs = args.batch_size
    test_bs = args.batch_size

    train_filepath = f"time_processed_data/{DATA}/Train_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{train_bs}_numneg_{args.train_num_neg}{negative_cache_suffix}.pkl"
    val_filepath = f"time_processed_data/{DATA}/Val_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{val_bs}_numneg_1{negative_cache_suffix}.pkl"
    test_filepath = f"time_processed_data/{DATA}/Test_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{test_bs}_numneg_99{negative_cache_suffix}.pkl"

    train_num_batch, train_delta_times_list, train_all_inds_list, train_batch_size_list = (
        process_time_data(
            "Train",
            finder,
            train_data,
            batch_size=train_bs,
            num_neg=args.train_num_neg,
            filepath=train_filepath,
            neg_edge_sampler=train_neg_edge_sampler,
        )
    )
    val_num_batch, val_delta_times_list, val_all_inds_list, val_batch_size_list = (
        process_time_data(
            "Val",
            finder,
            val_data,
            batch_size=val_bs,
            num_neg=1,
            filepath=val_filepath,
            neg_edge_sampler=val_neg_edge_sampler,
        )
    )
    hybrid_val_filepath = f"time_processed_data/{DATA}/HybridVal_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{val_bs}_numneg_99{negative_cache_suffix}.pkl"
    test_num_batch = test_delta_times_list = test_all_inds_list = test_batch_size_list = None
    hybrid_val_num_batch = hybrid_val_delta_times_list = None
    hybrid_val_all_inds_list = hybrid_val_batch_size_list = None


    def prepare_full_evaluation_data():
        test_payload = process_time_data(
            "Test",
            finder,
            test_data,
            batch_size=test_bs,
            num_neg=99,
            filepath=test_filepath,
            neg_edge_sampler=test_neg_edge_sampler,
        )
        hybrid_payload = process_time_data(
            "HybridVal",
            finder,
            val_data,
            batch_size=val_bs,
            num_neg=99,
            filepath=hybrid_val_filepath,
            neg_edge_sampler=hybrid_val_neg_edge_sampler,
        )
        return test_payload, hybrid_payload


    if not args.lazy_eval_preprocess:
        (
            (
                test_num_batch,
                test_delta_times_list,
                test_all_inds_list,
                test_batch_size_list,
            ),
            (
                hybrid_val_num_batch,
                hybrid_val_delta_times_list,
                hybrid_val_all_inds_list,
                hybrid_val_batch_size_list,
            ),
        ) = prepare_full_evaluation_data()


    ranking_selection = args.selection_metric == "strict_mrr"
    ranking_val_data = None
    ranking_val_num_batch = None
    ranking_val_delta_times_list = None
    ranking_val_all_inds_list = None
    ranking_val_batch_size_list = None
    if ranking_selection:
        if 0 < args.ranking_val_max_queries < val_data.n_interactions:
            ranking_indices = np.linspace(
                0,
                val_data.n_interactions - 1,
                num=args.ranking_val_max_queries,
                dtype=np.int64,
            )
            ranking_val_data = type(val_data)(
                val_data.sources[ranking_indices],
                val_data.destinations[ranking_indices],
                val_data.timestamps[ranking_indices],
                val_data.edge_idxs[ranking_indices],
                val_data.labels[ranking_indices],
            )
        else:
            ranking_val_data = val_data

        ranking_val_sampler = NegEdgeSampler(
            destinations=ranking_val_data.destinations,
            full_destinations=full_data.destinations,
            num_neg=args.ranking_val_num_neg,
            device=device,
            seed=2035,
            independent_rng=True,
        )
        ranking_mode = f"RankingValQ{ranking_val_data.n_interactions}"
        ranking_val_filepath = (
            f"time_processed_data/{DATA}/{ranking_mode}_topk_{args.topk}"
            f"_flag_{args.topk_sample_flag}_bs_{val_bs}"
            f"_numneg_{args.ranking_val_num_neg}{negative_cache_suffix}.pkl"
        )
        (
            ranking_val_num_batch,
            ranking_val_delta_times_list,
            ranking_val_all_inds_list,
            ranking_val_batch_size_list,
        ) = process_time_data(
            ranking_mode,
            finder,
            ranking_val_data,
            batch_size=val_bs,
            num_neg=args.ranking_val_num_neg,
            filepath=ranking_val_filepath,
            neg_edge_sampler=ranking_val_sampler,
        )


    num_epo = 0
    t_train = 0.0
    train_memory = 0.0
    t_val = 0.0
    f = f"log_learn_time/{DATA}/{filename}"
    val_time_score_filepath = paths["val"]
    test_time_score_filepath = paths["test"]
    hybrid_val_time_score_filepath = paths["hybrid_val"]


    def load_negative_matrix(mode, data, num_neg, bs):
        parts = []
        num_batches = math.ceil(data.n_interactions / bs)
        for batch_idx in range(num_batches):
            start = batch_idx * bs
            end = min(data.n_interactions, start + bs)
            path = (
                f"{data_directory(DATA, args.data_root)}/cache/batchneg/{mode}_neg{num_neg}_bs{bs}"
                f"{negative_cache_suffix}_batch{batch_idx}.pkl"
            )
            with open(path, "rb") as f:
                parts.append(
                    canonical_negative_matrix(pickle.load(f), end - start, num_neg, path)
                )
        return np.concatenate(parts, axis=0)


    def save_time_scores(path, payload, mode, data, num_neg, bs):
        payload = move_to_cpu(payload)
        atomic_pickle_dump(payload, path)
        metadata = make_score_metadata(
            DATA,
            mode,
            data.sources,
            data.destinations,
            data.timestamps,
            load_negative_matrix(mode, data, num_neg, bs),
            producer="Simple-LP-Time",
            config=vars(args).copy(),
        )
        meta_path = write_score_metadata(path, metadata)
        print(f"{mode} time score has been saved at {path}; metadata={meta_path}")


    for epoch in range(args.num_epochs):
        num_epo += 1

        t_train_epo, mem_train = run(
            model,
            "Train",
            epoch,
            optimizer,
            criterion,
            f,
            num_neg=args.train_num_neg,
            num_batch=train_num_batch,
            delta_times_list=train_delta_times_list,
            all_inds_list=train_all_inds_list,
            batch_size_list=train_batch_size_list,
        )
        t_train += t_train_epo
        train_memory = max(train_memory, mem_train)

        with torch.no_grad():
            val_ap, t_val_epo, val_all_pos_score, val_all_neg_score, mem_val = run(
                model,
                "Val",
                epoch,
                None,
                None,
                f,
                num_neg=1,
                num_batch=val_num_batch,
                delta_times_list=val_delta_times_list,
                all_inds_list=val_all_inds_list,
                batch_size_list=val_batch_size_list,
            )
            t_val += t_val_epo
            selection_value = val_ap
            if ranking_selection:
                (
                    selection_value,
                    ranking_val_time,
                    _,
                    _,
                    ranking_val_memory,
                ) = run(
                    model,
                    "RankingVal",
                    epoch,
                    None,
                    None,
                    f,
                    num_neg=args.ranking_val_num_neg,
                    num_batch=ranking_val_num_batch,
                    delta_times_list=ranking_val_delta_times_list,
                    all_inds_list=ranking_val_all_inds_list,
                    batch_size_list=ranking_val_batch_size_list,
                    k_list=[1, 10],
                )
                t_val += ranking_val_time
                mem_val = max(mem_val, ranking_val_memory)
                print(
                    f"Epoch{epoch}-Selection-{args.selection_metric}: "
                    f"{selection_value:.6f}"
                )
            if scheduler is not None:
                scheduler.step(selection_value)
            print(f"Epoch{epoch}-LR: {optimizer.param_groups[0]['lr']:.8g}")

            if early_stopper.early_stop_check(selection_value):
                break
            else:
                if epoch == early_stopper.best_epoch:
                    torch.save((model.state_dict()), best_checkpoint_path)
                    print("Saving the best model.")
                    val_time_score = val_all_pos_score, val_all_neg_score
                    save_time_scores(
                        val_time_score_filepath, val_time_score, "Val", val_data, 1, val_bs
                    )


    # Always evaluate the validation-selected checkpoint, even when the epoch loop
    # ends before patience is exhausted.
    model_parameters = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(model_parameters)
    model.eval()
    os.replace(best_checkpoint_path, best_model_path)

    if args.lazy_eval_preprocess:
        (
            (
                test_num_batch,
                test_delta_times_list,
                test_all_inds_list,
                test_batch_size_list,
            ),
            (
                hybrid_val_num_batch,
                hybrid_val_delta_times_list,
                hybrid_val_all_inds_list,
                hybrid_val_batch_size_list,
            ),
        ) = prepare_full_evaluation_data()

    with torch.no_grad():
        _, _, hybrid_val_all_pos_score, hybrid_val_all_neg_score, _ = run(
            model,
            "HybridVal",
            early_stopper.best_epoch,
            None,
            None,
            f,
            num_neg=99,
            num_batch=hybrid_val_num_batch,
            delta_times_list=hybrid_val_delta_times_list,
            all_inds_list=hybrid_val_all_inds_list,
            batch_size_list=hybrid_val_batch_size_list,
            k_list=[1, 10],
        )
        save_time_scores(
            hybrid_val_time_score_filepath,
            (hybrid_val_all_pos_score, hybrid_val_all_neg_score),
            "HybridVal",
            val_data,
            99,
            val_bs,
        )

        _, t_test, test_all_pos_score, test_all_neg_score, mem_test = run(
            model,
            "Test",
            early_stopper.best_epoch,
            None,
            None,
            f,
            num_neg=99,
            num_batch=test_num_batch,
            delta_times_list=test_delta_times_list,
            all_inds_list=test_all_inds_list,
            batch_size_list=test_batch_size_list,
            k_list=[1, 10],
        )
        test_time_score = test_all_pos_score, test_all_neg_score
        save_time_scores(
            test_time_score_filepath, test_time_score, "Test", test_data, 99, test_bs
        )

    positive, negative = combine_score_batches(test_all_pos_score, test_all_neg_score, 99)
    ap, mrr, hits = compute_metrics(positive, negative, device, k_list=(1, 10))
    result = {
        'test': {'AP': ap, 'MRR': mrr, 'H1': hits[0], 'H10': hits[1]},
        'best_epoch': early_stopper.best_epoch + 1, 'epochs_ran': num_epo,
        'train_time_s': t_train, 'train_peak_allocated_mb': train_memory,
        'test_infer_time_s': t_test, 'test_peak_allocated_mb': mem_test,
        'score_paths': paths, 'model_path': best_model_path,
    }
    if device.type == 'cpu':
        result.pop('train_peak_allocated_mb')
        result.pop('test_peak_allocated_mb')
        result['train_peak_rss_bytes'] = phase_rss['Train']
        result['test_peak_rss_bytes'] = phase_rss['Test']
    return write_result('time', args, result)


if __name__ == "__main__":
    main(parse_args())
