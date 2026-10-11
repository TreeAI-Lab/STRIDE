"""Shared data, temporal encoders, structural state, and score metadata."""
import argparse
import hashlib
import json
import os
import pickle
import random
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb
from numba import types, typed
from numba.experimental import jitclass
from sklearn.metrics import average_precision_score
import psutil

PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_DATASETS = ('Contacts', 'lastfm', 'wikipedia', 'reddit', 'askubuntu', 'superuser', 'wikitalk')


def data_directory(dataset, root='data'):
    path = Path(root)
    return (path if path.is_absolute() else PROJECT_DIR / path) / dataset


def load_best_params(dataset, path=None):
    path = Path(path) if path else PROJECT_DIR / 'configs/best-hyper-params' / f'{dataset}.yaml'
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def configure_args(parser, component, argv=None):
    parser.add_argument('--use_best_hyper_params', action='store_true')
    parser.add_argument('--best_hyper_params_file', default='')
    parser.add_argument('--data_root', default='data')
    if component != 'hybrid':
        parser.add_argument('--use_best_hyper_params_for_hybrid', action='store_true',
                            help='Load Hybrid-paired component parameters instead of standalone best parameters.')
    initial, _ = parser.parse_known_args(argv)
    paired = component != 'hybrid' and initial.use_best_hyper_params_for_hybrid
    if initial.use_best_hyper_params or initial.best_hyper_params_file or paired:
        preset = load_best_params(initial.dataset_name, initial.best_hyper_params_file or None)
        values = preset[component]
        if paired:
            values = preset.get('hybrid_components', {}).get(component, values)
        keys = {action.dest for action in parser._actions}
        unknown = set(values) - keys
        if unknown:
            raise ValueError(f'Unknown {component} preset keys: {sorted(unknown)}')
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def component_signature(args):
    ignored = {'gpu', 'data_root', 'use_best_hyper_params', 'use_best_hyper_params_for_hybrid', 'best_hyper_params_file', 'result_json'}
    if getattr(args, 'eval_batch_size', None) is None:
        ignored.add('eval_batch_size')
    config = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def score_paths(component, args):
    name = component_signature(args)
    root = Path(f'{component}_score_cache') / args.dataset_name
    return {split: str(root / f'{split}_{name}.pkl') for split in ('val', 'test', 'hybrid_val')}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.0


def write_result(component, args, result):
    root = Path('results') / args.dataset_name / component / component_signature(args)
    root.mkdir(parents=True, exist_ok=True)
    result['params'] = vars(args).copy()
    path = root / 'metrics.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Test: ' + ' '.join(f'{key}={value:.6f}' for key, value in result['test'].items()))
    if 'train_peak_allocated_mb' in result:
        train_memory = result['train_peak_allocated_mb']
        test_memory = result['test_peak_allocated_mb']
        memory_kind = 'allocated'
    else:
        train_memory = result['train_peak_rss_bytes'] / 1024**2
        test_memory = result['test_peak_rss_bytes'] / 1024**2
        memory_kind = 'RSS'
    print(f"Train: {result['train_time_s']:.3f}s, {memory_kind}={train_memory:.1f} MiB; "
          f"Test inference: {result['test_infer_time_s']:.3f}s, {memory_kind}={test_memory:.1f} MiB")
    print(f'Results: {path}')
    return result


class PhaseMonitor:
    """Wall time and sampled process RSS; CUDA allocated memory when requested."""
    def __init__(self, device=None):
        self.device = device
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self.stop = threading.Event()

    def sample(self):
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)

    def watch(self):
        while not self.stop.wait(0.1):
            self.sample()

    def __enter__(self):
        if self.device is not None:
            synchronize(self.device)
            reset_peak_memory(self.device)
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            synchronize(self.device)
        self.seconds = time.perf_counter() - self.started
        self.sample()
        self.stop.set()
        self.thread.join()
        self.allocated_mb = peak_memory_mb(self.device) if self.device is not None else 0.0


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0

    def sample(self, ratio):
        data_size = self.n_interactions
        sample_size = int(ratio * data_size)
        sample_inds = random.sample(range(data_size), sample_size)
        sample_inds = np.sort(sample_inds)
        sources = self.sources[sample_inds]
        destination = self.destinations[sample_inds]
        timestamps = self.timestamps[sample_inds]
        edge_idxs = self.edge_idxs[sample_inds]
        labels = self.labels[sample_inds]
        return Data(sources, destination, timestamps, edge_idxs, labels)


def get_data_transductive(dataset_name, use_validation=False, data_root="data"):
    graph_df = pd.read_csv(data_directory(dataset_name, data_root) / f"ml_{dataset_name}.csv")

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values.astype(np.float64)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    n_edges = len(sources)


    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time

    val_mask = (
        np.logical_and(timestamps <= test_time, timestamps > val_time)
        if use_validation
        else test_mask
    )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )

    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.n_interactions, full_data.n_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes
        )
    )

    return full_data, train_data, val_data, test_data, n_total_unique_nodes, n_edges

def set_random_seed(seed):
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False








def compute_metrics(
    pred_pos, pred_neg, device, k_list=(1, 10)
):
    with torch.no_grad():
        pred_pos = pred_pos.detach().reshape(-1).cpu()
        pred_neg = pred_neg.detach().reshape(-1).cpu()
        y_pred = torch.cat([pred_pos, pred_neg], dim=0)
        y_true = (
            torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
        )
        if not torch.isfinite(y_pred).all():
            raise ValueError("Non-finite prediction scores")
        ap = average_precision_score(y_true, y_pred)

        num_edge = int(pred_pos.numel())
        if num_edge == 0 or pred_neg.numel() % num_edge != 0:
            raise ValueError(
                f"invalid metric shapes: pos={tuple(pred_pos.shape)} "
                f"neg={tuple(pred_neg.shape)}"
            )
        num_neg = pred_neg.numel() // num_edge
        # Scores are stored negative-major: all queries for negative 1, then
        # all queries for negative 2, etc.
        neg_matrix = pred_neg.reshape(num_neg, num_edge).T
        # Strict/max rank: every negative tied with the positive is ahead.
        ranks = 1 + torch.sum(neg_matrix >= pred_pos[:, None], dim=1)
        mrr = float(torch.mean(1.0 / ranks.float()).item())
        hr_list = [float(torch.mean((ranks <= k).float()).item()) for k in k_list]

        return ap, mrr, hr_list


class EarlyStopMonitor(object):
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-8):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class NegEdgeSampler:
    def __init__(
        self,
        destinations: np.ndarray,
        full_destinations: np.ndarray,
        num_neg: int,
        device: str = 'cuda',
        seed: int = 2024,
        independent_rng: bool = True,
    ):
        self.seed = seed
        self.independent_rng = independent_rng
        self.destinations = destinations
        self.full_destinations = full_destinations
        self.num_neg = num_neg
        self.device = device

        assert num_neg <= len(full_destinations) - 1, f"num_neg should be <= {len(full_destinations) - 1}"

        self.unique_destinations = torch.tensor(np.unique(full_destinations), dtype=torch.long, device=self.device)
        self.num_dst_node = len(destinations)

        if self.independent_rng:
            # Keep negative sampling independent from model initialization and
            # from other samplers. This is the reproducible V2 behavior.
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
        else:
            # Optional compatibility path for V1, where every sampler reset
            # PyTorch's global RNG. The final sampler consequently controlled
            # model initialization in train_time.py.
            self.generator = None
            torch.manual_seed(self.seed)
    
    def sample(self, batch_destinations: np.ndarray):
        batch_size = len(batch_destinations)

        batch_destinations = torch.tensor(batch_destinations, dtype=torch.long, device=self.device)

        all_choices = self.unique_destinations.unsqueeze(0).expand(batch_size, -1)

        mask = (all_choices != batch_destinations.unsqueeze(1))

        all_choices = all_choices[mask].view(batch_size, -1) # [batch_size, len(self.unique_destinations)-1]

        random_scores = torch.rand(
            all_choices.shape,
            dtype=torch.float,
            device=self.device,
            generator=self.generator,
        )
        _, topk_indices = torch.topk(random_scores, self.num_neg, dim=1)

        batch_neg_samples = torch.gather(all_choices, 1, topk_indices)

        return batch_neg_samples


nb_tppr_dict2 = nb.typed.Dict.empty(
    key_type=types.int64,
    value_type=types.float64,
)
nb_dict_type2 = nb.typeof(nb_tppr_dict2)
list_dict2 = typed.List()
list_dict2.append(nb_tppr_dict2)
list_list_dict2 = typed.List()
list_list_dict2.append(list_dict2)

structure_spec = [
    ("num_nodes", types.int64),
    ("k", types.int64),
    ("n_tppr", types.int64),
    ("alpha_list", types.List(types.float64)),
    ("beta_list", types.List(types.float64)),
    ("norm_list", types.Array(types.float64, 2, "C")),
    ("PPR_list", nb.typeof(list_list_dict2)),
    ("sim", nb.typeof("123")),
]


@jitclass(structure_spec)
class StructureState:
    def __init__(self, num_nodes, k, alpha, beta, sim, n_tppr=1):
        self.num_nodes = num_nodes
        self.k = k
        self.n_tppr = n_tppr
        self.alpha_list = [alpha]
        self.beta_list = [beta]
        self.sim = sim
        self.reset_tppr()

    def reset_tppr(self):
        PPR_list = typed.List()
        for _ in range(self.n_tppr):
            temp_PPR_list = typed.List()
            for _ in range(self.num_nodes):
                tppr_dict = nb.typed.Dict.empty(
                    key_type=types.int64,
                    value_type=types.float64,
                )
                temp_PPR_list.append(tppr_dict)
            PPR_list.append(temp_PPR_list)
        self.norm_list = np.zeros((self.n_tppr, self.num_nodes), dtype=np.float64)
        self.PPR_list = PPR_list

    def get_similarity(self, tppr_index, source, target):
        PPR_list = self.PPR_list[tppr_index]
        source_tppr = PPR_list[source]
        target_tppr = PPR_list[target]
        similarity = 0

        for key, weight in source_tppr.items():
            if self.sim == "mul_wo_norm":
                if key in target_tppr:
                    similarity += weight * target_tppr[key]
        return similarity

    def precompute_link_prediction(self, source_nodes, num_neg, score_events=True):
        n_edges = len(source_nodes) // (2 + num_neg)
        scores = np.zeros(n_edges * (num_neg + 1))

        for index0, alpha in enumerate(self.alpha_list):
            beta = self.beta_list[index0]
            norm_list = self.norm_list[index0]
            PPR_list = self.PPR_list[index0]

            for i in range(n_edges):
                source = source_nodes[i]
                target = source_nodes[i + n_edges]
                sim = self.get_similarity(index0, source, target) if score_events else 0.0
                scores[i] = sim

                for neg_index in range(num_neg):
                    fake = source_nodes[i + (neg_index + 2) * n_edges]
                    sim = self.get_similarity(index0, source, fake)
                    scores[i + (neg_index + 1) * n_edges] = sim

                pairs = (
                    [(source, target), (target, source)]
                    if source != target
                    else [(source, target)]
                )

                for index, pair in enumerate(pairs):
                    s1 = pair[0] # node 1
                    s2 = pair[1] # node 2

                    if norm_list[s1] == 0:
                        t_s1_PPR = nb.typed.Dict.empty(
                            key_type=types.int64,
                            value_type=types.float64,
                        )
                        scale_s2 = 1 - alpha
                    else:
                        t_s1_PPR = PPR_list[s1].copy()
                        last_norm = norm_list[s1]
                        new_norm = last_norm * beta + beta
                        scale_s1 = last_norm / new_norm * beta
                        scale_s2 = beta / new_norm * (1 - alpha)
                        for key, value in t_s1_PPR.items():
                            t_s1_PPR[key] = value * scale_s1

                    if norm_list[s2] == 0:
                        t_s1_PPR[s2] = scale_s2 * alpha if alpha != 0 else scale_s2
                    else:

                        s2_PPR = PPR_list[s2]
                        for key, value in s2_PPR.items():
                            if key in t_s1_PPR:
                                t_s1_PPR[key] += value * scale_s2
                            else:
                                t_s1_PPR[key] = value * scale_s2

                        if s2 in t_s1_PPR:
                            t_s1_PPR[s2] += scale_s2 * alpha if alpha != 0 else scale_s2
                        else:
                            t_s1_PPR[s2] = scale_s2 * alpha if alpha != 0 else scale_s2

                    updated_tppr = nb.typed.Dict.empty(
                        key_type=types.int64, value_type=types.float64
                    )

                    tppr_size = len(t_s1_PPR)
                    if tppr_size <= self.k:
                        updated_tppr = t_s1_PPR
                    else:
                        keys = list(t_s1_PPR.keys())
                        values = np.array(list(t_s1_PPR.values()))
                        inds = np.argsort(values)[-self.k :]
                        for ind in inds:
                            key = keys[ind]
                            value = values[ind]
                            updated_tppr[key] = value

                    if index == 0:
                        new_s1_PPR = updated_tppr
                    else:
                        new_s2_PPR = updated_tppr

                if source != target:
                    PPR_list[source] = new_s1_PPR
                    PPR_list[target] = new_s2_PPR
                    norm_list[source] = norm_list[source] * beta + beta
                    norm_list[target] = norm_list[target] * beta + beta
                else:
                    PPR_list[source] = new_s1_PPR
                    norm_list[source] = norm_list[source] * beta + beta

        return scores


class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t))
        return output


class FeedForward(nn.Module):
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EventMLPBlock(nn.Module):
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')


        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_update(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)

        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_update(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_update(x)

        if 'channel' in self.module_spec:
            x = x + self.channel_update(x)
        return x


class EventMLPEncoder(nn.Module):
    def __init__(self, per_graph_size, time_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False,
                 device = 'cpu', masked_pooling=False,
                 use_time_summary=False, log_time_input=False,
                ):
        super().__init__()

        self.per_graph_size = per_graph_size
        self.num_layers = num_layers
        self.time_encoder = TimeEncoder(time_channels)
        self.feat_encoder = nn.Linear(time_channels, time_channels) 
        self.layernorm = nn.LayerNorm(time_channels)
        self.mlp_head = nn.Linear(time_channels,time_channels)
        self.event_blocks = torch.nn.ModuleList()
        self.time_channels = time_channels
        self.device = device
        self.masked_pooling = masked_pooling
        self.use_time_summary = use_time_summary
        self.log_time_input = log_time_input

        if self.use_time_summary:
            # log(count), count ratio, min/mean/max/std log-delta.  These
            # preserve information that the original fixed-topk mean loses.
            self.summary_norm = nn.LayerNorm(6)
            self.summary_projector = nn.Sequential(
                nn.Linear(6, time_channels),
                nn.GELU(),
                nn.Linear(time_channels, time_channels),
            )

        for ell in range(num_layers):
            if module_spec is None:
                self.event_blocks.append(
                    EventMLPBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer).to(self.device)
                    )
            else:
                self.event_blocks.append(
                    EventMLPBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer).to(self.device)
                    )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.event_blocks:
            layer.reset_parameters()
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()
        if self.use_time_summary:
            self.summary_norm.reset_parameters()
            for module in self.summary_projector:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()


    def forward(self, delta_times, inds, batch_size):
        encoded_delta_times = delta_times
        if self.log_time_input:
            encoded_delta_times = torch.log1p(torch.clamp(delta_times, min=0))

        time_encodings = self.time_encoder(encoded_delta_times)
        time_encodings = self.feat_encoder(time_encodings)

        x = torch.zeros((batch_size * self.per_graph_size, self.time_channels)).to(self.device)
        x[inds] = time_encodings # add 0 gaps if node N's num_nei < topk, add to max_edge

        mask = torch.zeros(
            (batch_size * self.per_graph_size, 1),
            dtype=x.dtype,
            device=self.device,
        )
        mask[inds] = 1
        mask = mask.reshape(batch_size, self.per_graph_size, 1)
        x = x.reshape(batch_size, self.per_graph_size, self.time_channels)

        for i in range(self.num_layers):
            x = self.event_blocks[i](x)
            if self.masked_pooling:
                x = x * mask
            
        x = self.layernorm(x)
        if self.masked_pooling:
            x = x * mask
            x = torch.sum(x, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        else:
            x = torch.mean(x, dim=1)

        if self.use_time_summary:
            dense_delta = torch.zeros(
                (batch_size * self.per_graph_size, 1),
                dtype=delta_times.dtype,
                device=self.device,
            )
            dense_delta[inds] = delta_times
            dense_delta = torch.log1p(torch.clamp(dense_delta, min=0)).reshape(
                batch_size, self.per_graph_size
            )
            valid = mask.squeeze(-1)
            count = torch.sum(valid, dim=1)
            safe_count = torch.clamp(count, min=1.0)
            mean = torch.sum(dense_delta * valid, dim=1) / safe_count
            variance = torch.sum(
                ((dense_delta - mean[:, None]) ** 2) * valid, dim=1
            ) / safe_count
            minimum = dense_delta.masked_fill(valid == 0, float("inf")).min(dim=1).values
            maximum = dense_delta.masked_fill(valid == 0, float("-inf")).max(dim=1).values
            has_history = count > 0
            minimum = torch.where(has_history, minimum, torch.zeros_like(minimum))
            maximum = torch.where(has_history, maximum, torch.zeros_like(maximum))
            summary = torch.stack(
                [
                    torch.log1p(count),
                    count / float(self.per_graph_size),
                    minimum,
                    mean,
                    maximum,
                    torch.sqrt(torch.clamp(variance, min=0)),
                ],
                dim=1,
            )
            x = x + self.summary_projector(self.summary_norm(summary))

        x = self.mlp_head(x)
        return x


class ScoreHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.linear1 = torch.nn.Linear(2, dim)
        self.linear2 = torch.nn.Linear(dim, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, h):
        num_edge = h.shape[0] // 2
        x = self.linear1(h)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x[:num_edge], x[num_edge:]


class EdgePredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.src_fc = torch.nn.Linear(dim, 100)
        self.dst_fc = torch.nn.Linear(dim, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, num_neg=1):
        num_edge = h.shape[0] // (num_neg + 2)
        
        # encode pos embeds
        h_src = self.src_fc(h[:num_edge])

        # encode pos and neg dest embeds
        h_pos_dst = self.dst_fc(h[num_edge:2*num_edge])

        h_neg_dst = self.dst_fc(h[2*num_edge:(num_neg+2)*num_edge]) # [num_neg*num_edge, embed_dim]
        h_src_repeated = h_src.repeat(num_neg, 1) # [num_neg*num_edge, embed_dim]

        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)

        h_neg_edge = torch.nn.functional.relu(h_src_repeated + h_neg_dst)

        # [batch_size, 1]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class TimeModel(nn.Module):
    def __init__(self, encoder_config, edge_predictor_configs):
        super(TimeModel, self).__init__()

        self.dim = edge_predictor_configs['dim']
        self.edge_predictor = EdgePredictor(**edge_predictor_configs)
        self.base_model = EventMLPEncoder(**encoder_config)
        self.combiner = ScoreHead(10)
        self.reset_parameters()            

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        self.combiner.reset_parameters()

    def forward(self, delta_times, all_inds, batch_size, num_neg=1):  
        x = self.base_model(delta_times, all_inds, batch_size)
        pred_pos, pred_neg = self.edge_predictor(x, num_neg)
        return pred_pos, pred_neg

l_int = typed.List()
l_float = typed.List()
a_int = np.array([1, 2], dtype=np.int32)
a_float = np.array([1, 2], dtype=np.float64)
l_int.append(a_int)
l_float.append(a_float)
spec = [
    ("node_to_neighbors", nb.typeof(l_int)),
    ("node_to_edge_idxs", nb.typeof(l_int)),
    ("node_to_edge_timestamps", nb.typeof(l_float)),
]

@jitclass(spec)
class NeighborFinder:
    def __init__(self, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps):
        self.node_to_neighbors = node_to_neighbors
        self.node_to_edge_idxs = node_to_edge_idxs
        self.node_to_edge_timestamps = node_to_edge_timestamps

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i],
        )

    def get_clean_delta_times(self, source_nodes, timestamps, n_neighbors, topk_sample_flag="last"):
        if topk_sample_flag not in ["last", "early", "random"]:
            raise ValueError("TopK sample flag must be in ['last', 'early', 'random']")

        if topk_sample_flag == "random":
            np.random.seed(2024)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        delta_times = np.zeros(len(source_nodes) * tmp_n_neighbors, dtype=np.float32)
        n_edges = np.zeros(len(source_nodes), dtype=np.int32)
        cum_sum = 0
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            _, _, edge_times = self.find_before(source_node, timestamp)
            n_ngh = len(edge_times)
            if n_ngh > 0:
                if topk_sample_flag == "last":
                    selected_times = edge_times[-n_neighbors:][
                        ::-1
                    ] # delta time from last to early
                elif topk_sample_flag == "early":
                    selected_times = edge_times[
                        :n_neighbors
                    ] # delta time from early to last
                elif topk_sample_flag == "random":
                    if n_ngh <= n_neighbors:
                        selected_times = edge_times
                    else:
                        selected_indices = np.random.choice(
                            n_ngh, n_neighbors, replace=False
                        )
                        selected_times = edge_times[selected_indices]
                        selected_times = np.sort(
                            selected_times
                        ) # delta time from early to last

                n_ngh = len(selected_times)
                delta_times[cum_sum : cum_sum + n_ngh] = timestamp - selected_times

            n_edges[i] = n_ngh
            cum_sum += n_ngh
        return delta_times, n_edges, cum_sum




def get_neighbor_finder(data):
    max_node_idx = max(data.sources.max(), data.destinations.max())
    adj_list = [[] for _ in range(max_node_idx + 1)]

    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    node_to_neighbors = typed.List()
    node_to_edge_idxs = typed.List()
    node_to_edge_timestamps = typed.List()

    for neighbors in adj_list:
        sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
        node_to_neighbors.append(
            np.array([x[0] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_idxs.append(
            np.array([x[1] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_timestamps.append(
            np.array([x[2] for x in sorted_neighbors], dtype=np.float64)
        )

    return NeighborFinder(node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def atomic_pickle_dump(value, path):
    """Write pickle caches without leaving a partial final file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "wb") as file_handle:
        pickle.dump(value, file_handle)
    os.replace(temporary_path, path)

SCORE_META_FORMAT = "eagle_tg_candidate_score_meta_v1"


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def canonical_negative_matrix(value, rows, num_neg, path=""):
    """Return the canonical [query, negative] candidate-ID matrix.

    Old Time caches with q>1 were written as a flattened negative-major
    vector.  That layout is unambiguous for this repository and is normalized
    in memory here; all newly written batch caches are two-dimensional.
    """

    rows = int(rows)
    num_neg = int(num_neg)
    raw = to_numpy(value)
    if raw.ndim == 2 and raw.shape == (rows, num_neg):
        return np.ascontiguousarray(raw, dtype=np.int64)
    if raw.ndim == 1 and raw.size == rows * num_neg:
        if num_neg == 1:
            matrix = raw.reshape(rows, 1)
        else:
            matrix = raw.reshape(num_neg, rows).T
        return np.ascontiguousarray(matrix, dtype=np.int64)
    raise ValueError(
        f"negative cache has incompatible shape at {path or '<memory>'}: "
        f"got={raw.shape}, expected={(rows, num_neg)}"
    )


def _hash_array(hasher, name, value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    hasher.update(str(name).encode("utf-8"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())


def candidate_fingerprint(sources, destinations, timestamps, negatives):
    hasher = hashlib.sha256()
    _hash_array(hasher, "sources", sources, np.int64)
    _hash_array(hasher, "destinations", destinations, np.int64)
    _hash_array(hasher, "timestamps", timestamps, np.float64)
    _hash_array(hasher, "negatives", negatives, np.int64)
    return hasher.hexdigest()


def make_score_metadata(
    dataset,
    split,
    sources,
    destinations,
    timestamps,
    negatives,
    producer,
    config,
):
    negatives = canonical_negative_matrix(
        negatives, len(sources), np.asarray(negatives).shape[-1], path="metadata"
    )
    return {
        "format": SCORE_META_FORMAT,
        "dataset": str(dataset),
        "split": str(split),
        "producer": str(producer),
        "num_queries": int(len(sources)),
        "num_negatives": int(negatives.shape[1]),
        "score_layout": "positive_vector_and_negative_major_vector",
        "candidate_fingerprint": candidate_fingerprint(
            sources, destinations, timestamps, negatives
        ),
        "config": dict(config),
    }


def metadata_path(score_path):
    return str(score_path) + ".meta.json"


def write_score_metadata(score_path, metadata):
    path = metadata_path(score_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp, path)
    return path


def load_score_metadata(score_path):
    path = metadata_path(score_path)
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("format") != SCORE_META_FORMAT:
        raise ValueError(f"unsupported score metadata format in {path}")
    return metadata
