import dgl
import numpy as np
import numpy
import argparse
import time
from dataset import Dataset
from paths import PRETRAINED_ROOT, cached_dataset, raw_dataset, pretrained_embedding
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
import random
from ppr import *
from EnhancedBernsteinNet import *
from config import DEFAULT_DATASET, apply_dataset_defaults
import os.path as osp
import os

# Device is selected after parsing the dataset-specific configuration.
DEVICE = torch.device('cpu')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print('set seed for random numpy and torch')
def load_pretrained_embedding(dataset_name, seed=0):
    """Load the pretrained embedding."""
    embedding_path = pretrained_embedding(dataset_name, seed)
    if not osp.exists(embedding_path):
        print(f"Warning: Pretrained embedding file not found at {embedding_path}")
        return None
    return torch.load(embedding_path)

# Select the F1 averaging method by dataset.
def calculate_f1(labels, preds, dataset_name, f1_average='auto'):
    """Select the F1 averaging method based on the dataset name."""
    if f1_average != 'auto':
        return f1_score(labels, preds, average=f1_average)
    if dataset_name == 'Twibot-20':
        # Twibot-20 uses binary F1.
        return f1_score(labels, preds, average='binary')
    else:
        # Twibot-22, MGTAB, Tfinance, and Tsocial use macro F1.
        return f1_score(labels, preds, average='macro')

def train(model, g, args, use_fp16=False):
    features = g.ndata['feature']
    labels = g.ndata['label']
    device = labels.device
    if use_fp16 and device.type == 'cuda':
        features = features.half()  # fp16 feature = ½ DGL message memory
    index = list(range(len(labels)))
    dataset_name = args.dataset
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    # sklearn train_test_split needs CPU – convert labels to CPU temporarily
    labels_cpu = labels.cpu() if labels.is_cuda else labels

    split_mode = getattr(args, 'split_mode', 'default')
    split_seed = args.split_seed if getattr(args, 'split_seed', None) is not None else args.seed
    mgtab_val_ratio = getattr(args, 'mgtab_val_ratio', 0.2)
    if dataset_name == 'MGTAB' and not (0.0 < mgtab_val_ratio < 0.3):
        raise ValueError('--mgtab_val_ratio must be in (0, 0.3) because MGTAB train ratio is fixed at 0.7')

    if dataset_name in ['tfinance', 'tsocial']:
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index],
                                                                stratify=labels_cpu[index],
                                                                train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)
    elif dataset_name == 'Twibot-20' and split_mode == 'stratified':
        idx_train, idx_rest, y_train, y_rest = train_test_split(index[:11826], labels_cpu[:11826],
                                                                stratify=labels_cpu[:11826],
                                                                train_size=8278,
                                                                random_state=split_seed, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=1183,
                                                                random_state=split_seed, shuffle=True)
    elif dataset_name == 'Twibot-20':
        idx_train = torch.arange(0, 8278)
        idx_valid = torch.arange(8278, 8278 + 2365)
        idx_test = torch.arange(8278 + 2365, 8278 + 2365 + 1183)
    elif dataset_name == 'MGTAB' and split_mode == 'stratified':
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index],
                                                                stratify=labels_cpu[index],
                                                                train_size=0.7,
                                                                random_state=split_seed, shuffle=True)
        mgtab_test_ratio = 1.0 - 0.7 - mgtab_val_ratio
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=mgtab_test_ratio / 0.3,
                                                                random_state=split_seed, shuffle=True)
    elif dataset_name == 'MGTAB' and split_mode == 'random':
        num_nodes = len(labels)
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(num_nodes)
        train_end = int(0.7 * num_nodes)
        valid_end = int((0.7 + mgtab_val_ratio) * num_nodes)
        idx_train = torch.from_numpy(perm[:train_end]).long()
        idx_valid = torch.from_numpy(perm[train_end:valid_end]).long()
        idx_test = torch.from_numpy(perm[valid_end:]).long()
    elif dataset_name == 'MGTAB':
        num_nodes = len(labels)
        train_end = int(0.7 * num_nodes)
        valid_end = int((0.7 + mgtab_val_ratio) * num_nodes)
        idx_train = torch.arange(0, train_end)
        idx_valid = torch.arange(train_end, valid_end)
        idx_test = torch.arange(valid_end, num_nodes)
    elif dataset_name == 'Twibot-22':
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        idx_train = train_mask.nonzero(as_tuple=True)[0]
        idx_valid = val_mask.nonzero(as_tuple=True)[0]
        idx_test = test_mask.nonzero(as_tuple=True)[0]
    else:
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index],
                                                                stratify=labels_cpu[index],
                                                                train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=0.67,
                                                                random_state=2, shuffle=True)

    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)

    train_mask[idx_train] = True
    val_mask[idx_valid] = True
    test_mask[idx_test] = True
    if dataset_name == 'MGTAB':
        print(f'MGTAB split mode={split_mode}, split_seed={split_seed}, val_ratio={mgtab_val_ratio:.2f}')
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    # Compute homophily ratios and set targets for Homophily-Aware MLPs
    model.set_homophily(labels, train_mask)
    for etype in model.etypes:
        print(f'  [{etype}] homophily={model._homophily[etype]:.4f}  omega_target={model._omega_target[etype]:.4f}')
    if args.auto_focal_alpha and hasattr(model, 'set_focal_alpha_from_labels'):
        alpha = model.set_focal_alpha_from_labels(labels, train_mask)
        print(f'auto focal alpha for positive class: {alpha:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, final_tacc, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    final_thres = 0.5
    final_probs_np = None
    final_labels_np = None

    lambda_freq = getattr(model, 'lambda_freq', 0.2)
    print(f'Focal Loss(alpha={model.focal_alpha:.4f}) + L_freq(lambda={lambda_freq}), lr={args.lr}, wd={args.weight_decay}')
    time_start = time.time()
    patience = args.patience
    patience_counter = 0

    for e in range(args.epoch):
        model.train()
        # autocast for forward: fp16 matmul saves memory, avoids dtype mismatch
        # Loss computed outside autocast in fp32 for numerical stability
        with torch.amp.autocast('cuda', enabled=use_fp16):
            if 'Twibot' in dataset_name:
                logits = model(None)
            else:
                logits = model(features)
        logits = logits.float()  # back to fp32 for stable loss

        focal = model.focal_loss(logits[train_mask], labels[train_mask])
        freq = model.get_freq_loss()
        loss = focal + lambda_freq * freq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_fp16):
                if 'Twibot' in dataset_name:
                    logits_eval = model(None)
                else:
                    logits_eval = model(features)
            probs = logits_eval.float().softmax(1)

        # sklearn metrics require CPU numpy
        labels_np = labels.cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        f1, thres = get_best_f1(labels_np[val_mask.cpu().numpy()], probs_np[val_mask.cpu().numpy()],
                                args.dataset, getattr(args, 'f1_average', 'auto'))

        preds_np = np.zeros_like(labels_np)
        preds_np[probs_np[:, 1] > thres] = 1
        tacc = accuracy_score(labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()])
        trec = recall_score(labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()])
        tpre = precision_score(labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()])
        tmf1 = calculate_f1(labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()],
                            args.dataset, getattr(args, 'f1_average', 'auto'))
        tauc = roc_auc_score(labels_np[test_mask.cpu().numpy()], probs_np[test_mask.cpu().numpy()][:, 1])

        if best_f1 < f1:
            best_f1 = f1
            patience_counter = 0  # Reset the patience counter.
            final_tacc = tacc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_thres = thres
            final_probs_np = probs_np.copy()
            final_labels_np = labels_np.copy()
        else:
            patience_counter += 1  # Increment the patience counter.
            
        print('Epoch {}, focal: {:.4f}, freq: {:.4f}, loss: {:.4f}, val mf1: {:.4f}, thres: {:.4f}, (best {:.4f})'.format(
            e, focal.item(), freq.item(), loss.item(), f1, thres, best_f1))

        del logits, logits_eval, probs, probs_np, preds_np, focal, freq, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if patience_counter >= patience:  # Check whether patience is exhausted.
            print('Early stopping at epoch {}, no improvement in last {} epochs'.format(e, patience))
            break  # Stop training.

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Best threshold: {:.4f}'.format(final_thres))
    print('Test: ACC {:.2f} REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_tacc*100,
                                                                                final_trec*100,
                                                                                final_tpre*100,
                                                                                final_tmf1*100,
                                                                                final_tauc*100))
    if getattr(args, 'save_probs_path', '') and final_probs_np is not None:
        save_dir = osp.dirname(args.save_probs_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            args.save_probs_path,
            dataset=np.array(args.dataset),
            labels=final_labels_np,
            probs=final_probs_np,
            train_mask=train_mask.cpu().numpy(),
            val_mask=val_mask.cpu().numpy(),
            test_mask=test_mask.cpu().numpy(),
            threshold=np.array(final_thres),
            val_f1=np.array(best_f1),
            test_acc=np.array(final_tacc),
            test_rec=np.array(final_trec),
            test_pre=np.array(final_tpre),
            test_mf1=np.array(final_tmf1),
            test_auc=np.array(final_tauc),
        )
        print(f'Saved best-epoch probabilities to {args.save_probs_path}')
    return final_tmf1, final_tauc


# threshold adjusting for best macro f1  (inputs: CPU numpy arrays)
def get_best_f1(labels, probs, dataset_name, f1_average='auto'):
    labels = labels.astype(np.int64, copy=False)
    scores = probs[:, 1]
    n = labels.shape[0]
    if n == 0:
        return 0.0, 0.5

    order = np.argsort(-scores, kind='mergesort')
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    pos_total = sorted_labels.sum()
    neg_total = n - pos_total

    if pos_total == 0 or neg_total == 0:
        return 0.0, 0.5

    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), n - 1]
    ks = group_ends + 1
    tp = np.cumsum(sorted_labels)[group_ends].astype(np.float64)
    fp = ks.astype(np.float64) - tp
    fn = pos_total - tp
    tn = neg_total - fp

    # Include the all-negative prediction case.
    ks = np.r_[0, ks]
    tp = np.r_[0.0, tp]
    fp = np.r_[0.0, fp]
    fn = np.r_[float(pos_total), fn]
    tn = np.r_[float(neg_total), tn]

    pos_f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                       out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    if f1_average == 'binary' or (f1_average == 'auto' and dataset_name == 'Twibot-20'):
        f1_values = pos_f1
    else:
        neg_f1 = np.divide(2 * tn, 2 * tn + fp + fn,
                           out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
        f1_values = 0.5 * (pos_f1 + neg_f1)

    best_idx = int(np.argmax(f1_values))
    best_k = int(ks[best_idx])
    if best_k == 0:
        best_thre = float(sorted_scores[0])
    elif best_k >= n:
        best_thre = float(sorted_scores[-1] - 1e-12)
    else:
        best_thre = float((sorted_scores[best_k - 1] + sorted_scores[best_k]) / 2.0)
    return float(f1_values[best_idx]), best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wavelet')
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Dataset for this model (Twibot-20/Twibot-22/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=None, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=None, help="Hidden layer dimension")
    parser.add_argument("--homo", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=None, help="Running times")
    parser.add_argument("--add_edges", action='store_true', default=None, help="Use PPR-added edges")
    parser.add_argument("--no_add_edges", dest="add_edges", action='store_false', help="Disable PPR-added edges")
    parser.add_argument("--ppr_topk", type=int, default=None, help="PPR top-k neighbors")
    parser.add_argument("--pretrain_seed", type=int, default=None, help="Pretrained embedding seed")
    parser.add_argument("--multi_hop", action='store_true', help="Use multi-hop PPR")
    parser.add_argument("--hop_count", type=int, default=2, help="Number of PPR hops (default: 2)")
    parser.add_argument("--fp16", action='store_true', default=None, help="Use mixed precision training")
    parser.add_argument("--no_fp16", action='store_true', help="Disable mixed precision")
    parser.add_argument("--cpu", action='store_true', help="Train on the CPU")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--lambda_freq", type=float, default=None, help="Frequency loss weight")
    parser.add_argument("--K", type=int, default=None, help="Bernstein polynomial order")
    parser.add_argument("--num_scales", type=int, default=None, help="Number of Gaussian windows")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Bernstein layers")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal loss gamma")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Random seed for stratified train/val/test split; defaults to --seed")
    parser.add_argument("--split_mode", type=str, default=None,
                        choices=["default", "stratified", "random"],
                        help="Dataset split mode for datasets without official masks")
    parser.add_argument("--f1_average", type=str, default="auto",
                        choices=["auto", "binary", "macro"],
                        help="F1 averaging mode for validation threshold and test reporting")
    parser.add_argument("--relation_fusion", type=str, default=None,
                        choices=["mean_final_semantic", "semantic", "homophily", "edge_weight", "mean"],
                        help="How to combine relation outputs in each layer")
    parser.add_argument("--homophily_temperature", type=float, default=8.0,
                        help="Softmax temperature for homophily relation fusion")
    parser.add_argument("--freq_loss_weighting", type=str, default="uniform",
                        choices=["uniform", "homophily", "edge_weight"],
                        help="Relation weighting mode inside frequency regularization")
    parser.add_argument("--use_edge_weight", action="store_true",
                        help="Use edge weight data in message passing when graph edges provide it")
    parser.add_argument("--final_residual_weight", type=float, default=None,
                        help="Blend final relation fusion with residual node representation; 0 keeps old behavior")
    parser.add_argument("--mgtab_feature_mode", type=str, default=None,
                        choices=["raw", "pretrained", "concat"],
                        help="MGTAB node feature source: raw features, pretrained embedding, or concatenation")
    parser.add_argument("--mgtab_feature_seed", type=int, default=None,
                        help="Which MGTAB pretrained embedding seed to use for feature replacement/concat")
    parser.add_argument("--mgtab_feature_seeds", type=str, default="",
                        help="Comma-separated MGTAB pretrained embedding seeds; overrides --mgtab_feature_seed")
    parser.add_argument("--mgtab_feature_agg", type=str, default="concat",
                        choices=["concat", "mean"],
                        help="How to combine multiple MGTAB pretrained embeddings")
    parser.add_argument("--mgtab_feature_norm", action="store_true", default=None,
                        help="Z-score normalize selected MGTAB node features before training")
    parser.add_argument("--no_mgtab_feature_norm", dest="mgtab_feature_norm", action="store_false",
                        help="Disable MGTAB feature normalization")
    parser.add_argument("--mgtab_embedding_scale", type=float, default=None,
                        help="Scale MGTAB pretrained embedding feature block after optional normalization")
    parser.add_argument("--mgtab_feature_clip", type=float, default=None,
                        help="MGTAB-only: clip z-score normalized features to +/- this value; 0 disables")
    parser.add_argument("--mgtab_val_ratio", type=float, default=None,
                        help="MGTAB-only validation ratio after 70% train split; default 0.2 keeps the old 70/20/10 protocol")
    parser.add_argument("--focal_alpha", type=float, default=None, help="Positive-class focal alpha before auto adjustment")
    parser.add_argument("--no_auto_focal_alpha", dest="auto_focal_alpha", action="store_false", default=None,
                        help="Disable train-split based focal alpha")
    parser.add_argument("--no_pretrained_skip", action="store_true",
                        help="Disable Twibot pretrained feature skip branch")
    parser.add_argument("--reuse_ppr_graph", action="store_true", default=None,
                        help="Load existing dataset/<name>_new_topkK/graph.bin if present")
    parser.add_argument("--no_reuse_ppr_graph", dest="reuse_ppr_graph", action="store_false",
                        help="Do not reuse an existing PPR graph")
    parser.add_argument("--edge_types", type=str, default=None,
                        help="Comma-separated graph edge types to use, e.g. _E0,_E1")
    parser.add_argument("--save_probs_path", type=str, default="",
                        help="Optional path to save best-validation-epoch labels/probabilities/masks for ensembling")
    parser.add_argument("--print_config", action="store_true",
                        help="Print the resolved dataset preset and exit")
    parser.set_defaults(auto_focal_alpha=None)

    args = parser.parse_args()
    args = apply_dataset_defaults(args)
    if args.print_config:
        print(args)
        raise SystemExit(0)
    set_seed(args.seed)
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    h_feats = args.hid_dim

    # ---- Device selection ----
    if args.cpu:
        DEVICE = torch.device('cpu')
    elif torch.cuda.is_available():
        # Use the GPU with most free memory
        best_gpu, best_free = 0, 0
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if free > best_free:
                best_gpu, best_free = i, free
        DEVICE = torch.device(f'cuda:{best_gpu}')
        print(f'Selected {DEVICE} ({torch.cuda.get_device_properties(DEVICE).total_memory/1e9:.1f}GB)')
    else:
        DEVICE = torch.device('cpu')

    # ---- fp16 decision ----
    # fp16 halves DGL message memory (features→fp16, model stays fp32).
    # No autocast needed — first nn.Linear upcasts to fp32 automatically.
    use_fp16 = args.fp16 and DEVICE.type == 'cuda'
    if args.no_fp16:
        use_fp16 = False
    if use_fp16:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        print('FP16 graph features enabled (½ message memory)')

    # Load the data.
    data = Dataset(dataset_name, homo).graph
    
    # Add edges when requested.
    if args.add_edges:
        print(f"Adding PPR edges to {dataset_name}...")
        # Load the pretrained embedding.
        hop_suffix = '_multi' if args.multi_hop else ''
        graph_path = cached_dataset(f'{dataset_name}_new_topk{args.ppr_topk}{hop_suffix}') / 'graph.bin'
        if args.reuse_ppr_graph and osp.exists(graph_path):
            print(f"Reusing the cached PPR graph: {graph_path}")
            new_graph, _ = dgl.load_graphs(str(graph_path))
            graph = new_graph[0]
        else:
            pretrained_embeddings = load_pretrained_embedding(dataset_name, args.pretrain_seed)
            if pretrained_embeddings is not None:
                # Add edges and save the augmented graph.
                # Configure multi-hop PPR.
                if args.multi_hop:
                    print(f"Computing PPR with {args.hop_count} hops...")
                    # Configure two-hop PPR.
                    if args.hop_count == 2:
                        # Use only two-hop PPR with the corresponding weights.
                        add_edges_and_save_graph(
                            data,
                            pretrained_embeddings,
                            topk=args.ppr_topk,
                            dataset_name=dataset_name,
                            use_multi_hop=True,
                            hops=(1, 2),
                            hop_weights=(0.7, 0.3)
                        )
                    else:
                        # Use the user-specified hop count.
                        if args.hop_count == 1:
                            hops = (1,)
                            hop_weights = (1.0,)
                        elif args.hop_count == 3:
                            hops = (1, 2, 3)
                            hop_weights = (0.6, 0.3, 0.1)
                        elif args.hop_count == 4:
                            hops = (1, 2, 3, 4)
                            hop_weights = (0.5, 0.3, 0.15, 0.05)
                        else:
                            # Fall back to two hops.
                            hops = (1, 2)
                            hop_weights = (0.7, 0.3)

                        add_edges_and_save_graph(
                            data,
                            pretrained_embeddings,
                            topk=args.ppr_topk,
                            dataset_name=dataset_name,
                            use_multi_hop=True,
                            hops=hops,
                            hop_weights=hop_weights
                        )
                else:
                    # Use traditional one-hop PPR.
                    add_edges_and_save_graph(
                        data,
                        pretrained_embeddings,
                        topk=args.ppr_topk,
                        dataset_name=dataset_name,
                        use_multi_hop=False
                    )

                # Reload the augmented graph.
                print(f"Loading the {dataset_name} graph with PPR edges...")
                new_graph, _ = dgl.load_graphs(str(graph_path))
                graph = new_graph[0]
            else:
                print("Pretrained embedding not found. Using the original graph.")
                graph = data
    else:
        graph = data

    mgtab_embedding_dim = 0
    if dataset_name == 'MGTAB' and args.mgtab_feature_mode != 'raw':
        if args.mgtab_feature_seeds:
            mgtab_feature_seeds = [int(seed.strip()) for seed in args.mgtab_feature_seeds.split(',')
                                   if seed.strip()]
        else:
            mgtab_feature_seeds = [args.mgtab_feature_seed]
        if not mgtab_feature_seeds:
            raise ValueError('No MGTAB pretrained feature seeds selected')

        mgtab_embs = []
        for feature_seed in mgtab_feature_seeds:
            emb_path = raw_dataset('MGTAB') / f'pretrained_embedding{feature_seed}.pt'
            if not osp.exists(emb_path):
                emb_path = pretrained_embedding('MGTAB', feature_seed)
            if not osp.exists(emb_path):
                raise FileNotFoundError(f'MGTAB pretrained feature not found for seed {feature_seed}')
            mgtab_embs.append(torch.load(emb_path).float().contiguous())

        if len(mgtab_embs) == 1:
            mgtab_emb = mgtab_embs[0]
        elif args.mgtab_feature_agg == 'mean':
            mgtab_emb = torch.stack(mgtab_embs, dim=0).mean(dim=0)
        else:
            mgtab_emb = torch.cat(mgtab_embs, dim=1)
        mgtab_embedding_dim = mgtab_emb.shape[1]

        if args.mgtab_feature_mode == 'pretrained':
            graph.ndata['feature'] = mgtab_emb
        elif args.mgtab_feature_mode == 'concat':
            graph.ndata['feature'] = torch.cat([graph.ndata['feature'].float(), mgtab_emb], dim=1)
        print(f'MGTAB feature mode={args.mgtab_feature_mode}, seeds={mgtab_feature_seeds}, '
              f'agg={args.mgtab_feature_agg}, dim={graph.ndata["feature"].shape[1]}')

    if dataset_name == 'MGTAB' and args.mgtab_feature_norm:
        feat = graph.ndata['feature'].float()
        graph.ndata['feature'] = (feat - feat.mean(dim=0, keepdim=True)) / feat.std(dim=0, keepdim=True).clamp(min=1e-6)
        print('MGTAB selected node features z-score normalized')

    if (dataset_name == 'MGTAB' and args.mgtab_feature_norm
            and args.mgtab_feature_clip > 0.0):
        graph.ndata['feature'] = graph.ndata['feature'].float().clamp(
            min=-args.mgtab_feature_clip, max=args.mgtab_feature_clip)
        print(f'MGTAB normalized node features clipped to +/-{args.mgtab_feature_clip}')

    if (dataset_name == 'MGTAB' and args.mgtab_feature_mode != 'raw'
            and args.mgtab_embedding_scale != 1.0):
        feat = graph.ndata['feature'].float().clone()
        if args.mgtab_feature_mode == 'concat':
            feat[:, -mgtab_embedding_dim:] *= args.mgtab_embedding_scale
        else:
            feat *= args.mgtab_embedding_scale
        graph.ndata['feature'] = feat
        print(f'MGTAB pretrained embedding feature block scaled by {args.mgtab_embedding_scale}')

    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    selected_etypes = None
    if args.edge_types:
        requested_etypes = [etype.strip() for etype in args.edge_types.split(',') if etype.strip()]
        available_etypes = set(graph.etypes)
        selected_etypes = [etype for etype in requested_etypes if etype in available_etypes]
        missing_etypes = [etype for etype in requested_etypes if etype not in available_etypes]
        if missing_etypes:
            print(f"Warning: requested edge types not found and ignored: {missing_etypes}")
        if not selected_etypes:
            raise ValueError(f"No usable edge types selected from {requested_etypes}; available={graph.etypes}")
        print(f"Using selected edge types: {selected_etypes}")
    active_etypes = selected_etypes if selected_etypes is not None else list(graph.etypes)

    # Pre-check GPU memory: DGL message passing + K=4 Laplacian powers
    # tsocial PPR (>200M edges) exceeds 24GB even with expandable_segments
    total_edges = sum(graph.num_edges(etype) for etype in active_etypes)
    max_etype_edges = max(graph.num_edges(etype) for etype in active_etypes)
    num_nodes = graph.num_nodes()
    in_feats = graph.ndata['feature'].shape[1]
    bpe = 2 if use_fp16 else 4
    num_etypes = len(active_etypes)
    est_vram = (total_edges * 16 + num_nodes * in_feats * bpe +
                max_etype_edges * h_feats * bpe * 2 +
                num_nodes * h_feats * bpe * (args.K + 1 + num_etypes + 2)) / 1e9
    # Heuristic: graphs >180M edges likely OOM on 24GB without fp16
    edge_limit = 220e6 if use_fp16 else 180e6
    if DEVICE.type == 'cuda':
        gpu_total = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    else:
        gpu_total = 0.0

    if DEVICE.type == 'cuda' and est_vram < gpu_total * 0.88 and total_edges < edge_limit:
        graph = graph.to(DEVICE)
        mode = 'fp16' if use_fp16 else 'fp32'
        print(f'Graph moved to {DEVICE} [{mode}] (est {est_vram:.1f}GB / {gpu_total:.1f}GB)')
    else:
        mode = 'fp16' if use_fp16 else 'fp32'
        if DEVICE.type == 'cuda':
            print(f'Graph too large for GPU [{mode}] (est {est_vram:.1f}GB / {gpu_total:.1f}GB), using CPU')
        else:
            print(f'Using CPU [{mode}] (est GPU memory would be {est_vram:.1f}GB)')
        DEVICE = torch.device('cpu')
        use_fp16 = False

    # Support multiple runs with different seeds
    for run_idx in range(args.run):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{args.run}")
        print(f"{'='*60}")

        if homo:
            model = EnhancedBernsteinNet(in_feats, h_feats, num_classes, graph,
                                         K=args.K,
                                         num_scales=args.num_scales,
                                         num_layers=args.num_layers,
                                         dropout_rate=args.dropout_rate,
                                         focal_gamma=args.focal_gamma,
                                         focal_alpha=args.focal_alpha,
                                         lambda_freq=args.lambda_freq,
                                         use_pretrained_skip=not args.no_pretrained_skip,
                                         etypes=selected_etypes,
                                         relation_fusion=args.relation_fusion,
                                         homophily_temperature=args.homophily_temperature,
                                         freq_loss_weighting=args.freq_loss_weighting,
                                         use_edge_weight=args.use_edge_weight,
                                         final_residual_weight=args.final_residual_weight)
        else:
            model = HeteroEnhancedBernsteinNet(in_feats, h_feats, num_classes, graph,
                                               K=args.K,
                                               num_scales=args.num_scales,
                                               num_layers=args.num_layers,
                                               dropout_rate=args.dropout_rate,
                                               focal_gamma=args.focal_gamma,
                                               focal_alpha=args.focal_alpha,
                                               lambda_freq=args.lambda_freq,
                                               use_pretrained_skip=not args.no_pretrained_skip,
                                               etypes=selected_etypes,
                                               relation_fusion=args.relation_fusion,
                                               homophily_temperature=args.homophily_temperature,
                                               freq_loss_weighting=args.freq_loss_weighting,
                                               use_edge_weight=args.use_edge_weight,
                                               final_residual_weight=args.final_residual_weight)
        model = model.to(DEVICE)
        train(model, graph, args, use_fp16=use_fp16)
