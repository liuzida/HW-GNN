"""Dataset-specific defaults for the benchmark runs.

Explicit command-line values take precedence over these defaults.
"""

from copy import deepcopy


DEFAULT_DATASET = "tsocial"


DATASET_PRESETS = {
    "tfinance": {
        "train_ratio": 0.4,
        "hid_dim": 32,
        "homo": 1,
        "epoch": 500,
        "run": 1,
        "add_edges": True,
        "ppr_topk": 32,
        "pretrain_seed": 0,
        "reuse_ppr_graph": True,
        "lr": 0.005,
        "patience": 80,
        "lambda_freq": 0.2,
        "seed": 3407,
        "split_mode": "default",
        "edge_types": "",
    },
    "tsocial": {
        "train_ratio": 0.4,
        "hid_dim": 10,
        "homo": 1,
        "epoch": 800,
        "run": 1,
        "add_edges": True,
        "ppr_topk": 64,
        "pretrain_seed": 0,
        "reuse_ppr_graph": True,
        "lr": 0.005,
        "patience": 120,
        "lambda_freq": 0.2,
        "seed": 3407,
        "split_mode": "default",
        "edge_types": "",
    },
    "Twibot-22": {
        "train_ratio": 0.4,
        "hid_dim": 64,
        "homo": 1,
        "epoch": 500,
        "run": 1,
        "add_edges": False,
        "ppr_topk": 8,
        "pretrain_seed": 0,
        "reuse_ppr_graph": False,
        "fp16": True,
        "lr": 0.003,
        "patience": 80,
        "lambda_freq": 0.1,
        "K": 3,
        "num_scales": 5,
        "seed": 3407,
        "split_mode": "default",
        "edge_types": "_E0,_E1",
    },
    "Twibot-20": {
        "train_ratio": 0.4,
        "hid_dim": 192,
        "homo": 1,
        "epoch": 500,
        "run": 1,
        "add_edges": False,
        "ppr_topk": 8,
        "pretrain_seed": 0,
        "reuse_ppr_graph": False,
        "fp16": True,
        "lr": 0.005,
        "patience": 100,
        "lambda_freq": 0.2,
        "seed": 1,
        "split_mode": "stratified",
        "edge_types": "",
    },
    # Single-model MGTAB configuration with a stratified split.
    "MGTAB": {
        "train_ratio": 0.4,
        "hid_dim": 112,
        "homo": 1,
        "epoch": 800,
        "run": 1,
        "add_edges": False,
        "ppr_topk": 8,
        "pretrain_seed": 0,
        "reuse_ppr_graph": False,
        "lr": 0.005,
        "patience": 180,
        "lambda_freq": 0.4,
        "dropout_rate": 0.25,
        "focal_gamma": 1.5,
        "seed": 1,
        "split_seed": 1,
        "split_mode": "stratified",
        "relation_fusion": "mean_final_semantic",
        "final_residual_weight": 0.30,
        "mgtab_feature_mode": "concat",
        "mgtab_feature_seed": 2,
        "mgtab_feature_norm": True,
        "mgtab_embedding_scale": 1.55,
        "mgtab_feature_clip": 0.0,
        "mgtab_val_ratio": 0.2,
        "auto_focal_alpha": True,
        "focal_alpha": 0.25,
        "edge_types": "",
    },
}


def apply_dataset_defaults(args):
    """Fill only omitted arguments from the selected dataset preset."""
    try:
        preset = DATASET_PRESETS[args.dataset]
    except KeyError as exc:
        supported = ", ".join(DATASET_PRESETS)
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Choose one of: {supported}") from exc

    for name, value in preset.items():
        if getattr(args, name, None) is None:
            setattr(args, name, deepcopy(value))

    common_defaults = {
        "K": 4,
        "num_scales": 5,
        "dropout_rate": 0.3,
        "focal_gamma": 2.0,
        "relation_fusion": "mean_final_semantic",
        "final_residual_weight": 0.0,
        "mgtab_feature_mode": "raw",
        "mgtab_feature_norm": False,
        "mgtab_feature_seed": 0,
        "mgtab_embedding_scale": 1.0,
        "mgtab_feature_clip": 0.0,
        "mgtab_val_ratio": 0.2,
        "focal_alpha": 0.25,
    }
    for name, value in common_defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, deepcopy(value))

    # Values that are common to all presets retain the original parser
    # defaults when a dataset does not override them.
    if args.split_seed is None:
        args.split_seed = args.seed if args.dataset == "Twibot-20" else None
    if args.edge_types is None:
        args.edge_types = ""
    if args.reuse_ppr_graph is None:
        args.reuse_ppr_graph = False
    if args.add_edges is None:
        args.add_edges = False
    if args.fp16 is None:
        args.fp16 = False
    if args.auto_focal_alpha is None:
        args.auto_focal_alpha = True

    return args


def preset_summary(dataset):
    """Return a copy suitable for documentation or diagnostics."""
    return deepcopy(DATASET_PRESETS[dataset])
