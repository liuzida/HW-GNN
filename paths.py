"""Portable project paths.

Set these environment variables when datasets or generated artifacts live
outside the repository:

  BOT_DATA_ROOT       raw dataset directories
  BOT_CACHE_ROOT      generated DGL graphs
  BOT_PRETRAINED_ROOT pretrained embeddings and MLP artifacts
  BOT_MLP_ROOT       optional MLP checkpoint directory
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_path(name, default):
    value = os.environ.get(name, default)
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


RAW_DATA_ROOT = _configured_path("BOT_DATA_ROOT", "dataset/raw")
DATASET_CACHE_ROOT = _configured_path("BOT_CACHE_ROOT", "dataset")
PRETRAINED_ROOT = _configured_path("BOT_PRETRAINED_ROOT", "pretrained")
MLP_ROOT = _configured_path("BOT_MLP_ROOT", "mlp_model")


def raw_dataset(name):
    return RAW_DATA_ROOT / name


def cached_dataset(name):
    return DATASET_CACHE_ROOT / name


def pretrained_embedding(dataset_name, seed):
    return PRETRAINED_ROOT / f"{dataset_name}_pretrained_embedding_{seed}.pt"
