#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def best_macro_f1(labels, scores):
    labels = labels.astype(np.int64, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n = labels.shape[0]
    pos_total = sorted_labels.sum()
    neg_total = n - pos_total
    if n == 0 or pos_total == 0 or neg_total == 0:
        return 0.0, 0.5

    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), n - 1]
    ks = group_ends + 1
    tp = np.cumsum(sorted_labels)[group_ends].astype(np.float64)
    fp = ks.astype(np.float64) - tp
    fn = pos_total - tp
    tn = neg_total - fp

    ks = np.r_[0, ks]
    tp = np.r_[0.0, tp]
    fp = np.r_[0.0, fp]
    fn = np.r_[float(pos_total), fn]
    tn = np.r_[float(neg_total), tn]

    pos_f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                       out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    neg_f1 = np.divide(2 * tn, 2 * tn + fp + fn,
                       out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
    values = 0.5 * (pos_f1 + neg_f1)
    best_idx = int(np.argmax(values))
    best_k = int(ks[best_idx])
    if best_k == 0:
        threshold = float(sorted_scores[0])
    elif best_k >= n:
        threshold = float(sorted_scores[-1] - 1e-12)
    else:
        threshold = float((sorted_scores[best_k - 1] + sorted_scores[best_k]) / 2.0)
    return float(values[best_idx]), threshold


def evaluate(labels, scores, mask, threshold):
    y = labels[mask]
    s = scores[mask]
    pred = (s > threshold).astype(np.int64)
    return {
        "ACC": accuracy_score(y, pred) * 100,
        "REC": recall_score(y, pred) * 100,
        "PRE": precision_score(y, pred, zero_division=0) * 100,
        "MF1": f1_score(y, pred, average="macro") * 100,
        "AUC": roc_auc_score(y, s) * 100,
    }


def load_npz(path):
    data = np.load(path, allow_pickle=False)
    return {
        "path": path,
        "labels": data["labels"],
        "scores": data["probs"][:, 1],
        "train_mask": data["train_mask"].astype(bool),
        "val_mask": data["val_mask"].astype(bool),
        "test_mask": data["test_mask"].astype(bool),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No npz files matched: {args.pattern}")

    runs = [load_npz(path) for path in paths]
    labels = runs[0]["labels"]
    train_mask = runs[0]["train_mask"]
    val_mask = runs[0]["val_mask"]
    test_mask = runs[0]["test_mask"]
    for run in runs[1:]:
        if not np.array_equal(labels, run["labels"]):
            raise ValueError(f"Label mismatch in {run['path']}")
        if not np.array_equal(train_mask, run["train_mask"]):
            raise ValueError(f"Train mask mismatch in {run['path']}")
        if not np.array_equal(val_mask, run["val_mask"]):
            raise ValueError(f"Val mask mismatch in {run['path']}")
        if not np.array_equal(test_mask, run["test_mask"]):
            raise ValueError(f"Test mask mismatch in {run['path']}")

    score_stack = np.stack([run["scores"] for run in runs], axis=0)
    candidates = {
        "mean_all": score_stack.mean(axis=0),
        "median_all": np.median(score_stack, axis=0),
    }

    if len(runs) >= 3:
        # Average each prefix to see how many repeats are useful without using test labels.
        for k in range(3, len(runs) + 1):
            candidates[f"mean_first{k:02d}"] = score_stack[:k].mean(axis=0)

    lines = []
    lines.append("=== MGTAB Ensemble Analysis ===")
    lines.append(f"Files: {len(paths)}")
    lines.append("")
    lines.append("Single-run validation-selected metrics from saved probabilities:")
    single_rows = []
    for run in runs:
        val_f1, threshold = best_macro_f1(labels[val_mask], run["scores"][val_mask])
        metrics = evaluate(labels, run["scores"], test_mask, threshold)
        single_rows.append((metrics["MF1"], os.path.basename(run["path"]), threshold, metrics))
    for mf1, name, threshold, metrics in sorted(single_rows, reverse=True):
        lines.append(
            f"SINGLE | {name} | thres={threshold:.4f} | "
            f"ACC {metrics['ACC']:.2f} REC {metrics['REC']:.2f} PRE {metrics['PRE']:.2f} "
            f"MF1 {metrics['MF1']:.2f} AUC {metrics['AUC']:.2f}"
        )

    lines.append("")
    lines.append("Ensemble metrics, threshold selected on validation split:")
    ensemble_rows = []
    for name, scores in candidates.items():
        val_f1, threshold = best_macro_f1(labels[val_mask], scores[val_mask])
        metrics = evaluate(labels, scores, test_mask, threshold)
        ensemble_rows.append((metrics["MF1"], name, threshold, val_f1, metrics))

    for mf1, name, threshold, val_f1, metrics in sorted(ensemble_rows, reverse=True):
        lines.append(
            f"ENSEMBLE | {name} | val_mf1={val_f1 * 100:.2f} thres={threshold:.4f} | "
            f"ACC {metrics['ACC']:.2f} REC {metrics['REC']:.2f} PRE {metrics['PRE']:.2f} "
            f"MF1 {metrics['MF1']:.2f} AUC {metrics['AUC']:.2f}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
