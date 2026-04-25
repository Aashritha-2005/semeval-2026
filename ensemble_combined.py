"""
Combine OOF logits from individually trained models into an optimized ensemble.
Run this AFTER training each model separately with train_single.py.

Usage:
  python ensemble_combine.py \
    --data_path tel_train.csv \
    --logits_dir ./output \
    --models mdeberta,xlmr,muril
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABEL_COLS = [
    "polarization", "political", "racial/ethnic", "religious",
    "gender/sexual", "other", "stereotype", "vilification",
    "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"
]


def apply_post_processing(preds):
    preds = preds.copy()
    preds[preds[:, 0] == 0, 1:] = 0
    return preds


def optimize_thresholds(logits, labels):
    n_labels = labels.shape[1]
    thresholds = np.zeros(n_labels)
    probs = 1 / (1 + np.exp(-logits))
    for i in range(n_labels):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            f = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        thresholds[i] = best_t
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--logits_dir", type=str, default="./output")
    parser.add_argument("--models", type=str, default="mdeberta,xlmr,muril")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    labels = df[LABEL_COLS].values
    model_keys = [k.strip() for k in args.models.split(",")]

    # Load OOF logits
    all_logits = {}
    for key in model_keys:
        path = os.path.join(args.logits_dir, f"{key}_oof_logits.npy")
        if os.path.exists(path):
            all_logits[key] = np.load(path)
            print(f"Loaded {key} logits: {all_logits[key].shape}")
        else:
            print(f"WARNING: {path} not found, skipping {key}")

    if not all_logits:
        print("No logits found. Train models first.")
        return

    available_keys = list(all_logits.keys())
    n_models = len(available_keys)

    # Grid search ensemble weights
    print(f"\nSearching ensemble weights for {n_models} models...")
    best_f1 = -1
    best_weights = None
    step = 0.05

    if n_models == 1:
        best_weights = {available_keys[0]: 1.0}
        combined = all_logits[available_keys[0]]
    elif n_models == 2:
        for w0 in np.arange(0, 1.01, step):
            w1 = 1.0 - w0
            combined = w0 * all_logits[available_keys[0]] + w1 * all_logits[available_keys[1]]
            thresholds = optimize_thresholds(combined, labels)
            probs = 1 / (1 + np.exp(-combined))
            preds = np.column_stack([(probs[:, i] >= thresholds[i]).astype(int) for i in range(len(LABEL_COLS))])
            preds = apply_post_processing(preds)
            mf1 = f1_score(labels, preds, average="macro", zero_division=0)
            if mf1 > best_f1:
                best_f1 = mf1
                best_weights = {available_keys[0]: w0, available_keys[1]: w1}
        combined = sum(best_weights[k] * all_logits[k] for k in available_keys)
    else:  # 3 models
        for w0 in np.arange(0, 1.01, step):
            for w1 in np.arange(0, 1.01 - w0, step):
                w2 = 1.0 - w0 - w1
                if w2 < -1e-6:
                    continue
                combined = (w0 * all_logits[available_keys[0]] +
                            w1 * all_logits[available_keys[1]] +
                            w2 * all_logits[available_keys[2]])
                thresholds = optimize_thresholds(combined, labels)
                probs = 1 / (1 + np.exp(-combined))
                preds = np.column_stack([(probs[:, i] >= thresholds[i]).astype(int) for i in range(len(LABEL_COLS))])
                preds = apply_post_processing(preds)
                mf1 = f1_score(labels, preds, average="macro", zero_division=0)
                if mf1 > best_f1:
                    best_f1 = mf1
                    best_weights = {available_keys[0]: w0, available_keys[1]: w1, available_keys[2]: w2}
        combined = sum(best_weights[k] * all_logits[k] for k in available_keys)

    print(f"\nBest weights: {best_weights}")

    # Final threshold calibration
    final_thresholds = optimize_thresholds(combined, labels)
    probs = 1 / (1 + np.exp(-combined))
    final_preds = np.column_stack([(probs[:, i] >= final_thresholds[i]).astype(int) for i in range(len(LABEL_COLS))])
    final_preds = apply_post_processing(final_preds)
    final_f1 = f1_score(labels, final_preds, average="macro", zero_division=0)

    print(f"\n*** Ensemble Macro F1: {final_f1:.4f} ***")
    print("\nPer-label F1:")
    for i, col in enumerate(LABEL_COLS):
        lf1 = f1_score(labels[:, i], final_preds[:, i], zero_division=0)
        print(f"  {col:20s}: {lf1:.4f}  (threshold={final_thresholds[i]:.2f})")

    # Save ensemble config
    model_name_map = {
        "mdeberta": "microsoft/mdeberta-v3-base",
        "xlmr": "xlm-roberta-base",
        "muril": "google/muril-base-cased",
    }
    config = {
        "ensemble_weights": best_weights,
        "thresholds": dict(zip(LABEL_COLS, final_thresholds.tolist())),
        "macro_f1": final_f1,
        "models": {k: model_name_map.get(k, k) for k in available_keys},
    }
    config_path = os.path.join(args.logits_dir, "calibration_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {config_path}")


if __name__ == "__main__":
    main()