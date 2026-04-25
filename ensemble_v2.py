"""
Combine OOF outputs from multiple v2 models into an ensemble.
Run AFTER training each model with train_v2.py.

Usage:
  python ensemble_v2.py --data_path tel_train.csv --models muril,xlmr
  python ensemble_v2.py --data_path tel_train.csv --models muril,xlmr,mdeberta
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

MANIFESTATION_COLS = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]


def optimize_thresholds(logits, labels):
    n_labels = labels.shape[1]
    probs = 1 / (1 + np.exp(-logits))
    thresholds = np.zeros(n_labels)
    for i in range(n_labels):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.15, 0.85, 0.01):
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
    parser.add_argument("--models", type=str, default="muril,xlmr")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    labels_polar = df["polarization"].values
    labels_manif = df[MANIFESTATION_COLS].values
    model_keys = [k.strip() for k in args.models.split(",")]

    # Load per-model outputs
    polar_probs = {}
    manif_logits = {}
    for key in model_keys:
        pp = os.path.join(args.logits_dir, f"{key}_polar_probs.npy")
        ml = os.path.join(args.logits_dir, f"{key}_manif_logits.npy")
        if os.path.exists(pp) and os.path.exists(ml):
            polar_probs[key] = np.load(pp)
            manif_logits[key] = np.load(ml)
            print(f"Loaded {key}")
        else:
            print(f"WARNING: {key} files not found, skipping")

    available = list(polar_probs.keys())
    if not available:
        print("No models found.")
        return

    n = len(df)
    step = 0.05

    best_macro_f1 = -1
    best_config = None

    # Grid search over weights
    print(f"\nGrid searching weights for {len(available)} models...")

    if len(available) == 1:
        weight_combos = [{available[0]: 1.0}]
    elif len(available) == 2:
        weight_combos = []
        for w0 in np.arange(0, 1.01, step):
            weight_combos.append({available[0]: w0, available[1]: 1.0 - w0})
    else:
        weight_combos = []
        for w0 in np.arange(0, 1.01, step):
            for w1 in np.arange(0, 1.01 - w0, step):
                w2 = 1.0 - w0 - w1
                if w2 >= -1e-6:
                    weight_combos.append({available[0]: w0, available[1]: w1, available[2]: max(0, w2)})

    for weights in weight_combos:
        # Combine polarization probs
        combined_polar = sum(weights[k] * polar_probs[k] for k in available)
        # Combine manifestation logits
        combined_manif = sum(weights[k] * manif_logits[k] for k in available)

        # Search polarization threshold
        for pt in np.arange(0.3, 0.7, 0.02):
            p_preds = (combined_polar >= pt).astype(int)

            # Search manifestation thresholds
            thresholds = optimize_thresholds(combined_manif, labels_manif)
            probs = 1 / (1 + np.exp(-combined_manif))
            m_preds = np.zeros_like(probs, dtype=int)
            for i in range(len(MANIFESTATION_COLS)):
                m_preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

            # Post-processing
            m_preds[p_preds == 0] = 0
            for idx in range(n):
                if p_preds[idx] == 1 and m_preds[idx].sum() == 0:
                    m_preds[idx, probs[idx].argmax()] = 1

            mf1 = f1_score(labels_manif, m_preds, average="macro", zero_division=0)
            if mf1 > best_macro_f1:
                best_macro_f1 = mf1
                best_config = {
                    "weights": dict(weights),
                    "polar_threshold": pt,
                    "manif_thresholds": dict(zip(MANIFESTATION_COLS, thresholds.tolist())),
                }

    print(f"\n*** Best Ensemble Macro F1: {best_macro_f1:.4f} ***")
    print(f"Weights: {best_config['weights']}")
    print(f"Polar threshold: {best_config['polar_threshold']:.2f}")

    # Reconstruct final predictions for per-label breakdown
    weights = best_config["weights"]
    combined_polar = sum(weights[k] * polar_probs[k] for k in available)
    combined_manif = sum(weights[k] * manif_logits[k] for k in available)
    p_preds = (combined_polar >= best_config["polar_threshold"]).astype(int)
    probs = 1 / (1 + np.exp(-combined_manif))
    thresholds = np.array([best_config["manif_thresholds"][c] for c in MANIFESTATION_COLS])
    m_preds = np.zeros_like(probs, dtype=int)
    for i in range(len(MANIFESTATION_COLS)):
        m_preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    m_preds[p_preds == 0] = 0
    for idx in range(n):
        if p_preds[idx] == 1 and m_preds[idx].sum() == 0:
            m_preds[idx, probs[idx].argmax()] = 1

    print("\nPer-label F1:")
    for i, col in enumerate(MANIFESTATION_COLS):
        lf1 = f1_score(labels_manif[:, i], m_preds[:, i], zero_division=0)
        print(f"  {col:20s}: {lf1:.4f}  (threshold={thresholds[i]:.2f})")

    # Save
    model_name_map = {
        "muril": "google/muril-base-cased",
        "xlmr": "xlm-roberta-base",
        "mdeberta": "microsoft/mdeberta-v3-base",
        "indicbert": "ai4bharat/indic-bert",
    }
    best_config["models"] = {k: model_name_map.get(k, k) for k in available}
    best_config["macro_f1"] = best_macro_f1

    config_path = os.path.join(args.logits_dir, "ensemble_v2_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved {config_path}")


if __name__ == "__main__":
    main()