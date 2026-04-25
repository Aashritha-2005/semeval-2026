"""
SemEval-2026 Task 9 - Subtask 3: Manifestation Identification (Telugu)
Multi-Stage Pipeline with Calibrated Ensembles and Lexical Post-Processing

Architecture:
  Stage 1: Multi-model ensemble (mDeBERTa-v3-base, XLM-RoBERTa-base, MuRIL)
  Stage 2: 5-fold CV with 2 seeds, out-of-fold logit averaging
  Stage 3: Per-class threshold calibration via macro-F1 optimization
  Stage 4: Lexical post-processing (polarization=0 => all others=0)

Usage:
  python train_pipeline.py --data_path tel_train.csv --output_dir ./output
  
  For Google Colab with GPU:
    python train_pipeline.py --data_path tel_train.csv --output_dir ./output --device cuda

  For Mac (MPS or CPU):
    python train_pipeline.py --data_path tel_train.csv --output_dir ./output --device mps
    python train_pipeline.py --data_path tel_train.csv --output_dir ./output --device cpu
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
LABEL_COLS = [
    "polarization", "political", "racial/ethnic", "religious",
    "gender/sexual", "other", "stereotype", "vilification",
    "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"
]

MODEL_CONFIGS = {
    "mdeberta": {
        "name": "microsoft/mdeberta-v3-base",
        "max_len": 128,
        "lr": 2e-5,
        "epochs": 8,
        "batch_size": 16,
        "weight": 0.4,
    },
    "xlmr": {
        "name": "xlm-roberta-base",
        "max_len": 128,
        "lr": 2e-5,
        "epochs": 8,
        "batch_size": 16,
        "weight": 0.35,
    },
    "muril": {
        "name": "google/muril-base-cased",
        "max_len": 128,
        "lr": 2e-5,
        "epochs": 8,
        "batch_size": 16,
        "weight": 0.25,
    },
}

N_FOLDS = 5
SEEDS = [42, 2026]


# ─────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────
class PolarizationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ─────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels=12, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ─────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(labels):
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# ─────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────
def optimize_thresholds(logits, labels):
    """Per-class threshold optimization to maximize macro F1."""
    n_labels = labels.shape[1]
    best_thresholds = np.zeros(n_labels)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    for i in range(n_labels):
        best_f1 = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[i] = best_t
    return best_thresholds


# ─────────────────────────────────────────────────
# LEXICAL POST-PROCESSING
# ─────────────────────────────────────────────────
def apply_post_processing(preds):
    """
    Rule: If polarization=0, set all other labels to 0.
    This is a hard constraint from the data: non-polarized text
    never has any manifestation labels.
    """
    preds = preds.copy()
    mask = preds[:, 0] == 0  # polarization column
    preds[mask, 1:] = 0
    return preds


# ─────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────
def run_single_model(model_key, config, df, device, output_dir):
    """Train a single model with K-fold CV and multiple seeds. Returns OOF logits."""
    print(f"\n{'='*60}")
    print(f"  Training: {config['name']}")
    print(f"{'='*60}")

    texts = df["text"].values
    labels = df[LABEL_COLS].values
    n_samples = len(df)
    n_labels = len(LABEL_COLS)

    # Accumulate OOF logits across seeds
    oof_logits_accum = np.zeros((n_samples, n_labels))

    for seed in SEEDS:
        print(f"\n  Seed: {seed}")
        set_seed(seed)

        oof_logits = np.zeros((n_samples, n_labels))
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(texts, labels[:, 0])  # stratify on polarization
        ):
            print(f"    Fold {fold+1}/{N_FOLDS}", end=" ")

            tokenizer = AutoTokenizer.from_pretrained(config["name"])

            train_dataset = PolarizationDataset(
                texts[train_idx], labels[train_idx], tokenizer, config["max_len"]
            )
            val_dataset = PolarizationDataset(
                texts[val_idx], labels[val_idx], tokenizer, config["max_len"]
            )

            train_loader = DataLoader(
                train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0
            )

            model = MultiLabelClassifier(config["name"], n_labels).to(device)

            pos_weight = get_class_weights(labels[train_idx]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["lr"], weight_decay=0.01
            )
            total_steps = len(train_loader) * config["epochs"]
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

            best_val_f1 = -1
            best_logits = None
            patience = 3
            patience_counter = 0

            for epoch in range(config["epochs"]):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler, criterion, device
                )
                val_logits = predict(model, val_loader, device)
                val_probs = 1 / (1 + np.exp(-val_logits))
                val_preds = (val_probs >= 0.5).astype(int)
                val_preds = apply_post_processing(val_preds)

                val_f1 = f1_score(labels[val_idx], val_preds, average="macro", zero_division=0)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_logits = val_logits.copy()
                    patience_counter = 0
                    # Save best model for this fold
                    save_path = os.path.join(
                        output_dir, f"{model_key}_seed{seed}_fold{fold}.pt"
                    )
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            oof_logits[val_idx] = best_logits
            print(f"  F1={best_val_f1:.4f}")

            # Clean up GPU memory
            del model, optimizer, scheduler, criterion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oof_logits_accum += oof_logits

    # Average across seeds
    oof_logits_avg = oof_logits_accum / len(SEEDS)
    return oof_logits_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to tel_train.csv")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, cuda, mps, or auto")
    parser.add_argument("--models", type=str, default="mdeberta,xlmr,muril",
                        help="Comma-separated model keys to train")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples, {len(LABEL_COLS)} labels")

    labels = df[LABEL_COLS].values
    model_keys = [k.strip() for k in args.models.split(",")]

    # ── Stage 1 & 2: Train each model, get OOF logits ──
    all_oof_logits = {}
    for model_key in model_keys:
        config = MODEL_CONFIGS[model_key]
        oof_logits = run_single_model(model_key, config, df, device, args.output_dir)
        all_oof_logits[model_key] = oof_logits

    # ── Stage 2b: Weighted ensemble of OOF logits ──
    print(f"\n{'='*60}")
    print("  Ensemble & Threshold Calibration")
    print(f"{'='*60}")

    # Optimize ensemble weights via grid search
    best_macro_f1 = -1
    best_weights = None

    model_keys_list = list(all_oof_logits.keys())
    n_models = len(model_keys_list)

    if n_models == 1:
        ensemble_logits = all_oof_logits[model_keys_list[0]]
        best_weights = {model_keys_list[0]: 1.0}
    else:
        # Grid search over weight combinations
        step = 0.05
        weight_range = np.arange(0.0, 1.01, step)
        
        for w0 in weight_range:
            for w1 in weight_range:
                if n_models == 2:
                    if abs(w0 + w1 - 1.0) > 1e-6:
                        continue
                    weights = [w0, w1]
                elif n_models == 3:
                    w2 = 1.0 - w0 - w1
                    if w2 < -1e-6 or w2 > 1.0 + 1e-6:
                        continue
                    weights = [w0, w1, w2]
                else:
                    continue

                combined = sum(
                    w * all_oof_logits[k]
                    for w, k in zip(weights, model_keys_list)
                )
                thresholds = optimize_thresholds(combined, labels)
                probs = 1 / (1 + np.exp(-combined))
                preds = np.zeros_like(probs)
                for i in range(len(LABEL_COLS)):
                    preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
                preds = apply_post_processing(preds)

                mf1 = f1_score(labels, preds, average="macro", zero_division=0)
                if mf1 > best_macro_f1:
                    best_macro_f1 = mf1
                    best_weights = {k: w for k, w in zip(model_keys_list, weights)}

        ensemble_logits = sum(
            best_weights[k] * all_oof_logits[k] for k in model_keys_list
        )

    print(f"\n  Best ensemble weights: {best_weights}")

    # ── Stage 3: Final threshold calibration ──
    final_thresholds = optimize_thresholds(ensemble_logits, labels)
    print(f"  Optimized thresholds: {dict(zip(LABEL_COLS, [f'{t:.2f}' for t in final_thresholds]))}")

    # ── Stage 4: Final evaluation with post-processing ──
    probs = 1 / (1 + np.exp(-ensemble_logits))
    final_preds = np.zeros_like(probs)
    for i in range(len(LABEL_COLS)):
        final_preds[:, i] = (probs[:, i] >= final_thresholds[i]).astype(int)
    final_preds = apply_post_processing(final_preds)

    macro_f1 = f1_score(labels, final_preds, average="macro", zero_division=0)
    print(f"\n  *** OOF Macro F1 (with post-processing): {macro_f1:.4f} ***")

    # Per-label F1
    print("\n  Per-label F1:")
    for i, col in enumerate(LABEL_COLS):
        lf1 = f1_score(labels[:, i], final_preds[:, i], zero_division=0)
        print(f"    {col:20s}: {lf1:.4f}")

    # ── Save calibration config ──
    calibration_config = {
        "ensemble_weights": best_weights,
        "thresholds": dict(zip(LABEL_COLS, final_thresholds.tolist())),
        "macro_f1": macro_f1,
        "models": {k: MODEL_CONFIGS[k]["name"] for k in model_keys},
    }
    config_path = os.path.join(args.output_dir, "calibration_config.json")
    with open(config_path, "w") as f:
        json.dump(calibration_config, f, indent=2)
    print(f"\n  Calibration config saved to {config_path}")
    print(f"\n  Model checkpoints saved to {args.output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()