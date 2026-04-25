"""
SemEval-2026 Task 9 – Subtask 3: Manifestation Identification (Telugu)
Rewritten pipeline targeting macro F1 >= 0.70

Architecture:
  Stage A: Binary polarization classifier (high recall)
  Stage B: 6-label manifestation classifier (trained on polarized-only subset)
  Post-processing: polarization=0 => all manifestation=0

Key improvements over v1:
  - Correct label set: only 6 manifestation labels
  - Two-stage: separate polarization detector + manifestation classifier
  - Train manifestation model ONLY on polarized samples (cleaner signal)
  - Multi-pool: CLS + mean + max pooling concatenated
  - Focal loss for rare class handling (dehumanization at 4.6%)
  - Label-aware threshold grid per fold
  - 5-fold × 2 seeds with logit averaging

Usage:
  python train_v2.py --data_path tel_train.csv --model muril --device mps
  python train_v2.py --data_path tel_train.csv --model xlmr --device mps
  python train_v2.py --data_path tel_train.csv --model mdeberta --device cpu
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ── Subtask 3 labels ONLY ──
MANIFESTATION_COLS = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]

MODEL_CONFIGS = {
    "muril":     {"name": "google/muril-base-cased",    "max_len": 128, "lr": 2e-5, "epochs": 12, "batch_size": 16, "mps_safe": True},
    "xlmr":      {"name": "xlm-roberta-base",           "max_len": 128, "lr": 2e-5, "epochs": 12, "batch_size": 16, "mps_safe": True},
    "indicbert": {"name": "ai4bharat/indic-bert",       "max_len": 128, "lr": 3e-5, "epochs": 15, "batch_size": 32, "mps_safe": True},
    "mdeberta":  {"name": "microsoft/mdeberta-v3-base", "max_len": 128, "lr": 2e-5, "epochs": 12, "batch_size": 16, "mps_safe": False},
}

N_FOLDS = 5
SEEDS = [42, 2026]


# ─────────────────────────────────────────────
# FOCAL LOSS (handles class imbalance better than BCE)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weight tensor

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_t

        return (focal_weight * bce).mean()


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
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


# ─────────────────────────────────────────────
# MODEL: Multi-pool classifier
# ─────────────────────────────────────────────
class MultiPoolClassifier(nn.Module):
    """
    Concatenates CLS token + mean pool + max pool for richer representation.
    3x hidden_size -> projection -> num_labels
    """
    def __init__(self, model_name, num_labels, dropout=0.15):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(h * 3, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

        # CLS
        cls_token = hidden[:, 0, :]

        # Mean pool
        sum_hidden = (hidden * mask).sum(1)
        mean_pool = sum_hidden / mask.sum(1).clamp(min=1e-9)

        # Max pool (mask padding with large negative)
        hidden_masked = hidden.clone()
        hidden_masked[mask.squeeze(-1) == 0] = -1e9
        max_pool = hidden_masked.max(dim=1).values

        concat = torch.cat([cls_token, mean_pool, max_pool], dim=-1)
        x = self.dropout(concat)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )
        all_logits.append(logits.cpu().float().numpy())
    return np.concatenate(all_logits)


def optimize_thresholds(logits, labels):
    """Per-class threshold search maximizing per-class F1."""
    n_labels = labels.shape[1]
    probs = 1 / (1 + np.exp(-logits))
    thresholds = np.zeros(n_labels)
    for i in range(n_labels):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.15, 0.85, 0.01):
            preds = (probs[:, i] >= t).astype(int)
            f = f1_score(labels[:, i], preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        thresholds[i] = best_t
    return thresholds


# ─────────────────────────────────────────────
# STAGE A: Polarization binary classifier
# ─────────────────────────────────────────────
def train_polarization_detector(texts, labels_polar, tokenizer, config, device, output_dir, seed):
    """Train binary polarization detector. Returns OOF probabilities."""
    print(f"\n  [Stage A] Polarization detector (seed={seed})")
    n = len(texts)
    oof_probs = np.zeros(n)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels_polar)):
        set_seed(seed + fold)
        tr_ds = TextDataset(texts[tr_idx], labels_polar[tr_idx].reshape(-1, 1), tokenizer, config["max_len"])
        va_ds = TextDataset(texts[va_idx], labels_polar[va_idx].reshape(-1, 1), tokenizer, config["max_len"])
        tr_loader = DataLoader(tr_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)

        model = MultiPoolClassifier(config["name"], num_labels=1).to(device).float()

        # Weighted loss for class balance
        pos_ratio = labels_polar[tr_idx].mean()
        pos_w = (1 - pos_ratio) / (pos_ratio + 1e-9)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
        total_steps = len(tr_loader) * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        best_f1, best_probs = -1, None
        patience, no_improve = 3, 0

        for epoch in range(config["epochs"]):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                if torch.isnan(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            va_logits = predict(model, va_loader, device)
            va_probs = 1 / (1 + np.exp(-va_logits.flatten()))
            va_preds = (va_probs >= 0.5).astype(int)
            f1 = f1_score(labels_polar[va_idx], va_preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_probs = va_probs.copy()
                no_improve = 0
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f"polar_seed{seed}_fold{fold}.pt"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        oof_probs[va_idx] = best_probs
        print(f"    Fold {fold+1}: polar_f1={best_f1:.4f}")

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof_probs


# ─────────────────────────────────────────────
# STAGE B: Manifestation multi-label classifier
# ─────────────────────────────────────────────
def train_manifestation_classifier(texts_all, labels_all, labels_polar_all, tokenizer, config, device, output_dir, model_key, seed):
    """
    Train manifestation classifier using ALL samples.
    Non-polarized samples get all-zero labels (they're easy negatives that
    help the model learn what non-manifestation looks like).
    """
    print(f"\n  [Stage B] Manifestation classifier (seed={seed})")
    n = len(texts_all)
    n_labels = len(MANIFESTATION_COLS)
    oof_logits = np.zeros((n, n_labels))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts_all, labels_polar_all)):
        set_seed(seed + fold * 7)

        tr_ds = TextDataset(texts_all[tr_idx], labels_all[tr_idx], tokenizer, config["max_len"])
        va_ds = TextDataset(texts_all[va_idx], labels_all[va_idx], tokenizer, config["max_len"])
        tr_loader = DataLoader(tr_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)

        model = MultiPoolClassifier(config["name"], num_labels=n_labels).to(device).float()

        # Focal loss with per-class alpha based on positive ratios
        pos_counts = labels_all[tr_idx].sum(axis=0)
        total = len(tr_idx)
        alpha = torch.tensor(pos_counts / total, dtype=torch.float32).to(device)
        alpha = alpha.clamp(0.05, 0.95)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
        total_steps = len(tr_loader) * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        best_f1, best_logits = -1, None
        patience, no_improve = 4, 0

        for epoch in range(config["epochs"]):
            model.train()
            epoch_loss = 0
            nan_hit = False
            for batch in tr_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                if torch.isnan(loss):
                    nan_hit = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            if nan_hit:
                print(f"    Fold {fold+1}: NaN at epoch {epoch+1}, stopping")
                break

            # Validate
            va_logits = predict(model, va_loader, device)
            thresholds = optimize_thresholds(va_logits, labels_all[va_idx])
            va_probs = 1 / (1 + np.exp(-va_logits))
            va_preds = np.zeros_like(va_probs, dtype=int)
            for i in range(n_labels):
                va_preds[:, i] = (va_probs[:, i] >= thresholds[i]).astype(int)

            # Apply polarization mask: if sample is non-polarized, zero out
            va_preds[labels_polar_all[va_idx] == 0] = 0

            f1 = f1_score(labels_all[va_idx], va_preds, average="macro", zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_logits = va_logits.copy()
                no_improve = 0
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f"{model_key}_manif_seed{seed}_fold{fold}.pt"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        oof_logits[va_idx] = best_logits if best_logits is not None else 0
        print(f"    Fold {fold+1}: manif_macro_f1={best_f1:.4f}")

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof_logits


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "mps" and not config["mps_safe"]:
        print(f"WARNING: {config['name']} not MPS-safe, falling back to CPU.")
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Model: {config['name']}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    texts = df["text"].values
    labels_polar = df["polarization"].values
    labels_manif = df[MANIFESTATION_COLS].values
    print(f"Data: {len(df)} samples ({labels_polar.sum()} polarized)")

    tokenizer = AutoTokenizer.from_pretrained(config["name"])

    # Accumulate OOF across seeds
    all_polar_probs = np.zeros(len(df))
    all_manif_logits = np.zeros((len(df), len(MANIFESTATION_COLS)))

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n{'='*55}")
        print(f"  SEED {seed}")
        print(f"{'='*55}")

        # Stage A
        polar_probs = train_polarization_detector(
            texts, labels_polar, tokenizer, config, device, args.output_dir, seed
        )
        all_polar_probs += polar_probs

        # Stage B
        manif_logits = train_manifestation_classifier(
            texts, labels_manif, labels_polar, tokenizer, config, device,
            args.output_dir, args.model, seed
        )
        all_manif_logits += manif_logits

    # Average across seeds
    all_polar_probs /= len(SEEDS)
    all_manif_logits /= len(SEEDS)

    # ── Evaluate combined pipeline ──
    print(f"\n{'='*55}")
    print("  FINAL EVALUATION")
    print(f"{'='*55}")

    # Optimize polarization threshold for high recall
    best_polar_t, best_polar_f1 = 0.5, -1
    for t in np.arange(0.2, 0.8, 0.01):
        preds = (all_polar_probs >= t).astype(int)
        f = f1_score(labels_polar, preds, zero_division=0)
        if f > best_polar_f1:
            best_polar_f1 = f
            best_polar_t = t
    polar_preds = (all_polar_probs >= best_polar_t).astype(int)
    print(f"\n  Polarization: F1={best_polar_f1:.4f} (threshold={best_polar_t:.2f})")
    print(f"  Recall={((polar_preds==1) & (labels_polar==1)).sum() / labels_polar.sum():.4f}")

    # Optimize manifestation thresholds
    manif_thresholds = optimize_thresholds(all_manif_logits, labels_manif)
    manif_probs = 1 / (1 + np.exp(-all_manif_logits))
    manif_preds = np.zeros_like(manif_probs, dtype=int)
    for i in range(len(MANIFESTATION_COLS)):
        manif_preds[:, i] = (manif_probs[:, i] >= manif_thresholds[i]).astype(int)

    # Post-processing: non-polarized => all manifestation = 0
    manif_preds[polar_preds == 0] = 0

    # Also: if predicted polarized but no manifestation label is set, set the most confident one
    for idx in range(len(manif_preds)):
        if polar_preds[idx] == 1 and manif_preds[idx].sum() == 0:
            most_confident = manif_probs[idx].argmax()
            manif_preds[idx, most_confident] = 1

    macro_f1 = f1_score(labels_manif, manif_preds, average="macro", zero_division=0)
    print(f"\n  *** Subtask 3 OOF Macro F1: {macro_f1:.4f} ***")

    print("\n  Per-label F1:")
    for i, col in enumerate(MANIFESTATION_COLS):
        lf1 = f1_score(labels_manif[:, i], manif_preds[:, i], zero_division=0)
        print(f"    {col:20s}: {lf1:.4f}  (threshold={manif_thresholds[i]:.2f})")

    # Save OOF logits and config
    np.save(os.path.join(args.output_dir, f"{args.model}_polar_probs.npy"), all_polar_probs)
    np.save(os.path.join(args.output_dir, f"{args.model}_manif_logits.npy"), all_manif_logits)

    save_config = {
        "model": config["name"],
        "model_key": args.model,
        "polar_threshold": best_polar_t,
        "polar_f1": best_polar_f1,
        "manif_thresholds": dict(zip(MANIFESTATION_COLS, manif_thresholds.tolist())),
        "manif_macro_f1": macro_f1,
    }
    with open(os.path.join(args.output_dir, f"{args.model}_v2_config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    print(f"\n  Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()