"""
Quick single-model trainer for fast iteration.
Use this to test one model at a time before running the full ensemble.

Usage (Mac, ~20 min per model):
  python train_single.py --data_path tel_train.csv --model mdeberta --device mps
  python train_single.py --data_path tel_train.csv --model xlmr --device mps
  python train_single.py --data_path tel_train.csv --model muril --device mps

Usage (Colab GPU, ~5 min per model):
  python train_single.py --data_path tel_train.csv --model mdeberta --device cuda
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

LABEL_COLS = [
    "polarization", "political", "racial/ethnic", "religious",
    "gender/sexual", "other", "stereotype", "vilification",
    "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"
]

MODEL_CONFIGS = {
    "mdeberta": {"name": "microsoft/mdeberta-v3-base", "max_len": 128, "lr": 2e-5, "epochs": 8, "batch_size": 16},
    "xlmr": {"name": "xlm-roberta-base", "max_len": 128, "lr": 2e-5, "epochs": 8, "batch_size": 16},
    "muril": {"name": "google/muril-base-cased", "max_len": 128, "lr": 2e-5, "epochs": 8, "batch_size": 16},
}


class PolarizationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels=12, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def apply_post_processing(preds):
    preds = preds.copy()
    preds[preds[:, 0] == 0, 1:] = 0
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    config = MODEL_CONFIGS[args.model]
    print(f"Model: {config['name']}")

    df = pd.read_csv(args.data_path)
    texts = df["text"].values
    labels = df[LABEL_COLS].values
    print(f"Data: {len(df)} samples")

    tokenizer = AutoTokenizer.from_pretrained(config["name"])

    oof_logits = np.zeros((len(df), len(LABEL_COLS)))
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels[:, 0])):
        print(f"\nFold {fold+1}/{args.n_folds}")

        train_ds = PolarizationDataset(texts[train_idx], labels[train_idx], tokenizer, config["max_len"])
        val_ds = PolarizationDataset(texts[val_idx], labels[val_idx], tokenizer, config["max_len"])
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"]*2, shuffle=False, num_workers=0)

        model = MultiLabelClassifier(config["name"], len(LABEL_COLS)).to(device)

        # Class weights
        pos_counts = labels[train_idx].sum(axis=0)
        neg_counts = len(train_idx) - pos_counts
        pos_weight = torch.tensor(np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 20.0), dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
        total_steps = len(train_loader) * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

        best_f1 = -1
        best_logits = None
        patience = 3
        no_improve = 0

        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            model.eval()
            val_logits_list = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    val_logits_list.append(logits.cpu().numpy())
            val_logits = np.concatenate(val_logits_list)

            val_probs = 1 / (1 + np.exp(-val_logits))
            val_preds = (val_probs >= 0.5).astype(int)
            val_preds = apply_post_processing(val_preds)
            val_f1 = f1_score(labels[val_idx], val_preds, average="macro", zero_division=0)

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  macro_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_logits = val_logits.copy()
                no_improve = 0
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f"{args.model}_seed{args.seed}_fold{fold}.pt"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        oof_logits[val_idx] = best_logits
        fold_f1s.append(best_f1)
        print(f"  Best fold F1: {best_f1:.4f}")

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Threshold calibration
    print(f"\n{'='*50}")
    print(f"Mean fold F1 (default 0.5 threshold): {np.mean(fold_f1s):.4f}")

    probs = 1 / (1 + np.exp(-oof_logits))
    best_thresholds = np.zeros(len(LABEL_COLS))
    for i in range(len(LABEL_COLS)):
        best_t, best_f = 0.5, -1
        for t in np.arange(0.1, 0.9, 0.01):
            f = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t
        best_thresholds[i] = best_t

    final_preds = np.zeros_like(probs, dtype=int)
    for i in range(len(LABEL_COLS)):
        final_preds[:, i] = (probs[:, i] >= best_thresholds[i]).astype(int)
    final_preds = apply_post_processing(final_preds)

    final_f1 = f1_score(labels, final_preds, average="macro", zero_division=0)
    print(f"OOF Macro F1 (calibrated thresholds): {final_f1:.4f}")

    print("\nPer-label F1:")
    for i, col in enumerate(LABEL_COLS):
        lf1 = f1_score(labels[:, i], final_preds[:, i], zero_division=0)
        print(f"  {col:20s}: {lf1:.4f}  (threshold={best_thresholds[i]:.2f})")

    # Save OOF logits for ensemble later
    np.save(os.path.join(args.output_dir, f"{args.model}_oof_logits.npy"), oof_logits)

    # Save thresholds
    thresh_config = {
        "model": config["name"],
        "model_key": args.model,
        "thresholds": dict(zip(LABEL_COLS, best_thresholds.tolist())),
        "macro_f1": final_f1,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, f"{args.model}_config.json"), "w") as f:
        json.dump(thresh_config, f, indent=2)

    print(f"\nSaved: {args.model}_oof_logits.npy, {args.model}_config.json")


if __name__ == "__main__":
    main()