"""
Inference for v2 pipeline (Subtask 3 only).

Usage:
  python inference_v2.py --test_path tel_test.csv --device mps
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

MANIFESTATION_COLS = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]

N_FOLDS = 5
SEEDS = [42, 2026]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class MultiPoolClassifier(nn.Module):
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
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        cls_token = hidden[:, 0, :]
        mean_pool = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        hidden_masked = hidden.clone()
        hidden_masked[mask.squeeze(-1) == 0] = -1e9
        max_pool = hidden_masked.max(dim=1).values
        concat = torch.cat([cls_token, mean_pool, max_pool], dim=-1)
        x = self.dropout(concat)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@torch.no_grad()
def predict_batch(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_logits.append(logits.cpu().float().numpy())
    return np.concatenate(all_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--config_path", type=str, default="./output/ensemble_v2_config.json")
    parser.add_argument("--output_path", type=str, default="./output/submission.csv")
    parser.add_argument("--device", type=str, default="auto")
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

    with open(args.config_path) as f:
        config = json.load(f)

    weights = config["weights"]
    polar_threshold = config["polar_threshold"]
    manif_thresholds = np.array([config["manif_thresholds"][c] for c in MANIFESTATION_COLS])
    model_names = config["models"]

    df_test = pd.read_csv(args.test_path)
    texts = df_test["text"].values
    print(f"Test samples: {len(df_test)}")

    # Inference per model
    all_polar_probs = {}
    all_manif_logits = {}

    for model_key, model_name in model_names.items():
        print(f"\nInference: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        test_ds = TextDataset(texts, tokenizer, max_len=128)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        # Polarization
        polar_accum = np.zeros(len(texts))
        polar_count = 0
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                path = os.path.join(args.model_dir, f"polar_seed{seed}_fold{fold}.pt")
                if not os.path.exists(path):
                    continue
                model = MultiPoolClassifier(model_name, num_labels=1).to(device).float()
                model.load_state_dict(torch.load(path, map_location=device))
                logits = predict_batch(model, test_loader, device)
                polar_accum += 1 / (1 + np.exp(-logits.flatten()))
                polar_count += 1
                del model
        if polar_count > 0:
            all_polar_probs[model_key] = polar_accum / polar_count

        # Manifestation
        manif_accum = np.zeros((len(texts), len(MANIFESTATION_COLS)))
        manif_count = 0
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                path = os.path.join(args.model_dir, f"{model_key}_manif_seed{seed}_fold{fold}.pt")
                if not os.path.exists(path):
                    continue
                model = MultiPoolClassifier(model_name, num_labels=len(MANIFESTATION_COLS)).to(device).float()
                model.load_state_dict(torch.load(path, map_location=device))
                logits = predict_batch(model, test_loader, device)
                manif_accum += logits
                manif_count += 1
                del model
        if manif_count > 0:
            all_manif_logits[model_key] = manif_accum / manif_count

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Weighted ensemble
    combined_polar = sum(weights[k] * all_polar_probs[k] for k in all_polar_probs)
    combined_manif = sum(weights[k] * all_manif_logits[k] for k in all_manif_logits)

    # Predictions
    polar_preds = (combined_polar >= polar_threshold).astype(int)
    manif_probs = 1 / (1 + np.exp(-combined_manif))
    manif_preds = np.zeros_like(manif_probs, dtype=int)
    for i in range(len(MANIFESTATION_COLS)):
        manif_preds[:, i] = (manif_probs[:, i] >= manif_thresholds[i]).astype(int)

    # Post-processing
    manif_preds[polar_preds == 0] = 0
    for idx in range(len(manif_preds)):
        if polar_preds[idx] == 1 and manif_preds[idx].sum() == 0:
            manif_preds[idx, manif_probs[idx].argmax()] = 1

    # Build submission
    submission = pd.DataFrame({"id": df_test["id"]})
    for i, col in enumerate(MANIFESTATION_COLS):
        submission[col] = manif_preds[:, i]

    submission.to_csv(args.output_path, index=False)
    print(f"\nSubmission saved to {args.output_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nPrediction distribution:")
    for col in MANIFESTATION_COLS:
        print(f"  {col:20s}: {submission[col].sum()}/{len(submission)}")


if __name__ == "__main__":
    main()