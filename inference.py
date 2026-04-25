"""
Inference script for SemEval-2026 Task 9 - Subtask 3 (Telugu)
Loads trained model checkpoints, runs ensemble inference, applies
calibrated thresholds and post-processing.

Usage:
  python inference.py \
    --test_path tel_test.csv \
    --model_dir ./output \
    --config_path ./output/calibration_config.json \
    --output_path ./output/submission.csv
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

LABEL_COLS = [
    "polarization", "political", "racial/ethnic", "religious",
    "gender/sexual", "other", "stereotype", "vilification",
    "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"
]

N_FOLDS = 5
SEEDS = [42, 2026]


class PolarizationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


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
    mask = preds[:, 0] == 0
    preds[mask, 1:] = 0
    return preds


@torch.no_grad()
def predict_batch(model, loader, device):
    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./output")
    parser.add_argument("--config_path", type=str, default="./output/calibration_config.json")
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
    print(f"Using device: {device}")

    # Load config
    with open(args.config_path) as f:
        config = json.load(f)

    ensemble_weights = config["ensemble_weights"]
    thresholds = np.array([config["thresholds"][col] for col in LABEL_COLS])
    model_names = config["models"]

    # Load test data
    df_test = pd.read_csv(args.test_path)
    texts = df_test["text"].values
    print(f"Loaded {len(df_test)} test samples")

    # Inference per model
    all_logits = {}
    for model_key, model_name in model_names.items():
        print(f"\nInference: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        test_dataset = PolarizationDataset(texts, tokenizer, max_len=128)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        model_logits_accum = np.zeros((len(texts), len(LABEL_COLS)))
        count = 0

        for seed in SEEDS:
            for fold in range(N_FOLDS):
                ckpt_path = os.path.join(
                    args.model_dir, f"{model_key}_seed{seed}_fold{fold}.pt"
                )
                if not os.path.exists(ckpt_path):
                    print(f"  WARNING: {ckpt_path} not found, skipping")
                    continue

                model = MultiLabelClassifier(model_name, len(LABEL_COLS)).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

                logits = predict_batch(model, test_loader, device)
                model_logits_accum += logits
                count += 1

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if count > 0:
            all_logits[model_key] = model_logits_accum / count
            print(f"  Averaged over {count} checkpoints")

    # Weighted ensemble
    ensemble_logits = sum(
        ensemble_weights[k] * all_logits[k] for k in all_logits
    )

    # Apply thresholds
    probs = 1 / (1 + np.exp(-ensemble_logits))
    preds = np.zeros_like(probs, dtype=int)
    for i in range(len(LABEL_COLS)):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

    # Post-processing
    preds = apply_post_processing(preds)

    # Build submission
    submission = pd.DataFrame({"id": df_test["id"]})
    for i, col in enumerate(LABEL_COLS):
        submission[col] = preds[:, i]

    submission.to_csv(args.output_path, index=False)
    print(f"\nSubmission saved to {args.output_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nLabel distribution in predictions:")
    for col in LABEL_COLS:
        print(f"  {col:20s}: {submission[col].sum()}/{len(submission)}")


if __name__ == "__main__":
    main()