import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import re

# =========================
# SETUP
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

LABEL_COLUMNS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    return re.sub(r'[^\u0C00-\u0C7F\s]', '', str(text))

# =========================
# DATASET
# =========================
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        texts = texts.apply(clean_text)

        self.encodings = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# =========================
# CLASS WEIGHTS (KEY FIX)
# =========================
def get_class_weights(labels):
    pos = labels.sum(axis=0)
    neg = len(labels) - pos

    weights = neg / (pos + 1)
    weights = np.clip(weights, 1.0, 3.0)

    return torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

# =========================
# TRAIN
# =========================
def train_model(model, loader, weights):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

    model.train()

    for epoch in range(4):
        print(f"\nEpoch {epoch+1}")
        loop = tqdm(loader)

        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            logits = model(**batch).logits
            loss = loss_fn(logits, batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    return model

# =========================
# PREDICT
# =========================
def predict(model, tokenizer, texts):
    model.eval()

    texts = [clean_text(t) for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits)

    return probs.cpu().numpy()

# =========================
# THRESHOLD TUNING
# =========================
def tune_thresholds(y_true, probs):
    thresholds = []

    for i in range(probs.shape[1]):
        best_t = 0.3
        best_f1 = 0

        for t in np.arange(0.1, 0.6, 0.05):
            pred = (probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return thresholds

# =========================
# MAIN
# =========================
def main():
    print("🚀 Loading cleaned dataset...")
    df = pd.read_csv("cleaned_train.csv")

    texts = df["text"]
    labels = df[LABEL_COLUMNS]

    print("\n📊 Label Distribution:\n")
    print(labels.sum())

    # SPLIT
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # MODEL
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification"
    ).to(DEVICE)

    # DATA
    dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 🔥 CLASS WEIGHTS
    weights = get_class_weights(train_labels)

    # TRAIN
    model = train_model(model, loader, weights)

    # EVALUATE
    print("\n🔍 Evaluating...")
    probs = predict(model, tokenizer, val_texts.tolist())

    thresholds = tune_thresholds(val_labels.values, probs)
    print("Tuned thresholds:", thresholds)

    preds = np.zeros_like(probs)

    for i in range(len(thresholds)):
        preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)

    score = f1_score(val_labels.values, preds, average="macro")

    print("\n🔥 FINAL MACRO F1:", score)


if __name__ == "__main__":
    main()