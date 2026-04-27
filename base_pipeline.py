import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLUMNS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# CLEAN TEXT
def clean_text(text):
    return re.sub(r'[^\u0C00-\u0C7F\s]', '', str(text))

# DATASET
class TeluguDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        texts = texts.apply(clean_text)

        self.encodings = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# MODEL
class ModelWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(LABEL_COLUMNS),
            problem_type="multi_label_classification",
            hidden_dropout_prob=0.2
        ).to(DEVICE)

# 🔥 BALANCED CLASS WEIGHTS (FINAL FIX)
def get_weights(labels):
    pos = labels.sum(axis=0)
    neg = len(labels) - pos

    weights = neg / (pos + 1)
    weights = np.clip(weights, 1.0, 2.5)   # NOT too high

    return torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

# TRAIN
def train_model(wrapper, loader, weights):
    optimizer = AdamW(wrapper.model.parameters(), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

    wrapper.model.train()

    for epoch in range(4):
        print(f"\nEpoch {epoch+1}")
        loop = tqdm(loader)

        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            logits = wrapper.model(**batch).logits
            loss = loss_fn(logits, batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    return wrapper

# PREDICT
def predict(wrapper, texts):
    wrapper.model.eval()

    texts = [clean_text(t) for t in texts]

    enc = wrapper.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt"
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = wrapper.model(**enc).logits
        probs = torch.sigmoid(logits)

    return probs.cpu().numpy()

# MAIN
def main():
    df = pd.read_csv("tel_train.csv")

    texts = df["text"]
    labels = df[LABEL_COLUMNS]

    print("\n📊 LABEL DISTRIBUTION:\n")
    print(labels.sum())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    wrapper = ModelWrapper()

    dataset = TeluguDataset(train_texts, train_labels, wrapper.tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    weights = get_weights(train_labels)

    wrapper = train_model(wrapper, loader, weights)

    print("\n🔍 Evaluating...")

    probs = predict(wrapper, val_texts.tolist())

    # 🔥 CRITICAL: LOWER THRESHOLD
    preds = (probs > 0.35).astype(int)

    score = f1_score(val_labels.values, preds, average="macro")

    print("\n🔥 FINAL MACRO F1:", score)


if __name__ == "__main__":
    main()