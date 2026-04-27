import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm
import re

# =========================
# SEED (STABILITY)
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# =========================
# MODEL
# =========================
class ModelWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(LABEL_COLUMNS),
            problem_type="multi_label_classification",
            hidden_dropout_prob=0.2
        ).to(DEVICE)

        # 🔥 FREEZE EMBEDDINGS (STABILITY)
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

# =========================
# CLASS WEIGHTS
# =========================
def get_weights(labels):
    pos = labels.sum(axis=0)
    neg = len(labels) - pos

    weights = neg / (pos + 1)
    weights = np.clip(weights, 1.0, 2.5)

    return torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

# =========================
# TRAIN
# =========================
def train_model(wrapper, loader, weights):
    optimizer = AdamW(wrapper.model.parameters(), lr=1e-5)
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

# =========================
# PREDICT
# =========================
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
    df = pd.read_csv("tel_train.csv")

    # 🔥 UPSAMPLE RARE LABELS
    rare_labels = ["dehumanization", "stereotype"]
    rare_df = df[df[rare_labels].sum(axis=1) > 0]

    df = pd.concat([df, rare_df, rare_df])

    texts = df["text"]
    labels = df[LABEL_COLUMNS]

    print("\n📊 LABEL DISTRIBUTION:\n")
    print(labels.sum())

    # 🔥 STRATIFIED SPLIT
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in mskf.split(texts, labels):
        train_texts = texts.iloc[train_idx]
        val_texts = texts.iloc[val_idx]
        train_labels = labels.iloc[train_idx]
        val_labels = labels.iloc[val_idx]
        break

    wrapper = ModelWrapper()

    dataset = TeluguDataset(train_texts, train_labels, wrapper.tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    weights = get_weights(train_labels)

    wrapper = train_model(wrapper, loader, weights)

    print("\n🔍 Evaluating...")

    probs = predict(wrapper, val_texts.tolist())

    thresholds = tune_thresholds(val_labels.values, probs)

    # 🔥 THRESHOLD FLOOR (STABILITY)
    thresholds = [max(t, 0.25) for t in thresholds]

    print("Final thresholds:", thresholds)

    preds = np.zeros_like(probs)

    for i in range(len(thresholds)):
        preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)

    score = f1_score(val_labels.values, preds, average="macro")

    print("\n🔥 FINAL MACRO F1:", score)


if __name__ == "__main__":
    main()
