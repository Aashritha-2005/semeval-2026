import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLUMNS = [
    "polarization", "political", "racial/ethnic", "religious",
    "gender/sexual", "other", "stereotype", "vilification",
    "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"
]

# =========================
# DATASET
# =========================
class TeluguDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
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
# MODEL
# =========================
class ModelWrapper:
    def __init__(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        ).to(DEVICE)

# =========================
# CLASS WEIGHTS (CRITICAL)
# =========================
def get_pos_weights(labels):
    pos_counts = labels.sum(axis=0)
    total = len(labels)
    weights = total / (pos_counts + 1)  # avoid divide by zero
    return torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

# =========================
# TRAIN
# =========================
def train_model(wrapper, train_loader, pos_weights):
    optimizer = AdamW(wrapper.model.parameters(), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    wrapper.model.train()

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}")
        loop = tqdm(train_loader)

        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = wrapper.model(**batch)
            logits = outputs.logits

            loss = loss_fn(logits, batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    return wrapper

# =========================
# PREDICT
# =========================
def get_probs(wrapper, texts):
    wrapper.model.eval()

    enc = wrapper.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = wrapper.model(**enc).logits
        probs = torch.sigmoid(logits)

    return probs.cpu().numpy()

# =========================
# ENSEMBLE
# =========================
def ensemble_predict(models, texts):
    probs = []
    for model in models:
        probs.append(get_probs(model, texts))
    return np.mean(probs, axis=0)

# =========================
# THRESHOLD TUNING (🔥 KEY)
# =========================
def tune_thresholds(y_true, probs):
    best_thresh = []

    for i in range(probs.shape[1]):
        best_f1 = 0
        best_t = 0.5

        for t in np.arange(0.1, 0.6, 0.05):
            pred = (probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresh.append(best_t)

    return best_thresh

# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("tel_train.csv")

    print("Columns:", df.columns)

    texts = df["text"]
    labels = df[LABEL_COLUMNS]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    num_labels = len(LABEL_COLUMNS)

    # 🔥 USE ONLY PUBLIC MODELS
    model_names = [
        "google/muril-base-cased",
        "xlm-roberta-base"
    ]

    pos_weights = get_pos_weights(train_labels)

    trained_models = []

    for name in model_names:
        print(f"\n🚀 Training {name}")

        wrapper = ModelWrapper(name, num_labels)

        dataset = TeluguDataset(train_texts, train_labels, wrapper.tokenizer)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        wrapper = train_model(wrapper, loader, pos_weights)
        trained_models.append(wrapper)

    # =========================
    # EVALUATION
    # =========================
    print("\n🔍 Evaluating...")

    probs = ensemble_predict(trained_models, val_texts.tolist())

    # 🔥 THRESHOLD TUNING
    thresholds = tune_thresholds(val_labels.values, probs)
    print("Best thresholds:", thresholds)

    preds = np.zeros_like(probs)
    for i in range(len(thresholds)):
        preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)

    score = f1_score(val_labels.values, preds, average="macro")

    print("\n🔥 FINAL MACRO F1:", score)


if __name__ == "__main__":
    main()