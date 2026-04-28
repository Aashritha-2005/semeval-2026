import pandas as pd

# =========================
# LOAD FILES
# =========================
train = pd.read_csv("tel_train.csv")
harmful = pd.read_csv("tel_harmful.csv")

# =========================
# FIX HARMFUL DATASET 🔥
# =========================
# Remove wrong text column
if "text" in harmful.columns:
    harmful = harmful.drop(columns=["text"])

# Rename translated_te → text
harmful = harmful.rename(columns={"translated_te": "text"})

# =========================
# KEEP ONLY REQUIRED COLUMNS
# =========================
LABEL_COLUMNS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

REQUIRED_COLUMNS = ["text"] + LABEL_COLUMNS

train = train[REQUIRED_COLUMNS]
harmful = harmful[REQUIRED_COLUMNS]

# =========================
# MERGE
# =========================
df = pd.concat([train, harmful], ignore_index=True)

# =========================
# CLEAN TEXT (remove empty)
# =========================
df = df[df["text"].notna()]
df = df[df["text"].str.strip() != ""]

# =========================
# LABEL SUM
# =========================
df["label_sum"] = df[LABEL_COLUMNS].sum(axis=1)

# =========================
# ⚠ KEEP SOME ZERO ROWS (IMPORTANT)
# =========================
positive_df = df[df["label_sum"] > 0]
zero_df = df[df["label_sum"] == 0]

# keep only 30% of zero rows
zero_df = zero_df.sample(frac=0.3, random_state=42)

df = pd.concat([positive_df, zero_df], ignore_index=True)

print("After balancing zeros:", len(df))

# =========================
# 🔥 OVERSAMPLE RARE LABELS
# =========================
rare_labels = ["dehumanization", "stereotype"]

rare_df = df[df[rare_labels].sum(axis=1) > 0]

# duplicate rare samples (controlled)
df = pd.concat([df, rare_df], ignore_index=True)

print("After oversampling:", len(df))

# =========================
# FINAL CLEANUP
# =========================
df = df.drop(columns=["label_sum"])

# shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# SAVE
# =========================
df.to_csv("cleaned_train.csv", index=False)

print("✅ FINAL DATASET READY: cleaned_train.csv")