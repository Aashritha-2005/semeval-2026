import pandas as pd

train = pd.read_csv("tel_train.csv")
harmful = pd.read_csv("tel_harmful.csv")

if "text" in harmful.columns:
    harmful = harmful.drop(columns=["text"])

harmful = harmful.rename(columns={"translated_te": "text"})


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

df = pd.concat([train, harmful], ignore_index=True)


df = df[df["text"].notna()]
df = df[df["text"].str.strip() != ""]

df["label_sum"] = df[LABEL_COLUMNS].sum(axis=1)


# KEEP SOME ZERO ROWS
positive_df = df[df["label_sum"] > 0]
zero_df = df[df["label_sum"] == 0]

# keep only 30% of zero rows
zero_df = zero_df.sample(frac=0.3, random_state=42)

df = pd.concat([positive_df, zero_df], ignore_index=True)

print("After balancing zeros:", len(df))

# OVERSAMPLE RARE LABELS
rare_labels = ["dehumanization", "stereotype"]

rare_df = df[df[rare_labels].sum(axis=1) > 0]

# duplicate rare samples (controlled)
df = pd.concat([df, rare_df], ignore_index=True)

print("After oversampling:", len(df))

df = df.drop(columns=["label_sum"])

# shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("cleaned_train.csv", index=False)

print("✅ FINAL DATASET READY: cleaned_train.csv")
