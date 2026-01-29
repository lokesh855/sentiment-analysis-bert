import re
import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_SAMPLES = 500  # set None to use full dataset

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:
    """
    Cleans input text by:
    - Removing URLs
    - Removing HTML tags
    - Removing special characters
    - Lowercasing
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Load IMDb Dataset from Kaggle
# -----------------------------
def load_imdb_dataset():
    print("Downloading IMDb dataset from Kaggle...")

    dataset_path = kagglehub.dataset_download(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

    csv_path = os.path.join(dataset_path, "IMDB Dataset.csv")
    df = pd.read_csv(csv_path)

    # Convert sentiment to label
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df[["review", "label"]]
    df.rename(columns={"review": "text"}, inplace=True)

    return df


# -----------------------------
# Preprocess Dataset
# -----------------------------
def preprocess_data(df):
    print("Cleaning text data...")
    df["text"] = df["text"].apply(clean_text)

    if MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)

    return df


# -----------------------------
# Train-Test Split & Save
# -----------------------------
def split_and_save(df):
    print("Splitting dataset...")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_imdb_dataset()
    df = preprocess_data(df)
    split_and_save(df)

    print("Data preprocessing completed successfully!")
