import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "model_output"
INPUT_CSV = "data/raw/input_texts.csv"
OUTPUT_CSV = "results/predictions.csv"
TEXT_COLUMN = "text"
MAX_LENGTH = 128

os.makedirs("results", exist_ok=True)

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Load Input Data
# -----------------------------
print(f"Loading input data from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Input CSV must contain a '{TEXT_COLUMN}' column")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return sentiment, round(confidence.item(), 4)

# -----------------------------
# Run Batch Prediction
# -----------------------------
print("Running batch predictions...")
sentiments = []
confidences = []

for text in df[TEXT_COLUMN]:
    sentiment, confidence = predict_sentiment(str(text))
    sentiments.append(sentiment)
    confidences.append(confidence)

df["sentiment"] = sentiments
df["confidence"] = confidences

# -----------------------------
# Save Output
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f" Batch predictions saved to {OUTPUT_CSV}")
