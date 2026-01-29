import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "model_output"
MODEL_NAME = "bert-base-uncased"

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for BERT-based sentiment analysis",
    version="1.0.0"
)

# -----------------------------
# Load Model & Tokenizer (Startup)
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# -----------------------------
# Request Schema
# -----------------------------
class PredictRequest(BaseModel):
    text: str

# -----------------------------
# Response Schema
# -----------------------------
class PredictResponse(BaseModel):
    sentiment: str
    confidence: float

# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    """
    Predict sentiment for input text.
    Returns sentiment label and confidence score.
    """

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence.item(), 4)
    }
