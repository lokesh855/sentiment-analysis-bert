import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "bert-base-uncased"
TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_OUTPUT_DIR = "model_output"
RESULTS_DIR = "results"

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SEED = 42

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# -----------------------------
# Metrics (HF v5 compatible)
# -----------------------------
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Starting BERT fine-tuning...")

    # Tokenizer & Model
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # Load datasets
    train_dataset, test_dataset = load_data()

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set torch format
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False
    )

    # Trainer (NO tokenizer arg in v5)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    # -----------------------------
    # Save Metrics
    # -----------------------------
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    # -----------------------------
    # Save Run Summary
    # -----------------------------
    run_summary = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "metrics": {
            "accuracy": eval_results["eval_accuracy"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            "f1": eval_results["eval_f1"]
        }
    }

    summary_path = os.path.join(RESULTS_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=4)

    # -----------------------------
    # Save Model Artifacts
    # -----------------------------
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("Training complete!")
    print(f"Metrics saved to {metrics_path}")
    print(f"Run summary saved to {summary_path}")
    print(f"Model saved to {MODEL_OUTPUT_DIR}")
