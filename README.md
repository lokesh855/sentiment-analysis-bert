
# Sentiment Analysis System using BERT (MLOps Project)

This project implements an end-to-end **sentiment analysis system** using a **pre-trained BERT model**. It demonstrates a complete **MLOps pipeline**, including data preprocessing, model fine-tuning, evaluation, batch inference, REST API deployment, and a web-based user interface.

The system is fully **containerized with Docker** and can be launched using a single command.

---

## 🚀 Features

- Pre-trained **BERT (bert-base-uncased)** fine-tuned for sentiment analysis
- Data preprocessing using Hugging Face Datasets
- Model training and evaluation with standard NLP metrics
- Experiment tracking via JSON-based run summaries
- REST API built with **FastAPI**
- Interactive web UI using **Streamlit**
- Batch prediction for large CSV files
- Fully Dockerized with health checks

---

## 🗂️ Project Structure

```text
sentiment-analysis-bert/
│
├── data/
│   ├── raw/                # Raw input datasets
│   └── processed/          # Cleaned train/test CSVs
│
├── model_output/           # Fine-tuned model artifacts
│
├── results/
│   ├── metrics.json        # Evaluation metrics
│   └── run_summary.json    # Experiment tracking output
│
├── scripts/
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model fine-tuning
│   └── batch_predict.py    # Batch inference
│
├── src/
│   ├── api.py              # FastAPI backend
│   └── ui.py               # Streamlit UI
│
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── env.example
├── requirements.txt
└── README.md

## 🧠 Model & Dataset

Model: bert-base-uncased

Dataset: IMDb Movie Reviews (via Hugging Face Datasets)

Task: Binary sentiment classification (Positive / Negative)

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone <repository-url>
cd sentiment-analysis-bert

2️⃣ (Optional) Create Virtual Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

🔄 Pipeline Usage
Step 1: Data Preprocessing
python scripts/preprocess.py


Output:

data/processed/train.csv

data/processed/test.csv

Step 2: Model Training & Evaluation
python scripts/train.py


Output:

Fine-tuned model → model_output/

Metrics → results/metrics.json

Run summary → results/run_summary.json

Step 3: Batch Prediction
python scripts/batch_predict.py


Input:

CSV with a text column

Output:

results/predictions.csv

🌐 API Usage
Start API (Local)
uvicorn src.api:app --host 0.0.0.0 --port 8000

Health Check
GET /health


Response:

{ "status": "ok" }

Sentiment Prediction
POST /predict


Request:

{
  "text": "This movie was amazing!"
}


Response:

{
  "sentiment": "Positive",
  "confidence": 0.98
}

🖥️ Web Interface (Streamlit)
streamlit run src/ui.py


Open: http://localhost:8501

Enter text and view predictions in real time

🐳 Docker Deployment (Recommended)
Build & Run Everything
docker-compose up --build

Access Services
Service	URL
API	http://localhost:8000


UI	http://localhost:8501


📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Metrics are saved to:

results/metrics.json

💼 Real-World Applications

Social media sentiment monitoring

Customer feedback analysis

Brand reputation tracking

Review classification systems

🏁 Conclusion

This project demonstrates how to build a production-ready NLP system using modern tools like Hugging Face, FastAPI, Docker, and Streamlit, following best practices in MLOps and model deployment.