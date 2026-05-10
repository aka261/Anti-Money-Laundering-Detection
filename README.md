# Anti-Money Laundering Detection System

An end to end machine learning project that detects suspicious money laundering transactions using multiple models, SHAP explainability, MLflow tracking and a REST API.

---

## Project Overview

Money laundering is the process of making illegally obtained money appear legitimate. Banks and financial institutions lose billions of dollars every year due to money laundering. This project builds an automated system that flags suspicious transactions in real time using machine learning.

---

## Features

- Generates 50,000 realistic bank transactions with normal and suspicious patterns
- Trains and compares 3 machine learning models
- Handles class imbalance using SMOTE
- SHAP values for model explainability
- MLflow experiment tracking for all model runs
- FastAPI REST API for real time transaction scoring
- Groq AI explanation of why a transaction is suspicious
- Interactive Streamlit dashboard with 6 pages

---

## Machine Learning Models

| Model | Description |
|---|---|
| Random Forest | Builds 100 decision trees and takes majority vote |
| XGBoost | Builds trees sequentially fixing errors of previous trees |
| Neural Network | Deep learning model with 3 layers using TensorFlow |

---

## Project Structure

```
AML_Detection/
  app.py                  - Main Streamlit application
  api.py                  - FastAPI REST API
  requirements.txt        - Required packages
  Dockerfile              - Docker configuration
  docker-compose.yml      - Docker Compose configuration
  Procfile                - Heroku deployment
  deploy.sh               - Deployment script
  .env.example            - Environment variables template
  data/
    data_generator.py     - Generates realistic bank transaction dataset
  models/
    trainer.py            - Trains all 3 models with MLflow tracking
  utils/
    explainer.py          - Groq AI explanation of results
    shap_explainer.py     - SHAP value calculation and visualization
  k8s/
    deployment.yml        - Kubernetes deployment
    secrets.yml           - Kubernetes secrets
    service.yml           - Kubernetes service
```

---

## Dashboard Pages

1. **Dataset Overview** - Charts showing transaction distribution, amount distribution, hour of day analysis
2. **Model Comparison** - Side by side comparison of all 3 models with confusion matrices
3. **Transaction Risk Checker** - Enter transaction details and get a risk score with gauge chart
4. **SHAP Explainability** - Feature importance charts showing what drives predictions
5. **Suspicious Transactions** - View all flagged suspicious transactions
6. **MLflow Tracking** - Instructions to view experiment tracking UI

---

## Tech Stack

- **Language** - Python 3.11
- **Machine Learning** - Scikit-learn, XGBoost, TensorFlow
- **Explainability** - SHAP
- **Experiment Tracking** - MLflow
- **API** - FastAPI, Uvicorn
- **Frontend** - Streamlit
- **Visualization** - Plotly
- **AI Explanation** - Groq (Llama 3.3 70B)
- **Deployment** - Docker, Kubernetes, Google Cloud Run

---

## Installation

**Step 1 - Clone the repository**
```
git clone https://github.com/aka261/Anti-Money-Laundering-Detection.git
cd Anti-Money-Laundering-Detection
```

**Step 2 - Create virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

**Step 3 - Install packages**
```
pip install -r requirements.txt
```

**Step 4 - Set up environment variables**
```
copy .env.example .env
```
Add your Groq API key to the `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Step 5 - Run the app**
```
streamlit run app.py
```

**Step 6 - Click Generate Data and Train Models button in the app**

---

## REST API

Run the API separately:
```
uvicorn api:app --reload
```

Open API documentation at:
```
http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Check if API is running |
| GET | /health | Health check |
| POST | /predict | Get risk score for a transaction |
| GET | /models | List available models |

### Sample API Request

```json
POST /predict
{
  "amount": 9500,
  "transaction_type": "transfer",
  "hour": 2,
  "day_of_week": 6,
  "num_transactions_sender": 80,
  "num_transactions_receiver": 75,
  "same_bank": 0,
  "international": 1,
  "model": "random_forest"
}
```

### Sample API Response

```json
{
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "model_used": "random_forest",
  "transaction": {...}
}
```

---

## MLflow Tracking

View all model experiments:
```
mlflow ui
```
Open http://localhost:5000 in your browser.

MLflow tracks:
- Model parameters (n_estimators, epochs, batch_size)
- Metrics (accuracy, precision, recall, F1, AUC)
- Model artifacts

---

## Docker Deployment

```
docker-compose up --build
```

App will be available at http://localhost:8501

---

## Risk Score Interpretation

| Risk Score | Risk Level | Action |
|---|---|---|
| 0.0 to 0.4 | LOW | No action needed |
| 0.4 to 0.7 | MEDIUM | Manual review recommended |
| 0.7 to 1.0 | HIGH | Flag for investigation |

---

## Money Laundering Patterns Detected

- High number of transactions per day (velocity)
- International transfers
- Unusual transaction timing
- Large transaction amounts
- Cross bank transfers

---

## Author

Akash
GitHub: https://github.com/aka261
