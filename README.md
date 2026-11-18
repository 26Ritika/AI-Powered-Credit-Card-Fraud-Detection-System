# AI-Powered Credit Card Fraud Detection System

An end-to-end machine learning project designed to identify potentially fraudulent credit card transactions using supervised learning, probability calibration, threshold optimization, model explainability, and a FastAPI prediction service.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem because legitimate transactions significantly outnumber fraudulent transactions.

This project aims to develop a machine learning pipeline that identifies potentially fraudulent transactions while considering the trade-off between false positives and false negatives.

The system includes:

- Data preprocessing and validation.
- Exploratory data analysis.
- Handling class imbalance.
- Training and comparison of multiple machine learning models.
- Model evaluation using fraud-focused metrics.
- Probability calibration.
- Business-cost-based threshold optimization.
- Model explainability using SHAP.
- REST API development using FastAPI.
- Data drift monitoring using Population Stability Index (PSI).
- Automated testing using Pytest.
- Containerization using Docker.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Model Explainability | SHAP |
| Visualization | Matplotlib |
| API Development | FastAPI, Pydantic |
| Model Serialization | Joblib |
| Testing | Pytest |
| Deployment | Docker |

## Dataset

This project uses the Credit Card Fraud Detection dataset available on Kaggle.

**Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset information:

- Total transactions: 284,807
- Fraudulent transactions: 492
- Total features: 30
- Target column: `Class`
- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset contains anonymized transaction features (`V1` to `V28`), transaction time, transaction amount, and the target class.

### Dataset Setup

1. Download `creditcard.csv` from Kaggle.
2. Create a `data` directory in the project root if it does not exist.
3. Place the downloaded CSV file inside the directory.

Expected location:

```text
data/creditcard.csv
```

## Project Structure

```text
AI-Fraud-Detection-System/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibration.py
│   ├── explainability.py
│   ├── threshold_optimization.py
│   └── pipeline.py
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       └── prediction.py
│
├── models/
│
├── monitoring/
│   ├── __init__.py
│   └── drift_detection.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_artifact.py
│   ├── test_preprocessing.py
│   └── test_drift_detection.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Fraud-Detection-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Kaggle dataset and place `creditcard.csv` inside the `data/` directory.

## Model Training Pipeline

Run the complete machine learning pipeline:

```bash
python -m src.pipeline
```

The pipeline is designed to perform the following operations:

1. Load and validate the dataset.
2. Perform data preprocessing.
3. Split the dataset into training, validation, and testing sets.
4. Scale features where required.
5. Train multiple machine learning models.
6. Compare models using validation PR-AUC.
7. Select the best-performing model.
8. Calibrate predicted probabilities.
9. Optimize the classification threshold using configurable business costs.
10. Evaluate the final model on the test dataset.
11. Save the trained model artifact.

### Models

The project compares the following algorithms:

- Logistic Regression
- Random Forest
- XGBoost

The best model is selected based on validation Precision-Recall AUC (PR-AUC).

### Model Artifact

The trained model is saved at:

```text
models/fraud_detection_model.joblib
```

The artifact contains the trained model, preprocessing information, classification threshold, feature names, and evaluation metadata.

## Model Evaluation

Because fraud detection involves highly imbalanced data, accuracy alone is not a reliable measure of model performance.

This project evaluates models using the following metrics:

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of flagged transactions that are actually fraudulent |
| Recall | Measures the proportion of actual fraud cases detected |
| F1-score | Balances precision and recall |
| F2-score | Places greater emphasis on recall |
| PR-AUC | Evaluates performance on imbalanced classification data |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| Brier Score | Measures the quality of predicted probabilities |

### Threshold Optimization

Instead of relying only on the default classification threshold of 0.5, the project optimizes the threshold using configurable false-positive and false-negative costs.

This helps demonstrate how business requirements can influence fraud detection decisions.

The example cost values in the implementation are illustrative assumptions, not actual banking loss estimates.

## Model Explainability

The project uses SHAP (SHapley Additive exPlanations) to understand the contribution of individual features to model predictions.

The explainability module supports:

- Feature importance analysis.
- SHAP summary plots.
- Individual transaction explanations.

These explanations help make model behavior easier to inspect.

## FastAPI Application

The trained model is exposed through a REST API built using FastAPI.

### Start the API

First, ensure that the model artifact has been generated successfully.

Run:

```bash
uvicorn app.main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

### Interactive API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns API information |
| GET | `/health` | Checks API health |
| POST | `/predict` | Predicts fraud probability for a transaction |

The prediction endpoint returns information such as:

- Transaction ID.
- Fraud probability.
- Fraud classification.
- Decision status.
- Model name.
- Classification threshold.

## Data Drift Monitoring

The project implements Population Stability Index (PSI) to monitor changes in incoming transaction feature distributions.

The monitoring module compares reference data with current data and identifies potential distribution shifts.

The system categorizes drift as:

- No Significant Drift
- Moderate Drift
- Significant Drift

Drift detection is a monitoring signal, not proof that model performance has degraded. Actual performance evaluation requires labeled transaction outcomes.

## Automated Testing

The project includes automated tests using Pytest.

Run the test suite:

```bash
pytest -q
```

The test suite covers:

- API endpoint behavior.
- Model training and prediction.
- Model artifact loading.
- Data preprocessing.
- Data drift detection.

## Docker Deployment

Docker is used to containerize the FastAPI application.

**Prerequisite:** Generate the trained model artifact before building the Docker image.

### Build the Docker image

```bash
docker build -t ai-fraud-detection .
```

### Run the container

```bash
docker run -p 8000:8000 ai-fraud-detection
```

Access the API documentation:

```text
http://localhost:8000/docs
```

## Limitations

- The project uses a publicly available historical dataset.
- The dataset may not represent the transaction patterns of every financial institution.
- Business cost values are illustrative and require domain-specific validation.
- Model performance must be established through actual experiments.
- Real-world deployment requires additional security, privacy, compliance, and monitoring measures.

## Future Improvements

Potential future enhancements include:

- Real-time transaction streaming.
- Database integration.
- Advanced model monitoring.
- Automated model retraining.
- Cloud deployment.
- API authentication and authorization.
- Advanced explainability techniques.
- Integration with production-grade data pipelines.

## Disclaimer

This project is developed for educational, research, and portfolio purposes only. It demonstrates an end-to-end machine learning approach to credit card fraud detection using a publicly available dataset.

The model predictions are experimental and should not be considered verified financial advice or used as the sole basis for real-world financial decisions.

This project is not a certified banking fraud prevention system. Any real-world deployment would require rigorous model validation, security testing, data privacy safeguards, regulatory compliance, continuous performance monitoring, and approval from relevant financial domain experts.

The author makes no guarantee of fraud detection accuracy or prevention of financial losses in real-world environments.

## Author

Ritika 

