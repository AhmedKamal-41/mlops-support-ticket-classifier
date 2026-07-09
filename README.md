<<<<<<< HEAD
# IT Service Desk Ticket Classification & Routing System

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogisticRegression-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen)

An **ML-powered IT service desk tool**. It uses **NLP (TF-IDF + Logistic
Regression)** to classify free-text IT support tickets into service desk
categories, then applies **service desk logic** on top of the model's prediction
to decide **priority, routing team, a suggested knowledge-base article, first
troubleshooting steps, and an escalation recommendation** — simulating how a real
Help Desk / Service Desk triages and routes tickets.

> Built an ML-powered IT service desk classifier that uses NLP to categorize
> support tickets, then applies routing, priority, and knowledge-base logic to
> simulate real service desk triage.

The **machine learning model is the core** of the project. The **IT workflow
layer** wraps the prediction so it's useful for Help Desk, Service Desk Analyst,
Technical Support, and IT Operations roles.

![IT Service Desk Ticket Triage demo](assets/images/demo_overview.png)

## What this project combines

- **Machine learning / NLP** — text classification with TF-IDF + Logistic Regression
- **Ticket classification** — 15 real IT service desk categories
- **IT service desk workflow** — turns a prediction into an actionable triage
- **Priority & routing logic** — priority level and the team that should own the ticket
- **Knowledge base recommendations** — a suggested KB article per category
- **Support operations thinking** — first troubleshooting steps and escalation rules

## How it works

```
   Free-text ticket
"My Outlook is not syncing and I cannot receive emails."
          │
          ▼
┌───────────────────────────┐
│   ML / NLP  (the core)     │   src/preprocessing.py  → clean text
│   TF-IDF  +  LogisticReg   │   src/model.py / train.py → predict category
└───────────────────────────┘
          │  predicted category + confidence
          ▼
┌───────────────────────────┐
│  Service desk workflow     │   src/service_desk.py
│  layer (rules)             │   priority · routing · KB · steps · escalation
└───────────────────────────┘
          │
          ▼
   Actionable triage (see example below)
```

- **The ML model predicts the category.** (This is the learned, data-driven part.)
- **The service desk layer applies rules** to derive priority, routing team, KB
  article, first steps, and escalation — plus it uses the model's **confidence**
  to flag low-confidence tickets for manual review.

## Worked example

**Input:** `My Outlook is not syncing and I cannot receive emails.`

**Output:**

| Field | Value |
|-------|-------|
| **ML Predicted Category** | Microsoft 365 / Outlook |
| **Confidence** | 0.81 |
| **Priority** | Medium |
| **Routing Team** | Microsoft 365 Support |
| **Suggested KB** | `kb-outlook-sync.md` |
| **First Steps** | Check internet · restart Outlook · test Outlook on the Web · confirm mailbox access |
| **Escalation** | Escalate to Messaging/Exchange if mailbox access fails in OWA or multiple users are affected |

## Demo

The Streamlit app (`it_service_desk_demo.py`) shows the full triage for any
ticket: the ML prediction and confidence, the priority, the routing team, the
suggested KB article, first troubleshooting steps, and the escalation guidance.

**Medium-priority example — routed to Microsoft 365 Support:**

![Outlook ticket triaged as Microsoft 365 / Outlook, Medium priority](assets/images/demo_triage_outlook.png)

**Critical-priority example — a phishing report routed to the Security / SOC team:**

![Phishing ticket triaged as Security / Phishing, Critical priority](assets/images/demo_triage_phishing.png)

## IT categories

The ML model classifies each ticket into one of these 15 categories, and each maps
to a service desk rule (`src/service_desk.py`):

| Category | Priority | Routing team | Suggested KB |
|----------|----------|--------------|--------------|
| Password Reset | Low | Service Desk (Tier 1) | `kb-password-reset.md` |
| Account Lockout | Medium | Service Desk (Tier 1) | `kb-account-lockout.md` |
| Microsoft 365 / Outlook | Medium | Microsoft 365 Support | `kb-outlook-sync.md` |
| Teams / OneDrive | Medium | Microsoft 365 Support | `kb-teams-onedrive.md` |
| Network / Wi-Fi | High | Network Operations | `kb-network-wifi.md` |
| VPN | High | Network Operations | `kb-vpn-connectivity.md` |
| Printer | Low | Service Desk (Tier 1) | `kb-printing.md` |
| Hardware | Medium | Desktop Support | `kb-hardware-troubleshooting.md` |
| Software Installation | Low | Desktop Support | `kb-software-install.md` |
| Access Request | Medium | Identity & Access Management | `kb-access-request.md` |
| Shared Folder | Medium | Identity & Access Management | `kb-shared-folder-access.md` |
| Security / Phishing | Critical | Security / SOC | `kb-phishing-response.md` |
| New Hire Setup | Medium | IT Onboarding | `kb-new-hire-setup.md` |
| Offboarding | High | IT Onboarding | `kb-offboarding.md` |
| Escalation Required | Critical | Major Incident / On-call | `kb-major-incident.md` |

## Quick start (the demo)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the CLI demo (trains the model automatically on first run)
python demo.py
```

More ways to run the demo:

```bash
python demo.py "My VPN won't connect from home"   # triage a single ticket
python demo.py --interactive                       # type tickets live
python demo.py --retrain                           # retrain the model first
```

Optional Streamlit demo (paste a ticket, see the full triage):

```bash
streamlit run it_service_desk_demo.py
```

## The machine learning pipeline

The ML core lives in a few small, readable modules:

| Step | Where | What it does |
|------|-------|--------------|
| **Dataset** | `data/raw/it_support_tickets.csv` | 210 labelled IT tickets across the 15 categories (`text,label`) |
| **Preprocessing** | `src/preprocessing.py` | Lowercase, strip, collapse whitespace |
| **Vectorization** | `src/preprocessing.py` | **TF-IDF** with unigrams + bigrams, English stop words |
| **Model** | `src/model.py`, `src/train.py` | **LogisticRegression** (multi-class) |
| **Train/test split** | `src/data_loader.py` | Stratified 70% / 15% / 15% train / val / test |
| **Evaluation** | `src/evaluation.py` | Accuracy, macro / weighted **F1**, per-class metrics, confusion matrix |
| **Prediction** | `src/model.py` | `classify()` and `classify_and_triage()` |

**Prediction function** (the one the demos call):

```python
from src.model import classify_and_triage

result = classify_and_triage("I can't connect to the VPN from home")
# {
#   'predicted_category': 'VPN',
#   'confidence': 0.78,
#   'priority': 'High',
#   'routing_team': 'Network Operations',
#   'suggested_kb': 'kb-vpn-connectivity.md',
#   'first_steps': [...],
#   'escalation': '...',
#   'needs_review': False,
# }
```

### Model performance

On the held-out **test split** (15-class problem, ~210 total samples):

- **Accuracy:** ~0.72
- **Macro-F1:** ~0.66

These are honest numbers for a small, deliberately simple and explainable
dataset. Most misclassifications are between naturally overlapping categories
(e.g. *Password Reset* vs *Account Lockout*, or *Escalation Required* vs the
specific outage type). The **confidence score** is used by the service desk layer:
tickets below a confidence threshold are flagged `needs_review` so a human analyst
confirms the category before auto-routing.

Retrain any time:

```bash
python -m src.model     # trains and saves models/classifier.pkl + vectorizer.pkl
```

## Tests

Simple, fast tests cover both the ML classifier and the service desk rules:

```bash
pytest tests/ -v
```

- `tests/test_model.py` — clear tickets classify into the right category; end-to-end triage works
- `tests/test_service_desk.py` — every category has a complete rule; routing, priority, and low-confidence review flag behave correctly

## Project structure

```
mlops-support-ticket-classifier/
├── data/raw/
│   └── it_support_tickets.csv      # IT ticket dataset (text, label)
├── src/
│   ├── config.py                   # Categories (LABELS) + paths
│   ├── data_loader.py              # Load + stratified train/val/test split
│   ├── preprocessing.py            # Text cleaning + TF-IDF vectorizer
│   ├── model.py                    # Self-contained train / load / classify + triage
│   ├── service_desk.py             # ★ Service desk workflow layer (rules)
│   ├── train.py                    # Full MLflow training pipeline
│   ├── evaluation.py               # Metrics + confusion matrix
│   ├── inference.py                # MLflow model loading + prediction
│   ├── drift_detection.py          # Simple data drift detection
│   └── api/                        # FastAPI serving layer
├── demo.py                         # ★ CLI demo (ML prediction + triage)
├── it_service_desk_demo.py         # ★ Streamlit demo
├── tests/                          # ★ Model + service desk tests
├── dashboard/                      # Streamlit ops/monitoring dashboard
├── monitoring/                     # Prometheus + Grafana
├── docker/ , docker-compose.yml    # Full-stack containerization
└── requirements.txt
```

(★ = added/changed to turn the classifier into an IT service desk tool.)

## The IT service desk workflow layer

`src/service_desk.py` is the layer that makes the ML prediction operationally
useful. It's intentionally **rule-based and easy to read** — the kind of triage
logic a real service desk encodes in its ITSM tool, kept here in one transparent
dictionary:

```python
"Microsoft 365 / Outlook": {
    "priority": "Medium",
    "routing_team": "Microsoft 365 Support",
    "kb_article": "kb-outlook-sync.md",
    "first_steps": [
        "Check the user's internet connection",
        "Restart Outlook and test send/receive",
        "Confirm mailbox access via Outlook on the Web (OWA)",
        "Check the Microsoft 365 service health dashboard",
    ],
    "escalation": "Escalate to Messaging/Exchange if mailbox access fails in OWA "
                  "or multiple users are affected.",
},
```

`triage_ticket(text, predicted_category, confidence)` merges the ML prediction
with the matching rule and returns the full, actionable triage — including the
low-confidence review flag.

---

## MLOps / production stack (optional depth)

Beyond the core classifier + triage, the project also includes a full MLOps stack
so it can be run as a production-style service. This part is **optional** — the
demo above runs without it.

- **Experiment tracking & model registry:** MLflow (`src/train.py` logs params,
  metrics, and artifacts to `mlruns/`)
- **Serving:** FastAPI (`src/api/main.py`) exposes a `/predict` endpoint
- **Monitoring:** Prometheus + Grafana (request rate, latency, error rate, drift)
- **Data drift detection:** `src/drift_detection.py` compares ticket-length
  distributions between training data and live traffic
- **Containerization:** Docker + Docker Compose (`docker-compose.yml`)
- **CI/CD:** GitHub Actions (`.github/workflows/ci-cd.yml`)
- **Ops dashboard:** `dashboard/app.py` (KPIs, model quality, monitoring, drift)

### Tech stack

- **Language:** Python 3.11+
- **ML / NLP:** scikit-learn (TF-IDF + LogisticRegression)
- **Experiment tracking:** MLflow
- **API:** FastAPI + Uvicorn
- **Monitoring:** Prometheus + Grafana
- **Containers / CI:** Docker Compose + GitHub Actions

### Run the full stack

```bash
# Train and register a model with MLflow
python -m src.train

# Serve predictions
uvicorn src.api.main:app --reload      # http://localhost:8000/docs

# Or bring up the whole stack
docker-compose up --build
```

The `/predict` endpoint classifies a batch of tickets and returns the predicted
category with a confidence score:
=======
# MLOps Support Ticket Classifier

A complete, end-to-end MLOps project for classifying customer support tickets into categories using machine learning. This project demonstrates the full ML lifecycle: data loading, training, model tracking, deployment, monitoring, drift detection, and retraining.

## 📋 Project Overview

This project builds an **MLOps pipeline** around a **customer support ticket classifier**. The system:

- **Input**: Free-text customer support tickets (e.g., "I was double charged", "App keeps crashing")
- **Output**: One of five categories:
  - `billing` - Billing and payment issues
  - `technical` - Technical problems and bugs
  - `account` - Account access and login issues
  - `shipping` - Shipping and delivery questions
  - `general` - General inquiries

The project focuses on **MLOps best practices**: experiment tracking, model versioning, API deployment, metrics monitoring, data drift detection, and automated retraining.

## 📊 System Overview

<div align="center">

![End-to-End MLOps Pipeline](assets/images/mlops-pipeline-infographic.png)

*Complete MLOps pipeline visualization showing the full lifecycle from data ingestion to production deployment with monitoring and automated retraining.*

</div>

## 🏗️ Architecture

```
┌─────────────┐
│   Dataset   │ (CSV files in data/raw/)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Training   │ (src/train.py)
│  Pipeline   │ → Trains LogisticRegression classifier
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   MLflow    │ (Experiment tracking & Model Registry)
│  Tracking   │ → Logs metrics, params, artifacts
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  FastAPI    │ (src/api/main.py)
│  Inference  │ → Serves predictions via REST API
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Prometheus  │ ←── │   Grafana   │
│  Metrics    │     │  Dashboard  │
└─────────────┘     └─────────────┘
```

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **ML Framework**: scikit-learn (LogisticRegression)
- **Experiment Tracking**: MLflow (local file-based)
- **API Framework**: FastAPI + Uvicorn
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Metrics**: prometheus-fastapi-instrumentator

## 📁 Project Structure

```
mlops-support-ticket-classifier/
├── data/
│   ├── raw/                    # Original dataset (CSV files)
│   └── processed/               # Processed data (optional)
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration constants
│   ├── data_loader.py          # Data loading and splitting
│   ├── preprocessing.py        # Text preprocessing and vectorization
│   ├── train.py                # Training script with MLflow
│   ├── evaluation.py           # Metrics computation
│   ├── inference.py            # Model loading and prediction
│   ├── drift_detection.py     # Data drift detection
│   ├── retrain.py              # Retraining script
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic request/response models
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── scripts/
│   └── generate_sample_logs.py # Script to generate sample inference logs
├── logs/                       # Inference logs (JSONL format)
├── reports/                     # Evaluation artifacts (reports, confusion matrices)
├── mlruns/                      # MLflow artifacts (created automatically)
├── monitoring/
│   ├── prometheus.yml           # Prometheus configuration
│   └── grafana/
│       ├── provisioning/       # Grafana auto-configuration
│       └── dashboards/          # Pre-configured dashboards
├── docker/
│   ├── Dockerfile.api          # FastAPI service container
│   └── Dockerfile.mlflow       # MLflow server container
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions pipeline
├── docker-compose.yml          # Full stack orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Docker and Docker Compose (for full stack)
- Git

### Option 1: Run Locally (No Docker)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-support-ticket-classifier
   ```

2. **Create a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Validate the dataset** (optional, but recommended)
   ```bash
   python -m src.validate_data --data data/raw/support_tickets_sample.csv
   ```
   This validates the dataset schema, labels, and class distribution before training.

5. **Train the model**
   ```bash
   python -m src.train
   ```
   This will:
   - Validate the dataset (schema, labels, class distribution)
   - Load the dataset from `data/raw/support_tickets_sample.csv`
   - Train a LogisticRegression classifier
   - Evaluate on validation and test sets
   - Save evaluation artifacts (confusion matrices, classification reports)
   - Log everything to MLflow (in `mlruns/` directory)
   - Register the model in MLflow Model Registry

6. **Start the FastAPI server**
   ```bash
   uvicorn src.api.main:app --reload
   ```
   The API will be available at `http://localhost:8000`

7. **View MLflow UI** (optional)
   ```bash
   mlflow ui --backend-store-uri file:///./mlruns
   ```
   Open `http://localhost:5000` in your browser
   
   In MLflow UI, you can:
   - View all training runs and metrics
   - Compare model versions
   - View evaluation artifacts (confusion matrices, classification reports)
   - See dataset fingerprints for reproducibility
   - Promote models to Production stage

### Option 2: Run with Docker Compose (Full Stack)

1. **Clone and navigate to the project**
   ```bash
   git clone <repository-url>
   cd mlops-support-ticket-classifier
   ```

2. **Train the model first** (before starting Docker)
   ```bash
   # Create venv and install dependencies
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   
   # Train the model
   python -m src.train
   ```

3. **Start all services with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the services**
   - **FastAPI API**: http://localhost:8000
     - API docs: http://localhost:8000/docs
     - Health check: http://localhost:8000/health
   - **MLflow UI**: http://localhost:5000
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000
     - Default login: `admin` / `admin`

## 📖 Usage Guide

### Making Predictions

Use the `/predict` endpoint to classify support tickets:
>>>>>>> d6dcda15bc146462239168e16e4b6c3da45b868a

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
<<<<<<< HEAD
     -d '{"tickets": ["My Outlook is not syncing and I cannot receive emails"]}'
```

### Ops dashboard sample data

```bash
python scripts/generate_sample_logs.py --num-logs 1000   # sample inference logs
python scripts/generate_sample_reports.py                # sample metrics/report
streamlit run dashboard/app.py                           # http://localhost:8501
```

---

## Résumé bullets

- Built an **ML-powered IT ticket classifier** using NLP (TF-IDF + Logistic
  Regression) to categorize help desk issues by support type.
- Added **service desk routing logic** for priority, escalation path, support
  team, and knowledge-base recommendations on top of the model's prediction.
- Classified tickets across **Outlook, VPN, printer, account access, network,
  hardware, and security** issue categories (15 IT categories total).
- Created a **CLI/demo workflow** showing the model prediction, confidence score,
  routing team, and suggested troubleshooting steps end-to-end.
- Wrapped the classifier in an optional **MLOps stack** (MLflow tracking, FastAPI
  serving, Prometheus/Grafana monitoring, Docker, CI/CD).
=======
     -d '{
       "tickets": [
         "I was double charged for my subscription",
         "The app keeps crashing when I open it",
         "I can't log into my account"
       ]
     }'
```

Response:
```json
{
  "results": [
    {
      "text": "I was double charged for my subscription",
      "predicted_label": "billing",
      "confidence": 0.95
    },
    {
      "text": "The app keeps crashing when I open it",
      "predicted_label": "technical",
      "confidence": 0.92
    },
    {
      "text": "I can't log into my account",
      "predicted_label": "account",
      "confidence": 0.88
    }
  ]
}
```

### Data Validation

Validate your dataset before training to catch issues early:

```bash
# Validate dataset (uses default path from config)
python -m src.validate_data

# Validate specific dataset file
python -m src.validate_data --data path/to/data.csv

# Validate and check train/val/test splits are non-overlapping
python -m src.validate_data --check-splits

# Save validation report to JSON
python -m src.validate_data --output reports/validation_report.json
```

The validation checks:
- Required columns exist (`text`, `label`)
- No empty text rows
- Label values are in allowed set (5 classes)
- Class distribution (warns if extreme imbalance)
- Train/val/test splits are non-overlapping (if `--check-splits` is used)
- Computes dataset fingerprint for reproducibility

### Training a Model

Train a new model and register it in MLflow:

```bash
python -m src.train
```

The script will:
- **Validate the dataset** (schema, labels, class distribution)
- Load and split the dataset (70% train, 15% val, 15% test)
- Compute and log dataset fingerprint
- Preprocess text using TF-IDF vectorization
- Train a LogisticRegression classifier
- Evaluate on validation and test sets
- **Save evaluation artifacts** to `reports/`:
  - `val_classification_report.json` and `val_confusion_matrix.png`
  - `test_classification_report.json` and `test_confusion_matrix.png`
- Log all metrics, parameters, and artifacts to MLflow
- Register the model in MLflow Model Registry

### Model Evaluation

Evaluate a trained model on test data and generate evaluation artifacts:

```bash
# Evaluate using MLflow Production model on test split
python -m src.evaluate --split

# Evaluate using MLflow Production model on full dataset
python -m src.evaluate

# Evaluate using specific MLflow run
python -m src.evaluate --mlflow-run-id <run_id> --split

# Evaluate using local model files
python -m src.evaluate --model-dir models/ --data data/raw/support_tickets_sample.csv

# Specify custom output directory
python -m src.evaluate --output custom_reports/
```

The evaluation script:
- Loads model and vectorizer (from MLflow or local files)
- Makes predictions on test data
- Computes metrics (accuracy, macro-F1, per-class F1, etc.)
- Saves artifacts to `reports/` (or custom directory):
  - `test_classification_report.json` - All metrics in JSON format
  - `test_confusion_matrix.png` - Confusion matrix visualization

### Retraining

Retrain the model with updated data:

```bash
python -m src.retrain
```

This script:
- Trains a new model using the same pipeline
- Compares performance with the current Production model
- Automatically promotes the new model to Production if it performs better

### Promoting Models in MLflow

To manually promote a model to Production:

1. **Via MLflow UI**:
   - Go to http://localhost:5000
   - Navigate to Models → support_ticket_classifier
   - Select a model version
   - Click "Stage" → "Production"

2. **Via MLflow CLI**:
   ```bash
   mlflow models transition-model-version-stage \
     --name support_ticket_classifier \
     --version <VERSION_NUMBER> \
     --stage Production
   ```

### Streamlit Dashboard

The project includes a premium, recruiter-ready Streamlit dashboard for visualizing model metrics, monitoring inference operations, and tracking system health.

**Running the Dashboard:**
```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

**Dashboard Tabs:**

1. **Overview**
   - KPI cards: Requests (24h), p95 Latency, Error Rate, Macro-F1
   - Interactive charts: Requests over time, Latency percentiles (p50/p95)
   - Prediction distribution and recent requests table

2. **Model Quality**
   - Per-class performance metrics table (precision, recall, F1)
   - Confusion matrix visualization
   - Top misclassified examples (if available in logs)

3. **Monitoring**
   - Sidebar filters: Time range, class filter, status code filter
   - Throughput chart: Requests per minute over time
   - Latency histogram: Distribution of request latencies
   - Errors over time: 500 error tracking
   - Searchable inference logs table

4. **Drift**
   - Drift score and status indicators
   - Reference vs recent statistics comparison
   - Visual drift comparison charts

5. **Runbook**
   - Operational guides: Training, evaluation, serving, MLflow
   - Command snippets and best practices
   - How to generate sample data

**Generating Sample Data:**

To populate the dashboard with sample data:

```bash
# Generate sample inference logs (1000 entries)
python scripts/generate_sample_logs.py --num-logs 1000

# Generate sample classification reports and confusion matrix
python scripts/generate_sample_reports.py

# Generate reports including drift report
python scripts/generate_sample_reports.py --include-drift

# Custom output directory
python scripts/generate_sample_reports.py --output-dir custom_reports/
```

**Sample Data Features:**
- **Inference Logs**: Realistic timestamps, latency values (50-500ms), status codes, prediction labels
- **Classification Reports**: Complete metrics with per-class breakdown
- **Confusion Matrix**: Visual 5x5 matrix for all labels
- **Drift Reports**: Sample drift statistics and comparisons

**Dashboard Configuration:**

The dashboard includes a sidebar configuration panel where you can customize file paths:
- Classification report: `reports/classification_report.json`
- Confusion matrix: `reports/confusion_matrix.png`
- Inference logs: `logs/inference_logs.jsonl`
- Drift report: `reports/drift_report.json` (optional)

**Screenshots Checklist (for Recruiters):**

Capture these views for your portfolio:
- ✅ Overview tab with KPI cards and charts
- ✅ Model Quality tab showing confusion matrix
- ✅ Monitoring tab with filtered analytics
- ✅ Drift tab (if drift report available)
- ✅ Runbook tab with operational guides

**Features:**
- Professional UI with custom CSS styling
- Interactive Plotly charts
- Real-time status badges (Healthy/Degraded/Missing Artifacts)
- Graceful handling of missing data with empty states
- Generate sample data buttons for quick demos

## 📊 Monitoring

### Prometheus Metrics

The FastAPI service exposes metrics at `/metrics`:

- `http_requests_total` - Total request count
- `http_request_duration_seconds` - Request latency histogram
- `support_ticket_drift_score` - Data drift score (custom metric)

### Grafana Dashboard

The pre-configured dashboard shows:
- **Request Rate**: Requests per second
- **Latency**: p50, p95, p99 percentiles
- **Error Rate**: 4xx and 5xx errors
- **Drift Score**: Data drift detection metric

Access Grafana at http://localhost:3000 (admin/admin)

## 🔍 Data Drift Detection

The system includes simple drift detection based on ticket length distribution:

- **How it works**: Compares the distribution of ticket lengths (number of words) between training data and recent requests
- **Drift Score**: A normalized difference score (0.0 = no drift, higher = more drift)
- **Exposed as**: Prometheus metric `support_ticket_drift_score`
- **View**: Check `/drift/stats` endpoint or Grafana dashboard

The drift detector maintains a rolling window of recent requests and updates statistics with each prediction batch.

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push/PR:

1. **Tests**: Verifies all modules can be imported
2. **Linting**: (Optional) Code quality checks
3. **Build**: Builds Docker images for API and MLflow services
4. **Push**: (Optional) Pushes images to Docker Hub if configured

To enable Docker Hub push:
1. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as GitHub secrets
2. Uncomment the push steps in `.github/workflows/ci-cd.yml`

## 📝 Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key configuration options:
- `MLFLOW_TRACKING_URI`: MLflow server URI
- `MLFLOW_EXPERIMENT_NAME`: Experiment name in MLflow
- `MODEL_NAME`: Model name in MLflow Model Registry
- `API_PORT`: FastAPI server port

## 🧪 Development

### Running Tests

Run the test suite using pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_validate_data.py

# Run with verbose output
pytest tests/ -v

# Run with coverage (requires pytest-cov)
pytest tests/ --cov=src --cov-report=html
```

The test suite includes:
- **Data validation tests**: Schema validation, label checks, class distribution
- **Evaluation tests**: Metrics computation, artifact saving

### Artifacts

Training and evaluation generate artifacts saved to the `reports/` directory:

- **Validation reports**: `validation_report.json` (if saved via CLI)
- **Evaluation reports**: 
  - `val_classification_report.json` - Validation set metrics
  - `test_classification_report.json` - Test set metrics
  - `val_confusion_matrix.png` - Validation confusion matrix
  - `test_confusion_matrix.png` - Test confusion matrix

These artifacts are also logged to MLflow and can be viewed in the MLflow UI.

### Code Style

The code is designed to be:
- **Beginner-friendly**: Heavily commented with clear explanations
- **Modular**: Each module has a single responsibility
- **Type-hinted**: Where appropriate for clarity
- **Well-documented**: Docstrings for all functions

## 🎓 Learning Resources

This project demonstrates:
- **MLOps fundamentals**: Experiment tracking, model versioning, deployment
- **API development**: FastAPI, REST endpoints, request/response validation
- **Monitoring**: Prometheus metrics, Grafana dashboards
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions workflows

>>>>>>> d6dcda15bc146462239168e16e4b6c3da45b868a
