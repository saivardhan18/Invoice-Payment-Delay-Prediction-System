# InvoiceIQ — Invoice Payment Delay Prediction

> ML-powered accounts receivable tool built during my **HighRadius** internship.
> Predicts invoice payment delays, assigns aging buckets, and surfaces risk levels.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![scikit-learn](https://img.shields.io/badge/sklearn-1.3-red)

---

## 🚀 Live Demo

| Page | Description |
|------|-------------|
| `/` | Landing page with pipeline overview |
| `/dashboard` | EDA analytics, model comparison, feature importance |
| `/predict` | Single-invoice prediction form |
| `/batch_predict` | Batch prediction over all open invoices |

---

## 🎯 Problem Statement

Accounts receivable teams need to know *when* customers will actually pay — not just
the stated due date. Late payments damage cash flow and inflate DSO. This project
uses historical invoice data to predict the **payment clear date** and classify
invoices into **aging buckets** (0-15 / 16-30 / 31-45 / 46-60 / 60+ days).

---

## 🔬 ML Pipeline

```
Raw Data (50k invoices)
    ↓ Cleaning & deduplication
    ↓ Currency normalisation (CAD → USD)
    ↓ Date feature extraction (day/month/year)
    ↓ Customer avg-delay mapping
    ↓ Label encoding (business_code, name, payment_terms)
    ↓ Variance threshold feature selection
    ↓ 60/20/20 train/val/test split
    ↓ Model training & comparison
    ↓ XGBoost selected (best R²)
    ↓ Predict avg_delay (seconds) → days
    ↓ Add to due_date → predicted clear date
    ↓ Assign aging bucket
```

### Models Compared

| Model | Test R² | Val RMSE |
|-------|---------|---------|
| Linear Regression | ~0.32 | ~521k |
| Decision Tree | 1.00 | ~484k |
| **Random Forest** | **1.00** | **~339k** |
| **XGBoost** | **1.00** | **~351k** |
| SVR | ~-0.005 | ~632k |

> XGBoost selected as final model for its generalisation and interpretability (feature importance).

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/your-username/invoiceiq.git
cd invoiceiq

# Install
pip install -r requirements.txt

# Generate dataset (first time)
python generate_data.py

# Train models
python ml_pipeline.py

# Run app
python app.py
# → http://localhost:5000
```

---

## 📁 Project Structure

```
invoiceiq/
├── app.py               # Flask app + prediction routes
├── ml_pipeline.py       # Data processing + model training
├── generate_data.py     # Synthetic dataset generator
├── requirements.txt
├── data/
│   └── dataset.csv      # 50k invoice records
├── models/
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── model_results.json
│   ├── feature_importance.json
│   └── eda_stats.json
└── templates/
    ├── base.html
    ├── index.html
    ├── dashboard.html
    └── predict.html
```

---

## 🛠 Skills Demonstrated

- **Pandas** — data cleaning, deduplication, date parsing, groupby engineering
- **Feature Engineering** — date decomposition, CAD→USD conversion, customer delay mapping
- **EDA** — distribution analysis, correlation heatmaps, delay-by-customer/year breakdowns
- **Scikit-learn** — LabelEncoder, VarianceThreshold, train_test_split, cross-validation
- **XGBoost** — gradient boosted trees, feature importance extraction
- **Random Forest** — ensemble bagging, OOB validation
- **Flask** — REST API, Jinja2 templating, JSON serialisation
- **Chart.js** — interactive visualisations (donut, bar, feature importance)

---

## 🏢 Context

Built as part of a **HighRadius** internship project focused on Accounts Receivable
automation and Cash Flow Prediction. The goal was to reduce manual effort in
collections prioritisation by predicting which invoices would be paid late.

---

## 📄 API Reference

### `POST /predict`

```json
{
  "business_code": "U001",
  "cust_number": "200769623",
  "name_customer": "WAL-MAR corp",
  "invoice_currency": "USD",
  "converted_usd": 54273,
  "cust_payment_terms": "NAH4",
  "posting_date": "2020-01-26",
  "due_in_date": "2020-02-10",
  "buisness_year": "2020",
  "doc_id": 1930438000
}
```

**Response:**
```json
{
  "status": "ok",
  "delay_days": 10,
  "predicted_date": "2020-02-20",
  "due_date": "2020-02-10",
  "aging_bucket": "0-15 days",
  "risk_level": "Low",
  "customer": "WAL-MAR corp",
  "amount": 54273.0
}
```

### `POST /batch_predict`

No body required. Runs prediction on all open invoices in the dataset.

---

*Made with Python, Flask, and XGBoost · HighRadius internship project*
