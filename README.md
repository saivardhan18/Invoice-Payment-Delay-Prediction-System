# Invoice Payment Delay Prediction System

This repository contains a machine learning project that predicts invoice payment delays and estimates when an invoice is likely to be cleared. The project combines data preprocessing, feature engineering, model training, and a Flask web application into one end-to-end solution.

## Project Overview

Accounts receivable teams often need to know whether an invoice will be paid late. This project uses historical invoice data to train a regression model that predicts delay days and converts that result into:

- an expected clearance date
- an aging bucket such as 0–15 days or 16–30 days
- a simple risk level for follow-up actions

## Features

- Synthetic invoice dataset generation for demo purposes
- Data cleaning and preprocessing
- Currency normalization and date-based feature engineering
- Comparison of multiple regression models
- Flask-based web interface for single and batch predictions
- Dashboard for basic analytics and model insights

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Chart.js

## Project Structure

```text
invoice-predictor/
├── app.py
├── generate_data.py
├── ml_pipeline.py
├── requirements.txt
├── data/
├── models/
└── templates/
```

## How to Run

```bash
cd invoice-predictor
pip install -r requirements.txt
python generate_data.py
python ml_pipeline.py
python app.py
```

Then open:

```text
http://localhost:5000
```

## Main Use Cases

- Predict invoice payment delay for a single invoice
- Run batch predictions for open invoices
- Explore dashboard analytics such as average delay and late-payment distribution

## Goal

The goal of this project is to demonstrate a practical machine learning workflow for business forecasting in a simple and understandable way.
