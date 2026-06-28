"""
Flask app for Invoice Payment Delay Prediction
"""
import os
import sys
import json
import pickle
import warnings
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from asgiref.wsgi import WsgiToAsgi
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Import EncoderExt from ml_pipeline so pickle can locate the class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_pipeline  # noqa — ensures EncoderExt is registered under ml_pipeline
from ml_pipeline import EncoderExt, FEATURE_COLS, load_and_clean  # noqa
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
flask_app = app

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')


def load_artifacts():
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        ml_pipeline.train_pipeline()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'encoders.json')) as f:
        enc_state = json.load(f)
    # Reconstruct encoders as plain LabelEncoder objects from saved classes
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for key, state in enc_state.items():
        le = LabelEncoder()
        le.classes_ = np.array(state['classes_'])
        encoders[key] = le
    with open(os.path.join(MODEL_DIR, 'model_results.json')) as f:
        model_results = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_importance.json')) as f:
        feature_importance = json.load(f)
    with open(os.path.join(MODEL_DIR, 'eda_stats.json')) as f:
        eda_stats = json.load(f)
    return model, encoders, model_results, feature_importance, eda_stats


# Load once at startup
model, encoders, model_results, feature_importance, eda_stats = load_artifacts()


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                           eda_stats=eda_stats,
                           model_results=model_results,
                           feature_importance=feature_importance)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    data = request.get_json(silent=True) or request.form.to_dict()

    try:
        result = run_prediction(data)
        return jsonify({'status': 'ok', **result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict on the full unlabelled dataset."""
    try:
        df = load_and_clean(DATA_PATH)
        null_df = df[df['clear_date'].isnull()].copy()
        null_df_backup = null_df.copy()

        # Encode using saved encoders (plain LabelEncoder)
        def safe_transform(enc, values):
            known = set(enc.classes_)
            mapped = [v if v in known else enc.classes_[0] for v in values]
            return enc.transform(mapped)

        null_df['business_code_enc'] = safe_transform(encoders['business_code'], null_df['business_code'])
        null_df['name_customer_enc'] = safe_transform(encoders['name_customer'], null_df['name_customer'])
        null_df['cust_payment_terms_enc'] = safe_transform(encoders['cust_payment_terms'], null_df['cust_payment_terms'])
        null_df['cust_number'] = null_df['cust_number'].astype(str).str.replace('CCCA', '1').str.replace('CCU', '2').str.replace('CC', '3').astype(int)

        for col, prefix in [('posting_date', 'postingdate'),
                              ('baseline_create_date', 'createdate'),
                              ('due_in_date', 'due')]:
            null_df[f'day_of_{prefix}']   = null_df[col].dt.day
            null_df[f'month_of_{prefix}'] = null_df[col].dt.month
            null_df[f'year_of_{prefix}']  = null_df[col].dt.year

        feats = [c for c in FEATURE_COLS if c in null_df.columns]
        X = null_df[feats]
        preds = model.predict(X)

        # Convert avg_delay (seconds) → days and add to due date
        null_df_backup = null_df_backup.reset_index(drop=True)
        null_df_backup['avg_delay_sec'] = preds
        null_df_backup['delay_days'] = (preds / (24 * 3600)).astype(int)
        null_df_backup['predicted_clear_date'] = (
            null_df_backup['due_in_date'] +
            pd.to_timedelta(null_df_backup['delay_days'], unit='D')
        )

        def bucket(d):
            if d < 0:   return 'Early'
            if d <= 15: return '0-15 days'
            if d <= 30: return '16-30 days'
            if d <= 45: return '31-45 days'
            if d <= 60: return '46-60 days'
            return 'Over 60 days'

        null_df_backup['aging_bucket'] = null_df_backup['delay_days'].apply(bucket)

        preview = null_df_backup[[
            'business_code', 'cust_number', 'name_customer',
            'due_in_date', 'delay_days', 'predicted_clear_date', 'aging_bucket',
            'converted_usd' if 'converted_usd' in null_df_backup.columns else 'total_open_amount'
        ]].head(100)

        # Safe JSON serialization
        def safe(v):
            if pd.isnull(v): return None
            if isinstance(v, (pd.Timestamp, datetime)): return str(v)[:10]
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return round(float(v), 2)
            return v

        records = [{k: safe(v) for k, v in row.items()} for row in preview.to_dict('records')]

        bucket_counts = null_df_backup['aging_bucket'].value_counts().to_dict()

        return jsonify({
            'status': 'ok',
            'total_processed': len(null_df_backup),
            'preview': records,
            'bucket_summary': bucket_counts
        })

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/api/eda_stats')
def api_eda_stats():
    return jsonify(eda_stats)


@app.route('/api/model_results')
def api_model_results():
    return jsonify(model_results)


@app.route('/api/feature_importance')
def api_feature_importance():
    return jsonify(feature_importance)


# ─── Prediction helper ────────────────────────────────────────────────────

def run_prediction(data):
    """Single-invoice prediction from form/JSON input."""
    business_code = data.get('business_code', 'U001')
    cust_number   = str(data.get('cust_number', '200769623'))
    name_customer = data.get('name_customer', 'WAL-MAR corp')
    biz_year      = float(data.get('buisness_year', 2020))
    doc_id        = float(data.get('doc_id', 1930438000))
    payment_terms = data.get('cust_payment_terms', 'NAA8')
    amount        = float(data.get('converted_usd', 10000))
    currency      = data.get('invoice_currency', 'USD')
    posting_date  = pd.to_datetime(data.get('posting_date', '2020-01-26'))
    due_date      = pd.to_datetime(data.get('due_in_date', '2020-02-10'))
    baseline_date = posting_date

    if currency == 'CAD':
        amount = round(amount * 0.7, 2)

    # Encode
    try:
        biz_enc = int(encoders['business_code'].transform([business_code])[0])
    except Exception:
        biz_enc = 0

    try:
        name_enc = int(encoders['name_customer'].transform([name_customer])[0])
    except Exception:
        name_enc = 0

    try:
        terms_enc = int(encoders['cust_payment_terms'].transform([payment_terms])[0])
    except Exception:
        terms_enc = 0

    cust_num_int = int(cust_number.replace('CCCA', '1').replace('CCU', '2').replace('CC', '3'))

    row = {
        'cust_number':             cust_num_int,
        'buisness_year':           biz_year,
        'doc_id':                  doc_id,
        'converted_usd':           amount,
        'business_code_enc':       biz_enc,
        'name_customer_enc':       name_enc,
        'cust_payment_terms_enc':  terms_enc,
        'day_of_postingdate':      posting_date.day,
        'month_of_postingdate':    posting_date.month,
        'year_of_postingdate':     posting_date.year,
        'day_of_createdate':       baseline_date.day,
        'month_of_createdate':     baseline_date.month,
        'year_of_createdate':      baseline_date.year,
        'day_of_due':              due_date.day,
        'month_of_due':            due_date.month,
        'year_of_due':             due_date.year,
    }

    X = pd.DataFrame([row])[[c for c in FEATURE_COLS if c in row]]
    avg_delay_sec = float(model.predict(X)[0])
    delay_days    = int(avg_delay_sec / (24 * 3600))
    predicted_date = due_date + timedelta(days=delay_days)

    if delay_days < 0:
        bucket = 'Early Payment'
    elif delay_days <= 15:
        bucket = '0-15 days'
    elif delay_days <= 30:
        bucket = '16-30 days'
    elif delay_days <= 45:
        bucket = '31-45 days'
    elif delay_days <= 60:
        bucket = '46-60 days'
    else:
        bucket = 'Over 60 days'

    risk = 'Low' if delay_days <= 15 else ('Medium' if delay_days <= 45 else 'High')

    return {
        'delay_days':       delay_days,
        'predicted_date':   str(predicted_date)[:10],
        'due_date':         str(due_date)[:10],
        'aging_bucket':     bucket,
        'risk_level':       risk,
        'avg_delay_sec':    round(avg_delay_sec, 0),
        'amount':           round(amount, 2),
        'customer':         name_customer,
    }


app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    flask_app.run(debug=True, port=5000)
