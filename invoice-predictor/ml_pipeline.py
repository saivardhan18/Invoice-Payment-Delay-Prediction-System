"""
Invoice Payment Delay Prediction — ML Pipeline
Trains models and saves artifacts for the Flask app.
"""
import os
import json
import warnings
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - runtime dependency may be absent
    xgb = None

warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Data Loading & Cleaning ───────────────────────────────────────────────

def load_and_clean(path):
    df = pd.read_csv(path)
    
    # Drop useless columns
    drop_cols = ['area_business', 'posting_id', 'invoice_id',
                 'document_create_date', 'isOpen', 'document type',
                 'document_create_date.1']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert dates
    df['clear_date'] = pd.to_datetime(df['clear_date'])
    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df['due_in_date'] = pd.to_datetime(df['due_in_date'], format='%Y%m%d')
    df['baseline_create_date'] = pd.to_datetime(df['baseline_create_date'], format='%Y%m%d')
    
    # Currency conversion CAD → USD
    df['converted_usd'] = np.where(
        df['invoice_currency'] == 'CAD',
        df['total_open_amount'] * 0.7,
        df['total_open_amount']
    )
    df.drop(['invoice_currency', 'total_open_amount'], axis=1, inplace=True)
    
    return df


# ─── Feature Engineering ──────────────────────────────────────────────────

def engineer_features(maindata, fit_encoders=True, encoders=None):
    df = maindata.copy()
    
    # Compute delay
    df['Delay'] = df['clear_date'] - df['due_in_date']
    
    # Customer avg delay in seconds
    avg_delay_map = df.groupby('name_customer')['Delay'].mean()
    df['avg_delay'] = df['name_customer'].map(avg_delay_map)
    df['avg_delay'] = df['avg_delay'].apply(lambda x: x.total_seconds() if pd.notnull(x) else 0)
    
    # Encode categorical columns
    if fit_encoders:
        encoders = {}
        
        enc_biz = LabelEncoder()
        df['business_code_enc'] = enc_biz.fit_transform(df['business_code'])
        encoders['business_code'] = enc_biz
        
        enc_name = _make_encoder_ext()
        enc_name.fit(df['name_customer'])
        df['name_customer_enc'] = enc_name.transform(df['name_customer'])
        encoders['name_customer'] = enc_name
        
        enc_terms = _make_encoder_ext()
        enc_terms.fit(df['cust_payment_terms'])
        df['cust_payment_terms_enc'] = enc_terms.transform(df['cust_payment_terms'])
        encoders['cust_payment_terms'] = enc_terms
    else:
        df['business_code_enc'] = encoders['business_code'].transform(df['business_code'])
        df['name_customer_enc'] = encoders['name_customer'].transform(df['name_customer'])
        df['cust_payment_terms_enc'] = encoders['cust_payment_terms'].transform(df['cust_payment_terms'])
    
    # Customer number cleanup
    df['cust_number'] = df['cust_number'].astype(str).str.replace('CCCA', '1').str.replace('CCU', '2').str.replace('CC', '3').astype(int)
    
    # Date feature extraction
    for col, prefix in [('posting_date', 'postingdate'),
                         ('baseline_create_date', 'createdate'),
                         ('due_in_date', 'due')]:
        df[f'day_of_{prefix}']   = df[col].dt.day
        df[f'month_of_{prefix}'] = df[col].dt.month
        df[f'year_of_{prefix}']  = df[col].dt.year
    
    # Drop raw columns
    drop = ['business_code', 'name_customer', 'cust_payment_terms',
            'posting_date', 'baseline_create_date', 'due_in_date',
            'clear_date', 'Delay']
    df.drop([c for c in drop if c in df.columns], axis=1, inplace=True)
    
    return df, encoders


# ─── Encoder helper ───────────────────────────────────────────────────────

class EncoderExt:
    """LabelEncoder that handles unseen categories as 'Unknown'."""
    def __init__(self):
        self.le = LabelEncoder()
        self.classes_ = None

    def fit(self, data):
        self.le.fit(list(data) + ['Unknown'])
        self.classes_ = self.le.classes_
        return self

    def transform(self, data):
        d = list(data)
        for item in np.unique(data):
            if item not in self.le.classes_:
                d = ['Unknown' if x == item else x for x in d]
        return self.le.transform(d)


def _make_encoder_ext():
    return EncoderExt()


# ─── Training ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'cust_number', 'buisness_year', 'doc_id', 'converted_usd',
    'business_code_enc', 'name_customer_enc', 'cust_payment_terms_enc',
    'day_of_postingdate', 'month_of_postingdate', 'year_of_postingdate',
    'day_of_createdate', 'month_of_createdate', 'year_of_createdate',
    'day_of_due', 'month_of_due', 'year_of_due'
]


def train_pipeline(data_path=DATA_PATH):
    print("Loading data...")
    df = load_and_clean(data_path)
    
    # Split into labelled / unlabelled
    maindata = df[df['clear_date'].notnull()].copy()
    nulldata  = df[df['clear_date'].isnull()].copy()
    
    print(f"  Labelled rows: {len(maindata)}, Unlabelled: {len(nulldata)}")
    
    # Feature engineering on labelled set
    print("Engineering features...")
    feat_df, encoders = engineer_features(maindata, fit_encoders=True)
    
    X = feat_df[[c for c in FEATURE_COLS if c in feat_df.columns]]
    y = feat_df['avg_delay']
    
    X_train, X_loc_test, y_train, y_loc_test = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_loc_test, y_loc_test, test_size=0.5, random_state=42)
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ── Train models ──
    models = {
        'Linear Regression':    LinearRegression(),
        'Decision Tree':        DecisionTreeRegressor(random_state=42),
        'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if xgb is not None:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    
    results = {}
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  → {name}")
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_val  = model.predict(X_val)
        results[name] = {
            'test_r2':   round(float(r2_score(y_test, pred_test)), 4),
            'val_r2':    round(float(r2_score(y_val, pred_val)), 4),
            'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 2),
            'val_rmse':  round(float(np.sqrt(mean_squared_error(y_val, pred_val))), 2),
        }
        print(f"     Test R²={results[name]['test_r2']:.4f}  RMSE={results[name]['test_rmse']:.0f}")
    
    best_model_name = max(results, key=lambda name: results[name]['val_r2'])
    best_model = models[best_model_name]
    
    # Feature importance
    feat_names = list(X_train.columns)
    if hasattr(best_model, 'feature_importances_'):
        importance_values = best_model.feature_importances_
    else:
        importance_values = np.abs(best_model.coef_)
    importance = dict(zip(feat_names,
                          [round(float(v), 4) for v in importance_values]))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # EDA stats for dashboard
    eda_stats = compute_eda_stats(maindata)
    
    # Save artifacts
    print("\nSaving artifacts...")
    with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save encoder state as plain dicts (avoids class pickle issues)
    encoder_state = {}
    for key, enc in encoders.items():
        # Works for both sklearn LabelEncoder and EncoderExt
        le = getattr(enc, 'le', enc)  # EncoderExt wraps a .le; plain LE is itself
        encoder_state[key] = {'classes_': list(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'encoders.json'), 'w') as f:
        json.dump(encoder_state, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(FEATURE_COLS, f)
    with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(importance, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'eda_stats.json'), 'w') as f:
        json.dump(eda_stats, f, indent=2)
    
    print("✓ All artifacts saved to models/")
    return results


def compute_eda_stats(df):
    stats = {}
    
    # Delay distribution
    df2 = df.copy()
    df2['Delay'] = (df2['clear_date'] - df2['due_in_date']).dt.days
    
    stats['delay_distribution'] = {
        'early': int((df2['Delay'] < 0).sum()),
        'on_time': int((df2['Delay'] == 0).sum()),
        'late_0_15': int(((df2['Delay'] > 0) & (df2['Delay'] <= 15)).sum()),
        'late_16_30': int(((df2['Delay'] > 15) & (df2['Delay'] <= 30)).sum()),
        'late_31_60': int(((df2['Delay'] > 30) & (df2['Delay'] <= 60)).sum()),
        'late_60plus': int((df2['Delay'] > 60).sum()),
    }
    
    stats['avg_delay_by_currency'] = (
        df2.groupby('invoice_currency' if 'invoice_currency' in df2.columns else 'business_code')['Delay']
        .mean().round(2).to_dict() if 'invoice_currency' in df2.columns
        else df2.groupby('business_code')['Delay'].mean().round(2).to_dict()
    )
    
    stats['avg_delay_by_year'] = (
        df2.groupby('buisness_year')['Delay'].mean().round(2).to_dict()
    )
    stats['avg_delay_by_year'] = {str(int(k)): v for k, v in stats['avg_delay_by_year'].items()}
    
    # Amount stats
    col = 'converted_usd' if 'converted_usd' in df2.columns else 'total_open_amount'
    stats['amount_stats'] = {
        'mean':   round(float(df2[col].mean()), 2),
        'median': round(float(df2[col].median()), 2),
        'p90':    round(float(df2[col].quantile(0.9)), 2),
    }
    
    stats['total_invoices'] = int(len(df2))
    stats['avg_delay_days'] = round(float(df2['Delay'].mean()), 1)
    stats['pct_late'] = round(float((df2['Delay'] > 0).mean() * 100), 1)
    
    # Top delayed customers
    top_delayed = (df2.groupby('name_customer')['Delay']
                   .mean().sort_values(ascending=False)
                   .head(10).round(1).to_dict())
    stats['top_delayed_customers'] = top_delayed
    
    return stats


if __name__ == '__main__':
    results = train_pipeline()
    print("\n=== Model Comparison ===")
    for name, r in results.items():
        print(f"{name:25s}  Test R²={r['test_r2']:.4f}  Val R²={r['val_r2']:.4f}")
