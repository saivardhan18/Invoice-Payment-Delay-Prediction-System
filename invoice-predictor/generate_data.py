"""
Generate synthetic invoice dataset for demo purposes.
Mirrors the structure of the HighRadius internship dataset.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

BUSINESS_CODES = ['U001', 'CA02', 'UK03', 'AU04']
CURRENCIES = ['USD', 'CAD', 'GBP', 'AUD']
PAYMENT_TERMS = ['NAA8', 'NAH4', 'NAD1', 'NAM4', 'CA10', 'NAU5', 'NAC6', 'NAM2', 'NAAX']
DOCUMENT_TYPES = ['RV', 'DR', 'CR']

CUSTOMER_NAMES = [
    'WAL-MAR corp', 'BEN E Foods', 'MDV Trust', 'SYSC llc', 'KROGER Inc',
    'SAFEW associates', "BJ'S Wholesale", 'COST foundation', 'TARG us',
    'MAINES llc', 'C&S WH trust', 'GROC associates', 'ASSOCIAT foundation',
    'DEC corp', 'SA corporation', 'RESTA co', 'WINC trust', 'MEIJ in',
    'DECA corporation', 'OK systems', 'DOLLA co', 'SYSCO co', 'CO corporation',
    'AM wholesale', 'RA trust', 'YEN BROS corp', 'ZIYAD us', 'SAFEW Inc',
    'YAEGER corp', 'LOB associates', 'THE corporation', 'SUPE in'
]


def random_date(start_year=2019, end_year=2020):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 5, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_dataset(n=50000):
    records = []
    for i in range(n):
        biz_code = random.choice(BUSINESS_CODES)
        currency = 'USD' if biz_code == 'U001' else ('CAD' if biz_code == 'CA02' else random.choice(CURRENCIES))
        cust_name = random.choice(CUSTOMER_NAMES)
        cust_num = f"0{random.randint(100000000, 299999999)}"
        
        create_date = random_date()
        posting_date = create_date + timedelta(days=random.randint(0, 2))
        baseline_date = posting_date
        due_date = posting_date + timedelta(days=random.randint(7, 60))
        
        biz_year = posting_date.year
        
        # 80% chance of being paid (not open)
        is_open = random.choices([0, 1], weights=[80, 20])[0]
        
        if is_open == 0:
            delay_days = int(np.random.normal(5, 15))  # avg 5 days late
            clear_date = due_date + timedelta(days=delay_days)
        else:
            clear_date = None
        
        amount = round(np.random.lognormal(10, 1.2), 2)
        if currency == 'CAD':
            converted_usd = round(amount * 0.7, 2)
        else:
            converted_usd = amount
        
        doc_id = random.randint(1928502000, 9500000000)
        posting_id = 1.0
        
        records.append({
            'business_code': biz_code,
            'cust_number': cust_num,
            'name_customer': cust_name,
            'clear_date': clear_date.strftime('%Y-%m-%d %H:%M:%S') if clear_date else None,
            'buisness_year': float(biz_year),
            'doc_id': float(doc_id),
            'posting_date': posting_date.strftime('%Y-%m-%d'),
            'document_create_date': int(create_date.strftime('%Y%m%d')),
            'document_create_date.1': int(posting_date.strftime('%Y%m%d')),
            'due_in_date': float(due_date.strftime('%Y%m%d')),
            'invoice_currency': currency,
            'document type': random.choice(DOCUMENT_TYPES),
            'posting_id': posting_id,
            'area_business': None,
            'total_open_amount': amount,
            'baseline_create_date': float(baseline_date.strftime('%Y%m%d')),
            'cust_payment_terms': random.choice(PAYMENT_TERMS),
            'invoice_id': float(doc_id),
            'isOpen': is_open
        })
    
    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':
    print("Generating synthetic dataset...")
    df = generate_dataset(50000)
    df.to_csv('/home/claude/invoice-predictor/data/dataset.csv', index=False)
    print(f"Dataset saved: {df.shape}")
    print(df.head())
