# TASK 3: FEATURE ENGINEERING
import pandas as pd
import numpy as np
from datetime import datetime
import os

print("="*60)
print("TASK 3: FEATURE ENGINEERING")
print("="*60)

# Load data
df = pd.read_csv('data/raw/train.csv')
print(f"✅ Loaded: {len(df):,} transactions, {df['CustomerId'].nunique():,} customers")

# 1. RFM FEATURES (Critical for Task 4)
print("\n1. Creating RFM features...")
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'CustomerId': 'count',
    'Amount': ['sum', 'mean', 'std', 'min', 'max']
}).round(2)

rfm.columns = ['recency', 'frequency', 'total_amount', 'avg_amount', 
               'std_amount', 'min_amount', 'max_amount']
rfm = rfm.reset_index()

# Derived RFM features
rfm['amount_range'] = rfm['max_amount'] - rfm['min_amount']
rfm['amount_variability'] = rfm['std_amount'] / (rfm['avg_amount'] + 1e-6)
rfm['avg_transaction_size'] = rfm['total_amount'] / rfm['frequency']

# 2. TIME-BASED FEATURES
print("2. Creating time-based features...")
df['transaction_hour'] = df['TransactionStartTime'].dt.hour
df['transaction_day'] = df['TransactionStartTime'].dt.day
df['transaction_month'] = df['TransactionStartTime'].dt.month
df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
df['transaction_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
df['time_of_day'] = pd.cut(df['transaction_hour'], 
                          bins=[-1, 6, 12, 18, 24],
                          labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# 3. CATEGORICAL FEATURES
print("3. Processing categorical features...")
# Most frequent category per customer
customer_cats = df.groupby('CustomerId').agg({
    'ProductCategory': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
    'ChannelId': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
    'CountryCode': 'first'
}).reset_index()

customer_cats.columns = ['CustomerId', 'fav_category', 'fav_channel', 'country']

# 4. MERGE ALL FEATURES
print("4. Merging all features...")
# Merge everything
customer_features = pd.merge(rfm, customer_cats, on='CustomerId', how='left')

# Save customer-level features (for Task 4 clustering)
os.makedirs('data/processed', exist_ok=True)
customer_features.to_csv('data/processed/customer_features.csv', index=False)

# Merge with transaction data
train_with_features = pd.merge(df, customer_features, on='CustomerId', how='left')
train_with_features.to_csv('data/processed/train_with_features.csv', index=False)

# 5. SUMMARY
print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"• Original data: {df.shape}")
print(f"• Customer features: {customer_features.shape[0]} customers, {customer_features.shape[1]} features")
print(f"• Enhanced data: {train_with_features.shape}")

print(f"\n🔑 KEY RFM FEATURES CREATED:")
rfm_features = ['recency', 'frequency', 'total_amount', 'avg_amount', 'std_amount', 
                'amount_variability', 'avg_transaction_size']
for feat in rfm_features[:5]:
    print(f"  • {feat}")

print(f"\n🕒 TIME FEATURES:")
time_features = ['transaction_hour', 'transaction_day', 'transaction_month', 
                 'transaction_dayofweek', 'transaction_weekend', 'time_of_day']
for feat in time_features[:4]:
    print(f"  • {feat}")

print(f"\n📊 SAMPLE RFM FEATURES:")
print(customer_features[['CustomerId', 'recency', 'frequency', 'total_amount', 
                        'avg_amount', 'fav_category']].head(3))

print(f"\n💾 FILES SAVED:")
print(f"  • data/processed/customer_features.csv")
print(f"  • data/processed/train_with_features.csv")

print("\n✅ TASK 3 COMPLETE - Ready for Task 4 (Clustering)")
