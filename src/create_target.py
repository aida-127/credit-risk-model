# src/create_target.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

print("="*60)
print("TASK 4: PROXY TARGET CREATION")
print("="*60)

# 1. Load RFM features from Task 3
print("1. Loading RFM features...")
try:
    rfm = pd.read_csv('data/processed/customer_features.csv')
    print(f"   ✅ Loaded {rfm.shape[0]:,} customers, {rfm.shape[1]} features")
except FileNotFoundError:
    print("   ❌ File not found: data/processed/customer_features.csv")
    print("   Run Task 3 first: python src/data_processing.py")
    exit()

# 2. Select RFM for clustering
print("\n2. Preparing data for clustering...")
X = rfm[['recency', 'frequency', 'total_amount']].copy()
print(f"   Features: recency, frequency, total_amount")

# 3. Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✅ Features scaled (mean=0, std=1)")

# 4. K-means clustering (3 clusters as required)
print("\n4. Running K-means clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(X_scaled)
print("   ✅ 3 clusters created")

# 5. Analyze clusters
print("\n5. Analyzing clusters...")
cluster_stats = rfm.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean', 
    'total_amount': 'mean',
    'CustomerId': 'count'
}).round(2)

print("\n📊 CLUSTER STATISTICS:")
print(cluster_stats)

# 6. Identify high-risk cluster
print("\n6. Identifying high-risk cluster...")
cluster_stats['risk_score'] = (
    cluster_stats['recency'] - 
    cluster_stats['frequency'] - 
    cluster_stats['total_amount'] / cluster_stats['total_amount'].max()
)

high_risk_cluster = cluster_stats['risk_score'].idxmax()
print(f"   🔴 High-risk cluster: {high_risk_cluster}")
print(f"      • Recency: {cluster_stats.loc[high_risk_cluster, 'recency']:.1f} days (high = inactive)")
print(f"      • Frequency: {cluster_stats.loc[high_risk_cluster, 'frequency']:.1f} transactions (low)")
print(f"      • Total amount: ${cluster_stats.loc[high_risk_cluster, 'total_amount']:,.2f} (low)")

# 7. Create binary target variable
print("\n7. Creating target variable...")
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# Statistics
risk_rate = rfm['is_high_risk'].mean() * 100
high_risk_count = rfm['is_high_risk'].sum()
low_risk_count = (rfm['is_high_risk'] == 0).sum()

print(f"   ✅ Target variable 'is_high_risk' created")
print(f"      • High-risk customers: {high_risk_count:,} ({risk_rate:.1f}%)")
print(f"      • Low-risk customers: {low_risk_count:,} ({100-risk_rate:.1f}%)")

# 8. Save target variable
print("\n8. Saving results...")
target_cols = ['CustomerId', 'recency', 'frequency', 'total_amount', 'cluster', 'is_high_risk']
rfm[target_cols].to_csv('data/processed/target_variable.csv', index=False)
print(f"   ✅ Saved: data/processed/target_variable.csv")

# 9. Visualization
print("\n9. Creating visualizations...")
plt.figure(figsize=(12, 5))

# Risk distribution
plt.subplot(1, 2, 1)
risk_counts = rfm['is_high_risk'].value_counts()
colors = ['green', 'red']
plt.bar(['Low Risk (0)', 'High Risk (1)'], risk_counts.values, color=colors)
plt.title('Risk Distribution')
plt.ylabel('Number of Customers')
for i, v in enumerate(risk_counts.values):
    plt.text(i, v, f'{v}\n({v/len(rfm)*100:.1f}%)', ha='center', va='bottom', fontsize=10)

# Recency vs Frequency
plt.subplot(1, 2, 2)
scatter = plt.scatter(rfm['recency'], rfm['frequency'], c=rfm['cluster'], 
                     cmap='viridis', alpha=0.6, s=30)
plt.title('Recency vs Frequency (Clusters)')
plt.xlabel('Recency (days since last transaction)')
plt.ylabel('Frequency (number of transactions)')
plt.colorbar(label='Cluster')

plt.tight_layout()
os.makedirs('notebooks', exist_ok=True)
plt.savefig('notebooks/task4_clusters.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"   ✅ Visualization saved: notebooks/task4_clusters.png")

# 10. Final summary
print("\n" + "="*60)
print("TASK 4 COMPLETE - SUMMARY")
print("="*60)
print(f"• Dataset: {rfm.shape[0]:,} customers")
print(f"• Clusters: 3 (as required)")
print(f"• High-risk cluster: {high_risk_cluster}")
print(f"• High-risk rate: {risk_rate:.1f}%")
print(f"• Files created:")
print(f"  - data/processed/target_variable.csv")
print(f"  - notebooks/task4_clusters.png")
print("\n✅ Ready for Task 5: Model Training")
print("="*60)

# Display sample
print("\n📋 SAMPLE OF TARGET VARIABLE (first 5 customers):")
print(rfm[['CustomerId', 'recency', 'frequency', 'total_amount', 'cluster', 'is_high_risk']].head())