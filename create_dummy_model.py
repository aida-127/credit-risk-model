# create_dummy_model.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

print("=== Creating Dummy Model for Credit Risk API ===\n")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Create dummy training data (100 samples, 7 features)
print("1. Generating training data...")
np.random.seed(42)
X = np.random.randn(100, 7)  # 7 features matching your API
y = np.random.randint(0, 2, 100)  # Binary classification

print(f"   Data shape: {X.shape}")
print(f"   Classes: {np.unique(y, return_counts=True)}")

# Create and fit scaler
print("\n2. Creating and fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Scaler fitted with {scaler.n_features_in_} features")

# Create and train a simple model
print("\n3. Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
    max_depth=5
)
model.fit(X_scaled, y)
print(f"   Model trained with {model.n_estimators} estimators")
print(f"   Training accuracy: {model.score(X_scaled, y):.3f}")

# Save model and scaler
print("\n4. Saving model files...")
model_path = "models/xgboost_final.pkl"
scaler_path = "models/scaler.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"   ✅ Model saved: {model_path}")
print(f"   ✅ Scaler saved: {scaler_path}")

# Verify files exist
print("\n5. Verifying saved files...")
print(f"   Model file exists: {os.path.exists(model_path)}")
print(f"   Scaler file exists: {os.path.exists(scaler_path)}")

print("\n" + "="*50)
print("✅ Dummy model creation complete!")
print("Now restart Docker with: docker-compose down && docker-compose up -d")