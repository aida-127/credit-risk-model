"""
TASK 5: COMPLETE MODEL TRAINING SCRIPT
- Trains multiple models (Logistic Regression, Random Forest, XGBoost)
- Logs everything to MLflow
- Handles missing data files gracefully
- Creates all required metrics and artifacts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
import os
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "credit-risk-modeling-v2"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# ============================================================================
# SETUP MLFLOW
# ============================================================================
def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("="*80)
    print("MLFLOW MODEL TRAINING - TASK 5")
    print("="*80)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data():
    """
    Prepare training data with real files or synthetic data
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    try:
        # Try to load real data from Task 4
        print("Attempting to load processed data from Task 4...")
        customer_features = pd.read_csv('data/processed/customer_features.csv')
        target = pd.read_csv('data/processed/target_variable.csv')
        
        # Merge features with target
        data = pd.merge(customer_features, target[['CustomerId', 'is_high_risk']], 
                       on='CustomerId', how='inner')
        
        print(f"✅ Successfully loaded real data!")
        print(f"   • Samples: {data.shape[0]}")
        print(f"   • Features: {data.shape[1] - 2}")  # Exclude CustomerId and target
        print(f"   • Target distribution: {data['is_high_risk'].value_counts().to_dict()}")
        
    except FileNotFoundError as e:
        print(f"⚠️  Real data not found: {e}")
        print("Creating synthetic RFM data for demonstration...")
        
        # Create realistic synthetic RFM data
        np.random.seed(RANDOM_STATE)
        n_samples = 5000
        
        # Generate realistic customer behavior patterns
        data = pd.DataFrame({
            'CustomerId': range(1, n_samples + 1),
            
            # RFM Features
            'recency': np.random.exponential(30, n_samples),  # Days since last purchase
            'frequency': np.random.poisson(15, n_samples),    # Purchase frequency
            'monetary': np.random.lognormal(7, 1.2, n_samples),  # Total spending
            
            # Transaction patterns
            'transaction_count': np.random.randint(1, 100, n_samples),
            'avg_transaction_value': np.random.uniform(20, 500, n_samples),
            'transaction_std': np.random.exponential(75, n_samples),
            'max_transaction': np.random.lognormal(6.5, 1.5, n_samples),
            'min_transaction': np.random.uniform(1, 50, n_samples),
            
            # Time-based features
            'days_since_first': np.random.uniform(30, 365, n_samples),
            'transactions_per_day': np.random.uniform(0.1, 2, n_samples),
            
            # Behavioral features
            'weekend_ratio': np.random.beta(2, 5, n_samples),
            'hour_variability': np.random.uniform(0.1, 0.9, n_samples),
            'category_diversity': np.random.randint(1, 10, n_samples),
            
            # Channel usage (encoded)
            'channel_web': np.random.binomial(1, 0.6, n_samples),
            'channel_mobile': np.random.binomial(1, 0.3, n_samples),
            'channel_pos': np.random.binomial(1, 0.1, n_samples),
            
            # Product preferences
            'category_financial': np.random.binomial(1, 0.4, n_samples),
            'category_retail': np.random.binomial(1, 0.7, n_samples),
            'category_digital': np.random.binomial(1, 0.3, n_samples),
            
            # Risk indicators
            'chargeback_ratio': np.random.beta(1, 50, n_samples),
            'failed_transactions': np.random.poisson(0.5, n_samples),
            'velocity_1d': np.random.exponential(2, n_samples),
            'velocity_7d': np.random.exponential(10, n_samples),
        })
        
        # Create realistic target variable based on RFM patterns
        risk_score = (
            0.3 * (data['recency'] > 60) +           # High recency = risky
            0.2 * (data['frequency'] < 5) +          # Low frequency = risky
            0.2 * (data['monetary'] < 100) +         # Low monetary = risky
            0.1 * (data['chargeback_ratio'] > 0.05) + # High chargebacks
            0.1 * (data['failed_transactions'] > 1) + # Failed transactions
            0.1 * np.random.randn(n_samples)          # Random noise
        )
        
        data['is_high_risk'] = (risk_score > risk_score.median()).astype(int)
        
        print(f"✅ Created synthetic data with {n_samples} samples")
        print(f"   • High-risk customers: {data['is_high_risk'].sum()} ({data['is_high_risk'].mean()*100:.1f}%)")
        print(f"   • Low-risk customers: {(data['is_high_risk'] == 0).sum()} ({(1-data['is_high_risk'].mean())*100:.1f}%)")
        
        # Save synthetic data for reference
        os.makedirs('data/synthetic', exist_ok=True)
        data.to_csv('data/synthetic/synthetic_rfm_data.csv', index=False)
        print(f"   • Synthetic data saved: data/synthetic/synthetic_rfm_data.csv")
    
    # Prepare features and target
    feature_cols = [col for col in data.columns if col not in ['CustomerId', 'is_high_risk']]
    X = data[feature_cols].copy()
    y = data['is_high_risk'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Encode any remaining categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    print(f"\n📊 Final dataset:")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Samples: {X.shape[0]}")
    print(f"   • Feature names: {list(X.columns[:5])}..." if X.shape[1] > 5 else list(X.columns))
    
    return X, y, list(X.columns)

# ============================================================================
# MODEL TRAINING WITH MLFLOW
# ============================================================================
def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train multiple models with hyperparameter tuning and MLflow logging
    """
    print("\n" + "="*60)
    print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Define models with hyperparameter grids
    models_config = {
        "Logistic_Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    results = []
    best_model_info = {"name": "", "model": None, "score": 0, "metrics": {}}
    
    for model_name, config in models_config.items():
        print(f"\n🔧 {model_name}")
        print("-" * 40)
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name, nested=True):
            print(f"   MLflow run started: {mlflow.active_run().info.run_id}")
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Hyperparameter tuning with GridSearchCV
            print(f"   Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Log best hyperparameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Make predictions
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            # Log artifacts
            log_artifacts(best_model, X_test_scaled, y_test, y_pred, y_pred_proba, 
                         model_name, feature_names, scaler)
            
            # Store results
            results.append({
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1_Score": round(metrics["f1"], 4),
                "ROC_AUC": round(metrics["roc_auc"], 4),
                "Best_Params": grid_search.best_params_
            })
            
            # Update best model
            if metrics["roc_auc"] > best_model_info["score"]:
                best_model_info = {
                    "name": model_name,
                    "model": best_model,
                    "score": metrics["roc_auc"],
                    "metrics": metrics,
                    "params": grid_search.best_params_
                }
            
            print(f"   ✅ Best params: {grid_search.best_params_}")
            print(f"   📊 ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    
    return results, best_model_info, scaler

# ============================================================================
# ARTIFACT LOGGING
# ============================================================================
def log_artifacts(model, X_test, y_test, y_pred, y_pred_proba, 
                  model_name, feature_names, scaler):
    """
    Create and log artifacts to MLflow
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create artifacts directory
    artifacts_dir = f"artifacts/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f"{artifacts_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    roc_path = f"{artifacts_dir}/roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path)
    
    # 3. Feature Importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.barh(range(min(20, len(feature_names))), 
                importance[indices][:20][::-1], 
                align='center')
        plt.yticks(range(min(20, len(feature_names))), 
                  [feature_names[i] for i in indices[:20]][::-1])
        plt.xlabel('Relative Importance')
        feature_path = f"{artifacts_dir}/feature_importance.png"
        plt.savefig(feature_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(feature_path)
        
        # Save feature importance as CSV
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        feature_csv_path = f"{artifacts_dir}/feature_importance.csv"
        feature_importance_df.to_csv(feature_csv_path, index=False)
        mlflow.log_artifact(feature_csv_path)
    
    # 4. Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = f"{artifacts_dir}/classification_report.csv"
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path)
    
    # 5. Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'predicted_probability': y_pred_proba
    })
    predictions_path = f"{artifacts_dir}/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    mlflow.log_artifact(predictions_path)
    
    # 6. Save scaler
    scaler_path = f"{artifacts_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    
    # Setup MLflow
    experiment = setup_mlflow()
    
    # Prepare data
    X, y, feature_names = prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📈 Data split:")
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Testing samples: {X_test.shape[0]}")
    print(f"   • Positive class in train: {y_train.mean():.2%}")
    print(f"   • Positive class in test: {y_test.mean():.2%}")
    
    # Train models
    results, best_model_info, scaler = train_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Display results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Model: {best_model_info['name']}")
    print(f"ROC-AUC: {best_model_info['score']:.4f}")
    print(f"Best Parameters: {best_model_info['params']}")
    
    # Register best model
    print("\n" + "="*80)
    print("REGISTERING BEST MODEL")
    print("="*80)
    
    with mlflow.start_run(run_name="best_model_registration") as run:
        mlflow.log_metrics(best_model_info["metrics"])
        mlflow.log_params(best_model_info["params"])
        
        if best_model_info['name'] == "XGBoost":
            mlflow.xgboost.log_model(best_model_info["model"], "best_model")
        else:
            mlflow.sklearn.log_model(best_model_info["model"], "best_model")
        
        # Log best model info
        model_info = {
            "best_model": best_model_info["name"],
            "roc_auc": float(best_model_info["score"]),
            "timestamp": datetime.now().isoformat(),
            "features_used": feature_names
        }
        
        with open("best_model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        mlflow.log_artifact("best_model_info.json")
        
        print(f"✅ Best model registered!")
        print(f"🔗 Model URI: runs:/{run.info.run_id}/best_model")
        
        # Save final model locally
        final_model_path = f"models/best_{best_model_info['name'].lower()}.pkl"
        joblib.dump(best_model_info["model"], final_model_path)
        print(f"💾 Model saved locally: {final_model_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TASK 5 COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n📊 MLflow Experiment: {experiment.name}")
    print(f"🔗 View results at: {MLFLOW_TRACKING_URI}")
    print(f"📈 Total models trained: {len(results)}")
    print(f"⭐ Best model: {best_model_info['name']} (ROC-AUC: {best_model_info['score']:.4f})")
    print("\n📁 Generated artifacts:")
    print("   • models/ - Trained models and scaler")
    print("   • artifacts/ - Evaluation plots and reports")
    print("   • data/synthetic/ - Synthetic training data")
    
    # Create prediction script
    create_prediction_script(best_model_info["name"], feature_names)

def create_prediction_script(best_model_name, feature_names):
    """Create a prediction script for the best model"""
    
    script_content = f'''"""
PREDICTION SCRIPT FOR CREDIT RISK MODEL
Best Model: {best_model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
import joblib
import numpy as np
import pandas as pd

class CreditRiskPredictor:
    def __init__(self):
        """Initialize the predictor with trained model and scaler"""
        self.model = joblib.load('models/best_{best_model_name.lower()}.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.feature_names = {feature_names}
        
    def predict(self, customer_data):
        """
        Predict credit risk for a customer.
        
        Args:
            customer_data: dict with feature values or pandas DataFrame
        
        Returns:
            dict with prediction results
        """
        # Convert to DataFrame if dict
        if isinstance(customer_data, dict):
            input_df = pd.DataFrame([customer_data])
        else:
            input_df = customer_data.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns
        input_df = input_df[self.feature_names]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        probability = self.model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if probability > 0.5 else 0
        
        # Calculate risk score (0-1000)
        risk_score = int(probability * 1000)
        
        # Risk interpretation
        if probability < 0.3:
            risk_level = "Low"
            recommendation = "Approve"
        elif probability < 0.7:
            risk_level = "Medium"
            recommendation = "Review"
        else:
            risk_level = "High"
            recommendation = "Decline"
        
        return {{
            "probability": float(probability),
            "prediction": int(prediction),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "threshold_used": 0.5
        }}
    
    def batch_predict(self, customers_data):
        """Predict for multiple customers"""
        results = []
        for customer in customers_data:
            results.append(self.predict(customer))
        return results

# Example usage
if __name__ == "__main__":
    # Example customer data
    example_customer = {{
        "recency": 45.2,
        "frequency": 12,
        "monetary": 1250.75,
        "transaction_count": 48,
        "avg_transaction_value": 156.34
        # Add all other features with realistic values
    }}
    
    predictor = CreditRiskPredictor()
    result = predictor.predict(example_customer)
    
    print("Credit Risk Prediction Result:")
    print(f"• Probability of high risk: {{result['probability']:.2%}}")
    print(f"• Prediction: {{'High Risk' if result['prediction'] == 1 else 'Low Risk'}}")
    print(f"• Risk Score: {{result['risk_score']}}/1000")
    print(f"• Risk Level: {{result['risk_level']}}")
    print(f"• Recommendation: {{result['recommendation']}}")
'''
    
    os.makedirs('src', exist_ok=True)
    with open('src/predict.py', 'w') as f:
        f.write(script_content)
    
    print(f"📝 Prediction script created: src/predict.py")

if __name__ == "__main__":
    main()