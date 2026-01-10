import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
from datetime import datetime

def load_and_preprocess_data(filepath):
    """Load and preprocess the churn data"""
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(['churn', 'customer_id'], axis=1)
    y = df['churn']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 'tech_support']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    return X, y, label_encoders

def train_model(X_train, y_train):
    """Train the Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics

def main():
    print("Starting model training...")
    
    # Load data
    X, y, label_encoders = load_and_preprocess_data('data/churn_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save model and artifacts
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'n_features': len(X.columns),
        'feature_names': list(X.columns),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'roc_auc': float(metrics['roc_auc'])
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nModel saved successfully!")
    print("Files created:")
    print("  - models/churn_model.pkl")
    print("  - models/label_encoders.pkl")
    print("  - models/feature_names.pkl")
    print("  - models/model_metadata.json")

if __name__ == "__main__":
    main()