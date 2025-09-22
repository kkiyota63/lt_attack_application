#!/usr/bin/env python3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np

def train_balanced_model():
    """Train a balanced XGBoost model that can properly classify fraud samples"""
    
    # Load and prepare data
    df = pd.read_csv("XGBoost/clean_data_sample_10percent.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Extract features and labels
    X = df.drop('fraud_bool', axis=1)
    y = df['fraud_bool']
    
    print(f"Original class distribution:")
    print(y.value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost model with class balancing
    model = xgb.XGBClassifier(
        n_estimators=50,  # Fewer trees for simpler model
        max_depth=4,      # Shallower trees
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Balance classes
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("Training balanced XGBoost model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
    
    print("\nTrain confusion matrix:")
    print(confusion_matrix(y_train, train_pred))
    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred))
    
    # Check fraud detection specifically
    fraud_mask = y_test == 1
    fraud_predictions = test_pred[fraud_mask]
    print(f"\nFraud detection performance:")
    print(f"Total fraud samples in test: {sum(fraud_mask)}")
    print(f"Correctly classified as fraud: {sum(fraud_predictions == 1)}")
    print(f"Misclassified as normal: {sum(fraud_predictions == 0)}")
    
    # Save model
    model.save_model("models/balanced_fraud_model.model")
    print("Saved balanced model to models/balanced_fraud_model.model")
    
    # Export to JSON format for lt_attack
    booster = model.get_booster()
    model_json = booster.get_dump(dump_format='json')
    
    # Get feature names to create mapping
    feature_names = list(X.columns)
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    def convert_tree_format(tree_node):
        """Recursively convert tree format to match expected format"""
        if 'leaf' in tree_node:
            return {"nodeid": tree_node.get('nodeid', 0), "leaf": tree_node['leaf']}
        
        converted = {
            "nodeid": tree_node.get('nodeid', 0),
            "depth": tree_node.get('depth', 0),
            "split": feature_name_to_idx.get(tree_node['split'], 0),
            "split_condition": tree_node['split_condition'],
            "yes": tree_node['yes'],
            "no": tree_node['no'],
            "missing": tree_node['missing']
        }
        
        if 'children' in tree_node:
            converted['children'] = [convert_tree_format(child) for child in tree_node['children']]
        
        return converted
    
    # Parse and reformat for lt_attack tool
    trees = []
    for tree_str in model_json:
        tree = json.loads(tree_str)
        converted_tree = convert_tree_format(tree)
        trees.append(converted_tree)
    
    # Save JSON
    with open("models/balanced_fraud_model.json", 'w') as f:
        json.dump(trees, f, indent=2)
    
    print(f"Model JSON saved to models/balanced_fraud_model.json")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of trees: {len(trees)}")
    
    return X.shape[1], len(trees)

if __name__ == "__main__":
    train_balanced_model()