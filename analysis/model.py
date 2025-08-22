# analysis/model.py
"""
Train and save machine learning models for loan default prediction.

This script performs the following steps:
1. Loads cleaned data
2. Defines features (X) and target (y)
3. One-hot encodes the 'purpose' column
4. Splits into train and test sets (stratified to preserve class balance)
5. Imputes missing values and scales features for Logistic Regression
6. Trains a balanced Logistic Regression and an XGBoost classifier
7. Saves the trained model artifacts to disk

Usage:
    python analysis/model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def model_train():
    """
    Executes the full training pipeline:
    - Load and normalize transformed dataset
    - Prepare feature matrix and target vector
    - One-hot encode categorical variables
    - Split data into training and test subsets
    - Handle missing values and scale numeric features
    - Train and persist two classification models
    """
    # Step 1: Load data
    df = pd.read_csv('data/extracted/lending_cleaned.csv')
    # Normalize header names by dropping type suffixes and converting to snake_case
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    # Step 2: Define features (X) and target (y)
    target = 'not_fully_paid'             # Binary label: 1 = default, 0 = fully paid
    X = df.drop(columns=[target])         # Feature matrix
    y = df[target]                        # Target vector

    # Step 3: One-hot encode categorical 'purpose'
    if 'purpose' in X.columns:
        X = pd.get_dummies(X, columns=['purpose'], drop_first=True)

    # Step 4: Split into training and testing sets
    # Stratify to maintain the same default ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 5: Impute missing values and scale features for Logistic Regression
    # Impute numeric columns with training set median to avoid NaNs
    num_cols = X_train.select_dtypes(include=['number']).columns
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)
    # Standardize numeric features to mean=0, std=1

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000))
    ])
    lr_pipe.fit(X_train, y_train)
    logging.info("Trained Logistic Regression.")

    # 6b: XGBoost with scale_pos_weight to handle imbalance
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=imbalance_ratio)
    xgb.fit(X_train, y_train)
    logging.info("Trained XGBoost.")

    # Step 7: Save trained model artifacts
    out_dir = 'data/analysis/models'
    os.makedirs(out_dir, exist_ok=True)
    # Persist Logistic Regression and XGBoost models as .pkl files
    joblib.dump(lr_pipe, os.path.join(out_dir, 'logistic_regression.pkl'))
    joblib.dump(xgb, os.path.join(out_dir, 'xgboost.pkl'))
    logging.info("✔ Models saved to %s", out_dir)


if __name__ == '__main__':
    model_train()
