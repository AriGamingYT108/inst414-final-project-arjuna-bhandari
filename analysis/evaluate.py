import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def evaluate_models():
    """
    Load the transformed dataset, re-split and preprocess features,
    load trained models, and output evaluation metrics for each.

    Steps:
    1. Load and normalize column headers
    2. Separate features (X) and target (y)
    3. One-hot encode categorical 'purpose' column
    4. Re-create train/test split for evaluation
    5. Scale test features for Logistic Regression
    6. Load saved model artifacts
    7. Print classification report, ROC AUC, and confusion matrix for:
       a) Logistic Regression  b) XGBoost
    """
    # Step 1: Load transformed data
    df = pd.read_csv('data/extracted/lending_cleaned.csv')
    # Normalize header names: drop type suffixes, lowercase, convert to snake_case
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    # Step 2: Define feature matrix X and target vector y
    target = 'not_fully_paid'           # Binary default label: 1=default, 0=fully paid
    X = df.drop(columns=[target])       # All other columns are model inputs
    y = df[target]                      # The label to predict

    # Step 3: One-hot encode 'purpose' if present
    if 'purpose' in X.columns:
        X = pd.get_dummies(X, columns=['purpose'], drop_first=True)

    # Step 4: Re-create the train/test split (80/20, stratified)
    # Stratifying ensures the default rate is consistent between sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Step 5: Load trained model artifacts
    models_dir = 'data/analysis/models'
    # Logistic Regression saved pipeline
    lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
    # XGBoost classifier
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost.pkl'))

    # Step 6a: Evaluate Logistic Regression
    logging.info("=== Logistic Regression Evaluation ===")
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    # Print precision, recall, F1-score per class
    logging.info("Classification Report (LR):\n%s", classification_report(y_test, y_pred_lr))
    # Print ROC AUC metric
    logging.info("ROC AUC (LR): %.4f", roc_auc_score(y_test, y_prob_lr))
    # Print confusion matrix
    logging.info("Confusion Matrix (LR):\n%s", confusion_matrix(y_test, y_pred_lr))

    # Step 6b: Evaluate XGBoost (no scaling needed)
    logging.info("=== XGBoost Evaluation ===")
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    # Print precision, recall, F1-score per class
    logging.info("Classification Report (XGB):\n%s", classification_report(y_test, y_pred_xgb))
    # Print ROC AUC metric
    logging.info("ROC AUC (XGB): %.4f", roc_auc_score(y_test, y_prob_xgb))
    # Print confusion matrix
    logging.info("Confusion Matrix (XGB):\n%s", confusion_matrix(y_test, y_pred_xgb))


if __name__ == '__main__':
    evaluate_models()
