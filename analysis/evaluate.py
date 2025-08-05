import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

def evaluate_models():
    """
    1. Load and normalize headers.
    2. Separate X/y, one-hot encode 'purpose'.
    3. Re-create the same train/test split.
    4. Scale test set for LR.
    5. Load models from data/analysis/models.
    6. Print classification_report, ROC AUC, confusion matrix.
    """
    df = pd.read_csv('data/extracted/transformed_data.csv')
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    target = 'not_fully_paid'
    X = df.drop(columns=[target])
    y = df[target]

    if 'purpose' in X.columns:
        X = pd.get_dummies(X, columns=['purpose'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_dir = 'data/analysis/models'
    lr = joblib.load(os.path.join(models_dir,'logistic_regression.pkl'))
    xgb = joblib.load(os.path.join(models_dir,'xgboost.pkl'))

    print("\n=== Logistic Regression Evaluation ===")
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:,1]
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC (LR): {roc_auc_score(y_test,y_prob):.4f}")
    print("Confusion Matrix (LR):")
    print(confusion_matrix(y_test,y_pred))

    print("\n=== XGBoost Evaluation ===")
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC (XGB): {roc_auc_score(y_test,y_prob):.4f}")
    print("Confusion Matrix (XGB):")
    print(confusion_matrix(y_test,y_pred))

if __name__ == '__main__':
    evaluate_models()
