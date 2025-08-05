import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

def model_train():
    """
    1. Load data/extracted/transformed_data.csv and normalize headers.
    2. Define X (features) and y (target='not_fully_paid').
    3. One-hot encode 'purpose'.
    4. Train/test split (80/20, stratified).
    5. Scale for Logistic Regression.
    6. Train LogisticRegression & XGBClassifier.
    7. Save models to data/analysis/models/*.pkl.
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    print("Trained Logistic Regression.")

    scale = (y_train==0).sum()/(y_train==1).sum()
    xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale)
    xgb.fit(X_train, y_train)
    print("Trained XGBoost.")

    models_dir = 'data/analysis/models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(models_dir,'logistic_regression.pkl'))
    joblib.dump(xgb, os.path.join(models_dir,'xgboost.pkl'))
    print(f"✔ Models saved to {models_dir}")

if __name__ == '__main__':
    model_train()
