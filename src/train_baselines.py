import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def train_baselines():
    models_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'models')
    data_path = os.path.join(models_dir, 'data_splits.npz')

    if not os.path.exists(data_path):
        print(
            f"Data file not found at {data_path}. Please run data_prep.py first.")
        return

    data = np.load(data_path)
    X_train_val = data['X_train_val']
    y_train_val = data['y_train_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_val, y_train_val)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, lr_probs):.4f}")
    print(f"F1 Score: {f1_score(y_test, lr_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_preds))

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_val, y_train_val)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, rf_probs):.4f}")
    print(f"F1 Score: {f1_score(y_test, rf_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

    # Save models
    joblib.dump(lr_model, os.path.join(models_dir, 'baseline_lr.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'baseline_rf.joblib'))
    print("\nBaseline models saved to models/ directory.")


if __name__ == "__main__":
    train_baselines()
