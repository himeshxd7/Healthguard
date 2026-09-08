import os
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    print("Loading prepared datasets...")
    data = np.load(os.path.join(models_dir, 'data_splits.npz'))
    X_train_val = data['X_train_val']
    y_train_val = data['y_train_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print("Training HistGradientBoostingClassifier...")
    # This model natively handles np.nan values (missing data)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    
    model.fit(X_train_val, y_train_val)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Test Set Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(models_dir, 'heart_disease_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
