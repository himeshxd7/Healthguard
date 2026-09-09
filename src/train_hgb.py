import os
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, brier_score_loss, f1_score
)
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None  # older sklearn — fallback


def train_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')

    data_path = os.path.join(models_dir, 'data_splits.npz')
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run data_prep.py first.")
        return

    print("Loading prepared datasets...")
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    X_test  = data['X_test']
    y_test  = data['y_test']
    X_train_val = data['X_train_val']
    y_train_val = data['y_train_val']

    print(f"  Training samples: {len(X_train)} (after SMOTE if applied)")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # STEP 1: HYPERPARAMETER SEARCH
    # GridSearchCV with Stratified K-Fold to find best parameters.
    # Uses the training split only (not test) — no leakage.
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Hyperparameter Search (GridSearchCV, 5-Fold CV) ---")

    param_grid = {
        'max_iter':       [200, 300, 500],
        'learning_rate':  [0.03, 0.05, 0.1],
        'max_leaf_nodes': [15, 31, 63],
        'min_samples_leaf': [10, 20],
    }

    base_model = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=False,   # We control stopping via CV
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_auc = grid_search.best_score_

    print(f"\nBest CV AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")

    # -------------------------------------------------------------------------
    # STEP 2: CROSS-VALIDATION REPORT on full train+val set
    # Gives us a reliable performance estimate before touching the test set.
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Cross-Validation Report (5-Fold on train+val) ---")

    best_hgb = HistGradientBoostingClassifier(
        **best_params,
        random_state=42,
        early_stopping=False,
    )

    cv_results = cross_validate(
        best_hgb,
        X_train_val,
        y_train_val,
        cv=cv,
        scoring=['roc_auc', 'f1', 'accuracy'],
        return_train_score=False,
        n_jobs=-1
    )

    print(f"  CV AUC:      {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  CV F1:       {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"  CV Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")

    # -------------------------------------------------------------------------
    # STEP 3: TRAIN FINAL MODEL + CALIBRATE
    # Train on full train+val, then calibrate probabilities with isotonic
    # regression so risk_score=0.73 truly means ~73% probability.
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Training Final Model on Train+Val and Calibrating ---")

    final_hgb = HistGradientBoostingClassifier(
        **best_params,
        random_state=42,
        early_stopping=False,
    )

    # CalibratedClassifierCV with cv='prefit' uses the already-fitted model
    # and calibrates on a held-out set (X_val).
    final_hgb.fit(X_train, y_train)  # Fit on training split

    # Calibrate probabilities so risk_score truly represents probability.
    # sklearn >= 1.6 uses FrozenEstimator to avoid the deprecation warning.
    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(final_hgb),
            method='isotonic',
            cv='prefit'
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            final_hgb,
            method='isotonic',
            cv='prefit'
        )
    calibrated_model.fit(X_val, y_val)

    # Tune decision threshold on validation set to maximise F1
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.2, 0.8, 0.02):
        preds_t = (val_probs >= thresh).astype(int)
        f_t = f1_score(y_val, preds_t, zero_division=0)
        if f_t > best_f1:
            best_f1, best_thresh = f_t, thresh
    print(f"  Optimal threshold (val set): {best_thresh:.2f} (F1={best_f1:.4f})")  
    # Save threshold alongside model so app.py can use it
    joblib.dump(best_thresh, os.path.join(models_dir, 'decision_threshold.joblib'))

    # -------------------------------------------------------------------------
    # STEP 4: EVALUATE ON HELD-OUT TEST SET
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Final Evaluation on Test Set ---")

    y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= best_thresh).astype(int)
    print(f"  Evaluation threshold: {best_thresh:.2f} (tuned on val set)")

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_pred_prob)
    f1     = f1_score(y_test, y_pred)
    brier  = brier_score_loss(y_test, y_pred_prob)

    print(f"\n  Accuracy:    {acc:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Brier Score: {brier:.4f}  (lower is better; 0.0 = perfect calibration)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

    # -------------------------------------------------------------------------
    # STEP 5: SAVE MODEL
    # -------------------------------------------------------------------------
    model_path = os.path.join(models_dir, 'heart_disease_model.joblib')
    joblib.dump(calibrated_model, model_path)
    print(f"\nCalibrated model saved to {model_path}")
    print("\nNext step: python src/shap_analysis.py")


if __name__ == "__main__":
    train_model()
