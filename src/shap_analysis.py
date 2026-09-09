import os
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt


def run_shap_analysis():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    charts_dir = os.path.join(base_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    model_path      = os.path.join(models_dir, 'heart_disease_model.joblib')
    data_path       = os.path.join(models_dir, 'data_splits.npz')
    feat_names_path = os.path.join(models_dir, 'feature_names.joblib')

    for path, name in [(model_path, 'Model'), (data_path, 'Data splits'), (feat_names_path, 'Feature names')]:
        if not os.path.exists(path):
            print(f"  {name} not found at {path}. Please run data_prep.py and train_hgb.py first.")
            return

    print("Loading model and data...")
    model        = joblib.load(model_path)
    feature_names = joblib.load(feat_names_path)
    data         = np.load(data_path, allow_pickle=True)
    X_test       = data['X_test']
    X_train_val  = data['X_train_val']

    print(f"  Model type: {type(model).__name__}")
    print(f"  Features:   {feature_names}")
    print(f"  Test samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # SHAP EXPLAINER
    # The production model is a CalibratedClassifierCV wrapping a
    # HistGradientBoostingClassifier. We extract the underlying HGB estimator
    # from the first calibrated classifier and use TreeExplainer on it.
    # TreeExplainer is fast and exact for tree-based models.
    # -------------------------------------------------------------------------
    print("\nExtracting base tree model for TreeExplainer...")
    try:
        # CalibratedClassifierCV wraps calibrated estimators.
        # In sklearn 1.6+ with FrozenEstimator, the chain is:
        # CalibratedClassifierCV -> calibrated_classifiers_[0] -> estimator (FrozenEstimator) -> estimator (HGB)
        raw_estimator = model.calibrated_classifiers_[0].estimator
        # Unwrap FrozenEstimator if present
        if hasattr(raw_estimator, 'estimator'):
            raw_estimator = raw_estimator.estimator
        print(f"  Base estimator: {type(raw_estimator).__name__}")
        explainer = shap.TreeExplainer(raw_estimator)
        explainer_type = "TreeExplainer"
    except (AttributeError, IndexError, Exception) as e:
        # Fallback: if model structure is different, use KernelExplainer
        print("  Falling back to KernelExplainer (slower but model-agnostic)...")
        background = X_train_val[np.random.choice(X_train_val.shape[0], 100, replace=False)]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        explainer_type = "KernelExplainer"

    print(f"  Using: {explainer_type}")

    # -------------------------------------------------------------------------
    # COMPUTE SHAP VALUES
    # -------------------------------------------------------------------------
    print("\nGenerating SHAP values for test set (this may take a moment)...")

    # For large test sets, sample 200 rows for the summary plot
    n_samples = min(200, len(X_test))
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[idx]

    shap_values = explainer.shap_values(X_sample)

    # Handle list output (binary classification with some explainers)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    print(f"  SHAP values shape: {shap_values.shape}")

    # -------------------------------------------------------------------------
    # HUMAN-READABLE FEATURE NAMES
    # -------------------------------------------------------------------------
    readable_names = {
        'age':      'Age',
        'sex':      'Sex (Male)',
        'cp':       'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure',
        'chol':     'Cholesterol',
        'fbs':      'Fasting Blood Sugar >120',
        'restecg':  'Resting ECG Result',
        'thalach':  'Max Heart Rate Achieved',
        'exang':    'Exercise-Induced Angina',
        'oldpeak':  'ST Depression (Exercise)',
        'slope':    'ST Segment Slope',
        'ca':       'Major Vessels Blocked',
        'thal':     'Thalassemia',
    }
    display_names = [readable_names.get(f, f) for f in feature_names]

    # -------------------------------------------------------------------------
    # SUMMARY PLOT — Global feature importance
    # -------------------------------------------------------------------------
    print("\nGenerating global SHAP summary plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=display_names,
        show=False,
        max_display=13
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=14, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(charts_dir, 'shap_summary_plot.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot saved to {summary_path}")

    # -------------------------------------------------------------------------
    # BAR PLOT — Mean absolute SHAP values (feature ranking)
    # -------------------------------------------------------------------------
    print("Generating feature importance bar chart...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        [display_names[i] for i in reversed(sorted_idx)],
        mean_abs_shap[sorted_idx[::-1]],
        color='#2196F3'
    )
    plt.xlabel('Mean |SHAP Value| (Average Impact on Risk Score)')
    plt.title('Feature Importance — HealthGuard AI Model', fontsize=13)
    plt.tight_layout()
    bar_path = os.path.join(charts_dir, 'shap_feature_importance.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance chart saved to {bar_path}")

    print("\nSHAP analysis complete!")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    run_shap_analysis()
