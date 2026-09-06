import os
import numpy as np
import tensorflow as tf
import shap
import joblib
import matplotlib.pyplot as plt


def run_shap_analysis():
    models_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'models')
    charts_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Load model and data
    model_path = os.path.join(models_dir, 'heart_disease_model.keras')
    data_path = os.path.join(models_dir, 'data_splits.npz')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Model or data not found. Please run train_ann.py first.")
        return

    model = tf.keras.models.load_model(model_path)
    data = np.load(data_path)
    preprocessor = joblib.load(preprocessor_path)

    X_train = data['X_train']
    X_test = data['X_test']

    print("Generating SHAP values (this might take a moment)...")

    # Keras models can use DeepExplainer or KernelExplainer.
    # DeepExplainer is often faster for neural networks.
    # We use a background dataset (subset of train) to integrate over
    background = X_train[np.random.choice(
        X_train.shape[0], 100, replace=False)]

    # Adjust for CNN if needed
    if len(model.input_shape) == 3:
        background = np.expand_dims(background, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    explainer = shap.DeepExplainer(model, background)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # SHAP values shape may be a list of arrays for Keras models.
    # We take the first element if it is a list (binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Flatten if 3D
    if len(shap_values.shape) > 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Get feature names from the preprocessor
    # The preprocessor is a ColumnTransformer
    feature_names = []
    # numeric features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    feature_names.extend(numeric_features)
    # categorical features (OneHotEncoded)
    cat_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out()
    feature_names.extend(cat_feature_names)

    # Create a summary plot and save it
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'shap_summary_plot.png'))
    plt.close()

    print("SHAP analysis complete. Summary plot saved to charts/shap_summary_plot.png.")


if __name__ == "__main__":
    run_shap_analysis()
