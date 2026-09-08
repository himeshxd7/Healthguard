import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo

# Set seeds for reproducibility
np.random.seed(42)

def prepare_data():
    print("Fetching UCI Heart Disease dataset (ID 45)...")
    heart_disease = fetch_ucirepo(id=45)
    df_uci = heart_disease.data.features.copy()
    df_uci['target'] = heart_disease.data.targets['num'].apply(lambda x: 1 if x > 0 else 0)

    print("Loading Heart Failure Clinical Records dataset...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    clin_path = os.path.join(base_dir, 'heart_failure_clinical_records_dataset.csv')
    df_clin = pd.read_csv(clin_path)
    df_clin.rename(columns={'DEATH_EVENT': 'target'}, inplace=True)
    # The 'time' column is a leakage variable for predicting death (follow-up period), we should drop it.
    df_clin.drop(columns=['time'], inplace=True, errors='ignore')

    print("Merging datasets...")
    df_merged = pd.concat([df_uci, df_clin], ignore_index=True)

    X = df_merged.drop('target', axis=1)
    y = df_merged['target']

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    # All other features are numeric
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Preprocessing pipeline WITHOUT Imputers!
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        # Use OrdinalEncoder and let NaNs pass through
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=np.nan))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    print("Splitting dataset into train, validation, and test sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    print("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    X_train_val_processed = preprocessor.fit_transform(X_train_val)

    # Save the pipeline
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    pipeline_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, pipeline_path)
    print(f"Preprocessor saved to {pipeline_path}")

    # Save processed datasets
    np.savez(os.path.join(models_dir, 'data_splits.npz'),
             X_train=X_train_processed, y_train=y_train.values.ravel(),
             X_val=X_val_processed, y_val=y_val.values.ravel(),
             X_test=X_test_processed, y_test=y_test.values.ravel(),
             X_train_val=X_train_val_processed, y_train_val=y_train_val.values.ravel())
    
    # Save feature names order for app.py
    feature_names = numeric_features + categorical_features
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    print("Processed datasets and feature names saved.")

if __name__ == "__main__":
    prepare_data()
