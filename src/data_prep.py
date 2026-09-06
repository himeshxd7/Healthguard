import os

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo


# Set seeds for reproducibility
np.random.seed(42)


def prepare_data():
    print("Fetching UCI Heart Disease dataset (ID 45)...")
    # Fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # Data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Convert target to binary classification (0: no disease, >0: disease)
    y = y.copy()
    y['num'] = y['num'].apply(lambda x: 1 if x > 0 else 0)

    # The dataset has some missing values (represented as NaNs typically if
    # fetched this way)
    print("Missing values before imputation:\n", X.isnull().sum())

    # Features categorization based on dataset info
    categorical_features = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'thal']
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    # We will build a preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    print("Splitting dataset into train, validation, and test sets...")
    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: From the 80%, split 75/25 for train/val (so val is 20% of
    # total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(
        f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    print("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Also fit on the entire training+validation set for final baseline models
    # that don't need early stopping
    X_train_val_processed = preprocessor.fit_transform(X_train_val)

    # Save the pipeline
    models_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)),
        'models')
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
    print("Processed datasets saved.")


if __name__ == "__main__":
    prepare_data()
