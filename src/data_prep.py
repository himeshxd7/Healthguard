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
    # -------------------------------------------------------------------------
    # DATASET STRATEGY: Use only clinically compatible datasets.
    # UCI Heart Disease (ID=45, Cleveland) predicts coronary artery disease.
    # Statlog Heart (ID=145) has the same features and the same clinical task.
    # We combine these two — ~573 clean rows — all answering the same question:
    # "Does this person have heart disease?"
    #
    # The Heart Failure Clinical Records dataset (predicting death post-failure)
    # is intentionally NOT used here — it is a different clinical task with
    # different biomarkers and would corrupt the model with noisy mixed signals.
    # -------------------------------------------------------------------------

    print("Fetching UCI Heart Disease dataset (ID=45, Cleveland subset)...")
    heart_disease = fetch_ucirepo(id=45)
    df_uci = heart_disease.data.features.copy()
    df_uci['target'] = heart_disease.data.targets['num'].apply(lambda x: 1 if x > 0 else 0)
    print(f"  UCI Heart Disease: {len(df_uci)} rows, class balance: {df_uci['target'].value_counts().to_dict()}")

    print("Fetching Statlog Heart dataset (ID=145)...")
    try:
        statlog = fetch_ucirepo(id=145)
        df_stat = statlog.data.features.copy()
        df_stat['target'] = statlog.data.targets.iloc[:, 0].apply(lambda x: 1 if int(x) == 2 else 0)

        # Statlog uses the same 13 features as UCI Cleveland but may have
        # different column names. Align them.
        col_map = {
            'age': 'age', 'sex': 'sex', 'chest_pain_type': 'cp',
            'resting_blood_pressure': 'trestbps', 'serum_cholestoral': 'chol',
            'fasting_blood_sugar': 'fbs', 'resting_electrocardiographic_results': 'restecg',
            'maximum_heart_rate_achieved': 'thalach', 'exercise_induced_angina': 'exang',
            'oldpeak': 'oldpeak', 'slope': 'slope',
            'number_of_major_vessels': 'ca', 'thal': 'thal',
            # alternate common names
            'cp': 'cp', 'trestbps': 'trestbps', 'chol': 'chol', 'fbs': 'fbs',
            'restecg': 'restecg', 'thalach': 'thalach', 'exang': 'exang',
            'slope': 'slope', 'ca': 'ca', 'thal': 'thal',
        }
        df_stat.rename(columns={k: v for k, v in col_map.items() if k in df_stat.columns}, inplace=True)
        print(f"  Statlog Heart: {len(df_stat)} rows, class balance: {df_stat['target'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  Warning: Could not fetch Statlog dataset ({e}). Proceeding with UCI only.")
        df_stat = pd.DataFrame()

    # Merge compatible datasets
    print("Merging compatible datasets (same clinical task)...")
    frames = [df_uci]
    if not df_stat.empty:
        frames.append(df_stat)
    df_merged = pd.concat(frames, ignore_index=True)
    print(f"  Combined dataset: {len(df_merged)} rows")

    # -------------------------------------------------------------------------
    # FEATURE SCHEMA
    # We use the 13 UCI Cleveland features. The Heart Failure dataset features
    # (ejection_fraction, creatinine_phosphokinase, etc.) are NOT used for
    # model training because they don't exist in the UCI data.
    # They will still appear in the app UI and be passed to the LLM for context,
    # but the model receives NaN for those fields (HGB handles NaN natively).
    # -------------------------------------------------------------------------
    uci_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Keep only UCI features that actually exist in merged df
    available = [f for f in uci_features if f in df_merged.columns]
    df_merged = df_merged[available + ['target']].copy()

    # Drop rows with no target
    df_merged.dropna(subset=['target'], inplace=True)
    print(f"  Final dataset after cleaning: {len(df_merged)} rows")
    print(f"  Features used: {available}")
    print(f"  Class balance: {df_merged['target'].value_counts().to_dict()}")

    X = df_merged[available]
    y = df_merged['target'].astype(int)

    # -------------------------------------------------------------------------
    # FEATURE TYPES
    # -------------------------------------------------------------------------
    categorical_features = [f for f in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'] if f in available]
    numeric_features = [col for col in available if col not in categorical_features]

    # -------------------------------------------------------------------------
    # PREPROCESSING PIPELINE
    # No imputers — HGB handles NaN natively.
    # OrdinalEncoder passes NaN through (encoded_missing_value=np.nan).
    # -------------------------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=np.nan
        ))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    # -------------------------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # -------------------------------------------------------------------------
    print("Splitting into train/val/test (60/20/20)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # -------------------------------------------------------------------------
    # SMOTE — Synthetic Minority Oversampling
    # Balances classes in training set by synthesising new minority examples.
    # Applied AFTER splitting to avoid data leakage.
    # -------------------------------------------------------------------------
    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    X_train_val_processed = preprocessor.fit_transform(X_train_val)

    try:
        from imblearn.over_sampling import SMOTE
        # SMOTE requires no NaN — fill NaN with column median for SMOTE step only
        X_train_smote = pd.DataFrame(X_train_processed).fillna(pd.DataFrame(X_train_processed).median())
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min() - 1))
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_smote, y_train)
        # Re-introduce NaN back based on original mask (preserve original NaN pattern for HGB)
        print(f"  SMOTE applied: {len(X_train)} -> {len(X_train_resampled)} training samples")
        print(f"  Class balance after SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
        use_smote = True
    except ImportError:
        print("  Warning: imbalanced-learn not installed. Skipping SMOTE. Run: pip install imbalanced-learn")
        X_train_resampled = X_train_processed
        y_train_resampled = y_train.values
        use_smote = False

    # -------------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save preprocessor
    pipeline_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, pipeline_path)
    print(f"\nPreprocessor saved to {pipeline_path}")

    # Save feature names (ordered: numeric first, then categorical)
    feature_names = numeric_features + categorical_features
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    print(f"Feature names saved: {feature_names}")

    # Save processed data splits
    np.savez(
        os.path.join(models_dir, 'data_splits.npz'),
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_val=X_val_processed,
        y_val=y_val.values.ravel(),
        X_test=X_test_processed,
        y_test=y_test.values.ravel(),
        X_train_val=X_train_val_processed,
        y_train_val=y_train_val.values.ravel()
    )
    print("Data splits saved to models/data_splits.npz")
    print("\nData preparation complete!")
    if use_smote:
        print("Note: SMOTE was applied to training split only (no leakage).")
    print("\nNext step: python src/train_hgb.py")


if __name__ == "__main__":
    prepare_data()
