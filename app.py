import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import os
import matplotlib.pyplot as plt
from utils.llm_explainer import generate_explanation
from utils.db import insert_prediction, get_user_history
import uuid

# Configuration
st.set_page_config(page_title="HealthGuard AI", page_icon="🫀", layout="wide")

# Initialize session state for user_id
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Load Models


@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        model = tf.keras.models.load_model(
            os.path.join(models_dir, 'heart_disease_model.keras'))
        preprocessor = joblib.load(
            os.path.join(
                models_dir,
                'preprocessor.joblib'))
        data = np.load(os.path.join(models_dir, 'data_splits.npz'))
        background = data['X_train'][np.random.choice(
            data['X_train'].shape[0], 50, replace=False)]

        # Adjust background if model expects 3D
        if len(model.input_shape) == 3:
            background = np.expand_dims(background, axis=-1)

        explainer = shap.DeepExplainer(model, background)
        return model, preprocessor, explainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


model, preprocessor, explainer = load_models()

st.title("🫀 HealthGuard AI")
st.markdown("**Heart Disease Risk & Health Assistant**")
st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. It is driven by an AI model and is NOT a substitute for professional medical advice, diagnosis, or treatment.")

if model is None:
    st.error("Models not found. Please run the training scripts first.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Patient Data Input")
    with st.form("patient_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                1,
                2,
                3,
                4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic"}[x])
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120)
        chol = st.number_input(
            "Serum Cholestoral (mg/dl)",
            min_value=100,
            max_value=600,
            value=200)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[
                "No",
                "Yes"])
        restecg = st.selectbox(
            "Resting Electrocardiographic Results",
            options=[
                0,
                1,
                2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"}[x])
        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=50,
            max_value=250,
            value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        oldpeak = st.number_input(
            "ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1)
        slope = st.selectbox(
            "Slope of the Peak Exercise ST Segment", options=[
                1, 2, 3], format_func=lambda x: {
                1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
        ca = st.number_input(
            "Number of Major Vessels Colored by Flourosopy",
            min_value=0,
            max_value=3,
            value=0)
        thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {
                            3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

        submit_button = st.form_submit_button("Analyze Risk")

with col2:
    if submit_button:
        # Prepare inputs
        inputs_dict = {
            'age': age, 'sex': 1 if sex == "Male" else 0, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': 1 if fbs == "Yes" else 0, 'restecg': restecg,
            'thalach': thalach, 'exang': 1 if exang == "Yes" else 0, 'oldpeak': oldpeak,
            'slope': slope, 'ca': ca, 'thal': thal
        }

        input_df = pd.DataFrame([inputs_dict])

        # Preprocess
        processed_input = preprocessor.transform(input_df)

        # Check model expected shape and reshape if needed
        if len(model.input_shape) == 3:
            processed_input = np.expand_dims(processed_input, axis=-1)

        # Predict
        risk_score = float(model.predict(processed_input, verbose=0)[0][0])

        st.header("Analysis Results")

        # Risk Gauge
        if risk_score < 0.3:
            st.success(f"### Risk Score: {risk_score:.1%} (Low Risk)")
        elif risk_score < 0.7:
            st.warning(f"### Risk Score: {risk_score:.1%} (Moderate Risk)")
        else:
            st.error(f"### Risk Score: {risk_score:.1%} (High Risk)")

        st.progress(risk_score)

        # SHAP Explanation
        st.subheader("Key Driving Factors")
        with st.spinner("Analyzing driving factors..."):
            shap_values = explainer.shap_values(processed_input)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Get feature names
            feature_names = []
            feature_names.extend(
                ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'])
            cat_encoder = preprocessor.transformers_[
                1][1].named_steps['encoder']
            feature_names.extend(cat_encoder.get_feature_names_out())

            # Map top factors
            shap_vals = shap_values[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.flatten()
            feature_impacts = {
                feature_names[i]: shap_vals[i] for i in range(
                    len(feature_names))}
            sorted_impacts = sorted(
                feature_impacts.items(),
                key=lambda x: abs(
                    x[1]),
                reverse=True)[
                :5]
            top_factors_dict = {k: v for k, v in sorted_impacts}

            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh([k for k, _ in reversed(sorted_impacts)], [
                           v for _, v in reversed(sorted_impacts)])
            for i, bar in enumerate(bars):
                bar.set_color('red' if sorted_impacts[len(
                    sorted_impacts) - 1 - i][1] > 0 else 'blue')
            ax.set_xlabel('SHAP Value (Impact on prediction)')
            ax.set_title(
                'Top 5 Contributing Factors (Red = Increases Risk, Blue = Decreases Risk)')
            st.pyplot(fig)

        # Groq Explanation
        st.subheader("AI Health Assistant Explanation")
        with st.spinner("Generating personalized explanation..."):
            original_inputs = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg,
                'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak
            }
            explanation = generate_explanation(
                risk_score, top_factors_dict, original_inputs)
            st.markdown(explanation)

        # Save to DB
        with st.spinner("Saving to history..."):
            inserted = insert_prediction(
                st.session_state.user_id,
                original_inputs,
                risk_score,
                top_factors_dict,
                explanation)
            if not inserted:
                st.caption(
                    "Note: Database not connected or failed to save history.")

    # History Tab
    st.divider()
    with st.expander("View Patient History"):
        history = get_user_history(st.session_state.user_id)
        if history:
            for record in history:
                st.markdown(
                    f"**Date:** {record['created_at'][:10]} | **Risk Score:** {record['risk_score']:.1%}")
        else:
            st.write("No previous history found.")
