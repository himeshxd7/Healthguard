import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import plotly.graph_objects as go
import plotly.express as px
from utils.llm_explainer import generate_explanation
from utils.db import insert_prediction, get_user_history
import uuid

# Configuration
st.set_page_config(page_title="HealthGuard AI", page_icon="🫀", layout="wide")
# Initialize session state for user_id
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        model = joblib.load(os.path.join(models_dir, 'heart_disease_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        
        # TreeExplainer is ideal for HistGradientBoosting
        explainer = shap.TreeExplainer(model)
        return model, preprocessor, explainer, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, preprocessor, explainer, feature_names = load_models()

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
    st.markdown("Please fill out your profile. Clinical tests can be left as 'Unknown' if you don't have the data. Our ML model natively handles missing data without auto-filling.")
    
    with st.expander("1. General Profile", expanded=True):
        age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Your chronological age in years.")
        sex = st.selectbox("Sex", options=["Male", "Female"])

    with st.expander("2. Lifestyle & Habits", expanded=True):
        smoking = st.selectbox("Do you smoke?", options=["No", "Yes"], help="Current or recent smoking habits.")
        cigarettes_per_day = 0
        if smoking == "Yes":
            cigarettes_per_day = st.number_input("How many cigarettes a day?", min_value=1, value=10, help="Average number of cigarettes smoked per day.")
        
        alcohol = st.selectbox("Do you consume alcohol?", options=["No", "Yes"])
        alcohol_freq = "None"
        if alcohol == "Yes":
            alcohol_freq = st.selectbox("How often do you consume alcohol?", options=["Occasionally", "Weekly", "Daily"])

    with st.expander("3. Basic Vitals & History", expanded=True):
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, help="Normal is around 120/80.")
        high_blood_pressure = st.selectbox("Diagnosed with High Blood Pressure?", options=["Unknown", "No", "Yes"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, help="Highest heart rate during exercise.")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        diabetes = st.selectbox("Diagnosed with Diabetes?", options=["Unknown", "No", "Yes"])

    with st.expander("4. Advanced Clinical Tests (Optional)", expanded=True):
        st.info("Leave fields blank or as 'Unknown' if you do not have these lab results. The model handles missing tests perfectly without guessing.")
        
        # Original UCI features
        cp = st.selectbox("Chest Pain Type", options=["Unknown", 1, 2, 3, 4], format_func=lambda x: "Unknown" if x == "Unknown" else {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x], help="Angina is chest pain caused by reduced blood flow to the heart.")
        chol_val = st.number_input("Serum Cholestoral (mg/dl)", min_value=100.0, max_value=600.0, value=None, placeholder="Type value if known...", help="The total amount of cholesterol in your blood. 'Serum' simply means the clear fluid part of the blood.")
            
        restecg = st.selectbox("Resting Electrocardiographic Results", options=["Unknown", 0, 1, 2], format_func=lambda x: "Unknown" if x == "Unknown" else {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x], help="Electrocardiographic (ECG) measures the electrical activity of your heart while at rest.")
        exang = st.selectbox("Exercise Induced Angina", options=["Unknown", "No", "Yes"], help="Chest pain triggered strictly by physical activity.")
        
        oldpeak_val = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=None, step=0.1, placeholder="Type value if known...", help="Measures changes in your heart's electrical pattern during peak exercise compared to rest, which can indicate if the heart muscle is getting enough oxygen. Normal is usually 0.0.")
            
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Unknown", 1, 2, 3], format_func=lambda x: "Unknown" if x == "Unknown" else {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x], help="The angle of the ST segment during peak exercise. Flat or downsloping can indicate heart issues.")
        ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", options=["Unknown", 0, 1, 2, 3], help="An X-ray test (fluoroscopy) that uses dye to see if your major blood vessels are blocked. 0 means clear.")
        thal = st.selectbox("Thalassemia", options=["Unknown", 3, 6, 7], format_func=lambda x: "Unknown" if x == "Unknown" else {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x], help="A blood disorder. 'Normal' means normal blood flow. 'Fixed defect' means no blood flow in some parts. 'Reversable defect' means blood flow is observed but not normal.")
        
        st.markdown("---")
        st.markdown("**Blood & Heart Failure Lab Results:**")
        anaemia = st.selectbox("Anaemia (Decrease of red blood cells)?", options=["Unknown", "No", "Yes"], help="A condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues.")
        platelets_val = st.number_input("Platelets (kiloplatelets/mL)", min_value=10000.0, max_value=900000.0, value=None, step=1000.0, placeholder="Type value if known...", help="Platelets are blood cells that help your body form clots to stop bleeding.")
            
        cpk_val = st.number_input("Creatinine Phosphokinase (CPK enzyme level) (mcg/L)", min_value=10.0, max_value=10000.0, value=None, placeholder="Type value if known...", help="Creatinine phosphokinase is an enzyme in the body. High levels can indicate stress or injury to the heart muscle.")
            
        ef_val = st.number_input("Ejection Fraction (%)", min_value=10.0, max_value=90.0, value=None, placeholder="Type value if known...", help="Percentage of blood leaving the heart at each contraction. A normal heart's ejection fraction is between 50 and 70 percent.")
            
        sc_val = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=None, step=0.1, placeholder="Type value if known...", help="Serum creatinine is a waste product in your blood. Higher levels can indicate poor kidney function, which affects heart health.")
            
        ss_val = st.number_input("Serum Sodium (mEq/L)", min_value=100.0, max_value=160.0, value=None, step=1.0, placeholder="Type value if known...", help="Serum sodium measures the amount of sodium in your blood. Abnormal levels can be linked to heart failure.")

    submit_button = st.button("Analyze Risk", type="primary", use_container_width=True)

with col2:
    if submit_button:
        # Helper to parse unknowns to np.nan so HGB can handle them natively
        def parse_opt(val):
            return np.nan if val == "Unknown" or val is None else val
        
        # Helper for binary Yes/No -> 1/0
        def parse_bin(val):
            if val == "Unknown" or val is None: return np.nan
            return 1 if val == "Yes" else 0

        # Construct inputs dict safely
        inputs_dict = {
            'age': age,
            'trestbps': trestbps,
            'chol': parse_opt(chol_val),
            'thalach': thalach,
            'oldpeak': parse_opt(oldpeak_val),
            'ca': parse_opt(ca),
            'anaemia': parse_bin(anaemia),
            'creatinine_phosphokinase': parse_opt(cpk_val),
            'diabetes': parse_bin(diabetes),
            'ejection_fraction': parse_opt(ef_val),
            'high_blood_pressure': parse_bin(high_blood_pressure),
            'platelets': parse_opt(platelets_val),
            'serum_creatinine': parse_opt(sc_val),
            'serum_sodium': parse_opt(ss_val),
            'smoking': 1 if smoking == "Yes" else 0,  # From lifestyle
            'sex': 1 if sex == "Male" else 0,
            'cp': parse_opt(cp),
            'fbs': 1 if fbs == "Yes" else 0,
            'restecg': parse_opt(restecg),
            'exang': parse_bin(exang),
            'slope': parse_opt(slope),
            'thal': parse_opt(thal)
        }

        # Create DataFrame ensuring column order matches training data exactly
        input_df = pd.DataFrame([inputs_dict])[feature_names]

        # Preprocess (StandardScaler + OrdinalEncoder)
        processed_input = preprocessor.transform(input_df)

        # Predict
        risk_score = float(model.predict_proba(processed_input)[0][1])

        st.header("Analysis Results")

        # Risk Gauge (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            number = {'suffix': "%", 'font': {'size': 50}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if risk_score < 0.3:
            st.success("You are in the **Low Risk** category. Keep up the good work!")
        elif risk_score < 0.7:
            st.warning("You are in the **Moderate Risk** category. Consider reviewing your lifestyle habits.")
        else:
            st.error("You are in the **High Risk** category. Please consult a healthcare professional.")

        # SHAP Explanation
        st.subheader("What drove this prediction?")
        st.markdown("This chart breaks down the most significant factors in your profile. Bars pointing to the right (red) increased your risk, while bars pointing to the left (green) decreased it.")
        
        with st.spinner("Analyzing driving factors..."):
            shap_values = explainer.shap_values(processed_input)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] # positive class
            else:
                shap_vals = shap_values[0]

            # Human readable feature names
            readable_names = {
                'age': 'Age', 'sex': 'Gender', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure',
                'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
                'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina', 'oldpeak': 'ST Depression (Exercise)',
                'slope': 'ST Segment Slope', 'ca': 'Major Vessels Blocked', 'thal': 'Thalassemia',
                'anaemia': 'Anaemia', 'creatinine_phosphokinase': 'CPK Enzyme Level', 'diabetes': 'Diabetes',
                'ejection_fraction': 'Ejection Fraction (Heart Pumping)', 'high_blood_pressure': 'High Blood Pressure',
                'platelets': 'Platelets Count', 'serum_creatinine': 'Serum Creatinine (Kidney)',
                'serum_sodium': 'Serum Sodium', 'smoking': 'Smoking Habit'
            }

            feature_impacts = {readable_names.get(feature_names[i], feature_names[i]): shap_vals[i] for i in range(len(feature_names))}
            sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            top_factors_dict = {k: v for k, v in sorted_impacts}

            # Plotly Horizontal Bar Chart
            factors = [k for k, _ in reversed(sorted_impacts)]
            impacts = [v for _, v in reversed(sorted_impacts)]
            colors = ['#ff4b4b' if v > 0 else '#00cc96' for v in impacts]
            
            fig_bar = go.Figure(go.Bar(
                x=impacts,
                y=factors,
                orientation='h',
                marker_color=colors,
                text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in impacts],
                textposition='auto'
            ))
            fig_bar.update_layout(
                title="Top 6 Contributing Factors",
                xaxis_title="Impact on Risk Score",
                yaxis_title="",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Groq Explanation
        st.subheader("AI Health Assistant Explanation")
        with st.spinner("Generating personalized explanation..."):
            # Original inputs sent to LLM
            original_inputs = inputs_dict.copy()
            
            # Since chol_val can be np.nan now directly from inputs_dict, LLM handles it as 'nan'
            original_inputs['cigarettes_per_day'] = cigarettes_per_day if smoking == "Yes" else 0
            original_inputs['alcohol'] = alcohol
            original_inputs['alcohol_freq'] = alcohol_freq if alcohol == "Yes" else "None"
            
            explanation = generate_explanation(risk_score, top_factors_dict, original_inputs)
            st.markdown(explanation)

        # Save to DB
        with st.spinner("Saving to history..."):
            # Make sure we don't pass np.nan directly if Supabase strict JSON doesn't like it.
            # db.py already has safe_int and safe_float handling though.
            inserted = insert_prediction(st.session_state.user_id, original_inputs, risk_score, top_factors_dict, explanation)
            if not inserted:
                st.caption("Note: Database not connected or failed to save history.")

    # History Tab
    st.divider()
    with st.expander("View Patient History"):
        history = get_user_history(st.session_state.user_id)
        if history:
            for record in history:
                st.markdown(f"**Date:** {record['created_at'][:10]} | **Risk Score:** {record['risk_score']:.1%}")
        else:
            st.write("No previous history found.")
