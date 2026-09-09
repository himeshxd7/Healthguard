import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import math
import plotly.graph_objects as go
from utils.llm_explainer import generate_explanation
from utils.db import insert_prediction, get_user_history
from utils.risk_rules import apply_clinical_rules
import uuid

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthGuard AI — Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark-mode health theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}

/* Top header card */
.header-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2444 100%);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* Form section cards */
.form-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* Risk level cards */
.risk-low    { background: linear-gradient(135deg, #065f46, #047857); border: 1px solid #10b981; border-radius: 12px; padding: 20px; }
.risk-mod    { background: linear-gradient(135deg, #78350f, #92400e); border: 1px solid #f59e0b; border-radius: 12px; padding: 20px; }
.risk-high   { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #ef4444; border-radius: 12px; padding: 20px; }

/* Test recommendation cards */
.test-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,179,237,0.25);
    border-left: 4px solid #63b3ed;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Progress bar styling */
.step-indicator {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    padding: 8px 16px;
    display: inline-block;
    font-size: 0.85rem;
    color: #90cdf4;
    margin-bottom: 12px;
    border: 1px solid rgba(99,179,237,0.2);
}

/* Clinical note box */
.clinical-note {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #fcd34d;
}

/* Disclaimer box */
.disclaimer {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 0.85rem;
    color: #fca5a5;
}

/* Streamlit button overrides */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    transform: translateY(-1px);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* Number input / selectbox */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'result_ready' not in st.session_state:
    st.session_state.result_ready = False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        model         = joblib.load(os.path.join(models_dir, 'heart_disease_model.joblib'))
        preprocessor  = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))

        # Extract the underlying HGB estimator for SHAP (unwrap FrozenEstimator if needed)
        raw_est = model.calibrated_classifiers_[0].estimator
        if hasattr(raw_est, 'estimator'):
            raw_est = raw_est.estimator
        explainer = shap.TreeExplainer(raw_est)

        # Load decision threshold (tuned on val set)
        thresh_path = os.path.join(models_dir, 'decision_threshold.joblib')
        threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5

        return model, preprocessor, explainer, feature_names, threshold
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, 0.5


model, preprocessor, explainer, feature_names, decision_threshold = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
  <h1 style="margin:0; font-size:2rem; font-weight:700; color:#90cdf4;">🫀 HealthGuard AI</h1>
  <p style="margin:6px 0 0 0; color:#a0aec0; font-size:1rem;">
    Heart Disease Risk Assessment &amp; AI-Powered Health Guidance
  </p>
  <p style="margin:8px 0 0 0; font-size:0.82rem; color:#fc8181;">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational awareness only.
    It is NOT a medical diagnosis and does NOT replace a qualified doctor.
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Models not found. Please run: `python src/data_prep.py` then `python src/train_hgb.py`")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def parse_opt(val):
    """Convert Unknown/None to np.nan for HGB native NaN handling."""
    if val == "Unknown" or val is None:
        return np.nan
    try:
        f = float(val)
        return np.nan if math.isnan(f) else f
    except (TypeError, ValueError):
        return np.nan


def parse_bin(val):
    """Convert Yes/No/Unknown to 1/0/NaN."""
    if val in ("Unknown", None):
        return np.nan
    return 1.0 if val == "Yes" else 0.0


def yn_to_bool(val) -> bool:
    return val == "Yes"


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STEP WIZARD LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_STEPS = 4
step = st.session_state.step

if not st.session_state.result_ready:
    # ── Progress indicator ─────────────────────────────────────────────────
    progress_pct = int((step - 1) / TOTAL_STEPS * 100)
    step_labels = ["About You", "Symptoms & Lifestyle", "Medical History", "Lab Results"]
    st.markdown(f"""
    <div class="step-indicator">
        Step {step} of {TOTAL_STEPS}: <strong>{step_labels[step-1]}</strong>
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress_pct / 100)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — About You
    # ══════════════════════════════════════════════════════════════════════════
    if step == 1:
        st.subheader("👤 About You")
        st.caption("Basic information helps personalise your risk assessment.")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Your Age", min_value=18, max_value=100, value=50,
                                  help="Your age in years.")
            sex = st.selectbox("Biological Sex", ["Male", "Female"],
                               help="Biological sex at birth affects cardiovascular risk.")
        with col2:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170,
                                     help="Used to calculate your BMI.")
            weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70,
                                     help="Used to calculate your BMI.")

        # ── Waist circumference ────────────────────────────────────────────────
        waist_cm = st.number_input(
            "Waist Circumference (cm)",
            min_value=40, max_value=200, value=85,
            help="Measure around your belly button with a tape. Visceral (belly) fat is the most dangerous type for heart health. "
                 "Risk thresholds: Men >94 cm = elevated, >102 cm = high. Women >80 cm = elevated, >88 cm = high."
        )

        # ── Derived metrics ────────────────────────────────────────────────────
        bmi = weight / ((height / 100) ** 2)
        bmi_category = (
            "Underweight" if bmi < 18.5 else
            "Normal weight" if bmi < 25 else
            "Overweight" if bmi < 30 else "Obese"
        )
        # Waist-to-height ratio — stronger CV predictor than BMI
        whr = waist_cm / height
        whr_risk = "High" if whr > 0.5 else "Acceptable"
        # Target heart rate zones (Karvonen method approximate)
        age_val = age
        max_hr_est = 220 - age_val
        thr_low  = int(max_hr_est * 0.50)
        thr_mod  = int(max_hr_est * 0.70)
        thr_high = int(max_hr_est * 0.85)
        # Ideal weight range (Hamwi formula)
        ideal_low  = round((height - 100) * 0.9 * 0.9, 1)
        ideal_high = round((height - 100) * 0.9 * 1.1, 1)

        # Display calculated metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMI", f"{bmi:.1f}", bmi_category,
                  delta_color="inverse" if bmi >= 25 else "normal")
        m2.metric("Waist-to-Height Ratio", f"{whr:.2f}", whr_risk,
                  delta_color="inverse" if whr > 0.5 else "normal")
        m3.metric("Estimated Max HR", f"{max_hr_est} bpm",
                  f"Target zone: {thr_low}–{thr_mod} bpm")
        m4.metric("Ideal Weight Range", f"{ideal_low}–{ideal_high} kg")

        if bmi >= 30:
            st.warning("⚠️ Obesity (BMI ≥ 30) is a major independent cardiovascular risk factor.")
        elif bmi >= 25:
            st.info("ℹ️ Overweight BMI. Even a 5–10% weight reduction significantly reduces heart risk.")

        if whr > 0.5:
            st.warning("⚠️ Your waist-to-height ratio > 0.5 indicates central obesity — a stronger CV risk marker than BMI alone.")

        st.divider()
        family_history = st.selectbox(
            "Has a parent or sibling been diagnosed with heart disease, stroke, or sudden cardiac death before age 60?",
            ["No", "Yes", "Not sure"],
            help="A first-degree family history significantly increases your personal risk (doubles it per AHA 2019)."
        )

        previous_heart_event = st.selectbox(
            "Have you ever had a heart attack, stroke, stent, or bypass surgery?",
            ["No", "Yes"],
            help="A previous cardiac event places you in a high-risk category automatically."
        )

        st.session_state.form_data.update({
            'age': age, 'sex': sex, 'height': height, 'weight': weight,
            'bmi': round(bmi, 1), 'bmi_category': bmi_category,
            'waist_cm': waist_cm, 'whr': round(whr, 2), 'whr_risk': whr_risk,
            'max_hr_est': max_hr_est,
            'thr_low': thr_low, 'thr_mod': thr_mod, 'thr_high': thr_high,
            'family_history': family_history,
            'previous_heart_event': yn_to_bool(previous_heart_event),
        })

        col_back, col_next = st.columns([1, 3])
        with col_next:
            if st.button("Next: Symptoms & Lifestyle →", type="primary"):
                st.session_state.step = 2
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Symptoms & Lifestyle
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 2:
        st.subheader("💓 Symptoms & Lifestyle")
        st.caption("Answer honestly — these questions are the most important part of your assessment.")

        # ── SYMPTOMS ──────────────────────────────────────────────────────────
        st.markdown("### 🫀 Symptoms")
        st.caption("These are classic cardiovascular warning signs. Even mild symptoms matter.")

        col1, col2 = st.columns(2)
        with col1:
            chest_pain_yn = st.selectbox(
                "Chest pain, pressure, or tightness?",
                ["No", "Yes — only during exercise/exertion", "Yes — at rest or anytime"],
                help="Angina is chest pain caused by reduced blood flow to the heart. Exercise-triggered pain is classic angina. Rest pain is more serious."
            )
            breathlessness = st.selectbox(
                "Unusual shortness of breath?",
                ["No", "Only during heavy exertion", "During normal activities (walking, stairs)", "At rest or lying flat"],
                help="Breathlessness during mild activity or at rest can indicate heart failure or poor cardiac output."
            )
            ankle_swelling = st.selectbox(
                "Swelling in your ankles, feet, or legs?",
                ["No", "Mild (shoes feel tighter by evening)", "Significant (visible puffiness)"],
                help="Ankle oedema is a classic sign of heart failure — fluid accumulates when the heart can't pump efficiently."
            )
        with col2:
            palpitations = st.selectbox(
                "Heart racing, pounding, or irregular beats?",
                ["No", "Occasionally (< once/week)", "Frequently (multiple times/week)"],
                help="Palpitations can indicate arrhythmia. Frequent palpitations with dizziness warrant an ECG."
            )
            leg_pain_walking = st.selectbox(
                "Leg pain or cramping when walking that stops with rest?",
                ["No", "Yes"],
                help="Intermittent claudication (leg cramps when walking) can indicate Peripheral Artery Disease — plaque in the leg arteries, which strongly predicts coronary artery disease."
            )
            fatigue = st.selectbox(
                "Unusual fatigue or exhaustion?",
                ["No", "Occasionally", "Frequently — gets in the way of daily life"],
                help="Persistent unexplained fatigue, especially in women, can be an early and often overlooked heart disease symptom."
            )

        # Map chest pain to UCI cp encoding
        if chest_pain_yn == "No":
            cp_val = 4   # Asymptomatic
        elif "exertion" in chest_pain_yn or "exercise" in chest_pain_yn:
            cp_val = 1   # Typical Angina
        else:
            cp_val = 3   # Non-anginal / Rest pain

        # Build symptom summary for LLM
        symptom_parts = []
        if chest_pain_yn != "No":        symptom_parts.append(f"Chest pain: {chest_pain_yn}")
        if breathlessness != "No":       symptom_parts.append(f"Breathlessness: {breathlessness}")
        if ankle_swelling != "No":       symptom_parts.append(f"Ankle swelling: {ankle_swelling}")
        if palpitations != "No":         symptom_parts.append(f"Palpitations: {palpitations}")
        if leg_pain_walking == "Yes":    symptom_parts.append("Intermittent claudication (leg pain on walking)")
        if fatigue != "No":              symptom_parts.append(f"Fatigue: {fatigue}")
        symptom_summary = "; ".join(symptom_parts) if symptom_parts else "None reported"

        st.divider()

        # ── PHYSICAL ACTIVITY (detailed) ──────────────────────────────────────
        st.markdown("### 🏃 Physical Activity")
        st.caption("Sedentary time is now considered an independent heart disease risk factor — even if you exercise.")

        col1, col2 = st.columns(2)
        with col1:
            walking_mins_day = st.number_input(
                "Average minutes of walking per day",
                min_value=0, max_value=480, value=30,
                help="Include all walking — to work, around home, etc. WHO recommends ≥ 150 min/week of moderate activity."
            )
            exercise_days_week = st.selectbox(
                "Days per week of deliberate exercise (gym, running, cycling, swimming...)",
                ["0 days — not at all", "1–2 days", "3–4 days", "5+ days"],
                help="Any structured exercise that raises your heart rate counts. Target: 3–5 days for cardiovascular benefit."
            )
        with col2:
            sitting_hours_day = st.number_input(
                "Hours of sitting/sedentary time per day (work + leisure)",
                min_value=0, max_value=20, value=8,
                help="Sitting >8 hours/day is independently linked to 147% higher risk of cardiovascular events (Biswas et al., Annals of Internal Medicine)."
            )
            job_type = st.selectbox(
                "What type of work/job do you have?",
                ["Primarily desk/sedentary (office, computer, driving)",
                 "Mix of sitting and standing/walking",
                 "Physically active (construction, farming, healthcare, manual labour)"],
                help="Occupational physical activity contributes meaningfully to total daily movement."
            )

        # Compute activity score for LLM context
        weekly_walk_mins = walking_mins_day * 7
        activity_who_compliant = weekly_walk_mins >= 150
        sitting_risk = sitting_hours_day >= 8

        if sitting_hours_day >= 10:
            st.error(f"⚠️ {sitting_hours_day} hours of sitting/day is a significant independent CV risk factor.")
        elif sitting_hours_day >= 8:
            st.warning(f"ℹ️ {sitting_hours_day} hours sitting/day — try to break it up with 5-min walks every hour.")

        if weekly_walk_mins >= 150:
            st.success(f"✅ {weekly_walk_mins:.0f} min/week walking meets the WHO physical activity guideline (≥150 min/week).")
        else:
            st.info(f"Your estimated weekly walking: **{weekly_walk_mins:.0f} min/week**. WHO recommends ≥ 150 min/week.")

        st.divider()

        # ── SLEEP ─────────────────────────────────────────────────────────────
        st.markdown("### 😴 Sleep")
        st.caption("Poor sleep quality is an underappreciated but independent cardiovascular risk factor.")

        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.number_input(
                "Average hours of sleep per night",
                min_value=2.0, max_value=14.0, value=7.0, step=0.5,
                help="Both too little (<6h) and too much (>9h) are linked to higher CV risk. Optimal: 7–8 hours."
            )
            sleep_quality = st.selectbox(
                "How would you rate your sleep quality?",
                ["Good — wake up feeling refreshed",
                 "Fair — sometimes tired during the day",
                 "Poor — frequently tired or unrefreshed"],
                help="Restorative sleep is when your cardiovascular system recovers and repairs. Poor sleep raises cortisol and blood pressure."
            )
        with col2:
            snoring = st.selectbox(
                "Do you snore loudly, or have you been told you stop breathing during sleep?",
                ["No", "Yes — I snore (mild/moderate)", "Yes — told I stop breathing or gasp"],
                help="Loud snoring with breathing pauses is a key symptom of Obstructive Sleep Apnea (OSA). OSA causes repeated oxygen drops during sleep, significantly stressing the heart and raising BP. It is associated with a 140% higher risk of developing heart failure."
            )

        if snoring == "Yes — told I stop breathing or gasp":
            st.error("⚠️ Suspected Sleep Apnea: This is a significant cardiac risk factor. Mention this to your doctor — a sleep study can diagnose it and CPAP treatment dramatically reduces heart risk.")
        elif snoring == "Yes — I snore (mild/moderate)":
            st.warning("ℹ️ Snoring can be a sign of mild sleep apnea. If you feel unrefreshed or your partner notices pauses in breathing, discuss a sleep study with your doctor.")

        if sleep_hours < 6:
            st.warning(f"⚠️ Sleeping only {sleep_hours}h/night is linked to 48% higher risk of developing heart disease (Harvard Sleep Study).")

        st.divider()

        # ── STRESS ────────────────────────────────────────────────────────────
        st.markdown("### 🧠 Stress & Mental Health")

        col1, col2 = st.columns(2)
        with col1:
            stress_level = st.selectbox(
                "How would you rate your typical daily stress level?",
                ["Low — generally calm and relaxed",
                 "Moderate — some pressure but manageable",
                 "High — frequently stressed or anxious",
                 "Very high — constant severe stress"],
                help="Chronic psychological stress raises cortisol and adrenaline, elevating blood pressure, promoting inflammation, and accelerating atherosclerosis."
            )
        with col2:
            depression_anxiety = st.selectbox(
                "Have you experienced depression or significant anxiety in the past year?",
                ["No", "Mild", "Moderate to severe"],
                help="Depression is an independent risk factor for heart disease — as significant as smoking. It also makes people less likely to follow a healthy lifestyle."
            )

        st.divider()

        # ── DIET (detailed) ───────────────────────────────────────────────────
        st.markdown("### 🥗 Diet & Nutrition")
        st.caption("Specific dietary habits are stronger predictors than a general 'healthy/unhealthy' category.")

        col1, col2 = st.columns(2)
        with col1:
            veg_fruit_servings = st.selectbox(
                "Fruits and vegetables per day (servings)",
                ["< 1 serving", "1–2 servings", "3–4 servings", "5+ servings (WHO target)"],
                help="Each additional serving of fruit/vegetables reduces heart disease mortality by ~4% (Aune et al., Int J Epidemiology). 5+ servings/day is the WHO target."
            )
            salt_intake = st.selectbox(
                "How would you describe your salt/sodium intake?",
                ["Low — rarely add salt, avoid salty foods",
                 "Moderate — some salt in cooking, occasional salty snacks",
                 "High — regularly eat salty foods, processed foods, pickles, chips"],
                help="High sodium intake raises blood pressure. WHO recommends < 5g/day (about 1 teaspoon). Most Indians consume 8–11g/day."
            )
            sugary_drinks = st.selectbox(
                "Sugary drinks (cola, juices, chai with lots of sugar) per day?",
                ["None or rarely", "1–2 per day", "3+ per day"],
                help="Each sugary drink per day increases diabetes risk by 26% and cardiovascular risk significantly."
            )
        with col2:
            red_meat_freq = st.selectbox(
                "Red meat consumption (mutton, beef, pork, processed meat)?",
                ["Rarely or never", "1–2 times/week", "3–4 times/week", "Daily"],
                help="Processed red meat is a Group 1 carcinogen and cardiovascular risk factor. Unprocessed red meat in moderation (1–2x/week) is less concerning."
            )
            fried_food_freq = st.selectbox(
                "Fried/deep-fried foods (samosa, pakora, chips, fried snacks)?",
                ["Rarely", "A few times/week", "Daily"],
                help="Trans fats from deep-frying directly raise LDL cholesterol and lower HDL, strongly promoting atherosclerosis."
            )
            dietary_pattern = st.selectbox(
                "Which best describes your overall dietary pattern?",
                ["Mostly plant-based / vegetarian",
                 "Mediterranean-style (fish, olive oil, vegetables, whole grains)",
                 "Mixed non-vegetarian (varied meat + vegetables)",
                 "High processed / fast food heavy"],
                help="The Mediterranean diet has the strongest evidence for cardiovascular protection (PREDIMED trial, 30% relative risk reduction)."
            )

        st.divider()

        # ── SMOKING & ALCOHOL ─────────────────────────────────────────────────
        st.markdown("### 🚬 Smoking & Alcohol")

        col1, col2 = st.columns(2)
        with col1:
            smoking = st.selectbox("Do you currently smoke?", ["No", "Yes", "Ex-smoker (quit)"],
                                   help="Smoking doubles heart disease risk. Even 1 cigarette/day significantly raises risk. Ex-smokers' risk drops substantially within 1 year of quitting.")
            cigarettes_per_day = 0
            years_smoked = 0
            if smoking == "Yes":
                cigarettes_per_day = st.number_input("How many cigarettes per day?",
                                                      min_value=1, max_value=100, value=10)
                years_smoked = st.number_input("For how many years?", min_value=1, max_value=70, value=10)
        with col2:
            alcohol = st.selectbox("Do you regularly consume alcohol?", ["No", "Yes"])
            alcohol_freq = "None"
            alcohol_units_week = 0
            if alcohol == "Yes":
                alcohol_freq = st.selectbox("How often?",
                                            ["Occasionally (< 1x/week)", "Weekly (1–3x/week)", "Daily"])
                alcohol_units_week = st.number_input(
                    "Approx. units per week? (1 unit = 1 beer / 1 small peg / 1 glass wine)",
                    min_value=1, max_value=100, value=5
                )

        st.session_state.form_data.update({
            # Symptoms
            'chest_pain_yn': chest_pain_yn != "No",
            'cp': cp_val,
            'breathlessness': breathlessness,
            'ankle_swelling': ankle_swelling,
            'palpitations': palpitations,
            'leg_pain_walking': leg_pain_walking == "Yes",
            'fatigue': fatigue,
            'symptom_summary': symptom_summary,
            # Activity
            'walking_mins_day': walking_mins_day,
            'exercise_days_week': exercise_days_week,
            'sitting_hours_day': sitting_hours_day,
            'job_type': job_type.split('(')[0].strip(),
            'weekly_walk_mins': weekly_walk_mins,
            'activity_who_compliant': activity_who_compliant,
            'sitting_risk': sitting_risk,
            'activity_level': exercise_days_week,
            # Sleep
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality.split(' — ')[0],
            'snoring': snoring,
            # Stress
            'stress_level': stress_level.split(' — ')[0],
            'depression_anxiety': depression_anxiety,
            # Diet
            'veg_fruit_servings': veg_fruit_servings,
            'salt_intake': salt_intake.split(' — ')[0],
            'sugary_drinks': sugary_drinks,
            'red_meat_freq': red_meat_freq,
            'fried_food_freq': fried_food_freq,
            'dietary_pattern': dietary_pattern,
            # Smoking/Alcohol
            'smoking': 1 if smoking == "Yes" else 0,
            'ex_smoker': smoking == "Ex-smoker (quit)",
            'cigarettes_per_day': cigarettes_per_day,
            'years_smoked': years_smoked,
            'alcohol': alcohol,
            'alcohol_freq': alcohol_freq,
            'alcohol_units_week': alcohol_units_week,
        })

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Next: Medical History →", type="primary"):
                st.session_state.step = 3
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Medical History & Basic Vitals
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3:
        st.subheader("🏥 Medical History & Vitals")
        st.caption("Diagnosed conditions and easily self-measured vitals from a home BP monitor, smartwatch, or your last doctor's visit.")

        # ── DIAGNOSED CONDITIONS ──────────────────────────────────────────────
        st.markdown("### 📋 Known Diagnoses")

        col1, col2 = st.columns(2)
        with col1:
            high_blood_pressure = st.selectbox(
                "High Blood Pressure (Hypertension)?  📌",
                ["No", "Yes", "Unknown"],
                help="Hypertension (BP ≥ 130/80) is the single biggest modifiable risk factor for heart attack and stroke. If you're on BP medication, select 'Yes'."
            )
            diabetes = st.selectbox(
                "Diabetes (Type 1 or Type 2)?  📌",
                ["No", "Yes — Type 1", "Yes — Type 2", "Prediabetes", "Unknown"],
                help="Diabetics have 2–4x higher cardiovascular risk. Even prediabetes significantly raises risk."
            )
            high_cholesterol_known = st.selectbox(
                "Ever been told you have high cholesterol?",
                ["No", "Yes — on medication (statins)", "Yes — not on medication", "Never checked"],
                help="High LDL cholesterol is a primary driver of atherosclerosis (plaque build-up in arteries)."
            )
        with col2:
            anaemia = st.selectbox(
                "Anaemia (low red blood cells / haemoglobin)?",
                ["No", "Yes", "Unknown"],
                help="Anaemia forces the heart to pump harder to deliver oxygen. It also worsens outcomes in existing heart disease."
            )
            kidney_disease = st.selectbox(
                "Chronic Kidney Disease (CKD) or poor kidney function?",
                ["No", "Yes", "Unknown"],
                help="The kidneys and heart are deeply linked. CKD is an independent and strong predictor of cardiovascular disease."
            )
            thyroid_disorder = st.selectbox(
                "Thyroid disorder (hypothyroidism or hyperthyroidism)?",
                ["No", "Yes — Hypothyroid (underactive)", "Yes — Hyperthyroid (overactive)", "Unknown"],
                help="Hypothyroidism raises LDL and blood pressure. Hyperthyroidism can cause arrhythmias. Both increase cardiovascular risk."
            )

        # Parse diabetes — map all diabetes types to binary for model
        diabetes_bin = np.nan
        if "Yes" in diabetes or diabetes == "Prediabetes":
            diabetes_bin = 1.0
        elif diabetes == "No":
            diabetes_bin = 0.0

        # ── EXERCISE-TRIGGERED SYMPTOMS ────────────────────────────────────────
        st.divider()
        st.markdown("### 💊 Medications")
        st.caption("Current medications provide important clinical context.")

        col1, col2 = st.columns(2)
        with col1:
            on_bp_meds = st.selectbox("Are you on blood pressure medication?",
                                      ["No", "Yes"],
                                      help="BP medications lower readings — your current BP with medication still reflects your risk category.")
            on_statins = st.selectbox("Are you on cholesterol-lowering medication (statins like Atorvastatin, Rosuvastatin)?",
                                      ["No", "Yes"],
                                      help="Statins lower LDL cholesterol and are prescribed when cholesterol or CV risk is elevated.")
        with col2:
            on_diabetes_meds = st.selectbox("Are you on diabetes medication or insulin?",
                                            ["No", "Yes"],
                                            help="Presence of diabetes medication confirms diabetes diagnosis.")
            on_blood_thinners = st.selectbox("Are you on blood thinners (Aspirin, Warfarin, Clopidogrel, etc.)?",
                                             ["No", "Yes"],
                                             help="Blood thinners are prescribed to prevent clots after a cardiac event or for atrial fibrillation.")

        exang_yn = st.selectbox(
            "Does physical exertion trigger your chest pain or breathlessness?",
            ["No", "Yes", "Unknown"],
            help="Exercise-induced angina is chest pain or breathlessness specifically triggered by physical effort and relieved by rest. This is a key predictor in the model."
        )

        st.divider()
        st.markdown("### 📡 Vitals")
        st.caption("From your BP monitor, smartwatch, fitness tracker, or last doctor's visit.")

        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input(
                "Resting Systolic Blood Pressure (mmHg)",
                min_value=70, max_value=250, value=120,
                help="The top number in your BP reading (e.g. 120 in 120/80). Normal: < 120. Elevated: 120–129. Stage 1 HBP: 130–139. Stage 2: ≥ 140."
            )
            # BP category display
            bp_cat = ("Normal" if trestbps < 120 else
                      "Elevated" if trestbps < 130 else
                      "Stage 1 Hypertension" if trestbps < 140 else
                      "Stage 2 Hypertension" if trestbps < 180 else
                      "Hypertensive Crisis")
            bp_color = "green" if trestbps < 120 else "orange" if trestbps < 140 else "red"
            st.markdown(f"<span style='color:{bp_color};font-size:0.85rem;font-weight:600'>BP Category: {bp_cat}</span>",
                        unsafe_allow_html=True)
        with col2:
            resting_hr = st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=30, max_value=200, value=72,
                help="Count your pulse at your wrist for 60 seconds when calm. Normal: 60–100 bpm. Athletes can be 40–60. High resting HR (>80) is independently linked to CV mortality."
            )
            hr_cat = ("Bradycardia (<60)" if resting_hr < 60 else
                      "Normal" if resting_hr <= 80 else
                      "Slightly elevated" if resting_hr <= 100 else "Tachycardia (>100)")
            hr_color = "orange" if resting_hr < 50 or resting_hr > 100 else "green"
            st.markdown(f"<span style='color:{hr_color};font-size:0.85rem;font-weight:600'>HR Category: {hr_cat}</span>",
                        unsafe_allow_html=True)
        with col3:
            thalach_input = st.number_input(
                "Max Heart Rate Achieved (bpm)",
                min_value=60, max_value=250, value=int(st.session_state.form_data.get('max_hr_est', 170)),
                help="Your highest heart rate during peak exercise (from fitness tracker, treadmill, or gym equipment). If unknown, your estimated maximum is pre-filled."
            )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL? (measured after 8+ hours of no food)",
            ["No", "Yes", "Unknown"],
            help="Fasting blood sugar > 120 mg/dL suggests diabetes or prediabetes. Normal is < 100 mg/dL. Prediabetes: 100–125. Diabetes: ≥ 126."
        )

        st.session_state.form_data.update({
            'high_blood_pressure': parse_bin(high_blood_pressure),
            'diabetes': diabetes_bin,
            'diabetes_detail': diabetes,
            'high_cholesterol_known': high_cholesterol_known,
            'anaemia': parse_bin(anaemia),
            'kidney_disease': kidney_disease,
            'thyroid_disorder': thyroid_disorder,
            'on_bp_meds': on_bp_meds == "Yes",
            'on_statins': on_statins == "Yes",
            'on_diabetes_meds': on_diabetes_meds == "Yes",
            'on_blood_thinners': on_blood_thinners == "Yes",
            'exang': parse_bin(exang_yn),
            'trestbps': trestbps,
            'bp_category': bp_cat,
            'resting_hr': resting_hr,
            'thalach': thalach_input,
            'fbs': 0.0 if fbs == "No" else (1.0 if fbs == "Yes" else np.nan),
        })

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                st.session_state.step = 2
                st.rerun()
        with col_next:
            if st.button("Next: Lab Results →", type="primary"):
                st.session_state.step = 4
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Lab Results (Optional)
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 4:
        st.subheader("🔬 Lab Results")
        st.markdown("""
        <p style="color:#a0aec0; font-size:0.92rem;">
        These are <strong>completely optional</strong>. If you have recent blood test results,
        entering them significantly improves the accuracy of your assessment.
        Leave blank or skip if you don't have them — the model still works without them.
        </p>
        """, unsafe_allow_html=True)

        st.info("💡 Tip: These values are usually on your lab report from your last blood test / hospital visit.")

        col1, col2 = st.columns(2)
        with col1:
            chol_val = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=0.0, max_value=700.0, value=None,
                placeholder="e.g. 200",
                help="Normal: < 200 mg/dL. Borderline high: 200–239. High: ≥ 240."
            )
            ejection_fraction = st.number_input(
                "Ejection Fraction (%)",
                min_value=0.0, max_value=100.0, value=None,
                placeholder="e.g. 60",
                help="How much blood your heart pumps per beat. Normal: 50–70%. "
                     "Found on your echocardiogram report."
            )
            serum_creatinine = st.number_input(
                "Serum Creatinine (mg/dL)",
                min_value=0.0, max_value=20.0, value=None, step=0.1,
                placeholder="e.g. 1.0",
                help="Kidney function marker. Normal: 0.6–1.2 mg/dL. Higher = worse kidney function."
            )

        with col2:
            platelets_val = st.number_input(
                "Platelet Count (kiloplatelets/mL)",
                min_value=0.0, max_value=1000000.0, value=None, step=1000.0,
                placeholder="e.g. 250000",
                help="Normal: 150,000–400,000 kiloplatelets/mL."
            )
            serum_sodium = st.number_input(
                "Serum Sodium (mEq/L)",
                min_value=100.0, max_value=180.0, value=None, step=1.0,
                placeholder="e.g. 137",
                help="Normal: 135–145 mEq/L. Low sodium can indicate heart failure."
            )
            cpk_val = st.number_input(
                "CPK / Creatinine Phosphokinase (mcg/L)",
                min_value=0.0, max_value=20000.0, value=None,
                placeholder="e.g. 582",
                help="Enzyme released when heart muscle is damaged. Normal: < 200 mcg/L."
            )

        st.divider()

        # ── Advanced optional clinical fields (collapsed by default) ────────
        with st.expander("🏥 Advanced Clinical Results (Specialist only — leave if unsure)", expanded=False):
            st.caption("These require specialist tests. Only fill if you have the actual report.")
            col1, col2 = st.columns(2)
            with col1:
                oldpeak_val = st.number_input(
                    "ST Depression (Exercise ECG)",
                    min_value=0.0, max_value=10.0, value=None, step=0.1,
                    placeholder="e.g. 1.5",
                    help="From a Treadmill Test / Stress ECG report. Normal: 0.0"
                )
                slope_val = st.selectbox(
                    "ST Segment Slope",
                    ["Unknown", 1, 2, 3],
                    format_func=lambda x: "Unknown" if x == "Unknown" else
                    {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
                    help="From a Treadmill Test report. Flat/Downsloping indicates higher risk."
                )
            with col2:
                ca_val = st.selectbox(
                    "Major Vessels Blocked (Fluoroscopy)",
                    ["Unknown", 0, 1, 2, 3],
                    help="From a cardiac fluoroscopy/angiography report. 0 = no blockages."
                )
                restecg_val = st.selectbox(
                    "Resting ECG Result",
                    ["Unknown", 0, 1, 2],
                    format_func=lambda x: "Unknown" if x == "Unknown" else
                    {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x],
                    help="From an ECG report read by a doctor."
                )
                thal_val = st.selectbox(
                    "Thalassemia",
                    ["Unknown", 3, 6, 7],
                    format_func=lambda x: "Unknown" if x == "Unknown" else
                    {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x],
                    help="From a nuclear stress test or cardiac MRI report."
                )

        # Save optional clinical fields
        oldpeak_v  = parse_opt(oldpeak_val) if 'oldpeak_val' in dir() else np.nan
        slope_v    = parse_opt(slope_val)   if 'slope_val'   in dir() else np.nan
        ca_v       = parse_opt(ca_val)      if 'ca_val'      in dir() else np.nan
        restecg_v  = parse_opt(restecg_val) if 'restecg_val' in dir() else np.nan
        thal_v     = parse_opt(thal_val)    if 'thal_val'    in dir() else np.nan

        st.session_state.form_data.update({
            'chol': chol_val,
            'ejection_fraction': ejection_fraction,
            'serum_creatinine': serum_creatinine,
            'platelets': platelets_val,
            'serum_sodium': serum_sodium,
            'creatinine_phosphokinase': cpk_val,
            'oldpeak': oldpeak_v,
            'slope': slope_v,
            'ca': ca_v,
            'restecg': restecg_v,
            'thal': thal_v,
        })

        col_back, col_submit = st.columns(2)
        with col_back:
            if st.button("← Back"):
                st.session_state.step = 3
                st.rerun()
        with col_submit:
            analyze_clicked = st.button("🔍 Analyse My Heart Risk", type="primary")

        if analyze_clicked:
            st.session_state.result_ready = True
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result_ready:
    fd = st.session_state.form_data

    # ── Build model input dict ───────────────────────────────────────────────
    inputs_dict = {
        'age':       fd.get('age'),
        'sex':       1 if fd.get('sex') == "Male" else 0,
        'cp':        parse_opt(fd.get('cp', np.nan)),
        'trestbps':  fd.get('trestbps'),
        'chol':      parse_opt(fd.get('chol')),
        'fbs':       fd.get('fbs', np.nan),
        'restecg':   fd.get('restecg', np.nan),
        'thalach':   fd.get('thalach'),
        'exang':     fd.get('exang', np.nan),
        'oldpeak':   fd.get('oldpeak', np.nan),
        'slope':     fd.get('slope', np.nan),
        'ca':        fd.get('ca', np.nan),
        'thal':      fd.get('thal', np.nan),
        # LLM context fields (not in model — passed to Groq for richer explanation)
        'smoking':                  fd.get('smoking', 0),
        'ex_smoker':                fd.get('ex_smoker', False),
        'cigarettes_per_day':       fd.get('cigarettes_per_day', 0),
        'years_smoked':             fd.get('years_smoked', 0),
        'alcohol':                  fd.get('alcohol', 'No'),
        'alcohol_freq':             fd.get('alcohol_freq', 'None'),
        'alcohol_units_week':       fd.get('alcohol_units_week', 0),
        'activity_level':           fd.get('activity_level', 'Not specified'),
        'walking_mins_day':         fd.get('walking_mins_day', 0),
        'exercise_days_week':       fd.get('exercise_days_week', '0 days'),
        'sitting_hours_day':        fd.get('sitting_hours_day', 0),
        'job_type':                 fd.get('job_type', 'Not specified'),
        'weekly_walk_mins':         fd.get('weekly_walk_mins', 0),
        'activity_who_compliant':   fd.get('activity_who_compliant', False),
        'sitting_risk':             fd.get('sitting_risk', False),
        'sleep_hours':              fd.get('sleep_hours', 7),
        'sleep_quality':            fd.get('sleep_quality', 'Not specified'),
        'snoring':                  fd.get('snoring', 'No'),
        'stress_level':             fd.get('stress_level', 'Not specified'),
        'depression_anxiety':       fd.get('depression_anxiety', 'No'),
        'veg_fruit_servings':       fd.get('veg_fruit_servings', 'Not specified'),
        'salt_intake':              fd.get('salt_intake', 'Not specified'),
        'sugary_drinks':            fd.get('sugary_drinks', 'Not specified'),
        'red_meat_freq':            fd.get('red_meat_freq', 'Not specified'),
        'fried_food_freq':          fd.get('fried_food_freq', 'Not specified'),
        'dietary_pattern':          fd.get('dietary_pattern', 'Not specified'),
        'family_history':           fd.get('family_history', 'No'),
        'previous_heart_event':     fd.get('previous_heart_event', False),
        'symptom_summary':          fd.get('symptom_summary', 'None reported'),
        'ankle_swelling':           fd.get('ankle_swelling', 'No'),
        'leg_pain_walking':         fd.get('leg_pain_walking', False),
        'bmi':                      fd.get('bmi'),
        'bmi_category':             fd.get('bmi_category'),
        'waist_cm':                 fd.get('waist_cm'),
        'whr':                      fd.get('whr'),
        'whr_risk':                 fd.get('whr_risk'),
        'thr_low':                  fd.get('thr_low'),
        'thr_mod':                  fd.get('thr_mod'),
        'thr_high':                 fd.get('thr_high'),
        'diabetes':                 fd.get('diabetes', np.nan),
        'diabetes_detail':          fd.get('diabetes_detail', 'No'),
        'high_blood_pressure':      fd.get('high_blood_pressure', np.nan),
        'bp_category':              fd.get('bp_category', 'Unknown'),
        'high_cholesterol_known':   fd.get('high_cholesterol_known', 'Never checked'),
        'anaemia':                  fd.get('anaemia', np.nan),
        'kidney_disease':           fd.get('kidney_disease', 'Unknown'),
        'thyroid_disorder':         fd.get('thyroid_disorder', 'Unknown'),
        'on_bp_meds':               fd.get('on_bp_meds', False),
        'on_statins':               fd.get('on_statins', False),
        'on_diabetes_meds':         fd.get('on_diabetes_meds', False),
        'on_blood_thinners':        fd.get('on_blood_thinners', False),
        'ejection_fraction':        parse_opt(fd.get('ejection_fraction')),
        'serum_creatinine':         parse_opt(fd.get('serum_creatinine')),
        'platelets':                parse_opt(fd.get('platelets')),
        'serum_sodium':             parse_opt(fd.get('serum_sodium')),
        'creatinine_phosphokinase': parse_opt(fd.get('creatinine_phosphokinase')),
    }

    # ── Run model ────────────────────────────────────────────────────────────
    model_input = {k: inputs_dict[k] for k in feature_names}
    input_df     = pd.DataFrame([model_input])[feature_names]
    processed    = preprocessor.transform(input_df)
    model_risk   = float(model.predict_proba(processed)[0][1])

    # ── Apply clinical rules safety net ──────────────────────────────────────
    rule_result  = apply_clinical_rules(inputs_dict, model_risk)
    risk_score   = max(model_risk, rule_result.risk_floor)

    # ── Risk level ───────────────────────────────────────────────────────────
    if risk_score < 0.30:
        risk_level = "LOW"
        risk_color = "#10b981"
        risk_card_class = "risk-low"
        risk_emoji = "✅"
        risk_msg   = "Your profile looks relatively healthy. Keep up the good work and maintain regular check-ups."
    elif risk_score < 0.65:
        risk_level = "MODERATE"
        risk_color = "#f59e0b"
        risk_card_class = "risk-mod"
        risk_emoji = "⚠️"
        risk_msg   = "Several risk factors have been identified. A consultation with your GP within 2–4 weeks is advisable."
    else:
        risk_level = "HIGH"
        risk_color = "#ef4444"
        risk_card_class = "risk-high"
        risk_emoji = "🚨"
        risk_msg   = "Your profile shows significant risk factors. Please consult a cardiologist as soon as possible."

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS LAYOUT
    # ─────────────────────────────────────────────────────────────────────────
    col_new, _ = st.columns([2, 1])
    with col_new:
        if st.button("← Start New Assessment"):
            st.session_state.result_ready = False
            st.session_state.step = 1
            st.session_state.form_data = {}
            st.rerun()

    st.markdown("---")

    # ── Risk Score section ───────────────────────────────────────────────────
    gauge_col, card_col = st.columns([1, 1])

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_score * 100,
            number = {'suffix': "%", 'font': {'size': 52, 'color': risk_color}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title  = {'text': "Heart Disease Risk Score", 'font': {'size': 18, 'color': '#a0aec0'}},
            gauge  = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0aec0",
                         'tickfont': {'color': '#a0aec0'}},
                'bar':  {'color': risk_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30],  'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [65, 100],'color': "rgba(239, 68, 68, 0.2)"},
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with card_col:
        st.markdown(f"""
        <div class="{risk_card_class}" style="height: 260px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">{risk_emoji}</div>
            <div style="font-size: 1.6rem; font-weight: 700; margin-bottom: 8px;">{risk_level} RISK</div>
            <div style="font-size: 1rem; opacity: 0.9; line-height: 1.5;">{risk_msg}</div>
            <div style="margin-top: 16px; font-size: 0.85rem; opacity: 0.7;">
                ML Model Score: {model_risk:.1%}
                {"&nbsp;&nbsp;|&nbsp;&nbsp;Clinical rules applied" if rule_result.triggered else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Clinical Rule Alerts ─────────────────────────────────────────────────
    if rule_result.triggered:
        st.markdown("#### ⚕️ Clinical Safety Alerts")
        st.caption("The following clinical findings triggered safety thresholds based on established medical guidelines:")
        for rule, note in zip(rule_result.triggered_rules, rule_result.clinical_notes):
            st.markdown(f"""
            <div class="clinical-note">
                <strong>⚡ {rule}</strong><br/>
                <span style="font-size:0.85rem; opacity:0.85;">{note}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP Feature Importance ──────────────────────────────────────────────
    st.subheader("📊 What Drove This Prediction?")
    st.caption("Bars pointing right (red) increased your risk. Bars pointing left (green) reduced it.")

    with st.spinner("Analysing contributing factors..."):
        shap_values = explainer.shap_values(processed)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        readable_names = {
            'age': 'Age', 'sex': 'Sex (Male)', 'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
            'oldpeak': 'ST Depression (Exercise)', 'slope': 'ST Segment Slope',
            'ca': 'Major Vessels Blocked', 'thal': 'Thalassemia',
        }

        feature_impacts = {
            readable_names.get(feature_names[i], feature_names[i]): shap_vals[i]
            for i in range(len(feature_names))
        }
        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        top_factors_dict = {k: v for k, v in sorted_impacts[:6]}

        factors = [k for k, _ in reversed(sorted_impacts)]
        impacts  = [v for _, v in reversed(sorted_impacts)]
        colors   = ['#ef4444' if v > 0 else '#10b981' for v in impacts]

        fig_bar = go.Figure(go.Bar(
            x=impacts, y=factors, orientation='h',
            marker_color=colors,
            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in impacts],
            textposition='auto',
            textfont=dict(color='white', size=11)
        ))
        fig_bar.update_layout(
            title="Top Contributing Factors (SHAP Analysis)",
            title_font=dict(color='#a0aec0', size=15),
            xaxis_title="Impact on Risk Score",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.2)',
                       tickfont=dict(color='#a0aec0'), gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(tickfont=dict(color='#e2e8f0'), gridcolor='rgba(255,255,255,0.05)'),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── AI Explanation ───────────────────────────────────────────────────────
    st.subheader("🤖 AI Health Assessment & Recommendations")
    with st.spinner("Generating personalised AI assessment (this takes 10–20 seconds)..."):
        explanation = generate_explanation(risk_score, top_factors_dict, inputs_dict)
        st.markdown(explanation)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> HealthGuard AI is an educational tool only.
        It uses a machine learning model trained on publicly available datasets and is NOT a medical device.
        Results are NOT a diagnosis and should NOT be used to make medical decisions without consulting
        a qualified healthcare professional. In a medical emergency, call <strong>102</strong> (ambulance)
        or <strong>112</strong> (emergency) in India.
    </div>
    """, unsafe_allow_html=True)

    # ── Save to DB ───────────────────────────────────────────────────────────
    inserted = insert_prediction(
        st.session_state.user_id, inputs_dict, risk_score, top_factors_dict, explanation
    )
    if not inserted:
        st.caption("📝 Note: Database not connected — history not saved.")

    st.markdown("---")

    # ── Print button ─────────────────────────────────────────────────────────
    st.markdown("""
    <button onclick="window.print()"
        style="background: linear-gradient(135deg, #2d3748, #1a202c);
               color: #a0aec0; border: 1px solid rgba(255,255,255,0.15);
               border-radius: 8px; padding: 10px 20px; cursor: pointer;
               font-size: 0.9rem; margin-top: 8px;">
        🖨️ Print / Save as PDF
    </button>
    """, unsafe_allow_html=True)

    # ── History ──────────────────────────────────────────────────────────────
    with st.expander("📋 View Previous Assessments"):
        history = get_user_history(st.session_state.user_id)
        if history:
            for record in history:
                score = record.get('risk_score', 0)
                level = "HIGH" if score >= 0.65 else "MODERATE" if score >= 0.3 else "LOW"
                st.markdown(f"**{record['created_at'][:10]}** — Risk: `{score:.1%}` ({level})")
        else:
            st.write("No previous assessments found in this session.")
