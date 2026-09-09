import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# MEDICAL TEST KNOWLEDGE BASE
# Maps model features / clinical findings to specific recommended tests.
# Used to build targeted test recommendations in the LLM prompt.
# ─────────────────────────────────────────────────────────────────────────────
MEDICAL_TESTS = {
    "lipid_panel": {
        "name": "Lipid Panel (Cholesterol Profile)",
        "why": "Measures LDL ('bad') cholesterol, HDL ('good') cholesterol, and triglycerides — the primary drivers of artery blockage.",
        "how": "A simple fasting blood test. Fast for 9–12 hours before the test.",
        "approx_cost": "Rs. 200–600 at any diagnostic lab",
        "urgency": "Routine",
    },
    "ecg": {
        "name": "ECG (Electrocardiogram)",
        "why": "Records the electrical activity of your heart. Detects arrhythmias, previous heart attacks, and heart enlargement.",
        "how": "Painless, 10-minute test. Electrodes are placed on chest, arms, and legs.",
        "approx_cost": "Rs. 100–500",
        "urgency": "Routine to Urgent depending on symptoms",
    },
    "echo": {
        "name": "Echocardiogram (Heart Ultrasound)",
        "why": "Shows the structure and pumping function of your heart (ejection fraction). Detects valve problems and weakened heart muscle.",
        "how": "Ultrasound probe placed on chest. No radiation. Takes 30–60 minutes.",
        "approx_cost": "Rs. 1,500–5,000",
        "urgency": "Moderate priority",
    },
    "tmt": {
        "name": "TMT / Stress Test (Treadmill Test)",
        "why": "Checks how your heart performs under physical stress. Reveals coronary artery disease that rests scans can miss.",
        "how": "You walk/run on a treadmill while connected to an ECG. Supervised by a doctor.",
        "approx_cost": "Rs. 800–2,500",
        "urgency": "Moderate priority — do NOT attempt if you have current chest pain",
    },
    "hba1c": {
        "name": "HbA1c (Glycated Hemoglobin)",
        "why": "Shows your average blood sugar over the past 3 months. Uncontrolled diabetes is a major heart disease amplifier.",
        "how": "A simple blood test. No fasting required.",
        "approx_cost": "Rs. 200–500",
        "urgency": "Routine",
    },
    "kidney_function": {
        "name": "Kidney Function Test (Creatinine + eGFR)",
        "why": "The kidneys and heart are closely linked. Poor kidney function increases the heart's workload and risk of failure.",
        "how": "Blood test. Measures creatinine levels to estimate kidney filtration rate.",
        "approx_cost": "Rs. 150–400",
        "urgency": "Routine to Moderate",
    },
    "cbc": {
        "name": "Complete Blood Count (CBC)",
        "why": "Checks for anaemia (low red blood cells), which forces the heart to pump harder to deliver oxygen.",
        "how": "A standard blood test drawn from a vein.",
        "approx_cost": "Rs. 150–300",
        "urgency": "Routine",
    },
    "bp_monitoring": {
        "name": "24-Hour Ambulatory Blood Pressure Monitoring",
        "why": "Detects high blood pressure patterns missed by a single reading, including 'masked hypertension' during daily activity.",
        "how": "Wear a portable BP cuff for 24 hours that takes readings automatically.",
        "approx_cost": "Rs. 800–2,000",
        "urgency": "Moderate — if BP is consistently above 130/80",
    },
    "cardiologist": {
        "name": "Cardiology Consultation",
        "why": "A cardiologist can order the right combination of tests based on your specific risk profile and interpret results accurately.",
        "how": "Book an appointment with a cardiologist (DM/DNB Cardiology).",
        "approx_cost": "Rs. 300–1,500 for consultation",
        "urgency": "High — prioritise within 1–2 weeks if risk is high",
    },
}


def _build_test_recommendations(inputs_dict: dict, risk_score: float) -> list[str]:
    """
    Determine which medical tests to recommend based on missing data,
    known risk factors, and overall risk score.
    Returns a list of test keys from MEDICAL_TESTS.
    """
    tests = []
    import math

    def is_missing(val):
        try:
            return val is None or math.isnan(float(val))
        except (TypeError, ValueError):
            return val in (None, "Unknown", "")

    # Always recommend ECG for anyone using this tool
    tests.append("ecg")

    # Cholesterol missing or high
    chol = inputs_dict.get('chol')
    if is_missing(chol) or (not is_missing(chol) and float(chol) > 200):
        tests.append("lipid_panel")

    # Diabetes unknown or diagnosed
    diabetes = inputs_dict.get('diabetes')
    if is_missing(diabetes) or diabetes == 1:
        tests.append("hba1c")

    # Serum creatinine missing or elevated
    sc = inputs_dict.get('serum_creatinine')
    if is_missing(sc) or (not is_missing(sc) and float(sc) > 1.2):
        tests.append("kidney_function")

    # Anaemia unknown
    if is_missing(inputs_dict.get('anaemia')):
        tests.append("cbc")

    # Ejection fraction missing or low
    ef = inputs_dict.get('ejection_fraction')
    if is_missing(ef) or (not is_missing(ef) and float(ef) < 50):
        tests.append("echo")

    # High blood pressure known or unknown
    if is_missing(inputs_dict.get('high_blood_pressure')) or inputs_dict.get('high_blood_pressure') == 1:
        tests.append("bp_monitoring")

    # Stress test recommended for moderate/high risk without known angina
    if risk_score >= 0.3 and is_missing(inputs_dict.get('exang')):
        tests.append("tmt")

    # High risk → cardiologist referral
    if risk_score >= 0.65:
        tests.append("cardiologist")

    # Deduplicate while preserving order
    seen = set()
    unique_tests = []
    for t in tests:
        if t not in seen:
            seen.add(t)
            unique_tests.append(t)

    return unique_tests


@st.cache_data(ttl=3600, show_spinner=False)
def generate_explanation(risk_score, top_factors_dict, inputs_dict):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_key_here":
        return "⚠️ Groq API key not configured. Please set GROQ_API_KEY in your .env file."

    # ── Build test recommendations ────────────────────────────────────────────
    recommended_test_keys = _build_test_recommendations(inputs_dict, risk_score)
    test_details_block = ""
    for key in recommended_test_keys:
        t = MEDICAL_TESTS[key]
        test_details_block += (
            f"\n- **{t['name']}** | Priority: {t['urgency']}"
            f"\n  - Why: {t['why']}"
            f"\n  - How: {t['how']}"
            f"\n  - Approximate cost: {t['approx_cost']}"
        )

    # ── Determine urgency level for the prompt ────────────────────────────────
    if risk_score < 0.30:
        urgency_instruction = (
            "Urgency: LOW. Tell the user to schedule a routine check-up with their GP within 3–6 months. "
            "Frame this positively — their current profile looks relatively healthy, but monitoring is still wise."
        )
    elif risk_score < 0.65:
        urgency_instruction = (
            "Urgency: MODERATE. Advise the user to book an appointment with their GP within 2–4 weeks. "
            "Several risk factors warrant professional evaluation soon. "
            "Tell them to monitor symptoms and seek emergency care if they experience sudden chest pain, "
            "breathlessness at rest, or palpitations."
        )
    else:
        urgency_instruction = (
            "Urgency: HIGH. Strongly advise the user to consult a cardiologist within 1–2 weeks — "
            "the sooner the better. Emphasise that this is NOT a diagnosis but the risk profile is "
            "serious enough to warrant urgent professional evaluation. "
            "Make clear: if they experience sudden chest pain, left arm pain, jaw pain, severe breathlessness, "
            "or feel faint — they must CALL EMERGENCY SERVICES (India: 102/112) IMMEDIATELY."
        )

    # ── Master prompt ─────────────────────────────────────────────────────────
    prompt = f"""
You are a compassionate, senior cardiologist AI assistant helping a patient understand their heart disease risk assessment from our HealthGuard AI tool.

## PATIENT PROFILE

### Demographics & Body Metrics
- Age: {inputs_dict.get('age')} | Sex: {'Male' if inputs_dict.get('sex') == 1 else 'Female'}
- BMI: {inputs_dict.get('bmi', 'Unknown')} ({inputs_dict.get('bmi_category', 'Unknown')})
- Waist: {inputs_dict.get('waist_cm', 'Unknown')} cm | Waist-to-Height Ratio: {inputs_dict.get('whr', 'Unknown')} ({inputs_dict.get('whr_risk', 'Unknown')})

### Cardiovascular Vitals
- Resting Blood Pressure: {inputs_dict.get('trestbps', 'Unknown')} mmHg ({inputs_dict.get('bp_category', 'Unknown')})
- Max Heart Rate Achieved: {inputs_dict.get('thalach', 'Unknown')} bpm
- Estimated Max HR for age: {inputs_dict.get('max_hr_est', 'Unknown')} bpm
- Target HR training zones: {inputs_dict.get('thr_low', '?')}-{inputs_dict.get('thr_mod', '?')} bpm (moderate) | {inputs_dict.get('thr_mod', '?')}-{inputs_dict.get('thr_high', '?')} bpm (vigorous)
- Cholesterol: {inputs_dict.get('chol', 'Not provided')} mg/dL

### Diagnosed Conditions
- Diabetes: {inputs_dict.get('diabetes_detail', 'No')}
- High Blood Pressure: {'Yes' if inputs_dict.get('high_blood_pressure') == 1 else 'No' if inputs_dict.get('high_blood_pressure') == 0 else 'Unknown'}
- Known High Cholesterol: {inputs_dict.get('high_cholesterol_known', 'Never checked')}
- Anaemia: {'Yes' if inputs_dict.get('anaemia') == 1 else 'No' if inputs_dict.get('anaemia') == 0 else 'Unknown'}
- Kidney Disease: {inputs_dict.get('kidney_disease', 'Unknown')}
- Thyroid Disorder: {inputs_dict.get('thyroid_disorder', 'Unknown')}

### Current Medications
- BP Medication: {'Yes' if inputs_dict.get('on_bp_meds') else 'No'}
- Statins: {'Yes' if inputs_dict.get('on_statins') else 'No'}
- Diabetes Medication/Insulin: {'Yes' if inputs_dict.get('on_diabetes_meds') else 'No'}
- Blood Thinners: {'Yes' if inputs_dict.get('on_blood_thinners') else 'No'}

### Symptoms
- {inputs_dict.get('symptom_summary', 'None reported')}
- Ankle/Leg Swelling: {inputs_dict.get('ankle_swelling', 'No')}
- Leg Pain When Walking (Claudication): {'Yes' if inputs_dict.get('leg_pain_walking') else 'No'}

### Physical Activity
- Daily Walking: {inputs_dict.get('walking_mins_day', 0)} min/day ({inputs_dict.get('weekly_walk_mins', 0):.0f} min/week — WHO target: 150 min/week)
- Structured Exercise: {inputs_dict.get('exercise_days_week', 'Not specified')}
- Sitting Time: {inputs_dict.get('sitting_hours_day', 0)} hours/day
- Job Type: {inputs_dict.get('job_type', 'Not specified')}
- WHO Activity Compliant: {'Yes' if inputs_dict.get('activity_who_compliant') else 'No — below 150 min/week'}

### Sleep
- Duration: {inputs_dict.get('sleep_hours', 'Unknown')} hours/night
- Quality: {inputs_dict.get('sleep_quality', 'Not specified')}
- Snoring/Sleep Apnea: {inputs_dict.get('snoring', 'No')}

### Stress & Mental Health
- Stress Level: {inputs_dict.get('stress_level', 'Not specified')}
- Depression/Anxiety: {inputs_dict.get('depression_anxiety', 'No')}

### Diet
- Dietary Pattern: {inputs_dict.get('dietary_pattern', 'Not specified')}
- Fruits & Vegetables: {inputs_dict.get('veg_fruit_servings', 'Not specified')}
- Salt Intake: {inputs_dict.get('salt_intake', 'Not specified')}
- Sugary Drinks: {inputs_dict.get('sugary_drinks', 'Not specified')}
- Red Meat: {inputs_dict.get('red_meat_freq', 'Not specified')}
- Fried Food: {inputs_dict.get('fried_food_freq', 'Not specified')}

### Smoking & Alcohol
- {'Smoker: ' + str(inputs_dict.get('cigarettes_per_day',0)) + ' cigarettes/day for ' + str(inputs_dict.get('years_smoked',0)) + ' years' if inputs_dict.get('smoking') == 1 else ('Ex-smoker (quit)' if inputs_dict.get('ex_smoker') else 'Non-smoker')}
- {'Alcohol: ' + str(inputs_dict.get('alcohol_freq','None')) + ', approx. ' + str(inputs_dict.get('alcohol_units_week',0)) + ' units/week' if inputs_dict.get('alcohol') == 'Yes' else 'No alcohol'}

### Family & History
- Family History of Heart Disease: {inputs_dict.get('family_history', 'No')}
- Previous Cardiac Event (heart attack/stent/bypass): {'Yes' if inputs_dict.get('previous_heart_event') else 'No'}

## MODEL OUTPUT
The machine learning model predicted a **heart disease risk probability of {risk_score:.1%}**.
This means: out of 100 people with a similar health profile, roughly {round(risk_score * 100)} may have underlying heart disease.

## TOP RISK-DRIVING FACTORS (SHAP Analysis)
The following factors had the most influence on this specific prediction.
Positive value = increases risk | Negative value = decreases risk:
"""
    for factor, value in top_factors_dict.items():
        direction = "increases" if value > 0 else "decreases"
        prompt += f"\n- **{factor}**: {value:+.3f} ({direction} risk)"

    prompt += f"""

## YOUR TASK — Write a comprehensive, structured patient health report with these EXACT sections:

### 1. What Your Risk Score Means
Explain in plain, compassionate language what {risk_score:.1%} means. Use the "out of 100 people" analogy.
Mention the BMI ({inputs_dict.get('bmi', 'Unknown')}) and waist-to-height ratio ({inputs_dict.get('whr', 'Unknown')}) in context.
Do NOT use scary or alarmist language unnecessarily. Be honest but kind.

### 2. Your Key Risk Drivers
For each of the top 3 SHAP factors above:
- Explain what the factor is in plain language
- State whether it is high, low, or normal for this patient specifically
- Explain specifically how it influenced the risk score

### 3. Recommended Medical Tests
Recommend specific tests for THIS patient based on their exact profile. For each test below, write 2-3 patient-friendly sentences about what it involves and why it's particularly important for them:
{test_details_block}

### 4. Personalised Heart-Healthy Diet Plan
Based on their specific diet profile (dietary pattern: {inputs_dict.get('dietary_pattern', 'unknown')}, salt: {inputs_dict.get('salt_intake', 'unknown')}, fruit/veg: {inputs_dict.get('veg_fruit_servings', 'unknown')}, fried food: {inputs_dict.get('fried_food_freq', 'unknown')}, sugary drinks: {inputs_dict.get('sugary_drinks', 'unknown')}, red meat: {inputs_dict.get('red_meat_freq', 'unknown')}):

a) **Foods to Prioritise** (list 6-8 specific heart-healthy foods with a 1-line explanation for each. Reference the DASH or Mediterranean diet. Make it practical for an Indian context where relevant.)

b) **Foods to Reduce or Avoid** (list 4-6 specific foods/habits to cut down on, with a brief reason. Be direct but not preachy. Reference their specific reported habits.)

c) **Practical Daily Meal Tips** (give 2-3 concrete, realistic meal suggestions or swaps for their dietary pattern — e.g. "Replace your afternoon chai with green tea 3 days a week" or "Swap white rice for brown rice or millets at dinner".)

d) **Hydration** (specific daily water intake recommendation, especially given their BP: {inputs_dict.get('trestbps', 'Unknown')} mmHg.)

### 5. Personalised Exercise Prescription
Based on their current activity (walking: {inputs_dict.get('walking_mins_day', 0)} min/day, exercise days: {inputs_dict.get('exercise_days_week', 'none')}, sitting: {inputs_dict.get('sitting_hours_day', 0)} hrs/day):

a) **Target Heart Rate Zones** - Tell them specifically:
   - Their estimated max HR is **{inputs_dict.get('max_hr_est', 220 - (inputs_dict.get('age') or 50))} bpm**
   - For fat-burning / cardiac rehab zone: **{inputs_dict.get('thr_low', '?')}-{inputs_dict.get('thr_mod', '?')} bpm** (50-70% max HR)
   - For cardiovascular strengthening: **{inputs_dict.get('thr_mod', '?')}-{inputs_dict.get('thr_high', '?')} bpm** (70-85% max HR)
   - Tell them how to check their pulse during exercise.

b) **Aerobic Exercise Prescription** (be specific: type of activity, intensity, duration, frequency per week. Calibrate to their current level. If sedentary: start with 10 min walks. If moderately active: build to Zone 2 training.)

c) **Strength/Resistance Training** (2x/week minimum, explain why it matters for heart health - improves insulin sensitivity, reduces visceral fat. Give 2-3 beginner-friendly exercises with no equipment needed.)

d) **Breaking Sedentary Time** (given they sit {inputs_dict.get('sitting_hours_day', 0)} hours/day - give specific practical strategies: standing every 30 min, walking meetings, etc.)

e) **Sleep & Exercise Connection** (if sleep quality is "{inputs_dict.get('sleep_quality', 'unknown')}" - briefly note how exercise timing affects sleep quality, and whether morning or evening exercise is recommended for them.)

### 6. Lifestyle & Mental Health
Based specifically on their stress ({inputs_dict.get('stress_level', 'unknown')}), sleep ({inputs_dict.get('sleep_hours', 7)} hrs, {inputs_dict.get('sleep_quality', 'unknown')}), and snoring/apnea status ({inputs_dict.get('snoring', 'No')}):
- 2-3 specific stress management techniques (not just "relax more")
- Sleep hygiene tips tailored to their situation
- If snoring suggests sleep apnea: specifically mention requesting a sleep study

### 7. Next Steps & Urgency
{urgency_instruction}
List the next 5 concrete actions the patient should take in order of priority.

### Important Medical Disclaimer
End with a clear disclaimer:
- You are an AI assistant, NOT a doctor
- This is NOT a medical diagnosis
- The patient must consult qualified medical professionals for any health decisions
- In a medical emergency in India: call **102** (ambulance) or **112** (emergency)

Keep the tone: warm, medically credible, specific, and empowering. Use simple language. Use markdown headers and bullet points. Make the diet and exercise sections feel like a real, actionable health plan - not generic advice.
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
            temperature=0.4,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error connecting to Groq AI: {str(e)}\n\nPlease check your GROQ_API_KEY in .env"
