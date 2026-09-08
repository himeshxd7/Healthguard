import os
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key or url == "your_url_here" or key == "your_key_here":
        return None
    return create_client(url, key)


def insert_prediction(user_id, inputs_dict, risk_score,
                      top_factors, explanation):
    supabase = get_supabase_client()
    if not supabase:
        return False

    import math
    def safe_int(val, default=0):
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return default
            return int(f)
        except (ValueError, TypeError):
            return default

    def safe_float(val, default=0.0):
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    data = {
        "user_id": user_id,
        "age": safe_int(inputs_dict.get('age')),
        "sex": inputs_dict.get('sex'),
        "chest_pain_type": str(inputs_dict.get('cp', '')),
        "resting_bp": safe_int(inputs_dict.get('trestbps')),
        "cholesterol": safe_int(inputs_dict.get('chol')),
        "fasting_blood_sugar": inputs_dict.get('fbs') == "Yes",
        "resting_ecg": str(inputs_dict.get('restecg', '')),
        "max_heart_rate": safe_int(inputs_dict.get('thalach')),
        "exercise_angina": inputs_dict.get('exang') == "Yes",
        "oldpeak": safe_float(inputs_dict.get('oldpeak')),
        "smoking": inputs_dict.get('smoking', 'No'),
        "cigarettes_per_day": safe_int(inputs_dict.get('cigarettes_per_day')),
        "alcohol": inputs_dict.get('alcohol', 'No'),
        "alcohol_freq": inputs_dict.get('alcohol_freq', 'None'),
        "anaemia": inputs_dict.get('anaemia') == 1,
        "creatinine_phosphokinase": safe_float(inputs_dict.get('creatinine_phosphokinase')),
        "diabetes": inputs_dict.get('diabetes') == 1,
        "ejection_fraction": safe_float(inputs_dict.get('ejection_fraction')),
        "high_blood_pressure": inputs_dict.get('high_blood_pressure') == 1,
        "platelets": safe_float(inputs_dict.get('platelets')),
        "serum_creatinine": safe_float(inputs_dict.get('serum_creatinine')),
        "serum_sodium": safe_float(inputs_dict.get('serum_sodium')),
        "risk_score": float(risk_score),
        "top_factors": json.dumps(top_factors),
        "explanation": explanation
    }

    try:
        supabase.table('predictions').insert(data).execute()
        return True
    except Exception as e:
        print(f"Supabase Insert Error: {e}")
        return False


def get_user_history(user_id):
    supabase = get_supabase_client()
    if not supabase:
        return []

    try:
        response = supabase.table('predictions').select(
            '*').eq('user_id', user_id).order('created_at', desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Supabase Select Error: {e}")
        return []
