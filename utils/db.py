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

    data = {
        "user_id": user_id,
        "age": int(inputs_dict.get('age', 0)),
        "sex": inputs_dict.get('sex'),
        "chest_pain_type": inputs_dict.get('cp'),
        "resting_bp": int(inputs_dict.get('trestbps', 0)),
        "cholesterol": int(inputs_dict.get('chol', 0)),
        "fasting_blood_sugar": inputs_dict.get('fbs') == "Yes",
        "resting_ecg": inputs_dict.get('restecg'),
        "max_heart_rate": int(inputs_dict.get('thalach', 0)),
        "exercise_angina": inputs_dict.get('exang') == "Yes",
        "oldpeak": float(inputs_dict.get('oldpeak', 0.0)),
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
