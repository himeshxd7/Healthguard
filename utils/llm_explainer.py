import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


@st.cache_data(ttl=3600, show_spinner=False)
def generate_explanation(risk_score, top_factors_dict, inputs_dict):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_key_here":
        return "Groq API key not configured. Please set GROQ_API_KEY in your .env file. The AI explanation feature requires a valid Groq API key."

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
        You are a helpful AI health assistant analyzing a heart disease risk prediction from a machine learning model.

        PATIENT INPUTS & LIFESTYLE:
        - Age: {inputs_dict.get('age')}
        - Sex: {inputs_dict.get('sex')}
        - Cholesterol: {inputs_dict.get('chol')}
        - Max Heart Rate: {inputs_dict.get('thalach')}
        - Resting BP: {inputs_dict.get('trestbps')}
        - Smoking: {inputs_dict.get('smoking')} ({inputs_dict.get('cigarettes_per_day')} cigarettes/day)
        - Alcohol Consumption: {inputs_dict.get('alcohol')} (Frequency: {inputs_dict.get('alcohol_freq')})

        MODEL OUTPUT:
        The machine learning model predicted a risk probability score of {risk_score:.2%} for heart disease.

        TOP CONTRIBUTING FACTORS (SHAP Analysis):
        The following factors had the most impact on this specific prediction (positive value = increases risk, negative = decreases risk):
        """

        for factor, value in top_factors_dict.items():
            prompt += f"\n        - {factor}: {value:.4f}"

        prompt += """

        INSTRUCTIONS:
        1. Explain what this risk score means in simple terms.
        2. Explain how the top contributing factors influenced the model's decision based on the SHAP values provided.
        3. Provide 3-4 general, non-prescriptive lifestyle suggestions. Make sure to tailor these specifically based on the provided lifestyle inputs (Smoking and Alcohol frequency).
        4. Conclude with a clear disclaimer that you are an AI, not a doctor, and this is NOT a diagnostic tool.

        Keep the explanation compassionate, clear, and easy to read. Use markdown formatting.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.8-27b",
            temperature=0.5,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"
