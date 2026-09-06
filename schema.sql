create table predictions (
    id serial primary key,
    user_id text not null,
    created_at timestamp default now(),
    age int,
    sex text,
    chest_pain_type text,
    resting_bp int,
    cholesterol int,
    fasting_blood_sugar boolean,
    resting_ecg text,
    max_heart_rate int,
    exercise_angina boolean,
    oldpeak float,
    risk_score float,
    top_factors text,       -- SHAP output, stored as JSON string
    explanation text        -- Groq-generated explanation
);
