"""
utils/risk_rules.py

Rule-based clinical safety net for HealthGuard AI.

This module implements hard medical thresholds that OVERRIDE the ML model's
output in specific dangerous combinations. This ensures the app cannot
give a "Low Risk" result when clinical guidelines clearly indicate otherwise.

These rules are based on established medical guidelines:
- ACC/AHA Cardiovascular Risk Guidelines
- ESC (European Society of Cardiology) Heart Failure Guidelines
- JNC 8 Hypertension Guidelines

IMPORTANT: These rules do not replace the ML model — they act as a floor.
If the ML model gives a higher risk than the rule, the ML score is used.
"""

import math
from dataclasses import dataclass


@dataclass
class RuleResult:
    """Result from the rule-based safety net."""
    triggered: bool          # Was any rule triggered?
    risk_floor: float        # Minimum risk score enforced (0.0–1.0)
    level: str               # "LOW", "MODERATE", "HIGH"
    triggered_rules: list    # Human-readable list of triggered rules
    clinical_notes: list     # Clinical rationale for patient display


def _is_missing(val) -> bool:
    """Returns True if value is missing/unknown."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return str(val).strip().lower() in ('', 'unknown', 'none', 'nan')


def apply_clinical_rules(inputs: dict, model_risk: float) -> RuleResult:
    """
    Apply clinical rule overrides to the ML model's risk score.

    Args:
        inputs:     The full inputs_dict as built in app.py
        model_risk: The probability from the ML model (0.0–1.0)

    Returns:
        RuleResult with final risk floor and any triggered rules.
    """
    triggered_rules = []
    clinical_notes  = []
    risk_floor      = 0.0   # Start with no floor — model score is the baseline

    age             = inputs.get('age', None)
    sex             = inputs.get('sex', None)   # 1 = Male, 0 = Female
    trestbps        = inputs.get('trestbps', None)
    chol            = inputs.get('chol', None)
    ef              = inputs.get('ejection_fraction', None)
    sc              = inputs.get('serum_creatinine', None)
    cp              = inputs.get('cp', None)
    diabetes        = inputs.get('diabetes', None)
    hbp             = inputs.get('high_blood_pressure', None)
    smoking         = inputs.get('smoking', 0)
    cigarettes      = inputs.get('cigarettes_per_day', 0)
    chest_pain_yn   = inputs.get('chest_pain_yn', False)   # from new symptom form
    breathless      = inputs.get('breathlessness', False)
    family_hx       = inputs.get('family_history', 'No')
    prev_heart_evt  = inputs.get('previous_heart_event', False)

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 1: Previous heart attack / stent / bypass
    # Anyone with a documented prior cardiac event is always HIGH RISK.
    # ──────────────────────────────────────────────────────────────────────────
    if prev_heart_evt:
        risk_floor = max(risk_floor, 0.75)
        triggered_rules.append("Previous cardiac event (heart attack / stent / bypass)")
        clinical_notes.append(
            "Patients with prior heart attacks or procedures have significantly elevated "
            "risk of recurrent events (up to 20% within 5 years per AHA guidelines)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 2: Critically low Ejection Fraction
    # EF < 35% = Class III/IV Heart Failure by ESC criteria.
    # ──────────────────────────────────────────────────────────────────────────
    if not _is_missing(ef) and float(ef) < 35:
        risk_floor = max(risk_floor, 0.80)
        triggered_rules.append(f"Critically low ejection fraction ({float(ef):.0f}%)")
        clinical_notes.append(
            f"An ejection fraction of {float(ef):.0f}% indicates severely reduced heart pumping function "
            "(normal is 50–70%). This meets criteria for Heart Failure with Reduced Ejection Fraction "
            "(HFrEF) per ESC guidelines and requires urgent cardiology evaluation."
        )
    elif not _is_missing(ef) and float(ef) < 50:
        risk_floor = max(risk_floor, 0.55)
        triggered_rules.append(f"Mildly reduced ejection fraction ({float(ef):.0f}%)")
        clinical_notes.append(
            f"An ejection fraction of {float(ef):.0f}% is below the normal range of 50–70%, "
            "suggesting mildly reduced heart pumping function that warrants monitoring."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 3: Severely elevated Serum Creatinine
    # sc > 2.0 mg/dL = Stage 3+ CKD, significantly increases cardiac risk.
    # ──────────────────────────────────────────────────────────────────────────
    if not _is_missing(sc) and float(sc) > 2.0:
        risk_floor = max(risk_floor, 0.60)
        triggered_rules.append(f"Severely elevated serum creatinine ({float(sc):.1f} mg/dL)")
        clinical_notes.append(
            f"Serum creatinine of {float(sc):.1f} mg/dL indicates significantly impaired kidney function "
            "(CKD Stage 3+). Chronic kidney disease is a major independent risk factor for cardiovascular "
            "disease and heart failure."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 4: Hypertensive Crisis / Stage 2 Hypertension
    # BP >= 160 mmHg systolic = Stage 2 hypertension (JNC 8).
    # ──────────────────────────────────────────────────────────────────────────
    if not _is_missing(trestbps) and float(trestbps) >= 160:
        risk_floor = max(risk_floor, 0.55)
        triggered_rules.append(f"Severely elevated resting BP ({float(trestbps):.0f} mmHg)")
        clinical_notes.append(
            f"A resting blood pressure of {float(trestbps):.0f} mmHg is Stage 2 Hypertension (JNC 8). "
            "Sustained BP at this level significantly increases risk of heart attack, stroke, and heart failure."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 5: Classic high-risk demographic + symptom profile
    # Male > 55, or Female > 65, with chest pain = always at least Moderate.
    # ──────────────────────────────────────────────────────────────────────────
    if not _is_missing(age) and not _is_missing(sex):
        age_val = float(age)
        sex_val = int(sex)
        is_high_risk_age = (sex_val == 1 and age_val > 55) or (sex_val == 0 and age_val > 65)
        has_symptoms = chest_pain_yn or (not _is_missing(cp) and int(float(cp)) in [1, 2, 3])
        if is_high_risk_age and has_symptoms:
            risk_floor = max(risk_floor, 0.45)
            sex_label = "Male" if sex_val == 1 else "Female"
            triggered_rules.append(
                f"High-risk age-sex-symptom profile ({sex_label}, {age_val:.0f} years, with chest symptoms)"
            )
            clinical_notes.append(
                f"Men over 55 and women over 65 presenting with chest pain or discomfort represent a "
                "high-risk demographic for coronary artery disease per ACC/AHA guidelines."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 6: Combined metabolic syndrome markers
    # Diabetes + High BP + High Cholesterol = very high CV risk per Framingham.
    # ──────────────────────────────────────────────────────────────────────────
    has_diabetes = diabetes == 1
    has_hbp      = hbp == 1
    has_hi_chol  = not _is_missing(chol) and float(chol) > 240

    metabolic_flags = sum([has_diabetes, has_hbp, has_hi_chol])
    if metabolic_flags >= 2:
        risk_floor = max(risk_floor, 0.50)
        flags_desc = []
        if has_diabetes: flags_desc.append("Diabetes")
        if has_hbp:      flags_desc.append("Hypertension")
        if has_hi_chol:  flags_desc.append(f"High Cholesterol ({float(chol):.0f} mg/dL)")
        triggered_rules.append(f"Multiple metabolic syndrome markers: {', '.join(flags_desc)}")
        clinical_notes.append(
            "Presence of multiple metabolic syndrome components (diabetes, hypertension, dyslipidemia) "
            "synergistically multiplies cardiovascular risk beyond the individual effects of each factor "
            "(Framingham Heart Study)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 7: Heavy smoking
    # > 20 cigarettes/day = major independent risk factor.
    # ──────────────────────────────────────────────────────────────────────────
    if smoking == 1 and not _is_missing(cigarettes) and float(cigarettes) >= 20:
        risk_floor = max(risk_floor, 0.40)
        triggered_rules.append(f"Heavy smoking ({float(cigarettes):.0f} cigarettes/day)")
        clinical_notes.append(
            f"Smoking {float(cigarettes):.0f} cigarettes per day more than doubles the risk of heart disease. "
            "Heavy smokers have a 70% higher rate of coronary heart disease than non-smokers (CDC / WHO)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # RULE 8: Positive family history + high-risk age
    # First-degree relative with heart disease before 55 (M) or 65 (F).
    # ──────────────────────────────────────────────────────────────────────────
    if family_hx == "Yes" and not _is_missing(age) and float(age) > 40:
        risk_floor = max(risk_floor, 0.35)
        triggered_rules.append("Positive family history of heart disease")
        clinical_notes.append(
            "A first-degree family history of premature heart disease (parent or sibling) approximately "
            "doubles your personal risk, even when other risk factors are controlled (AHA 2019)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL DETERMINATION
    # The effective risk is the maximum of the model score and the rule floor.
    # ──────────────────────────────────────────────────────────────────────────
    effective_risk = max(model_risk, risk_floor)

    if effective_risk >= 0.65:
        level = "HIGH"
    elif effective_risk >= 0.30:
        level = "MODERATE"
    else:
        level = "LOW"

    return RuleResult(
        triggered=len(triggered_rules) > 0,
        risk_floor=risk_floor,
        level=level,
        triggered_rules=triggered_rules,
        clinical_notes=clinical_notes,
    )
