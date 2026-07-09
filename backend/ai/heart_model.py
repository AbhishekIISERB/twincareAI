"""Heart disease risk prediction model.

PyTorch MLP trained on health biomarker features. This is the AMD GPU centerpiece:
- Trained on ROCm in the Jupyter notebook
- Inference runs on whatever device is available (GPU preferred, CPU fallback)

If no trained model file exists, falls back to a rule-based risk calculation
so the demo always works.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import settings

logger = logging.getLogger(__name__)


# Feature configuration — order matters, must match training
FEATURE_NAMES = [
    "age",
    "sex",                  # 0=female, 1=male
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "fasting_glucose",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
]

# Population medians for imputation of missing values
POPULATION_MEDIANS = {
    "age": 50.0,
    "sex": 0.5,
    "total_cholesterol": 200.0,
    "ldl_cholesterol": 100.0,
    "hdl_cholesterol": 55.0,
    "triglycerides": 130.0,
    "fasting_glucose": 95.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "bmi": 25.0,
}

# Feature normalization parameters (from training data statistics)
FEATURE_MEANS = {
    "age": 50.08, "sex": 0.56, "total_cholesterol": 200.48,
    "ldl_cholesterol": 115.93, "hdl_cholesterol": 51.75, "triglycerides": 140.63,
    "fasting_glucose": 105.51, "systolic_bp": 129.30, "diastolic_bp": 82.15, "bmi": 26.03,
}

FEATURE_STDS = {
    "age": 11.97, "sex": 0.50, "total_cholesterol": 38.99,
    "ldl_cholesterol": 34.43, "hdl_cholesterol": 14.68, "triglycerides": 57.55,
    "fasting_glucose": 24.40, "systolic_bp": 17.71, "diastolic_bp": 10.12, "bmi": 4.87,
}


class HeartDiseaseNet(nn.Module):
    """3-layer MLP for heart disease risk prediction."""

    def __init__(self, input_size: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Global model instance
_model: HeartDiseaseNet | None = None
_model_loaded: bool = False


def load_model() -> HeartDiseaseNet | None:
    """Load the trained heart disease model from disk."""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    model_path = Path(settings.HEART_MODEL_PATH)
    if model_path.exists():
        try:
            model = HeartDiseaseNet()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            _model = model
            _model_loaded = True
            logger.info(f"Heart disease model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _model_loaded = True
            return None
    else:
        logger.warning(f"No model file found at {model_path} — using rule-based fallback")
        _model_loaded = True
        return None


def prepare_features(biomarkers: dict, user_profile: dict) -> tuple[np.ndarray, dict]:
    """
    Prepare feature vector from biomarkers and user profile.
    
    Returns:
        Tuple of (normalized feature array, dict of raw feature values used)
    """
    from datetime import date

    # Extract raw feature values
    raw_features = {}

    # Age from date of birth
    dob = user_profile.get("date_of_birth")
    if dob:
        if isinstance(dob, str):
            dob = date.fromisoformat(dob)
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        raw_features["age"] = float(age)

    # Sex from gender
    gender = user_profile.get("gender", "")
    if gender:
        raw_features["sex"] = 1.0 if gender.lower() == "male" else 0.0

    # Map biomarkers to features
    biomarker_to_feature = {
        "total_cholesterol": "total_cholesterol",
        "ldl_cholesterol": "ldl_cholesterol",
        "hdl_cholesterol": "hdl_cholesterol",
        "triglycerides": "triglycerides",
        "fasting_glucose": "fasting_glucose",
    }

    for biomarker_name, feature_name in biomarker_to_feature.items():
        if biomarker_name in biomarkers:
            bio = biomarkers[biomarker_name]
            value = bio.get("value") if isinstance(bio, dict) else bio
            if value is not None:
                raw_features[feature_name] = float(value)

    # Build feature vector with imputation
    feature_vector = []
    features_used = {}
    for fname in FEATURE_NAMES:
        if fname in raw_features:
            val = raw_features[fname]
            features_used[fname] = {"value": val, "source": "actual"}
        else:
            val = POPULATION_MEDIANS[fname]
            features_used[fname] = {"value": val, "source": "imputed"}

        # Normalize
        normalized = (val - FEATURE_MEANS[fname]) / max(FEATURE_STDS[fname], 1e-6)
        feature_vector.append(normalized)

    return np.array(feature_vector, dtype=np.float32), features_used


def predict_risk(biomarkers: dict, user_profile: dict) -> dict:
    """
    Predict heart disease risk.
    
    Returns dict with: probability, confidence, risk_level, features_used
    """
    features, features_used = prepare_features(biomarkers, user_profile)

    model = load_model()

    if model is not None:
        # PyTorch model inference
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            probability = model(input_tensor).item()

        # Confidence based on how many features are actual vs imputed
        actual_count = sum(1 for v in features_used.values() if v["source"] == "actual")
        confidence = min(0.95, 0.5 + (actual_count / len(FEATURE_NAMES)) * 0.45)

    else:
        # Rule-based fallback
        probability, confidence = _rule_based_prediction(features_used)

    # Determine risk level
    if probability < 0.2:
        risk_level = "low"
    elif probability < 0.4:
        risk_level = "moderate"
    elif probability < 0.6:
        risk_level = "high"
    else:
        risk_level = "very_high"

    return {
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "features_used": features_used,
    }


def _rule_based_prediction(features_used: dict) -> tuple[float, float]:
    """
    Rule-based fallback prediction when no trained model is available.
    Uses clinical risk factor guidelines.
    """
    risk_score = 0.0
    max_score = 0.0

    # Age risk (men > 45, women > 55 have higher risk)
    age = features_used.get("age", {}).get("value", 50)
    sex = features_used.get("sex", {}).get("value", 0.5)
    if age > 55:
        risk_score += 2
    elif age > 45:
        risk_score += 1
    max_score += 2

    # Cholesterol risk
    chol = features_used.get("total_cholesterol", {}).get("value", 200)
    if chol > 240:
        risk_score += 2
    elif chol > 200:
        risk_score += 1
    max_score += 2

    # LDL risk
    ldl = features_used.get("ldl_cholesterol", {}).get("value", 100)
    if ldl > 160:
        risk_score += 2
    elif ldl > 130:
        risk_score += 1
    max_score += 2

    # HDL protection (higher is better)
    hdl = features_used.get("hdl_cholesterol", {}).get("value", 55)
    if hdl < 40:
        risk_score += 2
    elif hdl < 50:
        risk_score += 1
    max_score += 2

    # Triglycerides
    tg = features_used.get("triglycerides", {}).get("value", 130)
    if tg > 200:
        risk_score += 1.5
    elif tg > 150:
        risk_score += 0.5
    max_score += 1.5

    # Blood sugar
    glucose = features_used.get("fasting_glucose", {}).get("value", 95)
    if glucose > 126:
        risk_score += 2
    elif glucose > 100:
        risk_score += 1
    max_score += 2

    # Blood pressure
    sbp = features_used.get("systolic_bp", {}).get("value", 120)
    if sbp > 140:
        risk_score += 2
    elif sbp > 130:
        risk_score += 1
    max_score += 2

    # BMI
    bmi = features_used.get("bmi", {}).get("value", 25)
    if bmi > 30:
        risk_score += 1.5
    elif bmi > 25:
        risk_score += 0.5
    max_score += 1.5

    probability = risk_score / max(max_score, 1)
    # Lower confidence for rule-based
    actual_count = sum(1 for v in features_used.values() if v["source"] == "actual")
    confidence = min(0.75, 0.3 + (actual_count / len(FEATURE_NAMES)) * 0.45)

    return probability, confidence


def get_feature_names() -> list[str]:
    """Return the ordered feature names used by the model."""
    return FEATURE_NAMES.copy()
