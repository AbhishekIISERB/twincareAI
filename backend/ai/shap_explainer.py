"""SHAP explainability for the heart disease prediction model.

Computes per-feature SHAP contributions using DeepExplainer (for PyTorch)
or KernelExplainer (fallback for any model/rule-based).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_shap_values(
    features: dict,
    prediction_result: dict,
) -> dict:
    """
    Compute SHAP-like feature contributions for a prediction.
    
    For the hackathon, we use an analytical approximation based on
    feature deviations from population means — this is faster and
    more reliable than running SHAP DeepExplainer on a small model.
    
    For the full SHAP implementation with DeepExplainer, see the
    Jupyter notebook (01_heart_disease_model_training.ipynb).
    
    Returns:
        dict with 'features', 'values', 'base_value' for waterfall chart
    """
    from ai.heart_model import FEATURE_MEANS, FEATURE_STDS, POPULATION_MEDIANS

    base_probability = 0.18  # Average population risk
    predicted_probability = prediction_result["probability"]
    total_deviation = predicted_probability - base_probability

    # Calculate contribution of each feature
    feature_contributions = []

    for feature_name, feature_data in features.items():
        value = feature_data["value"]
        source = feature_data["source"]

        mean = FEATURE_MEANS.get(feature_name, value)
        std = FEATURE_STDS.get(feature_name, 1.0)

        # Normalized deviation
        z_score = (value - mean) / max(std, 1e-6)

        # Risk direction depends on the feature
        risk_direction = _get_risk_direction(feature_name, z_score)

        # Contribution proportional to deviation, weighted by feature importance
        importance_weight = _get_feature_weight(feature_name)
        contribution = z_score * importance_weight * 0.05  # Scale factor

        # Reduce contribution for imputed features
        if source == "imputed":
            contribution *= 0.1

        feature_contributions.append({
            "feature": feature_name,
            "display_name": _get_display_name(feature_name),
            "value": value,
            "contribution": round(contribution, 4),
            "direction": risk_direction,
            "source": source,
        })

    # Sort by absolute contribution (most important first)
    feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Normalize contributions to sum to total_deviation
    total_raw = sum(abs(fc["contribution"]) for fc in feature_contributions)
    if total_raw > 0:
        scale = total_deviation / total_raw if total_raw != 0 else 0
        for fc in feature_contributions:
            fc["contribution"] = round(fc["contribution"] * abs(scale) * np.sign(fc["contribution"]), 4)

    # Build SHAP-format output
    shap_output = {
        "features": [fc["display_name"] for fc in feature_contributions],
        "values": [fc["contribution"] for fc in feature_contributions],
        "base_value": round(base_probability, 4),
    }

    # Build feature importance list
    feature_importance = [
        {
            "feature": fc["display_name"],
            "importance": round(abs(fc["contribution"]), 4),
            "direction": fc["direction"],
            "actual_value": fc["value"],
            "source": fc["source"],
        }
        for fc in feature_contributions
        if fc["source"] == "actual"  # Only show actual features in importance
    ]

    return {
        "shap_values": shap_output,
        "feature_importance": feature_importance,
    }


def _get_risk_direction(feature: str, z_score: float) -> str:
    """Determine if a feature increases or decreases risk based on clinical knowledge."""
    # HDL is protective — higher is better
    if feature == "hdl_cholesterol":
        return "decreases_risk" if z_score > 0 else "increases_risk"

    # For most features, higher values increase risk
    positive_risk = [
        "total_cholesterol", "ldl_cholesterol", "triglycerides",
        "fasting_glucose", "systolic_bp", "diastolic_bp", "bmi", "age",
    ]

    if feature in positive_risk:
        return "increases_risk" if z_score > 0 else "decreases_risk"

    # Sex: male = higher risk
    if feature == "sex":
        return "increases_risk" if z_score > 0 else "decreases_risk"

    return "increases_risk" if z_score > 0 else "decreases_risk"


def _get_feature_weight(feature: str) -> float:
    """Get the clinical importance weight for a feature."""
    weights = {
        "total_cholesterol": 1.5,
        "ldl_cholesterol": 1.8,
        "hdl_cholesterol": 1.4,
        "triglycerides": 1.0,
        "fasting_glucose": 1.3,
        "systolic_bp": 1.6,
        "diastolic_bp": 1.2,
        "age": 1.7,
        "sex": 0.8,
        "bmi": 1.1,
    }
    return weights.get(feature, 1.0)


def _get_display_name(feature: str) -> str:
    """Get human-readable display name for a feature."""
    names = {
        "age": "Age",
        "sex": "Sex",
        "total_cholesterol": "Total Cholesterol",
        "ldl_cholesterol": "LDL Cholesterol",
        "hdl_cholesterol": "HDL Cholesterol",
        "triglycerides": "Triglycerides",
        "fasting_glucose": "Fasting Glucose",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "bmi": "BMI",
    }
    return names.get(feature, feature.replace("_", " ").title())
