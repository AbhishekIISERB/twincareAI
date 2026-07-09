"""Biomarker reference ranges and classification logic.

Maps common biomarker names to their normal reference ranges and organ system groupings.
Used for status determination (normal/low/high/critical) and organ score calculation.
"""

from dataclasses import dataclass


@dataclass
class BiomarkerRange:
    """Reference range for a biomarker."""
    low: float
    high: float
    unit: str
    critical_low: float | None = None
    critical_high: float | None = None
    organ_system: str = "general"
    display_name: str = ""


# Comprehensive biomarker reference ranges
BIOMARKER_RANGES: dict[str, BiomarkerRange] = {
    # Lipid Panel — Heart
    "total_cholesterol": BiomarkerRange(
        low=125, high=200, unit="mg/dL", critical_high=300,
        organ_system="heart", display_name="Total Cholesterol"
    ),
    "ldl_cholesterol": BiomarkerRange(
        low=0, high=100, unit="mg/dL", critical_high=190,
        organ_system="heart", display_name="LDL Cholesterol"
    ),
    "hdl_cholesterol": BiomarkerRange(
        low=40, high=100, unit="mg/dL", critical_low=20,
        organ_system="heart", display_name="HDL Cholesterol"
    ),
    "triglycerides": BiomarkerRange(
        low=0, high=150, unit="mg/dL", critical_high=500,
        organ_system="heart", display_name="Triglycerides"
    ),
    "vldl_cholesterol": BiomarkerRange(
        low=2, high=30, unit="mg/dL",
        organ_system="heart", display_name="VLDL Cholesterol"
    ),

    # Blood Sugar — Pancreas/Metabolic
    "fasting_glucose": BiomarkerRange(
        low=70, high=100, unit="mg/dL", critical_low=50, critical_high=250,
        organ_system="metabolic", display_name="Fasting Glucose"
    ),
    "hba1c": BiomarkerRange(
        low=4.0, high=5.7, unit="%", critical_high=10.0,
        organ_system="metabolic", display_name="HbA1c"
    ),
    "random_glucose": BiomarkerRange(
        low=70, high=140, unit="mg/dL", critical_high=400,
        organ_system="metabolic", display_name="Random Glucose"
    ),

    # CBC — Blood/Immune
    "hemoglobin": BiomarkerRange(
        low=12.0, high=17.5, unit="g/dL", critical_low=7.0, critical_high=20.0,
        organ_system="blood", display_name="Hemoglobin"
    ),
    "rbc_count": BiomarkerRange(
        low=4.0, high=6.0, unit="million/µL",
        organ_system="blood", display_name="RBC Count"
    ),
    "wbc_count": BiomarkerRange(
        low=4000, high=11000, unit="/µL", critical_low=2000, critical_high=30000,
        organ_system="blood", display_name="WBC Count"
    ),
    "platelet_count": BiomarkerRange(
        low=150000, high=400000, unit="/µL", critical_low=50000, critical_high=800000,
        organ_system="blood", display_name="Platelet Count"
    ),
    "hematocrit": BiomarkerRange(
        low=36.0, high=54.0, unit="%",
        organ_system="blood", display_name="Hematocrit"
    ),
    "mcv": BiomarkerRange(
        low=80, high=100, unit="fL",
        organ_system="blood", display_name="MCV"
    ),
    "mch": BiomarkerRange(
        low=27, high=33, unit="pg",
        organ_system="blood", display_name="MCH"
    ),
    "mchc": BiomarkerRange(
        low=32, high=36, unit="g/dL",
        organ_system="blood", display_name="MCHC"
    ),

    # Liver Function
    "sgot_ast": BiomarkerRange(
        low=0, high=40, unit="U/L", critical_high=200,
        organ_system="liver", display_name="SGOT (AST)"
    ),
    "sgpt_alt": BiomarkerRange(
        low=0, high=40, unit="U/L", critical_high=200,
        organ_system="liver", display_name="SGPT (ALT)"
    ),
    "alkaline_phosphatase": BiomarkerRange(
        low=44, high=147, unit="U/L",
        organ_system="liver", display_name="Alkaline Phosphatase"
    ),
    "total_bilirubin": BiomarkerRange(
        low=0.1, high=1.2, unit="mg/dL", critical_high=5.0,
        organ_system="liver", display_name="Total Bilirubin"
    ),
    "direct_bilirubin": BiomarkerRange(
        low=0.0, high=0.3, unit="mg/dL",
        organ_system="liver", display_name="Direct Bilirubin"
    ),
    "total_protein": BiomarkerRange(
        low=6.0, high=8.3, unit="g/dL",
        organ_system="liver", display_name="Total Protein"
    ),
    "albumin": BiomarkerRange(
        low=3.5, high=5.5, unit="g/dL", critical_low=2.0,
        organ_system="liver", display_name="Albumin"
    ),
    "globulin": BiomarkerRange(
        low=2.0, high=3.5, unit="g/dL",
        organ_system="liver", display_name="Globulin"
    ),

    # Kidney Function
    "creatinine": BiomarkerRange(
        low=0.6, high=1.2, unit="mg/dL", critical_high=4.0,
        organ_system="kidney", display_name="Creatinine"
    ),
    "blood_urea_nitrogen": BiomarkerRange(
        low=7, high=20, unit="mg/dL", critical_high=100,
        organ_system="kidney", display_name="Blood Urea Nitrogen"
    ),
    "urea": BiomarkerRange(
        low=15, high=45, unit="mg/dL", critical_high=200,
        organ_system="kidney", display_name="Urea"
    ),
    "uric_acid": BiomarkerRange(
        low=3.4, high=7.0, unit="mg/dL", critical_high=12.0,
        organ_system="kidney", display_name="Uric Acid"
    ),
    "egfr": BiomarkerRange(
        low=90, high=120, unit="mL/min/1.73m²", critical_low=15,
        organ_system="kidney", display_name="eGFR"
    ),

    # Thyroid
    "tsh": BiomarkerRange(
        low=0.4, high=4.0, unit="mIU/L", critical_high=10.0,
        organ_system="thyroid", display_name="TSH"
    ),
    "t3": BiomarkerRange(
        low=80, high=200, unit="ng/dL",
        organ_system="thyroid", display_name="T3"
    ),
    "t4": BiomarkerRange(
        low=5.0, high=12.0, unit="µg/dL",
        organ_system="thyroid", display_name="T4"
    ),

    # Electrolytes
    "sodium": BiomarkerRange(
        low=136, high=145, unit="mEq/L", critical_low=120, critical_high=160,
        organ_system="metabolic", display_name="Sodium"
    ),
    "potassium": BiomarkerRange(
        low=3.5, high=5.0, unit="mEq/L", critical_low=2.5, critical_high=6.5,
        organ_system="metabolic", display_name="Potassium"
    ),
    "calcium": BiomarkerRange(
        low=8.5, high=10.5, unit="mg/dL", critical_low=6.0, critical_high=13.0,
        organ_system="metabolic", display_name="Calcium"
    ),

    # Iron
    "iron": BiomarkerRange(
        low=60, high=170, unit="µg/dL",
        organ_system="blood", display_name="Serum Iron"
    ),
    "ferritin": BiomarkerRange(
        low=12, high=300, unit="ng/mL",
        organ_system="blood", display_name="Ferritin"
    ),

    # Vitamins
    "vitamin_d": BiomarkerRange(
        low=30, high=100, unit="ng/mL", critical_low=10,
        organ_system="metabolic", display_name="Vitamin D"
    ),
    "vitamin_b12": BiomarkerRange(
        low=200, high=900, unit="pg/mL", critical_low=100,
        organ_system="blood", display_name="Vitamin B12"
    ),
}

# Common aliases that map to canonical names
BIOMARKER_ALIASES: dict[str, str] = {
    "cholesterol": "total_cholesterol",
    "total chol": "total_cholesterol",
    "chol": "total_cholesterol",
    "ldl": "ldl_cholesterol",
    "ldl-c": "ldl_cholesterol",
    "hdl": "hdl_cholesterol",
    "hdl-c": "hdl_cholesterol",
    "tg": "triglycerides",
    "trigs": "triglycerides",
    "glucose": "fasting_glucose",
    "fasting blood sugar": "fasting_glucose",
    "fbs": "fasting_glucose",
    "blood sugar": "fasting_glucose",
    "glycated hemoglobin": "hba1c",
    "glycosylated hemoglobin": "hba1c",
    "hb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "rbc": "rbc_count",
    "wbc": "wbc_count",
    "white blood cells": "wbc_count",
    "red blood cells": "rbc_count",
    "platelets": "platelet_count",
    "plt": "platelet_count",
    "ast": "sgot_ast",
    "sgot": "sgot_ast",
    "alt": "sgpt_alt",
    "sgpt": "sgpt_alt",
    "alp": "alkaline_phosphatase",
    "bilirubin": "total_bilirubin",
    "bil": "total_bilirubin",
    "bun": "blood_urea_nitrogen",
    "blood urea": "blood_urea_nitrogen",
    "creat": "creatinine",
    "serum creatinine": "creatinine",
    "na": "sodium",
    "k": "potassium",
    "ca": "calcium",
    "vit d": "vitamin_d",
    "25-oh vitamin d": "vitamin_d",
    "vit b12": "vitamin_b12",
}

# Organ systems for grouping
ORGAN_SYSTEMS = {
    "heart": ["total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglycerides", "vldl_cholesterol"],
    "liver": ["sgot_ast", "sgpt_alt", "alkaline_phosphatase", "total_bilirubin", "direct_bilirubin", "total_protein", "albumin", "globulin"],
    "kidney": ["creatinine", "blood_urea_nitrogen", "urea", "uric_acid", "egfr"],
    "blood": ["hemoglobin", "rbc_count", "wbc_count", "platelet_count", "hematocrit", "mcv", "mch", "mchc", "iron", "ferritin", "vitamin_b12"],
    "metabolic": ["fasting_glucose", "hba1c", "random_glucose", "sodium", "potassium", "calcium", "vitamin_d"],
    "thyroid": ["tsh", "t3", "t4"],
}


def normalize_biomarker_name(name: str) -> str:
    """Normalize a biomarker name to its canonical form."""
    cleaned = name.lower().strip()
    # Remove common suffixes/prefixes
    cleaned = cleaned.replace("serum ", "").replace("plasma ", "").replace("blood ", "")
    cleaned = cleaned.replace("(", "").replace(")", "").replace("-", "_").replace(" ", "_")

    # Check aliases
    if cleaned in BIOMARKER_ALIASES:
        return BIOMARKER_ALIASES[cleaned]

    # Check if it directly matches a canonical name
    if cleaned in BIOMARKER_RANGES:
        return cleaned

    # Try partial matching
    for alias, canonical in BIOMARKER_ALIASES.items():
        if alias in cleaned or cleaned in alias:
            return canonical

    return cleaned


def classify_biomarker(name: str, value: float) -> str:
    """Classify a biomarker value as normal, low, high, or critical."""
    canonical = normalize_biomarker_name(name)
    ref = BIOMARKER_RANGES.get(canonical)

    if ref is None:
        return "normal"  # Unknown biomarker — default to normal

    # Check critical ranges first
    if ref.critical_low is not None and value < ref.critical_low:
        return "critical"
    if ref.critical_high is not None and value > ref.critical_high:
        return "critical"

    # Check normal range
    if value < ref.low:
        return "low"
    elif value > ref.high:
        return "high"

    return "normal"


def get_reference_range_str(name: str) -> str | None:
    """Get the reference range as a display string."""
    canonical = normalize_biomarker_name(name)
    ref = BIOMARKER_RANGES.get(canonical)
    if ref is None:
        return None
    return f"{ref.low}-{ref.high} {ref.unit}"


def get_display_name(name: str) -> str:
    """Get the display name for a biomarker."""
    canonical = normalize_biomarker_name(name)
    ref = BIOMARKER_RANGES.get(canonical)
    if ref and ref.display_name:
        return ref.display_name
    return name.replace("_", " ").title()


def get_biomarker_unit(name: str) -> str:
    """Get the expected unit for a biomarker."""
    canonical = normalize_biomarker_name(name)
    ref = BIOMARKER_RANGES.get(canonical)
    return ref.unit if ref else ""
