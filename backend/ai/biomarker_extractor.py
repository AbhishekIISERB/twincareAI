"""Biomarker extraction from raw OCR text.

Two-stage approach:
1. Regex patterns for common lab report formats (handles ~80% of clean reports)
2. LLM fallback via Fireworks API for ambiguous/messy text
"""

import re
import json
import logging
from dataclasses import dataclass

from utils.biomarker_ranges import (
    normalize_biomarker_name,
    classify_biomarker,
    get_reference_range_str,
    get_display_name,
    get_biomarker_unit,
    BIOMARKER_RANGES,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedBiomarker:
    """A single extracted biomarker from text."""
    name: str            # Canonical name
    display_name: str    # Human-readable name
    value: float
    unit: str
    reference_range: str | None
    status: str          # normal, low, high, critical


# --- Regex patterns for common biomarker formats ---
# Patterns match: "Biomarker Name    value    unit    reference_range"
# Examples:
#   "Total Cholesterol    240    mg/dL    125 - 200"
#   "Glucose: 95 mg/dL"
#   "HbA1c 5.4 %"

BIOMARKER_PATTERNS = [
    # Pattern: Name ... Value Unit (Reference)
    # Matches "Total Cholesterol    240.5    mg/dL    125 - 200"
    (
        r"(?:total\s*cholesterol|serum\s*cholesterol)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "total_cholesterol",
    ),
    (
        r"(?:ldl|ldl[\s\-]?c(?:holesterol)?)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "ldl_cholesterol",
    ),
    (
        r"(?:hdl|hdl[\s\-]?c(?:holesterol)?)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "hdl_cholesterol",
    ),
    (
        r"(?:triglycerides?|tg)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "triglycerides",
    ),
    (
        r"(?:vldl|vldl[\s\-]?c(?:holesterol)?)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "vldl_cholesterol",
    ),
    (
        r"(?:fasting\s*(?:blood\s*)?(?:sugar|glucose)|fbs|glucose\s*fasting|blood\s*sugar\s*fasting)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "fasting_glucose",
    ),
    (
        r"(?:random\s*(?:blood\s*)?(?:sugar|glucose)|rbs)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "random_glucose",
    ),
    (
        r"(?:hba1c|glycated\s*h(?:a?e)?moglobin|glycosylated\s*h(?:a?e)?moglobin)\s*[:\-]?\s*([\d.]+)\s*(%)",
        "hba1c",
    ),
    (
        r"(?:h(?:a?e)?moglobin|hb|hgb)(?!\s*a1c)\s*[:\-]?\s*([\d.]+)\s*(g/d[lL])",
        "hemoglobin",
    ),
    (
        r"(?:rbc|red\s*blood\s*cell)\s*(?:count)?\s*[:\-]?\s*([\d.]+)\s*(million/[µu][lL]|x10\^6/[µu][lL])",
        "rbc_count",
    ),
    (
        r"(?:wbc|white\s*blood\s*cell|total\s*leucocyte)\s*(?:count|tlc)?\s*[:\-]?\s*([\d,.]+)\s*(/[µu][lL]|cells/[µu][lL]|x10\^3/[µu][lL])",
        "wbc_count",
    ),
    (
        r"(?:platelet|plt)\s*(?:count)?\s*[:\-]?\s*([\d,.]+)\s*(/[µu][lL]|x10\^3/[µu][lL]|lakhs)",
        "platelet_count",
    ),
    (
        r"(?:h(?:a?e)?matocrit|hct|pcv)\s*[:\-]?\s*([\d.]+)\s*(%)",
        "hematocrit",
    ),
    (
        r"(?:mcv|mean\s*corpuscular\s*volume)\s*[:\-]?\s*([\d.]+)\s*(f[lL])",
        "mcv",
    ),
    (
        r"(?:mch(?!c)|mean\s*corpuscular\s*h(?:a?e)?moglobin(?!\s*conc))\s*[:\-]?\s*([\d.]+)\s*(pg)",
        "mch",
    ),
    (
        r"(?:mchc|mean\s*corpuscular\s*h(?:a?e)?moglobin\s*conc(?:entration)?)\s*[:\-]?\s*([\d.]+)\s*(g/d[lL]|%)",
        "mchc",
    ),
    (
        r"(?:sgot|ast|aspartate\s*(?:amino)?transaminase)\s*[:\-]?\s*([\d.]+)\s*(U/[lL]|IU/[lL])",
        "sgot_ast",
    ),
    (
        r"(?:sgpt|alt|alanine\s*(?:amino)?transaminase)\s*[:\-]?\s*([\d.]+)\s*(U/[lL]|IU/[lL])",
        "sgpt_alt",
    ),
    (
        r"(?:alkaline\s*phosphatase|alp)\s*[:\-]?\s*([\d.]+)\s*(U/[lL]|IU/[lL])",
        "alkaline_phosphatase",
    ),
    (
        r"(?:total\s*bilirubin|bilirubin\s*total)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "total_bilirubin",
    ),
    (
        r"(?:direct\s*bilirubin|bilirubin\s*direct|conjugated\s*bilirubin)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "direct_bilirubin",
    ),
    (
        r"(?:total\s*protein)\s*[:\-]?\s*([\d.]+)\s*(g/d[lL])",
        "total_protein",
    ),
    (
        r"(?:albumin)\s*[:\-]?\s*([\d.]+)\s*(g/d[lL])",
        "albumin",
    ),
    (
        r"(?:globulin)\s*[:\-]?\s*([\d.]+)\s*(g/d[lL])",
        "globulin",
    ),
    (
        r"(?:creatinine|creat)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "creatinine",
    ),
    (
        r"(?:blood\s*urea\s*nitrogen|bun)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "blood_urea_nitrogen",
    ),
    (
        r"(?:urea(?!\s*nitrogen))\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "urea",
    ),
    (
        r"(?:uric\s*acid)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "uric_acid",
    ),
    (
        r"(?:tsh|thyroid\s*stimulating\s*hormone)\s*[:\-]?\s*([\d.]+)\s*(mIU/[lL]|[µu]IU/m[lL])",
        "tsh",
    ),
    (
        r"(?:sodium|na)\s*[:\-]?\s*([\d.]+)\s*(mEq/[lL]|mmol/[lL])",
        "sodium",
    ),
    (
        r"(?:potassium|k)\s*[:\-]?\s*([\d.]+)\s*(mEq/[lL]|mmol/[lL])",
        "potassium",
    ),
    (
        r"(?:calcium|ca)\s*[:\-]?\s*([\d.]+)\s*(mg/d[lL])",
        "calcium",
    ),
    (
        r"(?:vitamin\s*d|vit\s*d|25[\-\s]?(?:oh|hydroxy)?\s*vitamin\s*d)\s*[:\-]?\s*([\d.]+)\s*(ng/m[lL])",
        "vitamin_d",
    ),
    (
        r"(?:vitamin\s*b12|vit\s*b12|cyanocobalamin)\s*[:\-]?\s*([\d.]+)\s*(pg/m[lL])",
        "vitamin_b12",
    ),
    (
        r"(?:iron|serum\s*iron)\s*[:\-]?\s*([\d.]+)\s*([µu]g/d[lL])",
        "iron",
    ),
    (
        r"(?:ferritin)\s*[:\-]?\s*([\d.]+)\s*(ng/m[lL]|[µu]g/[lL])",
        "ferritin",
    ),
]


def extract_biomarkers_regex(text: str) -> list[ExtractedBiomarker]:
    """Extract biomarkers using regex patterns."""
    results = []
    seen = set()
    text_lower = text.lower()

    for pattern, canonical_name in BIOMARKER_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if canonical_name in seen:
                continue
            try:
                value_str = match.group(1).replace(",", "")
                value = float(value_str)
                unit = match.group(2) if match.lastindex >= 2 else get_biomarker_unit(canonical_name)

                # Validate physiological plausibility
                ref = BIOMARKER_RANGES.get(canonical_name)
                if ref:
                    # Allow values up to 10x the high end (for clearly abnormal values)
                    if value < 0 or (ref.critical_high and value > ref.critical_high * 3):
                        continue

                status = classify_biomarker(canonical_name, value)
                results.append(ExtractedBiomarker(
                    name=canonical_name,
                    display_name=get_display_name(canonical_name),
                    value=value,
                    unit=unit,
                    reference_range=get_reference_range_str(canonical_name),
                    status=status,
                ))
                seen.add(canonical_name)
            except (ValueError, IndexError):
                continue

    logger.info(f"Regex extraction found {len(results)} biomarkers")
    return results


async def extract_biomarkers_llm(text: str) -> list[ExtractedBiomarker]:
    """
    Extract biomarkers using LLM (Fireworks API) for better accuracy on messy text.
    
    This is the fallback when regex extraction finds too few biomarkers.
    """
    from ai.llm_client import call_fireworks_api

    prompt = f"""You are a medical lab report parser. Extract ALL biomarker values from the following lab report text.

For each biomarker found, output a JSON array of objects with these fields:
- "name": the standard biomarker name (e.g., "Total Cholesterol", "Hemoglobin", "Fasting Glucose")
- "value": the numeric value (as a number, not string)
- "unit": the measurement unit (e.g., "mg/dL", "g/dL", "%")

Output ONLY the JSON array, no other text. If no biomarkers are found, output an empty array [].

Lab Report Text:
{text[:3000]}"""

    try:
        response = await call_fireworks_api(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        
        # Parse the LLM response
        response_text = response.strip()
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            biomarkers_data = json.loads(json_match.group())
        else:
            logger.warning("LLM response did not contain valid JSON array")
            return []

        results = []
        seen = set()
        for item in biomarkers_data:
            try:
                name = item.get("name", "")
                value = float(item.get("value", 0))
                unit = item.get("unit", "")

                canonical = normalize_biomarker_name(name)
                if canonical in seen:
                    continue

                status = classify_biomarker(canonical, value)
                ref_range = get_reference_range_str(canonical)

                results.append(ExtractedBiomarker(
                    name=canonical,
                    display_name=get_display_name(canonical),
                    value=value,
                    unit=unit or get_biomarker_unit(canonical),
                    reference_range=ref_range,
                    status=status,
                ))
                seen.add(canonical)
            except (ValueError, KeyError, TypeError):
                continue

        logger.info(f"LLM extraction found {len(results)} biomarkers")
        return results

    except Exception as e:
        logger.error(f"LLM biomarker extraction failed: {e}")
        return []


async def extract_biomarkers(text: str) -> list[ExtractedBiomarker]:
    """
    Extract biomarkers from text using regex first, then LLM fallback.
    
    Strategy:
    1. Try regex extraction (fast, no API cost)
    2. If regex finds < 3 biomarkers, try LLM extraction
    3. Merge results, preferring regex matches for duplicates
    """
    regex_results = extract_biomarkers_regex(text)

    if len(regex_results) >= 3:
        return regex_results

    # Try LLM fallback
    logger.info("Regex found few biomarkers, trying LLM extraction...")
    llm_results = await extract_biomarkers_llm(text)

    # Merge: keep regex results and add unique LLM results
    seen = {r.name for r in regex_results}
    merged = list(regex_results)
    for r in llm_results:
        if r.name not in seen:
            merged.append(r)
            seen.add(r.name)

    logger.info(f"Total biomarkers extracted: {len(merged)}")
    return merged
