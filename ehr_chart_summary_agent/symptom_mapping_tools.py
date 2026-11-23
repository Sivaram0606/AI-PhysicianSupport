"""
Progressive Clinical Symptom Extraction & Multi-Stage Etiology Mapping Tools.

Supports the dedicated Symptom Mapping Assistant UI and Google ADK Sub-Agents by:
1. Extracting structured presenting symptoms, abnormal triage vitals, and acute lab
   derangements from the patient's HL7 FHIR R4 `$everything` record.
2. Dynamically mapping each symptom across the physician's progressive workflow stages:
   - Stage 1 & 2: Longitudinal EHR Conditions, Imaging & Open Clinical Loops
   - Stage 3: Prescribed/Held/OTC Drugs, NIH RxNorm Duplicate Ingredients & OpenFDA Adverse Effects
   - Stage 4: Hyperlocal 5-Digit ZIP-Code Infectious Outbreaks & Air Quality Triggers
3. Highlighting Unmatched / Residual Symptoms so the physician can validate whether
   medications explain the presentation or whether ZIP-code outbreak investigation is warranted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
from typing import Any

try:
    from .clinical_tools import (
        check_drug_brand_or_generic,
        get_location_outbreak_and_disease_data,
    )
    from .ehr_chart_tools import (
        get_hospitalizations_and_clinical_notes,
        get_longitudinal_labs_and_imaging,
        get_medication_reconciliation_history,
        get_open_loops_and_care_gaps,
        get_patient_demographics_and_triage,
        resolve_canonical_patient_id,
    )
except ImportError:
    from clinical_tools import (  # type: ignore
        check_drug_brand_or_generic,
        get_location_outbreak_and_disease_data,
    )
    from ehr_chart_tools import (  # type: ignore
        get_hospitalizations_and_clinical_notes,
        get_longitudinal_labs_and_imaging,
        get_medication_reconciliation_history,
        get_open_loops_and_care_gaps,
        get_patient_demographics_and_triage,
        resolve_canonical_patient_id,
    )


@lru_cache(maxsize=32)
def _cached_ehr_core(pid: str) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_intake = pool.submit(get_patient_demographics_and_triage, pid)
        f_labs = pool.submit(get_longitudinal_labs_and_imaging, pid)
        f_loops = pool.submit(get_open_loops_and_care_gaps, pid)
        return {
            "intake": f_intake.result(),
            "labs": f_labs.result(),
            "loops": f_loops.result(),
        }


@lru_cache(maxsize=32)
def _cached_med_history(pid: str) -> dict[str, Any]:
    return get_medication_reconciliation_history(pid)


@lru_cache(maxsize=128)
def _cached_drug_check(med_name: str, rxcui: str) -> dict[str, Any]:
    return check_drug_brand_or_generic(med_name, rxcui)


@lru_cache(maxsize=32)
def _cached_outbreak_check(postal_code: str, city: str) -> dict[str, Any]:
    return get_location_outbreak_and_disease_data(postal_code=postal_code, city=city)



# Canonical clinical symptom keyword ontology used to cross-match FHIR symptoms against
# EHR conditions, OpenFDA adverse reaction / MedDRA terms, and ZIP-code respiratory/infectious outbreaks.
_SYMPTOM_ONTOLOGY: list[dict[str, Any]] = [
    {
        "key": "dyspnea",
        "label": "Shortness of Breath / Dyspnea",
        "triggers": ["shortness of breath", "dyspnea", "respiratory distress", "orthopnea"],
        "ehr_keywords": ["asthma", "heart failure", "hfref", "pulmonary", "edema", "copd", "effusion"],
        "drug_ae_keywords": ["dyspnoea", "dyspnea", "bronchospasm", "wheezing", "fluid retention", "cut lasix", "self-cut", "prior auth"],
        "outbreak_keywords": ["influenza", "covid", "sars-cov-2", "rsv", "pm2.5", "aqi"],
        "infectious_candidate": True,
    },
    {
        "key": "wheezing",
        "label": "Expiratory Wheezing / Bronchospasm",
        "triggers": ["wheezing", "wheeze", "bronchospasm"],
        "ehr_keywords": ["asthma", "spirometry", "ics", "eosinophil", "hyperinflated"],
        "drug_ae_keywords": ["bronchospasm", "wheezing", "cough", "ace-inhibitor", "nsaid"],
        "outbreak_keywords": ["influenza", "rsv", "covid", "pm2.5", "ozone"],
        "infectious_candidate": True,
    },
    {
        "key": "cough",
        "label": "Persistent Cough",
        "triggers": ["dry cough", "persistent cough", "cough"],
        "ehr_keywords": ["asthma", "nodule", "pulmonary", "pharyngitis", "bronchitis"],
        "drug_ae_keywords": ["cough", "lisinopril", "ace-inhibitor", "bradykinin", "throat tickle"],
        "outbreak_keywords": ["influenza", "covid", "sars-cov-2", "rsv", "measles"],
        "infectious_candidate": True,
    },
    {
        "key": "fever",
        "label": "Fever / Pyrexia",
        "triggers": ["fever", "low-grade fever", "febrile", "chills"],
        "ehr_keywords": ["infection", "pharyngitis", "sepsis", "pneumonia"],
        "drug_ae_keywords": ["drug fever", "hypersensitivity"],
        "outbreak_keywords": ["influenza", "covid", "sars-cov-2", "rsv", "measles", "mpox"],
        "infectious_candidate": True,
    },
    {
        "key": "muscle_weakness_myalgia",
        "label": "Muscle Weakness / Cramps / Myalgia",
        "triggers": ["thigh weakness", "muscle weakness", "calf cramps", "cramps", "weakness", "myalgia"],
        "ehr_keywords": ["rhabdomyolysis", "hypokalemia", "hyperaldosteronism", "conn", "aldosterone"],
        "drug_ae_keywords": ["rhabdomyolysis", "myopathy", "myalgia", "muscle spasms", "weakness", "lipitor", "atorvastatin", "clarithromycin", "chlorthalidone", "hypokalemia"],
        "outbreak_keywords": ["influenza", "covid"],
        "infectious_candidate": False,
    },
    {
        "key": "dark_urine",
        "label": "Dark / Tea-Colored Urine (Myoglobinuria)",
        "triggers": ["dark urine", "tea-colored urine", "hematuria", "myoglobinuria"],
        "ehr_keywords": ["rhabdomyolysis", "acute kidney injury", "aki", "myoglobin"],
        "drug_ae_keywords": ["rhabdomyolysis", "chromaturia", "renal impairment", "lipitor", "clarithromycin"],
        "outbreak_keywords": [],
        "infectious_candidate": False,
    },
    {
        "key": "dizziness_orthostasis",
        "label": "Dizziness / Orthostasis / Lightheadedness",
        "triggers": ["dizziness", "dizzy", "orthostasis", "orthostatic", "lightheaded"],
        "ehr_keywords": ["anemia", "gi bleed", "hemorrhage", "dehydration", "hypotension", "atrial fibrillation"],
        "drug_ae_keywords": ["dizziness", "hypotension", "haemorrhage", "bleeding", "advil", "ibuprofen", "antihypertensive", "rhabdomyolysis", "clarithromycin", "lipitor", "atorvastatin"],
        "outbreak_keywords": [],
        "infectious_candidate": False,
    },
    {
        "key": "gi_bleed_epigastric",
        "label": "Severe Epigastric Pain & Melena",
        "triggers": ["epigastric pain", "melena", "black tarry", "abdominal pain", "gi bleed"],
        "ehr_keywords": ["peptic ulcer", "gastritis", "gastrointestinal hemorrhage", "anemia"],
        "drug_ae_keywords": ["gastrointestinal haemorrhage", "ulcer", "melena", "nsaid", "advil", "ibuprofen", "stopped prilosec"],
        "outbreak_keywords": [],
        "infectious_candidate": False,
    },
    {
        "key": "bleeding_epistaxis",
        "label": "Persistent Epistaxis & Easy Bruising",
        "triggers": ["epistaxis", "nosebleed", "bruising", "ecchymoses"],
        "ehr_keywords": ["coagulopathy", "atrial fibrillation", "supratherapeutic inr"],
        "drug_ae_keywords": ["epistaxis", "haemorrhage", "contusion", "inr", "warfarin", "coumadin", "cranberry"],
        "outbreak_keywords": [],
        "infectious_candidate": False,
    },
    {
        "key": "edema_weight_gain",
        "label": "Ankle Edema & Acute Weight Gain",
        "triggers": ["ankle edema", "edema", "swelling", "weight gain"],
        "ehr_keywords": ["heart failure", "hfref", "volume overload", "cardiomyopathy"],
        "drug_ae_keywords": ["self-cut", "furosemide", "lasix", "prior auth", "jardiance", "held lisinopril", "oedema"],
        "outbreak_keywords": [],
        "infectious_candidate": False,
    },
    {
        "key": "fatigue_nausea",
        "label": "Severe Fatigue & Nausea",
        "triggers": ["fatigue", "nausea", "malaise", "lethargy"],
        "ehr_keywords": ["hypothyroidism", "hepatotoxicity", "transaminitis", "rheumatoid arthritis", "anemia"],
        "drug_ae_keywords": ["fatigue", "nausea", "hepatotoxicity", "methotrexate", "ferrous sulfate", "synthroid", "levothyroxine", "malabsorption"],
        "outbreak_keywords": ["influenza", "covid", "sars-cov-2"],
        "infectious_candidate": True,
    },
]


def _parse_float_prefix(raw: Any) -> float | None:
    if raw is None:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(raw))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_patient_symptoms_from_fhir(patient_id: str) -> dict[str, Any]:
    """
    Extracts structured presenting symptoms, abnormal triage vitals, and acute laboratory
    derangements from the patient's FHIR R4 `$everything` record.

    Args:
        patient_id: Patient FHIR ID (`pat-001`..`pat-007`) or name.

    Returns:
        Dictionary with normalized symptom items, categories, severities, and FHIR citations.
    """
    pid = resolve_canonical_patient_id(patient_id)
    core = _cached_ehr_core(pid)
    intake = core["intake"]
    if intake.get("status") != "success":
        return {"status": "error", "message": intake.get("message", "Patient not found.")}

    labs_resp = core["labs"]
    visit = intake.get("todays_visit", {})
    vitals = visit.get("triage_vitals_today", {})
    cc = visit.get("chief_complaint", "")
    ma_note = vitals.get("ma_intake_note", "")
    combined_narrative = f"{cc} {ma_note}".lower()
    enc_res = visit.get("fhir_resource", f"Encounter/triage-{pid}")

    extracted: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    # 1. Narrative symptoms from Chief Complaint & MA Intake Note
    for item in _SYMPTOM_ONTOLOGY:
        matched_phrase = next((t for t in item["triggers"] if t in combined_narrative), None)
        if matched_phrase and item["key"] not in seen_keys:
            seen_keys.add(item["key"])
            extracted.append(
                {
                    "symptom_id": f"sym-{item['key']}",
                    "key": item["key"],
                    "label": item["label"],
                    "category": "Presenting Symptom",
                    "observed_value": f"Reported at triage ('{matched_phrase}')",
                    "severity": "HIGH" if item["key"] in ("dyspnea", "wheezing", "gi_bleed_epigastric", "bleeding_epistaxis", "dark_urine") else "MODERATE",
                    "infectious_candidate": item["infectious_candidate"],
                    "fhir_citation": enc_res,
                }
            )

    # 2. Objective Triage Vital Sign Derangements
    spo2_val = _parse_float_prefix(vitals.get("spo2"))
    if spo2_val is not None and spo2_val < 94:
        extracted.append(
            {
                "symptom_id": "sym-hypoxia",
                "key": "hypoxia",
                "label": f"Hypoxemia (SpO2 {vitals.get('spo2')})",
                "category": "Triage Vital Abnormality",
                "observed_value": str(vitals.get("spo2")),
                "severity": "CRITICAL" if spo2_val <= 91 else "HIGH",
                "infectious_candidate": True,
                "fhir_citation": enc_res,
            }
        )

    temp_val = _parse_float_prefix(vitals.get("temp"))
    if temp_val is not None and temp_val >= 99.8:
        # Ensure fever is prominently tracked as an objective vital sign if not already or enrich it
        existing_fever = next((x for x in extracted if x["key"] == "fever"), None)
        if existing_fever:
            existing_fever["label"] = f"Low-Grade Fever / Pyrexia (Temp {vitals.get('temp')})"
            existing_fever["observed_value"] = f"Triage Temp {vitals.get('temp')}"
        else:
            extracted.append(
                {
                    "symptom_id": "sym-fever",
                    "key": "fever",
                    "label": f"Elevated Temperature ({vitals.get('temp')})",
                    "category": "Triage Vital Abnormality",
                    "observed_value": str(vitals.get("temp")),
                    "severity": "MODERATE",
                    "infectious_candidate": True,
                    "fhir_citation": enc_res,
                }
            )

    bp_raw = str(vitals.get("bp", ""))
    bp_match = re.match(r"(\d+)\s*/\s*(\d+)", bp_raw)
    if bp_match:
        sbp, dbp = int(bp_match.group(1)), int(bp_match.group(2))
        if sbp >= 150 or dbp >= 95:
            extracted.append(
                {
                    "symptom_id": "sym-hypertension",
                    "key": "hypertension_vital",
                    "label": f"Acute Blood Pressure Elevation ({bp_raw})",
                    "category": "Triage Vital Abnormality",
                    "observed_value": bp_raw,
                    "severity": "HIGH" if sbp >= 165 or dbp >= 100 else "MODERATE",
                    "infectious_candidate": False,
                    "fhir_citation": enc_res,
                }
            )
        elif sbp < 100:
            extracted.append(
                {
                    "symptom_id": "sym-hypotension",
                    "key": "hypotension_vital",
                    "label": f"Hypotension / Hemodynamic Instability ({bp_raw})",
                    "category": "Triage Vital Abnormality",
                    "observed_value": bp_raw,
                    "severity": "CRITICAL",
                    "infectious_candidate": False,
                    "fhir_citation": enc_res,
                }
            )

    hr_val = _parse_float_prefix(vitals.get("hr"))
    if hr_val is not None and hr_val > 100:
        extracted.append(
            {
                "symptom_id": "sym-tachycardia",
                "key": "tachycardia",
                "label": f"Tachycardia (HR {vitals.get('hr')})",
                "category": "Triage Vital Abnormality",
                "observed_value": str(vitals.get("hr")),
                "severity": "HIGH" if hr_val >= 110 else "MODERATE",
                "infectious_candidate": True,
                "fhir_citation": enc_res,
            }
        )

    # 3. Acute Biomarker / Lab Derangements (Latest Observation with H/HH/L/LL)
    for lab in labs_resp.get("labs_and_diagnostics", {}).get("longitudinal_labs", []):
        hist = lab.get("history", [])
        if not hist:
            continue
        latest = hist[-1]
        flag = str(latest.get("flag", "N")).upper()
        if flag in ("H", "HH", "L", "LL"):
            baseline = hist[0]
            trend_str = (
                f"{baseline['value']} ({baseline['date']}) -> {latest['value']} [{flag}] ({latest['date']})"
                if len(hist) > 1
                else f"{latest['value']} [{flag}] ({latest['date']})"
            )
            extracted.append(
                {
                    "symptom_id": f"sym-lab-{re.sub(r'[^a-z0-9]+', '-', lab['test'].lower()).strip('-')}",
                    "key": f"lab_{lab['test'].lower()}",
                    "label": f"Abnormal {lab['test']}: {latest['value']} [{flag}]",
                    "category": "Acute Biomarker / Lab Sign",
                    "observed_value": trend_str,
                    "severity": "CRITICAL" if flag in ("HH", "LL") else "HIGH",
                    "infectious_candidate": "eosinophil" in lab["test"].lower() or "pef" in lab["test"].lower() or "expiratory" in lab["test"].lower(),
                    "fhir_citation": latest.get("fhir_resource", "Observation"),
                }
            )

    return {
        "status": "success",
        "patient_id": pid,
        "patient_name": intake["demographics"].get("name", pid),
        "postal_code": intake["demographics"].get("postal_code", ""),
        "chief_complaint": cc,
        "symptom_count": len(extracted),
        "symptoms": extracted,
    }


def _match_symptom_to_ehr(
    sym: dict[str, Any],
    active_problems: list[dict[str, Any]],
    imaging: list[dict[str, Any]],
    open_loops: list[dict[str, Any]],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    ont = next((o for o in _SYMPTOM_ONTOLOGY if o["key"] == sym["key"]), None)
    keywords = list(ont["ehr_keywords"]) if ont else []
    label_low = sym["label"].lower()

    # Add targeted biomarker keywords
    if "alt" in label_low or "ast" in label_low:
        keywords.extend(["hepatotoxicity", "acetaminophen", "methotrexate", "liver", "transaminitis"])
    elif "creatinine" in label_low or "egfr" in label_low:
        keywords.extend(["kidney", "renal", "aki", "rhabdomyolysis", "heart failure"])
    elif "potassium" in label_low or "aldosterone" in label_low or "arr" in label_low:
        keywords.extend(["hyperaldosteronism", "conn", "hypokalemia", "hypertension"])
    elif "inr" in label_low:
        keywords.extend(["atrial fibrillation", "warfarin", "anticoagulation"])
    elif "a1c" in label_low:
        keywords.extend(["diabetes", "glycemic", "metabolic", "hyperglycemia"])
    elif "hemoglobin" in label_low or "bun" in label_low:
        keywords.extend(["bleed", "ulcer", "gastritis", "anemia"])
    elif "ck" in label_low or "creatine kinase" in label_low:
        keywords.extend(["rhabdomyolysis", "statin", "myopathy"])
    elif "bnp" in label_low:
        keywords.extend(["heart failure", "hfref", "volume", "edema"])
    elif "tsh" in label_low:
        keywords.extend(["hypothyroidism", "thyroid", "synthroid"])
    elif "eosinophil" in label_low or "expiratory" in label_low or "hypoxia" in sym["key"]:
        keywords.extend(["asthma", "spirometry", "hyperinflated"])
    elif "hypertension" in sym["key"]:
        keywords.extend(["hypertension", "hyperaldosteronism", "conn"])
    elif "hypotension" in sym["key"] or "tachycardia" in sym["key"]:
        keywords.extend(["bleed", "atrial fibrillation", "asthma", "distress", "hemorrhage"])

    for prob in active_problems:
        cond_text = prob.get("condition", "")
        if any(kw in cond_text.lower() for kw in keywords):
            matches.append(
                {
                    "source_type": "FHIR Active Condition",
                    "finding": cond_text,
                    "rationale": f"Active diagnosis on problem list ({prob.get('icd10') or prob.get('snomed_code') or 'FHIR'}).",
                    "citation": prob.get("fhir_resource", "Condition"),
                }
            )

    for img in imaging:
        findings_txt = f"{img.get('modality', '')}: {img.get('findings', '')}"
        if any(kw in findings_txt.lower() for kw in keywords):
            matches.append(
                {
                    "source_type": "Diagnostic Imaging",
                    "finding": img.get("modality", "Imaging"),
                    "rationale": img.get("findings", "")[:180],
                    "citation": img.get("fhir_resource", "DiagnosticReport"),
                }
            )

    for loop in open_loops:
        # Avoid leaking drug-specific causality before Stage 3 if we only want longitudinal/care-gap context,
        # OR include non-medication care gaps in Stage 1-2 and full open loops!
        loop_txt = f"{loop.get('category', '')}: {loop.get('details', '')}"
        if any(kw in loop_txt.lower() for kw in keywords):
            matches.append(
                {
                    "source_type": f"Open Clinical Loop [{loop.get('priority')}]",
                    "finding": loop.get("category", "Open Loop"),
                    "rationale": loop.get("details", "")[:190],
                    "citation": loop.get("fhir_resource", "Flag"),
                }
            )

    return matches[:3]


def _match_symptom_to_drugs(
    sym: dict[str, Any],
    enriched_meds: list[dict[str, Any]],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    ont = next((o for o in _SYMPTOM_ONTOLOGY if o["key"] == sym["key"]), None)
    keywords = [k.lower() for k in (ont["drug_ae_keywords"] if ont else [])]
    label_low = sym["label"].lower()

    # Fever in pat-001 should NOT falsely match antipyretic Tylenol as a drug-caused symptom!
    if sym["key"] == "fever":
        keywords = ["drug fever", "neuroleptic malignant", "serotonin syndrome"]

    if "alt" in label_low or "ast" in label_low:
        keywords.extend(["hepatotoxicity", "liver", "hepatic", "transaminases", "acetaminophen", "tylenol", "duplicate", "methotrexate"])
    elif "creatinine" in label_low or "egfr" in label_low:
        keywords.extend(["renal", "kidney", "nephrotoxicity", "rhabdomyolysis", "clarithromycin", "lipitor", "nsaid", "ibuprofen", "advil", "held for aki"])
    elif "potassium" in label_low:
        keywords.extend(["hypokalaemia", "hypokalemia", "potassium", "chlorthalidone", "furosemide", "lasix", "lisinopril"])
    elif "inr" in label_low:
        keywords.extend(["inr", "international normalised ratio", "haemorrhage", "bleeding", "warfarin", "coumadin", "cranberry"])
    elif ("hemoglobin" in label_low and "a1c" not in label_low) or "bun" in label_low:
        keywords.extend(["gastrointestinal haemorrhage", "bleeding", "ulcer", "melena", "ibuprofen", "advil", "nsaid", "prilosec"])
    elif "ck" in label_low or "creatine kinase" in label_low:
        keywords.extend(["rhabdomyolysis", "creatine phosphokinase", "myopathy", "lipitor", "atorvastatin", "clarithromycin", "cyp3a4"])
    elif "bnp" in label_low:
        keywords.extend(["self-cut", "furosemide", "lasix", "prior auth", "jardiance", "empagliflozin", "held"])
    elif "tsh" in label_low:
        keywords.extend(["ferrous", "iron", "chelation", "malabsorption", "synthroid", "levothyroxine"])
    elif "hypertension" in sym["key"]:
        keywords.extend(["chlorthalidone", "resistant", "lisinopril", "ibuprofen", "nsaid"])
    elif "hypotension" in sym["key"]:
        keywords.extend(["haemorrhage", "bleeding", "advil", "ibuprofen", "hypotension"])
    elif "tachycardia" in sym["key"]:
        keywords.extend(["gastrointestinal haemorrhage", "stopped prilosec", "melena"])

    if not keywords:
        return []

    for med in enriched_meds:
        med_name = med["medication"]
        adherence_and_reason = f"{med.get('change_type', '')} {med.get('adherence_or_reason', '')}".lower()
        rx = med.get("rxnorm_openfda", {})
        side_fx = rx.get("openfda_known_side_effects", {})
        faers_dict = side_fx.get("top_faers_reported_adverse_events", {})
        faers_terms = " ".join(faers_dict.keys()).lower() if isinstance(faers_dict, dict) else ""
        label_blob = " ".join(
            str(side_fx.get(k) or "")
            for k in ("boxed_warning_excerpt", "labeled_adverse_reactions", "warnings_and_cautions", "drug_interactions_excerpt")
        ).lower()
        full_blob = f"{med_name.lower()} {adherence_and_reason} {faers_terms} {label_blob}"

        hit_kws = [kw for kw in keywords if kw in full_blob]
        if hit_kws:
            classification = rx.get("classification", "MEDICATION")
            evidence_parts: list[str] = []
            if any(kw in adherence_and_reason for kw in hit_kws):
                evidence_parts.append(med.get("adherence_or_reason", ""))
            if side_fx.get("boxed_warning_excerpt") and any(kw in str(side_fx.get("boxed_warning_excerpt")).lower() for kw in hit_kws):
                evidence_parts.append(f"FDA Boxed Warning: {side_fx['boxed_warning_excerpt'][:140]}...")
            elif any(kw in label_blob for kw in hit_kws):
                snippet = (side_fx.get("warnings_and_cautions") or side_fx.get("labeled_adverse_reactions") or "")[:140]
                if snippet:
                    evidence_parts.append(f"OpenFDA Label: {snippet}...")
            if isinstance(faers_dict, dict):
                matched_faers = [t for t in faers_dict.keys() if any(kw in t.lower() for kw in hit_kws)]
                if matched_faers:
                    evidence_parts.append(f"FAERS Adverse Events: {', '.join(matched_faers[:3])}")

            matches.append(
                {
                    "medication": med_name,
                    "classification": classification,
                    "status": med.get("status_badge", "ACTIVE"),
                    "rationale": " | ".join(p for p in evidence_parts if p) or f"Matched pharmacology/adverse profile ({', '.join(hit_kws[:2])}).",
                    "citation": f"{med.get('fhir_resource', 'MedicationRequest')} + OpenFDA / NIH RxNorm",
                }
            )

    return matches[:3]


def _match_symptom_to_outbreaks(
    sym: dict[str, Any],
    outbreak_data: dict[str, Any],
) -> list[dict[str, str]]:
    if not outbreak_data or outbreak_data.get("error"):
        return []
    if not sym.get("infectious_candidate"):
        return []

    matches: list[dict[str, str]] = []
    zip_code = outbreak_data.get("patient_zip_location", {}).get("zip_code", "")
    ed_block = outbreak_data.get("active_outbreaks_in_zip_catchment", {}).get("local_emergency_department_trajectories", {})
    for path in ed_block.get("pathogen_ed_trajectories", []):
        trend = str(path.get("weekly_outbreak_trend", ""))
        pct = path.get("percent_of_local_ed_visits", 0.0)
        if "increas" in trend.lower() or (pct and float(pct) >= 1.0):
            matches.append(
                {
                    "signal_type": f"ZIP {zip_code} Local ED Surge",
                    "pathogen_or_factor": path.get("pathogen", "Respiratory Virus"),
                    "rationale": f"{pct}% of local ED visits in ZIP {zip_code} catchment (Weekly trend: {trend}).",
                    "citation": f"CDC NSSP Local ED (rdmq-nq56) — ZIP {zip_code}",
                }
            )

    for w in outbreak_data.get("active_outbreaks_in_zip_catchment", {}).get("local_sewershed_wastewater_pathogens", []):
        if w.get("detected_in_local_wastewater"):
            matches.append(
                {
                    "signal_type": f"ZIP {zip_code} Wastewater PCR Detection",
                    "pathogen_or_factor": w.get("pathogen", "Pathogen"),
                    "rationale": f"Detected in local sewershed ({w.get('short_term_trend', 'Active')}).",
                    "citation": f"CDC NWSS Sewershed — ZIP {zip_code}",
                }
            )

    aq = outbreak_data.get("zip_code_air_quality", {})
    aqi = aq.get("us_aqi")
    if aqi is not None and float(aqi) >= 50 and sym["key"] in ("dyspnea", "wheezing", "cough", "hypoxia"):
        matches.append(
            {
                "signal_type": f"ZIP {zip_code} Environmental Air Trigger",
                "pathogen_or_factor": f"US AQI {aqi} (PM2.5: {aq.get('pm2_5_ug_m3')} ug/m3)",
                "rationale": "Moderate/elevated particulate or ozone levels can exacerbate acute bronchospasm/dyspnea.",
                "citation": f"Open-Meteo Air Quality — ZIP {zip_code}",
            }
        )

    return matches[:3]


def compute_symptom_mappings(
    patient_id: str,
    unlocked_stage: int = 1,
    precomputed_drug_checks: dict[str, dict[str, Any]] | None = None,
    precomputed_outbreaks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Internal/UI symptom mapping engine supporting optional pre-fetched RxNorm/OpenFDA
    and ZIP-code outbreak caches to avoid duplicate HTTP requests in the Physician UI.
    """
    pid = resolve_canonical_patient_id(patient_id)
    extracted_res = extract_patient_symptoms_from_fhir(pid)
    if extracted_res.get("status") != "success":
        return extracted_res

    core = _cached_ehr_core(pid)
    intake = core["intake"]
    labs_resp = core["labs"]
    loops_resp = core["loops"]
    active_problems = intake.get("demographics", {}).get("active_problem_list", [])
    imaging = labs_resp.get("labs_and_diagnostics", {}).get("imaging_and_procedures", [])
    open_loops = loops_resp.get("unresolved_items_and_care_gaps", [])

    enriched_meds: list[dict[str, Any]] = []
    if unlocked_stage >= 3:
        med_hist = _cached_med_history(pid)
        drug_cache = dict(precomputed_drug_checks or {})
        active_list = med_hist.get("medications", {}).get("active_prescriptions", [])
        held_list = med_hist.get("medications", {}).get("held_or_discontinued_medications", [])
        missing_specs = [
            (item["medication"], item.get("rxcui_code", ""))
            for item in [*active_list, *held_list]
            if item["medication"] not in drug_cache
        ]
        if missing_specs:
            with ThreadPoolExecutor(max_workers=min(8, len(missing_specs))) as pool:
                futs = {
                    name: pool.submit(_cached_drug_check, name, rxcui)
                    for name, rxcui in missing_specs
                }
                for name, fut in futs.items():
                    drug_cache[name] = fut.result()

        for m in active_list:
            m_name = m["medication"]
            enriched_meds.append(
                {
                    "medication": m_name,
                    "status_badge": "ACTIVE",
                    "change_type": m.get("change_type", ""),
                    "adherence_or_reason": m.get("actual_patient_adherence", ""),
                    "fhir_resource": m.get("fhir_resource", "MedicationRequest"),
                    "rxnorm_openfda": drug_cache.get(m_name, {}),
                }
            )
        for h in held_list:
            h_name = h["medication"]
            enriched_meds.append(
                {
                    "medication": h_name,
                    "status_badge": h.get("status", "HELD/STOPPED"),
                    "change_type": h.get("status", ""),
                    "adherence_or_reason": f"{h.get('reason', '')} ({h.get('reinitiation_criteria', '')})",
                    "fhir_resource": h.get("fhir_resource", "MedicationRequest"),
                    "rxnorm_openfda": drug_cache.get(h_name, {}),
                }
            )

    outbreak_data: dict[str, Any] = {}
    postal_code = extracted_res.get("postal_code", "")
    if unlocked_stage >= 4 and postal_code:
        outbreak_data = precomputed_outbreaks or _cached_outbreak_check(
            postal_code,
            intake.get("demographics", {}).get("city", ""),
        )

    mapped_symptoms: list[dict[str, Any]] = []
    ehr_matched_count = 0
    drug_matched_count = 0
    outbreak_matched_count = 0
    unmatched_count = 0

    for sym in extracted_res["symptoms"]:
        ehr_matches = _match_symptom_to_ehr(sym, active_problems, imaging, open_loops)
        drug_matches = _match_symptom_to_drugs(sym, enriched_meds) if unlocked_stage >= 3 else []
        outbreak_matches = _match_symptom_to_outbreaks(sym, outbreak_data) if unlocked_stage >= 4 else []

        if ehr_matches:
            ehr_matched_count += 1
        if drug_matches:
            drug_matched_count += 1
        if outbreak_matches:
            outbreak_matched_count += 1

        # Determine primary resolution status at the currently unlocked stage
        if unlocked_stage >= 4 and outbreak_matches and not drug_matches:
            primary_status = "MATCHED_ZIP_OUTBREAK"
            status_label = "ZIP Outbreak Correlation"
            confidence = "HIGH"
        elif unlocked_stage >= 3 and drug_matches:
            primary_status = "MATCHED_DRUG_AE"
            status_label = "Drug / Adverse-Effect Match"
            confidence = "HIGH"
        elif ehr_matches:
            primary_status = "MATCHED_EHR"
            status_label = "Intrinsic EHR / Lab Match"
            confidence = "MODERATE" if unlocked_stage < 3 else "HIGH"
        elif unlocked_stage >= 4 and outbreak_matches:
            primary_status = "MATCHED_ZIP_OUTBREAK"
            status_label = "ZIP Outbreak Correlation"
            confidence = "MODERATE"
        else:
            primary_status = "UNMATCHED"
            status_label = (
                "No Drug Match — Check ZIP Outbreaks"
                if unlocked_stage == 3
                else "Unexplained at Current Stage"
            )
            confidence = "UNRESOLVED"
            unmatched_count += 1

        mapped_symptoms.append(
            {
                **sym,
                "short_label": _make_short_label(sym),
                "primary_status": primary_status,
                "status_label": status_label,
                "confidence": confidence,
                "ehr_matches": ehr_matches,
                "drug_matches": drug_matches,
                "outbreak_matches": outbreak_matches,
                "unmatched_by_drugs": unlocked_stage >= 3 and len(drug_matches) == 0,
            }
        )

    unmatched_by_drugs_list = [
        s["label"] for s in mapped_symptoms if unlocked_stage >= 3 and len(s["drug_matches"]) == 0
    ]
    competing = build_competing_hypotheses(
        mapped_symptoms=mapped_symptoms,
        unlocked_stage=unlocked_stage,
        postal_code=postal_code,
    )

    return {
        "status": "success",
        "patient_id": pid,
        "patient_name": extracted_res["patient_name"],
        "postal_code": postal_code,
        "unlocked_stage": unlocked_stage,
        "summary_counts": {
            "total_symptoms": len(mapped_symptoms),
            "ehr_matched": ehr_matched_count,
            "drug_ae_matched": drug_matched_count,
            "zip_outbreak_matched": outbreak_matched_count,
            "unmatched_overall": unmatched_count,
            "unmatched_by_drugs": len(unmatched_by_drugs_list),
        },
        "unmatched_by_drugs_labels": unmatched_by_drugs_list,
        "competing_hypotheses": competing,
        "mapped_symptoms": mapped_symptoms,
    }


_SHORT_KEY_LABELS: dict[str, str] = {
    "dyspnea": "Dyspnea",
    "wheezing": "Wheezing",
    "cough": "Persistent Cough",
    "muscle_weakness_myalgia": "Myalgia / Weakness",
    "dark_urine": "Dark Urine",
    "dizziness_orthostasis": "Orthostasis / Dizzy",
    "gi_bleed_epigastric": "Epigastric Pain & Melena",
    "bleeding_epistaxis": "Epistaxis & Bruising",
    "edema_weight_gain": "Edema & Wt Gain",
    "fatigue_nausea": "Fatigue & Nausea",
}


def _make_short_label(sym: dict[str, Any]) -> str:
    """Produces a compact 2-4 word clinical chip label for scannable Hypothesis Cards."""
    key = sym.get("key", "")
    label = str(sym.get("label", ""))
    obs = str(sym.get("observed_value", ""))

    if key in _SHORT_KEY_LABELS:
        return _SHORT_KEY_LABELS[key]
    if key == "fever":
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*°?\s*([FC])?", f"{label} {obs}", re.IGNORECASE)
        if m:
            unit = (m.group(2) or "F").upper()
            return f"Fever {m.group(1)}°{unit}"
        return "Fever / Pyrexia"
    if key == "hypoxia":
        m = re.search(r"(\d{2,3})\s*%", f"{label} {obs}")
        return f"SpO2 {m.group(1)}%" if m else "Hypoxemia"
    if key == "hypertension_vital":
        m = re.search(r"(\d{2,3}\s*/\s*\d{2,3})", f"{label} {obs}")
        return f"BP {m.group(1).replace(' ', '')}" if m else "Elevated BP"
    if key == "hypotension_vital":
        m = re.search(r"(\d{2,3}\s*/\s*\d{2,3})", f"{label} {obs}")
        return f"Low BP {m.group(1).replace(' ', '')}" if m else "Hypotension"
    if key == "tachycardia":
        m = re.search(r"(\d{2,3})", f"{label} {obs}")
        return f"HR {m.group(1)} bpm" if m else "Tachycardia"
    if key.startswith("lab_"):
        # Format e.g. "Abnormal Peak Expiratory Flow (% Predicted): 54 [LL]" -> "PEF 54 [LL]"
        val_m = re.search(r":\s*([^\[]+\[[A-Z]+\])", label)
        val_part = val_m.group(1).strip() if val_m else ""
        low = label.lower()
        if "expiratory" in low or "pef" in low:
            short_test = "PEF"
        elif "eosinophil" in low:
            short_test = "Eosinophils"
        elif "creatine kinase" in low or " ck" in low:
            short_test = "CK"
        elif "creatinine" in low:
            short_test = "Creatinine"
        elif "aldosterone" in low:
            short_test = "Aldosterone"
        elif "potassium" in low:
            short_test = "K+"
        elif "a1c" in low:
            short_test = "HbA1c"
        elif "hemoglobin" in low:
            short_test = "Hgb"
        elif "nt-probnp" in low or "bnp" in low:
            short_test = "NT-proBNP"
        else:
            raw_test = re.sub(r"^Abnormal\s+", "", label.split(":")[0], flags=re.IGNORECASE)
            raw_test = re.sub(r"\s*\([^)]*\)", "", raw_test).strip()
            short_test = raw_test[:18]
        return f"{short_test} {val_part}".strip()
    return label[:28]


def _compact_med_name(med_name: str) -> str:
    """Shortens verbose RxNorm display names into clean bedside chart shorthand."""
    s = re.sub(r"^24\s*HR\s+", "", str(med_name).strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(
        r"\s+(?:Extended\s+Release|Delayed\s+Release|Oral\s+Tablet|Oral\s+Capsule|Chewable\s+Tablet|Tablet|Capsule|hydrochloride).*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # Keep brand/ingredient + primary strength or concise ingredient
    s = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:MG|MCG|ML|G)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:MG|MCG|ML|G))?.*$", "", s, flags=re.IGNORECASE)
    return s.strip() or str(med_name).strip()[:24]


def build_competing_hypotheses(
    mapped_symptoms: list[dict[str, Any]],
    unlocked_stage: int = 1,
    postal_code: str = "",
    ruled_out_medications: set[str] | None = None,
) -> dict[str, Any]:
    """
    Aggregates per-symptom evidence into 2-3 ranked Root-Cause Hypothesis Cards
    (Medication/Regimen, Chronic EHR Baseline, and Hyperlocal ZIP Outbreak) plus
    top-level diagnostic coverage metrics for the Competing Hypotheses Scorecard UI.
    """
    ruled_out = {m.strip() for m in (ruled_out_medications or set()) if m}
    total = len(mapped_symptoms)

    ehr_explains: list[dict[str, Any]] = []
    ehr_gaps: list[dict[str, Any]] = []
    ehr_causes_counter: dict[str, int] = {}

    drug_explains: list[dict[str, Any]] = []
    drug_gaps: list[dict[str, Any]] = []
    drug_causes_counter: dict[str, int] = {}

    zip_explains: list[dict[str, Any]] = []
    zip_gaps: list[dict[str, Any]] = []
    zip_causes_counter: dict[str, int] = {}

    overall_explained = 0
    overall_unexplained_chips: list[str] = []

    for sym in mapped_symptoms:
        chip = sym.get("short_label") or _make_short_label(sym)
        sev = sym.get("severity", "MODERATE")
        is_red_flag = sev in ("CRITICAL", "HIGH") or bool(sym.get("infectious_candidate"))

        # 1. EHR Baseline
        ehr_m = sym.get("ehr_matches", [])
        if ehr_m:
            top_ehr = ehr_m[0]
            ehr_causes_counter[top_ehr["finding"]] = ehr_causes_counter.get(top_ehr["finding"], 0) + 1
            ehr_explains.append(
                {
                    "symptom_id": sym["symptom_id"],
                    "short_label": chip,
                    "label": sym["label"],
                    "detail": f"{top_ehr['finding']} — {top_ehr['rationale']}",
                    "citation": top_ehr["citation"],
                }
            )
        else:
            ehr_gaps.append(
                {
                    "symptom_id": sym["symptom_id"],
                    "short_label": chip,
                    "label": sym["label"],
                    "is_red_flag": is_red_flag,
                }
            )

        # 2. Medication / Adverse Effects (filtered dynamically by physician ruled-out drugs)
        raw_drug_m = sym.get("drug_matches", []) if unlocked_stage >= 3 else []
        eff_drug_m = [m for m in raw_drug_m if m["medication"] not in ruled_out]
        has_eff_drug = len(eff_drug_m) > 0
        if unlocked_stage >= 3:
            if has_eff_drug:
                for dm in eff_drug_m[:2]:
                    short_med = _compact_med_name(dm["medication"])
                    drug_causes_counter[short_med] = drug_causes_counter.get(short_med, 0) + 1
                top_dm = eff_drug_m[0]
                drug_explains.append(
                    {
                        "symptom_id": sym["symptom_id"],
                        "short_label": chip,
                        "label": sym["label"],
                        "detail": f"{_compact_med_name(top_dm['medication'])}: {top_dm['rationale']}",
                        "citation": top_dm["citation"],
                    }
                )
            else:
                drug_gaps.append(
                    {
                        "symptom_id": sym["symptom_id"],
                        "short_label": chip,
                        "label": sym["label"],
                        "is_red_flag": is_red_flag,
                    }
                )

        # 3. Hyperlocal ZIP Outbreak / Environmental
        out_m = sym.get("outbreak_matches", []) if unlocked_stage >= 4 else []
        if unlocked_stage >= 4:
            if out_m:
                top_out = out_m[0]
                zip_causes_counter[top_out["pathogen_or_factor"]] = (
                    zip_causes_counter.get(top_out["pathogen_or_factor"], 0) + 1
                )
                zip_explains.append(
                    {
                        "symptom_id": sym["symptom_id"],
                        "short_label": chip,
                        "label": sym["label"],
                        "resolves_drug_gap": not has_eff_drug,
                        "detail": f"{top_out['pathogen_or_factor']} — {top_out['rationale']}",
                        "citation": top_out["citation"],
                    }
                )
            else:
                zip_gaps.append(
                    {
                        "symptom_id": sym["symptom_id"],
                        "short_label": chip,
                        "label": sym["label"],
                        "is_red_flag": False,
                    }
                )

        # Overall explained at current unlocked stage
        explained_now = bool(ehr_m) or has_eff_drug or bool(out_m)
        if explained_now:
            overall_explained += 1
        else:
            overall_unexplained_chips.append(chip)

    top_ehr_names = [k for k, _ in sorted(ehr_causes_counter.items(), key=lambda x: -x[1])[:2]]
    ehr_summary = " + ".join(top_ehr_names) if top_ehr_names else "No Matching Chronic Condition on Problem List"

    top_drug_names = [k for k, _ in sorted(drug_causes_counter.items(), key=lambda x: -x[1])[:3]]
    if unlocked_stage < 3:
        drug_summary = "Active, Held & OTC Regimen (RxNorm + OpenFDA)"
    elif top_drug_names:
        drug_summary = " + ".join(top_drug_names)
    elif ruled_out:
        drug_summary = "Candidate Medications Ruled Out by Physician"
    else:
        drug_summary = "No Active/Stopped Medication Matches Presentation"

    top_zip_names = [k for k, _ in sorted(zip_causes_counter.items(), key=lambda x: -x[1])[:2]]
    zip_label_prefix = f"ZIP {postal_code}" if postal_code else "Local ZIP"
    if unlocked_stage < 4:
        zip_summary = f"{zip_label_prefix} ED Respiratory, Wastewater PCR & US AQI"
    elif top_zip_names:
        zip_summary = f"{zip_label_prefix}: " + " + ".join(top_zip_names)
    else:
        zip_summary = f"No Matching Infectious Outbreak Signal in {zip_label_prefix}"

    # Fit classifications
    if unlocked_stage < 3:
        drug_fit_level = "LOCKED"
        drug_badge = "Unlock in Step 3"
    elif total > 0 and len(drug_explains) == total:
        drug_fit_level = "STRONG"
        drug_badge = f"Explains All {len(drug_explains)}/{total}"
    elif len(drug_explains) > 0:
        drug_fit_level = "PARTIAL"
        drug_badge = f"Partial Fit · {len(drug_explains)}/{total}"
    else:
        drug_fit_level = "NONE"
        drug_badge = f"No Fit · 0/{total}"

    ehr_fit_level = "STRONG" if total > 0 and len(ehr_explains) == total else ("BASELINE" if ehr_explains else "NONE")
    ehr_badge = f"Baseline · {len(ehr_explains)}/{total}"

    resolves_gaps_cnt = sum(1 for z in zip_explains if z.get("resolves_drug_gap"))
    if unlocked_stage < 4:
        zip_fit_level = "GATED"
        zip_badge = "Gated · Step 4"
    elif resolves_gaps_cnt > 0:
        zip_fit_level = "RESOLVES_GAPS"
        zip_badge = f"Resolves {resolves_gaps_cnt} Drug Gap{'s' if resolves_gaps_cnt != 1 else ''} · {len(zip_explains)}/{total}"
    elif zip_explains:
        zip_fit_level = "PARTIAL"
        zip_badge = f"Local Signal · {len(zip_explains)}/{total}"
    else:
        zip_fit_level = "NONE"
        zip_badge = f"No Match · 0/{total}"

    cards_by_id: dict[str, dict[str, Any]] = {
        "drug_ae": {
            "id": "drug_ae",
            "domain_title": "Medication / Adverse-Effect Etiology",
            "source_tag": "NIH RxNorm · OpenFDA · Med Rec",
            "state": "LOCKED" if unlocked_stage < 3 else "ACTIVE",
            "culprit_summary": drug_summary,
            "fit_level": drug_fit_level,
            "fit_badge": drug_badge,
            "explained_count": len(drug_explains),
            "total_count": total,
            "explains": drug_explains,
            "cannot_explain": drug_gaps,
        },
        "ehr_baseline": {
            "id": "ehr_baseline",
            "domain_title": "Chronic EHR Baseline & Care Gaps",
            "source_tag": "HL7 FHIR R4 Conditions · Labs · Loops",
            "state": "ACTIVE",
            "culprit_summary": ehr_summary,
            "fit_level": ehr_fit_level,
            "fit_badge": ehr_badge,
            "explained_count": len(ehr_explains),
            "total_count": total,
            "explains": ehr_explains,
            "cannot_explain": ehr_gaps,
        },
        "zip_outbreak": {
            "id": "zip_outbreak",
            "domain_title": f"{zip_label_prefix} Outbreak & Environmental Trigger",
            "source_tag": "CDC NSSP · NWSS PCR · Open-Meteo AQI",
            "state": "GATED" if unlocked_stage < 4 else "ACTIVE",
            "culprit_summary": zip_summary,
            "fit_level": zip_fit_level,
            "fit_badge": zip_badge,
            "explained_count": len(zip_explains),
            "resolves_drug_gaps_count": resolves_gaps_cnt,
            "total_count": total,
            "explains": zip_explains,
            "cannot_explain": zip_gaps,
        },
    }

    if unlocked_stage < 3:
        ordered_ids = ["ehr_baseline", "drug_ae", "zip_outbreak"]
    elif unlocked_stage == 3:
        ordered_ids = ["drug_ae", "ehr_baseline", "zip_outbreak"]
    else:
        if resolves_gaps_cnt >= 2 or (resolves_gaps_cnt > 0 and len(zip_explains) >= len(drug_explains)):
            ordered_ids = ["zip_outbreak", "ehr_baseline", "drug_ae"]
        else:
            ordered_ids = sorted(
                ["drug_ae", "ehr_baseline", "zip_outbreak"],
                key=lambda cid: -cards_by_id[cid]["explained_count"],
            )

    coverage_pct = int(round((overall_explained / total) * 100)) if total else 100
    drug_coverage_pct = int(round((len(drug_explains) / total) * 100)) if total and unlocked_stage >= 3 else 0

    return {
        "total_findings": total,
        "overall_explained": overall_explained,
        "overall_unexplained_count": total - overall_explained,
        "overall_unexplained_chips": overall_unexplained_chips,
        "coverage_pct": coverage_pct,
        "drug_explained_count": len(drug_explains),
        "drug_unexplained_count": len(drug_gaps),
        "drug_unexplained_chips": [g["short_label"] for g in drug_gaps],
        "drug_coverage_pct": drug_coverage_pct,
        "resolves_drug_gaps_count": resolves_gaps_cnt,
        "hypotheses": [cards_by_id[cid] for cid in ordered_ids],
        "cards_by_id": cards_by_id,
    }


def map_symptoms_across_stages(
    patient_id: str,
    unlocked_stage: int = 1,
) -> dict[str, Any]:
    """
    Maps a patient's extracted symptoms progressively across unlocked clinical workflow stages:
      - `unlocked_stage = 1` or `2`: Maps only against Longitudinal EHR Compartments & Swimlane context.
      - `unlocked_stage = 3`: Adds Drug & Adverse Effects mapping (NIH RxNorm + OpenFDA) and flags
        any Unmatched / Residual Symptoms for the Physician Validation Gate.
      - `unlocked_stage >= 4`: Adds 5-Digit ZIP-Code Outbreak & Environmental Infection mapping.

    Args:
        patient_id: Patient FHIR ID (`pat-001`..`pat-007`) or name.
        unlocked_stage: Highest clinical step currently unlocked in the UI (1, 2, 3, or 4).

    Returns:
        Structured dictionary with competing root-cause hypotheses, per-symptom mappings, and coverage KPIs.
    """
    return compute_symptom_mappings(patient_id=patient_id, unlocked_stage=unlocked_stage)


