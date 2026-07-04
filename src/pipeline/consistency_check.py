"""
consistency_check.py
--------------------
Compares BioGPT report text against CheXNet pathology predictions.

MATCH    → status = CONSISTENT  → AUTO-APPROVE candidate
MISMATCH → status = INCONSISTENT → FLAG FOR REVIEW
"""

PATHOLOGY_KEYWORDS = {
    "normal"        : ["normal", "no acute", "unremarkable", "clear", "no evidence"],
    "effusion"      : ["effusion", "pleural fluid"],
    "pleural"       : ["pleural", "pleural thickening"],
    "pneumothorax"  : ["pneumothorax"],
    "consolidation" : ["consolidation", "consolidative"],
    "infiltrate"    : ["infiltrate", "infiltration"],
    "opacity"       : ["opacity", "opacification"],
    "atelectasis"   : ["atelectasis", "atelectatic", "collapse"],
    "edema"         : ["edema", "oedema", "vascular congestion"],
    "cardiomegaly"  : ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "nodule"        : ["nodule", "nodular"],
    "pneumonia"     : ["pneumonia", "pneumonic"],
    "fracture"      : ["fracture", "fractured"],
    "mass"          : [" mass ", "masses"],
    "calcification" : ["calcification", "calcified", "granuloma"],
    "emphysema"     : ["emphysema", "hyperinflation"],
    "hernia"        : ["hernia", "hiatal"],
    "fibrosis"      : ["fibrosis", "fibrotic", "scarring"],
}

# Per-class thresholds from ablation study evaluation
CLASSIFIER_THRESHOLDS = {
    "normal": 0.55, "effusion": 0.05, "pleural": 0.10,
    "pneumothorax": 0.10, "consolidation": 0.10, "infiltrate": 0.10,
    "opacity": 0.10, "atelectasis": 0.10, "edema": 0.10,
    "cardiomegaly": 0.10, "nodule": 0.10, "pneumonia": 0.10,
    "fracture": 0.10, "mass": 0.10, "calcification": 0.10,
    "emphysema": 0.10, "hernia": 0.10, "fibrosis": 0.05,
}

# Pathologies where mismatch is clinically significant
CLINICAL_PATHOLOGIES = {
    "effusion", "pneumothorax", "consolidation", "atelectasis",
    "edema", "cardiomegaly", "pneumonia", "mass", "emphysema",
}


def extract_report_pathologies(report_text: str) -> dict:
    text = report_text.lower()
    return {
        p: any(kw in text for kw in kws)
        for p, kws in PATHOLOGY_KEYWORDS.items()
    }


def check_consistency(report_text: str, classifier_probs: dict) -> dict:
    report_flags = extract_report_pathologies(report_text)
    matches, mismatches, details = [], [], {}

    for pathology, report_present in report_flags.items():
        prob     = classifier_probs.get(pathology, 0.0)
        clf_pos  = prob >= CLASSIFIER_THRESHOLDS[pathology]
        agree    = report_present == clf_pos

        details[pathology] = {
            "report_says"  : report_present,
            "classifier"   : round(prob, 3),
            "clf_positive" : clf_pos,
            "agree"        : agree,
        }
        (matches if agree else mismatches).append(pathology)

    clinical_mismatches = [p for p in mismatches if p in CLINICAL_PATHOLOGIES]
    status = "CONSISTENT" if not clinical_mismatches else "INCONSISTENT"

    return {
        "status"             : status,
        "matches"            : matches,
        "mismatches"         : mismatches,
        "clinical_mismatches": clinical_mismatches,
        "details"            : details,
    }
