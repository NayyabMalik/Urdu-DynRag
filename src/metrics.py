#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Advanced Evaluation Metrics Module (Phase 1)
Reference: Implementation Specification & Operational Architecture

Key Features:
  1. Key-Fact Accuracy (Primary Metric):
     - Normalizes Urdu/Arabic-Indic digits, commas, yeh/kaf variants, diacritics, and number words.
     - Strips introductory boilerplates ("جواب:", leading bullets, "سیاق و سباق کے مطابق...").
     - Extracts factual entity anchors (numbers, durations, currency, named entities).
     - Handles polarity (جی ہاں / نہیں) and detects self-contradictions.
  2. Grounded Retrieval Metrics:
     - Hit@1, Hit@5, MRR strictly computed over answerable grounded queries (S1-S4; n=76 in test split).
     - Explicit diagnostic reporting for S5 unanswerable distractors (top-1 passage == source passage).
  3. Grounded Abstention Metrics:
     - Regex-based abstention detection using documented phrase patterns from real outputs (no bare 'نہیں').
     - Precision, Recall, and F1 on unanswerable (S5) queries.
     - False-abstention rate on answerable queries (S1-S4 + parametric).
  4. Diagnostics & Controls:
     - Language leakage detection (CJK / Cyrillic script drift).
     - Prediction length statistics.
     - Echo-the-query and Oracle-span baselines under the single canonical tokenizer.
     - Macro (per-passage) and Micro (per-query) aggregation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# ==============================================================================
# 1. Unicode & Orthographic Dictionaries
# ==============================================================================

# Map Arabic-Indic and Eastern Arabic-Indic digits to ASCII standard Latin digits
DIGIT_MAP = {
    # Eastern Arabic-Indic (Urdu/Persian)
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    # Arabic-Indic
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

# Urdu written number words to standard numeric strings
URDU_NUMBER_WORDS = {
    "ایک": "1",
    "دو": "2",
    "تین": "3",
    "چار": "4",
    "پانچ": "5",
    "چھ": "6",
    "سات": "7",
    "آٹھ": "8",
    "نو": "9",
    "دس": "10",
    "گیارہ": "11",
    "بارہ": "12",
    "تیرہ": "13",
    "چودہ": "14",
    "پندرہ": "15",
    "سولہ": "16",
    "سترہ": "17",
    "اٹھارہ": "18",
    "انیس": "19",
    "بیس": "20",
    "پچیس": "25",
    "تیس": "30",
    "پینتیس": "35",
    "چالیس": "40",
    "پینتالیس": "45",
    "پچاس": "50",
    "ساٹھ": "60",
    "ستر": "70",
    "اسی": "80",
    "نوے": "90",
    "سو": "100",
    "ہزار": "1000",
    "لاکھ": "100000",
    "کروڑ": "10000000",
}

# Documented abstention phrase patterns taken from real generator outputs
ABSTENTION_PATTERNS = [
    r"معلومات\s*(دستیاب|موجود|فراہم)\s*نہیں",
    r"کوئی\s*(تفصیل|معلومات|ذکر|ریکارڈ|آخری\s*تاریخ)\s*نہیں",
    r"ذکر\s*(موجود\s*)?نہیں\s*(ہے|تھا)?",
    r"سیاق\s*(و\s*سباق)?\s*میں\s*.*نہیں\s*(ہے|ملتا|ملتی|آتا|ذکر)",
    r"درج\s*نہیں\s*(ہے|تھا)",
    r"بیان\s*نہیں\s*کیا\s*گیا",
    r"عبارت\s*میں\s*.*(ذکر|معلومات)\s*نہیں",
    r"جواب\s*موجود\s*نہیں\s*ہے",
    r"کوئی\s*معلومات\s*نہیں",
    r"کوئی\s*تفصیل\s*نہیں",
]

ABSTENTION_REGEX = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)

# Boilerplate patterns to strip from predictions
BOILERPLATE_PATTERNS = [
    r"^[\s\.\:\-۔،•\*]+",
    r"^(جواب|answer)\s*[:：\-۔]\s*",
    r"^(سیاق\s*و\s*سباق\s*(میں|کے\s*مطابق)?|سیاق\s*کے\s*مطابق)\s*[:：\-۔،]\s*",
    r"^(درج\s*ذیل|فراہم\s*کردہ\s*(عبارت|متن)\s*کے\s*مطابق)\s*[:：\-۔،]\s*",
    r"^(سوال\s*(میں)?\s*[:：\-۔])\s*",
    r"^[\s\.\:\-۔،•\*]+",
]


# ==============================================================================
# 2. Text Normalization & Tokenization
# ==============================================================================

def strip_boilerplate(text: str) -> str:
    """Strip common introductory model scaffolding and leading punctuation."""
    cleaned = text.strip()
    changed = True
    while changed:
        orig = cleaned
        for pat in BOILERPLATE_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        changed = (cleaned != orig)
    return cleaned


def normalize_urdu_text(text: str, map_number_words: bool = True) -> str:
    """
    Standardize Urdu orthography, numerals, diacritics, and character variants.
    """
    if not text:
        return ""

    s = text.strip()

    # 1. Unicode Normalization (NFC)
    s = unicodedata.normalize("NFC", s)

    # 2. Map Yeh and Kaf variants
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")

    # 3. Strip diacritics (Aerab) and tatweel
    s = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", s)

    # 4. Map digits (Eastern Arabic-Indic and Arabic-Indic to Latin 0-9)
    for orig_d, latin_d in DIGIT_MAP.items():
        s = s.replace(orig_d, latin_d)

    # 5. Remove commas in numbers (e.g. 6,000 -> 6000)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # 6. Map isolated Urdu number words to numbers
    if map_number_words:
        tokens = s.split()
        normalized_tokens = []
        for t in tokens:
            clean_t = re.sub(r"[^\w\u0600-\u06FF]", "", t)
            if clean_t in URDU_NUMBER_WORDS:
                normalized_tokens.append(URDU_NUMBER_WORDS[clean_t])
            else:
                normalized_tokens.append(t)
        s = " ".join(normalized_tokens)

    # 7. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_text(text: str) -> List[str]:
    """
    Single canonical tokenizer for all evaluation metrics.
    Preserves alphanumeric, Latin, and Perso-Arabic unicode characters.
    """
    norm = normalize_urdu_text(text, map_number_words=False)
    clean = re.sub(r"[^\w\s\u0600-\u06FF]", " ", norm.lower())
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 0]
    return tokens


# ==============================================================================
# 3. Linguistic & Behavioral Diagnostics
# ==============================================================================

def detect_language_leakage(text: str) -> bool:
    """
    Check if the prediction contains unwanted foreign scripts (CJK, Cyrillic).
    """
    # CJK Unified Ideographs: \u4E00-\u9FFF, Cyrillic: \u0400-\u04FF
    cjk_cyrillic = re.compile(r"[\u4E00-\u9FFF\u0400-\u04FF]")
    return bool(cjk_cyrillic.search(text))


def detect_abstention(text: str) -> bool:
    """
    Check if the prediction actively expresses abstention / unanswerability.
    Uses strict documented multi-token phrase patterns (never bare 'نہیں').
    """
    clean = strip_boilerplate(text)
    # Never match bare 'نہیں'
    if clean.strip() == "نہیں":
        return False
    return bool(ABSTENTION_REGEX.search(clean))


def detect_polarity_and_contradiction(prediction: str) -> Tuple[Optional[str], bool]:
    """
    Extract polar answer stance ('positive', 'negative', or None) and
    detect self-contradiction where both affirmative and negative are asserted.
    """
    clean = strip_boilerplate(prediction)
    toks = clean.split()
    first_few = " ".join(toks[:8]) if toks else ""

    pos_pattern = re.compile(r"(جی\s*ہاں|\bہاں\b|لازمی\s*ہے|ضروری\s*ہے|لازم\s*ہوتا\s*ہے|binding\s*ہوتا\s*ہے)", re.IGNORECASE)
    neg_pattern = re.compile(r"(\bنہیں\b|لازمی\s*نہیں|ضروری\s*نہیں|لازم\s*نہیں|binding\s*نہیں)", re.IGNORECASE)

    has_pos = bool(pos_pattern.search(clean))
    has_neg = bool(neg_pattern.search(clean))

    # Self-contradiction: asserts both binding and not binding, or yes and no
    contradiction = False
    if has_pos and has_neg:
        if ("لازم ہوتا ہے" in clean and "لازم نہیں" in clean) or \
           ("binding ہوتا ہے" in clean and "binding نہیں" in clean) or \
           ("لازمی ہے" in clean and "لازمی نہیں" in clean) or \
           ("لازم ہوتا ہے" in clean and "binding نہیں" in clean):
            contradiction = True

    # Polarity detection from the primary sentence head
    polarity = None
    if has_neg and not has_pos:
        polarity = "negative"
    elif has_pos and not has_neg:
        polarity = "positive"
    elif has_neg and has_pos:
        # Check first token
        if neg_pattern.search(first_few):
            polarity = "negative"
        elif pos_pattern.search(first_few):
            polarity = "positive"

    return polarity, contradiction


# ==============================================================================
# 4. Key-Fact Extraction & Accuracy (Primary Metric)
# ==============================================================================

def extract_key_facts_from_gold(
    gold_answer: str,
    gold_span: Optional[str] = None,
    key_entities: Optional[List[str]] = None,
) -> Set[str]:
    """
    Extract canonical factual units (numbers, dates, currency, named entities).
    """
    facts = set()

    # Priority 1: Use explicitly tagged key entities if provided
    if key_entities:
        for ent in key_entities:
            norm_ent = normalize_urdu_text(str(ent), map_number_words=True)
            if len(norm_ent) > 0:
                facts.add(norm_ent)

    # Priority 2: Extract numeric anchors, quantities, and durations
    full_text = f"{gold_answer} {gold_span or ''}"
    norm_full = normalize_urdu_text(full_text, map_number_words=True)

    # Find all standalone numbers and compounds (e.g. 6 ماہ, 5 سال, 6000 روپے, 4 خواتین)
    num_matches = re.findall(r"\b\d+\b", norm_full)
    for n in num_matches:
        facts.add(n)

    # Common Urdu legal/factual duration and currency collocations
    units = ["سال", "ماہ", "مہینے", "دن", "روپے", "روپئے", "نشستیں", "ہڈیاں", "قید", "جرمانہ", "قشر"]
    for u in units:
        # e.g., 6 ماہ or 5 سال
        matches = re.findall(rf"\b\d+\s*{u}\b", norm_full)
        for m in matches:
            facts.add(m)
        if u in norm_full and any(u == t for t in tokenize_text(norm_full)):
            facts.add(u)

    # If no numbers found, fall back to core noun/proper entity tokens from gold_answer
    if not facts:
        # Content tokens with stopwords removed
        ur_stopwords = {
            "کی", "کے", "کا", "کو", "ہے", "ہیں", "سے", "پر", "میں", "اور",
            "تھا", "تھی", "تھے", "ہو", "ہوتا", "ہوتی", "ہوتے", "یہ", "وہ",
            "اس", "ان", "نے", "لیے", "گے", "گی", "گا", "کر", "کرنا", "ہونا"
        }
        tokens = tokenize_text(norm_full)
        content_tokens = [t for t in tokens if t not in ur_stopwords and len(t) > 1]
        for ct in content_tokens:
            facts.add(ct)

    return facts


def compute_key_fact_accuracy(
    prediction: str,
    gold_answer: str,
    gold_span: Optional[str] = None,
    key_entities: Optional[List[str]] = None,
    query: str = "",
    is_unanswerable: bool = False,
) -> float:
    """
    Compute Key-Fact Accuracy: fraction of canonical gold facts recovered in prediction.
    Enforces polarity constraints on yes/no questions and handles abstention on S5.
    """
    # Case 1: Unanswerable distractor (S5)
    if is_unanswerable:
        # Correct if model properly abstains and does NOT hallucinate
        if detect_abstention(prediction) and not detect_language_leakage(prediction):
            return 1.0
        return 0.0

    # Case 2: Model abstained on an answerable question -> False abstention
    if detect_abstention(prediction):
        return 0.0

    # Case 3: Language leakage (e.g. Chinese tokens) -> Invalid answer
    if detect_language_leakage(prediction):
        return 0.0

    clean_pred = strip_boilerplate(prediction)
    norm_pred = normalize_urdu_text(clean_pred, map_number_words=True)

    # Check for self-contradiction
    pred_polarity, is_contradiction = detect_polarity_and_contradiction(clean_pred)
    if is_contradiction:
        return 0.0

    # Polarity check for yes/no questions
    gold_norm = normalize_urdu_text(gold_answer, map_number_words=True)
    gold_polarity, _ = detect_polarity_and_contradiction(gold_answer)
    if gold_polarity is not None:
        if pred_polarity != gold_polarity:
            return 0.0

    # Extract gold facts
    gold_facts = extract_key_facts_from_gold(gold_answer, gold_span, key_entities)
    if not gold_facts:
        return 1.0 if clean_pred else 0.0

    # Count how many gold facts are present in normalized prediction
    matched = 0
    for fact in gold_facts:
        if fact in norm_pred:
            matched += 1
        else:
            # Check if all tokens of compound fact exist
            fact_tokens = tokenize_text(fact)
            if fact_tokens and all(ft in tokenize_text(norm_pred) for ft in fact_tokens):
                matched += 1

    score = float(matched) / float(len(gold_facts))
    return float(min(1.0, max(0.0, score)))


# ==============================================================================
# 5. Retrieval & Abstention Metrics
# ==============================================================================

def compute_grounded_retrieval_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute Hit@1, Hit@5, and MRR strictly over answerable grounded queries (S1-S4).
    Report S5 source-passage alignment separately as a diagnostic.
    """
    # Answerable grounded queries: styles S1 to S4
    grounded_styles = {"style_1_clean_synthetic", "style_2_conversational", "style_3_code_switched", "style_4_roman_urdu"}
    grounded_records = [r for r in records if r.get("query_style") in grounded_styles and r.get("passage_id")]
    n_grounded = len(grounded_records)

    hit1_count = 0
    hit5_count = 0
    reciprocal_ranks = []

    for r in grounded_records:
        gold_pid = r.get("passage_id")
        retrieved_pids = r.get("retrieved_pids", [])
        if not retrieved_pids and r.get("retrieved_pid"):
            retrieved_pids = [r.get("retrieved_pid")]

        if retrieved_pids and retrieved_pids[0] == gold_pid:
            hit1_count += 1
        if gold_pid in retrieved_pids[:5]:
            hit5_count += 1

        if gold_pid in retrieved_pids:
            rank = retrieved_pids.index(gold_pid) + 1
            reciprocal_ranks.append(1.0 / float(rank))
        else:
            reciprocal_ranks.append(0.0)

    # Diagnostic for S5: does retrieved passage match original source passage?
    s5_records = [r for r in records if r.get("query_style") == "style_5_hard_distractor"]
    s5_source_matches = 0
    for r in s5_records:
        src_pid = r.get("passage_id")
        retrieved_pids = r.get("retrieved_pids", [])
        if not retrieved_pids and r.get("retrieved_pid"):
            retrieved_pids = [r.get("retrieved_pid")]
        if src_pid and retrieved_pids and retrieved_pids[0] == src_pid:
            s5_source_matches += 1

    return {
        "grounded_n": n_grounded,
        "hit1": (hit1_count / float(n_grounded)) if n_grounded > 0 else 0.0,
        "hit5": (hit5_count / float(n_grounded)) if n_grounded > 0 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "s5_n": len(s5_records),
        "s5_source_match_rate": (s5_source_matches / float(len(s5_records))) if s5_records else 0.0,
    }


def compute_abstention_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute Precision, Recall, and F1 on unanswerable (S5) queries,
    and false-abstention rate on answerable queries.
    """
    tp = 0  # Abstained on S5
    fp = 0  # Abstained on answerable (S1-S4, parametric)
    fn = 0  # Failed to abstain on S5
    tn = 0  # Answered on answerable

    for r in records:
        q_style = r.get("query_style", "")
        pred_text = r.get("pred_answer", "")
        abstained = detect_abstention(pred_text)
        is_s5 = (q_style == "style_5_hard_distractor")

        if is_s5:
            if abstained:
                tp += 1
            else:
                fn += 1
        else:
            if abstained:
                fp += 1
            else:
                tn += 1

    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
    false_abstention_rate = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "s5_precision": prec,
        "s5_recall": rec,
        "s5_f1": f1,
        "false_abstention_rate": false_abstention_rate,
        "s5_tp": tp,
        "s5_fn": fn,
        "answerable_fp": fp,
        "answerable_tn": tn,
    }


# ==============================================================================
# 6. Aggregators & Statistical Reporting
# ==============================================================================

def compute_passage_bootstrap_ci(
    records: List[Dict[str, Any]],
    metric_key: str = "key_fact_acc",
    num_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean and 95% bootstrap confidence intervals resampled at the PASSAGE level.
    """
    rng = np.random.RandomState(seed)

    # Group records by passage_id (or 'parametric' if empty)
    passage_groups: Dict[str, List[float]] = {}
    for r in records:
        pid = r.get("passage_id") or "parametric_unassigned"
        val = float(r.get(metric_key, 0.0))
        passage_groups.setdefault(pid, []).append(val)

    pids = list(passage_groups.keys())
    n_p = len(pids)
    if n_p == 0:
        return 0.0, 0.0, 0.0

    # Observed mean
    all_vals = [v for vals in passage_groups.values() for v in vals]
    observed_mean = float(np.mean(all_vals))

    boot_means = []
    for _ in range(num_resamples):
        sampled_pids = rng.choice(pids, size=n_p, replace=True)
        sample_vals = [v for pid in sampled_pids for v in passage_groups[pid]]
        boot_means.append(float(np.mean(sample_vals)))

    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return observed_mean, low, high
