#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Advanced Evaluation Metrics Module (Phase 1b)
Reference: Implementation Specification & Operational Architecture
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

# Common Roman Urdu to Urdu Nastaliq mappings to prevent query-word leakage
ROMAN_URDU_LEXICON = {
    "zameen": "زمین", "bahari": "بیرونی", "kaunsi": "کونسی", "tah": "تہہ",
    "hai": "ہے", "kya": "کیا", "kitni": "کتنی", "kitna": "کتنا", "kitne": "کتنے",
    "saza": "سزا", "eobi": "ای او بی آئی", "nadra": "نادرا", "passport": "پاسپورٹ",
    "license": "لائسنس", "challan": "چالان", "hec": "ایچ ای سی", "degree": "ڈگری"
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

UR_STOPWORDS = {
    "کی", "کے", "کا", "کو", "ہے", "ہیں", "سے", "پر", "میں", "اور",
    "تھا", "تھی", "تھے", "ہو", "ہوتا", "ہوتی", "ہوتے", "ہوا", "ہوئی", "ہوئے",
    "یہ", "وہ", "اس", "ان", "نے", "لیے", "گے", "گی", "گا", "کر", "کرنا", "ہونا",
    "تک", "بھی", "تو", "ہی", "یا", "کہ", "جس", "جن", "جو", "کسی", "کوئی",
    "سب", "بہت", "زیادہ", "کم", "نہیں", "ہاں", "جی", "والا", "والی", "والے",
    "کہلاتا", "کہلاتی", "کہلاتے", "کہتے", "کہتی", "کہا", "کہے", "کہوں",
    "جاتا", "جاتی", "جاتے", "گیا", "گئی", "گئے",
    "رکھا", "رکھی", "رکھے", "رہتا", "رہتی", "رہتے",
    "بنتا", "بنتی", "بنتے", "بنا", "بنی", "بنے",
    "مشتمل", "واقع", "موجود", "شامل", "حاصل",
    "مطابق", "تحت", "بارے", "ضمن", "باعث", "رو",
}

NUMERIC_UNITS = ["سال", "ماہ", "مہینے", "دن", "گھنٹے", "روپے", "روپئے", "نشستیں", "ہڈیاں", "فیصد", "%"]


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
    """Standardize Urdu orthography, numerals, diacritics, and character variants."""
    if not text:
        return ""

    s = text.strip()
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    s = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", s)

    for orig_d, latin_d in DIGIT_MAP.items():
        s = s.replace(orig_d, latin_d)

    s = re.sub(r"(?<=\d),(?=\d)", "", s)

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

    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_text(text: str) -> List[str]:
    """Single canonical tokenizer for all evaluation metrics."""
    norm = normalize_urdu_text(text, map_number_words=False)
    # Strip both ASCII and Unicode punctuation (including Arabic comma, semicolon, question mark, full stop)
    clean = re.sub(r"[\u060C\u061B\u061F\u06D4\u066A\u00AB\u00BB\u201C\u201D\u2018\u2019\u0640]", " ", norm)
    clean = re.sub(r"[^\w\s]|_", " ", clean)
    tokens = [t.strip() for t in clean.lower().split() if len(t.strip()) > 0]
    return tokens


# ==============================================================================
# 3. Linguistic & Behavioral Diagnostics
# ==============================================================================

def detect_language_leakage(text: str) -> bool:
    """Check if the prediction contains unwanted foreign scripts (CJK, Cyrillic)."""
    cjk_cyrillic = re.compile(r"[\u4E00-\u9FFF\u0400-\u04FF]")
    return bool(cjk_cyrillic.search(text))


def detect_abstention(text: str) -> bool:
    """Check if the prediction actively expresses abstention / unanswerability."""
    clean = strip_boilerplate(text)
    if clean.strip() == "نہیں":
        return False
    return bool(ABSTENTION_REGEX.search(clean))


def is_polar_query_or_answer(query: str, gold_answer: str) -> bool:
    """Determine if the item is genuinely a polar (yes/no) question."""
    q_norm = normalize_urdu_text(query)
    g_norm = normalize_urdu_text(gold_answer)
    wh_words = ["کتنے", "کتنی", "کتنا", "کب", "کہاں", "کون", "کونسا", "کونسی", "کیوں", "کس"]
    if any(re.search(rf"\b{w}\b", q_norm) for w in wh_words):
        return False
    q_tokens = tokenize_text(q_norm)
    if q_tokens and q_tokens[0] == "کیا":
        return True
    first_few = " ".join(tokenize_text(g_norm)[:3])
    if re.search(r"^(جی\s*ہاں|ہاں|نہیں)\b", first_few):
        return True
    return False


def detect_polarity_and_contradiction(prediction: str) -> Tuple[Optional[str], bool]:
    clean = strip_boilerplate(prediction)
    toks = clean.split()
    first_few = " ".join(toks[:8]) if toks else ""

    pos_pattern = re.compile(r"(جی\s*ہاں|\bہاں\b|لازمی\s*ہے|ضروری\s*ہے|لازم\s*ہوتا\s*ہے|binding\s*ہوتا\s*ہے)", re.IGNORECASE)
    neg_pattern = re.compile(r"(\bنہیں\b|لازمی\s*نہیں|ضروری\s*نہیں|لازم\s*نہیں|binding\s*نہیں)", re.IGNORECASE)

    has_pos = bool(pos_pattern.search(clean))
    has_neg = bool(neg_pattern.search(clean))

    contradiction = False
    if has_pos and has_neg:
        if ("لازم ہوتا ہے" in clean and "لازم نہیں" in clean) or \
           ("binding ہوتا ہے" in clean and "binding نہیں" in clean) or \
           ("لازمی ہے" in clean and "لازمی نہیں" in clean) or \
           ("لازم ہوتا ہے" in clean and "binding نہیں" in clean):
            contradiction = True

    polarity = None
    if has_neg and not has_pos:
        polarity = "negative"
    elif has_pos and not has_neg:
        polarity = "positive"
    elif has_neg and has_pos:
        if neg_pattern.search(first_few):
            polarity = "negative"
        elif pos_pattern.search(first_few):
            polarity = "positive"

    return polarity, contradiction


# ==============================================================================
# 4. Strict Key-Fact Extraction & Accuracy (Primary Metric)
# ==============================================================================

def extract_key_facts_from_gold(
    gold_answer: str,
    gold_span: Optional[str] = None,
    key_entities: Optional[List[str]] = None,
    query: str = "",
) -> Dict[str, Any]:
    span_text = gold_span.strip() if gold_span and gold_span.strip() else gold_answer.strip()
    norm_span = normalize_urdu_text(span_text, map_number_words=True)
    norm_ans = normalize_urdu_text(gold_answer, map_number_words=True)
    norm_query = normalize_urdu_text(query, map_number_words=True)

    query_tokens = set(tokenize_text(norm_query))
    for r_word, u_word in ROMAN_URDU_LEXICON.items():
        if r_word in query.lower():
            query_tokens.add(u_word)

    facts: Dict[str, Any] = {
        "numeric": [],       # list of dicts: {"num": str, "unit": str|None}
        "named_entities": [] # list of strings
    }

    # 1. Numeric facts
    unit_pattern = r"\b(\d+)\s*(" + "|".join(NUMERIC_UNITS) + r")\b"
    span_num_units = re.findall(unit_pattern, norm_span)
    ans_num_units = re.findall(unit_pattern, norm_ans) if norm_ans else []
    target_num_units = ans_num_units if ans_num_units else span_num_units

    matched_nums = set()
    for num, unit in target_num_units:
        if num not in query_tokens:
            facts["numeric"].append({"num": num, "unit": unit})
            matched_nums.add(num)

    # Standalone numbers
    span_nums = re.findall(r"\b\d+\b", norm_span)
    ans_nums = re.findall(r"\b\d+\b", norm_ans) if norm_ans else span_nums
    target_nums = [n for n in (ans_nums if ans_nums else span_nums) if n not in matched_nums and n not in query_tokens]
    for n in target_nums:
        facts["numeric"].append({"num": n, "unit": None})
        matched_nums.add(n)

    # 2. Named entities / key phrases from gold_span only if no numeric facts
    if not facts["numeric"]:
        # Extract named entities strictly from gold_span / gold_answer
        span_tokens = set(tokenize_text(norm_span)) if norm_span else set()
        ans_tokens = tokenize_text(norm_ans)
        cand_tokens = [
            t for t in ans_tokens
            if t not in UR_STOPWORDS and t not in query_tokens and len(t) > 1
        ]
        generic_terms = {"سیاق", "سباق", "جواب", "درج", "ذیل", "معلومات", "قانون", "قواعد", "عبارت", "متن"}
        cand_tokens = [t for t in cand_tokens if t not in generic_terms]
        overlap = [t for t in cand_tokens if t in span_tokens]
        distinctive = overlap if overlap else cand_tokens
        if distinctive:
            facts["named_entities"].extend(distinctive[:3])

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
    Compute Key-Fact Accuracy:
    - Answerable queries: Fraction of gold factual entities recovered in prediction.
      Numeric facts require exact number match. Generic words give no partial credit.
    - Unanswerable queries (S5): 1.0 if properly abstains without language leakage; else 0.0.
      (Note: S5 results must be reported separately from S1-S4 answerable accuracy).
    """
    if is_unanswerable:
        if detect_abstention(prediction) and not detect_language_leakage(prediction):
            return 1.0
        # Also check common Urdu refusal phrasing in truncated responses
        if any(term in prediction for term in ["سیاق و سباق میں نہیں", "ذیل سیاق و سباق", "معلومات دستیاب نہیں", "ذکر موجود نہیں"]):
            return 1.0
        return 0.0

    if detect_abstention(prediction):
        return 0.0
    if detect_language_leakage(prediction):
        return 0.0

    clean_pred = strip_boilerplate(prediction)
    norm_pred = normalize_urdu_text(clean_pred, map_number_words=True)
    pred_tokens = set(tokenize_text(norm_pred))

    # Polar stance check for genuine polar questions
    is_polar = is_polar_query_or_answer(query, gold_answer)

    facts = extract_key_facts_from_gold(
        gold_answer=gold_answer,
        gold_span=gold_span,
        key_entities=key_entities,
        query=query,
    )
    num_facts = facts["numeric"]
    ent_facts = facts["named_entities"]

    # For pure polar questions with no numeric facts: matching the polar stance gives 1.0
    if is_polar and not num_facts:
        pred_pol, is_contra = detect_polarity_and_contradiction(clean_pred)
        gold_pol, _ = detect_polarity_and_contradiction(gold_answer)
        if is_contra or gold_pol is None:
            return 0.0
        return 1.0 if pred_pol == gold_pol else 0.0

    if is_polar:
        pred_pol, is_contra = detect_polarity_and_contradiction(clean_pred)
        gold_pol, _ = detect_polarity_and_contradiction(gold_answer)
        if is_contra:
            return 0.0
        if gold_pol is not None and pred_pol != gold_pol:
            return 0.0

    total = len(num_facts) + len(ent_facts)
    if total == 0:
        return 1.0 if clean_pred else 0.0

    matched = 0
    # Numeric facts must match the exact number
    for nf in num_facts:
        num = nf["num"]
        if num in pred_tokens or re.search(rf"\b{num}\b", norm_pred):
            matched += 1

    # Named entity facts
    for ef in ent_facts:
        if ef in norm_pred:
            matched += 1
        else:
            ef_tokens = tokenize_text(ef)
            if ef_tokens and any(t in pred_tokens for t in ef_tokens):
                matched += 1

    return min(1.0, float(matched) / float(total))


# ==============================================================================
# 5. Retrieval & Abstention Metrics
# ==============================================================================

def compute_grounded_retrieval_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    grounded_styles = {
        "style_1_clean_synthetic",
        "style_2_conversational",
        "style_3_code_switched",
        "style_4_roman_urdu",
    }
    eval_records = [
        r for r in records
        if r.get("query_style") in grounded_styles and r.get("passage_id")
    ]
    n = len(eval_records)

    hit1_count = 0
    hit5_count = 0
    reciprocal_ranks = []

    for r in eval_records:
        gold_pid = r.get("passage_id")
        retrieved = r.get("retrieved_pids", [])
        if retrieved:
            if retrieved[0] == gold_pid:
                hit1_count += 1
            if gold_pid in retrieved[:5]:
                hit5_count += 1
            if gold_pid in retrieved:
                rank = retrieved.index(gold_pid) + 1
                reciprocal_ranks.append(1.0 / float(rank))
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

    s5_records = [r for r in records if r.get("query_style") == "style_5_hard_distractor"]
    s5_matches = sum(1 for r in s5_records if r.get("retrieved_pids") and r.get("retrieved_pids")[0] == r.get("passage_id"))
    s5_match_rate = float(s5_matches) / float(len(s5_records)) if s5_records else 0.0

    return {
        "hit1": float(hit1_count) / float(n) if n > 0 else 0.0,
        "hit5": float(hit5_count) / float(n) if n > 0 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "n": n,
        "grounded_n": n,
        "s5_n": len(s5_records),
        "s5_source_match_rate": s5_match_rate,
    }


def compute_abstention_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    s5_records = [r for r in records if r.get("query_style") == "style_5_hard_distractor"]
    answerable_records = [r for r in records if r.get("query_style") != "style_5_hard_distractor"]

    tp = sum(1 for r in s5_records if r.get("is_abstained", False))
    fn = len(s5_records) - tp
    fp = sum(1 for r in answerable_records if r.get("is_abstained", False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    false_abstention_rate = fp / len(answerable_records) if answerable_records else 0.0

    return {
        "abstention_precision": float(precision),
        "abstention_recall": float(recall),
        "abstention_f1": float(f1),
        "false_abstention_rate": float(false_abstention_rate),
        "s5_n": len(s5_records),
        "answerable_n": len(answerable_records),
    }


# ==============================================================================
# 6. Bootstrap Resampling & Paired Difference CIs (5,000 Iterations)
# ==============================================================================

def compute_passage_bootstrap_ci(
    records: List[Dict[str, Any]],
    metric_key: str = "key_fact_acc",
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0

    passage_groups: Dict[str, List[float]] = {}
    for r in records:
        pid = r.get("passage_id") or f"parametric_{r.get('query_id')}"
        if pid not in passage_groups:
            passage_groups[pid] = []
        passage_groups[pid].append(float(r.get(metric_key, 0.0)))

    unique_pids = list(passage_groups.keys())
    m = len(unique_pids)
    if m == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(unique_pids, size=m, replace=True)
        sampled_vals = []
        for p in sampled_pids:
            sampled_vals.extend(passage_groups[p])
        bootstrap_means.append(float(np.mean(sampled_vals)))

    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    obs_mean = float(np.mean([r.get(metric_key, 0.0) for r in records]))
    ci_low = float(np.percentile(bootstrap_means, low_p))
    ci_high = float(np.percentile(bootstrap_means, high_p))

    return obs_mean, ci_low, ci_high


def compute_paired_bootstrap_difference_ci(
    system_records: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]],
    metric_key: str = "key_fact_acc",
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    base_map = {r["query_id"]: float(r.get(metric_key, 0.0)) for r in baseline_records}
    
    passage_diff_groups: Dict[str, List[float]] = {}
    for r in system_records:
        qid = r["query_id"]
        pid = r.get("passage_id") or f"parametric_{qid}"
        sys_val = float(r.get(metric_key, 0.0))
        base_val = base_map.get(qid, 0.0)
        diff = sys_val - base_val
        if pid not in passage_diff_groups:
            passage_diff_groups[pid] = []
        passage_diff_groups[pid].append(diff)

    unique_pids = list(passage_diff_groups.keys())
    m = len(unique_pids)
    if m == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    bootstrap_diff_means = []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(unique_pids, size=m, replace=True)
        sampled_diffs = []
        for p in sampled_pids:
            sampled_diffs.extend(passage_diff_groups[p])
        bootstrap_diff_means.append(float(np.mean(sampled_diffs)))

    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    all_diffs = [
        float(r.get(metric_key, 0.0)) - base_map.get(r["query_id"], 0.0)
        for r in system_records if r["query_id"] in base_map
    ]
    obs_diff = float(np.mean(all_diffs)) if all_diffs else 0.0
    ci_low = float(np.percentile(bootstrap_diff_means, low_p))
    ci_high = float(np.percentile(bootstrap_diff_means, high_p))

    return obs_diff, ci_low, ci_high
