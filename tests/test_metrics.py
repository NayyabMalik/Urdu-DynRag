import json
import pytest
from pathlib import Path

from src.metrics import (
    normalize_urdu_text,
    strip_boilerplate,
    tokenize_text,
    detect_language_leakage,
    detect_abstention,
    detect_polarity_and_contradiction,
    compute_key_fact_accuracy,
    compute_grounded_retrieval_metrics,
    compute_abstention_metrics,
)


def test_digit_normalization():
    # Eastern Arabic-Indic (Urdu)
    assert normalize_urdu_text("۱۲۳۴۵") == "12345"
    # Arabic-Indic
    assert normalize_urdu_text("١٢٣٤٥") == "12345"
    # Comma removal in numbers
    assert normalize_urdu_text("6,000 روپے") == "6000 روپے"
    assert normalize_urdu_text("1,000") == "1000"


def test_orthographic_variants_and_diacritics():
    # Yeh and Kaf variants
    assert normalize_urdu_text("كتاب") == "کتاب"
    assert normalize_urdu_text("علي") == "علی"
    # Diacritics stripping
    assert normalize_urdu_text("قِشْرٌ") == "قشر"
    assert normalize_urdu_text("مُعَاوِن") == "معاون"


def test_number_words_mapping():
    assert normalize_urdu_text("چھ ماہ") == "6 ماہ"
    assert normalize_urdu_text("چار نشستیں") == "4 نشستیں"
    assert normalize_urdu_text("پندرہ دن") == "15 دن"
    assert normalize_urdu_text("پانچ سال") == "5 سال"


def test_boilerplate_stripping():
    assert strip_boilerplate("جواب: چھ ماہ") == "چھ ماہ"
    assert strip_boilerplate(". چار") == "چار"
    assert strip_boilerplate(". . \n\nجواب میں سیاق و سباق میں کوئی معلومات نہیں ہے۔") == "جواب میں سیاق و سباق میں کوئی معلومات نہیں ہے۔"


def test_language_leakage():
    assert detect_language_leakage("مجرم娜，根据您提供的上下文") is True
    assert detect_language_leakage("جواب: 6000 روپے") is False
    assert detect_language_leakage("Привет мир") is True


def test_abstention_detection():
    # Documented valid abstention phrases
    assert detect_abstention("سیاق و سباق میں اس کی کوئی تفصیل نہیں ہے۔") is True
    assert detect_abstention("معلومات دستیاب نہیں ہے۔") is True
    assert detect_abstention("کوئی معلومات نہیں ہے۔") is True
    assert detect_abstention("کوئی آخری تاریخ نہیں ہے۔") is True
    # Bare 'نہیں' must NOT be detected as abstention
    assert detect_abstention("نہیں") is False
    assert detect_abstention("نہیں، نادرا کے مطابق یہ لازمی نہیں ہے۔") is False


def test_polarity_and_contradiction():
    # Clear negative
    pol, contra = detect_polarity_and_contradiction("نہیں، یہ لازمی نہیں ہے۔")
    assert pol == "negative"
    assert contra is False

    # Clear positive
    pol, contra = detect_polarity_and_contradiction("جی ہاں، یہ لازمی ہے۔")
    assert pol == "positive"
    assert contra is False

    # Self-contradiction
    pol, contra = detect_polarity_and_contradiction(
        "یہ فریقین پر لازمی ہے اور لازم ہوتا ہے۔ نہیں، یہ parties کے لیے binding نہیں ہوتا ہے۔"
    )
    assert contra is True


def test_grounded_retrieval_denominator():
    sample_records = [
        {"query_style": "style_1_clean_synthetic", "passage_id": "doc1", "retrieved_pids": ["doc1", "doc2"]},
        {"query_style": "style_2_conversational", "passage_id": "doc2", "retrieved_pids": ["doc3", "doc2"]},
        {"query_style": "style_3_code_switched", "passage_id": "doc3", "retrieved_pids": ["doc3"]},
        {"query_style": "style_4_roman_urdu", "passage_id": "doc4", "retrieved_pids": ["doc5"]},
        {"query_style": "style_5_hard_distractor", "passage_id": "doc6", "retrieved_pids": ["doc6"]}, # Excluded
        {"query_style": "parametric_nastaliq", "passage_id": "", "retrieved_pids": ["doc1"]}, # Excluded
    ]
    res = compute_grounded_retrieval_metrics(sample_records)
    # Grounded count must be exactly 4 (styles 1-4)
    assert res["grounded_n"] == 4
    # Hit@1: doc1 (hit), doc2 (miss), doc3 (hit), doc4 (miss) -> 2/4 = 0.5
    assert res["hit1"] == 0.5
    # Hit@5: doc1 (hit), doc2 (hit), doc3 (hit), doc4 (miss) -> 3/4 = 0.75
    assert res["hit5"] == 0.75
    # S5 source match rate: 1/1 = 1.0
    assert res["s5_n"] == 1
    assert res["s5_source_match_rate"] == 1.0


def test_golden_25_agreement():
    golden_path = Path("tests/golden_25.json")
    assert golden_path.exists(), "golden_25.json must exist!"

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_items = json.load(f)

    assert len(golden_items) == 25

    agreements = 0
    disagreements = []

    for item in golden_items:
        qid = item["id"]
        q_style = item["style"]
        query = item["query"]
        gold_ans = item["gold_answer"]
        gold_span = item.get("gold_span")
        pred_ans = item["pred_answer"]
        hand_label = item["label"]
        is_s5 = (q_style == "style_5_hard_distractor")

        # Metric evaluation
        acc = compute_key_fact_accuracy(
            prediction=pred_ans,
            gold_answer=gold_ans,
            gold_span=gold_span,
            query=query,
            is_unanswerable=is_s5,
        )
        is_abstained = detect_abstention(pred_ans)
        is_leaked = detect_language_leakage(pred_ans)
        _, is_contra = detect_polarity_and_contradiction(pred_ans)

        # Decide automated category
        if is_leaked:
            metric_decision = "language_leakage"
        elif is_contra:
            metric_decision = "self_contradiction"
        elif is_s5:
            metric_decision = "correct_abstention" if is_abstained else "hallucinated_on_unanswerable"
        elif is_abstained:
            metric_decision = "false_abstention"
        elif acc >= 0.5:
            metric_decision = "correct"
        else:
            metric_decision = "incorrect"

        # Check binary alignment: does metric match hand label classification?
        # Hand labels mapping:
        # Correct category: correct, correct_but_verbose
        # Abstention category: correct_abstention, false_abstention
        # Failure category: wrong_fact, wrong_clause, retrieval_miss, language_leakage, hallucinated_on_unanswerable, self_contradiction
        is_hand_correct = hand_label in {"correct", "correct_but_verbose", "correct_abstention"}
        is_metric_correct = (acc >= 0.5)

        agrees = (is_hand_correct == is_metric_correct)
        if agrees:
            agreements += 1
        else:
            disagreements.append({
                "id": qid,
                "style": q_style,
                "hand_label": hand_label,
                "metric_acc": acc,
                "metric_decision": metric_decision,
                "pred": pred_ans,
            })

    agreement_pct = (agreements / 25.0) * 100.0
    print(f"\n[Golden 25 Test] Agreement: {agreements}/25 ({agreement_pct:.1f}%)")
    if disagreements:
        print("Disagreements noted:")
        for d in disagreements:
            print(f"  ID {d['id']} ({d['style']}): Hand='{d['hand_label']}' vs MetricAcc={d['metric_acc']} (Decision='{d['metric_decision']}')")

    # Strict requirement: agreement >= 96.0% (at least 24 / 25)
    assert agreement_pct >= 96.0, f"Golden agreement was {agreement_pct:.1f}%, expected >= 96.0%!"

