import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import random
import re
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Setup cache directory
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
cache_file = cache_dir / "audit_llm_cache.json"
llm_cache = {}
if cache_file.exists():
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
    except Exception:
        llm_cache = {}

# Load dataset
with open("data/urdu_dynrag_verified_dataset_v3.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
test_items = [d for d in dataset if d.get("split") == "test"]

# Load raw metadata for text lookup
with open("models/indices/raw_metadata.json", "r", encoding="utf-8") as f:
    raw_meta = json.load(f)
with open("models/indices/dr_metadata.json", "r", encoding="utf-8") as f:
    qa_meta = json.load(f)

# Sample 25 items reproducible seed 42
random.seed(42)
sampled = random.sample(test_items, 25)

# Fast retriever search without loading heavy sentence transformer if possible,
# or load FAISS and BGE-M3
from phase1_indexer import DualFAISSRetriever
from phase3_env import FrozenAnswerGenerator

retriever = DualFAISSRetriever(indices_dir="models/indices")
generator = FrozenAnswerGenerator()

print(f"Generating responses for 25 audit samples...")
examples = []

for idx, item in enumerate(sampled, 1):
    qid = item["id"]
    query = item["query"]
    gold_answer = item.get("gold_answer", "")
    gold_span = item.get("gold_span", "")
    gold_pid = item.get("passage_id", "")
    q_style = item.get("query_style", "general")

    # Retrieve top-1 raw chunk
    raw_res = retriever.search_raw(query, k=1)
    retrieved_pid = raw_res[0]["passage_id"] if raw_res else None
    retrieved_text = raw_res[0]["text"] if raw_res else ""

    cache_key = f"{query}|||{retrieved_text[:100]}"
    if cache_key in llm_cache:
        pred_answer = llm_cache[cache_key]
    else:
        pred_answer = generator.generate_answer(query, context=retrieved_text)
        llm_cache[cache_key] = pred_answer
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(llm_cache, f, ensure_ascii=False, indent=2)

    # Categorize failure
    failures = []
    # 1. Retrieval mismatch
    if gold_pid and retrieved_pid != gold_pid:
        failures.append("retrieval_miss")
    # 2. English / Roman Urdu mismatch in output
    has_urdu = any("\u0600" <= c <= "\u06FF" for c in pred_answer)
    has_latin = any("a" <= c.lower() <= "z" for c in pred_answer)
    if q_style == "style_4_roman_urdu" and not has_urdu:
        failures.append("output_not_in_urdu_script")
    elif not has_urdu:
        failures.append("wrong_language")
    # 3. Gold span not in prediction when retrieval succeeded
    if gold_span and gold_span.strip() not in pred_answer:
        if gold_pid and retrieved_pid == gold_pid:
            failures.append("missed_gold_span_despite_correct_context")
    # 4. Too long / verbose
    if len(pred_answer.split()) > 45:
        failures.append("too_verbose")
    # 5. Over-abstention or false abstention
    if ("معلومات دستیاب نہیں" in pred_answer or "نہیں ہے" in pred_answer) and q_style != "style_5_hard_distractor":
        failures.append("false_abstention")
    # 6. Failed to abstain on distractor
    if q_style == "style_5_hard_distractor" and not ("معلومات" in pred_answer or "نہیں" in pred_answer):
        failures.append("hallucinated_on_distractor")

    if not failures:
        failures.append("correct_or_acceptable")

    examples.append({
        "sample_num": idx,
        "id": qid,
        "style": q_style,
        "query": query,
        "gold_pid": gold_pid,
        "retrieved_pid": retrieved_pid,
        "gold_span": gold_span,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "failure_categories": failures,
    })

with open("docs/audit_25_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print("Generated and saved 25 examples to docs/audit_25_examples.json")

