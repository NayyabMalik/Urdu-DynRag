import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from phase1_indexer import DualFAISSRetriever
from phase2_state_extractor import UrduRAGStateExtractor
from phase3_env import FrozenAnswerGenerator, reformulate_query
from phase4_train import DQNPolicy
import torch

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
    compute_passage_bootstrap_ci,
)

# Load dataset
with open("data/urdu_dynrag_verified_dataset_v3.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
test_items = [d for d in dataset if d.get("split") == "test"]
print(f"Loaded {len(test_items)} test items.")

# Setup persistent LLM cache
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
cache_file = cache_dir / "benchmark_predictions.json"
llm_cache = {}
if cache_file.exists():
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
        print(f"Loaded {len(llm_cache)} cached LLM completions.")
    except Exception:
        llm_cache = {}

# Also load from audit cache if available
audit_cache_file = cache_dir / "audit_llm_cache.json"
if audit_cache_file.exists():
    try:
        with open(audit_cache_file, "r", encoding="utf-8") as f:
            audit_cache = json.load(f)
            for k, v in audit_cache.items():
                if k not in llm_cache:
                    llm_cache[k] = v
    except Exception:
        pass

# Initialize dependencies
retriever = DualFAISSRetriever(indices_dir="models/indices")
state_extractor = UrduRAGStateExtractor(retriever=retriever, generator_mode="auto")
# Load test state cache for instant feature lookup
state_extractor.load_or_precompute_cache(
    dataset_items=test_items,
    cache_file="models/indices/state_cache_test.pt",
    batch_size=32,
)

generator = FrozenAnswerGenerator()

policy_path = Path("models/policy_best.pt")
policy_net = DQNPolicy()
policy_net.load_state_dict(torch.load(policy_path, weights_only=True))
policy_net.eval()

def generate_cached(query: str, context: Optional[str] = None) -> str:
    ctx_key = (context or "").strip()[:150]
    key = f"{query.strip()}|||{ctx_key}"
    if key in llm_cache:
        return llm_cache[key]
    ans = generator.generate_answer(query, context=context)
    llm_cache[key] = ans
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(llm_cache, f, ensure_ascii=False, indent=2)
    return ans

# Define evaluation logic for each system
systems = [
    "Echo-the-Query (Control)",
    "Oracle-Span (Control)",
    "Direct Gen (No RAG)",
    "Vanilla Chunk RAG",
    "DR-RAG (Ahmad et al., 2026)",
    "Zero-Shot ReAct Agent",
    "Urdu-DynRAG (Ours)",
]

all_system_records = []

for sys_name in systems:
    print(f"\nEvaluating system: {sys_name}...")
    sys_records = []
    for item in test_items:
        qid = item["id"]
        query = item["query"]
        gold_ans = item.get("gold_answer", "")
        gold_span = item.get("gold_span", "")
        target_pid = item.get("passage_id", "")
        q_style = item.get("query_style", "general")
        key_entities = item.get("key_entities", [])
        is_s5 = (q_style == "style_5_hard_distractor")

        retrieved_pids = []
        retrieved_text = ""
        action = 0

        if sys_name == "Echo-the-Query (Control)":
            pred_y = query
        elif sys_name == "Oracle-Span (Control)":
            pred_y = gold_span if gold_span else gold_ans
            retrieved_pids = [target_pid] if target_pid else []
        elif sys_name == "Direct Gen (No RAG)":
            action = 0
            pred_y = generate_cached(query, context=None)
        elif sys_name == "Vanilla Chunk RAG":
            action = 1
            raw_res = retriever.search_raw(query, k=5)
            if raw_res:
                retrieved_pids = [r["passage_id"] for r in raw_res]
                retrieved_text = raw_res[0]["text"]
            pred_y = generate_cached(query, context=retrieved_text)
        elif sys_name == "DR-RAG (Ahmad et al., 2026)":
            # As implemented in results_v1: threshold 0.80 on qa_top1
            s_vec = state_extractor.extract_state(query=query, t=0).numpy()
            sim_qa_top1 = s_vec[4]
            action = 2 if sim_qa_top1 >= 0.80 else 1
            if action == 2:
                qa_res = retriever.search_qa(query, k=5)
                if qa_res:
                    retrieved_pids = [r["parent_passage_id"] for r in qa_res]
                    retrieved_text = qa_res[0]["parent_text"]
            else:
                raw_res = retriever.search_raw(query, k=5)
                if raw_res:
                    retrieved_pids = [r["passage_id"] for r in raw_res]
                    retrieved_text = raw_res[0]["text"]
            pred_y = generate_cached(query, context=retrieved_text)
        elif sys_name == "Zero-Shot ReAct Agent":
            # ReAct chooses tool via prompt
            react_prompt = (
                f"You are an AI search planner. For this query: '{query}', choose one tool:\n"
                f"0: [DIRECT] (parametric memory, no search)\n"
                f"1: [RAW_RAG] (search raw documents)\n"
                f"2: [QA_RAG] (search question bank)\n"
                f"3: [REFORMULATE] (clean slang/translate Roman Urdu)\n"
                f"Reply ONLY with the number 0, 1, 2, or 3:"
            )
            raw_act = generate_cached(react_prompt, context=None)
            m = re.search(r"[0-3]", raw_act)
            action = int(m.group(0)) if m else 1
            if action in [1, 3]:
                raw_res = retriever.search_raw(query, k=5)
                if raw_res:
                    retrieved_pids = [r["passage_id"] for r in raw_res]
                    retrieved_text = raw_res[0]["text"]
            elif action == 2:
                qa_res = retriever.search_qa(query, k=5)
                if qa_res:
                    retrieved_pids = [r["parent_passage_id"] for r in qa_res]
                    retrieved_text = qa_res[0]["parent_text"]
            pred_y = generate_cached(query, context=retrieved_text if retrieved_text else None)
        elif sys_name == "Urdu-DynRAG (Ours)":
            s_vec = state_extractor.extract_state(query=query, t=0).numpy()
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                action = int(np.argmax(q_vals))
            if action == 1:
                raw_res = retriever.search_raw(query, k=5)
                if raw_res:
                    retrieved_pids = [r["passage_id"] for r in raw_res]
                    retrieved_text = raw_res[0]["text"]
            elif action == 2:
                qa_res = retriever.search_qa(query, k=5)
                if qa_res:
                    retrieved_pids = [r["parent_passage_id"] for r in qa_res]
                    retrieved_text = qa_res[0]["parent_text"]
            pred_y = generate_cached(query, context=retrieved_text if retrieved_text else None)

        # Compute metrics under unified tokenizer & normalization
        key_fact_acc = compute_key_fact_accuracy(
            prediction=pred_y,
            gold_answer=gold_ans,
            gold_span=gold_span,
            key_entities=key_entities,
            query=query,
            is_unanswerable=is_s5,
        )

        p_tokens = tokenize_text(pred_y)
        g_tokens = tokenize_text(gold_ans)
        common = set(p_tokens).intersection(set(g_tokens))
        p = len(common) / len(p_tokens) if p_tokens else 0.0
        r = len(common) / len(g_tokens) if g_tokens else 0.0
        token_f1 = (2.0 * p * r) / (p + r + 1e-8) if (p + r) > 0 else 0.0

        is_abstained = detect_abstention(pred_y)
        is_leaked = detect_language_leakage(pred_y)
        pol, is_contra = detect_polarity_and_contradiction(pred_y)
        token_len = len(p_tokens)

        rec = {
            "system": sys_name,
            "query_id": qid,
            "passage_id": target_pid,
            "query_style": q_style,
            "query": query,
            "gold_answer": gold_ans,
            "pred_answer": pred_y,
            "action": action,
            "retrieved_pids": retrieved_pids,
            "key_fact_acc": key_fact_acc,
            "token_f1": token_f1,
            "token_len": token_len,
            "is_abstained": is_abstained,
            "is_leaked": is_leaked,
            "is_contra": is_contra,
        }
        sys_records.append(rec)
        all_system_records.append(rec)

# Save all raw records
df_all = pd.DataFrame(all_system_records)
df_all.to_csv("experiments/phase1_benchmark_records.csv", index=False)
print("\nSaved experiments/phase1_benchmark_records.csv")

# Compute summary table for Phase 1 report
summary_rows = []
for sys_name in systems:
    sub = [r for r in all_system_records if r["system"] == sys_name]
    
    # Key-fact acc with passage-level 95% bootstrap CI
    mean_kfa, low_ci, high_ci = compute_passage_bootstrap_ci(sub, metric_key="key_fact_acc")
    
    # Retrieval metrics (S1-S4 only, n=76)
    ret_res = compute_grounded_retrieval_metrics(sub)
    
    # Abstention metrics
    abs_res = compute_abstention_metrics(sub)
    
    # Diagnostic lengths & leaks
    lens = [r["token_len"] for r in sub]
    leak_rate = np.mean([1.0 if r["is_leaked"] else 0.0 for r in sub]) * 100.0
    f1_mean = np.mean([r["token_f1"] for r in sub]) * 100.0
    
    # Macro average (per passage)
    p_groups = {}
    for r in sub:
        pid = r["passage_id"] or "parametric"
        p_groups.setdefault(pid, []).append(r["key_fact_acc"])
    macro_kfa = np.mean([np.mean(vals) for vals in p_groups.values()]) * 100.0
    
    summary_rows.append({
        "System": sys_name,
        "Key-Fact Acc (%) [95% CI]": f"{mean_kfa*100:.1f} [{low_ci*100:.1f}, {high_ci*100:.1f}]",
        "Macro Fact Acc (%)": f"{macro_kfa:.1f}",
        "Token F1 (%)": f"{f1_mean:.1f}",
        "Hit@1 (%) [n=76]": f"{ret_res['hit1']*100:.1f}",
        "Hit@5 (%) [n=76]": f"{ret_res['hit5']*100:.1f}",
        "MRR [n=76]": f"{ret_res['mrr']:.3f}",
        "S5 Match [n=19]": f"{ret_res['s5_source_match_rate']*100:.1f}",
        "S5 Abs F1 (%)": f"{abs_res['s5_f1']*100:.1f}",
        "False Abs (%)": f"{abs_res['false_abstention_rate']*100:.1f}",
        "Lang Leak (%)": f"{leak_rate:.1f}",
        "Ans Len (Med, IQR)": f"{np.median(lens):.0f} ({np.percentile(lens, 75) - np.percentile(lens, 25):.0f})",
    })

sum_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 100)
print("PHASE 1 EVALUATION: KEY-FACT ACCURACY, RETRIEVAL & CONTROLS")
print("=" * 100)
print(sum_df.to_string(index=False))

sum_df.to_csv("experiments/phase1_summary_table.csv", index=False)
with open("experiments/phase1_summary_table.md", "w", encoding="utf-8") as f:
    f.write(sum_df.to_markdown(index=False))

print("\nSaved experiments/phase1_summary_table.csv and .md")
