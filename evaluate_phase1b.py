#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG Phase 1b Full Evaluation Script:
  - Evaluates 7 Systems with Greedy Decoding (T=0.0):
      1. Echo-the-Query (Control)
      2. Oracle-Span (Control)
      3. Direct Gen (No RAG)
      4. Vanilla Chunk RAG
      5. DR-RAG (Ahmad et al., 2026)
      6. Zero-Shot ReAct Agent
      7. Urdu-DynRAG (Ours)
  - Strict Metric Suite with Disaggregated Answerable (n=82) vs Unanswerable (n=19) reporting.
  - 5,000 Bootstrap Resampling Iterations over 19 test passages + 6 parametric units.
  - Paired Difference CIs vs Vanilla Chunk RAG.
  - Comprehensive Headroom Analysis (No-Context, Top-1 Retrieved, Gold-Passage, Gold-Span) per Style.
  - Saves all raw CSV records and Markdown summary tables.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from phase1_indexer import DualFAISSRetriever
from phase2_state_extractor import UrduRAGStateExtractor
from phase3_env import reformulate_query
from phase4_train import DQNPolicy
from src.generator import UnifiedGenerator
from src.metrics import (
    compute_key_fact_accuracy,
    compute_grounded_retrieval_metrics,
    compute_abstention_metrics,
    compute_passage_bootstrap_ci,
    compute_paired_bootstrap_difference_ci,
    detect_abstention,
    detect_language_leakage,
    detect_polarity_and_contradiction,
    tokenize_text,
)


def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize_text(pred)
    gold_tokens = tokenize_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = sum(min(count, gold_counts.get(t, 0)) for t, count in pred_counts.items())
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(gold_tokens)
    return (2.0 * p * r) / (p + r)


def run_benchmark(backend: str = "openrouter") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Load Dataset
    data_path = Path("data/urdu_dynrag_verified_dataset_v3.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    test_items = [d for d in dataset if d.get("split") == "test"]
    print(f"[Phase 1b] Loaded {len(test_items)} test items from {data_path}.")

    # 2. Initialize Retriever & DQN Policy
    retriever = DualFAISSRetriever(indices_dir="models/indices")
    state_extractor = UrduRAGStateExtractor(retriever=retriever, generator_mode="auto")
    state_extractor.load_or_precompute_cache(
        dataset_items=test_items,
        cache_file="models/indices/state_cache_test.pt",
        batch_size=32,
    )

    policy_net = DQNPolicy()
    policy_path = Path("models/policy_best.pt")
    policy_net.load_state_dict(torch.load(policy_path, weights_only=True))
    policy_net.eval()

    # 3. Initialize Unified Generator with Greedy Decoding (T=0.0)
    greedy_cache_file = Path("cache/benchmark_predictions_greedy.json")
    generator = UnifiedGenerator(
        backend=backend,
        cache_path=greedy_cache_file,
        max_new_tokens=80,
    )
    print(f"[Phase 1b] Generator initialized with backend='{backend}', max_new_tokens=80, greedy decoding (T=0.0).")

    systems = [
        "Echo-the-Query (Control)",
        "Oracle-Span (Control)",
        "Direct Gen (No RAG)",
        "Vanilla Chunk RAG",
        "DR-RAG (Ahmad et al., 2026)",
        "Zero-Shot ReAct Agent",
        "Urdu-DynRAG (Ours)",
    ]

    all_records: List[Dict[str, Any]] = []

    # Pre-index passages for fast gold-context lookup
    passages_by_id = {}
    corpus_path = Path("data/urdu_dynrag_verified_corpus_v3.json")
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
            for c in corpus_data:
                passages_by_id[c["passage_id"]] = c["text"]

    for sys_name in systems:
        print(f"\nEvaluating: {sys_name} ...")
        t0 = time.time()
        for idx, item in enumerate(test_items, 1):
            qid = item["id"]
            query = item["query"]
            gold_ans = item.get("gold_answer", "")
            gold_span = item.get("gold_span", "")
            gold_ctx = item.get("gold_context", "") or passages_by_id.get(item.get("passage_id", ""), "")
            target_pid = item.get("passage_id", "")
            q_style = item.get("query_style", "general")
            key_entities = item.get("key_entities", [])
            is_s5 = (q_style == "style_5_hard_distractor")

            retrieved_pids = []
            retrieved_text = ""
            action = 0

            if sys_name == "Echo-the-Query (Control)":
                pred_ans = query
                action = 0
            elif sys_name == "Oracle-Span (Control)":
                pred_ans = gold_span if gold_span else gold_ans
                retrieved_pids = [target_pid] if target_pid else []
                action = 0
            elif sys_name == "Direct Gen (No RAG)":
                action = 0
                pred_ans = generator.generate(query, context=None, temperature=0.0, do_sample=False)
            elif sys_name == "Vanilla Chunk RAG":
                action = 1
                raw_res = retriever.search_raw(query, k=5)
                if raw_res:
                    retrieved_pids = [r["passage_id"] for r in raw_res]
                    retrieved_text = raw_res[0]["text"]
                pred_ans = generator.generate(query, context=retrieved_text, temperature=0.0, do_sample=False)
            elif sys_name == "DR-RAG (Ahmad et al., 2026)":
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
                pred_ans = generator.generate(query, context=retrieved_text, temperature=0.0, do_sample=False)
            elif sys_name == "Zero-Shot ReAct Agent":
                react_prompt = (
                    f"You are an AI search planner. For this query: '{query}', choose one tool:\n"
                    f"0: [DIRECT] (parametric memory, no search)\n"
                    f"1: [RAW_RAG] (search raw documents)\n"
                    f"2: [QA_RAG] (search question bank)\n"
                    f"3: [REFORMULATE] (clean slang/translate Roman Urdu)\n"
                    f"Reply ONLY with the number 0, 1, 2, or 3:"
                )
                raw_act = generator.generate(react_prompt, context=None, max_new_tokens=5, temperature=0.0, do_sample=False)
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
                pred_ans = generator.generate(query, context=retrieved_text if retrieved_text else None, temperature=0.0, do_sample=False)
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
                elif action == 3:
                    ref_query = reformulate_query(query)
                    raw_res = retriever.search_raw(ref_query, k=5)
                    if raw_res:
                        retrieved_pids = [r["passage_id"] for r in raw_res]
                        retrieved_text = raw_res[0]["text"]
                pred_ans = generator.generate(query, context=retrieved_text if retrieved_text else None, temperature=0.0, do_sample=False)

            # Compute strictly standardized metrics
            key_fact_acc = compute_key_fact_accuracy(
                prediction=pred_ans,
                gold_answer=gold_ans,
                gold_span=gold_span,
                key_entities=key_entities,
                query=query,
                is_unanswerable=is_s5,
            )
            token_f1 = compute_token_f1(pred_ans, gold_ans)
            is_abs = detect_abstention(pred_ans)
            leak = detect_language_leakage(pred_ans)
            pol, contra = detect_polarity_and_contradiction(pred_ans)
            ans_tokens = tokenize_text(pred_ans)

            is_hit1 = (len(retrieved_pids) > 0 and retrieved_pids[0] == target_pid)
            is_hit5 = (target_pid in retrieved_pids[:5])
            rr = 0.0
            if target_pid in retrieved_pids:
                rr = 1.0 / (retrieved_pids.index(target_pid) + 1)

            rec = {
                "system": sys_name,
                "query_id": qid,
                "passage_id": target_pid,
                "query_style": q_style,
                "is_unanswerable": is_s5,
                "action": action,
                "prediction": pred_ans,
                "gold_answer": gold_ans,
                "gold_span": gold_span,
                "key_fact_acc": key_fact_acc,
                "token_f1": token_f1,
                "is_abstained": is_abs,
                "has_leakage": leak,
                "polarity": pol,
                "has_contradiction": contra,
                "answer_token_length": len(ans_tokens),
                "is_hit1": is_hit1,
                "is_hit5": is_hit5,
                "reciprocal_rank": rr,
                "retrieved_pids": retrieved_pids,
            }
            all_records.append(rec)

            if idx % 20 == 0 or idx == len(test_items):
                print(f"  [{idx}/{len(test_items)}] Evaluated (Elapsed: {time.time()-t0:.1f}s)")

    df_records = pd.DataFrame(all_records)
    records_csv = Path("experiments/phase1b_benchmark_records.csv")
    records_csv.parent.mkdir(parents=True, exist_ok=True)
    df_records.to_csv(records_csv, index=False, encoding="utf-8")
    print(f"\n[Phase 1b] Saved {len(df_records)} benchmark records to {records_csv}.")

    # ==============================================================================
    # 4. Headroom Analysis per Style (Answerable Queries, n=82)
    # ==============================================================================
    print("\nRunning Headroom Analysis across styles on answerable test items...")
    answerable_items = [d for d in test_items if d.get("query_style") != "style_5_hard_distractor"]
    headroom_records = []

    for item in answerable_items:
        qid = item["id"]
        query = item["query"]
        gold_ans = item.get("gold_answer", "")
        gold_span = item.get("gold_span", "")
        gold_ctx = item.get("gold_context", "") or passages_by_id.get(item.get("passage_id", ""), "")
        target_pid = item.get("passage_id", "")
        q_style = item.get("query_style", "general")
        key_entities = item.get("key_entities", [])

        # Cond A: No-context
        ans_a = generator.generate(query, context=None, temperature=0.0, do_sample=False)
        acc_a = compute_key_fact_accuracy(ans_a, gold_ans, gold_span, key_entities, query, False)

        # Cond B: Top-1 retrieved context (Vanilla RAG)
        raw_res = retriever.search_raw(query, k=1)
        top1_text = raw_res[0]["text"] if raw_res else ""
        ans_b = generator.generate(query, context=top1_text, temperature=0.0, do_sample=False)
        acc_b = compute_key_fact_accuracy(ans_b, gold_ans, gold_span, key_entities, query, False)

        # Cond C: Gold-passage context (Oracle retrieval)
        ans_c = generator.generate(query, context=gold_ctx, temperature=0.0, do_sample=False)
        acc_c = compute_key_fact_accuracy(ans_c, gold_ans, gold_span, key_entities, query, False)

        # Cond D: Gold-span only (Oracle generator)
        ans_d = gold_span if gold_span else gold_ans
        acc_d = compute_key_fact_accuracy(ans_d, gold_ans, gold_span, key_entities, query, False)

        headroom_records.append({
            "query_id": qid,
            "passage_id": target_pid,
            "query_style": q_style,
            "acc_no_context": acc_a,
            "acc_top1_retrieved": acc_b,
            "acc_gold_passage": acc_c,
            "acc_gold_span": acc_d,
            "retrieval_gain": acc_b - acc_a,
            "retrieval_gap": acc_c - acc_b,
            "generation_gap": acc_d - acc_c,
            "pred_no_context": ans_a,
            "pred_top1_retrieved": ans_b,
            "pred_gold_passage": ans_c,
            "pred_gold_span": ans_d,
        })

    df_headroom = pd.DataFrame(headroom_records)
    headroom_csv = Path("experiments/phase1b_headroom_records.csv")
    df_headroom.to_csv(headroom_csv, index=False, encoding="utf-8")
    print(f"[Phase 1b] Saved {len(df_headroom)} headroom records to {headroom_csv}.")

    # Aggregate Headroom Table per style
    style_names = {
        "style_1_clean_synthetic": "S1 (Clean Synthetic)",
        "style_2_conversational": "S2 (Conversational)",
        "style_3_code_switched": "S3 (Code-Switched)",
        "style_4_roman_urdu": "S4 (Roman Urdu)",
        "parametric_memory": "Parametric",
    }
    headroom_summary = []
    unique_styles = sorted(df_headroom["query_style"].unique())
    for st in unique_styles + ["OVERALL"]:
        sub = df_headroom if st == "OVERALL" else df_headroom[df_headroom["query_style"] == st]
        n_st = len(sub)
        headroom_summary.append({
            "Style": "Overall Answerable" if st == "OVERALL" else style_names.get(st, st),
            "n": n_st,
            "No-Context (%)": f"{sub['acc_no_context'].mean()*100:.1f}",
            "Top-1 Retrieved (%)": f"{sub['acc_top1_retrieved'].mean()*100:.1f}",
            "Gold Passage (%)": f"{sub['acc_gold_passage'].mean()*100:.1f}",
            "Gold Span (Oracle) (%)": f"{sub['acc_gold_span'].mean()*100:.1f}",
            "Retrieval Gap (%)": f"{(sub['acc_gold_passage'].mean() - sub['acc_top1_retrieved'].mean())*100:+.1f}",
            "Generation Gap (%)": f"{(sub['acc_gold_span'].mean() - sub['acc_gold_passage'].mean())*100:+.1f}",
        })
    df_headroom_table = pd.DataFrame(headroom_summary)
    df_headroom_table.to_csv("experiments/phase1b_headroom_table.csv", index=False, encoding="utf-8")
    df_headroom_table.to_markdown("experiments/phase1b_headroom_table.md", index=False)

    # ==============================================================================
    # 5. Table 1: Comprehensive Summary Table with 5,000 Bootstrap CIs
    # ==============================================================================
    print("\nComputing 5,000 Bootstrap CIs and disaggregated Table 1...")
    summary_rows = []
    vanilla_ans_records = [r for r in all_records if r["system"] == "Vanilla Chunk RAG" and not r["is_unanswerable"]]

    for sys_name in systems:
        sys_all = [r for r in all_records if r["system"] == sys_name]
        sys_ans = [r for r in sys_all if not r["is_unanswerable"]]
        sys_grounded = [r for r in sys_ans if r["passage_id"]]
        sys_s5 = [r for r in sys_all if r["is_unanswerable"]]

        # 5,000 Bootstrap iterations over 19 passages + 6 parametric items
        obs_acc, ci_low, ci_high = compute_passage_bootstrap_ci(
            records=sys_ans,
            metric_key="key_fact_acc",
            n_bootstrap=5000,
            seed=42,
        )

        obs_f1, f1_low, f1_high = compute_passage_bootstrap_ci(
            records=sys_ans,
            metric_key="token_f1",
            n_bootstrap=5000,
            seed=42,
        )

        # Retrieval metrics (n=76)
        ret_metrics = compute_grounded_retrieval_metrics(sys_all)
        # Abstention metrics
        abs_metrics = compute_abstention_metrics(sys_all)

        # Answer length median and IQR
        lens = [r["answer_token_length"] for r in sys_all]
        med_len = int(np.median(lens)) if lens else 0
        iqr_len = int(np.percentile(lens, 75) - np.percentile(lens, 25)) if lens else 0

        # Language leakage
        leak_rate = (sum(1 for r in sys_all if r["has_leakage"]) / len(sys_all)) * 100.0

        summary_rows.append({
            "System": sys_name,
            "Key-Fact Acc (%) [95% CI] (n=82)": f"{obs_acc*100:.1f} [{ci_low*100:.1f}, {ci_high*100:.1f}]",
            "Token F1 (%) [95% CI] (n=82)": f"{obs_f1*100:.1f} [{f1_low*100:.1f}, {f1_high*100:.1f}]",
            "False Abs (%) (n=82)": f"{abs_metrics['false_abstention_rate']*100:.1f}",
            "Hit@1 (%) (n=76)": f"{ret_metrics['hit1']*100:.1f}",
            "Hit@5 (%) (n=76)": f"{ret_metrics['hit5']*100:.1f}",
            "MRR (n=76)": f"{ret_metrics['mrr']:.3f}",
            "S5 Match (%) (n=19)": f"{ret_metrics['s5_source_match_rate']*100:.1f}",
            "S5 Abs F1 (%) (n=19)": f"{abs_metrics['abstention_f1']*100:.1f}",
            "Lang Leak (%) (n=101)": f"{leak_rate:.1f}",
            "Ans Len (Med, IQR)": f"{med_len} ({iqr_len})",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("experiments/phase1b_summary_table.csv", index=False, encoding="utf-8")
    df_summary.to_markdown("experiments/phase1b_summary_table.md", index=False)

    # ==============================================================================
    # 6. Paired Bootstrap Differences vs Vanilla Chunk RAG (5,000 Iterations)
    # ==============================================================================
    paired_diff_rows = []
    for sys_name in systems:
        if sys_name in ["Echo-the-Query (Control)", "Oracle-Span (Control)"]:
            continue
        sys_ans = [r for r in all_records if r["system"] == sys_name and not r["is_unanswerable"]]

        diff_acc, dacc_low, dacc_high = compute_paired_bootstrap_difference_ci(
            system_records=sys_ans,
            baseline_records=vanilla_ans_records,
            metric_key="key_fact_acc",
            n_bootstrap=5000,
            seed=42,
        )

        diff_f1, df1_low, df1_high = compute_paired_bootstrap_difference_ci(
            system_records=sys_ans,
            baseline_records=vanilla_ans_records,
            metric_key="token_f1",
            n_bootstrap=5000,
            seed=42,
        )

        sig_acc = "No" if (dacc_low <= 0.0 <= dacc_high) else "Yes"
        sig_f1 = "No" if (df1_low <= 0.0 <= df1_high) else "Yes"

        paired_diff_rows.append({
            "System": sys_name,
            "Δ Key-Fact Acc vs Vanilla (%) [95% CI]": f"{diff_acc*100:+.2f} [{dacc_low*100:+.2f}, {dacc_high*100:+.2f}]",
            "Acc Sig (p<0.05)?": sig_acc,
            "Δ Token F1 vs Vanilla (%) [95% CI]": f"{diff_f1*100:+.2f} [{df1_low*100:+.2f}, {df1_high*100:+.2f}]",
            "F1 Sig (p<0.05)?": sig_f1,
        })

    df_paired = pd.DataFrame(paired_diff_rows)
    df_paired.to_csv("experiments/phase1b_paired_diff_table.csv", index=False, encoding="utf-8")
    df_paired.to_markdown("experiments/phase1b_paired_diff_table.md", index=False)

    print("\n[Phase 1b] Evaluation Complete. Output artifacts ready in experiments/ directory.")
    return df_records, df_summary, df_paired, df_headroom_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="openrouter", choices=["openrouter", "local"])
    args = parser.parse_args()
    run_benchmark(backend=args.backend)

