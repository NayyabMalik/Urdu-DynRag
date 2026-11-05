#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Action-Outcome Lookup Table Builder (Phase 2)
Precomputes real outcomes for all 455 dataset items across all trajectories:
  - Trajectory '0': Direct Generation (context=None)
  - Trajectory '1': Top-1 D_raw on original query
  - Trajectory '2': Top-1 D_qa on original query
  - Trajectory '3->1': Reformulate query + Top-1 D_raw
  - Trajectory '3->2': Reformulate query + Top-1 D_qa

Persisted in cache/action_outcome_table.json with incremental checkpointing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from src.phase1_indexer import DualFAISSRetriever
from src.generator import UnifiedGenerator
from src.reformulator import DeterministicUrduReformulator
from src.metrics import (
    compute_key_fact_accuracy,
    detect_abstention,
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


def build_action_outcome_table(
    dataset_path: str = "data/urdu_dynrag_verified_dataset_v3.json",
    output_path: str = "cache/action_outcome_table.json",
    generator_backend: str = "openrouter",
) -> Dict[str, Any]:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing table if available for resumption
    outcome_table: Dict[str, Dict[str, Any]] = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                outcome_table = json.load(f)
            print(f"[OutcomeBuilder] Loaded existing outcome table with {len(outcome_table)} items.")
        except Exception:
            outcome_table = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[OutcomeBuilder] Total dataset items to process: {len(dataset)}.")

    retriever = DualFAISSRetriever(indices_dir="models/indices")
    reformulator = DeterministicUrduReformulator(generator_backend=generator_backend)
    generator = UnifiedGenerator(
        backend=generator_backend,
        cache_path="cache/benchmark_predictions_greedy.json",
        max_new_tokens=80,
    )

    t0 = time.time()
    save_interval = 20

    for idx, item in enumerate(dataset, 1):
        item_id = item["id"]
        if item_id in outcome_table and len(outcome_table[item_id]) == 5:
            continue

        query = item["query"]
        gold_ans = item.get("gold_answer", "")
        gold_span = item.get("gold_span", "")
        target_pid = item.get("passage_id", "")
        q_style = item.get("query_style", "general")
        key_entities = item.get("key_entities", [])
        is_s5 = (q_style == "style_5_hard_distractor")

        # Precompute reformulated query
        q_ref = reformulator.reformulate(query)

        # Retrieve candidates
        # 1. Raw on original query
        raw_res = retriever.search_raw(query, k=1)
        raw_text = raw_res[0]["text"] if raw_res else ""
        raw_pid = raw_res[0]["passage_id"] if raw_res else ""

        # 2. QA on original query
        qa_res = retriever.search_qa(query, k=1)
        qa_text = qa_res[0]["parent_text"] if qa_res else ""
        qa_pid = qa_res[0]["parent_passage_id"] if qa_res else ""

        # 3. Raw on reformulated query
        raw_ref_res = retriever.search_raw(q_ref, k=1)
        raw_ref_text = raw_ref_res[0]["text"] if raw_ref_res else ""
        raw_ref_pid = raw_ref_res[0]["passage_id"] if raw_ref_res else ""

        # 4. QA on reformulated query
        qa_ref_res = retriever.search_qa(q_ref, k=1)
        qa_ref_text = qa_ref_res[0]["parent_text"] if qa_ref_res else ""
        qa_ref_pid = qa_ref_res[0]["parent_passage_id"] if qa_ref_res else ""

        trajectories = [
            ("0", query, ""),
            ("1", query, raw_text),
            ("2", query, qa_text),
            ("3->1", q_ref, raw_ref_text),
            ("3->2", q_ref, qa_ref_text),
        ]

        retrieved_pids_map = {
            "0": "",
            "1": raw_pid,
            "2": qa_pid,
            "3->1": raw_ref_pid,
            "3->2": qa_ref_pid,
        }

        item_outcomes = {}
        for traj_key, prompt_q, ctx_text in trajectories:
            ret_pid = retrieved_pids_map[traj_key]
            is_hit1 = (bool(target_pid) and ret_pid == target_pid)
            ctx_tokens = len(ctx_text.split()) if ctx_text else 0

            # Generate or fetch cached prediction
            ans = generator.generate(
                query=prompt_q,
                context=ctx_text if ctx_text else None,
                temperature=0.0,
                do_sample=False,
            )

            acc = compute_key_fact_accuracy(
                prediction=ans,
                gold_answer=gold_ans,
                gold_span=gold_span,
                key_entities=key_entities,
                query=query,
                is_unanswerable=is_s5,
            )
            f1 = compute_token_f1(ans, gold_ans)
            is_abs = detect_abstention(ans)

            item_outcomes[traj_key] = {
                "trajectory": traj_key,
                "prediction": ans,
                "key_fact_acc": float(acc),
                "token_f1": float(f1),
                "is_abstained": bool(is_abs),
                "is_hit1": bool(is_hit1),
                "retrieved_pid": ret_pid,
                "context_tokens": ctx_tokens,
            }

        outcome_table[item_id] = item_outcomes

        if idx % save_interval == 0 or idx == len(dataset):
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(outcome_table, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - t0
            print(f"[OutcomeBuilder] Progress: {len(outcome_table)}/{len(dataset)} items precomputed (Elapsed: {elapsed:.1f}s)")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(outcome_table, f, ensure_ascii=False, indent=2)
    print(f"[OutcomeBuilder] Successfully generated Action-Outcome Lookup Table at {out_file}!")
    return outcome_table


if __name__ == "__main__":
    build_action_outcome_table()
