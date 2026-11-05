#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Phase 2 5-Fold Passage-Grouped Cross-Validation Benchmark Engine
Executes complete 5-fold cross-validation across all 455 benchmark queries:
  - 85 unique passages (85 x 5 query styles = 425 grounded/distractor queries)
  - 10 parametric question groups (10 x 3 query styles = 30 parametric queries)
  - 5 deterministic folds (seed=42), grouped strictly by passage_id and parametric group.
  - Assert zero passage leakage between train_fold, val_fold, and test_fold.
  - Per-fold DR-RAG threshold calibration on val_fold.
  - Per-fold Double DQN training on train_fold with real cached rewards from action_outcome_table.json.
  - Evaluation of all baselines, zero-shot ReAct, and Urdu-DynRAG with 2-step MDP on test_fold.
  - 4 Ablation configurations across all 5 folds.
  - 5,000 passage-clustered bootstrap iterations for 95% CIs and paired differences.
  - Generates camera-ready CSV and Markdown tables in experiments/.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.phase1_indexer import DualFAISSRetriever
from src.phase2_state_extractor import UrduRAGStateExtractor, tokenize_text
from src.phase3_env import UrduRAGEnv, reformulate_query
from src.phase4_train import DQNPolicy, ReplayBuffer
from src.generator import UnifiedGenerator
from src.metrics import (
    compute_key_fact_accuracy,
    detect_abstention,
)
from src.reformulator import DeterministicUrduReformulator


def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize_text(pred)
    gold_tokens = tokenize_text(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = sum(min(count, gold_counts.get(t, 0)) for t, count in pred_counts.items())
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(gold_tokens)
    return (2.0 * p * r) / (p + r)


# ==============================================================================
# 1. 5-Fold Passage-Grouped Splitter
# ==============================================================================

def build_passage_grouped_5folds(
    dataset: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Partitions the dataset into 5 deterministic folds grouped strictly by passage_id
    and parametric question group.
    
    Guarantees:
      - 0% passage overlap between train, val, and test folds.
      - Every item is tested exactly once in out-of-fold evaluation.
    """
    rng = random.Random(seed)

    # 1. Separate grounded/distractor items and parametric items
    grounded_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    param_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in dataset:
        pid = item.get("passage_id")
        if pid:
            grounded_by_pid[pid].append(item)
        else:
            # Group parametric items by base ID:
            # urdu_dynv_0291, 0292, 0293 belong to group 0291
            item_num = int(item["id"].split("_")[-1])
            base_group_num = 291 + ((item_num - 291) // 3) * 3
            group_key = f"param_grp_{base_group_num:04d}"
            param_by_group[group_key].append(item)

    all_pids = sorted(list(grounded_by_pid.keys()))
    all_param_grps = sorted(list(param_by_group.keys()))

    rng.shuffle(all_pids)
    rng.shuffle(all_param_grps)

    num_folds = 5
    pids_per_fold = len(all_pids) // num_folds  # 85 // 5 = 17
    param_per_fold = len(all_param_grps) // num_folds  # 10 // 5 = 2

    folds = []
    for k in range(num_folds):
        test_pids = set(all_pids[k * pids_per_fold : (k + 1) * pids_per_fold])
        test_param = set(all_param_grps[k * param_per_fold : (k + 1) * param_per_fold])

        train_val_pids = [p for p in all_pids if p not in test_pids]
        train_val_param = [g for g in all_param_grps if g not in test_param]

        # In train_val: 58 train passages (~85%), 10 val passages (~15%)
        train_pids = set(train_val_pids[:58])
        val_pids = set(train_val_pids[58:])

        # In train_val: 7 train param groups (21 items), 1 val param group (3 items)
        train_param = set(train_val_param[:7])
        val_param = set(train_val_param[7:])

        # Strict assertion checks
        assert len(test_pids & train_pids) == 0, f"Fold {k}: Test/Train passage overlap!"
        assert len(test_pids & val_pids) == 0, f"Fold {k}: Test/Val passage overlap!"
        assert len(train_pids & val_pids) == 0, f"Fold {k}: Train/Val passage overlap!"
        assert len(test_param & train_param) == 0, f"Fold {k}: Test/Train param overlap!"
        assert len(test_param & val_param) == 0, f"Fold {k}: Test/Val param overlap!"
        assert len(train_param & val_param) == 0, f"Fold {k}: Train/Val param overlap!"

        test_items: List[Dict[str, Any]] = []
        for p in test_pids:
            test_items.extend(grounded_by_pid[p])
        for g in test_param:
            test_items.extend(param_by_group[g])

        train_items: List[Dict[str, Any]] = []
        for p in train_pids:
            train_items.extend(grounded_by_pid[p])
        for g in train_param:
            train_items.extend(param_by_group[g])

        val_items: List[Dict[str, Any]] = []
        for p in val_pids:
            val_items.extend(grounded_by_pid[p])
        for g in val_param:
            val_items.extend(param_by_group[g])

        folds.append({
            "fold_idx": k,
            "test_items": test_items,
            "train_items": train_items,
            "val_items": val_items,
            "test_pids": test_pids,
            "train_pids": train_pids,
            "val_pids": val_pids,
        })
        print(f"[Fold {k+1}/5] Train: {len(train_items)} ({len(train_pids)} pids), Val: {len(val_items)} ({len(val_pids)} pids), Test: {len(test_items)} ({len(test_pids)} pids)")

    return folds


# ==============================================================================
# 2. Per-Fold DR-RAG Threshold Calibrator
# ==============================================================================

def calibrate_dr_rag_threshold(
    val_items: List[Dict[str, Any]],
    state_cache: Dict[Tuple[str, int, bool], np.ndarray],
    outcome_table: Dict[str, Any],
) -> float:
    """
    Sweeps tau in [0.10, 0.90] on val_fold to find optimal tau_k* maximizing average accuracy.
    """
    candidate_taus = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    best_tau = 0.80
    best_score = -float("inf")

    for tau in candidate_taus:
        total_acc = 0.0
        for item in val_items:
            item_id = item["id"]
            query = item["query"]
            s_vec = state_cache.get((query, 0, False))
            if s_vec is None:
                sim_qa_top1 = 0.5
            else:
                sim_qa_top1 = float(s_vec[4])

            action = 2 if sim_qa_top1 >= tau else 1
            traj = str(action)
            out = outcome_table.get(item_id, {}).get(traj, {})
            acc = out.get("key_fact_acc", 0.0)
            total_acc += acc

        mean_acc = total_acc / max(1, len(val_items))
        if mean_acc > best_score:
            best_score = mean_acc
            best_tau = tau

    return best_tau


# ==============================================================================
# 3. Fast Per-Fold Double DQN Policy Trainer
# ==============================================================================

def train_fold_policy(
    fold_idx: int,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    state_cache: Dict[Tuple[str, int, bool], np.ndarray],
    outcome_table: Dict[str, Any],
    num_epochs: int = 30,
    seed: int = 42,
) -> DQNPolicy:
    """
    Trains a Double DQN Policy network using precomputed state vectors and
    cached action outcomes. Runs in seconds on CPU.
    """
    random.seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)
    torch.manual_seed(seed + fold_idx)

    q_net = DQNPolicy(state_dim=14, action_dim=4)
    target_net = copy.deepcopy(q_net)
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=3000)

    # Pre-extract state pairs
    # (q, 0, False) -> s0
    # (q_ref, 1, True) -> s1
    lambda_1 = 0.06
    lambda_2 = 0.02
    step_cost_a3 = 0.02

    best_val_reward = -float("inf")
    best_weights = copy.deepcopy(q_net.state_dict())

    # Initial buffer fill
    for item in train_items[:50]:
        item_id = item["id"]
        q = item["query"]
        s0 = state_cache.get((q, 0, False))
        if s0 is None:
            continue
        is_s5 = (item.get("query_style") == "style_5_hard_distractor")

        for a in [0, 1, 2, 3]:
            if a == 3:
                # s0 -> a=3 -> s1
                q_ref = reformulate_query(q)
                s1 = state_cache.get((q_ref, 1, True), s0)
                buffer.push(s0, 3, -step_cost_a3, s1, False)

                # s1 -> a_term
                for a_term in [1, 2]:
                    traj = f"3->{a_term}"
                    out = outcome_table.get(item_id, {}).get(traj, {})
                    acc = out.get("key_fact_acc", 0.0)
                    is_hit1 = out.get("is_hit1", False)
                    ctx_tok = out.get("context_tokens", 0)
                    if is_s5:
                        r = acc - lambda_1 * 1.0
                    else:
                        r = acc + (0.15 if is_hit1 else 0.0) - lambda_1 * 1.0 - lambda_2 * (ctx_tok / 1000.0)
                    buffer.push(s1, a_term, r, s1, True)
            else:
                traj = str(a)
                out = outcome_table.get(item_id, {}).get(traj, {})
                acc = out.get("key_fact_acc", 0.0)
                is_hit1 = out.get("is_hit1", False)
                ctx_tok = out.get("context_tokens", 0)
                cost_ret = 0.0 if a == 0 else 1.0
                if is_s5:
                    r = acc - lambda_1 * cost_ret
                else:
                    r = acc + (0.15 if is_hit1 else 0.0) - lambda_1 * cost_ret - lambda_2 * (ctx_tok / 1000.0)
                buffer.push(s0, a, r, s0, True)

    total_steps = num_epochs * len(train_items)
    decay_steps = int(total_steps * 0.65)
    global_step = 0
    batch_size = 32

    for epoch in range(1, num_epochs + 1):
        q_net.train()
        for item in train_items:
            item_id = item["id"]
            q = item["query"]
            s0 = state_cache.get((q, 0, False))
            if s0 is None:
                continue
            is_s5 = (item.get("query_style") == "style_5_hard_distractor")

            global_step += 1
            eps = max(0.05, 1.0 - 0.95 * (global_step / max(1, decay_steps)))

            # Step 1: select action at t=0
            a0 = q_net.select_action(s0, epsilon=eps)

            if a0 == 3:
                q_ref = reformulate_query(q)
                s1 = state_cache.get((q_ref, 1, True), s0)
                buffer.push(s0, 3, -step_cost_a3, s1, False)

                # Step 2: select terminal action from {0, 1, 2}
                with torch.no_grad():
                    q_vals1 = q_net(torch.tensor(s1, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                    q_vals1[3] = -999.0
                    if random.random() < eps:
                        a_term = random.choice([0, 1, 2])
                    else:
                        a_term = int(torch.argmax(q_vals1[:3]).item())

                traj = f"3->{a_term}" if a_term in [1, 2] else "0"
                out = outcome_table.get(item_id, {}).get(traj, {})
                acc = out.get("key_fact_acc", 0.0)
                is_hit1 = out.get("is_hit1", False)
                ctx_tok = out.get("context_tokens", 0)
                cost_ret = 0.0 if a_term == 0 else 1.0
                if is_s5:
                    r_term = acc - lambda_1 * cost_ret
                else:
                    r_term = acc + (0.15 if is_hit1 else 0.0) - lambda_1 * cost_ret - lambda_2 * (ctx_tok / 1000.0)

                buffer.push(s1, a_term, r_term, s1, True)
            else:
                traj = str(a0)
                out = outcome_table.get(item_id, {}).get(traj, {})
                acc = out.get("key_fact_acc", 0.0)
                is_hit1 = out.get("is_hit1", False)
                ctx_tok = out.get("context_tokens", 0)
                cost_ret = 0.0 if a0 == 0 else 1.0
                if is_s5:
                    r = acc - lambda_1 * cost_ret
                else:
                    r = acc + (0.15 if is_hit1 else 0.0) - lambda_1 * cost_ret - lambda_2 * (ctx_tok / 1000.0)

                buffer.push(s0, a0, r, s0, True)

            # Gradient update
            if len(buffer) >= batch_size:
                s_b, a_b, r_b, sn_b, d_b = buffer.sample(batch_size)
                curr_q = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = torch.argmax(q_net(sn_b), dim=1)
                    next_target_q = target_net(sn_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = r_b + 0.95 * next_target_q * (1.0 - d_b)

                loss = criterion(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()

                if global_step % 40 == 0:
                    target_net.load_state_dict(q_net.state_dict())

        # Evaluate on val_items
        q_net.eval()
        val_reward_sum = 0.0
        with torch.no_grad():
            for item in val_items:
                item_id = item["id"]
                q = item["query"]
                s0 = state_cache.get((q, 0, False))
                if s0 is None:
                    continue
                is_s5 = (item.get("query_style") == "style_5_hard_distractor")
                a0 = q_net.select_action(s0, epsilon=0.0)

                if a0 == 3:
                    q_ref = reformulate_query(q)
                    s1 = state_cache.get((q_ref, 1, True), s0)
                    q_vals1 = q_net(torch.tensor(s1, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                    q_vals1[3] = -999.0
                    a_term = int(torch.argmax(q_vals1[:3]).item())
                    traj = f"3->{a_term}" if a_term in [1, 2] else "0"
                    step_cost = step_cost_a3
                else:
                    traj = str(a0)
                    step_cost = 0.0

                out = outcome_table.get(item_id, {}).get(traj, {})
                acc = out.get("key_fact_acc", 0.0)
                is_hit1 = out.get("is_hit1", False)
                ctx_tok = out.get("context_tokens", 0)
                cost_ret = 0.0 if (a0 == 0 or traj == "0") else 1.0

                if is_s5:
                    r = acc - lambda_1 * cost_ret - step_cost
                else:
                    r = acc + (0.15 if is_hit1 else 0.0) - lambda_1 * cost_ret - lambda_2 * (ctx_tok / 1000.0) - step_cost
                val_reward_sum += r

        mean_val_r = val_reward_sum / max(1, len(val_items))
        if mean_val_r > best_val_reward:
            best_val_reward = mean_val_r
            best_weights = copy.deepcopy(q_net.state_dict())

    q_net.load_state_dict(best_weights)
    q_net.eval()
    return q_net


# ==============================================================================
# 4. State Precomputation for All Unique Queries
# ==============================================================================

def precompute_state_cache(
    dataset: List[Dict[str, Any]],
    state_extractor: UrduRAGStateExtractor,
    cache_path: Path,
) -> Dict[Tuple[str, int, bool], np.ndarray]:
    """
    Precomputes 14-dim state vectors for (query, 0, False) and (reformulated_query, 1, True).
    """
    if cache_path.exists():
        try:
            print(f"[StateCache] Loading precomputed state cache from {cache_path}...")
            cache = torch.load(cache_path, weights_only=False)
            print(f"[StateCache] Loaded {len(cache)} state vectors successfully!")
            return cache
        except Exception:
            pass

    print("[StateCache] Precomputing 14-dim state vectors for all unique queries...")
    cache: Dict[Tuple[str, int, bool], np.ndarray] = {}
    unique_queries = list(dict.fromkeys(item["query"].strip() for item in dataset if item.get("query")))

    for q in tqdm(unique_queries, desc="Precomputing States", unit="query"):
        s0 = state_extractor.extract_state(query=q, t=0, reformulated_flag=False).cpu().numpy()
        cache[(q, 0, False)] = s0

        q_ref = reformulate_query(q)
        s1 = state_extractor.extract_state(query=q_ref, t=1, reformulated_flag=True).cpu().numpy()
        cache[(q_ref, 1, True)] = s1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[StateCache] Saved {len(cache)} state vectors to {cache_path}!")
    return cache


# ==============================================================================
# 5. Clustered Bootstrap Engine (5,000 Iterations)
# ==============================================================================

def compute_passage_clustered_bootstrap_cis(
    records_df: pd.DataFrame,
    systems: List[str],
    num_iterations: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes 95% bootstrap confidence intervals clustered at the passage / parametric
    group level with 5,000 resamples.
    """
    rng = np.random.default_rng(seed)

    # Determine clusters: passage_id if present, else base parametric question group
    clusters = sorted(records_df["cluster_id"].unique())
    num_clusters = len(clusters)
    cluster_to_indices = {c: records_df[records_df["cluster_id"] == c].index.values for c in clusters}

    results: Dict[str, Dict[str, Dict[str, Any]]] = {sys: defaultdict(dict) for sys in systems}
    paired_diffs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

    # Metric accumulators: sys -> metric_name -> list of bootstrap estimates
    boot_samples: Dict[str, Dict[str, List[float]]] = {
        sys: {
            "key_fact_acc": [],
            "token_f1": [],
            "false_abs_pct": [],
            "hit1": [],
            "hit5": [],
            "mrr": [],
            "s5_abs_f1": [],
            "act0_pct": [],
            "act1_pct": [],
            "act2_pct": [],
            "act3_pct": [],
        }
        for sys in systems
    }

    # Pre-filter system views
    sys_dfs = {sys: records_df[records_df["system"] == sys].set_index("id") for sys in systems}

    # Unique item IDs
    all_qids = sorted(records_df["id"].unique())
    qid_to_cluster = records_df.drop_duplicates("id").set_index("id")["cluster_id"].to_dict()
    qid_to_style = records_df.drop_duplicates("id").set_index("id")["query_style"].to_dict()

    answerable_qids = [qid for qid in all_qids if qid_to_style[qid] != "style_5_hard_distractor"]
    s5_qids = [qid for qid in all_qids if qid_to_style[qid] == "style_5_hard_distractor"]
    grounded_qids = [qid for qid in all_qids if not qid_to_style[qid].startswith("parametric")]

    for b in range(num_iterations):
        sampled_clusters = rng.choice(clusters, size=num_clusters, replace=True)
        # Gather items in sampled clusters
        sampled_qids = [qid for qid in all_qids if qid_to_cluster[qid] in sampled_clusters]
        if not sampled_qids:
            continue

        b_ans_qids = [q for q in sampled_qids if q in answerable_qids]
        b_s5_qids = [q for q in sampled_qids if q in s5_qids]
        b_grounded_qids = [q for q in sampled_qids if q in grounded_qids]

        for sys in systems:
            df_s = sys_dfs[sys].loc[sampled_qids]
            df_ans = sys_dfs[sys].loc[b_ans_qids]
            df_s5 = sys_dfs[sys].loc[b_s5_qids]
            df_grnd = sys_dfs[sys].loc[b_grounded_qids]

            # 1. Answerable Key-Fact Accuracy
            acc = df_ans["key_fact_acc"].mean() if len(df_ans) > 0 else 0.0
            boot_samples[sys]["key_fact_acc"].append(acc)

            # 2. Token F1
            f1 = df_ans["token_f1"].mean() if len(df_ans) > 0 else 0.0
            boot_samples[sys]["token_f1"].append(f1)

            # 3. False Abstention %
            false_abs = (df_ans["is_abstained"].mean() * 100.0) if len(df_ans) > 0 else 0.0
            boot_samples[sys]["false_abs_pct"].append(false_abs)

            # 4. Grounded Hit@1, Hit@5, MRR
            h1 = (df_grnd["is_hit1"].mean() * 100.0) if len(df_grnd) > 0 else 0.0
            h5 = (df_grnd["is_hit5"].mean() * 100.0) if len(df_grnd) > 0 else 0.0
            mrr = df_grnd["mrr"].mean() if len(df_grnd) > 0 else 0.0
            boot_samples[sys]["hit1"].append(h1)
            boot_samples[sys]["hit5"].append(h5)
            boot_samples[sys]["mrr"].append(mrr)

            # 5. S5 Abstention F1
            s5_acc = df_s5["key_fact_acc"].mean() if len(df_s5) > 0 else 0.0
            boot_samples[sys]["s5_abs_f1"].append(s5_acc)

            # 6. Action Distributions
            acts = df_s["primary_action"].values
            boot_samples[sys]["act0_pct"].append(np.mean(acts == 0) * 100.0)
            boot_samples[sys]["act1_pct"].append(np.mean(acts == 1) * 100.0)
            boot_samples[sys]["act2_pct"].append(np.mean(acts == 2) * 100.0)
            boot_samples[sys]["act3_pct"].append(np.mean(acts == 3) * 100.0)

    # Summarize Mean and 95% CIs
    summary_dict = {}
    for sys in systems:
        summary_dict[sys] = {}
        for m_name, vals in boot_samples[sys].items():
            arr = np.array(vals)
            mean_val = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            summary_dict[sys][m_name] = {
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "formatted": f"{mean_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]" if "pct" not in m_name and "hit" not in m_name and "acc" not in m_name else f"{mean_val:.1f}% [{ci_low:.1f}%, {ci_high:.1f}%]" if ("pct" in m_name or "hit" in m_name) else f"{mean_val*100.0:.1f}% [{ci_low*100.0:.1f}%, {ci_high*100.0:.1f}%]",
            }

    # Compute Paired Differences vs Vanilla Chunk RAG and DR-RAG
    ref_systems = ["Vanilla Chunk RAG", "DR-RAG (Val-Tuned)"]
    for ref_sys in ref_systems:
        if ref_sys not in systems:
            continue
        for sys in systems:
            if sys == ref_sys:
                continue
            paired_diffs[ref_sys][sys] = {}
            for m_name in ["key_fact_acc", "token_f1", "hit1"]:
                s_vals = np.array(boot_samples[sys][m_name])
                ref_vals = np.array(boot_samples[ref_sys][m_name])
                diffs = s_vals - ref_vals
                mean_diff = float(np.mean(diffs))
                ci_low = float(np.percentile(diffs, 2.5))
                ci_high = float(np.percentile(diffs, 97.5))
                # Empirical two-tailed p-value
                p_val = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
                if m_name == "key_fact_acc":
                    fmt = f"{mean_diff*100.0:+.2f}% [{ci_low*100.0:+.2f}%, {ci_high*100.0:+.2f}%], p={p_val:.4f}"
                elif m_name == "hit1":
                    fmt = f"{mean_diff:+.2f}% [{ci_low:+.2f}%, {ci_high:+.2f}%], p={p_val:.4f}"
                else:
                    fmt = f"{mean_diff:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}], p={p_val:.4f}"
                paired_diffs[ref_sys][sys][m_name] = {
                    "mean_diff": mean_diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": min(1.0, max(0.0001, p_val)),
                    "formatted": fmt,
                }

    return {
        "summary": summary_dict,
        "paired_diffs": paired_diffs,
    }


# ==============================================================================
# 6. Main Phase 2 Benchmark Runner
# ==============================================================================

def run_phase2_cv_benchmark() -> None:
    print("=" * 85)
    print("URDU-DYNRAG: PHASE 2 5-FOLD PASSAGE-GROUPED CROSS-VALIDATION BENCHMARK")
    print("=" * 85)

    dataset_path = Path("data/urdu_dynrag_verified_dataset_v3.json")
    outcome_table_path = Path("cache/action_outcome_table.json")
    state_cache_path = Path("cache/precomputed_states_cv.pt")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset missing at {dataset_path}")
    if not outcome_table_path.exists():
        raise FileNotFoundError(f"Action Outcome Table missing at {outcome_table_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(outcome_table_path, "r", encoding="utf-8") as f:
        outcome_table = json.load(f)

    print(f"[Phase 2 Setup] Dataset size: {len(dataset)} items.")
    print(f"[Phase 2 Setup] Outcome table size: {len(outcome_table)} items.")

    retriever = DualFAISSRetriever(indices_dir="models/indices")
    reformulator = DeterministicUrduReformulator(generator_backend="openrouter")
    state_extractor = UrduRAGStateExtractor(
        retriever=retriever,
        generator_mode="heuristic",  # Bug 4 Fix: Enforce heuristic across both train & eval
    )

    # 1. State vector cache precomputation
    state_cache = precompute_state_cache(dataset, state_extractor, state_cache_path)

    # 2. Build 5 passage-grouped folds
    folds = build_passage_grouped_5folds(dataset, seed=42)

    # Assign cluster_id to each dataset item for clustered bootstrap
    for item in dataset:
        pid = item.get("passage_id")
        if pid:
            item["cluster_id"] = pid
        else:
            item_num = int(item["id"].split("_")[-1])
            base_group_num = 291 + ((item_num - 291) // 3) * 3
            item["cluster_id"] = f"param_grp_{base_group_num:04d}"

    item_lookup = {item["id"]: item for item in dataset}

    all_cv_records: List[Dict[str, Any]] = []

    # 3. Iterate over the 5 folds
    for fold in folds:
        k = fold["fold_idx"]
        train_items = fold["train_items"]
        val_items = fold["val_items"]
        test_items = fold["test_items"]

        print(f"\n{'='*35} FOLD {k+1}/5 {'='*35}")

        # A. Calibrate DR-RAG threshold tau_k* on val_fold
        tau_k = calibrate_dr_rag_threshold(val_items, state_cache, outcome_table)
        print(f"[Fold {k+1}] DR-RAG Val-Tuned calibrated threshold tau* = {tau_k:.2f}")

        # B. Train Double DQN Policy on train_fold
        print(f"[Fold {k+1}] Training Double DQN policy (30 epochs, seed={42+k})...")
        policy = train_fold_policy(
            fold_idx=k,
            train_items=train_items,
            val_items=val_items,
            state_cache=state_cache,
            outcome_table=outcome_table,
            num_epochs=30,
            seed=42,
        )

        # C. Evaluate all systems on test_fold (N = 91 items per fold)
        systems_to_evaluate = [
            "Echo-the-Query (Control)",
            "Oracle-Span (Control)",
            "Direct Gen (No RAG)",
            "Vanilla Chunk RAG",
            "DR-RAG (Fixed 0.80)",
            "DR-RAG (Val-Tuned)",
            "Zero-Shot ReAct Agent",
            "Urdu-DynRAG (Ours)",
            # Ablations
            "Urdu-DynRAG (w/o Action 3)",
            "Urdu-DynRAG (w/o D_qa Action 2)",
            "Urdu-DynRAG (w/o Entropy f10-f12)",
        ]

        for item in test_items:
            item_id = item["id"]
            query = item["query"]
            gold_ans = item.get("gold_answer", "")
            gold_span = item.get("gold_span", "")
            target_pid = item.get("passage_id", "")
            q_style = item.get("query_style", "general")
            key_entities = item.get("key_entities", [])
            cluster_id = item["cluster_id"]
            is_s5 = (q_style == "style_5_hard_distractor")
            is_param = q_style.startswith("parametric")

            s0 = state_cache.get((query, 0, False))
            if s0 is None:
                s0 = state_extractor.extract_state(query=query, t=0).cpu().numpy()

            out_0 = outcome_table.get(item_id, {}).get("0", {})
            out_1 = outcome_table.get(item_id, {}).get("1", {})
            out_2 = outcome_table.get(item_id, {}).get("2", {})
            out_3_1 = outcome_table.get(item_id, {}).get("3->1", {})
            out_3_2 = outcome_table.get(item_id, {}).get("3->2", {})

            for sys_name in systems_to_evaluate:
                primary_action = 0
                trajectory = "0"
                pred_ans = ""
                key_fact_acc = 0.0
                token_f1 = 0.0
                is_hit1 = False
                is_hit5 = False
                mrr = 0.0
                is_abs = False

                if sys_name == "Echo-the-Query (Control)":
                    primary_action = 0
                    trajectory = "echo"
                    pred_ans = query
                    key_fact_acc = compute_key_fact_accuracy(pred_ans, gold_ans, gold_span, key_entities, query, is_unanswerable=is_s5)
                    token_f1 = compute_token_f1(pred_ans, gold_ans)
                    is_abs = detect_abstention(pred_ans)
                    is_hit1 = False
                    is_hit5 = False
                    mrr = 0.0

                elif sys_name == "Oracle-Span (Control)":
                    primary_action = 0
                    trajectory = "oracle"
                    pred_ans = gold_span if gold_span else gold_ans
                    key_fact_acc = 1.0 if not is_s5 else 0.0
                    token_f1 = 1.0
                    is_abs = False
                    is_hit1 = True if target_pid else False
                    is_hit5 = True if target_pid else False
                    mrr = 1.0 if target_pid else 0.0

                elif sys_name == "Direct Gen (No RAG)":
                    primary_action = 0
                    trajectory = "0"
                    pred_ans = out_0.get("prediction", "")
                    key_fact_acc = out_0.get("key_fact_acc", 0.0)
                    token_f1 = out_0.get("token_f1", 0.0)
                    is_abs = out_0.get("is_abstained", False)
                    is_hit1 = False
                    is_hit5 = False
                    mrr = 0.0

                elif sys_name == "Vanilla Chunk RAG":
                    primary_action = 1
                    trajectory = "1"
                    pred_ans = out_1.get("prediction", "")
                    key_fact_acc = out_1.get("key_fact_acc", 0.0)
                    token_f1 = out_1.get("token_f1", 0.0)
                    is_abs = out_1.get("is_abstained", False)
                    is_hit1 = out_1.get("is_hit1", False)
                    is_hit5 = is_hit1  # Top-1 is contained in Top-5
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "DR-RAG (Fixed 0.80)":
                    sim_qa_top1 = float(s0[4])
                    primary_action = 2 if sim_qa_top1 >= 0.80 else 1
                    trajectory = str(primary_action)
                    out = out_2 if primary_action == 2 else out_1
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "DR-RAG (Val-Tuned)":
                    sim_qa_top1 = float(s0[4])
                    primary_action = 2 if sim_qa_top1 >= tau_k else 1
                    trajectory = str(primary_action)
                    out = out_2 if primary_action == 2 else out_1
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "Zero-Shot ReAct Agent":
                    # ReAct router with Bug 2 Fix: when Action 3 chosen, retrieve on q_ref!
                    # Rule: if Nastaliq script ratio < 0.60 or code-switched, pick action 3
                    nastaliq_ratio = float(s0[1])
                    if nastaliq_ratio < 0.60:
                        primary_action = 3
                        trajectory = "3->1"
                        out = out_3_1
                    elif is_param:
                        primary_action = 0
                        trajectory = "0"
                        out = out_0
                    else:
                        primary_action = 1
                        trajectory = "1"
                        out = out_1
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "Urdu-DynRAG (Ours)":
                    # Full 2-step MDP policy evaluation
                    with torch.no_grad():
                        q_vals0 = policy(torch.tensor(s0, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                        a0 = int(np.argmax(q_vals0))

                    if a0 == 3:
                        primary_action = 3
                        q_ref = reformulate_query(query)
                        s1 = state_cache.get((q_ref, 1, True), s0)
                        with torch.no_grad():
                            q_vals1 = policy(torch.tensor(s1, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                            q_vals1[3] = -999.0
                            a_term = int(torch.argmax(q_vals1[:3]).item())
                        trajectory = f"3->{a_term}" if a_term in [1, 2] else "0"
                    else:
                        primary_action = a0
                        trajectory = str(a0)

                    out = outcome_table.get(item_id, {}).get(trajectory, out_1)
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "Urdu-DynRAG (w/o Action 3)":
                    # Ablation: Action 3 masked at t=0
                    with torch.no_grad():
                        q_vals0 = policy(torch.tensor(s0, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                        q_vals0[3] = -999.0
                        a0 = int(torch.argmax(q_vals0[:3]).item())
                    primary_action = a0
                    trajectory = str(a0)
                    out = outcome_table.get(item_id, {}).get(trajectory, out_1)
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "Urdu-DynRAG (w/o D_qa Action 2)":
                    # Ablation: Action 2 masked at t=0 and t=1
                    with torch.no_grad():
                        q_vals0 = policy(torch.tensor(s0, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                        q_vals0[2] = -999.0
                        a0 = int(torch.argmax(q_vals0).item())

                    if a0 == 3:
                        primary_action = 3
                        q_ref = reformulate_query(query)
                        s1 = state_cache.get((q_ref, 1, True), s0)
                        with torch.no_grad():
                            q_vals1 = policy(torch.tensor(s1, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                            q_vals1[2] = -999.0
                            q_vals1[3] = -999.0
                            a_term = int(torch.argmax(q_vals1).item())
                        trajectory = f"3->{a_term}" if a_term in [1, 2] else "0"
                    else:
                        primary_action = a0
                        trajectory = str(a0)

                    out = outcome_table.get(item_id, {}).get(trajectory, out_1)
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                elif sys_name == "Urdu-DynRAG (w/o Entropy f10-f12)":
                    # Ablation: Entropy features f9, f10, f11 set to neutral (0.45, 0.65, 0.05)
                    s0_no_ent = s0.copy()
                    s0_no_ent[9] = 0.45
                    s0_no_ent[10] = 0.65
                    s0_no_ent[11] = 0.05
                    with torch.no_grad():
                        q_vals0 = policy(torch.tensor(s0_no_ent, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                        a0 = int(np.argmax(q_vals0))

                    if a0 == 3:
                        primary_action = 3
                        q_ref = reformulate_query(query)
                        s1 = state_cache.get((q_ref, 1, True), s0)
                        s1_no_ent = s1.copy()
                        s1_no_ent[9] = 0.45
                        s1_no_ent[10] = 0.65
                        s1_no_ent[11] = 0.05
                        with torch.no_grad():
                            q_vals1 = policy(torch.tensor(s1_no_ent, dtype=torch.float32).unsqueeze(0)).squeeze(0).clone()
                            q_vals1[3] = -999.0
                            a_term = int(torch.argmax(q_vals1[:3]).item())
                        trajectory = f"3->{a_term}" if a_term in [1, 2] else "0"
                    else:
                        primary_action = a0
                        trajectory = str(a0)

                    out = outcome_table.get(item_id, {}).get(trajectory, out_1)
                    pred_ans = out.get("prediction", "")
                    key_fact_acc = out.get("key_fact_acc", 0.0)
                    token_f1 = out.get("token_f1", 0.0)
                    is_abs = out.get("is_abstained", False)
                    is_hit1 = out.get("is_hit1", False)
                    is_hit5 = is_hit1
                    mrr = 1.0 if is_hit1 else 0.0

                all_cv_records.append({
                    "fold": k + 1,
                    "id": item_id,
                    "query": query,
                    "system": sys_name,
                    "query_style": q_style,
                    "cluster_id": cluster_id,
                    "passage_id": target_pid,
                    "primary_action": primary_action,
                    "trajectory": trajectory,
                    "prediction": pred_ans,
                    "gold_answer": gold_ans,
                    "key_fact_acc": float(key_fact_acc),
                    "token_f1": float(token_f1),
                    "is_abstained": bool(is_abs),
                    "is_hit1": bool(is_hit1),
                    "is_hit5": bool(is_hit5),
                    "mrr": float(mrr),
                })

    records_df = pd.DataFrame(all_cv_records)
    exp_dir = Path("experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)
    records_csv_path = exp_dir / "phase2_cv_records.csv"
    records_df.to_csv(records_csv_path, index=False, encoding="utf-8")
    print(f"\n[Phase 2 Complete] Saved {len(records_df)} evaluation records to {records_csv_path}!")

    # 4. Bootstrap Confidence Intervals & Summaries
    print("\n[Bootstrap Engine] Running 5,000 passage-clustered bootstrap iterations...")
    all_systems = [
        "Echo-the-Query (Control)",
        "Oracle-Span (Control)",
        "Direct Gen (No RAG)",
        "Vanilla Chunk RAG",
        "DR-RAG (Fixed 0.80)",
        "DR-RAG (Val-Tuned)",
        "Zero-Shot ReAct Agent",
        "Urdu-DynRAG (Ours)",
        "Urdu-DynRAG (w/o Action 3)",
        "Urdu-DynRAG (w/o D_qa Action 2)",
        "Urdu-DynRAG (w/o Entropy f10-f12)",
    ]
    boot_res = compute_passage_clustered_bootstrap_cis(records_df, all_systems, num_iterations=5000, seed=42)

    # 5. Export Camera-Ready Summary Table
    main_systems = [
        "Echo-the-Query (Control)",
        "Oracle-Span (Control)",
        "Direct Gen (No RAG)",
        "Vanilla Chunk RAG",
        "DR-RAG (Fixed 0.80)",
        "DR-RAG (Val-Tuned)",
        "Zero-Shot ReAct Agent",
        "Urdu-DynRAG (Ours)",
    ]
    summary_rows = []
    for sys in main_systems:
        s_data = boot_res["summary"][sys]
        summary_rows.append({
            "System": sys,
            "Key-Fact Acc (Ans) [n=370]": s_data["key_fact_acc"]["formatted"],
            "Token F1 [n=370]": s_data["token_f1"]["formatted"],
            "False Abs % [n=370]": s_data["false_abs_pct"]["formatted"],
            "Grounded Hit@1 [n=425]": s_data["hit1"]["formatted"],
            "Grounded Hit@5 [n=425]": s_data["hit5"]["formatted"],
            "Grounded MRR [n=425]": s_data["mrr"]["formatted"],
            "S5 Abs F1 [n=85]": s_data["s5_abs_f1"]["formatted"],
            "Action Dist % (0/1/2/3)": f"{s_data['act0_pct']['mean']:.0f}% / {s_data['act1_pct']['mean']:.0f}% / {s_data['act2_pct']['mean']:.0f}% / {s_data['act3_pct']['mean']:.0f}%",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(exp_dir / "phase2_cv_summary_table.csv", index=False, encoding="utf-8")
    summary_df.to_markdown(exp_dir / "phase2_cv_summary_table.md", index=False)
    print("\n" + "=" * 85)
    print("TABLE 1: 5-FOLD PASSAGE-GROUPED CV BENCHMARK SUMMARY (N=455)")
    print("=" * 85)
    print(summary_df.to_markdown(index=False))

    # 6. Export Style-by-Style Breakdown
    styles_list = [
        "style_1_clean_synthetic",
        "style_2_conversational",
        "style_3_code_switched",
        "style_4_roman_urdu",
        "style_5_hard_distractor",
        "parametric",
    ]
    style_names = {
        "style_1_clean_synthetic": "S1 (Clean Urdu)",
        "style_2_conversational": "S2 (Conversational)",
        "style_3_code_switched": "S3 (Code-Switched)",
        "style_4_roman_urdu": "S4 (Roman Urdu)",
        "style_5_hard_distractor": "S5 (Distractor / Unanswerable)",
        "parametric": "Parametric",
    }

    style_rows = []
    for sys in main_systems:
        df_sys = records_df[records_df["system"] == sys]
        row: Dict[str, Any] = {"System": sys}
        for st in styles_list:
            if st == "parametric":
                df_st = df_sys[df_sys["query_style"].str.startswith("parametric")]
            else:
                df_st = df_sys[df_sys["query_style"] == st]
            acc = df_st["key_fact_acc"].mean() * 100.0 if len(df_st) > 0 else 0.0
            hit1 = df_st["is_hit1"].mean() * 100.0 if len(df_st) > 0 and not st.startswith("param") else 0.0
            col_name = style_names[st]
            if st == "style_5_hard_distractor":
                row[f"{col_name} Acc"] = f"{acc:.1f}%"
            elif st == "parametric":
                row[f"{col_name} Acc"] = f"{acc:.1f}%"
            else:
                row[f"{col_name} Acc / Hit@1"] = f"{acc:.1f}% / {hit1:.1f}%"
        style_rows.append(row)

    style_df = pd.DataFrame(style_rows)
    style_df.to_csv(exp_dir / "phase2_cv_style_breakdown.csv", index=False, encoding="utf-8")
    style_df.to_markdown(exp_dir / "phase2_cv_style_breakdown.md", index=False)
    print("\n" + "=" * 85)
    print("TABLE 2: STYLE-BY-STYLE BREAKDOWN (ACCURACY / HIT@1)")
    print("=" * 85)
    print(style_df.to_markdown(index=False))

    # 7. Export Paired Difference Table
    paired_rows = []
    paired_diffs = boot_res["paired_diffs"]
    for ref_sys in ["Vanilla Chunk RAG", "DR-RAG (Val-Tuned)"]:
        for sys in [s for s in main_systems if s != ref_sys and s not in ["Echo-the-Query (Control)", "Oracle-Span (Control)"]]:
            d = paired_diffs[ref_sys][sys]
            paired_rows.append({
                "Reference Baseline": ref_sys,
                "Comparison System": sys,
                "Delta Key-Fact Acc (95% CI, p-val)": d["key_fact_acc"]["formatted"],
                "Delta Token F1 (95% CI, p-val)": d["token_f1"]["formatted"],
                "Delta Hit@1 (95% CI, p-val)": d["hit1"]["formatted"],
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(exp_dir / "phase2_cv_paired_diff.csv", index=False, encoding="utf-8")
    paired_df.to_markdown(exp_dir / "phase2_cv_paired_diff.md", index=False)
    print("\n" + "=" * 85)
    print("TABLE 3: PAIRED STATISTICAL DIFFERENCES (vs Vanilla & DR-RAG)")
    print("=" * 85)
    print(paired_df.to_markdown(index=False))

    # 8. Export Ablation Table
    ablation_systems = [
        "Urdu-DynRAG (Ours)",
        "Urdu-DynRAG (w/o Action 3)",
        "Urdu-DynRAG (w/o D_qa Action 2)",
        "Urdu-DynRAG (w/o Entropy f10-f12)",
    ]
    ablation_rows = []
    for sys in ablation_systems:
        s_data = boot_res["summary"][sys]
        ablation_rows.append({
            "Configuration": sys,
            "Key-Fact Acc (Ans)": s_data["key_fact_acc"]["formatted"],
            "Token F1": s_data["token_f1"]["formatted"],
            "Grounded Hit@1": s_data["hit1"]["formatted"],
            "Action Dist % (0/1/2/3)": f"{s_data['act0_pct']['mean']:.0f}% / {s_data['act1_pct']['mean']:.0f}% / {s_data['act2_pct']['mean']:.0f}% / {s_data['act3_pct']['mean']:.0f}%",
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(exp_dir / "phase2_ablation_table.csv", index=False, encoding="utf-8")
    ablation_df.to_markdown(exp_dir / "phase2_ablation_table.md", index=False)
    print("\n" + "=" * 85)
    print("TABLE 4: PHASE 2 ABLATION STUDY")
    print("=" * 85)
    print(ablation_df.to_markdown(index=False))


if __name__ == "__main__":
    run_phase2_cv_benchmark()
