#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Dynamic RL-Guided Adaptive Retrieval for Under-Resourced QA
Phase 4: Double/Target DQN Policy Training & Multi-Baseline Comparative Evaluation
Reference: System Specification, Operational Architecture & Team Implementation Roadmap

Deliverables:
  1. Double DQN Policy & Target Network Training (Huber loss, linear epsilon decay, replay buffer)
  2. Multi-Baseline Comparative Evaluation on held-out test split (101 items):
     - Direct Gen (No RAG)
     - Vanilla Chunk RAG
     - DR-RAG (Ahmad et al., 2026)
     - Zero-Shot ReAct Agent
     - Urdu-DynRAG (Ours)
  3. Statistical Significance Testing (Paired Bootstrap Resampling, 1,000 trials)
  4. Publication Tables (CSV, Markdown, LaTeX Booktabs)
  5. Camera-Ready Publication Figures (300 DPI PNG + Vector PDF)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

try:
    from phase1_indexer import DualFAISSRetriever
    from phase2_state_extractor import UrduRAGStateExtractor, tokenize_text
    from phase3_env import FrozenAnswerGenerator, UrduRAGEnv, compute_token_f1, reformulate_query
except ImportError:
    from src.phase1_indexer import DualFAISSRetriever
    from src.phase2_state_extractor import UrduRAGStateExtractor, tokenize_text
    from src.phase3_env import FrozenAnswerGenerator, UrduRAGEnv, compute_token_f1, reformulate_query


# ==============================================================================
# 1. DQN Policy Network & Experience Replay Buffer
# ==============================================================================

class DQNPolicy(nn.Module):
    """
    2-Layer MLP Q-Network:
      nn.Linear(14, 64) -> ReLU -> nn.Linear(64, 32) -> ReLU -> nn.Linear(32, 4)
    Total Parameters: 3,172 (< 4,000 parameters, instant CPU inference).
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 4,
        hidden1: int = 64,
        hidden2: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.forward(s_t)
            return int(torch.argmax(q_vals, dim=-1).item())


class ReplayBuffer:
    """Fixed-capacity experience replay buffer for off-policy DQN transitions."""

    def __init__(self, capacity: int = 2000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 32) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ==============================================================================
# 2. Advanced Evaluation Metrics (ROUGE-L, chrF++, Hit@1, Faithfulness)
# ==============================================================================

def compute_lcs(s1: List[str], s2: List[str]) -> int:
    """Length of Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Token-level ROUGE-L F1 score."""
    p_toks = tokenize_text(prediction)
    g_toks = tokenize_text(ground_truth)
    if not p_toks or not g_toks:
        return 0.0
    lcs = compute_lcs(p_toks, g_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / float(len(p_toks))
    rec = lcs / float(len(g_toks))
    return float((2.0 * prec * rec) / (prec + rec + 1e-8))


def compute_chrf_score(prediction: str, ground_truth: str, n_order: int = 4) -> float:
    """
    chrF++ character n-gram F-score proxy (crucial for morphologically rich Urdu).
    """
    pred_chars = list(prediction.strip().replace(" ", ""))
    gold_chars = list(ground_truth.strip().replace(" ", ""))
    if not pred_chars or not gold_chars:
        return 0.0

    precisions = []
    recalls = []

    for n in range(1, n_order + 1):
        p_grams = [tuple(pred_chars[i : i + n]) for i in range(len(pred_chars) - n + 1)]
        g_grams = [tuple(gold_chars[i : i + n]) for i in range(len(gold_chars) - n + 1)]

        if not p_grams or not g_grams:
            continue

        p_set = set(p_grams)
        g_set = set(g_grams)
        common = p_set.intersection(g_set)

        p = len(common) / float(len(p_set)) if p_set else 0.0
        r = len(common) / float(len(g_set)) if g_set else 0.0
        precisions.append(p)
        recalls.append(r)

    if not precisions:
        return 0.0
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    return float((2.0 * avg_p * avg_r) / (avg_p + avg_r + 1e-8))


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    num_resamples: int = 1000,
) -> float:
    """
    Paired bootstrap hypothesis test for statistical significance.
    Returns p-value for hypothesis that scores_a is significantly different from scores_b.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = float(np.mean(a - b))

    n = len(a)
    diffs = []
    for _ in range(num_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        resampled_diff = np.mean(a[indices] - b[indices])
        diffs.append(resampled_diff)

    diffs = np.array(diffs)
    # Fraction of resampled differences crossing zero
    if observed_diff > 0:
        p_val = float(np.sum(diffs <= 0) / num_resamples)
    else:
        p_val = float(np.sum(diffs >= 0) / num_resamples)
    return max(0.001, p_val)


# ==============================================================================
# 3. Policy Training Loop (Double / Target DQN)
# ==============================================================================

def train_dqn(
    env_train: UrduRAGEnv,
    env_val: UrduRAGEnv,
    num_epochs: int = 25,
    gamma: float = 0.95,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    target_update_freq: int = 50,
    models_dir: str = "models",
    history_dir: str = "experiments",
    seed: int = 42,
) -> Tuple[DQNPolicy, Dict[str, Any]]:
    """
    Train Double DQN policy on train split with linear epsilon decay.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(history_dir).mkdir(parents=True, exist_ok=True)

    q_net = DQNPolicy()
    target_net = copy.deepcopy(q_net)
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=2000)

    # Pre-populate buffer with random initial transitions
    print("[DQN Training] Warm-up: Gathering initial replay transitions...")
    for _ in range(64):
        s, _ = env_train.reset()
        done = False
        while not done:
            a = random.randint(0, 3)
            s_next, r, term, trunc, _ = env_train.step(a)
            done = term or trunc
            buffer.push(s, a, r, s_next, done)
            s = s_next

    total_steps = num_epochs * len(env_train.dataset)
    decay_steps = int(total_steps * 0.60)
    global_step = 0

    best_val_reward = -float("inf")
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_reward": [],
        "val_reward": [],
        "val_f1": [],
        "td_loss": [],
        "epsilon": [],
    }

    print(f"\n[DQN Training] Starting {num_epochs} Epochs ({len(env_train.dataset)} items/epoch)...")
    print("-" * 80)

    for epoch in range(1, num_epochs + 1):
        epoch_rewards = []
        epoch_losses = []

        # Train epoch
        for _ in range(len(env_train.dataset)):
            s, _ = env_train.reset()
            ep_reward = 0.0
            done = False

            while not done:
                global_step += 1
                # Linear epsilon decay: 1.0 -> 0.05
                if global_step < decay_steps:
                    eps = 1.0 - (0.95 * (global_step / decay_steps))
                else:
                    eps = 0.05

                a = q_net.select_action(s, epsilon=eps)
                s_next, r, term, trunc, _ = env_train.step(a)
                done = term or trunc
                buffer.push(s, a, r, s_next, done)
                s = s_next
                ep_reward += r

                # Off-policy Q-update
                if len(buffer) >= batch_size:
                    s_b, a_b, r_b, sn_b, d_b = buffer.sample(batch_size)

                    # Online Q-values for taken actions
                    current_q = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                    # Double DQN target computation
                    with torch.no_grad():
                        next_actions = torch.argmax(q_net(sn_b), dim=1)
                        next_target_q = target_net(sn_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target_q = r_b + gamma * next_target_q * (1.0 - d_b)

                    loss = criterion(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())

                    # Periodic target network sync
                    if global_step % target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())

            epoch_rewards.append(ep_reward)

        # Validation evaluation (Greedy epsilon = 0.0)
        val_rewards = []
        val_f1s = []
        for _ in range(len(env_val.dataset)):
            s_val, _ = env_val.reset()
            done_val = False
            ep_r_val = 0.0
            while not done_val:
                a_val = q_net.select_action(s_val, epsilon=0.0)
                s_val_next, r_v, term_v, trunc_v, info_v = env_val.step(a_val)
                done_val = term_v or trunc_v
                ep_r_val += r_v
                s_val = s_val_next
                if done_val:
                    val_f1s.append(info_v.get("token_f1", 0.0))
            val_rewards.append(ep_r_val)

        mean_train_r = float(np.mean(epoch_rewards))
        mean_val_r = float(np.mean(val_rewards))
        mean_val_f1 = float(np.mean(val_f1s))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        history["epoch"].append(epoch)
        history["train_reward"].append(mean_train_r)
        history["val_reward"].append(mean_val_r)
        history["val_f1"].append(mean_val_f1)
        history["td_loss"].append(mean_loss)
        history["epsilon"].append(eps)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Rew: {mean_train_r:+.4f} | "
            f"Val Rew: {mean_val_r:+.4f} | "
            f"Val F1: {mean_val_f1:.4f} | "
            f"Loss: {mean_loss:.4f} | "
            f"Eps: {eps:.2f}"
        )

        # Save checkpoint if best validation reward
        if mean_val_r > best_val_reward:
            best_val_reward = mean_val_r
            torch.save(q_net.state_dict(), Path(models_dir) / "policy_best.pt")
            print(f"   -> [Checkpoint Saved] New best validation reward: {best_val_reward:+.4f}")

    # Save training history JSON
    history_path = Path(history_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DQN Training] Saved complete training history to {history_path}")

    # Reload best model for evaluation
    q_net.load_state_dict(torch.load(Path(models_dir) / "policy_best.pt", weights_only=True))
    q_net.eval()
    return q_net, history


# ==============================================================================
# 4. Comparative Evaluation Engine (5 Systems on Test Split)
# ==============================================================================

def evaluate_system_instance(
    system_name: str,
    item: Dict[str, Any],
    retriever: DualFAISSRetriever,
    state_extractor: UrduRAGStateExtractor,
    generator: FrozenAnswerGenerator,
    policy_net: Optional[DQNPolicy] = None,
    ablation_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one test item under a specific system architecture.
    """
    query = item["query"]
    gold_answer = item.get("gold_answer", "")
    gold_span = item.get("gold_span", "")
    target_pid = item.get("passage_id", "")
    q_style = item.get("query_style", "general")

    start_time = time.time()
    retrieved_pid = None
    retrieved_text = ""
    action_history = []
    current_q = query
    context_tokens = 0

    # Execute system-specific decision logic
    t = 0
    done = False
    final_action = 0

    while not done and t < 3:
        # Extract 14-dim state
        s_tensor = state_extractor.extract_state(
            query=current_q,
            t=t,
            reformulated_flag=(3 in action_history),
        )
        s_vec = s_tensor.numpy()

        # System Action Selection
        if system_name == "Direct Gen (No RAG)":
            action = 0

        elif system_name == "Vanilla Chunk RAG":
            action = 1

        elif system_name == "DR-RAG (Ahmad et al., 2026)":
            # Heuristic rule: inspect s_vec[4] (sim_qa_top1). If >= 0.80 -> Action 2, else Action 1
            sim_qa_top1 = s_vec[4]
            action = 2 if sim_qa_top1 >= 0.80 else 1

        elif system_name == "Zero-Shot ReAct Agent":
            # Prompt-based tool routing
            prompt = (
                f"You are an AI search planner. For this query: '{current_q}', choose one tool:\n"
                f"0: [DIRECT] (parametric memory, no search)\n"
                f"1: [RAW_RAG] (search raw documents)\n"
                f"2: [QA_RAG] (search question bank)\n"
                f"3: [REFORMULATE] (clean slang/translate Roman Urdu)\n"
                f"Reply ONLY with the number 0, 1, 2, or 3:"
            )
            raw_act = generator.generate_answer(prompt)
            match = re.search(r"[0-3]", raw_act)
            action = int(match.group(0)) if match else 1

        elif system_name.startswith("Urdu-DynRAG"):
            # Ablations
            if ablation_flags and ablation_flags.get("zero_entropy_features"):
                s_vec[9:12] = 0.5  # Zero out f10-f12
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                if ablation_flags and ablation_flags.get("disable_action3"):
                    q_vals[3] = -999.0
                if ablation_flags and ablation_flags.get("disable_action2"):
                    q_vals[2] = -999.0
                action = int(np.argmax(q_vals))

        action_history.append(action)
        final_action = action

        if action == 3:
            # Query Reformulation
            current_q = reformulate_query(current_q)
            t += 1
            if t >= 3 or action_history.count(3) > 1:
                done = True
                final_action = 1  # Fallback to chunk
        else:
            done = True

    # Execution of final retrieval action
    if final_action == 0:
        retrieved_text = ""
        retrieved_pid = None
    elif final_action == 1:
        raw_res = retriever.search_raw(current_q, k=1)
        if raw_res:
            retrieved_text = raw_res[0]["text"]
            retrieved_pid = raw_res[0]["passage_id"]
    elif final_action == 2:
        qa_res = retriever.search_qa(current_q, k=1)
        if qa_res:
            retrieved_text = qa_res[0]["parent_text"]
            retrieved_pid = qa_res[0]["parent_passage_id"]

    context_tokens = len(retrieved_text.split())

    # Generate answer y
    y_pred = generator.generate_answer(current_q, context=retrieved_text if retrieved_text else None)
    elapsed_ms = (time.time() - start_time) * 1000.0

    # Compute Metrics
    token_f1 = compute_token_f1(y_pred, gold_answer)
    rouge_l = compute_rouge_l(y_pred, gold_answer)
    chrf = compute_chrf_score(y_pred, gold_answer)

    # Gold Span Containment / Exact Match
    em = 1.0 if gold_span and gold_span.strip() in y_pred else 0.0

    # Retrieval Hit@1
    hit1 = 1.0 if target_pid and retrieved_pid == target_pid else 0.0

    # Context Faithfulness Proxy
    faithfulness = compute_token_f1(y_pred, retrieved_text) if retrieved_text else 1.0

    # Hard Distractor Abstention Accuracy
    abstention = 0.0
    if q_style == "style_5_hard_distractor":
        abstention = 1.0 if ("معلومات" in y_pred or "نہیں" in y_pred or hit1 == 0.0) else 0.0

    # Reward
    cost_retrieval = 0.0 if final_action == 0 else 1.0
    if 3 in action_history:
        cost_retrieval += 0.3
    cost_tokens = context_tokens / 1000.0
    reward = token_f1 - (0.15 * cost_retrieval) - (0.05 * cost_tokens)

    return {
        "system": system_name,
        "query_id": item.get("id"),
        "query_style": q_style,
        "token_f1": token_f1,
        "rouge_l": rouge_l,
        "chrf": chrf,
        "exact_match": em,
        "retrieval_hit1": hit1,
        "faithfulness": faithfulness,
        "abstention_acc": abstention,
        "retrieval_called": 1.0 if final_action in [1, 2] else 0.0,
        "reformulated": 1.0 if 3 in action_history else 0.0,
        "final_action": final_action,
        "latency_ms": elapsed_ms,
        "context_tokens": context_tokens,
        "reward": reward,
    }


def run_full_benchmark(
    test_items: List[Dict[str, Any]],
    retriever: DualFAISSRetriever,
    state_extractor: UrduRAGStateExtractor,
    generator: FrozenAnswerGenerator,
    policy_net: DQNPolicy,
    output_dir: str = "experiments",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run evaluation across all 5 systems and generate publication tables.
    """
    systems = [
        "Direct Gen (No RAG)",
        "Vanilla Chunk RAG",
        "DR-RAG (Ahmad et al., 2026)",
        "Zero-Shot ReAct Agent",
        "Urdu-DynRAG (Ours)",
    ]

    all_records: List[Dict[str, Any]] = []
    print(f"\n[Benchmarking] Evaluating {len(systems)} systems across {len(test_items)} test items...")

    for sys_name in systems:
        print(f" -> Evaluating System: {sys_name}...")
        for item in test_items:
            res = evaluate_system_instance(
                system_name=sys_name,
                item=item,
                retriever=retriever,
                state_extractor=state_extractor,
                generator=generator,
                policy_net=policy_net,
            )
            all_records.append(res)

    df_results = pd.DataFrame(all_records)

    # --------------------------------------------------------------------------
    # Statistical Significance Testing (Urdu-DynRAG vs. DR-RAG)
    # --------------------------------------------------------------------------
    f1_dynrag = df_results[df_results["system"] == "Urdu-DynRAG (Ours)"]["token_f1"].tolist()
    f1_drrag = df_results[df_results["system"] == "DR-RAG (Ahmad et al., 2026)"]["token_f1"].tolist()
    p_value = paired_bootstrap_test(f1_dynrag, f1_drrag, num_resamples=1000)
    sig_mark = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
    print(f"\n[Significance] Urdu-DynRAG vs. DR-RAG: p = {p_value:.4f} {sig_mark}")

    # --------------------------------------------------------------------------
    # Table 1: Overall Results
    # --------------------------------------------------------------------------
    t1_rows = []
    for sys_name in systems:
        sub = df_results[df_results["system"] == sys_name]
        is_ours = sys_name == "Urdu-DynRAG (Ours)"
        mark = sig_mark if is_ours else ""
        t1_rows.append({
            "System": sys_name,
            "Token F1 (%)": f"{sub['token_f1'].mean()*100:.1f} ± {sub['token_f1'].std()*100:.1f}{mark}",
            "ROUGE-L (%)": f"{sub['rouge_l'].mean()*100:.1f}",
            "chrF++ (%)": f"{sub['chrf'].mean()*100:.1f}",
            "Gold Span EM (%)": f"{sub['exact_match'].mean()*100:.1f}",
            "Hit@1 (%)": f"{sub['retrieval_hit1'].mean()*100:.1f}",
            "Faithfulness": f"{sub['faithfulness'].mean():.3f}",
            "Retrieval Rate (%)": f"{sub['retrieval_called'].mean()*100:.1f}",
            "Latency (ms)": f"{sub['latency_ms'].mean():.1f}",
            "Context Tokens": f"{sub['context_tokens'].mean():.0f}",
            "Net Reward": f"{sub['reward'].mean():+.3f}",
        })
    df_t1 = pd.DataFrame(t1_rows)

    # --------------------------------------------------------------------------
    # Table 2: Performance Breakdown by Query Style
    # --------------------------------------------------------------------------
    styles = [
        "style_1_clean_synthetic",
        "style_2_conversational",
        "style_3_code_switched",
        "style_4_roman_urdu",
        "style_5_hard_distractor",
        "parametric_nastaliq",
    ]
    t2_rows = []
    for sys_name in systems:
        row = {"System": sys_name}
        for st in styles:
            sub = df_results[(df_results["system"] == sys_name) & (df_results["query_style"] == st)]
            clean_name = st.replace("style_", "S").replace("_", " ").title()
            row[clean_name] = f"{sub['token_f1'].mean()*100:.1f}" if len(sub) > 0 else "N/A"
        t2_rows.append(row)
    df_t2 = pd.DataFrame(t2_rows)

    # --------------------------------------------------------------------------
    # Table 3: Ablation Study
    # --------------------------------------------------------------------------
    ablations = [
        ("Full Urdu-DynRAG", None),
        ("w/o Action 3 (No Reformulation)", {"disable_action3": True}),
        ("w/o D_qa (No Synthetic QA)", {"disable_action2": True}),
        ("w/o Generator Entropy (f10-f12 zeroed)", {"zero_entropy_features": True}),
    ]
    t3_rows = []
    for ab_name, flags in ablations:
        ab_records = []
        for item in test_items:
            res = evaluate_system_instance(
                system_name="Urdu-DynRAG",
                item=item,
                retriever=retriever,
                state_extractor=state_extractor,
                generator=generator,
                policy_net=policy_net,
                ablation_flags=flags,
            )
            ab_records.append(res)
        sub = pd.DataFrame(ab_records)
        t3_rows.append({
            "Configuration": ab_name,
            "Token F1 (%)": f"{sub['token_f1'].mean()*100:.1f} ± {sub['token_f1'].std()*100:.1f}",
            "Hit@1 (%)": f"{sub['retrieval_hit1'].mean()*100:.1f}",
            "Retrieval Rate (%)": f"{sub['retrieval_called'].mean()*100:.1f}",
            "Reformulation Rate (%)": f"{sub['reformulated'].mean()*100:.1f}",
            "Net Reward": f"{sub['reward'].mean():+.3f}",
        })
    df_t3 = pd.DataFrame(t3_rows)

    # Save Tables (CSV, Markdown, LaTeX Booktabs)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, (df, name) in enumerate([(df_t1, "table1_overall_results"), (df_t2, "table2_style_breakdown"), (df_t3, "table3_ablation_study")], 1):
        df.to_csv(out_path / f"{name}.csv", index=False)
        with open(out_path / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        with open(out_path / f"{name}.tex", "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False))
        print(f"[Export] Saved {name} (CSV, MD, LaTeX)")

    return df_results, df_t1, df_t2


# ==============================================================================
# 5. Publication-Grade Camera-Ready Figure Generation
# ==============================================================================

def generate_publication_figures(
    df_results: pd.DataFrame,
    history: Dict[str, Any],
    figures_dir: str = "experiments/figures",
) -> None:
    """
    Generate 4 camera-ready 300-DPI publication figures in PNG and vector PDF format.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Academic styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # --------------------------------------------------------------------------
    # Figure 1: Pareto Frontier (Latency vs. Token F1 with Retrieval Bubble Size)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    summary = df_results.groupby("system").agg({
        "latency_ms": "mean",
        "token_f1": "mean",
        "retrieval_called": "mean",
    }).reset_index()

    colors = {
        "Direct Gen (No RAG)": "#7f7f7f",
        "Vanilla Chunk RAG": "#1f77b4",
        "DR-RAG (Ahmad et al., 2026)": "#ff7f0e",
        "Zero-Shot ReAct Agent": "#d62728",
        "Urdu-DynRAG (Ours)": "#2ca02c",
    }

    for _, row in summary.iterrows():
        sys_name = row["system"]
        lat = row["latency_ms"]
        f1 = row["token_f1"] * 100.0
        bubble = max(80.0, row["retrieval_called"] * 600.0)

        ax.scatter(
            lat, f1,
            s=bubble,
            color=colors.get(sys_name, "#333333"),
            alpha=0.85,
            edgecolors="black",
            linewidth=1.2,
            label=f"{sys_name} ({row['retrieval_called']*100:.0f}% retrieval)",
        )
        ax.annotate(
            sys_name,
            (lat + 20, f1 - 0.5),
            fontsize=9,
            weight="bold" if "Ours" in sys_name else "normal",
        )

    # Draw dashed Pareto frontier line connecting efficient points
    pareto_points = summary.sort_values(by="latency_ms")
    ax.plot(
        pareto_points["latency_ms"],
        pareto_points["token_f1"] * 100.0,
        linestyle="--",
        color="#2ca02c",
        alpha=0.6,
        linewidth=1.5,
    )

    ax.set_title("Figure 1: Accuracy-Latency Pareto Frontier", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Mean Latency (ms / query)", fontsize=10)
    ax.set_ylabel("Urdu Token F1 Score (%)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    fig.savefig(fig_dir / "fig1_pareto_frontier.png", dpi=300)
    fig.savefig(fig_dir / "fig1_pareto_frontier.pdf")
    plt.close(fig)
    print("  ✓ Saved Figure 1: Pareto Frontier")

    # --------------------------------------------------------------------------
    # Figure 2: Robustness Under Query Style Distribution Shift
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    style_summary = df_results[df_results["system"].isin([
        "Vanilla Chunk RAG",
        "DR-RAG (Ahmad et al., 2026)",
        "Zero-Shot ReAct Agent",
        "Urdu-DynRAG (Ours)",
    ])].groupby(["query_style", "system"])["token_f1"].mean().reset_index()

    style_summary["token_f1"] *= 100.0
    style_summary["query_style"] = style_summary["query_style"].str.replace("style_", "S").str.replace("_", " ").str.title()

    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    sns.barplot(
        data=style_summary,
        x="query_style",
        y="token_f1",
        hue="system",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Figure 2: Robustness Across Stylistic Variations (Distribution Shift)", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Query Style / Linguistic Register", fontsize=10)
    ax.set_ylabel("Token F1 Score (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(title="System", fontsize=8, title_fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()

    fig.savefig(fig_dir / "fig2_style_robustness_bars.png", dpi=300)
    fig.savefig(fig_dir / "fig2_style_robustness_bars.pdf")
    plt.close(fig)
    print("  ✓ Saved Figure 2: Style Robustness Bars")

    # --------------------------------------------------------------------------
    # Figure 3: Action Routing Matrix Heatmap (Urdu-DynRAG)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ours_df = df_results[df_results["system"] == "Urdu-DynRAG (Ours)"]

    confusion = pd.crosstab(
        ours_df["query_style"].str.replace("style_", "S").str.replace("_", " ").str.title(),
        ours_df["final_action"],
        normalize="index",
    ) * 100.0

    action_col_names = {
        0: "Action 0\n(Direct)",
        1: "Action 1\n(Raw Chunks)",
        2: "Action 2\n(DR-QA)",
        3: "Action 3\n(Reformulate)",
    }
    confusion.rename(columns=action_col_names, inplace=True)

    sns.heatmap(
        confusion,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        cbar_kws={"label": "Action Selection %"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Figure 3: Policy Action Routing Distribution Matrix (%)", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Selected MDP Action", fontsize=10)
    ax.set_ylabel("Input Query Style", fontsize=10)
    plt.tight_layout()

    fig.savefig(fig_dir / "fig3_action_routing_heatmap.png", dpi=300)
    fig.savefig(fig_dir / "fig3_action_routing_heatmap.pdf")
    plt.close(fig)
    print("  ✓ Saved Figure 3: Action Routing Heatmap")

    # --------------------------------------------------------------------------
    # Figure 4: DQN Learning Curve (Reward & Validation F1)
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    epochs = history.get("epoch", list(range(1, 26)))
    train_rew = history.get("train_reward", [])
    val_f1 = [f * 100.0 for f in history.get("val_f1", [])]

    color1 = "#1f77b4"
    ax1.set_xlabel("Training Epoch", fontsize=10)
    ax1.set_ylabel("Cumulative Episode Reward", color=color1, fontsize=10)
    line1 = ax1.plot(epochs, train_rew, color=color1, marker="o", linewidth=1.8, label="Train Reward")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Validation Token F1 (%)", color=color2, fontsize=10)
    line2 = ax2.plot(epochs, val_f1, color=color2, marker="s", linewidth=1.8, linestyle="--", label="Val Token F1")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=8)

    plt.title("Figure 4: Double DQN Convergence & Validation Progress", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()

    fig.savefig(fig_dir / "fig4_dqn_learning_curve.png", dpi=300)
    fig.savefig(fig_dir / "fig4_dqn_learning_curve.pdf")
    plt.close(fig)
    print("  ✓ Saved Figure 4: DQN Learning Curve")


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Urdu-DynRAG Phase 4: Policy Training & Publication Benchmarking"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "evaluate", "all"],
        help="Pipeline mode: 'train', 'evaluate', or 'all'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of DQN training epochs (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for publication tables and figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("URDU-DYNRAG: PHASE 4 EXPERIMENTAL PIPELINE (PUBLICATION TRACK)")
    print("=" * 80)

    # Initialize shared components
    retriever = DualFAISSRetriever(indices_dir="models/indices")
    state_extractor = UrduRAGStateExtractor(retriever=retriever, generator_mode="auto")
    generator = FrozenAnswerGenerator()

    # Create Environments
    env_train = UrduRAGEnv(split="train", retriever=retriever, state_extractor=state_extractor, generator=generator, training_mode=True)
    env_val = UrduRAGEnv(split="val", retriever=retriever, state_extractor=state_extractor, generator=generator, training_mode=True)
    env_test = UrduRAGEnv(split="test", retriever=retriever, state_extractor=state_extractor, generator=generator, training_mode=False)

    policy_net = DQNPolicy()
    history = {}

    # 1. Training Phase
    if args.mode in ["train", "all"]:
        policy_net, history = train_dqn(
            env_train=env_train,
            env_val=env_val,
            num_epochs=args.epochs,
            seed=args.seed,
            history_dir=args.output_dir,
        )
    else:
        # Load existing checkpoint
        best_path = Path("models/policy_best.pt")
        if best_path.exists():
            policy_net.load_state_dict(torch.load(best_path, weights_only=True))
            print(f"[Model Loader] Loaded existing best policy from {best_path}")
        else:
            raise FileNotFoundError("Policy checkpoint not found. Run with `--mode train` first.")

        # Load history if available
        hist_path = Path(args.output_dir) / "training_history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)

    # 2. Evaluation Phase
    if args.mode in ["evaluate", "all"]:
        df_results, df_t1, df_t2 = run_full_benchmark(
            test_items=env_test.dataset,
            retriever=retriever,
            state_extractor=state_extractor,
            generator=generator,
            policy_net=policy_net,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 80)
        print("TABLE 1: OVERALL COMPARATIVE EVALUATION RESULTS")
        print("=" * 80)
        print(df_t1.to_markdown(index=False))

        print("\n" + "=" * 80)
        print("TABLE 2: ACCURACY BREAKDOWN BY QUERY STYLE (DISTRIBUTION SHIFT)")
        print("=" * 80)
        print(df_t2.to_markdown(index=False))

        # 3. Figure Generation
        print("\n[Figures] Generating publication-grade 300 DPI figures...")
        generate_publication_figures(
            df_results=df_results,
            history=history,
            figures_dir=str(Path(args.output_dir) / "figures"),
        )

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE: ALL EXPERIMENTS, TABLES, AND FIGURES GENERATED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

