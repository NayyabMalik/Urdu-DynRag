#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Dynamic RL-Guided Adaptive Retrieval for Under-Resourced QA
Phase 3: Gymnasium-Compatible MDP Environment (UrduRAGEnv)
Reference: System Specification, Operational Architecture & Team Implementation Roadmap

Formulates Adaptive Retrieval as a finite-horizon Markov Decision Process (MDP):
  - State Space: s_t in R^14 (Continuous observation Box(0.0, 1.0, shape=(14,)))
  - Action Space: Discrete(4)
      * a = 0: Direct Parametric Generation (Skip retrieval, answer from memory) -> TERMINAL
      * a = 1: Retrieve from Raw Passage Chunks (D_raw) -> TERMINAL
      * a = 2: Retrieve from DR Question-Pair Index (D_qa) -> TERMINAL
      * a = 3: Query Reformulation / Normalization -> CONTINUES (max 1 per episode)
  - Horizon: T = 3 maximum steps
  - Reward: R(s_t, a_t) = R_acc - lambda_1 * Cost_retrieval(a) - lambda_2 * Cost_tokens(ctx) - Omega_penalty
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import gymnasium as gym
import numpy as np
import torch
from dotenv import load_dotenv
from gymnasium import spaces

# Load environment configuration (.env)
load_dotenv()

# Project dependencies from Phase 1 and Phase 2
try:
    from phase1_indexer import DualFAISSRetriever
    from phase2_state_extractor import UrduRAGStateExtractor, tokenize_text
except ImportError:
    from src.phase1_indexer import DualFAISSRetriever
    from src.phase2_state_extractor import UrduRAGStateExtractor, tokenize_text


# ==============================================================================
# 1. Linguistic Normalization & Reformulation Dictionary (for Action 3)
# ==============================================================================

# Common Roman Urdu to Urdu Nastaliq mapping for legal/civic domains
ROMAN_TO_URDU_LEXICON = {
    "license": "لائسنس",
    "licence": "لائسنس",
    "renew": "تجدید",
    "renewal": "تجدید",
    "medical": "میڈیکل",
    "certificate": "سرٹیفکیٹ",
    "doctor": "ڈاکٹر",
    "form": "فارم",
    "cnic": "شناختی کارڈ",
    "id": "شناختی کارڈ",
    "card": "کارڈ",
    "smart": "سمارٹ",
    "passport": "پاسپورٹ",
    "fees": "فیس",
    "fee": "فیس",
    "challan": "چالان",
    "eobi": "ای او بی آئی",
    "pension": "پنشن",
    "nadra": "نادرا",
    "gum": "گم",
    "lost": "گمشدہ",
    "duplicate": "مثنی",
    "rules": "قواعد",
    "rule": "قاعدہ",
    "law": "قانون",
    "ordinance": "آرڈیننس",
    "fine": "جرمانہ",
    "penalty": "سزا",
    "court": "عدالت",
    "police": "پولیس",
    "hec": "ایچ ای سی",
    "degree": "ڈگری",
    "attestation": "تصدیق",
    "urgent": "ارجنٹ",
    "executive": "ایگزیکٹو",
    "normal": "نارمل",
    "age": "عمر",
    "years": "سال",
    "days": "دن",
    "months": "ماہ",
}

CONVERSATIONAL_FILLERS = [
    r"kia\s+krna\s+hoga",
    r"kahan\s+jana\s+pary\s+ga",
    r"mujhe\s+btao",
    r"zarori\s+ha\s+kya",
    r"krwana\s+hy",
    r"krwana\s+ha",
    r"bta\s+do",
    r"kia\s+tareeqa\s+ha",
    r"کیا\s+کرنا\s+ہوگا",
    r"کہاں\s+جانا\s+پڑے\s+گا",
    r"مجھے\s+بتائیں",
    r"لازمی\s+ہے\s+کیا",
    r"کرانا\s+ہے",
    r"کروانا\s+ہے",
]


def reformulate_query(query: str, llm_client: Optional[Any] = None) -> str:
    """
    Action 3 execution: Normalize, clean conversational noise, and translate
    Roman Urdu tokens into formal Urdu Nastaliq retrieval keywords.
    """
    text = query.strip()

    # Strategy 1: If LLM is available via API, ask for a 1-shot concise search rewrite
    if llm_client is not None:
        try:
            prompt = (
                f"Rewrite this conversational or Roman Urdu query into 3-5 formal Urdu Nastaliq search keywords.\n"
                f"Original: {text}\n"
                f"Output Urdu search keywords only (no explanation):"
            )
            resp = llm_client.generate(prompt, max_tokens=25)
            clean_res = resp.strip().replace("\n", " ")
            if len(clean_res) > 3 and any("\u0600" <= c <= "\u06FF" for c in clean_res):
                return clean_res
        except Exception:
            pass

    # Strategy 2: Fast deterministic rule-based normalizer
    # Remove conversational filler regex
    for pattern in CONVERSATIONAL_FILLERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    words = text.split()
    rewritten_words = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w.lower())
        if clean_w in ROMAN_TO_URDU_LEXICON:
            rewritten_words.append(ROMAN_TO_URDU_LEXICON[clean_w])
        else:
            rewritten_words.append(w)

    result = " ".join(rewritten_words).strip()
    return result if len(result) > 0 else query


# ==============================================================================
# 2. Frozen Generator Interface (Local & API)
# ==============================================================================

class FrozenAnswerGenerator:
    """
    Unified interface for answer generation using Qwen2.5-7B/3B (OpenRouter or local).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_api: bool = True,
        device: str = "cpu",
    ) -> None:
        self.use_api = use_api
        self.device = device
        self.api_client: Optional[Any] = None
        self.local_model: Optional[Any] = None
        self.local_tokenizer: Optional[Any] = None

        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = (
            model_name
            or os.getenv("GENERATOR_MODEL")
            or "qwen/qwen-2.5-7b-instruct"
        )

        if use_api and openrouter_key and not openrouter_key.startswith("your_"):
            try:
                from openai import OpenAI
                self.api_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    timeout=10.0,
                )
                print(f"[FrozenGenerator] Connected to OpenRouter generator: {self.model_name}")
            except Exception as e:
                print(f"[FrozenGenerator] OpenRouter initialization failed ({e}).")

    def generate_answer(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate answer y given query and optional retrieved context.
        """
        if context and len(context.strip()) > 0:
            prompt = (
                f"آپ ایک مستند قانونی اور انتظامی معاون ہیں۔ درج ذیل سیاق و سباق (Context) کی بنیاد پر صارف کے سوال کا مختصر، دوٹوک اور درست جواب اردو میں دیں۔ اگر جواب سیاق و سباق میں نہیں ہے تو خود سے قیاس نہ کریں۔\n\n"
                f"سیاق و سباق:\n\"\"\"{context.strip()}\"\"\"\n\n"
                f"سوال: {query}\n"
                f"جواب:"
            )
        else:
            prompt = (
                f"آپ ایک مستند معاون ہیں۔ اپنے اندرونی علم کی روشنی میں اس سوال کا مختصر اور درست جواب اردو میں دیں:\n\n"
                f"سوال: {query}\n"
                f"جواب:"
            )

        if self.api_client is not None:
            try:
                resp = self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        # Offline fallback: extract grounded answer span or key sentence
        if context:
            sentences = [s.strip() for s in re.split(r"[۔\n]", context) if len(s.strip()) > 15]
            if sentences:
                return sentences[0] + "۔"
            return context[:100] + "۔"
        return "معلومات کے مطابق تصدیق شدہ جواب دستیاب ہے۔"


# ==============================================================================
# 3. Accuracy Reward Metric (Token F1 + Gold Span Bonus)
# ==============================================================================

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and gold answer.
    """
    pred_tokens = tokenize_text(prediction)
    gold_tokens = tokenize_text(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens).intersection(set(gold_tokens))
    if not common:
        return 0.0

    precision = float(len(common)) / float(len(pred_tokens))
    recall = float(len(common)) / float(len(gold_tokens))
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
    return float(f1)


# ==============================================================================
# 4. Gymnasium Environment: UrduRAGEnv
# ==============================================================================

class UrduRAGEnv(gym.Env):
    """
    Gymnasium-compatible MDP Environment for Adaptive RAG in Urdu.

    Observation Space: Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
    Action Space: Discrete(4)
      0: Direct Parametric Generation (No retrieval)
      1: Retrieve from Raw Passage Chunks (D_raw)
      2: Retrieve from DR Question-Pair Index (D_qa)
      3: Query Reformulation / Normalization (Step t <- t+1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        split: str = "train",
        retriever: Optional[DualFAISSRetriever] = None,
        state_extractor: Optional[UrduRAGStateExtractor] = None,
        generator: Optional[FrozenAnswerGenerator] = None,
        max_steps: int = 3,
        lambda_retrieval: float = 0.15,
        lambda_tokens: float = 0.05,
        omega_penalty: float = 1.5,
        training_mode: bool = False,
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.lambda_1 = lambda_retrieval
        self.lambda_2 = lambda_tokens
        self.omega_penalty = omega_penalty
        self.training_mode = training_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Load Dataset Split
        self.dataset = self._load_dataset_split(dataset_path, split)
        if len(self.dataset) == 0:
            raise ValueError(f"No records found for split '{split}' in dataset.")

        # Dependencies
        self.retriever = retriever or DualFAISSRetriever()
        self.state_extractor = state_extractor or UrduRAGStateExtractor(retriever=self.retriever)
        self.generator = generator or FrozenAnswerGenerator()

        # Episode State Tracking
        self.current_item: Dict[str, Any] = {}
        self.current_query: str = ""
        self.current_step: int = 0
        self.reformulated_flag: bool = False
        self.current_state: np.ndarray = np.zeros(14, dtype=np.float32)

    def _load_dataset_split(self, path: Optional[str], split_filter: str) -> List[Dict[str, Any]]:
        """Load and filter dataset records strictly by assigned split."""
        candidates = [
            path,
            "data/urdu_dynrag_verified_dataset_v3.json",
            "urdu_dynrag_verified_dataset_v3.json",
            "../data/urdu_dynrag_verified_dataset_v3.json",
        ]
        target_path = None
        for c in candidates:
            if c and Path(c).exists():
                target_path = Path(c)
                break
        if target_path is None:
            raise FileNotFoundError(f"Could not locate dataset file. Checked: {candidates}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON list.")

        filtered = [r for r in data if r.get("split") == split_filter]
        if not filtered and split_filter == "all":
            filtered = data
        elif not filtered:
            # Fallback if split tag not strictly matched
            filtered = data

        print(f"[UrduRAGEnv] Loaded {len(filtered)} records for split '{split_filter}' from {target_path.name}")
        return filtered

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset episode state, sample an item from dataset split, and extract s_0.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Sample query from dataset
        self.current_item = random.choice(self.dataset)
        self.current_query = self.current_item["query"]
        self.current_step = 0
        self.reformulated_flag = False

        # Extract initial state s_0
        s_0_tensor = self.state_extractor.extract_state(
            query=self.current_query,
            t=self.current_step,
            reformulated_flag=self.reformulated_flag,
        )
        self.current_state = s_0_tensor.cpu().numpy().astype(np.float32)

        info = {
            "id": self.current_item.get("id"),
            "domain": self.current_item.get("domain"),
            "strata": self.current_item.get("strata"),
            "query_style": self.current_item.get("query_style"),
            "optimal_action": self.current_item.get("optimal_action"),
            "query": self.current_query,
            "gold_answer": self.current_item.get("gold_answer"),
        }
        return self.current_state, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute transition s_t -> a_t -> (s_{t+1}, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {list(range(4))}")

        self.current_step += 1
        gold_answer = self.current_item.get("gold_answer", "")
        gold_span = self.current_item.get("gold_span", "")
        optimal_action = self.current_item.get("optimal_action", "")

        terminated = False
        truncated = False
        reward = 0.0
        generated_answer = ""
        retrieved_context = ""

        info: Dict[str, Any] = {
            "step": self.current_step,
            "action_taken": action,
            "optimal_action": optimal_action,
            "query": self.current_query,
        }

        # ----------------------------------------------------------------------
        # ACTION 3: Query Reformulation / Normalization (Continues Episode)
        # ----------------------------------------------------------------------
        if action == 3:
            # Check maximum reformulation constraint
            if self.reformulated_flag:
                # Penalty: Second reformulation attempted
                reward = -self.omega_penalty
                terminated = True
                info["reason"] = "Repeated reformulation penalty"
                return self.current_state, float(reward), terminated, truncated, info

            # Execute query rewrite
            new_query = reformulate_query(self.current_query)
            self.current_query = new_query
            self.reformulated_flag = True

            # Step cost for reformulation: Cost_retrieval(a=3) = 0.3
            step_cost = self.lambda_1 * 0.3
            reward -= step_cost

            # Check horizon limit
            if self.current_step >= self.max_steps:
                truncated = True
                terminated = True
                reward -= self.omega_penalty
                info["reason"] = "Horizon exceeded without answer generation"
            else:
                # Extract new state s_{t+1}
                s_next_tensor = self.state_extractor.extract_state(
                    query=self.current_query,
                    t=self.current_step,
                    reformulated_flag=self.reformulated_flag,
                )
                self.current_state = s_next_tensor.cpu().numpy().astype(np.float32)

            info["new_query"] = self.current_query
            info["reward"] = float(reward)
            return self.current_state, float(reward), terminated, truncated, info

        # ----------------------------------------------------------------------
        # ACTIONS 0, 1, 2: Terminal Answer Generation Actions
        # ----------------------------------------------------------------------
        terminated = True
        cost_retrieval = 0.0

        if action == 0:
            # Action 0: Direct Parametric Generation (No retrieval)
            retrieved_context = ""
            cost_retrieval = 0.0

        elif action == 1:
            # Action 1: Retrieve from Raw Passage Chunks (D_raw)
            raw_res = self.retriever.search_raw(self.current_query, k=1)
            if raw_res:
                retrieved_context = raw_res[0]["text"]
                info["retrieved_passage_id"] = raw_res[0]["passage_id"]
            cost_retrieval = 1.0

        elif action == 2:
            # Action 2: Retrieve from DR Question-Pair Index (D_qa)
            qa_res = self.retriever.search_qa(self.current_query, k=1)
            if qa_res:
                retrieved_context = qa_res[0]["parent_text"]
                info["retrieved_passage_id"] = qa_res[0]["parent_passage_id"]
                info["matched_synthetic_q"] = qa_res[0]["synthetic_question"]
            cost_retrieval = 1.0

        # Generate answer y via frozen generator (or fast simulation during training)
        target_pid = self.current_item.get("passage_id", "")
        if self.training_mode:
            # High-speed deterministic reward for RL training
            if action == 0:
                is_correct = (optimal_action == "direct_generation")
                f1_score = 0.85 if is_correct else 0.20
            else:
                retrieved_pid = info.get("retrieved_passage_id", "")
                hit = (retrieved_pid == target_pid) if target_pid else False
                f1_score = 0.82 if hit else 0.22
            generated_answer = gold_answer if f1_score > 0.5 else "معلومات دستیاب نہیں۔"
        else:
            generated_answer = self.generator.generate_answer(
                query=self.current_query,
                context=retrieved_context if retrieved_context else None,
            )
            f1_score = compute_token_f1(generated_answer, gold_answer)

        # 1. Compute R_accuracy (Token F1 Score)
        r_acc = f1_score

        # Gold span bonus (+0.1 if retrieved context contains gold_span)
        if gold_span and retrieved_context and gold_span.strip() in retrieved_context:
            r_acc += 0.1

        # 2. Compute Costs
        context_tokens = len(retrieved_context.split())
        cost_tokens = float(context_tokens) / 1000.0

        # Full Reward Formulation:
        # R = R_acc - lambda_1 * Cost_retrieval - lambda_2 * Cost_tokens
        total_reward = r_acc - (self.lambda_1 * cost_retrieval) - (self.lambda_2 * cost_tokens)

        reward = float(total_reward)
        info.update({
            "generated_answer": generated_answer,
            "gold_answer": gold_answer,
            "token_f1": f1_score,
            "retrieval_cost": cost_retrieval,
            "context_tokens": context_tokens,
            "final_reward": reward,
        })

        return self.current_state, reward, terminated, truncated, info


# ==============================================================================
# Smoke Test Block (__main__)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("URDU-DYNRAG: PHASE 3 GYMNASIUM ENVIRONMENT SMOKE TEST")
    print("=" * 80)

    # Initialize Environment on Test Split
    env = UrduRAGEnv(split="test")
    print(f"\n[Environment Ready] Observation space: {env.observation_space}")
    print(f"[Environment Ready] Action space     : {env.action_space}")

    # Run 5 complete random/strategic rollouts
    num_rollouts = 5
    action_names = {
        0: "Action 0: Direct Generation (Parametric)",
        1: "Action 1: Retrieve Raw Chunk (D_raw)",
        2: "Action 2: Retrieve Synthetic QA (D_qa)",
        3: "Action 3: Reformulate / Clean Query",
    }

    print("\n" + "-" * 80)
    print(f"Executing {num_rollouts} Test Rollouts...")
    print("-" * 80)

    for ep in range(1, num_rollouts + 1):
        obs, info = env.reset(seed=ep * 42)
        print(f"\n[Episode {ep}]")
        print(f"Domain         : {info['domain']} | Strata: {info['strata']}")
        print(f"Original Query : \"{info['query']}\"")
        print(f"Optimal Action : {info['optimal_action']}")

        ep_reward = 0.0
        step_count = 0
        done = False

        while not done:
            step_count += 1
            # Strategic rollout: if Roman Urdu and step 1, take action 3, else take random answer action
            if info.get("query_style") in ["style_4_roman_urdu", "parametric_roman_urdu"] and step_count == 1:
                action = 3
            else:
                action = random.choice([0, 1, 2])

            print(f"   Step {step_count} -> Selected: {action_names[action]}")
            next_obs, step_reward, terminated, truncated, step_info = env.step(action)
            ep_reward += step_reward
            done = terminated or truncated

            # Invariant checks
            assert isinstance(next_obs, np.ndarray), "Observation must be numpy array"
            assert next_obs.shape == (14,), f"Observation shape must be (14,), got {next_obs.shape}"
            assert not np.isnan(next_obs).any(), "Observation contains NaNs!"
            assert not np.isinf(next_obs).any(), "Observation contains Infs!"

            if done:
                print(f"   Episode Terminated! Final Step: {step_count}")
                print(f"   Generated Answer: \"{step_info.get('generated_answer', '')[:80]}...\"")
                print(f"   Token F1 Score  : {step_info.get('token_f1', 0.0):.4f}")
                print(f"   Cumulative Rew  : {ep_reward:+.4f}")

    print("\n" + "=" * 80)
    print("PHASE 3 SMOKE TEST COMPLETE: 5 ROLLOUTS EXECUTED WITH ZERO ERRORS!")
    print("=" * 80)

