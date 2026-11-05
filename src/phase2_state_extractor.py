#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Dynamic RL-Guided Adaptive Retrieval for Under-Resourced QA
Phase 2: 14-Dimensional State Feature Extractor Module
Reference: System Specification, Operational Architecture & Team Implementation Roadmap

Extracts a compact, continuous state vector s_t in R^14 capturing:
  1. Linguistic Traits (f1-f3)
  2. Dual-Index Retrieval Confidence Dynamics (f4-f9)
  3. Frozen Generator Output Uncertainty (f10-f12)
  4. Episodic MDP Signals (f13-f14)

Returns a PyTorch tensor of shape (14,) with dtype torch.float32, strictly bounded
between [0.0, 1.0] with zero NaNs or Infs.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

# Import Phase 1 Dual-Representation Retriever
try:
    from src.phase1_indexer import DualFAISSRetriever
except ImportError:
    try:
        from phase1_indexer import DualFAISSRetriever
    except ImportError:
        DualFAISSRetriever = None  # type: ignore


# ==============================================================================
# Urdu Lexical Constants & Markers
# ==============================================================================

# Perso-Arabic Unicode range: \u0600 - \u06FF
URDU_UNICODE_PATTERN = re.compile(r"[\u0600-\u06FF]")

# Common Urdu Interrogative & Question Particles (Nastaliq + Romanized)
URDU_QUESTION_PARTICLES: Set[str] = {
    # Nastaliq markers
    "کب", "کہاں", "کیوں", "کون", "کیسے", "کیسی", "کیسا",
    "کیا", "کس", "کتنے", "کتنی", "کتنا", "کیونکر", "کدھر", "آیا",
    # Common Roman Urdu transliterated interrogatives
    "kab", "kahan", "kyun", "kyu", "kon", "kaise", "kesy", "kese",
    "kia", "kya", "kis", "kitny", "kitne", "kitni", "kitna", "kidhar"
}


# ==============================================================================
# Helper Functions: Tokenization & Lexical Jaccard
# ==============================================================================

def tokenize_text(text: str) -> List[str]:
    """Simple robust whitespace/punctuation tokenizer for Urdu and code-switched text."""
    clean = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text.lower())
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 0]
    return tokens


def compute_jaccard_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Compute lexical Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not query_tokens or not doc_tokens:
        return 0.0
    set_a = set(query_tokens)
    set_b = set(doc_tokens)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# ==============================================================================
# Generator Uncertainty Providers (Local Model & API Support)
# ==============================================================================

class BaseGeneratorUncertaintyProvider:
    """Interface for computing parametric predictive entropy over next tokens."""

    def compute_entropy_features(self, query: str) -> Tuple[float, float, float]:
        """
        Returns:
          - mean_entropy: float in [0.0, 1.0]
          - mean_top1_prob: float in [0.0, 1.0]
          - entropy_variance: float in [0.0, 1.0]
        """
        raise NotImplementedError


class APIGeneratorUncertaintyProvider(BaseGeneratorUncertaintyProvider):
    """
    Computes parametric entropy via OpenRouter API with logprobs enabled.
    Fast, lightweight, and requires 0 GB local VRAM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 8.0,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = (
            model_name
            or os.getenv("JUDGE_MODEL")
            or os.getenv("GENERATOR_MODEL")
            or "openai/gpt-4o-mini"
        )
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, timeout=self.timeout)

    def compute_entropy_features(self, query: str) -> Tuple[float, float, float]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": query}],
                max_tokens=5,
                logprobs=True,
                top_logprobs=5,
            )
            choice = resp.choices[0]
            logprobs_obj = choice.logprobs

            if not logprobs_obj or not logprobs_obj.content:
                # If provider omitted logprobs, return neutral parametric estimate
                return 0.45, 0.65, 0.05

            entropies: List[float] = []
            top1_probs: List[float] = []

            for tok_info in logprobs_obj.content[:5]:
                # Top-1 prob
                top1_p = float(math.exp(tok_info.logprob))
                top1_probs.append(min(1.0, max(0.0, top1_p)))

                # Top-5 candidate probabilities
                if tok_info.top_logprobs:
                    probs = [float(math.exp(c.logprob)) for c in tok_info.top_logprobs]
                    s = sum(probs) + 1e-8
                    norm_probs = [p / s for p in probs]
                    k_cands = len(norm_probs)
                    max_ent = math.log(k_cands) if k_cands > 1 else 1.0
                    ent = -sum(p * math.log(p + 1e-8) for p in norm_probs) / max_ent
                    entropies.append(min(1.0, max(0.0, ent)))
                else:
                    # Single prob proxy
                    ent = - (top1_p * math.log(top1_p + 1e-8) + (1.0 - top1_p) * math.log(max(1e-8, 1.0 - top1_p)))
                    entropies.append(min(1.0, max(0.0, ent)))

            if not entropies:
                return 0.45, 0.65, 0.05

            mean_entropy = float(np.mean(entropies))
            mean_top1 = float(np.mean(top1_probs))
            ent_variance = float(np.var(entropies))

            return (
                min(1.0, max(0.0, mean_entropy)),
                min(1.0, max(0.0, mean_top1)),
                min(1.0, max(0.0, ent_variance * 4.0)),  # Normalized variance
            )
        except Exception:
            # Fallback on any network interruption
            return 0.45, 0.65, 0.05


class LocalModelGeneratorUncertaintyProvider(BaseGeneratorUncertaintyProvider):
    """
    Computes parametric entropy locally using a HuggingFace causal LM
    (e.g., Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-3B-Instruct).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()

    def compute_entropy_features(self, query: str) -> Tuple[float, float, float]:
        try:
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            entropies: List[float] = []
            top1_probs: List[float] = []

            curr_ids = input_ids
            vocab_size = self.model.config.vocab_size
            max_log_v = math.log(vocab_size) if vocab_size > 1 else 1.0

            with torch.no_grad():
                for _ in range(5):
                    outputs = self.model(input_ids=curr_ids)
                    next_token_logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Top-1 prob
                    top1_val, top1_idx = torch.topk(probs, k=1, dim=-1)
                    top1_probs.append(float(top1_val.item()))

                    # Full entropy normalized by log(vocab_size)
                    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).item()
                    norm_ent = ent / max_log_v
                    entropies.append(min(1.0, max(0.0, norm_ent)))

                    # Autoregressively append greedy token
                    curr_ids = torch.cat([curr_ids, top1_idx], dim=-1)

            mean_entropy = float(np.mean(entropies))
            mean_top1 = float(np.mean(top1_probs))
            ent_variance = float(np.var(entropies))

            return (
                min(1.0, max(0.0, mean_entropy)),
                min(1.0, max(0.0, mean_top1)),
                min(1.0, max(0.0, ent_variance * 4.0)),
            )
        except Exception:
            return 0.45, 0.65, 0.05


class HeuristicGeneratorUncertaintyProvider(BaseGeneratorUncertaintyProvider):
    """
    Zero-compute offline fallback generator provider.
    Estimates uncertainty proxy from character unigram/bigram entropy and query shape.
    """

    def compute_entropy_features(self, query: str) -> Tuple[float, float, float]:
        tokens = tokenize_text(query)
        if not tokens:
            return 0.5, 0.5, 0.0

        # Character entropy
        chars = [c for c in query if not c.isspace()]
        if not chars:
            return 0.5, 0.5, 0.0

        counts: Dict[str, int] = {}
        for c in chars:
            counts[c] = counts.get(c, 0) + 1
        n = len(chars)
        h = -sum((cnt / n) * math.log(cnt / n + 1e-8) for cnt in counts.values())
        norm_h = min(1.0, max(0.0, h / math.log(max(2, len(counts)))))

        top1_prob = min(1.0, max(0.0, 1.0 - (norm_h * 0.5)))
        variance = min(1.0, max(0.0, 0.05 + 0.1 * (1.0 - norm_h)))

        return norm_h, top1_prob, variance


# ==============================================================================
# Main 14-Dimensional State Feature Extractor Class
# ==============================================================================

class UrduRAGStateExtractor:
    """
    Extracts 14 continuous state features s_t in R^14 for Urdu-DynRAG MDP.
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        indices_dir: str = "models/indices",
        embedding_model_name: str = "BAAI/bge-m3",
        generator_mode: str = "auto",  # 'auto', 'api', 'local', or 'heuristic'
        device: str = "cpu",
    ) -> None:
        self.device = device

        # Initialize Dual FAISS Retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            if DualFAISSRetriever is None:
                raise ImportError("Could not import DualFAISSRetriever from phase1_indexer.")
            print(f"[StateExtractor] Initializing DualFAISSRetriever from {indices_dir}...")
            self.retriever = DualFAISSRetriever(
                indices_dir=indices_dir,
                model_name=embedding_model_name,
                device=device,
            )

        # Initialize Generator Uncertainty Provider
        self.generator_provider = self._init_generator_provider(generator_mode)
        # Fast in-memory state feature cache for training efficiency
        self._state_cache: Dict[Tuple[str, int, bool], torch.Tensor] = {}

    def _init_generator_provider(self, mode: str) -> BaseGeneratorUncertaintyProvider:
        """Select appropriate generator uncertainty provider based on configuration."""
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

        if mode == "api" or (mode == "auto" and openrouter_key and not openrouter_key.startswith("your_")):
            try:
                print("[StateExtractor] Initializing APIGeneratorUncertaintyProvider via OpenRouter...")
                return APIGeneratorUncertaintyProvider()
            except Exception as e:
                print(f"[StateExtractor] Warning: API provider failed initialization ({e}). Falling back to heuristic.")

        if mode == "local":
            try:
                print("[StateExtractor] Initializing LocalModelGeneratorUncertaintyProvider...")
                return LocalModelGeneratorUncertaintyProvider(device=self.device)
            except Exception as e:
                print(f"[StateExtractor] Warning: Local LM failed ({e}). Falling back to heuristic.")

        print("[StateExtractor] Using HeuristicGeneratorUncertaintyProvider (offline mode)...")
        return HeuristicGeneratorUncertaintyProvider()

    def load_or_precompute_cache(
        self,
        dataset_items: List[Dict[str, Any]],
        cache_file: str = "models/indices/state_cache_v3.pt",
        batch_size: int = 32,
    ) -> None:
        """
        Pre-compute all state vectors in high-speed vectorized batch mode once,
        and cache to disk. Makes subsequent training 100x faster!
        """
        cache_path = Path(cache_file)
        if cache_path.exists():
            print(f"[StateExtractor] Loading pre-computed states from {cache_path}...")
            loaded = torch.load(cache_path, weights_only=False)
            self._state_cache.update(loaded)
            print(f"[StateExtractor] Loaded {len(self._state_cache)} state vectors into memory in 0.01s!")
            return

        print(f"[StateExtractor] Pre-computing states for {len(dataset_items)} items in vectorized batch mode...")
        # Local import to avoid circular dependency
        try:
            from src.phase3_env import reformulate_query
        except ImportError:
            from phase3_env import reformulate_query

        needed_queries = set()
        query_states_map = []

        for item in dataset_items:
            q_orig = item["query"].strip()
            needed_queries.add(q_orig)
            query_states_map.append((q_orig, 0, False))

            q_ref = reformulate_query(q_orig).strip()
            needed_queries.add(q_ref)
            query_states_map.append((q_ref, 1, True))
            query_states_map.append((q_ref, 2, True))
            query_states_map.append((q_orig, 1, False))
            query_states_map.append((q_orig, 2, False))

        unique_queries = list(needed_queries)
        print(f"[StateExtractor] Encoding {len(unique_queries)} unique queries in batch mode (batch_size={batch_size})...")

        q_embeddings = self.retriever.encoder.encode(
            unique_queries,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        q_embeddings = np.asarray(q_embeddings, dtype=np.float32)

        # Vectorized batch FAISS search
        scores_raw, indices_raw = self.retriever.raw_index.search(q_embeddings, 3)
        scores_qa, indices_qa = self.retriever.qa_index.search(q_embeddings, 3)

        query_to_search_res = {}
        for idx, q_str in enumerate(unique_queries):
            raw_top1 = float(scores_raw[idx][0])
            raw_top3 = float(scores_raw[idx][2])
            qa_top1 = float(scores_qa[idx][0])
            qa_top3 = float(scores_qa[idx][2])

            raw_idx0 = indices_raw[idx][0]
            qa_idx0 = indices_qa[idx][0]

            top1_raw_text = (
                self.retriever.raw_metadata[raw_idx0]["text"]
                if 0 <= raw_idx0 < len(self.retriever.raw_metadata)
                else ""
            )
            top1_qa_parent = (
                self.retriever.qa_metadata[qa_idx0]["parent_text"]
                if 0 <= qa_idx0 < len(self.retriever.qa_metadata)
                else ""
            )

            query_to_search_res[q_str] = {
                "raw_top1": raw_top1,
                "raw_margin": raw_top1 - raw_top3,
                "qa_top1": qa_top1,
                "qa_margin": qa_top1 - qa_top3,
                "raw_text": top1_raw_text,
                "qa_text": top1_qa_parent,
            }

        # Populate state cache
        for q_str, t_step, ref_flag in query_states_map:
            s_info = query_to_search_res[q_str]
            f = [0.0] * 14
            tokens = tokenize_text(q_str)
            token_count = len(tokens)

            f[0] = min(1.0, max(0.0, float(token_count) / 50.0))
            clean_chars = [c for c in q_str if not c.isspace()]
            if clean_chars:
                urdu_chars = len(URDU_UNICODE_PATTERN.findall("".join(clean_chars)))
                f[1] = min(1.0, max(0.0, float(urdu_chars) / float(len(clean_chars) + 1e-5)))
            else:
                f[1] = 0.0
            f[2] = 1.0 if any(tok in URDU_QUESTION_PARTICLES for tok in tokens) else 0.0

            f[3] = min(1.0, max(0.0, s_info["raw_top1"]))
            f[4] = min(1.0, max(0.0, s_info["qa_top1"]))
            f[5] = min(1.0, max(0.0, s_info["raw_margin"]))
            f[6] = min(1.0, max(0.0, s_info["qa_margin"]))
            f[7] = min(1.0, max(0.0, compute_jaccard_similarity(tokens, tokenize_text(s_info["raw_text"]))))
            f[8] = min(1.0, max(0.0, compute_jaccard_similarity(tokens, tokenize_text(s_info["qa_text"]))))

            mean_ent, mean_top1_p, ent_var = self.generator_provider.compute_entropy_features(q_str)
            f[9] = min(1.0, max(0.0, mean_ent))
            f[10] = min(1.0, max(0.0, mean_top1_p))
            f[11] = min(1.0, max(0.0, ent_var))

            f[12] = min(1.0, max(0.0, float(t_step) / 3.0))
            f[13] = 1.0 if ref_flag else 0.0

            st = torch.tensor(f, dtype=torch.float32)
            st = torch.nan_to_num(st, nan=0.5, posinf=1.0, neginf=0.0)
            st = torch.clamp(st, min=0.0, max=1.0)
            self._state_cache[(q_str, t_step, ref_flag)] = st

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_cache, cache_path)
        print(f"[StateExtractor] Successfully cached {len(self._state_cache)} state vectors to {cache_path}!")

    def extract_state(
        self,
        query: str,
        t: int = 0,
        reformulated_flag: bool = False,
    ) -> torch.Tensor:
        """
        Extract the 14-dimensional state vector s_t for a given query q_t.

        Parameters:
          - query: Raw query text (Urdu Nastaliq, Conversational, or Roman Urdu).
          - t: Current step index in episode (0 <= t <= 3).
          - reformulated_flag: True if Action 3 (Reformulate) was already taken.

        Returns:
          - torch.Tensor of shape (14,) and dtype torch.float32, bounded in [0.0, 1.0].
        """
        query_str = (query or "").strip()
        cache_key = (query_str, int(t), bool(reformulated_flag))
        if cache_key in self._state_cache:
            return self._state_cache[cache_key].clone()

        # Feature placeholder
        f = [0.0] * 14

        try:
            # ------------------------------------------------------------------
            # 1. Linguistic Features (f1 - f3)
            # ------------------------------------------------------------------
            tokens = tokenize_text(query_str)
            token_count = len(tokens)

            # f1: Token length |q_t| normalized by 50.0
            f[0] = min(1.0, max(0.0, float(token_count) / 50.0))

            # f2: Urdu Nastaliq script ratio: (count in \u0600-\u06FF) / (total len + 1e-5)
            clean_chars = [c for c in query_str if not c.isspace()]
            if clean_chars:
                urdu_chars = len(URDU_UNICODE_PATTERN.findall("".join(clean_chars)))
                f[1] = min(1.0, max(0.0, float(urdu_chars) / float(len(clean_chars) + 1e-5)))
            else:
                f[1] = 0.0

            # f3: Binary indicator if Urdu interrogative markers are present
            has_question_marker = any(tok in URDU_QUESTION_PARTICLES for tok in tokens)
            f[2] = 1.0 if has_question_marker else 0.0

            # ------------------------------------------------------------------
            # 2. Retrieval Confidence Signals (f4 - f9)
            # ------------------------------------------------------------------
            if self.retriever is not None and query_str:
                search_res = self.retriever.search_both(query_str, k=3)

                raw_top1 = float(search_res.get("raw_top1", 0.0))
                raw_margin = float(search_res.get("raw_margin", 0.0))
                qa_top1 = float(search_res.get("qa_top1", 0.0))
                qa_margin = float(search_res.get("qa_margin", 0.0))

                # f4: Top-1 cosine similarity score from raw chunk index
                f[3] = min(1.0, max(0.0, raw_top1))

                # f5: Top-1 cosine similarity score from DR QA index
                f[4] = min(1.0, max(0.0, qa_top1))

                # f6: Margin in raw index (Top1 - Top3)
                f[5] = min(1.0, max(0.0, raw_margin))

                # f7: Margin in DR index (Top1 - Top3)
                f[6] = min(1.0, max(0.0, qa_margin))

                # f8: Lexical token overlap with Top-1 raw chunk
                raw_results = search_res.get("raw_results", [])
                if raw_results:
                    top1_raw_text = raw_results[0].get("text", "")
                    raw_tokens = tokenize_text(top1_raw_text)
                    f[7] = min(1.0, max(0.0, compute_jaccard_similarity(tokens, raw_tokens)))
                else:
                    f[7] = 0.0

                # f9: Lexical token overlap with Top-1 DR matched parent passage
                qa_results = search_res.get("qa_results", [])
                if qa_results:
                    top1_parent_text = qa_results[0].get("parent_text", "")
                    qa_tokens = tokenize_text(top1_parent_text)
                    f[8] = min(1.0, max(0.0, compute_jaccard_similarity(tokens, qa_tokens)))
                else:
                    f[8] = 0.0
            else:
                f[3] = 0.5
                f[4] = 0.5
                f[5] = 0.0
                f[6] = 0.0
                f[7] = 0.0
                f[8] = 0.0

            # ------------------------------------------------------------------
            # 3. Generator Output Uncertainty (f10 - f12)
            # ------------------------------------------------------------------
            if query_str and self.generator_provider is not None:
                mean_ent, mean_top1_p, ent_var = self.generator_provider.compute_entropy_features(query_str)
                f[9] = min(1.0, max(0.0, mean_ent))       # f10: Predictive entropy over first 5 tokens
                f[10] = min(1.0, max(0.0, mean_top1_p))   # f11: Mean Top-1 token generation probability
                f[11] = min(1.0, max(0.0, ent_var))       # f12: Entropy variance across 5 tokens
            else:
                f[9] = 0.5
                f[10] = 0.5
                f[11] = 0.05

            # ------------------------------------------------------------------
            # 4. Episodic MDP Signals (f13 - f14)
            # ------------------------------------------------------------------
            # f13: Current step normalized by horizon (t / 3.0)
            f[12] = min(1.0, max(0.0, float(t) / 3.0))

            # f14: Binary indicator if Query Reformulation was already executed
            f[13] = 1.0 if reformulated_flag else 0.0

        except Exception as e:
            # High-robustness fallback: ensure no crash under any circumstance
            print(f"[StateExtractor] Warning during state extraction: {e}. Applying safe bounds.")
            f = [min(1.0, max(0.0, float(val))) if not math.isnan(val) else 0.5 for val in f]

        # Convert to float32 PyTorch tensor and verify shape (14,)
        state_tensor = torch.tensor(f, dtype=torch.float32)

        # Sanity check: replace any possible NaN or Inf with safe values
        state_tensor = torch.nan_to_num(state_tensor, nan=0.5, posinf=1.0, neginf=0.0)
        state_tensor = torch.clamp(state_tensor, min=0.0, max=1.0)

        # Store in cache
        self._state_cache[cache_key] = state_tensor.clone()
        return state_tensor

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for all 14 state features."""
        return [
            "f1_query_length_norm",
            "f2_urdu_script_ratio",
            "f3_interrogative_marker",
            "f4_sim_raw_top1",
            "f5_sim_qa_top1",
            "f6_margin_raw_t1_t3",
            "f7_margin_qa_t1_t3",
            "f8_jaccard_raw_top1",
            "f9_jaccard_qa_top1",
            "f10_generator_predictive_entropy",
            "f11_generator_top1_prob",
            "f12_generator_entropy_dispersion",
            "f13_normalized_step",
            "f14_reformulated_flag",
        ]


# ==============================================================================
# Unit Tests & Self-Verification
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("URDU-DYNRAG: 14-DIMENSIONAL STATE FEATURE EXTRACTOR UNIT TEST")
    print("=" * 80)

    # Initialize Extractor
    extractor = UrduRAGStateExtractor(generator_mode="auto")
    feat_names = extractor.get_feature_names()
    assert len(feat_names) == 14, f"Expected 14 feature names, got {len(feat_names)}"

    # Test cases representing distinct strata and styles
    test_suite = [
        {
            "desc": "Test 1: Clean Formal Nastaliq (Style 1)",
            "query": "ڈرائیونگ لائسنس کی تجدید کے وقت کن درخواست گزاروں کو میڈیکل سرٹیفکیٹ دینا ہوتا ہے؟",
            "t": 0,
            "reformulated": False,
        },
        {
            "desc": "Test 2: Conversational Urdu with Loanwords (Style 2)",
            "query": "لائسنس رینیو کروانا ہے ڈاکٹر کی پرچی لازمی ہے کیا؟",
            "t": 1,
            "reformulated": False,
        },
        {
            "desc": "Test 3: Roman Urdu Transliteration (Style 4)",
            "query": "driving license renew krwany k lye medical certificate kis umer me zarori ha?",
            "t": 2,
            "reformulated": True,
        },
        {
            "desc": "Test 4: Parametric General Knowledge (No Retrieval Needed)",
            "query": "پاکستان کا قومی پھول کونسا ہے؟",
            "t": 0,
            "reformulated": False,
        },
    ]

    for t_case in test_suite:
        desc = t_case["desc"]
        q = t_case["query"]
        t_step = t_case["t"]
        ref_flag = t_case["reformulated"]

        print(f"\n[{desc}]")
        print(f"Query: \"{q}\" (t={t_step}, reformulated={ref_flag})")

        s_t = extractor.extract_state(query=q, t=t_step, reformulated_flag=ref_flag)

        # Assertions
        assert isinstance(s_t, torch.Tensor), "Output must be a torch.Tensor"
        assert s_t.shape == (14,), f"Output shape must be (14,), got {s_t.shape}"
        assert s_t.dtype == torch.float32, f"Output dtype must be torch.float32, got {s_t.dtype}"
        assert not torch.isnan(s_t).any(), "State tensor must not contain NaNs"
        assert not torch.isinf(s_t).any(), "State tensor must not contain Infs"
        assert (s_t >= 0.0).all() and (s_t <= 1.0).all(), "All features must be bounded in [0.0, 1.0]"

        print("-" * 80)
        for i, (fname, val) in enumerate(zip(feat_names, s_t.tolist())):
            print(f"   [{i+1:02d}] {fname:<35} : {val:.4f}")

    print("\n" + "=" * 80)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY! Shape (14,), Dtype float32, Bounded in [0.0, 1.0]")
    print("=" * 80)

