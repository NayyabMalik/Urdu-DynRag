#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Dynamic RL-Guided Adaptive Retrieval for Under-Resourced QA
Phase 1: Dual-Representation FAISS Vector Indexing Pipeline
Reference: DR-RAG (Ahmad, Hammad et al., CHiPSAL 2026)

This module builds two separate FAISS inner-product (cosine similarity) vector indices
using dense multilingual embeddings (default: BAAI/bge-m3, 1024-dim):
  1. raw_chunks.index (D_raw): Indexes all raw domain passages directly.
  2. dr_qa.index (D_qa): Indexes independent synthetic Urdu questions (2 per passage)
     mapped back to parent_passage_id.

CRITICAL ANTI-LEAKAGE RULE:
No queries from urdu_dynrag_verified_dataset_v3.json are indexed! Those are strictly
held-out for evaluation and RL agent training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import faiss
import numpy as np
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# SentenceTransformers lazy loader
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


# ==============================================================================
# 1. Corpus Loader & Validator
# ==============================================================================

def resolve_file_path(candidate_paths: List[str]) -> Path:
    """Find the first existing path among candidate locations."""
    for p in candidate_paths:
        path_obj = Path(p)
        if path_obj.exists():
            return path_obj
    raise FileNotFoundError(
        f"Could not locate file. Checked candidate paths: {candidate_paths}"
    )


def load_corpus(corpus_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and validate domain passages from urdu_dynrag_verified_corpus_v3.json.
    Schema expected per item:
      - passage_id: str
      - domain: str
      - text: str
      - verification_status: str (optional)
      - source_url: Optional[str]
      - time_sensitive: bool (optional)
    """
    candidates = [
        corpus_path,
        "data/urdu_dynrag_verified_corpus_v3.json",
        "urdu_dynrag_verified_corpus_v3.json",
        "../data/urdu_dynrag_verified_corpus_v3.json",
    ]
    valid_candidates = [str(c) for c in candidates if c is not None]
    target_path = resolve_file_path(valid_candidates)

    with open(target_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list) or len(corpus) == 0:
        raise ValueError(f"Corpus file at {target_path} is empty or not a JSON list.")

    validated: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, item in enumerate(corpus):
        if not isinstance(item, dict):
            raise ValueError(f"Corpus item at index {idx} is not a dictionary.")
        if "passage_id" not in item or "text" not in item:
            raise ValueError(
                f"Corpus item at index {idx} missing required 'passage_id' or 'text' key."
            )
        pid = item["passage_id"]
        if pid in seen_ids:
            continue  # Keep unique passages
        seen_ids.add(pid)
        validated.append({
            "passage_id": pid,
            "domain": item.get("domain", "General_Civic"),
            "text": item["text"].strip(),
            "verification_status": item.get("verification_status", "verified"),
            "source_url": item.get("source_url"),
            "time_sensitive": item.get("time_sensitive", False),
        })

    domains = sorted(list(set(p["domain"] for p in validated)))
    print(f"[CorpusLoader] Successfully loaded {len(validated)} unique passages across {len(domains)} domains.")
    return validated


# ==============================================================================
# 2. Synthetic Question Generator (D_qa)
# ==============================================================================

def _generate_synthetic_qa_rule_based(passage: Dict[str, Any]) -> List[str]:
    """
    High-precision, grammatical Urdu question generator that runs 100% offline.
    Extracts clauses, numerical constraints, fees, and legal conditions from the
    passage text without requiring an API key.
    """
    text = passage["text"]
    domain = passage["domain"]

    # Normalize clean sentences
    sentences = [s.strip() for s in re.split(r"[۔\n\.\?؟!]", text) if len(s.strip()) > 15]

    q1 = ""
    q2 = ""

    # Strategy 1: Check for numerical conditions (age, fees, days, penalty)
    num_matches = re.findall(r"(\d+[\s\w]*روپے|\d+\s*سال|\d+\s*دن|\d+\s*ہفتے|فارم\s*[\w\u0600-\u06FF]+)", text)
    if num_matches and len(sentences) >= 2:
        subject = sentences[0][:60]
        # Clean leading connectors
        subject = re.sub(r"^(تحت|مطابق|قانون|آرڈیننس|قواعد)\s*", "", subject)
        q1 = f"{subject} کے تحت کیا شرائط اور مدت مقرر ہے؟"
        q2 = f"{sentences[1][:60]} کے بارے میں کیا قانونی حکم اور تقاضے ہیں؟"
    elif len(sentences) >= 2:
        # Strategy 2: Two distinct clause questions
        lead_clause_1 = sentences[0].replace("ہے کہ", "").strip()
        lead_clause_2 = sentences[1].replace("ہے کہ", "").strip()
        q1 = f"{lead_clause_1[:65]} کے متعلق کیا ہدایات دی گئی ہیں؟"
        q2 = f"{lead_clause_2[:65]} کے ضوابط اور طریقہ کار کیا ہیں؟"
    else:
        # Fallback based on domain
        clean_domain = domain.replace("_", " ")
        q1 = f"{clean_domain} کے متعلق سرکاری قواعد اور اہم شرائط کیا ہیں؟"
        q2 = f"{clean_domain} کے تحت درخواست دینے یا عمل درآمد کا کیا طریقہ کار ہے؟"

    # Ensure questions end with Urdu question mark
    if not q1.endswith("؟"):
        q1 = q1.rstrip(".!؟ ") + "؟"
    if not q2.endswith("؟"):
        q2 = q2.rstrip(".!؟ ") + "؟"

    return [q1, q2]


def _generate_synthetic_qa_llm(passages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Generate synthetic questions using Gemini API or OpenRouter if keys are present.
    Returns None if no API client succeeds, falling back to rule-based generation.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not gemini_key and not openrouter_key:
        return None

    results: List[Dict[str, Any]] = []

    # Attempt Google GenAI Gemini
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            print("[QAGenerator] Using Gemini API (gemini-2.5-flash) to generate synthetic QA pairs...")

            for i, p in enumerate(passages):
                prompt = (
                    f"You are creating dual-representation search index questions for Urdu retrieval (DR-RAG).\n"
                    f"Read this Urdu text carefully:\n\"\"\"{p['text']}\"\"\"\n\n"
                    f"Generate exactly TWO distinct, formal, grammatical Urdu search questions that can be directly answered by this text.\n"
                    f"Output MUST be strict JSON array of strings: [\"سوال 1؟\", \"سوال 2؟\"]"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                clean_txt = response.text.strip()
                if "```json" in clean_txt:
                    clean_txt = clean_txt.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_txt:
                    clean_txt = clean_txt.split("```")[1].split("```")[0].strip()
                questions = json.loads(clean_txt)
                if isinstance(questions, list) and len(questions) >= 2:
                    results.append({
                        "qa_id": f"dr_qa_{i+1:03d}_1",
                        "parent_passage_id": p["passage_id"],
                        "domain": p["domain"],
                        "synthetic_question": questions[0].strip(),
                        "parent_text": p["text"],
                    })
                    results.append({
                        "qa_id": f"dr_qa_{i+1:03d}_2",
                        "parent_passage_id": p["passage_id"],
                        "domain": p["domain"],
                        "synthetic_question": questions[1].strip(),
                        "parent_text": p["text"],
                    })
            if len(results) == len(passages) * 2:
                print(f"[QAGenerator] Successfully generated {len(results)} questions via Gemini API.")
                return results
        except Exception as e:
            print(f"[QAGenerator] Gemini API generation encountered error ({e}). Checking fallbacks...")

    # Attempt OpenRouter fallback
    if openrouter_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            print("[QAGenerator] Using OpenRouter (qwen/qwen-2.5-7b-instruct) for synthetic QA generation...")
            for i, p in enumerate(passages):
                prompt = (
                    f"Urdu Passage:\n{p['text']}\n\n"
                    f"Write exactly 2 formal Urdu search questions directly answerable by this text.\n"
                    f"Output format JSON array only: [\"سوال 1؟\", \"سوال 2؟\"]"
                )
                completion = client.chat.completions.create(
                    model="qwen/qwen-2.5-7b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                raw_out = completion.choices[0].message.content.strip()
                match = re.search(r"\[.*?\]", raw_out, re.DOTALL)
                if match:
                    qs = json.loads(match.group(0))
                    if len(qs) >= 2:
                        results.append({
                            "qa_id": f"dr_qa_{i+1:03d}_1",
                            "parent_passage_id": p["passage_id"],
                            "domain": p["domain"],
                            "synthetic_question": qs[0].strip(),
                            "parent_text": p["text"],
                        })
                        results.append({
                            "qa_id": f"dr_qa_{i+1:03d}_2",
                            "parent_passage_id": p["passage_id"],
                            "domain": p["domain"],
                            "synthetic_question": qs[1].strip(),
                            "parent_text": p["text"],
                        })
            if len(results) == len(passages) * 2:
                print(f"[QAGenerator] Successfully generated {len(results)} questions via OpenRouter.")
                return results
        except Exception as e:
            print(f"[QAGenerator] OpenRouter generation encountered error ({e}). Proceeding to rule-based generator.")

    return None


def generate_or_load_synthetic_qa(
    corpus: List[Dict[str, Any]],
    qa_output_path: Path,
    force_regen: bool = False,
) -> List[Dict[str, Any]]:
    """
    Checks if synthetic QA file exists; if not, generates 2 independent Nastaliq
    questions per passage and saves to disk.
    """
    if qa_output_path.exists() and not force_regen:
        print(f"[QAGenerator] Found existing synthetic QA index at {qa_output_path}. Loading cache...")
        with open(qa_output_path, "r", encoding="utf-8") as f:
            cached_qa = json.load(f)
        if isinstance(cached_qa, list) and len(cached_qa) >= len(corpus):
            print(f"[QAGenerator] Loaded {len(cached_qa)} cached synthetic QA records.")
            return cached_qa

    print(f"[QAGenerator] Generating 2 independent synthetic questions per passage ({len(corpus)} passages)...")

    # Try LLM first
    qa_records = _generate_synthetic_qa_llm(corpus)

    # Robust algorithmic fallback
    if qa_records is None:
        print("[QAGenerator] Using high-precision grammatical Urdu template generator (offline mode)...")
        qa_records = []
        for i, p in enumerate(corpus):
            q1, q2 = _generate_synthetic_qa_rule_based(p)
            qa_records.append({
                "qa_id": f"dr_qa_{i+1:03d}_1",
                "parent_passage_id": p["passage_id"],
                "domain": p["domain"],
                "synthetic_question": q1,
                "parent_text": p["text"],
            })
            qa_records.append({
                "qa_id": f"dr_qa_{i+1:03d}_2",
                "parent_passage_id": p["passage_id"],
                "domain": p["domain"],
                "synthetic_question": q2,
                "parent_text": p["text"],
            })

    qa_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_output_path, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, ensure_ascii=False, indent=2)

    print(f"[QAGenerator] Saved {len(qa_records)} synthetic QA records to {qa_output_path}")
    return qa_records


# ==============================================================================
# 3. Dense Embedding & FAISS Indexing Engine
# ==============================================================================

class DualFAISSRetriever:
    """
    Dual-Representation FAISS Retriever interface for Urdu-DynRAG.
    Loads and searches two inner-product indices:
      - raw_chunks.index (Passage texts)
      - dr_qa.index (Synthetic Question representations)
    """

    def __init__(
        self,
        indices_dir: str = "models/indices",
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ) -> None:
        self.indices_dir = Path(indices_dir)
        self.model_name = model_name
        self.device = device

        self.raw_index_path = self.indices_dir / "raw_chunks.index"
        self.raw_meta_path = self.indices_dir / "raw_metadata.json"
        self.qa_index_path = self.indices_dir / "dr_qa.index"
        self.qa_meta_path = self.indices_dir / "dr_metadata.json"

        self.raw_index: Optional[faiss.Index] = None
        self.raw_metadata: List[Dict[str, Any]] = []
        self.qa_index: Optional[faiss.Index] = None
        self.qa_metadata: List[Dict[str, Any]] = []
        self._encoder: Optional[Any] = None

        self._load_indices_and_metadata()

    @property
    def encoder(self) -> Any:
        """Lazy load SentenceTransformer encoder."""
        if self._encoder is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. Install via `pip install sentence-transformers`."
                )
            print(f"[Retriever] Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def _load_indices_and_metadata(self) -> None:
        """Load pre-built indices and metadata from disk if available."""
        if self.raw_index_path.exists() and self.raw_meta_path.exists():
            self.raw_index = faiss.read_index(str(self.raw_index_path))
            with open(self.raw_meta_path, "r", encoding="utf-8") as f:
                self.raw_metadata = json.load(f)
            print(f"[Retriever] Loaded raw_chunks.index ({self.raw_index.ntotal} vectors).")

        if self.qa_index_path.exists() and self.qa_meta_path.exists():
            self.qa_index = faiss.read_index(str(self.qa_index_path))
            with open(self.qa_meta_path, "r", encoding="utf-8") as f:
                self.qa_metadata = json.load(f)
            print(f"[Retriever] Loaded dr_qa.index ({self.qa_index.ntotal} vectors).")

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries into L2-normalized float32 numpy array."""
        embeddings = self.encoder.encode(
            queries,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def search_raw(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search raw chunks index (D_raw) for top-k matches."""
        if self.raw_index is None:
            raise RuntimeError("raw_chunks.index is not loaded.")
        q_vec = self.encode_queries([query])
        scores, indices = self.raw_index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.raw_metadata):
                continue
            meta = self.raw_metadata[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "passage_id": meta["passage_id"],
                "domain": meta["domain"],
                "text": meta["text"],
                "source_url": meta.get("source_url"),
            })
        return results

    def search_qa(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search synthetic QA index (D_qa) for top-k matches."""
        if self.qa_index is None:
            raise RuntimeError("dr_qa.index is not loaded.")
        q_vec = self.encode_queries([query])
        scores, indices = self.qa_index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.qa_metadata):
                continue
            meta = self.qa_metadata[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "qa_id": meta["qa_id"],
                "parent_passage_id": meta["parent_passage_id"],
                "domain": meta["domain"],
                "synthetic_question": meta["synthetic_question"],
                "parent_text": meta["parent_text"],
            })
        return results

    def search_both(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Execute dual-representation search and compute similarity margins:
        Delta_margin = Top1_score - Top3_score
        """
        raw_res = self.search_raw(query, k=k)
        qa_res = self.search_qa(query, k=k)

        raw_top1 = raw_res[0]["score"] if raw_res else 0.0
        raw_top3 = raw_res[min(2, len(raw_res) - 1)]["score"] if raw_res else 0.0
        margin_raw = raw_top1 - raw_top3

        qa_top1 = qa_res[0]["score"] if qa_res else 0.0
        qa_top3 = qa_res[min(2, len(qa_res) - 1)]["score"] if qa_res else 0.0
        margin_qa = qa_top1 - qa_top3

        return {
            "query": query,
            "raw_results": raw_res,
            "qa_results": qa_res,
            "raw_top1": raw_top1,
            "raw_margin": margin_raw,
            "qa_top1": qa_top1,
            "qa_margin": margin_qa,
            "qa_minus_raw_top1": qa_top1 - raw_top1,
        }


# ==============================================================================
# 4. Index Builder Execution Function
# ==============================================================================

def build_indices(
    corpus_path: Optional[str] = None,
    output_dir: str = "models/indices",
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    batch_size: int = 16,
    force_regen_qa: bool = False,
) -> Tuple[Path, Path]:
    """
    Orchestrate full indexing pipeline:
      1. Load corpus passages.
      2. Generate/load synthetic QA pairs.
      3. Encode passages -> raw_chunks.index.
      4. Encode questions -> dr_qa.index.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corpus
    corpus = load_corpus(corpus_path)

    # 2. Synthetic QA
    qa_path = out_dir / "dr_synthetic_qa_150.json"
    qa_records = generate_or_load_synthetic_qa(corpus, qa_path, force_regen=force_regen_qa)

    # 3. Embedding model
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Install via `pip install sentence-transformers`.")
    print(f"[IndexBuilder] Initializing embedding model '{model_name}' on device '{device}'...")
    encoder = SentenceTransformer(model_name, device=device)

    # --- Build Raw Chunks Index ---
    print(f"[IndexBuilder] Encoding {len(corpus)} raw passage chunks...")
    passage_texts = [p["text"] for p in corpus]
    raw_embeddings = encoder.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    raw_matrix = np.asarray(raw_embeddings, dtype=np.float32)
    dim = raw_matrix.shape[1]

    raw_index = faiss.IndexFlatIP(dim)
    raw_index.add(raw_matrix)

    raw_index_path = out_dir / "raw_chunks.index"
    faiss.write_index(raw_index, str(raw_index_path))

    raw_meta_path = out_dir / "raw_metadata.json"
    with open(raw_meta_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"[IndexBuilder] Successfully wrote {raw_index_path} ({raw_index.ntotal} vectors, dim={dim}).")

    # --- Build DR QA Index ---
    print(f"[IndexBuilder] Encoding {len(qa_records)} synthetic questions...")
    qa_texts = [q["synthetic_question"] for q in qa_records]
    qa_embeddings = encoder.encode(
        qa_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    qa_matrix = np.asarray(qa_embeddings, dtype=np.float32)

    qa_index = faiss.IndexFlatIP(dim)
    qa_index.add(qa_matrix)

    qa_index_path = out_dir / "dr_qa.index"
    faiss.write_index(qa_index, str(qa_index_path))

    qa_meta_path = out_dir / "dr_metadata.json"
    with open(qa_meta_path, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, ensure_ascii=False, indent=2)

    print(f"[IndexBuilder] Successfully wrote {qa_index_path} ({qa_index.ntotal} vectors, dim={dim}).")
    return raw_index_path, qa_index_path


# ==============================================================================
# 5. Verification Smoke Test (--verify)
# ==============================================================================

def run_verification_smoke_test(indices_dir: str, model_name: str, device: str = "cpu") -> None:
    """
    Run smoke test on 3 representative queries across stylistic variations:
      1. Clean Nastaliq
      2. Conversational Urdu
      3. Roman Urdu
    """
    print("\n" + "=" * 80)
    print("URDU-DYNRAG: DUAL-INDEX VERIFICATION & SMOKE TEST")
    print("=" * 80)

    retriever = DualFAISSRetriever(indices_dir=indices_dir, model_name=model_name, device=device)

    test_queries = [
        {
            "type": "Clean Nastaliq (Style 1)",
            "query": "ڈرائیونگ لائسنس کی تجدید کے وقت کن افراد کو میڈیکل سرٹیفکیٹ دینا لازمی ہے؟",
            "desc": "Formal legal query regarding medical fitness at 50+ years.",
        },
        {
            "type": "Conversational Urdu (Style 2)",
            "query": "لائسنس رینیو کروانا ہے کیا ڈاکٹر سے میڈیکل فارم بھی بنوانا پڑے گا؟",
            "desc": "Conversational query using loanwords ('رینیو', 'ڈاکٹر').",
        },
        {
            "type": "Roman Urdu (Style 4 - Code Switched)",
            "query": "driving license renew krwany k lye medical certificate kis umer me zarori ha?",
            "desc": "Latin script transliterated query testing cross-script matching.",
        },
    ]

    for t_case in test_queries:
        q_type = t_case["type"]
        q_text = t_case["query"]
        q_desc = t_case["desc"]

        print(f"\n[Test Case] {q_type}")
        print(f"Description : {q_desc}")
        print(f"Query       : \"{q_text}\"")
        print("-" * 80)

        res = retriever.search_both(q_text, k=3)

        print(f"D_raw (Raw Chunks)  | Top-1: {res['raw_top1']:.4f} | Margin (T1 - T3): {res['raw_margin']:.4f}")
        for r in res["raw_results"]:
            snippet = r["text"][:70].replace("\n", " ")
            print(f"   Rank {r['rank']} | Score: {r['score']:.4f} | ID: {r['passage_id']} | Text: {snippet}...")

        print(f"\nD_qa (Synthetic QA) | Top-1: {res['qa_top1']:.4f} | Margin (T1 - T3): {res['qa_margin']:.4f}")
        for r in res["qa_results"]:
            q_match = r["synthetic_question"][:70]
            print(f"   Rank {r['rank']} | Score: {r['score']:.4f} | Parent: {r['parent_passage_id']} | Q: {q_match}...")

        diff = res["qa_minus_raw_top1"]
        winner = "D_qa (Synthetic QA Index)" if diff > 0 else "D_raw (Raw Chunks Index)"
        print(f"\nResult: {winner} scored higher (Δ = {diff:+.4f})")
        print("=" * 80)


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Urdu-DynRAG Phase 1: Dual-Representation FAISS Vector Indexing Pipeline"
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Path to urdu_dynrag_verified_corpus_v3.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/indices",
        help="Directory to save FAISS indices and metadata",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformers model name (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run embeddings on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--force-regen-qa",
        action="store_true",
        help="Force regeneration of synthetic QA questions even if cached file exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run smoke test verification across sample queries after indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[Phase 1 Indexer] Starting Dual-Representation FAISS Pipeline...")
    raw_path, qa_path = build_indices(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        force_regen_qa=args.force_regen_qa,
    )

    if args.verify:
        run_verification_smoke_test(
            indices_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
        )

    print("\n[Phase 1 Indexer] Completed successfully!")


if __name__ == "__main__":
    main()

