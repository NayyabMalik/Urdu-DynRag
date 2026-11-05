#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Roman Urdu Spelling Variant & Orthographic Robustness Stress Test
Evaluates the robustness of DeterministicUrduReformulator and DualFAISSRetriever
against non-standard, informal, and noisy Roman Urdu spellings across all 85 S4 queries.

Perturbation Regimes Tested:
  1. Clean Roman Urdu (Baseline S4)
  2. Regime A: Vowel Alternations (ee <-> i, oo <-> u, aa <-> a, ai <-> ay)
  3. Regime B: Consonant Digraph Shifts (kh -> k, q -> k, ph -> f, gh -> g, c -> k)
  4. Regime C: Casual SMS Vowel Dropping (shnakhti, drkhwast, krna, hga)
  5. Regime D: Compound Typo & Orthographic Noise (Combined real-world informal typing)

Outputs:
  - experiments/roman_urdu_robustness_records.csv
  - experiments/roman_urdu_robustness_table.csv
  - experiments/roman_urdu_robustness_table.md
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from phase1_indexer import DualFAISSRetriever
from src.reformulator import DeterministicUrduReformulator, compute_urdu_script_ratio


# ==============================================================================
# 1. Systematic Orthographic Perturbation Generators
# ==============================================================================

def perturb_vowel_shifts(text: str) -> str:
    """
    Regime A: Systematic vowel alternations common in informal Roman Urdu.
    Examples:
      tareeqa -> tariqa, fees -> fis, zaroori -> zaruri, challan -> chalan
    """
    res = text
    # ee -> i (e.g. fees -> fis, tareeq -> tariq)
    res = re.sub(r"ee", "i", res, flags=re.IGNORECASE)
    # oo -> u (e.g. zaroori -> zaruri, kanoon -> kanun)
    res = re.sub(r"oo", "u", res, flags=re.IGNORECASE)
    # aa -> a (e.g. shanakhti -> shanakhti, challaan -> chalan)
    res = re.sub(r"aa", "a", res, flags=re.IGNORECASE)
    # ai -> ay or e (e.g. kaise -> kayse / kese)
    res = re.sub(r"ai", "ay", res, flags=re.IGNORECASE)
    return res


def perturb_consonant_digraphs(text: str) -> str:
    """
    Regime B: Consonant digraph simplifications and phonetic substitutions.
    Examples:
      shanakhti -> shanakti, tareeqa -> tareeka, card -> kard, khilaf -> kilaf
    """
    res = text
    # kh -> k
    res = re.sub(r"\bkh", "k", res, flags=re.IGNORECASE)
    res = re.sub(r"kh", "k", res, flags=re.IGNORECASE)
    # q -> k
    res = re.sub(r"q", "k", res, flags=re.IGNORECASE)
    # gh -> g
    res = re.sub(r"gh", "g", res, flags=re.IGNORECASE)
    # ph -> f
    res = re.sub(r"ph", "f", res, flags=re.IGNORECASE)
    # c before a/o/u or end of word -> k
    res = re.sub(r"\bc([aou])", r"k\1", res, flags=re.IGNORECASE)
    res = re.sub(r"card\b", "kard", res, flags=re.IGNORECASE)
    return res


def perturb_sms_vowel_dropping(text: str) -> str:
    """
    Regime C: Casual SMS typing / vowel omission in non-initial syllables.
    Examples:
      shanakhti -> shnakhti, darkhwast -> drkhwast, zarori -> zrori, karna -> krna
    """
    words = text.split()
    new_words = []
    sms_drop_map = {
        "shanakhti": "shnakhti",
        "shinakhti": "shnakhti",
        "darkhwast": "drkhwast",
        "darkhast": "drkhast",
        "zarori": "zrori",
        "zaroori": "zrori",
        "karna": "krna",
        "karne": "krne",
        "karein": "kren",
        "hoga": "hga",
        "hogi": "hgi",
        "honge": "hnge",
        "chahiye": "chahye",
        "tareeqa": "triqa",
        "police": "pols",
        "challan": "chlan",
        "dafa": "dfa",
        "jurmana": "jrmana",
        "saza": "sza",
        "kya": "kia",
        "hai": "hy",
        "hain": "hn",
    }
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w.lower())
        punct = w[len(clean_w):] if len(clean_w) < len(w) else ""
        if clean_w in sms_drop_map:
            new_words.append(sms_drop_map[clean_w] + punct)
        elif len(clean_w) > 5 and clean_w.count("a") > 1:
            # Drop the first medial unstressed 'a'
            parts = clean_w.split("a", 1)
            dropped = parts[0] + parts[1]
            new_words.append(dropped + punct)
        else:
            new_words.append(w)
    return " ".join(new_words)


def perturb_compound_noise(text: str) -> str:
    """
    Regime D: Combined real-world messy informal typing combining
    vowel shifts, consonant digraph simplifications, and SMS contractions.
    """
    t1 = perturb_sms_vowel_dropping(text)
    t2 = perturb_vowel_shifts(t1)
    t3 = perturb_consonant_digraphs(t2)
    return t3


# ==============================================================================
# 2. Main Stress Test Execution Engine
# ==============================================================================

def run_roman_urdu_stress_test(
    dataset_path: str = "data/urdu_dynrag_verified_dataset_v3.json",
    indices_dir: str = "models/indices",
    output_dir: str = "experiments",
) -> None:
    print("=" * 85)
    print("URDU-DYNRAG: ROMAN URDU ORTHOGRAPHIC ROBUSTNESS STRESS TEST")
    print("=" * 85)

    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter all 85 Roman Urdu (S4) queries
    s4_items = [d for d in dataset if d.get("query_style") == "style_4_roman_urdu"]
    print(f"[StressTest] Found {len(s4_items)} Roman Urdu (S4) benchmark queries.")

    retriever = DualFAISSRetriever(indices_dir=indices_dir)
    reformulator = DeterministicUrduReformulator(generator_backend="openrouter")

    regimes: List[Tuple[str, str, Callable[[str], str]]] = [
        ("Clean Roman Urdu (Baseline S4)", "Original benchmark queries without perturbation", lambda q: q),
        ("Regime A: Vowel Shift Noise", "Alternations (ee->i, oo->u, aa->a, ai->ay)", perturb_vowel_shifts),
        ("Regime B: Consonant Digraph Shifts", "Digraph shifts (kh->k, q->k, ph->f, gh->g, c->k)", perturb_consonant_digraphs),
        ("Regime C: SMS Vowel Dropping", "Casual SMS vowel contractions (shnakhti, drkhwast, krna)", perturb_sms_vowel_dropping),
        ("Regime D: Compound Informal Noise", "Combined vowel shifts + consonant shifts + SMS contractions", perturb_compound_noise),
    ]

    all_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for regime_name, regime_desc, perturb_fn in regimes:
        print(f"\n--- Testing: {regime_name} ---")
        t0 = time.time()

        vanilla_hits = 0
        dynrag_hits = 0
        script_ratios: List[float] = []

        sample_original = ""
        sample_perturbed = ""
        sample_reformulated = ""

        for idx, item in enumerate(s4_items, 1):
            qid = item["id"]
            orig_q = item["query"]
            target_pid = item.get("passage_id", "")

            # Apply perturbation
            perturbed_q = perturb_fn(orig_q)

            # 1. Vanilla Chunk RAG (search raw without reformulation)
            raw_res = retriever.search_raw(perturbed_q, k=1)
            raw_pid = raw_res[0]["passage_id"] if raw_res else ""
            vanilla_hit = bool(target_pid) and (raw_pid == target_pid)
            if vanilla_hit:
                vanilla_hits += 1

            # 2. Urdu-DynRAG Reformulation
            ref_q = reformulator.reformulate(perturbed_q)
            script_ratio = compute_urdu_script_ratio(ref_q)
            script_ratios.append(script_ratio)

            # Retrieve with reformulated query
            ref_raw_res = retriever.search_raw(ref_q, k=1)
            ref_pid = ref_raw_res[0]["passage_id"] if ref_raw_res else ""
            dynrag_hit = bool(target_pid) and (ref_pid == target_pid)
            if dynrag_hit:
                dynrag_hits += 1

            if idx == 1:
                sample_original = orig_q
                sample_perturbed = perturbed_q
                sample_reformulated = ref_q

            all_records.append({
                "id": qid,
                "regime": regime_name,
                "target_passage_id": target_pid,
                "original_query": orig_q,
                "perturbed_query": perturbed_q,
                "reformulated_query": ref_q,
                "script_ratio": float(script_ratio),
                "vanilla_hit1": bool(vanilla_hit),
                "dynrag_hit1": bool(dynrag_hit),
            })

        n_total = len(s4_items)
        vanilla_hit1_pct = (vanilla_hits / n_total) * 100.0
        dynrag_hit1_pct = (dynrag_hits / n_total) * 100.0
        delta_pct = dynrag_hit1_pct - vanilla_hit1_pct
        mean_script_ratio = float(sum(script_ratios) / len(script_ratios))
        elapsed = time.time() - t0

        print(f"  Vanilla Chunk RAG Hit@1:     {vanilla_hit1_pct:.1f}% ({vanilla_hits}/{n_total})")
        print(f"  Urdu-DynRAG (Ours) Hit@1:    {dynrag_hit1_pct:.1f}% ({dynrag_hits}/{n_total})")
        print(f"  Advantage (Delta):           {delta_pct:+.1f}%")
        print(f"  Mean Nastaliq Script Ratio:  {mean_script_ratio:.3f}")
        print(f"  Elapsed Time:                {elapsed:.1f}s")

        summary_rows.append({
            "Perturbation Regime": regime_name,
            "Description": regime_desc,
            "Vanilla RAG Hit@1": f"{vanilla_hit1_pct:.1f}% ({vanilla_hits}/{n_total})",
            "Urdu-DynRAG Hit@1": f"{dynrag_hit1_pct:.1f}% ({dynrag_hits}/{n_total})",
            "Delta Gain": f"{delta_pct:+.1f}%",
            "Nastaliq Ratio": f"{mean_script_ratio*100.0:.1f}%",
            "Sample Perturbed Query": sample_perturbed,
            "Sample Reformulated Query": sample_reformulated,
        })

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame(all_records)
    records_csv = out_dir / "roman_urdu_robustness_records.csv"
    records_df.to_csv(records_csv, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "roman_urdu_robustness_table.csv"
    summary_md = out_dir / "roman_urdu_robustness_table.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_df.to_markdown(summary_md, index=False)

    print("\n" + "=" * 85)
    print("TABLE 5: ROMAN URDU ORTHOGRAPHIC ROBUSTNESS STRESS TEST (N=85 queries per regime)")
    print("=" * 85)
    print(summary_df[["Perturbation Regime", "Vanilla RAG Hit@1", "Urdu-DynRAG Hit@1", "Delta Gain", "Nastaliq Ratio"]].to_markdown(index=False))
    print(f"\n[StressTest] Complete! Artifacts saved in {out_dir}/ directory.")


if __name__ == "__main__":
    run_roman_urdu_stress_test()

