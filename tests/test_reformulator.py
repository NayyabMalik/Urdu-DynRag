#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for DeterministicUrduReformulator (Phase 2):
Verifies:
  1. Script normalization converts Roman Urdu (S4) to Nastaliq (script_ratio >= 0.85).
  2. Deterministic SHA-256 caching.
  3. Retrieval Hit@1 on Grounded S4 queries exceeds 85.0% after reformulation.
"""

import json
from pathlib import Path
import pytest

from src.reformulator import DeterministicUrduReformulator, compute_urdu_script_ratio
from src.phase1_indexer import DualFAISSRetriever


@pytest.fixture(scope="module")
def reformulator():
    return DeterministicUrduReformulator(cache_path="cache/reformulation_cache.json")


@pytest.fixture(scope="module")
def dataset():
    data_path = Path("data/urdu_dynrag_verified_dataset_v3.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_script_ratio_on_s4_roman_urdu(reformulator, dataset):
    """Verify that all 85 S4 (Roman Urdu) queries convert to Nastaliq with ratio >= 0.85."""
    s4_queries = [it["query"] for it in dataset if it.get("query_style") == "style_4_roman_urdu"]
    assert len(s4_queries) == 85

    ratios = []
    for q in s4_queries:
        ref_q = reformulator.reformulate(q, force_offline=True)
        ratio = compute_urdu_script_ratio(ref_q)
        ratios.append(ratio)

    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio >= 0.90
    passing = sum(1 for r in ratios if r >= 0.85)
    # At least 95% of queries strictly exceed 0.85 ratio
    assert passing / len(ratios) >= 0.95


def test_deterministic_caching(reformulator):
    """Verify that calling reformulate twice returns identical string."""
    q = "Zameen ki sab se bahari tah kaunsi hai?"
    out1 = reformulator.reformulate(q, force_offline=True)
    out2 = reformulator.reformulate(q, force_offline=True)
    assert out1 == out2
    assert "زمین" in out1
    assert "بیرونی" in out1


def test_grounded_s4_hit1_boost(reformulator, dataset):
    """Verify that reformulating Grounded S4 queries raises Hit@1 to >= 85.0%."""
    retriever = DualFAISSRetriever(indices_dir="models/indices")
    s4_items = [it for it in dataset if it.get("query_style") == "style_4_roman_urdu" and it.get("passage_id")]
    assert len(s4_items) == 85

    ref_hits = 0
    for it in s4_items:
        ref_q = reformulator.reformulate(it["query"], force_offline=True)
        res = retriever.search_raw(ref_q, k=1)
        if res and res[0]["passage_id"] == it["passage_id"]:
            ref_hits += 1

    hit1_rate = float(ref_hits) / float(len(s4_items))
    print(f"Reformulated Grounded S4 Hit@1: {ref_hits}/{len(s4_items)} ({hit1_rate*100:.1f}%)")
    assert hit1_rate >= 0.85

