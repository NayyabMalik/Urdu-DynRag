#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Verification Audit: Phase 1 & Phase 2
Validates file integrity, index dimensions, retriever functionality,
and 14-dimensional state tensor constraints.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import faiss
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("URDU-DYNRAG: END-TO-END AUDIT FOR PHASE 1 & PHASE 2")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. File Existence & Size Checks
# ------------------------------------------------------------------------------
indices_dir = Path("models/indices")
required_files = [
    "raw_chunks.index",
    "raw_metadata.json",
    "dr_qa.index",
    "dr_metadata.json",
    "dr_synthetic_qa_150.json",
]

print("\n[Audit 1: File Existence & Storage Checks]")
all_files_exist = True
for fname in required_files:
    fpath = indices_dir / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024.0
        print(f"  ✓ {fname:<30} EXISTS ({size_kb:8.1f} KB)")
    else:
        print(f"  ✗ {fname:<30} MISSING!")
        all_files_exist = False

assert all_files_exist, "Critical failure: required index files are missing!"

# ------------------------------------------------------------------------------
# 2. FAISS Index Dimensionality & Metadata Alignment
# ------------------------------------------------------------------------------
print("\n[Audit 2: Index Matrix Integrity & Vector Alignment]")
raw_idx = faiss.read_index(str(indices_dir / "raw_chunks.index"))
qa_idx = faiss.read_index(str(indices_dir / "dr_qa.index"))

with open(indices_dir / "raw_metadata.json", "r", encoding="utf-8") as f:
    raw_meta = json.load(f)
with open(indices_dir / "dr_metadata.json", "r", encoding="utf-8") as f:
    qa_meta = json.load(f)

print(f"  ✓ Raw Index: {raw_idx.ntotal} vectors | Dimension: {raw_idx.d} | Metadata count: {len(raw_meta)}")
print(f"  ✓ QA Index : {qa_idx.ntotal} vectors | Dimension: {qa_idx.d} | Metadata count: {len(qa_meta)}")

assert raw_idx.ntotal == len(raw_meta), f"Count mismatch: {raw_idx.ntotal} vs {len(raw_meta)}"
assert qa_idx.ntotal == len(qa_meta), f"Count mismatch: {qa_idx.ntotal} vs {len(qa_meta)}"
assert raw_idx.d == 1024, f"Raw index dimension must be 1024, got {raw_idx.d}"
assert qa_idx.d == 1024, f"QA index dimension must be 1024, got {qa_idx.d}"

# ------------------------------------------------------------------------------
# 3. DualFAISSRetriever Execution
# ------------------------------------------------------------------------------
print("\n[Audit 3: DualFAISSRetriever Live Search Test]")
from phase1_indexer import DualFAISSRetriever

retriever = DualFAISSRetriever(indices_dir=str(indices_dir))
test_query = "پاسپورٹ بنوانے کے لیے کیا دستاویزات درکار ہیں؟"
res = retriever.search_both(test_query, k=3)

print(f"  Query: \"{test_query}\"")
print(f"  ✓ D_raw Top-1 Score : {res['raw_top1']:.4f} (ID: {res['raw_results'][0]['passage_id']})")
print(f"  ✓ D_qa  Top-1 Score : {res['qa_top1']:.4f} (ID: {res['qa_results'][0]['parent_passage_id']})")
print(f"  ✓ Raw Margin (T1-T3): {res['raw_margin']:.4f}")
print(f"  ✓ QA Margin  (T1-T3): {res['qa_margin']:.4f}")

assert len(res["raw_results"]) == 3, "Expected 3 raw search results"
assert len(res["qa_results"]) == 3, "Expected 3 QA search results"
assert 0.0 <= res["raw_top1"] <= 1.0, "Raw score out of bounds"
assert 0.0 <= res["qa_top1"] <= 1.0, "QA score out of bounds"

# ------------------------------------------------------------------------------
# 4. State Extractor Invariant Tests
# ------------------------------------------------------------------------------
print("\n[Audit 4: 14-Dimensional State Extractor Invariant Tests]")
from phase2_state_extractor import UrduRAGStateExtractor

extractor = UrduRAGStateExtractor(retriever=retriever, generator_mode="auto")
feat_names = extractor.get_feature_names()
assert len(feat_names) == 14, f"Must have 14 features, found {len(feat_names)}"

audit_queries = [
    # 1. Formal Nastaliq
    ("شناختی کارڈ گم ہو جائے تو نادرا میں کیا طریقہ کار ہے؟", 0, False),
    # 2. Conversational loanwords
    ("nadra office ja k duplicate card k lye konsy papers lgengy?", 1, True),
    # 3. Roman Urdu
    ("passport renew krwana ha kitny din me mily ga?", 2, True),
    # 4. Parametric Factoid
    ("پاکستان کا دارالحکومت کیا ہے؟", 0, False),
]

for idx, (q, step, ref) in enumerate(audit_queries, 1):
    state = extractor.extract_state(query=q, t=step, reformulated_flag=ref)

    # Invariants
    assert isinstance(state, torch.Tensor), f"Query {idx}: must return torch.Tensor"
    assert state.shape == (14,), f"Query {idx}: expected shape (14,), got {state.shape}"
    assert state.dtype == torch.float32, f"Query {idx}: expected float32, got {state.dtype}"
    assert not torch.isnan(state).any(), f"Query {idx}: found NaN in state vector!"
    assert not torch.isinf(state).any(), f"Query {idx}: found Inf in state vector!"
    assert (state >= 0.0).all() and (state <= 1.0).all(), f"Query {idx}: values outside [0.0, 1.0]!"

    print(f"  ✓ Sample {idx} passed | Range: [{state.min():.4f}, {state.max():.4f}] | Query: \"{q[:35]}...\"")

print("\n" + "=" * 80)
print("AUDIT RESULT: ALL PHASE 1 & PHASE 2 MODULES ARE 100% OPERATIONAL & VERIFIED!")
print("=" * 80)
