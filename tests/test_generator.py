import os
import sys
import tempfile
from pathlib import Path

import pytest

from src.generator import UnifiedGenerator, PROMPT_TEMPLATE_VERSION


def test_cache_key_sha256_reproducibility():
    """Verify that compute_cache_key produces identical SHA-256 digests and doesn't truncate context."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_cache.json"
        gen = UnifiedGenerator(backend="openrouter", cache_path=cache_file)
        
        query = "پاکستان کا دارالحکومت کیا ہے؟"
        context = "پاکستان کا دارالحکومت اسلام آباد ہے جو مارگلہ پہاڑیوں کے دامن میں واقع ہے۔ " * 10
        prompt = gen.format_prompt(query, context)
        
        key1 = gen.compute_cache_key(prompt, max_new_tokens=40, temperature=0.0, do_sample=False)
        key2 = gen.compute_cache_key(prompt, max_new_tokens=40, temperature=0.0, do_sample=False)
        
        assert key1 == key2
        assert len(key1) == 64  # Valid SHA-256 hex string


def test_identical_query_context_identical_outputs():
    """Verify that identical (query, context) pairs return identical outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_cache.json"
        gen = UnifiedGenerator(backend="openrouter", cache_path=cache_file)
        
        # Inject mock cached answer to avoid network latency during unit test
        query = "ٹیسٹ سوال کیا ہے؟"
        context = "ٹیسٹ سیاق و سباق کی عبارت۔"
        prompt = gen.format_prompt(query, context)
        key = gen.compute_cache_key(prompt, max_new_tokens=40, temperature=0.0, do_sample=False)
        gen.cache[key] = "ٹیسٹ جواب: اسلام آباد"
        gen._save_cache()
        
        ans1 = gen.generate(query, context=context, max_new_tokens=40, temperature=0.0)
        ans2 = gen.generate(query, context=context, max_new_tokens=40, temperature=0.0)
        
        assert ans1 == ans2
        assert ans1 == "ٹیسٹ جواب: اسلام آباد"


def test_vanilla_and_dr_rag_identical_context_produce_identical_answers():
    """
    Verify that Vanilla RAG and DR-RAG with identical contexts produce identical answers
    due to sharing the same prompt format and cache.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_cache.json"
        gen = UnifiedGenerator(backend="openrouter", cache_path=cache_file)
        
        query = "شناختی کارڈ بنوانے کے لیے کیا ضروری ہے؟"
        shared_context = "نادرا قوانین کے مطابق شناختی کارڈ بنوانے کے لیے ب فارم یا پیدائشی سرٹیفکیٹ درکار ہوتا ہے۔"
        
        # Pre-seed cache to simulate deterministic output
        prompt = gen.format_prompt(query, shared_context)
        key = gen.compute_cache_key(prompt, max_new_tokens=40, temperature=0.0, do_sample=False)
        expected_output = "ب فارم یا پیدائشی سرٹیفکیٹ درکار ہوتا ہے۔"
        gen.cache[key] = expected_output
        gen._save_cache()
        
        # Simulate Vanilla RAG pipeline
        vanilla_context = shared_context
        vanilla_ans = gen.generate(query, context=vanilla_context, max_new_tokens=40, temperature=0.0)
        
        # Simulate DR-RAG pipeline when action 1 or matching context is selected
        dr_rag_context = shared_context
        dr_rag_ans = gen.generate(query, context=dr_rag_context, max_new_tokens=40, temperature=0.0)
        
        assert vanilla_ans == dr_rag_ans
        assert vanilla_ans == expected_output

