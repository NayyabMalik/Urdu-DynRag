#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Unified Answer Generator Module (Phase 1b)
Supports:
  1. Local Backend: Qwen/Qwen2.5-3B-Instruct in 4-bit NF4 with greedy decoding (temperature=0, do_sample=False).
  2. OpenRouter Backend: qwen/qwen-2.5-7b-instruct via OpenRouter API with greedy decoding.
  3. Persistent SHA256 Caching: sha256(model_name, prompt_version, full_prompt, decoding_params).
     Context is never truncated in the cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

load_dotenv()


PROMPT_TEMPLATE_VERSION = "v1"

CONTEXT_PROMPT_TEMPLATE = (
    "آپ ایک مستند قانونی اور انتظامی معاون ہیں۔ درج ذیل سیاق و سباق (Context) کی بنیاد پر صارف کے سوال کا مختصر، دوٹوک اور درست جواب اردو میں دیں۔ اگر جواب سیاق و سباق میں نہیں ہے تو خود سے قیاس نہ کریں۔\n\n"
    "سیاق و سباق:\n\"\"\"{context}\"\"\"\n\n"
    "سوال: {query}\n"
    "جواب:"
)

DIRECT_PROMPT_TEMPLATE = (
    "آپ ایک مستند معاون ہیں۔ اپنے اندرونی علم کی روشنی میں اس سوال کا مختصر اور درست جواب اردو میں دیں:\n\n"
    "سوال: {query}\n"
    "جواب:"
)


class UnifiedGenerator:
    """
    Unified generator supporting local 4-bit NF4 and OpenRouter backends,
    with SHA-256 caching and greedy decoding.
    """

    def __init__(
        self,
        backend: str = "local",
        model_name: Optional[str] = None,
        cache_path: Union[str, Path] = "cache/benchmark_predictions.json",
        max_new_tokens: int = 40,
    ) -> None:
        self.backend = backend.lower()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.prompt_version = PROMPT_TEMPLATE_VERSION

        # Default model identifiers
        if self.backend == "local":
            self.model_name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        else:
            self.model_name = model_name or os.getenv("GENERATOR_MODEL", "qwen/qwen-2.5-7b-instruct")

        # Load cache
        self.cache: Dict[str, str] = {}
        self._load_cache()

        # Backend handles (lazy-loaded for local)
        self.local_tokenizer = None
        self.local_model = None
        self.openrouter_client = None

        if self.backend == "openrouter":
            self._init_openrouter()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[UnifiedGenerator] Warning: Could not save cache to {self.cache_path}: {e}")

    def _init_openrouter(self) -> None:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
        )

    def _init_local_model(self) -> None:
        if self.local_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[UnifiedGenerator] Loading local model '{self.model_name}' in 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Use device_map={'': 'cpu'} if CUDA is not available
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        self.local_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        print(f"[UnifiedGenerator] Local model '{self.model_name}' loaded successfully.")

    def format_prompt(self, query: str, context: Optional[str] = None) -> str:
        q_clean = query.strip()
        if context and len(context.strip()) > 0:
            c_clean = context.strip()
            return CONTEXT_PROMPT_TEMPLATE.format(context=c_clean, query=q_clean)
        return DIRECT_PROMPT_TEMPLATE.format(query=q_clean)

    def compute_cache_key(
        self,
        full_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """
        Deterministic SHA-256 cache key over model name, prompt template version,
        full prompt text, and decoding parameters. Never truncates context.
        """
        payload = (
            f"{self.model_name}|{self.prompt_version}|{full_prompt}|"
            f"temperature={temperature}|do_sample={do_sample}|max_new_tokens={max_new_tokens}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate answer with greedy decoding (temperature=0.0). Checks cache first.
        """
        tokens_budget = max_new_tokens or self.max_new_tokens
        prompt = self.format_prompt(query, context)
        key = self.compute_cache_key(
            full_prompt=prompt,
            max_new_tokens=tokens_budget,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Check cache
        if key in self.cache:
            return self.cache[key]

        # Generate fresh output
        if self.backend == "openrouter":
            if self.openrouter_client is None:
                self._init_openrouter()
            resp = self.openrouter_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=tokens_budget,
            )
            raw_text = resp.choices[0].message.content or ""
            ans = raw_text.strip()
        else:
            self._init_local_model()
            import torch
            inputs = self.local_tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=tokens_budget,
                    do_sample=do_sample,
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            ans = self.local_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Cache and save
        self.cache[key] = ans
        self._save_cache()
        return ans

