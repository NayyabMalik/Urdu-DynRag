# Experimental LLM Prompt Templates

This document details the exact prompt templates used in **Urdu-DynRAG** for generation, reformulation, and direct parametric baseline inference (as required by the ACL/EMNLP reproducibility checklist).

---

## 1. Grounded RAG Generation Prompt (Actions 1, 2)

Used when context is retrieved from either the raw chunk index ($D_{\text{raw}}$) or the synthetic QA index ($D_{\text{qa}}$). Evaluated using greedy decoding ($T = 0.0$, `max_new_tokens = 40`).

```
آپ ایک مستند قانونی اور انتظامی معاون ہیں۔ درج ذیل سیاق و سباق (Context) کی بنیاد پر صارف کے سوال کا مختصر، دوٹوک اور درست جواب اردو میں دیں۔ اگر جواب سیاق و سباق میں نہیں ہے تو خود سے قیاس نہ کریں۔

سیاق و سباق:
"""{context}"""

سوال: {query}
جواب:
```

*English Translation:*
> "You are an authoritative legal and administrative assistant. Based on the following context, provide a concise, direct, and accurate answer in Urdu to the user's question. If the answer is not in the context, do not speculate.  
> Context: \"\"\"{context}\"\"\"  
> Question: {query}  
> Answer:"

---

## 2. Direct Parametric Generation Prompt (Action 0)

Used when the policy selects Action 0 (no retrieval, purely internal model knowledge).

```
آپ ایک مستند معاون ہیں۔ اپنے اندرونی علم کی روشنی میں اس سوال کا مختصر اور درست جواب اردو میں دیں:

سوال: {query}
جواب:
```

*English Translation:*
> "You are an authoritative assistant. In light of your internal knowledge, provide a concise and accurate answer in Urdu to this question:  
> Question: {query}  
> Answer:"

---

## 3. Action 3: Nastaliq Transliteration & Reformulation Prompt

Used by `DeterministicUrduReformulator` (Action 3) when Roman Urdu ($S_4$) or code-switched ($S_3$) queries are detected. Executed with greedy decoding ($T=0.0$, `max_new_tokens = 40`) and deterministic SHA-256 caching.

```
Convert the following Roman Urdu or code-switched question into a formal, grammatical Urdu (Nastaliq script) question preserving all numbers, sections, and legal terms. Output ONLY the Urdu question:

Question: {query}
```

