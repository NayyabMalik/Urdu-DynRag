# Urdu-DynRAG: Phase 0 Audit & Reproduction Report

**Date:** 2026-11-04  
**Auditor:** Senior ML Engineer & Research Reviewer  
**Repository:** `Urdu-DynRAG`  
**Dataset:** `data/urdu_dynrag_verified_dataset_v3.json` (N=455: Train=306, Val=48, Test=101)  
**Corpus:** `data/urdu_dynrag_verified_corpus_v3.json` (85 unique verified passages)  
**Previous Run Archive:** `results_v1/`

---

## 1. Executive Summary

This document provides a thorough audit of the evaluation metrics, environment dynamics, baseline implementations, and experimental claims in `Urdu-DynRAG` prior to any code modifications.

Every metric computation, feature extraction pipeline, and known limitation was audited directly against the codebase (`phase1_indexer.py`, `phase2_state_extractor.py`, `phase3_env.py`, `phase4_train.py`, and `phase5_evaluate.py`) and verified against the raw execution records in `results_v1/raw_benchmark_records.csv`.

---

## 2. Metric Computation Audit (How Metrics Are Computed Today)

### 2.1 Urdu Token F1 Score
* **Implementation:** `phase3_env.py:compute_token_f1()`
* **Formula:**
  $$\text{Precision} = \frac{|T_{\text{pred}} \cap T_{\text{gold}}|}{|T_{\text{pred}}|}, \quad \text{Recall} = \frac{|T_{\text{pred}} \cap T_{\text{gold}}|}{|T_{\text{gold}}|}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R + 10^{-8}}$$
* **Tokenization:** `tokenize_text()` in `phase2_state_extractor.py`:
  ```python
  clean = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text.lower())
  tokens = [t.strip() for t in clean.split() if len(t.strip()) > 0]
  ```
* **Behavior:** Strips punctuation, preserves Perso-Arabic unicode, lowercases Latin characters, splits on whitespace.

### 2.2 ROUGE-L
* **Implementation:** `phase5_evaluate.py:compute_rouge_l()`
* **Formula:** Longest Common Subsequence (LCS) at the token level between `tokenize_text(prediction)` and `tokenize_text(gold_answer)`:
  $$P_{\text{LCS}} = \frac{\text{LCS}}{|T_{\text{pred}}|}, \quad R_{\text{LCS}} = \frac{\text{LCS}}{|T_{\text{gold}}|}, \quad \text{ROUGE-L} = \frac{2 \cdot P_{\text{LCS}} \cdot R_{\text{LCS}}}{P_{\text{LCS}} + R_{\text{LCS}} + 10^{-8}}$$

### 2.3 chrF++ Score
* **Implementation:** `phase5_evaluate.py:compute_chrf_score()`
* **Formula:** Character n-gram F-score across orders $n \in \{1, 2, 3, 4\}$ over whitespace-stripped character sequences. Averages precision and recall across orders and computes the harmonic mean.

### 2.4 Faithfulness
* **Implementation:** `phase5_evaluate.py:line 283`
* **Formula:**
  ```python
  faithfulness = compute_token_f1(y_pred, retrieved_text) if retrieved_text else 1.0
  ```
* **Flaw:**
  1. If `retrieved_text` is empty (Direct Gen), faithfulness returns `1.000` by default.
  2. For RAG systems, it computes token overlap between a ~15-word answer and a ~150-word retrieved passage chunk. The recall is artificially capped at $\sim 15 / 150 = 0.10$, forcing faithfulness down to $\sim 0.18$.

### 2.5 Retrieval Hit@1
* **Implementation:** `phase5_evaluate.py:line 282`
* **Formula:**
  ```python
  hit1 = 1.0 if target_pid and retrieved_pid == target_pid else 0.0
  ```
* **Flaw:** Evaluated across all 101 test instances, including 6 parametric queries that have no gold passage (`target_pid == ""`).

### 2.6 Abstention Accuracy
* **Implementation:** `phase5_evaluate.py:line 287`
* **Formula:** Evaluated strictly on `style_5_hard_distractor`:
  ```python
  abstention = 1.0 if ("معلومات" in y_pred or "نہیں" in y_pred or hit1 == 0.0) else 0.0
  ```
* **Flaw:** Counts `hit1 == 0.0` as successful abstention even if the model hallucinated an answer.

### 2.7 Net Reward
* **Implementation:** `phase5_evaluate.py:line 293`
* **Formula:**
  $$\text{Reward} = \text{Token } F_1 - 0.15 \cdot C_{\text{retrieval}} - 0.05 \cdot C_{\text{tokens}}$$
  Where $C_{\text{retrieval}} = 0.0$ if Action 0 else $1.0$ ($+0.3$ if reformulated), and $C_{\text{tokens}} = \frac{\text{len}(context.split())}{1000}$.

### 2.8 Latency
* **Implementation:** `phase5_evaluate.py:line 275`
* **Formula:** Total wall-clock time `(time.time() - start_time) * 1000.0` per query. Includes network request latency to OpenRouter API. No warm-up discard and no hardware synchronization.

### 2.9 DQN State Features (14 Dimensions)
Extracted in `phase2_state_extractor.py`:
* $f_0$: Normalized query token count: $\min(1.0, |Tokens(q)| / 50.0)$
* $f_1$: Perso-Arabic character script ratio: $|Chars_{\text{Urdu}}| / |Chars_{\text{clean}}|$
* $f_2$: Interrogative particle indicator: $1.0$ if any word is in a 31-token Urdu question lexicon, else $0.0$
* $f_3$: Top-1 cosine similarity on raw chunk index ($D_{\text{raw}}$)
* $f_4$: Top-1 cosine similarity on synthetic QA index ($D_{\text{qa}}$)
* $f_5$: Raw chunk retrieval margin: $s_{\text{raw}, 1} - s_{\text{raw}, 3}$
* $f_6$: Synthetic QA retrieval margin: $s_{\text{qa}, 1} - s_{\text{qa}, 3}$
* $f_7$: Lexical Jaccard similarity between query tokens and top-1 raw chunk text
* $f_8$: Lexical Jaccard similarity between query tokens and top-1 QA parent chunk text
* $f_9$: Mean generator predictive entropy (via OpenRouter logprobs)
* $f_{10}$: Mean generator top-1 token probability
* $f_{11}$: Predictive entropy variance across generated tokens
* $f_{12}$: Normalized MDP time step: $t / 3.0$
* $f_{13}$: Reformulation indicator: $1.0$ if Action 3 was previously executed, else $0.0$

### 2.10 Training & Validation Reward Discrepancy
* **Implementation:** `phase3_env.py:lines 497–506`
* In `UrduRAGEnv(training_mode=True)`:
  ```python
  if self.training_mode:
      if action == 0:
          is_correct = (optimal_action == "direct_generation")
          f1_score = 0.85 if is_correct else 0.20
      else:
          hit = (retrieved_pid == target_pid) if target_pid else False
          f1_score = 0.82 if hit else 0.22
  ```
* During `phase4_train.py`, validation reward was computed using this deterministic simulation rather than calling real LLM answer generation.

---

## 3. Reproduction of Table 1 (Current System Baseline)

All numbers below were re-computed directly from `results_v1/raw_benchmark_records.csv` across the 101 test queries:

| System | Token F1 (%) | ROUGE-L (%) | chrF++ (%) | Gold Span EM (%) | Hit@1 (%) | Faithfulness | Retrieval Rate (%) | Latency (ms) | Context Tokens | Net Reward |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Gen (No RAG)** | 31.9 ± 18.4 | 29.4 | 40.5 | 0.0 | 0.0 | 1.000 | 0.0 | 2053.3 | 0 | +0.319 |
| **Vanilla Chunk RAG** | 36.5 ± 23.6 | 35.9 | 40.5 | 1.0 | 79.2 | 0.183 | 100.0 | 1475.1 | 64 | +0.212 |
| **DR-RAG (Ahmad et al., 2026)** | 36.7 ± 23.4 | 35.8 | 40.2 | 0.0 | 79.2 | 0.184 | 100.0 | 1664.0 | 64 | +0.214 |
| **Zero-Shot ReAct Agent** | 36.1 ± 23.0 | 34.3 | 40.5 | 0.0 | 74.3 | 0.241 | 93.1 | 3841.7 | 58 | +0.182 |
| **Urdu-DynRAG (Ours)** | 37.4 ± 22.9 | 36.6 | 41.2 | 0.0 | 79.2 | 0.240 | 93.1 | 1145.4 | 60 | +0.232 |

*Status: Exact match (tolerance = 0.0).*

---

## 4. Confirmation / Refutation of Known Problems

### a) Token F1 Inflation via Question Restatement: **CONFIRMED**
* **Finding:** When a trivial "Echo-the-Query" baseline (predicting $y = q$) is evaluated against `gold_answer`, it scores:
  * Overall Test Split ($N=101$): **28.10% Token F1**
  * Clean Synthetic ($S_1$, $N=19$): **60.64% Token F1**
  * Conversational ($S_2$, $N=19$): **33.55% Token F1**
* **Root Cause:** Gold answers in the dataset restate the query almost verbatim before providing the answer clause. On average, **28.3% of tokens in the gold answer are already present in the question**.

### b) Faithfulness Calculation is Invalid: **CONFIRMED**
* **Finding:** Direct Gen scores **1.000** because `retrieved_text` is empty and defaults to 1.0. RAG models score **0.183–0.240** because token F1 is computed against the entire passage chunk (recall denominator = chunk length).
* **Verdict:** Current faithfulness metric is mathematically defective and must be replaced.

### c) Hit@1 Denominator Includes Ungrounded Queries: **CONFIRMED**
* **Finding:** 
  * Over all 101 queries: Hit@1 = **79.2%**
  * Over 95 grounded queries (excluding 6 parametric queries where `passage_id == ""`): Hit@1 = **84.2%**
* **Verdict:** Grounded Hit@1 should report $84.2\%$ with denominator $N=95$.

### d) DR-RAG Baseline Does Not Materially Use Synthetic-QA Index: **CONFIRMED**
* **Finding:** In `phase5_evaluate.py`, DR-RAG selected Action 2 (synthetic QA) on only **1 out of 101 queries** because the threshold `sim_qa_top1 >= 0.80` is too strict for BGE-M3. On 100 queries, it defaulted to Action 1.
* On the 1 query where Action 2 was selected, both indexes mapped to the identical parent passage chunk.
* **Verdict:** DR-RAG and Vanilla Chunk RAG retrieved identical contexts across 100% of test queries.

### e) Policy Seldom Reformulates and Ablations Match Full Model: **CONFIRMED**
* **Finding:** In `Urdu-DynRAG (Ours)`:
  * Final Action 0: 7 queries (6.9%)
  * Final Action 1: 89 queries (88.1%)
  * Final Action 2: 5 queries (5.0%)
  * Action 3 (Reformulation): **0 occurrences (0.0%)**
* **Ablations in Table 3:**
  * Full Urdu-DynRAG: 37.0% F1
  * w/o Action 3: 37.9% F1
  * w/o Synthetic QA: 37.9% F1
  * w/o Entropy: 37.9% F1
* Every ablation scored $\ge$ the full model because Action 3 was never taken and Action 2 was rarely chosen.

### f) Validation Reward (~0.73) and F1 (~0.78) Detached from Real Generation: **CONFIRMED**
* **Finding:** `UrduRAGEnv(training_mode=True)` hardcoded `f1_score = 0.85 if is_correct else 0.20` and `0.82 if hit else 0.22`, plus a $+0.1$ gold span bonus.
* Validation during training evaluated against this mock environment rather than real LLM generation. Real LLM generation F1 on test is ~0.37.

### g) Parametric Results in Table 2 Based on N=2: **CONFIRMED**
* **Finding:** The test split contains exactly:
  * `parametric_nastaliq`: $N=2$
  * `parametric_conversational`: $N=2$
  * `parametric_roman_urdu`: $N=2$
* A total of 6 parametric queries exist in test. Reporting separate columns for each with $N=2$ is statistically insufficient.

### h) Latency Distorted by Cold-Start and Run Order: **CONFIRMED**
* **Finding:** Direct Gen was evaluated first. Query 1 of Direct Gen took **9,981.6 ms** due to HTTPS/TCP handshake and API connection establishment.
* This inflated Direct Gen's mean latency to **2,053.3 ms** (median: 1,228.6 ms).
* Evaluating systems sequentially without warm-up queries skews comparative latency.

---

## 5. Failure Analysis: 25 Sampled Test Predictions

Twenty-five test instances were sampled (seed 42) and evaluated through the pipeline. The full records are stored in `docs/audit_25_examples.json`.

### Failure Taxonomy:

| Failure Mode | Frequency (in 25 samples) | Example Query ID | Description |
| :--- | :---: | :--- | :--- |
| **Extractive Conciseness Penalized by Token F1** | 14 / 25 (56%) | `urdu_dynv_0351`, `urdu_dynv_0151` | Model correctly predicted key facts (e.g., "چھ ماہ", "چار", "5 سال"), but received low Token F1 because the gold answer restates the entire question. |
| **Cross-Lingual Retrieval Failure on Roman Urdu** | 4 / 25 (16%) | `urdu_dynv_0049`, `urdu_dynv_0419` | Roman Urdu queries failed to match Urdu Nastaliq chunks in BGE-M3 dense space without transliteration. |
| **Language Leakage / Tokenizer Drift** | 1 / 25 (4%) | `urdu_dynv_0419` | Model generated mixed Chinese and Urdu tokens when prompted with Roman Urdu without script normalization. |
| **Distractor / Clause Confusion** | 2 / 25 (8%) | `urdu_dynv_0137` | Passage contained multiple legal clauses (e.g., 2 years vs 7 years); model extracted the wrong clause. |
| **Proper Grounded Abstention** | 3 / 25 (12%) | `urdu_dynv_0050` | Hard distractor query correctly triggered abstention phrase ("معلومات دستیاب نہیں"). |
| **Correct / Acceptable** | 1 / 25 (4%) | `urdu_dynv_0109` | Fully grounded correct answer matching gold entities. |

---

## 6. Phase 0 Conclusions & Action Plan for Subsequent Phases

1. **Phase 1 (Metrics):** Implement Key-Fact Accuracy (primary metric) with Urdu orthographic, numeric, and script normalizers. Drop flawed Faithfulness; report Grounded Hit@1 ($N=95$); add Echo and Oracle controls.
2. **Phase 2 (Protocol):** 5-fold cross-validation grouped by passage ID across all 85 passages. Dev validation only. Merge parametric queries to guarantee $N \ge 10$ per reported cell.
3. **Phase 3 (Baselines):** Calibrate DR-RAG's threshold so it actively queries the synthetic QA index. Randomize execution order with 5 warm-up queries.
4. **Phase 4 (Prompts):** Compare Urdu, English, and Extractive prompts on dev only.
5. **Phase 5 (Environment & Policy):** Pre-generate an offline Oracle transition table across all 4 actions. Train RL and supervised policies against real answer scores, not mock F1 heuristics. Implement real Roman Urdu transliteration for Action 3.
6. **Phase 6 (Reporting):** Fully automated generation of camera-ready tables, error analysis, and final summary report.
