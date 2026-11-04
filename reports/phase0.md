# Phase 0 Audit & Reproduction Report

**Status:** Complete  
**Phase Commit Tag:** `phase-0-audit`  
**Artifacts Generated:**  
- `docs/AUDIT.md`: Detailed computation formulas, code line audits, and point-by-point evaluations.
- `docs/audit_25_examples.json`: 25 sampled test predictions with retrieved passage IDs and categorized generator failures.
- `results_v1/`: Full archive of prior experiments, figures, and raw records.

---

## 1. What Was Audited

1. **Metric Implementations:**
   - Evaluated `compute_token_f1`, `compute_rouge_l`, `compute_chrf_score`, `retrieval_hit1`, `faithfulness`, and `latency_ms` in `phase3_env.py` and `phase5_evaluate.py`.
2. **Environment & Training Dynamics:**
   - Audited the 14 DQN state features in `phase2_state_extractor.py`.
   - Audited `UrduRAGEnv(training_mode=True)` in `phase3_env.py` and confirmed why training/validation rewards (~0.73) were detached from test scores (~0.23).
3. **Table 1 Reproduction:**
   - Successfully reproduced the exact Table 1 values from `results_v1/raw_benchmark_records.csv` across all 5 systems and 10 metrics with 0.0 tolerance.
4. **Baseline Independence:**
   - Evaluated DR-RAG's threshold condition and proved that DR-RAG and Vanilla Chunk RAG retrieved identical contexts across 100% of test queries.

---

## 2. Key Findings & Confirmed Issues (Points a - h)

| Point | Hypothesis | Status | Key Metric / Evidence |
| :---: | :--- | :---: | :--- |
| **a** | Token F1 is inflated by question restatement in gold answers | **CONFIRMED** | Echo-the-query baseline achieves **28.10% F1** overall, **60.64% F1** on Clean Synthetic ($S_1$), and **33.55% F1** on Conversational ($S_2$). 28.3% of gold answer tokens appear in the query. |
| **b** | Faithfulness is invalid (1.0 for Direct Gen, ~0.18 for RAG) | **CONFIRMED** | Direct Gen defaults to 1.0 because context is empty; RAG is computed as token F1 against full 150-word passage chunks, depressing score to 0.18. |
| **c** | Hit@1 denominator includes 6 ungrounded parametric queries | **CONFIRMED** | Overall Hit@1 is **79.2%** ($N=101$), while Grounded Hit@1 is **84.2%** ($N=95$). |
| **d** | DR-RAG does not materially use synthetic QA index | **CONFIRMED** | DR-RAG took Action 2 only once ($1/101$) due to over-strict 0.80 cosine threshold. Retained identical context to Vanilla RAG on 100% of queries. |
| **e** | Policy retrieves 93.1%, never reformulates, ablations score $\ge$ full | **CONFIRMED** | Policy took Action 3 (reformulation) 0 times (0.0%). All ablations in Table 3 scored 37.9% vs 37.0% for the full model. |
| **f** | Training/validation reward (~0.73) differs from test (~0.23) | **CONFIRMED** | `training_mode=True` hardcoded synthetic reward heuristics ($F_1 = 0.85/0.82$) without executing real LLM generation. |
| **g** | Parametric results in Table 2 based on $N=2$ per cell | **CONFIRMED** | Test set contains exactly 2 items each for `parametric_nastaliq`, `parametric_conversational`, and `parametric_roman_urdu`. |
| **h** | Latency distorted by cold-start and run order | **CONFIRMED** | First query of Direct Gen took 9,981 ms (API handshake), skewing mean latency to 2,053 ms vs median of 1,228 ms. |

---

## 3. Negative Results & Honest Assessment

1. **The current Token F1 metric is not trustworthy for evaluating QA quality.** A trivial echo baseline achieves 60.6% F1 on synthetic queries without answering anything.
2. **The current DQN policy is not truly adaptive.** It acts essentially as a binary switch between direct generation (on parametric queries) and raw chunk retrieval (on almost everything else). It never learned to reformulate ($0\%$).
3. **The reported DR-RAG baseline was not a genuine comparative test.** Because it retrieved the same chunks as Vanilla RAG, the comparison between DR-RAG and Vanilla RAG in Table 1 was measuring generator noise rather than retrieval differences.

---

## 4. Next Phase Readiness: Phase 1 (Metrics)

We are ready to build `src/metrics.py` with:
- **Key-Fact Accuracy:** Normalized fact extraction over numbers, dates, amounts, named entities, and Urdu numerals.
- **Controls:** Echo-the-query baseline and Oracle-span baseline in every table.
- **Abstention Metrics:** Precision, recall, F1 on unanswerable queries ($S_5$), plus false-abstention rate.
- **Grounded Retrieval:** Hit@1, Hit@5, MRR with explicit denominators ($N=95$).
- Comprehensive unit tests in `tests/test_metrics.py`.
