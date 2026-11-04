# Phase 0 & 0b Audit & Reproduction Report

**Status:** Complete  
**Phase Commit Tag:** `phase-0b`  
**Artifacts Generated:**  
- `docs/AUDIT.md`: Complete audit of metric formulas, generator facts, cost accounting table, feature indices, DR-RAG non-faithfulness, security audit, and 20-query real reproduction results.
- `tests/golden_25.json`: 25 hand-labeled test predictions with primary labels, reasons, and failure taxonomy.
- `docs/reproduction_20_report.json`: Quantitative metrics from the 20-query reproduction experiment.
- `results_v1/`: Full archive of prior experiments, figures, and raw records.

---

## 1. What Was Audited

1. **Metric Implementations:**
   - Evaluated `compute_token_f1`, `compute_rouge_l`, `compute_chrf_score`, `retrieval_hit1`, `faithfulness`, and `latency_ms` in `phase3_env.py` and `phase5_evaluate.py`.
2. **Generator Facts & Latency Warning:**
   - Documented OpenRouter provider, `qwen/qwen-2.5-7b-instruct`, temperature 0.2, max_new_tokens 80, and lack of generation caching.
   - Added explicit warning that remote network latency is not valid algorithmic efficiency evidence.
3. **Environment & Training Dynamics:**
   - Audited the 14 DQN state features in `phase2_state_extractor.py`.
   - Identified that Table 3 ablation "zeroed" features 9–11 by assigning them to 0.5 rather than 0.0.
   - Audited `UrduRAGEnv(training_mode=True)` in `phase3_env.py` and confirmed why training/validation rewards (~0.73) were detached from test scores (~0.23).
4. **Table 1 Reproduction:**
   - Successfully reproduced the exact Table 1 values from `results_v1/raw_benchmark_records.csv` across all 5 systems and 10 metrics with 0.0 tolerance.
5. **Real Reproduction (20 Queries Twice Without Cache):**
   - Action routing reproduced with 100% agreement.
   - Exact string match was 40%, with token F1 fluctuating by ~2–4% due to non-zero temperature ($T=0.2$) on cloud inference.
6. **Baseline Faithfulness:**
   - Evaluated DR-RAG's threshold condition and proved that DR-RAG and Vanilla Chunk RAG retrieved identical contexts across 100% of test queries. Marked DR-RAG as non-faithful.
7. **Security Verification:**
   - Verified that `.env`, API keys, and cache files are not in git history.

---

## 2. Key Findings & Confirmed Issues (Points a - h)

| Point | Hypothesis | Status | Key Metric / Evidence |
| :---: | :--- | :---: | :--- |
| **a** | Token F1 is inflated by question restatement in gold answers | **CONFIRMED** | Echo-the-query baseline achieves **28.10% F1** overall, **60.64% F1** on Clean Synthetic ($S_1$), and **33.55% F1** on Conversational ($S_2$). 28.3% of gold answer tokens appear in the query. |
| **b** | Faithfulness is invalid (1.0 for Direct Gen, ~0.18 for RAG) | **CONFIRMED** | Direct Gen defaults to 1.0 because context is empty; RAG is computed as token F1 against full 150-word passage chunks, depressing score to 0.18. |
| **c** | Hit@1 denominator includes 6 ungrounded parametric queries | **CONFIRMED** | Overall Hit@1 is **79.2%** ($N=101$), while Grounded Hit@1 is **84.2%** ($N=95$). |
| **d** | DR-RAG does not materially use synthetic QA index | **CONFIRMED** | DR-RAG took Action 2 only once ($1/101$) due to over-strict 0.80 cosine threshold. Retained identical context to Vanilla RAG on 100% of queries. Baseline marked non-faithful. |
| **e** | Policy retrieves 93.1%, never reformulates, ablations score $\ge$ full | **CONFIRMED** | Policy took Action 3 (reformulation) 0 times (0.0%). All ablations in Table 3 scored 37.9% vs 37.0% for the full model. |
| **f** | Training/validation reward (~0.73) differs from test (~0.23) | **CONFIRMED** | `training_mode=True` hardcoded synthetic reward heuristics ($F_1 = 0.85/0.82$) without executing real LLM generation. |
| **g** | Parametric results in Table 2 based on $N=2$ per cell | **CONFIRMED** | Test set contains exactly 2 items each for `parametric_nastaliq`, `parametric_conversational`, and `parametric_roman_urdu`. |
| **h** | Latency distorted by cold-start and run order | **CONFIRMED** | First query of Direct Gen took 9,981 ms (API handshake), skewing mean latency to 2,053 ms vs median of 1,228 ms. |

---

## 3. Hand-Labeled Failure Taxonomy (`tests/golden_25.json`)

All 25 samples hand-labeled with ONE primary label (parametric examples forced through RAG):

| Primary Label | Count (N=25) | Percentage | Description |
| :--- | :---: | :---: | :--- |
| `correct` | 10 | 40.0% | Accurately extracted gold key entities (`0351`, `0014`, `0151`, `0109`, `0371`, `0047`, `0374`, `0108`, `0347`, `0106`). |
| `correct_but_verbose` | 1 | 4.0% | Correct fact with redundant conditional explanation (`0073`). |
| `wrong_fact` | 0 | 0.0% | Direct factual errors on correct context. |
| `wrong_clause` | 1 | 4.0% | Extracted 2-year clause instead of 7-year death threat clause (`0137`). |
| `false_abstention` | 2 | 8.0% | Answerable parametric queries forced through irrelevant context, causing false abstention (`0320`, `0310`). |
| `hallucinated_on_unanswerable` | 2 | 8.0% | Fabricated answers ("5 سال", "25 دن") on unanswerable distractor queries (`0215`, `0110`). |
| `correct_abstention` | 6 | 24.0% | Properly abstained on unanswerable distractor queries (`0050`, `0285`, `0015`, `0270`, `0355`, `0375`). |
| `retrieval_miss` | 1 | 4.0% | Cross-lingual transliteration failure on Roman Urdu query (`0049`). |
| `language_leakage` | 1 | 4.0% | Model outputted Chinese tokens when responding to Roman Urdu query (`0419`). |
| `self_contradiction` | 1 | 4.0% | Asserted both that the decision is binding and not binding (`0453`). |
| **Total** | **25** | **100.0%** | |

---

## 4. Next Phase Readiness: Phase 1 (Metrics)

With all Phase 0 and 0b audits approved and committed, we are ready to implement `src/metrics.py` and `tests/test_metrics.py`.
