# Urdu-DynRAG: Phase 0 & Phase 0b Comprehensive Audit & Reproduction Report

**Date:** 2026-11-04  
**Auditor:** Senior ML Engineer & Research Reviewer  
**Repository:** `Urdu-DynRAG`  
**Dataset:** `data/urdu_dynrag_verified_dataset_v3.json` (N=455: Train=306, Val=48, Test=101)  
**Corpus:** `data/urdu_dynrag_verified_corpus_v3.json` (85 unique verified passages)  
**Previous Run Archive:** `results_v1/`  
**Phase Commit Tags:** `phase-0-audit`, `phase-0b`

---

## 1. Generator Facts & Execution Environment

* **Generator Model:** `qwen/qwen-2.5-7b-instruct` (specified via environment variable `GENERATOR_MODEL=qwen/qwen-2.5-7b-instruct` in `.env`).
* **Inference Provider:** `OpenRouter API` (remote cloud endpoint: `https://openrouter.ai/api/v1`).
* **Quantization:** Cloud-hosted remote model (bfloat16 / provider default serving precision). Not locally quantized.
* **Sampling Parameters:**
  * `temperature`: **0.2** (`phase3_env.py:line 225`)
  * `max_tokens`: **80** (`phase3_env.py:line 224`)
* **Logprobs Availability:**
  * Generation calls (`FrozenAnswerGenerator.generate_answer`): **No logprobs requested** (standard raw completion).
  * State feature extraction (`APIGeneratorUncertaintyProvider.compute_entropy_features`): **Logprobs requested** (`top_logprobs=5`, `max_tokens=15`) on a separate query-only draft completion.
* **Output Caching:** In the baseline `phase5_evaluate.py`, answer strings were **NOT cached to disk** (only scalar summary metrics were persisted to CSV).
* **CRITICAL LATENCY & EFFICIENCY WARNING:**  
  > [!WARNING]  
  > All generation and logprob calls in `results_v1` were transmitted over the public internet to OpenRouter's cloud API endpoint. The measured `latency_ms` includes non-deterministic network round-trip time (RTT), TLS/TCP handshake setup, remote server queuing, and cloud provider load spikes. **Therefore, the reported latency numbers in Table 1 do NOT constitute valid scientific evidence of algorithmic or computational efficiency.**

---

## 2. Security Audit (Git History Check)

A rigorous check on the repository git history was performed:
* Commands executed: `git log -p -S "sk-or-"`, `git log --all --full-history -- ".env"`, `git log --all --full-history -- "cache" "cache/*"`.
* **API keys in git history:** **NO**
* **`.env` committed in git history:** **NO**
* **Cache files with sensitive data in git history:** **NO**

---

## 3. Real Reproduction Experiment (20 Fixed Test Queries Re-run Twice)

With generation cache completely disabled, 20 fixed test queries (sampled across styles, seed 42) were evaluated twice through both `Vanilla Chunk RAG` and `Urdu-DynRAG` and compared against `results_v1`:

| Metric | Vanilla Chunk RAG | Urdu-DynRAG (Ours) |
| :--- | :---: | :---: |
| **Action Agreement (Run 1 vs Run 2)** | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** |
| **Exact String Match (Run 1 vs Run 2)** | **8 / 20 (40.0%)** | **8 / 20 (40.0%)** |
| **Mean Absolute Token F1 Difference** | **6.36%** | **8.16%** |
| **Run 1 Mean Token F1** | 34.18% | 33.69% |
| **Run 2 Mean Token F1** | 30.12% | 31.00% |
| **results_v1 Mean Token F1 (same 20 queries)** | 35.83% | 29.90% |
| **Discrepancy (Run 1 vs results_v1)** | **1.65%** | **3.78%** |

**Findings:**
1. Action routing and context retrieval reproduce with **100% deterministic precision**.
2. Because generation uses non-zero temperature ($T=0.2$) via remote cloud sampling, exact string reproduction occurs on 40% of queries, with a natural token F1 fluctuation of ~2–4% around the original `results_v1` baseline.

---

## 4. Cost Accounting Table

Computation required for each action, including state feature extraction:

| Action | State Extraction Overhead | Action-Specific Retrieval Overhead | Answer Generation Call | Reward Charged Today? |
| :--- | :--- | :--- | :--- | :--- |
| **Action 0: Direct Gen** | 1 BGE-M3 encode (1024-d)<br>2 FAISS lookups ($D_{raw}, D_{qa}$ top-3)<br>2 Jaccard lexical checks<br>1 OpenRouter draft call (15 tokens, logprobs) | **None** (0 tokens context) | 1 LLM call (max 80 tokens) | Charged $C_{retrieval}=0$, $C_{tokens}=0$. **State extraction overhead (draft LLM call + embedding) is NOT charged.** |
| **Action 1: Raw Chunk RAG** | Same state extraction overhead as Action 0 | 1 FAISS lookup on $D_{raw}$ (or reuse state top-1)<br>~64 context tokens fetched | 1 LLM call (query + 64 context tokens, max 80 tokens) | Charged $0.15 \times 1.0$ + $0.05 \times \frac{64}{1000} \approx 0.153$. **Draft LLM call and FAISS lookups not charged.** |
| **Action 2: Synthetic QA RAG** | Same state extraction overhead as Action 0 | 1 FAISS lookup on $D_{qa}$ (or reuse state top-1)<br>~64 parent context tokens fetched | 1 LLM call (query + 64 context tokens, max 80 tokens) | Charged $0.15 \times 1.0$ + $0.05 \times \frac{64}{1000} \approx 0.153$. **Draft LLM call and FAISS lookups not charged.** |
| **Action 3: Reformulate** | Same state extraction overhead as Action 0 | Rule-based lexical rewrite; episode advances $t \leftarrow t+1$<br>Full state extraction runs **again** ($1 \times$ BGE-M3, $2 \times$ FAISS, $1 \times$ draft LLM call) | None in step $t$; occurs in terminal step $t+1$ | Charged $0.15 \times 0.3 = 0.045$ for step 3. **The repeated state extraction (extra LLM draft call) is NOT charged.** |

---

## 5. Feature Audit & Ablation Discrepancy

### 5.1 Code Index to Name Mapping (`phase2_state_extractor.py:L433-L456`)

| 0-Based Index | 1-Based Name | Variable Name | Description | Bounded Range |
| :---: | :---: | :--- | :--- | :---: |
| `0` | $f_1$ | `token_count` | Query token length / 50.0 | $[0.0, 1.0]$ |
| `1` | $f_2$ | `urdu_chars` | Perso-Arabic unicode character ratio | $[0.0, 1.0]$ |
| `2` | $f_3$ | `has_question_particle` | Binary indicator for 31 Urdu interrogatives | $\{0.0, 1.0\}$ |
| `3` | $f_4$ | `raw_top1` | Top-1 cosine similarity on $D_{\text{raw}}$ | $[0.0, 1.0]$ |
| `4` | $f_5$ | `qa_top1` | Top-1 cosine similarity on $D_{\text{qa}}$ | $[0.0, 1.0]$ |
| `5` | $f_6$ | `raw_margin` | Top-1 to Top-3 cosine margin ($D_{\text{raw}}$) | $[0.0, 1.0]$ |
| `6` | $f_7$ | `qa_margin` | Top-1 to Top-3 cosine margin ($D_{\text{qa}}$) | $[0.0, 1.0]$ |
| `7` | $f_8$ | `jaccard_raw` | Lexical Jaccard similarity to top-1 raw text | $[0.0, 1.0]$ |
| `8` | $f_9$ | `jaccard_qa` | Lexical Jaccard similarity to top-1 QA parent text | $[0.0, 1.0]$ |
| `9` | $f_{10}$ | `generator_mean_entropy` | Mean next-token predictive entropy | $[0.0, 1.0]$ |
| `10` | $f_{11}$ | `generator_mean_top1_prob` | Mean top-1 token probability | $[0.0, 1.0]$ |
| `11` | $f_{12}$ | `generator_entropy_var` | Predictive entropy variance across tokens | $[0.0, 1.0]$ |
| `12` | $f_{13}$ | `step_ratio` | MDP episode step ratio: $t / 3.0$ | $[0.0, 1.0]$ |
| `13` | $f_{14}$ | `reformulated_flag` | Binary flag: $1.0$ if Action 3 was executed | $\{0.0, 1.0\}$ |

### 5.2 Ablation Mismatch Flag
In `phase5_evaluate.py:line 235`:
```python
if ablation_flags and ablation_flags.get("zero_entropy_features"):
    s_vec[9:12] = 0.5
```
1. **Index alignment:** Python slice `s_vec[9:12]` modifies 0-based indices `9`, `10`, `11`, which correspond to 1-based features $f_{10}, f_{11}, f_{12}$.
2. **Discrepancy:** The published Table 3 header reads `"w/o Generator Entropy (f10-f12 zeroed)"`, but the code actually assigned them to `0.5` (midpoint), **not zero**!

---

## 6. Baseline Faithfulness: DR-RAG (Ahmad et al., 2026)

* **Paper Description:** Dense Retrieval RAG for low-resource Urdu uses dual representations: text chunks ($D_{raw}$) and synthetic question-answer pairs ($D_{qa}$). For every query, it searches both representations and fuses/aggregates candidates to select optimal context passages.
* **Our Baseline Implementation (`phase5_evaluate.py:line 215`):**
  ```python
  sim_qa_top1 = s_vec[4]
  action = 2 if sim_qa_top1 >= 0.80 else 1
  ```
* **Evaluation:**
  * It does NOT query both representations and fuse results.
  * It employs a hard heuristic with a threshold ($0.80$) that was triggered on only 1 out of 101 queries.
  * On 100/101 queries, it executed standard Vanilla Chunk RAG.
* **VERDICT:** **The DR-RAG baseline in `results_v1` is NON-FAITHFUL to Ahmad et al. (2026).**

---

## 7. Hand-Labeled Failure Taxonomy (`tests/golden_25.json`)

All 25 sampled examples from the Phase 0 audit were manually examined and assigned exactly **ONE primary label** from the standardized taxonomy. Full per-item records and rationale are saved in [`tests/golden_25.json`](file:///C:/Users/PMLS/Documents/Urdu-DynRAG/tests/golden_25.json).

*(Note: Parametric examples in this test were forced through RAG to inspect generator behavior under irrelevant retrieval).*

| Primary Label | Count (N=25) | Percentage | Description / Sample IDs |
| :--- | :---: | :---: | :--- |
| `correct` | 10 | 40.0% | Accurately extracted gold key entities (`urdu_dynv_0351`, `0014`, `0151`, `0109`, `0371`, `0047`, `0374`, `0108`, `0347`, `0106`). |
| `correct_but_verbose` | 1 | 4.0% | Correct fact accompanied by redundant conditional clauses (`urdu_dynv_0073`). |
| `wrong_fact` | 0 | 0.0% | Direct factual errors when retrieving the correct context. |
| `wrong_clause` | 1 | 4.0% | Context contained multiple penalty clauses; model extracted the 2-year simple threat clause instead of the 7-year death threat clause (`urdu_dynv_0137`). |
| `false_abstention` | 2 | 8.0% | Parametric questions answerable from internal knowledge, but forced irrelevant context triggered false abstention (`urdu_dynv_0320`, `0310`). |
| `hallucinated_on_unanswerable` | 2 | 8.0% | S5 distractor queries; model hallucinated facts ("5 سال", "25 دن") instead of abstaining (`urdu_dynv_0215`, `0110`). |
| `correct_abstention` | 6 | 24.0% | Properly abstained on unanswerable distractors (`urdu_dynv_0050`, `0285`, `0015`, `0270`, `0355`, `0375`). |
| `retrieval_miss` | 1 | 4.0% | Roman Urdu cross-lingual gap caused retrieval of Indus River passage instead of Earth science passage (`urdu_dynv_0049`). |
| `language_leakage` | 1 | 4.0% | Model generated Chinese characters when responding to a Roman Urdu query (`urdu_dynv_0419`). |
| `self_contradiction` | 1 | 4.0% | Stated that the decision is binding, then immediately stated it is not binding (`urdu_dynv_0453`). |
| **Total** | **25** | **100.0%** | |

---

## 8. Summary of Confirmed Problems from Phase 0

1. **Token F1 Inflation:** Echo baseline achieves 28.10% overall and 60.64% on S1; 28.3% of gold tokens are in the query.
2. **Defective Faithfulness:** Returns 1.0 for Direct Gen and ~0.18 for RAG. Must be replaced.
3. **Hit@1 Denominator:** 79.2% ($N=101$) vs 84.2% ($N=95$ grounded).
4. **DR-RAG Non-Faithful:** Threshold 0.80 triggered only once; 100% context overlap with Vanilla RAG.
5. **DQN Non-Adaptive:** Action 3 taken 0 times (0.0%); all ablations scored $\ge$ full model.
6. **Training Reward Detached:** `training_mode=True` hardcoded synthetic reward heuristics ($F_1 = 0.85/0.82$).
7. **Parametric Underpowered:** Exactly $N=2$ per cell in Table 2.
8. **Latency Cold-Start:** Query 1 of Direct Gen took 9,981 ms, distorting mean latency.
