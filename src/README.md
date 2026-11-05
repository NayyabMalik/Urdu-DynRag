# Urdu-DynRAG Core Library (`src/`)

This directory contains the core library modules and canonical evaluation runners for **Urdu-DynRAG**: Dynamic RL-Guided Adaptive Retrieval for Under-Resourced and Mixed-Script QA.

## Architecture Overview

```
src/
├── __init__.py
│
├── phase1_indexer.py              # Phase 1: Dual-representation FAISS vector indexing (D_raw, D_qa)
├── phase2_state_extractor.py      # Phase 2: 14-dimensional linguistic, retrieval & uncertainty feature extractor
├── phase3_env.py                  # Phase 3: 2-step MDP Gymnasium environment with real outcome rewards
├── phase4_train.py                # Phase 4: Resumable Double DQN policy network and replay buffer
├── phase4_train_eval.py           # Phase 4/5: Baseline evaluation harness and policy evaluation
│
├── reformulator.py                # Phase 2: DeterministicUrduReformulator (350+ term legal lexicon & transliteration)
├── generator.py                   # Unified LLM generation client (OpenRouter, Greedy T=0.0)
├── metrics.py                     # Official evaluation suite (Key-Fact Accuracy, Token F1, Hit@1, Bootstrap CIs)
│
├── phase2_cv_benchmark.py         # Main Phase 2: 5-Fold Passage-Grouped Cross-Validation benchmark engine
├── test_roman_urdu_robustness.py  # Phase 2: Orthographic noise stress test (Table 5) across 5 noise regimes
├── generate_paper_figures.py      # Publication figure generator (300 DPI PNG & vector PDF Figures 1–4)
└── build_action_outcome_table.py  # Precomputes real generator outcomes across all 455 items and 5 actions
```

## Primary Entrypoints

| Task | Command | Description |
| :--- | :--- | :--- |
| **5-Fold Cross-Validation** | `python src/phase2_cv_benchmark.py` | 5-Fold passage-grouped CV ($N=455$), 5,000 bootstrap CIs, generates Tables 1–4. |
| **Orthographic Stress Test** | `python src/test_roman_urdu_robustness.py` | Evaluates Roman Urdu ($S_4$) resilience under 5 spelling noise regimes (Table 5). |
| **Publication Figures** | `python src/generate_paper_figures.py` | Generates camera-ready Figures 1–4 in `experiments/figures/`. |
| **Precompute Outcomes** | `python src/build_action_outcome_table.py` | Populates `cache/action_outcome_table.json` for deterministic offline training. |

