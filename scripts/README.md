# Scripts & Utilities

This directory contains standalone execution scripts, historical evaluation drivers, and audit tools.

## Structure

```
scripts/
├── audit/                   # Verification and audit reproduction scripts
│   ├── audit_analysis.py            # Token F1 and Echo Query baseline analysis
│   ├── audit_standalone.py          # Standalone verification suite
│   ├── generate_audit_examples.py   # Generates qualitative LLM outputs for audit inspection
│   ├── sample_audit_examples.py     # Samples 25 random qualitative queries across styles
│   └── test_reproduction_20.py      # 20-sample reproduction test script
│
└── legacy_eval/             # Historical evaluation drivers (Phases 1, 1b, and 5)
    ├── evaluate_phase1.py           # Phase 1 5-system baseline evaluation
    ├── evaluate_phase1b.py          # Phase 1b greedy evaluation with headroom analysis
    └── phase5_evaluate.py           # Multi-baseline benchmarking with bootstrap tests
```

> **Note**: For official Phase 2 cross-validation benchmarks and publication figures, use the canonical runners in `src/`:
> - `python src/phase2_cv_benchmark.py` (5-Fold Passage-Grouped CV)
> - `python src/test_roman_urdu_robustness.py` (Table 5 Orthographic Noise Stress Test)
> - `python src/generate_paper_figures.py` (Camera-Ready Figures 1–4)

