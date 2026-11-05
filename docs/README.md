# Documentation & Specifications

This directory contains research specifications, audit documentation, and architectural plans for the **Urdu-DynRAG** project.

## Directory Layout

```
docs/
├── specs/
│   └── Urdu-DynRAG_Specification_Master_Plan.pdf   # Research specification & master plan
│
├── prompt_templates.md                              # Scientific LLM prompt templates (ACL Reproducibility)
├── AUDIT.md                                         # Comprehensive Phase 1 & 1b code audit report
├── audit_25_examples.json                           # 25 qualitative audit examples across styles
└── reproduction_20_report.json                      # 20-sample reproduction audit report
```

### Key Milestones Documented:
1. **`specs/`**: System architecture, mathematical MDP formulations, and reward design.
2. **`prompt_templates.md`**: Exact system and task prompt templates used across all RAG and reformulation models.
3. **`AUDIT.md`**: Line-by-line audit detailing the 5 bugs in Phase 1b (lexicon gap, Action 3 stub, reward alignment) and how they were resolved in Phase 2.
4. **`reports/`**: Complete phase evaluation reports (`reports/phase1.md`, `reports/phase1b.md`, `reports/phase2.md`).
