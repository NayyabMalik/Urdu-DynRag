#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from phase1_indexer import (
    DualFAISSRetriever,
    build_indices,
    load_corpus,
    generate_or_load_synthetic_qa,
    run_verification_smoke_test,
    main,
)

if __name__ == "__main__":
    main()

