#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from phase2_state_extractor import (
    UrduRAGStateExtractor,
    compute_jaccard_similarity,
    tokenize_text,
)

if __name__ == "__main__":
    from phase2_state_extractor import main
    # Or run unit tests
    import subprocess
    subprocess.run([sys.executable, str(root_dir / "phase2_state_extractor.py")])

