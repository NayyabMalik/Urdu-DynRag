#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Root Forwarding Shim for Phase 2 State Extractor.
The canonical implementation resides in `src/phase2_state_extractor.py`.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.phase2_state_extractor import *

if __name__ == "__main__":
    from src.phase2_state_extractor import main
    main()
