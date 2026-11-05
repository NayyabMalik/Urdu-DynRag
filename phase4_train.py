#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Root Forwarding Shim for Phase 4 Double DQN Training.
The canonical implementation resides in `src/phase4_train.py`.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.phase4_train import *

if __name__ == "__main__":
    from src.phase4_train import main
    main()
