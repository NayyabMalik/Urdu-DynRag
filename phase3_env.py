#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Root Forwarding Shim for Phase 3 MDP Environment.
The canonical implementation resides in `src/phase3_env.py`.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.phase3_env import *

if __name__ == "__main__":
    from src.phase3_env import main
    main()
