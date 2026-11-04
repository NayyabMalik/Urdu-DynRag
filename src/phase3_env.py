#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from phase3_env import (
    UrduRAGEnv,
    FrozenAnswerGenerator,
    compute_token_f1,
    reformulate_query,
)

if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, str(root_dir / "phase3_env.py")])

