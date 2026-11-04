#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from phase4_train_eval import (
    DQNPolicy,
    ReplayBuffer,
    train_dqn,
    evaluate_system_instance,
    run_full_benchmark,
    generate_publication_figures,
    main,
)

if __name__ == "__main__":
    main()

