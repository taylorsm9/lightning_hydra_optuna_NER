"""Run an entry script with DeepSpeedCPUAdam swapped for torch.optim.AdamW (smoke test only).

Folders 04 and 05 hardcode DeepSpeedCPUAdam, which is meant to run inside DeepSpeed's ZeRO-offload engine on a
GPU machine. Used standalone on a CPU-only machine it segfaults in its first step (deepspeed 0.19.9), so the CPU
smoke test swaps it out here instead of changing the project code. The DeepSpeed path itself is untested on CPU.

Usage (from inside the folder): python run_without_deepspeed_optim.py <entry script> [hydra overrides ...]
"""
import runpy
import sys

import torch
import deepspeed.ops.adam


class AdamWStandIn(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, **_):
        super().__init__(params, lr=lr, weight_decay=weight_decay)


deepspeed.ops.adam.DeepSpeedCPUAdam = AdamWStandIn

entry = sys.argv[1]
sys.path.insert(0, ".")
sys.argv = [entry] + sys.argv[2:]
runpy.run_path(entry, run_name="__main__")
