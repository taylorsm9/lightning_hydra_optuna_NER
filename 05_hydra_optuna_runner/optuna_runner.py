import subprocess
import sys
import optuna
import shlex
import re
from optuna.samplers import TPESampler

def extract_score(stdout: str) -> float:
    pattern = r"Best Val Score:\s*([0-9]*\.?[0-9]+)"
    for line in reversed(stdout.strip().splitlines()):
        match = re.search(pattern, line)
        if match:
            return float(match.group(1))
    raise ValueError("No 'Best Val Score:' line found in stdout")

def objective(trial: optuna.Trial) -> float:
    cmd = [
        sys.executable, "train.py",  # the same interpreter/environment as this script "logger=wandb_optuna", "callbacks=optuna", "+trainer.enable_checkpointing=False",
        f"model.starting_lr={trial.suggest_float('model.starting_lr', 1e-6, 1e-5, log=True)}",
        f"model.max_lr={trial.suggest_float('model.max_lr', 5e-6, 1e-4, log=True)}",
        f"model.weight_decay={trial.suggest_float('model.weight_decay', 1e-5, 1e-2, log=True)}",
        f"model.hidden_dropout_prob={trial.suggest_float('model.hidden_dropout_prob', 0.1, 0.6)}",
        f"model.attention_probs_dropout_prob={trial.suggest_float('model.attention_probs_dropout_prob', 0.1, 0.5)}",
        f"datamodule.beta={trial.suggest_float('datamodule.beta', 0.9, 0.999, log=True)}",
        f"datamodule.o_class_mod={trial.suggest_float('datamodule.o_class_mod', 2, 20)}"
    ]

    print("[optuna] Running:", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return extract_score(result.stdout)
    except Exception as e:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise e

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50, n_jobs=1)