# 05 · Hydra + separate train/test + Optuna runner

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from uncommitted local work (April 2025). It was never on a branch.

## What it shows

04's setup, as it stood in April 2025 in uncommitted work that never made it to a branch:

- **Separate `train.py` and `test.py`.** `test.py` takes `+checkpoint_path=`: a Lightning `.ckpt`, a plain
  state dict, or a DeepSpeed checkpoint directory. Both scripts upload the resolved Hydra config to W&B.
- **`optuna_runner.py`** drives Optuna from a plain script. Each trial runs `train.py <overrides>` as a
  subprocess and reads the score from its stdout. That replaces the Hydra sweeper plugin, which is
  unmaintained and pins `optuna<3`.
- **Gradient checkpointing** as a model option (`use_gradient_checkpointing`).
- **`val_f1_micro_no_o`**, a micro-F1 that ignores tokens where both the label and the prediction are `O`.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python train.py
python optuna_runner.py                        # 50 trials
python test.py +checkpoint_path=logs/checkpoints/last.ckpt
```

The DeepSpeed notes from 04 apply; `test.py` merges a DeepSpeed checkpoint directory into fp32 weights and
tests without DeepSpeed.

## Notes

- `metric_level` (model setting) picks the F1 that checkpointing and Optuna use: `token` (default, `val_avg_f1`, credits partly found entities) or `entity` (`val_entity_f1`, exact CoNLL). Both are logged.
- With the default `IOB: 0` the model predicts IO tags, so two adjacent entities of the same type can't be told apart, and entity F1 is computed against IO-converted gold labels (slightly optimistic). Set `IOB: 1` for exact CoNLL F1.
- With `batch_size: 512` an epoch is only about 27 optimizer steps.
