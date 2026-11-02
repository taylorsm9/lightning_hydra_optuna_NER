# 05 · Hydra + separate train/test + Optuna runner

Part of [lightning_cli_ner](../README.md), an archived teaching project. This is 04's setup as it stood in
uncommitted local work in April 2025; it was never on a branch.

## What it shows

Training and testing are separate scripts. `test.py` takes `+checkpoint_path=`, which can be a Lightning
`.ckpt`, a plain state dict or a DeepSpeed checkpoint directory, and both scripts upload the resolved Hydra
config to W&B.

`optuna_runner.py` drives Optuna from a plain script instead of the Hydra sweeper plugin, which is
unmaintained and pins `optuna<3`. Each trial runs `train.py <overrides>` as a subprocess and reads the
score from its stdout.

The model can also use gradient checkpointing (`use_gradient_checkpointing`), and it logs
`val_f1_micro_no_o`, a micro-F1 that ignores tokens where both the label and the prediction are `O`.

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
