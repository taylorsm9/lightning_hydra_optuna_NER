# 04 · Hydra + DeepSpeed ZeRO-3

Part of [lightning_hydra_optuna_NER](../README.md), an archived teaching project. Copied from branch `lightning_hydra`.

## What it shows

03's Hydra setup, without the folds, plus two changes that save memory. The first is DeepSpeed ZeRO
stage 3 with optimizer and parameter offload to the CPU (`config/trainer/default.yaml`) and
`DeepSpeedCPUAdam` as the optimizer, set up to fit a batch of 512 on one GPU. The second is dynamic
padding: the datamodule's `collate_fn` pads each batch only to its longest sentence instead of padding
everything to 512 tokens, and pads labels with `-100` so the loss ignores them.

`diff -r ../03_hydra_optuna_kfold/src src` shows exactly what those took.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py
python trainer.py test_after_fit=true          # also test the best checkpoint
DS_SKIP_CUDA_CHECK=1 python trainer.py         # if DeepSpeed refuses to compile CPU Adam (nvcc/torch CUDA mismatch)
python trainer.py '~trainer.strategy'          # without DeepSpeed (uses torch AdamW)
python trainer.py +hparams_search=optuna
```

DeepSpeed needs a GPU and `deepspeed==0.16.4` (0.19.9 crashes under ZeRO-3 with this model). Checkpoints are
DeepSpeed directories; `test_after_fit` merges the best one into fp32 weights and tests it without DeepSpeed.

## Notes

- `metric_level` (model setting) picks the F1 that checkpointing and Optuna use: `token` (default, `val_avg_f1`, credits partly found entities) or `entity` (`val_entity_f1`, exact CoNLL). Both are logged.
- With the default `IOB: 0` the model predicts IO tags, so two adjacent entities of the same type can't be told apart, and entity F1 is computed against IO-converted gold labels (slightly optimistic). Set `IOB: 1` for exact CoNLL F1.
- With `batch_size: 512` an epoch is only about 27 optimizer steps.
