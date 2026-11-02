# 01 · LightningCLI

Part of [lightning_hydra_optuna_NER](../README.md), an archived teaching project. Copied from branch `master`.

## What it shows

`LightningCLI` reads one yaml (`config/conll_2003/conll.yaml`) and builds the trainer, model,
datamodule, logger and callbacks from its `class_path`/`init_args` entries. `main.py` is about 30 lines.
It only subclasses the CLI to link shared arguments (`IOB`, `num_labels`, `batch_size`, `model_class`)
from the data section to the model section, so they're set in one place.

Compare with [`02_plain_trainer/trainer.py`](../02_plain_trainer/trainer.py), which does the same wiring by hand.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python main.py fit --config config/conll_2003/conll.yaml
python main.py fit --config config/conll_2003/conll.yaml --trainer.max_epochs 5 --data.init_args.batch_size 16
python main.py test --config config/conll_2003/conll.yaml --ckpt_path logs/conll/checkpoints/<best>.ckpt
```

Checkpoints go to `logs/conll/checkpoints/` (the best by `val_score`, plus `last.ckpt`), test predictions
to `logs/conll/test_preds/test_data.pkl`, and the resolved config to `logs/conll/config.yaml`.
`bash_scripts/conll/conll_baseline.sh` is a SLURM script (fill in your own partition and email).
On a 24 GB GPU, use `--data.init_args.batch_size 16 --trainer.accumulate_grad_batches 2` (same effective batch).

## Notes

- `metric_level` (model setting) picks the F1 that checkpointing and Optuna use: `token` (default, `val_avg_f1`, credits partly found entities) or `entity` (`val_entity_f1`, exact CoNLL). Both are logged.
- With the default `IOB: 0` the model predicts IO tags, so two adjacent entities of the same type can't be told apart, and entity F1 is computed against IO-converted gold labels (slightly optimistic). Set `IOB: 1` for exact CoNLL F1.
