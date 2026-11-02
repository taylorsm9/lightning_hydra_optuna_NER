# 02 · Plain Lightning Trainer

Part of [lightning_hydra_optuna_NER](../README.md), an archived teaching project. Copied from branch `lightning_trainer`.

## What it shows

The same yaml as 01, but without LightningCLI. `trainer.py` reads it with `yaml.safe_load` and builds
every object by hand: it finds each callback's `init_args`, creates the `WandbLogger`, copies the shared
data arguments into the model arguments, converts the learning rates from strings to floats (`yaml.safe_load`
reads `1e-5` as a string), and passes what's left to `pl.Trainer`.

This shows what LightningCLI does for you, and why the project moved to Hydra next.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py                              # reads config/conll_2003/conll.yaml, trains, then tests the best checkpoint
python trainer.py path/to/another.yaml
```

## Notes

- `metric_level` (model setting) picks the F1 that checkpointing and Optuna use: `token` (default, `val_avg_f1`, credits partly found entities) or `entity` (`val_entity_f1`, exact CoNLL). Both are logged.
- With the default `IOB: 0` the model predicts IO tags, so two adjacent entities of the same type can't be told apart, and entity F1 is computed against IO-converted gold labels (slightly optimistic). Set `IOB: 1` for exact CoNLL F1.
- The default `batch_size` is 1 (slow, but it fits any GPU); raise it in the yaml.
