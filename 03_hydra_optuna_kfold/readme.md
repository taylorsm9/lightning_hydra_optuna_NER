# 03 · Hydra + Optuna sweeper + 5-fold ensemble

Part of [lightning_hydra_optuna_NER](../README.md), an archived teaching project. Copied from branch `lightning_hydra_ensemble` (also tag `archive/lightning_hydra_ensemble`).

## What it shows

The yaml is split into Hydra config groups (`config/model`, `datamodule`, `trainer`, `callbacks`,
`logger`) composed by `config/conll_config.yaml`. `trainer.py` builds every object with
`hydra.utils.instantiate` and returns the run's best `val_score`, which is what lets the Optuna sweeper plugin
use it as an objective (`config/hparams_search/optuna.yaml`).

It also has a 5-fold ensemble:
- `notebooks/dataset_preprocessing/conll_folds.ipynb` merges train and val and writes 5 folds
- `bash_scripts/conll/5_fold.sh` trains one model per fold and writes each one's test predictions
- `notebooks/testing/ensemble_test.ipynb` averages the models' probabilities, with weights tuned by Optuna

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py
python trainer.py trainer.max_epochs=5 datamodule.batch_size=8
python trainer.py test_after_fit=true                        # also test the best checkpoint
python trainer.py +hparams_search=optuna                     # 20-trial sweep; results in logs/multirun/
python trainer.py +hparams_search=optuna hydra.sweeper.n_trials=5
```

For the folds, run `conll_folds.ipynb` from its own directory, then `bash bash_scripts/conll/5_fold.sh` from
this folder, then `ensemble_test.ipynb`.

## Notes

- `metric_level` (model setting) picks the F1 that checkpointing and Optuna use: `token` (default, `val_avg_f1`, credits partly found entities) or `entity` (`val_entity_f1`, exact CoNLL). Both are logged.
- With the default `IOB: 0` the model predicts IO tags, so two adjacent entities of the same type can't be told apart, and entity F1 is computed against IO-converted gold labels (slightly optimistic). Set `IOB: 1` for exact CoNLL F1.
- The default `batch_size` is 1 (slow, but it fits any GPU); the sweep tries 1, 2 and 4.
