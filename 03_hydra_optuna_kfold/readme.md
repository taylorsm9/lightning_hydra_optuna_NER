# 03 · Hydra + Optuna sweeper + 5-fold ensemble

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from branch `lightning_hydra_ensemble` (also tag `archive/lightning_hydra_ensemble`).
The code is kept as it was. Only the changes listed below were made.

## What it shows

The yaml is split into Hydra config groups (`config/model`, `datamodule`, `trainer`, `callbacks`,
`logger`) composed by `config/conll_config.yaml`. `trainer.py` builds every object with
`hydra.utils.instantiate` and returns `val_avg_f1`, which is what lets the Optuna sweeper plugin use it
as an objective (`config/hparams_search/optuna.yaml`).

It also has the start of a 5-fold ensemble:
- `notebooks/dataset_preprocessing/conll_folds.ipynb` merges train and val and writes 5 folds
- `bash_scripts/conll/5_fold.sh` trains one model per fold
- `notebooks/testing/ensemble_test.ipynb` averages the models' probabilities, with weights tuned by Optuna

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py
python trainer.py trainer.max_epochs=5 datamodule.batch_size=8 model.batch_size=8
python trainer.py +hparams_search=optuna                     # 20-trial sweep; results in logs/multirun/
python trainer.py +hparams_search=optuna hydra.sweeper.n_trials=5
```

For the folds, create `../datasets/conll_2003/5_fold/`, run `conll_folds.ipynb` from its own directory,
then run `bash bash_scripts/conll/5_fold.sh` from this folder.

## Changes from the original

- Data paths point at `../datasets/conll_2003`, which `scripts/get_conll.py` writes.
- **Crash fix, label alignment.** `expand_encode_labels` matched subword tokens to words by counting tokens whose character offset starts at 0. DeBERTa-v3 sometimes emits a standalone `▁` token that also starts at 0, so the count ran past the end of the sentence and raised `IndexError` (11 CoNLL sentences, e.g. ones containing `Gurion` or `bacillus`). Labels now come from the tokenizer's `word_ids()`. On every sentence that didn't crash, the labels are identical to before.
- **Crash fix, metrics with `IOB: 1`.** The torchmetrics objects were sized with the raw `num_labels` (5) instead of the IOB-adjusted count (9), which crashed validation. With the default `IOB: 0` nothing changes.
- **Crash fix, saving test predictions.** `test_preds/` is created if it doesn't exist. The original repo had it only because prediction files were committed.
- `conll_folds.ipynb` and `5_fold.sh` use the new data path.

## Known issues (not fixed)

Kept on purpose, pending a later review.

- Every sentence is padded to 512 tokens (the `max_length` setting is ignored) and the padding is labelled `O`. Nothing marks it to be ignored, so most of each loss computation is padding, which acts like a large hidden weight on `O`.
- Validation and test metrics count those padding tokens as correct `O` predictions, so `val_avg_f1` (which checkpointing optimizes) reads high.
- The whole encoder is cast to bfloat16 (`.to(torch.bfloat16)`) and trained that way. bf16 holds about 3 significant digits, so many small AdamW updates may round away. Suspected cause of the below-published scores.
- Metrics are per subword token and include `O`, so they aren't comparable with entity-level CoNLL F1. The two encoders split words differently, so their scores aren't comparable with each other either.
- With `is_split_into_words=True`, ModernBERT tokenizes each word without its leading space, so it sees word pieces it rarely saw in pretraining.
- `remove_o: true` does nothing: `make_tensor` discards the result of `remove_only_o_sentences`.
- `test_dataloader` uses `drop_last=True`, so the last partial batch of test sentences is skipped.
- The DataLoaders hardcode `num_workers=8, persistent_workers=True`. Once during testing, the workers deadlocked in the sanity check; rerunning worked.
- The ensemble isn't wired up end to end. `trainer.py` only trains (never tests), so no fold writes predictions, and `ensemble_test.ipynb` reads `logs/conll/test_preds/val_data_{i}.pkl`, which nothing writes.
- `conll_folds.ipynb` doesn't create the `5_fold/` directory it saves into.
- The default `batch_size` is 1 (set in both `datamodule` and `model`; change both).
- `bash_scripts/conll/conll_baseline.sh` still calls 01's `main.py`.
