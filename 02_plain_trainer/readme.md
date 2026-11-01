# 02 · Plain Lightning Trainer

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from branch `lightning_trainer` (also tag `archive/lightning_trainer`).
The code is kept as it was. Only the changes listed below were made.

## What it shows

The same yaml as 01, but without LightningCLI. `trainer.py` reads it with `yaml.safe_load` and builds
every object by hand: it finds each callback's `init_args`, creates the `WandbLogger`, copies the shared
data arguments into the model arguments, converts the learning rates from strings to floats (`yaml.safe_load`
reads `1e-5` as a string), and passes what's left to `pl.Trainer`.

This shows what LightningCLI does for you, and why the project moved to Hydra next.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py                              # reads config/conll_2003/conll.yaml
python trainer.py path/to/another.yaml
```

Only training is wired up. There is no test command.

## Changes from the original

- Data paths point at `../datasets/conll_2003`, which `scripts/get_conll.py` writes.
- **Crash fix, label alignment.** `expand_encode_labels` matched subword tokens to words by counting tokens whose character offset starts at 0. DeBERTa-v3 sometimes emits a standalone `▁` token that also starts at 0, so the count ran past the end of the sentence and raised `IndexError` (11 CoNLL sentences, e.g. ones containing `Gurion` or `bacillus`). Labels now come from the tokenizer's `word_ids()`. On every sentence that didn't crash, the labels are identical to before.
- **Crash fix, metrics with `IOB: 1`.** The torchmetrics objects were sized with the raw `num_labels` (5) instead of the IOB-adjusted count (9), which crashed validation. With the default `IOB: 0` nothing changes.
- **Crash fix, saving test predictions.** `test_preds/` is created if it doesn't exist. The original repo had it only because prediction files were committed.
- `trainer.py` takes an optional yaml path as its first argument (the smoke tests use it). Without one it reads the original hardcoded path.

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
- `main.py` is a leftover copy of 01's LightningCLI entry point. The original readme below describes it, not `trainer.py`.
- The default `batch_size` is 1.
- The predictions file is named `test_data.csv` but written with `to_pickle`.
