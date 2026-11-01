# 01 · LightningCLI

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from branch `master` (also tag `archive/master`).
The code is kept as it was. Only the changes listed below were made.

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
python main.py test --config config/conll_2003/conll.yaml --ckpt_path logs/conll/checkpoints/last.ckpt
```

Checkpoints go to `logs/conll/checkpoints/`, test predictions to `logs/conll/test_preds/`.
`bash_scripts/conll/conll_baseline.sh` is the SLURM script used on the university cluster.

## Changes from the original

- Data paths point at `../datasets/conll_2003`, which `scripts/get_conll.py` writes.
- **Crash fix, label alignment.** `expand_encode_labels` matched subword tokens to words by counting tokens whose character offset starts at 0. DeBERTa-v3 sometimes emits a standalone `▁` token that also starts at 0, so the count ran past the end of the sentence and raised `IndexError` (11 CoNLL sentences, e.g. ones containing `Gurion` or `bacillus`). Labels now come from the tokenizer's `word_ids()`. On every sentence that didn't crash, the labels are identical to before.
- **Crash fix, metrics with `IOB: 1`.** The torchmetrics objects were sized with the raw `num_labels` (5) instead of the IOB-adjusted count (9), which crashed validation. With the default `IOB: 0` nothing changes.
- **Crash fix, saving test predictions.** `test_preds/` is created if it doesn't exist. The original repo had it only because prediction files were committed.

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
- Each run overwrites the committed `logs/conll/config.yaml` with the resolved config (`save_config_kwargs={"overwrite": True}`).
- The predictions file is named `test_data.csv` but written with `to_pickle`.
