# 04 · Hydra + DeepSpeed ZeRO-3

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from branch `lightning_hydra` (also tag `archive/lightning_hydra`).
The code is kept as it was. Only the changes listed below were made.

## What it shows

03's Hydra setup (without the folds) plus two memory-saving changes:

- **DeepSpeed ZeRO stage 3** with optimizer and parameter offload to the CPU (`config/trainer/default.yaml`),
  using `DeepSpeedCPUAdam` as the optimizer. It's set up to fit a larger batch (512) on one GPU.
- **Dynamic padding.** The datamodule's `collate_fn` pads each batch only to its longest sentence instead of
  padding everything to the full length, and pads labels with `-100` so the loss ignores them.
  Validation metrics are masked to real tokens too.

`diff -r ../03_hydra_optuna_kfold/src src` shows exactly what those took.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python trainer.py
DS_SKIP_CUDA_CHECK=1 python trainer.py         # if DeepSpeed refuses to compile CPU Adam (nvcc/torch CUDA mismatch)
python trainer.py +hparams_search=optuna
```

Needs a GPU and `deepspeed==0.16.4` (0.19.9 crashes under ZeRO-3 with this model). Checkpoints are
DeepSpeed directories. Convert one with `python <ckpt dir>/zero_to_fp32.py <ckpt dir> <out dir>`.

## Changes from the original

- Data paths point at `../datasets/conll_2003`, which `scripts/get_conll.py` writes.
- **Crash fix, label alignment.** `expand_encode_labels` matched subword tokens to words by counting tokens whose character offset starts at 0. DeBERTa-v3 sometimes emits a standalone `▁` token that also starts at 0, so the count ran past the end of the sentence and raised `IndexError` (11 CoNLL sentences, e.g. ones containing `Gurion` or `bacillus`). Labels now come from the tokenizer's `word_ids()`. On every sentence that didn't crash, the labels are identical to before.
- **Crash fix, metrics with `IOB: 1`.** The torchmetrics objects were sized with the raw `num_labels` (5) instead of the IOB-adjusted count (9), which crashed validation. With the default `IOB: 0` nothing changes.
- **Crash fix, saving test predictions.** `test_preds/` is created if it doesn't exist. The original repo had it only because prediction files were committed.
- `bash_scripts/conll/conll_baseline.sh`: the notes about the university's shared GPU cluster and its partition name were removed (`YOUR_PARTITION` is a placeholder), throughout the git history as well.

## Known issues (not fixed)

Kept on purpose, pending a later review.

- `collate_fn` pads `input_ids` with 0. That's DeBERTa's pad id, but for ModernBERT 0 is a real token (the pad id is 50283). The attention mask hides it, so the effect is small.
- Metrics are masked by the attention mask, so `[CLS]`/`[SEP]` (labelled `O`) are still counted.
- The whole encoder is cast to bfloat16 (`.to(torch.bfloat16)`) and trained that way. bf16 holds about 3 significant digits, so many small AdamW updates may round away. Suspected cause of the below-published scores.
- Metrics are per subword token and include `O`, so they aren't comparable with entity-level CoNLL F1. The two encoders split words differently, so their scores aren't comparable with each other either.
- With `is_split_into_words=True`, ModernBERT tokenizes each word without its leading space, so it sees word pieces it rarely saw in pretraining.
- `remove_o: true` does nothing: `make_tensor` discards the result of `remove_only_o_sentences`.
- `test_dataloader` uses `drop_last=True`, so the last partial batch of test sentences is skipped.
- The DataLoaders hardcode `num_workers=8, persistent_workers=True`. Once during testing, the workers deadlocked in the sanity check; rerunning worked.
- `model.py` imports `DeepSpeedCPUAdam` at the top and always uses it, so deepspeed must be installed and the DeepSpeed strategy kept, even to try this folder without DeepSpeed.
- Only training is wired up. There is no test command.
- The predictions file is named `test_data.csv` but written with `to_pickle`.
