# 05 · Hydra + separate train/test + Optuna runner

Part of [lightning_cli_ner](../README.md), an archived teaching project. Copied from uncommitted local work (April 2025). It was never on a branch.
The code is kept as it was. Only the changes listed below were made.

## What it shows

04's setup, as it stood in April 2025 in uncommitted work that never made it to a branch:

- **Separate `train.py` and `test.py`.** `test.py` takes `+checkpoint_path=` and accepts a Lightning
  `.ckpt`, a plain state dict, or (in principle) a DeepSpeed directory. Both upload the resolved Hydra config to W&B.
- **`optuna_runner.py`** drives Optuna from a plain script. Each trial runs `python train.py <overrides>`
  as a subprocess and reads the score from its stdout. That replaces the Hydra sweeper plugin, which is
  unmaintained and pins `optuna<3`.
- **Gradient checkpointing** as a model option (`use_gradient_checkpointing`).
- **`val_f1_micro_no_o`**, a micro-F1 that ignores tokens where both the label and the prediction are `O`.
  It's the checkpoint and Optuna objective here.

## How to run

From this folder, after the setup in the [root README](../README.md#setup):

```bash
python train.py
python optuna_runner.py                        # 50 trials; needs `python` on PATH to be this environment
python <ckpt dir>/zero_to_fp32.py <ckpt dir> <out dir>
python test.py +checkpoint_path=<out dir>/pytorch_model.bin
```

The DeepSpeed notes from 04 apply. `test.py` directly on a DeepSpeed checkpoint directory fails under
deepspeed 0.16.4 (`'DeepSpeedZeRoOffload' object has no attribute '_restore_from_bit16_weights'`).
Convert it first, as the docstring in `test.py` says.

## Changes from the original

- Data paths point at `../datasets/conll_2003`, which `scripts/get_conll.py` writes.
- **Crash fix, label alignment.** `expand_encode_labels` matched subword tokens to words by counting tokens whose character offset starts at 0. DeBERTa-v3 sometimes emits a standalone `▁` token that also starts at 0, so the count ran past the end of the sentence and raised `IndexError` (11 CoNLL sentences, e.g. ones containing `Gurion` or `bacillus`). Labels now come from the tokenizer's `word_ids()`. On every sentence that didn't crash, the labels are identical to before.
- **Crash fix, metrics with `IOB: 1`.** The torchmetrics objects were sized with the raw `num_labels` (5) instead of the IOB-adjusted count (9), which crashed validation. With the default `IOB: 0` nothing changes.
- **Crash fix, saving test predictions.** `test_preds/` is created if it doesn't exist. The original repo had it only because prediction files were committed.
- `bash_scripts/conll/conll_baseline.sh`: the notes about the university's shared GPU cluster and its partition name were removed (`YOUR_PARTITION` is a placeholder), throughout the git history as well.
- `train.py` and `test.py` load `conll_config` (they pointed at the GutBrain config).
- `config/model/conll_ner.yaml` gets `use_gradient_checkpointing: false`. The model requires the argument, and the config didn't set it.
- `5_fold.sh` and `conll_folds.ipynb` use the new data path.
- The W&B project, run and group names in `config/logger/` said `gutbrain`; they now say `conll`.
- Notebook outputs were stripped (the code was copied from a local checkout).

## Known issues (not fixed)

Kept on purpose, pending a later review.

- **Tokenizer mismatch.** `config/model` uses `DeBERTaV3LargeModel` but `config/datamodule` tokenizes for `ModernBertLargeModel`, so DeBERTa is fed ModernBERT token ids. Set both to the same `model_class`.
- `collate_fn` pads `input_ids` with 0. That's DeBERTa's pad id, but for ModernBERT 0 is a real token (the pad id is 50283). The attention mask hides it, so the effect is small.
- Metrics are masked by the attention mask, so `[CLS]`/`[SEP]` (labelled `O`) are still counted.
- The whole encoder is cast to bfloat16 (`.to(torch.bfloat16)`) and trained that way. bf16 holds about 3 significant digits, so many small AdamW updates may round away. Suspected cause of the below-published scores.
- Metrics are per subword token and include `O`, so they aren't comparable with entity-level CoNLL F1. The two encoders split words differently, so their scores aren't comparable with each other either.
- With `is_split_into_words=True`, ModernBERT tokenizes each word without its leading space, so it sees word pieces it rarely saw in pretraining.
- `remove_o: true` does nothing: `make_tensor` discards the result of `remove_only_o_sentences`.
- `test_dataloader` uses `drop_last=True`, so the last partial batch of test sentences is skipped.
- The DataLoaders hardcode `num_workers=8, persistent_workers=True`. Once during testing, the workers deadlocked in the sanity check; rerunning worked.
- `precision: 16-mixed` (fp16) here, while the model is cast to bf16.
- `optuna_runner.py` scores each trial by its last epoch, not its best.
- `bash_scripts/conll/5_fold.sh` calls `trainer.py`, which doesn't exist in this folder.
- `config/hparams_search/optuna_old.yaml` is a leftover of the Hydra-sweeper approach.
