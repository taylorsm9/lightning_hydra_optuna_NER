# lightning_cli_ner

> **Historical teaching project, archived, not maintained.**
> Each folder is a different way to train the same CoNLL-2003 named-entity-recognition model
> (ModernBERT-large or DeBERTa-v3-large) with PyTorch Lightning.

This started as a single private repo where each git branch tried a different training setup:
LightningCLI, a hand-built Trainer, Hydra configs, Optuna sweeps, a 5-fold ensemble, DeepSpeed.
It was written in 2025 for students in a university NLP lab who were learning these tools. For this
public version each branch has been copied into its own folder, so you can read and run the setups
side by side without digging through git history. The original branches are kept as `archive/<branch>` tags.

The code in each folder is **kept as it was**, rough edges included. The only changes are the ones
needed to make it run today (data paths, a few crash fixes), and each folder's README lists them.
It's a record of how the systems were built, not a polished library.

## What's in it

Beyond the different training setups, every folder shares the same model and tooling:

**Handling NER's long-tailed labels.** Most tokens are `O`, and some entity types are rare. Several
loss-side options deal with that:
- **Class-balanced weights** from the "effective number of samples"
  ([Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555)). Cui et al. proposed this weighting for
  long-tailed image classification. We adapted it to token-level NER, where the `O` class dominates, in
  March 2024, in an earlier group project on the ChEMU, TAC and CoNLL-2003 datasets.
  Each entity class gets the weight `(1 - β) / (1 - βⁿ)`, where `n` is its token count in the training
  set and `β` (default 0.99) is a hyperparameter. As `β` goes from 0 to 1 the weights shift from uniform
  toward inverse frequency, so `β` tunes how hard rare classes are boosted. `O` gets the smallest entity
  weight. Turn it on with `effective_sample_weights: true` (`compute_class_weights` in
  `src/utilities/data_preprocessing_functions.py`).
- **`o_class_mod`** divides the `O` weight further, with or without class-balanced weights.
- **Learned per-class logit scaling.** A trainable scale per class is added (as a log) to the logits
  inside the loss, so the model learns its own adjustment for the class priors.
- Label smoothing and loss flooding ([Ishida et al., 2020](https://arxiv.org/abs/2002.08709)) against overconfidence and overfitting.

**Experiment tracking with Weights & Biases.** Every run logs training loss and learning rate per
step, and per-class precision, recall and F1 each validation epoch. It also logs a confusion matrix
image every epoch, which `notebooks/confusion_matrix/confusion_matrix_gif.ipynb` turns into an animation
of training. 05 also uploads each run's resolved Hydra config to W&B and groups Optuna trials in their own project.

**Configuration and overrides.** Every setting lives in config, not code, and can be overridden for one
run without editing a file. In 01 that's LightningCLI flags (`--trainer.max_epochs 5`), with shared
arguments linked so `batch_size` or `model_class` are set once for both data and model. In 03-05 it's
Hydra config groups and overrides (`trainer.max_epochs=5`, `+hparams_search=optuna`). Both save the
resolved config with every run.

**Automated hyperparameter tuning with Optuna.** 03 and 04 run a TPE search through the Hydra Optuna
sweeper: learning rates, weight decay, dropout, batch size, epochs. 05 drives Optuna from its own
script, adds `β` and `o_class_mod` to the search, and uses early stopping within each trial.

Training uses a one-cycle learning-rate schedule, bf16 mixed precision, gradient clipping, and either
ModernBERT-large or DeBERTa-v3-large as the encoder (`model_class`).

## The folders

| Folder | What it shows | Entry point | Config | From |
|---|---|---|---|---|
| [`01_lightning_cli/`](01_lightning_cli/) | LightningCLI builds everything from one yaml | `main.py fit` | `config/conll_2003/conll.yaml` (`class_path`/`init_args`) | branch `master` |
| [`02_plain_trainer/`](02_plain_trainer/) | The same yaml, but the model, datamodule, callbacks and `Trainer` are built by hand | `trainer.py` | same yaml, read with `yaml.safe_load` | branch `lightning_trainer` |
| [`03_hydra_optuna_kfold/`](03_hydra_optuna_kfold/) | Hydra config groups, an Optuna sweep through the Hydra sweeper, 5-fold cross-validation and an ensemble notebook | `trainer.py` | `config/` groups | branch `lightning_hydra_ensemble` |
| [`04_hydra_deepspeed/`](04_hydra_deepspeed/) | Hydra plus DeepSpeed ZeRO stage 3 with CPU offload, and dynamic padding | `trainer.py` | `config/` groups | branch `lightning_hydra` |
| [`05_hydra_optuna_runner/`](05_hydra_optuna_runner/) | Separate `train.py`/`test.py`, Optuna driven from a plain Python script instead of the Hydra sweeper, gradient checkpointing, F1 without true-O tokens | `train.py`, `test.py`, `optuna_runner.py` | `config/` groups | uncommitted local work, April 2025 |

The same model code (`src/`) is copied into every folder on purpose, so each one stands alone. Diffing
two folders shows exactly what a setup changed:

```bash
diff -r 03_hydra_optuna_kfold/src 04_hydra_deepspeed/src      # what DeepSpeed and dynamic padding needed
diff -r 04_hydra_deepspeed/config 05_hydra_optuna_runner/config
```

### Same run, five ways

Each of these trains on CoNLL-2003 with that folder's defaults (ModernBERT-large, except 05, whose model config
says DeBERTa-v3-large; see its README). Run them from inside the folder.

```bash
cd 01_lightning_cli       && python main.py fit --config config/conll_2003/conll.yaml
cd 02_plain_trainer       && python trainer.py
cd 03_hydra_optuna_kfold  && python trainer.py                      # add +hparams_search=optuna to sweep
cd 04_hydra_deepspeed     && python trainer.py
cd 05_hydra_optuna_runner && python train.py                        # or: python optuna_runner.py
```

Overriding a setting looks different in each one, which is much of the point:

```bash
python main.py fit --config config/conll_2003/conll.yaml --trainer.max_epochs 5     # 01: CLI flags
python trainer.py my_copy_of_conll.yaml                                             # 02: edit (a copy of) the yaml
python trainer.py trainer.max_epochs=5 datamodule.batch_size=8                      # 03-05: Hydra overrides
```

| | LightningCLI (01) | Plain Trainer (02) | Hydra (03-05) |
|---|---|---|---|
| Boilerplate | Almost none | Every object built and wired by hand | A small `@hydra.main` script plus `instantiate` |
| Overrides | `--section.key value` | Edit the file (or pass another yaml) | `group.key=value`, add with `+`, remove with `~` |
| Config saved with the run | Yes (`config.yaml` next to the logs) | No | Yes (`.hydra/` in each run directory) |
| Sweeps | Not built in | Not built in | Optuna sweeper plugin (03, 04) or a script (05) |
| Good for | Getting started, one model, one config | Seeing what the CLI does for you | Many experiments, config groups, sweeps |

## Setup

Tested in November 2026 on Python 3.12 with an RTX 4090 and the package versions the project used in 2025.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA at pytorch.org
pip install -r requirements.txt
python scripts/get_conll.py          # writes datasets/conll_2003/
```

- **Weights & Biases.** Every folder logs to W&B. Either run `wandb login`, or set `WANDB_MODE=offline` to log locally.
- **DeepSpeed (04, 05).** DeepSpeed compiles its CPU Adam optimizer the first time you run. It refuses to
  if your installed CUDA toolkit (`nvcc`) has a different major version than the one torch was built with.
  If you see that error, `DS_SKIP_CUDA_CHECK=1` worked for us. Keep `deepspeed==0.16.4`: 0.19.9 crashes
  under ZeRO-3 with this model.
- **CPU only.** The full runs need a GPU. The smoke tests below run on CPU.

## The data

CoNLL-2003 is derived from the Reuters RCV1 corpus and is **not redistributed in this repo**. See the
[shared task page](https://www.clips.uantwerpen.be/conll2003/ner/) for its terms. `scripts/get_conll.py`
downloads the public copy on the Hugging Face Hub
([`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003)) and writes:

- `datasets/conll_2003/{train,val,test}.conll`: tab-separated tag, `1`, `1`, word (IOB2 tags), with a blank line between sentences
- `datasets/conll_2003/{train,val,test}.pkl`: one row per sentence, list columns `Word` and `Label`, which is what the datamodules read

The output is identical to the files the project trained on in 2025, including two quirks of the
original preprocessing: the words `NA` and `None` are missing from 14 sentences (pandas read them as
missing values), and sentences are sorted by their string id. The notebook that made them is in each
folder's `notebooks/dataset_preprocessing/preprocess_conll.ipynb`.

## Tests

```bash
pytest tests/                                                     # label alignment for both tokenizers
bash tests/smoke/run_smoke.sh                                     # tiny CPU run of every folder
SMOKE_DEVICE=gpu bash tests/smoke/run_smoke.sh 04 05              # with each folder's real GPU/DeepSpeed settings
```

The smoke runs use 20 hand-written sentences (`tests/fixtures/tiny_ner.conll`), so they need no data
download, only the encoder weights from the Hugging Face Hub. On CPU, 04 and 05 drop the DeepSpeed strategy
and swap `DeepSpeedCPUAdam` for `AdamW`. On GPU everything runs as configured.

## Known issues

These are **not fixed**, on purpose. The code is kept as it was, and a later review will decide what to
fix. Each folder's README has its own list.

- **Scores are below published results.** A 5-epoch run of 01 reaches entity-level F1 91.8 on val and
  87.6 on test, while large encoders typically reach about 92-93 on test. Suspected causes: the whole
  model is cast to bfloat16 and trained that way, and in 01-03 padding tokens are labelled `O` and counted in the loss.
- **The built-in metrics aren't CoNLL F1.** They are computed per subword token, include the `O` class,
  and in 01-03 include padding, so they read much higher than standard entity-level F1.
- With `is_split_into_words=True`, ModernBERT's tokenizer splits each word as if it had no leading space,
  so the model sees word pieces it rarely saw in pretraining.
- `test_dataloader` uses `drop_last=True`, so a few test sentences are skipped.

## Credits

Written by scott ([@taylorsm9](https://github.com/taylorsm9)). Charles wrote the original
datamodule, Charles and Osmon worked on earlier versions of the model and preprocessing, and Ethan Leyden
refactored some of the preprocessing functions.

An unfinished experiment with a second dataset (GutBrainIE) is not included. It survives only in the git
history and the `archive/gutbrain_dataloader` tag.

## License

[MIT](LICENSE)
