#!/usr/bin/env bash
# Tiny CPU run of every folder: 1 epoch, 2 train / 2 val batches of the hand-written fixture.
# Only runtime settings are overridden (on the command line, or in a generated copy of the yaml);
# no code is changed. Everything a run writes goes to tests/.smoke_out/.
#
# Usage (from the repo root):  bash tests/smoke/run_smoke.sh [01 02 03 04 05]
#   PYTHON=/path/to/python            interpreter to use (default: python)
#   MODEL_CLASS=DeBERTaV3LargeModel   use the other encoder (default: each folder's own config)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python}"
DATA="$ROOT/tests/.smoke_data"
OUT="$ROOT/tests/.smoke_out"
export WANDB_MODE=offline WANDB_SILENT=true TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA" "$OUT"
"$PY" "$ROOT/tests/smoke/make_smoke_data.py" "$DATA"

HYDRA_COMMON=(trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
              +trainer.limit_train_batches=2 +trainer.limit_val_batches=2 trainer.num_sanity_val_steps=0
              datamodule.batch_size=2 model.batch_size=2 paths.data_dir="$DATA")
if [[ -n "${MODEL_CLASS:-}" ]]; then
  HYDRA_COMMON+=(datamodule.model_class="$MODEL_CLASS" model.model_class="$MODEL_CLASS")
fi

run_01() {
  cd "$ROOT/01_lightning_cli"
  "$PY" "$ROOT/tests/smoke/make_smoke_yaml.py" config/conll_2003/conll.yaml "$OUT/01" "$DATA" "${MODEL_CLASS:-}"
  "$PY" main.py fit --config "$OUT/01/smoke.yaml"
}

run_02() {
  cd "$ROOT/02_plain_trainer"
  "$PY" "$ROOT/tests/smoke/make_smoke_yaml.py" config/conll_2003/conll.yaml "$OUT/02" "$DATA" "${MODEL_CLASS:-}"
  "$PY" trainer.py "$OUT/02/smoke.yaml"
}

run_hydra() {  # $1 folder, $2 entry script (or launcher + entry script), rest: extra overrides
  local folder=$1 entry=$2; shift 2
  cd "$ROOT/$folder"
  "$PY" $entry "${HYDRA_COMMON[@]}" paths.log_dir="$OUT/$folder" paths.output_dir="$OUT/$folder/outputs" \
    hydra.run.dir="$OUT/$folder/hydra" "$@"
}

run_03() { run_hydra 03_hydra_optuna_kfold trainer.py; }
# The DeepSpeed pieces need a GPU: drop DeepSpeedStrategy and swap DeepSpeedCPUAdam for AdamW
# (see run_without_deepspeed_optim.py). Everything else in 04/05 runs as committed.
NO_DS="$ROOT/tests/smoke/run_without_deepspeed_optim.py"
run_04() { run_hydra 04_hydra_deepspeed "$NO_DS trainer.py" '~trainer.strategy'; }
run_05() { run_hydra 05_hydra_optuna_runner "$NO_DS train.py" '~trainer.strategy'; }

folders=("$@")
[[ ${#folders[@]} -eq 0 ]] && folders=(01 02 03 04 05)
status=0
for f in "${folders[@]}"; do
  rm -rf "${OUT:?}/$f"*
  if ( "run_$f" ) > "$OUT/$f.log" 2>&1; then
    echo "PASS $f ${MODEL_CLASS:-}"
  else
    echo "FAIL $f ${MODEL_CLASS:-} (log: tests/.smoke_out/$f.log)"; tail -25 "$OUT/$f.log"; status=1
  fi
done
exit $status
