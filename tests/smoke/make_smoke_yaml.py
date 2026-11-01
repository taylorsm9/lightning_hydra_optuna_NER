"""Write a copy of a LightningCLI-style yaml (folders 01/02) set up for a tiny smoke run.

Usage: make_smoke_yaml.py <source yaml> <out dir> <data dir> <cpu|gpu> [model_class]
"""
import os
import sys

import yaml

src, out, data, device = sys.argv[1:5]
model_class = sys.argv[5] if len(sys.argv) > 5 else ""
os.makedirs(out, exist_ok=True)

with open(src) as f:
    cfg = yaml.safe_load(f)

cfg["trainer"].update(accelerator=device, devices=1, max_epochs=1, limit_train_batches=2, limit_val_batches=2,
                      num_sanity_val_steps=0, default_root_dir=out)
cfg["trainer"]["logger"]["init_args"]["save_dir"] = out
for cb in cfg["trainer"]["callbacks"]:
    if "ModelCheckpoint" in cb["class_path"]:
        cb["init_args"]["dirpath"] = f"{out}/checkpoints"
cfg["data"]["init_args"].update(batch_size=2, train_path=f"{data}/train.pkl", val_path=f"{data}/val.pkl",
                                test_path=f"{data}/test.pkl")
cfg["model"]["init_args"].update(pred_output_path=f"{out}/test_data.pkl",
                                 confusion_matrix_path=f"{out}/confusion_matrix.csv")
if model_class:
    cfg["data"]["init_args"]["model_class"] = model_class

with open(f"{out}/smoke.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
