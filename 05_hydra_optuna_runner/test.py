import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from pathlib import Path

from train import log_hydra_config, build_callbacks

"""
USAGE
python test.py +checkpoint_path=logs/checkpoints/last.ckpt/pytorch_model.bin
Note that when using deepspeed, you must run zero_to_fp32.py (in checkpoint dir) to make checkpoints into a single file

"""
def _load_model(cfg: DictConfig, ckpt: Path):
    """Instantiate the model and restore weights.

    Handles three cases:
    1. *ckpt* **directory** → return bare model, pass dir to Trainer so
       DeepSpeed/Lightning restore.
    2. *ckpt* **file** produced by *Lightning* (`*.ckpt`) → use
       `load_from_checkpoint`.
    3. *ckpt* **plain state‑dict* (`pytorch_model.bin`, `model.safetensors`, …)
       → `torch.load` + `model.load_state_dict`.
    """
    import torch
    model = hydra.utils.instantiate(cfg.model)

    if ckpt.is_dir():
        return model, ckpt  # Trainer will handle DeepSpeed restore

    # File path ------------------------------------------------------------
    try:
        # Case 2: Lightning .ckpt with metadata
        model_cls = model.__class__
        model = model_cls.load_from_checkpoint(str(ckpt), strict=False)
        return model, None
    except (KeyError, RuntimeError):
        # Fall back to plain state‑dict
        state_dict = torch.load(str(ckpt), map_location="cpu")
        # If it’s a Lightning‑style dict, strip the wrapper
        if "state_dict" in state_dict and all(not k.startswith("module.") for k in state_dict["state_dict"].keys()):
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[warning] missing keys:", missing)
        if unexpected:
            print("[warning] unexpected keys:", unexpected)
        return model, None


@hydra.main(version_base=None, config_path="config", config_name="conll_config")
def test(cfg: DictConfig) -> None:
    ckpt_input = cfg.get("checkpoint_path")
    if not ckpt_input:
        raise ValueError("Add +checkpoint_path=/path/to/ckpt or dir")
    ckpt_path = Path(ckpt_input).expanduser()

    pl.seed_everything(42, workers=True)

    # Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Model (handle file vs dir)
    model, trainer_ckpt = _load_model(cfg, ckpt_path)

    # Callbacks & logger
    callbacks = build_callbacks(cfg.callbacks)
    logger    = hydra.utils.instantiate(cfg.logger)

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log_hydra_config(logger)

    # Test
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path=str(trainer_ckpt) if trainer_ckpt else None)

    if results:
        print("Test metrics:")
        for k, v in results[0].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test()