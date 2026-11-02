import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from pathlib import Path

from train import log_hydra_config, build_callbacks

"""
USAGE
python test.py +checkpoint_path=logs/checkpoints/last.ckpt
The checkpoint can be a Lightning .ckpt file, a plain state dict (pytorch_model.bin), or a DeepSpeed checkpoint
directory, whose shards are merged into fp32 weights first (as zero_to_fp32.py in that directory would).
Testing runs without DeepSpeed; inference fits on one GPU.

"""
def _load_model(cfg: DictConfig, ckpt: Path):
    """Instantiate the model and load the checkpoint's weights into it."""
    import torch
    model = hydra.utils.instantiate(cfg.model)

    if ckpt.is_dir():
        # Lightning can't restore a DeepSpeed ZeRO-3 directory for testing (DeepSpeed's inference engine has no
        # _restore_from_bit16_weights), so merge the shards into fp32 weights
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ckpt))
        state_dict = {k.removeprefix("_forward_module."): v for k, v in state_dict.items()}
    else:
        # A Lightning checkpoint keeps the weights under "state_dict"; a plain state dict is the weights.
        # (load_from_checkpoint can't be used: the model doesn't save its constructor arguments.)
        state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state_dict = state_dict.get("state_dict", state_dict)
    state_dict.pop("criterion.weight", None)  # the loss is rebuilt in setup() from the datamodule's class weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[warning] missing keys:", missing)
    if unexpected:
        print("[warning] unexpected keys:", unexpected)
    return model


@hydra.main(version_base=None, config_path="config", config_name="conll_config")
def test(cfg: DictConfig) -> None:
    ckpt_input = cfg.get("checkpoint_path")
    if not ckpt_input:
        raise ValueError("Add +checkpoint_path=/path/to/ckpt or dir")
    ckpt_path = Path(ckpt_input).expanduser()

    pl.seed_everything(42, workers=True)

    # Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Model with the checkpoint's weights
    model = _load_model(cfg, ckpt_path)

    # Callbacks & logger
    callbacks = build_callbacks(cfg.callbacks)
    logger    = hydra.utils.instantiate(cfg.logger)

    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy="auto",  # no DeepSpeed for testing
        callbacks=callbacks,
        logger=logger,
    )

    log_hydra_config(logger)

    # Test
    results = trainer.test(model=model, datamodule=datamodule)

    if results:
        print("Test metrics:")
        for k, v in results[0].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test()