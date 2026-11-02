import os
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from typing import List, Optional


def build_callbacks(callbacks_cfg: DictConfig) -> List:
    """Instantiates callbacks from config. Returns a list of 2 callback objects for use by the lightning trainer"""
    callbacks: List = []

    for _, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def best_val_score(trainer, monitor="val_score"):
    """Best value of the monitored metric over the run, as tracked by ModelCheckpoint or EarlyStopping
    (the last epoch's value if neither tracks it). Optuna should compare trials by their best epoch."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.monitor == monitor and callback.best_model_score is not None:
            return callback.best_model_score
        if isinstance(callback, EarlyStopping) and callback.monitor == monitor and torch.isfinite(callback.best_score):
            return callback.best_score
    return trainer.callback_metrics.get(monitor, None)


@hydra.main(version_base=None, config_path="config", config_name="conll_config")
def train(cfg: DictConfig) -> Optional[float]:

    # Set seed
    pl.seed_everything(42, workers=True)

    # Initialize lightning datamodule and model with hydra configs
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    # Get callbacks and initialize trainer
    callbacks = build_callbacks(cfg.callbacks)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train
    trainer.fit(model=model, datamodule=datamodule)

    # Best validation score of the run (model.metric_level picks token or entity F1), for hyperparameter optimization
    score = best_val_score(trainer)

    # Optionally evaluate the best checkpoint on the test set (off for sweeps; 5_fold.sh turns it on)
    if cfg.get("test_after_fit"):
        best = trainer.checkpoint_callback.best_model_path
        if os.path.isdir(best):
            # DeepSpeed ZeRO-3 saves a directory of shards, which Lightning can't restore for testing (DeepSpeed's
            # inference engine has no _restore_from_bit16_weights). Merge the shards into fp32 weights, load them
            # into a fresh model and test without DeepSpeed; inference fits on one GPU.
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(best)
            state_dict = {k.removeprefix("_forward_module."): v for k, v in state_dict.items()}
            state_dict.pop("criterion.weight", None)  # the loss is rebuilt in setup()
            model = hydra.utils.instantiate(cfg.model)
            model.load_state_dict(state_dict, strict=False)
            trainer = hydra.utils.instantiate(cfg.trainer, strategy="auto", callbacks=[], logger=logger)
            trainer.test(model=model, datamodule=datamodule)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best)

    return score

if __name__ == "__main__":
    train()