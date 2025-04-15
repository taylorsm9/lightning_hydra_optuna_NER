import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from typing import List, Optional


def build_callbacks(callbacks_cfg: DictConfig) -> List:
    """Instantiates callbacks from config. Returns a list of 2 callback objects for use by the lightning trainer"""
    callbacks: List = []

    for _, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="gutbrain_config")
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

    # Get macro f1 for hyperparameter tuning with optuna
    score = trainer.callback_metrics.get("val_avg_f1", None)

    return score

if __name__ == "__main__":
    train()