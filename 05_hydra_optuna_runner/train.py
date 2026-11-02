import hydra
from omegaconf import DictConfig, ListConfig
import lightning.pytorch as pl
from pathlib import Path
from typing import Optional
from hydra.core.hydra_config import HydraConfig

def log_hydra_config(logger):
    """Place .hydra/config.yaml in the *Files* tab of the current W&B run."""

    run = logger.experiment

    # File Hydra already wrote
    trial_dir = Path(HydraConfig.get().runtime.output_dir)
    cfg_path  = trial_dir / ".hydra" / "config.yaml"

    if not cfg_path.exists():
        print(f"Hydra config not found at: {cfg_path}")
        return

    # Save to Files tab
    run.save(
        str(cfg_path),
        base_path=str(trial_dir),
        policy="now",
    )

def build_callbacks(cb_cfg: DictConfig | ListConfig):
    callbacks = []
    for _, conf in cb_cfg.items():
        # Skip keys like _delete_: true or anything not a mapping
        if not isinstance(conf, DictConfig):
            continue
        if "_target_" in conf:
            callbacks.append(hydra.utils.instantiate(conf))
    return callbacks

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

    # Save config to wandb
    log_hydra_config(logger)

    # Train
    trainer.fit(model=model, datamodule=datamodule)

    # Get entity-level F1 for hyperparameter tuning and print for Optuna
    score = trainer.callback_metrics.get("val_entity_f1", None)
    score = float(score)
    print(f"Last Epoch Val Score: {score}")
    return score

if __name__ == "__main__":
    train()