import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

from src.models.model import EncoderNERModel
from src.datamodules.CONLL_data_module import CONLLDataModule

if __name__ == "__main__":

    with open('config/conll_2003/conll.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']['init_args']
    data_config = config['data']['init_args']
    trainer_config = config['trainer']

    # Handle trainer callbacks
    callbacks = []
    # ModelCheckpoint callback
    checkpoint_callback_config = next(
        callback['init_args'] for callback in trainer_config['callbacks']
        if 'ModelCheckpoint' in callback['class_path']
    )
    callbacks.append(ModelCheckpoint(**checkpoint_callback_config))

    # LearningRateMonitor callback
    lr_callback_config = next(
        callback['init_args'] for callback in trainer_config['callbacks']
        if 'LearningRateMonitor' in callback['class_path']
    )
    callbacks.append(LearningRateMonitor(**lr_callback_config))

    logger_config = trainer_config['logger']['init_args']
    logger = WandbLogger(**logger_config)

    # Remove nested configurations that are handled separately
    trainer_params = trainer_config.copy()
    trainer_params.pop('callbacks', None)
    trainer_params.pop('logger', None)

    model_params = {
            'IOB': data_config['IOB'],
            'num_labels': data_config['num_labels'],
            'batch_size': data_config['batch_size'],
            'model_class': data_config['model_class'],
            'starting_lr': float(model_config['starting_lr']),  # Convert to float
            'max_lr': float(model_config['max_lr']),           # Convert to float
            'weight_decay': float(model_config['weight_decay']), # Convert to float
            'pct_start': float(model_config['pct_start']),     # Convert to float
            'flooding_val': float(model_config['flooding_val']), # Convert to float
            'pred_output_path': model_config['pred_output_path'],
            'confusion_matrix_path': model_config['confusion_matrix_path'],
            'hidden_dropout_prob': float(model_config['hidden_dropout_prob']),     # Convert to float
            'attention_probs_dropout_prob': float(model_config['attention_probs_dropout_prob']), # Convert to float
        }

    datamodule_params = {
        'train_path': data_config['train_path'],
        'val_path': data_config['val_path'],
        'test_path': data_config['test_path'],
        'batch_size': data_config['batch_size'],
        'num_labels': data_config['num_labels'],
        'IOB': data_config['IOB'],
        'remove_o': data_config['remove_o'],
        'o_class_mod': data_config['o_class_mod'],
        'effective_sample_weights': data_config['effective_sample_weights'],
        'beta': data_config['beta'],
        'model_class': data_config['model_class'],
        'max_length': data_config['max_length']
    }

    model = EncoderNERModel(**model_params)
    datamodule = CONLLDataModule(**datamodule_params)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_params
    )

    trainer.fit(model, datamodule)

    # TODO args for fit vs test, accept checkpoint in args, arg for config location
    # We will never do this here, just leaving the note for the hydra version