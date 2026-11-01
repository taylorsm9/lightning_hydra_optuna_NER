# Example usage:
# python main.py fit --config ChEMU.yaml
# python test --config ChEMU.yaml --ckpt_path checkpoints/last-v72.ckpt


import os
from lightning.pytorch.cli import LightningCLI, LightningDataModule
import torch
from lightning.fabric.utilities.seed import seed_everything
from src.datamodules.CONLL_data_module import CONLLDataModule


class CustomCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # data args are propagated to model args to simplify yaml changes
        parser.link_arguments("data.init_args.IOB", "model.init_args.IOB")
        parser.link_arguments("data.init_args.num_labels", "model.init_args.num_labels")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.init_args.model_class", "model.init_args.model_class")


    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        print("Current working directory: {0}".format(os.getcwd()))
        torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    print(f"Cuda Environment Check: {"Good" if torch.cuda.is_available() else "Environment Bad :("}")
    # accept any subclass of LightningDataModule, overwrite backup yaml. seed_everything_default sets workers=True
    # there are more reproducibility settings to add later (deterministic, isolate_rng, slows down training)
    cli = CustomCLI(datamodule_class=LightningDataModule, subclass_mode_data=True,
                    save_config_kwargs={"overwrite": True}, seed_everything_default=42)
