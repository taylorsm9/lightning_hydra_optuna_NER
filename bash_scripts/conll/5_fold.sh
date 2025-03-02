#!/bin/bash

# Loop through folds 0-4
for fold in {0..4}; do
  echo "Training fold $fold..."

  python trainer.py \
    datamodule.train_path=datasets/conll_2003/5_fold/train_fold_$fold.pkl \
    datamodule.val_path=datasets/conll_2003/5_fold/val_fold_$fold.pkl \
    callbacks.model_checkpoint.dirpath=logs/conll/checkpoints/fold$fold \
    trainer.max_epochs=5

  echo "Finished training fold $fold"
  echo "------------------------"
done