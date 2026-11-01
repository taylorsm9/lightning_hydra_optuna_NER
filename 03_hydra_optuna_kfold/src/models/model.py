from lightning.pytorch import LightningModule, callbacks
from transformers import AutoConfig, DebertaV2Model, ModernBertConfig, ModernBertModel

import torchmetrics

import torch
import torch.nn as nn
import pandas as pd

import wandb
import matplotlib.pyplot
matplotlib.use('Agg')


class DeBERTaV3LargeModel(nn.Module):
    def __init__(self, num_labels, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.1):
        super(DeBERTaV3LargeModel, self).__init__()
        configuration = AutoConfig.from_pretrained('microsoft/deberta-v3-large')

        configuration.hidden_dropout_prob = hidden_dropout_prob
        configuration.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large', config=configuration)
        self.linear = nn.Linear(self.deberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state)
        return logits

class ModernBertLargeModel(nn.Module):
    def __init__(self, num_labels, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.1):
        super(ModernBertLargeModel, self).__init__()
        configuration = AutoConfig.from_pretrained('answerdotai/ModernBERT-large')

        configuration.hidden_dropout_prob = hidden_dropout_prob
        configuration.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.modern_bert = ModernBertModel.from_pretrained('answerdotai/ModernBERT-large', config=configuration)
        self.linear = nn.Linear(self.modern_bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.modern_bert(input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state)
        return logits

class EncoderNERModel(LightningModule):
    def __init__(self, IOB, num_labels, batch_size, starting_lr, max_lr, weight_decay, pct_start, flooding_val,
                 pred_output_path, confusion_matrix_path, hidden_dropout_prob,
                 attention_probs_dropout_prob, model_class):
        super().__init__()
        self.IOB = IOB
        self.num_labels = num_labels + ((num_labels - 1) * self.IOB)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.model = None
        self.batch_size = batch_size
        self.starting_lr = starting_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.pct_start = pct_start
        self.flooding_val = float(flooding_val)
        self.class_weights = None
        self.test_data = []
        self.pred_output_path = pred_output_path
        self.criterion = None

        if model_class == "DeBERTaV3LargeModel":
            self.model = DeBERTaV3LargeModel(self.num_labels, self.hidden_dropout_prob,
                                             self.attention_probs_dropout_prob).to(torch.bfloat16)
        elif model_class == "ModernBertLargeModel":
            self.model = ModernBertLargeModel(self.num_labels, self.hidden_dropout_prob,
                                             self.attention_probs_dropout_prob).to(torch.bfloat16)

        # removed because we never looked at this
        # self.confusion_matrix_path = confusion_matrix_path

        # gradient accumulation was not used so this was removed
        # accumulation schedule is a dict or a list of dicts, so this has to be done outside YAML in a custom trainer
        # accumulation schedule has to match trainer schedule
        # a schedule could be inherited by model in the same way model inherits from dataModule
        # self.accumulation_schedule = accumulation_schedule

        # logit scaling, learned
        initial_scaling_o = 1.0  # option to change initial scaling for "O" class
        initial_scaling_other = 1.0  # Initial scaling for other classes
        initial_scalings = torch.full((self.num_labels,), initial_scaling_other)  # Comma denotes tuple,
        initial_scalings[0] = initial_scaling_o  # "O" class is at index 0
        self.logit_scaling = nn.Parameter(initial_scalings, requires_grad=True)

        # Metrics for training
        # these can be commented out if not troubleshooting performance
        # self.train_f1 = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average=None)
        # self.train_f1_micro = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average='micro')
        # self.train_precision = torchmetrics.Precision(num_classes=num_labels, task="multiclass", average=None)
        # self.train_recall = torchmetrics.Recall(num_classes=num_labels, task="multiclass", average=None)

        # Metrics for validation
        self.val_f1 = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average=None)
        self.val_f1_micro = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average='micro')
        self.val_precision = torchmetrics.Precision(num_classes=num_labels, task="multiclass", average=None)
        self.val_recall = torchmetrics.Recall(num_classes=num_labels, task="multiclass", average=None)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_labels, task="multiclass",
                                                                 normalize='true')

        # Metrics for testing
        self.test_f1 = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average=None)
        self.test_f1_micro = torchmetrics.F1Score(num_classes=num_labels, task="multiclass", average='micro')
        self.test_precision = torchmetrics.Precision(num_classes=num_labels, task="multiclass", average=None)
        self.test_recall = torchmetrics.Recall(num_classes=num_labels, task="multiclass", average=None)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_labels, task="multiclass",
                                                                  normalize='true')

    def setup(self, stage=None):
        self.class_weights = self.trainer.datamodule.class_weights.clone().detach()
        print("class weights:")
        print(self.class_weights)
        # now declare criterion with correct weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean', ignore_index=-100,
                                             label_smoothing=0.1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.starting_lr,
                                      weight_decay=self.weight_decay, )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr,
                                                        total_steps=self.trainer.estimated_stepping_batches,
                                                        pct_start=self.pct_start)
        # return optimizer
        return [optimizer], [{'scheduler': scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask):
        model_class: ModernBertLargeModel #DeBERTaV3LargeModel,ModernBertLargeModel
        logits = self.model(input_ids, attention_mask)
        return logits

    def custom_loss_function(self, logits, labels, train=True):
        # apply logit scaling
        logits = logits + torch.log(self.logit_scaling ** 1 + 1e-12)

        # get loss
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

        # b is a flooding value, ensuring loss doesn't fall below this point to prevent overfitting
        # in practice this value is so low it never happens on ChEMU. Don't set it too high
        b = self.flooding_val
        if train:
            loss = (loss - b).abs() + b  # b is the flooding level.

        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        train_loss = self.custom_loss_function(logits, labels, train=True)

        # Update train metrics for troubleshooting
        # preds = torch.argmax(logits, dim=-1)
        # comment this out along with full on_train_epoch_end method if not troubleshooting performance
        # self.train_f1(preds, labels)
        # self.train_f1_micro(preds, labels)
        # self.train_precision(preds, labels)
        # self.train_recall(preds, labels)

        # unscaled loss for logging each batch's loss
        return {"loss": train_loss, 'unscaled_loss': train_loss}

    def on_train_batch_end(self, out, batch, batch_idx) -> None:
        # use unscaled loss
        self.log("train_loss", out['unscaled_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

    # def on_train_epoch_end(self, outputs=None):
    #     # Comment this full method out along with training_step logging if not looking for problems
    #     # Compute per-class metrics
    #     f1_per_class = self.train_f1.compute()
    #     precision_vals = self.train_precision.compute()
    #     recall_vals = self.train_recall.compute()
    #
    #     # Log training metrics
    #     avg_f1 = f1_per_class.mean()
    #     self.log('train_avg_f1', avg_f1, on_epoch=True, prog_bar=True)
    #     self.log('train_f1_micro', self.train_f1_micro.compute(), on_epoch=True, prog_bar=True)
    #
    #     # Log per class metrics
    #     for i, (f1_score, prec, rec) in enumerate(zip(f1_per_class, precision_vals, recall_vals)):
    #         self.log(f'train_f1_class_{i}', f1_score, on_epoch=True, prog_bar=True)
    #         self.log(f'train_precision_class_{i}', prec, on_epoch=True, prog_bar=True)
    #         self.log(f'train_recall_class_{i}', rec, on_epoch=True, prog_bar=True)
    #
    #     # Reset metrics
    #     self.train_f1.reset()
    #     self.train_f1_micro.reset()
    #     self.train_precision.reset()
    #     self.train_recall.reset()

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.custom_loss_function(logits, labels, train=False)

        # Update validation metrics
        preds = torch.argmax(logits, dim=-1)
        self.val_f1(preds, labels)
        self.val_f1_micro(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        return {"valid_loss": val_loss}

    def on_validation_epoch_end(self):
        # Compute per-class F1 at the end of epoch
        f1_per_class = self.val_f1.compute()
        precision_vals = self.val_precision.compute()
        recall_vals = self.val_recall.compute()

        # Log F1 scores
        avg_f1 = f1_per_class.mean()

        self.log('val_avg_f1', avg_f1, on_epoch=True, prog_bar=True)
        self.log('valid_f1', self.val_f1_micro.compute(), on_epoch=True, prog_bar=True)

        # Log per class f1, precision, recall
        for i, (f1_score, prec, rec) in enumerate(zip(f1_per_class, precision_vals, recall_vals)):
            self.log(f'val_f1_class_{i}', f1_score, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'val_precision_class_{i}', prec, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'val_recall_class_{i}', rec, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log confusion matrix at the end of epoch
        fig, ax = self.val_confusion_matrix.plot(add_text=False)
        wandb.log({f'val_confusion_matrix': [wandb.Image(fig)]})
        # self.log_confusion_matrix(fig, ax, 'valid')
        matplotlib.pyplot.close(fig)
        self.val_f1.reset()
        self.val_f1_micro.reset()
        self.val_confusion_matrix.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        active_labels = labels.view(-1)[attention_mask.view(-1) == 1]
        active_preds = preds.view(-1)[attention_mask.view(-1) == 1]
        active_input_ids = input_ids.view(-1)[attention_mask.view(-1) == 1]
        active_logits = logits.view(-1, logits.size(-1))[attention_mask.view(-1) == 1] # For postprocessing
        active_probs = torch.nn.functional.softmax(active_logits, dim=-1) # For postprocessing

        self.test_f1.update(active_preds, active_labels)
        self.test_f1_micro.update(active_preds, active_labels)
        self.test_precision.update(active_preds, active_labels)
        self.test_recall.update(active_preds, active_labels)

        self.test_confusion_matrix.update(active_preds, active_labels)

        # Save active input IDs, predictions, and labels for detokenization and results examination
        self.test_data.extend(
            zip(active_input_ids.cpu().tolist(),
                active_preds.cpu().tolist(),
                active_labels.cpu().tolist(),
                active_probs.cpu().tolist())) # For postprocessing
    def on_test_epoch_end(self):
        # Compute per-class F1 scores
        f1_per_class = self.test_f1.compute()
        precision_vals = self.test_precision.compute()
        recall_vals = self.test_recall.compute()

        # Log scores
        self.log('test_avg_f1', f1_per_class.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1_micro', self.test_f1_micro.compute(), on_epoch=True, prog_bar=True, logger=True)

        # Log per class F1 score, precision, and recall
        for i, (f1_score, prec, rec) in enumerate(zip(f1_per_class, precision_vals, recall_vals)):
            self.log(f'test_f1_class_{i}', f1_score, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'test_precision_class_{i}', prec, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'test_recall_class_{i}', rec, on_step=False, on_epoch=True, prog_bar=True)

        # log confusion matrix for wandb heatmap
        fig, ax = self.test_confusion_matrix.plot(add_text=False)
        wandb.log({f'test_confusion_matrix': [wandb.Image(fig)]})
        matplotlib.pyplot.close(fig)

        # separately, log confusion matrix to a csv to see exact normalized values if need be
        # removed because we never needed this
        # cm_tensor = self.test_confusion_matrix.compute()
        # cm_np = cm_tensor.cpu().numpy()
        # df_cm = pd.DataFrame(cm_np)
        # df_cm.to_csv(self.confusion_matrix_path, index=False)

        # Reset metrics
        self.test_f1.reset()
        self.test_f1_micro.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_confusion_matrix.reset()

        if self.test_data:  # Ensure there's data to save
            df = pd.DataFrame(self.test_data, columns=['InputID', 'Prediction', 'Label', 'Probabilities'])
            num_classes = len(df['Probabilities'].iloc[0])
            df.to_pickle(self.pred_output_path)
            self.test_data.clear()  # Clear the list

    def on_train_end(self):
        model_checkpoint = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, callbacks.ModelCheckpoint):
                model_checkpoint = callback
                break

        if model_checkpoint:
            print(f"Last checkpoint path: {model_checkpoint.last_model_path}")
        else:
            print("ModelCheckpoint callback not found.")
