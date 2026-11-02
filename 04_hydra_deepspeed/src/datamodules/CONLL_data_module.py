"""
ChEMU Data Module

Defines a PyTorch Lightning LightningDataModule for the ChEMU dataset.

The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU 
shared NER (Named Entity Recognition) task seeks to identify chemical compounds along with their roles in a reaction.
Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than 
classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and 
"reaction product." Our goal is to label words correctly as being members of these classes, in order to glean information 
from sets of patents that are too large for a human to read. 

Authors: Scott and Charles
Date: 10/19/2023
"""

from lightning.pytorch import LightningDataModule
from transformers import DebertaV2Tokenizer, AutoTokenizer
# from data_preprocessing_functions import *
from ..utilities.data_preprocessing_functions import *
import os

from torch.utils.data import DataLoader


class CONLLDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size, num_labels, IOB, remove_o,
                 o_class_mod, effective_sample_weights, beta, model_class, max_length, num_workers=8):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        # self.fill_context = fill_context
        self.remove_o = remove_o
        self.IOB = IOB
        self.num_labels = num_labels + ((num_labels - 1) * self.IOB)
        self.class_weights = []
        self.beta = beta # for effective sample weights
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
        self.o_class_mod = o_class_mod # ECW -130 epoch: 2 increased f1 by 2. 10 epoch schedule:16=8>2>32
        self.effective_sample_weights = effective_sample_weights
        self.model_class = model_class
        self.max_length = max_length
        self.num_workers = num_workers  # 0 loads batches in the main process

    def prepare_data(self):

        if self.model_class == "DeBERTaV3LargeModel":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=True)
        elif self.model_class == "ModernBertLargeModel":
            # add_prefix_space: each pre-split word is tokenized as if preceded by a space, as it is in running
            # text ("Ġlamb"). Without it, ModernBERT's byte-level BPE splits words into pieces it rarely saw
            # in pretraining ("l" + "amb").
            self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-large', use_fast=True,
                                                           add_prefix_space=True)

        if self.IOB:
            label_idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                         'I-MISC': 8}
        else:
            label_idx = {'O': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'MISC': 4}

        print(self.train_path)
        label_idx = get_label_idx(self.train_path, self.IOB, label_idx)
        print(label_idx)

        if self.effective_sample_weights:
            self.class_weights = compute_class_weights(self.train_path, label_idx, self.IOB, self.beta)
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
            self.class_weights[0] = self.class_weights[0]/self.o_class_mod
        else:
            self.class_weights = torch.ones(self.num_labels)
            self.class_weights[0] = self.class_weights[0]/self.o_class_mod


        self.train = make_tensor(self.train_path, label_idx, self.IOB, self.tokenizer, self.max_length,
                                       remove_o=self.remove_o)
        self.val = make_tensor(self.val_path, label_idx, self.IOB, self.tokenizer, self.max_length,)
        self.test = make_tensor(self.test_path, label_idx, self.IOB, self.tokenizer, self.max_length,)

    def collate_fn(self, batch):
        # Unpack the batch
        input_ids, attention_masks, labels = zip(*batch)

        # Pad sequences to the maximum length in this batch
        # pad_sequence handles variable length sequences by padding to the max length in the batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return padded_input_ids, padded_attention_masks, padded_labels
    def _worker_kwargs(self):
        if self.num_workers == 0:
            return dict(num_workers=0)
        return dict(num_workers=self.num_workers, prefetch_factor=2, persistent_workers=True)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, **self._worker_kwargs(),
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        # validation and test keep every sentence (drop_last=True used to skip the last partial batch)
        return DataLoader(self.val, batch_size=self.batch_size,
                          drop_last=False, **self._worker_kwargs(),
                          pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          drop_last=False, **self._worker_kwargs(),
                          pin_memory=True, collate_fn=self.collate_fn)
