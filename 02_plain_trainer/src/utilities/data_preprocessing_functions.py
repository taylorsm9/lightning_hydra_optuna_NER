"""
Data Preprocessing Functions

This Python file contains a set of functions designed for data preprocessing tasks.
These functions are intended to be used to prepare and clean data for use in the ChEMUDataModule. 
Each function serves a specific purpose and contributes to the data preparation process. 

Functions:
- remove_only_o_sentences: Removes sentences containing only "O" class labels.
- strip_iob_tags: Removes "B-"/"I-" prefixes from labels.
- get_unique_list_labels: Lists the labels present in a DataFrame.
- compute_effective_sample_weights / compute_class_weights: Class-balanced loss weights from the
  "effective number of samples" (Cui et al., 2019).
- get_label_idx / reorder_label_indices: Builds the mapping between class labels and prediction indexes.
- tokenize_dataframe: Tokenizes each sentence and aligns word labels with tokens.
- create_single_tensor: Turns the tokenized DataFrame into a dataset.
- make_tensor: Runs the steps above for one data split.

Please refer to the function-specific docstrings for more detailed information on their implementation and usage.
Authors: Scott, Osmon, and Charles
Date: 10/19/2023

"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def remove_only_o_sentences(dataframe):
    """
    Removes samples from a DataFrame where the 'Label' column contains lists of labels
    and every label in a list is "O".

    Args:
        dataframe (DataFrame): DataFrame with 'Label' column containing lists of labels.

    Returns:
        DataFrame: Modified DataFrame without samples where the only label is "O".
    """

    # Function to check if a list contains labels other than "O"
    def contains_non_o(labels) -> bool:
        return not all(label == "O" for label in labels)

    # Copy the DataFrame
    dataframe_copy = dataframe.copy()

    # Filter rows where the 'Label' list contains labels other than "O"
    filtered_df = dataframe_copy[dataframe_copy['Label'].apply(contains_non_o)]

    return filtered_df



def strip_iob_tags(dataframe):
    """
    Strips 'B-' and 'I-' IOB tags from labels in a DataFrame where each 'Label'
    entry is a list of labels.

    Args:
        dataframe (DataFrame): DataFrame with 'Label' column containing lists of labels.

    Returns:
        DataFrame: Modified DataFrame with stripped labels.
    """

    # Nested function to strip IOB tags from a single label
    def strip_iob(label) -> str:
        if isinstance(label, str) and label.startswith(("B-", "I-")):
            return label[2:]
        return label

    # Nested function to process a list of labels
    def strip_iob_list(labels) -> list:
        return [strip_iob(label) for label in labels]

    # Copy the DataFrame to avoid modifying the original
    dataframe_copy = dataframe.copy()

    # Apply strip_iob_list function to each list in the 'Label' column
    dataframe_copy['Label'] = dataframe_copy['Label'].apply(strip_iob_list)

    return dataframe_copy


def get_unique_list_labels(dataframe):
    """
    Extracts a unique list of labels from a DataFrame's 'Label' column after sentences have been grouped.

    Args:
        dataframe (DataFrame): DataFrame with 'Label' column containing lists of labels.

    Returns:
        list: List of unique labels.
    """
    # Use a set comprehension to extract unique labels from each list in the 'Label' column
    unique_labels = {label for label_list in dataframe['Label'] for label in label_list}
    # Convert the set to a list to get your final list of unique labels
    return list(unique_labels)








def compute_effective_sample_weights(num_samples_per_class, beta):
    weights = (1 - beta) / (1 - np.power(beta, num_samples_per_class))
    return weights

def flatten_labels(label_lists):
    """Flatten a list of label lists into a single list of labels."""
    return [label for sublist in label_lists for label in sublist]


def compute_class_weights(data_path, label_idx, IOB, beta):
    data_df = pd.read_pickle(data_path)
    # data_df = remove_only_o_sentences(data_df)
    if not IOB:
        data_df = strip_iob_tags(data_df)

    data_df = data_df.dropna()

    # Flatten the 'Label' column lists into a single list of labels
    all_labels = flatten_labels(data_df['Label'])

    # Count occurrences of each label using Counter
    label_counts = Counter(all_labels)
    samples_per_class = [label_counts.get(label, 0) for label in label_idx]
    samples_per_class = samples_per_class[1:]
    weights = compute_effective_sample_weights(np.array(samples_per_class), beta)
    lowest = weights.argmin()
    weights = np.insert(weights, 0, weights[lowest])

    return weights


def reorder_label_indices(current_indices, order_indices):
    """
    Reorders the indices of labels according to the order specified in another dictionary, ensuring
    indices go from 0 to n for the labels present in the dataset, based on their appearance order
    in the order_indices dictionary.

    Parameters:
    - current_indices: dict, the current dictionary of labels with their indices.
    - order_indices: dict, the dictionary defining the desired order of labels.

    Returns:
    - A dictionary with labels reordered according to order_indices and indices from 0 to n.
    """
    ordered_indices = {}
    current_index = 0
    for label in order_indices.keys():
        if label in current_indices:
            ordered_indices[label] = current_index
            current_index += 1
    return ordered_indices


def get_label_idx(train_path, IOB, label_idx, label_column='Label'):
    """
    Assigns a unique index to each label found in a DataFrame column where each cell contains a list of labels,
    assuming there are no NaN values in the column.

    Parameters:
    - data_df: pandas.DataFrame containing the data.
    - label_column: str, the name of the column containing the lists of labels.

    Returns:
    - A dictionary with labels as keys and their assigned indices as values.
    """
    data_df = pd.read_pickle(train_path)
    if not IOB:
        data_df = strip_iob_tags(data_df)
    disorderly_idx = {}
    current_index = 0
    for labels in data_df[label_column]:
        for label in labels:
            if label not in disorderly_idx:
                disorderly_idx[label] = current_index
                current_index += 1
    label_idx = reorder_label_indices(disorderly_idx, label_idx)
    return label_idx


def create_single_tensor(df):
    """
    Creates a TensorDataset from tokenization data in a dictionary.

    Args:
        dict (dict): Contains 'token_ids', 'attention_masks', 'labels'.

    Returns:
        TensorDataset: Dataset with tokenization data.
    """
    token_ids = torch.tensor(df["token_ids"].tolist(), dtype=torch.long)
    attention_masks = torch.ByteTensor(df["attention_masks"].tolist())
    labels = torch.tensor(df["labels"].tolist(), dtype=torch.long)

    # Optionally, check that the first dimension (number of examples) is the same for all tensors
    assert token_ids.size(0) == attention_masks.size(0) == labels.size(0), "Size mismatch between tensors"

    return TensorDataset(token_ids, attention_masks, labels)

def tokenize_dataframe(df, label_idx, tokenizer, max_length) -> pd.DataFrame:
    """
    Takes a dataframe saved in a `.pkl` file and converts it into a tensor that can be used for train/dev/test/learning
        Args:
            data_path (str): filepath for the saved `.pkl` file, containing a "Word" and "Label" column.
            label_idx (dict): a dictionary used to encode labels into integers, required for returning a tensor. Example: {'0': 0, 'B-ORG': 1, ...}
            # TODO IOB (0 or 1): whether or not `B-` and `I-` prefixes to labels should be removed
            tokenizer (transformers.AutoTokenizer): the huggingface tokenizer that will be used on the "Word" column
            # TODO remove_o=False (Bool): whether or not samples with only `O` labels should be removed from the data
            testmode=False (Bool): If false, make_tensor will return a tensor ready to be used for training. If true, make_tensor will return a dataframe of 20 samples, so an inspector can review or validate functionality with ease.
    """

    def expand_encode_labels(labels, word_ids):
        # word_ids() gives each token the index of the word it came from, or None for special/pad tokens.
        # (The original version counted words by offset[0] == 0, which crashed when a tokenizer emitted two
        # tokens starting at offset 0 for one word, e.g. DeBERTa-v3 on "Gurion" or ModernBERT on "Çağrı".)
        # Only the first token of each word carries the word's label. Special tokens, padding and the rest of
        # a word's tokens get -100, which the loss (ignore_index=-100) and the metrics skip. (The original
        # labelled all of them, including padding, as "O" or the word's label, so most of each loss was
        # padding and the metrics counted every subword.)
        expanded_labels = []
        previous_word_index = None
        for word_index in word_ids:
            if word_index is None or word_index == previous_word_index:
                expanded_labels.append(-100)
            else:
                expanded_labels.append(label_idx[labels[word_index]])
            previous_word_index = word_index

        return expanded_labels

    def tokenize_row(tokenizer, row, max_length):
        encoded = tokenizer(row["Word"], is_split_into_words=True, padding="max_length", truncation=True, max_length=max_length, return_offsets_mapping=True)

        # print a row that has mismatched sequence lengths for tokenization, offset mapping, and attention mask
        if not (len(encoded["input_ids"]) == len(encoded["attention_mask"]) == len(encoded["offset_mapping"])):
            print(row["Word"])

        return encoded["input_ids"], encoded["attention_mask"], expand_encode_labels(row["Label"], encoded.word_ids())


    df[["token_ids", "attention_masks", "labels"]] = df.apply(lambda x: pd.Series(tokenize_row(tokenizer, x, max_length)), axis=1)

    return df

def make_tensor(data_path, label_idx, IOB, tokenizer, max_length, remove_o=False):
    """
    Processes and saves tensors for train/dev or test sets for continual learning steps.

    Args:
        data_dir (str): Directory containing data files.
        filename (str): Filename for saving tensors.
        fill_context (bool): Flag for context filling in tokenization.
        IOB (bool): Flag for stripping IOB tags.
        mode (str): Mode of operation ('train_dev' or 'test').

    Prepares and saves tensors by processing data from 'data_dir'.
    """
    print("preparing next dataframe")
    data_df = pd.read_pickle(data_path)

    if not (IOB):
        data_df = strip_iob_tags(data_df)

    if remove_o:
        data_df = remove_only_o_sentences(data_df)  # (the result used to be discarded, so remove_o did nothing)

    print("Labels present: ")
    print(get_unique_list_labels(data_df))
    print("sentences:", len(data_df))

    unknown = set(get_unique_list_labels(data_df)) - set(label_idx)
    if unknown:
        raise ValueError(f"{data_path} has labels the training set doesn't: {sorted(unknown)}")

    tokenized_df = tokenize_dataframe(data_df, label_idx, tokenizer, max_length)
    tensor = create_single_tensor(tokenized_df)
    # save_tensor(tensor, filename, tensor_dir)

    # return 0
    return tensor
