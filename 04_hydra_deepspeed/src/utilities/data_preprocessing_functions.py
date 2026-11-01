"""
Data Preprocessing Functions

This Python file contains a set of functions designed for data preprocessing tasks.
These functions are intended to be used to prepare and clean data for use in the ChEMUDataModule. 
Each function serves a specific purpose and contributes to the data preparation process. 

Functions:
- dir_to_df: Creates a dataframe from folder of raw data files.
- remove_only_o_sentences: Removes sentences containing only "O" class labels.
- add_sentence_numbers: Associates tokens with a sentence ID number.
- map_labels: Builds a mapping between class labels and prediction indexes.
- multi_sentence_BERT_preprocessing: Prepares data samples for use with BERT.
- single_sentence_BERT_preprocessing: Prepares data samples for use with BERT.
- run_data_preprocessing: Runs all the data preprocessing steps in order.

Please refer to the function-specific docstrings for more detailed information on their implementation and usage.
Authors: Scott, Osman, and Charles
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

import os
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
    Creates a list of tensor tuples from tokenization data with variable lengths.
    """
    dataset = []
    for i in range(len(df)):
        token_ids = torch.tensor(df.iloc[i]["token_ids"], dtype=torch.long)
        attention_masks = torch.tensor(df.iloc[i]["attention_masks"], dtype=torch.uint8)
        labels = torch.tensor(df.iloc[i]["labels"], dtype=torch.long)

        dataset.append((token_ids, attention_masks, labels))

    return dataset


def tokenize_dataframe(df, label_idx, tokenizer, max_length) -> pd.DataFrame:
    """
    Takes a dataframe and tokenizes the text without padding to fixed length.
    Returns a DataFrame with tokenization results.
    """

    def expand_encode_labels(labels, word_ids):
        # word_ids() gives each token the index of the word it came from, or None for special/pad tokens.
        # (The original version counted words by offset[0] == 0, which crashed when a tokenizer emitted two
        # tokens starting at offset 0 for one word, e.g. DeBERTa-v3 on "Gurion" or ModernBERT on "Çağrı".)
        expanded_labels = []
        for word_index in word_ids:
            if word_index is None:  # Special tokens like [CLS], [SEP]
                expanded_labels.append(label_idx["O"])  # Ignored during loss computation
            else:
                # Assign the label of the current word to its tokens
                expanded_labels.append(label_idx[labels[word_index]])

        return expanded_labels
    token_ids_list = []
    attention_masks_list = []
    labels_list = []

    for _, row in df.iterrows():
        encoded = tokenizer(
            row["Word"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,  # Max length for truncation only, padding is dynamic
            return_offsets_mapping=True
        )

        # Check for mismatched lengths
        if not (len(encoded["input_ids"]) == len(encoded["attention_mask"]) == len(encoded["offset_mapping"])):
            print(row["Word"])

        # Use existing expand_encode_labels function to map word labels to token labels
        labels = expand_encode_labels(row["Label"], encoded.word_ids())

        token_ids_list.append(encoded["input_ids"])
        attention_masks_list.append(encoded["attention_mask"])
        labels_list.append(labels)

    # Create a new DataFrame with the tokenized data
    result_df = df.copy()
    result_df["token_ids"] = token_ids_list
    result_df["attention_masks"] = attention_masks_list
    result_df["labels"] = labels_list

    return result_df

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
        remove_only_o_sentences(data_df)

    print("Labels present: ")
    print(get_unique_list_labels(data_df))
    print("df items:", data_df.size)

    tokenized_df = tokenize_dataframe(data_df, label_idx, tokenizer, max_length)
    tensor = create_single_tensor(tokenized_df)
    # save_tensor(tensor, filename, tensor_dir)

    # return 0
    return tensor
