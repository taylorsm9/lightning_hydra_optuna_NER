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


def flatten_tokens_and_labels(sentence, labels, IOB, tokenizer):
    """
    Flattens a list of tokens and their corresponding labels. Handles IOB tagging if required.

    Args:
        sentence (list): A list of tokens in the sentence.
        labels (list): A list of corresponding labels for the tokens.
        IOB (bool): Flag to process IOB tags.

    Returns:
        list, list: Flattened lists of tokens and their corresponding labels.
    """
    flat_tokens, flat_labels = [], []
    for token, label in zip(sentence, labels):
        sub_words = tokenizer.tokenize(token)
        flat_tokens.extend(sub_words)

        if IOB and label != "O":
            token_sub_labels = [label] + ["I-" + label[2:]] * (len(sub_words) - 1)
        else:
            token_sub_labels = [label] * len(sub_words)

        flat_labels.extend(token_sub_labels)

    return flat_tokens, flat_labels


def process_batch(tokens, labels, encoded_token_ids, encoded_attention_masks, encoded_sub_labels, label_idx, tokenizer):
    """
    Maps pre-tokenized tokens to their labels, and manually constructs model inputs
    for pre-tokenized and subword-tokenized inputs.

    Args:
        tokens (list of str): Pre-tokenized tokens.
        labels (list of str): Labels for each token.
        encoded_token_ids (list): To store encoded token IDs.
        encoded_attention_masks (list): To store attention masks.
        encoded_sub_labels (list): To store labels for each sub-token, mapped by label_idx.
        label_idx (dict): Mapping of labels to their indices.
        tokenizer: Tokenizer instance compatible with BERT (from Hugging Face's Transformers).
    """

    # Convert pre-tokenized tokens to token IDs, handling subwords correctly
    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    # Padding: Ensure length matches max_length
    padding_length = 512 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length

    # Update encoded inputs and attention masks
    encoded_token_ids.append(input_ids)
    encoded_attention_masks.append(attention_mask)

    # Align labels with encoded tokens, accounting for special tokens
    adjusted_labels = [label_idx.get("O", -100)]  # Start with label for [CLS]
    # Adjust labels according to provided mapping, excluding special tokens at start/end
    adjusted_labels += [label_idx.get(label, label_idx.get("O", -100)) for label in labels]
    # Add label for [SEP]
    adjusted_labels.append(label_idx.get("O", -100))

    # Padding: Ensure label length matches token length
    adjusted_labels += [label_idx.get("O", -100)] * padding_length

    encoded_sub_labels.append(adjusted_labels)


def BERT_preprocessing(sentences, label_sets, label_idx, IOB, fill_context, tokenizer):
    """
    Processes tokenized sentences for BERT input, with or without context filling.
    Input IDs are generated in process_batch

    Args:
        sentences (list): Tokenized sentences to process.
        label_sets (list): Corresponding label sequences for each sentence.
        IOB (bool): Flag to indicate usage of IOB tagging format.
        fill_context (bool): If True, fills BERT's context window with sentences.

    Returns:
        dict: Contains 'token_ids', 'attention_masks', and 'labels' as tensors.
    """
    encoded_token_ids, encoded_attention_masks, encoded_sub_labels = [], [], []
    temp_tokens, temp_labels = [], []
    total_length = 0  # Track total length for fill_context

    for sentence, labels in tqdm(zip(sentences, label_sets), desc="Tokenizing sentences"):
        flat_tokens, flat_labels = flatten_tokens_and_labels(sentence, labels, IOB, tokenizer)

        if fill_context:
            # Calculate the length of the tokenized sentence (including special tokens)
            tokenized_length = len(tokenizer.encode(" ".join(flat_tokens), add_special_tokens=True, truncation=False))

            # Check if adding this sentence exceeds the max length
            if total_length + tokenized_length > 512:
                # Process the batch accumulated so far
                process_batch(temp_tokens, temp_labels, encoded_token_ids, encoded_attention_masks,
                              encoded_sub_labels, label_idx, tokenizer)
                # Reset for new batch
                temp_tokens, temp_labels, total_length = flat_tokens, flat_labels, tokenized_length
            else:
                # Accumulate this sentence
                temp_tokens.extend(flat_tokens)
                temp_labels.extend(flat_labels)
                total_length += tokenized_length
        else:
            # Process each sentence individually
            # first check if tokenized length is > 512
            process_batch(flat_tokens, flat_labels, encoded_token_ids, encoded_attention_masks,
                          encoded_sub_labels, label_idx, tokenizer)

    # Process any remaining sentences in the batch for fill_context mode
    if fill_context and temp_tokens:
        process_batch(temp_tokens, temp_labels, encoded_token_ids, encoded_attention_masks,
                      encoded_sub_labels, label_idx, tokenizer)

    return {
        "token_ids": torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks": torch.ByteTensor(encoded_attention_masks),
        "labels": torch.tensor(encoded_sub_labels, dtype=torch.long)
    }


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


def tokenize_dataframe(dataframe, label_idx, IOB, fill_context, tokenizer):
    """
    Tokenizes a DataFrame's sentences and labels for BERT processing.

    Args:
        dataframe (DataFrame): DataFrame with 'Word' and 'Label' columns.
        label_to_index (dict): Mapping of labels to indices.
        fill_context (bool): Flag to determine the type of tokenization.

    Returns:
        list: Tokenized data suitable for BERT input.
    """
    labels = dataframe["Label"].tolist()
    sentences = dataframe["Word"].tolist()
    tokenized_data = BERT_preprocessing(sentences, labels, label_idx, IOB, fill_context, tokenizer)
    return tokenized_data


def create_single_tensor(df):
    """
    Creates a TensorDataset from tokenization data in a dictionary.

    Args:
        dict (dict): Contains 'token_ids', 'attention_masks', 'labels'.

    Returns:
        TensorDataset: Dataset with tokenization data.
    """
    single_tensor = TensorDataset(df["token_ids"],
                                  df["attention_masks"],
                                  df["labels"])
    return single_tensor


# def save_tensor(tensor, filename, tensor_dir):
#     """
#     Saves a list of tensors into a directory, each in its own subfolder.
#
#     Args:
#         tensor_list (list): List of tensors to save.
#         tensor_dir (str): Base directory for saving tensors.
#         filename (str): Filename for saving each tensor.
#
#     Creates subfolders in 'tensor_dir' for each tensor and saves them with 'filename'.
#     """
#     # Ensure the base directory exists
#     if not os.path.exists(tensor_dir):
#         os.makedirs(tensor_dir)
#
#     # Construct the full file path for the current tensor
#     file_path = os.path.join(tensor_dir, filename)
#
#     # Save the tensor
#     torch.save(tensor, file_path)


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


def make_tensor(data_path, label_idx, IOB, tokenizer, remove_o = False, fill_context = False):
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
    if remove_o:
        data_df = remove_only_o_sentences(data_df)
    # data_df = add_sentence_numbers(data_df)
    if not (IOB):
        data_df = strip_iob_tags(data_df)

    print("Continual Learning Step labels: ")
    print(get_unique_list_labels(data_df))
    print("df items:", data_df.size)

    tokenized_df = tokenize_dataframe(data_df, label_idx, IOB, fill_context, tokenizer)
    tensor = create_single_tensor(tokenized_df)
    # save_tensor(tensor, filename, tensor_dir)

    # return 0
    return tensor


