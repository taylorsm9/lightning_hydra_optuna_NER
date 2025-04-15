import os
import pandas as pd
from transformers import AutoTokenizer
import jsonargparse
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def get_encodings(tokenizer, text):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return encoding['input_ids'], encoding['offset_mapping']

def split_entities(entities):
    "takes in a list of dicts where a dict contains metadata about entities"
    title_spans = []
    title_labels = []
    abstract_spans = []
    abstract_labels = []

    # Split up the entities by abstract/title
    for entity in entities:
        # get the span of the entity
        span_tuple = (entity['start_idx'], entity['end_idx'])
        if entity['location'] == 'title':
            title_spans.append(span_tuple)
            title_labels.append(entity['label'])
        else:
            abstract_spans.append(span_tuple)
            abstract_labels.append(entity['label'])

    return title_spans, title_labels, abstract_spans, abstract_labels

def add_labels_and_spans(df, labels_path):
    label_df = pd.read_json(labels_path)
    label_df = label_df.transpose()
    results = label_df['entities'].apply(split_entities)
    label_df['title_spans'] = [r[0] for r in results]
    label_df['title_entities_only'] = [r[1] for r in results]
    label_df['abstract_spans'] = [r[2] for r in results]
    label_df['abstract_entities_only'] = [r[3] for r in results]

    # Drop columns we don't need for NER
    cols_to_drop = ['metadata', 'entities','relations','binary_tag_based_relations', 'ternary_tag_based_relations', 'ternary_mention_based_relations']
    for col in cols_to_drop:
        label_df = label_df.drop(col, axis=1)
    df = df.merge(label_df, left_index=True, right_index=True, how='left')
    return df

def add_label_idx_column(df, label_idx):
    """
    Adds a 'title_label_idx' column to df where each entry
    is a list of indices corresponding to the labels
    in the 'title_labels' column.
    """

    inverse_label_idx = {v: k for k, v in label_idx.items()}
    df["title_entities_only_idx"] = df["title_entities_only"].apply(
        lambda label_list: [inverse_label_idx[label] for label in label_list]
    )
    df["abstract_entities_only_idx"] = df["abstract_entities_only"].apply(
    lambda label_list: [inverse_label_idx[label] for label in label_list]
    )

    return df

def clean_offset_tuple(offset_tuple, sample_text):
    """
    Clean a single offset tuple by adjusting for whitespace at boundaries.

    Args:
        offset_tuple: A tuple of (start, end) character positions
        sample_text: The original text string

    Returns:
        A new tuple with adjusted (start, end) positions
    """
    start, end = offset_tuple

    # If there's text and the start character is whitespace, increment start
    if start < len(sample_text) and (
        sample_text[start].isspace() or
        sample_text[start] == '\u00a0'  # Non-breaking space
    ):
        start += 1

    # If there's content and the character just before the end is whitespace, decrement end
    if end > 0 and end <= len(sample_text) and (
        sample_text[end-1].isspace() or
        sample_text[end-1] == '\u00a0'  # Non-breaking space
    ):
        end -= 1

    end -= 1 # Handle exclusive tuple

    return (start, end)

def clean_offsets_column(df, offsets_column, text_column):
    """
    Clean all offset tuples in a specific column based on a corresponding text column.

    Args:
        df: The pandas DataFrame
        offsets_column: Name of the column containing lists of offset tuples
        text_column: Name of the column containing the text to reference

    Returns:
        The modified DataFrame with cleaned offsets
    """
    # Make a copy to avoid the SettingWithCopyWarning
    df_copy = df.copy()

    # Define a function to clean a single row's offsets
    def clean_row_offsets(row):
        text = row[text_column]
        row[offsets_column] = [clean_offset_tuple(offset, text) for offset in row[offsets_column]]
        return row

    # Apply the function to each row
    return df_copy.apply(clean_row_offsets, axis=1)

def make_label_list(entity_spans, entity_labels, offsets):
    """ takes in 2 equal length lists of entity span tuples (int) and entity label ints (not full sentence labels, just entities), along with a list of offsets for the full sample, including 'O' class tokens
    returns a list of labels for the full sample
    """
    full_labels = [0] * len(offsets)
    j = 0
    for i, offset in enumerate(offsets): # for every token in the sample
     if j < len(entity_spans): # for every entity in the solution
         # The last item in an offset tuple is - get this - itself offset by 1. So if an offset stands for a single character at index 0, its offset will be 0-1. We must subtract 1 to align with the data
         entity_start = entity_spans[j][0]
         entity_end = entity_spans[j][1]
         offset_last_char = offsets[i][1]
         if entity_start <= offset_last_char <= entity_end: # if we are somewhere within the entity bounds (inclusive)
            full_labels[i] = entity_labels[j]
         if offset_last_char == entity_end: # if we have reached the end of the entity
             j += 1
    return full_labels

def process_label_row(row):
    """Apply make_label_list to a single DataFrame row"""
    entity_spans = row['abstract_spans']
    entity_labels = row['abstract_entities_only_idx']
    offsets = row['abstract_offset']
    return make_label_list(entity_spans, entity_labels, offsets)

def make_full_sample(tokenizer, row):
    TITLE_START_TOKEN_ID = tokenizer.convert_tokens_to_ids('<TITLE>')
    TITLE_END_TOKEN_ID = tokenizer.convert_tokens_to_ids('</TITLE>')
    ABSTRACT_START_TOKEN_ID = tokenizer.convert_tokens_to_ids('<ABSTRACT>')
    ABSTRACT_END_TOKEN_ID = tokenizer.convert_tokens_to_ids('</ABSTRACT>')

    # Add special tokens around the title
    title_with_boundaries = [TITLE_START_TOKEN_ID] + row['full_title_labels'] + [TITLE_END_TOKEN_ID]

    # Add special tokens around the abstract
    abstract_with_boundaries = [ABSTRACT_START_TOKEN_ID] + row['full_abstract_labels'] + [ABSTRACT_END_TOKEN_ID]

    # Combine sample
    combined_sample = title_with_boundaries + abstract_with_boundaries

    return combined_sample

def make_full_sample_labels(row):
    # 'O' class index
    O_CLASS_INDEX = 0

    # Add 'O' labels at the beginning and end of title labels (for the special tokens)
    title_labels_with_boundaries = [O_CLASS_INDEX] + row['full_title_labels'] + [O_CLASS_INDEX]

    # Add 'O' labels at the beginning and end of abstract labels (for the special tokens)
    abstract_labels_with_boundaries = [O_CLASS_INDEX] + row['full_abstract_labels'] + [O_CLASS_INDEX]

    # Combine labels
    combined_labels = title_labels_with_boundaries + abstract_labels_with_boundaries

    return combined_labels

if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser(description="Process dataset from config")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')

    # Load config from args
    args = parser.parse_args()
    config = load_config(args.config)

    # Get paths from the config
    dataset_path = config['dataset_path']
    labels_path = config['labels_path']
    save_path = config['save_path']

    # Index of labels from NER set
    label_idx = {
        0: "O",
        1: "anatomical location",
        2: "animal",
        3: "bacteria",
        4: "biomedical technique",
        5: "chemical",
        6: "DDF",
        7: "dietary supplement",
        8: "drug",
        9: "food",
        10: "gene",
        11: "human",
        12: "microbiome",
        13: "statistical technique"
    }

    df = pd.read_json(dataset_path, orient='index')

    # Use a tokenizer with added special tokens for title and abstract boundaries
    tokenizer = AutoTokenizer.from_pretrained('gutbrain_tokenizer')

    # Get token IDs
    df[['title_encoded', 'title_offset']] = df['title'].apply(lambda x: pd.Series(get_encodings(tokenizer, x)))
    df[['abstract_encoded', 'abstract_offset']] = df['abstract'].apply(lambda x: pd.Series(get_encodings(tokenizer, x)))

    # Add the labels and spans
    df = add_labels_and_spans(df, labels_path)

    # Convert the labels into indices
    df = add_label_idx_column(df, label_idx)

    # Handle the spacing indices included by the tokenizer, removing indices of spaces that are included in offsets
    df = clean_offsets_column(df, 'abstract_offset', 'abstract')
    df = clean_offsets_column(df, 'title_offset', 'title')

    # Use the label indices from the labels and the cleaned offsets to make a list of labels for the samples
    df['full_abstract_labels'] = df.apply(process_label_row, axis=1)
    df['full_title_labels'] = df.apply(process_label_row, axis=1)

    # Combine title and abstract samples with special tokens for boundaries
    df['Word'] = df.apply(lambda row: make_full_sample(tokenizer, row), axis=1)

    df['Label'] = df.apply(make_full_sample_labels, axis=1)
    df.to_pickle(save_path)