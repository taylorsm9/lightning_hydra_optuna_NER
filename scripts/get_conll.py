"""
Download CoNLL-2003 and write it in the layout every folder in this repo expects.

CoNLL-2003 is derived from the Reuters RCV1 corpus and is not redistributed in this repo.
This script fetches a public copy from the Hugging Face Hub (eriktks/conll2003) and writes:

    datasets/conll_2003/{train,val,test}.conll   tab-separated: tag, 1, 1, word (IOB2), blank line between sentences
    datasets/conll_2003/{train,val,test}.pkl     pandas DataFrame, one row per sentence, list columns Word and Label

The .pkl files are what the datamodules read. Both are byte-for-byte / row-for-row identical to the
files the project was trained on in 2025, including two quirks of the original pipeline:

  - the .conll files escape double quotes as \\"
  - the .pkl files were built with pandas.read_csv (see notebooks/dataset_preprocessing/preprocess_conll.ipynb),
    which reads the words "NA" and "None" as missing values; dropna() then removed those tokens from 14 sentences.
    Sentences are also ordered by their string id ("Sentence: 1", "Sentence: 10", ...), as groupby left them.

Usage (from the repo root):
    python scripts/get_conll.py
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def conll_to_df(conll_path):
    """Same steps as dir_to_df + add_sentence_numbers + groupby in preprocess_conll.ipynb."""
    data = pd.read_csv(conll_path, sep="\t", encoding="utf-8", header=None, skip_blank_lines=False,
                       names=["Label", "Start", "End", "Word"], escapechar="\\")
    data = pd.concat([data, pd.DataFrame([[None] * 4], columns=data.columns)], ignore_index=True)

    # a NaN Label marks a sentence break; row 0 always starts "Sentence: 1"
    breaks = data["Label"].isna() & (data.index > 0)
    data["Sentence #"] = "Sentence: " + (1 + breaks.cumsum()).astype(str)
    data = data.dropna()

    grouped = data.groupby("Sentence #").agg({"Label": list, "Start": list, "End": list, "Word": list})
    return grouped.reset_index()[["Word", "Label"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=Path(__file__).resolve().parent.parent / "datasets" / "conll_2003",
                        type=Path, help="where to write the files (default: <repo>/datasets/conll_2003)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # the dataset repo uses a loading script, which datasets>=4 no longer runs; use the Hub's parquet conversion
    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
    tag_names = ds["train"].features["ner_tags"].feature.names

    for name, hf_split in SPLITS.items():
        sentences = []
        for words, tags in zip(ds[hf_split]["tokens"], ds[hf_split]["ner_tags"]):
            sentences.append([f"{tag_names[t]}\t1\t1\t" + w.replace('"', '\\"') for w, t in zip(words, tags)])
        conll_path = args.out_dir / f"{name}.conll"
        conll_path.write_text("\n\n".join("\n".join(s) for s in sentences) + "\n", encoding="utf-8")

        conll_to_df(conll_path).to_pickle(args.out_dir / f"{name}.pkl")
        print(f"{name}: {len(sentences)} sentences -> {args.out_dir / name}.{{conll,pkl}}")


if __name__ == "__main__":
    main()
