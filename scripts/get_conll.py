"""
Download CoNLL-2003 and write it in the layout every folder in this repo expects.

CoNLL-2003 is derived from the Reuters RCV1 corpus and is not redistributed in this repo.
This script fetches a public copy from the Hugging Face Hub (eriktks/conll2003) and writes:

    datasets/conll_2003/{train,val,test}.conll   tab-separated: tag, 1, 1, word (IOB2), blank line between sentences
    datasets/conll_2003/{train,val,test}.pkl     pandas DataFrame, one row per sentence, list columns Word and Label

The .pkl files are what the datamodules read. Every token is kept, in the dataset's sentence order. (The 2025
files were built with pandas.read_csv in notebooks/dataset_preprocessing/preprocess_conll.ipynb, which reads the
words "NA" and "None" as missing values; dropna() then removed those tokens from 14 sentences, and groupby
sorted the sentences by their string id. Double quotes in the .conll files are still escaped as \\" so the
notebook can read them.)

Usage (from the repo root):
    python scripts/get_conll.py
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def conll_to_df(conll_path):
    """One row per sentence of a .conll file written by this script, with list columns Word and Label."""
    rows, words, labels = [], [], []
    for line in Path(conll_path).read_text(encoding="utf-8").split("\n") + [""]:
        if not line.strip():  # a blank line ends a sentence
            if words:
                rows.append((words, labels))
            words, labels = [], []
            continue
        tag, _, _, word = line.split("\t", 3)
        words.append(word.replace('\\"', '"'))
        labels.append(tag)
    # plain object column labels, so the pickle loads under pandas 2 (which the project used) and pandas 3
    return pd.DataFrame(rows, columns=pd.Index(["Word", "Label"], dtype=object))


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
