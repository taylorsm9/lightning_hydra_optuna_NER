"""
Regression test for how every folder's preprocessing aligns word labels with subword tokens.

The original expand_encode_labels counted words by `offset[0] == 0` and crashed with IndexError when a
tokenizer gave one word two tokens starting at offset 0 (DeBERTa-v3 on "Gurion", "bacillus", "ASIAN", ...;
ModernBERT on "Çağrı"). It also labelled padding and special tokens "O" and gave every subword its word's
label. The fixed version uses encoding.word_ids() and labels only each word's first token. This test checks,
for every folder and both tokenizers, that
  1. nothing crashes on the fixture sentences, which are built to crash the original logic,
  2. each word's first token carries the word's label, and
  3. every other token (special, padding, rest of a word) is -100.

Needs the two tokenizers (downloaded on first run). Run from the repo root: pytest tests/
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from get_conll import conll_to_df  # noqa: E402

FOLDERS = sorted(p.name for p in ROOT.glob("0*_*") if p.is_dir())
# created the way the datamodules create them
TOKENIZERS = {"microsoft/deberta-v3-large": {}, "answerdotai/ModernBERT-large": {"add_prefix_space": True}}
LABEL_IDX = {"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4}


def load_preprocessing(folder):
    path = ROOT / folder / "src" / "utilities" / "data_preprocessing_functions.py"
    spec = importlib.util.spec_from_file_location(f"preprocessing_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_expand_encode_labels(labels, offset_mapping):
    """The original logic, verbatim apart from taking label_idx from the module scope."""
    expanded_labels = []
    word_index = 0
    for offset in offset_mapping:
        if offset == (0, 0):
            expanded_labels.append(LABEL_IDX["O"])
        else:
            if offset[0] == 0:
                expanded_labels.append(LABEL_IDX[labels[word_index]])
                word_index += 1
            else:
                expanded_labels.append(LABEL_IDX[labels[word_index - 1]])
    return expanded_labels


@pytest.fixture(scope="module")
def fixture_df():
    df = conll_to_df(ROOT / "tests" / "fixtures" / "tiny_ner.conll")
    df["Label"] = [[l[2:] if l[:2] in ("B-", "I-") else l for l in labels] for labels in df["Label"]]
    return df


@pytest.fixture(scope="module", params=list(TOKENIZERS))
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param, use_fast=True, **TOKENIZERS[request.param])


@pytest.mark.parametrize("folder", FOLDERS)
def test_first_subword_labels(folder, tokenizer, fixture_df):
    module = load_preprocessing(folder)
    result = module.tokenize_dataframe(fixture_df.copy(), LABEL_IDX, tokenizer, 512)

    crashed_before = 0
    for (_, row), token_ids, labels in zip(fixture_df.iterrows(), result["token_ids"], result["labels"]):
        assert len(labels) == len(token_ids)

        # same tokenizer call as the folder makes, to get word_ids and to run the original logic
        padded = "padding=\"max_length\"" in Path(module.__file__).read_text()
        kwargs = dict(padding="max_length", max_length=512) if padded else dict(max_length=512)
        enc = tokenizer(row["Word"], is_split_into_words=True, truncation=True, return_offsets_mapping=True, **kwargs)

        expected, previous = [], None
        for w in enc.word_ids():
            expected.append(-100 if w is None or w == previous else LABEL_IDX[row["Label"][w]])
            previous = w
        assert labels == expected, " ".join(row["Word"])
        assert sum(l != -100 for l in labels) == len(row["Word"])  # one labelled token per word (no truncation here)

        try:
            original_expand_encode_labels(row["Label"], [tuple(o) for o in enc["offset_mapping"]])
        except IndexError:
            crashed_before += 1

    # the fixture is built to hit the original crash with both tokenizers
    assert crashed_before > 0
