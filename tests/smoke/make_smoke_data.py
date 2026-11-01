"""Turn tests/fixtures/tiny_ner.conll into train/val/test .pkl files, the same way get_conll.py does."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from get_conll import conll_to_df  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
df = conll_to_df(ROOT / "tests" / "fixtures" / "tiny_ner.conll")
for split in ["train", "val", "test"]:
    df.to_pickle(out / f"{split}.pkl")
