import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.splits import make_student_level_splits, attach_split

if __name__ == "__main__":
    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    meta = pd.read_parquet(PROCESSED_DIR / "ati_v1_metadata.parquet")

    split_table = make_student_level_splits(meta)
    meta_with_split = attach_split(meta, split_table)

    meta_with_split.to_parquet(PROCESSED_DIR / "ati_v1_metadata.parquet", index=False)
    print(meta_with_split.groupby("split")["label"].agg(["mean", "count"]))