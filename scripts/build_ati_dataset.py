import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.build_dataset import build_full_dataset
from src.ati.data.splits import make_student_level_splits, attach_split

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "oulad"
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

    matrices_arr, meta_df = build_full_dataset(DATA_DIR, OUTPUT_DIR)

    print("Building student-level train/val/test split...")
    split_table = make_student_level_splits(meta_df)
    meta_with_split = attach_split(meta_df, split_table)

    meta_with_split.to_parquet(OUTPUT_DIR / "ati_v1_metadata.parquet", index=False)
    print(meta_with_split.groupby("split")["label"].agg(["mean", "count"]))