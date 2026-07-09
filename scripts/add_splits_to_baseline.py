import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.splits import make_student_level_splits, attach_split

if __name__ == "__main__":
    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

    ati_meta = pd.read_parquet(PROCESSED_DIR / "ati_v1_metadata.parquet")
    split_table = ati_meta[["code_module", "code_presentation", "id_student", "split"]].drop_duplicates()

    baseline_meta = pd.read_parquet(PROCESSED_DIR / "baseline_metadata.parquet")
    baseline_meta = baseline_meta.merge(
        split_table, on=["code_module", "code_presentation", "id_student"], how="left"
    )
    assert baseline_meta["split"].isna().sum() == 0, "Có dòng không map được split"

    baseline_meta.to_parquet(PROCESSED_DIR / "baseline_metadata.parquet", index=False)
    print(baseline_meta.groupby("split")["label"].agg(["mean", "count"]))