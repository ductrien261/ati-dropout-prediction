"""
Build raw/tabular representation cho toàn bộ dataset.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.grouping import iter_week_daily_groups, count_groups
from src.ati.data.oulad_loader import N_ACTIVITIES_EXPECTED
from src.ati.transforms.raw_features import build_raw_features

if __name__ == "__main__":
    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    daily = pd.read_parquet(PROCESSED_DIR / "daily_activity.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    n = N_ACTIVITIES_EXPECTED

    n_samples = count_groups(daily)
    print(f"Total samples: {n_samples:,}")

    features = np.zeros((n_samples, n), dtype=np.float32)
    meta_rows = []

    for i, ((module, presentation, student_id, week), week_daily) in enumerate(
        tqdm(iter_week_daily_groups(daily, n), total=n_samples, desc="Building raw features")
    ):
        features[i] = build_raw_features(week_daily, n=n)
        meta_rows.append({
            "code_module": module, "code_presentation": presentation,
            "id_student": student_id, "week": week,
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.merge(labels, on=["code_module", "code_presentation", "id_student"], how="left")
    assert meta_df["label"].isna().sum() == 0

    ati_meta = pd.read_parquet(PROCESSED_DIR / "ati_v1_metadata.parquet")
    split_table = ati_meta[["code_module", "code_presentation", "id_student", "split"]].drop_duplicates()
    meta_df = meta_df.merge(split_table, on=["code_module", "code_presentation", "id_student"], how="left")
    assert meta_df["split"].isna().sum() == 0

    np.save(PROCESSED_DIR / "raw_features.npy", features)
    meta_df.to_parquet(PROCESSED_DIR / "raw_features_metadata.parquet", index=False)
    print(f"Saved raw_features: {features.shape}")
    print(meta_df.groupby("split")["label"].agg(["mean", "count"]))