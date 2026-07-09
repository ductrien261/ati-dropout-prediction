"""
Build 5 baseline datasets (GAF, RP, MTF, Heatmap, Radar)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ati.data.grouping import iter_week_daily_groups, count_groups
from src.ati.data.oulad_loader import N_ACTIVITIES_EXPECTED
from src.ati.transforms.gaf import build_gaf_image
from src.ati.transforms.rp import build_rp_image
from src.ati.transforms.mtf import build_mtf_image
from src.ati.transforms.heatmap import build_heatmap_image
from src.ati.transforms.radar import build_radar_image

OUTPUT_SPECS = {
    "baseline_gaf.npy":     ((N_ACTIVITIES_EXPECTED, 7, 7), build_gaf_image),
    "baseline_rp.npy":      ((N_ACTIVITIES_EXPECTED, 7, 7), build_rp_image),
    "baseline_mtf.npy":     ((N_ACTIVITIES_EXPECTED, 7, 7), build_mtf_image),
    "baseline_heatmap.npy": ((4, 5), build_heatmap_image),
    "baseline_radar.npy":   ((64, 64), build_radar_image),
}


def build_baseline_datasets(daily, labels, output_dir: Path):
    output_dir = Path(output_dir)
    n = N_ACTIVITIES_EXPECTED

    print("Counting total (student, week) groups...")
    n_samples = count_groups(daily)
    print(f"Total samples: {n_samples:,}")

    memmaps = {}
    for fname, (shape, _) in OUTPUT_SPECS.items():
        full_shape = (n_samples, *shape)
        memmaps[fname] = np.lib.format.open_memmap(
            output_dir / fname, mode="w+", dtype=np.float32, shape=full_shape
        )

    meta_rows = []
    for i, ((module, presentation, student_id, week), week_daily) in enumerate(
        tqdm(iter_week_daily_groups(daily, n), total=n_samples, desc="Building baseline images")
    ):
        for fname, (_, build_fn) in OUTPUT_SPECS.items():
            memmaps[fname][i] = build_fn(week_daily, n=n)
        meta_rows.append({
            "code_module": module, "code_presentation": presentation,
            "id_student": student_id, "week": week,
        })

    for m in memmaps.values():
        m.flush()

    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.merge(labels, on=["code_module", "code_presentation", "id_student"], how="left")
    assert meta_df["label"].isna().sum() == 0, "Có sample không map được nhãn"
    meta_df.to_parquet(output_dir / "baseline_metadata.parquet", index=False)

    print("Done. Shapes:")
    for fname, (shape, _) in OUTPUT_SPECS.items():
        print(f"  {fname}: ({n_samples}, {shape})")
    return meta_df


if __name__ == "__main__":
    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    daily = pd.read_parquet(PROCESSED_DIR / "daily_activity.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")

    build_baseline_datasets(daily, labels, PROCESSED_DIR)