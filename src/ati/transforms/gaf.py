"""
GAF (Gramian Angular Field) per activity type.
Áp dụng lên chuỗi 7-ngày của TỪNG activity riêng biệt -> 20 ảnh 7x7 xếp chồng.
"""
import numpy as np
from pyts.image import GramianAngularField

from src.ati.transforms.common import (
    N_ACTIVITIES, DAYS_PER_WEEK, is_sparse_week, build_weekly_sequence,
)


def build_gaf_image(week_daily: dict, n: int = N_ACTIVITIES,
                     days_per_week: int = DAYS_PER_WEEK,
                     method: str = "summation") -> np.ndarray:
    """Trả về tensor (n, days_per_week, days_per_week)."""
    if not week_daily or is_sparse_week(week_daily):
        return np.zeros((n, days_per_week, days_per_week), dtype=np.float32)

    seq = build_weekly_sequence(week_daily, n, days_per_week)
    gaf = GramianAngularField(image_size=days_per_week, method=method)
    with np.errstate(invalid="ignore", divide="ignore"):
        images = gaf.fit_transform(seq)
    images = np.nan_to_num(images, nan=0.0) 
    return images.astype(np.float32)