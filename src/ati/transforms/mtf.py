"""
MTF (Markov Transition Field) per activity type.
"""
import numpy as np
from pyts.image import MarkovTransitionField

from src.ati.transforms.common import (
    N_ACTIVITIES, DAYS_PER_WEEK, is_sparse_week, build_weekly_sequence,
)


def build_mtf_image(week_daily: dict, n: int = N_ACTIVITIES,
                     days_per_week: int = DAYS_PER_WEEK,
                     n_bins: int = 4) -> np.ndarray:
    if not week_daily or is_sparse_week(week_daily):
        return np.zeros((n, days_per_week, days_per_week), dtype=np.float32)

    seq = build_weekly_sequence(week_daily, n, days_per_week)
    mtf = MarkovTransitionField(image_size=days_per_week, n_bins=n_bins, strategy="uniform")
    with np.errstate(invalid="ignore", divide="ignore"):
        images = mtf.fit_transform(seq)
    images = np.nan_to_num(images, nan=0.0)
    return images.astype(np.float32)