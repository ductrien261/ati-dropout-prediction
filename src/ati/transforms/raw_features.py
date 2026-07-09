import numpy as np

from src.ati.transforms.common import N_ACTIVITIES, is_sparse_week, build_weekly_vector


def build_raw_features(week_daily: dict, n: int = N_ACTIVITIES) -> np.ndarray:
    """Trả về vector (n,) - tổng click mỗi activity trong tuần, KHÔNG chuẩn hóa"""
    if not week_daily or is_sparse_week(week_daily):
        return np.zeros(n, dtype=np.float32)
    return build_weekly_vector(week_daily, n).astype(np.float32)