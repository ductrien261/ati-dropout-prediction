"""
Hàm dùng chung cho toàn bộ baseline (GAF, RP, MTF, Heatmap, Radar).
Tất cả nhận cùng 1 kiểu input: week_daily = dict[date:int -> np.ndarray(20)],
"""
import numpy as np

N_ACTIVITIES = 20
DAYS_PER_WEEK = 7
SPARSE_WEEK_THRESHOLD = 5


def is_sparse_week(week_daily: dict, k: int = SPARSE_WEEK_THRESHOLD) -> bool:
    total_clicks = sum(vec.sum() for vec in week_daily.values())
    return total_clicks < k


def build_weekly_sequence(week_daily: dict, n: int = N_ACTIVITIES,
                           days_per_week: int = DAYS_PER_WEEK) -> np.ndarray:
    """
    Trả về ma trận (n, days_per_week): mỗi hàng là chuỗi click theo thứ-trong-tuần (offset 0..6) của 1 activity type.
    """
    seq = np.zeros((n, days_per_week), dtype=np.float64)
    for date, vec in week_daily.items():
        offset = date % days_per_week
        seq[:, offset] += vec
    return seq


def build_weekly_vector(week_daily: dict, n: int = N_ACTIVITIES) -> np.ndarray:
    """Tổng click mỗi activity trong cả tuần (dùng cho Heatmap, Radar)."""
    vec = np.zeros(n, dtype=np.float64)
    for v in week_daily.values():
        vec += v
    return vec