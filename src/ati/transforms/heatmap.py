"""
Heatmap baseline: reshape vector tổng-click-mỗi-activity-trong-tuần (20,) thành lưới cố định 4x5, chuẩn hóa [0,1].
"""
import numpy as np

from src.ati.transforms.common import N_ACTIVITIES, is_sparse_week, build_weekly_vector

GRID_SHAPE = (4, 5)  # 4*5 = 20 = N_ACTIVITIES


def build_heatmap_image(week_daily: dict, n: int = N_ACTIVITIES,
                         grid_shape: tuple = GRID_SHAPE) -> np.ndarray:
    """Trả về ảnh (grid_shape[0], grid_shape[1]), giá trị chuẩn hóa [0,1]."""
    assert grid_shape[0] * grid_shape[1] == n, "grid_shape phải khớp N_ACTIVITIES"

    if not week_daily or is_sparse_week(week_daily):
        return np.zeros(grid_shape, dtype=np.float32)

    vec = build_weekly_vector(week_daily, n)
    total = vec.sum()
    normalized = vec / total if total > 0 else vec
    return normalized.reshape(grid_shape).astype(np.float32)