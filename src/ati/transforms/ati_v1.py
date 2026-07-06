"""
ATI-v1 (row-normalized): xây ma trận chuyển đổi.
"""
import numpy as np

N_ACTIVITIES = 20
EPSILON = 1e-6
SPARSE_WEEK_THRESHOLD = 5


def is_sparse_week(week_daily: dict, k: int = SPARSE_WEEK_THRESHOLD) -> bool:
    total_clicks = sum(vec.sum() for vec in week_daily.values())
    return total_clicks < k


def uniform_matrix(n: int = N_ACTIVITIES) -> np.ndarray:
    return np.full((n, n), 1.0 / n)


def build_transition_matrix(week_daily: dict, n: int = N_ACTIVITIES,
                             epsilon: float = EPSILON,
                             k: int = SPARSE_WEEK_THRESHOLD) -> np.ndarray:
    """
    week_daily: dict[date:int -> np.ndarray(n)] vector số click theo
    activity, cho từng ngày, của MỘT (student, week).

    Trả về ma trận n x n đã chuẩn hóa theo hàng (ATI-v1).

    Trả về uniform_matrix nếu:
      - week_daily rỗng
      - tổng click < k (sparse week)
      - không có cặp ngày liên tiếp nào (không tạo được transition thực)
    """
    if not week_daily or is_sparse_week(week_daily, k):
        return uniform_matrix(n)

    counts = np.zeros((n, n), dtype=np.float64)
    days = sorted(week_daily.keys())
    n_transitions = 0
    for d in days:
        d_next = d + 1
        if d_next in week_daily:
            vec_t = week_daily[d]
            vec_t1 = week_daily[d_next]
            counts += np.outer(vec_t, vec_t1)
            n_transitions += 1

    if n_transitions == 0:
        return uniform_matrix(n)

    counts += epsilon
    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = counts / row_sums
    return matrix