import numpy as np
from src.ati.transforms.ati_v1 import (
    build_transition_matrix, uniform_matrix, N_ACTIVITIES
)


def make_vec(active_idx, click, n=N_ACTIVITIES):
    v = np.zeros(n)
    v[active_idx] = click
    return v


def test_uniform_matrix_rows_sum_to_one():
    m = uniform_matrix(5)
    assert m.shape == (5, 5)
    np.testing.assert_allclose(m.sum(axis=1), np.ones(5))


def test_sparse_week_returns_uniform():
    week_daily = {0: make_vec(0, 2)}  # tổng click = 2 < ngưỡng 5
    m = build_transition_matrix(week_daily)
    np.testing.assert_allclose(m, uniform_matrix())


def test_empty_week_returns_uniform():
    m = build_transition_matrix({})
    np.testing.assert_allclose(m, uniform_matrix())


def test_rows_sum_to_one_for_dense_week():
    week_daily = {0: make_vec(0, 10), 1: make_vec(1, 10), 2: make_vec(2, 10)}
    m = build_transition_matrix(week_daily)
    np.testing.assert_allclose(m.sum(axis=1), np.ones(N_ACTIVITIES), atol=1e-6)


def test_known_transition_direction():
    # ngày 0 chỉ có activity 0; ngày 1 chỉ có activity 1
    # -> hàng 0 phải tập trung vào cột 1
    week_daily = {0: make_vec(0, 100), 1: make_vec(1, 100)}
    m = build_transition_matrix(week_daily)
    assert np.argmax(m[0]) == 1


def test_non_adjacent_days_do_not_transition():
    # ngày 0 và ngày 2 KHÔNG liền kề (thiếu ngày 1) -> không tạo transition
    week_daily = {0: make_vec(0, 100), 2: make_vec(1, 100)}
    m = build_transition_matrix(week_daily)
    np.testing.assert_allclose(m, uniform_matrix(), atol=1e-6)