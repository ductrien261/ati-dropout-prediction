import numpy as np
from src.ati.transforms.gaf import build_gaf_image
from src.ati.transforms.rp import build_rp_image
from src.ati.transforms.mtf import build_mtf_image
from src.ati.transforms.heatmap import build_heatmap_image
from src.ati.transforms.radar import build_radar_image
from src.ati.transforms.common import N_ACTIVITIES


def make_vec(active_idx, click, n=N_ACTIVITIES):
    v = np.zeros(n)
    v[active_idx] = click
    return v


def make_dense_week():
    return {d: make_vec(d % N_ACTIVITIES, 20) for d in range(7)}


def test_empty_week_returns_zero_for_all_baselines():
    empty = {}
    assert build_gaf_image(empty).sum() == 0
    assert build_rp_image(empty).sum() == 0
    assert build_mtf_image(empty).sum() == 0
    assert build_heatmap_image(empty).sum() == 0
    assert build_radar_image(empty).sum() == 0


def test_gaf_shape_and_no_nan():
    img = build_gaf_image(make_dense_week())
    assert img.shape == (N_ACTIVITIES, 7, 7)
    assert not np.isnan(img).any()


def test_rp_shape_and_no_nan():
    img = build_rp_image(make_dense_week())
    assert img.shape == (N_ACTIVITIES, 7, 7)
    assert not np.isnan(img).any()


def test_mtf_shape_and_no_nan():
    img = build_mtf_image(make_dense_week())
    assert img.shape == (N_ACTIVITIES, 7, 7)
    assert not np.isnan(img).any()


def test_heatmap_shape_and_normalized():
    img = build_heatmap_image(make_dense_week())
    assert img.shape == (4, 5)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert abs(img.sum() - 1.0) < 1e-5 


def test_radar_shape_and_range():
    img = build_radar_image(make_dense_week())
    assert img.shape == (64, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert img.sum() > 0 