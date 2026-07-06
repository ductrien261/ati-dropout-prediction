import numpy as np
import pandas as pd

from src.ati.data.oulad_loader import (
    clean_studentvle, build_activity_index, build_daily_activity_table,
    build_labels, build_valid_day_range, filter_daily_activity, add_week_column,
)
from src.ati.data.splits import make_student_level_splits, attach_split


def make_fake_oulad():
    """Data giả tối giản: 3 sinh viên, đủ để test toàn bộ pipeline."""
    vle = pd.DataFrame({
        "id_site": [1, 2, 3],
        "activity_type": ["quiz", "resource", "forumng"],
    })

    svle = pd.DataFrame([
        # Student 1 (Withdrawn): hoạt động ngày 0,1 rồi rút học ngày 10
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 1, "date": 0, "sum_click": 10},
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 2, "date": 1, "sum_click": 5},
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 3, "date": 20, "sum_click": 3},  # sau khi rút học -> phải bị lọc

        # Student 2 (Pass): hoạt động đều đặn
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 1, "date": 0, "sum_click": 8},
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 2, "date": 1, "sum_click": 6},

        # Student 3 (Pass): có dòng trùng key -> phải bị gộp
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 3, "id_site": 1, "date": 0, "sum_click": 2},
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 3, "id_site": 1, "date": 0, "sum_click": 3},

        # Student 3: tương tác trước khai giảng -> phải bị lọc
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 3, "id_site": 2, "date": -5, "sum_click": 100},
    ])

    reg = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "date_unregistration": [10, np.nan, np.nan],
    })

    info = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "final_result": ["Withdrawn", "Pass", "Pass"],
    })

    return vle, svle, reg, info


def test_duplicate_rows_are_merged():
    vle, svle, reg, info = make_fake_oulad()
    svle_clean = clean_studentvle(svle)
    student3_day0 = svle_clean[
        (svle_clean["id_student"] == 3) & (svle_clean["date"] == 0)
    ]
    assert len(student3_day0) == 1
    assert student3_day0["sum_click"].iloc[0] == 5  # 2 + 3


def test_pre_course_start_rows_are_dropped():
    vle, svle, reg, info = make_fake_oulad()
    svle_clean = clean_studentvle(svle)
    activity_index = build_activity_index(vle, n_expected=3)
    daily = build_daily_activity_table(svle_clean, vle, activity_index)
    valid_range = build_valid_day_range(reg)
    filtered = filter_daily_activity(daily, valid_range)

    assert (filtered["date"] < 0).sum() == 0


def test_post_withdrawal_rows_are_dropped():
    vle, svle, reg, info = make_fake_oulad()
    svle_clean = clean_studentvle(svle)
    activity_index = build_activity_index(vle, n_expected=3)
    daily = build_daily_activity_table(svle_clean, vle, activity_index)
    valid_range = build_valid_day_range(reg)
    filtered = filter_daily_activity(daily, valid_range)

    student1_rows = filtered[filtered["id_student"] == 1]
    assert (student1_rows["date"] > 10).sum() == 0
    assert 20 not in student1_rows["date"].values


def test_labels_map_correctly():
    vle, svle, reg, info = make_fake_oulad()
    labels = build_labels(info)
    label_map = dict(zip(labels["id_student"], labels["label"]))
    assert label_map[1] == 1  # Withdrawn
    assert label_map[2] == 0  # Pass
    assert label_map[3] == 0  # Pass


def test_activity_index_size_check_raises_on_wrong_count():
    bad_vle = pd.DataFrame({"id_site": [1, 2], "activity_type": ["quiz", "resource"]})
    try:
        build_activity_index(bad_vle)
        assert False, "Phải raise ValueError khi không đủ 20 loại activity"
    except ValueError:
        pass


def test_student_level_split_no_leakage():
    n_students = 40
    meta = pd.DataFrame({
        "code_module": ["AAA"] * (n_students * 2),
        "code_presentation": ["2013J"] * (n_students * 2),
        "id_student": np.repeat(np.arange(n_students), 2),
        "week": list(range(2)) * n_students,
        "label": np.repeat([0, 1] * (n_students // 2), 2),
    })
    split_table = make_student_level_splits(
        meta, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=0
    )
    assert split_table["id_student"].duplicated().sum() == 0

    meta_with_split = attach_split(meta, split_table)
    for student_id, group in meta_with_split.groupby("id_student"):
        assert group["split"].nunique() == 1