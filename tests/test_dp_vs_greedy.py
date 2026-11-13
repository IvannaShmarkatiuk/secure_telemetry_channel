from dp_vs_greedy import compare_windows
from run_session import load_config, load_frames


def test_one_row_per_window():
    rows = compare_windows(load_frames(), load_config())
    assert len(rows) == 47  # 3000 кадрів / 64 = 46 повних вікон + 1 неповне


def test_dp_value_never_below_greedy():
    for row in compare_windows(load_frames(), load_config()):
        assert row["value_dp"] >= row["value_greedy"]


def test_both_choices_fit_capacity():
    for row in compare_windows(load_frames(), load_config()):
        assert row["weight_dp"] <= row["capacity_bits"]
        assert row["weight_greedy"] <= row["capacity_bits"]
