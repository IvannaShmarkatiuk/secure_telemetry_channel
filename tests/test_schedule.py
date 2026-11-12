import pytest

from run_session import frame_value, load_config, schedule


def test_values_match_priorities():
    assert frame_value(0, 0) == 1
    assert frame_value(2, 0) == 3
    assert frame_value(1, 0) == 5


def test_critical_bonus():
    assert frame_value(0, 1) == 6
    assert frame_value(1, 1) == 10


def test_values_follow_config():
    config = load_config()
    config["пріоритети_кадрів"]["норма"] = 7
    assert frame_value(0, 0, config) == 7


@pytest.mark.parametrize("cls", [-1, 3])
def test_invalid_class_rejected(cls):
    with pytest.raises(ValueError):
        frame_value(cls, 0)


@pytest.mark.parametrize("critical", [-1, 2])
def test_invalid_critical_rejected(critical):
    with pytest.raises(ValueError):
        frame_value(0, critical)


def test_schedule_most_valuable_first():
    groups = [([0, 1], 4), ([2, 3], 20), ([4], 9)]
    assert schedule(groups) == [[2, 3], [4], [0, 1]]


def test_schedule_ties_keep_original_order():
    groups = [([0], 5), ([1], 5), ([2], 5)]
    assert schedule(groups) == [[0], [1], [2]]


def test_schedule_empty():
    assert schedule([]) == []


def test_schedule_does_not_change_input():
    groups = [([0], 1), ([1], 9)]
    schedule(groups)
    assert groups == [([0], 1), ([1], 9)]