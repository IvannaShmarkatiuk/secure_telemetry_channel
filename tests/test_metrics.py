import pytest

from frame import Frame
from metrics import critical_recovery_r, efficiency_s, quality_q, speed_t


def test_efficiency_nothing_sent_is_one():
    assert efficiency_s(1000, 0) == 1.0


def test_efficiency_equal_is_half():
    assert efficiency_s(1000, 1000) == pytest.approx(0.5)


def test_efficiency_less_sent_is_better():
    assert efficiency_s(1000, 500) > efficiency_s(1000, 2000)


def test_speed_zero_time_is_one():
    assert speed_t(20.0, 0.0) == 1.0


def test_speed_equal_to_reference_is_half():
    assert speed_t(20.0, 20.0) == pytest.approx(0.5)


def _frames():
    return [
        Frame(0, 0, 0, 0, 0, 1),
        Frame(1, 0, 0, 0, 0, 0),
        Frame(2, 0, 0, 0, 0, 1),
        Frame(3, 0, 0, 0, 0, 1),
        Frame(4, 0, 0, 0, 0, 0),
    ]


def test_recovery_all_restored_is_one():
    frames = _frames()
    assert critical_recovery_r(frames, frames) == 1.0


def test_recovery_nothing_restored_is_zero():
    assert critical_recovery_r(_frames(), []) == 0.0


def test_recovery_partial():
    frames = _frames()
    restored = [frames[0], frames[1], frames[4]]
    assert critical_recovery_r(frames, restored) == pytest.approx(1 / 3)


def test_recovery_ignores_normal_frames():
    frames = _frames()
    normal_only = [f for f in frames if f.critical == 0]
    assert critical_recovery_r(frames, normal_only) == 0.0


def test_recovery_no_critical_in_original_is_one():
    normal = [Frame(0, 0, 0, 0, 0, 0)]
    assert critical_recovery_r(normal, []) == 1.0


def test_quality_formula():
    q = quality_q(0.6, 0.8, 0.9, 0.5, 0.35, 0.15)
    assert q == pytest.approx(0.5 * 0.6 + 0.35 * 0.8 + 0.15 * 0.9)


def test_quality_is_zero_when_no_critical_recovered():
    assert quality_q(0.9, 0.0, 0.9, 0.5, 0.35, 0.15) == 0.0