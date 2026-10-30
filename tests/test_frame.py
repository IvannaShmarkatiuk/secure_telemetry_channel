from frame import Frame
import pytest


def test_spec_frame_to_bytes():
    f = Frame(0, 1000, 215, -3, 1, 1)
    assert f.to_bytes() == bytes.fromhex("00000000" "03E8" "00D7" "FFFD" "01" "01")

def test_spec_frame_from_bytes():
    data = bytes.fromhex("00000000" "03E8" "00D7" "FFFD" "01" "01")
    f = Frame.from_bytes(data)
    assert f == Frame(0, 1000, 215, -3, 1, 1)

def test_round_trip():
    f = Frame(123456, -500, -35, 7, 3, 0)
    assert Frame.from_bytes(f.to_bytes()) == f

def test_from_bytes_wrong_length():
    with pytest.raises(ValueError):
        Frame.from_bytes(b"123")