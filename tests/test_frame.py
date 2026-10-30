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

@pytest.mark.parametrize("mode", [-1, 8, 255])
def test_invalid_mode(mode):
    with pytest.raises(ValueError):
        Frame(0, 0, 0, 0, mode, 0)

@pytest.mark.parametrize("critical", [-1, 2, 255])
def test_invalid_critical(critical):
    with pytest.raises(ValueError):
        Frame(0, 0, 0, 0, 0, critical)

def test_from_bytes_invalid_mode():
    data = bytearray(Frame(0, 0, 0, 0, 0, 0).to_bytes())
    data[10] = 9                     # байт mode 
    with pytest.raises(ValueError):
        Frame.from_bytes(bytes(data))