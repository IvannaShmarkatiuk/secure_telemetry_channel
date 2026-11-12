import json
from pathlib import Path

from frame import Frame
from knapsack import capacity_bits, frame_weight, window_lengths

REF_DIR = Path(__file__).parent.parent / "data" / "reference"


def load_ref04():
    """Кадри і таблиця кодів з ref_04."""
    meta = json.loads((REF_DIR / "ref_04_session.json").read_text(encoding="utf-8"))
    frames = [Frame(**f) for f in meta["кадри"]]
    table = {t["символ"]: t["довжина"] for t in meta["таблиця_кодів"]}
    return frames, table


def test_capacity_formula():
    # W = 8 * (437 - 14 - 2*129) = 8 * 165 = 1320
    assert capacity_bits(437, 129) == 1320


def test_frame_weight_is_sum_of_code_lengths():
    frame = Frame(1, 0, 0, 0, 0, 0)
    # байти кадру: 00 00 00 01 | 00 00 | 00 00 | 00 00 | 00 | 00
    lengths = {0x00: 1, 0x01: 4}
    # 11 нулів по 1 біту + один байт 0x01 на 4 біти
    assert frame_weight(frame, lengths) == 11 * 1 + 4


def test_window_table_matches_teacher_ref04():
    frames, table = load_ref04()
    assert window_lengths(frames) == table


def test_ref04_total_weight_and_capacity():
    frames, _ = load_ref04()
    lengths = window_lengths(frames)
    assert len(lengths) == 129
    assert sum(frame_weight(f, lengths) for f in frames) == 3389
    assert capacity_bits(437, len(lengths)) == 1320
