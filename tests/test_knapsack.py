import json
from pathlib import Path

from knapsack import capacity_bits

REF_DIR = Path(__file__).parent.parent / "data" / "reference"


def test_capacity_formula():
    # W = 8 * (437 - 14 - 2*129) = 8 * 165 = 1320
    assert capacity_bits(437, 129) == 1320


def test_capacity_for_ref04_window():
    # у ref_04 рівно 64 кадри — це одне вікно кандидатів; його таблиця має 129 символів
    meta = json.loads((REF_DIR / "ref_04_session.json").read_text(encoding="utf-8"))
    assert capacity_bits(437, len(meta["таблиця_кодів"])) == 1320