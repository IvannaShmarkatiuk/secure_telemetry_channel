import json
import random
from itertools import combinations
from pathlib import Path

from knapsack import capacity_bits, knapsack_dp

REF_DIR = Path(__file__).parent.parent / "data" / "reference"


def test_capacity_formula():
    # W = 8 * (437 - 14 - 2*129) = 8 * 165 = 1320
    assert capacity_bits(437, 129) == 1320


def test_capacity_for_ref04_window():
    # у ref_04 рівно 64 кадри — це одне вікно кандидатів; його таблиця має 129 символів
    meta = json.loads((REF_DIR / "ref_04_session.json").read_text(encoding="utf-8"))
    assert capacity_bits(437, len(meta["таблиця_кодів"])) == 1320


def test_dp_beats_greedy_example():
    # А(3, 5), Б(4, 6), В(4, 6), W = 8: найкраще Б + В = 12, а не А + Б = 11
    assert knapsack_dp([3, 4, 4], [5, 6, 6], 8) == [1, 2]


def test_dp_nothing_fits():
    assert knapsack_dp([10, 20], [5, 5], 9) == []


def test_dp_everything_fits():
    assert knapsack_dp([1, 2, 3], [1, 1, 1], 100) == [0, 1, 2]


def test_dp_empty_input():
    assert knapsack_dp([], [], 50) == []


def test_dp_matches_brute_force():
    # на маленьких випадкових наборах перебираємо ВСІ підмножини і порівнюємо з ДП
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(0, 8)
        weights = [rng.randint(1, 15) for _ in range(n)]
        values = [rng.randint(1, 10) for _ in range(n)]
        capacity = rng.randint(0, 40)

        best = 0
        for k in range(n + 1):
            for subset in combinations(range(n), k):
                if sum(weights[i] for i in subset) <= capacity:
                    best = max(best, sum(values[i] for i in subset))

        chosen = knapsack_dp(weights, values, capacity)
        assert sum(weights[i] for i in chosen) <= capacity
        assert sum(values[i] for i in chosen) == best    