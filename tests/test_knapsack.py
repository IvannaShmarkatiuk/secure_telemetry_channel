import csv
import json
import random
from itertools import combinations
from pathlib import Path

from frame import Frame
from huffman import frame_weight, window_lengths
from knapsack import capacity_bits, knapsack_dp, knapsack_greedy
from packet import build_packet, parse_packet
from run_session import frame_value

REF_DIR = Path(__file__).parent.parent / "data" / "reference"


def test_capacity_formula():
    # W = 8 * (437 - 14 - 2*129) = 8 * 165 = 1320
    assert capacity_bits(437, 129) == 1320


def test_capacity_for_ref04_window():
    # у ref_04 рівно 64 кадри — це одне вікно кандидатів; його таблиця має 129 символів
    meta = json.loads((REF_DIR / "ref_04_session.json").read_text(encoding="utf-8"))
    assert capacity_bits(437, len(meta["таблиця_кодів"])) == 1320


# ---------- Крок 3: ДП ----------

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


# ---------- Крок 4: жадібний ----------

def test_greedy_example_is_worse_than_dp():
    # той самий приклад: жадібний бере А + Б = 11 (ДП дає Б + В = 12)
    assert knapsack_greedy([3, 4, 4], [5, 6, 6], 8) == [0, 1]


def test_greedy_skips_heavy_and_takes_lighter():
    # вигідність: 2.0, 1.5, 1.0; кадр 1 не влазить, але кадр 2 після нього — так
    assert knapsack_greedy([5, 10, 2], [10, 15, 2], 8) == [0, 2]


def test_greedy_nothing_fits():
    assert knapsack_greedy([10, 20], [5, 5], 9) == []


def test_greedy_empty_input():
    assert knapsack_greedy([], [], 50) == []


def test_greedy_negative_capacity():
    assert knapsack_greedy([1], [1], -8) == []


# ---------- Крок 5: ДП не гірший за жадібний (кейс C1 №8) ----------

def total(values, chosen):
    return sum(values[i] for i in chosen)


def test_dp_never_worse_than_greedy_random():
    rng = random.Random(2)
    for _ in range(300):
        n = rng.randint(0, 64)
        weights = [rng.randint(30, 70) for _ in range(n)]       # як реальні ваги кадрів у бітах
        values = [rng.choice([1, 3, 5, 6, 8, 10]) for _ in range(n)]  # усі можливі цінності
        capacity = rng.randint(0, 1400)
        dp = knapsack_dp(weights, values, capacity)
        greedy = knapsack_greedy(weights, values, capacity)
        assert sum(weights[i] for i in dp) <= capacity
        assert sum(weights[i] for i in greedy) <= capacity
        assert total(values, dp) >= total(values, greedy)


def test_dp_never_worse_than_greedy_on_session():
    # справжній сеанс: вікна по 64 кадри, ваги за таблицею вікна, цінність за справжнім класом
    with open(REF_DIR.parent / "session.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for start in range(0, len(rows), 64):
        window = rows[start:start + 64]
        frames = [Frame(*(int(r[k]) for k in
                          ("timestamp_ms", "altitude_dm", "temp_c10", "vibration", "mode", "critical")))
                  for r in window]
        frames_bytes = [fr.to_bytes() for fr in frames]
        lengths = window_lengths(frames_bytes)
        weights = [frame_weight(b, lengths) for b in frames_bytes]
        values = [frame_value(int(r["label"]), int(r["critical"])) for r in window]
        capacity = capacity_bits(437, len(lengths))
        dp = knapsack_dp(weights, values, capacity)
        greedy = knapsack_greedy(weights, values, capacity)
        assert total(values, dp) >= total(values, greedy)


def test_dp_choice_fits_into_512_byte_packet():
    # вікно ref_04: 64 кадри; ДП обирає кадри, пакет кодуємо таблицею всього вікна
    meta = json.loads((REF_DIR / "ref_04_session.json").read_text(encoding="utf-8"))
    frames = [Frame(**f) for f in meta["кадри"]]
    frames_bytes = [fr.to_bytes() for fr in frames]
    lengths = window_lengths(frames_bytes)
    weights = [frame_weight(b, lengths) for b in frames_bytes]
    values = [frame_value(0, fr.critical) for fr in frames]
    chosen = knapsack_dp(weights, values, capacity_bits(437, len(lengths)))
    picked = [frames[i] for i in chosen]

    packet = build_packet(picked, "algo2026", 1, 0, bytes(16), bytes(12), lengths=lengths)

    assert len(packet) <= 512                                # ДП не помилився з розміром
    assert parse_packet(packet, "algo2026") == picked        # приймач читає кадри без втрат
