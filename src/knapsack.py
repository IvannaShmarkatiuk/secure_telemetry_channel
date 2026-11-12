"""Пакування кадрів у пакет: рюкзак 0/1 (FORMAT_SPEC, розділ 3.5).

Ваги кадрів рахує huffman.py (window_lengths, frame_weight),
цінності — run_session.py (frame_value). Тут лише місткість і вибір кадрів.
"""

FIXED_PAYLOAD_BYTES = 14  # 6 байтів початку навантаження + 8 байтів лічильників


def capacity_bits(max_payload_bytes: int, table_nsym: int) -> int:
    """Місткість рюкзака в бітах: W = 8 * (max_payload_bytes - 14 - 2 * table_nsym)."""
    return 8 * (max_payload_bytes - FIXED_PAYLOAD_BYTES - 2 * table_nsym)

def knapsack_dp(weights: list[int], values: list[int], capacity: int) -> list[int]:
    """Рюкзак 0/1 через ДП, O(n·W). Повертає індекси обраних кадрів за зростанням."""
    if capacity < 0:
        return []
    n = len(weights)
    # best[c] — найбільша цінність, яку можна набрати з місткістю c біт
    best = [0] * (capacity + 1)
    # took[i][c] — чи вигідно було взяти кадр i, коли місткість c
    took = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        w, v = weights[i], values[i]
        # c іде згори вниз, щоб кожен кадр узяти не більше одного разу
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
                took[i][c] = True

    # Відновлюємо вибір: ідемо від останнього кадру до першого
    chosen = []
    c = capacity
    for i in range(n - 1, -1, -1):
        if took[i][c]:
            chosen.append(i)
            c -= weights[i]
    return sorted(chosen)    

def knapsack_greedy(weights: list[int], values: list[int], capacity: int) -> list[int]:
    """Жадібний рюкзак за v/w (спадання). Повертає індекси обраних кадрів за зростанням."""
    # Найвигідніші кадри — першими; sorted стабільний, тож за рівної вигідності
    # раніше йде кадр, що стоїть раніше у списку
    order = sorted(range(len(weights)), key=lambda i: values[i] / weights[i], reverse=True)
    chosen = []
    left = capacity
    for i in order:
        if weights[i] <= left:  # влазить — беремо, ні — пропускаємо і дивимося далі
            chosen.append(i)
            left -= weights[i]
    return sorted(chosen)    