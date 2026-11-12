"""Пакування кадрів у пакет: рюкзак 0/1 (FORMAT_SPEC, розділ 3.5).

Ваги кадрів рахує huffman.py (window_lengths, frame_weight),
цінності — run_session.py (frame_value). Тут лише місткість і вибір кадрів.
"""

FIXED_PAYLOAD_BYTES = 14  # 6 байтів початку навантаження + 8 байтів лічильників


def capacity_bits(max_payload_bytes: int, table_nsym: int) -> int:
    """Місткість рюкзака в бітах: W = 8 * (max_payload_bytes - 14 - 2 * table_nsym)."""
    return 8 * (max_payload_bytes - FIXED_PAYLOAD_BYTES - 2 * table_nsym)