from frame import Frame
from huffman import count_frequencies, limited_code_lengths

FIXED_PAYLOAD_BYTES = 14  # 6 байтів початку навантаження + 8 байтів лічильників


def window_lengths(frames: list[Frame]) -> dict[int, int]:
    """Одна таблиця Гаффмана {байт: довжина коду} для всього вікна кандидатів."""
    data = b"".join(frame.to_bytes() for frame in frames)
    return limited_code_lengths(count_frequencies(data))


def frame_weight(frame: Frame, lengths: dict[int, int]) -> int:
    """Вага кадру: скільки біт займуть його 12 байтів за таблицею вікна."""
    return sum(lengths[b] for b in frame.to_bytes())


def capacity_bits(max_payload_bytes: int, table_nsym: int) -> int:
    """Місткість рюкзака в бітах: W = 8 * (max_payload_bytes - 14 - 2 * table_nsym)."""
    return 8 * (max_payload_bytes - FIXED_PAYLOAD_BYTES - 2 * table_nsym)
