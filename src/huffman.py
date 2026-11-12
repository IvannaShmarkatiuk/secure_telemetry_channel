"""Кодування Гаффмана за FORMAT_SPEC_1, розділи 3.2 і 3.3."""

import heapq

MAX_CODE_LEN = 15  # розділ 3.2: максимальна довжина коду


class HuffmanError(Exception):
    """Помилка декодування; пакетний рівень перетворює її на E_CODE."""


class HuffmanTrailingDataError(HuffmanError):
    """Зайві байти в потоці після закодованих даних; пакетний рівень дає E_PAYLOAD."""


# ---------- Крок 1. Частоти ----------

def count_frequencies(data: bytes) -> dict[int, int]:
    freqs: dict[int, int] = {}
    for b in data:
        freqs[b] = freqs.get(b, 0) + 1
    return freqs


# ---------- Крок 2. Довжини кодів (дерево через heapq) ----------


def build_code_lengths(freqs: dict[int, int]) -> dict[int, int]:
    if not freqs:
        return {}
    if len(freqs) == 1:
        (sym,) = freqs
        return {sym: 1}  # розділ 3.2: вироджений випадок, довжина 1

    heap = [(f, sym, [sym]) for sym, f in freqs.items()]
    heapq.heapify(heap)
    lengths = {sym: 0 for sym in freqs}
    next_id = 256  # ключі внутрішніх вузлів більші за будь-який символ

    while len(heap) > 1:
        f1, _, syms1 = heapq.heappop(heap)
        f2, _, syms2 = heapq.heappop(heap)
        for s in syms1 + syms2:
            lengths[s] += 1
        heapq.heappush(heap, (f1 + f2, next_id, syms1 + syms2))
        next_id += 1
    return lengths


# ---------- Крок 3. Обмеження довжини до 15 біт ----------

def limited_code_lengths(freqs: dict[int, int]) -> dict[int, int]:
    lengths = build_code_lengths(freqs)
    current = dict(freqs)
    while lengths and max(lengths.values()) > MAX_CODE_LEN:
        current = {s: max(1, f // 2) for s, f in current.items()}
        lengths = build_code_lengths(current)
    return lengths


# ---------- Крок 4. Канонічні коди (розділ 3.3) ----------

def canonical_codes(lengths: dict[int, int]) -> dict[int, int]:
    bl_count = [0] * (MAX_CODE_LEN + 1)
    for L in lengths.values():
        bl_count[L] += 1
    bl_count[0] = 0

    next_code = [0] * (MAX_CODE_LEN + 1)
    code = 0
    for L in range(1, MAX_CODE_LEN + 1):
        code = (code + bl_count[L - 1]) << 1
        next_code[L] = code

    codes: dict[int, int] = {}
    for s in sorted(lengths):
        L = lengths[s]
        codes[s] = next_code[L]
        next_code[L] += 1
    return codes


# ---------- Крок 5. BitWriter / BitReader ----------

class BitWriter:
    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0      # біти неповного байта
        self._nbits = 0    # скільки їх

    def write(self, code: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            self._acc = (self._acc << 1) | bit
            self._nbits += 1
            if self._nbits == 8:
                self._out.append(self._acc)
                self._acc = 0
                self._nbits = 0

    def getvalue(self) -> bytes:
        if self._nbits == 0:
            return bytes(self._out)
        return bytes(self._out) + bytes([self._acc << (8 - self._nbits)])


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0      # номер наступного біта

    def read_bit(self) -> int:
        if self._pos >= len(self._data) * 8:
            raise HuffmanError("бітовий потік обірвано")
        byte = self._data[self._pos // 8]
        bit = (byte >> (7 - self._pos % 8)) & 1
        self._pos += 1
        return bit

    def bytes_used(self) -> int:
        return (self._pos + 7) // 8        # скільки байтів зачеплено прочитаними бітами

    def padding_is_zero(self) -> bool:
        used = self._pos % 8               # скільки бітів поточного байта прочитано
        if used == 0:
            return True                    # стоїмо на межі байта: доповнення немає
        byte = self._data[self._pos // 8]
        mask = (1 << (8 - used)) - 1       # одиниці на місцях доповнення
        return byte & mask == 0


# ---------- Крок 6. huffman_encode / huffman_decode ----------

def encode_with_lengths(data: bytes, lengths: dict[int, int]) -> bytes:
    codes = canonical_codes(lengths)
    writer = BitWriter()
    for b in data:
        writer.write(codes[b], lengths[b])
    return writer.getvalue()


def huffman_encode(data: bytes) -> tuple[dict[int, int], bytes]:
    lengths = limited_code_lengths(count_frequencies(data))
    return lengths, encode_with_lengths(data, lengths)


def huffman_decode(lengths: dict[int, int], bitstream: bytes,
                   symbol_count: int) -> bytes:
    for s, L in lengths.items():
        if not (0 <= s <= 255) or not (1 <= L <= MAX_CODE_LEN):
            raise HuffmanError(f"недопустимий запис таблиці: {s}:{L}")

    if len(lengths) == 1:
        if next(iter(lengths.values())) != 1:     # розділ 3.2: один символ -> довжина 1
            raise HuffmanError("єдиний символ має мати довжину 1")
    elif lengths:
        kraft = sum(1 << (MAX_CODE_LEN - L) for L in lengths.values())
        if kraft != 1 << MAX_CODE_LEN:            # сума 2^(-L) має дорівнювати 1
            raise HuffmanError("таблиця не утворює повного префіксного дерева (сума Крафта ≠ 1)")

    codes = canonical_codes(lengths)
    lookup = {(lengths[s], c): s for s, c in codes.items()}

    reader = BitReader(bitstream)
    out = bytearray()
    for _ in range(symbol_count):
        code = 0
        for L in range(1, MAX_CODE_LEN + 1):
            code = (code << 1) | reader.read_bit()
            sym = lookup.get((L, code))
            if sym is not None:
                out.append(sym)
                break
        else:
            raise HuffmanError("біти не відповідають жодному символу таблиці")

    if not reader.padding_is_zero():       # розділ 3.3: доповнення мають бути нулі
        raise HuffmanError("ненульові біти доповнення")
    if reader.bytes_used() != len(bitstream):   # розділ 3.3: доповнюється лише останній байт
        raise HuffmanTrailingDataError("зайві байти після закодованих даних")
    return bytes(out)