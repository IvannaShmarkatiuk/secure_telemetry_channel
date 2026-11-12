import random

import pytest

from huffman import (
    BitReader,
    BitWriter,
    HuffmanError,
    HuffmanTrailingDataError,
    build_code_lengths,
    canonical_codes,
    count_frequencies,
    encode_with_lengths,
    frame_weight,
    huffman_decode,
    huffman_encode,
    limited_code_lengths,
    window_lengths,
)

# Контрольний приклад, FORMAT_SPEC розділ 7
SPEC_FRAME = bytes.fromhex("00000000 03E8 00D7 FFFD 01 01")
SPEC_LENGTHS = {0x00: 1, 0x01: 3, 0xFF: 3, 0x03: 4, 0xD7: 4, 0xE8: 4, 0xFD: 4}
SPEC_BITSTREAM = bytes.fromhex("0CE6DF90")


# ---------- Крок 1. Частоти ----------

def test_frequencies_empty():
    assert count_frequencies(b"") == {}


def test_frequencies_spec_frame():
    assert count_frequencies(SPEC_FRAME) == {
        0x00: 5, 0x01: 2, 0xFF: 1, 0x03: 1, 0xD7: 1, 0xE8: 1, 0xFD: 1,
    }


# ---------- Крок 2. Довжини кодів (дерево через heapq) ----------

def kraft_sum(lengths):
    return sum(2 ** -L for L in lengths.values())


def test_lengths_empty():
    assert build_code_lengths({}) == {}


def test_lengths_single_symbol_is_one():
    assert build_code_lengths({0x41: 10}) == {0x41: 1}


def test_lengths_two_symbols():
    assert build_code_lengths({0x61: 3, 0x62: 1}) == {0x61: 1, 0x62: 1}


def test_lengths_spec_frame_optimal():
    freqs = count_frequencies(SPEC_FRAME)
    lengths = build_code_lengths(freqs)
    assert set(lengths) == set(freqs)
    assert kraft_sum(lengths) == 1          # повне префіксне дерево
    assert sum(freqs[s] * lengths[s] for s in freqs) == 30  # оптимум: 30 біт


# ---------- Крок 3. Обмеження довжини до 15 біт ----------

def fibonacci_freqs(n):
    a, b = 1, 1
    freqs = {}
    for sym in range(n):
        freqs[sym] = a
        a, b = b, a + b
    return freqs


def test_fibonacci_tree_is_too_deep_without_limit():
    # переконуємося, що тест справді перевіряє обмеження
    assert max(build_code_lengths(fibonacci_freqs(20)).values()) > 15


def test_limited_lengths_at_most_15():
    freqs = fibonacci_freqs(20)
    lengths = limited_code_lengths(freqs)
    assert max(lengths.values()) <= 15
    assert set(lengths) == set(freqs)   # жоден символ не злитий і не втрачений
    assert kraft_sum(lengths) <= 1      # префіксний код існує


def test_limited_lengths_unchanged_when_not_needed():
    freqs = count_frequencies(SPEC_FRAME)
    assert limited_code_lengths(freqs) == build_code_lengths(freqs)


# ---------- Крок 4. Канонічні коди (розділ 3.3) ----------

def test_canonical_codes_spec_example():
    codes = canonical_codes(SPEC_LENGTHS)
    as_text = {s: format(codes[s], f"0{SPEC_LENGTHS[s]}b") for s in codes}
    assert as_text == {
        0x00: "0", 0x01: "100", 0xFF: "101",
        0x03: "1100", 0xD7: "1101", 0xE8: "1110", 0xFD: "1111",
    }


def test_canonical_codes_single_symbol_is_zero():
    assert canonical_codes({0x41: 1}) == {0x41: 0}


# ---------- Крок 5. BitWriter / BitReader ----------

def test_bitwriter_first_bit_is_0x80():
    w = BitWriter()
    w.write(1, 1)
    assert w.getvalue() == b"\x80"


def test_bitwriter_msb_first_and_full_byte():
    w = BitWriter()
    w.write(0b101, 3)
    w.write(0b11111, 5)
    assert w.getvalue() == b"\xBF"     # 101 11111


def test_bitwriter_pads_last_byte_with_zeros():
    w = BitWriter()
    w.write(0b111111111, 9)
    assert w.getvalue() == b"\xFF\x80"


def test_bitwriter_empty():
    assert BitWriter().getvalue() == b""


def test_bitreader_reads_msb_first():
    r = BitReader(b"\xA0")              # 1010 0000
    assert [r.read_bit() for _ in range(8)] == [1, 0, 1, 0, 0, 0, 0, 0]


def test_bitreader_past_end_raises():
    r = BitReader(b"\x00")
    for _ in range(8):
        r.read_bit()
    with pytest.raises(HuffmanError):
        r.read_bit()


# ---------- Крок 6. huffman_encode / huffman_decode ----------

def test_empty_stream():
    lengths, bitstream = huffman_encode(b"")
    assert lengths == {} and bitstream == b""
    assert huffman_decode(lengths, bitstream, 0) == b""


def test_single_distinct_symbol():
    data = b"\x41" * 10
    lengths, bitstream = huffman_encode(data)
    assert lengths == {0x41: 1}               # довжина 1, код 0
    assert bitstream == b"\x00\x00"           # 10 нульових бітів + 6 нулів доповнення
    assert huffman_decode(lengths, bitstream, len(data)) == data


def test_two_symbols():
    data = b"aab"
    lengths, bitstream = huffman_encode(data)
    assert lengths == {0x61: 1, 0x62: 1}      # a -> 0, b -> 1
    assert bitstream == b"\x20"               # 001 00000
    assert huffman_decode(lengths, bitstream, len(data)) == data


def test_all_256_bytes_equiprobable():
    data = bytes(range(256)) * 4
    lengths, bitstream = huffman_encode(data)
    assert lengths == {s: 8 for s in range(256)}
    # усі довжини 8 -> канонічний код символу s дорівнює s
    assert bitstream == data
    assert huffman_decode(lengths, bitstream, len(data)) == data


@pytest.mark.parametrize("seed,size", [(1, 1), (2, 100), (3, 1200), (4, 5000)])
def test_random_round_trip(seed, size):
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(size))
    lengths, bitstream = huffman_encode(data)
    assert huffman_decode(lengths, bitstream, len(data)) == data


def test_round_trip_with_length_limit():
    data = b"".join(bytes([s]) * f for s, f in fibonacci_freqs(20).items())
    lengths, bitstream = huffman_encode(data)
    assert max(lengths.values()) <= 15
    assert huffman_decode(lengths, bitstream, len(data)) == data


def test_spec_example_bitstream():
    assert encode_with_lengths(SPEC_FRAME, SPEC_LENGTHS) == SPEC_BITSTREAM


def test_spec_example_decode():
    assert huffman_decode(SPEC_LENGTHS, SPEC_BITSTREAM, 12) == SPEC_FRAME


def test_spec_example_huffman_encode_is_optimal():
    lengths, bitstream = huffman_encode(SPEC_FRAME)
    freqs = count_frequencies(SPEC_FRAME)
    assert sum(freqs[s] * lengths[s] for s in freqs) == 30
    assert huffman_decode(lengths, bitstream, 12) == SPEC_FRAME


def test_decode_symbol_not_in_table():
    # у таблиці лише код "0"; після біта 1 жоден код не збігається
    with pytest.raises(HuffmanError, match="не відповідають"):
         huffman_decode({0x41: 1}, b"\x80\x00", 1)


def test_decode_truncated_stream():
    with pytest.raises(HuffmanError):
        huffman_decode(SPEC_LENGTHS, SPEC_BITSTREAM[:3], 12)


def test_decode_length_over_15():
    with pytest.raises(HuffmanError):
        huffman_decode({0x41: 16}, b"\x00\x00\x00", 1)


# ---------- Біти доповнення мають бути нулями (розділ 3.3) ----------

def test_bitreader_padding_is_zero():
    r = BitReader(b"\x90")              # 100100 00
    for _ in range(6):
        r.read_bit()
    assert r.padding_is_zero()


def test_bitreader_padding_not_zero():
    r = BitReader(b"\x91")              # 100100 01
    for _ in range(6):
        r.read_bit()
    assert not r.padding_is_zero()


def test_bitreader_padding_on_byte_boundary():
    r = BitReader(b"\xFF")
    for _ in range(8):
        r.read_bit()
    assert r.padding_is_zero()          # доповнення немає


def test_decode_nonzero_padding_spec_example():
    bad = SPEC_BITSTREAM[:3] + b"\x91"  # останній біт доповнення = 1
    with pytest.raises(HuffmanError):
        huffman_decode(SPEC_LENGTHS, bad, 12)


def test_decode_nonzero_padding_single_symbol():
    # 10 нульових бітів даних, у доповненні другого байта одиниця
    with pytest.raises(HuffmanError):
        huffman_decode({0x41: 1}, b"\x00\x01", 10)



# ---------- Повнота таблиці (нерівність Крафта) і зайві байти ----------

def test_decode_kraft_over_one():
    # три коди довжини 1 неможливі: сума Крафта 3/2
    with pytest.raises(HuffmanError, match="Крафта"):
        huffman_decode({1: 1, 2: 1, 3: 1}, b"\x00", 1)


def test_decode_kraft_under_one():
    # 1/2 + 1/4 = 3/4: дерево неповне, Гаффман такого не дає
    with pytest.raises(HuffmanError, match="Крафта"):
        huffman_decode({1: 1, 2: 2}, b"\x00", 1)


def test_decode_single_symbol_length_one_is_valid():
    # вироджений випадок: сума 1/2
    assert huffman_decode({0x41: 1}, b"\x00", 3) == b"AAA"


def test_decode_single_symbol_wrong_length():
    with pytest.raises(HuffmanError, match="довжину 1"):
        huffman_decode({0x41: 2}, b"\x00", 1)


def test_decode_extra_byte_after_data():
    with pytest.raises(HuffmanTrailingDataError):
        huffman_decode(SPEC_LENGTHS, SPEC_BITSTREAM + b"\x00", 12)


def test_trailing_error_is_huffman_error():
    assert issubclass(HuffmanTrailingDataError, HuffmanError)




# ---------- Чекпоінт 2: ваги кадрів для рюкзака ----------

def test_frame_weight_spec_frame():
    # розділ 7: кадр за своєю таблицею займає 30 біт
    assert frame_weight(SPEC_FRAME, SPEC_LENGTHS) == 30


def test_window_lengths_empty():
    assert window_lengths([]) == {}


def test_window_lengths_single_frame_matches_spec():
    lengths = window_lengths([SPEC_FRAME])
    freqs = count_frequencies(SPEC_FRAME)
    assert sum(freqs[s] * lengths[s] for s in freqs) == 30


def test_window_table_is_complete_and_limited():
    rng = random.Random(5)
    frames = [bytes(rng.randrange(256) for _ in range(12)) for _ in range(64)]
    lengths = window_lengths(frames)
    assert max(lengths.values()) <= 15
    assert kraft_sum(lengths) == 1
    assert set(lengths) == set(b"".join(frames))


def test_weights_sum_equals_encoded_size():
    # ваги рахуються за тією самою таблицею, якою потім кодуються кадри,
    # тому сума ваг = фактична довжина бітового потоку 
    rng = random.Random(6)
    frames = [bytes(rng.choice(b"\x00\x00\x00\x01\x03\xff") for _ in range(12)) for _ in range(64)]
    lengths = window_lengths(frames)
    chosen = frames[::3]
    total_bits = sum(frame_weight(f, lengths) for f in chosen)
    stream = encode_with_lengths(b"".join(chosen), lengths)
    assert len(stream) == (total_bits + 7) // 8