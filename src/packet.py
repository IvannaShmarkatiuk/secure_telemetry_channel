import struct

from crypto import aes_ctr, derive_keys, make_tag, verify_tag
from frame import Frame
from huffman import (HuffmanError, HuffmanTrailingDataError, encode_with_lengths, huffman_decode,
                     huffman_encode)

MAGIC = b"TLM1"
VERSION = 1
HEADER_LEN = 43
TAG_LEN = 32
MIN_PACKET_LEN = HEADER_LEN + TAG_LEN  # 75 = заголовок + порожній шифротекст + HMAC

# Заголовок (розділ 2.1). ">" означає big-endian без вирівнювання.
# 4s magic, B version, H device_id, I packet_seq, 16s salt, 12s nonce, I ciphertext_len
HEADER_FORMAT = ">4sBHI16s12sI"

# Навантаження (розділ 3): початок і середина, між ними таблиця довжин кодів.
# H frame_count, B flags, B reserved, H table_nsym
PAYLOAD_START_FORMAT = ">HBBH"
# I symbol_count, I bitstream_len
PAYLOAD_COUNTS_FORMAT = ">II"
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08
# Біти flags, яких наш приймач не знає: 0x01, 0x02 (командні розширення) і 0x10..0x80 (reserved)
UNKNOWN_FLAGS = 0xFF & ~(FLAG_CLASS_TABLE | FLAG_NO_COMPRESSION)


class PacketError(Exception):
    """Відмова приймача; .code — один із кодів розділу 6 FORMAT_SPEC."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def build_header(device_id: int, packet_seq: int, salt: bytes, nonce: bytes,
                 ciphertext_len: int) -> bytes:
    """Збирає відкритий заголовок, 43 байти."""
    return struct.pack(HEADER_FORMAT, MAGIC, VERSION, device_id, packet_seq, salt, nonce,
                       ciphertext_len)


def build_payload(frames: list[Frame], lengths=None) -> bytes:
    """Збирає навантаження з кадрів.

    lengths — таблиця вікна кандидатів (FORMAT_SPEC 3.5, п. 5): якщо передана, кадри кодуються
    саме нею і в пакет пишеться вся таблиця; інакше таблиця будується з самих кадрів.
    """
    data = b"".join(frame.to_bytes() for frame in frames)
    if lengths is None:
        lengths, bitstream = huffman_encode(data)
    else:
        bitstream = encode_with_lengths(data, lengths)
    # Таблиця: лише символи з ненульовою довжиною, за зростанням символу (розділ 3.2)
    symbols = sorted(s for s in lengths if lengths[s] > 0)
    table = b"".join(struct.pack(">BB", s, lengths[s]) for s in symbols)
    start = struct.pack(PAYLOAD_START_FORMAT, len(frames), 0, 0, len(symbols))
    counts = struct.pack(PAYLOAD_COUNTS_FORMAT, len(data), len(bitstream))
    return start + table + counts + bitstream


def parse_payload(payload: bytes) -> list[Frame]:
    """Розбирає розшифроване навантаження назад у кадри (розділ 3)."""
    if len(payload) < 6:
        raise PacketError("E_PAYLOAD")
    frame_count, flags, reserved, table_nsym = struct.unpack(PAYLOAD_START_FORMAT, payload[:6])
    if reserved != 0 or flags & UNKNOWN_FLAGS:
        raise PacketError("E_PAYLOAD")

    table_end = 6 + 2 * table_nsym
    if len(payload) < table_end + 8:
        raise PacketError("E_PAYLOAD")
    symbols = list(payload[6:table_end:2])
    # Символи мають іти строго за зростанням (розділ 3.2): це ловить і повтори, і безлад
    if any(a >= b for a, b in zip(symbols, symbols[1:])):
        raise PacketError("E_CODE")
    lengths = {payload[i]: payload[i + 1] for i in range(6, table_end, 2)}
    symbol_count, bitstream_len = struct.unpack(PAYLOAD_COUNTS_FORMAT, payload[table_end:table_end + 8])
    if symbol_count != 12 * frame_count:
        raise PacketError("E_PAYLOAD")

    bitstream_start = table_end + 8
    bitstream_end = bitstream_start + bitstream_len
    # class_table: по байту на кадр після потоку; нам не потрібна, тому пропускаємо
    class_table_len = frame_count if flags & FLAG_CLASS_TABLE else 0
    if len(payload) != bitstream_end + class_table_len:
        raise PacketError("E_PAYLOAD")
    bitstream = payload[bitstream_start:bitstream_end]

    if flags & FLAG_NO_COMPRESSION:
        # таблиці немає, у потоці рівно symbol_count сирих байтів
        if table_nsym != 0 or bitstream_len != symbol_count:
            raise PacketError("E_PAYLOAD")
        data = bitstream
    else:
        try:
            data = huffman_decode(lengths, bitstream, symbol_count)
        except HuffmanTrailingDataError:
            raise PacketError("E_PAYLOAD")  # зайві байти в потоці: неузгоджені довжини секцій
        except HuffmanError:
            raise PacketError("E_CODE")
    try:
        return [Frame.from_bytes(data[i:i + 12]) for i in range(0, len(data), 12)]
    except ValueError:
        raise PacketError("E_PAYLOAD")  # кадр із mode > 7 або critical > 1


def build_packet(frames: list[Frame], password: str, device_id: int, packet_seq: int,
                 salt: bytes, nonce: bytes, lengths=None) -> bytes:
    """Збирає повний пакет: заголовок + шифротекст + HMAC (розділи 2 і 4, Encrypt-then-MAC)."""
    # struct.pack мовчки доповнив би коротку сіль нулями, тому перевіряємо довжини самі
    if len(salt) != 16:
        raise ValueError("сіль має бути 16 байтів")
    if len(nonce) != 12:
        raise ValueError("nonce має бути 12 байтів")

    payload = build_payload(frames, lengths)
    k_enc, k_mac = derive_keys(password, salt)
    ciphertext = aes_ctr(k_enc, nonce, payload)
    header = build_header(device_id, packet_seq, salt, nonce, len(ciphertext))
    tag = make_tag(k_mac, header + ciphertext)
    return header + ciphertext + tag


def parse_packet(packet: bytes, password: str) -> list["Frame"]:
    # Крок 1: пакет не може бути коротшим за заголовок + HMAC
    if len(packet) < MIN_PACKET_LEN:
        raise PacketError("E_SHORT")

    # Крок 2: відкритий заголовок
    magic, version, device_id, packet_seq, salt, nonce, ciphertext_len = struct.unpack(
        HEADER_FORMAT, packet[:HEADER_LEN]
    )
    if magic != MAGIC or version != VERSION:
        raise PacketError("E_MAGIC")

    # Крок 3: фактична довжина має збігатися з тією, що записана в заголовку
    if len(packet) != HEADER_LEN + ciphertext_len + TAG_LEN:
        raise PacketError("E_LENGTH")

    # Кроки 4-5: HMAC перевіряємо ДО розшифрування. Невірний пароль теж дає E_HMAC.
    ciphertext_end = HEADER_LEN + ciphertext_len
    k_enc, k_mac = derive_keys(password, salt)
    if not verify_tag(k_mac, packet[:ciphertext_end], packet[ciphertext_end:]):
        raise PacketError("E_HMAC")

    # Кроки 6-8: розшифрувати, розібрати навантаження, відновити кадри
    payload = aes_ctr(k_enc, nonce, packet[HEADER_LEN:ciphertext_end])
    return parse_payload(payload)
