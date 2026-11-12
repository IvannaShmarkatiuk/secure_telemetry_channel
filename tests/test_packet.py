import hashlib
import struct

import pytest

import packet
from frame import Frame
import huffman
from huffman import HuffmanError
from packet import (HEADER_FORMAT, HEADER_LEN, PacketError, build_header, build_packet,
                    build_payload, parse_packet, parse_payload)

# Контрольний пакет із FORMAT_SPEC, розділ 7 (ref_01_single_frame.bin), 107 байтів
REF_01 = bytes.fromhex(
    "544C4D31" "01" "0001" "00000000"
    "000102030405060708090A0B0C0D0E0F"
    "010203040506070800000000"
    "00000020"
    "8E51D20BA4B608DFA5B8A5748C541098" "9DB93D2BFD04DD1CEEB0C925DAE946AF"
    "DD881BA276FB275C767958046D3EB9B7" "5D4F7C95EFA3F8CAC0B9411E18461A8E"
)
PASSWORD = "algo2026"


def error_code(packet: bytes) -> str:
    with pytest.raises(PacketError) as info:
        parse_packet(packet, PASSWORD)
    return info.value.code


def test_ref_packet_has_expected_length():
    assert len(REF_01) == 107


def test_header_format_is_43_bytes():
    assert struct.calcsize(HEADER_FORMAT) == HEADER_LEN == 43


def test_packet_error_keeps_code():
    assert PacketError("E_HMAC").code == "E_HMAC"


def test_empty_packet_is_short():
    assert error_code(b"") == "E_SHORT"


def test_74_bytes_is_short():
    assert error_code(REF_01[:74]) == "E_SHORT"


def test_short_is_checked_before_magic():
    # 74 байти сміття: спершу має спрацювати перевірка довжини (крок 1), а не magic (крок 2)
    assert error_code(b"\xff" * 74) == "E_SHORT"


def test_bad_magic():
    assert error_code(b"X" + REF_01[1:]) == "E_MAGIC"


def test_bad_version():
    assert error_code(REF_01[:4] + b"\x02" + REF_01[5:]) == "E_MAGIC"


def test_truncated_packet():
    assert error_code(REF_01[:-1]) == "E_LENGTH"


def test_packet_with_extra_byte():
    assert error_code(REF_01 + b"\x00") == "E_LENGTH"


# ---------- Крок 1: навантаження (розділ 3) ----------
# Контрольний приклад, FORMAT_SPEC розділ 7
REF_FRAME = Frame(timestamp_ms=0, altitude_dm=1000, temp_c10=215, vibration=-3, mode=1, critical=1)
REF_FRAME_BYTES = bytes.fromhex("00000000 03E8 00D7 FFFD 01 01")
REF_LENGTHS = {0x00: 1, 0x01: 3, 0x03: 4, 0xD7: 4, 0xE8: 4, 0xFD: 4, 0xFF: 3}
REF_BITSTREAM = bytes.fromhex("0CE6DF90")
REF_PAYLOAD = bytes.fromhex(
    "0001" "00" "00" "0007"
    "0001 0103 0304 D704 E804 FD04 FF03"
    "0000000C" "00000004"
    "0CE6DF90"
)
EMPTY_PAYLOAD = bytes(14)  # frame_count = 0, усі поля нулі, таблиці й потоку немає


# Підробки Гаффмана: повертають те, що за специфікацією мав би повернути справжній модуль.
def fake_encode(data):
    if data == b"":
        return {}, b""
    assert data == REF_FRAME_BYTES
    return REF_LENGTHS, REF_BITSTREAM


def fake_decode(lengths, bitstream, symbol_count):
    if symbol_count == 0:
        return b""
    assert (lengths, bitstream, symbol_count) == (REF_LENGTHS, REF_BITSTREAM, 12)
    return REF_FRAME_BYTES


@pytest.fixture(autouse=True)
def fake_huffman(monkeypatch):
    monkeypatch.setattr(packet, "huffman_encode", fake_encode)
    monkeypatch.setattr(packet, "huffman_decode", fake_decode)


def payload_error_code(payload: bytes) -> str:
    with pytest.raises(PacketError) as info:
        parse_payload(payload)
    return info.value.code


def test_ref_payload_is_32_bytes():
    assert len(REF_PAYLOAD) == 32


def test_frame_stub_gives_ref_bytes():
    assert REF_FRAME.to_bytes() == REF_FRAME_BYTES


def test_build_ref_payload():
    assert build_payload([REF_FRAME]) == REF_PAYLOAD


def test_parse_ref_payload():
    assert parse_payload(REF_PAYLOAD) == [REF_FRAME]


def test_build_empty_payload():
    assert build_payload([]) == EMPTY_PAYLOAD


def test_parse_empty_payload():
    assert parse_payload(EMPTY_PAYLOAD) == []


def test_parse_no_compression():
    payload = bytes.fromhex("0001 08 00 0000 0000000C 0000000C") + REF_FRAME_BYTES
    assert parse_payload(payload) == [REF_FRAME]


def test_no_compression_with_wrong_bitstream_len():
    payload = bytes.fromhex("0001 08 00 0000 0000000C 0000000B") + REF_FRAME_BYTES[:11]
    assert payload_error_code(payload) == "E_PAYLOAD"


def test_reserved_not_zero():
    assert payload_error_code(REF_PAYLOAD[:3] + b"\x01" + REF_PAYLOAD[4:]) == "E_PAYLOAD"


@pytest.mark.parametrize("flags", [0x01, 0x02, 0x10, 0x80])
def test_unknown_flags(flags):
    assert payload_error_code(REF_PAYLOAD[:2] + bytes([flags]) + REF_PAYLOAD[3:]) == "E_PAYLOAD"


def test_symbol_count_not_12_times_frame_count():
    # frame_count = 2, а symbol_count лишився 12
    assert payload_error_code(b"\x00\x02" + REF_PAYLOAD[2:]) == "E_PAYLOAD"


def test_payload_too_short():
    assert payload_error_code(REF_PAYLOAD[:5]) == "E_PAYLOAD"


def test_payload_cut_inside_bitstream():
    assert payload_error_code(REF_PAYLOAD[:-1]) == "E_PAYLOAD"


def test_payload_with_extra_byte():
    assert payload_error_code(REF_PAYLOAD + b"\x00") == "E_PAYLOAD"


def test_huffman_error_becomes_e_code(monkeypatch):
    def broken_decode(lengths, bitstream, symbol_count):
        raise HuffmanError("потік обірвано")
    monkeypatch.setattr(packet, "huffman_decode", broken_decode)
    assert payload_error_code(REF_PAYLOAD) == "E_CODE"


def test_bad_frame_mode_gives_packet_error():
    bad_frame = REF_FRAME_BYTES[:10] + b"\x09\x01"  # mode = 9
    payload = bytes.fromhex("0001 08 00 0000 0000000C 0000000C") + bad_frame
    assert payload_error_code(payload) == "E_PAYLOAD"


# ---------- Крок 2: заголовок (розділ 2.1) ----------
REF_SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
REF_NONCE = bytes.fromhex("010203040506070800000000")


def test_build_ref_header():
    header = build_header(device_id=1, packet_seq=0, salt=REF_SALT, nonce=REF_NONCE, ciphertext_len=32)
    assert header == REF_01[:43]


def test_header_fields_are_big_endian():
    header = build_header(device_id=0x0102, packet_seq=0x03040506, salt=REF_SALT, nonce=REF_NONCE,
                          ciphertext_len=0x0708090A)
    assert header[0:4] == b"TLM1"
    assert header[4] == 1
    assert header[5:7] == bytes.fromhex("0102")
    assert header[7:11] == bytes.fromhex("03040506")
    assert header[11:27] == REF_SALT
    assert header[27:39] == REF_NONCE
    assert header[39:43] == bytes.fromhex("0708090A")


# ---------- Крок 3: build_packet (розділи 2 і 4) ----------
# Ключі тут умовні: справжні рахує derive_keys Наді. Підробки перевіряють, ЩО packet.py
# передає криптографії, і повертають шифротекст і тег із REF_01.
FAKE_K_ENC = b"E" * 32
FAKE_K_MAC = b"M" * 32


def fake_derive_keys(password, salt):
    assert (password, salt) == (PASSWORD, REF_SALT)
    return FAKE_K_ENC, FAKE_K_MAC


def fake_aes_ctr(k_enc, nonce, data):
    assert (k_enc, nonce, data) == (FAKE_K_ENC, REF_NONCE, REF_PAYLOAD)
    return REF_01[43:75]


def fake_make_tag(k_mac, data):
    # HMAC рахується над заголовком і шифротекстом разом (Encrypt-then-MAC)
    assert (k_mac, data) == (FAKE_K_MAC, REF_01[:75])
    return REF_01[75:]


@pytest.fixture
def fake_crypto(monkeypatch):
    monkeypatch.setattr(packet, "derive_keys", fake_derive_keys)
    monkeypatch.setattr(packet, "aes_ctr", fake_aes_ctr)
    monkeypatch.setattr(packet, "make_tag", fake_make_tag)


def build_ref_packet(salt=REF_SALT, nonce=REF_NONCE) -> bytes:
    return build_packet([REF_FRAME], PASSWORD, device_id=1, packet_seq=0, salt=salt, nonce=nonce)


def test_build_ref_packet(fake_crypto):
    assert build_ref_packet() == REF_01


def test_build_packet_rejects_short_salt(fake_crypto):
    with pytest.raises(ValueError):
        build_ref_packet(salt=REF_SALT[:15])


def test_build_packet_rejects_long_nonce(fake_crypto):
    with pytest.raises(ValueError):
        build_ref_packet(nonce=REF_NONCE + b"\x00")


# ---------- Крок 4: parse_packet у порядку розділу 5 ----------
def test_parse_payload_with_class_table():
    # flags = CLASS_TABLE (0x04): після потоку ще 1 байт класу на кадр, приймач його пропускає
    payload = REF_PAYLOAD[:2] + b"\x04" + REF_PAYLOAD[3:] + b"\x02"
    assert parse_payload(payload) == [REF_FRAME]


def test_class_table_missing():
    payload = REF_PAYLOAD[:2] + b"\x04" + REF_PAYLOAD[3:]
    assert payload_error_code(payload) == "E_PAYLOAD"


def test_table_with_repeated_symbol():
    # другий запис таблиці (01 03) замінено повтором першого (00 01)
    payload = REF_PAYLOAD[:8] + b"\x00\x01" + REF_PAYLOAD[10:]
    assert payload_error_code(payload) == "E_CODE"


def test_table_not_sorted():
    # перші два записи переставлено: 01 03, 00 01
    payload = REF_PAYLOAD[:6] + b"\x01\x03\x00\x01" + REF_PAYLOAD[10:]
    assert payload_error_code(payload) == "E_CODE"


# Іграшкова криптографія: оборотна і чутлива до пароля, щоб перевірити ЛОГІКУ приймача
# без коду Наді. Справжній AES і HMAC перевірить крок 5 на еталонних файлах.
def toy_derive_keys(password, salt):
    master = hashlib.sha256(password.encode() + salt).digest()
    return master[:16], master[16:]


def toy_aes_ctr(k_enc, nonce, data):
    toy_aes_ctr.calls += 1
    return bytes(b ^ 0x5A for b in data)


def toy_make_tag(k_mac, data):
    return hashlib.sha256(k_mac + data).digest()


def toy_verify_tag(k_mac, data, tag):
    return toy_make_tag(k_mac, data) == tag


@pytest.fixture
def toy_crypto(monkeypatch):
    toy_aes_ctr.calls = 0
    monkeypatch.setattr(packet, "derive_keys", toy_derive_keys)
    monkeypatch.setattr(packet, "aes_ctr", toy_aes_ctr)
    monkeypatch.setattr(packet, "make_tag", toy_make_tag)
    monkeypatch.setattr(packet, "verify_tag", toy_verify_tag)


def packet_from_payload(payload: bytes) -> bytes:
    """Чесно шифрує і пломбує БУДЬ-ЯКЕ навантаження, навіть зіпсоване."""
    k_enc, k_mac = toy_derive_keys(PASSWORD, REF_SALT)
    ciphertext = toy_aes_ctr(k_enc, REF_NONCE, payload)
    header = build_header(1, 0, REF_SALT, REF_NONCE, len(ciphertext))
    return header + ciphertext + toy_make_tag(k_mac, header + ciphertext)


def test_single_frame_round_trip(toy_crypto):
    assert parse_packet(build_ref_packet(), PASSWORD) == [REF_FRAME]


def test_empty_packet_round_trip(toy_crypto):
    pkt = build_packet([], PASSWORD, device_id=1, packet_seq=0, salt=REF_SALT, nonce=REF_NONCE)
    assert len(pkt) == 43 + 14 + 32
    assert parse_packet(pkt, PASSWORD) == []


def test_tampered_ciphertext(toy_crypto):
    pkt = bytearray(build_ref_packet())
    pkt[50] ^= 0x01
    assert error_code(bytes(pkt)) == "E_HMAC"


def test_tampered_tag(toy_crypto):
    pkt = bytearray(build_ref_packet())
    pkt[-1] ^= 0x01
    assert error_code(bytes(pkt)) == "E_HMAC"


def test_tampered_header(toy_crypto):
    pkt = bytearray(build_ref_packet())
    pkt[6] ^= 0x01  # device_id: заголовок теж під пломбою
    assert error_code(bytes(pkt)) == "E_HMAC"


def test_wrong_password(toy_crypto):
    with pytest.raises(PacketError) as info:
        parse_packet(build_ref_packet(), "wrong")
    assert info.value.code == "E_HMAC"


def test_no_decryption_before_hmac(toy_crypto):
    pkt = bytearray(build_ref_packet())
    pkt[-1] ^= 0x01
    calls_after_build = toy_aes_ctr.calls
    error_code(bytes(pkt))
    assert toy_aes_ctr.calls == calls_after_build  # розділ 5: при поганому HMAC не розшифровуємо


def test_reserved_not_zero_in_packet(toy_crypto):
    payload = REF_PAYLOAD[:3] + b"\x01" + REF_PAYLOAD[4:]
    assert error_code(packet_from_payload(payload)) == "E_PAYLOAD"


def test_huffman_error_in_packet(toy_crypto, monkeypatch):
    def broken_decode(lengths, bitstream, symbol_count):
        raise HuffmanError("потік обірвано")
    monkeypatch.setattr(packet, "huffman_decode", broken_decode)
    assert error_code(packet_from_payload(REF_PAYLOAD)) == "E_CODE"


def test_big_packet_is_accepted(toy_crypto):
    # ліміт max_packet_bytes стосується лише передавача
    payload = bytes.fromhex("0040 08 00 0000 00000300 00000300") + REF_FRAME_BYTES * 64
    pkt = packet_from_payload(payload)
    assert len(pkt) > 512
    assert parse_packet(pkt, PASSWORD) == [REF_FRAME] * 64

def test_extra_bitstream_byte_is_rejected(monkeypatch):
    # тут потрібен справжній декодер Маші, а не підробка з фікстури
    monkeypatch.setattr(packet, "huffman_decode", huffman.huffman_decode)
    # контрольне навантаження + один цілий зайвий нульовий байт; bitstream_len чесно 5 замість 4
    bad = REF_PAYLOAD[:20] + bytes.fromhex("0000000C 00000005") + REF_BITSTREAM + b"\x00"
    assert payload_error_code(bad) == "E_PAYLOAD"