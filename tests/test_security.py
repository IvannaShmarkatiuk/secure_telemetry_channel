import pytest

import packet
from crypto import aes_ctr, derive_keys, make_tag
from frame import Frame
from packet import PacketError, build_packet, parse_packet

PASSWORD = "algo2026"
SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")
NONCE = bytes.fromhex("010203040506070800000000")
FRAMES = [Frame(0, 1000, 215, -3, 1, 1),
          Frame(50, 1003, 216, 12, 1, 0),
          Frame(100, 1010, 216, -40, 2, 0)]


def make_packet() -> bytes:
    return build_packet(FRAMES, PASSWORD, device_id=1, packet_seq=0, salt=SALT, nonce=NONCE)


def code_of(pkt: bytes, password: str = PASSWORD) -> str:
    with pytest.raises(PacketError) as info:
        parse_packet(pkt, password)
    return info.value.code


@pytest.fixture
def decrypt_spy(monkeypatch):
    """Обгортка над справжнім aes_ctr у приймачі: рахує, скільки разів він викликався."""
    calls = []

    def spy(k_enc, nonce, data):
        calls.append(len(data))
        return aes_ctr(k_enc, nonce, data)

    monkeypatch.setattr(packet, "aes_ctr", spy)
    return calls


# ---------- правильний пароль + оригінальний пакет ----------
def test_correct_password_original_packet_restores_frames():
    assert parse_packet(make_packet(), PASSWORD) == FRAMES


def test_spec_section7_ciphertext_and_tag():
    """Контрольний приклад розділу 7: шифротекст і тег збігаються байт у байт."""
    pkt = build_packet([Frame(0, 1000, 215, -3, 1, 1)], PASSWORD, 1, 0, SALT, NONCE)
    assert pkt[43:75].hex().upper().startswith("8E51D20BA4B608DF")
    assert pkt[75:].hex().upper() == ("DD881BA276FB275C767958046D3EB9B7"
                                      "5D4F7C95EFA3F8CAC0B9411E18461A8E")


# ---------- неправильний пароль ----------
@pytest.mark.parametrize("wrong", ["algo2027", "", "ALGO2026", "algo2026 "])
def test_wrong_password_gives_e_hmac(wrong):
    assert code_of(make_packet(), wrong) == "E_HMAC"


def test_wrong_password_does_not_decrypt(decrypt_spy):
    pkt = make_packet()
    decrypt_spy.clear()               # шифрування під час збирання пакета не рахуємо
    code_of(pkt, "wrong-password")
    assert decrypt_spy == []          # розшифрування навіть не почалося


# ---------- змінений пакет ----------
@pytest.mark.parametrize("pos", [5, 7, 11, 27, 43, 50, -33])   # device_id, seq, salt, nonce, шифротекст
def test_any_changed_byte_gives_e_hmac(pos):
    pkt = bytearray(make_packet())
    pkt[pos] ^= 0x01
    assert code_of(bytes(pkt)) == "E_HMAC"


def test_changed_packet_does_not_decrypt(decrypt_spy):
    pkt = bytearray(make_packet())
    pkt[50] ^= 0x80
    decrypt_spy.clear()
    code_of(bytes(pkt))
    assert decrypt_spy == []


# ---------- правильний і неправильний HMAC ----------
def test_correct_hmac_decrypts_exactly_once(decrypt_spy):
    pkt = make_packet()
    decrypt_spy.clear()
    parse_packet(pkt, PASSWORD)
    assert len(decrypt_spy) == 1


def test_wrong_hmac_rejected_without_decrypt(decrypt_spy):
    pkt = bytearray(make_packet())
    pkt[-1] ^= 0x01                    # зіпсовано останній байт тегу
    decrypt_spy.clear()
    assert code_of(bytes(pkt)) == "E_HMAC"
    assert decrypt_spy == []


def test_hmac_made_with_enc_key_is_rejected():
    """Якщо пломбу поставити ключем шифрування замість K_mac — пакет не приймається."""
    pkt = make_packet()
    k_enc, _ = derive_keys(PASSWORD, SALT)
    forged = pkt[:-32] + make_tag(k_enc, pkt[:-32])
    assert code_of(forged) == "E_HMAC"


# ---------- чому nonce не можна повторювати ----------
def test_nonce_reuse_leaks_xor_of_plaintexts():
    """Однаковий (ключ, nonce) -> однаковий keystream: c1 XOR c2 = p1 XOR p2, ключ не потрібен."""
    k_enc, _ = derive_keys(PASSWORD, SALT)
    p1, p2 = b"altitude=1000 temp=215", b"altitude=9999 temp=999"
    c1, c2 = aes_ctr(k_enc, NONCE, p1), aes_ctr(k_enc, NONCE, p2)
    assert bytes(a ^ b for a, b in zip(c1, c2)) == bytes(a ^ b for a, b in zip(p1, p2))


def test_different_nonce_per_packet_changes_ciphertext():
    a = build_packet(FRAMES, PASSWORD, 1, 0, SALT, NONCE)
    b = build_packet(FRAMES, PASSWORD, 1, 1, SALT, NONCE[:8] + (1).to_bytes(4, "big"))
    assert a[43:-32] != b[43:-32]
