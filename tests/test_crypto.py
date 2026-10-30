import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from crypto import derive_keys, aes_ctr, make_tag, verify_tag

PASSWORD = "algo2026"
SALT = bytes.fromhex("000102030405060708090A0B0C0D0E0F")


# ---------- Крок 1: derive_keys ----------

def test_key_lengths():
    k_enc, k_mac = derive_keys(PASSWORD, SALT)
    assert len(k_enc) == 32
    assert len(k_mac) == 32


def test_keys_are_different():
    k_enc, k_mac = derive_keys(PASSWORD, SALT)
    assert k_enc != k_mac


def test_derive_keys_is_deterministic():
    assert derive_keys(PASSWORD, SALT) == derive_keys(PASSWORD, SALT)


def test_different_salt_gives_different_keys():
    other_salt = bytes(16)
    assert derive_keys(PASSWORD, SALT) != derive_keys(PASSWORD, other_salt)


def test_derive_keys_is_cached():
    derive_keys.cache_clear()
    derive_keys(PASSWORD, SALT)
    derive_keys(PASSWORD, SALT)
    info = derive_keys.cache_info()
    assert info.misses == 1  # KDF порахований один раз
    assert info.hits == 1    # другий виклик узято з кешу


# ---------- Крок 2: aes_ctr ----------

NONCE = bytes.fromhex("010203040506070800000000")


def test_aes_ctr_round_trip():
    k_enc, _ = derive_keys(PASSWORD, SALT)
    for n in (0, 1, 15, 16, 17, 1000):
        data = bytes(range(256)) * 4
        data = data[:n]
        assert aes_ctr(k_enc, NONCE, aes_ctr(k_enc, NONCE, data)) == data


def test_aes_ctr_keeps_length():
    k_enc, _ = derive_keys(PASSWORD, SALT)
    for n in (0, 1, 15, 16, 17, 1000):
        assert len(aes_ctr(k_enc, NONCE, bytes(n))) == n


def test_aes_ctr_different_nonce_different_ciphertext():
    k_enc, _ = derive_keys(PASSWORD, SALT)
    data = bytes(32)
    other_nonce = bytes.fromhex("010203040506070900000000")
    assert aes_ctr(k_enc, NONCE, data) != aes_ctr(k_enc, other_nonce, data)


def test_aes_ctr_counter_starts_at_one():
    # Шифруємо нулі: результат = чиста гама. Перший блок гами має дорівнювати
    # AES-ECB(K_enc, nonce || 00000001), а не nonce || 00000000.
    k_enc, _ = derive_keys(PASSWORD, SALT)
    ecb = Cipher(algorithms.AES(k_enc), modes.ECB()).encryptor()
    expected_block1 = ecb.update(NONCE + (1).to_bytes(4, "big"))
    expected_block2 = ecb.update(NONCE + (2).to_bytes(4, "big"))
    keystream = aes_ctr(k_enc, NONCE, bytes(32))
    assert keystream[:16] == expected_block1
    assert keystream[16:] == expected_block2


# ---------- Крок 3: make_tag / verify_tag ----------

def test_tag_length_is_32():
    _, k_mac = derive_keys(PASSWORD, SALT)
    assert len(make_tag(k_mac, b"hello")) == 32


def test_verify_tag_accepts_correct_tag():
    _, k_mac = derive_keys(PASSWORD, SALT)
    data = b"some packet bytes"
    assert verify_tag(k_mac, data, make_tag(k_mac, data)) is True


def test_verify_tag_rejects_flipped_data_bit():
    _, k_mac = derive_keys(PASSWORD, SALT)
    data = bytearray(b"some packet bytes")
    tag = make_tag(k_mac, bytes(data))
    data[3] ^= 0x01  # змінили один біт
    assert verify_tag(k_mac, bytes(data), tag) is False


def test_verify_tag_rejects_flipped_tag_bit():
    _, k_mac = derive_keys(PASSWORD, SALT)
    data = b"some packet bytes"
    tag = bytearray(make_tag(k_mac, data))
    tag[0] ^= 0x01
    assert verify_tag(k_mac, data, bytes(tag)) is False


def test_verify_tag_rejects_wrong_key():
    _, k_mac = derive_keys(PASSWORD, SALT)
    _, other_mac = derive_keys("wrong-password", SALT)
    data = b"some packet bytes"
    assert verify_tag(other_mac, data, make_tag(k_mac, data)) is False
