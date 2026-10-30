"""Криптографія пакета TLM1 (FORMAT_SPEC, розділ 4)."""
import functools
import hashlib
import hmac

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


@functools.lru_cache(maxsize=None)
def derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """Повертає (K_enc, K_mac). Кешується: KDF рахується раз на (пароль, сіль)."""
    master = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
        dklen=64,
    )
    return master[:32], master[32:]


def aes_ctr(k_enc: bytes, nonce: bytes, data: bytes) -> bytes:
    """AES-256-CTR. Шифрування і розшифрування — одна й та сама операція.

    Блок лічильника = nonce (12 байтів) || uint32_BE(i), i починається з 1.
    """
    initial_counter_block = nonce + (1).to_bytes(4, "big")
    cipher = Cipher(algorithms.AES(k_enc), modes.CTR(initial_counter_block))
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()


def make_tag(k_mac: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 над data ключем K_mac. Повертає 32 байти."""
    return hmac.new(k_mac, data, hashlib.sha256).digest()


def verify_tag(k_mac: bytes, data: bytes, tag: bytes) -> bool:
    """Перевіряє тег у постійний час (без витоку через час відповіді)."""
    return hmac.compare_digest(make_tag(k_mac, data), tag)
