"""Криптографія пакета TLM1 (FORMAT_SPEC, розділ 4)."""
import functools
import hashlib


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
