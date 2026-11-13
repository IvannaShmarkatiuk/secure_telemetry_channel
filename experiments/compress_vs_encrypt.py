import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto import aes_ctr, derive_keys          # noqa: E402
from huffman import count_frequencies, huffman_encode  # noqa: E402
from run_session import load_frames              # noqa: E402

PASSWORD = "algo2026"
SALT = bytes(range(16))
CHUNKS = [1, 8, 38, 64, 256, 3000]               # кадрів у порції; 38 ≈ пакет послідовного пакування


def entropy_bits_per_byte(data: bytes) -> float:
    """Ентропія Шеннона: скільки біт інформації в середньому несе один байт (макс. 8)."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in count_frequencies(data).values())


def compressed_size(data: bytes) -> int:
    """Байтів після Гаффмана разом із таблицею: 2·table_nsym + len(bitstream)."""
    lengths, bitstream = huffman_encode(data)
    return 2 * len(lengths) + len(bitstream)


def run():
    frames = load_frames()
    k_enc, _ = derive_keys(PASSWORD, SALT)
    rows = []
    for n in CHUNKS:
        raw_total = a_total = b_total = 0
        h_raw = h_enc = 0.0
        chunks = 0
        for start in range(0, len(frames), n):
            raw = b"".join(f.to_bytes() for f in frames[start:start + n])
            nonce = b"EXPERIMT" + chunks.to_bytes(4, "big")       # свій nonce на кожну порцію, відтворювано
            # A: стиснути -> зашифрувати (так робить наш формат)
            a = compressed_size(raw)                               # CTR не змінює довжину
            # B: зашифрувати -> стиснути
            enc = aes_ctr(k_enc, nonce, raw)
            b = compressed_size(enc)
            raw_total += len(raw); a_total += a; b_total += b
            h_raw += entropy_bits_per_byte(raw); h_enc += entropy_bits_per_byte(enc)
            chunks += 1
        rows.append({
            "кадрів_у_порції": n,
            "порцій": chunks,
            "сирі_байти": raw_total,
            "стиснути_потім_шифрувати": a_total,
            "шифрувати_потім_стиснути": b_total,
            "коеф_A": round(a_total / raw_total, 4),
            "коеф_B": round(b_total / raw_total, 4),
            "ентропія_сирих_біт_на_байт": round(h_raw / chunks, 3),
            "ентропія_шифротексту_біт_на_байт": round(h_enc / chunks, 3),
        })
    return rows


def save_csv(rows, path):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)                     # атомарний запис: або старий файл, або новий цілий


if __name__ == "__main__":
    rows = run()
    print(f"{'кадрів':>7} {'сирі, Б':>9} {'стиснути→шифр.':>15} {'шифр.→стиснути':>15} "
          f"{'коеф A':>7} {'коеф B':>7} {'H сирих':>8} {'H шифр.':>8}")
    for r in rows:
        print(f"{r['кадрів_у_порції']:>7} {r['сирі_байти']:>9} {r['стиснути_потім_шифрувати']:>15} "
              f"{r['шифрувати_потім_стиснути']:>15} {r['коеф_A']:>7} {r['коеф_B']:>7} "
              f"{r['ентропія_сирих_біт_на_байт']:>8} {r['ентропія_шифротексту_біт_на_байт']:>8}")
    out = ROOT / "results" / "compress_vs_encrypt.csv"
    save_csv(rows, out)
    print(f"\nЗбережено: {out.relative_to(ROOT)}")
