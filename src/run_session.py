"""Наскрізний прогін: кадри -> пакети -> канал -> приймач -> відновлені кадри.

Запуск: python src/run_session.py
"""
import json
import os
import time
from pathlib import Path

from channel import Channel
from frame import Frame
from metrics import critical_recovery_r, efficiency_s, quality_q, speed_t
from packet import PacketError, build_packet, parse_packet

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEVICE_ID = 1
FRAME_SIZE = 12


def load_config(path=None):
    """Читає team_config.json."""
    path = Path(path) if path else DATA / "team_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_frames(path=None):
    """Читає session.bin: файл складається з кадрів по 12 байтів."""
    path = Path(path) if path else DATA / "session.bin"
    raw = path.read_bytes()
    if len(raw) % FRAME_SIZE != 0:
        raise ValueError("довжина файлу не кратна 12: останній кадр обірвано")
    return [Frame.from_bytes(raw[i:i + FRAME_SIZE]) for i in range(0, len(raw), FRAME_SIZE)]


def make_nonce(session_prefix, packet_seq):
    """nonce = 8 байтів сеансу + номер пакета (4 байти, big-endian) = 12 байтів."""
    return session_prefix + packet_seq.to_bytes(4, "big")


def make_packets(frames, config, salt, session_prefix):
    """Жадібна нарізка: у пакет додаємо кадри, поки він не перевищить ліміт.

    Повертає список пар (пакет, кадри, що в ньому). Пробні пакети, які не
    влізли, нікуди не відправляються, а packet_seq росте на 1 лише для
    фінального пакета.
    """
    password = config["пароль_сеансу"]
    limit = config["пакування"]["max_packet_bytes"]
    result = []
    pos = 0
    seq = 0
    while pos < len(frames):
        nonce = make_nonce(session_prefix, seq)
        best_packet = None
        best_count = 0
        for n in range(1, len(frames) - pos + 1):
            trial = build_packet(frames[pos:pos + n], password, DEVICE_ID, seq, salt, nonce)
            if len(trial) > limit:
                break
            best_packet, best_count = trial, n
        if best_packet is None:
            raise ValueError("навіть один кадр не вміщається в ліміт пакета")
        result.append((best_packet, frames[pos:pos + best_count]))
        pos += best_count
        seq += 1
    return result


def run_scenario(frames, config, name):
    """Один повний прогін у вибраному сценарії каналу. Повертає словник результатів."""
    password = config["пароль_сеансу"]
    started = time.perf_counter()

    salt = os.urandom(16)
    session_prefix = os.urandom(8)
    packets = make_packets(frames, config, salt, session_prefix)

    scenario = config["канал"]["сценарії"][name]
    channel = Channel(scenario, config["канал"]["seed_моделі_каналу"])

    restored = []
    bits_sent = 0
    for packet, _frames_in_packet in packets:
        before = channel.sent
        delivered, _arrival = channel.send(packet)
        if channel.sent > before:  # пакет реально пішов у канал (вклався в бюджет)
            bits_sent += 8 * len(packet)
        if delivered:
            try:
                restored.extend(parse_packet(packet, password))
            except PacketError:
                pass  # пошкоджений пакет приймач відкидає

    elapsed = time.perf_counter() - started

    coef = config["метрика_Q"]
    b_ref = FRAME_SIZE * 8 * len(frames)
    s = efficiency_s(b_ref, bits_sent)
    r = critical_recovery_r(frames, restored)
    t = speed_t(coef["t_ref_секунд"], elapsed)
    q = quality_q(s, r, t, coef["alpha"], coef["beta"], coef["gamma"])
    return {
        "сценарій": name,
        "пакетів": len(packets),
        "кадрів_відновлено": len(restored),
        "S": s, "R": r, "T": t, "Q": q,
        "час_с": elapsed,
        "канал": channel.stats(),
        "restored_frames": restored,
    }


def main():
    config = load_config()
    frames = load_frames()
    print(f"Кадрів у сеансі: {len(frames)}")
    print(f"{'сценарій':<11}{'пакетів':>8}{'кадрів':>8}{'S':>8}{'R':>8}{'T':>8}{'Q':>8}{'час, с':>9}")
    for name in config["канал"]["сценарії"]:
        res = run_scenario(frames, config, name)
        print(f"{name:<11}{res['пакетів']:>8}{res['кадрів_відновлено']:>8}"
              f"{res['S']:>8.4f}{res['R']:>8.4f}{res['T']:>8.4f}{res['Q']:>8.4f}{res['час_с']:>9.2f}")


if __name__ == "__main__":
    main()