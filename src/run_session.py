"""Наскрізний прогін: кадри -> пакети -> канал -> приймач -> відновлені кадри.

Запуск: python src/run_session.py
"""
import json
import os
import time
from pathlib import Path
from huffman import frame_weight, window_lengths
from knapsack import capacity_bits, knapsack_dp, knapsack_greedy

from channel import Channel
from frame import Frame
from metrics import critical_recovery_r, efficiency_s, quality_q, speed_t
from packet import PacketError, build_packet, parse_packet
from classifier import load_labeled, predict_classes, train_classifier

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

def frame_value(cls, critical, config=None):
    """Цінність кадру: пріоритет за класом + бонус за critical.

    cls: 0 норма, 1 аномалія, 2 перехідний режим.
    Числа беруться з team_config.json (пріоритети_кадрів).
    """
    if cls not in (0, 1, 2):
        raise ValueError(f"клас має бути 0, 1 або 2, а не {cls}")
    if critical not in (0, 1):
        raise ValueError(f"critical має бути 0 або 1, а не {critical}")
    p = (config or load_config())["пріоритети_кадрів"]
    by_class = {0: p["норма"], 1: p["аномалія"], 2: p["перехідний_режим"]}
    return by_class[cls] + p["critical_бонус"] * critical


def schedule(groups):
    """Порядок відправки: групи з більшою сумарною цінністю йдуть першими.

    groups: список пар (індекси кадрів, сумарна цінність).
    Повертає список списків індексів. При однаковій цінності зберігається
    початковий порядок. packet_seq призначають уже в цьому порядку: 0, 1, 2...
    """
    ordered = sorted(groups, key=lambda g: -g[1])
    return [list(indices) for indices, _ in ordered]

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
def plan_windows(frames, config, classes=None, solver=knapsack_dp):
    """Рюкзак по вікнах кандидатів із перенесенням (FORMAT_SPEC 3.5).

    Вікно: перші 64 кадри черги. Рюкзак обирає кадри в пакет, а решта
    переходить у наступне вікно, тож жоден кадр не губиться.
    classes: клас кожного кадру (0, 1, 2); без нього всі кадри класу 0.
    solver: knapsack_dp або knapsack_greedy.
    Повертає список груп (індекси кадрів, сумарна цінність, таблиця довжин).
    """
    window = config["пакування"]["candidate_window_frames"]
    max_payload = config["пакування"]["max_payload_bytes"]
    if classes is None:
        classes = [0] * len(frames)
    queue = list(range(len(frames)))
    groups = []
    while queue:
        idx = queue[:window]
        raw = [frames[i].to_bytes() for i in idx]
        lengths = window_lengths(raw)
        weights = [frame_weight(b, lengths) for b in raw]
        values = [frame_value(classes[i], frames[i].critical, config) for i in idx]
        cap = capacity_bits(max_payload, len(lengths))
        chosen = solver(weights, values, cap)
        if not chosen:
            raise ValueError("жоден кадр вікна не вміщається в пакет")
        chosen_idx = [idx[j] for j in chosen]
        groups.append((chosen_idx, sum(values[j] for j in chosen), lengths))
        taken = set(chosen_idx)
        queue = [i for i in queue if i not in taken]
    return groups

def make_packets_knapsack(frames, config, salt, session_prefix, classes=None, solver=knapsack_dp):
    """Пакування через рюкзак: один пакет на вікно, найцінніші пакети йдуть першими.

    Повертає список пар (пакет, кадри в пакеті), як і make_packets.
    """
    password = config["пароль_сеансу"]
    limit = config["пакування"]["max_packet_bytes"]
    groups = plan_windows(frames, config, classes, solver)
    tables = {tuple(g[0]): g[2] for g in groups}
    order = schedule([(g[0], g[1]) for g in groups])
    result = []
    for seq, indices in enumerate(order):
        chosen = [frames[i] for i in indices]
        nonce = make_nonce(session_prefix, seq)
        packet = build_packet(chosen, password, DEVICE_ID, seq, salt, nonce,
                              tables[tuple(indices)])
        if len(packet) > limit:
            raise ValueError(f"пакет {seq}: {len(packet)} байтів, ліміт {limit}")
        result.append((packet, chosen))
    return result

def run_scenario(frames, config, name, packer="knapsack", classes=None, solver=knapsack_dp):
    """Один повний прогін у вибраному сценарії каналу. Повертає словник результатів."""
    password = config["пароль_сеансу"]
    started = time.perf_counter()

    salt = os.urandom(16)
    session_prefix = os.urandom(8)
    if packer == "knapsack":
        packets = make_packets_knapsack(frames, config, salt, session_prefix, classes, solver)
    else:
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

    frames_train, labels_train = load_labeled(str(DATA / "train.csv"))
    model = train_classifier(frames_train, labels_train)
    classes = predict_classes(model, frames)

    print(f"Кадрів у сеансі: {len(frames)}")
    print(f"{'сценарій':<11}{'пакетів':>8}{'кадрів':>8}{'S':>8}{'R':>8}{'T':>8}{'Q':>8}{'час, с':>9}")
    for name in config["канал"]["сценарії"]:
        res = run_scenario(frames, config, name, classes=classes)
        print(f"{name:<11}{res['пакетів']:>8}{res['кадрів_відновлено']:>8}"
              f"{res['S']:>8.4f}{res['R']:>8.4f}{res['T']:>8.4f}{res['Q']:>8.4f}{res['час_с']:>9.2f}")

if __name__ == "__main__":
    main()