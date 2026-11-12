"""Порівняння пакувань і рюкзаків ДП та жадібного.

Запуск з кореня репозиторію: python experiments/compare_packers.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import load_labeled, predict_classes, train_classifier  # noqa: E402
from huffman import frame_weight, window_lengths  # noqa: E402
from knapsack import capacity_bits, knapsack_dp, knapsack_greedy  # noqa: E402
from run_session import (DATA, frame_value, load_config, load_frames,  # noqa: E402
                         run_scenario)


def solver_table(frames, config, classes):
    """ДП проти жадібного на однакових вікнах по 64 кадри: цінність, час, похибка."""
    window = config["пакування"]["candidate_window_frames"]
    max_payload = config["пакування"]["max_payload_bytes"]
    totals = {"ДП": [0, 0.0], "жадібний": [0, 0.0]}
    for start in range(0, len(frames), window):
        idx = range(start, min(start + window, len(frames)))
        raw = [frames[i].to_bytes() for i in idx]
        lengths = window_lengths(raw)
        weights = [frame_weight(b, lengths) for b in raw]
        values = [frame_value(classes[i], frames[i].critical, config) for i in idx]
        cap = capacity_bits(max_payload, len(lengths))
        for name, fn in (("ДП", knapsack_dp), ("жадібний", knapsack_greedy)):
            t = time.perf_counter()
            chosen = fn(weights, values, cap)
            totals[name][1] += time.perf_counter() - t
            totals[name][0] += sum(values[j] for j in chosen)
    dp_value = totals["ДП"][0]
    print("| алгоритм | сумарна цінність | час, с | похибка жадібного |")
    print("|---|---|---|---|")
    for name, (value, secs) in totals.items():
        err = "-" if name == "ДП" else f"{100 * (dp_value - value) / dp_value:.2f}%"
        print(f"| {name} | {value} | {secs:.3f} | {err} |")


def packer_table(frames, config, classes):
    variants = [
        ("послідовне (таблиця на пакет)", {"packer": "greedy"}),
        ("рюкзак ДП + вікно", {"classes": classes}),
        ("рюкзак жадібний + вікно", {"classes": classes, "solver": knapsack_greedy}),
    ]
    print("| сценарій | варіант | пакетів | кадрів | S | R | T | Q |")
    print("|---|---|---|---|---|---|---|---|")
    for name in config["канал"]["сценарії"]:
        for label, kwargs in variants:
            r = run_scenario(frames, config, name, **kwargs)
            print(f"| {name} | {label} | {r['пакетів']} | {r['кадрів_відновлено']} | "
                  f"{r['S']:.4f} | {r['R']:.4f} | {r['T']:.4f} | {r['Q']:.4f} |")


def main():
    config = load_config()
    frames = load_frames()
    frames_train, labels_train = load_labeled(str(DATA / "train.csv"))
    classes = predict_classes(train_classifier(frames_train, labels_train), frames)
    print("## Рюкзак: ДП проти жадібного (вікна по 64 кадри)\n")
    solver_table(frames, config, classes)
    print("\n## Варіанти пакування за сценаріями\n")
    packer_table(frames, config, classes)


if __name__ == "__main__":
    main()