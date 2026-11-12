"""Порівняння пакувань: жадібне, рюкзак із заглушкою класів, рюкзак із MLP.

Запуск з кореня репозиторію: python experiments/compare_packers.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import load_labeled, predict_classes, train_classifier  # noqa: E402
from run_session import DATA, load_config, load_frames, run_scenario  # noqa: E402


def main():
    config = load_config()
    frames = load_frames()
    frames_train, labels_train = load_labeled(str(DATA / "train.csv"))
    classes = predict_classes(train_classifier(frames_train, labels_train), frames)

    variants = [
        ("жадібне", {"packer": "greedy"}),
        ("рюкзак, класи=0", {"packer": "knapsack"}),
        ("рюкзак + MLP", {"packer": "knapsack", "classes": classes}),
    ]
    print("| сценарій | варіант | пакетів | S | R | T | Q |")
    print("|---|---|---|---|---|---|---|")
    for name in config["канал"]["сценарії"]:
        for label, kwargs in variants:
            r = run_scenario(frames, config, name, **kwargs)
            print(f"| {name} | {label} | {r['пакетів']} | {r['S']:.4f} | "
                  f"{r['R']:.4f} | {r['T']:.4f} | {r['Q']:.4f} |")


if __name__ == "__main__":
    main()