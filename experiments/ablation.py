"""Результати за сценаріями каналу і абляція класифікатора.

Порівнює три варіанти в кожному сценарії:
  послідовне          - кадри по черзі, таблиця Гаффмана на кожен пакет (без рюкзака);
  рюкзак без класиф.  - рюкзак ДП + вікно, усі кадри класу 0 (цінність лише від critical);
  рюкзак з класиф.    - те саме, але класи дає перцептрон.
Пише results/channel_scenarios.csv.

"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import load_labeled, predict_classes, train_classifier  # noqa: E402
from run_session import DATA, load_config, load_frames, run_scenario  # noqa: E402

RESULTS = ROOT / "results"
COLUMNS = ["сценарій", "варіант", "пакетів", "кадрів_відновлено",
           "втрачено", "не_вміщено_в_бюджет", "S", "R", "T", "Q"]


def collect_rows(frames, config, classes):
    variants = [
        ("послідовне", {"packer": "sequential"}),
        ("рюкзак без класифікатора", {"packer": "knapsack", "classes": None}),
        ("рюкзак з класифікатором", {"packer": "knapsack", "classes": classes}),
    ]
    rows = []
    for name in config["канал"]["сценарії"]:
        for label, kwargs in variants:
            r = run_scenario(frames, config, name, **kwargs)
            stats = r["канал"]
            rows.append({
                "сценарій": name, "варіант": label,
                "пакетів": r["пакетів"], "кадрів_відновлено": r["кадрів_відновлено"],
                "втрачено": stats.get("втрачено", 0),
                "не_вміщено_в_бюджет": stats.get("не_вміщено_в_бюджет", 0),
                "S": round(r["S"], 4), "R": round(r["R"], 4),
                "T": round(r["T"], 4), "Q": round(r["Q"], 4),
            })
    return rows


def main():
    config = load_config()
    frames = load_frames()
    train_frames, train_labels = load_labeled(str(DATA / "train.csv"))
    classes = predict_classes(train_classifier(train_frames, train_labels), frames)

    rows = collect_rows(frames, config, classes)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "channel_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in COLUMNS) + " |")
    print(f"\nЗбережено: {RESULTS / 'channel_scenarios.csv'}")


if __name__ == "__main__":
    main()
