"""Одна команда оцінки: Q для baseline, поточної системи і системи з параметрами GA.

Запуск з кореня репозиторію: python src/evaluate.py
Результат друкується в таблицю і зберігається в results/final_metrics.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import load_labeled, predict_classes, train_classifier  # noqa: E402
from ga import apply_genes  # noqa: E402
from run_session import DATA, load_config, load_frames, run_scenario  # noqa: E402

RESULTS = ROOT / "results"


def best_ga_genes():
    """Гени з найкращого запуску GA (results/ga_vs_random.json) або None."""
    path = RESULTS / "ga_vs_random.json"
    if not path.exists():
        return None
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    return max(runs, key=lambda r: r["ga_q"])["ga_genes"]


def evaluate(frames, config, label, **kwargs):
    """Прогін усіх сценаріїв. Повертає список словників S, R, T, Q."""
    rows = []
    for name in config["канал"]["сценарії"]:
        res = run_scenario(frames, config, name, **kwargs)
        threshold = config["пороги_втрати_критичних_кадрів"][name]
        loss = 1.0 - res["R"]
        rows.append({
            "variant": label, "scenario": name,
            "packets": res["пакетів"], "S": res["S"], "R": res["R"],
            "T": res["T"], "Q": res["Q"],
            "critical_loss": loss, "threshold": threshold,
            "threshold_met": loss <= threshold + 1e-9,
        })
    return rows


def main():
    config = load_config()
    frames = load_frames()
    train_frames, labels = load_labeled(str(DATA / "train.csv"))
    classes = predict_classes(train_classifier(train_frames, labels), frames)

    rows = evaluate(frames, config, "baseline (послідовне)", packer="greedy")
    rows += evaluate(frames, config, "поточне (рюкзак ДП + MLP)", classes=classes)
    genes = best_ga_genes()
    if genes:
        tuned = apply_genes(config, genes)
        label = (f"GA (вікно {genes['window']}, бонус {genes['critical_bonus']}, "
                 f"аномалія {genes['anomaly_weight']})")
        rows += evaluate(frames, tuned, label, classes=classes)

    print(f"{'варіант':<44}{'сценарій':<11}{'пак.':>5}{'S':>8}{'R':>8}{'T':>8}{'Q':>8}  поріг втрат")
    for r in rows:
        mark = "ok" if r["threshold_met"] else "НЕ виконано"
        print(f"{r['variant']:<44}{r['scenario']:<11}{r['packets']:>5}{r['S']:>8.4f}"
              f"{r['R']:>8.4f}{r['T']:>8.4f}{r['Q']:>8.4f}  "
              f"{r['critical_loss']:.3f} <= {r['threshold']:.2f}: {mark}")

    summary = {}
    for r in rows:
        summary.setdefault(r["variant"], []).append(r["Q"])
    print()
    for variant, qs in summary.items():
        print(f"Q (середнє за сценаріями): {sum(qs) / len(qs):.4f}  {variant}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "final_metrics.json").write_text(
        json.dumps({"rows": rows,
                    "mean_q": {v: sum(q) / len(q) for v, q in summary.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()