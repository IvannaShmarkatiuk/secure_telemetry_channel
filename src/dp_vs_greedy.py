import csv
import time

from huffman import frame_weight, window_lengths
from knapsack import capacity_bits, knapsack_dp, knapsack_greedy
from run_session import DATA, ROOT, frame_value, load_config, load_frames

FIELDS = ["window", "frames", "capacity_bits",
          "value_dp", "value_greedy", "weight_dp", "weight_greedy",
          "time_dp_ms", "time_greedy_ms"]


def load_labels():
    """Справжні класи кадрів із session.csv."""
    with open(DATA / "session.csv", encoding="utf-8") as f:
        return [int(r["label"]) for r in csv.DictReader(f)]


def compare_windows(frames, config, classes=None):
    """Для кожного вікна запускає обидва рюкзаки на ОДНАКОВИХ даних. Повертає список рядків."""
    if classes is None:
        classes = load_labels()
    window = config["пакування"]["candidate_window_frames"]
    max_payload = config["пакування"]["max_payload_bytes"]
    rows = []
    for start in range(0, len(frames), window):
        idx = range(start, min(start + window, len(frames)))
        raw = [frames[i].to_bytes() for i in idx]
        lengths = window_lengths(raw)
        weights = [frame_weight(b, lengths) for b in raw]
        values = [frame_value(classes[i], frames[i].critical, config) for i in idx]
        cap = capacity_bits(max_payload, len(lengths))
        row = {"window": start // window, "frames": len(idx), "capacity_bits": cap}
        for name, solver in (("dp", knapsack_dp), ("greedy", knapsack_greedy)):
            t = time.perf_counter()
            chosen = solver(weights, values, cap)
            row[f"time_{name}_ms"] = round(1000 * (time.perf_counter() - t), 3)
            row[f"value_{name}"] = sum(values[j] for j in chosen)
            row[f"weight_{name}"] = sum(weights[j] for j in chosen)
        rows.append(row)
    return rows


def main():
    rows = compare_windows(load_frames(), load_config())
    out = ROOT / "results" / "dp_vs_greedy.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    better = sum(r["value_dp"] > r["value_greedy"] for r in rows)
    print(f"Вікон: {len(rows)}, ДП краще у {better}, однаково у {len(rows) - better}")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
