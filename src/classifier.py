"""classifier.py — класифікатор кадрів (Надя, чекпоінт 2).

Запуск звіту: python src/classifier.py

Власний багатошаровий перцептрон (1 прихований шар, ReLU, softmax).
Прямий прохід, backprop і SGD написані вручну. numpy використовується
лише як "калькулятор" для матриць — жодних бібліотек навчання
(sklearn / torch / keras тут немає).

Класи: 0 — норма, 1 — аномалія, 2 — перехідний.
"""
from __future__ import annotations

import csv
import math
import random
import json
import sys
from pathlib import Path

import numpy as np

from frame import Frame

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_CLASSES = 3
WINDOW = 8            # скільки останніх кадрів дивимось для ковзних ознак
F1_MIN = 0.58         # запасне значення; у звіті береться з team_config.json
FEATURE_NAMES = [
    "d_altitude",     # Δaltitude: висота зараз мінус висота попереднього кадру
    "d_temp",         # Δtemp
    "abs_vibration",  # |vibration|
    "std_vibration",  # ковзне стандартне відхилення вібрації за вікном
    "std_altitude",   # ковзне стандартне відхилення висоти за вікном
    "mode_change_freq",  # частка змін mode у вікні
    "critical",       # прапорець critical (0/1)
]


# --------------------------------------------------------------------------
# 1. Завантаження даних
# --------------------------------------------------------------------------
def load_labeled(path: str) -> tuple[list[Frame], list[int]]:
    """CSV: timestamp_ms,altitude_dm,temp_c10,vibration,mode,critical,label"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frame = Frame(
                timestamp_ms=int(r["timestamp_ms"]),
                altitude_dm=int(r["altitude_dm"]),
                temp_c10=int(r["temp_c10"]),
                vibration=int(r["vibration"]),
                mode=int(r["mode"]),
                critical=int(r["critical"]),
            )
            rows.append((frame, int(r["label"])))
    # ознаки з вікна історії мають сенс лише у хронологічному порядку
    rows.sort(key=lambda fr: fr[0].timestamp_ms)
    return [fr for fr, _ in rows], [lab for _, lab in rows]


# --------------------------------------------------------------------------
# 2. Ознаки з вікна історії
# --------------------------------------------------------------------------
def extract_features(frames: list[Frame], window: int = WINDOW) -> np.ndarray:
    """Матриця n × 7. Для кадру i використовуються лише кадри i-window+1 … i
    (тільки минуле), тож ознаки не «підглядають» у майбутнє."""
    n = len(frames)
    alt = [f.altitude_dm for f in frames]
    tmp = [f.temp_c10 for f in frames]
    vib = [f.vibration for f in frames]
    mode = [f.mode for f in frames]

    X = np.zeros((n, len(FEATURE_NAMES)), dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        d_alt = alt[i] - alt[i - 1] if i > 0 else 0
        d_tmp = tmp[i] - tmp[i - 1] if i > 0 else 0
        std_vib = float(np.std(vib[lo:i + 1]))
        std_alt = float(np.std(alt[lo:i + 1]))
        changes = sum(1 for j in range(max(lo, 1), i + 1) if mode[j] != mode[j - 1])
        X[i] = [d_alt, d_tmp, abs(vib[i]), std_vib, std_alt,
                changes / window, frames[i].critical]
    return X


# --------------------------------------------------------------------------
# 3. Метрики
# --------------------------------------------------------------------------
def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    """cm[true][pred] — рядок = справжній клас, стовпчик = передбачений."""
    k = max([N_CLASSES - 1, *y_true, *y_pred]) + 1
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def f1_per_class(y_true: list[int], y_pred: list[int]) -> list[float | None]:
    """F1 для кожного класу; None — клас не трапився ні в true, ні в pred."""
    cm = confusion_matrix(y_true, y_pred)
    k = len(cm)
    res: list[float | None] = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k)) - tp
        fn = sum(cm[c]) - tp
        denom = 2 * tp + fp + fn
        res.append(None if denom == 0 else 2 * tp / denom)
    return res


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Середнє арифметичне F1 по класах (кожен клас важить однаково,
    незалежно від того, скільки в ньому кадрів)."""
    vals = [v for v in f1_per_class(y_true, y_pred) if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------
# 4. Сам перцептрон
# --------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)       # стабільність: не буде exp(1000)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _init_params(n_in: int, n_hidden: int, n_out: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, math.sqrt(2.0 / n_in), (n_in, n_hidden)),   # He-ініціалізація під ReLU
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0.0, math.sqrt(1.0 / n_hidden), (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def _forward(p: dict, X: np.ndarray):
    z1 = X @ p["W1"] + p["b1"]
    h = np.maximum(z1, 0.0)                    # ReLU
    probs = _softmax(h @ p["W2"] + p["b2"])
    return z1, h, probs


def _loss_and_grads(p: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float):
    """Зважена крос-ентропія + L2. Повертає (loss, градієнти)."""
    m = len(y)
    z1, h, probs = _forward(p, X)
    loss = -np.sum(w * np.log(probs[np.arange(m), y] + 1e-12)) / m
    loss += 0.5 * l2 * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))

    # d loss / d logits = (softmax - one_hot) * вага_прикладу / m
    dz2 = probs.copy()
    dz2[np.arange(m), y] -= 1.0
    dz2 *= (w / m)[:, None]

    grads = {"W2": h.T @ dz2 + l2 * p["W2"], "b2": dz2.sum(axis=0)}
    dh = dz2 @ p["W2"].T
    dz1 = dh * (z1 > 0)                        # похідна ReLU
    grads["W1"] = X.T @ dz1 + l2 * p["W1"]
    grads["b1"] = dz1.sum(axis=0)
    return loss, grads


def _class_weights(y: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    """w_c = n / (k · n_c): рідкісні класи важать більше. Середня вага = 1."""
    counts = np.bincount(y, minlength=k).astype(float)
    counts[counts == 0] = 1.0
    return len(y) / (k * counts)


def _stratified_split(labels: list[int], val_frac: float, split_seed: int):
    """80/20 так, щоб у валідації були всі класи в тій самій пропорції."""
    rnd = random.Random(split_seed)
    train_idx, val_idx = [], []
    for c in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == c]
        rnd.shuffle(idx)
        n_val = max(1, round(len(idx) * val_frac)) if len(idx) > 1 else 0
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    return sorted(train_idx), sorted(val_idx)


def _train_one_seed(Xtr, ytr, Xval, yval, seed, hidden, epochs, lr, batch, l2):
    params = _init_params(Xtr.shape[1], hidden, N_CLASSES, seed)
    cw = _class_weights(ytr)
    rng = np.random.default_rng(seed)
    best_f1, best_params, best_epoch = -1.0, None, -1

    for epoch in range(epochs):
        lr_t = lr / (1.0 + 0.02 * epoch)              # повільне зменшення кроку
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            _, g = _loss_and_grads(params, Xtr[b], ytr[b], cw[ytr[b]], l2)
            for key in params:                        # SGD: θ ← θ − η·∇θ
                params[key] -= lr_t * g[key]

        pred = _forward(params, Xval)[2].argmax(axis=1)
        f1 = macro_f1(yval.tolist(), pred.tolist())
        if f1 > best_f1:                              # запам'ятовуємо найкращу епоху
            best_f1, best_epoch = f1, epoch
            best_params = {k: v.copy() for k, v in params.items()}
    return best_params, best_f1, best_epoch


def train_classifier(frames: list[Frame], labels: list[int], seeds=(0, 1, 2), *,
                     hidden: int = 16, epochs: int = 60, lr: float = 0.05,
                     batch: int = 32, l2: float = 1e-4, window: int = WINDOW,
                     val_frac: float = 0.2, split_seed: int = 42,
                     verbose: bool = False) -> dict:
    """Власний MLP. Ділить 80/20, вчить з кожного seed, повертає модель
    з найкращим macro-F1 на валідаційній частині."""
    X = extract_features(frames, window)
    y = np.asarray(labels, dtype=int)
    tr, va = _stratified_split(labels, val_frac, split_seed)

    # нормалізація: середнє і std рахуємо ТІЛЬКИ по train, щоб не було витоку
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    runs = []
    for seed in seeds:
        params, f1, ep = _train_one_seed(Xn[tr], y[tr], Xn[va], y[va],
                                         seed, hidden, epochs, lr, batch, l2)
        runs.append({"seed": seed, "val_macro_f1": f1, "best_epoch": ep, "params": params})
        if verbose:
            print(f"seed={seed}: val macro-F1 = {f1:.4f} (епоха {ep})")

    best = max(runs, key=lambda r: r["val_macro_f1"])
    return {
        **best["params"],
        "mean": mean, "std": std, "window": window,
        "seed": best["seed"], "val_macro_f1": best["val_macro_f1"],
        "all_runs": [{k: v for k, v in r.items() if k != "params"} for r in runs],
    }


def predict_classes(model: dict, frames: list[Frame]) -> list[int]:
    """Клас кожного кадру: 0 норма, 1 аномалія, 2 перехідний. Ознаки з вікна історії."""
    if not frames:
        return []
    X = (extract_features(frames, model["window"]) - model["mean"]) / model["std"]
    return _forward(model, X)[2].argmax(axis=1).tolist()


# --------------------------------------------------------------------------
# 5. Звіт: python src/classifier.py  (з кореня репозиторію)
# --------------------------------------------------------------------------
def _report(train_path, valid_path, config_path) -> float:
    with open(config_path, encoding="utf-8") as f:
        perc = json.load(f)["перцептрон"]
    f1_min = perc["F1_min"]
    frames, labels = load_labeled(train_path)
    print(f"train.csv: {len(frames)} кадрів, розподіл класів {np.bincount(labels, minlength=3).tolist()}")
    model = train_classifier(frames, labels, verbose=True)
    f1s = [r["val_macro_f1"] for r in model["all_runs"]]
    print(f"Розкид між ініціалізаціями: {min(f1s):.4f} … {max(f1s):.4f}")
    print(f"Обрано seed={model['seed']} (val macro-F1 = {model['val_macro_f1']:.4f})\n")

    vf, vl = load_labeled(valid_path)
    pred = predict_classes(model, vf)
    score = macro_f1(vl, pred)
    names = ["норма", "аномалія", "перехідний"]
    print(f"valid.csv: macro-F1 = {score:.4f}  (поріг F1_min = {f1_min}) "
          f"{'OK' if score >= f1_min else 'НИЖЧЕ ПОРОГУ'}")
    print(f"  базові лінії: мажоритарний клас {perc['базова_лінія_мажоритарний_клас']}, "
          f"випадковий {perc['базова_лінія_випадковий_вибір']}")
    for c, v in enumerate(f1_per_class(vl, pred)):
        print(f"  F1[{names[c]}] = {'—' if v is None else f'{v:.4f}'}")
    print("Матриця помилок (рядок = справжній, стовпчик = передбачений):")
    print("             " + "".join(f"{n:>12}" for n in names))
    for c, row in enumerate(confusion_matrix(vl, pred)):
        print(f"{names[c]:>12} " + "".join(f"{x:>12}" for x in row))
    return score


if __name__ == "__main__":
    # python src/classifier.py                       -> data/train.csv, data/valid.csv
    # python src/classifier.py train.csv valid.csv   -> свої файли
    args = sys.argv[1:]
    if len(args) not in (0, 2):
        print("Використання: python src/classifier.py [train.csv valid.csv]")
        sys.exit(1)
    train_p, valid_p = args if args else (DATA / "train.csv", DATA / "valid.csv")
    _report(train_p, valid_p, DATA / "team_config.json")
