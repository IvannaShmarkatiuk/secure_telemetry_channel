"""Тести для classifier.py: python -m pytest tests/test_classifier.py -v"""
import random
from pathlib import Path

import numpy as np
import pytest

from classifier import (Frame, load_labeled, F1_MIN, confusion_matrix, macro_f1, extract_features,
                        train_classifier, predict_classes, _init_params,
                        _loss_and_grads, _class_weights)


# ---------- метрики на прикладах, порахованих вручну ----------
def test_confusion_matrix():
    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 0, 2]
    assert confusion_matrix(y_true, y_pred) == [[1, 1, 0],
                                                [0, 1, 0],
                                                [1, 0, 2]]


def test_macro_f1_hand_example():
    # F1_0 = 2·1/(2+1+1)=0.5, F1_1 = 2·1/(2+1+0)=2/3, F1_2 = 2·2/(4+0+1)=0.8
    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 0, 2]
    assert abs(macro_f1(y_true, y_pred) - (0.5 + 2 / 3 + 0.8) / 3) < 1e-12


def test_macro_f1_perfect_and_all_majority():
    y = [0] * 8 + [1, 2]
    assert macro_f1(y, y) == 1.0
    # "все норма" має гарну accuracy (80%), але поганий macro-F1
    assert macro_f1(y, [0] * 10) < 0.35


# ---------- ознаки ----------
def test_features_shape_and_values():
    fr = [Frame(0, 100, 200, 3, 1, 0),
          Frame(100, 110, 195, -7, 2, 1)]
    X = extract_features(fr, window=4)
    assert X.shape == (2, 7)
    assert X[0][0] == 0 and X[0][1] == 0          # для першого кадру Δ = 0
    assert X[1][0] == 10 and X[1][1] == -5        # Δaltitude, Δtemp
    assert X[1][2] == 7                           # |vibration|
    assert X[1][5] == 1 / 4                       # одна зміна mode у вікні 4
    assert X[1][6] == 1                           # critical


# ---------- backprop: аналітичний градієнт = чисельному ----------
def test_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 7))
    y = np.array([0, 1, 2, 1, 0])
    w = _class_weights(y)[y]
    p = _init_params(7, 4, 3, seed=1)
    _, g = _loss_and_grads(p, X, y, w, l2=1e-3)
    eps = 1e-6
    for key in p:
        it = np.nditer(p[key], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[key][i]
            p[key][i] = old + eps
            lp, _ = _loss_and_grads(p, X, y, w, 1e-3)
            p[key][i] = old - eps
            lm, _ = _loss_and_grads(p, X, y, w, 1e-3)
            p[key][i] = old
            assert abs((lp - lm) / (2 * eps) - g[key][i]) < 1e-5


# ---------- навчання на синтетиці ----------
def make_synthetic(n=1500, seed=0):
    """Норма: тихо; аномалія: сплеск вібрації; перехідний: кадри після зміни mode."""
    r = random.Random(seed)
    frames, labels = [], []
    alt, temp, mode, since_change = 1000, 200, 0, 99
    for i in range(n):
        if r.random() < 0.02:
            mode = (mode + 1) % 3
            since_change = 0
        else:
            since_change += 1
        if since_change < 4:
            alt += r.randint(20, 40); lab = 2; vib = r.randint(-5, 5)
        elif r.random() < 0.08:
            vib = r.choice([-1, 1]) * r.randint(60, 120); lab = 1
        else:
            vib = r.randint(-5, 5); lab = 0
        alt += r.randint(-2, 2); temp += r.randint(-1, 1)
        frames.append(Frame(i * 100, alt, temp, vib, mode, int(lab == 1 and abs(vib) > 100)))
        labels.append(lab)
    return frames, labels


def test_train_beats_threshold_on_synthetic():
    frames, labels = make_synthetic()
    model = train_classifier(frames, labels, seeds=(0, 1, 2), epochs=30)
    assert len(model["all_runs"]) == 3                       # 3 ініціалізації
    assert model["val_macro_f1"] == max(r["val_macro_f1"] for r in model["all_runs"])
    vf, vl = make_synthetic(600, seed=7)
    pred = predict_classes(model, vf)
    assert len(pred) == len(vf) and set(pred) <= {0, 1, 2}
    assert macro_f1(vl, pred) >= 0.58


def test_predict_is_deterministic():
    frames, labels = make_synthetic(500)
    m1 = train_classifier(frames, labels, seeds=(0,), epochs=5)
    m2 = train_classifier(frames, labels, seeds=(0,), epochs=5)
    assert predict_classes(m1, frames) == predict_classes(m2, frames)


# ---------- справжні дані команди (data/train.csv, data/valid.csv) ----------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.skipif(not (DATA_DIR / "train.csv").exists(), reason="немає data/train.csv")
def test_real_data_load_and_threshold():
    frames, labels = load_labeled(DATA_DIR / "train.csv")
    assert len(frames) == 8000 and set(labels) == {0, 1, 2}
    ts = [f.timestamp_ms for f in frames]
    assert ts == sorted(ts)
    model = train_classifier(frames, labels)            # 3 seed, 80/20
    vf, vl = load_labeled(DATA_DIR / "valid.csv")
    assert macro_f1(vl, predict_classes(model, vf)) >= F1_MIN


def test_save_metrics_writes_json_atomically(tmp_path):
    import json
    from classifier import save_metrics
    out = save_metrics({"valid_macro_f1": 0.9, "матриця": [[1, 0], [0, 1]]}, tmp_path / "m.json")
    assert json.loads(out.read_text(encoding="utf-8"))["valid_macro_f1"] == 0.9
    assert not (tmp_path / "m.tmp").exists()          # тимчасовий файл не лишився
