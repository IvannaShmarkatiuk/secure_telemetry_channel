"""Фінальні приймальні перевірки: 12 базових кейсів C1 (умова проєкту, розділ 9.1).

Кожен тест має номер кейсу в назві, щоб на демо показати пункти по черзі:
    python -m pytest tests/test_final_checks.py -v
Прогін іде основним трактом run_session: рюкзак ДП + класи перцептрона.
"""
import json
import random
from pathlib import Path

import pytest

from classifier import load_labeled, macro_f1, predict_classes, train_classifier
from frame import Frame
from huffman import canonical_codes, huffman_decode, huffman_encode
from knapsack import knapsack_dp, knapsack_greedy
from packet import PacketError, build_packet, parse_packet
from run_session import DATA, load_config, load_frames, run_scenario

PASSWORD = "algo2026"
REF_DIR = DATA / "reference"
SALT = bytes(range(16))
NONCE = bytes.fromhex("010203040506070800000000")
FRAME = Frame(0, 1000, 215, -3, 1, 1)          # кадр із розділу 7 FORMAT_SPEC


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def frames():
    return load_frames()


@pytest.fixture(scope="module")
def model():
    train_frames, train_labels = load_labeled(str(DATA / "train.csv"))
    return train_classifier(train_frames, train_labels)


@pytest.fixture(scope="module")
def classes(model, frames):
    return predict_classes(model, frames)


def test_c01_single_frame_round_trip():
    packet = build_packet([FRAME], PASSWORD, 1, 0, SALT, NONCE)
    assert parse_packet(packet, PASSWORD) == [FRAME]
    assert parse_packet(packet, PASSWORD)[0].to_bytes() == FRAME.to_bytes()


def test_c02_empty_stream_does_not_crash():
    packet = build_packet([], PASSWORD, 1, 0, SALT, NONCE)
    assert parse_packet(packet, PASSWORD) == []


def test_c03_calm_scenario_restores_everything(frames, config, classes):
    result = run_scenario(frames, config, "спокійний", classes=classes)
    restored = sorted(result["restored_frames"], key=lambda f: f.timestamp_ms)
    assert restored == sorted(frames, key=lambda f: f.timestamp_ms)


def test_c04_working_scenario_meets_critical_loss_threshold(frames, config, classes):
    result = run_scenario(frames, config, "робочий", classes=classes)
    threshold = config["пороги_втрати_критичних_кадрів"]["робочий"]
    assert 1 - result["R"] <= threshold


def test_c05_emergency_scenario_runs_and_computes_q(frames, config, classes):
    result = run_scenario(frames, config, "аварійний", classes=classes)
    assert 0.0 <= result["Q"] <= 1.0


def test_c06_canonical_codes_from_lengths():
    # розділ 7: таблиця довжин -> канонічні коди -> бітовий потік 0C E6 DF 90
    data = FRAME.to_bytes()
    lengths = {0x00: 1, 0x01: 3, 0xFF: 3, 0x03: 4, 0xD7: 4, 0xE8: 4, 0xFD: 4}
    codes = canonical_codes(lengths)
    assert format(codes[0x01], "03b") == "100" and format(codes[0xFF], "03b") == "101"
    assert huffman_decode(lengths, bytes.fromhex("0CE6DF90"), 12) == data


def test_c07_degenerate_alphabet():
    data = b"\x41" * 24
    lengths, bitstream = huffman_encode(data)
    assert lengths == {0x41: 1}
    assert huffman_decode(lengths, bitstream, len(data)) == data


def test_c08_dp_not_worse_than_greedy():
    rng = random.Random(8)
    for _ in range(200):
        n = rng.randint(1, 20)
        weights = [rng.randint(1, 60) for _ in range(n)]
        values = [rng.randint(1, 30) for _ in range(n)]
        cap = rng.randint(0, 200)
        dp = sum(values[i] for i in knapsack_dp(weights, values, cap))
        greedy = sum(values[i] for i in knapsack_greedy(weights, values, cap))
        assert dp >= greedy


def test_c09_classifier_labels_every_frame(model, frames, classes, config):
    assert len(classes) == len(frames)
    assert set(classes) <= {0, 1, 2}
    valid_frames, valid_labels = load_labeled(str(DATA / "valid.csv"))
    f1 = macro_f1(valid_labels, predict_classes(model, valid_frames))
    assert f1 >= config["перцептрон"]["F1_min"]


def test_c10_tampered_byte_and_tag_are_rejected():
    for name in ("bad_01_tampered_byte.bin", "bad_02_tampered_tag.bin"):
        with pytest.raises(PacketError) as error:
            parse_packet((REF_DIR / name).read_bytes(), PASSWORD)
        assert error.value.code == "E_HMAC"


def test_c11_wrong_password_and_truncated_packet_are_rejected():
    packet = build_packet([FRAME], PASSWORD, 1, 0, SALT, NONCE)
    with pytest.raises(PacketError) as error:
        parse_packet(packet, "wrong-password")
    assert error.value.code == "E_HMAC"
    with pytest.raises(PacketError) as error:
        parse_packet(packet[:-9], PASSWORD)
    assert error.value.code == "E_LENGTH"


def test_c12_reference_packets_are_accepted():
    for meta_path in sorted(REF_DIR.glob("ref_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        packet = meta_path.with_suffix(".bin").read_bytes()
        assert parse_packet(packet, meta["пароль"]) == [Frame(**f) for f in meta["кадри"]]
