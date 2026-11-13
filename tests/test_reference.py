import json
from pathlib import Path

import pytest

from frame import Frame
from packet import PacketError, build_packet, parse_packet

# Папка з еталонними файлами 
REF_DIR = Path(__file__).parent.parent / "data" / "reference"

# Правильні пакети: ref_01 ... ref_06
REF_NAMES = [
    "ref_01_single_frame",
    "ref_02_empty",
    "ref_03_degenerate",
    "ref_04_session",
    "ref_05_no_compression",
    "ref_06_class_table",
]

# Зіпсовані пакети і коди
EXPECTED_FAILURES = json.loads((REF_DIR / "expected_failures.json").read_text(encoding="utf-8"))


def load_ref(name):
    """Повертає байти пакета і його опис із json-файлу."""
    data = (REF_DIR / f"{name}.bin").read_bytes()
    meta = json.loads((REF_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return data, meta


@pytest.mark.parametrize("name", REF_NAMES)
def test_reference_packet_is_parsed(name):
    data, meta = load_ref(name)
    frames = parse_packet(data, meta["пароль"])
    expected = [Frame(**f) for f in meta["кадри"]]
    assert frames == expected


@pytest.mark.parametrize("name", [n for n in REF_NAMES if load_ref(n)[1]["flags"] == 0])
def test_reference_packet_is_built(name):
    data, meta = load_ref(name)
    frames = [Frame(**f) for f in meta["кадри"]]
    device_id = int.from_bytes(data[5:7], "big")
    packet_seq = int.from_bytes(data[7:11], "big")
    packet = build_packet(frames, meta["пароль"], device_id, packet_seq,
                          bytes.fromhex(meta["salt_hex"]), bytes.fromhex(meta["nonce_hex"]))
    assert packet == data


@pytest.mark.parametrize("name", sorted(EXPECTED_FAILURES))
def test_bad_packet_is_rejected(name):
    data = (REF_DIR / name).read_bytes()
    with pytest.raises(PacketError) as error:
        parse_packet(data, "algo2026")
    assert error.value.code == EXPECTED_FAILURES[name]["очікуваний_код"]


def test_no_compression_packet_is_built():
    # ref_05: flags = 0x08, таблиці немає, кадри лежать сирими байтами
    data, meta = load_ref("ref_05_no_compression")
    frames = [Frame(**f) for f in meta["кадри"]]
    device_id = int.from_bytes(data[5:7], "big")
    packet_seq = int.from_bytes(data[7:11], "big")
    packet = build_packet(frames, meta["пароль"], device_id, packet_seq,
                          bytes.fromhex(meta["salt_hex"]), bytes.fromhex(meta["nonce_hex"]),
                          compress=False)
    assert packet == data
