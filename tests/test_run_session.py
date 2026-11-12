from frame import Frame
from packet import parse_packet
from run_session import load_config, load_frames, make_packets, run_scenario
from run_session import make_packets_knapsack

def test_session_has_3000_frames():
    assert len(load_frames()) == 3000


def test_packets_fit_limit_and_cover_all_frames():
    config = load_config()
    frames = load_frames()
    packets = make_packets(frames, config, bytes(16), bytes(8))
    limit = config["пакування"]["max_packet_bytes"]
    assert all(len(p) <= limit for p, _ in packets)
    joined = [f for _, fs in packets for f in fs]
    assert joined == frames


def test_packets_are_parsable_and_contain_their_frames():
    config = load_config()
    frames = load_frames()[:200]
    packets = make_packets(frames, config, bytes(16), bytes(8))
    for packet, fs in packets:
        assert parse_packet(packet, config["пароль_сеансу"]) == fs


def test_calm_scenario_restores_all_critical():
    config = load_config()
    frames = load_frames()
    res = run_scenario(frames, config, "спокійний")
    assert res["R"] == 1.0
    assert 0.0 < res["Q"] <= 1.0


def test_greedy_packer_restores_everything_in_calm():
    config = load_config()
    frames = load_frames()
    res = run_scenario(frames, config, "спокійний", packer="greedy")
    assert res["restored_frames"] == frames


def test_emergency_scenario_loses_some_but_still_valid():
    config = load_config()
    frames = load_frames()
    res = run_scenario(frames, config, "аварійний")
    assert len(res["restored_frames"]) < len(frames)
    assert 0.0 <= res["R"] <= 1.0
    assert all(isinstance(f, Frame) for f in res["restored_frames"])

def test_knapsack_packets_fit_limit_and_parse():
    config = load_config()
    frames = load_frames()[:200]
    packets = make_packets_knapsack(frames, config, bytes(16), bytes(8))
    limit = config["пакування"]["max_packet_bytes"]
    for packet, fs in packets:
        assert len(packet) <= limit
        assert parse_packet(packet, config["пароль_сеансу"]) == fs


def test_knapsack_keeps_all_critical_frames():
    config = load_config()
    frames = load_frames()
    packets = make_packets_knapsack(frames, config, bytes(16), bytes(8))
    sent = {f.to_bytes() for _, fs in packets for f in fs}
    assert all(f.to_bytes() in sent for f in frames if f.critical)