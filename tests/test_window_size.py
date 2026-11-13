import copy

import pytest

from packet import parse_packet
from run_session import load_config, load_frames, make_packets_knapsack, plan_windows


def config_with_window(window):
    config = copy.deepcopy(load_config())
    config["пакування"]["candidate_window_frames"] = window
    return config


@pytest.mark.parametrize("window", [1, 16, 40, 64])
def test_every_frame_planned_exactly_once(window):
    frames = load_frames()
    groups = plan_windows(frames, config_with_window(window))
    planned = sorted(i for g in groups for i in g[0])
    assert planned == list(range(len(frames)))


def test_window_40_packets_fit_limit_and_parse():
    config = config_with_window(40)
    frames = load_frames()[:400]
    packets = make_packets_knapsack(frames, config, bytes(16), bytes(8))
    for packet, chosen in packets:
        assert len(packet) <= config["пакування"]["max_packet_bytes"]
        assert parse_packet(packet, config["пароль_сеансу"]) == chosen
