import copy

import pytest

from knapsack import knapsack_dp, knapsack_greedy
from run_session import load_config, load_frames, make_packets, plan_windows

SALT = bytes(16)
PREFIX = bytes(8)


def test_empty_stream_gives_no_packets():
    config = load_config()
    assert plan_windows([], config) == []
    assert make_packets([], config, SALT, PREFIX) == []


def test_single_frame_goes_into_one_packet():
    config = load_config()
    frames = load_frames()[:1]
    groups = plan_windows(frames, config)
    assert [g[0] for g in groups] == [[0]]


def test_zero_capacity_takes_nothing():
    assert knapsack_dp([5, 7], [1, 1], 0) == []
    assert knapsack_greedy([5, 7], [1, 1], 0) == []


def test_frame_bigger_than_payload_knapsack():
    # 20 байтів навантаження: після 14 байтів службових полів і таблиці місця для кадру нема
    config = copy.deepcopy(load_config())
    config["пакування"]["max_payload_bytes"] = 20
    with pytest.raises(ValueError):
        plan_windows(load_frames()[:5], config)


def test_frame_bigger_than_packet_sequential():
    # 80 байтів на весь пакет: 43 заголовок + 32 HMAC, на кадр лишається 5 байтів
    config = copy.deepcopy(load_config())
    config["пакування"]["max_packet_bytes"] = 80
    with pytest.raises(ValueError):
        make_packets(load_frames()[:5], config, SALT, PREFIX)
