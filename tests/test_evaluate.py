from evaluate import best_ga_genes, evaluate
from run_session import load_config, load_frames


def test_evaluate_returns_row_per_scenario_with_valid_metrics():
    config = load_config()
    frames = load_frames()[:300]
    rows = evaluate(frames, config, "test", packer="greedy")
    assert [r["scenario"] for r in rows] == list(config["канал"]["сценарії"])
    for r in rows:
        assert 0.0 <= r["Q"] <= 1.0
        assert 0.0 <= r["R"] <= 1.0
        assert r["threshold_met"] == (r["critical_loss"] <= r["threshold"] + 1e-9)


def test_best_ga_genes_has_all_genes():
    genes = best_ga_genes()
    assert genes is None or set(genes) == {"window", "critical_bonus", "anomaly_weight"}