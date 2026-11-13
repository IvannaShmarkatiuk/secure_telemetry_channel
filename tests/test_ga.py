import random

from classifier import predict_classes
from ga import (BASELINE, BOUNDS, GENES, Evaluator, apply_genes, crossover,
                genetic_algorithm, mutate, random_genes, random_search)
from run_session import load_config, load_frames


def small_evaluator():
    config = load_config()
    frames = load_frames()[:200]
    return Evaluator(frames, config, [0] * len(frames))


def in_bounds(genes):
    return all(BOUNDS[g][0] <= genes[g] <= BOUNDS[g][1] for g in GENES)


def test_random_genes_in_bounds():
    rng = random.Random(0)
    assert all(in_bounds(random_genes(rng)) for _ in range(100))


def test_mutation_and_crossover_stay_in_bounds():
    rng = random.Random(1)
    a, b = random_genes(rng), random_genes(rng)
    for _ in range(200):
        assert in_bounds(mutate(crossover(a, b, rng), rng, rate=1.0))


def test_crossover_takes_genes_from_parents():
    rng = random.Random(2)
    a, b = random_genes(rng), random_genes(rng)
    child = crossover(a, b, rng)
    assert all(child[g] in (a[g], b[g]) for g in GENES)


def test_apply_genes_does_not_change_original_config():
    config = load_config()
    before = config["пакування"]["candidate_window_frames"]
    cfg = apply_genes(config, {"window": 16, "critical_bonus": 9, "anomaly_weight": 3})
    assert cfg["пакування"]["candidate_window_frames"] == 16
    assert cfg["пріоритети_кадрів"]["critical_бонус"] == 9
    assert config["пакування"]["candidate_window_frames"] == before


def test_fitness_is_in_0_1_and_cached():
    fitness = small_evaluator()
    q = fitness(BASELINE)
    assert 0.0 < q <= 1.0
    assert fitness(BASELINE) == q
    assert fitness.evaluations == 1


def test_ga_elitism_best_never_decreases():
    fitness = small_evaluator()
    _, best, history = genetic_algorithm(fitness, random.Random(3), pop_size=4, budget=12)
    bests = [h[1] for h in history]
    assert bests[-1] >= bests[0]
    assert all(later >= earlier for earlier, later in zip(bests, bests[1:]))
    assert best == max(fitness.cache.values())


def test_ga_and_random_search_respect_budget():
    ga_fit, rs_fit = small_evaluator(), small_evaluator()
    genetic_algorithm(ga_fit, random.Random(4), pop_size=4, budget=10)
    random_search(rs_fit, random.Random(4), budget=10)
    assert ga_fit.evaluations <= 10 + 4  # поколінню дозволено дорахувати вже відомих
    assert rs_fit.evaluations == 10