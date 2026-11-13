"""Генетичний алгоритм для підбору параметрів передавача і порівняння з random search.

Хромосома з трьох генів:
    window          - розмір вікна кандидатів (8..64 кадри);
    critical_bonus  - бонус цінності кадру з critical = 1 (0..20);
    anomaly_weight  - пріоритет класу "аномалія" (1..10).
fitness = середнє Q за трьома сценаріями каналу (відкритий seed).
Після пошуку найкращі параметри перевіряються на інших seed каналу.

Запуск з кореня репозиторію: python src/ga.py
"""
import copy
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import load_labeled, predict_classes, train_classifier  # noqa: E402
from run_session import DATA, load_config, load_frames, run_scenario  # noqa: E402

BOUNDS = {"window": (8, 64), "critical_bonus": (0, 20), "anomaly_weight": (1, 10)}
GENES = list(BOUNDS)
BASELINE = {"window": 64, "critical_bonus": 5, "anomaly_weight": 5}
CHANNEL_SEEDS = (425243,)  # відкритий seed каналу: fitness
VALIDATION_SEEDS = (7, 99, 2026)  # інші seed: перевірка, що результат не підігнано


def apply_genes(config, genes):
    """Копія конфігурації з підставленими генами."""
    cfg = copy.deepcopy(config)
    cfg["пакування"]["candidate_window_frames"] = genes["window"]
    cfg["пріоритети_кадрів"]["critical_бонус"] = genes["critical_bonus"]
    cfg["пріоритети_кадрів"]["аномалія"] = genes["anomaly_weight"]
    return cfg


class Evaluator:
    """fitness = середнє Q. Рахує унікальні обчислення (бюджет) і кешує результати."""

    def __init__(self, frames, config, classes, seeds=CHANNEL_SEEDS):
        self.frames, self.config, self.classes, self.seeds = frames, config, classes, seeds
        self.cache = {}
        self.evaluations = 0

    def __call__(self, genes):
        key = tuple(genes[g] for g in GENES)
        if key not in self.cache:
            self.evaluations += 1
            cfg = apply_genes(self.config, genes)
            qs = []
            for seed in self.seeds:
                cfg["канал"]["seed_моделі_каналу"] = seed
                for name in cfg["канал"]["сценарії"]:
                    qs.append(run_scenario(self.frames, cfg, name, classes=self.classes)["Q"])
            self.cache[key] = sum(qs) / len(qs)
        return self.cache[key]


def random_genes(rng):
    return {g: rng.randint(*BOUNDS[g]) for g in GENES}


def crossover(a, b, rng):
    """Рівномірне схрещування: кожен ген береться від випадкового батька."""
    return {g: (a if rng.random() < 0.5 else b)[g] for g in GENES}


def mutate(genes, rng, rate=0.3):
    """Кожен ген із ймовірністю rate зсувається на випадкову величину в межах."""
    child = dict(genes)
    for g in GENES:
        if rng.random() < rate:
            lo, hi = BOUNDS[g]
            step = max(1, (hi - lo) // 4)
            child[g] = min(hi, max(lo, child[g] + rng.randint(-step, step)))
    return child


def tournament(population, scores, rng, k=3):
    picks = rng.sample(range(len(population)), k)
    return population[max(picks, key=lambda i: scores[i])]


def genetic_algorithm(fitness, rng, pop_size=8, budget=40, elite=1):
    """GA: популяція, турнірна селекція, схрещування, мутація, елітизм.

    Зупиняється, коли використано budget унікальних обчислень fitness.
    Повертає (найкращі гени, найкращий fitness, історія за поколіннями).
    """
    population = [dict(BASELINE)] + [random_genes(rng) for _ in range(pop_size - 1)]
    scores = [fitness(p) for p in population]
    history = [(0, max(scores), sum(scores) / len(scores))]
    generation = 0
    while fitness.evaluations < budget:
        generation += 1
        order = sorted(range(len(population)), key=lambda i: -scores[i])
        new_pop = [population[i] for i in order[:elite]]  # елітизм
        while len(new_pop) < pop_size:
            child = mutate(crossover(tournament(population, scores, rng),
                                     tournament(population, scores, rng), rng), rng)
            new_pop.append(child)
        population = new_pop
        scores = []
        for p in population:
            if fitness.evaluations >= budget and tuple(p[g] for g in GENES) not in fitness.cache:
                scores.append(float("-inf"))  # бюджет вичерпано, нового не рахуємо
            else:
                scores.append(fitness(p))
        valid = [s for s in scores if s != float("-inf")]
        history.append((generation, max(valid), sum(valid) / len(valid)))
    best = max(range(len(population)), key=lambda i: scores[i])
    return population[best], scores[best], history


def random_search(fitness, rng, budget=40):
    """Випадковий пошук з тим самим бюджетом обчислень fitness."""
    best_genes, best_score = None, float("-inf")
    while fitness.evaluations < budget:
        genes = random_genes(rng)
        score = fitness(genes)
        if score > best_score:
            best_genes, best_score = genes, score
    return best_genes, best_score


def main(budget=40, seeds=(0, 1, 2)):
    config = load_config()
    frames = load_frames()
    train_frames, labels = load_labeled(str(DATA / "train.csv"))
    classes = predict_classes(train_classifier(train_frames, labels), frames)

    base = Evaluator(frames, config, classes)(BASELINE)
    validator = Evaluator(frames, config, classes, seeds=VALIDATION_SEEDS)
    base_val = validator(BASELINE)
    print(f"Q_baseline (window=64, bonus=5, anomaly=5): {base:.4f}, на інших seed: {base_val:.4f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for seed in seeds:
        ga_fit = Evaluator(frames, config, classes)
        genes, score, history = genetic_algorithm(ga_fit, random.Random(seed), budget=budget)
        rs_fit = Evaluator(frames, config, classes)
        rs_genes, rs_score = random_search(rs_fit, random.Random(1000 + seed), budget=budget)
        initial, final = history[0][1], history[-1][1]
        assert final >= initial, "елітизм порушено"
        print(f"seed {seed}: GA {score:.4f} {genes} | random {rs_score:.4f} {rs_genes} "
              f"| поколінь {len(history) - 1}, оцінок {ga_fit.evaluations}/{rs_fit.evaluations}")
        ga_val, rs_val = validator(genes), validator(rs_genes)
        print(f"   на інших seed: GA {ga_val:.4f}, random {rs_val:.4f}")
        summary.append({"seed": seed, "ga_q": score, "ga_genes": genes,
                        "random_q": rs_score, "random_genes": rs_genes,
                        "ga_q_other_seeds": ga_val, "random_q_other_seeds": rs_val,
                        "initial_best": initial, "final_best": final})
        if seed == seeds[0]:
            with open(out_dir / "ga_history.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["generation", "best_fitness", "mean_fitness"])
                w.writerows(history)
    with open(out_dir / "ga_vs_random.json", "w", encoding="utf-8") as f:
        json.dump({"q_baseline": base, "q_baseline_other_seeds": base_val,
                   "budget": budget, "runs": summary}, f,
                  ensure_ascii=False, indent=2)
    ga_mean = sum(r["ga_q"] for r in summary) / len(summary)
    rs_mean = sum(r["random_q"] for r in summary) / len(summary)
    print(f"Середнє за {len(seeds)} запусками: GA {ga_mean:.4f}, random {rs_mean:.4f}, baseline {base:.4f}")


if __name__ == "__main__":
    main()