"""
Visualization utilities for GameAgent experiments.

Generates publication-quality plots for:
- Training trajectory (reward over iterations)
- Pareto front analysis (multi-objective trade-offs)
- Game performance heatmaps
- Benchmark comparison tables
- Nash-DPO convergence plots
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

COLORS = {
    "grpo": "#2196F3",
    "nash_dpo": "#FF5722",
    "baseline": "#9E9E9E",
    "sft": "#4CAF50",
    "accuracy": "#E91E63",
    "safety": "#00BCD4",
    "efficiency": "#FF9800",
    "creativity": "#9C27B0",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def plot_training_trajectory(log_path: str, output_path: str, title: str = "GRPO Training"):
    """Plot reward trajectory over GRPO iterations from training_log.json."""
    with open(log_path) as f:
        log = json.load(f)

    iterations = [entry["iteration"] for entry in log]
    games = list(log[0]["avg_payoffs_per_game"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Average payoff per game
    ax = axes[0]
    for game in games[:6]:
        payoffs = [entry["avg_payoffs_per_game"].get(game, 0) for entry in log]
        ax.plot(iterations, payoffs, marker="o", markersize=4, label=game.replace("_", " ").title())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Payoff")
    ax.set_title("Payoff per Game")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Strategy diversity
    ax = axes[1]
    for game in games[:6]:
        divs = [entry["eval_results"].get(game, {}).get("strategy_diversity", 0) for entry in log]
        ax.plot(iterations, divs, marker="s", markersize=4, label=game.replace("_", " ").title())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Shannon Entropy")
    ax.set_title("Strategy Diversity")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Nash distance
    ax = axes[2]
    for game in games[:6]:
        dists = [entry["eval_results"].get(game, {}).get("nash_distance", 0) for entry in log]
        ax.plot(iterations, dists, marker="^", markersize=4, label=game.replace("_", " ").title())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Nash Distance (L2)")
    ax.set_title("Nash Equilibrium Distance")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved training trajectory: %s", output_path)


def plot_nash_dpo_convergence(stats_dir: str, output_path: str):
    """Plot Elo ratings and preference pair counts across Nash-DPO iterations."""
    stats_files = sorted(Path(stats_dir).glob("iter*_stats.json"))
    if not stats_files:
        logger.warning("No Nash-DPO stats found in %s", stats_dir)
        return

    iterations = []
    elo_data = {}
    pair_counts = []

    for sf in stats_files:
        with open(sf) as f:
            data = json.load(f)
        iterations.append(data["iteration"])
        pair_counts.append(data["num_pairs"])
        for agent, rating in data["elo"].items():
            elo_data.setdefault(agent, []).append(rating)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for agent, ratings in elo_data.items():
        color = COLORS.get(agent, "#333333")
        ax1.plot(iterations, ratings, marker="o", color=color, linewidth=2, label=agent.title())
    ax1.axhline(y=1500, color="gray", linestyle="--", alpha=0.5, label="Baseline (1500)")
    ax1.set_xlabel("Nash-DPO Iteration")
    ax1.set_ylabel("Elo Rating")
    ax1.set_title("Agent Elo Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(iterations, pair_counts, color=COLORS["nash_dpo"], alpha=0.7)
    ax2.set_xlabel("Nash-DPO Iteration")
    ax2.set_ylabel("Preference Pairs")
    ax2.set_title("Training Signal per Iteration")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Nash-DPO Training Dynamics", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved Nash-DPO convergence: %s", output_path)


def plot_benchmark_comparison(results_path: str, output_path: str):
    """Bar chart comparing models across benchmarks from benchmark_results.json."""
    with open(results_path) as f:
        results = json.load(f)

    benchmarks = ["arc", "strategyqa", "bbh", "gsm8k", "truthfulqa", "mt_bench"]
    benchmark_labels = ["ARC", "StrategyQA", "BBH", "GSM8K", "TruthfulQA", "MT-Bench"]
    models = list(results.keys())

    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        values = []
        for bench in benchmarks:
            data = results[model].get(bench, {})
            if bench == "mt_bench":
                val = data.get("avg_score", 0) / 10
            else:
                val = data.get("accuracy", 0)
            values.append(val)
        color = COLORS.get(model, f"C{i}")
        bars = ax.bar(x + i * width, values, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Score")
    ax.set_title("GameAgent: Benchmark Comparison")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benchmark_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved benchmark comparison: %s", output_path)


def plot_pareto_front(agent_scores: dict, output_path: str):
    """
    Plot 2D projections of the 4D Pareto front.

    agent_scores: {model_name: {correctness: float, safety: float, efficiency: float, creativity: float}}
    """
    dimensions = ["correctness", "safety", "efficiency", "creativity"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        for model_name, scores in agent_scores.items():
            x = scores.get(dimensions[i], 0)
            y = scores.get(dimensions[j], 0)
            color = COLORS.get(model_name, "gray")
            ax.scatter(x, y, s=100, c=color, label=model_name, zorder=5)
            ax.annotate(model_name, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

        ax.set_xlabel(dimensions[i].title())
        ax.set_ylabel(dimensions[j].title())
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(agent_scores),
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Pareto Front Analysis: Multi-Objective Scores", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved Pareto front: %s", output_path)


def plot_game_heatmap(eval_results: dict, metric: str, output_path: str):
    """Heatmap of model × game performance."""
    models = list(eval_results.keys())
    games = []
    for model in models:
        for game in eval_results[model]:
            if game != "_summary" and game not in games:
                games.append(game)

    matrix = np.zeros((len(models), len(games)))
    for i, model in enumerate(models):
        for j, game in enumerate(games):
            matrix[i, j] = eval_results[model].get(game, {}).get(metric, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(games)), max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(games)))
    ax.set_xticklabels([g.replace("_", "\n") for g in games], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(games)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label=metric.replace("_", " ").title())
    ax.set_title(f"Game Performance: {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved game heatmap: %s", output_path)


def plot_transfer_matrix(transfer_results_path: str, output_path: str):
    """Plot cross-game transfer matrix from transfer_results.json.

    Supports two JSON layouts:
    - Nested: single_game[train][eval][eval_game][avg_payoff]
    - Flat:   single_game[train][avg_payoff] (per-game results directly)
    """
    with open(transfer_results_path) as f:
        data = json.load(f)

    if "single_game" not in data:
        logger.warning("No single_game data in transfer results")
        return

    train_games = list(data["single_game"].keys())
    first = data["single_game"][train_games[0]]

    if "eval" in first and isinstance(first["eval"], dict):
        eval_games = list(first["eval"].keys())
        matrix = np.zeros((len(train_games), len(eval_games)))
        for i, tg in enumerate(train_games):
            for j, eg in enumerate(eval_games):
                matrix[i, j] = data["single_game"][tg].get("eval", {}).get(eg, {}).get("avg_payoff", 0)
    else:
        eval_games = train_games
        matrix = np.zeros((len(train_games), len(eval_games)))
        for i, tg in enumerate(train_games):
            matrix[i, i] = data["single_game"][tg].get("avg_payoff", 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(eval_games)))
    ax.set_xticklabels([g.replace("_", "\n") for g in eval_games], rotation=45, ha="right")
    ax.set_yticks(range(len(train_games)))
    ax.set_yticklabels(train_games)
    ax.set_xlabel("Evaluation Game")
    ax.set_ylabel("Training Game")

    for i in range(len(train_games)):
        for j in range(len(eval_games)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                   fontsize=8, color="white" if matrix[i, j] < matrix.mean() else "black")

    plt.colorbar(im, label="Avg Payoff")
    ax.set_title("Cross-Game Transfer Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved transfer matrix: %s", output_path)
