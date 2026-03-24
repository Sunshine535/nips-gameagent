#!/usr/bin/env python3
"""Collect experiment results and generate visualizations.

Run after pipeline completes to create plots and a progress report.

Usage:
    python scripts/collect_and_visualize.py --results_dir results
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_results(results_dir: str) -> dict:
    """Scan results directory and collect all JSON result files."""
    results_dir = Path(results_dir)
    collected = {}

    # GRPO training log
    grpo_log = results_dir / "grpo_self_play" / "training_log.json"
    if grpo_log.exists():
        with open(grpo_log) as f:
            collected["grpo_training"] = json.load(f)
        logger.info("Found GRPO training log: %d iterations", len(collected["grpo_training"]))

    # Nash-DPO stats
    nash_dir = results_dir / "nash_dpo"
    if nash_dir.exists():
        stats = []
        for sf in sorted(nash_dir.glob("iter*_stats.json")):
            with open(sf) as f:
                stats.append(json.load(f))
        if stats:
            collected["nash_dpo_stats"] = stats
            logger.info("Found Nash-DPO stats: %d iterations", len(stats))

        summary = nash_dir / "nash_dpo_summary.json"
        if summary.exists():
            with open(summary) as f:
                collected["nash_dpo_summary"] = json.load(f)

    # Benchmark results
    bench_path = results_dir / "eval_benchmarks" / "benchmark_results.json"
    if bench_path.exists():
        with open(bench_path) as f:
            collected["benchmarks"] = json.load(f)
        logger.info("Found benchmark results: %d models", len(collected["benchmarks"]))

    # Cross-game transfer
    transfer_path = results_dir / "cross_game_transfer" / "transfer_results.json"
    if transfer_path.exists():
        with open(transfer_path) as f:
            collected["transfer"] = json.load(f)
        logger.info("Found transfer results")

    # GRPO vs Nash comparison
    comparison_path = results_dir / "grpo_vs_nash" / "grpo_vs_nash_results.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            collected["comparison"] = json.load(f)
        logger.info("Found GRPO vs Nash comparison")

    # Game evaluation
    game_eval_path = results_dir / "game_eval" / "game_eval_results.json"
    if game_eval_path.exists():
        with open(game_eval_path) as f:
            collected["game_eval"] = json.load(f)
        logger.info("Found game evaluation results")

    # Ablations
    ablation_dirs = list(results_dir.glob("ablation_*"))
    if ablation_dirs:
        collected["ablations"] = {}
        for ad in ablation_dirs:
            name = ad.name
            for jf in ad.glob("*.json"):
                with open(jf) as f:
                    collected["ablations"][f"{name}/{jf.name}"] = json.load(f)
        logger.info("Found %d ablation result files", len(collected.get("ablations", {})))

    return collected


def generate_visualizations(collected: dict, output_dir: str, results_dir: str = "results"):
    """Generate all plots from collected results."""
    from src.visualization import (
        plot_training_trajectory,
        plot_nash_dpo_convergence,
        plot_benchmark_comparison,
        plot_game_heatmap,
        plot_transfer_matrix,
    )

    output_dir = Path(output_dir)
    results_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "grpo_training" in collected:
        log_path = results_dir / "grpo_self_play" / "training_log.json"
        if log_path.exists():
            plot_training_trajectory(str(log_path), str(output_dir / "grpo_trajectory.png"))

    if "nash_dpo_stats" in collected:
        plot_nash_dpo_convergence(str(results_dir / "nash_dpo"),
                                 str(output_dir / "nash_dpo_convergence.png"))

    if "benchmarks" in collected:
        bench_path = results_dir / "eval_benchmarks" / "benchmark_results.json"
        plot_benchmark_comparison(str(bench_path),
                                 str(output_dir / "benchmark_comparison.png"))

    if "transfer" in collected:
        transfer_path = results_dir / "cross_game_transfer" / "transfer_results.json"
        plot_transfer_matrix(str(transfer_path),
                            str(output_dir / "transfer_matrix.png"))

    if "game_eval" in collected:
        plot_game_heatmap(collected["game_eval"], "avg_payoff",
                         str(output_dir / "game_payoff_heatmap.png"))
        plot_game_heatmap(collected["game_eval"], "nash_rate",
                         str(output_dir / "game_nash_rate_heatmap.png"))


def generate_progress_report(collected: dict, output_path: str):
    """Generate an HTML progress report."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GameAgent: Research Progress</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
h2 { color: #283593; margin-top: 30px; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
table { border-collapse: collapse; width: 100%; }
th { background: #1a237e; color: white; padding: 10px; text-align: left; }
td { padding: 8px; border-bottom: 1px solid #eee; }
tr:hover { background: #f5f5f5; }
.metric { font-size: 2em; font-weight: bold; color: #1a237e; }
.label { color: #666; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
img { max-width: 100%; border-radius: 4px; }
.status-ok { color: #2e7d32; font-weight: bold; }
.status-pending { color: #f57f17; font-weight: bold; }
</style></head><body>
<h1>GameAgent: Research Progress Report</h1>
<p>Generated: """ + __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
"""

    # Summary
    html += '<div class="card"><h2>Summary</h2><div class="grid">'
    has_grpo = "grpo_training" in collected
    has_nash = "nash_dpo_stats" in collected
    has_bench = "benchmarks" in collected

    html += f'<div><div class="metric">{"✓" if has_grpo else "○"}</div><div class="label">GRPO Training</div></div>'
    html += f'<div><div class="metric">{"✓" if has_nash else "○"}</div><div class="label">Nash-DPO Training</div></div>'
    html += f'<div><div class="metric">{"✓" if has_bench else "○"}</div><div class="label">Benchmark Evaluation</div></div>'
    html += '</div></div>'

    # Benchmark table
    if "benchmarks" in collected:
        html += '<div class="card"><h2>Benchmark Results</h2><table><tr><th>Model</th><th>ARC</th><th>StrategyQA</th><th>BBH</th><th>GSM8K</th><th>TruthfulQA</th><th>MT-Bench</th></tr>'
        for label, res in collected["benchmarks"].items():
            html += f'<tr><td><b>{label}</b></td>'
            for bench in ["arc", "strategyqa", "bbh", "gsm8k", "truthfulqa", "mt_bench"]:
                data = res.get(bench, {})
                if bench == "mt_bench":
                    val = data.get("avg_score", "-")
                    html += f'<td>{val:.2f}</td>' if isinstance(val, float) else f'<td>{val}</td>'
                else:
                    val = data.get("accuracy", "-")
                    html += f'<td>{val:.4f}</td>' if isinstance(val, float) else f'<td>{val}</td>'
            html += '</tr>'
        html += '</table></div>'

    # GRPO training
    if "grpo_training" in collected:
        html += '<div class="card"><h2>GRPO Self-Play Training</h2>'
        html += '<img src="grpo_trajectory.png" alt="GRPO Training Trajectory">'
        log = collected["grpo_training"]
        html += f'<p>Iterations completed: {len(log)}</p>'
        if log:
            last = log[-1]
            html += '<table><tr><th>Game</th><th>Avg Payoff</th><th>Diversity</th><th>Nash Distance</th></tr>'
            for game, payoff in last.get("avg_payoffs_per_game", {}).items():
                div = last.get("eval_results", {}).get(game, {}).get("strategy_diversity", 0)
                nd = last.get("eval_results", {}).get(game, {}).get("nash_distance", 0)
                html += f'<tr><td>{game}</td><td>{payoff:.3f}</td><td>{div:.3f}</td><td>{nd:.3f}</td></tr>'
            html += '</table>'
        html += '</div>'

    # Nash-DPO
    if "nash_dpo_stats" in collected:
        html += '<div class="card"><h2>Nash-DPO Training</h2>'
        html += '<img src="nash_dpo_convergence.png" alt="Nash-DPO Convergence">'
        stats = collected["nash_dpo_stats"]
        if stats:
            last = stats[-1]
            html += f'<p>Iterations: {len(stats)}, Final pairs: {last.get("num_pairs", 0)}</p>'
            html += '<p>Final Elo ratings: ' + ", ".join(
                f'{k}: {v:.0f}' for k, v in last.get("elo", {}).items()
            ) + '</p>'
        html += '</div>'

    # Plots
    plots_dir = Path(output_path).parent
    plot_files = list(plots_dir.glob("*.png"))
    if plot_files:
        html += '<div class="card"><h2>Visualizations</h2>'
        for pf in plot_files:
            html += f'<h3>{pf.stem.replace("_", " ").title()}</h3>'
            html += f'<img src="{pf.name}" alt="{pf.stem}">'
        html += '</div>'

    html += '</body></html>'

    with open(output_path, "w") as f:
        f.write(html)
    logger.info("Progress report saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Collect results and generate visualizations")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="to_human")
    args = parser.parse_args()

    logger.info("Collecting results from %s...", args.results_dir)
    collected = collect_results(args.results_dir)

    if not collected:
        logger.warning("No results found. Pipeline may not have completed yet.")
        return

    logger.info("Generating visualizations...")
    generate_visualizations(collected, args.output_dir, args.results_dir)

    logger.info("Generating progress report...")
    generate_progress_report(collected, os.path.join(args.output_dir, "progress_report.html"))

    # Save collected data
    with open(os.path.join(args.output_dir, "collected_results.json"), "w") as f:
        json.dump(collected, f, indent=2, default=str)

    logger.info("All done! Open %s/progress_report.html to view.", args.output_dir)


if __name__ == "__main__":
    main()
