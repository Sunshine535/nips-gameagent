#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables and figures from experiment results.

Run after pipeline completes:
    python scripts/generate_paper_content.py --results_dir results --output_dir paper/generated

Generates:
  - LaTeX tables (Table 2-5 from outline)
  - Matplotlib figures (Figure 1-4)
  - Summary statistics for Abstract/Introduction
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

GAME_ORDER = [
    'prisoners_dilemma', 'coordination_game', 'battle_of_sexes',
    'stag_hunt', 'chicken', 'matching_pennies',
    'public_goods', 'auction', 'ultimatum', 'negotiation'
]
GAME_SHORT = {
    'prisoners_dilemma': 'PD', 'coordination_game': 'CG',
    'battle_of_sexes': 'BoS', 'stag_hunt': 'SH',
    'chicken': 'Chk', 'matching_pennies': 'MP',
    'public_goods': 'PG', 'auction': 'Auc',
    'ultimatum': 'Ult', 'negotiation': 'Neg'
}

BENCHMARK_ORDER = ['arc', 'strategyqa', 'bbh', 'gsm8k', 'truthfulqa', 'mt_bench']
BENCHMARK_LABELS = ['ARC-C', 'StrategyQA', 'BBH', 'GSM8K', 'TruthfulQA', 'MT-Bench']

CONDITION_ORDER = ['base', 'sft', 'grpo', 'nash_dpo', 'grpo+nash']
CONDITION_LABELS = ['Base', 'SFT-only', 'A-only (GRPO)', 'B-only (Nash-DPO)', 'A+B']


def load_all_results(results_dir):
    rd = Path(results_dir)
    data = {}

    # GRPO training log
    p = rd / 'grpo_self_play' / 'training_log.json'
    if p.exists():
        with open(p) as f:
            data['grpo_log'] = json.load(f)

    # Nash-DPO stats
    nash_dir = rd / 'nash_dpo'
    if nash_dir.exists():
        stats = []
        for sf in sorted(nash_dir.glob('iter*_stats.json')):
            with open(sf) as f:
                stats.append(json.load(f))
        if stats:
            data['nash_stats'] = stats
        summary = nash_dir / 'nash_dpo_summary.json'
        if summary.exists():
            with open(summary) as f:
                data['nash_summary'] = json.load(f)

    # Benchmarks
    p = rd / 'eval_benchmarks' / 'benchmark_results.json'
    if p.exists():
        with open(p) as f:
            data['benchmarks'] = json.load(f)

    # Transfer
    p = rd / 'cross_game_transfer' / 'transfer_results.json'
    if p.exists():
        with open(p) as f:
            data['transfer'] = json.load(f)

    # GRPO vs Nash
    p = rd / 'grpo_vs_nash' / 'grpo_vs_nash_results.json'
    if p.exists():
        with open(p) as f:
            data['comparison'] = json.load(f)

    # Ablations
    for abl_dir in rd.glob('ablation_*'):
        for jf in abl_dir.glob('*.json'):
            key = f'ablation_{abl_dir.name}_{jf.stem}'
            with open(jf) as f:
                data[key] = json.load(f)

    return data


# ========= LaTeX Table Generators =========

def generate_table2_game_performance(data, outdir):
    """Table 2: Game performance across 10 games for all conditions."""
    if 'grpo_log' not in data:
        logger.warning('No GRPO log for Table 2')
        return

    log = data['grpo_log']
    if not log:
        return

    last_iter = log[-1]
    games = [g for g in GAME_ORDER if g in last_iter.get('avg_payoffs_per_game', {})]

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Game-theoretic performance across 10 strategic games after GRPO self-play training (3 iterations, 2000 episodes/iter). We report average payoff, strategy diversity (Shannon entropy), and Nash equilibrium distance ($).}')
    lines.append(r'\label{tab:game_performance}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l' + 'ccc' * 1 + '}')
    lines.append(r'\toprule')
    lines.append(r'Game & Avg Payoff & Diversity & Nash Dist. \')
    lines.append(r'\midrule')

    for game in games:
        payoff = last_iter['avg_payoffs_per_game'].get(game, 0)
        evals = last_iter.get('eval_results', {}).get(game, {})
        diversity = evals.get('strategy_diversity', 0)
        nash_dist = evals.get('nash_distance', 0)
        short = GAME_SHORT.get(game, game)
        lines.append(f'{short} & {payoff:.3f} & {diversity:.3f} & {nash_dist:.3f} \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(outdir / 'table2_game_performance.tex', 'w') as f:
        f.write('\n'.join(lines))
    logger.info('Generated Table 2')


def generate_table3_benchmarks(data, outdir):
    """Table 3: Factorial benchmark results."""
    if 'benchmarks' not in data:
        logger.warning('No benchmark data for Table 3')
        return

    bench = data['benchmarks']
    models = list(bench.keys())

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Downstream transfer evaluation on standard benchmarks. Factorial design: Base, SFT-only, GRPO-only (Track A), Nash-DPO-only (Track B), and A+B combined.}')
    lines.append(r'\label{tab:benchmarks}')
    lines.append(r'\small')
    ncols = len(BENCHMARK_ORDER)
    lines.append(r'\begin{tabular}{l' + 'c' * ncols + '}')
    lines.append(r'\toprule')
    header = 'Model & ' + ' & '.join(BENCHMARK_LABELS) + r' \'
    lines.append(header)
    lines.append(r'\midrule')

    for model in models:
        row = [model.replace('_', r'\_')]
        for bm in BENCHMARK_ORDER:
            d = bench[model].get(bm, {})
            if bm == 'mt_bench':
                val = d.get('avg_score', None)
                row.append(f'{val:.2f}' if val is not None else '--')
            else:
                val = d.get('accuracy', None)
                row.append(f'{val:.4f}' if val is not None else '--')
        lines.append(' & '.join(row) + r' \')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(outdir / 'table3_benchmarks.tex', 'w') as f:
        f.write('\n'.join(lines))
    logger.info('Generated Table 3')


def generate_table5_ablations(data, outdir):
    """Table 5: Ablation study results."""
    ablation_keys = [k for k in data if k.startswith('ablation_')]
    if not ablation_keys:
        logger.warning('No ablation data for Table 5')
        return

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation study results. Each ablation removes one component from the full pipeline.}')
    lines.append(r'\label{tab:ablations}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lcc}')
    lines.append(r'\toprule')
    lines.append(r'Condition & Avg Payoff & Benchmark Avg \')
    lines.append(r'\midrule')

    for k in sorted(ablation_keys):
        name = k.replace('ablation_', '').replace('_', ' ').title()
        d = data[k]
        payoff = d.get('avg_payoff', '--')
        bench = d.get('benchmark_avg', '--')
        if isinstance(payoff, (int, float)):
            payoff = f'{payoff:.3f}'
        if isinstance(bench, (int, float)):
            bench = f'{bench:.4f}'
        lines.append(f'{name} & {payoff} & {bench} \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(outdir / 'table5_ablations.tex', 'w') as f:
        f.write('\n'.join(lines))
    logger.info('Generated Table 5')


# ========= Figure Generators =========

def generate_figure1_training_trajectory(data, outdir):
    """Figure 1: GRPO training trajectory (payoff + diversity over iterations)."""
    if 'grpo_log' not in data:
        return

    log = data['grpo_log']
    iterations = [e['iteration'] for e in log]
    games = [g for g in GAME_ORDER if g in log[0].get('avg_payoffs_per_game', {})][:6]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(games)))

    for gi, game in enumerate(games):
        payoffs = [e['avg_payoffs_per_game'].get(game, 0) for e in log]
        axes[0].plot(iterations, payoffs, 'o-', color=colors[gi], markersize=5,
                     label=game.replace('_', ' ').title(), linewidth=1.5)

    axes[0].set_xlabel('GRPO Iteration')
    axes[0].set_ylabel('Average Payoff')
    axes[0].set_title('(a) Payoff Trajectory')
    axes[0].legend(fontsize=7, ncol=2, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    for gi, game in enumerate(games):
        divs = [e.get('eval_results', {}).get(game, {}).get('strategy_diversity', 0) for e in log]
        axes[1].plot(iterations, divs, 's-', color=colors[gi], markersize=5,
                     label=game.replace('_', ' ').title(), linewidth=1.5)

    axes[1].set_xlabel('GRPO Iteration')
    axes[1].set_ylabel('Shannon Entropy')
    axes[1].set_title('(b) Strategy Diversity')
    axes[1].legend(fontsize=7, ncol=2, loc='lower right')
    axes[1].grid(True, alpha=0.3)

    for gi, game in enumerate(games):
        dists = [e.get('eval_results', {}).get(game, {}).get('nash_distance', 0) for e in log]
        axes[2].plot(iterations, dists, '^-', color=colors[gi], markersize=5,
                     label=game.replace('_', ' ').title(), linewidth=1.5)

    axes[2].set_xlabel('GRPO Iteration')
    axes[2].set_ylabel('Nash Distance ($)')
    axes[2].set_title('(c) Nash Equilibrium Distance')
    axes[2].legend(fontsize=7, ncol=2, loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'figure1_training_trajectory.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(outdir / 'figure1_training_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Generated Figure 1')


def generate_figure2_transfer_heatmap(data, outdir):
    """Figure 2: Cross-game transfer heatmap."""
    if 'transfer' not in data:
        return

    td = data['transfer']
    if 'single_game' not in td:
        return

    sg = td['single_game']
    train_games = [g for g in GAME_ORDER if g in sg]

    first = sg[train_games[0]]
    if 'eval' in first and isinstance(first['eval'], dict):
        eval_games = list(first['eval'].keys())
        eval_games = [g for g in GAME_ORDER if g in eval_games]
        matrix = np.zeros((len(train_games), len(eval_games)))
        for i, tg in enumerate(train_games):
            for j, eg in enumerate(eval_games):
                matrix[i, j] = sg[tg].get('eval', {}).get(eg, {}).get('avg_payoff', 0)
    else:
        eval_games = train_games
        matrix = np.zeros((len(train_games), len(eval_games)))
        for i, tg in enumerate(train_games):
            matrix[i, i] = sg[tg].get('avg_payoff', 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(eval_games)))
    ax.set_xticklabels([GAME_SHORT.get(g, g) for g in eval_games], rotation=45, ha='right')
    ax.set_yticks(range(len(train_games)))
    ax.set_yticklabels([GAME_SHORT.get(g, g) for g in train_games])
    ax.set_xlabel('Evaluation Game')
    ax.set_ylabel('Training Game')

    for i in range(len(train_games)):
        for j in range(len(eval_games)):
            color = 'white' if matrix[i, j] < matrix.mean() else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, label='Average Payoff', shrink=0.8)
    ax.set_title('Cross-Game Transfer Matrix')
    plt.tight_layout()
    plt.savefig(outdir / 'figure2_transfer_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(outdir / 'figure2_transfer_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Generated Figure 2')


def generate_figure3_nash_convergence(data, outdir):
    """Figure 3: Nash-DPO Elo convergence."""
    if 'nash_stats' not in data:
        return

    stats = data['nash_stats']
    iterations = [s['iteration'] for s in stats]
    agents = list(stats[0].get('elo', {}).keys())

    agent_colors = {'accuracy': '#E91E63', 'safety': '#00BCD4',
                    'efficiency': '#FF9800', 'creativity': '#9C27B0'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for agent in agents:
        elos = [s['elo'].get(agent, 1500) for s in stats]
        c = agent_colors.get(agent, 'gray')
        ax1.plot(iterations, elos, 'o-', color=c, linewidth=2, markersize=6, label=agent.title())

    ax1.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xlabel('Nash-DPO Iteration')
    ax1.set_ylabel('Elo Rating')
    ax1.set_title('(a) Agent Elo Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    pairs = [s.get('num_pairs', 0) for s in stats]
    ax2.bar(iterations, pairs, color='#FF5722', alpha=0.7, width=0.6)
    ax2.set_xlabel('Nash-DPO Iteration')
    ax2.set_ylabel('Preference Pairs')
    ax2.set_title('(b) Training Signal per Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'figure3_nash_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(outdir / 'figure3_nash_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Generated Figure 3')


def generate_figure4_benchmark_comparison(data, outdir):
    """Figure 4: Grouped bar chart of benchmark comparisons."""
    if 'benchmarks' not in data:
        return

    bench = data['benchmarks']
    models = list(bench.keys())
    model_colors = {'base': '#9E9E9E', 'sft': '#4CAF50', 'grpo': '#2196F3',
                    'nash_dpo': '#FF5722', 'grpo+nash': '#673AB7'}

    x = np.arange(len(BENCHMARK_ORDER))
    width = 0.15
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for mi, model in enumerate(models):
        values = []
        for bm in BENCHMARK_ORDER:
            d = bench[model].get(bm, {})
            if bm == 'mt_bench':
                val = d.get('avg_score', 0) / 10
            else:
                val = d.get('accuracy', 0)
            values.append(val)

        offset = (mi - n_models / 2 + 0.5) * width
        color = model_colors.get(model, f'C{mi}')
        bars = ax.bar(x + offset, values, width, label=model, color=color, alpha=0.85)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Score (accuracy or normalized)')
    ax.set_title('Downstream Benchmark Performance: Factorial Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARK_LABELS)
    ax.legend(loc='upper left', ncol=n_models)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outdir / 'figure4_benchmark_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(outdir / 'figure4_benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Generated Figure 4')


# ========= Summary Stats =========

def generate_summary_stats(data, outdir):
    """Generate key statistics for Abstract and Introduction."""
    stats = {}

    if 'grpo_log' in data and data['grpo_log']:
        log = data['grpo_log']
        first = log[0]
        last = log[-1]

        first_avg = np.mean(list(first.get('avg_payoffs_per_game', {}).values()))
        last_avg = np.mean(list(last.get('avg_payoffs_per_game', {}).values()))
        improvement = ((last_avg - first_avg) / max(abs(first_avg), 1e-6)) * 100

        stats['grpo_iterations'] = len(log)
        stats['grpo_games'] = len(last.get('avg_payoffs_per_game', {}))
        stats['grpo_initial_payoff'] = first_avg
        stats['grpo_final_payoff'] = last_avg
        stats['grpo_payoff_improvement_pct'] = improvement

        if last.get('eval_results'):
            nash_dists = [v.get('nash_distance', 0) for v in last['eval_results'].values()]
            stats['grpo_final_avg_nash_dist'] = np.mean(nash_dists)

    if 'benchmarks' in data:
        bench = data['benchmarks']
        for model, results in bench.items():
            accs = []
            for bm in ['arc', 'strategyqa', 'bbh', 'gsm8k', 'truthfulqa']:
                val = results.get(bm, {}).get('accuracy')
                if val is not None:
                    accs.append(val)
            if accs:
                stats[f'bench_avg_{model}'] = np.mean(accs)

    if 'nash_stats' in data:
        last_nash = data['nash_stats'][-1]
        stats['nash_iterations'] = len(data['nash_stats'])
        stats['nash_final_elo'] = last_nash.get('elo', {})

    with open(outdir / 'summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    with open(outdir / 'summary_stats.md', 'w') as f:
        f.write('# Key Statistics for Paper\n\n')
        for k, v in sorted(stats.items()):
            if isinstance(v, float):
                f.write(f'- **{k}**: {v:.4f}\n')
            else:
                f.write(f'- **{k}**: {v}\n')

    logger.info('Generated summary stats: %d metrics', len(stats))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--output_dir', default='paper/generated')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading results from %s...', args.results_dir)
    data = load_all_results(args.results_dir)
    logger.info('Loaded %d result categories', len(data))

    if not data:
        logger.warning('No results found. Pipeline may not have completed yet.')
        return

    generate_table2_game_performance(data, outdir)
    generate_table3_benchmarks(data, outdir)
    generate_table5_ablations(data, outdir)

    generate_figure1_training_trajectory(data, outdir)
    generate_figure2_transfer_heatmap(data, outdir)
    generate_figure3_nash_convergence(data, outdir)
    generate_figure4_benchmark_comparison(data, outdir)

    generate_summary_stats(data, outdir)

    logger.info('All paper content generated in %s', outdir)


if __name__ == '__main__':
    main()
