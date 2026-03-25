#!/usr/bin/env python3
"""Generate all plots for the Craftax research report."""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
experiments = []
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        experiments.append(row)

def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

COLORS = {
    'PPO': '#4e79a7',
    'PPO-LSTM': '#f28e2b',
    'PQN': '#76b7b2',
    'PQN-LSTM': '#e15759',
    'PPO-GTrXL': '#59a14f',
    'PPO-GRU': '#edc948',
    'Best': '#b07aa1',
}

# ============================================================
# Plot 1: Baseline Comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

baselines = {}
for e in experiments:
    if e['hypothesis_id'] == 'baseline':
        algo = e['algorithm']
        ret = safe_float(e['avg_episode_return'])
        if algo not in baselines:
            baselines[algo] = []
        baselines[algo].append(ret)

algos = ['PPO', 'PPO-LSTM', 'PQN', 'PQN-LSTM']
means = [np.mean(baselines.get(a, [0])) for a in algos]
stds = [np.std(baselines.get(a, [0])) for a in algos]
colors = [COLORS.get(a, '#999') for a in algos]

bars = ax1.bar(algos, means, yerr=stds, color=colors, capsize=5, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Average Episode Return')
ax1.set_title('Baseline Algorithms (1B Steps, 3 Seeds)')
for bar, m in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{m:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Dungeon entry rates for baselines
dungeon_rates = {}
for e in experiments:
    if e['hypothesis_id'] == 'baseline' and e.get('enter_dungeon'):
        algo = e['algorithm']
        rate = safe_float(e['enter_dungeon'])
        if algo not in dungeon_rates:
            dungeon_rates[algo] = []
        dungeon_rates[algo].append(rate)

d_algos = [a for a in algos if a in dungeon_rates]
d_means = [np.mean(dungeon_rates.get(a, [0])) for a in d_algos]
bars2 = ax2.bar(d_algos, d_means, color=[COLORS.get(a, '#999') for a in d_algos],
                edgecolor='white', linewidth=1.5)
ax2.set_ylabel('Dungeon Entry Rate (%)')
ax2.set_title('Dungeon Entry (Floor 1) — Baselines')
for bar, m in zip(bars2, d_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{m:.0f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'baselines.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 2: Performance Progression (1B results over research timeline)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Key 1B results in research order
progression = [
    ('PPO\nbaseline', 26.83, 'PPO'),
    ('PQN\nbaseline', 20.95, 'PQN'),
    ('PQN-LSTM\nbaseline', 24.23, 'PQN-LSTM'),
    ('PPO-LSTM\nbaseline', 33.88, 'PPO-LSTM'),
    ('h007\nGTrXL ref', 19.01, 'PPO-GTrXL'),
    ('h009\nGTrXL+CNN', 18.17, 'PPO-GTrXL'),
    ('h012\nGTrXL+PopArt', 19.78, 'PPO-GTrXL'),
    ('h021\nLSTM+CNN', 32.59, 'PPO-LSTM'),
    ('h023\nLSTM+128s', 38.10, 'PPO-LSTM'),
    ('h031\nGRU base', 32.90, 'PPO-GRU'),
    ('h032\nLSTM+ent', 30.03, 'PPO-LSTM'),
    ('h040\nGRU+128+gn', 39.55, 'PPO-GRU'),
    ('h043\nLSTM+combo', 36.54, 'PPO-LSTM'),
    ('h044\nGRU+ent+128', 39.03, 'PPO-GRU'),
    ('h069\ncurriculum', 37.98, 'PPO-GRU'),
    ('h085\nRND 0.01', 38.71, 'PPO-GRU'),
    ('h096\nRND 0.005', 40.50, 'PPO-GRU'),
]

x = np.arange(len(progression))
labels = [p[0] for p in progression]
values = [p[1] for p in progression]
colors = [COLORS.get(p[2], '#999') for p in progression]

bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1)
ax.axhline(y=26.83, color=COLORS['PPO'], linestyle='--', alpha=0.5, label='PPO baseline (26.83)')
ax.axhline(y=33.88, color=COLORS['PPO-LSTM'], linestyle='--', alpha=0.5, label='PPO-LSTM baseline (33.88)')

# Highlight best
best_idx = np.argmax(values)
bars[best_idx].set_edgecolor('red')
bars[best_idx].set_linewidth(3)

for i, (bar, v) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold' if i == best_idx else 'normal')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Episode Return (3 Seeds)')
ax.set_title('Performance Progression — 1B Step Experiments')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'performance_progression.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 3: Architecture Comparison (LSTM vs GRU vs GTrXL)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

arch_results = {
    'PPO\n(feedforward)': {'mean': 26.83, 'std': 1.31, 'color': COLORS['PPO']},
    'PPO-LSTM\nbaseline': {'mean': 33.88, 'std': 0.50, 'color': COLORS['PPO-LSTM']},
    'PPO-GTrXL\n(h007 ref)': {'mean': 19.01, 'std': 0.84, 'color': COLORS['PPO-GTrXL']},
    'PPO-LSTM\n+CNN+128s (h023)': {'mean': 38.10, 'std': 2.61, 'color': '#ff9e4a'},
    'PPO-GRU\n+128s+gn (h040)': {'mean': 39.55, 'std': 2.32, 'color': COLORS['PPO-GRU']},
    'PPO-LSTM\n+combo (h043)': {'mean': 36.54, 'std': 4.43, 'color': '#ffa573'},
    'PPO-GRU\n+ent+128s (h044)': {'mean': 39.03, 'std': 2.91, 'color': '#d4a017'},
    'PPO-GRU\n+RND0.005 (h096)': {'mean': 40.50, 'std': 0.48, 'color': '#c8b900'},
}

labels = list(arch_results.keys())
means = [arch_results[k]['mean'] for k in labels]
stds = [arch_results[k]['std'] for k in labels]
colors = [arch_results[k]['color'] for k in labels]

x = np.arange(len(labels))
bars = ax.barh(x, means, xerr=stds, color=colors, capsize=4, edgecolor='white', linewidth=1.5, height=0.7)

for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
    ax.text(m + s + 0.3, bar.get_y() + bar.get_height()/2, f'{m:.1f}±{s:.1f}',
            va='center', fontsize=9, fontweight='bold')

ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Mean Episode Return (1B Steps)')
ax.set_title('Architecture Comparison — 1B Step Evaluations')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'architecture_comparison.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 4: Pilot Results — Top 20 at 200M
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

pilots = []
for e in experiments:
    ts = safe_float(e.get('total_timesteps', 0))
    if 190000000 < ts < 210000000:
        ret = safe_float(e['avg_episode_return'])
        if ret > 0:
            pilots.append((e['experiment_id'], e['hypothesis_id'], ret, safe_float(e.get('enter_dungeon', 0)), e.get('algorithm', '')))

pilots.sort(key=lambda x: x[2], reverse=True)
top = pilots[:25]

labels = [f"{p[1]}" for p in top]
values = [p[2] for p in top]
dungeon = [p[3] for p in top]

x = np.arange(len(top))
bars = ax.barh(x, values, color=['#2ca02c' if d > 50 else '#ff7f0e' if d > 0 else '#d62728' for d in dungeon],
               edgecolor='white', linewidth=1, height=0.7)

for i, (bar, v, d) in enumerate(zip(bars, values, dungeon)):
    label = f'{v:.1f} ({d:.0f}% dng)' if d > 0 else f'{v:.1f}'
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, label,
            va='center', fontsize=8)

ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Average Episode Return')
ax.set_title('Top 25 Pilot Experiments (200M Steps)')
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ca02c', label='>50% dungeon entry'),
                   Patch(facecolor='#ff7f0e', label='1-50% dungeon entry'),
                   Patch(facecolor='#d62728', label='0% dungeon entry')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'pilot_results.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 5: Exploration Methods Comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# RND sweep at 200M
rnd_data = []
for e in experiments:
    hid = e['hypothesis_id']
    ts = safe_float(e.get('total_timesteps', 0))
    if ts < 210000000 and ts > 190000000:
        ret = safe_float(e['avg_episode_return'])
        cmd = e.get('command', '')
        if 'rnd-coef' in cmd and ret > 0:
            # extract coef
            parts = cmd.split()
            for i, p in enumerate(parts):
                if p == '--rnd-coef' and i+1 < len(parts):
                    coef = float(parts[i+1])
                    rnd_data.append((coef, ret, hid))

# Group by coef
from collections import defaultdict
rnd_by_coef = defaultdict(list)
for coef, ret, hid in rnd_data:
    rnd_by_coef[coef].append(ret)

if rnd_by_coef:
    coefs = sorted(rnd_by_coef.keys())
    rnd_means = [np.mean(rnd_by_coef[c]) for c in coefs]
    rnd_stds = [np.std(rnd_by_coef[c]) if len(rnd_by_coef[c]) > 1 else 0 for c in coefs]

    ax1.bar([f'{c}' for c in coefs], rnd_means, yerr=rnd_stds, color='#4e79a7', capsize=5,
            edgecolor='white', linewidth=1.5)
    ax1.axhline(y=30.86, color='red', linestyle='--', alpha=0.7, label='h040 no-RND baseline (30.86)')
    for i, (m, s) in enumerate(zip(rnd_means, rnd_stds)):
        ax1.text(i, m + s + 0.3, f'{m:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('RND Coefficient')
    ax1.set_ylabel('Avg Episode Return (200M)')
    ax1.set_title('RND Coefficient Sweep (200M Pilots)')
    ax1.legend(fontsize=8)

# Failed exploration methods at 200M
failed = [
    ('Go-Explore\n(h051)', 0),  # crashed/no-op
    ('PBRS\n(h054)', 0),  # no-op due to bug
    ('Obs Augment\n(h057)', 33.18),
    ('Residual MLP\n(h053)', 20.42),
    ('Obs Norm\n(h052)', 30.62),
    ('Aux Kill Pred\n(h059)', 22.88),
    ('SIL replay\n(h101)', 0),  # OOM
    ('NovelD\n(h110)', 29.34),
]

failed_labels = [f[0] for f in failed]
failed_vals = [f[1] for f in failed]
bars2 = ax2.barh(range(len(failed)), failed_vals, color=['#d62728' if v < 25 else '#ff7f0e' if v < 32.62 else '#2ca02c' for v in failed_vals],
                 edgecolor='white', linewidth=1, height=0.6)
ax2.axvline(x=32.62, color='green', linestyle='--', alpha=0.7, label='h044 baseline (32.62)')
for i, (bar, v) in enumerate(zip(bars2, failed_vals)):
    label = f'{v:.1f}' if v > 0 else 'Failed'
    ax2.text(max(v, 0.5) + 0.3, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9)
ax2.set_yticks(range(len(failed)))
ax2.set_yticklabels(failed_labels, fontsize=9)
ax2.set_xlabel('Avg Episode Return (200M)')
ax2.set_title('Exploration & Auxiliary Methods')
ax2.legend(fontsize=8)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'exploration_methods.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 6: Failed Approaches Summary
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

failed_approaches = [
    ('h002 - Reward shaping', 6.42, 'Reward hacking'),
    ('h003 - gamma+shaping', 9.06, 'Combined failure'),
    ('h005 - Kitchen sink', 0, 'Crashed'),
    ('h008 - GAE lambda=0.95', 14.26, 'Higher variance'),
    ('h014 - Large GTrXL 512h', 0, 'OOM'),
    ('h016 - Symlog value', 15.58, 'Underperforms PopArt'),
    ('h020 - LSTM+PopArt', 18.94, 'Over-conservative'),
    ('h036 - 8 PPO epochs', 19.14, 'Over-optimization'),
    ('h048 - Aggressive ent', 19.06, 'Instability'),
    ('h050 - 4 minibatches', 22.26, 'Less signal diversity'),
    ('h049 - Cosine LR', 24.98, 'Suboptimal schedule'),
    ('h047 - Clip=0.1', 27.30, 'Too conservative'),
    ('h127 - BatchNorm@1B', 31.06, 'Pilot gains vanish'),
]

labels = [f[0] for f in failed_approaches]
values = [f[1] for f in failed_approaches]
reasons = [f[2] for f in failed_approaches]

x = np.arange(len(failed_approaches))
colors = ['#d62728' if v < 15 else '#ff7f0e' if v < 25 else '#edc948' for v in values]
bars = ax.barh(x, values, color=colors, edgecolor='white', linewidth=1, height=0.7)

ax.axvline(x=32.62, color='#2ca02c', linestyle='--', alpha=0.7, linewidth=2, label='Best pilot baseline (h044: 32.62)')

for i, (bar, v, r) in enumerate(zip(bars, values, reasons)):
    label = f'{v:.1f} — {r}' if v > 0 else f'Failed — {r}'
    ax.text(max(v, 0) + 0.5, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=8)

ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Average Episode Return (200M steps)')
ax.set_title('Failed / Negative Approaches')
ax.legend(fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'failed_approaches.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 7: Hypothesis Outcomes — Success/Failure pie
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# From data: status distribution
hypotheses = []
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'hypotheses.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        hypotheses.append(row)

status_counts = {}
for h in hypotheses:
    s = h.get('status', 'unknown')
    status_counts[s] = status_counts.get(s, 0) + 1

# Pie chart
labels_pie = list(status_counts.keys())
values_pie = list(status_counts.values())
colors_pie = {'closed': '#d62728', 'open': '#ff7f0e', 'promising': '#2ca02c', 'proposed': '#4e79a7'}
pie_colors = [colors_pie.get(l, '#999') for l in labels_pie]

ax1.pie(values_pie, labels=[f'{l}\n({v})' for l, v in zip(labels_pie, values_pie)],
        colors=pie_colors, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10})
ax1.set_title(f'Hypothesis Status Distribution\n({len(hypotheses)} total)')

# Achievements progression: baseline vs best
achievements = ['enter_dungeon', 'find_bow', 'fire_bow', 'collect_diamond', 'make_iron_sword']
ach_labels = ['Enter\nDungeon', 'Find\nBow', 'Fire\nBow', 'Collect\nDiamond', 'Make Iron\nSword']

# PPO baseline (average)
ppo_ach = {}
for a in achievements:
    vals = []
    for e in experiments:
        if e['hypothesis_id'] == 'baseline' and e['algorithm'] == 'PPO' and e.get(a):
            vals.append(safe_float(e[a]))
    ppo_ach[a] = np.mean(vals) if vals else 0

# Best method: h040 (highest mean)
best_ach = {}
for a in achievements:
    vals = []
    for e in experiments:
        if e['hypothesis_id'] == 'h040' and e.get(a):
            vals.append(safe_float(e[a]))
    best_ach[a] = np.mean(vals) if vals else 0

x_ach = np.arange(len(achievements))
width = 0.35
ax2.bar(x_ach - width/2, [ppo_ach[a] for a in achievements], width, label='PPO Baseline', color=COLORS['PPO'])
ax2.bar(x_ach + width/2, [best_ach[a] for a in achievements], width, label='Best (h040 GRU)', color=COLORS['PPO-GRU'])
ax2.set_xticks(x_ach)
ax2.set_xticklabels(ach_labels, fontsize=9)
ax2.set_ylabel('Achievement Rate (%)')
ax2.set_title('Achievement Rates: Baseline vs Best')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hypothesis_outcomes.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 8: Top Configs at 1B (detailed)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

top_1b = [
    ('h040-1B-s2\n(GRU+128+gn)', 41.46, 96, 'PPO-GRU'),
    ('h085-1B-s1\n(GRU+RND0.01)', 41.38, 96, 'PPO-GRU'),
    ('h096-1B-s3\n(GRU+RND0.005)', 40.98, 100, 'PPO-GRU'),
    ('h023-1B-s2\n(LSTM+128s)', 40.98, 96, 'PPO-LSTM'),
    ('h044-1B-s2\n(GRU+ent+128)', 40.90, 100, 'PPO-GRU'),
    ('h043-1B-s3\n(LSTM+combo)', 40.38, 100, 'PPO-LSTM'),
    ('h040-1B-s1\n(GRU+128+gn)', 40.14, 96, 'PPO-GRU'),
    ('h096-1B-s2\n(GRU+RND0.005)', 40.02, 96, 'PPO-GRU'),
    ('h070-1B-s3\n(curriculum)', 39.38, 88, 'PPO-GRU'),
    ('h069-1B-s2\n(curriculum)', 38.46, 92, 'PPO-GRU'),
]

labels = [t[0] for t in top_1b]
returns = [t[1] for t in top_1b]
dungeon_rates = [t[2] for t in top_1b]
cats = [t[3] for t in top_1b]

x = np.arange(len(top_1b))
bar_colors = [COLORS.get(c, '#999') for c in cats]
bars = ax.barh(x, returns, color=bar_colors, edgecolor='white', linewidth=1, height=0.7)

for i, (bar, r, d) in enumerate(zip(bars, returns, dungeon_rates)):
    ax.text(r + 0.1, bar.get_y() + bar.get_height()/2, f'{r:.2f}  ({d}% dng)',
            va='center', fontsize=9, fontweight='bold' if i == 0 else 'normal')

ax.axvline(x=33.88, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='PPO-LSTM baseline (33.88)')
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Average Episode Return')
ax.set_title('Top 10 Individual Seed Results at 1B Steps')
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.set_xlim(32, 43)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'top_configs_1b.png'), bbox_inches='tight')
plt.close()

# ============================================================
# Plot 9: RND coefficient sweep at 1B
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

rnd_1b = {
    'No RND\n(h040)': {'mean': 39.55, 'std': 2.32},
    'RND 0.003\n(h106)': {'mean': 25.54, 'std': 0},  # single pilot extrapolation
    'RND 0.005\n(h096)': {'mean': 40.50, 'std': 0.48},
    'RND 0.01\n(h085)': {'mean': 38.71, 'std': 2.31},
    'RND 0.015\n(h129)': {'mean': 35.46, 'std': 0},  # partial
}

rnd_labels = list(rnd_1b.keys())
rnd_means = [rnd_1b[k]['mean'] for k in rnd_labels]
rnd_stds = [rnd_1b[k]['std'] for k in rnd_labels]
rnd_colors = ['#4e79a7' if 'No' in k else '#2ca02c' if rnd_1b[k]['mean'] > 39.55 else '#ff7f0e' for k in rnd_labels]

bars = ax.bar(rnd_labels, rnd_means, yerr=rnd_stds, color=rnd_colors, capsize=5,
              edgecolor='white', linewidth=1.5)
ax.axhline(y=39.55, color='red', linestyle='--', alpha=0.5, label='h040 no-RND (39.55)')

for bar, m in zip(bars, rnd_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{m:.1f}',
            ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Mean Episode Return (1B Steps)')
ax.set_title('RND Coefficient Comparison at 1B Steps')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rnd_sweep.png'), bbox_inches='tight')
plt.close()

print("All plots generated successfully!")
print(f"Plots saved to: {PLOTS_DIR}")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  - {f}")
