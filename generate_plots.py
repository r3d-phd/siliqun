#!/usr/bin/env python3
"""
SiliQun Experiment Plot Generator
Generates all plots for the experiments/ folder from existing JSON result files.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'seed42':  '#2196F3',
    'seed123': '#FF5722',
    'seed456': '#4CAF50',
    'mean':    '#9C27B0',
    'ghz':     '#2196F3',
    'w':       '#FF5722',
    'cluster': '#4CAF50',
    'dicke':   '#FF9800',
    'ppo':     '#2196F3',
    'sac':     '#FF5722',
    'grape':   '#4CAF50',
    'gaa':     '#9C27B0',
}

BASE = Path('/home/ubuntu/siliqun-main-repo')
OUT  = BASE / 'experiments' / 'plots'

def save(fig, path, name):
    p = path / name
    fig.savefig(p)
    plt.close(fig)
    print(f'  Saved: {p.relative_to(BASE)}')
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 1. PPO Reward / Fidelity Curves  (3 seeds)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ppo_curves():
    out = OUT / 'ppo_reward_curves'
    out.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    seed_data = {}
    for s in seeds:
        f = BASE / f'results/json/siliqun_r1_ppo_results__r1_ppo_seed{s}.json'
        if f.exists():
            seed_data[s] = json.loads(f.read_text())

    if not seed_data:
        print('  [SKIP] PPO seed files not found')
        return

    # ── 1a. Training curve fidelity per seed (from training_curve list) ────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle('PPO Training — Fidelity Convergence per Seed', fontweight='bold')

    for ax, s in zip(axes, seeds):
        if s not in seed_data:
            continue
        d = seed_data[s]
        curve = d.get('training_curve', [])
        if not curve:
            continue
        episodes = [pt['episode'] for pt in curve]
        means    = [pt['mean_fidelity'] for pt in curve]
        bests    = [pt['best_fidelity'] for pt in curve]
        pcts     = [pt.get('pct_above_999', 0) for pt in curve]

        ax.plot(episodes, bests,  color=COLORS[f'seed{s}'], linewidth=2, label='Best fidelity')
        ax.plot(episodes, means,  color=COLORS[f'seed{s}'], linewidth=1.5, linestyle='--', alpha=0.7, label='Mean fidelity')
        ax.fill_between(episodes, means, bests, color=COLORS[f'seed{s}'], alpha=0.12)
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99 threshold')
        ax.set_title(f'Seed {s}  (best={d["final_best_fidelity"]:.4f})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fidelity' if s == 42 else '')
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)

    save(fig, out, 'ppo_fidelity_per_seed.png')

    # ── 1b. All seeds on one plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title('PPO Training — Best Fidelity Convergence (All Seeds)', fontweight='bold')

    for s in seeds:
        if s not in seed_data:
            continue
        d = seed_data[s]
        curve = d.get('training_curve', [])
        if not curve:
            continue
        episodes = [pt['episode'] for pt in curve]
        bests    = [pt['best_fidelity'] for pt in curve]
        ax.plot(episodes, bests, color=COLORS[f'seed{s}'], label=f'Seed {s}', linewidth=2)

    ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99 threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Best Fidelity')
    ax.legend()
    save(fig, out, 'ppo_reward_all_seeds.png')

    # ── 1c. Final fidelity bar chart ────────────────────────────────────────
    summary_f = BASE / 'results/json/siliqun_r1_ppo_results__r1_ppo_results.json'
    if summary_f.exists():
        summary = json.loads(summary_f.read_text())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title('PPO Final Fidelity — Summary', fontweight='bold')
        labels = [f'Seed {s}' for s in seeds]
        vals = [seed_data[s].get('final_best_fidelity', 0) if s in seed_data else 0 for s in seeds]
        bars = ax.bar(labels, vals, color=[COLORS[f'seed{s}'] for s in seeds], width=0.5, edgecolor='white')
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
        ax.set_ylim(0.8, 1.02)
        ax.set_ylabel('Best Fidelity')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
        mean_F = summary.get('mean_F_across_seeds', np.mean(vals))
        ax.set_xlabel(f'Mean across seeds: {mean_F:.4f}')
        ax.legend()
        save(fig, out, 'ppo_final_fidelity_bar.png')

    print('  [OK] PPO curves generated')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fidelity vs Episode  (DRL training demo + r3 curves)
# ══════════════════════════════════════════════════════════════════════════════
def plot_fidelity_vs_episode():
    out = OUT / 'fidelity_vs_episode'
    out.mkdir(parents=True, exist_ok=True)

    # ── 2a. DRL training results (2Q GHZ + W) ──────────────────────────────
    f = BASE / 'validation/drl_training_results.json'
    if f.exists():
        data = json.loads(f.read_text())
        fig, axes = plt.subplots(1, len(data), figsize=(6*len(data), 4), squeeze=False)
        fig.suptitle('DRL Training — Fidelity vs Episode', fontweight='bold')
        for i, run in enumerate(data):
            ax = axes[0][i]
            ep_f = run.get('episode_fidelities', [])
            ep_r = run.get('episode_rewards', [])
            target = run.get('target_state', f'Run {i+1}')
            n_q = run.get('n_qubits', '?')
            episodes = np.arange(1, len(ep_f) + 1)
            window = max(1, len(ep_f) // 15)
            if len(ep_f) >= window:
                smoothed = np.convolve(ep_f, np.ones(window)/window, mode='valid')
                ep_s = np.arange(window, len(ep_f) + 1)
                ax.plot(ep_s, smoothed, color=COLORS.get(target.lower(), '#2196F3'),
                        label=f'Fidelity (smooth w={window})', linewidth=2)
            ax.scatter(episodes, ep_f, s=3, alpha=0.2, color='gray')
            ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
            ax.set_title(f'{n_q}Q {target}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Fidelity')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
        save(fig, out, 'drl_fidelity_vs_episode.png')

    # ── 2b. R3 multi-seed convergence curves ───────────────────────────────
    curve_files = sorted((BASE / 'results/json').glob('r3_logs__seed_*_curve.json'))
    if curve_files:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_title('R3 Multi-Seed Convergence — Mean Fidelity vs Episode', fontweight='bold')
        palette = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
        for i, cf in enumerate(curve_files):
            curve = json.loads(cf.read_text())
            seed_label = cf.stem.replace('r3_logs__seed_', 'Seed ').replace('_curve', '')
            eps   = [pt['episode'] for pt in curve]
            means = [pt['mean_fidelity'] for pt in curve]
            bests = [pt['best_fidelity'] for pt in curve]
            c = palette[i % len(palette)]
            ax.plot(eps, means, color=c, label=f'{seed_label} (mean)', linewidth=2)
            ax.plot(eps, bests, color=c, linestyle='--', alpha=0.5, label=f'{seed_label} (best)')
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
        ax.set_xlabel('Episode (×1000)')
        ax.set_ylabel('Fidelity')
        ax.legend(fontsize=8, ncol=2)
        save(fig, out, 'r3_convergence_curves.png')

    # ── 2c. E4 v5 GHZ convergence (3 seeds) ────────────────────────────────
    f = BASE / 'results/json/siliqun_e4_v5_results__siliqun_e4_v5_results.json'
    if f.exists():
        d = json.loads(f.read_text())
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_title('E4 Experiment — GHZ Fidelity Convergence (3 Seeds)', fontweight='bold')
        for key, color in [('ghz_s42', COLORS['seed42']), ('ghz_s123', COLORS['seed123']), ('ghz_s456', COLORS['seed456'])]:
            if key not in d:
                continue
            run = d[key]
            history = run.get('history', [])
            if not history:
                continue
            eps  = [h.get('episode', i) for i, h in enumerate(history)]
            fids = [h.get('best_fidelity', h.get('mean_fidelity', 0)) for h in history]
            seed_n = key.split('_s')[1]
            ax.plot(eps, fids, color=color, label=f'Seed {seed_n}', linewidth=2)
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99 threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Best Fidelity')
        ax.legend()
        save(fig, out, 'e4_ghz_convergence_3seeds.png')

    print('  [OK] Fidelity vs episode plots generated')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scalability Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_scalability():
    out = OUT / 'scalability'
    out.mkdir(parents=True, exist_ok=True)

    # ── 3a. Per-target scalability from scaling__*.json ────────────────────
    scaling_files = sorted((BASE / 'results/json').glob('scaling__*.json'))
    # Group by family
    from collections import defaultdict
    by_family = defaultdict(list)
    for sf in scaling_files:
        if 'summary' in sf.name:
            continue
        d = json.loads(sf.read_text())
        family = d.get('family', sf.stem.split('_')[2] if len(sf.stem.split('_')) > 2 else 'unknown')
        by_family[family].append(d)

    if by_family:
        fig, axes = plt.subplots(1, len(by_family), figsize=(5*len(by_family), 4), squeeze=False)
        fig.suptitle('SiliQun Scalability — Fidelity vs Qubit Count', fontweight='bold')
        family_colors = {'ghz': COLORS['ghz'], 'w': COLORS['w'],
                         'cluster_linear': COLORS['cluster'], 'dicke_k3': COLORS['dicke']}
        for ax, (family, runs) in zip(axes[0], by_family.items()):
            runs_sorted = sorted(runs, key=lambda x: x.get('n_qubits', 0))
            n_q   = [r['n_qubits'] for r in runs_sorted]
            means = [r.get('mean_F', r.get('best_F', 0)) for r in runs_sorted]
            stds  = [r.get('std_F', 0) for r in runs_sorted]
            c = family_colors.get(family, '#2196F3')
            ax.errorbar(n_q, means, yerr=stds, marker='o', color=c,
                        capsize=4, linewidth=2, markersize=6)
            ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2)
            ax.set_title(family.replace('_', ' ').title())
            ax.set_xlabel('Qubits (N)')
            ax.set_ylabel('Mean Fidelity')
            ax.set_ylim(0.5, 1.05)
            ax.set_xticks(n_q)
        save(fig, out, 'scalability_per_target.png')

    # ── 3b. Combined scalability from v11_scaling_summary.json ─────────────
    summary_f = BASE / 'results/json/v11_scaling_summary.json'
    if summary_f.exists():
        summary = json.loads(summary_f.read_text())
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title('SiliQun v11 — Scalability Summary (All Targets)', fontweight='bold')
        target_map = {'ghz': ('GHZ', COLORS['ghz']),
                      'w': ('W-State', COLORS['w']),
                      'cluster_linear': ('Cluster', COLORS['cluster']),
                      'dicke_k3': ('Dicke-k3', COLORS['dicke'])}
        for key, (label, color) in target_map.items():
            if key not in summary:
                continue
            data = summary[key]
            if isinstance(data, dict):
                # structure: {"2": {n_qubits:2, mean_F:..., std_F:...}, "3": {...}, ...}
                if all(str(k).isdigit() for k in data.keys()):
                    n_q  = sorted([int(k) for k in data.keys()])
                    fids = [float(data[str(n)]['mean_F']) if isinstance(data[str(n)], dict) else float(data[str(n)]) for n in n_q]
                    stds = [float(data[str(n)].get('std_F', 0)) if isinstance(data[str(n)], dict) else 0 for n in n_q]
                    ax.errorbar(n_q, fids, yerr=stds, marker='o', color=color, label=label,
                                linewidth=2, markersize=6, capsize=3)
            elif isinstance(data, list):
                n_q  = [d.get('n_qubits', i) for i, d in enumerate(data)]
                fids = [float(d.get('mean_F', d.get('best_F', 0))) for d in data]
                ax.plot(n_q, fids, marker='o', color=color, label=label, linewidth=2, markersize=6)
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
        ax.set_xlabel('Number of Qubits (N)')
        ax.set_ylabel('Mean Fidelity')
        ax.legend()
        save(fig, out, 'v11_scalability_summary.png')

    print('  [OK] Scalability plots generated')


# ══════════════════════════════════════════════════════════════════════════════
# 4. Algorithm Comparison / Ablation
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation():
    out = OUT / 'ablation'
    out.mkdir(parents=True, exist_ok=True)

    # ── 4a. GRAPE vs GAA vs PPO vs SAC comparison ──────────────────────────
    grape_f = BASE / 'results/json/results__grape_crab_results.json'
    gaa_f   = BASE / 'results/json/results__gaa_seeds_234_results.json'

    methods, fidelities, colors_list = [], [], []

    if grape_f.exists():
        grape = json.loads(grape_f.read_text())
        grape_fids = [r['fidelity'] for r in grape if 'fidelity' in r]
        if grape_fids:
            methods.append('GRAPE/CRAB')
            fidelities.append(np.mean(grape_fids))
            colors_list.append(COLORS['grape'])

    if gaa_f.exists():
        gaa = json.loads(gaa_f.read_text())
        gaa_fids = [r.get('final_fidelity', 0) for r in gaa]
        if gaa_fids:
            methods.append('GAA')
            fidelities.append(np.mean(gaa_fids))
            colors_list.append(COLORS['gaa'])

    # PPO from summary
    ppo_f = BASE / 'results/json/siliqun_r1_ppo_results__r1_ppo_results.json'
    if ppo_f.exists():
        ppo = json.loads(ppo_f.read_text())
        methods.append('PPO (SiliQun)')
        fidelities.append(ppo.get('mean_F_across_seeds', 0))
        colors_list.append(COLORS['ppo'])

    # SAC from r3_v5
    r3_f = BASE / 'results/json/siliqun_r3_v5_results__r3_v5_results.json'
    if r3_f.exists():
        r3 = json.loads(r3_f.read_text())
        sac_fid = r3.get('final_mean_fidelity', r3.get('best_fidelity', 0))
        if sac_fid:
            methods.append('SAC (SiliQun)')
            fidelities.append(sac_fid)
            colors_list.append(COLORS['sac'])

    if methods:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('Algorithm Comparison — Final Mean Fidelity', fontweight='bold')
        bars = ax.bar(methods, fidelities, color=colors_list, width=0.5, edgecolor='white')
        ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99 threshold')
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel('Mean Fidelity')
        for bar, v in zip(bars, fidelities):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
                    ha='center', fontsize=9, fontweight='bold')
        ax.legend()
        save(fig, out, 'algorithm_comparison.png')

    # ── 4b. Noise robustness (from r3_v5 if available) ─────────────────────
    if r3_f.exists():
        r3 = json.loads(r3_f.read_text())
        noise_curve = r3.get('noise_curve', r3.get('noise_fidelity_curve', []))
        if noise_curve and isinstance(noise_curve, list) and len(noise_curve) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title('Noise Robustness — Fidelity vs Noise Level', fontweight='bold')
            noise_levels = [pt.get('noise', pt.get('noise_level', i)) for i, pt in enumerate(noise_curve)]
            fids = [pt.get('fidelity', pt.get('mean_fidelity', 0)) for pt in noise_curve]
            ax.plot(noise_levels, fids, marker='o', color=COLORS['sac'], linewidth=2, markersize=6)
            ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Fidelity')
            ax.legend()
            save(fig, out, 'noise_robustness.png')

    # ── 4c. E4 multi-target bar chart ───────────────────────────────────────
    e4_f = BASE / 'results/json/siliqun_e4_results.json'
    if e4_f.exists():
        e4 = json.loads(e4_f.read_text())
        if isinstance(e4, list) and e4:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title('E4 Multi-Target — Mean Fidelity per Target', fontweight='bold')
            targets = [r.get('target', f'T{i}') for i, r in enumerate(e4)]
            means   = [r.get('mean_fidelity', 0) for r in e4]
            stds    = [r.get('std_fidelity', 0) for r in e4]
            target_colors = [COLORS.get(t.lower().replace('-','').replace('_',''), '#2196F3') for t in targets]
            bars = ax.bar(targets, means, yerr=stds, color=target_colors,
                          width=0.5, capsize=5, edgecolor='white')
            ax.axhline(0.99, color='red', linestyle=':', linewidth=1.2, label='F=0.99')
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel('Mean Fidelity')
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
                        ha='center', fontsize=9)
            ax.legend()
            save(fig, out, 'e4_multitarget_bar.png')

    print('  [OK] Ablation / comparison plots generated')


# ══════════════════════════════════════════════════════════════════════════════
# 5. TensorBoard-style scalar export (CSV + plot)
# ══════════════════════════════════════════════════════════════════════════════
def export_tensorboard_scalars():
    out = OUT / 'tensorboard_scalars'
    out.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    all_rows = []

    for s in seeds:
        f = BASE / f'results/json/siliqun_r1_ppo_results__r1_ppo_seed{s}.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        curve = d.get('training_curve', [])
        for pt in curve:
            all_rows.append({'seed': s, 'step': pt['episode'],
                             'fidelity': pt['mean_fidelity'],
                             'reward': pt['best_fidelity']})

    if all_rows:
        import csv
        csv_path = out / 'ppo_scalars.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['seed', 'step', 'fidelity', 'reward'])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'  Saved: {csv_path.relative_to(BASE)}')

        # Plot from CSV
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('TensorBoard Scalars — PPO Training (All Seeds)', fontweight='bold')
        for s in seeds:
            rows = [r for r in all_rows if r['seed'] == s]
            if not rows:
                continue
            steps = [r['step'] for r in rows]
            fids  = [r['fidelity'] for r in rows]
            rews  = [r['reward'] for r in rows]
            window = max(1, len(steps) // 20)
            sf = np.convolve(fids, np.ones(window)/window, mode='valid')
            sr = np.convolve(rews, np.ones(window)/window, mode='valid')
            ss = np.arange(window, len(steps)+1)
            ax1.plot(ss, sf, color=COLORS[f'seed{s}'], label=f'Seed {s}', linewidth=1.5)
            ax2.plot(ss, sr, color=COLORS[f'seed{s}'], label=f'Seed {s}', linewidth=1.5)
        ax1.axhline(0.99, color='red', linestyle=':', linewidth=1.2)
        ax1.set_title('Fidelity'); ax1.set_xlabel('Step'); ax1.set_ylabel('Fidelity'); ax1.legend()
        ax2.set_title('Reward');   ax2.set_xlabel('Step'); ax2.set_ylabel('Reward');   ax2.legend()
        save(fig, out, 'tensorboard_ppo_scalars.png')

    print('  [OK] TensorBoard scalar export done')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating SiliQun experiment plots...')
    OUT.mkdir(parents=True, exist_ok=True)

    print('\n[1/5] PPO reward curves...')
    plot_ppo_curves()

    print('\n[2/5] Fidelity vs episode...')
    plot_fidelity_vs_episode()

    print('\n[3/5] Scalability plots...')
    plot_scalability()

    print('\n[4/5] Ablation / comparison...')
    plot_ablation()

    print('\n[5/5] TensorBoard scalar export...')
    export_tensorboard_scalars()

    print(f'\nAll plots saved to: {OUT}')
    all_plots = list(OUT.rglob('*.png'))
    print(f'Total: {len(all_plots)} PNG files')
    for p in sorted(all_plots):
        print(f'  {p.relative_to(BASE)}')
