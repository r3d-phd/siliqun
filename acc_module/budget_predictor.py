"""
QUASAR Budget Predictor v2
==========================
Extracts fidelity trajectories from QUASAR training logs, fits a saturating
exponential model F(t) = F_inf * (1 - exp(-t / tau)), and predicts the optimal
step budget required to reach F = 0.99 for each (target, n_qubits, alpha).

Key improvements over v1:
- Tracks separate "runs" (each DEHB trial + full-budget run resets step counter)
- Uses only the RISING portion of each run for fitting (stops at plateau)
- Extracts alpha per run for per-hyperparameter analysis
- Handles multiple seeds and multiple targets
"""

import re, os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

LOG_FILE    = "/home/ubuntu/budget_tool/all_logs.txt"
OUTPUT_DIR  = "/home/ubuntu/budget_tool/figures"
F_THRESHOLD = 0.99
SAFETY      = 1.20
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Parse: group by (target, nQ, seed, alpha) runs
# Each "run" is a contiguous block of step=10k,20k,... entries for same alpha
# ─────────────────────────────────────────────────────────────────────────────
STEP_RE = re.compile(
    r'(\d+)Q/([^/\s]+)/s(\d+)\s+step=([\d,]+)\s+best_F=([\d.]+)\s+plateau=([\d,]+)\s+alpha=([\d.]+)'
)
DONE_RE = re.compile(r'✓\s+(\d+)Q/([^/\s]+)/s(\d+)\s+→\s+best_F=([\d.]+)')
FULL_RE = re.compile(r'Best HP.*?running full budget')

# Group records into runs: same (tgt, nq, seed, alpha) and monotonically increasing steps
all_runs = []   # list of dicts: {tgt, nq, seed, alpha, steps[], fids[], is_full_budget}

current_run = None
prev_step   = -1

with open(LOG_FILE, 'r') as f:
    lines = f.readlines()

is_full_budget_next = False
for line in lines:
    if FULL_RE.search(line):
        is_full_budget_next = True

    m = STEP_RE.search(line)
    if m:
        nq, tgt, seed, step_str, f_str, plat_str, alpha_str = m.groups()
        nq    = int(nq)
        seed  = int(seed)
        step  = int(step_str.replace(',', ''))
        F     = float(f_str)
        alpha = float(alpha_str)
        tgt   = tgt.upper()

        # New run if: different alpha, different target/nq/seed, or step reset
        if (current_run is None or
            current_run['alpha'] != alpha or
            current_run['tgt']   != tgt   or
            current_run['nq']    != nq    or
            current_run['seed']  != seed  or
            step <= prev_step):

            if current_run and len(current_run['steps']) >= 3:
                all_runs.append(current_run)
            current_run = {
                'tgt': tgt, 'nq': nq, 'seed': seed, 'alpha': alpha,
                'steps': [], 'fids': [],
                'is_full_budget': is_full_budget_next
            }
            is_full_budget_next = False

        current_run['steps'].append(step)
        current_run['fids'].append(F)
        prev_step = step

if current_run and len(current_run['steps']) >= 3:
    all_runs.append(current_run)

print(f"Parsed {len(all_runs)} runs total")
for r in all_runs:
    print(f"  {r['tgt']:16s} {r['nq']}Q s{r['seed']} α={r['alpha']:.4f}  "
          f"{len(r['steps']):3d} pts  maxF={max(r['fids']):.4f}  "
          f"{'[FULL]' if r['is_full_budget'] else '[DEHB]'}")

# ─────────────────────────────────────────────────────────────────────────────
# Fit model to rising portion only
# ─────────────────────────────────────────────────────────────────────────────
def sat_exp(t, F_inf, tau):
    return F_inf * (1.0 - np.exp(-t / tau))

def fit_run(steps, fids):
    """Fit sat_exp to the rising portion (up to first plateau)."""
    steps = np.array(steps, dtype=float)
    fids  = np.array(fids,  dtype=float)

    # Find rising portion: stop when F stops increasing for 2 consecutive points
    rising_end = len(fids)
    for i in range(2, len(fids)):
        if fids[i] <= fids[i-1] and fids[i-1] <= fids[i-2]:
            rising_end = i
            break

    s_rise = steps[:rising_end]
    f_rise = fids[:rising_end]

    if len(s_rise) < 3:
        return None

    F_max = f_rise[-1]
    F0    = min(F_max + 0.01, 0.9999)
    tau0  = s_rise[len(s_rise)//2]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(sat_exp, s_rise, f_rise,
                                p0=[F0, tau0],
                                bounds=([F_max * 0.99, 1.0], [1.0, s_rise[-1] * 20]),
                                maxfev=20000)
        F_inf, tau = popt
        f_pred = sat_exp(s_rise, *popt)
        ss_res = np.sum((f_rise - f_pred)**2)
        ss_tot = np.sum((f_rise - f_rise.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        return {'F_inf': F_inf, 'tau': tau, 'r2': r2,
                'rising_end': rising_end, 'n_rise': len(s_rise)}
    except Exception:
        return None

for run in all_runs:
    fit = fit_run(run['steps'], run['fids'])
    run['fit'] = fit
    if fit:
        F_inf = fit['F_inf']
        tau   = fit['tau']
        if F_inf > F_THRESHOLD:
            T_star = -tau * np.log(1.0 - F_THRESHOLD / F_inf)
            run['T_star']      = T_star
            run['T_star_safe'] = T_star * SAFETY
            run['reachable']   = True
        else:
            run['T_star']      = None
            run['T_star_safe'] = None
            run['reachable']   = False
    else:
        run['T_star'] = run['T_star_safe'] = None
        run['reachable'] = False

# ─────────────────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*110)
print(f"{'Target':16s} {'nQ':>3} {'Seed':>4} {'Alpha':>7} {'Type':>5} "
      f"{'F_inf':>7} {'tau':>8} {'T*':>10} {'T*+20%':>10} {'R²':>6} {'MaxF':>7}")
print("="*110)

full_budget_runs = [r for r in all_runs if r['is_full_budget']]
dehb_runs        = [r for r in all_runs if not r['is_full_budget']]

for run in sorted(all_runs, key=lambda r: (r['tgt'], r['nq'], r['seed'], r['alpha'])):
    fit = run.get('fit')
    typ = 'FULL' if run['is_full_budget'] else 'DEHB'
    if not fit:
        print(f"{run['tgt']:16s} {run['nq']:>3} {run['seed']:>4} {run['alpha']:>7.4f} "
              f"{typ:>5}  [no fit]")
        continue
    T_str  = f"{run['T_star']:>10,.0f}" if run['T_star'] else f"{'∞':>10}"
    Ts_str = f"{run['T_star_safe']:>10,.0f}" if run['T_star_safe'] else f"{'∞':>10}"
    print(f"{run['tgt']:16s} {run['nq']:>3} {run['seed']:>4} {run['alpha']:>7.4f} "
          f"{typ:>5} {fit['F_inf']:>7.4f} {fit['tau']:>8.0f} {T_str} {Ts_str} "
          f"{fit['r2']:>6.3f} {max(run['fids']):>7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Best-alpha analysis: for each (tgt, nq, seed) pick the run with highest F_inf
# ─────────────────────────────────────────────────────────────────────────────
best_per_cell = {}
for run in all_runs:
    key = (run['tgt'], run['nq'], run['seed'])
    fit = run.get('fit')
    if not fit:
        continue
    if key not in best_per_cell or fit['F_inf'] > best_per_cell[key]['fit']['F_inf']:
        best_per_cell[key] = run

print("\n" + "="*80)
print("BEST-ALPHA BUDGET RECOMMENDATIONS PER CELL")
print("="*80)
print(f"{'Target':16s} {'nQ':>3} {'Seed':>4} {'Best α':>8} {'F_inf':>7} "
      f"{'T*':>10} {'T*+20%':>10} {'R²':>6}")
print("-"*80)

agg = defaultdict(list)
for key, run in sorted(best_per_cell.items()):
    tgt, nq, seed = key
    fit = run['fit']
    T_str  = f"{run['T_star']:>10,.0f}" if run['T_star'] else f"{'∞':>10}"
    Ts_str = f"{run['T_star_safe']:>10,.0f}" if run['T_star_safe'] else f"{'∞':>10}"
    print(f"{tgt:16s} {nq:>3} {seed:>4} {run['alpha']:>8.4f} "
          f"{fit['F_inf']:>7.4f} {T_str} {Ts_str} {fit['r2']:>6.3f}")
    if run['reachable']:
        agg[(tgt, nq)].append(run['T_star'])

print("\n" + "="*70)
print("AGGREGATE RECOMMENDED BUDGETS (mean T* +20% across seeds)")
print("="*70)
current_budget = {2: 500_000, 3: 750_000, 4: 1_125_000, 5: 1_687_500,
                  6: 2_531_250, 7: 3_796_875, 8: 5_695_312}
recs = {}
for (tgt, nq) in sorted(agg.keys()):
    vals = agg[(tgt, nq)]
    mean_T = np.mean(vals)
    rec    = int(mean_T * SAFETY)
    cur    = current_budget.get(nq, int(mean_T * 2))
    recs[(tgt, nq)] = rec
    ratio = rec / cur
    print(f"  {tgt:16s} {nq}Q  mean_T*={mean_T:>10,.0f}  rec={rec:>10,}  "
          f"current={cur:>10,}  {'SAVE' if ratio < 1 else 'NEED MORE':>9} "
          f"({abs(ratio-1)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'GHZ': '#2196F3', 'W': '#4CAF50',
          'CLUSTER_LINEAR': '#FF9800', 'DICKE-K3': '#9C27B0'}
LABELS = {'GHZ': 'GHZ', 'W': 'W', 'CLUSTER_LINEAR': 'Cluster', 'DICKE-K3': 'Dicke-k3'}

# ── Fig 1: Best-alpha full-budget runs with fit ──
full_best = {k: v for k, v in best_per_cell.items() if v['is_full_budget']}
if not full_best:
    full_best = best_per_cell  # fall back to all runs

n_panels = len(full_best)
if n_panels > 0:
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (key, run) in zip(axes, sorted(full_best.items())):
        tgt, nq, seed = key
        color = COLORS.get(tgt, '#607D8B')
        steps = np.array(run['steps'], dtype=float)
        fids  = np.array(run['fids'],  dtype=float)
        fit   = run.get('fit')

        ax.plot(steps / 1000, fids, 'o', markersize=4, color=color,
                alpha=0.7, label=f'Observed (seed {seed})')

        if fit and fit['r2'] > 0.3:
            t_fit = np.linspace(0, steps.max() * 1.2, 400)
            f_fit = sat_exp(t_fit, fit['F_inf'], fit['tau'])
            ax.plot(t_fit / 1000, f_fit, '--', color=color, linewidth=2,
                    label=f'Fit: F∞={fit["F_inf"]:.4f}, τ={fit["tau"]/1000:.1f}k')
            if run['T_star']:
                ax.axvline(run['T_star'] / 1000, color='red', linestyle=':',
                           linewidth=1.5, alpha=0.8,
                           label=f'T*={run["T_star"]/1000:.0f}k steps')
                ax.axvline(run['T_star_safe'] / 1000, color='darkred',
                           linestyle=':', linewidth=1.5, alpha=0.6,
                           label=f'T*+20%={run["T_star_safe"]/1000:.0f}k')

        ax.axhline(F_THRESHOLD, color='gray', linestyle='--', linewidth=1,
                   alpha=0.6, label='F=0.99 target')
        ax.set_xlabel('Training Steps (×10³)', fontsize=10)
        ax.set_ylabel('Best Fidelity', fontsize=10)
        ax.set_title(f'{LABELS.get(tgt, tgt)} — {nq}Q  (α={run["alpha"]:.4f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower right')
        ax.set_ylim(max(0.4, fids.min() - 0.05), 1.02)
        ax.grid(True, alpha=0.3)

    plt.suptitle('QUASAR Learning Curves: Observed vs Saturating Exponential Fit',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/LC1_learning_curves_fits.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

# ── Fig 2: DEHB α search — F_inf vs alpha ──
dehb_by_cell = defaultdict(list)
for run in dehb_runs:
    if run.get('fit') and run['fit']['r2'] > 0.3:
        dehb_by_cell[(run['tgt'], run['nq'], run['seed'])].append(
            (run['alpha'], run['fit']['F_inf'], run['T_star']))

if dehb_by_cell:
    n = len(dehb_by_cell)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (key, pts) in zip(axes, sorted(dehb_by_cell.items())):
        tgt, nq, seed = key
        color = COLORS.get(tgt, '#607D8B')
        pts_sorted = sorted(pts, key=lambda x: x[0])
        alphas = [p[0] for p in pts_sorted]
        finfs  = [p[1] for p in pts_sorted]
        ax.scatter(alphas, finfs, color=color, s=60, zorder=5)
        best_idx = np.argmax(finfs)
        ax.scatter([alphas[best_idx]], [finfs[best_idx]], color='red', s=120,
                   zorder=6, marker='*', label=f'Best α={alphas[best_idx]:.4f}')
        ax.axhline(F_THRESHOLD, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Entropy Coefficient α', fontsize=10)
        ax.set_ylabel('Asymptotic Fidelity F∞', fontsize=10)
        ax.set_title(f'DEHB α Search: {LABELS.get(tgt, tgt)} {nq}Q seed={seed}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/LC2_dehb_alpha_Finf.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ── Fig 3: Budget comparison bar chart ──
if recs:
    keys_sorted = sorted(recs.keys())
    labels = [f"{LABELS.get(t,t)}\n{n}Q" for t, n in keys_sorted]
    t_stars = [recs[k] / 1000 for k in keys_sorted]
    cur_buds = [current_budget.get(k[1], recs[k]) / 1000 for k in keys_sorted]
    colors_bar = [COLORS.get(k[0], '#607D8B') for k in keys_sorted]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, cur_buds, w, label='Current fixed budget',
           color='#CFD8DC', edgecolor='#607D8B', linewidth=0.8)
    bars = ax.bar(x + w/2, t_stars, w, label='Predicted T* (+20% margin)',
                  color=colors_bar, edgecolor='#333', linewidth=0.8, alpha=0.85)
    for bar, val in zip(bars, t_stars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}k', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Step Budget (×10³)', fontsize=11)
    ax.set_title('Predicted Optimal Budget vs Current Fixed Budget\n'
                 '(Model: F(t) = F∞·(1−e^{−t/τ}), threshold F=0.99)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/LC3_budget_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ── Save JSON ──
out = {'runs': [], 'recommendations': {}}
for run in all_runs:
    entry = {k: run[k] for k in ('tgt','nq','seed','alpha','is_full_budget','reachable')}
    entry['max_F']     = round(float(max(run['fids'])), 5)
    entry['max_step']  = int(max(run['steps']))
    entry['n_points']  = len(run['steps'])
    if run.get('fit'):
        entry['F_inf']  = round(run['fit']['F_inf'], 5)
        entry['tau']    = round(run['fit']['tau'], 1)
        entry['r2']     = round(run['fit']['r2'], 4)
    if run.get('T_star'):
        entry['T_star']      = round(run['T_star'])
        entry['T_star_safe'] = round(run['T_star_safe'])
    out['runs'].append(entry)

for (tgt, nq), rec in recs.items():
    out['recommendations'][f"{tgt}_{nq}Q"] = {
        'recommended_budget': rec,
        'current_budget': current_budget.get(nq),
        'savings_pct': round((1 - rec / current_budget.get(nq, rec)) * 100, 1)
    }

with open("/home/ubuntu/budget_tool/budget_predictions.json", 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved: budget_predictions.json")
print("Done.")
