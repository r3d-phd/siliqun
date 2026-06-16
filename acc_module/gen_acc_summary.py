"""Generate ACC summary figure: 3-panel comparison of the four decision scenarios."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from acc import AdaptiveConvergenceController, StopReason, _sat_exp

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Adaptive Convergence Controller (ACC) — Decision Scenarios',
             fontsize=13, fontweight='bold', y=1.02)

COLORS = {
    'observed':   '#2196F3',
    'fit':        '#FF9800',
    'threshold':  '#9E9E9E',
    'T_star':     '#F44336',
    'T_safe':     '#B71C1C',
    'F_inf':      '#FF9800',
    'stop':       '#4CAF50',
    'old_budget': '#CFD8DC',
}

steps_dense = np.linspace(0, 500_000, 1000)

# ── Panel 1: Threshold met (GHZ-like, fast convergence) ──────────────────────
ax = axes[0]
F_inf, tau = 0.9986, 5_000
F_obs = _sat_exp(steps_dense, F_inf, tau) + np.random.default_rng(1).normal(0, 0.001, 1000)
F_obs = np.clip(F_obs, 0, 1)
checkpoints = np.arange(10_000, 60_001, 10_000)
F_ck = np.array([_sat_exp(s, F_inf, tau) for s in checkpoints])

ax.fill_betweenx([0, 1.02], 0, 500_000/1000, alpha=0.07, color=COLORS['old_budget'],
                 label='Old: 500k budget')
ax.plot(steps_dense/1000, F_obs, color=COLORS['observed'], linewidth=1.5,
        alpha=0.6, label='Training curve')
ax.scatter(checkpoints/1000, F_ck, color=COLORS['observed'], s=40, zorder=5)
ax.axhline(0.99, color=COLORS['threshold'], linestyle='--', linewidth=1.2,
           label='F=0.99 threshold')

# ACC stops at first checkpoint where F≥0.99
stop_step = checkpoints[F_ck >= 0.99][0]
ax.axvline(stop_step/1000, color=COLORS['stop'], linewidth=2.5, alpha=0.9,
           label=f'ACC stop: step {stop_step//1000}k')
ax.annotate(f'STOP\n{stop_step//1000}k steps\n(saved {int((500_000-stop_step)/500_000*100)}%)',
            xy=(stop_step/1000, 0.995), xytext=(stop_step/1000+30, 0.94),
            fontsize=8, color=COLORS['stop'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['stop'], lw=1.5))

ax.set_xlim(0, 120); ax.set_ylim(0.80, 1.02)
ax.set_xlabel('Training Steps (×10³)', fontsize=10)
ax.set_ylabel('Best Fidelity', fontsize=10)
ax.set_title('Scenario 1: Threshold Met\n(GHZ-like, fast convergence)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(True, alpha=0.3)
ax.text(0.03, 0.12, 'StopReason:\nTHRESHOLD_MET', transform=ax.transAxes,
        fontsize=8, color=COLORS['stop'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['stop']))

# ── Panel 2: Unreachable (F∞ < 0.99) ─────────────────────────────────────────
ax = axes[1]
F_inf2, tau2 = 0.9500, 15_000
F_obs2 = _sat_exp(steps_dense, F_inf2, tau2) + np.random.default_rng(2).normal(0, 0.002, 1000)
F_obs2 = np.clip(F_obs2, 0, 1)
checkpoints2 = np.arange(10_000, 500_001, 10_000)
F_ck2 = np.array([_sat_exp(s, F_inf2, tau2) for s in checkpoints2])

ax.fill_betweenx([0, 1.02], 0, 500, alpha=0.07, color=COLORS['old_budget'],
                 label='Old: 500k budget (wasted)')
ax.plot(steps_dense/1000, F_obs2, color=COLORS['observed'], linewidth=1.5,
        alpha=0.6, label='Training curve')
ax.axhline(0.99, color=COLORS['threshold'], linestyle='--', linewidth=1.2,
           label='F=0.99 threshold')
ax.axhline(F_inf2, color=COLORS['F_inf'], linestyle=':', linewidth=1.5,
           label=f'F∞={F_inf2} (unreachable)')

# Fit line
ax.plot(steps_dense/1000, _sat_exp(steps_dense, F_inf2, tau2), '--',
        color=COLORS['fit'], linewidth=2, alpha=0.8, label='Saturating fit')

# ACC stops when slope flat AND F∞ < 0.99
acc_stop = 120_000
ax.axvline(acc_stop/1000, color=COLORS['stop'], linewidth=2.5, alpha=0.9,
           label=f'ACC stop: step {acc_stop//1000}k')
ax.annotate(f'STOP\n{acc_stop//1000}k steps\n(saved {int((500_000-acc_stop)/500_000*100)}%)',
            xy=(acc_stop/1000, 0.945), xytext=(acc_stop/1000+40, 0.88),
            fontsize=8, color=COLORS['stop'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['stop'], lw=1.5))

ax.set_xlim(0, 500); ax.set_ylim(0.70, 1.02)
ax.set_xlabel('Training Steps (×10³)', fontsize=10)
ax.set_title('Scenario 2: Unreachable Target\n(F∞ < 0.99)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(True, alpha=0.3)
ax.text(0.03, 0.12, 'StopReason:\nFLAT_UNREACHABLE', transform=ax.transAxes,
        fontsize=8, color=COLORS['stop'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['stop']))

# ── Panel 3: Converged past T* ────────────────────────────────────────────────
ax = axes[2]
F_inf3, tau3 = 0.9997, 8_000
F_obs3 = _sat_exp(steps_dense, F_inf3, tau3) + np.random.default_rng(3).normal(0, 0.0005, 1000)
F_obs3 = np.clip(F_obs3, 0, 1)

T_star = -tau3 * np.log(1 - 0.99 / F_inf3)
T_safe = T_star * 1.20

ax.fill_betweenx([0, 1.02], 0, 500, alpha=0.07, color=COLORS['old_budget'],
                 label='Old: 500k budget')
ax.plot(steps_dense/1000, F_obs3, color=COLORS['observed'], linewidth=1.5,
        alpha=0.6, label='Training curve')
ax.plot(steps_dense/1000, _sat_exp(steps_dense, F_inf3, tau3), '--',
        color=COLORS['fit'], linewidth=2, alpha=0.8,
        label=f'Fit: F∞={F_inf3:.4f}, τ={tau3//1000}k')
ax.axhline(0.99, color=COLORS['threshold'], linestyle='--', linewidth=1.2,
           label='F=0.99 threshold')
ax.axhline(F_inf3, color=COLORS['F_inf'], linestyle=':', linewidth=1.2, alpha=0.6,
           label=f'F∞={F_inf3}')
ax.axvline(T_star/1000, color=COLORS['T_star'], linestyle=':', linewidth=1.5,
           label=f'T*={T_star/1000:.0f}k')
ax.axvline(T_safe/1000, color=COLORS['stop'], linewidth=2.5, alpha=0.9,
           label=f'ACC stop: T*+20%={T_safe/1000:.0f}k')
ax.annotate(f'STOP\n{T_safe/1000:.0f}k steps\n(saved {int((500_000-T_safe)/500_000*100)}%)',
            xy=(T_safe/1000, 0.9960), xytext=(T_safe/1000+30, 0.975),
            fontsize=8, color=COLORS['stop'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['stop'], lw=1.5))

ax.set_xlim(0, 120); ax.set_ylim(0.90, 1.005)
ax.set_xlabel('Training Steps (×10³)', fontsize=10)
ax.set_title('Scenario 3: Converged (past T*)\n(Diminishing returns)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(True, alpha=0.3)
ax.text(0.03, 0.12, 'StopReason:\nCONVERGED', transform=ax.transAxes,
        fontsize=8, color=COLORS['stop'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['stop']))

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
out = 'figures/ACC3_decision_scenarios.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
