"""
Generate publication-quality benchmark plots for MPS vs MPO comparison.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load results
with open(os.path.join(RESULTS_DIR, "mps_vs_mpo_benchmark.json")) as f:
    data = json.load(f)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# ── Figure 1: Performance comparison ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 1a: Wall-clock time
perf = data["performance"]
n_qubits = [r["n_qubits"] for r in perf]
mps_times = [r["mps_time_s"] * 1000 for r in perf]  # ms
mpo_times = [r["mpo_time_s"] * 1000 for r in perf]  # ms

x = np.arange(len(n_qubits))
width = 0.35
axes[0].bar(x - width/2, mps_times, width, label="MPS", color="#2196F3", alpha=0.85)
axes[0].bar(x + width/2, mpo_times, width, label="MPO", color="#FF5722", alpha=0.85)
axes[0].set_xlabel("Number of Qubits")
axes[0].set_ylabel("Time (ms)")
axes[0].set_title("(a) Gate Application Time")
axes[0].set_xticks(x)
axes[0].set_xticklabels(n_qubits)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# 1b: Slowdown factor
slowdowns = [r["slowdown_factor"] for r in perf]
axes[1].plot(n_qubits, slowdowns, "o-", color="#9C27B0", linewidth=2, markersize=8)
axes[1].set_xlabel("Number of Qubits")
axes[1].set_ylabel("Slowdown Factor (MPO/MPS)")
axes[1].set_title("(b) MPO Overhead")
axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_performance.png"), bbox_inches="tight")
plt.close()
print("Saved: fig_performance.png")


# ── Figure 2: Bond dimension growth ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

bd = data["bond_dim_growth"]
steps = list(range(1, len(bd["mps_bond_dims"]) + 1))
ax.plot(steps, bd["mps_bond_dims"], "o-", label="MPS", color="#2196F3",
        linewidth=2, markersize=4)
ax.plot(steps, bd["mpo_bond_dims"], "s-", label="MPO", color="#FF5722",
        linewidth=2, markersize=4)
ax.set_xlabel("Circuit Depth (gates)")
ax.set_ylabel("Maximum Bond Dimension")
ax.set_title(f"Bond Dimension Growth ({bd['n_qubits']} qubits, noisy)")
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_bond_dims.png"), bbox_inches="tight")
plt.close()
print("Saved: fig_bond_dims.png")


# ── Figure 3: Purity decay ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

pd = data["purity_decay"]
steps = list(range(len(pd["purities"])))

axes[0].plot(steps, pd["purities"], "-", color="#4CAF50", linewidth=2)
axes[0].set_xlabel("Circuit Depth (gates)")
axes[0].set_ylabel("Purity Tr(rho^2)")
axes[0].set_title("(a) Purity Decay Under Noise")
axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Maximally mixed (d=2)")
axes[0].axhline(y=0.25, color="orange", linestyle="--", alpha=0.5, label="Maximally mixed (d=4)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.05)

axes[1].plot(steps, [abs(t - 1.0) for t in pd["traces"]], "-", color="#E91E63", linewidth=2)
axes[1].set_xlabel("Circuit Depth (gates)")
axes[1].set_ylabel("|Tr(rho) - 1|")
axes[1].set_title("(b) Trace Preservation Error")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_purity_trace.png"), bbox_inches="tight")
plt.close()
print("Saved: fig_purity_trace.png")


# ── Figure 4: Accuracy comparison ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

acc = data["accuracy"]
n_q = [r["n_qubits"] for r in acc]
z_err = [r["z_error_mean"] for r in acc]
zz_err = [r["zz_error_mean"] for r in acc]
fid_err = [r["fid_error_mean"] for r in acc]

x = np.arange(len(n_q))
width = 0.25
ax.bar(x - width, z_err, width, label="<Z> error", color="#2196F3", alpha=0.85)
ax.bar(x, zz_err, width, label="<ZZ> error", color="#FF5722", alpha=0.85)
ax.bar(x + width, fid_err, width, label="Fidelity error", color="#4CAF50", alpha=0.85)
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("MPS vs MPO Observable Agreement (Noiseless)")
ax.set_xticks(x)
ax.set_xticklabels(n_q)
ax.legend()
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_accuracy.png"), bbox_inches="tight")
plt.close()
print("Saved: fig_accuracy.png")


# ── Figure 5: Noise comparison ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

nc = data["noise_comparison"]
qubits = list(range(nc["n_qubits"]))
mpo_z = nc["mpo_z"]
mps_z_mean = nc["mps_z_mean"]
mps_z_std = nc["mps_z_std"]

x = np.arange(len(qubits))
width = 0.35
ax.bar(x - width/2, mpo_z, width, label="MPO (exact)", color="#FF5722", alpha=0.85)
ax.bar(x + width/2, mps_z_mean, width, label=f"MPS (mean, n={nc['n_trials']})",
       color="#2196F3", alpha=0.85, yerr=mps_z_std, capsize=5)
ax.set_xlabel("Qubit Index")
ax.set_ylabel("<Z>")
ax.set_title("Noisy <Z> Comparison: Exact MPO vs Stochastic MPS")
ax.set_xticks(x)
ax.set_xticklabels(qubits)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_noise_comparison.png"), bbox_inches="tight")
plt.close()
print("Saved: fig_noise_comparison.png")

print("\nAll benchmark figures generated successfully.")
