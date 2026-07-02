# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for SiliQun SoftwareX paper.
Uses real benchmark data from Aziz HPC and local comparison benchmarks.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# Figure 2: Speed Scaling (Aziz HPC data)
# ============================================================
def fig_speed_scaling():
    # Aziz HPC data: Intel Xeon E5-2695 v2 @ 2.40GHz
    qubits_aziz = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    tp_aziz = [90153, 90580, 90801, 89795, 91578, 91263, 92031, 92151, 89995, 91283]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(qubits_aziz, [t/1000 for t in tp_aziz], 'o-', color='#2196F3',
            linewidth=2, markersize=7, label='SiliQun (MPS)', zorder=3)

    # Add a horizontal reference line at the mean
    mean_tp = np.mean(tp_aziz) / 1000
    ax.axhline(y=mean_tp, color='#2196F3', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(16.3, mean_tp, f'{mean_tp:.1f}K', fontsize=8, color='#2196F3', va='center')

    # Theoretical exponential scaling for statevector
    qubits_sv = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    # Normalize to match at 2 qubits
    sv_tp = 90000 * (4.0 / 2**qubits_sv) * 2**2
    ax.plot(qubits_sv, [max(t/1000, 0.1) for t in sv_tp], 's--', color='#FF5722',
            linewidth=1.5, markersize=5, alpha=0.7, label='Statevector (theoretical $O(2^n)$)')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Throughput (K steps/s)')
    ax.set_title('SiliQun Throughput Scaling on Aziz HPC\n(Intel Xeon E5-2695 v2 @ 2.40 GHz)')
    ax.set_xticks(qubits_aziz)
    ax.set_ylim(0, 120)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)

    fig.savefig('/home/ubuntu/siliqun/paper/fig_speed_scaling.png')
    plt.close()
    print("Saved fig_speed_scaling.png")


# ============================================================
# Figure 3: Comparison with Qiskit Aer and QuTiP (local data)
# ============================================================
def fig_comparison():
    # Local sandbox data
    qubits = [2, 4, 6, 8, 10, 12, 14]
    siliqun_tp = [460839, 468882, 477126, 465071, 463393, 458520, 468677]
    qiskit_tp = [83104, 84028, 84548, 80815, 69540, 42314, 16359]
    qutip_tp = [500115, 447117, 376387, 283657, 160242, 49203, 11533]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Throughput comparison (log scale)
    ax1.semilogy(qubits, [t/1000 for t in siliqun_tp], 'o-', color='#2196F3',
                 linewidth=2, markersize=7, label='SiliQun (MPS)')
    ax1.semilogy(qubits, [t/1000 for t in qiskit_tp], 's-', color='#FF5722',
                 linewidth=2, markersize=7, label='Qiskit Aer (SV)')
    ax1.semilogy(qubits, [t/1000 for t in qutip_tp], '^-', color='#4CAF50',
                 linewidth=2, markersize=7, label='QuTiP (SV)')

    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Throughput (K gates/s, log scale)')
    ax1.set_title('Gate Throughput Comparison')
    ax1.set_xticks(qubits)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3, which='both')

    # Right: Speedup ratio
    speedup_qiskit = [s/q for s, q in zip(siliqun_tp, qiskit_tp)]
    speedup_qutip = [s/q for s, q in zip(siliqun_tp, qutip_tp)]

    x = np.arange(len(qubits))
    width = 0.35
    bars1 = ax2.bar(x - width/2, speedup_qiskit, width, label='vs Qiskit Aer',
                    color='#FF5722', alpha=0.8)
    bars2 = ax2.bar(x + width/2, speedup_qutip, width, label='vs QuTiP',
                    color='#4CAF50', alpha=0.8)

    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('SiliQun Speedup Factor')
    ax2.set_title('SiliQun Speedup Over Statevector Simulators')
    ax2.set_xticks(x)
    ax2.set_xticklabels(qubits)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 5:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.0f}x', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 5:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.0f}x', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig('/home/ubuntu/siliqun/paper/fig_comparison.png')
    plt.close()
    print("Saved fig_comparison.png")


# ============================================================
# Figure 4: Noise Overhead (Aziz HPC data)
# ============================================================
def fig_noise_overhead():
    qubits = [2, 4, 6, 8, 10, 12]
    clean_tp = [90788, 103197, 123359, 120495, 121582, 122683]
    noisy_tp = [69717, 85818, 88973, 90833, 90443, 89913]
    overhead = [30.2, 20.3, 38.6, 32.7, 34.4, 36.4]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))

    x = np.arange(len(qubits))
    width = 0.3

    bars1 = ax1.bar(x - width/2, [t/1000 for t in clean_tp], width,
                    label='Noiseless', color='#2196F3', alpha=0.85)
    bars2 = ax1.bar(x + width/2, [t/1000 for t in noisy_tp], width,
                    label='Noisy (depolarizing + dephasing)', color='#FF9800', alpha=0.85)

    ax2 = ax1.twinx()
    ax2.plot(x, overhead, 'D-', color='#F44336', linewidth=2, markersize=6,
             label='Noise overhead (%)')
    ax2.set_ylabel('Noise Overhead (%)', color='#F44336')
    ax2.tick_params(axis='y', labelcolor='#F44336')
    ax2.set_ylim(0, 60)

    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Throughput (K steps/s)')
    ax1.set_title('Noise Model Overhead on Aziz HPC')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qubits)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True, alpha=0.3, axis='y')
    fig.savefig('/home/ubuntu/siliqun/paper/fig_noise_overhead.png')
    plt.close()
    print("Saved fig_noise_overhead.png")


# ============================================================
# Figure 5: Bond Dimension Scaling (Aziz HPC data)
# ============================================================
def fig_bond_dim():
    chi_values = [2, 4, 8, 16, 32, 64]
    tp = [121924, 123329, 121836, 124272, 121506, 123787]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(chi_values, [t/1000 for t in tp], 'o-', color='#9C27B0',
            linewidth=2, markersize=8)

    ax.set_xlabel('Maximum Bond Dimension ($\\chi$)')
    ax.set_ylabel('Throughput (K steps/s)')
    ax.set_title('Throughput vs Bond Dimension (8 qubits, Aziz HPC)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(chi_values)
    ax.set_xticklabels(chi_values)
    ax.set_ylim(100, 140)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Nearly constant throughput\nacross bond dimensions',
                xy=(16, 124.3), xytext=(8, 135),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.savefig('/home/ubuntu/siliqun/paper/fig_bond_dim.png')
    plt.close()
    print("Saved fig_bond_dim.png")


if __name__ == "__main__":
    fig_speed_scaling()
    fig_comparison()
    fig_noise_overhead()
    fig_bond_dim()
    print("\nAll figures generated!")
