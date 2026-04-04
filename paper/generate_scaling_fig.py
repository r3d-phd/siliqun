"""Generate SiliQun scaling analysis figure for the paper."""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Memory requirements - DFS projected vs full physical space
n_logical = np.arange(5, 46)
n_physical = 3 * n_logical
proj_mem_gb = (2.0**n_logical * 16) / (1024**3)
full_mem_gb = (2.0**(3*n_logical) * 16) / (1024**3)

ax1.semilogy(n_logical, proj_mem_gb, 'b-o', markersize=4, linewidth=2, label='DFS projected ($2^n$)', zorder=5)
ax1.semilogy(n_logical, full_mem_gb, 'r-s', markersize=4, linewidth=2, label='Full physical ($2^{3n}$)', zorder=5)

# GPU memory lines
gpu_configs = [
    (24, 'RTX 3090 (24 GB)', 'green', '--'),
    (80, 'A100/H100 (80 GB)', 'orange', '--'),
    (192, 'B200 (192 GB)', 'purple', '--'),
    (640, '8x H100 DGX (640 GB)', 'brown', ':'),
    (80*1024, '1024x H100 (80 TB)', 'gray', ':'),
]

for mem, label, color, ls in gpu_configs:
    ax1.axhline(y=mem, color=color, linestyle=ls, alpha=0.7, linewidth=1)
    # Find intersection with projected line
    max_n = int(np.floor(np.log2(mem * (1024**3) / 16)))
    if max_n <= 45:
        ax1.plot(max_n, mem, 'v', color=color, markersize=10, zorder=10)
        ax1.annotate(f'{label}\n({max_n}q)', xy=(max_n, mem), 
                    xytext=(max_n+1.5, mem*3), fontsize=7,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                    color=color, fontweight='bold')

# Osaka/Fixstars reference point
osaka_qubits = 41
osaka_mem = (2**41 * 16) / (1024**3)  # ~32 TB
ax1.plot(osaka_qubits, osaka_mem, '*', color='red', markersize=15, zorder=10)
ax1.annotate('Osaka/Fixstars\n(41q full, 1024 GPUs)', 
            xy=(osaka_qubits, osaka_mem), xytext=(35, osaka_mem*50),
            fontsize=7, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

ax1.set_xlabel('Number of Logical Qubits ($n$)', fontsize=11)
ax1.set_ylabel('Memory Requirement (GB)', fontsize=11)
ax1.set_title('(a) Memory Scaling: DFS Projected vs Full Space', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(5, 45)
ax1.set_ylim(1e-6, 1e30)
ax1.grid(True, alpha=0.3)

# Panel (b): DRL training time feasibility
n_qubits_time = np.arange(20, 41)
base_time_ms = 430  # ms per episode at 25 qubits
episode_times_s = (base_time_ms / 1000) * 2.0**(n_qubits_time - 25)
training_1M_days = (episode_times_s * 1e6) / (3600 * 24)

ax2.semilogy(n_qubits_time, training_1M_days, 'b-o', markersize=5, linewidth=2, zorder=5)

# Feasibility zones
ax2.axhspan(0, 7, alpha=0.15, color='green', zorder=0)
ax2.axhspan(7, 30, alpha=0.15, color='yellow', zorder=0)
ax2.axhspan(30, 1e8, alpha=0.10, color='red', zorder=0)

ax2.axhline(y=7, color='green', linestyle='--', alpha=0.7, linewidth=1)
ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax2.axhline(y=365, color='red', linestyle='--', alpha=0.7, linewidth=1)

ax2.text(38, 3, 'Feasible\n(<1 week)', fontsize=8, color='green', fontweight='bold', ha='center')
ax2.text(38, 14, 'Challenging\n(1-4 weeks)', fontsize=8, color='goldenrod', fontweight='bold', ha='center')
ax2.text(38, 200, 'Infeasible\n(>1 month)', fontsize=8, color='red', fontweight='bold', ha='center')

# Mark key points
for n, label in [(25, '25q: 5 days'), (27, '27q: 20 days'), (30, '30q: 5.3 mo')]:
    idx = n - 20
    ax2.annotate(label, xy=(n, training_1M_days[idx]),
                xytext=(n+2, training_1M_days[idx]*0.3),
                fontsize=7, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=0.8))

ax2.set_xlabel('Number of Logical Qubits ($n$)', fontsize=11)
ax2.set_ylabel('Wall Time for $10^6$ Episodes (days)', fontsize=11)
ax2.set_title('(b) DRL Training Feasibility (Single A100)', fontsize=12, fontweight='bold')
ax2.set_xlim(20, 40)
ax2.set_ylim(0.1, 1e6)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_scaling_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/siliqun/paper/fig_scaling_analysis.pdf', bbox_inches='tight')
print("Scaling figure saved.")
