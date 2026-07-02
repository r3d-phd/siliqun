"""Generate publication-quality figures for SiliQun SoftwareX paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Simulation Speed vs Qubit Count
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

qubits = [2, 4, 6, 8, 10, 12]
speeds = [37060.4, 26188.1, 18777.6, 17676.7, 11038.8, 14470.5]
bond_dims = [2, 4, 8, 16, 32, 16]

color = '#2166ac'
ax.plot(qubits, speeds, 'o-', color=color, linewidth=2, markersize=8, label='Steps/second')

# Annotate bond dimensions
for i, (q, s, bd) in enumerate(zip(qubits, speeds, bond_dims)):
    ax.annotate(f'$\\chi$={bd}', (q, s), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, color='#666666')

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Simulation Speed (steps/s)')
ax.set_title('SiliQun Simulation Throughput vs. System Size')
ax.set_xticks(qubits)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 45000)

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_speed_scaling.png')
plt.close()
print("Generated: fig_speed_scaling.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Noisy vs Noiseless Throughput
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

qubits_noise = [2, 4, 6, 8]
noiseless = [153901.7, 152839.5, 177892.0, 174443.3]
noisy = [87051.3, 82917.5, 87796.2, 80728.9]

x = np.arange(len(qubits_noise))
width = 0.35

bars1 = ax.bar(x - width/2, noiseless, width, label='Noiseless', color='#4393c3', edgecolor='white')
bars2 = ax.bar(x + width/2, noisy, width, label='Noisy', color='#d6604d', edgecolor='white')

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Throughput (steps/s)')
ax.set_title('Simulation Throughput: Noiseless vs. Noisy')
ax.set_xticks(x)
ax.set_xticklabels(qubits_noise)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add overhead percentage labels
for i in range(len(qubits_noise)):
    overhead = (1 - noisy[i] / noiseless[i]) * 100
    mid_x = x[i] + width/2
    ax.annotate(f'{overhead:.0f}% overhead', (mid_x, noisy[i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color='#666666')

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_noise_overhead.png')
plt.close()
print("Generated: fig_noise_overhead.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: DRL Environment Throughput
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

# Env throughput data (2q only completed before crash)
# We'll show the comparison with typical DRL requirements
categories = ['SiliQun\n(2Q, Clean)', 'SiliQun\n(2Q, Noisy)', 'Typical DRL\nRequirement', 'Atari\nBenchmark']
throughputs = [5303.4, 5327.2, 1000, 3000]
colors = ['#4393c3', '#d6604d', '#999999', '#999999']

bars = ax.bar(categories, throughputs, color=colors, edgecolor='white', width=0.6)

ax.set_ylabel('Environment Steps/second')
ax.set_title('DRL Environment Throughput Comparison')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
            f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=1000, color='green', linestyle='--', alpha=0.5, label='Min. viable for DRL')
ax.legend()

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_env_throughput.png')
plt.close()
print("Generated: fig_env_throughput.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Gate Accuracy Table (as figure)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 3))
ax.axis('off')

gate_data = [
    ['$R_x(\\pi)|0\\rangle \\to |1\\rangle$', '$\\langle Z \\rangle$', '$-1.0$', '$-1.0$', '$0$'],
    ['$R_y(\\pi/2)|0\\rangle \\to |+\\rangle$', '$\\langle Z \\rangle$', '$0.0$', '$0.0$', '$2.2 \\times 10^{-16}$'],
    ['$R_z(\\pi)$ roundtrip', '$\\langle Z \\rangle$', '$-1.0$', '$-1.0$', '$0$'],
    ['Bell state $|\\Phi^+\\rangle$', '$\\langle ZZ \\rangle$', '$1.0$', '$1.0$', '$0$'],
    ['CNOT$|00\\rangle$', 'Fidelity', '$1.0$', '$1.0$', '$0$'],
    ['Bell entropy', '$S$', '$1.0$', '$1.0$', '$0$'],
]

table = ax.table(
    cellText=gate_data,
    colLabels=['Operation', 'Observable', 'Measured', 'Expected', 'Error'],
    cellLoc='center',
    loc='center',
    colWidths=[0.3, 0.15, 0.15, 0.15, 0.25],
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style header
for j in range(5):
    table[0, j].set_facecolor('#2166ac')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(gate_data) + 1):
    for j in range(5):
        if i % 2 == 0:
            table[i, j].set_facecolor('#f0f0f0')

ax.set_title('Gate Fidelity Validation Results', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_gate_accuracy.png')
plt.close()
print("Generated: fig_gate_accuracy.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Device Profile Comparison (conceptual)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

devices = ['Donor\n(P in Si)', 'SiMOS\n(QD)', 'GAA\n(Nanowire)']
t1_values = [30.0, 0.01, 0.1]  # seconds
t2_star = [2e-3, 2e-5, 5e-4]  # seconds

x = np.arange(len(devices))
width = 0.35

ax_t1 = ax
ax_t2 = ax.twinx()

bars1 = ax_t1.bar(x - width/2, t1_values, width, label='$T_1$ (s)', color='#4393c3', alpha=0.8)
bars2 = ax_t2.bar(x + width/2, [t * 1000 for t in t2_star], width, label='$T_2^*$ (ms)', color='#d6604d', alpha=0.8)

ax_t1.set_xlabel('Device Architecture')
ax_t1.set_ylabel('$T_1$ Relaxation Time (s)', color='#4393c3')
ax_t2.set_ylabel('$T_2^*$ Dephasing Time (ms)', color='#d6604d')
ax_t1.set_xticks(x)
ax_t1.set_xticklabels(devices)
ax_t1.set_yscale('log')
ax_t2.set_yscale('log')
ax_t1.set_title('Silicon Spin Qubit Device Profiles in SiliQun')

lines1, labels1 = ax_t1.get_legend_handles_labels()
lines2, labels2 = ax_t2.get_legend_handles_labels()
ax_t1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_device_profiles.png')
plt.close()
print("Generated: fig_device_profiles.png")

print("\nAll figures generated successfully!")
