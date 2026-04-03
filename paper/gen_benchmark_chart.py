#!/usr/bin/env python3
"""Generate GPU benchmark comparison chart for SiliQun paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from actual Aziz A100 benchmarks
operations = ['1Q Gate\n(Rx)', '2Q Gate\n(CNOT)', 'Z Expectation\n(⟨Z⟩)']
cpu_times = [317.2, 555.9, 122.8]  # ms
gpu_times = [10.6, 6.2, 1.3]      # ms
speedups = [30, 89, 93]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 2]})

# Left panel: CPU vs GPU times (log scale)
x = np.arange(len(operations))
width = 0.35

bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU (NumPy)', 
                color='#DAE8FC', edgecolor='#6C8EBF', linewidth=1.2)
bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU (CuPy/A100)', 
                color='#D5E8D4', edgecolor='#82B366', linewidth=1.2)

ax1.set_yscale('log')
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('25-Qubit Gate Execution Time', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(operations, fontsize=10)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0.5, 1000)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Right panel: Speedup factors
colors = ['#4472C4', '#ED7D31', '#A5A5A5']
bars3 = ax2.barh(operations, speedups, color=colors, edgecolor='black', linewidth=0.8, height=0.5)
ax2.set_xlabel('Speedup (×)', fontsize=12)
ax2.set_title('GPU Speedup Factor', fontsize=13, fontweight='bold')
ax2.set_xlim(0, 110)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, s in zip(bars3, speedups):
    ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f'{s}×', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_benchmark.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/siliqun/paper/fig_benchmark.png', dpi=300, bbox_inches='tight')
print("Benchmark chart saved.")
