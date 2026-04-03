#!/usr/bin/env python3
"""Generate DFS validation figure for SoftwareX paper."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

# Load validation results
with open('/home/ubuntu/siliqun/validation/dfs_validation_results.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Perturbative regime - leakage vs angle
pert = data['tests']['perturbative_regime']
angles_rad = np.array(pert['angles_rad'])
leakage = np.array(pert['leakage'])
fidelity = np.array(pert['projected_fidelity'])

ax = axes[0]
ax.plot(angles_rad * 180 / np.pi, leakage * 100, 'b-o', markersize=3, linewidth=1.5, label='Measured leakage')
# Fit quadratic
theta_fit = np.linspace(0, angles_rad[-1], 100)
# leakage ~ C * theta^2
C = leakage[-1] / angles_rad[-1]**2
ax.plot(theta_fit * 180 / np.pi, C * theta_fit**2 * 100, 'r--', linewidth=1.2, 
        label=f'$O(\\theta^2)$ fit (exp={pert["leakage_scaling_exponent"]:.2f})')
ax.set_xlabel('Inter-qubit coupling angle (degrees)')
ax.set_ylabel('Leakage (%)')
ax.set_title('(a) Perturbative Regime')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Full range - leakage and fidelity vs angle
inter = data['tests']['inter_qubit_exchange']
angles_deg = np.array(inter['angles_deg'])
leak_full = np.array(inter['leakage'])
fid_full = np.array(inter['projected_fidelity'])

ax = axes[1]
ax.plot(angles_deg, fid_full * 100, 'b-s', markersize=4, linewidth=1.5, label='Projected fidelity')
ax.plot(angles_deg, (1 - leak_full) * 100, 'r-^', markersize=4, linewidth=1.5, label='DFS population')
ax.axhline(y=99, color='green', linestyle=':', linewidth=1, alpha=0.7, label='99% threshold')
ax.axvline(x=17, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.annotate('Perturbative\nregime', xy=(8, 50), fontsize=8, ha='center', color='gray')
ax.set_xlabel('Inter-qubit coupling angle (degrees)')
ax.set_ylabel('Percentage (%)')
ax.set_title('(b) Full Coupling Range')
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

# Panel (c): Encoded SWAP verification
swap = data['tests']['encoded_swap']
states = [r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$']
keys = ['|00_L>', '|01_L>', '|10_L>', '|11_L>']

# Build the decoded state matrix
swap_matrix = np.zeros((4, 4))
for i, k in enumerate(keys):
    decoded = swap[k]['decoded_state']
    for j in range(4):
        swap_matrix[i, j] = decoded[j]

ax = axes[2]
im = ax.imshow(swap_matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(4))
ax.set_xticklabels(states, fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(states, fontsize=10)
ax.set_xlabel('Output state')
ax.set_ylabel('Input state')
ax.set_title('(c) Encoded SWAP Verification')

# Add text annotations
for i in range(4):
    for j in range(4):
        val = swap_matrix[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

plt.colorbar(im, ax=ax, shrink=0.8, label='Probability')

plt.tight_layout()
plt.savefig('/home/ubuntu/siliqun/paper/fig_dfs_validation.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/siliqun/paper/fig_dfs_validation.pdf', bbox_inches='tight')
print("DFS validation figure saved!")
