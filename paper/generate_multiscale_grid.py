#!/usr/bin/env python3
"""Generate multi-scale SLEDGE grid figure for the SiliQun paper.
Shows 3x3, 4x4, 5x5, and 6x6 DFS-encoded qubit grids with benchmarked/projected labels."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

fig = plt.figure(figsize=(16, 5.5))

# Grid configurations
configs = [
    {"rows": 3, "cols": 3, "label": "(a) 3x3 Grid", "qubits": 9, "spins": 27,
     "status": "Benchmarked", "time": "< 1 ms/ep", "color": "#d5e8d4", "edge_color": "#82b366"},
    {"rows": 4, "cols": 4, "label": "(b) 4x4 Grid", "qubits": 16, "spins": 48,
     "status": "Benchmarked", "time": "~5 ms/ep", "color": "#d5e8d4", "edge_color": "#82b366"},
    {"rows": 5, "cols": 5, "label": "(c) 5x5 Grid", "qubits": 25, "spins": 75,
     "status": "Benchmarked", "time": "430 ms/ep", "color": "#d5e8d4", "edge_color": "#82b366"},
    {"rows": 6, "cols": 6, "label": "(d) 6x6 Grid", "qubits": 36, "spins": 108,
     "status": "Projected (H200)", "time": "~110 s/ep", "color": "#fff2cc", "edge_color": "#d6b656"},
]

# Create subplots with custom widths
gs = fig.add_gridspec(1, 4, width_ratios=[3, 4, 5, 6], wspace=0.25)

for idx, cfg in enumerate(configs):
    ax = fig.add_subplot(gs[idx])
    rows, cols = cfg["rows"], cfg["cols"]
    
    # Draw background panel
    bg_color = cfg["color"]
    is_projected = "Projected" in cfg["status"]
    
    ax.set_xlim(-0.8, cols - 0.2)
    ax.set_ylim(-0.8, rows + 1.2)
    
    # Background rectangle
    bg = mpatches.FancyBboxPatch(
        (-0.7, -0.7), cols - 0.2 + 0.5, rows + 1.8,
        boxstyle="round,pad=0.1",
        facecolor=bg_color, edgecolor=cfg["edge_color"],
        linewidth=2, linestyle='--' if is_projected else '-',
        alpha=0.3, zorder=0
    )
    ax.add_patch(bg)
    
    # Draw exchange coupling lines (horizontal and vertical)
    for r in range(rows):
        for c in range(cols):
            y = rows - 1 - r  # flip y for visual
            if c + 1 < cols:
                ax.plot([c, c+1], [y, y], '-', color='#6c8ebf', linewidth=1.5, alpha=0.6, zorder=1)
            if r + 1 < rows:
                ax.plot([c, c], [y, y-1], '-', color='#6c8ebf', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Draw qubit circles
    for r in range(rows):
        for c in range(cols):
            y = rows - 1 - r
            # Outer circle (DFS logical qubit)
            circle = plt.Circle((c, y), 0.32, facecolor='#dae8fc', edgecolor='#6c8ebf',
                              linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            
            # Three small dots inside (representing 3 physical spins)
            offsets = [(-0.1, 0.08), (0.1, 0.08), (0.0, -0.1)]
            for ox, oy in offsets:
                dot = plt.Circle((c + ox, y + oy), 0.06, facecolor='#0050ef',
                               edgecolor='#001DBC', linewidth=0.5, zorder=4)
                ax.add_patch(dot)
    
    # Title
    ax.set_title(cfg["label"], fontsize=13, fontweight='bold', pad=8)
    
    # Info text below grid
    info_text = f"{cfg['qubits']} logical qubits ({cfg['spins']} spins)"
    ax.text((cols-1)/2, -0.55, info_text, ha='center', va='top', fontsize=9, fontweight='bold')
    ax.text((cols-1)/2, -0.78, cfg["time"], ha='center', va='top', fontsize=8, color='#555555')
    
    # Status badge
    badge_color = '#d5e8d4' if 'Benchmarked' in cfg["status"] else '#fff2cc'
    badge_edge = '#82b366' if 'Benchmarked' in cfg["status"] else '#d6b656'
    badge_text_color = '#2D7600' if 'Benchmarked' in cfg["status"] else '#9C6500'
    
    if 'Projected' in cfg["status"]:
        badge_color = '#f8cecc'
        badge_edge = '#b85450'
        badge_text_color = '#AE0000'
    
    badge = mpatches.FancyBboxPatch(
        ((cols-1)/2 - 0.9, -1.15), 1.8, 0.3,
        boxstyle="round,pad=0.05",
        facecolor=badge_color, edgecolor=badge_edge,
        linewidth=1.2, zorder=5
    )
    ax.add_patch(badge)
    ax.text((cols-1)/2, -1.0, cfg["status"], ha='center', va='center',
           fontsize=8, fontweight='bold', color=badge_text_color, zorder=6)
    
    ax.set_aspect('equal')
    ax.axis('off')

# Add legend at the bottom
legend_elements = [
    mpatches.Patch(facecolor='#dae8fc', edgecolor='#6c8ebf', linewidth=1.5,
                  label='DFS logical qubit (3 physical spins)'),
    mlines.Line2D([0], [0], color='#6c8ebf', linewidth=1.5, alpha=0.6,
                 label='Exchange coupling'),
    mpatches.Patch(facecolor='#d5e8d4', edgecolor='#82b366', linewidth=1.5,
                  label='Benchmarked on A100'),
    mpatches.Patch(facecolor='#f8cecc', edgecolor='#b85450', linewidth=1.5,
                  linestyle='--', label='Projected (memory limit)'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
          frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/home/ubuntu/siliqun/paper/fig_multiscale_grid.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.savefig('/home/ubuntu/siliqun/paper/fig_multiscale_grid.pdf', bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("Multi-scale SLEDGE grid figure saved.")
