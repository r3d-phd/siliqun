#!/usr/bin/env python3
"""Create multi-scale SLEDGE grid figure using Draw.io MCP client."""
import sys
sys.path.insert(0, "/home/ubuntu/skills/drawio-diagramming/scripts")
from drawio_client import DrawioClient

client = DrawioClient()
client.initialize()

# ========== Style Definitions ==========
# DFS logical qubit (triplet of 3 physical spins) - blue circle
qubit_style = "whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;rounded=1;fontSize=9;fontStyle=1;aspect=fixed;shape=ellipse;"
# Physical spin dot inside qubit - small dark circle
spin_style = "whiteSpace=wrap;html=1;fillColor=#0050ef;strokeColor=#001DBC;fontColor=#ffffff;rounded=1;fontSize=7;aspect=fixed;shape=ellipse;"
# Exchange coupling line between qubits
coupling_style = "edgeStyle=entityRelationEdgeStyle;rounded=0;orthogonalLoop=1;html=1;strokeWidth=2;strokeColor=#6c8ebf;endArrow=none;endFill=0;"
# Container for each grid configuration
container_benchmarked = "whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;rounded=1;dashed=0;verticalAlign=top;fontSize=13;fontStyle=1;opacity=30;"
container_projected = "whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=1;dashed=1;dashPattern=8 8;verticalAlign=top;fontSize=13;fontStyle=1;opacity=30;"
# Title and label styles
title_style = "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=18;fontStyle=1;"
label_style = "text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=11;"
badge_benchmarked = "whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;rounded=1;fontSize=10;fontStyle=1;fontColor=#2D7600;"
badge_projected = "whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;rounded=1;fontSize=10;fontStyle=1;fontColor=#9C6500;dashed=1;"
badge_max = "whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;rounded=1;fontSize=10;fontStyle=1;fontColor=#AE0000;dashed=1;"

# ========== Grid Configurations ==========
configs = [
    {"rows": 3, "cols": 3, "label": "3x3 Grid", "qubits": 9, "spins": 27, 
     "status": "Benchmarked", "time": "< 1 ms/ep", "badge": badge_benchmarked},
    {"rows": 4, "cols": 4, "label": "4x4 Grid", "qubits": 16, "spins": 48,
     "status": "Benchmarked", "time": "~5 ms/ep", "badge": badge_benchmarked},
    {"rows": 5, "cols": 5, "label": "5x5 Grid", "qubits": 25, "spins": 75,
     "status": "Benchmarked", "time": "430 ms/ep", "badge": badge_benchmarked},
    {"rows": 6, "cols": 6, "label": "6x6 Grid", "qubits": 36, "spins": 108,
     "status": "Projected (H200)", "time": "~110 s/ep", "badge": badge_max},
]

# Layout parameters
qsize = 36       # qubit circle size
spacing = 52     # spacing between qubit centers
panel_gap = 60   # gap between panels
start_x = 40
start_y = 80

def draw_grid(x_offset, y_offset, rows, cols, config):
    """Draw a SLEDGE grid at the given offset."""
    ids = {}
    
    # Draw qubit circles
    for r in range(rows):
        for c in range(cols):
            cx = x_offset + c * spacing
            cy = y_offset + r * spacing
            qid = client.add_rectangle(
                x=cx, y=cy, width=qsize, height=qsize,
                text=f"Q<sub>{r*cols+c+1}</sub>",
                style=qubit_style
            )
            ids[(r, c)] = qid
    
    # Draw exchange coupling edges (horizontal and vertical)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                client.add_edge(ids[(r, c)], ids[(r, c+1)], style=coupling_style)
            if r + 1 < rows:
                client.add_edge(ids[(r, c)], ids[(r+1, c)], style=coupling_style)
    
    return ids

# ========== Draw Each Configuration ==========
x_cursor = start_x

for i, cfg in enumerate(configs):
    rows, cols = cfg["rows"], cfg["cols"]
    grid_w = (cols - 1) * spacing + qsize
    grid_h = (rows - 1) * spacing + qsize
    panel_w = max(grid_w + 40, 180)
    panel_h = grid_h + 120
    
    # Container box
    is_benchmarked = "Benchmarked" in cfg["status"]
    cstyle = container_benchmarked if is_benchmarked else container_projected
    client.add_rectangle(
        x=x_cursor - 10, y=start_y - 10,
        width=panel_w, height=panel_h,
        text="", style=cstyle
    )
    
    # Grid title
    client.add_rectangle(
        x=x_cursor, y=start_y - 5,
        width=panel_w - 20, height=20,
        text=f"<b>{cfg['label']}</b>",
        style=title_style.replace("fontSize=18", "fontSize=14")
    )
    
    # Draw the actual grid
    grid_x = x_cursor + (panel_w - grid_w) // 2 - 10
    grid_y = start_y + 25
    draw_grid(grid_x, grid_y, rows, cols, cfg)
    
    # Info labels below grid
    info_y = grid_y + grid_h + 10
    client.add_rectangle(
        x=x_cursor, y=info_y,
        width=panel_w - 20, height=18,
        text=f"{cfg['qubits']} logical qubits ({cfg['spins']} spins)",
        style=label_style
    )
    client.add_rectangle(
        x=x_cursor, y=info_y + 18,
        width=panel_w - 20, height=18,
        text=f"{cfg['time']}",
        style=label_style.replace("fontSize=11", "fontSize=10")
    )
    
    # Status badge
    client.add_rectangle(
        x=x_cursor + (panel_w - 130) // 2 - 10, y=info_y + 40,
        width=120, height=22,
        text=cfg["status"],
        style=cfg["badge"]
    )
    
    x_cursor += panel_w + panel_gap

# ========== Legend ==========
legend_y = start_y + 400
client.add_rectangle(x=start_x, y=legend_y, width=120, height=22,
    text="Benchmarked", style=badge_benchmarked)
client.add_rectangle(x=start_x + 140, y=legend_y, width=160, height=22,
    text="Projected (memory limit)", style=badge_max)
client.add_rectangle(x=start_x + 320, y=legend_y, width=30, height=30,
    text="Q", style=qubit_style)
client.add_rectangle(x=start_x + 360, y=legend_y, width=200, height=22,
    text="= DFS logical qubit (3 spins)", style=label_style)

print("Multi-scale SLEDGE grid diagram created successfully!")
print("Open http://localhost:3000/ to view and export.")
