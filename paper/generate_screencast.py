#!/usr/bin/env python3
"""
Generate an explanatory screencast (MP4) for SiliQun SoftwareX submission.

This creates a series of annotated frames showing:
1. Title slide with SiliQun branding
2. Installation and import
3. Creating a simulation environment
4. Running a DRL training loop
5. GPU benchmark results
6. Architecture overview

Uses matplotlib animation to produce a clean, professional video.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Video settings
FPS = 2  # Slow enough to read
DPI = 150
FIGSIZE = (16, 9)  # 16:9 aspect ratio

# Color scheme
BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
CODE_BG = '#161b22'
ACCENT = '#58a6ff'
GREEN = '#3fb950'
ORANGE = '#d29922'
RED = '#f85149'
PURPLE = '#bc8cff'

def make_frame_title():
    """Frame 1: Title slide"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 6.5, 'SiliQun', fontsize=72, fontweight='bold',
            color=ACCENT, ha='center', va='center', fontfamily='monospace')
    ax.text(8, 5.2, 'GPU-Accelerated Tensor-Network Simulator', fontsize=24,
            color=TEXT_COLOR, ha='center', va='center')
    ax.text(8, 4.5, 'for Silicon Spin Qubit DRL Control', fontsize=24,
            color=TEXT_COLOR, ha='center', va='center')
    ax.text(8, 3.2, 'v2.0.0  |  Open Source (MIT)  |  Python 3.8+', fontsize=16,
            color='#8b949e', ha='center', va='center', fontfamily='monospace')
    ax.text(8, 1.5, 'Raad Alshehri', fontsize=18,
            color=TEXT_COLOR, ha='center', va='center')
    ax.text(8, 1.0, 'King Abdulaziz University', fontsize=14,
            color='#8b949e', ha='center', va='center')
    return fig

def make_frame_install():
    """Frame 2: Installation"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 8.2, 'Quick Start: Installation', fontsize=32, fontweight='bold',
            color=ACCENT, ha='center', va='center')

    # Terminal-style code block
    code_lines = [
        ('$', ' pip install siliqun', '#8b949e', GREEN),
        ('$', ' pip install siliqun[gpu]    # For CUDA support', '#8b949e', GREEN),
        (None, None, None, None),
        ('>>>', ' import siliqun', ORANGE, TEXT_COLOR),
        ('>>>', ' from siliqun.engine.gym_env import SiliQunEnv', ORANGE, TEXT_COLOR),
        ('>>>', ' from siliqun.engine.statevector_simulator import StateVectorSimulator', ORANGE, TEXT_COLOR),
        (None, None, None, None),
        ('>>>', ' print(siliqun.__version__)', ORANGE, TEXT_COLOR),
        ('', "  '2.0.0'", BG_COLOR, GREEN),
    ]

    y = 6.8
    for prompt, code, prompt_color, code_color in code_lines:
        if prompt is not None and (prompt or code):
            ax.text(1.5, y, prompt, fontsize=14, color=prompt_color,
                    fontfamily='monospace', va='center')
            ax.text(1.5 + len(prompt) * 0.12, y, code, fontsize=14,
                    color=code_color, fontfamily='monospace', va='center')
        y -= 0.55

    # Add box around code
    rect = FancyBboxPatch((1, 2.2), 14, 5.2, boxstyle="round,pad=0.3",
                          facecolor=CODE_BG, edgecolor='#30363d', linewidth=2)
    ax.add_patch(rect)
    return fig

def make_frame_env_setup():
    """Frame 3: Environment setup"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 8.2, 'Creating a Simulation Environment', fontsize=32,
            fontweight='bold', color=ACCENT, ha='center', va='center')

    code_lines = [
        ('# Configure the simulator', '#8b949e'),
        ('config = SimConfig(', TEXT_COLOR),
        ('    noise_enabled=True,', TEXT_COLOR),
        ('    sim_mode="auto"       # MPS for 1D, GPU-SV for 2D', '#8b949e'),
        (')', TEXT_COLOR),
        (None, None),
        ('# Create the Gymnasium environment', '#8b949e'),
        ('env = SiliQunEnv(', TEXT_COLOR),
        ('    device="sledge_5x5",  # 25-qubit 2D grid', '#8b949e'),
        ('    n_qubits=25,', TEXT_COLOR),
        ('    target_state="ghz",', TEXT_COLOR),
        ('    config=config', TEXT_COLOR),
        (')', TEXT_COLOR),
        (None, None),
        ('obs, info = env.reset()', GREEN),
        ('print(f"State dim: {env.observation_space.shape}")', GREEN),
    ]

    rect = FancyBboxPatch((1, 1.0), 14, 6.5, boxstyle="round,pad=0.3",
                          facecolor=CODE_BG, edgecolor='#30363d', linewidth=2)
    ax.add_patch(rect)

    y = 7.0
    for code, color in code_lines:
        if code is not None and code:
            ax.text(1.8, y, code, fontsize=13, color=color,
                    fontfamily='monospace', va='center')
        y -= 0.42
    return fig

def make_frame_training():
    """Frame 4: DRL training loop"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 8.2, 'DRL Training with Stable-Baselines3', fontsize=32,
            fontweight='bold', color=ACCENT, ha='center', va='center')

    code_lines = [
        ('from stable_baselines3 import PPO', TEXT_COLOR),
        (None, None),
        ('# Train a PPO agent on the 5x5 grid', '#8b949e'),
        ('model = PPO("MlpPolicy", env, verbose=1,', TEXT_COLOR),
        ('            learning_rate=3e-4,', TEXT_COLOR),
        ('            n_steps=2048)', TEXT_COLOR),
        (None, None),
        ('model.learn(total_timesteps=1_000_000)', GREEN),
        (None, None),
        ('# Evaluate the trained agent', '#8b949e'),
        ('obs, _ = env.reset()', TEXT_COLOR),
        ('for step in range(100):', TEXT_COLOR),
        ('    action, _ = model.predict(obs)', TEXT_COLOR),
        ('    obs, reward, done, _, info = env.step(action)', TEXT_COLOR),
        ('    print(f"Step {step}: F={info[\'fidelity\']:.4f}")', GREEN),
    ]

    rect = FancyBboxPatch((1, 1.2), 14, 6.2, boxstyle="round,pad=0.3",
                          facecolor=CODE_BG, edgecolor='#30363d', linewidth=2)
    ax.add_patch(rect)

    y = 6.9
    for code, color in code_lines:
        if code is not None and code:
            ax.text(1.8, y, code, fontsize=13, color=color,
                    fontfamily='monospace', va='center')
        y -= 0.40
    return fig

def make_frame_benchmarks():
    """Frame 5: GPU benchmark results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=BG_COLOR)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    fig.suptitle('GPU Benchmark Results (NVIDIA A100)', fontsize=28,
                 fontweight='bold', color=ACCENT, y=0.95)

    # Left: execution times
    operations = ['1Q Gate\n(Rx)', '2Q Gate\n(CNOT)', 'Z Expect']
    cpu_times = [317.2, 555.9, 122.8]
    gpu_times = [10.6, 6.2, 1.3]

    x = np.arange(len(operations))
    width = 0.35

    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU (NumPy)',
                    color=RED, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU (CuPy/A100)',
                    color=GREEN, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Execution Time (ms)', color=TEXT_COLOR, fontsize=14)
    ax1.set_title('25-Qubit Gate Times', color=TEXT_COLOR, fontsize=16, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, color=TEXT_COLOR)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12, facecolor=CODE_BG, edgecolor='#30363d',
               labelcolor=TEXT_COLOR)
    ax1.grid(axis='y', alpha=0.2, color='#30363d')

    # Right: speedup factors
    speedups = [30, 89, 93]
    colors = [ACCENT, PURPLE, ORANGE]
    bars = ax2.bar(operations, speedups, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{val}x', ha='center', va='bottom', color=TEXT_COLOR,
                fontsize=16, fontweight='bold')

    ax2.set_ylabel('Speedup Factor', color=TEXT_COLOR, fontsize=14)
    ax2.set_title('GPU Speedup (vs CPU)', color=TEXT_COLOR, fontsize=16, pad=10)
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.2, color='#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

def make_frame_architecture():
    """Frame 6: Architecture overview"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 8.5, 'SiliQun Architecture', fontsize=32, fontweight='bold',
            color=ACCENT, ha='center', va='center')

    # Draw 4 layers as boxes
    layers = [
        ('Environment Layer', 'Gymnasium API  |  Observation / Action / Reward', ACCENT, 7.0),
        ('Physics Layer', 'Device Profiles  |  Noise Models  |  DFS Encoding', PURPLE, 5.5),
        ('Simulation Core', 'MPS Engine (1D)  |  GPU State Vector Engine (2D)', GREEN, 4.0),
        ('Backend Layer', 'NumPy (CPU)  |  CuPy (CUDA GPU)', ORANGE, 2.5),
    ]

    for name, desc, color, y in layers:
        rect = FancyBboxPatch((2, y - 0.5), 12, 1.0, boxstyle="round,pad=0.15",
                              facecolor=color + '22', edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(8, y + 0.15, name, fontsize=18, fontweight='bold',
                color=color, ha='center', va='center')
        ax.text(8, y - 0.2, desc, fontsize=12,
                color='#8b949e', ha='center', va='center')

    # Arrows between layers
    for i in range(3):
        y_top = layers[i][3] - 0.5
        y_bot = layers[i+1][3] + 0.5
        ax.annotate('', xy=(8, y_bot + 0.05), xytext=(8, y_top - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#8b949e',
                                   lw=2, connectionstyle='arc3'))

    ax.text(8, 1.2, 'Modular design: swap backends, add devices, or extend physics without touching other layers',
            fontsize=12, color='#8b949e', ha='center', va='center', style='italic')
    return fig

def make_frame_scaling():
    """Frame 7: Scaling capabilities"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 8.2, 'Scaling: From 2 to 25 Qubits', fontsize=32,
            fontweight='bold', color=ACCENT, ha='center', va='center')

    # Grid visualizations
    grids = [
        ('2x2', 4, 2, 3.0),
        ('3x3', 9, 3, 6.5),
        ('4x4', 16, 4, 10.0),
        ('5x5', 25, 5, 13.5),
    ]

    for label, qubits, size, cx in grids:
        # Draw grid
        spacing = 0.5
        start_x = cx - (size - 1) * spacing / 2
        start_y = 5.5 - (size - 1) * spacing / 2

        for i in range(size):
            for j in range(size):
                x = start_x + j * spacing
                y = start_y + i * spacing
                circle = plt.Circle((x, y), 0.15, color=GREEN, alpha=0.8)
                ax.add_patch(circle)

                # Horizontal connections
                if j < size - 1:
                    ax.plot([x + 0.15, x + spacing - 0.15], [y, y],
                            color='#8b949e', linewidth=1.5, alpha=0.6)
                # Vertical connections
                if i < size - 1:
                    ax.plot([x, x], [y + 0.15, y + spacing - 0.15],
                            color='#8b949e', linewidth=1.5, alpha=0.6)

        ax.text(cx, 3.2, f'{label}', fontsize=16, fontweight='bold',
                color=TEXT_COLOR, ha='center', va='center')
        ax.text(cx, 2.7, f'{qubits}q', fontsize=14,
                color=ACCENT, ha='center', va='center')

    # Memory requirements
    ax.text(3.0, 1.5, '~0 MB', fontsize=12, color='#8b949e', ha='center')
    ax.text(6.5, 1.5, '~0 MB', fontsize=12, color='#8b949e', ha='center')
    ax.text(10.0, 1.5, '1 MB', fontsize=12, color=ORANGE, ha='center')
    ax.text(13.5, 1.5, '512 MB', fontsize=12, color=RED, ha='center')

    # Arrow showing progression
    ax.annotate('', xy=(14.5, 2.0), xytext=(1.5, 2.0),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=3))

    return fig

def make_frame_summary():
    """Frame 8: Summary"""
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    ax.text(8, 7.8, 'SiliQun: Key Highlights', fontsize=32, fontweight='bold',
            color=ACCENT, ha='center', va='center')

    highlights = [
        (GREEN, 'Dual Engine', 'MPS for 1D chains + GPU State Vector for 2D grids'),
        (ACCENT, '25-Qubit Simulation', 'Exact logical-subspace dynamics on 5x5 SLEDGE grid'),
        (PURPLE, '30-93x GPU Speedup', 'NVIDIA A100 acceleration via CuPy backend'),
        (ORANGE, 'DFS Encoding', 'Logical subspace projection with perturbative leakage'),
        (RED, '3 Device Profiles', 'Donor, SiMOS, and GAA with calibrated noise models'),
        ('#8b949e', 'Gymnasium Native', 'Drop-in compatible with Stable-Baselines3, RLlib, etc.'),
    ]

    y = 6.5
    for color, title, desc in highlights:
        # Bullet
        circle = plt.Circle((2.0, y), 0.12, color=color, alpha=0.9)
        ax.add_patch(circle)
        ax.text(2.8, y + 0.05, title, fontsize=18, fontweight='bold',
                color=color, va='center')
        ax.text(2.8, y - 0.35, desc, fontsize=13,
                color='#8b949e', va='center')
        y -= 0.95

    ax.text(8, 0.8, 'github.com/raad-alshehri/siliqun', fontsize=16,
            color=ACCENT, ha='center', va='center', fontfamily='monospace')

    return fig


def main():
    outdir = '/home/ubuntu/siliqun/paper'

    # Generate all frames
    frame_funcs = [
        make_frame_title,
        make_frame_install,
        make_frame_env_setup,
        make_frame_training,
        make_frame_benchmarks,
        make_frame_architecture,
        make_frame_scaling,
        make_frame_summary,
    ]

    frames = []
    for i, func in enumerate(frame_funcs):
        print(f"Generating frame {i+1}/{len(frame_funcs)}: {func.__name__}")
        fig = func()
        # Save as temporary PNG
        path = os.path.join(outdir, f'_frame_{i:02d}.png')
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        frames.append(path)

    # Create video from frames using matplotlib animation
    print("Assembling video...")
    fig_vid, ax_vid = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    ax_vid.axis('off')

    # Load all frame images
    images = []
    for path in frames:
        img = plt.imread(path)
        images.append(img)

    im = ax_vid.imshow(images[0])

    # Each frame shown for 4 seconds at 2 FPS = 8 ticks per frame
    frame_duration = 4  # seconds per slide
    ticks_per_frame = frame_duration * FPS

    def animate(tick):
        frame_idx = min(tick // ticks_per_frame, len(images) - 1)
        im.set_array(images[frame_idx])
        return [im]

    total_ticks = ticks_per_frame * len(images)
    anim = animation.FuncAnimation(fig_vid, animate, frames=total_ticks,
                                   interval=1000//FPS, blit=True)

    output_path = os.path.join(outdir, 'siliqun_screencast.mp4')
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)

    try:
        anim.save(output_path, writer=writer, dpi=DPI,
                  savefig_kwargs={'facecolor': BG_COLOR})
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"FFmpeg failed: {e}")
        # Fallback: try Pillow writer
        try:
            writer2 = animation.PillowWriter(fps=FPS)
            gif_path = os.path.join(outdir, 'siliqun_screencast.gif')
            anim.save(gif_path, writer=writer2, dpi=100,
                      savefig_kwargs={'facecolor': BG_COLOR})
            print(f"GIF saved to {gif_path}")
        except Exception as e2:
            print(f"Pillow also failed: {e2}")
            print("Individual frames saved as _frame_XX.png")

    plt.close(fig_vid)

    # Clean up temp frames
    # Keep them for now as backup
    print("Done! Frame PNGs also available in paper/ directory.")


if __name__ == '__main__':
    main()
