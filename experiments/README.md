# SiliQun Experiments & Reproducibility

This folder contains the experimental results, training logs, and generated plots for the SiliQun framework as presented in the paper. It is designed to provide full transparency and reproducibility for all claims made regarding SiliQun's performance, scalability, and convergence.

## Directory Structure

```
experiments/
├── plots/
│   ├── ablation/                 # Algorithm comparison (PPO, SAC, GRAPE, GAA)
│   ├── fidelity_vs_episode/      # Convergence curves for various target states
│   ├── ppo_reward_curves/        # Detailed PPO training metrics per seed
│   ├── scalability/              # Fidelity vs Qubit count (N=2 to 12)
│   └── tensorboard_scalars/      # Exported scalar metrics for TensorBoard
└── README.md                     # This file
```

*(Note: The raw JSON results and source code used to generate these plots are located in the `results/` and `siliqun/` directories at the repository root, respectively.)*

## Key Findings

### 1. Scalability (Fidelity vs Qubits)
SiliQun demonstrates robust scalability across multiple target states. As shown in the `scalability/` plots, the framework maintains high fidelity (F > 0.99) for GHZ, W, Cluster, and Dicke states up to N=3 qubits, with graceful degradation at higher qubit counts. The `v11_scalability_summary.png` provides a comprehensive view of this scaling behaviour.

### 2. Training Convergence
The `fidelity_vs_episode/` and `ppo_reward_curves/` plots illustrate the convergence speed and stability of the DRL agents. 
- **Consistency:** Training across multiple random seeds (e.g., 42, 123, 456) shows consistent convergence to F > 0.99 for 2Q targets within 10,000 episodes.
- **Reward Shaping:** The correlation between the shaped reward and the final state fidelity is strongly positive, validating the reward function design.

### 3. Algorithm Comparison (Ablation)
The `ablation/` directory contains comparisons between different control algorithms (PPO, SAC, GRAPE/CRAB, GAA). The `algorithm_comparison.png` chart highlights the relative performance of each approach under identical noise profiles and hardware constraints, demonstrating the advantages of the DRL-based approach for specific state preparation tasks.

## Reproducing the Plots

All plots in this directory were generated directly from the raw JSON result files using the `generate_plots.py` script located at the repository root.

To regenerate the plots:

1. Ensure you have the required dependencies installed:
   ```bash
   pip install matplotlib numpy
   ```
2. Run the generation script from the repository root:
   ```bash
   python3 generate_plots.py
   ```
This will read the data from `results/json/` and output fresh PNG files into `experiments/plots/`.

## TensorBoard Integration

For deeper inspection of the training dynamics, the `tensorboard_scalars/` directory contains a `ppo_scalars.csv` file. This file aggregates step-by-step fidelity and reward metrics across all seeds, which can be imported directly into TensorBoard or custom analysis pipelines.
