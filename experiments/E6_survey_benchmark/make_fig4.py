#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate fig4_benchmark from E6 receipts (results/e6_*.json).

Produces two panels matching the survey Figure 4 layout:
  (a) sample efficiency — first step with held-out F >= 0.99 (log y),
      mean over seeds with min--max whiskers; unresolved algorithms are
      shown at the budget limit with hatching;
  (b) final held-out F at end of budget (mean +/- std over seeds).

Output: results/fig4_benchmark.png (and .pdf).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ALGOS = ["dqn", "dqn_nas", "ppo", "ppo_ent", "ppo_hier", "sac", "ddpg"]
LABELS = {
    "dqn": "DQN",
    "dqn_nas": "DQN+NAS",
    "ppo": "PPO",
    "ppo_ent": "PPO+ent",
    "ppo_hier": "PPO-hier",
    "sac": "SAC",
    "ddpg": "DDPG",
}
BUDGET = 500_000
THRESH = 0.99


def load():
    data = {a: [] for a in ALGOS}
    for f in os.listdir(RESULTS):
        if not (f.startswith("e6_") and f.endswith(".json")) or "random" in f or "launch_manifest" in f:
            continue
        with open(os.path.join(RESULTS, f)) as fh:
            r = json.load(fh)
        if "algo" in r and r["algo"] in data:
            data[r["algo"]].append(r)
    return data


def main():
    data = load()
    n_found = {a: len(v) for a, v in data.items()}
    print("receipts per algo:", n_found)
    if sum(n_found.values()) < 35:
        print("INCOMPLETE — refusing to plot (need 35/35)")
        return

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- (a) sample efficiency ----
    means, errs, unresolved = [], [], []
    for a in ALGOS:
        rs = data[a]
        solved = [r["first_solve_step"] for r in rs if r["solved"]]
        if len(solved) < len(rs):
            unresolved.append(a)
            v = solved + [BUDGET] * (len(rs) - len(solved))
        else:
            v = solved
        v = np.array(v, dtype=float)
        means.append(v.mean())
        errs.append((v.mean() - v.min(), v.max() - v.mean()))
        print(f"{a:10s} first-solve mean {v.mean():10.0f}  "
              f"(min {v.min():.0f}, max {v.max():.0f}, n={len(v)})")
    x = np.arange(len(ALGOS))
    yerr = np.array(errs).T  # (2, n)
    bars = ax1.bar(x, means, yerr=yerr, capsize=3, color="#4878a8",
                   edgecolor="k", linewidth=0.5)
    for a in unresolved:
        bars[ALGOS.index(a)].set_hatch("//")
        bars[ALGOS.index(a)].set_alpha(0.55)
    ax1.set_yscale("log")
    ax1.set_ylim(100, BUDGET * 3)
    ax1.set_ylabel("Steps to first held-out F $\\geq$ 0.99")
    ax1.set_title("(a) Sample efficiency")
    ax1.set_xticks(x, [LABELS[a] for a in ALGOS], rotation=30, ha="right")
    ax1.axhline(BUDGET, color="r", lw=0.8, ls=":")
    ax1.text(len(ALGOS) - 0.4, BUDGET * 1.15, "budget", color="r",
             fontsize=7, ha="right")

    # ---- (b) final fidelity ----
    fin_means, fin_stds = [], []
    for a in ALGOS:
        rs = data[a]
        fin = np.array([r["final_F"] for r in rs], dtype=float)
        fin_means.append(fin.mean())
        fin_stds.append(fin.std(ddof=0))
        print(f"{a:10s} final F mean {fin.mean():.4f} std {fin.std(ddof=0):.4f}")
    ax2.bar(x, fin_means, yerr=np.array([fin_stds, fin_stds]), capsize=3,
            color="#6aa84f", edgecolor="k", linewidth=0.5)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Final held-out fidelity")
    ax2.set_title("(b) Final fidelity")
    ax2.set_xticks(x, [LABELS[a] for a in ALGOS], rotation=30, ha="right")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax2.text(len(ALGOS) - 0.4, 0.515, "maximally mixed", color="gray",
             fontsize=7, ha="right")

    fig.tight_layout()
    png = os.path.join(RESULTS, "fig4_benchmark.png")
    fig.savefig(png)
    fig.savefig(os.path.join(RESULTS, "fig4_benchmark.pdf"))
    print("wrote", png)


if __name__ == "__main__":
    main()
