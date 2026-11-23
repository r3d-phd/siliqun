#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate E6 receipts (results/e6_*.json) into a summary table."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ALGOS = ["dqn", "dqn_nas", "ppo", "ppo_ent", "ppo_hier", "sac", "ddpg"]
FID = 0.99


def load_receipts():
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if not (f.startswith("e6_") and f.endswith(".json")):
            continue
        if "random_baseline" in f or "launch_manifest" in f:
            continue
        with open(os.path.join(RESULTS, f)) as fh:
            d = json.load(fh)
        if "algo" not in d:
            continue
        rows.append(d)
    return rows


def main():
    rows = load_receipts()
    print(f"receipts found: {len(rows)}")
    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algo"], []).append(r)

    hdr = (f"{'algo':10s} {'solved':>7s} {'first_solve (mean)':>20s} "
           f"{'final_F mean':>12s} {'best_F mean':>12s} {'wall_h':>7s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for a in ALGOS:
        rs = by_algo.get(a, [])
        if not rs:
            print(f"{a:10s} {'--':>7s}  (no receipts yet)")
            summary[a] = None
            continue
        solved = [r["first_solve_step"] for r in rs if r["solved"]]
        nsolved = sum(1 for r in rs if r["solved"])
        mean_first = (sum(solved) / len(solved)) if solved else None
        fin = [r["final_F"] for r in rs]
        best = [r["best_F"] for r in rs]
        stable = [r for r in rs if r["final_F"] >= FID]
        wall = sum(r["wall_seconds"] for r in rs) / 3600.0
        # post-solve retention: for solved seeds, fraction of evals after
        # first solve that are still >= threshold
        ret = []
        for r in rs:
            if not r["solved"]:
                continue
            post = [e["F"] for e in r["eval_curve"]
                    if e["step"] >= r["first_solve_step"]]
            if post:
                ret.append(sum(1 for f in post if f >= FID) / len(post))
        mean_ret = (sum(ret) / len(ret)) if ret else None
        ms = f"{mean_first/1000:.1f}k" if mean_first else "n/a"
        mr = f"{mean_ret:.2f}" if mean_ret is not None else "n/a"
        print(f"{a:10s} solve {nsolved}/{len(rs)}  stable {len(stable)}/{len(rs)}  "
              f"first {ms:>7s}  retention {mr:>5s}  "
              f"final {sum(fin)/len(fin):.4f}  best {sum(best)/len(best):.4f}  "
              f"wall {wall:.1f}h")
        summary[a] = {
            "n_receipts": len(rs),
            "solved_count": nsolved,
            "stable_final_count": len(stable),
            "first_solve_steps": solved,
            "first_solve_mean": mean_first,
            "post_solve_retention_mean": mean_ret,
            "final_F_mean": sum(fin) / len(fin),
            "best_F_mean": sum(best) / len(best),
            "wall_hours": wall,
        }
    # random baseline
    rb_path = os.path.join(RESULTS, "e6_random_baseline.json")
    if os.path.exists(rb_path):
        with open(rb_path) as fh:
            rb = json.load(fh)
        print(f"\nrandom baseline: mean best F = {rb['mean_best_F']:.4f} "
              f"(std {rb['std_best_F']:.4f}, max {rb['fidelity_max']:.4f}, "
              f"{rb['n_episodes']} episodes)")
        summary["random_baseline"] = rb
    out = os.path.join(HERE, "E6_SUMMARY.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
