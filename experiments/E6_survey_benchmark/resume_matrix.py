#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume the E6 matrix: run only the missing (algo, seed) receipts.

Spawns one worker per missing seed for the missing algos (SAC + DDPG),
each worker running its SAC then DDPG run sequentially. Waits for all,
writes results/e6_launch_manifest_resume.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:/Users/raada/venvs/survey_bench/Scripts/python"
RESULTS = os.path.join(HERE, "results")
SEEDS = [0, 1, 2, 3, 4]
MISSING_ALGOS = ["sac", "ddpg"]  # the slow pair; verify below

# determine actually-missing jobs from disk
jobs = []
for algo in ["dqn", "dqn_nas", "ppo", "ppo_ent", "ppo_hier", "sac", "ddpg"]:
    for s in SEEDS:
        if not os.path.exists(os.path.join(RESULTS, f"e6_{algo}_s{s}.json")):
            jobs.append((algo, s))
print(f"missing jobs: {jobs}", flush=True)
if not jobs:
    print("nothing to do", flush=True)
    sys.exit(0)

env = dict(os.environ)
env["TORCH_THREADS"] = "2"
env["OMP_NUM_THREADS"] = "2"
env["E6_DEVICE"] = "cpu"

# one worker per seed: SAC(seed) then DDPG(seed)
t0 = time.time()
procs = []
for s in SEEDS:
    mine = [a for a, ss in jobs if ss == s]
    if not mine:
        continue
    log = os.path.join(RESULTS, "logs", f"e6_resume_s{s}.log")
    fh = open(log, "wb")
    cmd = [PY, os.path.join(HERE, "run_e6.py")]
    for a in mine:
        cmd += ["--algo", a, "--seed", str(s)]
    # run_e6 main only accepts one --algo/--seed pair; loop via shell-free
    # approach: invoke once per algo sequentially in a small wrapper
    wrapper = f'''
import subprocess, sys
sys.exit(sum(subprocess.call([sys.executable, {os.path.join(HERE, "run_e6.py")!r},
    "--algo", a, "--seed", "{s}"]) for a in {mine!r}) != 0)
'''
    wp = os.path.join(RESULTS, "logs", f"e6_resume_s{s}.py")
    with open(wp, "w") as wf:
        wf.write(wrapper)
    p = subprocess.Popen([PY, wp], cwd=HERE, env=env,
                         stdout=fh, stderr=subprocess.STDOUT)
    procs.append((s, p, fh))
    print(f"launched seed {s} ({'+'.join(mine)}) pid {p.pid}", flush=True)

results = []
for s, p, fh in procs:
    rc = p.wait()
    fh.close()
    results.append({"seed": s, "pid": p.pid, "exit_code": rc})
    print(f"seed {s} pid {p.pid} exit {rc}", flush=True)

wall = time.time() - t0
ok = all(r["exit_code"] == 0 for r in results)
manifest = {
    "launched": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)),
    "wall_seconds": wall,
    "jobs": jobs,
    "all_ok": ok,
    "results": results,
}
with open(os.path.join(RESULTS, "e6_launch_manifest_resume.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2), flush=True)
sys.exit(0 if ok else 1)
