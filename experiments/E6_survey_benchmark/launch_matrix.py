#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the 35-run E6 matrix as 10 parallel chunk workers.

Each worker runs `python run_e6.py --chunk i/10` with pinned thread
counts; logs stream to results/logs/e6_chunk_i.log. The launcher waits
for all workers and reports per-worker exit codes, then writes a
launch manifest. Exits 0 only if every worker exited 0.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:/Users/raada/venvs/survey_bench/Scripts/python"
N = 10
t0 = time.time()

env = dict(os.environ)
env["TORCH_THREADS"] = "2"
env["OMP_NUM_THREADS"] = "2"
env["E6_DEVICE"] = "cpu"

procs = []
for i in range(N):
    log = os.path.join(HERE, "results", "logs", f"e6_chunk_{i}.log")
    fh = open(log, "wb")
    p = subprocess.Popen(
        [PY, os.path.join(HERE, "run_e6.py"), "--chunk", f"{i}/{N}"],
        cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT,
    )
    procs.append((i, p, fh))
    print(f"launched chunk {i}/{N} pid {p.pid}", flush=True)

results = []
for i, p, fh in procs:
    rc = p.wait()
    fh.close()
    results.append({"chunk": i, "pid": p.pid, "exit_code": rc})
    print(f"chunk {i}/{N} pid {p.pid} exit {rc}", flush=True)

wall = time.time() - t0
ok = all(r["exit_code"] == 0 for r in results)
manifest = {
    "launched": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)),
    "wall_seconds": wall,
    "workers": N,
    "all_ok": ok,
    "results": results,
}
with open(os.path.join(HERE, "results", "e6_launch_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2), flush=True)
sys.exit(0 if ok else 1)
