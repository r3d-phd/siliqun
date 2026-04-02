"""
HPC runner for distributed SiliQun simulations.

Provides utilities for running SiliQun on HPC clusters like Aziz:
- PBS job generation and submission
- Multi-GPU distribution
- Checkpoint/restart support
- Parameter sweep orchestration

Usage:
    from siliqun.hpc import HPCRunner, PBSConfig
    runner = HPCRunner(PBSConfig(queue="A100", nodes=1, gpus=1))
    runner.submit_sweep(configs, script="train.py")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os
import logging
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class PBSConfig:
    """PBS job configuration for Aziz HPC.

    Parameters
    ----------
    queue : str
        PBS queue name (e.g., "A100", "V100", "cpu").
    nodes : int
        Number of compute nodes.
    cpus_per_node : int
        CPUs per node.
    gpus : int
        Number of GPUs per node.
    memory_gb : int
        Memory per node in GB.
    walltime : str
        Maximum walltime in HH:MM:SS format.
    job_name : str
        PBS job name.
    conda_env : str
        Conda environment to activate.
    modules : list of str
        Module files to load.
    """
    queue: str = "A100"
    nodes: int = 1
    cpus_per_node: int = 32
    gpus: int = 1
    memory_gb: int = 64
    walltime: str = "24:00:00"
    job_name: str = "siliqun"
    conda_env: str = "mqcf"
    modules: List[str] = field(default_factory=lambda: [
        "cuda/12.2",
        "anaconda3/2024.02",
    ])
    extra_env: Dict[str, str] = field(default_factory=lambda: {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "4",
    })


@dataclass
class CheckpointConfig:
    """Checkpoint configuration for fault-tolerant execution.

    Parameters
    ----------
    checkpoint_dir : str
        Directory for saving checkpoints.
    checkpoint_interval : int
        Save checkpoint every N episodes.
    resume : bool
        Whether to resume from the latest checkpoint.
    """
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    resume: bool = True


class HPCRunner:
    """Orchestrates SiliQun simulations on HPC clusters.

    Parameters
    ----------
    pbs_config : PBSConfig
        PBS job configuration.
    checkpoint_config : CheckpointConfig, optional
        Checkpoint configuration.
    """

    def __init__(
        self,
        pbs_config: PBSConfig,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        self.pbs = pbs_config
        self.ckpt = checkpoint_config or CheckpointConfig()

    def generate_pbs_script(
        self,
        python_script: str,
        args: Dict[str, Any] = None,
        output_dir: str = "./results",
    ) -> str:
        """Generate a PBS submission script.

        Parameters
        ----------
        python_script : str
            Path to the Python training/simulation script.
        args : dict
            Command-line arguments to pass to the script.
        output_dir : str
            Directory for job output logs.

        Returns
        -------
        str
            PBS script content.
        """
        args = args or {}

        # Build argument string
        arg_str = " ".join(
            f"--{k} {v}" for k, v in args.items()
        )

        # Module loads
        module_lines = "\n".join(
            f"module load {m}" for m in self.pbs.modules
        )

        # Environment variables
        env_lines = "\n".join(
            f"export {k}={v}" for k, v in self.pbs.extra_env.items()
        )

        script = f"""#!/bin/bash
#PBS -N {self.pbs.job_name}
#PBS -q {self.pbs.queue}
#PBS -l select={self.pbs.nodes}:ncpus={self.pbs.cpus_per_node}:ngpus={self.pbs.gpus}:mem={self.pbs.memory_gb}gb
#PBS -l walltime={self.pbs.walltime}
#PBS -o {output_dir}/{self.pbs.job_name}.out
#PBS -e {output_dir}/{self.pbs.job_name}.err
#PBS -j oe

echo "=== SiliQun HPC Job ==="
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================"

# Load modules
{module_lines}

# Activate conda environment
source activate {self.pbs.conda_env}

# Set environment variables
{env_lines}

# GPU diagnostics
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Change to working directory
cd $PBS_O_WORKDIR

# Run simulation
python3 {python_script} {arg_str} \\
    --checkpoint_dir {self.ckpt.checkpoint_dir} \\
    --checkpoint_interval {self.ckpt.checkpoint_interval} \\
    {"--resume" if self.ckpt.resume else ""}

EXIT_CODE=$?
echo "========================"
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
exit $EXIT_CODE
"""
        return script

    def generate_sweep_script(
        self,
        python_script: str,
        sweep_configs: List[Dict[str, Any]],
        output_dir: str = "./results",
    ) -> str:
        """Generate a PBS script that runs a parameter sweep.

        Parameters
        ----------
        python_script : str
            Path to the Python script.
        sweep_configs : list of dict
            List of configuration dictionaries for each run.
        output_dir : str
            Directory for results.

        Returns
        -------
        str
            PBS script content.
        """
        # Save sweep configs to JSON
        configs_json = json.dumps(sweep_configs, indent=2)

        # Build the run loop
        run_commands = []
        for i, cfg in enumerate(sweep_configs):
            arg_str = " ".join(
                f"--{k} {v}" for k, v in cfg.items()
            )
            run_commands.append(f"""
echo "=== Run {i+1}/{len(sweep_configs)} ==="
echo "Config: {json.dumps(cfg)}"
RUN_START=$(date +%s)

python3 {python_script} {arg_str} \\
    --checkpoint_dir {self.ckpt.checkpoint_dir}/run_{i} \\
    --checkpoint_interval {self.ckpt.checkpoint_interval} \\
    --run_id {i} \\
    --output_dir {output_dir}/run_{i} \\
    {"--resume" if self.ckpt.resume else ""} 2>&1 || echo "Run {i+1} FAILED"

RUN_END=$(date +%s)
echo "Run {i+1} completed in $((RUN_END - RUN_START)) seconds"
echo ""
""")

        module_lines = "\n".join(
            f"module load {m}" for m in self.pbs.modules
        )
        env_lines = "\n".join(
            f"export {k}={v}" for k, v in self.pbs.extra_env.items()
        )

        script = f"""#!/bin/bash
#PBS -N {self.pbs.job_name}_sweep
#PBS -q {self.pbs.queue}
#PBS -l select={self.pbs.nodes}:ncpus={self.pbs.cpus_per_node}:ngpus={self.pbs.gpus}:mem={self.pbs.memory_gb}gb
#PBS -l walltime={self.pbs.walltime}
#PBS -o {output_dir}/{self.pbs.job_name}_sweep.out
#PBS -e {output_dir}/{self.pbs.job_name}_sweep.err
#PBS -j oe

echo "=== SiliQun Parameter Sweep ==="
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Total runs: {len(sweep_configs)}"
echo "================================"

# Load modules
{module_lines}
source activate {self.pbs.conda_env}
{env_lines}

nvidia-smi
cd $PBS_O_WORKDIR
mkdir -p {output_dir}

# Save sweep configuration
cat > {output_dir}/sweep_config.json << 'SWEEP_EOF'
{configs_json}
SWEEP_EOF

TOTAL_START=$(date +%s)
SUCCESSES=0
FAILURES=0

{"".join(run_commands)}

TOTAL_END=$(date +%s)
echo "================================"
echo "Sweep complete in $((TOTAL_END - TOTAL_START)) seconds"
echo "Successes: $SUCCESSES / {len(sweep_configs)}"
echo "Failures: $FAILURES / {len(sweep_configs)}"
echo "End: $(date)"
"""
        return script

    def write_and_submit(
        self,
        script_content: str,
        script_path: str = "submit.pbs",
        dry_run: bool = False,
    ) -> Optional[str]:
        """Write PBS script and optionally submit it.

        Parameters
        ----------
        script_content : str
            PBS script content.
        script_path : str
            Path to save the script.
        dry_run : bool
            If True, only write the script without submitting.

        Returns
        -------
        str or None
            PBS job ID if submitted, None if dry_run.
        """
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        logger.info(f"PBS script written to {script_path}")

        if dry_run:
            logger.info("Dry run — script not submitted")
            return None

        try:
            result = subprocess.run(
                ["qsub", script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip()
                logger.info(f"Job submitted: {job_id}")
                return job_id
            else:
                logger.error(f"qsub failed: {result.stderr}")
                return None
        except FileNotFoundError:
            logger.warning("qsub not found — not on an HPC cluster")
            return None
        except subprocess.TimeoutExpired:
            logger.error("qsub timed out")
            return None

    def check_job_status(self, job_id: str) -> Dict[str, str]:
        """Check PBS job status.

        Returns
        -------
        dict with keys: state, walltime, queue, node
        """
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return {"state": "unknown", "error": result.stderr}

            info = {}
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "job_state" in line:
                    info["state"] = line.split("=")[-1].strip()
                elif "resources_used.walltime" in line:
                    info["walltime"] = line.split("=")[-1].strip()
                elif "exec_host" in line:
                    info["node"] = line.split("=")[-1].strip()
                elif "queue" in line and "queue" not in info:
                    info["queue"] = line.split("=")[-1].strip()
            return info

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"state": "unavailable"}
