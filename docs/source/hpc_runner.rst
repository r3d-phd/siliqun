HPC Runner
==========

SiliQun includes utilities for generating and submitting job scripts on HPC systems.

Generating PBS Job Scripts
--------------------------

.. code-block:: python

   from siliqun.hpc_runner import HPCRunner

   runner = HPCRunner(scheduler="pbs")
   script = runner.generate_job_script(
       job_name="siliqun_ghz3",
       n_cpus=32,
       memory_gb=64,
       walltime_hours=12,
       script_path="train_ghz3.py",
       conda_env="quantum_drl_gpu",
   )
   runner.submit(script)

Generating SLURM Job Scripts
-----------------------------

.. code-block:: python

   runner = HPCRunner(scheduler="slurm")
   script = runner.generate_job_script(
       job_name="siliqun_ghz5",
       n_cpus=32,
       memory_gb=128,
       walltime_hours=24,
       script_path="train_ghz5.py",
       partition="gpu",
   )
