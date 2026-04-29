Installation
============

Requirements
------------

- Python 3.9 or later
- NumPy 1.24 or later
- SciPy 1.10 or later
- Gymnasium 0.29 or later

Optional (for GPU acceleration):

- CUDA 11.x or 12.x
- CuPy (``cupy-cuda11x`` or ``cupy-cuda12x``)

Basic Installation (CPU only)
------------------------------

.. code-block:: bash

   git clone https://github.com/r3d-phd/siliqun.git
   cd siliqun
   pip install -e .

GPU Installation (recommended for 16+ qubits)
----------------------------------------------

For NVIDIA GPUs with CUDA 12.x:

.. code-block:: bash

   pip install cupy-cuda12x
   git clone https://github.com/r3d-phd/siliqun.git
   cd siliqun
   pip install -e .

For CUDA 11.x:

.. code-block:: bash

   pip install cupy-cuda11x
   git clone https://github.com/r3d-phd/siliqun.git
   cd siliqun
   pip install -e .

Verifying the Installation
---------------------------

.. code-block:: python

   import siliqun
   env = siliqun.SiliQunEnv(n_qubits=3, device="SiMOS")
   obs, info = env.reset()
   print(f"Observation shape: {obs.shape}")
   print(f"Action space: {env.action_space}")

HPC Installation (Aziz Supercomputer / SLURM / PBS)
-----------------------------------------------------

On HPC systems, load the appropriate modules before installation:

.. code-block:: bash

   module load python/3.10
   module load cuda/12.0
   pip install --user cupy-cuda12x
   pip install --user -e .

See :doc:`hpc_runner` for automated job generation utilities.
