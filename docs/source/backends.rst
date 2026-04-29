Simulation Backends
===================

SiliQun provides two simulation backends that can be selected automatically or manually.

MPS Backend (Tensor Network)
-----------------------------

The MPS (Matrix Product State) backend uses tensor network contraction for approximate simulation. It is recommended for systems with 4 or fewer qubits where memory efficiency is prioritised.

.. code-block:: python

   env = siliqun.SiliQunEnv(n_qubits=4, backend="mps")

GPU State Vector Backend
------------------------

The GPU state vector backend performs exact simulation using CuPy for CUDA-accelerated matrix operations. It is recommended for systems with 5 or more qubits where a GPU is available.

.. code-block:: python

   env = siliqun.SiliQunEnv(n_qubits=9, backend="statevector_gpu")

Automatic Selection
-------------------

The ``"auto"`` mode selects the MPS backend for systems with 4 or fewer qubits and the GPU state vector backend for larger systems.

.. code-block:: python

   env = siliqun.SiliQunEnv(n_qubits=9, backend="auto")
