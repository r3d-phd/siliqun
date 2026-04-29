DFS Logical Encoding
====================

SiliQun implements Decoherence-Free Subspace (DFS) encoding to protect logical qubits from collective dephasing noise. Physical spin triplets are mapped to logical qubits within the DFS subspace, with perturbative leakage tracking.

Enabling DFS Encoding
---------------------

.. code-block:: python

   env = siliqun.SiliQunEnv(n_qubits=6, use_dfs=True, topology="2x3")

DFS Basis Construction
----------------------

The DFS basis is constructed from the degenerate subspace of the collective dephasing operator. For a triplet of spin-1/2 particles, the logical states are:

.. math::

   |0_L\rangle = \frac{1}{\sqrt{2}}(|\uparrow\downarrow\uparrow\rangle - |\downarrow\uparrow\uparrow\rangle)

   |1_L\rangle = \frac{1}{\sqrt{6}}(2|\uparrow\uparrow\downarrow\rangle - |\uparrow\downarrow\uparrow\rangle - |\downarrow\uparrow\uparrow\rangle)

Leakage Tracking
----------------

SiliQun tracks perturbative leakage out of the DFS subspace at each timestep. The leakage probability is included in the ``info`` dictionary returned by ``env.step()``.
