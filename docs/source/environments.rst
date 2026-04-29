Environments
============

SiliQun provides a single unified environment class, ``SiliQunEnv``, which wraps the quantum simulation backends and exposes the standard Gymnasium API.

SiliQunEnv
----------

.. autoclass:: siliqun.SiliQunEnv
   :members:
   :undoc-members:
   :show-inheritance:

Observation Space
-----------------

The observation vector contains:

- Real and imaginary parts of the density matrix diagonal elements (2 × 2^n values)
- Current step index (normalised to [0, 1])
- Cumulative fidelity (scalar)
- Device noise level indicator (scalar)

Action Space
------------

The action space is a continuous ``Box(-1, 1, shape=(n_actions,))`` where each dimension controls a gate parameter:

- Dimensions 0–2: Rotation angles (θ_x, θ_y, θ_z) for each qubit
- Dimensions 3–5: CNOT coupling strengths for nearest-neighbour pairs
- Dimensions 6–7: DFS leakage correction pulses (when ``use_dfs=True``)

Reward Function
---------------

The default reward at each step is:

.. math::

   r_t = F(\rho_t, \rho_{\text{target}}) - F(\rho_{t-1}, \rho_{\text{target}}) - \lambda \cdot N_{\text{gates}}

where :math:`F` is the quantum state fidelity, :math:`\rho_{\text{target}}` is the target state, and :math:`\lambda` is a gate-count penalty coefficient (default: 0.01).

Episode Termination
-------------------

An episode terminates when:

- The fidelity exceeds the success threshold (default: 0.99)
- The maximum number of steps is reached (default: 20)
- A decoherence event causes irreversible state collapse (when ``strict_decoherence=True``)
