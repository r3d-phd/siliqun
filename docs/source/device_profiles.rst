Device Profiles
===============

SiliQun includes four physically calibrated silicon spin qubit device profiles. Each profile specifies the T1/T2* coherence times, charge noise spectrum, gate fidelities, and crosstalk parameters based on published experimental data.

Donor (P-in-Si)
---------------

Based on Muhonen et al. (2014) and Laucht et al. (2017).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - T1
     - 3.0 s
     - Longitudinal relaxation time
   * - T2*
     - 120 μs
     - Dephasing time (free induction decay)
   * - T2_echo
     - 1.0 ms
     - Echo coherence time
   * - Gate fidelity (1Q)
     - 99.6%
     - Single-qubit gate fidelity
   * - Charge noise (A)
     - 1.0 × 10⁻⁴ eV/√Hz
     - 1/f charge noise amplitude

SiMOS (Silicon MOS)
-------------------

Based on Veldhorst et al. (2015), Huang et al. (2019), and Noiri et al. (2022).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - T1
     - 100 ms
     - Longitudinal relaxation time
   * - T2*
     - 20 μs
     - Dephasing time
   * - T2_echo
     - 120 μs
     - Echo coherence time
   * - Gate fidelity (1Q)
     - 99.9%
     - Single-qubit gate fidelity
   * - Gate fidelity (2Q)
     - 98.0%
     - Two-qubit CNOT fidelity

GAA (Gate-All-Around Nanowire)
------------------------------

Based on Zwerver et al. (2022) and Philips et al. (2022).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - T1
     - 50 ms
     - Longitudinal relaxation time
   * - T2*
     - 10 μs
     - Dephasing time
   * - Gate fidelity (1Q)
     - 99.5%
     - Single-qubit gate fidelity

RIKEN 5Q
--------

Based on Noiri et al. (2022) — 5-qubit linear chain with nearest-neighbour exchange coupling.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - T1
     - 80 ms
     - Longitudinal relaxation time
   * - T2*
     - 15 μs
     - Dephasing time
   * - Gate fidelity (2Q)
     - 99.5%
     - Two-qubit CZ fidelity
   * - Topology
     - Linear chain
     - 5 qubits, nearest-neighbour coupling
