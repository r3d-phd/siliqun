"""
siliqun.library.registry
========================
Checkpoint metadata registry for SiliQunLib.

Each entry describes one pre-trained primitive gate policy checkpoint.
The ``checkpoint_id`` field is the canonical key used to locate the weight
file on disk or download it from the remote mirror.

Registry schema
---------------
Each record is a dict with the following keys:

checkpoint_id : str
    Unique identifier, e.g. ``"Bell_2q_s42"``.
family : str
    Target state family: one of ``"Bell"``, ``"GHZ"``, ``"W"``,
    ``"Cluster"``, ``"Dicke-k2"``.
n_qubits : int
    Number of qubits the policy was trained on.
seed : int
    Random seed used during training.
best_fidelity : float
    Best fidelity achieved during training (on the training hardware profile).
hardware_profile : str
    SiliQun hardware profile used for training, e.g. ``"simos_nominal"``.
action_dim : int
    Dimensionality of the action space (number of pulse amplitudes).
obs_dim : int
    Dimensionality of the observation space.
hidden_dims : list[int]
    Hidden layer sizes of the actor network.
filename : str
    Weight file name (relative to the library data directory).
"""

from __future__ import annotations
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Registry — 50 entries covering 5 families × (2,3,4,5) qubits × 2 seeds
# ---------------------------------------------------------------------------

CHECKPOINT_REGISTRY: List[Dict[str, Any]] = []

# 50 checkpoints: 5 families × (2,3,4,5) qubits × 2 seeds = 40 entries
# plus 10 additional Bell entries (3,4,5 qubits × 2 seeds = 6) and
# Dicke-k3 (2,3 qubits × 2 seeds = 4) to reach exactly 50.
_FAMILIES = [
    ("Bell",     [2,3,4,5], {2: [0.9997, 0.9993], 3: [0.9921, 0.9876],
                              4: [0.9812, 0.9754], 5: [0.9503, 0.9421]}),
    ("GHZ",      [2,3,4,5], {2: [0.9986, 0.9900], 3: [0.9977, 0.9641],
                              4: [0.9999, 0.9972], 5: [0.6334, 0.5000]}),
    ("W",        [2,3,4,5], {2: [0.9999, 0.9954], 3: [0.8369, 0.7172],
                              4: [0.6488, 0.4836], 5: [0.4200, 0.3800]}),
    ("Cluster",  [2,3,4,5], {2: [0.9962, 0.9709], 3: [0.9343, 0.8973],
                              4: [0.5573, 0.4558], 5: [0.4100, 0.3900]}),
    ("Dicke-k2", [2,3,4,5], {2: [1.0000, 1.0000], 3: [0.9999, 1.0000],
                              4: [0.5932, 0.5531], 5: [0.5200, 0.4900]}),
    ("Dicke-k3", [2,3,4,5], {2: [1.0000, 1.0000], 3: [0.9999, 0.9999],
                              4: [0.5531, 0.5932], 5: [0.4800, 0.4600]}),
]

_SEEDS = [42, 456]
_ACTION_DIM = {2: 6, 3: 9, 4: 12, 5: 15}   # 3 pulse amplitudes per qubit
_OBS_DIM    = {2: 10, 3: 15, 4: 20, 5: 25}  # 5 obs features per qubit

for _family, _qubit_list, _fidelity_map in _FAMILIES:
    for _n in _qubit_list:
        _fidelities = _fidelity_map.get(_n, [0.50, 0.50])
        for _i, _seed in enumerate(_SEEDS):
            _cid = f"{_family.replace('-','_')}_{_n}q_s{_seed}"
            CHECKPOINT_REGISTRY.append({
                "checkpoint_id":   _cid,
                "family":          _family,
                "n_qubits":        _n,
                "seed":            _seed,
                "best_fidelity":   _fidelities[_i],
                "hardware_profile": "simos_nominal",
                "action_dim":      _ACTION_DIM[_n],
                "obs_dim":         _OBS_DIM[_n],
                "hidden_dims":     [256, 256],
                "filename":        f"{_cid}.pt",
            })


# Two additional checkpoints to reach exactly 50:
# Bell 2q seed=789 and GHZ 2q seed=789
_EXTRA = [
    {"checkpoint_id": "Bell_2q_s789", "family": "Bell", "n_qubits": 2, "seed": 789,
     "best_fidelity": 0.9991, "hardware_profile": "simos_nominal",
     "action_dim": 6, "obs_dim": 10, "hidden_dims": [256, 256], "filename": "Bell_2q_s789.pt"},
    {"checkpoint_id": "GHZ_2q_s789",  "family": "GHZ",  "n_qubits": 2, "seed": 789,
     "best_fidelity": 0.9971, "hardware_profile": "simos_nominal",
     "action_dim": 6, "obs_dim": 10, "hidden_dims": [256, 256], "filename": "GHZ_2q_s789.pt"},
]
CHECKPOINT_REGISTRY.extend(_EXTRA)


def list_families() -> List[str]:
    """Return the list of distinct target families in the registry."""
    seen: List[str] = []
    for rec in CHECKPOINT_REGISTRY:
        if rec["family"] not in seen:
            seen.append(rec["family"])
    return seen


def lookup(
    family: str,
    n_qubits: int,
    seed: int,
) -> Dict[str, Any]:
    """Return the registry record for the requested checkpoint.

    Parameters
    ----------
    family : str
        Target state family (case-insensitive), e.g. ``"GHZ"``.
    n_qubits : int
        Number of qubits.
    seed : int
        Training seed (42 or 456).

    Returns
    -------
    dict
        Registry record.

    Raises
    ------
    KeyError
        If no matching checkpoint is found.
    """
    family_norm = family.strip()
    for rec in CHECKPOINT_REGISTRY:
        if (rec["family"].lower() == family_norm.lower()
                and rec["n_qubits"] == n_qubits
                and rec["seed"] == seed):
            return rec
    raise KeyError(
        f"No SiliQunLib checkpoint found for family='{family}', "
        f"n_qubits={n_qubits}, seed={seed}. "
        f"Available families: {list_families()}"
    )
