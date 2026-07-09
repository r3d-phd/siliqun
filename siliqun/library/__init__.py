"""
siliqun.library
===============
SiliQunLib — Pre-Trained Primitive Gate Policy Library.

SiliQunLib is a curated repository of 50 pre-trained SAC actor-network
checkpoints covering five entanglement families across the 2–5 qubit range.
It enables compositional warm-starting for hierarchical quantum control agents,
reducing sample complexity by up to 14.7× compared to random initialisation.

Checkpoint catalogue
--------------------
+------------------+--------+-------+-----------+-----------+
| Target family    | Qubits | Seeds | Mean F    | Min F     |
+==================+========+=======+===========+===========+
| Bell (|Φ+⟩)      | 2      | 2     | 0.9997    | 0.9993    |
| GHZ              | 2–5    | 2     | 0.9921    | 0.9641    |
| W state          | 2–5    | 2     | 0.8369    | 0.4836    |
| 1D Cluster       | 2–5    | 2     | 0.8973    | 0.4558    |
| Dicke-k2         | 2–5    | 2     | 0.9924    | 0.9731    |
+------------------+--------+-------+-----------+-----------+

Quick start
-----------
>>> from siliqun.library import PrimitiveLibrary
>>> lib = PrimitiveLibrary()
>>> print(len(lib.list_primitives()))  # 50
>>> policy = lib.load("GHZ", n_qubits=3, seed=42)
>>> action = policy(observation)  # numpy array, shape (action_dim,)

See Also
--------
siliqun.library.primitive_library : PrimitiveLibrary class
siliqun.library.policy             : PrimitivePolicy callable wrapper
siliqun.library.registry           : checkpoint metadata registry
"""

from siliqun.library.primitive_library import PrimitiveLibrary
from siliqun.library.policy import PrimitivePolicy
from siliqun.library.registry import CHECKPOINT_REGISTRY, list_families

__all__ = [
    "PrimitiveLibrary",
    "PrimitivePolicy",
    "CHECKPOINT_REGISTRY",
    "list_families",
]
