"""
Program Database for AlphaEvolve.

Implements a MAP-Elites-inspired archive combined with an island-based
population model, following the AlphaEvolve paper (Novikov et al., 2025).

The database serves two purposes:
  1. **Exploitation** -- resurface the highest-scoring programs so the LLM
     can refine them further.
  2. **Exploration** -- maintain behavioural diversity so the search does not
     collapse to a single local optimum.

Behavioural features used for MAP-Elites binning:
  - gate_count   : total number of gates in the sequence (proxy for circuit depth)
  - corr_usage   : whether the strategy uses correlation parameters (0 or 1)
  - fidelity_tier: discretised fidelity bucket (low / mid / high)

Each island is an independent MAP-Elites grid.  Periodically the best
programs migrate between islands to share discoveries.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aedb.program_db")


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Program:
    """A single program stored in the database."""

    code: str
    scores: Dict[str, float] = field(default_factory=dict)
    primary_score: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    model: str = ""
    source: str = "seed"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Deterministic unique identifier based on code content."""
        return hashlib.md5(self.code.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "uid": self.uid,
            "code": self.code,
            "scores": self.scores,
            "primary_score": self.primary_score,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "model": self.model,
            "source": self.source,
        }


# ──────────────────────────────────────────────────────────────────────
# Feature extraction for MAP-Elites binning
# ──────────────────────────────────────────────────────────────────────

def _extract_features(program: Program) -> Tuple[int, int, int]:
    """Extract behavioural features from a program for MAP-Elites binning.

    Returns a tuple (gate_count_bin, corr_usage, fidelity_tier) that
    serves as the cell key in the MAP-Elites grid.

    Parameters
    ----------
    program : Program
        The program to extract features from.

    Returns
    -------
    tuple of int
        (gate_count_bin, corr_usage, fidelity_tier)
    """
    code = program.code

    # Feature 1: gate count bin (0-2, 3-5, 6-10, 11+)
    gate_keywords = [
        "Gate(", "gates.append", ".append(Gate",
    ]
    gate_count = sum(code.count(kw) for kw in gate_keywords)
    if gate_count <= 2:
        gate_bin = 0
    elif gate_count <= 5:
        gate_bin = 1
    elif gate_count <= 10:
        gate_bin = 2
    else:
        gate_bin = 3

    # Feature 2: correlation usage (binary)
    corr_keywords = [
        "nn_correlation", "corr_length", "qubit_spacing",
        "correlation", "spacing",
    ]
    corr_usage = 1 if any(kw in code for kw in corr_keywords) else 0

    # Feature 3: fidelity tier
    fid = program.primary_score
    if fid < 0.3:
        fid_tier = 0
    elif fid < 0.6:
        fid_tier = 1
    elif fid < 0.85:
        fid_tier = 2
    else:
        fid_tier = 3

    return (gate_bin, corr_usage, fid_tier)


# ──────────────────────────────────────────────────────────────────────
# MAP-Elites Island
# ──────────────────────────────────────────────────────────────────────

class _Island:
    """A single MAP-Elites island.

    Each cell in the grid stores up to ``cell_capacity`` programs,
    ranked by primary score.
    """

    def __init__(self, island_id: int, cell_capacity: int = 3):
        self.island_id = island_id
        self.cell_capacity = cell_capacity
        self.grid: Dict[Tuple[int, int, int], List[Program]] = {}
        self.all_programs: List[Program] = []

    @property
    def size(self) -> int:
        return len(self.all_programs)

    @property
    def best_program(self) -> Optional[Program]:
        if not self.all_programs:
            return None
        return max(self.all_programs, key=lambda p: p.primary_score)

    def add(self, program: Program) -> bool:
        """Add a program to the island.

        Returns True if the program was accepted (either into an empty
        cell or by displacing a weaker program).
        """
        features = _extract_features(program)
        cell = self.grid.setdefault(features, [])

        # Check for duplicates
        for existing in cell:
            if existing.uid == program.uid:
                return False

        if len(cell) < self.cell_capacity:
            cell.append(program)
            self.all_programs.append(program)
            return True

        # Replace the weakest if the new program is better
        weakest = min(cell, key=lambda p: p.primary_score)
        if program.primary_score > weakest.primary_score:
            cell.remove(weakest)
            self.all_programs.remove(weakest)
            cell.append(program)
            self.all_programs.append(program)
            return True

        return False

    def sample_parent(self, rng: np.random.RandomState) -> Optional[Program]:
        """Sample a parent program using fitness-proportionate selection."""
        if not self.all_programs:
            return None

        scores = np.array(
            [max(p.primary_score, 0.01) for p in self.all_programs]
        )
        probs = scores / scores.sum()
        idx = rng.choice(len(self.all_programs), p=probs)
        return self.all_programs[idx]

    def sample_inspirations(
        self,
        rng: np.random.RandomState,
        n: int = 2,
        exclude_uid: Optional[str] = None,
    ) -> List[Program]:
        """Sample diverse inspiration programs.

        Prefers programs from different MAP-Elites cells to maximise
        behavioural diversity.
        """
        candidates = [
            p for p in self.all_programs
            if p.uid != exclude_uid
        ]
        if not candidates:
            return []

        # Try to pick from different cells for diversity
        cells = list(self.grid.values())
        rng.shuffle(cells)

        inspirations = []
        seen_uids = set()
        for cell in cells:
            for p in sorted(cell, key=lambda x: x.primary_score, reverse=True):
                if p.uid != exclude_uid and p.uid not in seen_uids:
                    inspirations.append(p)
                    seen_uids.add(p.uid)
                    if len(inspirations) >= n:
                        return inspirations
                    break  # one per cell for diversity

        # Fill remaining from all programs
        remaining = [
            p for p in candidates if p.uid not in seen_uids
        ]
        if remaining:
            rng.shuffle(remaining)
            inspirations.extend(remaining[: n - len(inspirations)])

        return inspirations[:n]

    def get_top_k(self, k: int = 5) -> List[Program]:
        """Return the top-k programs by primary score."""
        return sorted(
            self.all_programs,
            key=lambda p: p.primary_score,
            reverse=True,
        )[:k]


# ──────────────────────────────────────────────────────────────────────
# Program Database (multi-island MAP-Elites)
# ──────────────────────────────────────────────────────────────────────

class ProgramDatabase:
    """Multi-island MAP-Elites program database.

    Parameters
    ----------
    n_islands : int
        Number of independent islands.
    cell_capacity : int
        Maximum programs per MAP-Elites cell.
    migration_interval : int
        Migrate top programs between islands every N additions.
    migration_count : int
        Number of top programs to migrate per interval.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_islands: int = 3,
        cell_capacity: int = 3,
        migration_interval: int = 10,
        migration_count: int = 2,
        seed: int = 42,
    ):
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.rng = np.random.RandomState(seed)

        self.islands = [
            _Island(i, cell_capacity) for i in range(n_islands)
        ]
        self._add_counter = 0
        self._best_ever: Optional[Program] = None

    @property
    def total_programs(self) -> int:
        return sum(isl.size for isl in self.islands)

    @property
    def best_program(self) -> Optional[Program]:
        return self._best_ever

    def add(self, program: Program, island_id: Optional[int] = None) -> bool:
        """Add a program to the database.

        If ``island_id`` is None, the program is assigned to the island
        with the fewest programs (load balancing).
        """
        if island_id is None:
            island_id = min(
                range(self.n_islands),
                key=lambda i: self.islands[i].size,
            )

        accepted = self.islands[island_id].add(program)

        if accepted:
            self._add_counter += 1

            # Update global best
            if (
                self._best_ever is None
                or program.primary_score > self._best_ever.primary_score
            ):
                self._best_ever = program
                logger.info(
                    f"New global best: {program.primary_score:.4f} "
                    f"(island={island_id}, gen={program.generation})"
                )

            # Periodic migration
            if self._add_counter % self.migration_interval == 0:
                self._migrate()

        return accepted

    def _migrate(self):
        """Migrate top programs between islands (ring topology)."""
        for i in range(self.n_islands):
            src = self.islands[i]
            dst = self.islands[(i + 1) % self.n_islands]

            migrants = src.get_top_k(self.migration_count)
            for prog in migrants:
                migrant = copy.deepcopy(prog)
                migrant.metadata["migrated_from"] = i
                dst.add(migrant)

        logger.debug(
            f"Migration complete. Sizes: "
            + ", ".join(
                f"island_{i}={self.islands[i].size}"
                for i in range(self.n_islands)
            )
        )

    def sample(
        self,
        n_inspirations: int = 2,
    ) -> Tuple[Program, List[Program], int]:
        """Sample a parent and inspiration programs for the next mutation.

        Returns
        -------
        parent : Program
            The parent program to mutate.
        inspirations : list of Program
            Diverse inspiration programs for the prompt.
        island_id : int
            The island the parent was sampled from.
        """
        # Pick a random island (weighted by size)
        sizes = np.array([max(isl.size, 1) for isl in self.islands], dtype=float)
        probs = sizes / sizes.sum()
        island_id = int(self.rng.choice(self.n_islands, p=probs))
        island = self.islands[island_id]

        parent = island.sample_parent(self.rng)
        if parent is None:
            # Fallback: try other islands
            for isl in self.islands:
                parent = isl.sample_parent(self.rng)
                if parent is not None:
                    break

        if parent is None:
            raise RuntimeError("Database is empty, cannot sample")

        # Sample inspirations from ALL islands for cross-pollination
        all_programs = []
        for isl in self.islands:
            all_programs.extend(isl.all_programs)

        # Deduplicate and exclude parent
        seen = {parent.uid}
        unique = []
        for p in all_programs:
            if p.uid not in seen:
                unique.append(p)
                seen.add(p.uid)

        # Prefer diverse, high-scoring inspirations
        if unique:
            unique.sort(key=lambda p: p.primary_score, reverse=True)
            # Take from different feature bins
            inspirations = []
            seen_features = set()
            for p in unique:
                feat = _extract_features(p)
                if feat not in seen_features:
                    inspirations.append(p)
                    seen_features.add(feat)
                    if len(inspirations) >= n_inspirations:
                        break

            # Fill remaining
            if len(inspirations) < n_inspirations:
                remaining = [p for p in unique if p not in inspirations]
                inspirations.extend(remaining[: n_inspirations - len(inspirations)])
        else:
            inspirations = []

        return parent, inspirations, island_id

    def get_all_programs(self) -> List[Program]:
        """Return all programs across all islands."""
        programs = []
        for isl in self.islands:
            programs.extend(isl.all_programs)
        return programs

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        all_progs = self.get_all_programs()
        if not all_progs:
            return {"total": 0}

        scores = [p.primary_score for p in all_progs]
        return {
            "total": len(all_progs),
            "best": max(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "islands": [isl.size for isl in self.islands],
            "cells": sum(
                len(isl.grid) for isl in self.islands
            ),
        }
