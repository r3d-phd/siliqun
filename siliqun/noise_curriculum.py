"""
Noise Curriculum for Progressive Quantum Control Training.

This module implements the five-stage noise curriculum used in the QUASAR
training pipeline. It provides a standardised interface for gradually
increasing noise during DRL training, enabling stable convergence from
near-ideal to physically realistic noise levels.

The curriculum is extracted from QUASAR v26/v27 and is independent of the
DRL algorithm — it can be used with any Gymnasium-compatible training loop.

Noise Stages
------------
Stage 1 (ns=1) : Near-ideal — depolarising probability p ≈ 0.001
Stage 2 (ns=2) : Low noise   — p ≈ 0.005
Stage 3 (ns=3) : Medium noise — p ≈ 0.010
Stage 4 (ns=4) : High noise  — p ≈ 0.020
Stage 5 (ns=5) : Realistic   — p ≈ 0.050 (calibrated to Weinstein 2023)

References
----------
Weinstein et al., Nature 615, 817-822 (2023).
Al-Shehri, R. (2025). QUASAR: Quantum Universal Adaptive State-preparation
    via Reinforcement learning. PhD Thesis, King Abdulaziz University.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = [
    "NoiseCurriculum",
    "get_noise_prob",
    "NOISE_STAGE_PARAMS",
]

# ---------------------------------------------------------------------------
# Stage parameter table (calibrated from QUASAR v26/v27 experiments)
# ---------------------------------------------------------------------------

NOISE_STAGE_PARAMS: Dict[int, Dict] = {
    1: {
        "noise_prob":          0.001,
        "t1_scale":            100.0,   # T1 multiplier relative to stage 5
        "t2_scale":            100.0,   # T2* multiplier relative to stage 5
        "charge_noise_scale":  0.01,
        "description":         "Near-ideal (p=0.001) — warm-up phase",
    },
    2: {
        "noise_prob":          0.005,
        "t1_scale":            20.0,
        "t2_scale":            20.0,
        "charge_noise_scale":  0.05,
        "description":         "Low noise (p=0.005)",
    },
    3: {
        "noise_prob":          0.010,
        "t1_scale":            10.0,
        "t2_scale":            10.0,
        "charge_noise_scale":  0.10,
        "description":         "Medium noise (p=0.010)",
    },
    4: {
        "noise_prob":          0.020,
        "t1_scale":            5.0,
        "t2_scale":            5.0,
        "charge_noise_scale":  0.20,
        "description":         "High noise (p=0.020)",
    },
    5: {
        "noise_prob":          0.050,
        "t1_scale":            1.0,
        "t2_scale":            1.0,
        "charge_noise_scale":  1.00,
        "description":         "Realistic (p=0.050, Weinstein 2023 calibration)",
    },
}


def get_noise_prob(noise_stage: int) -> float:
    """Return the depolarising probability for a given noise stage.

    Parameters
    ----------
    noise_stage : int
        Noise stage in [1, 5].

    Returns
    -------
    float
        Depolarising probability p.

    Raises
    ------
    ValueError
        If noise_stage is not in [1, 5].

    Examples
    --------
    >>> get_noise_prob(1)
    0.001
    >>> get_noise_prob(5)
    0.05
    """
    if noise_stage not in NOISE_STAGE_PARAMS:
        raise ValueError(
            f"noise_stage must be in [1, 5], got {noise_stage}"
        )
    return NOISE_STAGE_PARAMS[noise_stage]["noise_prob"]


@dataclass
class NoiseCurriculum:
    """Progressive noise curriculum controller.

    Manages automatic advancement through noise stages based on a
    fidelity threshold criterion. The agent must achieve a mean fidelity
    above ``advance_threshold`` over the last ``window`` episodes before
    the curriculum advances to the next stage.

    Parameters
    ----------
    initial_stage : int
        Starting noise stage (default 1).
    advance_threshold : float
        Mean fidelity required to advance to the next stage (default 0.90).
    window : int
        Number of recent episodes used to compute the rolling mean fidelity
        (default 100).
    max_stage : int
        Maximum noise stage (default 5).
    auto_advance : bool
        Whether to automatically advance stages (default True).
        Set to False for fixed-stage training (e.g., ablation studies).

    Examples
    --------
    >>> curriculum = NoiseCurriculum(initial_stage=1, advance_threshold=0.90)
    >>> curriculum.current_stage
    1
    >>> curriculum.noise_prob
    0.001
    >>> # Simulate training loop
    >>> for ep in range(200):
    ...     fidelity = 0.95  # hypothetical high fidelity
    ...     advanced = curriculum.step(fidelity)
    >>> curriculum.current_stage
    2
    """

    initial_stage: int = 1
    advance_threshold: float = 0.90
    window: int = 100
    max_stage: int = 5
    auto_advance: bool = True

    # Internal state (not part of constructor)
    _current_stage: int = field(init=False, repr=False)
    _fidelity_history: list = field(init=False, repr=False)
    _total_advances: int = field(init=False, repr=False)

    def __post_init__(self):
        if self.initial_stage not in NOISE_STAGE_PARAMS:
            raise ValueError(
                f"initial_stage must be in [1, 5], got {self.initial_stage}"
            )
        self._current_stage = self.initial_stage
        self._fidelity_history = []
        self._total_advances = 0

    @property
    def current_stage(self) -> int:
        """Current noise stage."""
        return self._current_stage

    @property
    def noise_prob(self) -> float:
        """Depolarising probability for the current stage."""
        return NOISE_STAGE_PARAMS[self._current_stage]["noise_prob"]

    @property
    def params(self) -> Dict:
        """Full parameter dictionary for the current stage."""
        return NOISE_STAGE_PARAMS[self._current_stage]

    @property
    def is_final_stage(self) -> bool:
        """True if the curriculum is at the maximum noise stage."""
        return self._current_stage >= self.max_stage

    @property
    def rolling_mean_fidelity(self) -> Optional[float]:
        """Rolling mean fidelity over the last ``window`` episodes."""
        if not self._fidelity_history:
            return None
        hist = self._fidelity_history[-self.window:]
        return sum(hist) / len(hist)

    def step(self, episode_fidelity: float) -> bool:
        """Record an episode fidelity and advance the stage if warranted.

        Parameters
        ----------
        episode_fidelity : float
            Best fidelity achieved in the most recent episode.

        Returns
        -------
        bool
            True if the curriculum advanced to the next stage.
        """
        self._fidelity_history.append(float(episode_fidelity))

        if not self.auto_advance or self.is_final_stage:
            return False

        if len(self._fidelity_history) < self.window:
            return False

        mean_f = self.rolling_mean_fidelity
        if mean_f is not None and mean_f >= self.advance_threshold:
            self._current_stage = min(self._current_stage + 1, self.max_stage)
            self._fidelity_history.clear()
            self._total_advances += 1
            return True

        return False

    def force_stage(self, stage: int) -> None:
        """Manually set the noise stage (bypasses threshold check).

        Parameters
        ----------
        stage : int
            Target noise stage in [1, 5].
        """
        if stage not in NOISE_STAGE_PARAMS:
            raise ValueError(f"stage must be in [1, 5], got {stage}")
        self._current_stage = stage
        self._fidelity_history.clear()

    def state_dict(self) -> Dict:
        """Serialise curriculum state for checkpointing."""
        return {
            "current_stage":   self._current_stage,
            "total_advances":  self._total_advances,
            "fidelity_history": self._fidelity_history[-self.window:],
            "config": {
                "initial_stage":     self.initial_stage,
                "advance_threshold": self.advance_threshold,
                "window":            self.window,
                "max_stage":         self.max_stage,
                "auto_advance":      self.auto_advance,
            },
        }

    @classmethod
    def from_state_dict(cls, d: Dict) -> "NoiseCurriculum":
        """Restore curriculum from a checkpoint dictionary."""
        cfg = d["config"]
        obj = cls(
            initial_stage=cfg["initial_stage"],
            advance_threshold=cfg["advance_threshold"],
            window=cfg["window"],
            max_stage=cfg["max_stage"],
            auto_advance=cfg["auto_advance"],
        )
        obj._current_stage = d["current_stage"]
        obj._total_advances = d["total_advances"]
        obj._fidelity_history = list(d.get("fidelity_history", []))
        return obj

    def __repr__(self) -> str:
        mean_f = self.rolling_mean_fidelity
        mean_str = f"{mean_f:.4f}" if mean_f is not None else "N/A"
        return (
            f"NoiseCurriculum("
            f"stage={self._current_stage}/{self.max_stage}, "
            f"p={self.noise_prob:.4f}, "
            f"rolling_F={mean_str}, "
            f"advances={self._total_advances})"
        )
