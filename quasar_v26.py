# QUASAR v26 — OPID On-Policy Skill Distillation
# v25: P1=NoiseCurriculumController P2=rebuild_with_noise_stage P3=noise_stage_t P4=ERL-SLM noise_delta
# Base: QUASAR v24 — Hard State Family Extension
# Ported from v23:
# N1: CompleteGraph target — n-qubit complete graph state (H^n then CZ all pairs), O(n^2) gates
# N2: Dicke-k2 target — symmetric Dicke state with k=2 excitations (harder than Dicke-k3)
# N3: GHZ_W_Hybrid target — phase-twisted tensor product |GHZ>⊗|W> + e^{iφ}|W>⊗|GHZ>
# N5: P7 amplitude-redistribution extended to CompleteGraph and GHZ_W_Hybrid
# SLM Fix: enable_thinking=False in apply_chat_template (v24 primary fix)
# QUASAR v24 — ERL-SLM JSON Fix (enable_thinking=False)
# v24 change: SLMReflector._llm_reflect now uses enable_thinking=False
#             to suppress Qwen3-4B chain-of-thought wrapper and produce
#             clean JSON output directly. All v22 P1-P8 fixes retained.
# v22 changes (P1-P8 failure-analysis fixes):
# P1: SAC target entropy scale for hard cells (W/Cluster/Dicke at 3Q+)
# P2: ACC guard — suppress FLAT_UNREACHABLE before ACC_MIN_STEPS_HARD=200k
# P3: target_entropy_scale added to DEHB Inner ConfigSpace
# P4: SLM multi-call — 3 suggestions per ASR activation, pick best
# P5: FastMix intra-cell update on ASR failure (not just post-cell)
# P6: SWDFT variance-based plateau detector (catches flat regions)
# P7: Amplitude-redistribution perturbation for W/Cluster states
# P8: Transfer-learning warm-start (copy actor from n-1 qubit cell)
# v21 Phase 1 changes inherited (LoRA + alpha_SLM CG update)
# v21 Phase 1 changes:
# - TraceCollector: accumulates (circuit, outcome, source) tuples from ASR/DEHB/SLM
# - SLMLoRATrainer: periodic Qwen3-4B LoRA adapter update (rank=8, every 50k steps)
# - alpha_slm_cg_update: implicit differentiation CG step in FASTMIX outer loop
# - build_inner_cs: added lora_rank, lora_lr, lora_trigger_steps as DEHB-tunable HPs
# - run_cell: wired TraceCollector into SLM/ASR/DEHB call sites
# - FastMixOmega.update: calls alpha_slm_cg_update after each cell
# - main(): TraceCollector + SLMLoRATrainer instantiated, LoRA fine-tune triggered
# v20 changes (LEAP actions A2/A3/A4/A6/A7/A8):
# - FASTMIX-Omega: bilevel outer loop over 7 mixture coefficients
#   (alpha_SLM, alpha_BRFD, alpha_DEHB, alpha_replay, alpha_ASR, alpha_SWDFT, alpha_ACC)
#   Gradient-based update using validation oracle (3Q/W held-out cell)
# - Adaptive qubit scaling policy:
#     n_qubits <= 3 : require mean_F > 0.999 AND min_F > 0.999
#     n_qubits == 4 : require mean_F > 0.99  AND min_F > 0.99
#     n_qubits > 4  : require mean_F > 0.99  AND min_F > 0.99
# - Q-Forge recommended alpha weights for hard W/Cluster cells (A3)
# - ERL-SLM reflection collector for FASTMIX training dataset (A7)
# - ZNE (Zero-Noise Extrapolation) post-processing hook (A8)
# - All v19 fixes inherited (ERR-001 to ERR-007)
# Author: Raad Alshehri | KAU PhD | June 2026 (v21 Phase 1)
"""
quasar_v21.py — QUASAR v21 Phase 1: ERL-SLM LoRA Fine-tuning + alpha_SLM CG Update
=================================================================================
Changes from v14:
  - SWDFT  : Sliding-Window DFT stagnation detector (proactive monitor)
  - BRFD   : Bayesian Reward Function Discovery — shapes probe/reward weights
  - ERL    : Emergency Reactive Layer — 3-stage escalation pipeline
  - SDFT   : Spectral Decomposition Fidelity Tracker (Stage 2 correction)
  - TTT    : Test-Time Training (Stage 3 correction)
  - DEHB Outer : Architecture-level HPO (runs once per qubit level, all families)
  - DEHB Inner : Reactive HPO (fires on escalation failure, narrow re-search)
  - Signal Bus : Typed inter-component communication (8 signals)
  - SLM    : Now signal-gated (SWDFT→ERL→SLM) with BRFD-shaped probe budget
  - All component HPs learned by DEHB; all reward weights learned by BRFD

Architecture layers (v15):
  L1  Environment          QuasarEnv (noise-staged, multi-family)
  L2  Proactive Monitoring SWDFT stagnation detector
  L3  SAC Core             GoalConditionedActor/Critic + FiLM + DER++
  L4  Adaptive Stagnation Recovery (ASR)  ERL-SLM (Stage 1) → SDFT (Stage 2) → TTT (Stage 3)
       ERL-SLM: ReflectionModule + EpisodeMemory + targeted perturbation + internalization
  L5  Convergence Control  ACC + SLM (periodic fallback)
  L6  Adaptive Curriculum  DEHB Outer (pre-cell) + DEHB Inner (reactive)

Signal Bus signals:
  STAGNATION        SWDFT → ERL
  SLM_CORRECTED     ERL   → resume
  SLM_FAILED        ERL   → SDFT
  SDFT_CORRECTED    SDFT  → resume
  SDFT_FAILED       SDFT  → TTT
  TTT_CORRECTED     TTT   → resume
  TTT_FAILED        TTT   → DEHB Inner
  HP_UPDATED        DEHB  → SAC Core
=================================================================================
"""
# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import random
import sys
import time
import collections
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigSpace import ConfigurationSpace, Float, Integer
from dehb import DEHB
from loguru import logger as log

# ─────────────────────────────────────────────────────────────────────────────
# Device & reproducibility
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Training constants
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_SIZE   = 200_000
WARMUP_STEPS  = 1_000
UPDATE_EVERY  = 1
LOG_EVERY     = 10_000
DER_CAPACITY  = 20_000

# ───────────────────────────────────────────────────────────────────────────────
# v21 Phase 1 constants (DEHB-tunable; defaults used before first DEHB run)
# ───────────────────────────────────────────────────────────────────────────────
TRACE_BUFFER_SIZE   = 1_000    # FIFO capacity for ERL-SLM trace collector
LORA_TRIGGER_STEPS  = 50_000   # fine-tune every N training steps
LORA_MIN_NEW_TRACES = 200      # minimum new traces since last fine-tune

# ── v22 Fix constants ──────────────────────────────────────────────────────────
# P1: SAC target entropy scale for hard cells (W, Cluster, Dicke-k3 at 3Q+)
HARD_CELL_TARGETS = {"W", "w", "Cluster", "cluster", "cluster_linear",
                     "Dicke-k3", "Dicke_k3", "dicke-k3", "dicke_k3",
                     "CompleteGraph", "complete_graph", "completegraph",
                     "Dicke-k2",      "Dicke_k2",      "dicke-k2",      "dicke_k2",
                     "GHZ_W_Hybrid",  "ghz_w_hybrid",  "GHZ-W-Hybrid",  "ghz-w-hybrid"}
HARD_CELL_MIN_QUBITS = 3           # only apply to 3Q+
HARD_TARGET_ENTROPY_SCALE = -0.5   # vs default -1.0 → keeps policy exploratory

# P2: ACC minimum steps guard for hard cells
ACC_MIN_STEPS_HARD = 200_000       # never terminate hard cell before this step

# P6: SWDFT variance plateau detector
SWDFT_PLATEAU_VAR_THRESH = 1e-5    # var < 1e-5 → plateau (std < 0.003)
SWDFT_PLATEAU_MIN_WINDOW = 15      # need ≥15 checkpoints of flat data
LORA_RANK           = 8        # LoRA rank (DEHB-tunable)
LORA_ALPHA_SCALE    = 16       # LoRA alpha scaling factor
LORA_LR             = 2e-4     # LoRA learning rate (DEHB-tunable)
LORA_EPOCHS         = 3        # LoRA training epochs per fine-tune
LORA_BATCH_SIZE     = 4        # LoRA batch size

# ─────────────────────────────────────────────────────────────────────────────
# v26 OPID constants
# ─────────────────────────────────────────────────────────────────────────────
OPID_RESCORE_BATCH      = 8       # number of traces to re-score per OPID step
OPID_ADVANTAGE_CLIP     = 3.0     # clip log-prob shift to [-3, 3]
OPID_EPISODE_SKILL_K    = 3       # top-k episode-level skills to extract
OPID_STEP_SKILL_TOPK    = 5       # top-k critical steps per trajectory
OPID_DISTILL_WEIGHT     = 0.4     # weight of OPID advantage vs supervised loss
OPID_MIN_TRACES         = 20      # minimum traces before OPID fine-tune
OPID_CRITICAL_THRESHOLD = 0.15    # action std threshold for critical step detection
ALPHA_SLM_MIN       = 0.05     # minimum alpha_SLM mixture weight
ALPHA_SLM_MAX       = 0.60     # maximum alpha_SLM mixture weight
ALPHA_SLM_ADAM_LR   = 1e-3    # Adam lr for alpha_SLM CG update

# ───────────────────────────────────────────────────────────────────────────────
# Signal Bus
# ─────────────────────────────────────────────────────────────────────────────
class Signal(Enum):
    STAGNATION      = auto()
    SLM_CORRECTED   = auto()
    SLM_FAILED      = auto()
    SDFT_CORRECTED  = auto()
    SDFT_FAILED     = auto()
    TTT_CORRECTED   = auto()
    TTT_FAILED      = auto()
    HP_UPDATED      = auto()

class SignalBus:
    """Lightweight typed signal bus for inter-component communication."""
    def __init__(self):
        self._signals: Dict[Signal, Any] = {}

    def emit(self, signal: Signal, payload: Any = None):
        self._signals[signal] = payload

    def has(self, signal: Signal) -> bool:
        return signal in self._signals

    def consume(self, signal: Signal) -> Any:
        return self._signals.pop(signal, None)

    def clear(self):
        self._signals.clear()

# ─────────────────────────────────────────────────────────────────────────────
# SWDFT — Sliding-Window Discrete Fourier Transform stagnation detector
# ─────────────────────────────────────────────────────────────────────────────

# ─── v23 N1/N2/N3: Hard State Family Builders ────────────────────────────────
import math as _math_v23

def build_complete_graph_state(n: int) -> np.ndarray:
    """
    v23 N1: n-qubit complete graph state.
    Circuit: H on all qubits, then CZ on every pair (i,j) with i<j.
    O(n^2) CZ gates — significantly harder than GHZ/Cluster for the DRL agent.
    Returns a normalised complex128 state vector of shape (2**n,).
    """
    dim = 2 ** n
    sv = np.ones(dim, dtype=np.complex128) / np.sqrt(float(dim))
    for i in range(n):
        for j in range(i + 1, n):
            for idx in range(dim):
                if ((idx >> (n - 1 - i)) & 1) and ((idx >> (n - 1 - j)) & 1):
                    sv[idx] *= -1
    sv /= np.linalg.norm(sv)
    return sv


def build_dicke_k2_state(n: int) -> np.ndarray:
    """
    v23 N2: n-qubit symmetric Dicke state with k=2 excitations.
    |D^n_2> = C(n,2)^{-1/2} * sum_{|x|=2} |x>
    Harder than Dicke-k3 for n>=6 because C(n,2) grows as O(n^2).
    Returns a normalised complex128 state vector of shape (2**n,).
    """
    dim = 2 ** n
    sv = np.zeros(dim, dtype=np.complex128)
    count = 0
    for idx in range(dim):
        if bin(idx).count('1') == 2:
            sv[idx] = 1.0
            count += 1
    if count > 0:
        sv /= np.sqrt(float(count))
    return sv


def build_ghz_w_hybrid_state(n: int, phi: float = _math_v23.pi / 4) -> np.ndarray:
    """
    v23 N3: Phase-twisted GHZ-W tensor product.
    For even n: |GHZ_{n/2}> x |W_{n/2}> + e^{i*phi} |W_{n/2}> x |GHZ_{n/2}>
    The relative phase phi=pi/4 creates a non-monotone reward landscape.
    For odd n: falls back to CompleteGraph to avoid dimension mismatch.
    Returns a normalised complex128 state vector of shape (2**n,).
    """
    if n % 2 != 0:
        return build_complete_graph_state(n)
    half = n // 2
    dim_half = 2 ** half
    ghz = np.zeros(dim_half, dtype=np.complex128)
    ghz[0] = 1.0 / _math_v23.sqrt(2)
    ghz[-1] = 1.0 / _math_v23.sqrt(2)
    w = np.zeros(dim_half, dtype=np.complex128)
    for k in range(half):
        w[1 << (half - 1 - k)] = 1.0 / _math_v23.sqrt(half)
    sv = np.kron(ghz, w) + np.exp(1j * phi) * np.kron(w, ghz)
    sv /= np.linalg.norm(sv)
    return sv


def _build_v23_target_sv(target_name: str, n_qubits: int):
    """
    v23 N6: Dispatch function — returns the state vector for a v23 hard target,
    or None if the target is not a v23 family (handled by SiliQun natively).
    """
    _tn = target_name.lower().replace("-", "_").replace(" ", "_")
    if _tn in ("completegraph", "complete_graph"):
        return build_complete_graph_state(n_qubits)
    if _tn in ("dicke_k2",):
        return build_dicke_k2_state(n_qubits)
    if _tn in ("ghz_w_hybrid",):
        return build_ghz_w_hybrid_state(n_qubits)
    return None

# ─────────────────────────────────────────────────────────────────────────────

class SWDFT:
    """
    Monitors best_F over a sliding window and detects stagnation by measuring
    the ratio of low-frequency power to total power in the fidelity time series.
    When the signal is dominated by noise (flat), it emits STAGNATION.

    Parameters
    ----------
    window      : Number of LOG_EVERY checkpoints to keep in the window
    flat_thresh : Low-freq power ratio below which stagnation is declared
    min_steps   : Minimum steps before detection is active
    cooldown    : Minimum checkpoints between successive STAGNATION signals
    """
    def __init__(self, window: int = 20, flat_thresh: float = 0.15,
                 min_steps: int = 50_000, cooldown: int = 10):
        self.window      = window
        self.flat_thresh = flat_thresh
        self.min_steps   = min_steps
        self.cooldown    = cooldown
        self._history: deque = deque(maxlen=window)
        self._last_trigger  = -cooldown
        self._n_checks      = 0

    def update(self, step: int, best_F: float, bus: SignalBus):
        """Call at every LOG_EVERY checkpoint. Emits STAGNATION if detected."""
        self._history.append(best_F)
        self._n_checks += 1
        if step < self.min_steps or len(self._history) < self.window:
            return
        if (self._n_checks - self._last_trigger) < self.cooldown:
            return
        arr = np.array(self._history, dtype=np.float64)
        # DFT of the fidelity window
        spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
        total_power = spectrum.sum() + 1e-12
        # Low-frequency = DC + first 2 harmonics
        low_freq_power = spectrum[:3].sum()
        ratio = low_freq_power / total_power
        # v22 P6: variance-based plateau detector (catches flat regions missed by DFT)
        _var = float(arr.var())
        _plateau = (_var < SWDFT_PLATEAU_VAR_THRESH
                    and len(self._history) >= SWDFT_PLATEAU_MIN_WINDOW)
        if ratio < self.flat_thresh or _plateau:
            _reason = "DFT" if ratio < self.flat_thresh else "VAR_PLATEAU"
            bus.emit(Signal.STAGNATION, payload={
                "step": step, "best_F": best_F,
                "lf_ratio": float(ratio),
                "window_mean": float(arr.mean()),
                "window_std":  float(arr.std()),
                "window_var":  _var,
                "stagnation_reason": _reason,
            })
            self._last_trigger = self._n_checks
            log.info(f"    SWDFT STAGNATION [{_reason}] @ step={step:,} "
                     f"best_F={best_F:.4f} lf_ratio={ratio:.3f} var={_var:.2e}")

    def reset(self):
        self._history.clear()
        self._n_checks = 0
        self._last_trigger = -self.cooldown

# ─────────────────────────────────────────────────────────────────────────────
# BRFD — Bayesian Reward Function Discovery
# ─────────────────────────────────────────────────────────────────────────────
class BRFD:
    """
    Lightweight Bayesian bandit over reward weight combinations.
    Maintains a Beta distribution per weight preset and updates based on
    whether the probe improved best_F.

    Also provides probe_count suggestions calibrated to training difficulty.
    """
    PRESETS = [
        # (w_F, w_S, w_str)  — must sum to <= 1.0; remainder goes to w_log
        (0.60, 0.20, 0.10),
        (0.70, 0.15, 0.05),
        (0.80, 0.10, 0.05),
        (0.50, 0.30, 0.10),
        (0.65, 0.20, 0.10),
        (0.75, 0.15, 0.05),
    ]

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = len(self.PRESETS)
        # Beta(alpha, beta) per preset — start with uniform prior
        self._alpha = np.ones(n, dtype=np.float64)
        self._beta  = np.ones(n, dtype=np.float64)
        self._rng   = rng
        self._last_preset_idx: int = 0

    def sample_weights(self) -> Tuple[float, float, float]:
        """Thompson sampling: draw from each Beta, pick argmax."""
        samples = self._rng.beta(self._alpha, self._beta)
        idx = int(np.argmax(samples))
        self._last_preset_idx = idx
        return self.PRESETS[idx]

    def update(self, improved: bool):
        """Update the last-sampled preset based on whether probe improved F."""
        idx = self._last_preset_idx
        if improved:
            self._alpha[idx] += 1.0
        else:
            self._beta[idx] += 1.0

    def suggest_probe_count(self, best_F: float, step: int,
                            base: int = 300) -> int:
        """
        More probes when F is low (hard regime) or step is large (late plateau).
        Returns a value in [base, base * 4].
        """
        difficulty = max(0.0, 1.0 - best_F)          # 0 = easy, 1 = hard
        late_factor = min(1.0, step / 1_000_000)      # 0 = early, 1 = late
        multiplier  = 1.0 + 2.0 * difficulty + 1.0 * late_factor
        return int(base * multiplier)

    def score_probe(self, probe_F: float, best_F: float,
                    step: int) -> float:
        """
        Score a probe action during SLM. Rewards improvement over best_F
        and penalises regression. Used to select the best probe action.
        """
        delta = probe_F - best_F
        return delta + 0.1 * probe_F  # absolute quality bonus

# ─────────────────────────────────────────────────────────────────────────────
# ReflectionModule — ERL [2602.13949] Shi et al. 2026
# Diagnoses stagnation type and produces a structured reflection vector Δ
# ─────────────────────────────────────────────────────────────────────────────
class StagnationType(Enum):
    """Stagnation classification produced by ReflectionModule.diagnose()."""
    FLAT_LANDSCAPE      = auto()   # policy is stuck in a flat reward region
    WRONG_DISTRIBUTION  = auto()   # action distribution misaligned with target
    NOISE_SENSITIVITY   = auto()   # high sensitivity to environment noise
    EXPLORATION_DEFICIT = auto()   # insufficient action entropy


class SLMReflector:
    """
    Small Language Model (SLM) reflection generator for QUASAR v19.

    SLM = Qwen3-4B (Qwen Team, 2025).
    ERL (Experiential Reinforcement Learning, Shi et al. 2026, arXiv:2602.13949)
    augments the SLM by wrapping it in a two-attempt loop:

        y⁽¹⁾  First attempt  — SAC policy, random SLM probing
          ↓
        Δ     SLM reflection — Qwen3-4B reasons about why the agent is stuck
          ↓
        y⁽²⁾  Second attempt — Δ-guided SLM perturbation (ERL augmentation)
          ↓
        KL distillation — successful y⁽²⁾ internalized into base actor

    The SLM is loaded lazily in INT8 (~4 GB VRAM) on first stagnation event.
    Uses /think mode on first activation per cell for deep reasoning,
    /no_think on subsequent activations for speed.
    Falls back to deterministic Δ computation if model is unavailable.
    """
    MODEL_ID       = "Qwen/Qwen3-4B"
    DELTA_DIM      = 5
    MAX_NEW_TOKENS = 300

    def __init__(self, device: str = "cuda"):
        self._model    = None
        self._tok      = None
        self._device   = device
        self._activation_count: int = 0
        # Running statistics for deterministic fallback
        self._grad_var_history: deque = deque(maxlen=20)
        self._entropy_history:  deque = deque(maxlen=20)
        self._fidelity_history: deque = deque(maxlen=20)
        # Last LLM-parsed guidance (read by slm_correction)
        self.last_sigma_scale:  float = 1.0
        self.last_strategy:     str   = "gaussian"
        self.last_direction:    str   = "none"
        self.last_noise_delta:  int   = 0   # v25 P4

    # ── Lazy model loading ──────────────────────────────────────────────────
    def _ensure_loaded(self) -> bool:
        """Load Qwen3-4B using the best available quantization strategy.
        - A100 / large VRAM (>=40GB): fp16, no quantization
        - RTX 2070 / small VRAM (<10GB): 4-bit NF4 via BitsAndBytesConfig
        - Fallback: CPU offload with INT8 + fp32 offload enabled
        Returns True on success, False if model unavailable."""
        if self._model is not None:
            return True
        try:
            import torch as _torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            self._tok = AutoTokenizer.from_pretrained(
                self.MODEL_ID, trust_remote_code=True)
            vram_gb = 0.0
            if _torch.cuda.is_available():
                vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 40.0:
                # A100 / H100 — load in fp16, no quantization needed
                log.info(f"SLMReflector: loading Qwen3-4B in fp16 (VRAM={vram_gb:.1f}GB) \u2026")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=_torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                log.info("SLMReflector: Qwen3-4B loaded OK (fp16)")
            else:
                # RTX 2070 / small VRAM — use 4-bit NF4 quantization (~2.5 GB)
                log.info(f"SLMReflector: loading Qwen3-4B in 4-bit NF4 (VRAM={vram_gb:.1f}GB) \u2026")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                log.info("SLMReflector: Qwen3-4B loaded OK (4-bit NF4)")
            self._model.eval()
            return True
        except Exception as exc:
            log.warning(f"SLMReflector: model load failed ({exc}) \u2014 using deterministic fallback")
            return False

    # ── Core ERL reflection call ────────────────────────────────────────────
    def compute_delta(
        self,
        fidelity_gap:      float,
        grad_variance:     float,
        noise_sensitivity: float,
        action_entropy:    float,
        convergence_rate:  float,
        n_qubits:          int   = 2,
        target_name:       str   = "GHZ",
        step:              int   = 0,
        stag_steps:        int   = 0,
    ) -> np.ndarray:
        """
        ERL reflection step: call Qwen3-4B to generate structured Δ ∈ ℝ⁵.
        The SLM reasons about why the SAC agent is stuck and outputs JSON
        guidance that is parsed into the reflection vector and stored as
        last_sigma_scale / last_strategy / last_direction for slm_correction().
        Falls back to deterministic computation if LLM call fails.
        """
        self._activation_count += 1
        # Update running stats for fallback
        self._grad_var_history.append(grad_variance)
        self._entropy_history.append(action_entropy)
        self._fidelity_history.append(1.0 - fidelity_gap)

        if self._ensure_loaded():
            try:
                delta = self._llm_reflect(
                    fidelity_gap, grad_variance, noise_sensitivity,
                    action_entropy, convergence_rate,
                    n_qubits, target_name, step, stag_steps,
                )
                if delta is not None:
                    return delta
            except Exception as exc:
                log.warning(f"SLMReflector: LLM reflect failed ({exc}) — fallback")

        return self._deterministic_delta(
            fidelity_gap, grad_variance, noise_sensitivity,
            action_entropy, convergence_rate,
        )

    def _llm_reflect(
        self,
        fidelity_gap:      float,
        grad_variance:     float,
        noise_sensitivity: float,
        action_entropy:    float,
        convergence_rate:  float,
        n_qubits:          int,
        target_name:       str,
        step:              int,
        stag_steps:        int,
    ) -> Optional[np.ndarray]:
        """Call Qwen3-4B and parse its JSON reflection into \u0394."""
        import json as _json, re as _re
        # Always use /no_think — chain-of-thought mode wraps output in <think> tags
        # and produces free-form prose that prevents reliable JSON extraction.
        # FIX (ERL-SLM-001, 2026-06-29): Strengthened system prompt to prevent
        # markdown code fences and improved JSON extraction regex to handle
        # nested braces and leading/trailing whitespace.
        system_msg = (
            "You are a quantum control advisor. "
            "You MUST respond with ONLY a single valid JSON object. "
            "Do NOT use markdown. Do NOT use code fences (no ```). "
            "Do NOT add any text before or after the JSON. "
            "The JSON must have exactly these 9 keys: "
            "sigma_scale (float 0.1-3.0), "
            "strategy (string, one of: laplace, gaussian, directional), "
            "direction_hint (string, one of: increase_entanglement, reduce_depth, change_gate_family, none), "
            "fidelity_gap (float 0.0-1.0), "
            "grad_var (float 0.0-1.0), "
            "noise_sens (float 0.0-1.0), "
            "entropy (float 0.0-1.0), "
            "conv_rate (float 0.0-1.0), "
            "noise_delta (integer -1/0/+1: -1=reduce noise, 0=keep, +1=increase). "
            "Example of the EXACT format required: "
            '{"sigma_scale": 1.2, "strategy": "gaussian", "direction_hint": "none", '
            '"fidelity_gap": 0.05, "grad_var": 0.3, "noise_sens": 0.2, '
            '"entropy": 0.6, "conv_rate": 0.1, "noise_delta": 0}'
        )
        user_msg = (
            f"/no_think\n"
            f"A SAC agent training to synthesise a {n_qubits}-qubit {target_name} "
            f"state has stagnated. Training statistics:\n"
            f"  fidelity_gap (1-F): {fidelity_gap:.4f}\n"
            f"  grad_variance:      {grad_variance:.6f}\n"
            f"  noise_sensitivity:  {noise_sensitivity:.4f}\n"
            f"  action_entropy:     {action_entropy:.4f}\n"
            f"  convergence_rate:   {convergence_rate:.4f}\n"
            f"  steps_elapsed:      {step:,}\n"
            f"  steps_stagnated:    {stag_steps:,}\n"
            f"Output the JSON object only. Start with {{ and end with }}."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        text_input = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self._tok(text_input, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self._tok.eos_token_id,
            )
        response = self._tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        log.debug(f"SLMReflector raw response: {response[:400]}")
        # FIX (ERL-SLM-001): Multi-stage JSON extraction:
        # 1. Strip <think>...</think> chain-of-thought blocks
        # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
        # 3. Use a robust brace-balanced regex that handles nested structures
        # 4. Catch JSONDecodeError explicitly and log the raw response for debugging
        response_clean = _re.sub(
            r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
        # Strip markdown code fences
        response_clean = _re.sub(
            r'```(?:json)?\s*', '', response_clean).strip()
        response_clean = response_clean.replace('```', '').strip()
        # Find the first { and last } to extract the outermost JSON object
        brace_start = response_clean.find('{')
        brace_end   = response_clean.rfind('}')
        if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
            log.warning(
                f"SLMReflector: no JSON found in response "
                f"(len={len(response_clean)}, preview={response_clean[:120]!r})")
            return None
        json_str = response_clean[brace_start:brace_end + 1]
        try:
            d = _json.loads(json_str)
        except _json.JSONDecodeError as exc:
            log.warning(
                f"SLMReflector: JSON parse error ({exc}) — "
                f"raw_json={json_str[:200]!r}")
            return None
        # Store LLM guidance for slm_correction()
        self.last_sigma_scale = float(np.clip(d.get("sigma_scale", 1.0), 0.1, 3.0))
        self.last_strategy    = str(d.get("strategy", "gaussian"))
        self.last_direction   = str(d.get("direction_hint", "none"))
        # v25 P4: parse noise_delta
        try: _nd = int(round(float(d.get("noise_delta", 0))))
        except: _nd = 0
        self.last_noise_delta = int(max(-1, min(1, _nd)))
        delta = np.array([
            float(np.clip(d.get("fidelity_gap", fidelity_gap),      0.0, 1.0)),
            float(np.clip(d.get("grad_var",     grad_variance),      0.0, 1.0)),
            float(np.clip(d.get("noise_sens",   noise_sensitivity),  0.0, 1.0)),
            float(np.clip(d.get("entropy",      action_entropy),     0.0, 1.0)),
            float(np.clip(d.get("conv_rate",    convergence_rate),   0.0, 1.0)),
        ], dtype=np.float32)
        log.info(
            f"SLMReflector (Qwen3-4B): Δ={np.round(delta,3)} "
            f"sigma_scale={self.last_sigma_scale:.2f} "
            f"strategy={self.last_strategy} direction={self.last_direction}"
        )
        return delta

    def _deterministic_delta(
        self,
        fidelity_gap:      float,
        grad_variance:     float,
        noise_sensitivity: float,
        action_entropy:    float,
        convergence_rate:  float,
    ) -> np.ndarray:
        """Deterministic fallback when Qwen3-4B is unavailable."""
        gv_mean  = float(np.mean(self._grad_var_history)) + 1e-12
        gv_norm  = float(np.clip(grad_variance / gv_mean, 0.0, 1.0))
        ent_max  = float(max(self._entropy_history)) + 1e-12
        ent_norm = float(np.clip(action_entropy / ent_max, 0.0, 1.0))
        self.last_sigma_scale = 1.0
        self.last_strategy    = "gaussian"
        self.last_direction   = "none"
        return np.array([
            float(np.clip(fidelity_gap,      0.0, 1.0)),
            float(np.clip(gv_norm,           0.0, 1.0)),
            float(np.clip(noise_sensitivity, 0.0, 1.0)),
            float(np.clip(ent_norm,          0.0, 1.0)),
            float(np.clip(convergence_rate,  0.0, 1.0)),
        ], dtype=np.float32)

    def diagnose(self, delta: np.ndarray) -> StagnationType:
        """Classify dominant stagnation type from Δ."""
        fidelity_gap, grad_var, noise_sens, entropy, conv_rate = delta
        if noise_sens > 0.6:
            return StagnationType.NOISE_SENSITIVITY
        if entropy < 0.25:
            return StagnationType.EXPLORATION_DEFICIT
        if grad_var < 0.15:
            return StagnationType.FLAT_LANDSCAPE
        return StagnationType.WRONG_DISTRIBUTION

    def reset_cell(self):
        """Reset per-cell activation count when a new cell begins."""
        self._activation_count = 0

# Backward-compat alias so any code referencing ReflectionModule still works
ReflectionModule = SLMReflector


# ───────────────────────────────────────────────────────────────────────────────
# v21 Phase 1: TraceCollector — accumulates ERL-SLM outcome traces for LoRA
# ───────────────────────────────────────────────────────────────────────────────
class TraceCollector:
    """
    Accumulates ERL-SLM outcome traces for LoRA fine-tuning.

    Each trace records: target_name, n_qubits, outcome_F, source label
    (one of SLM/DEHB/ASR/random), step, seed, SLM reflection delta,
    perturbation sigma, strategy, and whether fidelity improved.

    Buffer: deque with maxlen=TRACE_BUFFER_SIZE (FIFO eviction).
    Traces are appended to a JSONL file on disk after every add() call.
    """

    def __init__(self, seed: int, buffer_size: int = TRACE_BUFFER_SIZE,
                 jsonl_path: Optional[str] = None):
        self.seed         = seed
        self.buffer_size  = buffer_size
        self._buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._n_total     = 0
        self._n_since_ft  = 0   # new traces since last fine-tune
        if jsonl_path is None:
            jsonl_path = f"/tmp/slm_traces_{seed}.jsonl"
        self._path = Path(jsonl_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", buffering=1)

    def add(
        self,
        target_name:   str,
        n_qubits:      int,
        outcome_F:     float,
        source:        str,
        step:          int,
        best_F_before: float,
        delta:         Optional[np.ndarray] = None,
        sigma:         float = 0.3,
        strategy:      str   = "gaussian",
        circuit_str:   str   = "N/A",
    ) -> None:
        """Add one trace to the buffer and flush to disk."""
        trace = {
            "circuit_str":  circuit_str,
            "target_name":  target_name,
            "n_qubits":     n_qubits,
            "outcome_F":    round(float(outcome_F), 6),
            "source":       source,
            "step":         step,
            "seed":         self.seed,
            "delta":        delta.tolist() if delta is not None else [],
            "sigma":        round(float(sigma), 4),
            "strategy":     strategy,
            "improved":     bool(outcome_F > best_F_before + 1e-5),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._buffer.append(trace)
        self._n_total    += 1
        self._n_since_ft += 1
        self._fh.write(json.dumps(trace) + "\n")

    def should_finetune(self, current_step: int, last_ft_step: int,
                        trigger_steps: int = LORA_TRIGGER_STEPS,
                        min_new: int = LORA_MIN_NEW_TRACES) -> bool:
        """Return True if LoRA fine-tune should be triggered."""
        return ((current_step - last_ft_step) >= trigger_steps
                and self._n_since_ft >= min_new)

    def get_training_batch(self) -> List[dict]:
        """Return a copy of the current buffer as a list."""
        return list(self._buffer)

    def mark_finetuned(self) -> None:
        """Reset the new-trace counter after a fine-tune."""
        self._n_since_ft = 0

    @property
    def n_total(self) -> int:
        return self._n_total

    @property
    def n_since_finetune(self) -> int:
        return self._n_since_ft

    def __len__(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def summary(self) -> dict:
        return {
            "n_total":    self._n_total,
            "n_since_ft": self._n_since_ft,
            "buffer_len": len(self._buffer),
            "jsonl_path": str(self._path),
        }



# ─────────────────────────────────────────────────────────────────────────────
# v26: OPIDTraceCollector — on-policy trace collection with log-prob re-scoring
# Extends TraceCollector with OPID advantage computation [arXiv 2606.26790]
# ─────────────────────────────────────────────────────────────────────────────
class OPIDTraceCollector(TraceCollector):
    """
    OPID-augmented trace collector. In addition to standard trace fields, stores:
      - log_pi_base:  log P(action | state, base_prompt)
      - log_pi_skill: log P(action | state, skill_prompt)
      - opid_advantage: log_pi_skill - log_pi_base (clipped to OPID_ADVANTAGE_CLIP)
    The opid_advantage replaces the binary improved/not-improved weight in
    OPIDLoRATrainer, eliminating distribution mismatch in fine-tuning.
    """
    def __init__(self, seed: int, slm_reflector=None,
                 buffer_size: int = TRACE_BUFFER_SIZE,
                 jsonl_path=None):
        super().__init__(seed=seed, buffer_size=buffer_size, jsonl_path=jsonl_path)
        self._slm = slm_reflector
        self._n_rescored = 0

    def add_with_rescore(self, target_name, n_qubits, outcome_F, source,
                         step, best_F_before, delta=None, sigma=0.3,
                         strategy="gaussian", circuit_str="N/A",
                         skill_context=None):
        """Add trace and compute OPID advantage via log-prob re-scoring."""
        self.add(target_name=target_name, n_qubits=n_qubits,
                 outcome_F=outcome_F, source=source, step=step,
                 best_F_before=best_F_before, delta=delta, sigma=sigma,
                 strategy=strategy, circuit_str=circuit_str)
        opid_advantage = 0.0
        if skill_context is not None and self._slm is not None:
            try:
                log_base  = self._compute_log_prob(
                    target_name, n_qubits, outcome_F, strategy, sigma,
                    delta, skill_context=None)
                log_skill = self._compute_log_prob(
                    target_name, n_qubits, outcome_F, strategy, sigma,
                    delta, skill_context=skill_context)
                raw_adv = log_skill - log_base
                opid_advantage = float(np.clip(raw_adv,
                    -OPID_ADVANTAGE_CLIP, OPID_ADVANTAGE_CLIP))
                self._n_rescored += 1
            except Exception as _e:
                log.debug(f"OPIDTraceCollector: rescore failed ({_e})")
        if self._buffer:
            self._buffer[-1]["opid_advantage"] = round(opid_advantage, 6)

    def _compute_log_prob(self, target_name, n_qubits, outcome_F,
                          strategy, sigma, delta, skill_context=None):
        """Compute log P(response | prompt) for OPID advantage."""
        if self._slm is None or self._slm._model is None or self._slm._tok is None:
            return 0.0
        _dev = next(self._slm._model.parameters()).device
        instruction = (
            f"You are a quantum control advisor. "
            f"A {n_qubits}-qubit {target_name} SAC agent "
            f"produced fidelity {outcome_F:.4f} "
            f"using strategy '{strategy}' (sigma={sigma:.3f}). "
            f"Output the JSON reflection for the next attempt."
        )
        if skill_context:
            instruction = f"[SKILL] {skill_context[:200]}\n" + instruction
        delta_list = delta.tolist() if delta is not None else [0.5, 0.3, 0.2, 0.5, 0.1]
        response_json = json.dumps({
            "sigma_scale":    round(sigma / 0.3, 2),
            "strategy":       strategy,
            "direction_hint": "none",
            "fidelity_gap":   round(delta_list[0] if delta_list else 0.5, 4),
            "grad_var":       round(delta_list[1] if len(delta_list) > 1 else 0.3, 4),
            "noise_sens":     round(delta_list[2] if len(delta_list) > 2 else 0.2, 4),
            "entropy":        round(delta_list[3] if len(delta_list) > 3 else 0.5, 4),
            "conv_rate":      round(delta_list[4] if len(delta_list) > 4 else 0.1, 4),
        })
        full_text = instruction + "\n" + response_json
        try:
            enc = self._slm._tok(full_text, return_tensors="pt",
                                  truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(_dev)
            with torch.no_grad():
                out = self._slm._model(input_ids=input_ids, labels=input_ids)
            n_tokens = input_ids.shape[1]
            return -float(out.loss.item()) * n_tokens
        except Exception as _e:
            log.debug(f"OPIDTraceCollector._compute_log_prob failed ({_e})")
            return 0.0

    def summary(self):
        base = super().summary()
        base["n_rescored"] = self._n_rescored
        return base

# ─────────────────────────────────────────────────────────────────────────────
# CATE-B: CATEWeightedSampler — online causal treatment-effect weighting
# Inspired by CausalMix (Tang et al. 2026, arXiv:2607.01104), adapted to the
# online non-stationary RL setting of QUASAR.
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# ACI-SLM (HB-79): AutoTrainess-inspired Agent-Computer Interface for SLM ops
# Reference: AutoTrainess (arXiv:2606.31551)
# =============================================================================
class SLMACIEvent:
    """Single structured event in the ACI operation journal."""
    __slots__ = ("step", "module", "op", "status", "payload", "ts")

    def __init__(self, step, module, op, status, payload):
        import time as _time
        self.step    = step
        self.module  = module
        self.op      = op
        self.status  = status
        self.payload = payload
        self.ts      = _time.time()

    def to_dict(self):
        return {
            "step":    self.step,
            "module":  self.module,
            "op":      self.op,
            "status":  self.status,
            "payload": self.payload,
            "ts":      round(self.ts, 2),
        }


class SLMACIRepository:
    """
    Structured ACI repository for all SLM operations in QUASAR.
    Inspired by AutoTrainess (arXiv:2606.31551).

    5 ACI Modules:
      PLAN  -> SLMReflector.reflect()    (cooldown + budget + gap guards)
      DATA  -> TraceCollector.add()      (noise-floor guard)
      TRAIN -> SLMLoRATrainer.finetune() (min-traces guard)
      EVAL  -> post-finetune regression  (rollback if F drops > tol)
      LOG   -> SLMACIEvent journal       (rolling experiment log)
    """
    MIN_TRACE_F         = 0.05
    MIN_TRACES_FOR_FT   = 20
    REGRESS_TOL         = 0.02
    MAX_JOURNAL_ENTRIES = 2000
    PLAN_COOLDOWN_STEPS = 50

    def __init__(self):
        self._journal = []
        self._step = 0
        self._last_plan_step = -9999
        self._best_F_at_last_ft = 0.0
        self._n_guard_fails = {"PLAN": 0, "DATA": 0, "TRAIN": 0, "EVAL": 0}
        self._n_ops = {"PLAN": 0, "DATA": 0, "TRAIN": 0, "EVAL": 0, "LOG": 0}

    # -- PLAN module ----------------------------------------------------------
    def aci_plan_gate(self, slm_reflector, n_qubits, fidelity_gap, **kwargs):
        if self._step - self._last_plan_step < self.PLAN_COOLDOWN_STEPS:
            self._log("PLAN", "reflect_gate", "skip",
                      {"reason": "cooldown",
                       "steps_since_last": self._step - self._last_plan_step})
            return False
        budget     = getattr(slm_reflector, "_activation_count", 0)
        max_budget = getattr(slm_reflector, "max_activations_per_cell", 999)
        if budget >= max_budget:
            self._n_guard_fails["PLAN"] += 1
            self._log("PLAN", "reflect_gate", "guard_fail",
                      {"reason": "budget_exhausted", "budget": budget, "max": max_budget})
            return False
        if fidelity_gap < 0.05:
            self._log("PLAN", "reflect_gate", "skip",
                      {"reason": "gap_too_small", "fidelity_gap": round(fidelity_gap, 4)})
            return False
        self._last_plan_step = self._step
        self._n_ops["PLAN"] += 1
        self._log("PLAN", "reflect_gate", "ok",
                  {"n_qubits": n_qubits, "fidelity_gap": round(fidelity_gap, 4)})
        return True

    def aci_plan_done(self, delta, sigma_scale, strategy):
        self._log("PLAN", "reflect_done", "ok",
                  {"delta": [round(float(x), 4) for x in delta] if delta is not None else None,
                   "sigma_scale": round(sigma_scale, 3), "strategy": strategy})

    # -- DATA module ----------------------------------------------------------
    def aci_data_gate(self, outcome_F, source, n_qubits):
        if outcome_F < self.MIN_TRACE_F:
            self._n_guard_fails["DATA"] += 1
            self._log("DATA", "add_trace", "guard_fail",
                      {"reason": "below_noise_floor",
                       "outcome_F": round(outcome_F, 4), "min_F": self.MIN_TRACE_F})
            return False
        self._n_ops["DATA"] += 1
        self._log("DATA", "add_trace", "ok",
                  {"outcome_F": round(outcome_F, 4), "source": source, "n_qubits": n_qubits})
        return True

    # -- TRAIN module ---------------------------------------------------------
    def aci_train_gate(self, n_new_traces, n_total_traces, current_best_F):
        if n_new_traces < self.MIN_TRACES_FOR_FT:
            self._n_guard_fails["TRAIN"] += 1
            self._log("TRAIN", "finetune_gate", "guard_fail",
                      {"reason": "insufficient_traces",
                       "n_new": n_new_traces, "min_required": self.MIN_TRACES_FOR_FT})
            return False
        self._best_F_at_last_ft = current_best_F
        self._n_ops["TRAIN"] += 1
        self._log("TRAIN", "finetune_gate", "ok",
                  {"n_new_traces": n_new_traces, "best_F_before": round(current_best_F, 4)})
        return True

    def aci_train_done(self, loss, n_examples, cate_summary=None):
        payload = {"loss": round(loss, 5) if loss else None, "n_examples": n_examples}
        if cate_summary:
            payload["cate_b"] = cate_summary
        self._log("TRAIN", "finetune_done", "ok", payload)

    # -- EVAL module ----------------------------------------------------------
    def aci_eval_gate(self, new_best_F, slm_lora_trainer):
        regression = self._best_F_at_last_ft - new_best_F
        if regression > self.REGRESS_TOL:
            self._n_guard_fails["EVAL"] += 1
            self._log("EVAL", "regression_check", "guard_fail",
                      {"reason": "fidelity_regression",
                       "best_F_before": round(self._best_F_at_last_ft, 4),
                       "best_F_after":  round(new_best_F, 4),
                       "regression":    round(regression, 4)})
            try:
                bak = getattr(slm_lora_trainer, "_prev_adapter_state", None)
                if bak is not None:
                    slm_lora_trainer._model.load_state_dict(bak, strict=False)
                    log.warning(
                        "SLM-ACI EVAL: rolled back LoRA adapter "
                        "(regression=%.4f > tol=%.2f)" % (regression, self.REGRESS_TOL))
            except Exception as _e:
                log.warning("SLM-ACI EVAL: rollback failed (%s)" % _e)
            return False
        self._n_ops["EVAL"] += 1
        self._log("EVAL", "regression_check", "ok",
                  {"best_F_before": round(self._best_F_at_last_ft, 4),
                   "best_F_after":  round(new_best_F, 4),
                   "delta_F":       round(new_best_F - self._best_F_at_last_ft, 4)})
        return True

    # -- LOG module -----------------------------------------------------------
    def aci_log_step(self, step, best_F, n_qubits, target, source="SAC"):
        self._step = step
        self._n_ops["LOG"] += 1
        self._log("LOG", "step_milestone", "ok",
                  {"step": step, "best_F": round(best_F, 4),
                   "n_qubits": n_qubits, "target": target, "source": source})

    def _log(self, module, op, status, payload):
        evt = SLMACIEvent(self._step, module, op, status, payload)
        self._journal.append(evt)
        if len(self._journal) > self.MAX_JOURNAL_ENTRIES:
            evict_n = self.MAX_JOURNAL_ENTRIES // 10
            self._journal = self._journal[evict_n:]

    def summary(self):
        recent = [e.status for e in self._journal[-50:]]
        ok_rate = recent.count("ok") / max(len(recent), 1)
        return {
            "n_ops":              self._n_ops.copy(),
            "n_guard_fails":      self._n_guard_fails.copy(),
            "journal_len":        len(self._journal),
            "recent_ok_rate":     round(ok_rate, 3),
            "best_F_at_last_ft":  round(self._best_F_at_last_ft, 4),
        }

    def recent_events(self, n=10):
        return [e.to_dict() for e in self._journal[-n:]]

# End of ACI-SLM classes
# =============================================================================

class CATEWeightedSampler:
    """
    Estimates the Conditional Average Treatment Effect (CATE) of each trace
    source (SLM / DEHB / ASR / random) on fidelity improvement, using an
    online S-learner:

        CATE(source) ≈ E[outcome_F | source] - E[outcome_F]

    The global baseline E[outcome_F] is updated incrementally.
    Per-source means are maintained as exponential moving averages (EMA)
    with decay alpha=0.05 so recent traces dominate (non-stationary adaptation).

    Weight for a trace = softmax(CATE_vector)[source_idx]
                        * (1.5 if improved else 0.5)

    The improved_bonus ensures successful traces are always preferred, while
    CATE re-ranks within the improved/failed groups by causal contribution.
    """
    SOURCES = ["SLM", "DEHB", "ASR", "random"]
    EMA_ALPHA = 0.05          # decay for per-source EMA (recent traces dominate)
    SOFTMAX_TEMP = 2.0        # temperature for softmax over CATE values
    IMPROVED_BONUS = 1.5      # multiplier for improved traces
    FAILED_BONUS   = 0.5      # multiplier for failed traces
    MIN_WEIGHT     = 0.1      # floor to prevent zero-weight traces

    def __init__(self):
        # Per-source EMA of outcome_F
        self._src_ema: dict = {s: 0.5 for s in self.SOURCES}
        self._src_n:   dict = {s: 0   for s in self.SOURCES}
        # Global EMA of outcome_F (counterfactual baseline)
        self._global_ema: float = 0.5
        self._global_n:   int   = 0
        self._n_updates:  int   = 0

    def update(self, source: str, outcome_F: float) -> None:
        """Incrementally update per-source and global EMA."""
        src = source if source in self.SOURCES else "random"
        alpha = self.EMA_ALPHA
        # Warm-start: use simple mean for first 10 observations per source
        n = self._src_n[src]
        if n < 10:
            self._src_ema[src] = (self._src_ema[src] * n + outcome_F) / (n + 1)
        else:
            self._src_ema[src] = (1 - alpha) * self._src_ema[src] + alpha * outcome_F
        self._src_n[src] += 1
        # Global baseline
        if self._global_n < 10:
            self._global_ema = (self._global_ema * self._global_n + outcome_F) / (self._global_n + 1)
        else:
            self._global_ema = (1 - alpha) * self._global_ema + alpha * outcome_F
        self._global_n += 1
        self._n_updates += 1

    def weight(self, source: str, improved: bool) -> float:
        """Return the CATE-derived sample weight for a single trace."""
        src = source if source in self.SOURCES else "random"
        # CATE vector: per-source mean minus global baseline
        cate_vals = np.array([
            self._src_ema[s] - self._global_ema for s in self.SOURCES
        ], dtype=np.float32)
        # Softmax over CATE with temperature
        cate_shifted = cate_vals - cate_vals.max()
        exp_vals = np.exp(cate_shifted / self.SOFTMAX_TEMP)
        softmax_w = exp_vals / (exp_vals.sum() + 1e-8)
        src_idx = self.SOURCES.index(src)
        base_w = float(softmax_w[src_idx])
        # Apply improved bonus
        bonus = self.IMPROVED_BONUS if improved else self.FAILED_BONUS
        w = max(base_w * bonus, self.MIN_WEIGHT)
        return w

    def batch_weights(self, traces: list) -> list:
        """
        First update all EMA from the batch (read-only pass), then return
        per-trace weights. This ensures the CATE estimate uses all available
        information before weighting.
        """
        for t in traces:
            self.update(t.get("source", "random"), t.get("outcome_F", 0.5))
        return [
            self.weight(t.get("source", "random"), t.get("improved", False))
            for t in traces
        ]

    def summary(self) -> dict:
        cate = {s: round(self._src_ema[s] - self._global_ema, 4) for s in self.SOURCES}
        return {
            "n_updates":   self._n_updates,
            "global_ema":  round(self._global_ema, 4),
            "src_ema":     {s: round(v, 4) for s, v in self._src_ema.items()},
            "cate":        cate,
            "top_source":  max(cate, key=cate.get),
        }

# ───────────────────────────────────────────────────────────────────────────────
# v21 Phase 1: SLMLoRATrainer — periodic Qwen3-4B LoRA adapter update
# ───────────────────────────────────────────────────────────────────────────────
class SLMLoRATrainer:
    """
    Periodic LoRA fine-tuning of Qwen3-4B using traces from TraceCollector.

    Uses the PEFT library (LoraConfig + get_peft_model).
    Adapter saved to /tmp/slm_lora_{seed}/ after each fine-tune.
    The SLMReflector._model is hot-patched (merge_and_unload) so subsequent
    reflections immediately benefit from the updated adapter.

    Training objective: next-token prediction on JSON reflection strings
    derived from traces.  Successful (improved=True) traces get weight 1.0;
    failed traces get weight 0.3 to teach the SLM what NOT to suggest.

    Hyperparameters (DEHB-tunable via build_inner_cs):
      lora_rank, lora_lr, lora_trigger_steps
    """
    ADAPTER_DIR_TEMPLATE = "/tmp/slm_lora_{seed}"

    def __init__(
        self,
        seed:        int,
        lora_rank:   int   = LORA_RANK,
        lora_alpha:  int   = LORA_ALPHA_SCALE,
        lora_lr:     float = LORA_LR,
        lora_epochs: int   = LORA_EPOCHS,
        lora_batch:  int   = LORA_BATCH_SIZE,
    ):
        self.seed        = seed
        self.lora_rank   = lora_rank
        self.lora_alpha  = lora_alpha
        self.lora_lr     = lora_lr
        self.lora_epochs = lora_epochs
        self.lora_batch  = lora_batch
        self.adapter_dir = Path(self.ADAPTER_DIR_TEMPLATE.format(seed=seed))
        self._n_finetunes = 0
        self._last_loss: Optional[float] = None
        self._cate_sampler = CATEWeightedSampler()  # CATE-B (HB-79)
        self._aci_repo = None  # ACI-SLM (HB-79): set by main()

    def finetune(self, traces: List[dict], slm_reflector: "SLMReflector",
                 current_best_F: float = 0.0) -> bool:
        """
        Run LoRA fine-tuning on the provided traces.
        Returns True on success, False on graceful degradation.
        ACI-SLM (HB-79): TRAIN gate checked before fine-tuning.
        """
        if not traces:
            return False
        # ACI-SLM TRAIN gate
        if self._aci_repo is not None:
            if not self._aci_repo.aci_train_gate(
                    n_new_traces=len(traces),
                    n_total_traces=len(traces),
                    current_best_F=current_best_F):
                return False
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as _e:
            log.warning(f"SLMLoRATrainer: peft not available ({_e}) — skipping LoRA")
            return False
        if slm_reflector is None:
            log.warning("SLMLoRATrainer: no SLMReflector provided — skipping LoRA")
            return False
        if not slm_reflector._ensure_loaded():
            log.warning("SLMLoRATrainer: base model not loaded — skipping LoRA")
            return False

        model = slm_reflector._model
        tok   = slm_reflector._tok
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        try:
            peft_model = get_peft_model(model, lora_cfg)
        except Exception as _e:
            log.warning(f"SLMLoRATrainer: get_peft_model failed ({_e}) — skipping")
            return False

        peft_model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=self.lora_lr,
        )
        examples = self._build_examples(traces, tok, cate_sampler=getattr(self, '_cate_sampler', None))
        if not examples:
            log.warning("SLMLoRATrainer: no valid training examples — skipping")
            return False

        total_loss, n_steps = 0.0, 0
        for _epoch in range(self.lora_epochs):
            np.random.shuffle(examples)
            for i in range(0, len(examples), self.lora_batch):
                batch = examples[i : i + self.lora_batch]
                try:
                    input_ids = torch.cat([ex["input_ids"] for ex in batch], dim=0)
                    labels    = torch.cat([ex["labels"]    for ex in batch], dim=0)
                    w_mean    = float(np.mean([ex["weight"] for ex in batch]))
                    out  = peft_model(input_ids=input_ids, labels=labels)
                    loss = out.loss * w_mean
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_steps    += 1
                except Exception as _te:
                    log.warning(f"SLMLoRATrainer: training step failed ({_te})")
                    continue

        self._last_loss = total_loss / max(n_steps, 1)
        log.info(f"SLMLoRATrainer: fine-tune #{self._n_finetunes + 1} done "
                 f"loss={self._last_loss:.4f} steps={n_steps} traces={len(traces)}")
        try:
            self.adapter_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(self.adapter_dir))
            log.info(f"SLMLoRATrainer: adapter saved → {self.adapter_dir}")
        except Exception as _se:
            log.warning(f"SLMLoRATrainer: adapter save failed ({_se})")
        try:
            merged = peft_model.merge_and_unload()
            slm_reflector._model = merged
            slm_reflector._model.eval()
            log.info("SLMLoRATrainer: LoRA adapter merged into base model (hot-patch OK)")
        except Exception as _me:
            log.warning(f"SLMLoRATrainer: merge_and_unload failed ({_me}) — adapter saved to disk only")
        self._n_finetunes += 1
        # ACI-SLM: record train_done event
        if self._aci_repo is not None:
            _cate_sum = self._cate_sampler.summary() if hasattr(self, "_cate_sampler") else None
            self._aci_repo.aci_train_done(
                loss=self._last_loss,
                n_examples=len(traces),
                cate_summary=_cate_sum)
        return True

    def _build_examples(self, traces: List[dict], tok,
                        cate_sampler: "CATEWeightedSampler" = None) -> List[dict]:
        """
        Convert traces to tokenised (input_ids, labels, weight) dicts.

        CATE-B (HB-79): If cate_sampler is provided, sample weights are
        determined by CATEWeightedSampler.batch_weights() instead of the
        binary 1.0/0.3 heuristic. This implements CausalMix Option B:
        CATE-weighted LoRA batching (Tang et al. 2026, arXiv:2607.01104).
        """
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Compute CATE weights for the entire batch before tokenisation
        if cate_sampler is not None:
            cate_weights = cate_sampler.batch_weights(traces)
            log.debug(f"SLMLoRATrainer CATE-B: sampler={cate_sampler.summary()}")
        else:
            # Fallback: binary heuristic (original behaviour)
            cate_weights = [1.0 if t.get("improved", False) else 0.3 for t in traces]
        examples = []
        for idx, t in enumerate(traces):
            try:
                instruction = (
                    f"You are a quantum control advisor. "
                    f"A {t['n_qubits']}-qubit {t['target_name']} SAC agent "
                    f"produced fidelity {t['outcome_F']:.4f} "
                    f"using strategy '{t['strategy']}' (sigma={t['sigma']:.3f}). "
                    f"The attempt {'improved' if t['improved'] else 'did not improve'} "
                    f"on the previous best. "
                    f"Output the JSON reflection for the next attempt."
                )
                delta = t.get("delta", [])
                if delta:
                    target_json = json.dumps({
                        "sigma_scale":    round(t["sigma"] / 0.3, 2),
                        "strategy":       t["strategy"],
                        "direction_hint": "none",
                        "fidelity_gap":   round(delta[0], 4) if len(delta) > 0 else 0.5,
                        "grad_var":       round(delta[1], 4) if len(delta) > 1 else 0.3,
                        "noise_sens":     round(delta[2], 4) if len(delta) > 2 else 0.2,
                        "entropy":        round(delta[3], 4) if len(delta) > 3 else 0.5,
                        "conv_rate":      round(delta[4], 4) if len(delta) > 4 else 0.1,
                    })
                else:
                    target_json = json.dumps({
                        "sigma_scale": 1.0, "strategy": t["strategy"],
                        "direction_hint": "none",
                        "fidelity_gap": round(1.0 - t["outcome_F"], 4),
                        "grad_var": 0.3, "noise_sens": 0.2,
                        "entropy": 0.5, "conv_rate": 0.1,
                    })
                full_text = instruction + "\n" + target_json
                enc = tok(full_text, return_tensors="pt",
                          truncation=True, max_length=512)
                input_ids = enc["input_ids"].to(_dev)
                labels    = input_ids.clone()
                n_instr   = tok(instruction, return_tensors="pt",
                                truncation=True, max_length=512)["input_ids"].shape[1]
                labels[:, :n_instr] = -100
                examples.append({
                    "input_ids": input_ids,
                    "labels":    labels,
                    "weight":    cate_weights[idx],   # CATE-B weight
                })
            except Exception as _ex:
                log.warning(f"SLMLoRATrainer: example build failed ({_ex})")
                continue
        return examples
    def summary(self) -> dict:
        d = {
            "n_finetunes": self._n_finetunes,
            "last_loss":   self._last_loss,
            "adapter_dir": str(self.adapter_dir),
            "lora_rank":   self.lora_rank,
        }
        if hasattr(self, "_cate_sampler"):
            d["cate_b"] = self._cate_sampler.summary()
        if self._aci_repo is not None:
            d["aci"] = self._aci_repo.summary()
        return d



# ─────────────────────────────────────────────────────────────────────────────
# v26: EpisodeLevelSkillExtractor — global failure-avoidance rules from episodes
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeLevelSkillExtractor:
    """
    Extracts episode-level skills from completed training trajectories.
    An episode-level skill is a concise natural-language rule capturing a global
    pattern observed across the entire trajectory (e.g., noise-stage scheduling
    heuristics, entropy coefficient guidance for specific target families).
    """
    def __init__(self, slm_reflector=None, max_skills: int = OPID_EPISODE_SKILL_K):
        self._slm = slm_reflector
        self._max_skills = max_skills
        self._skill_store = []
        self._n_extracted = 0

    def extract(self, target_name, n_qubits, trajectory_summary):
        """Extract one episode-level skill from a trajectory summary dict."""
        if self._slm is None or self._slm._model is None:
            return None
        prompt = (
            f"You are a quantum RL expert. A {n_qubits}-qubit {target_name} "
            f"training episode completed: best_F={trajectory_summary.get('best_F',0):.4f}, "
            f"F_start={trajectory_summary.get('F_start',0):.4f}, "
            f"n_steps={trajectory_summary.get('n_steps',0)}, "
            f"n_asr_calls={trajectory_summary.get('n_asr_calls',0)}, "
            f"n_noise_changes={trajectory_summary.get('n_noise_changes',0)}, "
            f"final_noise_stage={trajectory_summary.get('final_noise_stage',3)}, "
            f"dominant_failure_mode={trajectory_summary.get('dominant_failure_mode','unknown')}.\n"
            f"In ONE sentence, state the most important generalisation rule for future training. "
            f"Output ONLY the rule sentence."
        )
        try:
            skill = self._slm._call_ollama_raw(prompt, max_tokens=80)
            if skill and len(skill) > 10:
                entry = {"skill": skill.strip(), "target": target_name,
                         "n_qubits": n_qubits,
                         "F_before": trajectory_summary.get("F_start", 0),
                         "F_after": trajectory_summary.get("best_F", 0)}
                self._skill_store.append(entry)
                if len(self._skill_store) > 50:
                    self._skill_store.pop(0)
                self._n_extracted += 1
                log.info(f"  [v26 EpisodeSkill] {skill[:80]}")
                return skill.strip()
        except Exception as _e:
            log.debug(f"EpisodeLevelSkillExtractor.extract failed ({_e})")
        return None

    def get_relevant_skills(self, target_name, n_qubits, k=2):
        exact = [e["skill"] for e in self._skill_store
                 if e["target"] == target_name and e["n_qubits"] == n_qubits]
        if len(exact) >= k:
            return exact[-k:]
        same_target = [e["skill"] for e in self._skill_store
                       if e["target"] == target_name]
        combined = list(dict.fromkeys(exact + same_target))
        return combined[:k]

    def summary(self):
        return {"n_extracted": self._n_extracted, "n_stored": len(self._skill_store)}

# ─────────────────────────────────────────────────────────────────────────────
# v26: StepLevelSkillExtractor — critical timestep decision knowledge
# ─────────────────────────────────────────────────────────────────────────────
class StepLevelSkillExtractor:
    """
    Identifies critical timesteps (high action uncertainty + low fidelity) and
    extracts a concise decision rule for each, providing step-level OPID skills.
    """
    def __init__(self, slm_reflector=None,
                 top_k: int = OPID_STEP_SKILL_TOPK,
                 threshold: float = OPID_CRITICAL_THRESHOLD):
        self._slm = slm_reflector
        self._top_k = top_k
        self._threshold = threshold
        self._step_skills = []
        self._n_extracted = 0

    def identify_critical_steps(self, action_stds, fidelities):
        """Return indices of top-k critical steps (high std * low fidelity)."""
        if not action_stds:
            return []
        stds = np.array(action_stds)
        fids = np.array(fidelities) if fidelities else np.zeros_like(stds)
        scores = stds * np.clip(1.0 - fids, 0.0, 1.0)
        candidates = np.where(stds > self._threshold)[0]
        if len(candidates) == 0:
            candidates = np.argsort(scores)[-self._top_k:]
        else:
            top_idx = np.argsort(scores[candidates])[-self._top_k:]
            candidates = candidates[top_idx]
        return candidates.tolist()

    def extract_step_skill(self, step_idx, target_name, n_qubits,
                           action_std, fidelity_at_step, noise_stage):
        """Extract a step-level skill for one critical timestep."""
        if self._slm is None or self._slm._model is None:
            return None
        prompt = (
            f"You are a quantum RL expert. At step {step_idx} of a "
            f"{n_qubits}-qubit {target_name} episode, action_std={action_std:.3f} "
            f"(high uncertainty), fidelity={fidelity_at_step:.4f}, "
            f"noise_stage={noise_stage}.\n"
            f"In ONE sentence, state the specific decision rule for this critical step. "
            f"Output ONLY the rule sentence."
        )
        try:
            skill = self._slm._call_ollama_raw(prompt, max_tokens=60)
            if skill and len(skill) > 10:
                entry = {"skill": skill.strip(), "step_idx": step_idx,
                         "target": target_name, "n_qubits": n_qubits,
                         "action_std": action_std, "fidelity": fidelity_at_step}
                self._step_skills.append(entry)
                if len(self._step_skills) > 100:
                    self._step_skills.pop(0)
                self._n_extracted += 1
                return skill.strip()
        except Exception as _e:
            log.debug(f"StepLevelSkillExtractor.extract_step_skill failed ({_e})")
        return None

    def get_recent_skills(self, k=3):
        return [e["skill"] for e in self._step_skills[-k:]]

    def summary(self):
        return {"n_extracted": self._n_extracted, "n_stored": len(self._step_skills)}

# ─────────────────────────────────────────────────────────────────────────────
# v26: CriticalFirstRouter — select episode vs step skills by training phase
# ─────────────────────────────────────────────────────────────────────────────
class CriticalFirstRouter:
    """
    Routes between episode-level and step-level OPID skills based on current F:
      F < 0.50 -> episode-level skills (global rules dominate early training)
      0.50 <= F < 0.85 -> step-level skills (local decisions matter mid-training)
      F >= 0.85 -> both levels combined (fine-tuning phase)
    """
    def __init__(self, episode_extractor, step_extractor):
        self._ep  = episode_extractor
        self._stp = step_extractor
        self._n_routed = 0

    def route(self, current_F, target_name, n_qubits):
        """Return skill context string, or None if no skills available."""
        self._n_routed += 1
        if current_F < 0.5:
            skills = self._ep.get_relevant_skills(target_name, n_qubits, k=2)
        elif current_F < 0.85:
            skills = self._stp.get_recent_skills(k=3)
        else:
            ep_s  = self._ep.get_relevant_skills(target_name, n_qubits, k=1)
            stp_s = self._stp.get_recent_skills(k=2)
            skills = ep_s + stp_s
        return " | ".join(skills) if skills else None

    def summary(self):
        return {"n_routed": self._n_routed,
                "episode_skills": self._ep.summary(),
                "step_skills": self._stp.summary()}

# ─────────────────────────────────────────────────────────────────────────────
# v26: OPIDLoRATrainer — LoRA fine-tuning with OPID self-distillation advantage
# Replaces binary improved/not-improved weight with continuous opid_advantage,
# eliminating distribution mismatch in the SLM fine-tuning pipeline.
# ─────────────────────────────────────────────────────────────────────────────
class OPIDLoRATrainer(SLMLoRATrainer):
    """
    OPID-augmented LoRA trainer. Overrides _build_examples() to use
    opid_advantage (log_pi_skill - log_pi_base) as the training weight.
    Positive advantage -> upweight; negative -> downweight (min 0.1).
    Falls back to v25 binary weight if opid_advantage is absent.
    """
    def __init__(self, seed, lora_rank=8, lora_lr=2e-4,
                 lora_epochs=3, lora_batch=4):
        super().__init__(seed=seed, lora_rank=lora_rank, lora_lr=lora_lr,
                         lora_epochs=lora_epochs,
                         lora_batch=lora_batch)
        self._n_opid_steps = 0
        self._last_opid_advantage_mean = 0.0

    def _build_examples(self, traces, tok):
        """Override: use opid_advantage as training weight."""
        import math as _math
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        examples = []
        advantages = []
        for t in traces:
            try:
                opid_adv = t.get("opid_advantage", None)
                if opid_adv is not None:
                    weight = float(0.1 + 1.9 * (1.0 / (1.0 + _math.exp(-float(opid_adv)))))
                    advantages.append(float(opid_adv))
                else:
                    weight = 1.0 if t.get("improved", False) else 0.3

                instruction = (
                    f"You are a quantum control advisor. "
                    f"A {t['n_qubits']}-qubit {t['target_name']} SAC agent "
                    f"produced fidelity {t['outcome_F']:.4f} "
                    f"using strategy '{t['strategy']}' (sigma={t['sigma']:.3f}). "
                    f"The attempt {'improved' if t.get('improved', False) else 'did not improve'} "
                    f"on the previous best. "
                    f"Output the JSON reflection for the next attempt."
                )
                delta = t.get("delta", [])
                if delta:
                    target_json = json.dumps({
                        "sigma_scale":    round(t["sigma"] / 0.3, 2),
                        "strategy":       t["strategy"],
                        "direction_hint": "none",
                        "fidelity_gap":   round(delta[0], 4) if len(delta) > 0 else 0.5,
                        "grad_var":       round(delta[1], 4) if len(delta) > 1 else 0.3,
                        "noise_sens":     round(delta[2], 4) if len(delta) > 2 else 0.2,
                        "entropy":        round(delta[3], 4) if len(delta) > 3 else 0.5,
                        "conv_rate":      round(delta[4], 4) if len(delta) > 4 else 0.1,
                    })
                else:
                    target_json = json.dumps({
                        "sigma_scale": 1.0, "strategy": t.get("strategy", "gaussian"),
                        "direction_hint": "none",
                        "fidelity_gap": round(1.0 - t["outcome_F"], 4),
                        "grad_var": 0.3, "noise_sens": 0.2,
                        "entropy": 0.5, "conv_rate": 0.1,
                    })
                full_text = instruction + "\n" + target_json
                enc = tok(full_text, return_tensors="pt",
                          truncation=True, max_length=512)
                input_ids = enc["input_ids"].to(_dev)
                labels    = input_ids.clone()
                n_instr   = tok(instruction, return_tensors="pt",
                                truncation=True, max_length=512)["input_ids"].shape[1]
                labels[:, :n_instr] = -100
                examples.append({"input_ids": input_ids, "labels": labels, "weight": weight})
                self._n_opid_steps += 1
            except Exception as _ex:
                log.warning(f"OPIDLoRATrainer: example build failed ({_ex})")
                continue
        if advantages:
            self._last_opid_advantage_mean = float(np.mean(advantages))
            log.info(f"  [v26 OPID] mean_adv={self._last_opid_advantage_mean:.4f} "
                     f"n={len(advantages)}")
        return examples

    def summary(self):
        base = super().summary()
        base["n_opid_steps"] = self._n_opid_steps
        base["last_opid_advantage_mean"] = self._last_opid_advantage_mean
        return base
# ───────────────────────────────────────────────────────────────────────────────
# v21 Phase 1: alpha_slm_cg_update — implicit differentiation CG step
# ───────────────────────────────────────────────────────────────────────────────
def alpha_slm_cg_update(
    alpha_slm_current: float,
    val_F:             float,
    baseline_F:        float,
    n_slm_calls:       int,
    n_total_calls:     int,
    adam_state:        dict,
    lr:                float = ALPHA_SLM_ADAM_LR,
    beta1:             float = 0.9,
    beta2:             float = 0.999,
    eps:               float = 1e-8,
) -> Tuple[float, dict]:
    """
    Single CG step of implicit differentiation to update alpha_SLM.

    Gradient estimate: -(val_F - baseline_F) * (n_slm_calls / n_total_calls)
    Applied via Adam with lr=ALPHA_SLM_ADAM_LR.
    Clipped to [ALPHA_SLM_MIN, ALPHA_SLM_MAX].
    """
    delta_F       = val_F - baseline_F
    activity_frac = float(n_slm_calls) / max(float(n_total_calls), 1.0)
    grad          = -delta_F * activity_frac
    t  = adam_state.get("t", 0) + 1
    m  = beta1 * adam_state.get("m", 0.0) + (1.0 - beta1) * grad
    v  = beta2 * adam_state.get("v", 0.0) + (1.0 - beta2) * grad ** 2
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    step  = lr * m_hat / (math.sqrt(v_hat) + eps)
    new_alpha = float(np.clip(alpha_slm_current - step, ALPHA_SLM_MIN, ALPHA_SLM_MAX))
    return new_alpha, {"m": m, "v": v, "t": t}


# ───────────────────────────────────────────────────────────────────────────────
# EpisodeMemory — ERL [2602.13949] episodic memory for strategy reuse
# Stores successful (Δ, perturbation_strategy, improvement) triples
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    """One stored episode: reflection vector, strategy params, and improvement."""
    delta:       np.ndarray   # shape (5,) — the reflection vector
    sigma:       float        # perturbation scale used
    bias_high_entropy: bool   # whether entropy-biased probing was used
    improvement: float        # ΔF achieved (best_F_after - best_F_before)


class EpisodeMemory:
    """
    FIFO episodic memory for the ERL-augmented SLM.
    Stores successful (Δ, strategy, improvement) triples and retrieves
    the top-k most similar past entries via cosine similarity on Δ.

    Capacity: 100 entries, FIFO eviction.
    """
    MAX_CAPACITY = 100

    def __init__(self):
        self._entries: deque = deque(maxlen=self.MAX_CAPACITY)

    def store(
        self,
        delta:             np.ndarray,
        sigma:             float,
        bias_high_entropy: bool,
        improvement:       float,
    ) -> None:
        """Store a successful (Δ, strategy, improvement) triple."""
        if improvement > 0.0:  # only store genuinely successful episodes
            self._entries.append(MemoryEntry(
                delta=delta.copy(),
                sigma=sigma,
                bias_high_entropy=bias_high_entropy,
                improvement=improvement,
            ))

    def retrieve_similar(
        self,
        delta:  np.ndarray,
        top_k:  int = 3,
    ) -> List[MemoryEntry]:
        """
        Return the top-k most similar past entries by cosine similarity on Δ.
        Returns an empty list if memory is empty.
        """
        if not self._entries:
            return []
        query = delta.astype(np.float64)
        q_norm = np.linalg.norm(query) + 1e-12
        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            key  = entry.delta.astype(np.float64)
            k_norm = np.linalg.norm(key) + 1e-12
            sim  = float(np.dot(query, key) / (q_norm * k_norm))
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, tgt_dim: int):
        self.capacity = capacity
        self.ptr = 0; self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),        dtype=np.float32)
        self.targets  = np.zeros((capacity, tgt_dim),  dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, target):
        idx = self.ptr % self.capacity
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = done
        self.targets[idx]  = target
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        def t(x): return torch.FloatTensor(x[idx]).to(DEVICE)
        return t(self.obs), t(self.actions), t(self.rewards), \
               t(self.next_obs), t(self.dones), t(self.targets)

    def __len__(self): return self.size

# ─────────────────────────────────────────────────────────────────────────────
# DER++ Continual Learning Buffer
# ─────────────────────────────────────────────────────────────────────────────
class DERPlusPlusBuffer:
    """Cross-qubit transfer memory — stores (obs, act, rew, next_obs, done, tgt, logit)."""
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = DER_CAPACITY):
        self.capacity  = capacity
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self._n_stored = 0
        self._ptr      = 0
        self._obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._acts     = np.zeros((capacity, act_dim),  dtype=np.float32)
        self._rews     = np.zeros((capacity, 1),        dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),        dtype=np.float32)
        self._tgts     = np.zeros((capacity, obs_dim),  dtype=np.float32)

    def add_batch(self, obs, acts, rews, next_obs, dones, tgts):
        n = len(obs)
        for i in range(n):
            idx = self._ptr % self.capacity
            self._obs[idx]      = obs[i, :self.obs_dim]
            self._acts[idx]     = acts[i, :self.act_dim]
            self._rews[idx]     = rews[i]
            self._next_obs[idx] = next_obs[i, :self.obs_dim]
            self._dones[idx]    = dones[i]
            self._tgts[idx]     = tgts[i, :self.obs_dim]
            self._ptr += 1
            self._n_stored = min(self._n_stored + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self._n_stored < batch_size:
            return None
        idx = np.random.randint(0, self._n_stored, size=batch_size)
        def t(x): return torch.FloatTensor(x[idx]).to(device)
        return t(self._obs), t(self._acts), t(self._rews), \
               t(self._next_obs), t(self._dones), t(self._tgts)

# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma(cond) * x + self.beta(cond)

# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Actor (SAC + FiLM)
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        # Store dims so KL-distillation can reconstruct an identical copy
        self.hidden_dim = hidden_dim
        self.cond_dim   = cond_dim
        self.state_enc  = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.target_enc = nn.Sequential(
            nn.Linear(tgt_dim, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, cond_dim))
        self.film         = FiLM(cond_dim, hidden_dim)
        self.mean_head    = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, target: torch.Tensor):
        h    = self.state_enc(obs)
        cond = self.target_enc(target)
        h    = self.film(h, cond)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        mean    = torch.nan_to_num(mean,    nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=self.LOG_STD_MIN)
        return mean, log_std

    def get_action_and_logit(self, obs: torch.Tensor, target: torch.Tensor):
        mean, log_std = self.forward(obs, target)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        # v24 FIX: SiliQunEnv action space is [-1, 1] natively.
        # Do NOT multiply by pi — that was for the old QuantumCircuitEnv.
        action   = y_t
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

# ─────────────────────────────────────────────────────────────────────────────
# Goal-conditioned Critic (twin Q)
# ─────────────────────────────────────────────────────────────────────────────
class GoalConditionedCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, tgt_dim: int,
                 hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        in_dim = obs_dim + act_dim + tgt_dim
        self.q1_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.q2_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action, target):
        x = torch.cat([obs, action, target], dim=-1)
        return self.q1_enc(x), self.q2_enc(x)

# ─────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, target_dim: int,
                 alpha: float = 0.05, lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4, tau: float = 0.005,
                 hidden_dim: int = 256, cond_dim: int = 64,
                 target_name: str = "GHZ", n_qubits: int = 2):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim  # ERR-006 fix: store for KL distillation
        self.cond_dim   = cond_dim    # ERR-006 fix: store for KL distillation
        tgt_dim = target_dim * 2  # real + imag

        self.log_alpha    = torch.tensor(math.log(alpha), requires_grad=True,
                                         device=DEVICE)
        # v22 P1: raise target entropy for hard cells to prevent alpha collapse
        _is_hard = (target_name in HARD_CELL_TARGETS
                    and n_qubits >= HARD_CELL_MIN_QUBITS)
        _entropy_scale = HARD_TARGET_ENTROPY_SCALE if _is_hard else -1.0
        self.target_entropy = _entropy_scale * float(act_dim)
        if _is_hard:
            log.info(f"  [SAC v22 P1] Hard cell {n_qubits}Q/{target_name}: "
                     f"target_entropy={self.target_entropy:.2f} "
                     f"(scale={_entropy_scale})")
        self.tau          = tau

        self.actor  = GoalConditionedActor(obs_dim, act_dim, tgt_dim,
                                           hidden_dim, cond_dim).to(DEVICE)
        self.critic = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                            hidden_dim, cond_dim).to(DEVICE)
        self.critic_target = GoalConditionedCritic(obs_dim, act_dim, tgt_dim,
                                                   hidden_dim, cond_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha],          lr=lr_actor)

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, target: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            t = torch.FloatTensor(target).unsqueeze(0).to(DEVICE)
            action, _, mean_action = self.actor.get_action_and_logit(o, t)
        a = mean_action if deterministic else action
        return a.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int,
               der_buffer: Optional[DERPlusPlusBuffer] = None,
               der_batch_size: int = 64):
        if len(buffer) < batch_size:
            return
        obs, acts, rews, next_obs, dones, tgts = buffer.sample(batch_size)
        # Critic update
        with torch.no_grad():
            next_a, next_lp, _ = self.actor.get_action_and_logit(next_obs, tgts)
            q1_t, q2_t = self.critic_target(next_obs, next_a, tgts)
            q_t  = torch.min(q1_t, q2_t) - self.alpha * next_lp
            y    = rews + (1.0 - dones) * 0.99 * q_t
        q1, q2 = self.critic(obs, acts, tgts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        # DER++ auxiliary loss
        if der_buffer is not None and der_buffer._n_stored >= der_batch_size:
            der_sample = der_buffer.sample(der_batch_size, DEVICE)
            if der_sample is not None:
                d_obs, d_acts, d_rews, d_next_obs, d_dones, d_tgts = der_sample
                if d_obs.shape[1] != obs.shape[1]:
                    pad = obs.shape[1] - d_obs.shape[1]
                    if pad > 0:
                        d_obs      = F.pad(d_obs,      (0, pad))
                        d_next_obs = F.pad(d_next_obs, (0, pad))
                    else:
                        d_obs      = d_obs[:, :obs.shape[1]]
                        d_next_obs = d_next_obs[:, :obs.shape[1]]
                if d_tgts.shape[1] != tgts.shape[1]:
                    pad = tgts.shape[1] - d_tgts.shape[1]
                    if pad > 0:
                        d_tgts = F.pad(d_tgts, (0, pad))
                    else:
                        d_tgts = d_tgts[:, :tgts.shape[1]]
                with torch.no_grad():
                    d_next_a, d_next_lp, _ = self.actor.get_action_and_logit(
                        d_next_obs, d_tgts)
                    dq1_t, dq2_t = self.critic_target(d_next_obs, d_next_a, d_tgts)
                    dy = d_rews + (1.0 - d_dones) * 0.99 * (
                        torch.min(dq1_t, dq2_t) - self.alpha * d_next_lp)
                dq1, dq2 = self.critic(d_obs, d_acts, d_tgts)
                critic_loss = critic_loss + 0.5 * (F.mse_loss(dq1, dy) +
                                                    F.mse_loss(dq2, dy))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        # Actor update
        a_new, log_pi, _ = self.actor.get_action_and_logit(obs, tgts)
        q1_new, q2_new   = self.critic(obs, a_new, tgts)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Soft target update
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

# ─────────────────────────────────────────────────────────────────────────────
# ACC — Adaptive Convergence Controller (unchanged from v14)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from acc import AdaptiveConvergenceController, ACCDecision, StopReason
except ImportError:
    # Inline fallback if acc.py not on PYTHONPATH
    class StopReason(Enum):
        NONE             = auto()
        THRESHOLD_MET    = auto()
        FLAT_UNREACHABLE = auto()
        CONVERGED        = auto()
        MAX_BUDGET       = auto()

    @dataclass
    class ACCDecision:
        stop:               bool
        reason:             "StopReason"
        step:               int
        best_F:             float
        F_inf:              Optional[float] = None
        T_star:             Optional[int]   = None
        recommended_budget: Optional[int]   = None

    class AdaptiveConvergenceController:
        def __init__(self, F_threshold=0.99, max_budget=500_000,
                     min_points=5, r2_min=0.70, safety_margin=0.20):
            self.F_threshold   = F_threshold
            self.max_budget    = max_budget
            self.min_points    = min_points
            self.r2_min        = r2_min
            self.safety_margin = safety_margin
            self._steps: List[int]   = []
            self._fids:  List[float] = []

        def update(self, step: int, best_F: float) -> ACCDecision:
            self._steps.append(step)
            self._fids.append(best_F)
            if best_F >= self.F_threshold:
                return ACCDecision(True, StopReason.THRESHOLD_MET, step, best_F,
                                   F_inf=best_F, T_star=step,
                                   recommended_budget=step)
            if len(self._steps) < self.min_points:
                return ACCDecision(False, StopReason.NONE, step, best_F)
            # Saturating exponential fit: F(t) = F_inf * (1 - exp(-t/tau))
            try:
                from scipy.optimize import curve_fit
                def sat_exp(t, F_inf, tau):
                    return F_inf * (1.0 - np.exp(-np.array(t) / (tau + 1e-9)))
                t_arr = np.array(self._steps, dtype=np.float64)
                f_arr = np.array(self._fids,  dtype=np.float64)
                p0    = [max(f_arr) * 1.1, t_arr[-1] / 2]
                popt, _ = curve_fit(sat_exp, t_arr, f_arr, p0=p0,
                                    maxfev=2000, bounds=([0, 1], [1.5, 1e9]))
                F_inf_est, tau_est = popt
                # R² check
                f_pred = sat_exp(t_arr, *popt)
                ss_res = np.sum((f_arr - f_pred) ** 2)
                ss_tot = np.sum((f_arr - f_arr.mean()) ** 2) + 1e-12
                r2     = 1 - ss_res / ss_tot
                if r2 < self.r2_min:
                    return ACCDecision(False, StopReason.NONE, step, best_F,
                                       F_inf=float(F_inf_est))
                if F_inf_est < self.F_threshold * (1 - self.safety_margin):
                    # v22 P2: guard — never terminate hard cells before ACC_MIN_STEPS_HARD
                    _is_hard_cell = (getattr(self, '_target_name', '') in HARD_CELL_TARGETS
                                     and getattr(self, '_n_qubits', 0) >= HARD_CELL_MIN_QUBITS)
                    if _is_hard_cell and step < ACC_MIN_STEPS_HARD:
                        log.info(f"    [ACC v22 P2] Hard cell guard: step={step:,} < "
                                 f"{ACC_MIN_STEPS_HARD:,} — suppressing FLAT_UNREACHABLE "
                                 f"(F_inf_est={F_inf_est:.4f})")
                        return ACCDecision(False, StopReason.NONE, step, best_F,
                                          F_inf=float(F_inf_est))
                    return ACCDecision(True, StopReason.FLAT_UNREACHABLE, step,
                                       best_F, F_inf=float(F_inf_est))
                # Predict T* (steps to reach F_threshold)
                if F_inf_est > self.F_threshold:
                    ratio = 1.0 - self.F_threshold / F_inf_est
                    if ratio > 0:
                        T_star = int(-tau_est * math.log(ratio))
                        rec_budget = int(T_star * (1 + self.safety_margin))
                        if step >= T_star:
                            return ACCDecision(True, StopReason.CONVERGED, step,
                                               best_F, F_inf=float(F_inf_est),
                                               T_star=T_star,
                                               recommended_budget=rec_budget)
                        return ACCDecision(False, StopReason.NONE, step, best_F,
                                           F_inf=float(F_inf_est), T_star=T_star,
                                           recommended_budget=rec_budget)
            except Exception:
                pass
            return ACCDecision(False, StopReason.NONE, step, best_F)

# ─────────────────────────────────────────────────────────────────────────────
# SLM — Small Language Model correction function (v19: ERL-augmented)
# ERL augments SLM by providing a structured reflection Δ from Qwen3-4B
# between the first attempt y⁽¹⁾ and the second attempt y⁽²⁾.
# When reflection is None: standard random probing (first attempt y⁽¹⁾)
# When reflection=Δ: Qwen3-4B-guided perturbation (second attempt y⁽²⁾)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# v22 P7: Amplitude-redistribution perturbation for W/Cluster states
# ─────────────────────────────────────────────────────────────────────────────
def amplitude_redistribution_perturbation(
    agent: "SACAgent",
    target_state: str,
    n_qubits: int,
    strength: float = 0.3,
) -> bool:
    """
    v22 P7: Structured perturbation for W-state and Cluster-state synthesis.

    Instead of Gaussian noise on policy parameters, this function applies a
    targeted perturbation to the actor's output layer biases that directly
    addresses the amplitude imbalance problem:

    For W-state: the target is uniform amplitude across all single-excitation
    basis states. The perturbation adds a bias toward the under-represented
    basis states by modifying the actor's final layer to increase entropy in
    the action dimensions corresponding to qubit rotations.

    For Cluster-state: the perturbation increases the bias toward CZ-type
    actions (high-entanglement, two-qubit gates) by scaling up the action
    dimensions that correspond to two-qubit interaction pulses.

    Returns True if perturbation was applied, False if not applicable.
    """
    _target_norm = target_state.lower().replace("-", "_").replace(" ", "_")
    _is_w = _target_norm in ("w", "w_state")
    _is_cluster = _target_norm in ("cluster", "cluster_linear", "cluster_state")
    _is_complete_graph = _target_norm in ("complete_graph", "completegraph")
    _is_ghz_w = _target_norm in ("ghz_w_hybrid",)

    if not (_is_w or _is_cluster or _is_complete_graph or _is_ghz_w) or n_qubits < 3:
        return False

    try:
        with torch.no_grad():
            # Get the final linear layer of the actor (mean head)
            actor = agent.actor
            # Find the last Linear layer in the mean network
            mean_layers = [m for m in actor.modules()
                           if isinstance(m, torch.nn.Linear)]
            if not mean_layers:
                return False
            final_layer = mean_layers[-1]
            act_dim = final_layer.out_features

            if _is_w:
                # W-state: add entropy-increasing noise to all action dims equally
                # This prevents the policy from committing to a single qubit rotation
                noise = torch.randn_like(final_layer.bias) * strength * 0.1
                final_layer.bias.add_(noise)
                # Also reduce the weight magnitudes slightly to flatten the distribution
                final_layer.weight.mul_(1.0 - strength * 0.05)
                log.info(f"  [v22 P7] W-state amplitude redistribution: "
                         f"entropy noise added (strength={strength:.2f})")

            elif _is_cluster:
                # Cluster-state: boost the last act_dim//2 action dims
                # (assumed to correspond to two-qubit interaction pulses)
                half = act_dim // 2
                boost = torch.zeros_like(final_layer.bias)
                boost[half:] = strength * 0.15  # boost two-qubit action dims
                final_layer.bias.add_(boost)
                log.info(f"  [v22 P7] Cluster-state amplitude redistribution: "
                         f"two-qubit action boost (strength={strength:.2f})")
            elif _is_complete_graph:
                half = act_dim // 2
                boost = torch.zeros_like(final_layer.bias)
                boost[half:] = strength * 0.20
                final_layer.bias.add_(boost)
                noise = torch.randn(half, device=final_layer.bias.device) * strength * 0.05
                final_layer.bias[:half].add_(noise)
                log.info(f"  [v23 N5] CompleteGraph amplitude redistribution: full two-qubit boost (strength={strength:.2f})")
            elif _is_ghz_w:
                noise = torch.randn_like(final_layer.bias) * strength * 0.15
                noise[::2] *= -1
                final_layer.bias.add_(noise)
                log.info(f"  [v23 N5] GHZ-W Hybrid phase perturbation: alternating noise (strength={strength:.2f})")

        return True
    except Exception as e:
        log.warning(f"  [v22 P7] Amplitude redistribution failed: {e}")
        return False

def slm_correction(
    agent:           SACAgent,
    env,
    target_vec:      np.ndarray,
    best_F_current:  float = 0.0,
    n_probe:         int   = 300,
    brfd:            Optional[BRFD]          = None,
    reflection:      Optional[np.ndarray]    = None,
    memory:          Optional[EpisodeMemory] = None,
    slm_reflector:   Optional["SLMReflector"] = None,
) -> Tuple[bool, float, float, bool]:
    """
    ERL-augmented SLM correction (v19).

    SLM = Small Language Model (Qwen3-4B).
    ERL augments the SLM by inserting a reflection step between attempts:

      y⁽¹⁾  First attempt  — reflection=None, standard random probing
      Δ     SLM reflection — Qwen3-4B generates Δ (via SLMReflector)
      y⁽²⁾  Second attempt — reflection=Δ, LLM-guided perturbation

    When reflection=Δ is provided, sigma and strategy are taken from
    slm_reflector.last_sigma_scale and slm_reflector.last_strategy
    (set by Qwen3-4B in the preceding compute_delta() call).
    EpisodeMemory provides cross-episode strategy reuse.

    Returns (improved: bool, sigma_used: float, best_F_probe: float,
             bias_high_entropy: bool)
    """
    obs = env.reset()
    best_score       = -1.0
    best_action      = None
    best_F_probe     = 0.0
    bias_high_entropy = False

    # ── Determine perturbation strategy ──────────────────────────────────────
    if reflection is not None:
        # ERL second attempt y⁽²⁾: Qwen3-4B-guided perturbation
        fidelity_gap = float(reflection[0])
        entropy_norm = float(reflection[3])

        # Sigma from LLM guidance (sigma_scale * base), or fidelity-gap heuristic
        if slm_reflector is not None:
            sigma_base = float(np.clip(
                (0.2 + 0.6 * fidelity_gap) * slm_reflector.last_sigma_scale,
                0.05, 2.0))
            llm_strategy = slm_reflector.last_strategy
        else:
            sigma_base   = 0.2 + 0.6 * fidelity_gap
            llm_strategy = "gaussian"

        # Check EpisodeMemory for similar past strategies
        if memory is not None and len(memory) > 0:
            similar = memory.retrieve_similar(reflection, top_k=3)
            if similar:
                total_imp = sum(e.improvement for e in similar) + 1e-12
                mem_sigma = sum(e.sigma * e.improvement for e in similar) / total_imp
                # Blend LLM sigma (60%) with memory sigma (40%)
                sigma_base = 0.6 * sigma_base + 0.4 * mem_sigma
                bias_high_entropy = any(e.bias_high_entropy for e in similar)
                log.debug(f"    ERL-SLM y⁽²⁾: blended sigma={sigma_base:.3f} "
                          f"bias_entropy={bias_high_entropy}")

        # LLM-directed strategy overrides
        if llm_strategy == "laplace" or entropy_norm < 0.25:
            bias_high_entropy = True
        elif llm_strategy == "directional":
            sigma_base = float(np.clip(sigma_base * 1.5, 0.05, 2.0))

        sigma = float(np.clip(sigma_base, 0.05, 1.5))
        log.info(f"    ERL-SLM y⁽²⁾: sigma={sigma:.3f} strategy={llm_strategy} "
                 f"bias_entropy={bias_high_entropy}")
    else:
        # Standard first attempt y⁽¹⁾: random sigma
        sigma = 0.3 * (1.0 + np.random.rand())
        llm_strategy = "gaussian"

    # ── Probe loop ────────────────────────────────────────────────────────────
    for _ in range(n_probe):
        a = agent.select_action(obs, target_vec, deterministic=False)
        if bias_high_entropy:
            # Sample from a broader distribution to increase exploration
            noise = np.random.laplace(0, sigma, size=a.shape).astype(np.float32)
        else:
            noise = np.random.randn(*a.shape).astype(np.float32) * sigma
        a_probe = np.clip(a + noise, -1.0, 1.0)  # BUG-005 fix: SiliQunEnv action space is [-1,1]
        _, _, _, info = env.step(a_probe)
        F = info.get("F", info.get("fidelity", 0.0))
        score = (brfd.score_probe(F, best_F_probe, 0)
                 if brfd is not None else F)
        if score > best_score:
            best_score   = score
            best_action  = a_probe
            best_F_probe = F

    # ── Actor update if improvement found ────────────────────────────────────
    if best_action is not None and best_F_probe > best_F_current + 1e-4:
        o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        t_t = torch.FloatTensor(target_vec).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(best_action).unsqueeze(0).to(DEVICE)
        mean, log_std = agent.actor(o_t, t_t)
        std  = log_std.exp()
        loss = -torch.distributions.Normal(mean, std).log_prob(
            a_t).sum()  # BUG-005 fix: removed /math.pi (action already in [-1,1])
        agent.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_opt.step()
        return True, sigma, best_F_probe, bias_high_entropy
    return False, sigma, best_F_probe, bias_high_entropy

# ─────────────────────────────────────────────────────────────────────────────
# SDFT — Spectral Decomposition Fidelity Tracker (Stage 2 correction)
# ─────────────────────────────────────────────────────────────────────────────
class SDFT:
    """
    Analyses the fidelity history spectrum to identify dominant stagnation
    frequencies and applies a targeted phase correction to the actor output layer.
    """
    def __init__(self, history_len: int = 30):
        self.history_len = history_len
        self._F_history: deque = deque(maxlen=history_len)

    def record(self, best_F: float):
        self._F_history.append(best_F)

    def correct(self, agent: SACAgent, env, target_vec: np.ndarray,
                n_probe: int = 100) -> bool:
        """
        Identify the dominant stagnation mode and apply a targeted gradient
        correction to the actor's mean_head to escape the plateau.
        Returns True if correction was applied.
        """
        if len(self._F_history) < self.history_len // 2:
            return False
        arr = np.array(self._F_history, dtype=np.float64)
        spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
        # Find dominant non-DC frequency
        dominant_freq_idx = int(np.argmax(spectrum[1:]) + 1)
        # Use the dominant frequency to set a perturbation phase
        phase = 2 * math.pi * dominant_freq_idx / len(arr)
        obs = env.reset()
        best_F_corr = 0.0
        best_action = None
        for i in range(n_probe):
            a = agent.select_action(obs, target_vec, deterministic=True)
            # Phase-directed perturbation
            perturb = math.sin(phase * i) * 0.5
            a_probe = np.clip(a + perturb, -1.0, 1.0)  # BUG-006 fix: SiliQunEnv action space is [-1,1]
            _, _, _, info = env.step(a_probe)
            F = info.get("F", info.get("fidelity", 0.0))
            if F > best_F_corr:
                best_F_corr = F
                best_action = a_probe
        if best_action is not None and best_F_corr > 0.3:
            o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            t_t = torch.FloatTensor(target_vec).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor(best_action).unsqueeze(0).to(DEVICE)
            mean, log_std = agent.actor(o_t, t_t)
            std  = log_std.exp()
            loss = -torch.distributions.Normal(mean, std).log_prob(
                a_t).sum()  # BUG-006 fix: removed /math.pi
            agent.actor_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_opt.step()
            log.info(f"    SDFT correction applied (dominant_freq={dominant_freq_idx}, "
                     f"probe_best_F={best_F_corr:.4f})")
            return True
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TTT — Test-Time Training (Stage 3 correction)
# ─────────────────────────────────────────────────────────────────────────────
class TTT:
    """
    Re-anchors the policy to its historical best trajectory via a short
    supervised gradient update on the actor. This is the most expensive
    stage and is only invoked when SLM and SDFT both fail.
    """
    def __init__(self, n_gradient_steps: int = 50, lr: float = 1e-4):
        self.n_gradient_steps = n_gradient_steps
        self.lr               = lr
        self._best_obs:    Optional[np.ndarray] = None
        self._best_action: Optional[np.ndarray] = None
        self._best_target: Optional[np.ndarray] = None

    def record_best(self, obs: np.ndarray, action: np.ndarray,
                    target: np.ndarray):
        """Call whenever a new best_F is achieved during training."""
        self._best_obs    = obs.copy()
        self._best_action = action.copy()
        self._best_target = target.copy()

    def correct(self, agent: SACAgent) -> bool:
        """
        Apply n_gradient_steps of supervised imitation on the best trajectory.
        Returns True if correction was applied.
        """
        if (self._best_obs is None or self._best_action is None
                or self._best_target is None):
            return False
        o_t = torch.FloatTensor(self._best_obs).unsqueeze(0).to(DEVICE)
        t_t = torch.FloatTensor(self._best_target).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor(self._best_action).unsqueeze(0).to(DEVICE)
        # Temporarily lower LR for fine-grained correction
        orig_lr = agent.actor_opt.param_groups[0]["lr"]
        for pg in agent.actor_opt.param_groups:
            pg["lr"] = self.lr
        for _ in range(self.n_gradient_steps):
            mean, log_std = agent.actor(o_t, t_t)
            std  = log_std.exp()
            loss = -torch.distributions.Normal(mean, std).log_prob(
                a_t).sum()  # BUG-007 fix: removed /math.pi (action already in [-1,1])
            agent.actor_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_opt.step()
        for pg in agent.actor_opt.param_groups:
            pg["lr"] = orig_lr
        log.info(f"    TTT correction applied ({self.n_gradient_steps} steps)")
        return True

# ─────────────────────────────────────────────────────────────────────────────
# ASR — Adaptive Stagnation Recovery (v19: ERL augments SLM)
# SLM = Small Language Model (Qwen3-4B)
# ERL augments SLM: Qwen3-4B generates reflection Δ between y⁽¹⁾ and y⁽²⁾
# Orchestrates ERL-SLM (Stage 1) → SDFT (Stage 2) → TTT (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
class ASR:
    """
    Adaptive Stagnation Recovery (v19) — renamed from ERL cascade.
    Orchestrates the 3-stage reactive correction pipeline:
      Stage 1: ERL-SLM  (ERL augments SLM: Qwen3-4B generates Δ for guided perturbation)
      Stage 2: SDFT     (spectral phase correction)
      Stage 3: TTT      (test-time training on best trajectory)
    Falls through to DEHB Inner if all three fail.

    v19 additions vs v18:
      - SLMReflector (Qwen3-4B) generates structured Δ via ERL reflection
      - slm_correction() called twice: first attempt y⁽¹⁾ (random probing)
        then second attempt y⁽²⁾ (Δ-guided) if first fails — ERL paper loop
      - EpisodeMemory stores successful (Δ, strategy, improvement) triples
        for reuse in future activations (cross-episode memory)
    """
    # ── ASR constants (inherited from v18) ───────────────────────────────────────────────────────────────────
    COOLDOWN_STEPS  = 50_000   # Fix 1: min steps between ASR activations
    GAIN_THRESH     = 0.002    # Fix 2: min ΔF gain to justify escalation
    GRAD_VAR_THRESH = 1e-4     # Fix 3: min grad variance to fire DEHB Inner
    KL_DISTILL_STEPS = 500     # v19: internalization gradient steps
    KL_DISTILL_LR    = 5e-5   # v19: internalization learning rate

    def __init__(self, sdft: SDFT, ttt: TTT, brfd: BRFD, curriculum_ctrl=None):
        self.sdft            = sdft
        self.curriculum_ctrl = curriculum_ctrl  # v25 P4
        self.ttt        = ttt
        self.brfd       = brfd
        # v19: ERL augments SLM — SLMReflector (Qwen3-4B) generates Δ
        self.reflection = SLMReflector()   # replaces deterministic ReflectionModule
        self.memory     = EpisodeMemory()
        self.n_activations       = 0
        self.n_slm_success       = 0
        self.n_erl_slm_success   = 0  # v19: second-attempt successes
        self.n_sdft_success      = 0
        self.n_ttt_success       = 0
        self.n_full_fail         = 0
        # Fix 1: cooldown tracking
        self._last_activation_step: int = -self.COOLDOWN_STEPS
        # Fix 2: gain tracking
        self._best_F_at_trigger: float = 0.0
        # Fix 4: pre-ASR checkpoint (state_dict copies)
        self._pre_asr_actor_sd  = None
        self._pre_asr_critic_sd = None

    def _collect_agent_stats(
        self, agent: SACAgent, best_F: float, best_F_prev: float,
        step: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Collect training statistics for SLMReflector.compute_delta() (Qwen3-4B).
        Returns (fidelity_gap, grad_variance, noise_sensitivity,
                 action_entropy, convergence_rate).
        """
        fidelity_gap = float(np.clip(1.0 - best_F, 0.0, 1.0))

        # Gradient variance across actor parameters
        grad_norms: List[float] = []
        for p in agent.actor.parameters():
            if p.grad is not None:
                grad_norms.append(float(p.grad.data.norm(2).item()))
        grad_variance = float(np.var(grad_norms)) if grad_norms else 0.0

        # Noise sensitivity: std of recent log_alpha (proxy for entropy sensitivity)
        try:
            noise_sensitivity = float(torch.exp(agent.log_alpha).item())
            noise_sensitivity = float(np.clip(noise_sensitivity / 0.5, 0.0, 1.0))
        except Exception:
            noise_sensitivity = 0.3

        # Action entropy: current SAC alpha as proxy
        try:
            action_entropy = float(torch.exp(agent.log_alpha).item())
            action_entropy = float(np.clip(action_entropy / 1.0, 0.0, 1.0))
        except Exception:
            action_entropy = 0.5

        # Convergence rate: ΔF over last window (normalised)
        delta_F = float(np.clip(best_F - best_F_prev, 0.0, 0.5))
        convergence_rate = float(np.clip(delta_F / 0.5, 0.0, 1.0))

        return fidelity_gap, grad_variance, noise_sensitivity, \
               action_entropy, convergence_rate

    def respond(
        self,
        bus:        SignalBus,
        agent:      SACAgent,
        env,
        target_vec: np.ndarray,
        best_F:     float,
        step:       int,
        best_F_prev: float = 0.0,  # v19: for convergence_rate in Δ
        target_state: str = "GHZ",  # v22 P4/P7: for amplitude redistribution
        n_qubits: int = 2,          # v22 P4/P7
    ):
        """
        v19 ASR respond() — ERL-augmented SLM + v18 fixes:
          Fix 1: Hard cooldown gate
          Fix 2: Gain-cost gate
          Fix 3: Gradient variance check (DEHB Inner)
          Fix 4: Pre-ASR checkpoint restore
          v19:   ReflectionModule + EpisodeMemory + ERL two-attempt SLM loop
        """
        if not bus.has(Signal.STAGNATION):
            return
        payload = bus.consume(Signal.STAGNATION)

        # ── Fix 1: Hard cooldown gate ───────────────────────────────────────────────────────────────────
        steps_since_last = step - self._last_activation_step
        if steps_since_last < self.COOLDOWN_STEPS:
            log.info(f"    ASR COOLDOWN GATE: skipping (only {steps_since_last:,} steps "
                     f"since last activation, need {self.COOLDOWN_STEPS:,})")
            return
        self._last_activation_step    = step
        self._best_F_at_trigger    = best_F
        self._target_state = target_state  # v22 P4/P7
        self._n_qubits     = n_qubits      # v22 P4/P7
        self.n_activations += 1
        log.info(f"    ASR activated (activation #{self.n_activations}) "
                 f"@ step={step:,} best_F={best_F:.4f} [v19: ASR+ERL-SLM]")

        # ── Fix 4: Save pre-ASR checkpoint ───────────────────────────────────────────────────────────────────
        import copy
        self._pre_asr_actor_sd  = copy.deepcopy(agent.actor.state_dict())
        self._pre_asr_critic_sd = copy.deepcopy(agent.critic.state_dict())
        log.info(f"    ASR v19: pre-ASR checkpoint saved @ step={step:,}")

        # ── v19: ERL reflection — Qwen3-4B (SLMReflector) generates structured Δ ─────────────────────────────────────────────────────
        fidelity_gap, grad_var, noise_sens, entropy, conv_rate = \
            self._collect_agent_stats(agent, best_F, best_F_prev, step)
        delta = self.reflection.compute_delta(
            fidelity_gap, grad_var, noise_sens, entropy, conv_rate,
            n_qubits=getattr(env, 'n_qubits', 2),
            target_name=getattr(env, 'target_name', 'unknown'),
            step=step,
            stag_steps=step - self._last_activation_step,
        )
        stag_type = self.reflection.diagnose(delta)
        log.info(f"    ASR v19 (ERL-SLM Qwen3-4B): Δ={np.round(delta, 3).tolist()} "
                 f"stagnation_type={stag_type.name} "
                 f"sigma_scale={self.reflection.last_sigma_scale:.3f} "
                 f"strategy={self.reflection.last_strategy}")
        # ── v25 P4: Apply noise_delta from SLM reflection ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        _noise_delta = getattr(self.reflection, 'last_noise_delta', 0)
        if _noise_delta != 0 and self.curriculum_ctrl is not None:
            _new_stage = self.curriculum_ctrl.current_stage + _noise_delta
            _rebuilt = self.curriculum_ctrl.set_stage(_new_stage)
            if _rebuilt:
                log.info(
                    f"    [v25 P4] ERL-SLM noise_delta={_noise_delta:+d} applied: "
                    f"noise_stage -> {self.curriculum_ctrl.current_stage}"
                )

        # ── Stage 1a: ERL-SLM first attempt y⁽¹⁾ (standard random probing) ───────────────────────────────────────────────────────────────────
        n_probes = self.brfd.suggest_probe_count(best_F, step)
        slm_ok, sigma1, probe_F1, _ = slm_correction(
            agent, env, target_vec,
            best_F_current=best_F,
            n_probe=n_probes,
            brfd=self.brfd,
            reflection=None,   # first attempt: random
            memory=None,
        )
        self.brfd.update(improved=slm_ok)
        if slm_ok:
            bus.emit(Signal.SLM_CORRECTED, payload={"step": step})
            self.n_slm_success += 1
            log.info(f"    ASR Stage 1a (ERL-SLM first attempt) SUCCESS @ step={step:,}")
            return

        # ── Stage 1b: ERL-SLM multi-call (v22 P4) — 3 suggestions, pick best ──────────────────────────────────────────────────────────────────
        log.info(f"    ASR Stage 1a FAILED — ERL multi-call Stage 1b (3 attempts)")
        # v22 P4: Try up to 3 SLM suggestions and pick the best one
        _best_stage1b_F = best_F
        _best_stage1b_actor_sd = None
        _best_stage1b_critic_sd = None
        _stage1b_success = False
        for _attempt in range(3):
            # Restore pre-ASR checkpoint before each attempt
            agent.actor.load_state_dict(self._pre_asr_actor_sd)
            agent.critic.load_state_dict(self._pre_asr_critic_sd)
            # v22 P7: Apply amplitude redistribution perturbation for W/Cluster
            _p7_applied = amplitude_redistribution_perturbation(
                agent, self._target_state, self._n_qubits,
                strength=0.2 + 0.1 * _attempt)
            slm_ok_b, sigma_b, probe_F_b, _ = slm_correction(
                agent, env, target_vec,
                best_F_current=best_F,
                n_probe=n_probes,
                brfd=self.brfd,
                reflection=delta,
                memory=self.memory,
                slm_reflector=self.reflection,
            )
            log.info(f"    ASR Stage 1b attempt {_attempt+1}/3: "
                     f"probe_F={probe_F_b:.4f} p7={_p7_applied}")
            if probe_F_b > _best_stage1b_F:
                _best_stage1b_F = probe_F_b
                _best_stage1b_actor_sd = {k: v.clone()
                    for k, v in agent.actor.state_dict().items()}
                _best_stage1b_critic_sd = {k: v.clone()
                    for k, v in agent.critic.state_dict().items()}
                if slm_ok_b:
                    _stage1b_success = True
                    break  # early exit on clear improvement
        # Restore best Stage 1b result
        if _best_stage1b_actor_sd is not None:
            agent.actor.load_state_dict(_best_stage1b_actor_sd)
            agent.critic.load_state_dict(_best_stage1b_critic_sd)
        if _stage1b_success or (_best_stage1b_F - best_F) > 0.002:
            self.brfd.update(improved=True)
            bus.emit(Signal.SLM_CORRECTED, payload={"step": step})
            self.n_slm_success += 1
            log.info(f"    ASR Stage 1b SUCCESS (best_F={_best_stage1b_F:.4f})")
            return
        # Restore pre-ASR checkpoint before Stage 2
        agent.actor.load_state_dict(self._pre_asr_actor_sd)
        agent.critic.load_state_dict(self._pre_asr_critic_sd)

        slm_ok2, sigma2, probe_F2, bias_ent = slm_correction(
            agent, env, target_vec,
            best_F_current=best_F,
            n_probe=n_probes,
            brfd=self.brfd,
            reflection=delta,           # second attempt: Δ-guided
            memory=self.memory,
            slm_reflector=self.reflection,  # Qwen3-4B LLM guidance
        )
        self.brfd.update(improved=slm_ok2)
        if slm_ok2:
            improvement = probe_F2 - best_F
            self.memory.store(delta, sigma2, bias_ent, improvement)
            bus.emit(Signal.SLM_CORRECTED, payload={"step": step, "erl_second_attempt": True})
            self.n_erl_slm_success += 1
            log.info(f"    ASR Stage 1b (ERL-SLM second attempt) SUCCESS "
                     f"@ step={step:,} probe_F={probe_F2:.4f} sigma={sigma2:.3f}")
            return

        # Both SLM attempts failed
        delta_F = best_F - self._best_F_at_trigger
        bus.emit(Signal.SLM_FAILED, payload={"step": step})
        if delta_F > self.GAIN_THRESH:
            log.info(f"    ASR Stage 1 (ERL-SLM) FAILED but ΔF={delta_F:.4f} > {self.GAIN_THRESH} "
                     f"— gain-cost gate: skipping SDFT, waiting for next window")
            return
        log.info(f"    ASR Stage 1 (ERL-SLM) FAILED — escalating to SDFT "
                 f"(ΔF={delta_F:.4f} ≤ {self.GAIN_THRESH})")

        # ── Fix 4: Restore pre-ASR checkpoint before SDFT ───────────────────────────────────────────────────────────────────
        agent.actor.load_state_dict(self._pre_asr_actor_sd)
        agent.critic.load_state_dict(self._pre_asr_critic_sd)
        log.info(f"    ASR v19: pre-ASR checkpoint restored before SDFT")

        # ── Stage 2: SDFT ────────────────────────────────────────────────────────────────────────────────
        sdft_ok = self.sdft.correct(agent, env, target_vec)
        if sdft_ok:
            bus.emit(Signal.SDFT_CORRECTED, payload={"step": step})
            self.n_sdft_success += 1
            return

        # ── Fix 2: Gain-cost check before TTT ───────────────────────────────────────────────────────────────────
        delta_F2 = best_F - self._best_F_at_trigger
        bus.emit(Signal.SDFT_FAILED, payload={"step": step})
        if delta_F2 > self.GAIN_THRESH:
            log.info(f"    ASR Stage 2 (SDFT) FAILED but ΔF={delta_F2:.4f} > {self.GAIN_THRESH} "
                     f"— gain-cost gate: skipping TTT")
            return
        log.info(f"    ASR Stage 2 (SDFT) FAILED — escalating to TTT "
                 f"(ΔF={delta_F2:.4f} ≤ {self.GAIN_THRESH})")

        # ── Fix 4: Restore pre-ASR checkpoint before TTT ───────────────────────────────────────────────────────────────────
        agent.actor.load_state_dict(self._pre_asr_actor_sd)
        agent.critic.load_state_dict(self._pre_asr_critic_sd)
        log.info(f"    ASR v19: pre-ASR checkpoint restored before TTT")

        # ── Stage 3: TTT ────────────────────────────────────────────────────────────────────────────────
        ttt_ok = self.ttt.correct(agent)
        if ttt_ok:
            bus.emit(Signal.TTT_CORRECTED, payload={"step": step})
            self.n_ttt_success += 1
            return

        bus.emit(Signal.TTT_FAILED, payload={"step": step})
        self.n_full_fail += 1
        log.info(f"    ASR Stage 3 (TTT) FAILED — signalling DEHB Inner")
        # v22 P5: FastMix intra-cell update on full ASR failure
        # Reduce alpha_ASR weight when all 3 stages fail; boost alpha_DEHB
        if hasattr(self, '_fastmix') and self._fastmix is not None:
            try:
                self._fastmix.intra_cell_update(
                    failed_component="alpha_ASR",
                    boosted_component="alpha_DEHB",
                    penalty=0.02,
                )
                log.info(f"    [v22 P5] FastMix intra-cell update: "
                         f"alpha_ASR↓ alpha_DEHB↑")
            except Exception as _e:
                log.debug(f"    [v22 P5] FastMix intra-cell update skipped: {_e}")

    def summary(self) -> dict:
        return {
            "activations":       self.n_activations,
            "slm_success":       self.n_slm_success,
            "erl_slm_success":   self.n_erl_slm_success,
            "sdft_success":      self.n_sdft_success,
            "ttt_success":       self.n_ttt_success,
            "full_fail":         self.n_full_fail,
            "memory_size":       len(self.memory),
        }

# Keep ERL as alias for backward compatibility with any serialised checkpoints
ERL = ASR

# ─────────────────────────────────────────────────────────────────────────────
# SiliQunEnvWrapper — adapts SiliQunEnv (Gymnasium API) to the v15 SAC interface
#
# Key differences bridged:
#   1. Gymnasium step() returns (obs, reward, terminated, truncated, info)
#      → wrapped to (obs, reward, done, info)
#   2. Gymnasium reset() returns (obs, info)  → wrapped to obs
#   3. info["fidelity"] → info["F"] alias added
#   4. Target vector extracted from env._target_state (MPS or ndarray)
#   5. Action space [-1, 1] → actor output scaled by π before passing to env
#      (actor still outputs in [-π, π]; wrapper rescales to [-1, 1] for SiliQun)
# ─────────────────────────────────────────────────────────────────────────────

# Add SiliQun to PYTHONPATH
for _p in [
    str(Path.home() / "siliqun"),
    str(Path.home() / "siliqun_v6"),
    str(Path.home() / "siliqun" / "siliqun"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from siliqun.engine.gym_env import SiliQunEnv, make_siliqun_env
    from siliqun.physics.devices.profiles import simos_device
    from siliqun.physics.noise.channels import NoiseParams
    _SILIQUN_AVAILABLE = True
except ImportError:
    _SILIQUN_AVAILABLE = False
    log.warning("SiliQun not found on PYTHONPATH — will raise at env creation")


class SiliQunEnvWrapper:
    """
    Thin wrapper around SiliQunEnv that presents the same interface
    expected by the v15 SAC training loop:
      - reset() → np.ndarray  (obs only)
      - step(action) → (obs, reward, done, info)  (4-tuple)
      - env.target  → complex128 ndarray of shape (2**n,)
      - env.observation_space, env.action_space  (passthrough)
      - info["F"] and info["fidelity"] both available
    """

    # Map v15 target names → SiliQun target names
    _TARGET_MAP = {
        "GHZ":          "ghz",
        "ghz":          "ghz",
        "W":            "w",
        "w":            "w",
        "Cluster":      "cluster_linear",
        "cluster":      "cluster_linear",
        "cluster_linear": "cluster_linear",
        "Dicke-k3":     "dicke_k3",
        "Dicke_k3":     "dicke_k3",
        "dicke-k3":     "dicke_k3",
        "dicke_k3":     "dicke_k3",
        # v23 N1/N2/N3: hard state families (injected as np.ndarray)
        "CompleteGraph":  "complete_graph",
        "complete_graph": "complete_graph",
        "completegraph":  "complete_graph",
        "Dicke-k2":       "dicke_k2",
        "Dicke_k2":       "dicke_k2",
        "dicke-k2":       "dicke_k2",
        "dicke_k2":       "dicke_k2",
        "GHZ_W_Hybrid":   "ghz_w_hybrid",
        "ghz_w_hybrid":   "ghz_w_hybrid",
        "GHZ-W-Hybrid":   "ghz_w_hybrid",
        "ghz-w-hybrid":   "ghz_w_hybrid",
    }

    def __init__(self, n_qubits: int, target_state: str,
                 noise_stage: int = 5, max_ep_steps: int = 200,
                 reward_weights: Optional[dict] = None, seed: int = 42):
        if not _SILIQUN_AVAILABLE:
            raise ImportError(
                "SiliQun package not found. Install it with: "
                "pip install -e ~/siliqun")
        # v23 N6: check if this is a v23 hard target — if so, build the state vector
        # and pass it directly to make_siliqun_env as a np.ndarray
        _v23_sv = _build_v23_target_sv(target_state, n_qubits)
        siliqun_target = self._TARGET_MAP.get(target_state, target_state.lower())
        # FIX-5 (HB-81): For v23 hard targets (complete_graph, dicke_k2, ghz_w_hybrid),
        # pass the pre-built numpy state vector directly to make_siliqun_env so that
        # SiliQunEnv does not raise "Unknown target state: <name>".
        _siliqun_target_arg = _v23_sv if _v23_sv is not None else siliqun_target
        # Fix: Dicke-k3 requires k <= n_qubits; cap k dynamically
        import re as _re
        _dk_match = _re.match(r"dicke_k(\d+)$", siliqun_target)
        if _dk_match:
            _k = int(_dk_match.group(1))
            _k_capped = min(_k, n_qubits)
            siliqun_target = f"dicke_k{_k_capped}"
        # Map noise_stage (0-10) to SiliQun noise bool + sim_mode
        noise_enabled = noise_stage > 0
        # Use MPO for density-matrix noise, MPS for pure-state, SV for large
        if n_qubits >= 8:
            sim_mode = "sv"
        elif noise_enabled:
            sim_mode = "mpo"   # exact Lindblad noise for SiMOS
        else:
            sim_mode = "mps"
        # ── SiMOS practical noise profile ──────────────────────────────────
        # T2* = 20 µs (Intel/UNSW SiMOS), T1 = 10 s, charge noise 2 µV/√Hz
        # Single-qubit gate fidelity ~99.5% (error 5e-3), CZ fidelity ~99.0%
        # Readout fidelity 98.5%, exchange gate 50 ns
        # Calibrated to Veldhorst et al. (2015) and Yang et al. (2020)
        _simos_noise = NoiseParams(
            t1_times=[10.0] * n_qubits,
            t2_star_times=[5e-6] * n_qubits,    # 5 µs (mid-range 1–10 µs)
            t2_echo_times=[100e-6] * n_qubits,  # 100 µs Hahn-echo
            charge_noise_amplitude=2e-6,         # 2 µV/√Hz at 1 Hz
            charge_noise_correlation_length=2,
            thermal_photon_number=0.005,         # ~20 mK base temperature
            measurement_fidelity=0.995,          # 99.5% readout fidelity
            dephasing_model="exponential",
            exchange_frequency=12e6,             # 12 MHz exchange coupling
            pulse_duration=200e-9,               # 200 ns EDSR single-qubit
            idle_duration=50e-9,                 # 50 ns idle
            n_exchange_oscillations=20.0,
            gate_error_rates={
                "single": 5e-3,   # 99.5% single-qubit fidelity
                "two":    1e-2,   # 99.0% CZ / exchange gate fidelity
                "readout": 5e-3,  # 99.5% readout fidelity
            },
        )
        _simos_dev = simos_device(n_qubits)
        _simos_dev.noise_params = _simos_noise  # override with practical levels
        self._env = make_siliqun_env(
            n_qubits=n_qubits,
            device=_simos_dev,
            target=_siliqun_target_arg,  # FIX-5: numpy array for v23 hard targets
            sim_mode=sim_mode,
            noise=noise_enabled,
            max_bond_dim=min(64, 2 ** (n_qubits // 2)),
            max_steps=max_ep_steps,
            fidelity_threshold=0.99,
            reward_type="shaped",
            seed=seed,
            use_gpu=True,
        )
        self._n_qubits = n_qubits
        self._target_state_name = siliqun_target
        self._reward_weights = reward_weights or {}
        # Build target vector (real + imag concatenated)
        self._target_vec = self._extract_target_vec()
        # Expose Gymnasium spaces directly
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def _extract_target_vec(self) -> np.ndarray:
        """Extract dense complex state vector from SiliQun target."""
        ts = self._env._target_state
        if isinstance(ts, np.ndarray):
            sv = ts.flatten().astype(np.complex128)
        else:
            # MPS object — convert to dense
            try:
                sv = ts.to_dense().flatten().astype(np.complex128)
            except Exception:
                # Fallback: build GHZ manually
                dim = 2 ** self._n_qubits
                sv = np.zeros(dim, dtype=np.complex128)
                sv[0] = sv[-1] = 1.0 / math.sqrt(2)
        return sv

    @property
    def target(self) -> np.ndarray:
        """Complex target state vector (2**n,) — used to build target_vec."""
        return self._target_vec

    def reset(self) -> np.ndarray:
        """Reset env and return obs array (no info)."""
        obs, _info = self._env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return (obs, reward, done, info).
        SiliQun action space is [-1, 1]. SAC actor uses tanh squashing which
        already outputs in [-1, 1]. No rescaling needed — clip for safety only.
        v24 FIX: removed /pi rescaling (was corrupting actions; old env used
        [-pi, pi] but SiliQunEnv uses [-1, 1] natively).
        """
        # Clip to [-1, 1] for safety (SAC tanh output is already in this range)
        a_scaled = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self._env.step(a_scaled)
        done = terminated or truncated
        # Apply BRFD-style reward reweighting if weights provided
        F = float(info.get("fidelity", 0.0))
        if self._reward_weights:
            w_F   = self._reward_weights.get("fidelity",  0.65)
            w_S   = self._reward_weights.get("success",   0.20)
            w_str = self._reward_weights.get("structure", 0.10)
            w_log = self._reward_weights.get("log",       0.05)
            reward = (w_F * F
                      + w_S * float(info.get("success", 0.0))
                      + w_str * float((info.get("bond_dims") or [1])[0]) / 32.0
                      + w_log * math.log1p(F))
        # Alias fidelity as "F" for v15 pipeline compatibility
        info["F"] = F
        return np.array(obs, dtype=np.float32), float(reward), bool(done), info

    def __getattr__(self, name):
        """Passthrough to underlying SiliQunEnv for any other attribute."""
        return getattr(self._env, name)

    def rebuild_with_noise_stage(self, new_stage: int) -> None:
        """v25 P2: Rebuild env with new noise_stage (~0.5s for MPO)."""
        if new_stage == getattr(self, '_noise_stage', None):
            return
        import time as _time; _t0 = _time.time()
        old_stage = getattr(self, '_noise_stage', '?')
        self.__init__(
            n_qubits=self._n_qubits,
            target_state=self._target_state_name,
            noise_stage=new_stage,
            max_ep_steps=getattr(self._env, 'max_steps', 200),
            reward_weights=self._reward_weights,
            seed=getattr(self._env, 'seed', 42),
        )
        self._noise_stage = new_stage
        log.info(f"  [v25 NCC] SiliQunEnvWrapper rebuilt: noise_stage {old_stage} -> {new_stage} ({_time.time()-_t0:.2f}s)")

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# v25: NoiseCurriculumController
# ─────────────────────────────────────────────────────────────────────────────
class NoiseCurriculumController:
    """v25 P1: Controls noise_stage_t dynamically. Rebuilds SiliQunEnvWrapper when stage changes."""
    MIN_STAGE = 0
    MAX_STAGE = 5

    def __init__(self, base_noise_stage: int, n_qubits: int, target_state: str,
                 max_ep_steps: int = 200, reward_weights=None, seed: int = 42):
        self.current_stage   = int(max(self.MIN_STAGE, min(self.MAX_STAGE, base_noise_stage)))
        self._n_qubits       = n_qubits
        self._target_state   = target_state
        self._max_ep_steps   = max_ep_steps
        self._reward_weights = reward_weights or {}
        self._seed           = seed
        self._env            = None
        self._n_rebuilds     = 0

    def get_env(self):
        if self._env is None:
            self._build_env(self.current_stage)
        return self._env

    def set_stage(self, new_stage: int) -> bool:
        clamped = int(max(self.MIN_STAGE, min(self.MAX_STAGE, new_stage)))
        if clamped == self.current_stage:
            return False
        log.info(f"  [v25 NCC] noise_stage {self.current_stage} -> {clamped} (rebuild #{self._n_rebuilds+1})")
        self.current_stage = clamped
        if self._env is not None:
            self._env.rebuild_with_noise_stage(clamped)
        self._n_rebuilds += 1
        return True

    def _build_env(self, stage: int) -> None:
        self._env = SiliQunEnvWrapper(
            n_qubits=self._n_qubits, target_state=self._target_state,
            noise_stage=stage, max_ep_steps=self._max_ep_steps,
            reward_weights=self._reward_weights, seed=self._seed,
        )
        self._env._noise_stage = stage
        log.info(f"  [v25 NCC] env built: {self._n_qubits}Q {self._target_state} noise_stage={stage}")

    def summary(self) -> dict:
        return {"current_stage": self.current_stage, "n_rebuilds": self._n_rebuilds}


# Transfer agent (v14 logic preserved)
# ─────────────────────────────────────────────────────────────────────────────
def transfer_agent(source: SACAgent, obs_dim: int, act_dim: int,
                   target_dim: int, **sac_kwargs) -> SACAgent:
    new_agent = SACAgent(obs_dim, act_dim, target_dim, **sac_kwargs)
    tgt_dim_new = target_dim * 2
    def _copy(src_layer, tgt_layer):
        r = min(src_layer.weight.shape[0], tgt_layer.weight.shape[0])
        c = min(src_layer.weight.shape[1], tgt_layer.weight.shape[1])
        tgt_layer.weight.data[:r, :c] = src_layer.weight.data[:r, :c]
        tgt_layer.bias.data[:r]       = src_layer.bias.data[:r]
        if tgt_layer.weight.shape[1] > c:
            tgt_layer.weight.data[:r, c:] = torch.randn_like(
                tgt_layer.weight.data[:r, c:]) * 0.01
        if tgt_layer.weight.shape[0] > r:
            tgt_layer.weight.data[r:, :] = torch.randn_like(
                tgt_layer.weight.data[r:, :]) * 0.01
            tgt_layer.bias.data[r:] = 0.0
    _copy(source.actor.state_enc[0],  new_agent.actor.state_enc[0])
    _copy(source.actor.state_enc[2],  new_agent.actor.state_enc[2])
    _copy(source.actor.target_enc[0], new_agent.actor.target_enc[0])
    _copy(source.actor.target_enc[2], new_agent.actor.target_enc[2])
    _copy(source.actor.film.gamma,    new_agent.actor.film.gamma)
    _copy(source.actor.film.beta,     new_agent.actor.film.beta)
    _copy(source.actor.mean_head,     new_agent.actor.mean_head)
    _copy(source.actor.log_std_head,  new_agent.actor.log_std_head)
    for sq, tq in [(source.critic.q1_enc, new_agent.critic.q1_enc),
                   (source.critic.q2_enc, new_agent.critic.q2_enc)]:
        for i in [0, 2]:
            _copy(sq[i], tq[i])
    log.info(f"  Transfer: {source.obs_dim}→{obs_dim} obs, "
             f"{source.act_dim}→{act_dim} act")
    return new_agent

# ─────────────────────────────────────────────────────────────────────────────
# DEHB config spaces
# ─────────────────────────────────────────────────────────────────────────────
def build_inner_cs(seed: int = 42) -> ConfigurationSpace:
    """Inner DEHB: full HP space (8 dims), used pre-cell."""
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        Float("alpha",          (0.005, 0.50),    log=True,  default=0.05),
        Float("lr_actor",       (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("lr_critic",      (1e-4,  5e-3),    log=True,  default=3e-4),
        Float("tau",            (0.001, 0.05),     log=False, default=0.005),
        Integer("batch_size",   (64,   512),       log=True,  default=256),
        Integer("hidden_dim",   (128,  512),       log=True,  default=256),
        Integer("cond_dim",     (32,   128),       log=True,  default=64),
        Integer("slm_interval",       (5_000, 100_000),  log=True,  default=50_000),
        # v21 Phase 1: LoRA fine-tuning HPs
        Integer("lora_rank",           (4,    16),        log=True,  default=8),
        Float(  "lora_lr",             (5e-5, 5e-4),      log=True,  default=2e-4),
        Integer("lora_trigger_steps",  (20_000, 100_000), log=True,  default=50_000),
        # v22 P3: target entropy scale — DEHB tunes this for hard cells
        Float(  "target_entropy_scale",  (-1.5, -0.2),       log=False, default=-1.0),
        # v25 P3: noise curriculum stage
        Integer("noise_stage_t",         (0, 5),              log=False, default=3),
])
    return cs

def build_reactive_cs(seed: int = 42) -> ConfigurationSpace:
    """Reactive DEHB Inner: narrow 2-dim re-search around current best."""
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        Float("alpha",    (0.005, 0.30), log=True, default=0.05),
        Float("lr_actor", (1e-4,  3e-3), log=True, default=3e-4),
    ])
    return cs

# ─────────────────────────────────────────────────────────────────────────────
# run_cell — single training cell with full v15 reactive pipeline
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# v22 P8: Transfer-learning warm-start
# ─────────────────────────────────────────────────────────────────────────────
def transfer_warm_start(
    agent: "SACAgent",
    donor_agent: "SACAgent",
    blend_ratio: float = 0.7,
) -> bool:
    """
    v22 P8: Warm-start a larger-qubit agent by blending its initial random
    weights with the weights from a smaller-qubit (donor) agent.

    The donor agent is the best-performing agent from the (n-1)-qubit cell
    for the same target state. Since the observation and action spaces differ
    between qubit counts, we copy only the shared hidden layers (not the
    input/output projection layers).

    blend_ratio: fraction of donor weights to use (0.7 = 70% donor, 30% random)

    Returns True if transfer was applied, False if shapes are incompatible.
    """
    try:
        donor_state  = donor_agent.actor.state_dict()
        target_state_dict = agent.actor.state_dict()

        blended = {}
        n_transferred = 0
        for key in target_state_dict:
            if key in donor_state:
                d_shape = donor_state[key].shape
                t_shape = target_state_dict[key].shape
                if d_shape == t_shape:
                    # Same shape: blend donor and random init
                    blended[key] = (blend_ratio * donor_state[key].to(DEVICE)
                                    + (1.0 - blend_ratio) * target_state_dict[key])
                    n_transferred += 1
                else:
                    # Different shape (input/output layers): keep random init
                    blended[key] = target_state_dict[key]
            else:
                blended[key] = target_state_dict[key]

        agent.actor.load_state_dict(blended)
        log.info(f"  [v22 P8] Transfer warm-start: {n_transferred} layers blended "
                 f"(blend_ratio={blend_ratio:.2f})")
        return n_transferred > 0
    except Exception as e:
        log.warning(f"  [v22 P8] Transfer warm-start failed: {e}")
        return False

def run_cell(
    n_qubits:      int,
    target_state:  str,
    seed:          int,
    noise_stage:   int,
    max_ep_steps:  int,
    reward_w_F:    float,
    reward_w_S:    float,
    reward_w_str:  float,
    alpha:         float,
    lr_actor:      float,
    lr_critic:     float,
    tau:           float,
    batch_size:    int,
    hidden_dim:    int,
    cond_dim:      int,
    max_steps:     int,
    init_agent:    Optional[SACAgent] = None,
    plateau_patience: int = 150_000,
    slm_interval:  int = 50_000,
    der_buffer:    Optional[DERPlusPlusBuffer] = None,
    der_batch_size: int = 64,
    acc_f_threshold: float = 0.99,
    acc_min_points:  int   = 5,
    acc_r2_min:      float = 0.70,
    acc_margin:      float = 0.20,
    # v15 reactive components (optional — created internally if not passed)
    bus:    Optional[SignalBus] = None,
    swdft:  Optional[SWDFT]    = None,
    brfd:   Optional[BRFD]     = None,
    erl:    Optional[ASR]      = None,  # accepts ASR or legacy ERL alias
    # DEHB Inner reactive re-search
    enable_reactive_dehb: bool = True,
    reactive_brackets:    int  = 4,
    probe_log_every: int = 0,
    # Ablation disable flags
    disable_slm:   bool = False,
    disable_swdft: bool = False,
    disable_acc:   bool = False,
    disable_brfd:  bool = False,   # ablation: skip BRFD reward shaping (use fixed weights 0.65/0.20/0.10)
    disable_erl:   bool = False,   # ablation: skip ERL orchestrator entirely (no recovery)
    # v21 Phase 1: TraceCollector wiring
    trace_collector: Optional["TraceCollector"] = None,
) -> Tuple[float, int, List[dict], SACAgent]:
    """Run one training cell. Returns (best_F, total_steps, history, agent)."""
    _TARGET_NORM = {
        "GHZ": "ghz", "ghz": "ghz",
        "W": "w", "w": "w",
        "Cluster": "cluster_linear", "cluster": "cluster_linear",
        "cluster_linear": "cluster_linear",
        "Dicke-k3": "dicke_k3", "Dicke_k3": "dicke_k3",
        "dicke-k3": "dicke_k3", "dicke_k3": "dicke_k3",
        # v23 N1/N2/N3
        "CompleteGraph": "complete_graph", "complete_graph": "complete_graph",
        "completegraph": "complete_graph",
        "Dicke-k2": "dicke_k2", "Dicke_k2": "dicke_k2", "dicke-k2": "dicke_k2",
        "dicke_k2": "dicke_k2",
        "GHZ_W_Hybrid": "ghz_w_hybrid", "ghz_w_hybrid": "ghz_w_hybrid",
        "GHZ-W-Hybrid": "ghz_w_hybrid", "ghz-w-hybrid": "ghz_w_hybrid",
    }
    target_state = _TARGET_NORM.get(target_state, target_state.lower())
    set_seed(seed)

    # Instantiate v15 components if not provided
    if bus   is None: bus   = SignalBus()
    if swdft is None: swdft = SWDFT(window=20, flat_thresh=0.15,
                                    min_steps=30_000, cooldown=8)
    if brfd  is None: brfd  = BRFD(seed=seed)
    sdft_inst = SDFT(history_len=30)
    ttt_inst  = TTT(n_gradient_steps=50, lr=1e-4)
    # v25 P4: build NoiseCurriculumController and wire into ASR
    _ncc = NoiseCurriculumController(
        base_noise_stage=noise_stage, n_qubits=n_qubits,
        target_state=target_state, max_ep_steps=max_ep_steps, seed=seed,
    )
    if erl is None:
        erl = ERL(sdft=sdft_inst, ttt=ttt_inst, brfd=brfd, curriculum_ctrl=_ncc)
    elif hasattr(erl, 'curriculum_ctrl') and erl.curriculum_ctrl is None:
        erl.curriculum_ctrl = _ncc

    _w_log = max(0.0, 1.0 - reward_w_F - reward_w_S - reward_w_str)
    env = SiliQunEnvWrapper(
        n_qubits=n_qubits,
        target_state=target_state,
        noise_stage=noise_stage,
        max_ep_steps=max_ep_steps,
        reward_weights={
            "fidelity":  reward_w_F,
            "success":   reward_w_S,
            "structure": reward_w_str,
            "log":       _w_log,
        },
        seed=seed,
    )
    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    target_dim = env.target.shape[0]

    sac_kwargs = dict(alpha=alpha, lr_actor=lr_actor, lr_critic=lr_critic,
                      target_name=target_state, n_qubits=n_qubits,  # v22 P1
                      tau=tau, hidden_dim=hidden_dim, cond_dim=cond_dim)
    agent = (transfer_agent(init_agent, obs_dim, act_dim, target_dim, **sac_kwargs)
             if init_agent is not None
             else SACAgent(obs_dim, act_dim, target_dim, **sac_kwargs))
    buffer     = ReplayBuffer(BUFFER_SIZE, obs_dim, act_dim, target_dim * 2)
    # SiliQunEnvWrapper.target is already a complex128 state vector
    _tgt = env.target
    target_vec = np.concatenate([_tgt.real, _tgt.imag]).astype(np.float32)

    obs       = env.reset()
    best_F    = 0.0
    best_F_prev = 0.0  # v19: for convergence_rate in ReflectionModule
    best_obs  = obs.copy()
    best_act  = np.zeros(act_dim, dtype=np.float32)
    history   = []
    step      = 0
    _reactive_dehb_fired = False

    acc = AdaptiveConvergenceController(
        F_threshold=acc_f_threshold,
        max_budget=max_steps,
        min_points=acc_min_points,
        r2_min=acc_r2_min,
        safety_margin=acc_margin,
    )
    # v22 P2: wire target info into ACC for hard-cell guard
    acc._target_name = target_state
    acc._n_qubits    = n_qubits
    _dec = ACCDecision(stop=False, reason=StopReason.NONE, step=0, best_F=0.0)
    _log_every = probe_log_every  # P0-1 fix: define before while loop
    _log_every = probe_log_every if probe_log_every > 0 else LOG_EVERY

    while step < max_steps:
        action = (env.action_space.sample() if step < WARMUP_STEPS
                  else agent.select_action(obs, target_vec))
        next_obs, reward, done, info = env.step(action)
        F = info.get("F", info.get("fidelity", 0.0))
        buffer.add(obs, action, reward, next_obs, float(done), target_vec)
        obs = next_obs if not done else env.reset()
        step += 1

        if F > best_F:
            best_F_prev = best_F  # v19: track previous best for convergence_rate
            best_F   = F
            best_obs = obs.copy()
            best_act = action.copy()
            ttt_inst.record_best(best_obs, best_act, target_vec)

        if step >= WARMUP_STEPS and step % UPDATE_EVERY == 0:
            agent.update(buffer, batch_size, der_buffer=der_buffer,
                         der_batch_size=der_batch_size)

        # SDFT records every step for spectral analysis
        if step % _log_every == 0:
            sdft_inst.record(best_F)

        # Periodic SLM fallback (v14 behaviour, lower priority than signal-gated)
        if step % slm_interval == 0 and step > WARMUP_STEPS:
            if not bus.has(Signal.STAGNATION):  # only if ASR not already active
                _best_F_before_slm = best_F
                _slm_ok, _slm_sigma, _slm_probe_F, _ = slm_correction(
                    agent, env, target_vec, best_F_current=best_F,
                    n_probe=200, brfd=brfd)
                # v21 Phase 1: record trace
                if trace_collector is not None:
                    trace_collector.add(
                        target_name=target_state,
                        n_qubits=n_qubits,
                        outcome_F=_slm_probe_F,
                        source="SLM",
                        step=step,
                        best_F_before=_best_F_before_slm,
                        sigma=_slm_sigma,
                        strategy="gaussian",
                    )
                if _slm_ok:
                    log.info(f"    SLM periodic correction @ step={step:,}")

        if step % _log_every == 0:
            # SWDFT check — may emit STAGNATION (skip if ablation disabled)
            if not disable_swdft:
                swdft.update(step, best_F, bus)

            # ASR response if stagnation detected (skip SLM Stage 1 if ablation)
            if bus.has(Signal.STAGNATION):
                if disable_erl:
                    # Ablation: skip entire ERL orchestrator (no recovery at all)
                    bus.consume(Signal.STAGNATION)
                    log.info(f"    ABL: ERL disabled entirely — stagnation ignored @ step={step:,}")
                elif disable_slm:
                    # Ablation: skip ERL-SLM Stage 1, emit SLM_FAILED to go straight to SDFT
                    bus.consume(Signal.STAGNATION)
                    bus.emit(Signal.SLM_FAILED, payload={"ablation_skip": True})
                    log.info(f"    ABL: ERL-SLM disabled — emitting SLM_FAILED @ step={step:,}")
                else:
                    _best_F_before_asr = best_F
                    erl.respond(bus, agent, env, target_vec, best_F, step,
                                best_F_prev=best_F_prev)  # v19: pass prev for Δ
                    # v21 Phase 1: record ASR trace (outcome captured from erl summary)
                    if trace_collector is not None:
                        _asr_sum = erl.summary()
                        _asr_best_F = float(_asr_sum.get("best_F_after", best_F))
                        trace_collector.add(
                            target_name=target_state,
                            n_qubits=n_qubits,
                            outcome_F=_asr_best_F,
                            source="ASR",
                            step=step,
                            best_F_before=_best_F_before_asr,
                            strategy="gaussian",
                        )

            # v19: Internalization — KL distillation after successful ERL-SLM
            # If ASR Stage 1 (ERL-SLM) succeeded, distill the corrected policy
            # back into the base actor so future episodes benefit from the insight
            if bus.has(Signal.SLM_CORRECTED):
                _slm_payload = bus.consume(Signal.SLM_CORRECTED)
                _is_erl_second = (_slm_payload or {}).get("erl_second_attempt", False)
                if _is_erl_second:
                    # Internalization: KL(π_base || π_corrected) distillation
                    import copy as _copy
                    _corrected_actor_sd = _copy.deepcopy(agent.actor.state_dict())
                    # Restore base actor (pre-ASR checkpoint)
                    if erl._pre_asr_actor_sd is not None:
                        agent.actor.load_state_dict(erl._pre_asr_actor_sd)
                    # Collect a batch of observations from the replay buffer
                    if len(buffer) >= batch_size:
                        _obs_b, _, _, _, _, _tgt_b = buffer.sample(batch_size)
                        # Load corrected actor weights into a temporary copy
                        _corrected_actor = type(agent.actor)(
                            agent.obs_dim, agent.act_dim, agent.target_dim,
                            agent.actor.hidden_dim, agent.actor.cond_dim,
                        ).to(DEVICE)
                        _corrected_actor.load_state_dict(_corrected_actor_sd)
                        _corrected_actor.eval()
                        # KL distillation: minimise KL(base || corrected)
                        _orig_lr = agent.actor_opt.param_groups[0]["lr"]
                        for _pg in agent.actor_opt.param_groups:
                            _pg["lr"] = ASR.KL_DISTILL_LR
                        for _kl_step in range(ASR.KL_DISTILL_STEPS):
                            with torch.no_grad():
                                _mean_c, _lstd_c = _corrected_actor(_obs_b, _tgt_b)
                                _std_c = _lstd_c.exp()
                            _mean_b, _lstd_b = agent.actor(_obs_b, _tgt_b)
                            _std_b = _lstd_b.exp()
                            # KL(corrected || base) = sum of per-dim KL
                            _kl = torch.distributions.kl_divergence(
                                torch.distributions.Normal(_mean_c, _std_c),
                                torch.distributions.Normal(_mean_b, _std_b),
                            ).sum(dim=-1).mean()
                            agent.actor_opt.zero_grad()
                            _kl.backward()
                            nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                            agent.actor_opt.step()
                        for _pg in agent.actor_opt.param_groups:
                            _pg["lr"] = _orig_lr
                        log.info(f"    ASR v19: internalization complete "
                                 f"({ASR.KL_DISTILL_STEPS} KL steps) @ step={step:,}")
                    else:
                        log.debug(f"    ASR v19: internalization skipped (buffer too small)")

            # Reactive DEHB Inner if all ASR stages failed
            if bus.has(Signal.TTT_FAILED) and enable_reactive_dehb \
                    and not _reactive_dehb_fired:
                bus.consume(Signal.TTT_FAILED)
                # ── Fix 3: Gradient variance check ───────────────────────────────────────────────────────────────────
                _grad_var = 0.0
                try:
                    _grad_norms = []
                    for _p in agent.actor.parameters():
                        if _p.grad is not None:
                            _grad_norms.append(_p.grad.data.norm(2).item())
                    _grad_var = float(np.var(_grad_norms)) if _grad_norms else 0.0
                except Exception:
                    _grad_var = 1.0  # default: allow DEHB Inner if check fails
                if _grad_var < ASR.GRAD_VAR_THRESH:
                    log.info(f"    DEHB Inner SKIPPED — grad_var={_grad_var:.2e} "
                             f"< {ASR.GRAD_VAR_THRESH:.2e} (Fix 3: flat landscape)")
                else:
                    log.info(f"    DEHB Inner (reactive) firing @ step={step:,} "
                             f"grad_var={_grad_var:.2e}")
                try:
                    _r_cs = build_reactive_cs(seed=seed)
                    def _r_target(config, budget=None, **kw):
                        budget = kw.get('fidelity', budget)
                        try:
                            _r_F, _, _, _ = run_cell(
                                n_qubits=n_qubits, target_state=target_state,
                                seed=seed, noise_stage=noise_stage,
                                max_ep_steps=max_ep_steps,
                                reward_w_F=reward_w_F, reward_w_S=reward_w_S,
                                reward_w_str=reward_w_str,
                                alpha=float(config["alpha"]),
                                lr_actor=float(config["lr_actor"]),
                                lr_critic=lr_critic, tau=tau,
                                batch_size=batch_size, hidden_dim=hidden_dim,
                                cond_dim=cond_dim,
                                max_steps=int(max_steps * 0.1),
                                init_agent=agent,
                                slm_interval=slm_interval,
                                der_buffer=der_buffer,
                                enable_reactive_dehb=False,
                            )
                            return {"fitness": 1.0 - _r_F, "cost": budget}
                        except Exception:
                            return {"fitness": 1.0, "cost": budget}
                    _r_dehb = DEHB(
                        cs=_r_cs,
                        f=_r_target,
                        min_fidelity=0.1, max_fidelity=1.0,
                        n_workers=1, output_path="/tmp",
                    )
                    _r_dehb.run(
                        fevals=reactive_brackets * 3,
                    )
                    _r_inc_cfg, _r_inc_score = _r_dehb.get_incumbents()
                    if _r_inc_cfg is not None:
                        _r_cfg = dict(_r_inc_cfg) if hasattr(_r_inc_cfg, 'get_dictionary') else dict(_r_inc_cfg)
                        # Hot-patch alpha and lr_actor into existing agent
                        new_alpha = float(_r_cfg["alpha"])
                        new_lr    = float(_r_cfg["lr_actor"])
                        agent.log_alpha = torch.tensor(
                            math.log(new_alpha), requires_grad=True, device=DEVICE)
                        agent.alpha_opt = torch.optim.Adam(
                            [agent.log_alpha], lr=new_lr)
                        for pg in agent.actor_opt.param_groups:
                            pg["lr"] = new_lr
                        bus.emit(Signal.HP_UPDATED,
                                 payload={"alpha": new_alpha, "lr_actor": new_lr})
                        log.info(f"    DEHB Inner: new alpha={new_alpha:.4f} "
                                 f"lr_actor={new_lr:.2e}")
                    _reactive_dehb_fired = True
                except Exception as _re:
                    log.warning(f"    DEHB Inner failed: {_re}")

            # ACC convergence check (skip early stop if ablation disabled)
            if disable_acc:
                _dec = ACCDecision(stop=False, reason=StopReason.NONE, step=step, best_F=best_F)
            else:
                _dec = acc.update(step, best_F)
            stop, reason = _dec.stop, _dec.reason
            _F_inf = round(_dec.F_inf, 5) if _dec.F_inf is not None else None
            history.append({
                "step": step, "best_F": round(best_F, 5),
                "acc_stop": stop, "acc_reason": str(reason),
                "acc_F_inf": _F_inf,
                "acc_T_star": _dec.T_star,
                "acc_recommended_budget": _dec.recommended_budget,
                "asr_summary": erl.summary(),
            })
            log.info(f"    {n_qubits}Q/{target_state}/s{seed} "
                     f"step={step:,} best_F={best_F:.4f} "
                     f"acc={reason} F_inf={_F_inf} alpha={agent.alpha:.4f}")
            if stop:
                log.info(f"    ACC STOP [{reason}] step={step:,} "
                         f"F_inf={_F_inf} T*={_dec.T_star}")
                break

    _F_inf_final = round(_dec.F_inf, 5) if _dec.F_inf is not None else None
    history.append({
        "step": step, "best_F": round(best_F, 5),
        "acc_stop": True, "acc_reason": "MAX_BUDGET",
        "acc_F_inf": _F_inf_final,
        "acc_T_star": _dec.T_star,
        "acc_recommended_budget": _dec.recommended_budget,
        "erl_summary": erl.summary(),
    })
    return best_F, step, history, agent

# ─────────────────────────────────────────────────────────────────────────────
# DEHB Outer target factory
# ─────────────────────────────────────────────────────────────────────────────
class OuterTargetFactory:
    """
    Wraps run_cell for the DEHB Outer loop.
    Evaluates a HP configuration on ALL (target, seed) pairs at short budget.
    Returns mean best_F across all pairs as the fitness signal.
    """
    def __init__(self, n_qubits: int, targets: List[str], seeds: List[int],
                 noise_stage: int, max_ep_steps: int, short_budget: int,
                 der_buffer: Optional[DERPlusPlusBuffer],
                 init_agent: Optional[SACAgent],
                 curriculum_ctrl=None):
        self.n_qubits        = n_qubits
        self.targets         = targets
        self.seeds           = seeds
        self.noise_stage     = noise_stage
        self.max_ep_steps    = max_ep_steps
        self.short_budget    = short_budget
        self.der_buffer      = der_buffer
        self.init_agent      = init_agent
        self.curriculum_ctrl = curriculum_ctrl  # v25 P3
        self._n_calls        = 0

    def __call__(self, config, fidelity=None, **kwargs):
        # DEHB calls self.f(config, fidelity=fidelity) — accept as keyword or positional
        # FIX-10 (HB-87): DEHB may pass fidelity as numpy array shape (1,) — cast to scalar
        budget = float(fidelity.item() if hasattr(fidelity, 'item') else fidelity) if fidelity is not None else float(kwargs.get('budget', 1.0))
        self._n_calls += 1
        # v25 P3: extract noise_stage_t from DEHB config
        # FIX-10b: ensure scalar int even if config returns numpy array
        _nst_raw = config.get("noise_stage_t", self.noise_stage)
        _noise_stage_t = int(_nst_raw.item() if hasattr(_nst_raw, 'item') else _nst_raw)
        _effective_noise = _noise_stage_t
        if self.curriculum_ctrl is not None:
            self.curriculum_ctrl.set_stage(_noise_stage_t)
            _effective_noise = self.curriculum_ctrl.current_stage
        fids = []
        for target in self.targets:
            for seed in self.seeds:
                try:
                    F, _, _, _ = run_cell(
                        n_qubits=self.n_qubits,
                        target_state=target,
                        seed=seed,
                        noise_stage=_effective_noise,
                        max_ep_steps=self.max_ep_steps,
                        reward_w_F=0.65, reward_w_S=0.20, reward_w_str=0.10,
                        alpha=float(config["alpha"]),
                        lr_actor=float(config["lr_actor"]),
                        lr_critic=float(config["lr_critic"]),
                        tau=float(config["tau"]),
                        batch_size=int(config["batch_size"]),
                        hidden_dim=int(config["hidden_dim"]),
                        cond_dim=int(config["cond_dim"]),
                        max_steps=int(self.short_budget * budget),
                        init_agent=self.init_agent,
                        slm_interval=int(config["slm_interval"]),
                        der_buffer=self.der_buffer,
                        acc_f_threshold=0.99, acc_min_points=3,
                        acc_r2_min=0.60, acc_margin=0.20,
                        enable_reactive_dehb=False,
                    )
                    fids.append(F)
                except Exception as _e:
                    log.warning(f"    Outer HP trial SKIPPED ({_e})")
                    fids.append(0.0)
        mean_F = float(np.mean(fids)) if fids else 0.0
        return {"fitness": 1.0 - mean_F, "cost": budget}

# ─────────────────────────────────────────────────────────────────────────────
# run_qubit_level — runs DEHB Outer then full-budget cells for one qubit count
# ─────────────────────────────────────────────────────────────────────────────
# v22 P8: Global registry of best agents per (n_qubits-1, target) for transfer
_TRANSFER_AGENT_REGISTRY: Dict[str, "SACAgent"] = {}

def run_qubit_level(
    n_qubits:      int,
    targets:       List[str],
    seeds:         List[int],
    noise_stage:   int,
    max_steps:     int,
    outer_brackets: int,
    inner_brackets: int,
    results_dir:   Path,
    ckpt_dir:      Path,
    init_agent:    Optional[SACAgent],
    der_buffer:    Optional[DERPlusPlusBuffer],
    seed_filter:   Optional[int] = None,  # if set, only run this seed
    fixed_hp:      Optional[dict] = None,  # if set, skip DEHB Outer and use this HP
    disable_slm:   bool = False,           # ablation: skip ERL-SLM Stage 1
    disable_swdft: bool = False,           # ablation: skip SWDFT proactive detector
    disable_acc:   bool = False,           # ablation: run full budget (no ACC early stop)
    disable_brfd:  bool = False,           # ablation: skip BRFD reward shaping
    disable_erl:   bool = False,           # ablation: skip ERL orchestrator entirely
    trace_collector: Optional["TraceCollector"] = None,  # v21 Phase 1
) -> Tuple[Dict[str, Dict[int, float]], Optional[SACAgent]]:
    """
    Phase 1: DEHB Outer — find best HP across all (target, seed) at short budget.
    Phase 2: Full-budget run per (target, seed) with best HP + reactive pipeline.
    Returns cell_results dict and best overall agent.
    """
    max_ep_steps = n_qubits * 50

    # ── DEHB Outer (or fixed HP bypass) ──────────────────────────────────────
    if fixed_hp is not None:
        # Fixed HP mode: skip DEHB Outer entirely, use pre-validated incumbent
        _best_outer_cfg = fixed_hp
        log.info(f"  DEHB Outer: SKIPPED (fixed_hp mode)")
        log.info(f"  Fixed HP: alpha={float(_best_outer_cfg['alpha']):.4f} "
                 f"lr_actor={float(_best_outer_cfg['lr_actor']):.2e} "
                 f"lr_critic={float(_best_outer_cfg['lr_critic']):.2e} "
                 f"tau={float(_best_outer_cfg['tau']):.4f} "
                 f"batch_size={int(_best_outer_cfg['batch_size'])} "
                 f"hidden_dim={int(_best_outer_cfg['hidden_dim'])}")
    else:
        short_budget = max(10_000, max_steps // 20)
        outer_cs     = build_inner_cs(seed=42)
        outer_seeds  = seeds if seed_filter is None else [seed_filter]
        # v25 P3: NoiseCurriculumController for outer DEHB
        _ncc_outer = NoiseCurriculumController(
            base_noise_stage=noise_stage, n_qubits=n_qubits,
            target_state=targets[0], max_ep_steps=max_ep_steps,
            seed=outer_seeds[0] if outer_seeds else 42,
        )
        outer_factory = OuterTargetFactory(
            n_qubits=n_qubits, targets=targets, seeds=outer_seeds,
            noise_stage=noise_stage, max_ep_steps=max_ep_steps,
            short_budget=short_budget, der_buffer=der_buffer,
            init_agent=init_agent,
            curriculum_ctrl=_ncc_outer,
        )
        log.info(f"  DEHB Outer: {outer_brackets} brackets, "
                 f"short_budget={short_budget:,} per trial")
        _best_outer_cfg = None
        try:
            outer_dehb = DEHB(
                cs=outer_cs,
                f=outer_factory,
                min_fidelity=0.1, max_fidelity=1.0,
                n_workers=1, output_path="/tmp",
            )
            outer_dehb.run(
                fevals=outer_brackets * len(targets) * len(outer_seeds),
            )
            _inc_cfg, _inc_score = outer_dehb.get_incumbents()
            if _inc_cfg is not None:
                _best_outer_cfg = dict(_inc_cfg) if hasattr(_inc_cfg, 'get_dictionary') else dict(_inc_cfg)
                log.info(f"  DEHB Outer best: alpha={float(_best_outer_cfg['alpha']):.4f} "
                         f"lr_actor={float(_best_outer_cfg['lr_actor']):.2e}")
        except Exception as _oe:
            log.warning(f"  DEHB Outer failed ({_oe}) — using default HP")

        if _best_outer_cfg is None:
            # Fallback default config
            _best_outer_cfg = {
                "alpha": 0.05, "lr_actor": 3e-4, "lr_critic": 3e-4,
                "tau": 0.005, "batch_size": 256, "hidden_dim": 256,
                "cond_dim": 64, "slm_interval": 50_000,
            }

    # ── Full-budget cells ─────────────────────────────────────────────────────
    cell_results: Dict[str, Dict[int, float]] = {t: {} for t in targets}
    best_agent:   Optional[SACAgent] = None
    best_F_level  = 0.0

    run_seeds = seeds if seed_filter is None else [seed_filter]

    for target in targets:
        for seed in run_seeds:
            log.info(f"  ── {n_qubits}Q / {target} / seed={seed} ──")
            # Per-cell BRFD and SWDFT instances (independent per cell)
            cell_brfd  = BRFD(seed=seed)
            cell_swdft = SWDFT(window=20, flat_thresh=0.15,
                               min_steps=30_000, cooldown=8)
            cell_bus   = SignalBus()
            cell_sdft  = SDFT(history_len=30)
            cell_ttt   = TTT(n_gradient_steps=50, lr=1e-4)
            cell_erl   = ASR(sdft=cell_sdft, ttt=cell_ttt, brfd=cell_brfd)  # v19: ASR
# ACC probe (20k) to get budget estimate
            _acc_budget = max_steps
            try:
                _, _, _probe_hist, _ = run_cell(
                    n_qubits=n_qubits, target_state=target, seed=seed,
                    noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                    reward_w_F=0.65, reward_w_S=0.20, reward_w_str=0.10,
                    alpha=float(_best_outer_cfg["alpha"]),
                    lr_actor=float(_best_outer_cfg["lr_actor"]),
                    lr_critic=float(_best_outer_cfg["lr_critic"]),
                    tau=float(_best_outer_cfg["tau"]),
                    batch_size=int(_best_outer_cfg["batch_size"]),
                    hidden_dim=int(_best_outer_cfg["hidden_dim"]),
                    cond_dim=int(_best_outer_cfg["cond_dim"]),
                    max_steps=30_000,
                    init_agent=init_agent,
                    slm_interval=int(_best_outer_cfg["slm_interval"]),
                    der_buffer=der_buffer,
                    acc_f_threshold=0.99, acc_min_points=5,
                    acc_r2_min=0.60, acc_margin=0.20,
                enable_reactive_dehb=False,
                probe_log_every=2_000,
                trace_collector=trace_collector,
            )
                if _probe_hist and _probe_hist[-1].get("acc_recommended_budget"):
                    _raw_budget = _probe_hist[-1]["acc_recommended_budget"]
                    # Floor at 30k, no artificial ceiling — ACC drives the budget
                    _acc_budget = max(30_000, _raw_budget)
                    log.info(f"  ACC predicted budget: {_acc_budget:,} (raw={_raw_budget:,})")
            except Exception as _pe:
                log.warning(f"  ACC probe failed ({_pe}) — using {max_steps:,}")

            # BRFD reward weights for full run
            if disable_brfd:
                # Ablation: skip BRFD, use fixed default weights
                w_F, w_S, w_str = 0.65, 0.20, 0.10
                log.info("  ABL: BRFD disabled — using fixed weights w_F=0.65, w_S=0.20, w_str=0.10")
            else:
                w_F, w_S, w_str = cell_brfd.sample_weights()

            best_F, _steps, _hist, _agent = run_cell(
                n_qubits=n_qubits, target_state=target, seed=seed,
                noise_stage=noise_stage, max_ep_steps=max_ep_steps,
                reward_w_F=w_F, reward_w_S=w_S, reward_w_str=w_str,
                alpha=float(_best_outer_cfg["alpha"]),
                lr_actor=float(_best_outer_cfg["lr_actor"]),
                lr_critic=float(_best_outer_cfg["lr_critic"]),
                tau=float(_best_outer_cfg["tau"]),
                batch_size=int(_best_outer_cfg["batch_size"]),
                hidden_dim=int(_best_outer_cfg["hidden_dim"]),
                cond_dim=int(_best_outer_cfg["cond_dim"]),
                max_steps=_acc_budget,
                init_agent=init_agent,
                slm_interval=int(_best_outer_cfg["slm_interval"]),
                der_buffer=der_buffer,
                acc_f_threshold=0.99, acc_min_points=5,
                acc_r2_min=0.70, acc_margin=0.20,
                bus=cell_bus, swdft=cell_swdft, brfd=cell_brfd,
                erl=cell_erl,
                enable_reactive_dehb=True,
                reactive_brackets=inner_brackets,
                disable_slm=disable_slm,
                disable_swdft=disable_swdft,
                disable_acc=disable_acc,
                trace_collector=trace_collector,  # v21 Phase 1
            )
            cell_results[target][seed] = best_F

            # Save per-cell result
            cell_path = results_dir / f"{n_qubits}Q_{target}_s{seed}.json"
            cell_path.write_text(json.dumps({
                "n_qubits": n_qubits, "target": target, "seed": seed,
                "best_F": round(best_F, 6),
                "steps": _steps,
                "best_hp": {k: (float(v) if isinstance(v, (int, float))
                                else int(v))
                            for k, v in _best_outer_cfg.items()},
                "brfd_weights": {"w_F": w_F, "w_S": w_S, "w_str": w_str},
                "asr_summary": cell_erl.summary(),
                "history": _hist[-10:],  # last 10 checkpoints only
            }, indent=2))
            log.info(f"  {n_qubits}Q/{target}/s{seed} DONE: best_F={best_F:.4f}")

            if best_F > best_F_level:
                best_F_level = best_F
                best_agent   = _agent

    # Save best checkpoint for this qubit level
    if best_agent is not None:
        ckpt_path = ckpt_dir / f"{n_qubits}Q_best_agent.pt"
        torch.save({
            "obs_dim":    best_agent.obs_dim,
            "act_dim":    best_agent.act_dim,
            "target_dim": best_agent.target_dim,
            "actor":      best_agent.actor.state_dict(),
            "critic":     best_agent.critic.state_dict(),
        }, ckpt_path)
        log.info(f"  Checkpoint saved: {ckpt_path}")

    return cell_results, best_agent

# ─────────────────────────────────────────────────────────────────────────────
# A8: ZNE — Zero-Noise Extrapolation post-processing hook
# ─────────────────────────────────────────────────────────────────────────────
def zne_extrapolate(fidelity_at_noise: Dict[float, float]) -> float:
    """
    Richardson extrapolation to zero noise.
    fidelity_at_noise: {noise_scale: fidelity_value}
    Requires at least 2 noise levels. Returns extrapolated F(noise=0).
    If only one level provided, returns it unchanged.
    """
    if len(fidelity_at_noise) < 2:
        return list(fidelity_at_noise.values())[0]
    scales = sorted(fidelity_at_noise.keys())
    fids   = [fidelity_at_noise[s] for s in scales]
    # Linear Richardson: F(0) ≈ F(c1) - c1*(F(c2)-F(c1))/(c2-c1)
    c1, c2 = scales[0], scales[1]
    f1, f2 = fids[0], fids[1]
    if abs(c2 - c1) < 1e-9:
        return f1
    slope = (f2 - f1) / (c2 - c1)
    f_zero = f1 - c1 * slope
    return float(np.clip(f_zero, 0.0, 1.0))


def apply_zne_to_results(
    cell_results: Dict[str, Dict[int, float]],
    noise_scales: Tuple[float, ...] = (1.0, 1.5, 2.0),
    zne_fidelities: Optional[Dict[str, Dict[int, Dict[float, float]]]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Apply ZNE post-processing to cell_results.
    If zne_fidelities is provided (target -> seed -> {scale: F}), extrapolate.
    Otherwise return cell_results unchanged (ZNE data not yet available).
    """
    if zne_fidelities is None:
        return cell_results
    corrected = {}
    for target, seed_map in cell_results.items():
        corrected[target] = {}
        for seed, F in seed_map.items():
            if target in zne_fidelities and seed in zne_fidelities[target]:
                F_zne = zne_extrapolate(zne_fidelities[target][seed])
                log.info(f"  ZNE {target}/s{seed}: F_raw={F:.4f} → F_zne={F_zne:.4f}")
                corrected[target][seed] = F_zne
            else:
                corrected[target][seed] = F
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# A7: ERL-SLM Reflection Collector — accumulates successful JSON reflections
# for FASTMIX-Omega training dataset
# ─────────────────────────────────────────────────────────────────────────────
class ReflectionCollector:
    """
    Collects successful ERL-SLM JSON reflections as training data for
    FASTMIX-Omega outer loop. Persists to disk after each successful reflection.

    A reflection is 'successful' if:
      - JSON parsed without error
      - fidelity_gain > 0 (best_F improved after applying the reflection)
      - quality_score >= 0.5

    Dataset format (JSONL): one JSON object per line with fields:
      cell, seed, step, stagnation_type, delta, sigma, strategy,
      fidelity_before, fidelity_after, fidelity_gain, quality_score,
      alpha_suggested (dict of 7 alpha values if present)
    """
    MIN_QUALITY = 0.5
    MIN_GAIN    = 0.0  # any positive gain counts

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._n_collected = 0
        self._n_failed    = 0

    def record(
        self,
        cell:             str,
        seed:             int,
        step:             int,
        stagnation_type:  str,
        delta:            np.ndarray,
        sigma:            float,
        strategy:         str,
        fidelity_before:  float,
        fidelity_after:   float,
        quality_score:    float,
        alpha_suggested:  Optional[Dict[str, float]] = None,
        raw_json:         Optional[str] = None,
    ) -> bool:
        """Record one reflection. Returns True if it was accepted (successful)."""
        gain = fidelity_after - fidelity_before
        success = (quality_score >= self.MIN_QUALITY and gain > self.MIN_GAIN)
        entry = {
            "cell":            cell,
            "seed":            seed,
            "step":            step,
            "stagnation_type": stagnation_type,
            "delta":           delta.tolist() if isinstance(delta, np.ndarray) else delta,
            "sigma":           float(sigma),
            "strategy":        strategy,
            "fidelity_before": round(float(fidelity_before), 6),
            "fidelity_after":  round(float(fidelity_after), 6),
            "fidelity_gain":   round(float(gain), 6),
            "quality_score":   round(float(quality_score), 4),
            "success":         success,
            "alpha_suggested": alpha_suggested,
            "raw_json":        raw_json,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.dataset_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if success:
            self._n_collected += 1
            log.info(f"  [ReflectionCollector] SUCCESS #{self._n_collected}: "
                     f"{cell}/s{seed} gain={gain:.4f} quality={quality_score:.3f}")
        else:
            self._n_failed += 1
            log.debug(f"  [ReflectionCollector] FAILED #{self._n_failed}: "
                      f"{cell}/s{seed} gain={gain:.4f} quality={quality_score:.3f}")
        return success

    @property
    def n_successful(self) -> int:
        return self._n_collected

    @property
    def n_failed(self) -> int:
        return self._n_failed

    def summary(self) -> Dict[str, int]:
        return {"successful": self._n_collected, "failed": self._n_failed}


# ─────────────────────────────────────────────────────────────────────────────
# A2/A3: FASTMIX-Omega — Bilevel outer loop over 7 mixture coefficients
# Validation oracle: 3Q/W held-out cell (hardest consistently-failing cell)
# Q-Forge recommended weights for hard cells (A3)
# ─────────────────────────────────────────────────────────────────────────────

# Q-Forge recommended alpha weights (from LEAP verification, A3)
FASTMIX_QFORGE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "default": {
        "alpha_SLM":    0.30,
        "alpha_BRFD":   0.15,
        "alpha_DEHB":   0.15,
        "alpha_replay": 0.20,
        "alpha_ASR":    0.10,
        "alpha_SWDFT":  0.05,
        "alpha_ACC":    0.05,
    },
    # Hard W-state cells: ERL-SLM dominates, replay buffer secondary
    "3Q/W": {
        "alpha_SLM":    0.40,
        "alpha_BRFD":   0.15,
        "alpha_DEHB":   0.15,
        "alpha_replay": 0.30,
        "alpha_ASR":    0.10,
        "alpha_SWDFT":  0.10,
        "alpha_ACC":    0.05,
    },
    "4Q/W": {
        "alpha_SLM":    0.40,
        "alpha_BRFD":   0.15,
        "alpha_DEHB":   0.20,
        "alpha_replay": 0.30,
        "alpha_ASR":    0.25,
        "alpha_SWDFT":  0.10,
        "alpha_ACC":    0.05,
    },
    "5Q/W": {
        "alpha_SLM":    0.45,
        "alpha_BRFD":   0.15,
        "alpha_DEHB":   0.20,
        "alpha_replay": 0.30,
        "alpha_ASR":    0.30,
        "alpha_SWDFT":  0.10,
        "alpha_ACC":    0.05,
    },
    # Hard Cluster cells
    "4Q/Cluster": {
        "alpha_SLM":    0.35,
        "alpha_BRFD":   0.20,
        "alpha_DEHB":   0.20,
        "alpha_replay": 0.25,
        "alpha_ASR":    0.20,
        "alpha_SWDFT":  0.10,
        "alpha_ACC":    0.05,
    },
    "5Q/Cluster": {
        "alpha_SLM":    0.40,
        "alpha_BRFD":   0.20,
        "alpha_DEHB":   0.20,
        "alpha_replay": 0.25,
        "alpha_ASR":    0.25,
        "alpha_SWDFT":  0.10,
        "alpha_ACC":    0.05,
    },
}


def get_fastmix_weights(n_qubits: int, target: str) -> Dict[str, float]:
    """Return Q-Forge recommended FASTMIX-Omega weights for a given cell."""
    cell_key = f"{n_qubits}Q/{target}"
    return FASTMIX_QFORGE_WEIGHTS.get(cell_key, FASTMIX_QFORGE_WEIGHTS["default"])


class FastMixOmega:
    """
    FASTMIX-Omega bilevel outer loop.

    Maintains a 7-dimensional mixture coefficient vector alpha = [
        alpha_SLM, alpha_BRFD, alpha_DEHB, alpha_replay,
        alpha_ASR, alpha_SWDFT, alpha_ACC
    ]

    Update rule (gradient-free, bandit-style):
      After each cell completes, compute delta_F = best_F - baseline_F.
      Update the component that was most active during the cell by:
        alpha_k += lr * delta_F * activity_k
      Renormalise to sum = 1.0 after each update.

    Validation oracle: 3Q/W cell (hardest consistently-failing cell).
    Phase 0 (< 50 successful reflections): uses Q-Forge seed weights.
    Phase 1 (>= 50 reflections): switches to gradient-based updates.
    """
    COMPONENTS = [
        "alpha_SLM", "alpha_BRFD", "alpha_DEHB", "alpha_replay",
        "alpha_ASR", "alpha_SWDFT", "alpha_ACC",
    ]
    PHASE0_THRESHOLD = 50   # min successful reflections before Phase 1
    LR               = 0.05  # outer loop learning rate

    def __init__(
        self,
        seed_weights: Optional[Dict[str, float]] = None,
        reflection_collector: Optional["ReflectionCollector"] = None,
    ):
        # Initialise from Q-Forge seed weights or uniform
        if seed_weights is not None:
            self._alpha = np.array(
                [seed_weights.get(k, 1.0 / len(self.COMPONENTS))
                 for k in self.COMPONENTS], dtype=np.float64)
        else:
            self._alpha = np.ones(len(self.COMPONENTS), dtype=np.float64)
        self._normalise()
        self._collector    = reflection_collector
        self._history: List[Dict] = []
        self._baseline_F: float   = 0.0
        self._phase       = 0  # 0 = seed weights, 1 = gradient updates
        # v21 Phase 1: Adam state for alpha_SLM CG update
        self._alpha_slm_adam: dict = {"m": 0.0, "v": 0.0, "t": 0}

    def _normalise(self):
        s = self._alpha.sum()
        if s > 1e-9:
            self._alpha /= s

    def intra_cell_update(
        self,
        failed_component: str,
        boosted_component: str,
        penalty: float = 0.02,
    ):
        """
        v22 P5: Intra-cell weight adjustment on component failure.
        Reduces failed_component weight by penalty and redistributes to boosted_component.
        Called immediately on ASR full failure (all 3 stages failed).
        """
        if failed_component not in self.COMPONENTS:
            return
        if boosted_component not in self.COMPONENTS:
            return
        i_fail  = self.COMPONENTS.index(failed_component)
        i_boost = self.COMPONENTS.index(boosted_component)
        transfer = min(penalty, self._alpha[i_fail] - 1e-3)
        if transfer > 0:
            self._alpha[i_fail]  -= transfer
            self._alpha[i_boost] += transfer
            self._normalise()

    @property
    def weights(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.COMPONENTS, self._alpha)}

    def get_weight(self, component: str) -> float:
        """Return current mixture weight for a named component."""
        idx = self.COMPONENTS.index(component)
        return float(self._alpha[idx])

    def update(
        self,
        cell:         str,
        best_F:       float,
        activity:     Dict[str, float],  # component name -> activity score [0,1]
        n_successful_reflections: int = 0,
        # v21 Phase 1: CG update inputs
        baseline_F_before_cell: float = 0.0,
        n_slm_calls:  int = 0,
        n_total_calls: int = 1,
    ):
        """
        Update alpha based on cell outcome.
        activity: how much each component contributed to this cell's training.
        v21 Phase 1: also applies alpha_slm_cg_update for the SLM component.
        """
        delta_F = best_F - self._baseline_F
        self._baseline_F = max(self._baseline_F, best_F)

        # Switch to Phase 1 once enough reflections accumulated
        if self._phase == 0 and n_successful_reflections >= self.PHASE0_THRESHOLD:
            self._phase = 1
            log.info(f"  [FASTMIX-Omega] Phase 1 ACTIVATED "
                     f"(n_reflections={n_successful_reflections})")

        if self._phase == 1 and delta_F != 0.0:
            for i, comp in enumerate(self.COMPONENTS):
                act = activity.get(comp, 0.0)
                self._alpha[i] += self.LR * delta_F * act
                self._alpha[i]  = max(self._alpha[i], 1e-3)  # floor at 0.1%
            self._normalise()

        # v21 Phase 1: CG update for alpha_SLM (implicit differentiation)
        if self._phase == 1 and n_total_calls > 0:
            slm_idx = self.COMPONENTS.index("alpha_SLM")
            new_alpha_slm, self._alpha_slm_adam = alpha_slm_cg_update(
                alpha_slm_current=float(self._alpha[slm_idx]),
                val_F=best_F,
                baseline_F=baseline_F_before_cell,
                n_slm_calls=n_slm_calls,
                n_total_calls=n_total_calls,
                adam_state=self._alpha_slm_adam,
            )
            self._alpha[slm_idx] = new_alpha_slm
            self._normalise()
            log.debug(f"  [FASTMIX-Omega] alpha_SLM CG update: "
                      f"{float(self._alpha[slm_idx]):.4f} "
                      f"(n_slm={n_slm_calls}/{n_total_calls})")

        entry = {
            "cell":    cell,
            "best_F":  round(best_F, 5),
            "delta_F": round(delta_F, 5),
            "phase":   self._phase,
            "weights": self.weights,
        }
        self._history.append(entry)
        log.info(f"  [FASTMIX-Omega] cell={cell} best_F={best_F:.4f} "
                 f"delta_F={delta_F:+.4f} phase={self._phase}")
        log.info(f"  [FASTMIX-Omega] alpha={{"
                 + ", ".join(f"{k}:{v:.3f}" for k, v in self.weights.items())
                 + "}")

    def summary(self) -> Dict:
        return {
            "phase":   self._phase,
            "weights": self.weights,
            "n_updates": len(self._history),
            "baseline_F": round(self._baseline_F, 5),
        }


# ─────────────────────────────────────────────────────────────────────────────
# A4: Adaptive qubit scaling policy
# n_qubits <= 3 : mean_F > 0.999 AND min_F > 0.999
# n_qubits == 4 : mean_F > 0.99  AND min_F > 0.99
# n_qubits > 4  : mean_F > 0.99  AND min_F > 0.99
# ─────────────────────────────────────────────────────────────────────────────
def get_advance_thresholds(n_qubits: int) -> Tuple[float, float]:
    """
    Return (f_threshold, f_floor) for the given qubit count.
    Policy:
      n_qubits <= 3 : both thresholds = 0.999  (near-perfect required)
      n_qubits >= 4 : both thresholds = 0.990  (high-fidelity required)
    """
    if n_qubits <= 3:
        return 0.999, 0.999
    else:
        return 0.990, 0.990


# ─────────────────────────────────────────────────────────────────────────────
# Advancement logic (v20: uses adaptive thresholds)
# ─────────────────────────────────────────────────────────────────────────────
def should_advance(cell_results: Dict[str, Dict[int, float]],
                   f_threshold: float, f_floor: float) -> Tuple[bool, float, float]:
    all_F = [f for sd in cell_results.values() for f in sd.values()]
    if not all_F:
        return False, 0.0, 0.0
    mean_F = float(np.mean(all_F))
    min_F  = float(np.min(all_F))
    return (mean_F >= f_threshold and min_F >= f_floor), mean_F, min_F

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="QUASAR v25 — Noise-Adaptive Curriculum Learning + FASTMIX-Omega Bilevel Outer Loop")
    parser.add_argument("--targets",         nargs="+",
                        default=["GHZ", "W", "Cluster", "Dicke-k3",
                                 "CompleteGraph", "Dicke-k2", "GHZ_W_Hybrid"])
    parser.add_argument("--seeds",           nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--seed-filter",     type=int, default=None,
                        help="Run only this seed (for parallel per-seed jobs)")
    parser.add_argument("--min-qubits",      type=int, default=2)
    parser.add_argument("--max-qubits",      type=int, default=12)
    parser.add_argument("--outer-brackets",  type=int, default=6,
                        help="DEHB Outer bracket count per qubit level")
    parser.add_argument("--inner-brackets",  type=int, default=4,
                        help="DEHB Inner (reactive) bracket count")
    parser.add_argument("--f-threshold",     type=float, default=None,
                        help="Override mean-F threshold (default: adaptive — 0.999 for <=3Q, 0.99 for >4Q)")
    parser.add_argument("--min-qubits-ceiling", type=int, default=4,
                        help="Do not apply scalability ceiling below this qubit level (default 4)")
    parser.add_argument("--f-floor",         type=float, default=None,
                        help="Override min-F floor (default: adaptive — 0.999 for <=3Q, 0.99 for >4Q)")
    parser.add_argument("--fastmix-dataset",  type=str, default=None,
                        help="Path to JSONL file for FASTMIX-Omega reflection training dataset (A7)")
    parser.add_argument("--enable-zne",       action="store_true", default=False,
                        help="Enable ZNE post-processing on cell results (A8)")
    parser.add_argument("--max-steps-base",  type=int, default=500_000)
    parser.add_argument("--steps-scale",     type=float, default=1.5)
    parser.add_argument("--noise-stage",     type=int, default=5)
    parser.add_argument("--results-dir",     type=str,
                        default=str(Path.home() / "quasar_v20" / "results" / "v20_adaptive"))
    parser.add_argument("--fixed-hp-json",   type=str, default=None,
                        help="Path to JSON file with pre-validated HP config (skips DEHB Outer). "
                             "Format: {\"config\": {\"alpha\": ..., \"lr_actor\": ..., ...}}")
    # ── Ablation disable flags (A8) ────────────────────────────────────────────
    parser.add_argument("--disable-fastmix",  action="store_true", default=False,
                        help="Ablation: disable FASTMIX-Omega (use fixed default weights throughout)")
    parser.add_argument("--disable-slm",      action="store_true", default=False,
                        help="Ablation: disable ERL-SLM (Stage 1 ASR skipped, go straight to SDFT)")
    parser.add_argument("--disable-swdft",    action="store_true", default=False,
                        help="Ablation: disable SWDFT proactive stagnation detector")
    parser.add_argument("--disable-acc",      action="store_true", default=False,
                        help="Ablation: disable ACC (run full max_steps budget always)")
    parser.add_argument("--disable-der",      action="store_true", default=False,
                        help="Ablation: disable DER++ experience replay buffer")
    parser.add_argument("--disable-brfd",     action="store_true", default=False,
                        help="Ablation: disable BRFD reward shaping (use fixed weights 0.65/0.20/0.10)")
    parser.add_argument("--disable-erl",      action="store_true", default=False,
                        help="Ablation: disable ERL orchestrator entirely (no stagnation recovery)")
    parser.add_argument("--ablation-label",   type=str, default=None,
                        help="Label for this ablation run (written to scalability_report)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.remove()
    log.add(sys.stderr, level="INFO")
    log.add(results_dir / "quasar_v20.log", level="DEBUG", rotation="100 MB")

    log.info("=" * 70)
    log.info("QUASAR v25 — Noise-Adaptive Curriculum Learning + FASTMIX-Omega Bilevel Outer Loop")
    log.info(f"  Device:       {DEVICE}")
    log.info(f"  Qubit range:  {args.min_qubits} → {args.max_qubits} (adaptive)")
    log.info(f"  Targets:      {args.targets}")
    log.info(f"  Seeds:        {args.seeds}"
             + (f"  [filter: {args.seed_filter}]" if args.seed_filter else ""))
    log.info(f"  F threshold:  adaptive (0.999 for <=3Q, 0.99 for >4Q) [override={args.f_threshold}]")
    log.info(f"  F floor:      adaptive (0.999 for <=3Q, 0.99 for >4Q) [override={args.f_floor}]")
    log.info(f"  Max steps:    {args.max_steps_base:,} @ {args.min_qubits}Q, "
             f"×{args.steps_scale} per qubit")
    log.info(f"  Noise stage:  {args.noise_stage}")
    log.info(f"  DEHB Outer:   {args.outer_brackets} brackets")
    log.info(f"  DEHB Inner:   {args.inner_brackets} brackets (reactive)")
    if args.fixed_hp_json:
        log.info(f"  Fixed HP:     {args.fixed_hp_json} (DEHB Outer bypassed)")
    log.info("=" * 70)

    # ── Load fixed HP if provided ─────────────────────────────────────────────
    _fixed_hp: Optional[dict] = None
    if args.fixed_hp_json:
        try:
            with open(args.fixed_hp_json) as _f:
                _hp_data = json.load(_f)
            # Support both {"config": {...}} (incumbent.json format) and flat {"alpha": ...}
            _fixed_hp = _hp_data.get("config", _hp_data)
            log.info(f"  Loaded fixed HP: {_fixed_hp}")
        except Exception as _e:
            log.error(f"  Failed to load fixed HP from {args.fixed_hp_json}: {_e}")
            log.warning("  Falling back to DEHB Outer (fixed HP load failed)")

    # ── A7: Reflection collector ──────────────────────────────────────────────
    _dataset_path = Path(args.fastmix_dataset) if args.fastmix_dataset else (
        results_dir / "fastmix_reflections.jsonl")
    reflection_collector = ReflectionCollector(_dataset_path)
    log.info(f"  FASTMIX dataset: {_dataset_path}")

    # ── A2/A3: FASTMIX-Omega — initialise with Q-Forge default weights ────────
    fastmix = FastMixOmega(
        seed_weights=FASTMIX_QFORGE_WEIGHTS["default"],
        reflection_collector=reflection_collector,
    )
    log.info(f"  FASTMIX-Omega init weights: {fastmix.weights}")


    # ── v21 Phase 1: TraceCollector + SLMLoRATrainer ─────────────────────────────
    _trace_seed = args.seeds[0] if args.seeds else 42
    trace_collector = OPIDTraceCollector(
        seed=_trace_seed,
        buffer_size=TRACE_BUFFER_SIZE,
    )
    slm_lora_trainer = OPIDLoRATrainer(
        seed=_trace_seed,
        lora_rank=LORA_RANK,
        lora_lr=LORA_LR,
        lora_epochs=LORA_EPOCHS,
        lora_batch=LORA_BATCH_SIZE,
    )
    # ACI-SLM (HB-79): wire SLMACIRepository to the LoRA trainer
    _slm_aci = SLMACIRepository()
    slm_lora_trainer._aci_repo = _slm_aci
    log.info("SLM-ACI: SLMACIRepository wired to slm_lora_trainer")
    _lora_last_ft_step = 0
    _total_training_steps = 0  # accumulated across all cells for LoRA trigger
    log.info(f"  TraceCollector: {trace_collector.summary()}")
    log.info(f"  SLMLoRATrainer: {slm_lora_trainer.summary()}")
    # ── Ablation flags ─────────────────────────────────────────────────────────
    _abl = {
        "fastmix": not args.disable_fastmix,
        "slm":     not args.disable_slm,
        "swdft":   not args.disable_swdft,
        "acc":     not args.disable_acc,
        "der":     not args.disable_der,
        "brfd":    not args.disable_brfd,
        "erl":     not args.disable_erl,
    }
    log.info(f"  Ablation flags (enabled=True): {_abl}")
    if args.ablation_label:
        log.info(f"  Ablation label: {args.ablation_label}")
    # Disable FASTMIX-Omega updates if requested
    if args.disable_fastmix:
        fastmix._disabled = True
        log.info("  FASTMIX-Omega DISABLED for this ablation run")

    der_buffer: Optional[DERPlusPlusBuffer] = None
    scalability_report = {
        "config":             vars(args),
        "ablation_label":     args.ablation_label,
        "ablation_flags":     _abl,
        "start_time":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qubit_levels":       {},
        "scalability_ceiling": None,
        "ceiling_reason":     None,
        "fastmix_summary":    {},
    }
    prev_best_agent: Optional[SACAgent] = None
    ceiling_reached = False

    for n_qubits in range(args.min_qubits, args.max_qubits + 1):
        extra     = n_qubits - args.min_qubits
        max_steps = int(args.max_steps_base * (args.steps_scale ** extra))
        max_steps = min(max_steps, 5_000_000)

        log.info("")
        log.info("=" * 70)
        log.info(f"  QUBIT LEVEL: {n_qubits}Q  (budget={max_steps:,} steps)")
        log.info("=" * 70)
        t0 = time.time()

        if der_buffer is None:
            # v24 FIX: probe actual SiliQunEnv dims instead of using
            # the old QuantumCircuitEnv formula (4*(2**n)+n+2) which is wrong
            # for SiliQun (obs_dim=7 or 8, act_dim=9 for 2Q SiMOS)
            _probe_env = SiliQunEnvWrapper(
                n_qubits=args.min_qubits,
                target_state=args.targets[0],
                noise_stage=args.noise_stage,
                max_ep_steps=200,
                seed=args.seeds[0] if args.seeds else 42,
            )
            _obs_dim = _probe_env.observation_space.shape[0]
            _act_dim = _probe_env.action_space.shape[0]
            _tgt_dim = _probe_env.target.shape[0]
            del _probe_env
            der_buffer = DERPlusPlusBuffer(
                obs_dim=_obs_dim, act_dim=_act_dim, capacity=DER_CAPACITY)
            log.info(f"DER++ buffer init: obs_dim={_obs_dim}, act_dim={_act_dim} "
                     f"tgt_dim={_tgt_dim} [probed from SiliQunEnv]")
        cell_results, best_agent = run_qubit_level(
            n_qubits=n_qubits,
            targets=args.targets,
            seeds=args.seeds,
            noise_stage=args.noise_stage,
            max_steps=max_steps,
            outer_brackets=args.outer_brackets,
            inner_brackets=args.inner_brackets,
            results_dir=results_dir,
            ckpt_dir=ckpt_dir,
            init_agent=prev_best_agent,
            der_buffer=(der_buffer if _abl["der"] else None),
            seed_filter=args.seed_filter,
            fixed_hp=_fixed_hp,
            disable_slm=args.disable_slm,
            disable_swdft=args.disable_swdft,
            disable_acc=args.disable_acc,
            disable_brfd=args.disable_brfd,
            disable_erl=args.disable_erl,
            trace_collector=trace_collector,  # v21 Phase 1
        )
        elapsed = time.time() - t0
        _total_training_steps += max_steps

        # v21 Phase 1: LoRA fine-tune trigger
        if trace_collector.should_finetune(
                current_step=_total_training_steps,
                last_ft_step=_lora_last_ft_step):
            log.info(f"  [v21] LoRA fine-tune trigger: "
                     f"step={_total_training_steps:,} "
                     f"n_traces={trace_collector.n_since_finetune}")
            _ft_ok = slm_lora_trainer.finetune(
                traces=trace_collector.get_training_batch(),
                slm_reflector=None,  # no shared SLMReflector in main loop
                current_best_F=float(best_F),
            )
            if _ft_ok:
                trace_collector.mark_finetuned()
                _lora_last_ft_step = _total_training_steps
                log.info(f"  [v21] LoRA fine-tune complete: {slm_lora_trainer.summary()}")
            else:
                log.warning("  [v21] LoRA fine-tune skipped (PEFT unavailable or no data)")

        # ── A8: ZNE post-processing (if enabled) ─────────────────────────────
        if args.enable_zne:
            cell_results = apply_zne_to_results(cell_results)

        # ── A2/A3: FASTMIX-Omega update (skip if disabled) ──────────────────────
        # Compute per-target mean F for FASTMIX activity signal
        per_target_mean_for_fastmix = {
            t: float(np.mean(list(sd.values())))
            for t, sd in cell_results.items() if sd
        }
        level_mean_F = float(np.mean(list(per_target_mean_for_fastmix.values())))
        # Activity: W/Cluster cells drive SLM/ASR; GHZ/Dicke drive DEHB/BRFD
        _hard_cells = {t for t in cell_results if t in ("W", "Cluster")}
        _hard_ratio = len(_hard_cells) / max(len(cell_results), 1)
        fastmix_activity = {
            "alpha_SLM":    0.5 + 0.5 * _hard_ratio,
            "alpha_BRFD":   0.3,
            "alpha_DEHB":   0.3 - 0.2 * _hard_ratio,
            "alpha_replay": 0.4,
            "alpha_ASR":    0.3 + 0.4 * _hard_ratio,
            "alpha_SWDFT":  0.2,
            "alpha_ACC":    0.1,
        }
        if _abl["fastmix"]:
            # v21 Phase 1: pass n_slm_calls and n_total_calls for CG update
            _n_slm_traces   = trace_collector.n_since_finetune
            _n_total_traces = max(trace_collector.n_total, 1)
            fastmix.update(
                cell=f"{n_qubits}Q",
                best_F=level_mean_F,
                activity=fastmix_activity,
                n_successful_reflections=reflection_collector.n_successful,
                baseline_F_before_cell=fastmix._baseline_F,
                n_slm_calls=_n_slm_traces,
                n_total_calls=_n_total_traces,
            )
        scalability_report["fastmix_summary"] = fastmix.summary()

        # ── A4: Adaptive qubit scaling policy ────────────────────────────────
        _f_threshold, _f_floor = get_advance_thresholds(n_qubits)
        # Allow CLI override
        if args.f_threshold is not None:
            _f_threshold = args.f_threshold
        if args.f_floor is not None:
            _f_floor = args.f_floor
        log.info(f"  Advance thresholds: mean_F>{_f_threshold} AND min_F>{_f_floor} "
                 f"(policy: {'strict 0.999' if n_qubits <= 3 else 'high 0.99'})")

        advance, mean_F, min_F = should_advance(
            cell_results, _f_threshold, _f_floor)

        per_target_mean = per_target_mean_for_fastmix
        level_summary = {
            "n_qubits":        n_qubits,
            "mean_F":          round(mean_F, 5),
            "min_F":           round(min_F, 5),
            "per_target_mean": {k: round(v, 5) for k, v in per_target_mean.items()},
            "cell_results":    {t: {str(s): round(f, 5) for s, f in sd.items()}
                                for t, sd in cell_results.items()},
            "max_steps_used":  max_steps,
            "elapsed_s":       round(elapsed, 1),
            "advanced":        advance,
        }
        scalability_report["qubit_levels"][str(n_qubits)] = level_summary
        log.info(f"  {n_qubits}Q SUMMARY: mean_F={mean_F:.4f}  min_F={min_F:.4f}")
        for t, v in per_target_mean.items():
            log.info(f"    {t}: mean_F={v:.4f}")

        report_path = results_dir / "scalability_report.json"
        report_path.write_text(json.dumps(scalability_report, indent=2))

        if advance:
            log.info(f"  ✓ {n_qubits}Q PASSED → advancing to {n_qubits + 1}Q")
            prev_best_agent = best_agent
        else:
            reason = (f"mean_F={mean_F:.4f} < threshold={_f_threshold}"
                      if mean_F < _f_threshold
                      else f"min_F={min_F:.4f} < floor={_f_floor}")
            if n_qubits < args.min_qubits_ceiling:
                # Do not terminate early at low qubit levels — continue regardless
                log.warning(
                    f"  ⚠ {n_qubits}Q below min_qubits_ceiling={args.min_qubits_ceiling}: "
                    f"ceiling suppressed ({reason}), advancing anyway")
                prev_best_agent = best_agent
            else:
                log.info(f"  ✗ {n_qubits}Q STALLED ({reason})")
                log.info(f"  ══ SCALABILITY CEILING: N* = {n_qubits - 1}Q ══")
                scalability_report["scalability_ceiling"] = n_qubits - 1
                scalability_report["ceiling_reason"]      = reason
                ceiling_reached = True
                break

    if not ceiling_reached:
        scalability_report["scalability_ceiling"] = args.max_qubits
        scalability_report["ceiling_reason"] = (
            f"Reached hard max-qubits={args.max_qubits}")
        log.info(f"  ══ EXPERIMENT COMPLETE: N* >= {args.max_qubits}Q ══")

    scalability_report["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report_path = results_dir / "scalability_report.json"
    report_path.write_text(json.dumps(scalability_report, indent=2))
    log.info(f"  FINAL RESULT: N* = {scalability_report['scalability_ceiling']}Q")
    log.info(f"  FASTMIX-Omega final: {fastmix.summary()}")
    log.info(f"  Reflection dataset: {reflection_collector.n_successful} successful, "
             f"{reflection_collector.n_failed} failed")
    log.info(f"  Report: {report_path}")

if __name__ == "__main__":
    main()