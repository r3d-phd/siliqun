"""
Adaptive Convergence Controller (ACC)
======================================
A unified drop-in replacement for QUASAR's fixed step budget and naive plateau
counter.  It combines two complementary components:

  1. Budget Predictor (BP)  — fits F(t) = F∞·(1−e^{−t/τ}) to the rising
     portion of the learning curve and predicts T*, the minimum steps needed
     to reach F_threshold.

  2. Persistent Plateau Detector (PPD) — applies three tests to decide whether
     training should stop:
       a. Slope test      : is the curve still rising?
       b. Asymptote test  : is F_threshold reachable at all (F∞ ≥ threshold)?
       c. Convergence test: has the agent already passed T* (diminishing returns)?

Usage (drop-in for any training loop)
--------------------------------------
    from acc import AdaptiveConvergenceController, StopReason

    acc = AdaptiveConvergenceController(
        F_threshold   = 0.99,
        safety_margin = 1.20,    # T_budget = T* × 1.20
        window        = 5,       # checkpoints for slope test
        min_points    = 5,       # minimum checkpoints before asymptote test
        r2_min        = 0.70,    # minimum R² to trust the fit
        slope_eps     = 1e-7,    # steps/step threshold for "flat"
        diminish_frac = 0.05,    # stop if predicted remaining gain < 5% of T*
        max_budget    = 500_000, # hard ceiling (never exceed this)
        label         = "2Q/GHZ/s42",
    )

    for step in range(0, 500_001, 10_000):
        best_F = train(step)
        decision = acc.update(step, best_F)
        if decision.stop:
            print(f"Stop at step {step}: {decision.reason.name}")
            print(f"  Recommended budget for next run: {acc.recommended_budget:,}")
            break

    acc.save_log("acc_log_2Q_GHZ_s42.json")
    fig = acc.plot()
    fig.savefig("acc_2Q_GHZ_s42.png")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── optional scipy; fall back to numpy polyfit if unavailable ──────────────
try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Public enums and data classes
# ─────────────────────────────────────────────────────────────────────────────

class StopReason(Enum):
    NONE            = auto()   # no stop yet
    THRESHOLD_MET   = auto()   # F ≥ F_threshold observed directly
    CONVERGED       = auto()   # past T*: diminishing returns below threshold
    UNREACHABLE     = auto()   # F∞ < F_threshold: target is unreachable
    FLAT_UNREACHABLE= auto()   # slope flat AND F∞ < threshold
    MAX_BUDGET      = auto()   # hard ceiling reached
    MANUAL          = auto()   # caller forced a stop


@dataclass
class ACCDecision:
    stop:               bool
    reason:             StopReason
    step:               int
    best_F:             float
    F_inf:              Optional[float]   = None
    tau:                Optional[float]   = None
    T_star:             Optional[float]   = None
    recommended_budget: Optional[int]     = None
    r2:                 Optional[float]   = None
    slope:              Optional[float]   = None
    message:            str               = ""


@dataclass
class _Checkpoint:
    step:   int
    best_F: float


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveConvergenceController:
    """
    Unified Budget Predictor + Persistent Plateau Detector.

    Call ``update(step, best_F)`` at every checkpoint (e.g. every 10k steps).
    The returned ``ACCDecision`` tells you whether to stop and why.
    """

    def __init__(
        self,
        F_threshold:    float = 0.99,
        safety_margin:  float = 1.20,
        window:         int   = 5,
        min_points:     int   = 5,
        r2_min:         float = 0.70,
        slope_eps:      float = 1e-7,
        diminish_frac:  float = 0.05,
        max_budget:     int   = 500_000,
        label:          str   = "run",
    ):
        self.F_threshold   = F_threshold
        self.safety_margin = safety_margin
        self.window        = window
        self.min_points    = min_points
        self.r2_min        = r2_min
        self.slope_eps     = slope_eps
        self.diminish_frac = diminish_frac
        self.max_budget    = max_budget
        self.label         = label

        self._history:  List[_Checkpoint] = []
        self._decisions: List[ACCDecision] = []
        self._stopped   = False
        self._stop_decision: Optional[ACCDecision] = None

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, step: int, best_F: float) -> ACCDecision:
        """
        Register a new checkpoint and return a stop decision.

        Parameters
        ----------
        step   : current training step
        best_F : best fidelity observed so far

        Returns
        -------
        ACCDecision — check ``.stop`` to decide whether to break the loop.
        """
        if self._stopped:
            return self._stop_decision  # type: ignore

        self._history.append(_Checkpoint(step, best_F))

        # ── Test 0: threshold already met ────────────────────────────────────
        if best_F >= self.F_threshold:
            # Try to compute a fit even at threshold for recommended_budget
            n = len(self._history)
            rec_budget = None
            F_inf_val = None
            tau_val = None
            r2_val = None
            if n >= self.min_points:
                steps_arr = np.array([c.step   for c in self._history], dtype=float)
                fids_arr  = np.array([c.best_F for c in self._history], dtype=float)
                fit0 = self._fit_rising(steps_arr, fids_arr)
                if fit0 and fit0['r2'] >= self.r2_min and fit0['F_inf'] > self.F_threshold:
                    T_star0 = -fit0['tau'] * np.log(1 - self.F_threshold / fit0['F_inf'])
                    rec_budget = int(T_star0 * self.safety_margin)
                    F_inf_val = fit0['F_inf']
                    tau_val   = fit0['tau']
                    r2_val    = fit0['r2']
            dec = self._make_decision(
                stop=True, reason=StopReason.THRESHOLD_MET,
                step=step, best_F=best_F,
                F_inf=F_inf_val, tau=tau_val, r2=r2_val,
                T_star=rec_budget / self.safety_margin if rec_budget else None,
                recommended_budget=rec_budget,
                message=f"F={best_F:.5f} >= threshold {self.F_threshold}"
            )
            return self._record(dec)

        # ── Test 1: hard ceiling ──────────────────────────────────────────────
        if step >= self.max_budget:
            dec = self._make_decision(
                stop=True, reason=StopReason.MAX_BUDGET,
                step=step, best_F=best_F,
                message=f"Hard ceiling {self.max_budget:,} reached"
            )
            return self._record(dec)

        n = len(self._history)

        # ── Need enough data for statistical tests ────────────────────────────
        if n < self.min_points:
            return self._make_decision(
                stop=False, reason=StopReason.NONE,
                step=step, best_F=best_F,
                message=f"Collecting data ({n}/{self.min_points} checkpoints)"
            )

        steps = np.array([c.step   for c in self._history], dtype=float)
        fids  = np.array([c.best_F for c in self._history], dtype=float)

        # ── Slope test ────────────────────────────────────────────────────────
        recent_steps = steps[-self.window:]
        recent_fids  = fids[-self.window:]
        slope = float(np.polyfit(recent_steps, recent_fids, 1)[0])
        is_flat = slope < self.slope_eps

        # ── Asymptote fit ─────────────────────────────────────────────────────
        fit = self._fit_rising(steps, fids)

        if fit is None or fit['r2'] < self.r2_min:
            # Not enough signal yet — continue
            return self._make_decision(
                stop=False, reason=StopReason.NONE,
                step=step, best_F=best_F, slope=slope,
                message=f"Fit unreliable (R²={(fit['r2'] if fit else 0):.3f})"
            )

        F_inf = fit['F_inf']
        tau   = fit['tau']
        r2    = fit['r2']

        # Compute T* and recommended budget
        T_star = None
        rec_budget = None
        if F_inf > self.F_threshold:
            T_star = -tau * np.log(1.0 - self.F_threshold / F_inf)
            rec_budget = int(T_star * self.safety_margin)

        # ── Test 2: unreachable ───────────────────────────────────────────────
        if F_inf < self.F_threshold:
            if is_flat:
                dec = self._make_decision(
                    stop=True, reason=StopReason.FLAT_UNREACHABLE,
                    step=step, best_F=best_F,
                    F_inf=F_inf, tau=tau, r2=r2, slope=slope,
                    T_star=T_star, recommended_budget=rec_budget,
                    message=(f"Slope flat AND F∞={F_inf:.4f} < threshold "
                             f"{self.F_threshold} — target unreachable")
                )
                return self._record(dec)
            else:
                # Still rising but F∞ < threshold — warn but continue
                return self._make_decision(
                    stop=False, reason=StopReason.NONE,
                    step=step, best_F=best_F,
                    F_inf=F_inf, tau=tau, r2=r2, slope=slope,
                    T_star=T_star, recommended_budget=rec_budget,
                    message=(f"WARNING: F∞={F_inf:.4f} < threshold — "
                             f"still rising (slope={slope:.2e}), continuing")
                )

        # F∞ ≥ threshold — target is reachable
        assert T_star is not None

        # ── Test 3: convergence (past T*) ─────────────────────────────────────
        remaining_gain = T_star - step
        if remaining_gain < 0 or (is_flat and step > T_star * self.safety_margin):
            dec = self._make_decision(
                stop=True, reason=StopReason.CONVERGED,
                step=step, best_F=best_F,
                F_inf=F_inf, tau=tau, r2=r2, slope=slope,
                T_star=T_star, recommended_budget=rec_budget,
                message=(f"Past T*={T_star:,.0f} — "
                         f"F∞={F_inf:.5f}, current F={best_F:.5f}, "
                         f"slope={slope:.2e}")
            )
            return self._record(dec)

        # ── No stop ───────────────────────────────────────────────────────────
        return self._make_decision(
            stop=False, reason=StopReason.NONE,
            step=step, best_F=best_F,
            F_inf=F_inf, tau=tau, r2=r2, slope=slope,
            T_star=T_star, recommended_budget=rec_budget,
            message=(f"Running: F∞={F_inf:.4f}, T*={T_star:,.0f}, "
                     f"step={step:,}, remaining≈{max(0,remaining_gain):,.0f}")
        )

    @property
    def recommended_budget(self) -> Optional[int]:
        """Best estimate of the recommended budget for this cell."""
        for dec in reversed(self._decisions):
            if dec.recommended_budget:
                return dec.recommended_budget
        return None

    @property
    def history(self) -> List[_Checkpoint]:
        return list(self._history)

    def save_log(self, path: str) -> None:
        """Save full decision log to JSON."""
        log = {
            'label':      self.label,
            'F_threshold': self.F_threshold,
            'safety_margin': self.safety_margin,
            'checkpoints': [
                {'step': c.step, 'best_F': c.best_F}
                for c in self._history
            ],
            'decisions': [
                {
                    'step':               d.step,
                    'best_F':             d.best_F,
                    'stop':               d.stop,
                    'reason':             d.reason.name,
                    'F_inf':              d.F_inf,
                    'tau':                d.tau,
                    'T_star':             d.T_star,
                    'recommended_budget': d.recommended_budget,
                    'r2':                 d.r2,
                    'slope':              d.slope,
                    'message':            d.message,
                }
                for d in self._decisions
            ],
        }
        with open(path, 'w') as f:
            json.dump(log, f, indent=2)

    def plot(self, figsize: Tuple[float, float] = (10, 5)) -> plt.Figure:
        """
        Generate a diagnostic figure with two panels:
          Left  — fidelity trajectory + saturating fit + T* marker
          Right — slope over time + flat threshold
        """
        if not self._history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig

        steps = np.array([c.step   for c in self._history], dtype=float)
        fids  = np.array([c.best_F for c in self._history], dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── Left panel: fidelity + fit ────────────────────────────────────────
        ax1.plot(steps / 1000, fids, 'o-', color='#2196F3', markersize=5,
                 linewidth=1.5, label='Observed best_F')
        ax1.axhline(self.F_threshold, color='gray', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'F={self.F_threshold}')

        fit = self._fit_rising(steps, fids)
        if fit and fit['r2'] >= self.r2_min:
            t_fit = np.linspace(0, steps.max() * 1.3, 400)
            f_fit = _sat_exp(t_fit, fit['F_inf'], fit['tau'])
            ax1.plot(t_fit / 1000, f_fit, '--', color='#FF9800', linewidth=2,
                     label=f"Fit: F∞={fit['F_inf']:.4f}, τ={fit['tau']/1000:.1f}k")
            ax1.axhline(fit['F_inf'], color='#FF9800', linestyle=':',
                        linewidth=1, alpha=0.5)

            if fit['F_inf'] > self.F_threshold:
                T_star = -fit['tau'] * np.log(1 - self.F_threshold / fit['F_inf'])
                T_safe = T_star * self.safety_margin
                ax1.axvline(T_star / 1000, color='red', linestyle=':',
                            linewidth=1.5, alpha=0.8,
                            label=f'T*={T_star/1000:.0f}k')
                ax1.axvline(T_safe / 1000, color='darkred', linestyle=':',
                            linewidth=1.5, alpha=0.5,
                            label=f'T*+{int((self.safety_margin-1)*100)}%='
                                  f'{T_safe/1000:.0f}k')

        # Mark stop point if any
        if self._stop_decision and self._stop_decision.stop:
            sd = self._stop_decision
            ax1.axvline(sd.step / 1000, color='green', linewidth=2, alpha=0.8,
                        label=f'ACC stop ({sd.reason.name})')

        ax1.set_xlabel('Training Steps (×10³)', fontsize=10)
        ax1.set_ylabel('Best Fidelity', fontsize=10)
        ax1.set_title(f'ACC: {self.label}\nFidelity + Saturating Fit',
                      fontsize=10, fontweight='bold')
        ax1.legend(fontsize=7.5, loc='lower right')
        ax1.set_ylim(max(0.4, fids.min() - 0.05), 1.02)
        ax1.grid(True, alpha=0.3)

        # ── Right panel: slope over time ──────────────────────────────────────
        slopes = []
        slope_steps = []
        for i in range(self.window, len(steps) + 1):
            s = steps[i - self.window:i]
            f = fids[i - self.window:i]
            slope = float(np.polyfit(s, f, 1)[0])
            slopes.append(slope)
            slope_steps.append(steps[i - 1])

        if slopes:
            ax2.semilogy(np.array(slope_steps) / 1000,
                         np.maximum(np.array(slopes), 1e-10),
                         color='#4CAF50', linewidth=1.5)
            ax2.axhline(self.slope_eps, color='red', linestyle='--',
                        linewidth=1, alpha=0.7,
                        label=f'Flat threshold ε={self.slope_eps:.0e}')
            ax2.set_xlabel('Training Steps (×10³)', fontsize=10)
            ax2.set_ylabel('dF/dt (slope, log scale)', fontsize=10)
            ax2.set_title('PPD: Slope Over Time', fontsize=10, fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, which='both')

        plt.suptitle(f'Adaptive Convergence Controller — {self.label}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    # ── private helpers ───────────────────────────────────────────────────────

    def _fit_rising(self, steps: np.ndarray, fids: np.ndarray) -> Optional[dict]:
        """Fit sat_exp to the rising portion of the curve."""
        # Find rising end: stop when F flat for 2 consecutive points
        rising_end = len(fids)
        for i in range(2, len(fids)):
            if fids[i] <= fids[i-1] and fids[i-1] <= fids[i-2]:
                rising_end = i
                break

        s = steps[:rising_end]
        f = fids[:rising_end]

        if len(s) < 3:
            return None

        F_max = float(f[-1])
        F0    = min(F_max + 0.005, 0.9999)
        tau0  = float(s[len(s)//2])

        try:
            if _HAS_SCIPY:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = _scipy_curve_fit(
                        _sat_exp, s, f,
                        p0=[F0, tau0],
                        bounds=([F_max * 0.98, 1.0], [1.0, s[-1] * 20]),
                        maxfev=20000
                    )
                F_inf, tau = float(popt[0]), float(popt[1])
            else:
                # Linearise: ln(1 - F/F0) = -t/tau → linear regression
                # Use F0 = observed max as proxy for F_inf
                F_inf = min(F_max * 1.01, 0.9999)
                y = np.log(np.maximum(1.0 - f / F_inf, 1e-10))
                coeffs = np.polyfit(s, y, 1)
                tau = float(-1.0 / coeffs[0])

            f_pred = _sat_exp(s, F_inf, tau)
            ss_res = float(np.sum((f - f_pred)**2))
            ss_tot = float(np.sum((f - f.mean())**2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            return {'F_inf': F_inf, 'tau': tau, 'r2': r2}
        except Exception:
            return None

    def _make_decision(self, stop: bool, reason: StopReason,
                       step: int, best_F: float, **kwargs) -> ACCDecision:
        dec = ACCDecision(stop=stop, reason=reason, step=step, best_F=best_F,
                          **kwargs)
        self._decisions.append(dec)
        return dec

    def _record(self, dec: ACCDecision) -> ACCDecision:
        """Mark as stopped and return decision."""
        self._stopped = True
        self._stop_decision = dec
        return dec


# ─────────────────────────────────────────────────────────────────────────────
# Helper (module-level so it can be pickled)
# ─────────────────────────────────────────────────────────────────────────────

def _sat_exp(t: np.ndarray, F_inf: float, tau: float) -> np.ndarray:
    return F_inf * (1.0 - np.exp(-t / tau))
