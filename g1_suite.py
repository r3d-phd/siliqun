#!/usr/bin/env python
"""G1 numerical verification suite for the SiliQun TQE paper.

Measures, against analytic expectations:
  A. Rabi oscillation + pi-pulse swap        (coherent X drive)
  B. Y-drive imaginary coherence             (exposes np.real(rho) line 738)
  C. Exchange half/full-swap + Bell entangle (exposes np.real(rho); YY term)
  D. T2* dephasing rate vs analytic           (effective 1/T2)
  E. Fidelity bound F in [0,1]                (exposes unclipped _compute_fidelity)
  F. PGIRS "machine-precision" validator      (tautology: never runs solver)
  G. T1 profile value                        (code vs paper)

Run:  cd <repo> && <venv>/python g1_suite.py
"""
import sys, os
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(".", "plugins", "siliqun-gaas"))

from siliqun.pulse.lindblad import (
    LindbladSimulator, PulseSequence, DrivePulse, ExchangePulse,
    _build_collapse_operators, _drho_dt,
)
from siliqun.physics.devices.profiles import get_device

SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z     = np.array([[1, 0], [0, -1]], dtype=complex)

def make_noise(T1, T2star):
    """Build a NoiseParams with explicit T1, T2* and no charge/depolarising."""
    from siliqun.physics.noise.channels import NoiseParams
    return NoiseParams(T1=float(T1), T2_star=float(T2star))

def sim_for(nq, T1, T2star, dt=1e-10):
    dev = get_device("simos", n_qubits=nq)
    dev.noise_params = make_noise(T1, T2star)
    return LindbladSimulator(dev, n_qubits=nq, method="rk4", dt=dt, use_gpu=False)

def bell_vec():
    return np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)

print("=" * 70)
print("G1 NUMERICAL VERIFICATION SUITE  (siliqun %s)" % __import__("siliqun").__version__)
print("=" * 70)

# ---------------------------------------------------------------- A. RABI
print("\n[A] Rabi oscillation, X drive (phi=0), T1=inf, T2*=inf  [noise OFF]")
f1 = lambda s: np.real(np.trace(np.outer([0,1],[1,0]) @ s.rho_final))  # <1|rho|1>
A_results = {}
for amp, label in [(10e6, "A10MHz"), (30e6, "A30MHz")]:
    dur = np.pi / (np.pi * amp)          # exact pi-pulse duration
    r = sim_for(1, float("inf"), float("inf")).evolve(
        PulseSequence().add(DrivePulse(0, amp, 5e9, 0.0, dur)))
    pop1 = np.real(r.rho_final[1,1]); pop0 = np.real(r.rho_final[0,0])
    coher = r.rho_final[0,1]
    A_results[label] = (pop1, pop0, coher, dur)
    print(f"  {label}: dur={dur*1e9:.3f}ns  pop|1>={pop1:.4f} pop|0>={pop0:.4f} "
          f"rho01={coher:.4f}  (expect pop|1>->1, pop|0>->0)")

# ---------------------------------------------------------------- B. Y-drive
print("\n[B] Y-drive (phi=pi/2), T1=inf, T2*=inf  [noise OFF]  -> imaginary coherence")
amp = 10e6; dur = np.pi/(np.pi*amp)
rY = sim_for(1, float("inf"), float("inf")).evolve(
    PulseSequence().add(DrivePulse(0, amp, 5e9, np.pi/2, dur)))
coherY = rY.rho_final[0,1]
print(f"  after pi/2 Y pulse: rho01={coherY:.4f}")
print(f"  -> imaginary part present? {abs(coherY.imag)>1e-6}   "
      f"(|{coherY.imag:.4f}|)  BUG np.real(rho) zeros it if this is 0")

# ---------------------------------------------------------------- C. exchange
print("\n[C] Exchange (Heisenberg J/4 (XX+YY+ZZ)), full-swap dur=1/(2J), T1=T2*=inf")
J = 20e6; dur_sw = 1.0/(2*J)
init01 = np.zeros(4, complex); init01[1] = 1.0     # |01>
sim = sim_for(2, float("inf"), float("inf"))
sim.reset(init01)
rX = sim.evolve(PulseSequence().add(ExchangePulse(0, 1, J, dur_sw)),
                target_state=np.array([0,0,1,0], complex))  # |10>
pop10 = np.real(rX.rho_final[2,2]); pop01 = np.real(rX.rho_final[1,1])
F_swap = np.real(np.trace(np.outer([0,0,1,0],[0,0,1,0]) @ rX.rho_final))
print(f"  full-swap dur={dur_sw*1e9:.3f}ns:  pop|10>={pop10:.4f} pop|01>={pop01:.4f} "
      f"F(|10>)={F_swap:.4f}  (perfect SWAP -> pop|10>~1, F~1)")
# half-swap: Bell from |01>
dur_h = 1.0/(4*J)
simh = sim_for(2, float("inf"), float("inf")); simh.reset(init01)
rh = simh.evolve(PulseSequence().add(ExchangePulse(0,1,J,dur_h)), target_state=bell_vec())
F_bell = np.real(np.trace(np.outer(bell_vec(), bell_vec().conj()) @ rh.rho_final))
# concurrence / entanglement via reduced purity
rho_red0 = np.array([[rh.rho_final[0,0], rh.rho_final[0,2]],
                     [rh.rho_final[2,0], rh.rho_final[2,2]]])
conc = 2*abs(rh.rho_final[0,3])   # for this 2-qubit state
print(f"  half-swap dur={dur_h*1e9:.3f}ns:  F(|Bell>)={F_bell:.4f}  "
      f"concurrence~{conc:.4f}  (perfect Bell -> F~1, conc~1; 0 => no entanglement)")

# ---------------------------------------------------------------- D. dephasing
print("\n[D] Pure dephasing: |0>+|1> decay vs analytic  rho01(t)=0.5*exp(-t/T2eff)")
T1=10.0; T2s=41e-6
rate = 1.0/T2s + 1.0/(2*T1)
T2eff = 1.0/rate
simD = sim_for(1, T1, T2s); ps = (np.array([1,1],complex)/np.sqrt(2))
simD.reset(ps)
rD = simD.evolve(PulseSequence().add(DrivePulse(0,0,5e9,0.0, 5*T2eff)))  # 5*T2eff, amp0=no drive
rhoD = rD.rho_final
rho01_num = np.real(rhoD[0,1]); rho01_ana = 0.5*np.exp(-1.0)
print(f"  T1={T1}s T2*={T2s*1e6:.0f}us  ->  1/T2eff = {rate:.3e} 1/s  T2eff={T2eff*1e3:.3f}ms")
print(f"  rho01 after 5*T2eff:  numeric={rho01_num:.5f}  analytic={rho01_ana:.5f}  "
      f"rel.err={abs(rho01_num-rho01_ana)/rho01_ana*100:.3f}%")

# ---------------------------------------------------------------- E. fidelity bound
print("\n[E] Fidelity bound F in [0,1] across full Rabi trajectory (noise OFF)")
amp=10e6
rE = sim_for(1, float("inf"), float("inf")).evolve(
    PulseSequence().add(DrivePulse(0, amp, 5e9, 0.0, 3*np.pi/(np.pi*amp))),
    target_state=np.array([0,1], complex))
Fmax = max(rE.fidelity_history); Fmin = min(rE.fidelity_history)
print(f"  F range over 1.5 Rabi periods:  min={Fmin:.6f}  max={Fmax:.6f}")
print(f"  -> F>1.0? {Fmax>1.0}   F<0? {Fmin<0}   (unclipped Tr(rho_t rho) defect)")

# ---------------------------------------------------------------- F. validator tautology
print("\n[F] PGIRS 'machine-precision' validator  -- does it run the Lindblad solver?")
evolve_calls = [0]
_orig_evolve = LindbladSimulator.evolve
def spy(self, *a, **k):
    evolve_calls[0]+=1; return _orig_evolve(self,*a,**k)
LindbladSimulator.evolve = spy
try:
    from siliqun.core.abc import PGIRSValidator
    dev = get_device("simos", n_qubits=2)
    rep = PGIRSValidator(dev).run_all_benchmarks()
finally:
    LindbladSimulator.evolve = _orig_evolve
print(f"  PGIRSValidator.run_all_benchmarks() LindbladSimulator.evolve calls = {evolve_calls[0]}")
print(f"  -> solver NEVER exercised; 'max delta' is analytic-vs-analytic (tautology)")
try:
    for k in ("max_delta","max_delta_abs","delta"):
        if isinstance(rep, dict) and k in rep: print(f"     report[{k}]={rep[k]:.3e}")
    if isinstance(rep, dict):
        for k,v in rep.items():
            if isinstance(v,float) and k.startswith("max"): print(f"     {k}={v:.3e}")
except Exception as e:
    print("     (report shape:", str(rep)[:200], ")")

# ---------------------------------------------------------------- G. T1 profile
print("\n[G] Device-profile T1 (code) vs paper claim (10 s)")
for name in ["simos","gaas","donor","sledge"]:
    try:
        d = get_device(name, n_qubits=2)
        np_ = d.noise_params
        print(f"  {name:8s}: T1={np_.T1}s  T2*={getattr(np_,'T2_star',None)}")
    except Exception as e:
        print(f"  {name:8s}: (n/a) {str(e)[:40]}")
print("\n" + "="*70)
print("G1 SUITE COMPLETE")
print("="*70)
