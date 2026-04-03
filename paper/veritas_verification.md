# Veritas Verification Results — SiliQun SoftwareX Paper

## Section 1: Motivation and Significance

**Risk Level:** MEDIUM (2 flagged, 5 claims to verify)

### Flagged Issues

1. **Coherence times exceeding 30 seconds** (confidence_concern)
   - Concern: While long coherence times are reported, 30s is at the extreme end
   - Resolution: This is correct — Muhonen et al. (2014) Nature Nanotechnology reports T1 = 30s for phosphorus donors in isotopically enriched 28Si. VALID.

2. **SLEDGE architecture** (fabricated concern)
   - Concern: Veritas couldn't find "SLEDGE architecture" in public literature
   - Resolution: SLEDGE is our internal name for the device architecture. We should either:
     a) Clarify it's our proposed architecture, OR
     b) Replace with a more general description of 2D DFS-encoded arrays
   - ACTION NEEDED: Update the paper to clarify SLEDGE is our proposed architecture

### Safe Claims (all verified)
- Silicon spin qubits as promising platform ✓
- CMOS compatibility ✓
- Gate fidelities >99.9% (1Q) and >99% (2Q) ✓
- Charge noise sensitivity ✓
- DRL as alternative to optimal control ✓
- MPS limitations for 2D volume-law entanglement ✓
- 2^75 intractability ✓

## Section 2: Software Description

**Risk Level:** MEDIUM (2 flagged, 4 claims to verify)

### Flagged Issues

1. **"Novel" SV engine claim** (exaggerated)
   - Concern: Novelty claim needs substantiation vs existing SV engines
   - Resolution: Qualify as "novel in the context of DFS-encoded silicon spin qubits" — the novelty is the logical subspace projection, not the SV concept itself
   - ACTION: Add qualifier "novel for DFS-encoded systems"

2. **"Exact" simulation with perturbative leakage** (misleading)
   - Concern: Calling it "exact" while tracking leakage perturbatively is contradictory
   - Resolution: Clarify that simulation is exact within the logical subspace, with leakage tracked as a perturbative correction
   - ACTION: Change "exact" to "exact within the logical subspace"

### All other claims verified as safe ✓

## Section 3: GPU Performance Benchmarks

**Risk Level:** LOW (0 flagged, 5 claims to verify)

All benchmark claims verified as internally consistent and arithmetically correct.
No fabrications, exaggerations, or technical inaccuracies detected.

## Section 4: Impact and Conclusions

**Risk Level:** MEDIUM (2 flagged, 4 claims to verify)

### Flagged Issues

1. **"Prior to SiliQun... required building custom simulators"** (exaggerated)
   - Concern: Overstates the gap — tools like QuTiP exist
   - Resolution: Soften to "lacked integrated solutions combining silicon-specific physics with DRL compatibility"
   - ACTION: Revise wording

2. **"Far beyond capabilities of MPS backends"** (exaggerated)
   - Concern: MPS can handle much larger systems approximately; comparison is misleading
   - Resolution: Clarify the comparison is about exactness, not scale
   - ACTION: Revise to "offering exactness where MPS would provide only approximate solutions"

## Summary of Required Revisions

| # | Section | Issue | Fix |
|---|---------|-------|-----|
| 1 | Motivation | SLEDGE not in public literature | Clarify as proposed/internal architecture |
| 2 | Software Desc | "Novel" overclaimed | Qualify as novel for DFS-encoded systems |
| 3 | Software Desc | "Exact" contradicts perturbative leakage | Clarify exact within logical subspace |
| 4 | Impact | Prior art overstated | Soften language |
| 5 | Impact | MPS comparison misleading | Clarify exactness vs scale |
