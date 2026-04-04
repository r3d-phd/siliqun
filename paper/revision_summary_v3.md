# SiliQun SoftwareX Paper v3 - Revision Summary

## Overview

Paper v3 comprehensively addresses all 10 issues identified by the DeepSeek reviewer evaluation. The paper grew from 12 pages (v2) to 17 pages (v3), with 3 new figures, 3 new tables, 3 new references, and 2 new sections.

## Changes by Issue

### CRITICAL Issues (All Resolved)

| Issue | Description | Resolution | Location |
|-------|-------------|------------|----------|
| #1 | Missing DFS validation against ground truth | New Section 3.2 with systematic validation study comparing projected vs full physical-space dynamics for 2-logical-qubit system (6 physical spins). Three-panel Figure 5 shows: (a) perturbative leakage scaling as O(theta^2), (b) full coupling range behavior, (c) encoded SWAP verification | Section 3.2, Figure 5 |
| #2 | No DRL training results shown | New learning curves from REINFORCE agent on 2q Bell and 4q GHZ tasks. 2q achieves >0.99 fidelity in ~50 episodes. Figure 4 with reward and fidelity panels | Section 3.1, Figure 4 |
| #3 | Leakage tracking is diagnostic only | Quantified leakage regimes: <2% for theta < 17 degrees (perturbative), scaling as O(theta^2) with measured exponent 2.00. Limitations section explicitly warns about regimes where projection breaks down | Section 3.2, Section 5 |

### IMPORTANT Issues (All Resolved)

| Issue | Description | Resolution | Location |
|-------|-------------|------------|----------|
| #4 | Benchmarking details incomplete | CPU baseline specified: single-threaded NumPy on AMD EPYC 7763. New installation subsection (Section 2.3). New API documentation table (Table 2) | Section 2.3, Section 3.3, Table 2 |
| #5 | Limited novelty in MPS engine | Clarified MPS scope: "lightweight, purpose-built implementation... not intended to compete with general-purpose tensor network libraries such as ITensor or quimb." Added ITensor and quimb references | Section 2.1, Section 5 |
| #6 | Single-GPU focus | New Limitations section discusses multi-GPU scaling via NCCL/cuQuantum as planned extension. Notes 30-qubit limit on single 80 GB GPU | Section 5 |

### MINOR Issues (All Resolved)

| Issue | Description | Resolution | Location |
|-------|-------------|------------|----------|
| #7 | Author affiliation | Student email (stu.kau.edu.sa) is appropriate for SoftwareX; no change needed | N/A |
| #8 | Fig 4 typo ("igure 4") | All figure references now use consistent "Figure X" format | Throughout |
| #9 | Reference [8] incomplete | Schaal et al. now has full author list and correct journal info (Nature Electronics, 2(6), 236-242) | Reference [8] |
| #10 | Abstract "exact simulations" | Added qualification: "Within this logical subspace, the projected dynamics are exact for intra-qubit operations, while inter-qubit coupling introduces leakage that scales as O(theta^2)" | Abstract |

## New Content Added

### New Sections
- **Section 2.3: Installation** - Git clone + pip install instructions, CuPy setup guidance
- **Section 3.4: Computational Complexity** - O(2^n) scaling analysis with Table 5
- **Section 5: Limitations and Future Work** - Honest discussion of single-GPU, leakage regimes, MPS scope, device extensibility
- **Code Availability** - Explicit statement with GitHub URL and version

### New Figures
- **Figure 4**: DRL learning curves (reward + fidelity, 2q Bell + 4q GHZ)
- **Figure 5**: DFS validation (3 panels: perturbative regime, full range, encoded SWAP)

### New Tables
- **Table 2**: API documentation (7 modules with descriptions)
- **Table 5**: Computational complexity (7 operations with time/memory scaling)
- **Table 6**: Expanded comparison (now includes PennyLane, 7 features vs 5 in v2)

### New References
- [18] ITensor (Fishman et al., 2022)
- [19] quimb (Gray, 2018)
- [20] PennyLane (Bergholm et al., 2022)

## Paper Statistics

| Metric | v2 | v3 |
|--------|----|----|
| Pages | 12 | 17 |
| Figures | 4 | 7 |
| Tables | 4 | 6 |
| References | 17 | 20 |
| Sections | 5 | 7 |

## Files

- `softwarex_siliqun_v3.tex` - LaTeX source
- `softwarex_siliqun_v3.pdf` - Compiled PDF (17 pages)
- `fig_dfs_validation.png` - DFS validation figure
- `fig_learning_curves.png` - DRL learning curves figure
- `deepseek_evaluation_feedback.md` - Original reviewer feedback
