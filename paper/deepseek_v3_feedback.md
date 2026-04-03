# DeepSeek Evaluation of SiliQun v3 (Revised Manuscript)

**Source:** https://chat.deepseek.com/share/mhmeuzaf07slbxcy99

## Overall Verdict: Accept with minor formatting fixes

The revised version (v3) directly addresses nearly all of the major concerns raised in the previous evaluation. The manuscript is substantially stronger and now meets the bar for publication in a reputable software journal (e.g., SoftwareX).

## What Was Improved (and Done Well)

| Previous Weakness | How v3 Addresses It | Status |
|---|---|---|
| Missing validation of DFS projection | New Section 3.2: Systematic comparison of logical-subspace vs. full physical simulation for 2 logical qubits. Shows intra-qubit fidelity > 1-1e-15, inter-qubit leakage scaling as O(theta^2) with exponent 2.00, and encoded SWAP verification. | Resolved |
| No actual DRL results | New Section 3.1: Learning curves for REINFORCE on Bell state (converges >0.99 fidelity) and 4-qubit GHZ (shows expected difficulty). Confirms environment produces meaningful gradients. | Resolved |
| CPU benchmark baseline unclear | Explicitly states: single-threaded NumPy on AMD EPYC 7763 (one core, no BLAS parallelism). Conservative, reproducible baseline. | Resolved |
| Limited comparison with other frameworks | Expanded Table 6: adds PennyLane and clarifies "Manual" vs. native support. Fairer, more informative comparison. | Resolved |
| Leakage tracking is only diagnostic | Still diagnostic, but now quantified with scaling law and regime boundary (theta < 17 degrees gives leakage <2%). Paper honestly states leakage is not dynamically simulated. | Acceptable |
| Missing complexity analysis | New Table 5: clear O(2^n) scaling for memory and gates, plus concrete 25-qubit estimates. | Resolved |
| MPS engine novelty limited | Acknowledged explicitly: "lightweight, purpose-built... not intended to compete with ITensor or quimb". Honest positioning. | Resolved |

## Remaining Minor Issues

### 1. Page numbering and formatting glitches
Pages 7 and 14 contain long runs of "1 1 1 ..." and line numbers that appear to be formatting artifacts. Clean up before final submission.

### 2. Reference [8]
The citation (Schaal et al., Nature Electronics 2019) still lacks volume/page numbers. Please verify.

### 3. Single-GPU focus
Paper does not discuss multi-GPU or distributed simulation. Not a flaw for SoftwareX, but a brief sentence on future work would be nice.
**NOTE:** This IS already discussed in Section 5 (Limitations). DeepSeek may have missed it.

### 4. Author affiliation
Email @stu.kau.edu.sa suggests student status, but affiliation is "Faculty". Clarify (e.g., "PhD candidate, Faculty of...").

### 5. Minor typos
- Page 1: "physically moti" (cut off) - should be "physically motivated"
  **NOTE:** This is just a page break, not a typo. The word continues on page 2.
- Page 8: Code snippet uses PPO but text says REINFORCE - consistent? DeepSeek notes no contradiction.
