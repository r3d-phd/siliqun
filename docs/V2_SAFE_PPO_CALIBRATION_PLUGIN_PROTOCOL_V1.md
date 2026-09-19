# SiliQun V2 Safety-Constrained PPO Calibration Plugin Protocol V1

**Status:** Design and static-validation protocol only. No policy is trained, no checkpoint is created, no provider is contacted, and no hardware command is issued under this protocol.

## Research question

> After a named silicon device has passed the requisite identity, calibration, and offline-data gates, can one masked PPO policy select safe, pre-approved calibration-support actions more effectively than fixed nominal, independent Bayesian, and physics-informed MPC or estimator baselines under the same probe, time, and intervention budgets?

This is a future calibration-support question. It is **not** a SiMORA decoding, syndrome-frame, QEC, or algorithm-execution question. SiMORA remains a deterministic supervisor whose initial silicon action set is `HOLD_FRAME`, `COMMIT_FRAME`, and `ABORT`.

In this protocol, **SiMORA** denotes the separate deterministic syndrome-frame supervisor. It validates a complete detector frame and then holds, commits a frozen decoder-derived software frame, or aborts. The PPO preparation package has no path to alter that policy.

## Scope lock

The protocol uses **PPO as the sole DRL algorithm**, restricts all studied models to at most **five qubits**, and reserves **TE-PWS** as the only advanced information feature. TE-PWS is not implemented or measured here. The protocol does not authorize SAC, DQN, multi-agent learning, autonomous characterization, full gate-set tomography, or any controller modification.

The previous local PPO and SAC results do not license a hardware policy. They motivate a stricter new formulation: a partial-observation, multi-step calibration-support task with hard safety constraints, fixed non-learning comparators, time-split held-out evaluation, and hardware shadow mode before active use.

## Information firewall

A future plugin may receive only a signed `CalibrationObservation` derived from a frozen device and calibration manifest. Its allowed fields are: device and manifest hashes; profile identifier; a quantized calibration-age bucket; quantized and predeclared readout, interaction, coherence, and drift indicators; remaining probe/time/intervention budgets; action-validity mask; and decision index. It must not receive raw waveforms, unbounded control values, final logical labels, decoded success labels, postselected outcomes, comparator actions, held-out target outcomes, or an unblinded independent-audit result.

A named device must satisfy the silicon adapter’s identity, patch, detector, calibration, decoder, and latency gates before it may supply any data to shadow mode. The plugin has no direct provider/API handle.

## Safe action contract

The policy’s output is a `CalibrationRecommendation`, not a hardware command. It may select exactly one action from the following action-template registry:

| Action template | Meaning | Adapter condition |
|---|---|---|
| `select_diagnostic_template` | Request one approved diagnostic template identifier. | Template is allow-listed and within remaining budget. |
| `select_approved_pulse_template` | Recommend one already-approved pulse-template identifier. | Adapter independently validates the template and keeps hardware command mode disabled until a separate active-pilot authority exists. |
| `request_recalibration_review` | Request human or deterministic recalibration review. | No autonomous calibration change occurs. |
| `abstain` | Return no recommendation. | Adapter preserves the current safe state. |

The policy cannot request arbitrary voltages, microwave waveforms, reset parameters, calibration modifications, provider jobs, decoder changes, SiMORA frame decisions, or physical corrections. The adapter owns the action mask, rejects malformed outputs, and records every refusal. Any illegal probability mass, invalid action, stale manifest, or budget breach is terminal and fails closed.

## PPO convergence and scaling evidence framework

The future training package must implement the required convergence framework from its first run. It will retain aggregate policy loss, value loss, entropy, approximate KL, clipping fraction, explained variance, gradient norm, episodic utility, invalid-action rate, mask-leakage rate, and budget consumption. It must use masked logits, a frozen old-policy reference for the PPO clipped objective, bounded rollout storage, advantage normalization, value clipping where specified, and global gradient clipping. A conventional replay buffer or neural target network is not introduced because that would change the declared on-policy PPO method; the retained rollout and frozen old-policy reference serve the corresponding traceability functions for this protocol.

The companion scaling ledger records worker count, generated samples, update count, sample age, training throughput, and stale-rollout count from the start. The initial offline implementation is single-worker with zero permitted staleness. A later multi-worker scale-out must preserve the same evidence fields and predeclare its staleness limit. This is an accounting and diagnostics requirement, not a claim that local training reproduces a distributed scaling framework.

## Required comparators and endpoints

Every future execution must compare the PPO policy against a fixed nominal strategy, an independent Bayesian strategy, and a physics-informed MPC or estimator when the latter is technically applicable. Each comparator must have exactly the same observation availability, action templates, safety shield, probe budget, time budget, and held-out calibration records as PPO.

The primary endpoint must be a predeclared held-out calibration-support utility that combines recovery quality, probe cost, elapsed time, and safety interventions. Secondary endpoints include action-refusal rate, safe-abstention rate, uncertainty, latency, and temporal robustness after a different calibration snapshot. Training reward, simulated state fidelity, or a development trajectory may not substitute for the held-out primary endpoint.

## Sequential gates

| Gate | Required evidence | Current status |
|---|---|---|
| D0 — static protocol | Versioned action templates, observation schema, masks, comparators, convergence ledger, and source hashes. | This package prepares D0 only. |
| D1 — device eligibility | Named device identity, topology, calibration freshness, and safe adapter schema. | Not started. |
| D2 — offline data eligibility | Partitioned historical calibration data with a future time-split holdout. | Not started. |
| D3 — offline PPO evaluation | Fixed training partition, all comparators, diagnostics, safety gates, and held-out evaluation. | Not authorized. |
| D4 — shadow recommendation | Blinded or live completed observations; no device action from PPO. | Not authorized. |
| D5 — restricted active pilot | Separate authority, immutable allow-list, adapter shield, and controlled endpoint. | Not authorized. |
| D6 — temporal replication | New calibration snapshot with no policy or decoder tuning. | Not authorized. |

Failure at any gate is terminal for that authority. It must not be repaired by tuning after a held-out result; a materially new protocol and authority are required.

## TE-PWS boundary

TE-PWS is retained as the single advanced feature for a later journal-stage analysis. It may be added only when a separately reviewed stochastic-path estimator, likelihood definition, sampling plan, and held-out interpretation are available. It is not a PPO objective and this protocol makes no transfer-entropy, causal-discovery, or information-flow claim.

## Claim ceiling

The present package supports only the statement that **SiliQun V2 has a static, safety-constrained PPO calibration-plugin design with a defined action boundary, comparators, convergence ledger, scaling ledger, and staged shadow-to-active path**. It does not support a PPO effectiveness claim, SiMOS calibration claim, hardware-control claim, QEC claim, SiMORA runtime claim, or physical-performance claim.

## References

[1]: https://arxiv.org/abs/1707.06347 "Proximal Policy Optimization Algorithms"
[2]: https://www.annualreviews.org/content/journals/10.1146/annurev-control-042920-020211 "Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning"
[3]: https://docs.cleanrl.dev/rl-algorithms/ppo/ "CleanRL PPO documentation"
