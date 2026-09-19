# SiliQun V2 DRL Preparation Authority V1

`AUTHORIZE_SILIQUN_V2_DRL_DESIGN_AND_STATIC_VALIDATION_ONLY`

This authority permits only the following work on the `v2-standalone-baseline` branch:

1. Create and validate the V2 profile–circuit translation contract.
2. Create and validate a design-only safety-constrained PPO calibration-plugin protocol.
3. Implement static contract validators and unit tests that do not train a policy or contact an external system.
4. Publish source, tests, manifests, and static validation receipts.
5. Obtain independent scope review.

This authority does **not** permit PPO training, policy checkpoint creation, TE-PWS estimation, device discovery, authenticated metadata access, provider API access, hardware job submission, waveform or pulse export, calibration change, SiMORA policy modification, QEC experiment, decoder fitting, or any physical-performance claim.

Any transition from D0 static preparation to D1 device eligibility requires a new authority and a named-device evidence package. Any active pilot requires the independent silicon adapter gates and a separate safety and latency authority.
