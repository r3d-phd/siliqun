# SiliQun V2 B1 Cross-Simulator Sanity Export Authority V1

`AUTHORIZE_SILIQUN_V2_B1_FIXED_IDEAL_EXPORT_ONLY`

This authority permits one local run of the frozen two-circuit B1 suite on the committed SiliQun V2 standalone baseline. It permits construction of the two declared native-gate circuits, translation through `TranslationRequest` and `TranslationReceipt`, ideal state-vector evaluation, canonical probability-digest creation, aggregate receipt writing, and local tests.

This authority does **not** permit a named-device profile, hardware access, provider API, metadata access, circuit submission, pulse export, calibration, learned-policy training, PPO use, SiMORA import or invocation, detector generation, decoder binding, frame action, algorithm ingress, noise-model validation, QEC evaluation, or any physical-performance claim. The state vector and probability vector are transient process-local values and must not be retained in the export receipt.
