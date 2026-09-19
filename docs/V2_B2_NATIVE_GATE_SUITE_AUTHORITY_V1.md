# SiliQun V2 B2 Native-Gate Suite Export Authority V1

`AUTHORIZE_SILIQUN_V2_B2_FIXED_IDEAL_EXPORT_ONLY`

This authority permits one local execution of the prespecified five-circuit B2 native-gate suite on the committed SiliQun V2 standalone baseline lineage. It permits construction of the declared `rx`, `ry`, `rz`, and `cz` circuits; translation through `TranslationRequest` and `TranslationReceipt`; ideal state-vector evaluation; canonical probability-digest creation; digest-only receipt writing; and local source/readiness audits and tests.

The authority does **not** permit a named-device profile, hardware access, provider API, metadata access, circuit submission, pulse export, calibration, learned-policy training, PPO use, SiMORA import or invocation, detector generation, decoder binding, frame action, algorithm ingress, noise-model validation, QEC evaluation, or a physical-performance claim. State vectors and probability vectors are transient process-local values and must not be retained in the export receipt.
