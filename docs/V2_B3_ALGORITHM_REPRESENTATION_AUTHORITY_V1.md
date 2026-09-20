# SiliQun V2 B3 Algorithm-Representation Export Authority V1

`AUTHORIZE_SILIQUN_V2_B3_FIXED_IDEAL_ALGORITHM_REPRESENTATION_EXPORT_ONLY`

This authority permits one local execution of the prespecified three-representation B3 suite on the committed SiliQun V2 standalone baseline lineage. It permits construction of the declared native `rx`, `ry`, `rz`, and `cz` representations, translation through `TranslationRequest` and `TranslationReceipt`, ideal state-vector evaluation, canonical probability-digest creation, digest-only receipt writing, and static/source audits and tests.

This authority does **not** permit treating an ideal circuit representation as algorithm execution or success. It does not permit a named-device profile, hardware access, provider API, metadata access, circuit submission, pulse export, calibration, learned-policy training, PPO use, SiMORA import or invocation, detector generation, decoder binding, frame action, algorithm ingress, noise-model validation, QEC evaluation, factoring output, oracle answer, or a physical-performance claim. State vectors and probability vectors are transient process-local values and must not be retained in the export receipt.
