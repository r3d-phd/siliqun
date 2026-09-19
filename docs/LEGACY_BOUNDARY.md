# Baseline Source and Claim Boundary

This branch is an orphan software baseline anchored to the source revision recorded in [`BASELINE_MANIFEST.json`](../BASELINE_MANIFEST.json). Its purpose is to make the standalone V2 package inspectable without carrying forward unrelated historical source trees, experiment logs, training artifacts, or application wrappers.

Only the MIT license is retained directly from that recorded revision. The Python package, documentation, manifest, and tests in this branch are newly authored for the baseline scope. The branch is therefore a **software-boundary change**. It is not a claim that previous numerical outputs, device profiles, or experimental records have been reproduced, validated, or superseded.

The baseline must remain free of training environments, controller integrations, checkpoint formats, hardware-execution paths, and virtual-QEC bridge code. Any future addition must declare its source provenance, interface inputs and outputs, retained data, validation plan, and claim ceiling before it is merged.

The branch supports only the claim that a minimal standalone V2 core has been specified and tested. It does not establish named-device calibration, physical noise fidelity, hardware control, or QEC performance.
