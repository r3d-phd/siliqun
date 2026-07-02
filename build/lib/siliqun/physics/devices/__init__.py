"""
Device profiles for silicon spin qubit architectures.
"""

from .profiles import (
    DeviceProfile, get_device_profile,
    donor_device, simos_device, gaa_device,
    DEVICE_REGISTRY,
)
