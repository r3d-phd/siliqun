"""
SiliQun device profiles module.
Provides DEVICE_PROFILES dict for hardware-specific simulation.
"""
from siliqun.physics.devices.profiles import DEVICE_REGISTRY as DEVICE_PROFILES
from siliqun.physics.devices.profiles import get_device_profile, DeviceProfile

__all__ = ["DEVICE_PROFILES", "get_device_profile", "DeviceProfile"]
