"""
siliqun.core
============
Core abstractions for the SiliQun plugin architecture.

The central abstraction is :class:`TechnologyProfile`, an abstract base
class that every hardware-platform plugin must subclass.  The
:mod:`siliqun.core.abc` module defines the contract; this package
re-exports it for convenience.
"""

from .abc import TechnologyProfile, CalibrationRecord, PGIRSValidator

__all__ = ["TechnologyProfile", "CalibrationRecord", "PGIRSValidator"]
