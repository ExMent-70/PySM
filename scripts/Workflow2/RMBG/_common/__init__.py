"""Shared contracts for the PySM RMBG subsystem."""

from .config_schema import RMBG_SCHEMA_VERSION, RmbgSettings, default_settings

__all__ = ["RMBG_SCHEMA_VERSION", "RmbgSettings", "default_settings"]
