"""AI clip providers. See base.py for the interface."""

from .base import Provider
from .manual import ManualProvider
from .placeholder import PlaceholderProvider

__all__ = ["Provider", "ManualProvider", "PlaceholderProvider"]
