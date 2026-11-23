"""FastAPI service layer for the SQL Injection Detector.

The training scripts in ``src/`` remain the source of truth for model
creation. This package only loads a saved artifact and exposes it over HTTP.
"""

from api.config import Settings, get_settings
from api.main import create_app

__all__ = ["Settings", "create_app", "get_settings"]
__version__ = "1.0.0"
