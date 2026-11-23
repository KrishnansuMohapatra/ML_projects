"""HTTP routers, mounted by :func:`api.main.create_app`."""

from api.routers import files, health, predict

__all__ = ["files", "health", "predict"]
