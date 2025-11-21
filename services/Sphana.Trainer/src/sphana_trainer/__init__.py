"""Top-level package for the Sphana trainer service."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sphana_trainer")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]


