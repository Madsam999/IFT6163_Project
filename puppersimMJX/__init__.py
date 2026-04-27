"""MJX/Brax training package for Pupper v2."""

from pathlib import Path


def get_assets_path() -> Path:
    return Path(__file__).resolve().parent / "assets"
