"""Tile dialect: register-resident shaped values."""

from loom.builtin_types import tile_type
from loom.dialect.tile.defs import (
    ALL_TILE_OPS,
    ALL_TILE_TYPES,
    tile_duplicate,
    tile_ops,
)

__all__ = [
    "tile_ops",
    "tile_type",
    "tile_duplicate",
    "ALL_TILE_OPS",
    "ALL_TILE_TYPES",
]
