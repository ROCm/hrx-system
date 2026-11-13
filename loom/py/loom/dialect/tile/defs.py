# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tile operation definitions."""

from loom.dsl import TILE, Dialect, Op, TypeDef, value_duplication_op, value_refinement_op

tile_ops = Dialect(
    "tile",
    dialect_id=0x03,
    doc="Register-resident shaped values and their structural operations.",
)

tile_refine = value_refinement_op(
    "tile.refine",
    group=tile_ops,
    carrier=TILE,
    doc=(
        "Assert more precise shape and encoding information for the same tile elements. Source and result describe the same value and ownership identity; this operation performs no conversion or copy."
    ),
    examples=["%refined = tile.refine %source : tile<[%count]xf32> -> tile<16xf32>"],
)

tile_duplicate = value_duplication_op(
    "tile.duplicate",
    group=tile_ops,
    carrier=TILE,
    doc=("Create another local owner of a tile and its dependent dimensions or encoding. The duplicate carries the same elements through a distinct SSA ownership identity."),
    examples=["%copy_m, %copy_layout, %copy = tile.duplicate %m, %layout, %source : index, encoding<layout>, tile<[%copy_m]xf32, %copy_layout>"],
)

ALL_TILE_TYPES: tuple[TypeDef, ...] = ()
ALL_TILE_OPS: tuple[Op, ...] = (tile_refine, tile_duplicate)
