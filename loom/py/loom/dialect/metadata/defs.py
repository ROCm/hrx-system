# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Metadata dialect operation definitions."""

from loom.assembly import EQUALS, Attr
from loom.dsl import (
    MODULE_SCOPE,
    AttrDef,
    Dialect,
    KeyedModuleRecord,
    Op,
    OpPhase,
)

# ============================================================================
# Op group
# ============================================================================

metadata_ops = Dialect(
    "metadata",
    dialect_id=0x21,
    doc="Stable keyed program metadata records.",
    default_phase=OpPhase.MODULE_METADATA,
)

# ============================================================================
# metadata.module - module metadata
# ============================================================================

metadata_module = Op(
    "metadata.module",
    group=metadata_ops,
    doc=(
        "Attach one stable typed metadata value to the module. Keys are exact "
        "strings and unique within the module metadata scope. Values are "
        "ordinary Loom attributes so targets and tooling can project the "
        "subset their output contracts support."
    ),
    traits=[MODULE_SCOPE, KeyedModuleRecord("key")],
    attrs=[
        AttrDef("key", "string", doc="Exact public metadata key."),
        AttrDef("value", "any", doc="Stable typed metadata value."),
    ],
    format=[Attr("key"), EQUALS, Attr("value")],
    examples=[
        'metadata.module "model.revision" = u64(3)',
        'metadata.module "help.summary" = "Example model"',
    ],
)

ALL_METADATA_OPS: tuple[Op, ...] = (metadata_module,)
