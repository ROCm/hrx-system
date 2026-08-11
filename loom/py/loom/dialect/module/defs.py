# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Module composition operation definitions."""

from loom.assembly import Attr
from loom.dsl import (
    ATTR_TYPE_STRING,
    ATTR_TYPE_SYMBOL_SET,
    MODULE_SCOPE,
    AttrDef,
    Dialect,
    Op,
    SymbolReference,
    SymbolReferenceRole,
)

# ============================================================================
# Op group
# ============================================================================

module_ops = Dialect(
    "module",
    dialect_id=0x1F,
    doc="Compile-time source composition metadata.",
)

# ============================================================================
# module.import - source provider availability
# ============================================================================

module_import = Op(
    "module.import",
    group=module_ops,
    doc=(
        "Name a source provider that may define the listed module symbols. "
        "The provider is an opaque resolver-defined key, and the symbol list "
        "records availability without creating dependency or liveness edges. "
        "Provider identity and import order never participate in template or "
        "symbol selection. This metadata is consumed only by compile-time "
        "linking."
    ),
    traits=[MODULE_SCOPE],
    attrs=[
        AttrDef(
            "provider",
            ATTR_TYPE_STRING,
            doc="Opaque resolver-defined source provider key.",
        ),
        AttrDef(
            "symbols",
            ATTR_TYPE_SYMBOL_SET,
            symbol_ref=SymbolReference(
                "symbol",
                [],
                role=SymbolReferenceRole.AVAILABILITY,
            ),
            doc="Canonical symbol anchors that the provider may define.",
        ),
    ],
    format=[
        Attr("provider"),
        Attr("symbols"),
    ],
    examples=[
        'module.import "motif/format/ggml.loom" [@decode_q4, @decode_q6]',
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_MODULE_OPS: tuple[Op, ...] = (module_import,)
