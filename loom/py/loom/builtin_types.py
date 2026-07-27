# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in loom type declarations.

These TypeDefs define the textual format for the core loom types.
Scalar types (f32, i32, index) and the first-class ``encoding`` SSA
type are keywords, not TypeDefs. Core named types such as tile<...>,
tensor<...>, vector<...>, view<...>, bare buffer, pool<...>, and
group<...>, plus dotted dialect types such as hal.buffer and vm.ref<...>,
are TypeDefs.

Dialect-specific types are declared in their respective dialect files
(e.g., dialect/hal/, dialect/vm/) using the same TypeDef pattern.
"""

from loom.assembly import (
    COMMA,
    Attr,
    EncodingOf,
    OptionalGroup,
    ScalarOf,
    ShapeOf,
    kw,
)
from loom.dsl import (
    ANY,
    EncodingParam,
    ScalarParam,
    ShapeParam,
    TypeDef,
    TypeParam,
)

__all__ = [
    "ALL_BUILTIN_TYPES",
    # Shaped types.
    "tile_type",
    "tensor_type",
    "vector_type",
    "view_type",
    # Buffer type.
    "buffer_type",
    # Pool type.
    "pool_type",
    # Group type.
    "group_type",
    # Storage type.
    "storage_type",
]

# ============================================================================
# tile<...> — tile-level aggregate value
# ============================================================================

tile_type = TypeDef(
    name="tile",
    doc=(
        "Tile-level aggregate SSA value with static/dynamic dims and optional encoding."
    ),
    ir_kind="tile",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
        EncodingParam("encoding"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
        OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
    ],
)

# ============================================================================
# tensor<...> — logical tensor value
# ============================================================================

tensor_type = TypeDef(
    name="tensor",
    doc=("Logical tensor SSA value with static/dynamic dims and optional encoding."),
    ir_kind="tensor",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
        EncodingParam("encoding"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
        OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
    ],
)

# ============================================================================
# vector<...> — register lane grid
# ============================================================================

vector_type = TypeDef(
    name="vector",
    doc="Register lane-grid SSA value with static/dynamic lane dimensions.",
    ir_kind="vector",
    fact_domain="loom_vector_fact_domain",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
    ],
)

# ============================================================================
# view<...> — typed addressable logical coordinate space
# ============================================================================

view_type = TypeDef(
    name="view",
    doc=(
        "Typed non-owning logical coordinate space over buffer storage with "
        "optional address layout."
    ),
    ir_kind="view",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
        EncodingParam("layout"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
        OptionalGroup([COMMA, EncodingOf("layout")], anchor="layout"),
    ],
)

# ============================================================================
# buffer — opaque storage identity
# ============================================================================

buffer_type = TypeDef(
    name="buffer",
    doc="Opaque untyped storage identity used as the root for typed views.",
    ir_kind="buffer",
)

# ============================================================================
# pool<...> — block-managed device memory pool
# ============================================================================

pool_type = TypeDef(
    name="pool",
    doc="Block-managed device memory pool with a single block size dimension.",
    ir_kind="pool",
    params=[
        ShapeParam("block_size"),
    ],
    format=[
        ShapeOf("block_size"),
    ],
)

# ============================================================================
# group<scope> — barrier scoping
# ============================================================================

group_type = TypeDef(
    name="group",
    doc="Barrier scoping type.",
    ir_kind="group",
    params=[TypeParam("scope", ANY)],
    format=[Attr("scope")],
)

# ============================================================================
# low.storage<space> — function-local byte storage
# ============================================================================

storage_type = TypeDef(
    name="low.storage",
    doc="Function-local byte storage handle.",
    ir_kind="storage",
    params=[TypeParam("space", ANY)],
    format=[Attr("space")],
    fact_domain="loom_storage_fact_domain",
)

# ============================================================================
# Registry
# ============================================================================

ALL_BUILTIN_TYPES: tuple[TypeDef, ...] = (
    tile_type,
    tensor_type,
    vector_type,
    view_type,
    buffer_type,
    pool_type,
    group_type,
    storage_type,
)
