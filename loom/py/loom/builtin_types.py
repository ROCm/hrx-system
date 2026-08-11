# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in loom type declarations.

These TypeDefs define the textual format for the core loom types.
Scalar types (f32, i32, index) are keywords, not TypeDefs. Core named types
such as encoding, tile<...>, tensor<...>, vector<...>, view<...>, bare buffer,
and pool<...>, plus dotted dialect types such as hal.buffer and test.ref<...>,
are TypeDefs.

Dialect-specific types are declared in their respective dialect files
(e.g., dialect/hal/ and dialect/kernel/) using the same TypeDef pattern.
"""

from loom.assembly import (
    COMMA,
    EncodingOf,
    OptionalGroup,
    Param,
    ScalarOf,
    ShapeOf,
    kw,
)
from loom.dsl import (
    ATTR_TYPE_ENUM,
    AttrDef,
    EncodingParam,
    EnumCase,
    EnumDef,
    ScalarParam,
    ShapeParam,
    TypeDef,
)
from loom.ir import EncodingRole, EncodingType, StorageSpace, StorageType, TypeKind

__all__ = [
    "ALL_BUILTIN_TYPES",
    # Encoding type.
    "encoding_type",
    # Shaped types.
    "tile_type",
    "tensor_type",
    "vector_type",
    "view_type",
    # Buffer type.
    "buffer_type",
    # Pool type.
    "pool_type",
    # Storage type.
    "storage_type",
]

# ============================================================================
# encoding<role> - encoding SSA value role
# ============================================================================

EncodingRoleAttr = EnumDef(
    "EncodingRole",
    [
        EnumCase("layout", EncodingRole.LAYOUT.value),
        EnumCase("schema", EncodingRole.SCHEMA.value),
        EnumCase("storage", EncodingRole.STORAGE.value),
        EnumCase("transform", EncodingRole.TRANSFORM.value),
    ],
    doc="Semantic role carried by an encoding SSA value type.",
    c_type="loom_encoding_role_t",
    c_const_prefix="LOOM_ENCODING_ROLE",
    c_include="loom/ir/types.h",
)

encoding_type = TypeDef(
    name="encoding",
    doc="Encoding SSA value with an optional semantic role.",
    ir_kind="encoding",
    python_type=EncodingType,
    params=[
        AttrDef(
            "role",
            ATTR_TYPE_ENUM,
            enum_def=EncodingRoleAttr,
            optional=True,
        )
    ],
    format=[OptionalGroup([Param("role")], anchor="role")],
)

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
    fact_domain="loom_buffer_fact_domain",
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
# low.storage<space> — function-local byte storage
# ============================================================================

StorageSpaceAttr = EnumDef(
    "StorageSpace",
    [
        EnumCase("stack", StorageSpace.STACK.value),
        EnumCase("scratch", StorageSpace.SCRATCH.value),
        EnumCase("private", StorageSpace.PRIVATE.value),
        EnumCase("workgroup", StorageSpace.WORKGROUP.value),
    ],
    doc="Function-local byte storage space.",
    c_type="loom_storage_space_t",
    c_const_prefix="LOOM_STORAGE_SPACE",
    c_include="loom/ir/types.h",
)

storage_type = TypeDef(
    name="low.storage",
    doc="Function-local byte storage handle.",
    ir_kind="storage",
    python_type=StorageType,
    params=[AttrDef("space", ATTR_TYPE_ENUM, enum_def=StorageSpaceAttr)],
    format=[Param("space")],
    fact_domain="loom_storage_fact_domain",
)

# ============================================================================
# Registry
# ============================================================================

ALL_BUILTIN_TYPES: tuple[TypeDef, ...] = (
    encoding_type,
    tile_type,
    tensor_type,
    vector_type,
    view_type,
    buffer_type,
    pool_type,
    storage_type,
)

# Direct Python representation class to declaration lookup for compact
# descriptor-backed built-ins. Derived from the declarations so text formatting
# never needs a parallel family switch or spelling table.
BUILTIN_TYPE_BY_PYTHON_TYPE: dict[type[object], TypeDef] = {
    type_def.python_type: type_def
    for type_def in ALL_BUILTIN_TYPES
    if type_def.python_type is not None
}

# Direct compact shape representation kind to declaration lookup. Derived from
# the declarations so parsers and printers do not maintain parallel family or
# spelling tables.
BUILTIN_COMPACT_SHAPE_TYPE_BY_KIND: dict[TypeKind, TypeDef] = {
    TypeKind[type_def.ir_kind.upper()]: type_def
    for type_def in ALL_BUILTIN_TYPES
    if type_def.uses_compact_shape_format
}
