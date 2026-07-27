# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Encoding dialect op definitions."""

from loom.assembly import (
    COLON,
    COMMA,
    Attr,
    AttrDict,
    IndexList,
    OperandDict,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ANY,
    ANY_ENCODING,
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENCODING,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_STRING,
    ENCODING_LAYOUT,
    I1,
    INDEX,
    PURE,
    AttrDef,
    Dialect,
    Op,
    Operand,
    OpPhase,
    Result,
    SameType,
)

# ============================================================================
# Group
# ============================================================================

encoding_ops = Dialect("encoding", dialect_id=0x09, doc="Encoding definition and query ops.")

# ============================================================================
# encoding.layout.dense — dense logical-to-physical address layout
# ============================================================================

encoding_layout_dense = Op(
    name="encoding.layout.dense",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Construct a dense row-major address layout. The consuming view type provides the rank and logical extents."),
    results=[Result("result", ENCODING_LAYOUT, doc="Dense address-layout value.")],
    traits=[PURE],
    facts="loom_encoding_layout_dense_facts",
    format=[COLON, ResultType("result")],
    examples=[
        "%layout = encoding.layout.dense : encoding<layout>",
    ],
)

# ============================================================================
# encoding.layout.strided — explicit element-stride address layout
# ============================================================================

encoding_layout_strided = Op(
    name="encoding.layout.strided",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Construct an address layout from per-dimension element strides. Static and dynamic stride values are interleaved in one bracket list."),
    operands=[Operand("strides", INDEX, doc="Dynamic element strides.", variadic=True)],
    results=[Result("result", ENCODING_LAYOUT, doc="Strided address-layout value.")],
    attrs=[
        AttrDef(
            "static_strides",
            ATTR_TYPE_I64_ARRAY,
            doc="Static element strides with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[PURE],
    verify="loom_encoding_layout_strided_verify",
    facts="loom_encoding_layout_strided_facts",
    format=[
        IndexList("strides", "static_strides"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%layout = encoding.layout.strided [%row_stride, 1] : encoding<layout>",
        "%layout = encoding.layout.strided [4096, 1] : encoding<layout>",
    ],
)

# ============================================================================
# encoding.layout.assume.* — local layout fact refinement
# ============================================================================

encoding_layout_assume_dense = Op(
    name="encoding.layout.assume.dense",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Refine an existing address-layout encoding value with the fact that it is dense row-major. The result is the same encoding value in SSA form with stronger local facts."),
    operands=[Operand("layout", ENCODING_LAYOUT, doc="Address-layout value to refine.")],
    results=[Result("result", ENCODING_LAYOUT, doc="Layout value with dense-layout facts.")],
    constraints=[SameType("layout", "result")],
    traits=[PURE],
    facts="loom_encoding_layout_assume_dense_facts",
    format=[
        Ref("layout"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%dense = encoding.layout.assume.dense %layout : encoding<layout>",
    ],
)

encoding_layout_assume_strided = Op(
    name="encoding.layout.assume.strided",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Refine an existing address-layout encoding value with the fact that "
        "it is strided and has the given rank. Per-axis stride values remain "
        "unknown unless a concrete encoding.layout.strided value is available."
    ),
    operands=[Operand("layout", ENCODING_LAYOUT, doc="Address-layout value to refine.")],
    results=[Result("result", ENCODING_LAYOUT, doc="Layout value with strided-layout facts.")],
    attrs=[AttrDef("rank", ATTR_TYPE_I64, doc="Required strided layout rank.")],
    constraints=[SameType("layout", "result")],
    traits=[PURE],
    verify="loom_encoding_layout_assume_strided_verify",
    facts="loom_encoding_layout_assume_strided_facts",
    format=[
        Ref("layout"),
        AttrDict(),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%strided = encoding.layout.assume.strided %layout {rank = 2} : encoding<layout>",
    ],
)

# ============================================================================
# encoding.define — create an encoding value from a static specification
# ============================================================================

encoding_define = Op(
    name="encoding.define",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Create an encoding value from a static encoding specification.",
    operands=[
        Operand("params", ANY, variadic=True),
    ],
    results=[Result("result", ANY_ENCODING)],
    attrs=[
        AttrDef("spec", ATTR_TYPE_ENCODING, doc="Static encoding specification."),
        AttrDef(
            "param_names",
            ATTR_TYPE_DICT,
            optional=True,
            doc="Sorted dynamic parameter names mapped to operand ordinals.",
        ),
    ],
    traits=[PURE],
    verify="loom_encoding_define_verify",
    facts="loom_encoding_define_facts",
    format=[
        Attr("spec"),
        OperandDict("params", "param_names"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%enc = encoding.define #q8_0<block=32> : encoding<schema>",
        "%enc = encoding.define #q8_0<block=32> {group_size = %group_size : index} : encoding<schema>",
    ],
)

# ============================================================================
# encoding.assume.spec — local exact static encoding refinement
# ============================================================================

encoding_assume_spec = Op(
    name="encoding.assume.spec",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Refine an existing encoding value with an exact static encoding "
        "specification. Dynamic values remain ordinary SSA operands elsewhere; "
        "this op only states the selected static family and static parameters."
    ),
    operands=[Operand("enc", ANY_ENCODING, doc="Encoding value to refine.")],
    results=[Result("result", ANY_ENCODING, doc="Encoding value with exact static-spec facts.")],
    attrs=[AttrDef("spec", ATTR_TYPE_ENCODING, doc="Exact static encoding specification.")],
    constraints=[SameType("enc", "result")],
    traits=[PURE],
    verify="loom_encoding_assume_spec_verify",
    facts="loom_encoding_assume_spec_facts",
    format=[
        Ref("enc"),
        COMMA,
        Attr("spec"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%schema2 = encoding.assume.spec %schema, #ggml_q4_0<block_elems=32, storage_bytes=18> : encoding<schema>",
    ],
)

# ============================================================================
# encoding.isa — test if an encoding belongs to a category
# ============================================================================

encoding_isa = Op(
    name="encoding.isa",
    group=encoding_ops,
    doc="Test if an encoding belongs to a category.",
    operands=[Operand("enc", ANY_ENCODING)],
    results=[Result("result", I1)],
    attrs=[AttrDef("category", ATTR_TYPE_STRING)],
    traits=[PURE],
    format=[Ref("enc"), COMMA, Attr("category"), COLON, TypeOf("result")],
    examples=[
        '%is_quantized = encoding.isa %enc, "quantized" : i1',
    ],
)

# ============================================================================
# Registry: all encoding ops in declaration order
# ============================================================================

ALL_ENCODING_OPS: tuple[Op, ...] = (
    encoding_layout_dense,
    encoding_layout_strided,
    encoding_define,
    encoding_isa,
    encoding_layout_assume_dense,
    encoding_layout_assume_strided,
    encoding_assume_spec,
)
