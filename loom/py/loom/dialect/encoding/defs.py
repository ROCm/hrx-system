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
    ATTR_TYPE_BOOL,
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
    EncodingFamilyDef,
    EncodingFamilyRole,
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
# Static encoding families
# ============================================================================

_BLOCK_STORAGE_PARAMETERS = (
    AttrDef("block_elems", ATTR_TYPE_I64, optional=True),
    AttrDef("storage_bytes", ATTR_TYPE_I64, optional=True),
)

_NAMED_FP8_PARAMETERS = (
    AttrDef(
        "rounding",
        ATTR_TYPE_STRING,
        optional=True,
        bare_identifier=True,
    ),
    AttrDef("storage_bits", ATTR_TYPE_I64, optional=True),
)

_MATRIX_OPERAND_PARAMETERS = (
    AttrDef("affine", ATTR_TYPE_STRING, optional=True, bare_identifier=True),
    AttrDef("codebook", ATTR_TYPE_STRING, optional=True, bare_identifier=True),
    AttrDef("element_format", ATTR_TYPE_STRING, bare_identifier=True),
    AttrDef("payload_elements", ATTR_TYPE_I64),
    AttrDef(
        "payload_packing",
        ATTR_TYPE_STRING,
        optional=True,
        bare_identifier=True,
    ),
    AttrDef("payload_registers", ATTR_TYPE_I64),
    AttrDef("rounding", ATTR_TYPE_STRING, optional=True, bare_identifier=True),
    AttrDef("scale_format", ATTR_TYPE_STRING, optional=True, bare_identifier=True),
    AttrDef("scale_group_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("scale_group_shape", ATTR_TYPE_I64_ARRAY, optional=True),
    AttrDef("scale_operands", ATTR_TYPE_I64, optional=True),
    AttrDef(
        "scale_topology",
        ATTR_TYPE_STRING,
        optional=True,
        bare_identifier=True,
    ),
    AttrDef(
        "secondary_scale_format",
        ATTR_TYPE_STRING,
        optional=True,
        bare_identifier=True,
    ),
    AttrDef("sparsity", ATTR_TYPE_STRING, optional=True, bare_identifier=True),
    AttrDef("sparsity_group_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("sparsity_group_nonzero_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("zero_scale_fallback", ATTR_TYPE_BOOL, optional=True),
)

_NUMERIC_TRANSFORM_PARAMETERS = (
    AttrDef("family", ATTR_TYPE_STRING),
    AttrDef("input_elems", ATTR_TYPE_I64, optional=True),
    AttrDef("normalization", ATTR_TYPE_STRING, optional=True),
    AttrDef("output_elems", ATTR_TYPE_I64, optional=True),
)

_TURBOQUANT_KV_PARAMETERS = (
    AttrDef("first_stage_bits", ATTR_TYPE_I64),
    AttrDef("logical_element", ATTR_TYPE_STRING),
    AttrDef("logical_elems", ATTR_TYPE_I64),
    AttrDef("pack_order", ATTR_TYPE_STRING),
    AttrDef("qjl_rows", ATTR_TYPE_I64),
    AttrDef("record_bytes", ATTR_TYPE_I64),
    AttrDef("residual_bits", ATTR_TYPE_I64),
    AttrDef("scalar_quantizer", ATTR_TYPE_STRING),
    AttrDef("transform_family", ATTR_TYPE_STRING),
)

ALL_ENCODING_FAMILIES: tuple[EncodingFamilyDef, ...] = (
    EncodingFamilyDef(
        "physical_storage",
        group=encoding_ops,
        role=EncodingFamilyRole.PHYSICAL_STORAGE,
        parameters=(
            AttrDef("layout", ATTR_TYPE_ENCODING, optional=True),
            AttrDef("schema", ATTR_TYPE_ENCODING, optional=True),
        ),
        doc="Composes an address layout and storage schema.",
    ),
    EncodingFamilyDef(
        "dense",
        group=encoding_ops,
        role=EncodingFamilyRole.ADDRESS_LAYOUT,
        doc="Dense row-major address layout.",
    ),
    EncodingFamilyDef(
        "strided",
        group=encoding_ops,
        role=EncodingFamilyRole.ADDRESS_LAYOUT,
        parameters=(
            AttrDef("stride", ATTR_TYPE_I64, optional=True),
            AttrDef("strides", ATTR_TYPE_I64_ARRAY, optional=True),
        ),
        doc="Explicit element-stride address layout.",
    ),
    EncodingFamilyDef(
        "q8_0",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=(AttrDef("block", ATTR_TYPE_I64, optional=True),),
        doc="Blockwise eight-bit quantized storage schema.",
    ),
    *(
        EncodingFamilyDef(
            name,
            group=encoding_ops,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            parameters=_BLOCK_STORAGE_PARAMETERS,
            doc="GGML-compatible block storage schema.",
        )
        for name in ("ggml_q4_0", "ggml_q8_0")
    ),
    EncodingFamilyDef(
        "q6_k",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        doc="Six-bit K-quant storage schema.",
    ),
    EncodingFamilyDef(
        "ggml_q6_k",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=_BLOCK_STORAGE_PARAMETERS,
        doc="GGML-compatible block storage schema.",
    ),
    EncodingFamilyDef(
        "ggml_iq_grid",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=(
            AttrDef("code_bits", ATTR_TYPE_I64),
            AttrDef("grid_elems", ATTR_TYPE_I64),
        ),
        doc="GGML indexed-grid storage schema.",
    ),
    EncodingFamilyDef(
        "loom_fp4_table",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=(
            AttrDef("code_bits", ATTR_TYPE_I64),
            AttrDef("table_elems", ATTR_TYPE_I64),
        ),
        doc="Table-decoded four-bit storage schema.",
    ),
    *(
        EncodingFamilyDef(
            name,
            group=encoding_ops,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            parameters=_NAMED_FP8_PARAMETERS,
            doc="Named eight-bit floating-point storage schema.",
        )
        for name in (
            "ieee_fp8_e4m3",
            "ieee_fp8_e5m2",
            "fp8_e4m3fn",
            "fp8_e4m3fnuz",
            "fp8_e5m2fnuz",
        )
    ),
    EncodingFamilyDef(
        "matrix_operand",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=_MATRIX_OPERAND_PARAMETERS,
        doc="Target-independent encoded matrix operand schema.",
    ),
    EncodingFamilyDef(
        "numeric_transform",
        group=encoding_ops,
        role=EncodingFamilyRole.NUMERIC_TRANSFORM,
        parameters=_NUMERIC_TRANSFORM_PARAMETERS,
        doc="Numerical transform with static shape and policy parameters.",
    ),
    EncodingFamilyDef(
        "orthogonal_transform",
        group=encoding_ops,
        role=EncodingFamilyRole.NUMERIC_TRANSFORM,
        doc="Orthogonal numerical transform.",
    ),
    EncodingFamilyDef(
        "turboquant_kv",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=_TURBOQUANT_KV_PARAMETERS,
        doc="TurboQuant key/value storage schema.",
    ),
)

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
