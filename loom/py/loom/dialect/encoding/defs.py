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
    AttrParams,
    IndexList,
    OperandDict,
    Ref,
    ResultType,
    TemplateParam,
    TypeOf,
)
from loom.dialect.encoding.numeric_formats import NUMERIC_FORMAT_KEYWORDS
from loom.dsl import (
    ANY,
    ANY_ENCODING,
    ATTR_TYPE_BOOL,
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENCODING,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_PARAMETERIZED,
    ENCODING_LAYOUT,
    ENCODING_SCHEMA,
    FACT_IDENTITY,
    I1,
    INDEX,
    PURE,
    AttrDef,
    ConditionRefinement,
    ConditionRefinementTruth,
    Dialect,
    EncodingAliasDef,
    EncodingFamilyDef,
    EncodingFamilyRole,
    EncodingOperandSummaryDef,
    EncodingRecordDef,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    ParameterizedAttrDef,
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


def _enum_fact(enum_def: EnumDef, keyword: str) -> int:
    """Returns the one-hot fact bit for one nonzero enum case."""
    value = enum_def.case(keyword).value
    return 0 if value == 0 else 1 << (value - 1)


NumericFormat = EnumDef(
    "NumericFormat",
    [EnumCase(keyword, ordinal) for ordinal, keyword in enumerate(NUMERIC_FORMAT_KEYWORDS)],
    doc="Target-independent encoded numeric format.",
)

AuxiliaryKey = EnumDef(
    "AuxiliaryKey",
    [
        EnumCase("scale", 0),
        EnumCase("secondary_scale", 1),
        EnumCase("scale2", 2),
        EnumCase("scale3", 3),
        EnumCase("scale4", 4),
        EnumCase("scale5", 5),
        EnumCase("scale6", 6),
        EnumCase("scale7", 7),
        EnumCase("zero_point", 8),
        EnumCase("minimum", 9),
        EnumCase("bias", 10),
        EnumCase("sum_correction", 11),
        EnumCase("codebook", 12),
        EnumCase("sparsity", 13),
        EnumCase("metadata", 14),
        EnumCase("indices", 15),
        EnumCase("offsets", 16),
        EnumCase("mask", 17),
        EnumCase("signs", 18),
        EnumCase("residual", 19),
        EnumCase("amax", 20),
        EnumCase("thresholds", 21),
        EnumCase("centroids", 22),
        EnumCase("outliers", 23),
    ],
    doc="Auxiliary SSA operand key shared by encoded storage schemas.",
)

PayloadPacking = EnumDef(
    "PayloadPacking",
    [
        EnumCase(keyword, ordinal)
        for ordinal, keyword in enumerate(
            (
                "dense_lanes",
                "little_endian_nibbles",
                "big_endian_nibbles",
                "bitfield_stream",
                "bitplane_stream",
                "multi_stream",
                "base_n_packed",
                "codebook_indices",
                "target_fragment",
                "interleaved_scale_payload",
                "separate_scale_payload",
            ),
            start=1,
        )
    ],
    doc="Physical payload bit and field layout.",
)

ScaleTopology = EnumDef(
    "ScaleTopology",
    [
        EnumCase(keyword, ordinal)
        for ordinal, keyword in enumerate(
            (
                "none",
                "tensor_global",
                "row",
                "column",
                "channel",
                "group_1d",
                "block_1d",
                "block_2d",
                "subblock_in_superblock",
                "hierarchical",
                "per_token",
                "per_head",
                "per_page",
                "runtime_amax_derived",
            )
        )
    ],
    doc="Scale sharing topology for encoded values.",
)

AffinePolicy = EnumDef(
    "AffinePolicy",
    [
        EnumCase(keyword, ordinal)
        for ordinal, keyword in enumerate(
            (
                "none",
                "scale_only",
                "scale_plus_min",
                "scale_plus_zero_point",
                "scale_plus_bias",
                "super_scale_times_subscale",
                "sum_correction",
            )
        )
    ],
    doc="Affine transform applied to encoded values.",
)

RoundingPolicy = EnumDef(
    "RoundingPolicy",
    [
        EnumCase("none", 0),
        EnumCase("nearest_even", 1),
        EnumCase("nearest_away", 2),
        EnumCase("toward_zero", 3),
        EnumCase("down", 4),
        EnumCase("up", 5),
        EnumCase("stochastic", 6),
        EnumCase("satfinite", 7),
        EnumCase("overflow_to_inf", 8),
        EnumCase("overflow_to_nan", 9),
        EnumCase("flush_subnormal", 10),
        EnumCase("preserve_subnormal", 11),
        EnumCase("relu_clamp", 12),
        EnumCase("finite_only", 13),
        EnumCase("finite_flush_subnormal", 14),
    ],
    doc="Rounding and exceptional-value policy for encoded values.",
)

CodebookPolicy = EnumDef(
    "CodebookPolicy",
    [
        EnumCase(keyword, ordinal)
        for ordinal, keyword in enumerate(
            (
                "none",
                "static_builtin_table",
                "static_symbol_table",
                "global_data_table",
                "dynamic_table_operand",
                "per_superblock_table",
            )
        )
    ],
    doc="Codebook source for indexed encoded values.",
)

SparsityPolicy = EnumDef(
    "SparsityPolicy",
    [
        EnumCase(keyword, ordinal)
        for ordinal, keyword in enumerate(
            (
                "none",
                "mask",
                "n_m_structured",
                "block_sparse",
                "bsr",
                "csr",
                "coo",
                "page_table",
                "moe_routing",
                "outlier_side_stream",
            )
        )
    ],
    doc="Sparse payload organization for encoded values.",
)

TransformNormalization = EnumDef(
    "TransformNormalization",
    [
        EnumCase("none", 0),
        EnumCase("orthonormal", 1),
    ],
    doc="Normalization applied by a numeric transform.",
)

_OPERAND_PARAMETERS = (
    AttrDef("affine", ATTR_TYPE_ENUM, enum_def=AffinePolicy, optional=True),
    AttrDef("codebook", ATTR_TYPE_ENUM, enum_def=CodebookPolicy, optional=True),
    AttrDef("element_format", ATTR_TYPE_ENUM, enum_def=NumericFormat),
    AttrDef("payload_elements", ATTR_TYPE_I64),
    AttrDef(
        "payload_packing",
        ATTR_TYPE_ENUM,
        enum_def=PayloadPacking,
        optional=True,
    ),
    AttrDef("payload_registers", ATTR_TYPE_I64, optional=True),
    AttrDef("rounding", ATTR_TYPE_ENUM, enum_def=RoundingPolicy, optional=True),
    AttrDef("scale_format", ATTR_TYPE_ENUM, enum_def=NumericFormat, optional=True),
    AttrDef("scale_group_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("scale_group_shape", ATTR_TYPE_I64_ARRAY, optional=True),
    AttrDef("scale_operands", ATTR_TYPE_I64, optional=True),
    AttrDef(
        "scale_topology",
        ATTR_TYPE_ENUM,
        enum_def=ScaleTopology,
        optional=True,
    ),
    AttrDef(
        "secondary_scale_format",
        ATTR_TYPE_ENUM,
        enum_def=NumericFormat,
        optional=True,
    ),
    AttrDef("sparsity", ATTR_TYPE_ENUM, enum_def=SparsityPolicy, optional=True),
    AttrDef("sparsity_group_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("sparsity_group_nonzero_elements", ATTR_TYPE_I64, optional=True),
    AttrDef("zero_scale_fallback", ATTR_TYPE_BOOL, optional=True),
)

_CANONICAL_NUMERIC_SCHEMA_FORMATS = (
    "f64",
    "f32",
    "tf32",
    "f16",
    "bf16",
    "i32",
    "u32",
    "i16",
    "u16",
    "i8",
    "u8",
    "i6",
    "u6",
    "i5",
    "u5",
    "i4",
    "u4",
    "i3",
    "u3",
    "i2",
    "u2",
    "i1",
    "u1",
    "f8e4m3",
    "f8e5m2",
    "f8e4m3fn",
    "f8e4m3fnuz",
    "f8e5m2fnuz",
    "e8m0",
    "bf8",
    "f6e3m2",
    "f6e2m3",
    "bf6",
    "f4e2m1",
)

ALL_ENCODING_FAMILIES: tuple[EncodingFamilyDef, ...] = (
    EncodingFamilyDef(
        "encoding.storage",
        group=encoding_ops,
        role=EncodingFamilyRole.PHYSICAL_STORAGE,
        parameters=(
            AttrDef("layout", ATTR_TYPE_ENCODING, optional=True),
            AttrDef("schema", ATTR_TYPE_ENCODING, optional=True),
        ),
        dynamic_parameters=(
            Operand("layout", ENCODING_LAYOUT),
            Operand("schema", ENCODING_SCHEMA),
        ),
        doc="Composes an address layout and storage schema.",
    ),
    EncodingFamilyDef(
        "encoding.layout.dense",
        group=encoding_ops,
        role=EncodingFamilyRole.ADDRESS_LAYOUT,
        implicit_shaped_attachment=True,
        doc="Dense row-major address layout.",
    ),
    EncodingFamilyDef(
        "encoding.layout.strided",
        group=encoding_ops,
        role=EncodingFamilyRole.ADDRESS_LAYOUT,
        parameters=(AttrDef("strides", ATTR_TYPE_I64_ARRAY),),
        doc="Explicit element-stride address layout.",
    ),
    EncodingFamilyDef(
        "ggml.q4_0",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        fixed_record=EncodingRecordDef(32, 18, required_alignment=2),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=_enum_fact(NumericFormat, "quant_i4"),
            scale_format=_enum_fact(NumericFormat, "f16"),
            payload_packing=_enum_fact(PayloadPacking, "little_endian_nibbles"),
            scale_topology=_enum_fact(ScaleTopology, "block_1d"),
            affine_policy=_enum_fact(AffinePolicy, "scale_only"),
            payload_element_count=32,
            scale_group_shape=(32,),
            scale_operand_count=1,
        ),
        auxiliary_key_enum=AuxiliaryKey,
        required_auxiliary_keys=(AuxiliaryKey.case("scale"),),
        doc="GGML Q4_0 block storage schema.",
    ),
    EncodingFamilyDef(
        "ggml.q8_0",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        fixed_record=EncodingRecordDef(32, 34, required_alignment=2),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=_enum_fact(NumericFormat, "quant_i8"),
            scale_format=_enum_fact(NumericFormat, "f16"),
            payload_packing=_enum_fact(PayloadPacking, "dense_lanes"),
            scale_topology=_enum_fact(ScaleTopology, "block_1d"),
            affine_policy=_enum_fact(AffinePolicy, "scale_only"),
            payload_element_count=32,
            scale_group_shape=(32,),
            scale_operand_count=1,
        ),
        auxiliary_key_enum=AuxiliaryKey,
        required_auxiliary_keys=(AuxiliaryKey.case("scale"),),
        doc="GGML Q8_0 block storage schema.",
    ),
    EncodingFamilyDef(
        "ggml.q4_k",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        fixed_record=EncodingRecordDef(256, 144, required_alignment=2),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=_enum_fact(NumericFormat, "u4"),
            scale_format=_enum_fact(NumericFormat, "f16"),
            secondary_scale_format=_enum_fact(NumericFormat, "u6"),
            payload_packing=_enum_fact(PayloadPacking, "multi_stream"),
            scale_topology=_enum_fact(ScaleTopology, "hierarchical"),
            affine_policy=_enum_fact(AffinePolicy, "scale_plus_min"),
            payload_element_count=256,
            scale_group_shape=(32,),
            scale_operand_count=2,
        ),
        auxiliary_key_enum=AuxiliaryKey,
        required_auxiliary_keys=(
            AuxiliaryKey.case("scale"),
            AuxiliaryKey.case("secondary_scale"),
            AuxiliaryKey.case("minimum"),
        ),
        doc="GGML Q4_K super-block storage schema.",
    ),
    EncodingFamilyDef(
        "ggml.q6_k",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        fixed_record=EncodingRecordDef(256, 210, required_alignment=2),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=_enum_fact(NumericFormat, "quant_i6"),
            scale_format=_enum_fact(NumericFormat, "f16"),
            secondary_scale_format=_enum_fact(NumericFormat, "i8"),
            payload_packing=_enum_fact(PayloadPacking, "multi_stream"),
            scale_topology=_enum_fact(ScaleTopology, "hierarchical"),
            affine_policy=_enum_fact(AffinePolicy, "super_scale_times_subscale"),
            payload_element_count=256,
            scale_group_shape=(16,),
            scale_operand_count=2,
        ),
        auxiliary_key_enum=AuxiliaryKey,
        required_auxiliary_keys=(
            AuxiliaryKey.case("scale"),
            AuxiliaryKey.case("secondary_scale"),
        ),
        doc="GGML Q6_K super-block storage schema.",
    ),
    EncodingFamilyDef(
        "ggml.q8_1_x4",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        fixed_record=EncodingRecordDef(128, 144, required_alignment=16),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=_enum_fact(NumericFormat, "quant_i8"),
            scale_format=_enum_fact(NumericFormat, "f16"),
            payload_packing=_enum_fact(PayloadPacking, "separate_scale_payload"),
            scale_topology=_enum_fact(ScaleTopology, "block_1d"),
            affine_policy=_enum_fact(AffinePolicy, "sum_correction"),
            payload_element_count=128,
            scale_group_shape=(32,),
            scale_operand_count=1,
        ),
        auxiliary_key_enum=AuxiliaryKey,
        required_auxiliary_keys=(
            AuxiliaryKey.case("scale"),
            AuxiliaryKey.case("sum_correction"),
        ),
        doc="GGML Vulkan Q8_1 x4 block storage schema.",
    ),
    EncodingFamilyDef(
        "encoding.operand",
        group=encoding_ops,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=_OPERAND_PARAMETERS,
        aliases=tuple(
            EncodingAliasDef(
                f"encoding.{numeric_format}",
                fixed_parameters={"element_format": numeric_format},
                default_parameters={
                    "payload_elements": 1,
                    "payload_packing": "dense_lanes",
                },
            )
            for numeric_format in _CANONICAL_NUMERIC_SCHEMA_FORMATS
        ),
        auxiliary_key_enum=AuxiliaryKey,
        doc="Target-independent encoded operand schema.",
    ),
    EncodingFamilyDef(
        "transform.hadamard",
        group=encoding_ops,
        role=EncodingFamilyRole.NUMERIC_TRANSFORM,
        parameters=(
            AttrDef(
                "normalization",
                ATTR_TYPE_ENUM,
                enum_def=TransformNormalization,
                optional=True,
            ),
        ),
        doc="Sylvester Hadamard transform over the final vector axis.",
    ),
)

# ============================================================================
# Semantic encoding query attributes
# ============================================================================

encoding_match_requirements = ParameterizedAttrDef(
    "encoding.match",
    group=encoding_ops,
    parameters=(
        AttrDef(
            "element_format",
            ATTR_TYPE_ENUM,
            enum_def=NumericFormat,
            optional=True,
        ),
        AttrDef(
            "payload_packing",
            ATTR_TYPE_ENUM,
            enum_def=PayloadPacking,
            optional=True,
        ),
        AttrDef(
            "affine",
            ATTR_TYPE_ENUM,
            enum_def=AffinePolicy,
            optional=True,
        ),
    ),
    doc="Typed semantic requirements for an encoded storage schema.",
)

ALL_ENCODING_PARAMETERIZED_ATTRS: tuple[ParameterizedAttrDef, ...] = (encoding_match_requirements,)

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
        "%enc = encoding.define #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes> : encoding<schema>",
        "%enc = encoding.define #encoding.operand<element_format=i8, payload_elements=32, payload_packing=dense_lanes> {group_size = %group_size : index} : encoding<schema>",
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
    traits=[PURE, FACT_IDENTITY],
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
        "%schema2 = encoding.assume.spec %schema, #ggml.q4_0 : encoding<schema>",
    ],
)

# ============================================================================
# encoding.isa — test an exact static encoding specification
# ============================================================================

encoding_isa = Op(
    name="encoding.isa",
    group=encoding_ops,
    phase=OpPhase.COMPILE_TIME_QUERY,
    doc="Test if an encoding exactly matches a static encoding specification.",
    operands=[Operand("enc", ANY_ENCODING, doc="Encoding value to query.")],
    results=[Result("result", I1)],
    attrs=[AttrDef("spec", ATTR_TYPE_ENCODING, doc="Exact static encoding specification.")],
    traits=[PURE],
    condition_refinement=ConditionRefinement(
        source="enc",
        truth=ConditionRefinementTruth.TRUE,
        materialize="loom_encoding_isa_materialize_refinement",
    ),
    verify="loom_encoding_isa_verify",
    facts="loom_encoding_isa_facts",
    format=[TemplateParam("spec"), Ref("enc"), COLON, TypeOf("enc")],
    examples=[
        "%is_q4 = encoding.isa<#ggml.q4_k> %schema : encoding<schema>",
    ],
)

# ============================================================================
# encoding.matches — test typed semantic encoding facts
# ============================================================================

encoding_matches = Op(
    name="encoding.matches",
    group=encoding_ops,
    phase=OpPhase.COMPILE_TIME_QUERY,
    doc="Test whether a storage schema satisfies typed semantic requirements.",
    operands=[Operand("enc", ENCODING_SCHEMA, doc="Storage schema to query.")],
    results=[Result("result", I1)],
    attrs=[
        AttrDef(
            "requirements",
            ATTR_TYPE_PARAMETERIZED,
            parameterized_attr=encoding_match_requirements,
            doc="Authored semantic requirements; omitted fields are wildcards.",
        ),
    ],
    traits=[PURE],
    condition_refinement=ConditionRefinement(
        source="enc",
        truth=ConditionRefinementTruth.TRUE,
        materialize="loom_encoding_matches_materialize_refinement",
    ),
    verify="loom_encoding_matches_verify",
    facts="loom_encoding_matches_facts",
    format=[AttrParams("requirements"), Ref("enc"), COLON, TypeOf("enc")],
    examples=[
        "%supports = encoding.matches<element_format = u4, payload_packing = multi_stream, affine = scale_plus_min> %schema : encoding<schema>",
    ],
)

# ============================================================================
# encoding.assume.match — local semantic encoding refinement
# ============================================================================

encoding_assume_match = Op(
    name="encoding.assume.match",
    group=encoding_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Refine a storage schema with typed semantic requirements. Omitted fields retain their existing facts."),
    operands=[Operand("enc", ENCODING_SCHEMA, doc="Storage schema to refine.")],
    results=[Result("result", ENCODING_SCHEMA, doc="Schema with stronger semantic facts.")],
    attrs=[
        AttrDef(
            "requirements",
            ATTR_TYPE_PARAMETERIZED,
            parameterized_attr=encoding_match_requirements,
            doc="Authored semantic requirements; omitted fields are unchanged.",
        ),
    ],
    constraints=[SameType("enc", "result")],
    traits=[PURE, FACT_IDENTITY],
    verify="loom_encoding_assume_match_verify",
    facts="loom_encoding_assume_match_facts",
    format=[AttrParams("requirements"), Ref("enc"), COLON, TypeOf("enc")],
    examples=[
        "%schema2 = encoding.assume.match<element_format = u4, affine = scale_plus_min> %schema : encoding<schema>",
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
    encoding_matches,
    encoding_assume_match,
)
