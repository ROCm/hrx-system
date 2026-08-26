# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom.dsl — op declaration DSL."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import loom.dsl as dsl
import loom.ir as ir
from loom.assembly import (
    COLON,
    COMMA,
    EQUALS,
    Attr,
    BlockRef,
    Clause,
    EncodingOf,
    FuncArgs,
    IndexList,
    Keyword,
    OptionalGroup,
    Param,
    Ref,
    Region,
    ResultType,
    ScalarOf,
    Scope,
    ScopedEnumRef,
    ShapeOf,
    SymbolRef,
    TypeOf,
    kw,
)
from loom.dsl import (
    ADDRESS,
    ANY,
    ANY_ENCODING,
    ATTR_TYPE_PARAMETERIZED,
    ATTR_TYPE_SYMBOL,
    ATTR_TYPE_SYMBOL_ARRAY,
    ATTR_TYPE_SYMBOL_SET,
    BUFFER,
    BY_REFERENCE,
    BYTE_PATTERN_SCALAR,
    COMMAND_EFFECT,
    COMMUTATIVE,
    COMPILE_TIME_ONLY,
    CONSTANT_LIKE,
    CONVERGENT,
    DECOMPOSABLE,
    ELEMENTWISE,
    ENCODING_LAYOUT,
    ENCODING_SCHEMA,
    ENCODING_STORAGE,
    ENCODING_TRANSFORM,
    FACT_IDENTITY,
    FLOAT,
    FLOAT_ELEMENT,
    HINT,
    I1,
    I1_ELEMENT,
    IDEMPOTENT,
    INDEX,
    INTEGER,
    INTEGER_ELEMENT,
    INVOLUTION,
    MEMORY_FENCE,
    MODULE_SCOPE,
    NO_RETURN,
    NON_DETERMINISTIC,
    OFFSET,
    POISON,
    POOL,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    REGISTER,
    SAFE_TO_SPECULATE,
    STORAGE,
    STORAGE_RELATION,
    SYMBOL_DEFINE,
    TENSOR,
    TERMINATOR,
    TILE,
    UNIQUE_IDENTITY,
    UNKNOWN_EFFECTS,
    AliasResult,
    AllShapesMatch,
    AllTypesMatch,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    BlockArgCount,
    BlockArgsMatchElementTypes,
    BlockArgsMatchTypes,
    BlockArgsSatisfy,
    Borrow,
    CallLikeInterface,
    CallLikeKind,
    Consume,
    ContractFamily,
    Dialect,
    DimIndexInBounds,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    ElementWidthLessThan,
    EncodingAliasDef,
    EncodingFamilyDef,
    EncodingFamilyRole,
    EncodingOperandSummaryDef,
    EncodingParam,
    EncodingRecordDef,
    EnumCase,
    EnumDef,
    FreshResult,
    FuncLikeInterface,
    HasAllStaticRankOneVector,
    HasAllStaticVector,
    HasAncestor,
    HasBitwiseElement,
    HasBitwiseScalar,
    HasF16OrBf16Element,
    HasF32Element,
    HasFloatElement,
    HasI1Element,
    HasI8Element,
    HasI32Element,
    HasIndexOrNonI1IntegerElement,
    HasIndexOrNonI1IntegerScalar,
    HasIntegerElement,
    HasParent,
    HasRankOneVector,
    HasRegister,
    ImplicitTerminator,
    KeyedModuleRecord,
    LastAxisGroupedBy,
    LegacyFieldDefault,
    LegacyFieldMapping,
    LegacyFormat,
    LiteralMatchesElementType,
    MemoryAccessInterface,
    MovedResult,
    NoAncestor,
    OffsetCountMatchesRank,
    Op,
    OpCategory,
    Operand,
    OperandOwnershipEffect,
    OpPhase,
    PackedPayloadBitCountMatchesStorage,
    ParameterizedAttrDef,
    PositiveBitWidthAttr,
    RanksMatch,
    Reads,
    RegionDef,
    Result,
    ResultOwnershipEffect,
    SameElementType,
    SameEncoding,
    SameKind,
    SameShape,
    SameType,
    ScalarParam,
    ShapeParam,
    Successor,
    SymbolDefinition,
    SymbolKernelContract,
    SymbolReference,
    SymbolReferenceRole,
    TargetFactSpecialization,
    TargetLikeInterface,
    TotalBitCountEqual,
    TypeConstraint,
    TypeDef,
    TypeSemantic,
    UnpackedPayloadBitCountMatchesStorage,
    ValueCountMatchesStaticElementCount,
    Writes,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
    binary_op,
    cast_op,
    comparison_op,
    type_constraint_name,
    unary_op,
)
from loom.ir import (
    BF16,
    F16,
    F32,
    I8,
    I16,
    I32,
    DynamicDim,
    RegisterType,
    ScalarType,
    ShapedType,
    StaticDim,
    TypeKind,
)


@contextmanager
def _raises(
    exception_type: type[BaseException], match: str | None = None
) -> Iterator[None]:
    try:
        yield
    except exception_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"exception message {str(exc)!r} did not match {match!r}"
            ) from exc
        return
    raise AssertionError(f"{exception_type.__name__} was not raised")


# ============================================================================
# Test fixtures
# ============================================================================

_scalar_ops = Dialect("scalar", doc="Scalar ops.")
_tile_ops = Dialect("tile", doc="Tile ops.")
_func_ops = Dialect("func", doc="Function ops.")
_scf_ops = Dialect("scf", doc="Structured control flow.")

_cmpi_preds = EnumDef(
    "CmpIPredicate",
    [
        EnumCase("eq", 0),
        EnumCase("ne", 1),
        EnumCase("slt", 2),
        EnumCase("sle", 3),
        EnumCase("sgt", 4),
        EnumCase("sge", 5),
    ],
)


# ============================================================================
# Type constraints
# ============================================================================


class TestTypeConstraints:
    def test_all_constraints_have_names(self) -> None:
        for tc in TypeConstraint:
            name = type_constraint_name(tc)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_singleton_equality(self) -> None:
        assert TILE == TypeConstraint.TILE
        assert INTEGER == TypeConstraint.INTEGER
        assert OFFSET == TypeConstraint.OFFSET
        assert ADDRESS == TypeConstraint.ADDRESS
        assert BUFFER == TypeConstraint.BUFFER
        assert BYTE_PATTERN_SCALAR == TypeConstraint.BYTE_PATTERN_SCALAR
        assert dsl.I32 == TypeConstraint.I32
        assert STORAGE == TypeConstraint.STORAGE
        assert ANY_ENCODING == TypeConstraint.ANY_ENCODING
        assert ENCODING_LAYOUT == TypeConstraint.ENCODING_LAYOUT
        assert INTEGER_ELEMENT == TypeConstraint.INTEGER_ELEMENT
        assert TILE != TENSOR

    def test_values(self) -> None:
        assert TILE.value == "tile"
        assert FLOAT.value == "float"
        assert ANY_ENCODING.value == "encoding"
        assert ENCODING_LAYOUT.value == "encoding<layout>"
        assert ENCODING_SCHEMA.value == "encoding<schema>"
        assert ENCODING_STORAGE.value == "encoding<storage>"
        assert ENCODING_TRANSFORM.value == "encoding<transform>"
        assert FLOAT_ELEMENT.value == "float_element"
        assert I1_ELEMENT.value == "i1_element"
        assert INDEX.value == "index"
        assert OFFSET.value == "offset"
        assert ADDRESS.value == "address"
        assert STORAGE.value == "storage"
        assert BYTE_PATTERN_SCALAR.value == "byte_pattern_scalar"
        assert dsl.I32.value == "i32"

    def test_element_family_constraints_are_shaped_specific(self) -> None:
        element_family_constraints = {
            constraint
            for constraint in TypeConstraint
            if constraint.value.startswith(("integer_", "float_", "i1_"))
        }
        assert element_family_constraints == {
            TypeConstraint.INTEGER_ELEMENT,
            TypeConstraint.FLOAT_ELEMENT,
            TypeConstraint.I1_ELEMENT,
        }


# ============================================================================
# Operands and results
# ============================================================================


class TestOperand:
    def test_basic(self) -> None:
        o = Operand("lhs", INTEGER)
        assert o.name == "lhs"
        assert o.type_constraint == INTEGER
        assert not o.variadic
        assert not o.optional

    def test_variadic(self) -> None:
        o = Operand("inputs", TILE, variadic=True)
        assert o.variadic

    def test_optional(self) -> None:
        o = Operand("acc", TILE, optional=True)
        assert o.optional

    def test_with_doc(self) -> None:
        o = Operand("lhs", INTEGER, doc="Left operand.")
        assert o.doc == "Left operand."


class TestResult:
    def test_basic(self) -> None:
        r = Result("result", FLOAT)
        assert r.name == "result"
        assert r.type_constraint == FLOAT
        assert not r.variadic

    def test_variadic(self) -> None:
        r = Result("results", ANY, variadic=True)
        assert r.variadic


# ============================================================================
# Attributes and enums
# ============================================================================


class TestAttrDef:
    def test_basic(self) -> None:
        a = AttrDef("axis", "i64")
        assert a.name == "axis"
        assert a.attr_type == "i64"
        assert a.default is None
        assert not a.optional

    def test_with_default(self) -> None:
        a = AttrDef("transpose", "bool", default="false")
        assert a.default == "false"

    def test_elide_default(self) -> None:
        a = AttrDef("offset", "i64", default=0, elide_default=True)
        assert a.default == 0
        assert a.elide_default

    def test_elide_default_requires_default(self) -> None:
        with _raises(ValueError, match="elide_default requires default"):
            AttrDef("offset", "i64", elide_default=True)

    def test_enum_attr(self) -> None:
        a = AttrDef("predicate", "enum", enum_def=_cmpi_preds)
        assert a.enum_def is not None
        assert a.enum_def.name == "CmpIPredicate"

    def test_invalid_attr_type_rejected(self) -> None:
        with _raises(ValueError, match="invalid attr_type"):
            AttrDef("axis", "i64array")  # Invalid attr_type.

    def test_enum_without_enum_def_rejected(self) -> None:
        with _raises(ValueError, match="requires enum_def"):
            AttrDef("pred", "enum")  # Missing enum_def!

        with _raises(ValueError, match="requires enum_def"):
            AttrDef("modes", "enum_array")

        with _raises(ValueError, match="requires enum_def"):
            AttrDef("features", "signed_enum_set")

    def test_all_valid_attr_types(self) -> None:
        """All documented attr_type values are accepted."""
        for attr_type in [
            "i64",
            "f64",
            "string",
            "bool",
            "type",
            "i64_array",
            "any",
            "scoped_enum",
        ]:
            AttrDef("test", attr_type)  # Should not raise.
        AttrDef("test", "enum", enum_def=_cmpi_preds)  # enum needs enum_def.
        AttrDef("test", "enum_array", enum_def=_cmpi_preds)
        AttrDef("test", "signed_enum_set", enum_def=_cmpi_preds)
        AttrDef(
            "test",
            ATTR_TYPE_SYMBOL_ARRAY,
            symbol_ref=SymbolReference("record", ["record"]),
        )
        AttrDef(
            "test",
            ATTR_TYPE_SYMBOL_SET,
            symbol_ref=SymbolReference("record", ["record"]),
        )

    def test_symbol_array_requires_reference_contract(self) -> None:
        with _raises(ValueError, match="symbol_array.*requires symbol_ref"):
            AttrDef("providers", ATTR_TYPE_SYMBOL_ARRAY)

    def test_symbol_set_requires_reference_contract(self) -> None:
        with _raises(ValueError, match="symbol_set.*requires symbol_ref"):
            AttrDef("providers", ATTR_TYPE_SYMBOL_SET)

    def test_symbol_set_cannot_be_a_parameter(self) -> None:
        with _raises(ValueError, match="unsupported kind 'symbol_set'"):
            ParameterizedAttrDef(
                "test.providers",
                group=Dialect("test"),
                parameters=[
                    AttrDef(
                        "providers",
                        ATTR_TYPE_SYMBOL_SET,
                        symbol_ref=SymbolReference("record", ["record"]),
                    )
                ],
            )

    def test_symbol_reference_contract_requires_symbol_kind(self) -> None:
        with _raises(ValueError, match="symbol_ref requires"):
            AttrDef(
                "providers",
                "i64_array",
                symbol_ref=SymbolReference("record", ["record"]),
            )

    def test_open_enum_array(self) -> None:
        attr = AttrDef("modes", "enum_array", enum_def=_cmpi_preds, open_enum=True)
        assert attr.open_enum

    def test_signed_enum_set_rejects_open_domain(self) -> None:
        with _raises(ValueError, match="open_enum requires an enum attribute"):
            AttrDef(
                "features",
                "signed_enum_set",
                enum_def=_cmpi_preds,
                open_enum=True,
            )

    def test_scoped_enum_may_be_optional_but_never_defaulted(self) -> None:
        assert AttrDef("descriptor", "scoped_enum", optional=True).optional
        with _raises(ValueError, match="scoped_enum attributes cannot have defaults"):
            AttrDef("descriptor", "scoped_enum", default=0)

    def test_bare_identifier_requires_string(self) -> None:
        assert AttrDef("mode", "string", bare_identifier=True).bare_identifier
        with _raises(ValueError, match="bare_identifier requires"):
            AttrDef("mode", "i64", bare_identifier=True)


class TestParameterizedAttrDef:
    def test_declares_namespaced_typed_parameters(self) -> None:
        tile = ParameterizedAttrDef(
            "test.tile",
            group=Dialect("test"),
            parameters=[AttrDef("width", "i64")],
        )

        assert tile.name == "test.tile"
        assert tile.parameters == (AttrDef("width", "i64"),)
        assert tile.primary_parameter is None

    def test_declares_required_primary_parameter_by_stable_name(self) -> None:
        compact = ParameterizedAttrDef(
            "test.compact",
            group=Dialect("test"),
            parameters=[
                AttrDef("label", "string", optional=True),
                AttrDef("value", "i64"),
            ],
            primary_parameter="value",
        )

        assert compact.primary_parameter_index == 1
        assert compact.primary_parameter == AttrDef("value", "i64")

    def test_declares_typed_target_condition_descriptor(self) -> None:
        condition = ParameterizedAttrDef(
            "target.subgroup.size",
            group=Dialect("target"),
            parameters=[AttrDef("size", "i64")],
            primary_parameter="size",
            target_condition="loom_target_subgroup_size_condition",
        )

        assert condition.target_condition == "loom_target_subgroup_size_condition"

    def test_rejects_invalid_target_condition_symbol(self) -> None:
        with _raises(ValueError, match="must be a C symbol name"):
            ParameterizedAttrDef(
                "target.subgroup.size",
                group=Dialect("target"),
                target_condition="target.subgroup.size",
            )

    def test_rejects_missing_or_optional_primary_parameter(self) -> None:
        with _raises(ValueError, match="primary parameter 'missing' is not declared"):
            ParameterizedAttrDef(
                "test.compact",
                group=Dialect("test"),
                parameters=[AttrDef("value", "i64")],
                primary_parameter="missing",
            )
        with _raises(ValueError, match="primary parameter 'value' must be required"):
            ParameterizedAttrDef(
                "test.compact",
                group=Dialect("test"),
                parameters=[AttrDef("value", "i64", optional=True)],
                primary_parameter="value",
            )

    def test_rejects_wrong_namespace(self) -> None:
        with _raises(ValueError, match="must begin with the owning dialect"):
            ParameterizedAttrDef("other.tile", group=Dialect("test"))

    def test_rejects_duplicate_parameters(self) -> None:
        with _raises(ValueError, match="duplicate parameter 'width'"):
            ParameterizedAttrDef(
                "test.tile",
                group=Dialect("test"),
                parameters=[AttrDef("width", "i64"), AttrDef("width", "i64")],
            )

    def test_rejects_enclosing_operation_parameter_kinds(self) -> None:
        with _raises(ValueError, match="unsupported kind 'scoped_enum'"):
            ParameterizedAttrDef(
                "test.tile",
                group=Dialect("test"),
                parameters=[AttrDef("scope", "scoped_enum")],
            )

    def test_nested_parameter_allows_open_family(self) -> None:
        options = ParameterizedAttrDef(
            "test.options",
            group=Dialect("test"),
            parameters=[AttrDef("value", ATTR_TYPE_PARAMETERIZED)],
        )

        assert options.parameters[0].parameterized_attr is None

    def test_nested_parameter_retains_exact_family(self) -> None:
        dialect = Dialect("test")
        tile = ParameterizedAttrDef("test.tile", group=dialect)
        options = ParameterizedAttrDef(
            "test.options",
            group=dialect,
            parameters=[
                AttrDef(
                    "tile",
                    ATTR_TYPE_PARAMETERIZED,
                    optional=True,
                    parameterized_attr=tile,
                )
            ],
        )

        assert options.parameters[0].parameterized_attr is tile


class TestEncodingFamilyDef:
    def test_declares_implicit_shaped_attachment(self) -> None:
        family = EncodingFamilyDef(
            "encoding.layout.dense",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.ADDRESS_LAYOUT,
            implicit_shaped_attachment=True,
        )

        assert family.implicit_shaped_attachment

    def test_rejects_parameterized_implicit_shaped_attachment(self) -> None:
        with _raises(ValueError, match="cannot have static or dynamic parameters"):
            EncodingFamilyDef(
                "encoding.layout.parameterized",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.ADDRESS_LAYOUT,
                parameters=[AttrDef("value", "i64")],
                implicit_shaped_attachment=True,
            )

    def test_rejects_non_layout_implicit_shaped_attachment(self) -> None:
        with _raises(ValueError, match="must have the ADDRESS_LAYOUT role"):
            EncodingFamilyDef(
                "encoding.schema.dense",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                implicit_shaped_attachment=True,
            )

    def test_declares_canonical_alias_with_typed_fixed_parameters(self) -> None:
        numeric_format = EnumDef(
            "NumericFormat", [EnumCase("f16", 0), EnumCase("bf16", 1)]
        )
        family = EncodingFamilyDef(
            "encoding.operand",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            parameters=[
                AttrDef("payload_elements", "i64"),
                AttrDef("element_format", "enum", enum_def=numeric_format),
            ],
            aliases=[
                EncodingAliasDef(
                    "encoding.f16",
                    fixed_parameters={"element_format": "f16"},
                    default_parameters={"payload_elements": 1},
                )
            ],
        )

        assert family.aliases[0].fixed_parameters == (("element_format", "f16"),)
        assert family.aliases[0].default_parameters == (("payload_elements", 1),)
        assert family.alias_discriminator is not None
        assert family.alias_discriminator.name == "element_format"

    def test_rejects_alias_that_fixes_unknown_parameter(self) -> None:
        with _raises(ValueError, match="references unknown parameter 'missing'"):
            EncodingFamilyDef(
                "encoding.operand",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                aliases=[
                    EncodingAliasDef("encoding.f16", fixed_parameters={"missing": 1})
                ],
            )

    def test_rejects_aliases_without_one_direct_enum_discriminator(self) -> None:
        numeric_format = EnumDef(
            "NumericFormat", [EnumCase("f16", 0), EnumCase("bf16", 1)]
        )
        with _raises(ValueError, match="require one shared enum fixed parameter"):
            EncodingFamilyDef(
                "encoding.operand",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                parameters=[
                    AttrDef("element_format", "enum", enum_def=numeric_format),
                    AttrDef("payload_elements", "i64"),
                ],
                aliases=[
                    EncodingAliasDef(
                        "encoding.f16",
                        fixed_parameters={"element_format": "f16"},
                    ),
                    EncodingAliasDef(
                        "encoding.f16_scalar",
                        fixed_parameters={
                            "element_format": "f16",
                            "payload_elements": 1,
                        },
                    ),
                ],
            )

    def test_rejects_more_than_uint8_alias_ordinals(self) -> None:
        numeric_format = EnumDef(
            "NumericFormat",
            [EnumCase(f"f{value}", value) for value in range(256)],
        )
        with _raises(ValueError, match="256 aliases exceed the uint8_t ordinal limit"):
            EncodingFamilyDef(
                "encoding.operand",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                parameters=[
                    AttrDef("element_format", "enum", enum_def=numeric_format),
                ],
                aliases=[
                    EncodingAliasDef(
                        f"encoding.f{value}",
                        fixed_parameters={"element_format": f"f{value}"},
                    )
                    for value in range(256)
                ],
            )

    def test_declares_lexically_indexed_sparse_parameters(self) -> None:
        family = EncodingFamilyDef(
            "operand",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            parameters=[
                AttrDef("rounding", "string", optional=True),
                AttrDef("payload_elements", "i64"),
            ],
        )

        assert family.name == "operand"
        assert family.role is EncodingFamilyRole.STORAGE_SCHEMA
        assert tuple(parameter.name for parameter in family.parameters) == (
            "payload_elements",
            "rounding",
        )

    def test_declares_lexically_indexed_dynamic_parameters(self) -> None:
        family = EncodingFamilyDef(
            "storage",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.PHYSICAL_STORAGE,
            dynamic_parameters=[
                Operand("schema", TypeConstraint.ENCODING_SCHEMA),
                Operand("layout", TypeConstraint.ENCODING_LAYOUT),
            ],
        )

        assert tuple(parameter.name for parameter in family.dynamic_parameters) == (
            "layout",
            "schema",
        )

    def test_rejects_duplicate_dynamic_parameters_after_sorting(self) -> None:
        with _raises(ValueError, match="duplicate dynamic parameter 'value'"):
            EncodingFamilyDef(
                "schema",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                dynamic_parameters=[
                    Operand("value", TypeConstraint.INDEX),
                    Operand("value", TypeConstraint.INDEX),
                ],
            )

    def test_accepts_qualified_family_name(self) -> None:
        family = EncodingFamilyDef(
            "ggml.q4_k",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.STORAGE_SCHEMA,
        )

        assert family.name == "ggml.q4_k"

    def test_rejects_invalid_qualified_family_name(self) -> None:
        with _raises(ValueError, match="dotted ASCII identifier"):
            EncodingFamilyDef(
                "ggml..q4_k",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
            )

    def test_rejects_duplicate_parameters_after_sorting(self) -> None:
        with _raises(ValueError, match="duplicate parameter 'mode'"):
            EncodingFamilyDef(
                "schema",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                parameters=[
                    AttrDef("mode", "string"),
                    AttrDef("mode", "string"),
                ],
            )

    def test_declares_fixed_storage_metadata(self) -> None:
        auxiliary_keys = EnumDef(
            "AuxiliaryKey",
            [EnumCase("scale", 0), EnumCase("minimum", 1)],
        )
        family = EncodingFamilyDef(
            "ggml.q4_k",
            group=Dialect("encoding"),
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            fixed_record=EncodingRecordDef(
                logical_element_count=256,
                storage_byte_count=144,
                required_alignment=16,
            ),
            fixed_operand_summary=EncodingOperandSummaryDef(
                payload_packing=4,
                payload_element_count=256,
                scale_group_shape=(16, 16),
            ),
            auxiliary_key_enum=auxiliary_keys,
            required_auxiliary_keys=[auxiliary_keys.case("scale")],
        )

        assert family.fixed_record is not None
        assert family.fixed_record.storage_byte_count == 144
        assert family.fixed_operand_summary is not None
        assert family.fixed_operand_summary.scale_group_element_count == 256
        assert family.required_auxiliary_keys == (auxiliary_keys.case("scale"),)

    def test_rejects_fixed_metadata_on_non_storage_family(self) -> None:
        with _raises(ValueError, match="requires the STORAGE_SCHEMA role"):
            EncodingFamilyDef(
                "dense",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.ADDRESS_LAYOUT,
                fixed_record=EncodingRecordDef(1, 1),
            )

    def test_rejects_auxiliary_key_from_another_enum(self) -> None:
        expected_keys = EnumDef("ExpectedKey", [EnumCase("scale", 0)])
        other_keys = EnumDef("OtherKey", [EnumCase("scale", 0)])
        with _raises(ValueError, match="is not in ExpectedKey"):
            EncodingFamilyDef(
                "schema",
                group=Dialect("encoding"),
                role=EncodingFamilyRole.STORAGE_SCHEMA,
                auxiliary_key_enum=expected_keys,
                required_auxiliary_keys=[other_keys.case("scale")],
            )


class TestEncodingRecordDef:
    def test_rejects_non_power_of_two_alignment(self) -> None:
        with _raises(ValueError, match="must be a power of two"):
            EncodingRecordDef(16, 8, required_alignment=3)


class TestEncodingOperandSummaryDef:
    def test_canonicalizes_scale_group_element_count(self) -> None:
        summary = EncodingOperandSummaryDef(scale_group_shape=(4, 8))

        assert summary.scale_group_element_count == 32

    def test_rejects_inconsistent_scale_group_element_count(self) -> None:
        with _raises(ValueError, match="must equal the scale_group_shape product"):
            EncodingOperandSummaryDef(
                scale_group_element_count=16,
                scale_group_shape=(4, 8),
            )

    def test_rejects_out_of_range_fact_bits(self) -> None:
        with _raises(ValueError, match="unsigned 32-bit integer"):
            EncodingOperandSummaryDef(payload_packing=1 << 32)


class TestEnumDef:
    def test_basic(self) -> None:
        e = _cmpi_preds
        assert e.name == "CmpIPredicate"
        assert len(e.cases) == 6

    def test_keywords(self) -> None:
        assert _cmpi_preds.keywords == ("eq", "ne", "slt", "sle", "sgt", "sge")

    def test_accepts_list(self) -> None:
        e = EnumDef("Test", [EnumCase("a", 0)])
        assert isinstance(e.cases, tuple)

    def test_empty_name_rejected(self) -> None:
        with _raises(ValueError, match="name must be non-empty"):
            EnumDef("", [EnumCase("a", 0)])

    def test_empty_cases_rejected(self) -> None:
        with _raises(ValueError, match="cases must be non-empty"):
            EnumDef("Bad", [])

    def test_empty_keyword_rejected(self) -> None:
        with _raises(ValueError, match="case keyword must be non-empty"):
            EnumDef("Bad", [EnumCase("", 0)])

    def test_value_domain_rejected(self) -> None:
        for value in [False, -1, 256, "1"]:
            with _raises(ValueError, match=r"integer in \[0, 255\]"):
                EnumDef("Bad", [EnumCase("a", value)])  # type: ignore[arg-type]

    def test_full_byte_domain_accepted(self) -> None:
        e = EnumDef(
            "Byte",
            [EnumCase(f"v{i}", i) for i in range(256)],
        )
        assert len(e.cases) == 256
        assert e.cases[-1].value == 255

    def test_duplicate_keyword_rejected(self) -> None:
        with _raises(ValueError, match="duplicate keyword 'eq'"):
            EnumDef("Bad", [EnumCase("eq", 0), EnumCase("eq", 1)])

    def test_duplicate_value_rejected(self) -> None:
        with _raises(ValueError, match="duplicate value 0"):
            EnumDef("Bad", [EnumCase("a", 0), EnumCase("b", 0)])

    def test_external_c_enum_alias_metadata(self) -> None:
        e = EnumDef(
            "Mode",
            [EnumCase("fast", 0)],
            c_type="loom_shared_mode_t",
            c_const_prefix="LOOM_SHARED_MODE",
            c_include="loom/shared/mode.h",
        )

        assert e.c_type == "loom_shared_mode_t"
        assert e.c_const_prefix == "LOOM_SHARED_MODE"
        assert e.c_include == "loom/shared/mode.h"

    def test_external_c_enum_alias_requires_type_and_prefix(self) -> None:
        with _raises(
            ValueError,
            match="c_type and c_const_prefix must be provided together",
        ):
            EnumDef(
                "Mode",
                [EnumCase("fast", 0)],
                c_type="loom_shared_mode_t",
            )

        with _raises(ValueError, match="c_include requires c_type"):
            EnumDef(
                "Mode",
                [EnumCase("fast", 0)],
                c_include="loom/shared/mode.h",
            )


class TestEnumCase:
    def test_basic(self) -> None:
        c = EnumCase("slt", 2, doc="Signed less than.")
        assert c.keyword == "slt"
        assert c.value == 2
        assert c.doc == "Signed less than."


# ============================================================================
# Regions
# ============================================================================


class TestRegionDef:
    def test_basic(self) -> None:
        r = RegionDef("body")
        assert r.name == "body"
        assert not r.single_block

    def test_single_block(self) -> None:
        r = RegionDef("body", single_block=True)
        assert r.single_block

    def test_arg_source(self) -> None:
        r = RegionDef("body", arg_source="inputs")
        assert r.arg_source == "inputs"


# ============================================================================
# Traits
# ============================================================================


class TestTraits:
    def test_simple_trait(self) -> None:
        assert PURE.name == "Pure"
        assert PURE.args == ()

    def test_trait_repr(self) -> None:
        assert repr(PURE) == "Pure"
        assert repr(COMMUTATIVE) == "Commutative"

    def test_parameterized_trait(self) -> None:
        t = AllTypesMatch("lhs", "rhs", "result")
        assert t.name == "AllTypesMatch"
        assert t.args == ("lhs", "rhs", "result")
        assert repr(t) == "AllTypesMatch(lhs, rhs, result)"

    def test_has_parent(self) -> None:
        t = HasParent("scf.for")
        assert t.args == ("scf.for",)

    def test_ancestor_placement_traits(self) -> None:
        required = HasAncestor("low.func.def")
        forbidden = NoAncestor("low.func.def")
        assert required.name == "HasAncestor"
        assert required.args == ("low.func.def",)
        assert forbidden.name == "NoAncestor"
        assert forbidden.args == ("low.func.def",)

    def test_implicit_terminator(self) -> None:
        t = ImplicitTerminator("scf.yield")
        assert t.args == ("scf.yield",)

    def test_all_standard_traits(self) -> None:
        standard = [
            PURE,
            COMMUTATIVE,
            IDEMPOTENT,
            INVOLUTION,
            TERMINATOR,
            CONSTANT_LIKE,
            ELEMENTWISE,
            DECOMPOSABLE,
            CONVERGENT,
            SAFE_TO_SPECULATE,
            REFINABLE_RESULT_TYPE_REFS,
        ]
        names = [t.name for t in standard]
        assert len(set(names)) == len(names), "Duplicate trait names"


# ============================================================================
# Interfaces
# ============================================================================


class TestInterfaces:
    def test_call_like_interface_defaults_to_semantic(self) -> None:
        interface = CallLikeInterface(
            callee="callee",
            operands="operands",
            results=None,
        )
        assert interface.callee == "callee"
        assert interface.operands == "operands"
        assert interface.results is None
        assert interface.purity is None
        assert interface.kind == CallLikeKind.SEMANTIC

    def test_call_like_kind_values(self) -> None:
        assert CallLikeKind.SEMANTIC.value == "semantic"
        assert CallLikeKind.LOW_INTERNAL.value == "low_internal"
        assert CallLikeKind.LOW_INVOKE.value == "low_invoke"

    def test_target_like_interface_defaults_to_no_extensions(self) -> None:
        interface = TargetLikeInterface(symbol="symbol", selector="kind")

        assert interface.symbol == "symbol"
        assert interface.selector == "kind"
        assert interface.extensions is None
        assert interface.descriptor is None
        assert interface.fact_type is None
        assert interface.fact_projector is None
        assert interface.fact_specialization == TargetFactSpecialization.EXACT

    def test_memory_access_interface_uses_soft_field_defaults(self) -> None:
        interface = MemoryAccessInterface()

        assert interface.view == "view"
        assert interface.byte_offset == "byte_offset"
        assert interface.value == "value"
        assert interface.indices == "indices"
        assert interface.static_indices == "static_indices"
        assert interface.cache_scope == "cache_scope"
        assert interface.cache_temporal == "cache_temporal"
        assert interface.atomic_kind == "kind"
        assert interface.atomic_ordering == "ordering"
        assert interface.atomic_success_ordering == "success_ordering"
        assert interface.atomic_failure_ordering == "failure_ordering"
        assert interface.atomic_scope == "scope"
        assert interface._explicit_fields == frozenset()

    def test_memory_access_interface_tracks_explicit_overrides(self) -> None:
        interface = MemoryAccessInterface(value="stored", cache_scope=None)

        assert interface.value == "stored"
        assert interface.cache_scope is None
        assert interface._explicit_fields == frozenset({"value", "cache_scope"})


# ============================================================================
# Constraints
# ============================================================================


class TestConstraints:
    def test_same_type_repr(self) -> None:
        c = SameType("lhs", "rhs")
        assert repr(c) == "SameType(lhs, rhs)"
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_001"

    def test_same_type_validation_pass(self) -> None:
        c = SameType("a", "b")

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        ok, msg = c.check({"a": FakeValue("f32"), "b": FakeValue("f32")})
        assert ok
        assert msg == ""

    def test_same_type_validation_fail(self) -> None:
        c = SameType("a", "b")

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        ok, msg = c.check({"a": FakeValue("f32"), "b": FakeValue("i32")})
        assert not ok
        assert "'b'" in msg
        assert "'a'" in msg

    def test_same_type_missing_values(self) -> None:
        c = SameType("a", "b")
        ok, _msg = c.check({"a": None})
        assert ok, "Missing values should pass (can't check)"

    def test_same_kind(self) -> None:
        c = SameKind("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_001"

    def test_same_element_type(self) -> None:
        c = SameElementType("x", "y")
        assert c.error is not None
        assert c.error.error_id == "ERR_TYPE_002"

    def test_same_encoding(self) -> None:
        c = SameEncoding("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_ENCODING_001"

    def test_same_shape(self) -> None:
        c = SameShape("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_002"

    def test_ranks_match(self) -> None:
        c = RanksMatch("a", "b")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_001"

    def test_element_family_constraints(self) -> None:
        assert HasIntegerElement("x").name == "HasIntegerElement"
        assert HasFloatElement("x").name == "HasFloatElement"
        assert HasIndexOrNonI1IntegerScalar("x").name == "HasIndexOrNonI1IntegerScalar"
        assert (
            HasIndexOrNonI1IntegerElement("x").name == "HasIndexOrNonI1IntegerElement"
        )
        assert HasI1Element("x").name == "HasI1Element"
        assert HasI8Element("x").name == "HasI8Element"
        assert HasI32Element("x").name == "HasI32Element"
        assert HasF16OrBf16Element("x").name == "HasF16OrBf16Element"
        assert HasF32Element("x").name == "HasF32Element"
        assert HasRegister("x").name == "HasRegister"
        assert HasRankOneVector("x").name == "HasRankOneVector"
        assert HasAllStaticVector("x").name == "HasAllStaticVector"
        assert HasAllStaticRankOneVector("x").name == "HasAllStaticRankOneVector"

    def test_register_constraint_validates_register_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        assert HasRegister("x").check({"x": FakeValue(RegisterType(1, 0))})[0]
        assert not HasRegister("x").check({"x": FakeValue(I32)})[0]

    def test_exact_element_constraints_validate_shaped_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        assert HasI8Element("x").check({"x": FakeValue(vector_type(I8))})[0]
        assert not HasI8Element("x").check({"x": FakeValue(vector_type(I16))})[0]
        assert HasI32Element("x").check({"x": FakeValue(vector_type(I32))})[0]
        assert not HasI32Element("x").check({"x": FakeValue(vector_type(I8))})[0]
        assert HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(F16))})[0]
        assert HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(BF16))})[0]
        assert not HasF16OrBf16Element("x").check({"x": FakeValue(vector_type(F32))})[0]
        assert HasF32Element("x").check({"x": FakeValue(vector_type(F32))})[0]
        assert not HasF32Element("x").check({"x": FakeValue(I32)})[0]

    def test_index_or_non_i1_integer_constraints_validate_types(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        assert HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.INDEX)})[0]
        assert HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(I32)})[0]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.I1)})[0]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(ir.OFFSET)})[
            0
        ]
        assert not HasIndexOrNonI1IntegerScalar("x").check({"x": FakeValue(F32)})[0]
        assert HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.INDEX))}
        )[0]
        assert HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(I8))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.I1))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(ir.OFFSET))}
        )[0]
        assert not HasIndexOrNonI1IntegerElement("x").check(
            {"x": FakeValue(vector_type(F32))}
        )[0]

    def test_bitwise_constraints_accept_integer_and_float_payloads(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        assert HasBitwiseScalar("x").check({"x": FakeValue(ir.INDEX)})[0]
        assert HasBitwiseScalar("x").check({"x": FakeValue(I32)})[0]
        assert HasBitwiseScalar("x").check({"x": FakeValue(F32)})[0]
        assert not HasBitwiseScalar("x").check({"x": FakeValue(ir.I1)})[0]
        assert not HasBitwiseScalar("x").check({"x": FakeValue(ir.OFFSET)})[0]
        assert HasBitwiseElement("x").check({"x": FakeValue(vector_type(I8))})[0]
        assert HasBitwiseElement("x").check({"x": FakeValue(vector_type(F32))})[0]
        assert not HasBitwiseElement("x").check({"x": FakeValue(vector_type(ir.I1))})[0]

    def test_vector_shape_constraints_validate_vector_shape(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        vector_1d = ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))
        vector_2d = ShapedType(TypeKind.VECTOR, F32, (StaticDim(2), StaticDim(2)))
        vector_dynamic = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(),))
        tile_1d = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))

        assert HasRankOneVector("x").check({"x": FakeValue(vector_1d)})[0]
        assert not HasRankOneVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasRankOneVector("x").check({"x": FakeValue(tile_1d)})[0]
        assert HasAllStaticVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasAllStaticVector("x").check({"x": FakeValue(vector_dynamic)})[0]
        assert HasAllStaticRankOneVector("x").check({"x": FakeValue(vector_1d)})[0]
        assert not HasAllStaticRankOneVector("x").check({"x": FakeValue(vector_2d)})[0]
        assert not HasAllStaticRankOneVector("x").check(
            {"x": FakeValue(vector_dynamic)}
        )[0]

    def test_element_width_order_constraints(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType) -> ShapedType:
            return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(4),))

        greater = ElementWidthGreaterThan("result", "input")
        assert greater.name == "ElementWidthGreaterThan"
        assert greater.args == ("result", "input")
        assert greater.check(
            {
                "input": FakeValue(vector_type(F16)),
                "result": FakeValue(vector_type(F32)),
            }
        )[0]
        ok, message = greater.check(
            {
                "input": FakeValue(vector_type(F32)),
                "result": FakeValue(vector_type(F16)),
            }
        )
        assert not ok
        assert "result" in message
        assert "input" in message

        less = ElementWidthLessThan("result", "input")
        assert less.name == "ElementWidthLessThan"
        assert less.args == ("result", "input")
        assert less.check(
            {
                "input": FakeValue(vector_type(I32)),
                "result": FakeValue(vector_type(I8)),
            }
        )[0]
        assert not less.check(
            {
                "input": FakeValue(vector_type(I8)),
                "result": FakeValue(vector_type(I8)),
            }
        )[0]

    def test_element_width_at_least_attr(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = ElementWidthAtLeastAttr("result", "width")
        assert constraint.name == "ElementWidthAtLeastAttr"
        assert constraint.args == ("result", "width")
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 4,
            }
        )[0]
        assert not constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 16,
            }
        )[0]
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "width": 0,
            }
        )[0]

    def test_bit_range_within_element_width(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = BitRangeWithinElementWidth("source", "offset", "width")
        assert constraint.name == "BitRangeWithinElementWidth"
        assert constraint.args == ("source", "offset", "width")
        source = FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),)))
        assert constraint.check({"source": source, "offset": 2, "width": 4})[0]
        assert not constraint.check({"source": source, "offset": -1, "width": 4})[0]
        assert not constraint.check({"source": source, "offset": 2, "width": 0})[0]
        assert not constraint.check({"source": source, "offset": 6, "width": 4})[0]

    def test_positive_bit_width_attr(self) -> None:
        constraint = PositiveBitWidthAttr("width")
        assert constraint.name == "PositiveBitWidthAttr"
        assert constraint.args == ("width",)
        assert constraint.check({"width": 1})[0]
        assert not constraint.check({"width": 0})[0]
        assert not constraint.check({"width": -1})[0]

    def test_attr_matches_element_type(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = AttrMatchesElementType("value", "result")
        assert constraint.name == "AttrMatchesElementType"
        assert constraint.args == ("value", "result")
        assert constraint.check(
            {
                "value": 0.0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check(
            {
                "value": 7,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 128,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": -129,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": True,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check(
            {
                "value": True,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert constraint.check(
            {
                "value": 1,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 2,
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, ir.I1, (StaticDim(4),))
                ),
            }
        )[0]
        assert constraint.check({"value": 4, "result": FakeValue(ir.INDEX)})[0]
        assert constraint.check({"value": 4, "result": FakeValue(ir.OFFSET)})[0]
        assert not constraint.check({"value": True, "result": FakeValue(ir.INDEX)})[0]
        assert not constraint.check({"value": -1, "result": FakeValue(ir.OFFSET)})[0]

    def test_literal_matches_element_type(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = LiteralMatchesElementType("value", "result")
        assert constraint.name == "LiteralMatchesElementType"
        assert constraint.args == ("value", "result")
        assert constraint.check(
            {
                "value": 1.0,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert not constraint.check(
            {
                "value": 1,
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
            }
        )[0]
        assert constraint.check({"value": 127, "result": FakeValue(I8)})[0]
        assert not constraint.check({"value": 128, "result": FakeValue(I8)})[0]
        assert constraint.check({"value": 0, "result": FakeValue(ir.OFFSET)})[0]
        assert not constraint.check({"value": -1, "result": FakeValue(ir.OFFSET)})[0]

    def test_total_bit_count_equal(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = TotalBitCountEqual("input", "result")
        assert constraint.name == "TotalBitCountEqual"
        assert constraint.args == ("input", "result")
        assert constraint.check(
            {
                "input": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(4),))),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I32, (StaticDim(1),))),
            }
        )[0]
        assert constraint.check(
            {
                "input": FakeValue(I32),
                "result": FakeValue(F32),
            }
        )[0]
        assert not constraint.check(
            {
                "input": FakeValue(ShapedType(TypeKind.VECTOR, I8, (StaticDim(3),))),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I16, (StaticDim(1),))),
            }
        )[0]
        assert not constraint.check(
            {
                "input": FakeValue(I16),
                "result": FakeValue(F32),
            }
        )[0]
        assert constraint.check(
            {
                "input": FakeValue(
                    ShapedType(TypeKind.VECTOR, I8, (DynamicDim(), StaticDim(2)))
                ),
                "result": FakeValue(ShapedType(TypeKind.VECTOR, I16, (DynamicDim(),))),
            }
        )[0]

    def test_payload_bit_count_matches_storage_constraints(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(element_type: ScalarType, *sizes: int | None) -> ShapedType:
            return ShapedType(
                TypeKind.VECTOR,
                element_type,
                tuple(
                    DynamicDim() if size is None else StaticDim(size) for size in sizes
                ),
            )

        pack = PackedPayloadBitCountMatchesStorage(
            "source", "width", "result", "result"
        )
        assert pack.name == "PackedPayloadBitCountMatchesStorage"
        assert pack.args == ("source", "width", "result", "result")
        assert pack.check(
            {
                "source": FakeValue(vector_type(I8, 32)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 16)),
            }
        )[0]
        assert not pack.check(
            {
                "source": FakeValue(vector_type(I8, 32)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 15)),
            }
        )[0]
        assert pack.check(
            {
                "source": FakeValue(vector_type(I8, None)),
                "width": 4,
                "result": FakeValue(vector_type(I8, None)),
            }
        )[0]

        unpack = UnpackedPayloadBitCountMatchesStorage(
            "result", "width", "source", "result"
        )
        assert unpack.name == "UnpackedPayloadBitCountMatchesStorage"
        assert unpack.args == ("result", "width", "source", "result")
        assert unpack.check(
            {
                "source": FakeValue(vector_type(I8, 16)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 32)),
            }
        )[0]
        assert not unpack.check(
            {
                "source": FakeValue(vector_type(I8, 16)),
                "width": 4,
                "result": FakeValue(vector_type(I8, 31)),
            }
        )[0]

    def test_offset_count_matches_rank(self) -> None:
        c = OffsetCountMatchesRank("src", "offsets")
        assert c.error is not None
        assert c.error.error_id == "ERR_SUBRANGE_001"

    def test_value_count_matches_static_element_count(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        constraint = ValueCountMatchesStaticElementCount("result", "elements")
        assert constraint.error is not None
        assert constraint.error.error_id == "ERR_STRUCTURE_013"
        assert constraint.check(
            {
                "result": FakeValue(
                    ShapedType(TypeKind.VECTOR, F32, (StaticDim(2), StaticDim(2)))
                ),
                "elements": [object(), object(), object(), object()],
            }
        )[0]
        assert not constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (StaticDim(4),))),
                "elements": [object(), object(), object()],
            }
        )[0]
        assert constraint.check(
            {
                "result": FakeValue(ShapedType(TypeKind.VECTOR, F32, (DynamicDim(),))),
                "elements": [object()],
            }
        )[0]

    def test_dim_index_in_bounds(self) -> None:
        c = DimIndexInBounds("src", "dim")
        assert c.error is not None
        assert c.error.error_id == "ERR_SUBRANGE_002"

    def test_all_shapes_match(self) -> None:
        c = AllShapesMatch("inputs")
        assert c.error is not None
        assert c.error.error_id == "ERR_SHAPE_003"

    def test_last_axis_grouped_by(self) -> None:
        class FakeValue:
            def __init__(self, value_type: object):
                self.type = value_type

        def vector_type(*sizes: int) -> ShapedType:
            return ShapedType(
                TypeKind.VECTOR, F32, tuple(StaticDim(size) for size in sizes)
            )

        constraint = LastAxisGroupedBy("lhs", "result", 4)

        assert constraint.name == "LastAxisGroupedBy"
        assert constraint.args == ("lhs", "result")
        assert constraint.data == 4
        assert constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(2, 4)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 10)),
                "result": FakeValue(vector_type(2, 2)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(3, 4)),
            }
        )[0]
        assert not constraint.check(
            {
                "lhs": FakeValue(vector_type(2, 16)),
                "result": FakeValue(vector_type(2, 5)),
            }
        )[0]

    def test_region_constraints(self) -> None:
        block_arg_count = BlockArgCount("body", "inputs")
        assert block_arg_count.error is not None
        assert block_arg_count.error.error_id == "ERR_STRUCTURE_007"

        block_args_satisfy = BlockArgsSatisfy("body", REGISTER)
        assert block_args_satisfy.error is not None
        assert block_args_satisfy.error.error_id == "ERR_TYPE_014"
        assert block_args_satisfy.data == REGISTER

        block_args_match = BlockArgsMatchElementTypes("body", "inputs")
        assert block_args_match.error is not None
        assert block_args_match.error.error_id == "ERR_TYPE_008"

        block_args_type_match = BlockArgsMatchTypes("body", "inputs")
        assert block_args_type_match.error is not None
        assert block_args_type_match.error.error_id == "ERR_TYPE_013"

        yield_count = YieldCountMatchesResults("body", "results")
        assert yield_count.error is not None
        assert yield_count.error.error_id == "ERR_STRUCTURE_008"

        yield_types = YieldTypesMatchResults("body", "results")
        assert yield_types.error is not None
        assert yield_types.error.error_id == "ERR_TYPE_009"

    # --- Variadic field handling ---

    def test_same_type_variadic_pass(self) -> None:
        """SameType across a variadic list and a scalar field."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32"), FakeValue("f32"), FakeValue("f32")],
                "result": FakeValue("f32"),
            }
        )
        assert ok
        assert msg == ""

    def test_same_type_variadic_mismatch_within_list(self) -> None:
        """SameType detects mismatch within a variadic list."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32"), FakeValue("i32")],
                "result": FakeValue("f32"),
            }
        )
        assert not ok
        assert "inputs[1]" in msg
        assert "inputs[0]" in msg

    def test_same_type_variadic_mismatch_against_scalar(self) -> None:
        """SameType detects mismatch between variadic element and scalar."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeValue("f32")],
                "result": FakeValue("i32"),
            }
        )
        assert not ok
        assert "result" in msg
        assert "inputs[0]" in msg

    def test_same_type_empty_variadic(self) -> None:
        """SameType with empty variadic list passes (nothing to compare)."""

        class FakeValue:
            def __init__(self, t: str):
                self.type = t

        c = SameType("inputs", "result")
        ok, _msg = c.check({"inputs": [], "result": FakeValue("f32")})
        assert ok, "Single value with empty list should pass"

    def test_same_element_type_variadic_pass(self) -> None:
        """SameElementType across variadic and scalar."""

        class FakeTile:
            def __init__(self, dt: str):
                self.dtype = dt

        c = SameElementType("inputs", "result")
        ok, _msg = c.check(
            {
                "inputs": [FakeTile("f32"), FakeTile("f32")],
                "result": FakeTile("f32"),
            }
        )
        assert ok

    def test_same_element_type_variadic_fail(self) -> None:
        """SameElementType detects mismatch in variadic list."""

        class FakeTile:
            def __init__(self, dt: str):
                self.dtype = dt

        c = SameElementType("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeTile("f32"), FakeTile("i32")],
                "result": FakeTile("f32"),
            }
        )
        assert not ok
        assert "inputs[1]" in msg

    def test_same_shape_variadic_pass(self) -> None:
        """SameShape across variadic and scalar."""

        class FakeTile:
            def __init__(self, s: tuple[int, ...]):
                self.shape = s

        c = SameShape("inputs", "result")
        ok, _msg = c.check(
            {
                "inputs": [FakeTile((4, 8)), FakeTile((4, 8))],
                "result": FakeTile((4, 8)),
            }
        )
        assert ok

    def test_same_shape_variadic_fail(self) -> None:
        """SameShape detects mismatch in variadic list."""

        class FakeTile:
            def __init__(self, s: tuple[int, ...]):
                self.shape = s

        c = SameShape("inputs", "result")
        ok, msg = c.check(
            {
                "inputs": [FakeTile((4, 8)), FakeTile((4, 16))],
                "result": FakeTile((4, 8)),
            }
        )
        assert not ok
        assert "inputs[1]" in msg
        assert "inputs[0]" in msg

    def test_ranks_match_variadic_pass(self) -> None:
        """RanksMatch with variadic field."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, _msg = c.check(
            {
                "a": [FakeTile(2), FakeTile(2)],
                "b": FakeTile(2),
            }
        )
        assert ok

    def test_ranks_match_variadic_fail(self) -> None:
        """RanksMatch detects rank mismatch with variadic field."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, msg = c.check(
            {
                "a": [FakeTile(2), FakeTile(3)],
                "b": FakeTile(2),
            }
        )
        assert not ok
        assert "a[1]" in msg

    def test_ranks_match_both_variadic(self) -> None:
        """RanksMatch with both fields variadic."""

        class FakeTile:
            def __init__(self, rank: int):
                self.ndim = rank

        c = RanksMatch("a", "b")
        ok, msg = c.check(
            {
                "a": [FakeTile(2)],
                "b": [FakeTile(3)],
            }
        )
        assert not ok
        assert "a[0]" in msg
        assert "b[0]" in msg


# ============================================================================
# Op group
# ============================================================================


class TestDialect:
    def test_basic(self) -> None:
        g = Dialect("scalar", doc="Scalar ops.")
        assert g.name == "scalar"
        assert g.doc == "Scalar ops."

    def test_with_enums(self) -> None:
        g = Dialect("scalar", enums=[_cmpi_preds])
        assert len(g.enums) == 1
        assert isinstance(g.enums, tuple)

    def test_default_phase(self) -> None:
        g = Dialect("target", default_phase=OpPhase.MODULE_METADATA)
        assert g.default_phase == OpPhase.MODULE_METADATA

    def test_categories(self) -> None:
        structural = OpCategory("structure", doc="Structural ops.")
        compute = OpCategory("compute", doc="Compute ops.")

        g = Dialect(
            "vector",
            categories=[structural, compute],
            default_category=compute,
        )

        assert g.categories == (structural, compute)
        assert g.default_category == compute

    def test_default_category_must_be_declared(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"Dialect 'vector': default_category 'compute' "
                r"is not declared in categories"
            ),
        ):
            Dialect(
                "vector",
                categories=[OpCategory("structure")],
                default_category=OpCategory("compute"),
            )


class TestOpCategory:
    def test_basic(self) -> None:
        category = OpCategory("memory.atomic", doc="Atomic memory ops.")

        assert category.key == "memory.atomic"
        assert category.doc == "Atomic memory ops."

    def test_rejects_empty_key(self) -> None:
        with _raises(ValueError, match="op category key must not be empty"):
            OpCategory("")

    def test_rejects_unstable_key_spelling(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"op category key 'Memory/Atomic' must contain only lowercase "
                r"letters, digits, '.', '_', or '-'"
            ),
        ):
            OpCategory("Memory/Atomic")


# ============================================================================
# Type declaration
# ============================================================================


class TestTypeDef:
    def test_semantic_metadata_defaults_to_ordinary(self) -> None:
        type_def = TypeDef("test.handle")

        assert type_def.semantic == TypeSemantic.ORDINARY
        assert type_def.contracts == ()

    def test_semantic_metadata_is_stored_as_tuples(self) -> None:
        type_def = TypeDef(
            "kernel.async.token",
            semantic=TypeSemantic.CONTROL_TOKEN,
            contracts=[ContractFamily.KERNEL_ASYNC],
        )

        assert type_def.semantic == TypeSemantic.CONTROL_TOKEN
        assert type_def.contracts == (ContractFamily.KERNEL_ASYNC,)

    def test_managed_reference_semantic_is_explicit(self) -> None:
        type_def = TypeDef(
            "test.ref",
            semantic=TypeSemantic.MANAGED_REFERENCE,
        )

        assert type_def.semantic == TypeSemantic.MANAGED_REFERENCE

    def test_descriptor_parameters_support_positional_and_keyed_formats(self) -> None:
        type_def = TypeDef(
            "test.matrix",
            params=[AttrDef("element_type", "type"), AttrDef("rows", "i64")],
            format=[
                Param("element_type"),
                COMMA,
                kw("rows"),
                EQUALS,
                Param("rows"),
            ],
        )

        assert type_def.param("rows") == AttrDef("rows", "i64")
        assert type_def.uses_attribute_parameters

    def test_descriptor_parameters_reject_mixed_storage_schemas(self) -> None:
        from loom.dsl import TypeParam

        with _raises(ValueError, match="cannot be mixed"):
            TypeDef(
                "test.wrapper",
                params=[TypeParam("value"), AttrDef("mode", "i64")],
                format=[TypeOf("value"), COMMA, Param("mode")],
            )

    def test_compact_shape_format_must_match_representation(self) -> None:
        type_def = TypeDef(
            "test.vector",
            ir_kind="vector",
            params=[ShapeParam("dims"), ScalarParam("element_type")],
            format=[ShapeOf("dims"), kw("x"), ScalarOf("element_type")],
        )
        assert type_def.uses_compact_shape_format

        with _raises(ValueError, match="does not match its compact shape"):
            TypeDef(
                "test.vector",
                ir_kind="vector",
                params=[ShapeParam("dims"), ScalarParam("element_type")],
                format=[ScalarOf("element_type"), kw("x"), ShapeOf("dims")],
            )

        with _raises(ValueError, match="requires shape and scalar parameters"):
            TypeDef("test.vector", ir_kind="vector")

    def test_vector_compact_shape_format_rejects_encoding(self) -> None:
        with _raises(ValueError, match="vector representation cannot carry"):
            TypeDef(
                "test.vector",
                ir_kind="vector",
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

    def test_descriptor_parameters_share_family_schema_validation(self) -> None:
        with _raises(ValueError, match="must be a bare ASCII identifier"):
            TypeDef(
                "test.wrapper",
                params=[AttrDef("not-valid", "i64")],
                format=[Param("not-valid")],
            )

        open_wrapper = TypeDef(
            "test.wrapper",
            params=[AttrDef("value", ATTR_TYPE_PARAMETERIZED)],
            format=[Param("value")],
        )
        assert open_wrapper.params[0].parameterized_attr is None

        with _raises(ValueError, match="unsupported kind 'scoped_enum'"):
            TypeDef(
                "test.wrapper",
                params=[AttrDef("scope", "scoped_enum")],
                format=[Param("scope")],
            )

    def test_descriptor_parameters_require_exactly_one_format_reference(self) -> None:
        with _raises(ValueError, match="must appear exactly once"):
            TypeDef(
                "test.wrapper",
                params=[AttrDef("value", "type")],
                format=[Param("value"), COMMA, Param("value")],
            )

    def test_compact_enum_parameter_constructs_registered_python_type(self) -> None:
        class CompactValue:
            def __init__(self, mode: int = 0) -> None:
                self.mode = mode

        mode = EnumDef(
            "Mode",
            [EnumCase("fast", 1), EnumCase("precise", 2)],
        )
        type_def = TypeDef(
            "test.compact",
            ir_kind="encoding",
            python_type=CompactValue,
            params=[AttrDef("mode", "enum", enum_def=mode, optional=True)],
            format=[OptionalGroup([Param("mode")], anchor="mode")],
        )

        assert type_def.uses_inline_enum_parameter
        assert type_def.omits_empty_parameter_list
        assert type_def().mode == 0
        assert type_def(mode="precise").mode == 2
        assert type_def(mode=1).mode == 1

    def test_compact_enum_parameter_rejects_ambiguous_absence(self) -> None:
        class CompactValue:
            pass

        mode = EnumDef(
            "Mode",
            [EnumCase("unknown", 0), EnumCase("fast", 1)],
        )
        with _raises(ValueError, match="reserves value zero for absence"):
            TypeDef(
                "test.compact",
                ir_kind="encoding",
                python_type=CompactValue,
                params=[AttrDef("mode", "enum", enum_def=mode, optional=True)],
                format=[OptionalGroup([Param("mode")], anchor="mode")],
            )

    def test_compact_enum_parameter_requires_closed_single_enum(self) -> None:
        class CompactValue:
            pass

        mode = EnumDef("Mode", [EnumCase("fast", 1)])
        with _raises(ValueError, match="exactly one parameter"):
            TypeDef(
                "test.compact",
                ir_kind="encoding",
                python_type=CompactValue,
                params=[
                    AttrDef("mode", "enum", enum_def=mode),
                    AttrDef("other", "enum", enum_def=mode),
                ],
                format=[Param("mode"), COMMA, Param("other")],
            )
        with _raises(ValueError, match="must be an enum"):
            TypeDef(
                "test.compact",
                ir_kind="encoding",
                python_type=CompactValue,
                params=[AttrDef("mode", "i64")],
                format=[Param("mode")],
            )
        with _raises(ValueError, match="cannot be open"):
            TypeDef(
                "test.compact",
                ir_kind="encoding",
                python_type=CompactValue,
                params=[
                    AttrDef(
                        "mode",
                        "enum",
                        enum_def=mode,
                        optional=True,
                        open_enum=True,
                    )
                ],
                format=[OptionalGroup([Param("mode")], anchor="mode")],
            )

        with _raises(ValueError, match="assembly format omits"):
            TypeDef(
                "test.wrapper",
                params=[AttrDef("value", "type")],
                format=[kw("empty")],
            )

    def test_param_format_requires_descriptor_parameter(self) -> None:
        from loom.dsl import TypeParam

        with _raises(ValueError, match="requires an AttrDef parameter"):
            TypeDef(
                "test.wrapper",
                params=[TypeParam("value")],
                format=[Param("value")],
            )

    def test_format_rejects_unknown_parameter(self) -> None:
        with _raises(ValueError, match="format references unknown parameter"):
            TypeDef(
                "test.wrapper",
                params=[AttrDef("value", "type")],
                format=[Param("missing")],
            )


# ============================================================================
# Op declaration
# ============================================================================


class TestOp:
    def test_basic(self) -> None:
        op = Op(
            "scalar.addi",
            group=_scalar_ops,
            doc="Integer addition.",
            operands=[Operand("lhs", INTEGER), Operand("rhs", INTEGER)],
            results=[Result("result", INTEGER)],
        )
        assert op.name == "scalar.addi"
        assert op.namespace == "scalar"
        assert op.short_name == "addi"
        assert len(op.operands) == 2
        assert len(op.results) == 1

    def test_semantic_metadata(self) -> None:
        op = Op(
            "vector.iota",
            phase=OpPhase.EXECUTABLE,
            contracts=[ContractFamily.VECTOR_COORDINATE],
        )

        assert op.phase == OpPhase.EXECUTABLE
        assert op.contracts == (ContractFamily.VECTOR_COORDINATE,)

    def test_effective_phase_uses_dialect_default(self) -> None:
        op = Op(
            "target.generic",
            group=Dialect("target", default_phase=OpPhase.MODULE_METADATA),
        )

        assert op.effective_phase == OpPhase.MODULE_METADATA

    def test_category_metadata(self) -> None:
        coordinate = OpCategory("coordinate")
        memory = OpCategory("memory")
        dialect = Dialect("vector", categories=[coordinate, memory])
        op = Op("vector.iota", group=dialect, category=coordinate)

        assert op.category == coordinate
        assert op.effective_category == coordinate

    def test_effective_category_uses_dialect_default(self) -> None:
        compute = OpCategory("compute")
        op = Op(
            "vector.addf",
            group=Dialect(
                "vector",
                categories=[compute],
                default_category=compute,
            ),
        )

        assert op.effective_category == compute

    def test_category_must_be_declared_by_dialect(self) -> None:
        with _raises(
            ValueError,
            match=(
                r"Op 'vector\.load': category 'memory' is not declared "
                r"by dialect 'vector'"
            ),
        ):
            Op(
                "vector.load",
                group=Dialect("vector", categories=[OpCategory("compute")]),
                category=OpCategory("memory"),
            )

    def test_repr(self) -> None:
        op = Op("test.op")
        assert repr(op) == "Op('test.op')"

    def test_tuples_stored(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("x", ANY)],
            results=[Result("y", ANY)],
            successors=[Successor("dest")],
            traits=[PURE],
            constraints=[SameType("x", "y")],
            format=[Ref("x"), BlockRef("dest"), COLON, TypeOf("y")],
            legacy_formats=[
                LegacyFormat(
                    "test.op.old",
                    format=[Ref("x"), COLON, TypeOf("y")],
                    replaced_by="loom-source-format-2026-06-09",
                    expires_after="loom-source-format-2026-07-01",
                )
            ],
            examples=["example"],
        )
        assert isinstance(op.operands, tuple)
        assert isinstance(op.results, tuple)
        assert isinstance(op.successors, tuple)
        assert isinstance(op.traits, tuple)
        assert isinstance(op.constraints, tuple)
        assert isinstance(op.format, tuple)
        assert isinstance(op.legacy_formats, tuple)
        assert isinstance(op.examples, tuple)

    def test_legacy_format_metadata(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("input", ANY)],
            results=[Result("result", ANY)],
            attrs=[AttrDef("mode", "string"), AttrDef("policy", "string")],
            format=[Ref("input"), COLON, TypeOf("result")],
            legacy_formats=[
                LegacyFormat(
                    "test.op.mode-attr",
                    format=[
                        Ref("legacy_input"),
                        Attr("legacy_mode"),
                        COLON,
                        TypeOf("result"),
                    ],
                    field_mappings=[
                        LegacyFieldMapping("legacy_input", "input"),
                        LegacyFieldMapping("legacy_mode", "mode"),
                    ],
                    field_defaults=[LegacyFieldDefault("policy", "default")],
                    replaced_by="loom-source-format-2026-06-09",
                    expires_after="loom-source-format-2026-07-01",
                    rewrite_hook="migrate_test_op_mode_attr",
                )
            ],
        )

        legacy_format = op.legacy_formats[0]
        assert legacy_format.rule_id == "test.op.mode-attr"
        assert legacy_format.format[0] == Ref("legacy_input")
        assert legacy_format.field_mappings == (
            LegacyFieldMapping("legacy_input", "input"),
            LegacyFieldMapping("legacy_mode", "mode"),
        )
        assert legacy_format.field_defaults == (
            LegacyFieldDefault("policy", "default"),
        )
        assert legacy_format.expires_after == "loom-source-format-2026-07-01"
        assert legacy_format.rewrite_hook == "migrate_test_op_mode_attr"

    def test_legacy_format_expiration_is_optional(self) -> None:
        op = Op(
            "test.op",
            operands=[Operand("input", ANY)],
            legacy_formats=[
                LegacyFormat(
                    "test.op.old",
                    format=[Ref("input")],
                    replaced_by="loom-source-format-2026-06-09",
                )
            ],
        )

        legacy_format = op.legacy_formats[0]
        assert legacy_format.replaced_by == "loom-source-format-2026-06-09"
        assert legacy_format.expires_after == ""

    def test_duplicate_legacy_format_rule_id_is_rejected(self) -> None:
        with _raises(ValueError, match="duplicate legacy format rule_id"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.old",
                        format=[Ref("input")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    ),
                    LegacyFormat(
                        "test.op.old",
                        format=[Ref("input")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    ),
                ],
            )

    def test_legacy_format_field_validation_catches_typo(self) -> None:
        with _raises(ValueError, match="undeclared current fields"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("missing")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_unused_field_mapping_is_rejected(self) -> None:
        with _raises(ValueError, match="not referenced"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_mappings=[
                            LegacyFieldMapping("old_input", "input"),
                        ],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_duplicate_current_mapping_is_rejected(self) -> None:
        with _raises(ValueError, match="more than one legacy field"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("old_input"), Ref("older_input")],
                        field_mappings=[
                            LegacyFieldMapping("old_input", "input"),
                            LegacyFieldMapping("older_input", "input"),
                        ],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_default_field_must_exist(self) -> None:
        with _raises(ValueError, match="defaults undeclared field"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_defaults=[LegacyFieldDefault("missing", "value")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_legacy_format_default_cannot_replace_parsed_field(self) -> None:
        with _raises(ValueError, match="already parsed"):
            Op(
                "test.op",
                operands=[Operand("input", ANY)],
                legacy_formats=[
                    LegacyFormat(
                        "test.op.bad",
                        format=[Ref("input")],
                        field_defaults=[LegacyFieldDefault("input", "value")],
                        replaced_by="loom-source-format-2026-06-09",
                        expires_after="loom-source-format-2026-07-01",
                    )
                ],
            )

    def test_lookup_operand(self) -> None:
        op = Op(
            "test.op",
            operands=[
                Operand("lhs", INTEGER),
                Operand("rhs", INTEGER),
            ],
        )
        lhs_operand = op.operand("lhs")
        assert lhs_operand is not None
        assert lhs_operand.name == "lhs"
        assert op.operand("missing") is None

    def test_lookup_result(self) -> None:
        op = Op("test.op", results=[Result("out", FLOAT)])
        assert op.result("out") is not None
        assert op.result("missing") is None

    def test_lookup_attr(self) -> None:
        op = Op("test.op", attrs=[AttrDef("axis", "i64")])
        assert op.attr("axis") is not None
        assert op.attr("missing") is None

    def test_lookup_successor(self) -> None:
        op = Op("test.op", successors=[Successor("dest")])
        assert op.successor("dest") is not None
        assert op.successor("missing") is None

    def test_lookup_region(self) -> None:
        op = Op("test.op", regions=[RegionDef("body")])
        assert op.region("body") is not None
        assert op.region("missing") is None

    def test_trait_queries(self) -> None:
        op = Op("test.op", traits=[PURE, COMMUTATIVE, TERMINATOR])
        assert op.is_pure
        assert op.is_commutative
        assert op.is_terminator
        assert op.has_trait("Pure")
        assert not op.has_trait("Idempotent")

    def test_keyed_module_record_contract(self) -> None:
        op = Op(
            "module.record",
            attrs=[AttrDef("key", "string"), AttrDef("payload", "i64")],
            traits=[MODULE_SCOPE, KeyedModuleRecord("key")],
        )
        assert op.keyed_module_record_attr == "key"

    def test_keyed_module_record_requires_attr_only_module_metadata(self) -> None:
        with _raises(ValueError, match="requires the ModuleScope trait"):
            Op(
                "test.record",
                attrs=[AttrDef("key", "string")],
                traits=[KeyedModuleRecord("key")],
            )
        with _raises(ValueError, match="key attr 'key' must be a required string"):
            Op(
                "module.record",
                attrs=[AttrDef("key", "string", optional=True)],
                traits=[MODULE_SCOPE, KeyedModuleRecord("key")],
            )
        with _raises(ValueError, match="must be attr-only"):
            Op(
                "module.record",
                operands=[Operand("input", ANY)],
                attrs=[AttrDef("key", "string")],
                traits=[MODULE_SCOPE, KeyedModuleRecord("key")],
            )

    def test_no_traits(self) -> None:
        op = Op("test.op")
        # An op with no effects, no traits, and no allocating results
        # is derived pure — it does nothing observable.
        assert op.is_pure
        assert not op.is_terminator

    def test_namespace_no_dot(self) -> None:
        op = Op("nodot")
        assert op.namespace == ""
        assert op.short_name == "nodot"

    def test_keyword_only_args(self) -> None:
        """Op constructor requires keyword arguments after name."""
        # This should work:
        Op("test.op", doc="hello")
        # Positional args after name should fail:
        with _raises(TypeError):
            Op("test.op", "hello")  # type: ignore[arg-type, misc]

    def test_format_field_validation_catches_typo(self) -> None:
        """Format referencing undeclared field is caught at declaration time."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("input", ANY)],
                results=[Result("result", ANY)],
                format=[Ref("source"), COLON, TypeOf("result")],  # Wrong name.
            )

    def test_format_field_validation_accepts_successor(self) -> None:
        """BlockRef fields are validated against declared successors."""
        op = Op(
            "test.br",
            successors=[Successor("dest")],
            format=[BlockRef("dest")],
        )
        assert op.successor("dest") is not None

    def test_format_field_validation_descends_into_clause(self) -> None:
        """Clause payload fields are validated like top-level format fields."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("input", ANY)],
                format=[Clause("value", Ref("missing"))],
            )

    def test_format_field_validation_allows_implicit(self) -> None:
        """Implicit fields (iv, args, predicates) are allowed."""
        # Should not raise — "iv" and "args" are implicit.
        Op(
            "test.loop",
            operands=[Operand("lower_bound", INDEX)],
            results=[Result("results", ANY, variadic=True)],
            regions=[RegionDef("body")],
            format=[Ref("iv"), EQUALS, Ref("lower_bound"), Region("body")],
        )

    def test_format_field_validation_index_list(self) -> None:
        """IndexList fields (both dynamic and static) are validated."""
        with _raises(ValueError, match="undeclared fields"):
            Op(
                "test.bad",
                operands=[Operand("source", TILE)],
                results=[Result("result", TILE)],
                format=[
                    Ref("source"),
                    IndexList("offsets", "static_offsets"),  # Neither declared!
                ],
            )

    def test_region_arg_source_must_be_value_field(self) -> None:
        """Region arg_source must name a variadic value or FuncArgs field."""
        with _raises(ValueError, match="arg_source references non-value/non-FuncArgs"):
            Op(
                "test.bad",
                attrs=[AttrDef("types", "string")],
                regions=[RegionDef("body", arg_source="types")],
                format=[Region("body")],
            )

    def test_region_arg_source_validates_without_format(self) -> None:
        """Region arg_source is an op contract, not an assembly-format detail."""
        with _raises(ValueError, match="arg_source references non-value/non-FuncArgs"):
            Op(
                "test.bad",
                regions=[RegionDef("body", arg_source="missing")],
            )

    def test_region_arg_source_must_be_variadic(self) -> None:
        """Region arg_source maps one region arg per source value."""
        with _raises(ValueError, match="must reference a variadic"):
            Op(
                "test.bad",
                operands=[Operand("input", INTEGER)],
                regions=[RegionDef("body", arg_source="input")],
                format=[Ref("input"), Region("body")],
            )

    def test_nested_scope_rejected(self) -> None:
        """Scope(...) is one-level only; nested Scope is a declaration error."""
        with _raises(ValueError, match="nested Scope is not supported"):
            Op(
                "test.bad",
                results=[Result("result", ANY)],
                format=[Scope([Scope([ResultType("result")])])],
            )


# ============================================================================
# Memory effects
# ============================================================================


class TestEffects:
    def test_valid_read_effect(self) -> None:
        op = Op(
            "test.load",
            operands=[Operand("source", POOL)],
            results=[Result("result", ANY)],
            effects=[Reads("source")],
        )
        assert not op.is_pure
        assert op.effects[0].operand == "source"

    def test_valid_write_effect(self) -> None:
        op = Op(
            "test.store",
            operands=[Operand("target", POOL), Operand("data", TILE)],
            effects=[Writes("target")],
        )
        assert not op.is_pure

    def test_pure_with_effects_raises(self) -> None:
        with _raises(ValueError, match="PURE.*effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[PURE],
                effects=[Reads("pool")],
            )

    def test_effect_on_nonexistent_operand_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                operands=[Operand("input", POOL)],
                effects=[Reads("nonexistent")],
            )

    def test_effect_on_non_resource_operand_raises(self) -> None:
        with _raises(ValueError, match="not allowed"):
            Op(
                "test.bad",
                operands=[Operand("value", INTEGER)],
                effects=[Reads("value")],
            )

    def test_unknown_effects_with_pure_raises(self) -> None:
        with _raises(ValueError, match="PURE.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[PURE, UNKNOWN_EFFECTS])

    def test_unknown_effects_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="UNKNOWN_EFFECTS.*explicit"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[UNKNOWN_EFFECTS],
                effects=[Reads("pool")],
            )

    def test_pure_with_non_deterministic_raises(self) -> None:
        with _raises(ValueError, match="PURE.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[PURE, NON_DETERMINISTIC])

    def test_non_deterministic_not_pure(self) -> None:
        op = Op("test.rng", traits=[NON_DETERMINISTIC])
        assert not op.is_pure

    def test_unknown_effects_not_pure(self) -> None:
        op = Op("test.call", traits=[UNKNOWN_EFFECTS])
        assert not op.is_pure

    def test_command_effect_classifies_unknown_effects(self) -> None:
        op = Op("test.command", traits=[UNKNOWN_EFFECTS, COMMAND_EFFECT])
        assert not op.is_pure

    def test_command_effect_requires_observable_effects(self) -> None:
        with _raises(ValueError, match="COMMAND_EFFECT requires"):
            Op("test.bad", traits=[COMMAND_EFFECT])

    def test_constant_like_requires_pure_source_shape(self) -> None:
        with _raises(ValueError, match="CONSTANT_LIKE requires PURE"):
            Op(
                "test.bad",
                results=[Result("result", INTEGER)],
                traits=[CONSTANT_LIKE],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                operands=[Operand("input", INTEGER)],
                results=[Result("result", INTEGER)],
                traits=[PURE, CONSTANT_LIKE],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                results=[Result("result", INTEGER)],
                regions=[RegionDef("body")],
                traits=[PURE, CONSTANT_LIKE],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op("test.bad", traits=[PURE, CONSTANT_LIKE])
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                results=[Result("results", INTEGER, variadic=True)],
                traits=[PURE, CONSTANT_LIKE],
            )

    def test_poison_requires_pure_source_shape(self) -> None:
        with _raises(ValueError, match="POISON requires PURE"):
            Op(
                "test.bad",
                results=[Result("result", INTEGER)],
                traits=[POISON],
            )
        with _raises(ValueError, match="cannot also be CONSTANT_LIKE"):
            Op(
                "test.bad",
                results=[Result("result", INTEGER)],
                traits=[PURE, CONSTANT_LIKE, POISON],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                operands=[Operand("input", INTEGER)],
                results=[Result("result", INTEGER)],
                traits=[PURE, POISON],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                results=[Result("result", INTEGER)],
                regions=[RegionDef("body")],
                traits=[PURE, POISON],
            )
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op("test.bad", traits=[PURE, POISON])
        with _raises(ValueError, match="no operands or regions.*one result"):
            Op(
                "test.bad",
                results=[Result("results", INTEGER, variadic=True)],
                traits=[PURE, POISON],
            )


    def test_no_return_requires_terminator_with_or_without_effects(self) -> None:
        with _raises(ValueError, match="NO_RETURN requires the TERMINATOR trait"):
            Op("test.bad", traits=[NO_RETURN])
        with _raises(ValueError, match="NO_RETURN requires the TERMINATOR trait"):
            Op(
                "test.bad_effect",
                operands=[Operand("pool", POOL)],
                traits=[NO_RETURN],
                effects=[Reads("pool")],
            )

        Op("test.no_return", traits=[TERMINATOR, NO_RETURN, UNKNOWN_EFFECTS])

    def test_allocating_result(self) -> None:
        op = Op(
            "test.alloc",
            results=[Result("pool", POOL, allocates=True)],
        )
        assert not op.is_pure
        assert op.results[0].allocates

    def test_unique_identity_not_pure(self) -> None:
        op = Op("test.handle", traits=[UNIQUE_IDENTITY])
        assert not op.is_pure

    def test_pure_with_unique_identity_raises(self) -> None:
        with _raises(ValueError, match="PURE.*UNIQUE_IDENTITY"):
            Op("test.bad", traits=[PURE, UNIQUE_IDENTITY])

    def test_hint_not_pure(self) -> None:
        op = Op("test.hint", traits=[HINT])
        assert not op.is_pure

    def test_hint_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="HINT.*explicit effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[HINT],
                effects=[Reads("pool")],
            )

    def test_hint_with_pure_raises(self) -> None:
        with _raises(ValueError, match="HINT.*PURE"):
            Op("test.bad", traits=[HINT, PURE])

    def test_hint_with_unknown_effects_raises(self) -> None:
        with _raises(ValueError, match="HINT.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[HINT, UNKNOWN_EFFECTS])

    def test_hint_with_non_deterministic_raises(self) -> None:
        with _raises(ValueError, match="HINT.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[HINT, NON_DETERMINISTIC])

    def test_hint_with_convergent_raises(self) -> None:
        with _raises(ValueError, match="HINT.*CONVERGENT"):
            Op("test.bad", traits=[HINT, CONVERGENT])

    def test_compile_time_only_can_refine_values(self) -> None:
        op = Op(
            "test.assume",
            operands=[Operand("value", ANY)],
            results=[Result("result", ANY)],
            traits=[PURE, FACT_IDENTITY, STORAGE_RELATION, COMPILE_TIME_ONLY],
        )
        assert op.is_pure

    def test_compile_time_only_is_redundant_with_hint(self) -> None:
        with _raises(ValueError, match="COMPILE_TIME_ONLY.*redundant.*HINT"):
            Op("test.hint", traits=[HINT, COMPILE_TIME_ONLY])

    def test_compile_time_only_with_runtime_trait_raises(self) -> None:
        with _raises(ValueError, match="COMPILE_TIME_ONLY.*Convergent"):
            Op("test.bad", traits=[COMPILE_TIME_ONLY, CONVERGENT])

    def test_compile_time_only_result_requires_identity_storage(self) -> None:
        with _raises(
            ValueError,
            match="COMPILE_TIME_ONLY.*FACT_IDENTITY.*STORAGE_RELATION",
        ):
            Op(
                "test.bad",
                operands=[Operand("value", ANY)],
                results=[Result("result", ANY)],
                traits=[PURE, COMPILE_TIME_ONLY],
            )

    def test_compile_time_only_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="COMPILE_TIME_ONLY.*explicit effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[COMPILE_TIME_ONLY],
                effects=[Reads("pool")],
            )

    def test_convergent_can_be_pure(self) -> None:
        op = Op("test.convergent", traits=[PURE, CONVERGENT])
        assert op.is_pure

    def test_memory_fence_can_be_convergent(self) -> None:
        op = Op("test.barrier", traits=[MEMORY_FENCE, CONVERGENT])
        assert not op.is_pure

    def test_memory_fence_conflicts_with_pure(self) -> None:
        with _raises(ValueError, match="PURE.*MEMORY_FENCE"):
            Op("test.bad", traits=[PURE, MEMORY_FENCE])

    def test_memory_fence_conflicts_with_hint(self) -> None:
        with _raises(ValueError, match="HINT.*MEMORY_FENCE"):
            Op("test.bad", traits=[HINT, MEMORY_FENCE])

    def test_memory_fence_conflicts_with_safe_to_speculate(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*MEMORY_FENCE"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, MEMORY_FENCE])

    def test_safe_to_speculate_conflicts_with_hint(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*HINT"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, HINT])


# ============================================================================
# Ownership effects
# ============================================================================


class TestOwnershipEffects:
    def test_valid_operand_and_result_ownership_effects(self) -> None:
        op = Op(
            "test.resource.retain",
            operands=[Operand("resource", POOL)],
            results=[Result("result", POOL)],
            ownership_effects=[
                Borrow("resource", BY_REFERENCE),
                AliasResult("result", "resource"),
            ],
        )
        assert not op.is_pure
        operand_effect = op.ownership_effects[0]
        result_effect = op.ownership_effects[1]
        assert isinstance(operand_effect, OperandOwnershipEffect)
        assert isinstance(result_effect, ResultOwnershipEffect)
        assert operand_effect.operand == "resource"
        assert operand_effect.carrier == BY_REFERENCE
        assert result_effect.source == "resource"

    def test_fresh_result_ownership_effect(self) -> None:
        op = Op(
            "test.resource.alloc",
            results=[Result("result", POOL, allocates=True)],
            ownership_effects=[FreshResult("result")],
        )
        assert not op.is_pure
        result_effect = op.ownership_effects[0]
        assert isinstance(result_effect, ResultOwnershipEffect)
        assert result_effect.result == "result"

    def test_moved_result_ownership_effect(self) -> None:
        op = Op(
            "test.resource.move",
            operands=[Operand("source", POOL)],
            results=[Result("result", POOL)],
            ownership_effects=[MovedResult("result", "source")],
        )
        result_effect = op.ownership_effects[0]
        assert isinstance(result_effect, ResultOwnershipEffect)
        assert result_effect.result == "result"
        assert result_effect.source == "source"

    def test_moved_result_ownership_effect_requires_fixed_fields(self) -> None:
        with _raises(ValueError, match="fixed operand/result"):
            Op(
                "test.resource.move_many",
                operands=[Operand("sources", POOL, variadic=True)],
                results=[Result("results", POOL, variadic=True)],
                ownership_effects=[MovedResult("results", "sources")],
            )

    def test_moved_result_source_may_only_move_once(self) -> None:
        with _raises(ValueError, match="only one moved result"):
            Op(
                "test.resource.move_twice",
                operands=[Operand("source", POOL)],
                results=[Result("lhs", POOL), Result("rhs", POOL)],
                ownership_effects=[
                    MovedResult("lhs", "source"),
                    MovedResult("rhs", "source"),
                ],
            )

    def test_moved_result_source_cannot_have_operand_effect(self) -> None:
        with _raises(ValueError, match="may not also declare"):
            Op(
                "test.resource.bad_move",
                operands=[Operand("source", POOL)],
                results=[Result("result", POOL)],
                ownership_effects=[
                    Consume("source"),
                    MovedResult("result", "source"),
                ],
            )

    def test_alias_result_ownership_effect_requires_fixed_fields(self) -> None:
        with _raises(ValueError, match="fixed operand/result"):
            Op(
                "test.resource.alias_many",
                operands=[Operand("resources", POOL, variadic=True)],
                results=[Result("results", POOL, variadic=True)],
                ownership_effects=[AliasResult("results", "resources")],
            )

    def test_operand_ownership_effect_on_missing_operand_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                operands=[Operand("resource", POOL)],
                ownership_effects=[Consume("missing")],
            )

    def test_pure_with_ownership_effects_raises(self) -> None:
        with _raises(ValueError, match="PURE.*ownership"):
            Op(
                "test.bad",
                operands=[Operand("resource", POOL)],
                traits=[PURE],
                ownership_effects=[Consume("resource")],
            )

    def test_result_ownership_effect_on_missing_result_raises(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                results=[Result("result", POOL)],
                ownership_effects=[FreshResult("missing")],
            )

    def test_alias_result_requires_source_operand(self) -> None:
        with _raises(ValueError, match="not declared"):
            Op(
                "test.bad",
                results=[Result("result", POOL)],
                ownership_effects=[AliasResult("result", "missing")],
            )

    def test_safe_to_speculate_conflicts_with_unknown_effects(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*UNKNOWN_EFFECTS"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, UNKNOWN_EFFECTS])

    def test_safe_to_speculate_conflicts_with_non_deterministic(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*NON_DETERMINISTIC"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, NON_DETERMINISTIC])

    def test_safe_to_speculate_conflicts_with_unique_identity(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*UNIQUE_IDENTITY"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, UNIQUE_IDENTITY])

    def test_safe_to_speculate_conflicts_with_convergent(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*CONVERGENT"):
            Op("test.bad", traits=[SAFE_TO_SPECULATE, CONVERGENT])

    def test_safe_to_speculate_with_explicit_effects_raises(self) -> None:
        with _raises(ValueError, match="SAFE_TO_SPECULATE.*explicit effects"):
            Op(
                "test.bad",
                operands=[Operand("pool", POOL)],
                traits=[SAFE_TO_SPECULATE],
                effects=[Reads("pool")],
            )


# ============================================================================
# Helper functions
# ============================================================================


class TestBinaryOp:
    def test_basic(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Integer addition.",
        )
        assert op.name == "scalar.addi"
        assert len(op.operands) == 2
        assert op.operands[0].name == "lhs"
        assert op.operands[1].name == "rhs"
        assert len(op.results) == 1
        assert op.is_pure
        assert not op.is_commutative

    def test_commutative(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Integer addition.",
            commutative=True,
        )
        assert op.is_commutative

    def test_format(self) -> None:
        op = binary_op(
            "scalar.addi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            doc="Add.",
        )
        assert len(op.format) == 5
        assert isinstance(op.format[0], Ref)
        assert isinstance(op.format[1], Keyword)
        assert isinstance(op.format[2], Ref)
        assert isinstance(op.format[3], Keyword)
        assert isinstance(op.format[4], TypeOf)


class TestUnaryOp:
    def test_basic(self) -> None:
        op = unary_op(
            "scalar.negf",
            group=_scalar_ops,
            type_constraint=FLOAT,
            doc="Negate.",
        )
        assert len(op.operands) == 1
        assert op.operands[0].name == "input"
        assert op.is_pure

    def test_format(self) -> None:
        op = unary_op(
            "scalar.negf",
            group=_scalar_ops,
            type_constraint=FLOAT,
            doc="Negate.",
        )
        assert len(op.format) == 3
        assert isinstance(op.format[0], Ref)
        assert isinstance(op.format[1], Keyword)
        assert isinstance(op.format[2], TypeOf)


class TestCastOp:
    def test_basic(self) -> None:
        op = cast_op(
            "scalar.sitofp",
            group=_scalar_ops,
            from_constraint=INTEGER,
            to_constraint=FLOAT,
            doc="Signed int to float.",
        )
        assert op.operands[0].type_constraint == INTEGER
        assert op.results[0].type_constraint == FLOAT
        assert op.is_pure

    def test_format(self) -> None:
        op = cast_op(
            "scalar.sitofp",
            group=_scalar_ops,
            from_constraint=INTEGER,
            to_constraint=FLOAT,
            doc="Cast.",
        )
        # Expected format: Ref(input) COLON TypeOf(input) kw(to) TypeOf(result)
        assert len(op.format) == 5
        assert op.format[3] == kw("to")


class TestComparisonOp:
    def test_basic(self) -> None:
        op = comparison_op(
            "scalar.cmpi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            predicates=_cmpi_preds,
            doc="Integer comparison.",
        )
        assert len(op.operands) == 2
        assert len(op.attrs) == 1
        assert op.attrs[0].enum_def is _cmpi_preds
        assert op.results[0].type_constraint == I1

    def test_format(self) -> None:
        op = comparison_op(
            "scalar.cmpi",
            group=_scalar_ops,
            type_constraint=INTEGER,
            predicates=_cmpi_preds,
            doc="Compare.",
        )
        # Expected format: Attr(predicate) COMMA Ref(lhs) COMMA Ref(rhs)
        # COLON TypeOf(lhs)
        assert len(op.format) == 7
        assert isinstance(op.format[0], Attr)


class TestScopedEnumOp:
    def test_scoped_enum_requires_one_top_level_reference(self) -> None:
        with _raises(ValueError, match="exactly one ScopedEnumRef"):
            Op("test.packet", attrs=[AttrDef("descriptor", "scoped_enum")])
        op = Op(
            "test.packet",
            attrs=[AttrDef("descriptor", "scoped_enum")],
            format=[ScopedEnumRef("descriptor")],
        )
        assert op.attr("descriptor") is not None

    def test_optional_scoped_enum_requires_matching_optional_group(self) -> None:
        with _raises(ValueError, match="anchored to itself"):
            Op(
                "test.packet",
                attrs=[AttrDef("descriptor", "scoped_enum", optional=True)],
                format=[ScopedEnumRef("descriptor")],
            )
        op = Op(
            "test.packet",
            attrs=[AttrDef("descriptor", "scoped_enum", optional=True)],
            format=[OptionalGroup([ScopedEnumRef("descriptor")], anchor="descriptor")],
        )
        assert op.attr("descriptor") is not None

    def test_op_has_at_most_one_scoped_enum_identity(self) -> None:
        with _raises(ValueError, match="at most one scoped_enum attr"):
            Op(
                "test.packet",
                attrs=[
                    AttrDef("first", "scoped_enum"),
                    AttrDef("second", "scoped_enum"),
                ],
                format=[ScopedEnumRef("first"), ScopedEnumRef("second")],
            )

    def test_scoped_enum_reference_requires_scoped_enum_attr(self) -> None:
        with _raises(ValueError, match="must name a scoped_enum attr"):
            Op(
                "test.packet",
                attrs=[AttrDef("descriptor", "string")],
                format=[ScopedEnumRef("descriptor")],
            )


class TestSymbolReference:
    def test_defaults_to_dependency_role(self) -> None:
        reference = SymbolReference("function", ["callable"])
        assert reference.interfaces == ("callable",)
        assert reference.role is SymbolReferenceRole.DEPENDENCY

    def test_accepts_availability_role(self) -> None:
        reference = SymbolReference(
            "record", ["record"], role=SymbolReferenceRole.AVAILABILITY
        )
        assert reference.role is SymbolReferenceRole.AVAILABILITY

    def test_accepts_unconstrained_symbol_interface(self) -> None:
        reference = SymbolReference("symbol", [], role=SymbolReferenceRole.AVAILABILITY)
        assert reference.interfaces == ()

    def test_rejects_untyped_role(self) -> None:
        with _raises(ValueError, match="role must be a SymbolReferenceRole"):
            SymbolReference("record", ["record"], role="availability")  # type: ignore[arg-type]


class TestSymbolKernelContract:
    def test_callable_interface_requires_func_like_interface(self) -> None:
        with _raises(ValueError, match="requires the func_like interface"):
            SymbolDefinition(
                field="callee",
                name="function",
                interfaces=["callable"],
            )

    def test_func_like_requires_one_signature_representation(self) -> None:
        for func_like in (
            FuncLikeInterface(callee="callee"),
            FuncLikeInterface(callee="callee", body="body", args="args"),
        ):
            with _raises(ValueError, match="exactly one signature representation"):
                Op(
                    "test.function",
                    traits=[SYMBOL_DEFINE],
                    attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                    symbol_def=SymbolDefinition(
                        field="callee",
                        name="function",
                        interfaces=["func_like"],
                    ),
                    interfaces=[func_like],
                )

    def test_func_like_requires_explicit_func_args_format(self) -> None:
        with _raises(ValueError, match="requires an explicit FuncArgs"):
            Op(
                "test.function",
                traits=[SYMBOL_DEFINE],
                attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="function",
                    interfaces=["func_like", "callable"],
                ),
                regions=[RegionDef("body")],
                interfaces=[FuncLikeInterface(callee="callee", body="body")],
                format=[SymbolRef("callee"), Region("body")],
            )

    def test_body_signature_may_have_contiguous_argument_groups(self) -> None:
        op = Op(
            "test.partitioned_function",
            traits=[SYMBOL_DEFINE],
            attrs=[
                AttrDef("callee", ATTR_TYPE_SYMBOL),
                AttrDef("specialization_count", "i64"),
            ],
            symbol_def=SymbolDefinition(
                field="callee",
                name="function",
                interfaces=["func_like", "callable"],
            ),
            regions=[RegionDef("body")],
            interfaces=[FuncLikeInterface(callee="callee", body="body")],
            format=[
                SymbolRef("callee"),
                FuncArgs(
                    "args",
                    group="specializations",
                    end_attr="specialization_count",
                ),
                FuncArgs(
                    "args",
                    group="arguments",
                    start_attr="specialization_count",
                ),
                Region("body"),
            ],
        )

        assert op.attr("specialization_count") is not None

    def test_body_signature_partition_requires_matching_boundaries(self) -> None:
        with _raises(ValueError, match="must use 'specialization_count'"):
            Op(
                "test.partitioned_function",
                traits=[SYMBOL_DEFINE],
                attrs=[
                    AttrDef("callee", ATTR_TYPE_SYMBOL),
                    AttrDef("specialization_count", "i64"),
                    AttrDef("wrong_count", "i64"),
                ],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="function",
                    interfaces=["func_like", "callable"],
                ),
                regions=[RegionDef("body")],
                interfaces=[FuncLikeInterface(callee="callee", body="body")],
                format=[
                    SymbolRef("callee"),
                    FuncArgs(
                        "args",
                        group="specializations",
                        end_attr="specialization_count",
                    ),
                    FuncArgs(
                        "args",
                        group="arguments",
                        start_attr="wrong_count",
                    ),
                    Region("body"),
                ],
            )

    def test_body_signature_partition_requires_i64_boundary(self) -> None:
        with _raises(ValueError, match="must name a required i64 attribute"):
            Op(
                "test.partitioned_function",
                traits=[SYMBOL_DEFINE],
                attrs=[
                    AttrDef("callee", ATTR_TYPE_SYMBOL),
                    AttrDef("specialization_count", "string"),
                ],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="function",
                    interfaces=["func_like", "callable"],
                ),
                regions=[RegionDef("body")],
                interfaces=[FuncLikeInterface(callee="callee", body="body")],
                format=[
                    SymbolRef("callee"),
                    FuncArgs(
                        "args",
                        group="specializations",
                        end_attr="specialization_count",
                    ),
                    FuncArgs(
                        "args",
                        group="arguments",
                        start_attr="specialization_count",
                    ),
                    Region("body"),
                ],
            )

    def test_requires_exactly_one_workload_signature_storage(self) -> None:
        with _raises(ValueError, match="exactly one"):
            SymbolKernelContract()
        with _raises(ValueError, match="exactly one"):
            SymbolKernelContract(workload_region="config", workload_operands="args")

    def test_kernel_interface_requires_contract(self) -> None:
        with _raises(ValueError, match="must be declared together"):
            SymbolDefinition(
                field="callee",
                name="kernel",
                interfaces=["func_like", "kernel"],
            )

    def test_kernel_contract_requires_kernel_interface(self) -> None:
        with _raises(ValueError, match="must be declared together"):
            SymbolDefinition(
                field="callee",
                name="function",
                interfaces=["func_like"],
                kernel_contract=SymbolKernelContract(workload_region="config"),
            )

    def test_kernel_interface_requires_launch_abi(self) -> None:
        with _raises(ValueError, match="requires the func_like interface"):
            SymbolDefinition(
                field="callee",
                name="kernel",
                interfaces=["kernel"],
                kernel_contract=SymbolKernelContract(workload_region="config"),
            )

    def test_kernel_entry_interface_requires_device_abi(self) -> None:
        with _raises(ValueError, match="requires the func_like interface"):
            SymbolDefinition(
                field="callee",
                name="kernel entry",
                interfaces=["kernel_entry"],
            )

    def test_operand_backed_signature_owns_every_operand(self) -> None:
        with _raises(ValueError, match="must own every operand field"):
            Op(
                "test.decl",
                traits=[SYMBOL_DEFINE],
                operands=[
                    Operand("args", ANY, variadic=True),
                    Operand("ordinary_use", ANY),
                ],
                attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="function",
                    interfaces=["func_like"],
                ),
                interfaces=[FuncLikeInterface(callee="callee", args="args")],
                format=[FuncArgs("args")],
            )

    def test_kernel_contract_requires_func_like_implementation(self) -> None:
        with _raises(ValueError, match="requires a FuncLikeInterface"):
            Op(
                "test.kernel",
                traits=[SYMBOL_DEFINE],
                attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="kernel",
                    interfaces=["func_like", "kernel"],
                    kernel_contract=SymbolKernelContract(workload_region="config"),
                ),
                regions=[RegionDef("config")],
            )

    def test_kernel_signatures_require_distinct_operand_fields(self) -> None:
        with _raises(ValueError, match="distinct operand fields"):
            Op(
                "test.kernel_decl",
                traits=[SYMBOL_DEFINE],
                operands=[Operand("args", ANY, variadic=True)],
                attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="kernel",
                    interfaces=["func_like", "kernel"],
                    kernel_contract=SymbolKernelContract(workload_operands="args"),
                ),
                interfaces=[FuncLikeInterface(callee="callee", args="args")],
                format=[FuncArgs("args")],
            )

    def test_kernel_signatures_require_distinct_regions(self) -> None:
        with _raises(ValueError, match="distinct regions"):
            Op(
                "test.kernel",
                traits=[SYMBOL_DEFINE],
                attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
                symbol_def=SymbolDefinition(
                    field="callee",
                    name="kernel",
                    interfaces=["func_like", "kernel"],
                    kernel_contract=SymbolKernelContract(workload_region="body"),
                ),
                interfaces=[FuncLikeInterface(callee="callee", body="body")],
                regions=[RegionDef("body")],
                format=[FuncArgs("args")],
            )
