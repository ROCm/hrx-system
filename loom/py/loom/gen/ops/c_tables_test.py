# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.assembly import (
    COLON,
    COMMA,
    AlignedRefs,
    Attr,
    AttrDict,
    AttrParams,
    AttrTable,
    BlockRef,
    Clause,
    EncodingOf,
    Flags,
    FuncArgs,
    OperandDict,
    OptionalGroup,
    Param,
    PredicateList,
    Ref,
    Region,
    ResultType,
    ResultTypeList,
    ScalarOf,
    ScopedEnumRef,
    ShapeOf,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
    TypesOf,
    kw,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_BYTES,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_ENUM_ARRAY,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_PARAMETERIZED,
    ATTR_TYPE_PARAMETERIZED_ARRAY,
    ATTR_TYPE_PREDICATE_LIST,
    ATTR_TYPE_STRING,
    ATTR_TYPE_SYMBOL,
    ATTR_TYPE_SYMBOL_ARRAY,
    ATTR_TYPE_SYMBOL_SET,
    CONSTANT_LIKE,
    DECOMPOSABLE,
    ELEMENTWISE,
    HINT,
    INTEGER,
    MODULE_SCOPE,
    POOL,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    VECTOR,
    AliasResult,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    Borrow,
    CallLikeInterface,
    CallLikeKind,
    ConditionRefinement,
    ConditionRefinementTruth,
    Constraint,
    ContractFamily,
    Dialect,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    EncodingAliasDef,
    EncodingFamilyDef,
    EncodingFamilyRole,
    EncodingOperandSummaryDef,
    EncodingParam,
    EncodingRecordDef,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    HasParent,
    IterArgsMatchResults,
    KeyedModuleRecord,
    LiteralMatchesElementType,
    LoopLikeInterface,
    MemoryAccessInterface,
    MovedResult,
    Op,
    OpCategory,
    Operand,
    OperandRole,
    OpPhase,
    PackedPayloadBitCountMatchesStorage,
    ParameterizedAttrDef,
    PositiveBitWidthAttr,
    Reads,
    ReadWrites,
    RegionDef,
    Result,
    Retain,
    RetainedResult,
    SameShape,
    SameType,
    ScalarParam,
    ShapeParam,
    Successor,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolKernelContract,
    SymbolReference,
    SymbolReferenceRole,
    SymbolValueContract,
    TargetFactSpecialization,
    TargetLikeInterface,
    TotalBitCountEqual,
    TypeConstraint,
    TypeDef,
    TypeSemantic,
    UnpackedPayloadBitCountMatchesStorage,
    Writes,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
)
from loom.gen.ops import c_traits
from loom.gen.ops import model as c_table_model
from loom.gen.ops.c_builders import generate_builders_c
from loom.gen.ops.c_enums import TYPE_CONSTRAINT_MAP
from loom.gen.ops.c_location_tags import generate_location_tag_table_inc
from loom.gen.ops.c_registry import generate_op_registry
from loom.gen.ops.c_scalar_types import generate_scalar_type_table_inc
from loom.gen.ops.c_tables import (
    checked_in_file_set,
    generate_dialect_type_registry,
    generate_ops_h,
    generate_sharded_tables_c,
    generate_tables_aggregator_c,
    generate_tables_c,
    generate_type_registry,
)
from loom.location_tag import BuiltinLocationTag
from loom.scalar_type import SCALAR_TYPE_NONE, ScalarTypeKind


@contextmanager
def _raises_value_error(pattern: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!s} did not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected ValueError matching {pattern!r}")


def _compact_tensor_type_def(name: str) -> TypeDef:
    return TypeDef(
        name=name,
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


def test_constant_like_ops_derive_safe_to_speculate() -> None:
    constant = Op(
        "test.constant",
        results=[Result("result", INTEGER)],
        traits=[PURE, CONSTANT_LIKE],
    )

    assert c_traits.trait_flags(constant) == ("LOOM_TRAIT_PURE | LOOM_TRAIT_CONSTANT_LIKE | LOOM_TRAIT_SAFE_TO_SPECULATE")


def test_load_dialect_generation_calls_only_requested_loader() -> None:
    calls: list[str] = []
    expected = c_table_model.DialectGeneration(dialect=object(), ops=[], table_shards=None)
    other = c_table_model.DialectGeneration(dialect=object(), ops=[], table_shards=None)

    def load_other() -> c_table_model.DialectGeneration:
        calls.append("other")
        return other

    def load_wanted() -> c_table_model.DialectGeneration:
        calls.append("wanted")
        return expected

    original_loaders = c_table_model._DIALECT_GENERATION_LOADERS
    try:
        c_table_model._DIALECT_GENERATION_LOADERS = (
            ("other", load_other),
            ("wanted", load_wanted),
        )
        actual = c_table_model.load_dialect_generation("wanted")
    finally:
        c_table_model._DIALECT_GENERATION_LOADERS = original_loaders

    assert actual is expected
    assert calls == ["wanted"]


def test_checked_in_file_set_separates_public_artifacts_from_build_outputs() -> None:
    dialect = Dialect("artifact_test", dialect_id=0x7D)
    build_generated_dialect = Dialect(
        "build_generated_test",
        dialect_id=0x7C,
        checked_in_headers=False,
    )
    model = c_table_model.GenerationModel(
        dialects=[
            c_table_model.DialectGeneration(
                dialect=dialect,
                ops=[],
                table_shards=None,
            ),
            c_table_model.DialectGeneration(
                dialect=build_generated_dialect,
                ops=[],
                table_shards=None,
            ),
        ],
        types=[],
    )

    generated_file_set = checked_in_file_set(model)

    assert "loom/src/loom/ops/artifact_test/ops.h" in generated_file_set.output_paths
    assert "loom/src/loom/ops/build_generated_test/ops.h" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry.h" in generated_file_set.output_paths
    assert "loom/src/loom/ir/scalar_type_table.inc" in generated_file_set.output_paths
    assert "loom/src/loom/ops/artifact_test/builders.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/artifact_test/tables.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry_tables.c" in generated_file_set.obsolete_paths
    assert "loom/src/loom/ops/op_registry_tables.h" in generated_file_set.obsolete_paths
    assert all(not path.endswith(".c") for path in generated_file_set.output_paths)
    assert set(generated_file_set.output_paths).isdisjoint(generated_file_set.obsolete_paths)


def test_type_constraint_map_covers_every_constraint() -> None:
    assert set(TYPE_CONSTRAINT_MAP) == set(TypeConstraint)


def test_generate_scalar_type_table_uses_length_partitioned_classification() -> None:
    generated = generate_scalar_type_table_inc()

    assert SCALAR_TYPE_NONE == 0
    assert "loom_scalar_type_names[LOOM_SCALAR_TYPE_COUNT_]" in generated
    assert f"LOOM_SCALAR_TYPE_NONE == {SCALAR_TYPE_NONE}" in generated
    assert "return LOOM_SCALAR_TYPE_NONE;" in generated
    assert "switch (name.size)" in generated
    assert "iree_string_view_equal" in generated
    assert "for (" not in generated
    assert generated.count("ordinal does not match Python") == len(ScalarTypeKind)


def test_generate_location_tag_table_preserves_open_enum_boundary() -> None:
    generated = generate_location_tag_table_inc()

    assert "switch (tag)" in generated
    assert "switch (name.size)" in generated
    assert "LOOM_LOCATION_TAG_USER_BASE" in generated
    assert "iree_string_view_equal" in generated
    assert generated.count("value does not match Python") == len(BuiltinLocationTag)


def test_generate_type_registry_emits_fact_domain_pointer() -> None:
    type_def = TypeDef(
        name="test.handle",
        fact_domain="loom_test_handle_fact_domain",
    )

    type_registry_h, type_registry_tables_h, type_registry_tables_c = generate_type_registry([type_def])

    assert "loom_type_registry_configure_fact_context" in type_registry_h
    assert "loom_type_registry_entries_storage" in type_registry_tables_h
    assert "void* user_data, const loom_fact_context_t* context," in type_registry_h
    assert "extern const loom_value_fact_domain_t loom_test_handle_fact_domain;" in type_registry_tables_c
    assert ".fact_domain = &loom_test_handle_fact_domain," in type_registry_tables_c
    assert "loom_value_fact_type_domain_resolver_callback_make" not in type_registry_tables_c


def test_generate_type_registry_omits_zero_default_descriptor_fields() -> None:
    type_def = TypeDef(name="test.handle")

    _, _, type_registry_tables_c = generate_type_registry([type_def])

    assert ".param_count = 0," not in type_registry_tables_c
    assert ".fact_domain = NULL," not in type_registry_tables_c
    assert ".semantic = LOOM_TYPE_SEMANTIC_ORDINARY," not in type_registry_tables_c
    assert ".contract_families = 0," not in type_registry_tables_c
    assert ".format_elements = NULL," not in type_registry_tables_c
    assert ".format_element_count = 0," not in type_registry_tables_c


def test_generate_type_registry_emits_type_semantics() -> None:
    type_def = TypeDef(
        name="test.token",
        semantic=TypeSemantic.CONTROL_TOKEN,
        contracts=[ContractFamily.KERNEL_ASYNC],
    )

    type_registry_h, _, type_registry_tables_c = generate_type_registry([type_def])

    assert '#include "loom/ir/type_descriptor.h"' in type_registry_h
    assert ".semantic = LOOM_TYPE_SEMANTIC_CONTROL_TOKEN," in type_registry_tables_c
    assert ".contract_families = LOOM_CONTRACT_KERNEL_ASYNC," in type_registry_tables_c

    reference_type = TypeDef(
        name="test.ref",
        semantic=TypeSemantic.MANAGED_REFERENCE,
    )
    _, _, reference_tables_c = generate_type_registry([reference_type])
    assert ".semantic = LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE," in reference_tables_c


def test_generate_dialect_type_registry_emits_owned_shard() -> None:
    dialect = Dialect(
        "target_test",
        dialect_id=0x1E,
        c_path="target/arch/test/ops",
        register_by_default=False,
    )
    type_def = TypeDef(name="target_test.handle")

    types_h, types_c = generate_dialect_type_registry(dialect, [type_def])

    assert '#include "loom/ops/type_registry.h"' in types_h
    assert "loom_target_test_type_registry_entries[]" in types_h
    assert "loom_target_test_type_registry_entry_count" in types_h
    assert '#include "loom/target/arch/test/ops/types.h"' in types_c
    assert 'IREE_SVL("target_test.handle")' in types_c


def test_generate_type_registry_emits_builtin_type_descriptor_table() -> None:
    type_def = _compact_tensor_type_def("test.tensor")

    _, _, type_registry_tables_c = generate_type_registry([type_def])

    assert "loom_type_registry_builtin_descriptors[LOOM_TYPE_COUNT_]" in type_registry_tables_c
    assert "[LOOM_TYPE_TENSOR] = &loom_type_test_tensor_descriptor," in type_registry_tables_c


def test_generate_type_registry_emits_compact_enum_descriptor() -> None:
    class CompactValue:
        def __init__(self, mode: int = 0) -> None:
            self.mode = mode

    mode = EnumDef(
        "Mode",
        [EnumCase("fast", 1), EnumCase("precise", 2)],
        c_type="loom_mode_t",
        c_const_prefix="LOOM_MODE",
        c_include="loom/mode.h",
    )
    type_def = TypeDef(
        name="test.compact",
        ir_kind="encoding",
        python_type=CompactValue,
        params=[AttrDef("mode", ATTR_TYPE_ENUM, enum_def=mode, optional=True)],
        format=[OptionalGroup([Param("mode")], anchor="mode")],
    )

    type_registry_h, _, type_registry_tables_c = generate_type_registry([type_def])

    assert "loom_test_compact_type_mode_name" in type_registry_h
    assert "loom_test_compact_type_mode_parse" in type_registry_h
    assert "loom_test_compact_type_make" not in type_registry_h
    assert ".ir_kind = LOOM_TYPE_ENCODING," in type_registry_tables_c
    assert ".type_flags = LOOM_TYPE_FLAG_INLINE_DIMS," in type_registry_tables_c
    assert ".flags = LOOM_PARAMETERIZED_TYPE_OMIT_EMPTY_PARAMETER_LIST," in type_registry_tables_c
    assert "[LOOM_TYPE_ENCODING] = &loom_type_test_compact_descriptor," in type_registry_tables_c


def test_generate_type_registry_emits_signed_enum_set_parameter() -> None:
    feature = EnumDef(
        "Feature",
        [EnumCase("low", 1), EnumCase("high", 255)],
    )
    type_def = TypeDef(
        name="test.featured",
        params=[AttrDef("features", "signed_enum_set", enum_def=feature)],
        format=[Param("features")],
    )

    type_registry_h, _, type_registry_tables_c = generate_type_registry([type_def])

    assert "loom_signed_enum_set_t features" in type_registry_h
    assert "loom_attr_as_signed_enum_set(" in type_registry_h
    assert "LOOM_ATTR_SIGNED_ENUM_SET" in type_registry_tables_c
    assert ".enum_case_names" in type_registry_tables_c


def test_generate_type_registry_deduplicates_external_enum_include() -> None:
    mode = EnumDef(
        "Mode",
        [EnumCase("fast", 1)],
        c_type="loom_mode_t",
        c_const_prefix="LOOM_MODE",
        c_include="loom/ir/types.h",
    )
    type_def = TypeDef(
        name="test.mode",
        params=[AttrDef("mode", ATTR_TYPE_ENUM, enum_def=mode)],
        format=[Param("mode")],
    )

    type_registry_h, _, _ = generate_type_registry([type_def])

    assert type_registry_h.count('#include "loom/ir/types.h"') == 1


def test_generate_type_registry_rejects_duplicate_builtin_type_names() -> None:
    type_defs = [
        _compact_tensor_type_def("test.tensor"),
        _compact_tensor_type_def("test.other_tensor"),
    ]

    with _raises_value_error(r"Type kind 'tensor' has duplicate registry names"):
        generate_type_registry(type_defs)


def test_generate_type_registry_rejects_invalid_fact_domain_symbol() -> None:
    type_def = TypeDef(
        name="test.handle",
        fact_domain="loom.test.handle.fact_domain",
    )

    with _raises_value_error(r"TypeDef 'test\.handle': fact_domain must be a C symbol name"):
        generate_type_registry([type_def])


def test_generate_op_registry_emits_registration_tables() -> None:
    dialect = Dialect(
        "test",
        dialect_id=0x01,
        default_phase=OpPhase.EXECUTABLE,
    )

    op_registry_h, op_registry_tables_h, op_registry_tables_c = generate_op_registry([(dialect, [], ())])

    assert "loom_op_registry_register_all_dialects" in op_registry_h
    assert "loom_op_registry_dialects[]" in op_registry_tables_h
    assert "loom_test_dialect_vtables" in op_registry_tables_c
    assert "loom_op_registry_register_dialect" not in op_registry_tables_c
    assert "iree_make_status" not in op_registry_tables_c


def test_generate_op_registry_wires_condition_refinement_tables() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    op = Op(
        "test.is_positive",
        group=dialect,
        operands=[Operand("value", INTEGER)],
        results=[Result("matches", TypeConstraint.I1)],
        condition_refinement=ConditionRefinement(
            source="value",
            truth=ConditionRefinementTruth.TRUE,
            materialize="loom_test_is_positive_materialize",
        ),
    )

    _, tables_h, tables_c = generate_op_registry([(dialect, [op], ())])

    assert "loom_op_registry_condition_refinements_fn_t" in tables_h
    assert "condition_refinements_fn" in tables_h
    assert "loom_test_dialect_condition_refinements" in tables_c


def test_generate_parameterized_attribute_family_metadata() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    scope = EnumDef(
        "Scope",
        [EnumCase("workgroup", 1), EnumCase("subgroup", 2)],
    )
    tile = ParameterizedAttrDef(
        "test.tile",
        group=dialect,
        parameters=[AttrDef("width", ATTR_TYPE_I64)],
        primary_parameter="width",
        target_condition="loom_test_tile_condition",
    )
    options = ParameterizedAttrDef(
        "test.options",
        group=dialect,
        parameters=[
            AttrDef(
                "scopes",
                ATTR_TYPE_ENUM_ARRAY,
                enum_def=scope,
                optional=True,
                open_enum=True,
            ),
            AttrDef(
                "tile",
                ATTR_TYPE_PARAMETERIZED,
                optional=True,
                parameterized_attr=tile,
            ),
        ],
    )
    holder = Op(
        "test.holder",
        group=dialect,
        attrs=[AttrDef("payload", ATTR_TYPE_PARAMETERIZED)],
        format=[Attr("payload")],
    )

    ops_h = generate_ops_h("test", 0x01, [holder], [tile, options])
    builders_c = generate_builders_c("test", [holder], [tile, options])
    tables_c = generate_tables_c("test", 0x01, [holder], [tile, options])

    assert "LOOM_PARAMETERIZED_ATTR_TEST_TILE" in ops_h
    assert "extern const loom_target_condition_descriptor_t loom_test_tile_condition;" in ops_h
    assert "loom_test_tile_attr_make(" in ops_h
    assert "loom_test_options_attr_make(" in ops_h
    assert "LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_SCOPES" in ops_h
    assert "loom_test_options_attr_has_scopes(" in ops_h
    assert "loom_test_options_attr_scopes(" in ops_h
    assert "loom_test_dialect_parameterized_attrs" in ops_h
    assert "loom_module_make_parameterized_attr(" in builders_c
    assert "slots[0] = loom_attr_i64(width);" in builders_c
    assert "slots[1] = tile;" in builders_c
    assert '.name = _BSTRING(12, "test.options")' in tables_c
    assert ".target_condition = &loom_test_tile_condition" in tables_c
    assert ".primary_parameter_index = 0" in tables_c
    assert tables_c.count(".primary_parameter_index = LOOM_PARAMETERIZED_ATTR_NO_PRIMARY_PARAMETER") == 1
    assert ".attr_kind = LOOM_ATTR_ENUM_ARRAY" in tables_c
    assert ".flags = LOOM_ATTR_OPTIONAL | LOOM_ATTR_OPEN_ENUM" in tables_c
    assert ".attr_kind = LOOM_ATTR_PARAMETERIZED" in tables_c
    assert ".reference.parameterized_attr_kind = LOOM_PARAMETERIZED_ATTR_TEST_TILE" in tables_c
    assert ".reference.parameterized_attr_kind = LOOM_PARAMETERIZED_ATTR_KIND_ANY" in tables_c


def test_generate_symbol_reference_roles_for_all_descriptor_owners() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    dependency = SymbolReference("record", ["record"])
    availability = SymbolReference("symbol", [], role=SymbolReferenceRole.AVAILABILITY)
    options = ParameterizedAttrDef(
        "test.options",
        group=dialect,
        parameters=[AttrDef("target", ATTR_TYPE_SYMBOL, symbol_ref=availability)],
    )
    holder = Op(
        "test.holder",
        group=dialect,
        attrs=[
            AttrDef("dependency", ATTR_TYPE_SYMBOL, symbol_ref=dependency),
            AttrDef("available", ATTR_TYPE_SYMBOL, symbol_ref=availability),
        ],
    )
    handle = TypeDef(
        "test.handle",
        params=[AttrDef("target", ATTR_TYPE_SYMBOL, symbol_ref=availability)],
        format=[Param("target")],
    )

    tables_c = generate_tables_c("test", 0x01, [holder], [options])
    _, _, type_tables_c = generate_type_registry([handle])

    assert tables_c.count(".role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY,") == 1
    assert tables_c.count(".role = LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY,") == 2
    assert ".role = LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY," in type_tables_c
    assert tables_c.count(".interfaces = 0,") == 2
    assert ".interfaces = 0," in type_tables_c


def test_generate_encoding_family_metadata() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    rounding = EnumDef(
        "RoundingPolicy",
        [EnumCase("none", 0), EnumCase("nearest_even", 1)],
        doc="Encoded value rounding policy.",
    )
    auxiliary_keys = EnumDef(
        "AuxiliaryKey",
        [EnumCase("scale", 0), EnumCase("minimum", 3)],
    )
    family = EncodingFamilyDef(
        "operand",
        group=dialect,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=[
            AttrDef("rounding", ATTR_TYPE_ENUM, enum_def=rounding, optional=True),
            AttrDef("payload_elements", ATTR_TYPE_I64),
        ],
        aliases=[
            EncodingAliasDef(
                "encoding.scalar_nearest",
                fixed_parameters={"rounding": "nearest_even"},
                default_parameters={"payload_elements": 1},
            )
        ],
        dynamic_parameters=[
            Operand("matrix", TypeConstraint.VECTOR),
            Operand("seed", TypeConstraint.INDEX),
        ],
        fixed_record=EncodingRecordDef(
            logical_element_count=256,
            storage_byte_count=144,
            required_alignment=16,
        ),
        fixed_operand_summary=EncodingOperandSummaryDef(
            element_format=0x10000,
            payload_packing=0x2,
            zero_scale_fallback=True,
            sparsity_group_nonzero_element_count=2,
            sparsity_group_element_count=4,
            payload_register_count=8,
            payload_element_count=256,
            scale_group_shape=(16, 16),
            scale_operand_count=2,
        ),
        auxiliary_key_enum=auxiliary_keys,
        required_auxiliary_keys=[
            auxiliary_keys.case("scale"),
            auxiliary_keys.case("minimum"),
        ],
    )

    ops_h = generate_ops_h("encoding", 0x09, [], (), [family])
    tables_c = generate_tables_c("encoding", 0x09, [], (), [family])

    assert '#include "loom/ir/encoding.h"' in ops_h
    assert "typedef enum loom_encoding_rounding_policy_e" in ops_h
    assert "LOOM_ENCODING_ROUNDING_POLICY_NEAREST_EVEN = 1" in ops_h
    assert "typedef enum loom_encoding_auxiliary_key_e" in ops_h
    assert "LOOM_ENCODING_AUXILIARY_KEY_SCALE = 0" in ops_h
    assert "LOOM_ENCODING_AUXILIARY_KEY_MINIMUM = 3" in ops_h
    assert "loom_encoding_auxiliary_key_descriptors[LOOM_ENCODING_AUXILIARY_KEY_COUNT_]" in ops_h
    assert "LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_ELEMENTS = 0" in ops_h
    assert "LOOM_ENCODING_OPERAND_PARAMETER_ROUNDING = 1" in ops_h
    assert "LOOM_ENCODING_OPERAND_PARAMETER_COUNT_ = 2" in ops_h
    assert "LOOM_ENCODING_OPERAND_DYNAMIC_PARAMETER_MATRIX = 0" in ops_h
    assert "LOOM_ENCODING_OPERAND_DYNAMIC_PARAMETER_SEED = 1" in ops_h
    assert "LOOM_ENCODING_OPERAND_DYNAMIC_PARAMETER_COUNT_ = 2" in ops_h
    assert "LOOM_ENCODING_FAMILY_DYNAMIC_PARAMETER_COUNT_MAX_ = 2" in ops_h
    assert "extern const loom_encoding_family_descriptor_t loom_encoding_operand_family_descriptor;" in ops_h
    assert '.name = _BSTRING(7, "operand")' in tables_c
    assert ".role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA" in tables_c
    assert '.name = _BSTRING(16, "payload_elements")' in tables_c
    assert ".attr_kind = LOOM_ATTR_I64" in tables_c
    assert '.name = _BSTRING(8, "rounding")' in tables_c
    assert ".attr_kind = LOOM_ATTR_ENUM" in tables_c
    assert ".flags = LOOM_ATTR_OPTIONAL" in tables_c
    assert ".parameter_count = IREE_ARRAYSIZE(loom_encoding_operand_parameter_desc)" in tables_c
    assert "loom_encoding_operand_alias_0_parameters[]" in tables_c
    assert ".parameter_index = 0" in tables_c
    assert ".value = {.kind = LOOM_ATTR_I64, .i64 = INT64_C(1)}" in tables_c
    assert ".parameter_index = 1" in tables_c
    assert ".flags = LOOM_ENCODING_ALIAS_PARAMETER_FIXED" in tables_c
    assert ".value = {.kind = LOOM_ATTR_ENUM, .raw = 1}" in tables_c
    assert '.name = _BSTRING(23, "encoding.scalar_nearest")' in tables_c
    assert ".alias_count = IREE_ARRAYSIZE(loom_encoding_operand_aliases)" in tables_c
    assert ".aliases = loom_encoding_operand_aliases" in tables_c
    assert "static const uint8_t loom_encoding_operand_alias_ordinals[2]" in tables_c
    assert "[1] = 1" in tables_c
    assert ".alias_discriminator_parameter_index = 1" in tables_c
    assert ".alias_ordinals_by_discriminator = loom_encoding_operand_alias_ordinals" in tables_c
    assert "static const loom_encoding_dynamic_parameter_descriptor_t loom_encoding_operand_dynamic_parameter_desc[]" in tables_c
    assert ".type_constraint = LOOM_TYPE_CONSTRAINT_VECTOR" in tables_c
    assert ".type_constraint = LOOM_TYPE_CONSTRAINT_INDEX" in tables_c
    assert ".dynamic_parameter_count = IREE_ARRAYSIZE(loom_encoding_operand_dynamic_parameter_desc)" in tables_c
    assert "static const loom_encoding_family_fixed_metadata_t loom_encoding_operand_fixed_metadata" in tables_c
    assert ".element_format = UINT64_C(0x10000)" in tables_c
    assert ".payload_packing = UINT32_C(0x2)" in tables_c
    assert ".flags = UINT32_C(0x1)" in tables_c
    assert ".nonzero_element_count = 2" in tables_c
    assert ".payload_register_count = 8" in tables_c
    assert ".payload_element_count = 256" in tables_c
    assert ".element_count = 256" in tables_c
    assert ".shape = {16, 16}" in tables_c
    assert ".scale_operand_count = 2" in tables_c
    assert ".required_auxiliary_keys = UINT64_C(0x9)" in tables_c
    assert "loom_encoding_auxiliary_key_descriptors[LOOM_ENCODING_AUXILIARY_KEY_COUNT_]" in tables_c
    assert '.name = _BSTRING(5, "scale")' in tables_c
    assert ".stable_id = UINT64_C(0x6aacb9fbb71a1d91)" in tables_c
    assert '.name = _BSTRING(7, "minimum")' in tables_c
    assert ".logical_element_count = 256" in tables_c
    assert ".storage_byte_count = 144" in tables_c
    assert ".required_alignment = 16" in tables_c
    assert ".fixed_metadata = &loom_encoding_operand_fixed_metadata" in tables_c


def test_generate_implicit_shaped_attachment_metadata() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    family = EncodingFamilyDef(
        "encoding.layout.dense",
        group=dialect,
        role=EncodingFamilyRole.ADDRESS_LAYOUT,
        implicit_shaped_attachment=True,
    )

    tables_c = generate_tables_c("encoding", 0x09, [], encoding_families=[family])

    assert ".family_flags = LOOM_ENCODING_FAMILY_IMPLICIT_SHAPED_ATTACHMENT" in tables_c


def test_encoding_families_require_one_auxiliary_key_vocabulary() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    first_keys = EnumDef("FirstKeys", [EnumCase("scale", 0)])
    second_keys = EnumDef("SecondKeys", [EnumCase("minimum", 0)])
    families = [
        EncodingFamilyDef(
            "first",
            group=dialect,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            auxiliary_key_enum=first_keys,
        ),
        EncodingFamilyDef(
            "second",
            group=dialect,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            auxiliary_key_enum=second_keys,
        ),
    ]

    with _raises_value_error("encoding families in one dialect must share one auxiliary key enum"):
        generate_ops_h("encoding", 0x09, [], (), families)


def test_generate_qualified_encoding_family_metadata() -> None:
    family = EncodingFamilyDef(
        "ggml.q4_k",
        group=Dialect("encoding", dialect_id=0x09),
        role=EncodingFamilyRole.STORAGE_SCHEMA,
    )

    ops_h = generate_ops_h("encoding", 0x09, [], (), [family])
    tables_c = generate_tables_c("encoding", 0x09, [], (), [family])

    assert "loom_encoding_ggml_q4_k_family_descriptor" in ops_h
    assert '.name = _BSTRING(9, "ggml.q4_k")' in tables_c


def test_generate_dialect_qualified_encoding_family_metadata() -> None:
    family = EncodingFamilyDef(
        "encoding.operand",
        group=Dialect("encoding", dialect_id=0x09),
        role=EncodingFamilyRole.STORAGE_SCHEMA,
    )

    ops_h = generate_ops_h("encoding", 0x09, [], (), [family])
    tables_c = generate_tables_c("encoding", 0x09, [], (), [family])

    assert "loom_encoding_operand_family_descriptor" in ops_h
    assert "loom_encoding_encoding_operand" not in ops_h
    assert '.name = _BSTRING(16, "encoding.operand")' in tables_c


def test_rejects_colliding_encoding_family_c_names() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    families = [
        EncodingFamilyDef(
            name,
            group=dialect,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
        )
        for name in ("ggml.q4_k", "ggml_q4_k")
    ]

    with _raises_value_error("encoding families 'ggml.q4_k' and 'ggml_q4_k' both generate C prefix"):
        generate_ops_h("encoding", 0x09, [], (), families)
    with _raises_value_error("encoding families 'ggml.q4_k' and 'ggml_q4_k' both generate C prefix"):
        generate_tables_c("encoding", 0x09, [], (), families)


def test_rejects_duplicate_encoding_family_names() -> None:
    family = EncodingFamilyDef(
        "ggml.q4_k",
        group=Dialect("encoding", dialect_id=0x09),
        role=EncodingFamilyRole.STORAGE_SCHEMA,
    )

    with _raises_value_error("duplicate encoding family 'ggml.q4_k'"):
        generate_ops_h("encoding", 0x09, [], (), [family, family])


def test_generate_encoding_families_share_enum_vocabulary() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    rounding = EnumDef(
        "RoundingPolicy",
        [EnumCase("none", 0), EnumCase("nearest_even", 1)],
    )
    families = [
        EncodingFamilyDef(
            name,
            group=dialect,
            role=EncodingFamilyRole.STORAGE_SCHEMA,
            parameters=[AttrDef("rounding", ATTR_TYPE_ENUM, enum_def=rounding)],
        )
        for name in ("first", "second")
    ]

    ops_h = generate_ops_h("encoding", 0x09, [], (), families)
    tables_c = generate_tables_c("encoding", 0x09, [], (), families)

    assert ops_h.count("typedef enum loom_encoding_rounding_policy_e") == 1
    assert tables_c.count("loom_encoding_rounding_policy_names[]") == 1
    assert tables_c.count(".enum_case_names = loom_encoding_rounding_policy_names") == 2


def test_generate_parameterized_attrs_share_encoding_enum_vocabulary() -> None:
    dialect = Dialect("encoding", dialect_id=0x09)
    numeric_format = EnumDef(
        "NumericFormat",
        [EnumCase("f16", 1), EnumCase("bf16", 2)],
    )
    requirements = ParameterizedAttrDef(
        "encoding.requirements",
        group=dialect,
        parameters=[
            AttrDef(
                "element_format",
                ATTR_TYPE_ENUM,
                enum_def=numeric_format,
                optional=True,
            ),
        ],
    )
    family = EncodingFamilyDef(
        "operand",
        group=dialect,
        role=EncodingFamilyRole.STORAGE_SCHEMA,
        parameters=[
            AttrDef("element_format", ATTR_TYPE_ENUM, enum_def=numeric_format),
        ],
    )

    ops_h = generate_ops_h("encoding", 0x09, [], [requirements], [family])
    builders_c = generate_builders_c(
        "encoding",
        [],
        [requirements],
        encoding_families=[family],
    )
    tables_c = generate_tables_c("encoding", 0x09, [], [requirements], [family])

    assert ops_h.count("typedef enum loom_encoding_numeric_format_e") == 1
    assert "loom_encoding_requirements_element_format_t" not in ops_h
    assert "static inline loom_encoding_numeric_format_t loom_encoding_requirements_attr_element_format" in ops_h
    assert "loom_encoding_numeric_format_t element_format" in ops_h
    assert "loom_encoding_numeric_format_t element_format" in builders_c
    assert tables_c.count("loom_encoding_numeric_format_names[]") == 1
    assert tables_c.count(".enum_case_names = loom_encoding_numeric_format_names") == 2


def test_generate_parameterized_attribute_array_surface() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    tile = ParameterizedAttrDef(
        "test.tile",
        group=dialect,
        parameters=[AttrDef("width", ATTR_TYPE_I64)],
        primary_parameter="width",
    )
    node = ParameterizedAttrDef(
        "test.node",
        group=dialect,
        parameters=[
            AttrDef("value", ATTR_TYPE_I64),
            AttrDef(
                "children",
                ATTR_TYPE_PARAMETERIZED_ARRAY,
                optional=True,
            ),
            AttrDef(
                "tiles",
                ATTR_TYPE_PARAMETERIZED_ARRAY,
                optional=True,
                parameterized_attr=tile,
            ),
        ],
        primary_parameter="value",
    )
    holder = Op(
        "test.holder",
        group=dialect,
        attrs=[
            AttrDef("values", ATTR_TYPE_PARAMETERIZED_ARRAY),
            AttrDef(
                "tiles",
                ATTR_TYPE_PARAMETERIZED_ARRAY,
                optional=True,
                parameterized_attr=tile,
            ),
        ],
        format=[
            Attr("values"),
            OptionalGroup([Attr("tiles")], anchor="tiles"),
        ],
    )
    wrapper = TypeDef(
        "test.wrapper",
        params=[
            AttrDef(
                "children",
                ATTR_TYPE_PARAMETERIZED_ARRAY,
            )
        ],
        format=[Param("children")],
    )

    ops_h = generate_ops_h("test", 0x01, [holder], [tile, node])
    builders_c = generate_builders_c("test", [holder], [tile, node])
    tables_c = generate_tables_c("test", 0x01, [holder], [tile, node])
    type_registry_h, _, type_registry_tables_c = generate_type_registry([wrapper])

    assert "LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(loom_test_holder_values, 0)" in ops_h
    assert "loom_parameterized_attr_array_t values" in ops_h
    assert "loom_optional loom_parameterized_attr_array_t tiles" in ops_h
    assert "loom_module_make_parameterized_attr_array(" in builders_c
    assert "loom_attr_parameterized_array(children.values, children.count)" in builders_c
    assert "loom_attr_parameterized_array(tiles.values, tiles.count)" in builders_c
    assert ".attr_kind = LOOM_ATTR_PARAMETERIZED_ARRAY" in tables_c
    assert ".reference.parameterized_attr_kind = LOOM_PARAMETERIZED_ATTR_KIND_ANY" in tables_c
    assert ".reference.parameterized_attr_kind = LOOM_PARAMETERIZED_ATTR_TEST_TILE" in tables_c
    assert "loom_parameterized_attr_array_t children" in type_registry_h
    assert "loom_attr_as_parameterized_array(" in type_registry_h
    assert ".reference.parameterized_attr_kind = LOOM_PARAMETERIZED_ATTR_KIND_ANY" in type_registry_tables_c


def test_generate_dialect_tables_emit_dense_op_semantics() -> None:
    dialect = Dialect(
        "test",
        dialect_id=0x01,
        default_phase=OpPhase.EXECUTABLE,
    )
    op = Op(
        "test.iota",
        group=dialect,
        contracts=[ContractFamily.VECTOR_COORDINATE],
    )

    ops_h = generate_ops_h("test", 0x01, [op])
    tables_c = generate_tables_c("test", 0x01, [op])

    assert "loom_test_dialect_op_semantics" in ops_h
    assert "loom_op_semantics_t loom_test_op_semantics(" in ops_h
    assert "static const loom_op_semantics_t loom_test_semantics_array[] = {" in tables_c
    assert ".phase = LOOM_OP_PHASE_EXECUTABLE," in tables_c
    assert ".contract_families = LOOM_CONTRACT_VECTOR_COORDINATE," in tables_c
    assert "loom_dialect_semantics_lookup(" in tables_c
    assert "kind, LOOM_DIALECT_TEST, loom_test_semantics_array," in tables_c
    assert "loom_op_dialect_id(kind)" not in tables_c


def test_generate_dialect_tables_emit_sparse_condition_refinements() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    op = Op(
        "test.is_positive",
        group=dialect,
        operands=[Operand("value", INTEGER)],
        results=[Result("matches", TypeConstraint.I1)],
        condition_refinement=ConditionRefinement(
            source="value",
            truth=ConditionRefinementTruth.TRUE,
            materialize="loom_test_is_positive_materialize",
        ),
    )

    ops_h = generate_ops_h("test", 0x01, [op])
    tables_c = generate_tables_c("test", 0x01, [op])

    assert "iree_status_t loom_test_is_positive_materialize(" in ops_h
    assert "loom_test_dialect_condition_refinements(" in ops_h
    assert "loom_test_condition_refinement_array[]" in tables_c
    assert ".materialize = loom_test_is_positive_materialize," in tables_c
    assert ".source_operand_index = 0," in tables_c
    assert ".truth_flags = LOOM_CONDITION_REFINEMENT_TRUTH_TRUE," in tables_c
    assert ".condition_refinement_index = 1," in tables_c


def test_condition_refinement_requires_a_required_operand_and_i1_result() -> None:
    dialect = Dialect("test")
    refinement = ConditionRefinement(
        source="value",
        truth=ConditionRefinementTruth.TRUE,
        materialize="loom_test_is_positive_materialize",
    )

    with _raises_value_error("does not name an operand"):
        Op(
            "test.missing_source",
            group=dialect,
            results=[Result("matches", TypeConstraint.I1)],
            condition_refinement=refinement,
        )
    with _raises_value_error("must be a required non-variadic operand"):
        Op(
            "test.optional_source",
            group=dialect,
            operands=[Operand("value", INTEGER, optional=True)],
            results=[Result("matches", TypeConstraint.I1)],
            condition_refinement=refinement,
        )
    with _raises_value_error("requires exactly one non-variadic i1 result"):
        Op(
            "test.non_boolean",
            group=dialect,
            operands=[Operand("value", INTEGER)],
            results=[Result("result", INTEGER)],
            condition_refinement=refinement,
        )


def test_generate_tables_omits_zero_default_vtable_fields() -> None:
    op = Op("test.noop", group=Dialect("test"))

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "static const loom_op_vtable_t loom_test_noop_vtable = {" in tables_c
    assert "#define _BSTRING(length, value)" in tables_c
    assert '.name = _OP_NAME(9, 4, "test.noop"),' in tables_c
    assert ".fixed_operand_count = 0," not in tables_c
    assert ".symbol_kind = LOOM_SYMBOL_NONE," not in tables_c
    assert ".canonicalize = NULL," not in tables_c
    assert ".format_elements = NULL," not in tables_c
    assert ".format_element_count = 0," not in tables_c
    assert ".contract_families = 0," not in tables_c


def _target_projection_test_op(
    *,
    fact_type: str | None = None,
    fact_projector: str | None = None,
    fact_specialization: TargetFactSpecialization = TargetFactSpecialization.EXACT,
) -> Op:
    kind = EnumDef("TargetKind", [EnumCase("generic", 0)])
    abi = EnumDef("TargetAbi", [EnumCase("unknown", 0)])
    linkage = EnumDef("TargetLinkage", [EnumCase("default", 0)])
    return Op(
        "test.target",
        group=Dialect("test"),
        attrs=[
            AttrDef("symbol", ATTR_TYPE_SYMBOL),
            AttrDef("kind", ATTR_TYPE_ENUM, enum_def=kind),
            AttrDef("max_workgroup_size_x", ATTR_TYPE_I64, optional=True),
            AttrDef("subgroup_size", ATTR_TYPE_I64, optional=True),
            AttrDef("abi", ATTR_TYPE_ENUM, enum_def=abi, optional=True),
            AttrDef("export_symbol", ATTR_TYPE_STRING, optional=True),
            AttrDef("linkage", ATTR_TYPE_ENUM, enum_def=linkage, optional=True),
        ],
        interfaces=[
            TargetLikeInterface(
                symbol="symbol",
                selector="kind",
                bundle_table="loom_test_target_bundles",
                fact_type=fact_type,
                fact_projector=fact_projector,
                fact_specialization=fact_specialization,
            )
        ],
    )


def test_generate_target_projection_emits_typed_fields() -> None:
    tables_c = generate_tables_c(
        "test",
        0,
        [_target_projection_test_op()],
    )

    assert ("snapshot.max_workgroup_size.x), 2, LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_X, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32}") in tables_c
    assert ("snapshot.subgroup_size), 3, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE, LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32}") in tables_c
    assert ("export_plan.abi_kind), 4, LOOM_TARGET_FACT_FIELD_ABI, LOOM_TARGET_PROJECTION_VALUE_ENUM_U8}") in tables_c
    assert ("export_plan.export_symbol), 5, LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL, LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW}") in tables_c
    assert ("export_plan.linkage), 6, LOOM_TARGET_FACT_FIELD_LINKAGE, LOOM_TARGET_PROJECTION_VALUE_ENUM_U8}") in tables_c


def test_generate_target_projection_emits_typed_fact_contract() -> None:
    ops_h = generate_ops_h(
        "test",
        0,
        [_target_projection_test_op(fact_specialization=TargetFactSpecialization.STRUCTURAL)],
    )
    tables_c = generate_tables_c(
        "test",
        0,
        [_target_projection_test_op(fact_specialization=TargetFactSpecialization.STRUCTURAL)],
    )

    assert "extern const loom_target_fact_type_t loom_test_target_fact_type;" in ops_h
    assert '#include "loom/ops/target/facts.h"' in tables_c
    assert "const loom_target_fact_type_t loom_test_target_fact_type = {" in tables_c
    assert '.name = IREE_SVL("test"),' in tables_c
    assert ".storage_size = sizeof(loom_target_facts_t)," in tables_c
    assert ".satisfies_identity_requirement = loom_target_facts_selector_satisfies_identity_requirement," in tables_c
    assert (".satisfies_specialization_requirement = loom_target_facts_structural_satisfy_specialization_requirement,") in tables_c
    assert ".fact_type = &loom_test_target_fact_type," in tables_c


def test_generate_target_projection_accepts_family_owned_fact_type() -> None:
    tables_c = generate_tables_c(
        "test",
        0,
        [
            _target_projection_test_op(
                fact_type="loom_test_custom_fact_type",
                fact_projector="loom_test_custom_fact_projector",
            )
        ],
    )

    assert "extern const loom_target_fact_type_t loom_test_custom_fact_type;" in tables_c
    assert "extern const loom_target_fact_projector_t loom_test_custom_fact_projector;" in tables_c
    assert "static const loom_target_fact_type_t" not in tables_c
    assert ".fact_type = &loom_test_custom_fact_type," in tables_c
    assert ".fact_projector = &loom_test_custom_fact_projector," in tables_c


def test_generate_target_projection_rejects_projector_without_family_facts() -> None:
    with _raises_value_error(r"family fact projector requires an external fact type"):
        generate_tables_c(
            "test",
            0,
            [_target_projection_test_op(fact_projector="loom_test_fact_projector")],
        )


def test_generate_target_projection_rejects_split_fact_ownership() -> None:
    with _raises_value_error(r"external fact type owns its specialization"):
        generate_tables_c(
            "test",
            0,
            [
                _target_projection_test_op(
                    fact_type="loom_test_custom_fact_type",
                    fact_specialization=TargetFactSpecialization.STRUCTURAL,
                )
            ],
        )


def test_generate_tables_omits_type_propagation_flag_for_scalar_only_constraints() -> None:
    op = Op(
        "test.addi",
        group=Dialect("test"),
        operands=[Operand("lhs", INTEGER), Operand("rhs", INTEGER)],
        results=[Result("result", INTEGER)],
        constraints=[SameType("lhs", "rhs", "result")],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE" not in tables_c
    assert ".vtable_flags =" not in tables_c


def test_generate_tables_emits_keyed_module_record_metadata() -> None:
    op = Op(
        "module.record",
        group=Dialect("module"),
        attrs=[AttrDef("payload", ATTR_TYPE_I64), AttrDef("key", ATTR_TYPE_STRING)],
        traits=[MODULE_SCOPE, KeyedModuleRecord("key")],
    )

    tables_c = generate_tables_c("module", 0x01, [op])

    assert ".vtable_flags = LOOM_OP_VTABLE_KEYED_MODULE_RECORD," in tables_c
    assert ".module_record_key_attr_index = 1," in tables_c


def test_generate_tables_derives_decomposable_for_same_type_elementwise_vector_ops() -> None:
    op = Op(
        "test.add",
        group=Dialect("test"),
        operands=[Operand("lhs", VECTOR), Operand("rhs", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameType("lhs", "rhs", "result")],
        traits=[ELEMENTWISE],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "LOOM_TRAIT_DECOMPOSABLE" in tables_c


def test_generate_tables_derives_decomposable_for_shape_preserving_vector_ops() -> None:
    op = Op(
        "test.cast",
        group=Dialect("test"),
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameShape("input", "result")],
        traits=[ELEMENTWISE],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "LOOM_TRAIT_DECOMPOSABLE" in tables_c


def test_generate_tables_accepts_explicit_decomposable_for_same_type_elementwise_vector_ops() -> None:
    op = Op(
        "test.add",
        group=Dialect("test"),
        operands=[Operand("lhs", VECTOR), Operand("rhs", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameType("lhs", "rhs", "result")],
        traits=[ELEMENTWISE, DECOMPOSABLE],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "LOOM_TRAIT_DECOMPOSABLE" in tables_c


def test_generate_tables_does_not_derive_decomposable_for_mixed_result_elementwise_ops() -> None:
    op = Op(
        "test.cmp",
        group=Dialect("test"),
        operands=[Operand("lhs", VECTOR), Operand("rhs", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameType("lhs", "rhs")],
        traits=[ELEMENTWISE],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "LOOM_TRAIT_DECOMPOSABLE" not in tables_c


def test_generate_tables_rejects_explicit_decomposable_for_mixed_result_elementwise_ops() -> None:
    op = Op(
        "test.cmp",
        group=Dialect("test"),
        operands=[Operand("lhs", VECTOR), Operand("rhs", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameType("lhs", "rhs")],
        traits=[ELEMENTWISE, DECOMPOSABLE],
    )

    with _raises_value_error("Decomposable requires"):
        generate_tables_c("test", 0x01, [op])


def test_generate_tables_marks_type_propagation_candidate_for_refinable_types() -> None:
    op = Op(
        "test.refine",
        group=Dialect("test"),
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[SameType("input", "result")],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert ".vtable_flags = LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE," in tables_c


def test_generate_tables_marks_type_transfer_as_type_propagation_candidate() -> None:
    op = Op(
        "test.transfer",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        type_transfer="loom_test_transfer_types",
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert ".vtable_flags = LOOM_OP_VTABLE_TYPE_PROPAGATION_CANDIDATE," in tables_c


def test_generate_tables_omits_zero_default_symbol_definition_fields() -> None:
    dialect = Dialect("test")
    op = Op(
        "test.symbol",
        group=dialect,
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("name", ATTR_TYPE_SYMBOL)],
        symbol_def=SymbolDefinition(
            field="name",
            name="test symbol",
            interfaces=["record"],
        ),
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert "static const loom_symbol_definition_descriptor_t loom_test_symbol_symbol_def = {" in tables_c
    assert '.name = _BSTRING(11, "test symbol"),' in tables_c
    assert ".name_attr_index = 0," not in tables_c
    assert ".interfaces = LOOM_SYMBOL_INTERFACE_RECORD," in tables_c
    assert ".bytecode_kind = LOOM_SYMBOL_NONE," not in tables_c
    assert ".fact_domain = NULL," not in tables_c


def test_generate_tables_emits_generic_symbol_definition_flags() -> None:
    op = Op(
        "test.symbol",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("name", ATTR_TYPE_SYMBOL)],
        symbol_def=SymbolDefinition(
            field="name",
            name="test symbol",
            interfaces=["record"],
            flags=[
                SymbolDefinitionFlag.DECLARATION,
                SymbolDefinitionFlag.TEST_ONLY,
            ],
        ),
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert (".flags = LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION | LOOM_SYMBOL_DEFINITION_FLAG_TEST_ONLY,") in tables_c


def test_generate_tables_emits_generic_symbol_visibility() -> None:
    visibility = EnumDef("Visibility", [EnumCase("public", 1)])
    op = Op(
        "test.symbol",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[
            AttrDef("name", ATTR_TYPE_SYMBOL),
            AttrDef("visibility", ATTR_TYPE_ENUM, enum_def=visibility, optional=True),
        ],
        symbol_def=SymbolDefinition(
            field="name",
            name="test symbol",
            interfaces=["record"],
            visibility="visibility",
        ),
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert ".visibility_attr_index_plus_one = 2," in tables_c


def test_generate_tables_inherits_func_like_symbol_visibility() -> None:
    visibility = EnumDef("Visibility", [EnumCase("public", 1)])
    op = Op(
        "test.function",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[
            AttrDef("callee", ATTR_TYPE_SYMBOL),
            AttrDef("visibility", ATTR_TYPE_ENUM, enum_def=visibility, optional=True),
        ],
        symbol_def=SymbolDefinition(
            field="callee",
            name="function",
            interfaces=["func_like", "callable"],
        ),
        regions=[RegionDef("body")],
        interfaces=[FuncLikeInterface(callee="callee", visibility="visibility", body="body")],
        format=[SymbolRef("callee"), FuncArgs("args"), Region("body")],
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert ".visibility_attr_index_plus_one = 2," in tables_c


def test_generate_kernel_declaration_preserves_both_signatures() -> None:
    op = Op(
        "test.kernel_decl",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        operands=[
            Operand("workloads", ANY, variadic=True),
            Operand("args", ANY, variadic=True),
        ],
        attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
        symbol_def=SymbolDefinition(
            field="callee",
            name="kernel",
            interfaces=["func_like", "kernel"],
            kernel_contract=SymbolKernelContract(workload_operands="workloads"),
        ),
        interfaces=[FuncLikeInterface(callee="callee", args="args")],
        format=[
            SymbolRef("callee"),
            FuncArgs("workloads"),
            FuncArgs("args"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])
    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert ".interfaces = LOOM_SYMBOL_INTERFACE_FUNC_LIKE | LOOM_SYMBOL_INTERFACE_KERNEL," in tables_c
    assert ".kernel_workload_operand_field_index_plus_one = 1," in tables_c
    assert ".args_operand_field_index = 1," in tables_c
    assert ".args_operand_segment_count = 2," in tables_c
    assert "const loom_type_t* workloads_types" in ops_h
    assert "iree_host_size_t workloads_types_count" in ops_h
    assert "const loom_type_t* args_types" in ops_h
    assert "iree_host_size_t args_types_count" in ops_h
    assert "operand_segment_counts[0] = (uint16_t)workloads_types_count;" in builders_c
    assert "operand_segment_counts[1] = (uint16_t)args_types_count;" in builders_c
    assert "loom_builder_define_value(builder, workloads_types[_i]" in builders_c
    assert "loom_builder_define_value(builder, args_types[_i]" in builders_c


def test_generate_callable_symbol_interface() -> None:
    op = Op(
        "test.function",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
        symbol_def=SymbolDefinition(
            field="callee",
            name="function",
            interfaces=["func_like", "callable"],
        ),
        regions=[RegionDef("body")],
        interfaces=[FuncLikeInterface(callee="callee", body="body")],
        format=[SymbolRef("callee"), FuncArgs("args"), Region("body")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert (".interfaces = LOOM_SYMBOL_INTERFACE_FUNC_LIKE | LOOM_SYMBOL_INTERFACE_CALLABLE,") in tables_c


def test_generate_command_program_symbol_interface() -> None:
    op = Op(
        "test.command_program",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
        symbol_def=SymbolDefinition(
            field="callee",
            name="command program",
            interfaces=["func_like", "command_program"],
        ),
        regions=[RegionDef("body")],
        interfaces=[FuncLikeInterface(callee="callee", body="body")],
        format=[SymbolRef("callee"), FuncArgs("args"), Region("body")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert (".interfaces = LOOM_SYMBOL_INTERFACE_FUNC_LIKE | LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM,") in tables_c


def test_generate_body_signature_partition() -> None:
    op = Op(
        "test.partitioned_function",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[
            AttrDef("callee", ATTR_TYPE_SYMBOL),
            AttrDef("specialization_count", ATTR_TYPE_I64),
        ],
        symbol_def=SymbolDefinition(
            field="callee",
            name="function",
            interfaces=["func_like", "callable"],
        ),
        regions=[RegionDef("body")],
        interfaces=[
            FuncLikeInterface(
                callee="callee",
                specialization_count="specialization_count",
                body="body",
            )
        ],
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

    tables_c = generate_tables_c("test", 0, [op])
    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert ("{LOOM_FORMAT_KIND_FUNC_ARGS, 255, LOOM_FORMAT_FUNC_ARGS_DATA(255, 1)}") in tables_c
    assert ("{LOOM_FORMAT_KIND_FUNC_ARGS, 255, LOOM_FORMAT_FUNC_ARGS_DATA(1, 255)}") in tables_c
    assert ".specialization_count_attr_index = 1," in tables_c
    assert "const loom_type_t* specializations_types" in ops_h
    assert "iree_host_size_t specializations_types_count" in ops_h
    assert "const loom_type_t* arguments_types" in ops_h
    assert "iree_host_size_t arguments_types_count" in ops_h
    assert "int64_t specialization_count" not in ops_h
    assert "_i < specializations_types_count" in builders_c
    assert "_i < arguments_types_count" in builders_c
    assert ("loom_op_attrs(*out_op)[1] = loom_attr_i64((int64_t)(specializations_types_count));") in builders_c


def test_generate_operand_signature_partition() -> None:
    op = Op(
        "test.partitioned_declaration",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        operands=[Operand("args", ANY, variadic=True)],
        attrs=[
            AttrDef("callee", ATTR_TYPE_SYMBOL),
            AttrDef("specialization_count", ATTR_TYPE_I64),
        ],
        symbol_def=SymbolDefinition(
            field="callee",
            name="function",
            interfaces=["func_like", "callable"],
            flags=[SymbolDefinitionFlag.DECLARATION],
        ),
        interfaces=[FuncLikeInterface(callee="callee", args="args")],
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
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert ("{LOOM_FORMAT_KIND_FUNC_ARGS, 0, LOOM_FORMAT_FUNC_ARGS_DATA(255, 1)}") in tables_c
    assert ("{LOOM_FORMAT_KIND_FUNC_ARGS, 0, LOOM_FORMAT_FUNC_ARGS_DATA(1, 255)}") in tables_c
    assert "uint32_t variadic_operand_count_32 = 0;" in builders_c
    assert ("variadic_operand_count_32 += (uint32_t)specializations_types_count;") in builders_c
    assert ("variadic_operand_count_32 += (uint32_t)arguments_types_count;") in builders_c
    assert "uint16_t variadic_operand_offset = 0;" in builders_c
    assert ("variadic_operand_offset += (uint16_t)specializations_types_count;") in builders_c
    assert ("loom_op_attrs(*out_op)[1] = loom_attr_i64((int64_t)(specializations_types_count));") in builders_c


def test_generate_tables_emits_symbol_value_contract_indices() -> None:
    op = Op(
        "test.value",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[
            AttrDef("name", ATTR_TYPE_SYMBOL),
            AttrDef("value", "any"),
            AttrDef("predicates", ATTR_TYPE_PREDICATE_LIST),
        ],
        results=[Result("type", ANY)],
        symbol_def=SymbolDefinition(
            field="name",
            name="test value",
            interfaces=["record"],
            value_contract=SymbolValueContract(
                result="type",
                value="value",
                predicates="predicates",
            ),
        ),
    )

    tables_c = generate_tables_c("test", 0x01, [op])

    assert ".value_contract_result_index_plus_one = 1," in tables_c
    assert ".value_contract_value_attr_index_plus_one = 2," in tables_c
    assert ".value_contract_predicates_attr_index_plus_one = 3," in tables_c


def test_generate_tables_rejects_variadic_symbol_value_contract_result() -> None:
    op = Op(
        "test.value",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("name", ATTR_TYPE_SYMBOL)],
        results=[Result("types", ANY, variadic=True)],
        symbol_def=SymbolDefinition(
            field="name",
            name="test value",
            interfaces=["record"],
            value_contract=SymbolValueContract(result="types"),
        ),
    )

    with _raises_value_error("value contract result 'types' must not be variadic"):
        generate_tables_c("test", 0x01, [op])


def test_generate_tables_rejects_non_predicate_value_contract_attr() -> None:
    op = Op(
        "test.value",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[
            AttrDef("name", ATTR_TYPE_SYMBOL),
            AttrDef("predicates", ATTR_TYPE_I64),
        ],
        results=[Result("type", ANY)],
        symbol_def=SymbolDefinition(
            field="name",
            name="test value",
            interfaces=["record"],
            value_contract=SymbolValueContract(result="type", predicates="predicates"),
        ),
    )

    with _raises_value_error("predicates 'predicates' must name a predicate_list"):
        generate_tables_c("test", 0x01, [op])


def test_generate_sharded_tables_exports_vtables_and_keeps_dense_aggregator() -> None:
    dialect = Dialect(
        "test",
        dialect_id=0x01,
        categories=[OpCategory("structure"), OpCategory("math")],
    )
    first = Op("test.first", group=dialect, category=dialect.categories[0])
    second = Op("test.second", group=dialect, category=dialect.categories[1])

    table_files = generate_sharded_tables_c(
        "test",
        0x01,
        [
            (dialect.categories[0], [first]),
            (dialect.categories[1], [second]),
        ],
    )

    assert sorted(table_files) == ["tables.c", "tables.h", "tables/math.c", "tables/structure.c"]
    assert '#include "loom/ops/test/tables.h"' in table_files["tables/structure.c"]
    assert "#define _BSTRING(length, value)" not in table_files["tables/structure.c"]
    assert "#define _BSTRING(length, value)" in table_files["tables.h"]
    assert "const loom_op_vtable_t loom_test_first_vtable = {" in table_files["tables/structure.c"]
    assert "loom_test_dialect_vtables" not in table_files["tables/structure.c"]
    assert "extern const loom_op_vtable_t loom_test_first_vtable;" in table_files["tables.h"]
    assert "extern const loom_op_vtable_t loom_test_second_vtable;" in table_files["tables.h"]
    assert '#include "loom/ops/test/tables.h"' in table_files["tables.c"]
    assert "    &loom_test_first_vtable," in table_files["tables.c"]
    assert "    &loom_test_second_vtable," in table_files["tables.c"]
    assert "loom_op_semantics_t loom_test_op_semantics(" in table_files["tables.c"]


def test_generate_tables_aggregator_delegates_semantics_lookup() -> None:
    dialect = Dialect("test", dialect_id=0x01)
    op = Op("test.first", group=dialect)

    tables_c = generate_tables_aggregator_c("test", 0x01, [op])

    assert "loom_dialect_semantics_lookup(" in tables_c
    assert "kind, LOOM_DIALECT_TEST, loom_test_semantics_array," in tables_c
    assert "loom_op_dialect_id(kind)" not in tables_c


def test_generate_tables_rejects_constraint_field_index_above_6_bit_max() -> None:
    op = Op(
        "test.wide",
        group=Dialect("test"),
        operands=[Operand(f"input_{i}", INTEGER) for i in range(65)],
        constraints=[SameType("input_0", "input_64")],
    )

    with _raises_value_error(
        r"Op 'test\.wide' constraint SameType: field 'input_64' "
        r"index 64 exceeds LOOM_FIELD_REF 6-bit max 63"
    ):
        generate_tables_c("test", 0, [op])


def test_generate_tables_emits_element_width_constraint() -> None:
    op = Op(
        "test.ext",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        constraints=[ElementWidthGreaterThan("result", "input")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ELEMENT_WIDTH_ORDER" in tables_c
    assert "LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN" in tables_c
    assert "LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(0, 0)" in tables_c


def test_generate_tables_expands_clause_format() -> None:
    op = Op(
        "test.copy",
        group=Dialect("test"),
        operands=[
            Operand("source", INTEGER),
            Operand("target", INTEGER),
        ],
        format=[
            Clause("source", Ref("source")),
            Clause("target", Ref("target")),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_KW_SOURCE" in tables_c
    assert "LOOM_KW_TARGET" in tables_c
    assert "LOOM_FORMAT_KIND_GLUE" in tables_c
    assert "LOOM_FORMAT_KIND_OPERAND_REF" in tables_c


def test_generate_tables_emits_literal_matches_element_type_constraint() -> None:
    op = Op(
        "test.literal",
        group=Dialect("test"),
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("value", "any")],
        constraints=[LiteralMatchesElementType("value", "result")],
        format=[Clause("value", Attr("value")), COLON, ResultType("result")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE" in tables_c
    assert "LOOM_PROPERTY_ELEMENT_TYPE" in tables_c


def test_generate_tables_emits_bit_width_attr_constraints() -> None:
    op = Op(
        "test.bitfield",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("offset", "i64"), AttrDef("width", "i64")],
        constraints=[
            PositiveBitWidthAttr("width"),
            ElementWidthAtLeastAttr("result", "width"),
            BitRangeWithinElementWidth("input", "offset", "width"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ATTR_I64_PREDICATE" in tables_c
    assert "LOOM_PROPERTY_BIT_WIDTH_POSITIVE" in tables_c
    assert "LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR" in tables_c
    assert "LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH" in tables_c
    assert "LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(2, 1)" in tables_c


def test_generate_tables_emits_attr_matches_element_type_constraint() -> None:
    op = Op(
        "test.constant",
        group=Dialect("test"),
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("value", "any")],
        constraints=[AttrMatchesElementType("value", "result")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE" in tables_c
    assert "LOOM_PROPERTY_ELEMENT_TYPE" in tables_c
    assert "LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(1, 0)" in tables_c
    assert "LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TYPE, 5)" in tables_c
    assert "loom_err_" not in tables_c


def test_generate_tables_emits_bit_count_constraints() -> None:
    op = Op(
        "test.bitstream",
        group=Dialect("test"),
        operands=[Operand("source", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("width", "i64")],
        constraints=[
            TotalBitCountEqual("source", "result"),
            PackedPayloadBitCountMatchesStorage("source", "width", "result", "result"),
            UnpackedPayloadBitCountMatchesStorage("result", "width", "source", "result"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL" in tables_c
    assert "LOOM_PROPERTY_TOTAL_BIT_COUNT" in tables_c
    assert "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(1, 0)" in tables_c
    assert ("LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(1, 0)") in tables_c
    assert ("LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(1, 0)") in tables_c


def test_generate_builders_use_explicit_flags_for_optional_scalar_attrs() -> None:
    op = Op(
        "test.optional",
        group=Dialect("test"),
        attrs=[AttrDef("priority", "i64", optional=True)],
        format=[Attr("priority")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "enum loom_test_optional_build_flag_bits_e {" in ops_h
    assert "LOOM_TEST_OPTIONAL_BUILD_FLAG_HAS_PRIORITY = 1u << 0," in ops_h
    assert "typedef uint32_t loom_test_optional_build_flags_t;" in ops_h
    assert "loom_test_optional_build_flags_t build_flags" in ops_h
    assert "loom_test_optional_build_flags_t build_flags" in builders_c
    assert ("iree_any_bit_set(build_flags, LOOM_TEST_OPTIONAL_BUILD_FLAG_HAS_PRIORITY)") in builders_c
    assert "priority != 0" not in builders_c


def test_generate_builders_escape_c_and_cpp_keyword_parameters() -> None:
    op = Op(
        "test.keyword",
        group=Dialect("test"),
        attrs=[AttrDef("requires", "i64")],
        format=[Attr("requires")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "int64_t requires_" in ops_h
    assert "int64_t requires_" in builders_c
    assert "loom_attr_i64(requires_)" in builders_c
    assert "LOOM_DEFINE_ATTR_I64(loom_test_keyword_requires," in ops_h


def test_generate_builders_use_explicit_flags_for_optional_symbol_refs() -> None:
    op = Op(
        "test.targeted",
        group=Dialect("test"),
        attrs=[AttrDef("target", "symbol", optional=True)],
        format=[OptionalGroup([SymbolRef("target")], anchor="target")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "enum loom_test_targeted_build_flag_bits_e {" in ops_h
    assert "LOOM_TEST_TARGETED_BUILD_FLAG_HAS_TARGET = 1u << 0," in ops_h
    assert "loom_test_targeted_build_flags_t build_flags" in ops_h
    assert "loom_optional loom_symbol_ref_t target" in ops_h
    assert "loom_test_targeted_build_flags_t build_flags" in builders_c
    assert ("iree_any_bit_set(build_flags, LOOM_TEST_TARGETED_BUILD_FLAG_HAS_TARGET)") in builders_c
    assert "loom_op_attrs(*out_op)[0] = loom_attr_symbol(target);" in builders_c


def test_generate_builders_use_macros_for_32_bit_build_flags() -> None:
    op = Op(
        "test.wide32",
        group=Dialect("test"),
        attrs=[AttrDef(f"value_{i}", "i64", optional=True) for i in range(32)],
        format=[AttrDict()],
    )

    ops_h = generate_ops_h("test", 0, [op])

    assert "enum loom_test_wide32_build_flag_bits_e" not in ops_h
    assert "#define LOOM_TEST_WIDE32_BUILD_FLAG_HAS_VALUE_0 (1u << 0)" in ops_h
    assert "#define LOOM_TEST_WIDE32_BUILD_FLAG_HAS_VALUE_31 (1u << 31)" in ops_h
    assert "typedef uint32_t loom_test_wide32_build_flags_t;" in ops_h


def test_generate_builders_use_macros_for_64_bit_build_flags() -> None:
    op = Op(
        "test.wide64",
        group=Dialect("test"),
        attrs=[AttrDef(f"value_{i}", "i64", optional=True) for i in range(33)],
        format=[AttrDict()],
    )

    ops_h = generate_ops_h("test", 0, [op])

    assert "enum loom_test_wide64_build_flag_bits_e" not in ops_h
    assert "#define LOOM_TEST_WIDE64_BUILD_FLAG_HAS_VALUE_0 (UINT64_C(1) << 0)" in ops_h
    assert "#define LOOM_TEST_WIDE64_BUILD_FLAG_HAS_VALUE_32 (UINT64_C(1) << 32)" in ops_h
    assert "typedef uint64_t loom_test_wide64_build_flags_t;" in ops_h


def test_generate_builders_define_mixed_fixed_results() -> None:
    op = Op(
        "test.mixed_results",
        group=Dialect("test"),
        results=[
            Result("payload", ANY),
            Result("valid", TypeConstraint.I1),
        ],
        format=[],
    )

    builders_c = generate_builders_c("test", [op])

    assert "loom_type_t result_type" in builders_c
    assert "loom_type_t result_types_storage[2] = {" in builders_c
    assert "result_type," in builders_c
    assert "loom_type_scalar(LOOM_SCALAR_TYPE_I1)," in builders_c
    assert "builder, result_types_storage, 2," in builders_c


def test_generate_builders_synthesize_exact_result_types() -> None:
    op = Op(
        "test.add",
        group=Dialect("test"),
        operands=[
            Operand("lhs", TypeConstraint.I32),
            Operand("rhs", TypeConstraint.I32),
        ],
        results=[Result("result", TypeConstraint.I32)],
        format=[Ref("lhs"), COMMA, Ref("rhs")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "loom_type_t result_type" not in ops_h
    assert "const loom_type_t* result_types" not in ops_h
    assert "loom_type_scalar(LOOM_SCALAR_TYPE_I32)" in builders_c
    assert "loom_builder_define_result" in builders_c


def test_generate_builders_keep_array_for_multiple_dynamic_fixed_results() -> None:
    op = Op(
        "test.dynamic_results",
        group=Dialect("test"),
        results=[
            Result("lhs", ANY),
            Result("rhs", INTEGER),
        ],
        format=[ResultTypeList("lhs", parens=False)],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "const loom_type_t* result_types" in ops_h
    assert "const loom_type_t* result_types" in builders_c
    assert "builder, result_types, 2," in builders_c
    assert "result_types_storage" not in builders_c


def test_generate_builders_derive_loop_results_from_iter_args() -> None:
    op = _make_counted_loop_op()

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "const loom_type_t* result_types" not in ops_h
    assert "iree_host_size_t result_count" not in ops_h
    assert "(uint16_t)iter_args_count" in builders_c
    assert "loom_module_value_type(builder->module, iter_args[_i])" in builders_c


def test_generate_tables_uses_template_param_for_symbol_attrs() -> None:
    op = Op(
        "test.targeted",
        group=Dialect("test"),
        attrs=[AttrDef("target", "symbol")],
        format=[TemplateParam("target")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "{LOOM_FORMAT_KIND_TEMPLATE_PARAM, 0, 0}" in tables_c
    assert "{LOOM_FORMAT_KIND_SYMBOL_REF, 0, 0}" not in tables_c


def test_generate_tables_uses_template_param_flags_for_symbol_attrs() -> None:
    flags = EnumDef("Flags", [EnumCase("debug", 1), EnumCase("trace", 2)])
    op = Op(
        "test.targeted",
        group=Dialect("test"),
        attrs=[
            AttrDef("target", "symbol"),
            AttrDef("flags", ATTR_TYPE_FLAGS, optional=True, enum_def=flags),
        ],
        format=[TemplateParamFlags("target", "flags")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "{LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS, 0, 0}" in tables_c
    assert "test_targeted_instance_flags_names" in tables_c
    assert "{LOOM_FORMAT_KIND_SYMBOL_REF, 0, 0}" not in tables_c


def test_generate_builders_use_explicit_flags_for_optional_operands() -> None:
    op = Op(
        "test.optional_operand",
        group=Dialect("test"),
        operands=[Operand("extent", INTEGER, optional=True)],
        format=[OptionalGroup([Ref("extent")], anchor="extent")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_OPTIONAL_OPERAND(loom_test_optional_operand_extent, 0)" in ops_h
    assert "LOOM_TEST_OPTIONAL_OPERAND_BUILD_FLAG_HAS_EXTENT = 1u << 0," in ops_h
    assert "loom_optional loom_value_id_t extent" in ops_h
    assert "uint16_t operand_count = 0;" in builders_c
    assert "operand_count = 1;" in builders_c
    assert "loom_op_operands(*out_op)[0] = extent;" in builders_c
    assert ".operand_descriptor_count = IREE_ARRAYSIZE(loom_test_optional_operand_operand_desc)," in tables_c
    assert "LOOM_OPERAND_OPTIONAL" in tables_c


def test_generate_builders_use_explicit_flags_for_optional_regions() -> None:
    op = Op(
        "test.optional_region",
        group=Dialect("test"),
        regions=[
            RegionDef("body"),
            RegionDef("else_region", optional=True),
        ],
        format=[
            Region("body"),
            OptionalGroup([Region("else_region")], anchor="else_region"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_REGION(loom_test_optional_region_body, 0)" in ops_h
    assert ("LOOM_DEFINE_OPTIONAL_REGION(loom_test_optional_region_else_region, 1)") in ops_h
    assert ("LOOM_TEST_OPTIONAL_REGION_BUILD_FLAG_HAS_ELSE_REGION = 1u << 0,") in ops_h
    assert "uint8_t region_count = 1;" in builders_c
    assert "region_count = 2;" in builders_c
    assert "LOOM_REGION_OPTIONAL" in tables_c


def test_generate_tables_encodes_region_argument_uniform_scope() -> None:
    op = Op(
        "test.kernel_region",
        group=Dialect("test"),
        regions=[RegionDef("body", arg_uniform_scope="workgroup")],
        format=[Region("body")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_REGION_WORKGROUP_UNIFORM_ARGS" in tables_c

    cluster_op = Op(
        "test.cluster_kernel_region",
        group=Dialect("test"),
        regions=[RegionDef("body", arg_uniform_scope="cluster")],
        format=[Region("body")],
    )

    cluster_tables_c = generate_tables_c("test", 0, [cluster_op])

    assert "LOOM_REGION_CLUSTER_UNIFORM_ARGS" in cluster_tables_c


def test_generate_tables_rejects_unknown_region_argument_uniform_scope() -> None:
    op = Op(
        "test.bad_region_scope",
        group=Dialect("test"),
        regions=[RegionDef("body", arg_uniform_scope="device")],
        format=[Region("body")],
    )

    with _raises_value_error(
        r"Op 'test\.bad_region_scope' region 'body' has unsupported "
        r"arg_uniform_scope 'device'"
    ):
        generate_tables_c("test", 0, [op])


def test_has_parent_generates_direct_parent_placement() -> None:
    parent = Op(
        "test.parent",
        group=Dialect("test"),
    )
    child = Op(
        "test.child",
        group=Dialect("test"),
        traits=[HasParent("test.parent")],
    )

    tables_c = generate_tables_c("test", 0, [parent, child])

    assert "loom_test_child_required_parents" in tables_c
    assert ".required_parents = loom_test_child_required_parents," in tables_c
    assert (".required_parent_count = IREE_ARRAYSIZE(loom_test_child_required_parents),") in tables_c


def test_generate_builders_use_explicit_flags_for_optional_aggregate_attrs() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[
            AttrDef("dict", "dict", optional=True),
            AttrDef("values", ATTR_TYPE_I64_ARRAY, optional=True),
            AttrDef("payload", ATTR_TYPE_BYTES, optional=True),
            AttrDef("predicates", ATTR_TYPE_PREDICATE_LIST, optional=True),
        ],
        format=[
            AttrDict("dict"),
            OptionalGroup([Attr("values")], anchor="values"),
            OptionalGroup([Attr("payload")], anchor="payload"),
            OptionalGroup([PredicateList("predicates")], anchor="predicates"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT" in ops_h
    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_VALUES" in ops_h
    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_PAYLOAD" in ops_h
    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_PREDICATES" in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" in builders_c

    dict_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT)")
    dict_copy = builders_c.index("loom_module_make_canonical_attr_dict(")
    values_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_VALUES)")
    values_copy = builders_c.index("loom_builder_copy_i64_array_attr_storage(")
    payload_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_PAYLOAD)")
    payload_copy = builders_c.index("loom_builder_copy_bytes_attr_storage(")
    predicates_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_ATTRS_BUILD_FLAG_HAS_PREDICATES)")
    predicates_copy = builders_c.index("loom_builder_copy_predicate_list_attr_storage(")

    assert dict_guard < dict_copy
    assert values_guard < values_copy
    assert payload_guard < payload_copy
    assert predicates_guard < predicates_copy
    assert "dict.count > 0" not in builders_c
    assert "values_count > 0" not in builders_c
    assert "payload.data_length > 0" not in builders_c
    assert "predicates_count > 0" not in builders_c


def test_optional_aggregate_flags_do_not_renumber_existing_fields() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[
            AttrDef("before", ATTR_TYPE_I64, optional=True),
            AttrDef("dict", "dict", optional=True),
            AttrDef("after", ATTR_TYPE_STRING, optional=True),
        ],
        format=[Attr("before"), AttrDict("dict"), Attr("after")],
    )

    ops_h = generate_ops_h("test", 0, [op])

    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_BEFORE = 1u << 0" in ops_h
    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_AFTER = 1u << 1" in ops_h
    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT = 1u << 2" in ops_h


def test_generate_descriptor_backed_enum_array_surface() -> None:
    mode = EnumDef(
        "Mode",
        [EnumCase("low", 1), EnumCase("high", 255)],
    )
    op = Op(
        "test.enum_arrays",
        group=Dialect("test"),
        attrs=[
            AttrDef("required_values", ATTR_TYPE_ENUM_ARRAY, enum_def=mode),
            AttrDef(
                "optional_values",
                ATTR_TYPE_ENUM_ARRAY,
                enum_def=mode,
                optional=True,
                open_enum=True,
            ),
        ],
        format=[
            Attr("required_values"),
            OptionalGroup([Attr("optional_values")], anchor="optional_values"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_ATTR_ENUM_ARRAY(loom_test_enum_arrays_required_values, 0)" in ops_h
    assert "loom_enum_array_t required_values" in ops_h
    assert "loom_optional loom_enum_array_t optional_values" in ops_h
    assert "LOOM_TEST_ENUM_ARRAYS_BUILD_FLAG_HAS_OPTIONAL_VALUES" in ops_h
    assert "loom_builder_copy_enum_array_attr_storage" in builders_c
    assert "loom_attr_enum_array(_optional_values_storage" in builders_c
    optional_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_ENUM_ARRAYS_BUILD_FLAG_HAS_OPTIONAL_VALUES)")
    optional_copy = builders_c.index("loom_builder_copy_enum_array_attr_storage(", optional_guard)
    optional_assignment = builders_c.index("loom_attr_enum_array(_optional_values_storage", optional_copy)
    assert optional_guard < optional_copy < optional_assignment
    assert "LOOM_ATTR_ENUM_ARRAY" in tables_c
    assert "(uint8_t)(IREE_ARRAYSIZE(" in tables_c


def test_generate_descriptor_backed_signed_enum_set_surface() -> None:
    feature = EnumDef(
        "Feature",
        [EnumCase("low", 1), EnumCase("high", 255)],
    )
    op = Op(
        "test.features",
        group=Dialect("test"),
        attrs=[
            AttrDef("required", "signed_enum_set", enum_def=feature),
            AttrDef("optional", "signed_enum_set", enum_def=feature, optional=True),
        ],
        format=[
            Attr("required"),
            OptionalGroup([Attr("optional")], anchor="optional"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_ATTR_SIGNED_ENUM_SET(loom_test_features_required, 0)" in ops_h
    assert "loom_signed_enum_set_t required" in ops_h
    assert "loom_optional loom_signed_enum_set_t optional" in ops_h
    assert "LOOM_TEST_FEATURES_BUILD_FLAG_HAS_OPTIONAL" in ops_h
    assert "loom_builder_copy_signed_enum_set_attr_storage" in builders_c
    assert "loom_attr_signed_enum_set(_optional_storage" in builders_c
    optional_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_FEATURES_BUILD_FLAG_HAS_OPTIONAL)")
    optional_copy = builders_c.index("loom_builder_copy_signed_enum_set_attr_storage(", optional_guard)
    optional_assignment = builders_c.index("loom_attr_signed_enum_set(_optional_storage", optional_copy)
    assert optional_guard < optional_copy < optional_assignment
    assert "LOOM_ATTR_SIGNED_ENUM_SET" in tables_c
    assert ".enum_case_names" in tables_c


def test_generate_descriptor_backed_symbol_collection_surface() -> None:
    op = Op(
        "test.symbol_arrays",
        group=Dialect("test"),
        attrs=[
            AttrDef(
                "dependencies",
                ATTR_TYPE_SYMBOL_SET,
                symbol_ref=SymbolReference("record", ["record"]),
            ),
            AttrDef(
                "available",
                ATTR_TYPE_SYMBOL_ARRAY,
                symbol_ref=SymbolReference(
                    "record",
                    ["record"],
                    role=SymbolReferenceRole.AVAILABILITY,
                ),
                optional=True,
            ),
        ],
        format=[
            Attr("dependencies"),
            OptionalGroup([Attr("available")], anchor="available"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_ATTR_SYMBOL_SET(loom_test_symbol_arrays_dependencies, 0)" in ops_h
    assert "loom_symbol_ref_array_t dependencies" in ops_h
    assert "loom_optional loom_symbol_ref_array_t available" in ops_h
    assert "LOOM_TEST_SYMBOL_ARRAYS_BUILD_FLAG_HAS_AVAILABLE" in ops_h
    assert "loom_builder_copy_symbol_array_attr_storage" in builders_c
    assert "loom_module_try_make_symbol_set" in builders_c
    assert "loom_attr_symbol_array(_available_storage" in builders_c
    build_set = builders_c.index("loom_module_try_make_symbol_set")
    allocate = builders_c.index("loom_builder_allocate_op")
    set_assignment = builders_c.index("loom_op_attrs(*out_op)[0] = _dependencies_set")
    assert build_set < allocate < set_assignment
    optional_guard = builders_c.index("iree_any_bit_set(build_flags, LOOM_TEST_SYMBOL_ARRAYS_BUILD_FLAG_HAS_AVAILABLE)")
    optional_copy = builders_c.index("loom_builder_copy_symbol_array_attr_storage(", optional_guard)
    optional_assignment = builders_c.index("loom_attr_symbol_array(_available_storage", optional_copy)
    assert optional_guard < optional_copy < optional_assignment
    assert "LOOM_ATTR_SYMBOL_SET" in tables_c
    assert ".role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY" in tables_c
    assert ".role = LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY" in tables_c


def test_generate_symbol_array_parameter_surface() -> None:
    family = ParameterizedAttrDef(
        "test.providers",
        group=Dialect("test"),
        parameters=[
            AttrDef(
                "values",
                ATTR_TYPE_SYMBOL_ARRAY,
                symbol_ref=SymbolReference("record", ["record"]),
            )
        ],
    )

    ops_h = generate_ops_h("test", 0, [], parameterized_attrs=[family])
    builders_c = generate_builders_c("test", [], parameterized_attrs=[family])
    tables_c = generate_tables_c("test", 0, [], parameterized_attrs=[family])

    assert "loom_symbol_ref_array_t loom_test_providers_attr_values" in ops_h
    assert "loom_symbol_ref_array_t values" in ops_h
    assert "loom_attr_as_symbol_array" in ops_h
    assert "loom_attr_symbol_array(values.values" in builders_c
    assert "values.count > UINT16_MAX" in builders_c
    assert "LOOM_ATTR_SYMBOL_ARRAY" in tables_c
    assert ".role = LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY" in tables_c


def test_generate_builders_copy_i64_array_attrs_into_builder_arena() -> None:
    op = Op(
        "test.attr_table",
        group=Dialect("test"),
        operands=[
            Operand("selector", INTEGER),
            Operand("values", ANY, variadic=True),
        ],
        results=[Result("results", ANY, variadic=True)],
        attrs=[AttrDef("case_keys", ATTR_TYPE_I64_ARRAY)],
        format=[
            Ref("selector"),
            AttrTable("case_keys", "values"),
            COLON,
            ResultTypeList("results", parens=False),
        ],
    )

    builders_c = generate_builders_c("test", [op])

    assert "int64_t* _case_keys_storage = NULL;" in builders_c
    assert "loom_builder_copy_i64_array_attr_storage(" in builders_c
    assert 'IREE_SV("test.attr_table case_keys")' in builders_c
    assert "loom_attr_i64_array(_case_keys_storage, (uint16_t)case_keys_count)" in builders_c
    assert "(int64_t*)case_keys" not in builders_c


def test_generate_builders_copy_predicate_list_attrs_into_builder_arena() -> None:
    op = Op(
        "test.assume",
        group=Dialect("test"),
        operands=[Operand("value", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("predicates", ATTR_TYPE_PREDICATE_LIST)],
        format=[Ref("value"), PredicateList("predicates"), ResultType("result")],
    )

    builders_c = generate_builders_c("test", [op])

    assert "loom_predicate_t* _predicates_storage = NULL;" in builders_c
    assert "loom_builder_copy_predicate_list_attr_storage(" in builders_c
    assert 'IREE_SV("test.assume predicates")' in builders_c
    assert ("loom_attr_predicate_list(_predicates_storage, (uint16_t)predicates_count)") in builders_c
    assert "(loom_predicate_t*)predicates" not in builders_c


def test_generate_builders_copy_bytes_attrs_into_builder_arena() -> None:
    op = Op(
        "test.rodata",
        group=Dialect("test"),
        attrs=[AttrDef("contents", ATTR_TYPE_BYTES)],
        format=[Attr("contents")],
    )

    builders_c = generate_builders_c("test", [op])

    assert "const uint8_t* _contents_storage = NULL;" in builders_c
    assert "loom_builder_copy_bytes_attr_storage(" in builders_c
    assert 'IREE_SV("test.rodata contents")' in builders_c
    assert "loom_attr_bytes(_contents_storage, (uint32_t)contents.data_length)" in builders_c
    assert "(const uint8_t*)contents.data" not in builders_c


def test_generate_builders_preserve_named_operands_for_non_binary_shapes() -> None:
    op = Op(
        "test.lookup",
        group=Dialect("test"),
        operands=[
            Operand("table", INTEGER),
            Operand("indices", INTEGER),
        ],
        results=[Result("result", INTEGER)],
        format=[Ref("table"), Ref("indices"), ResultType("result")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "loom_value_id_t table" in ops_h
    assert "loom_value_id_t indices" in ops_h
    assert "LOOM_DEFINE_BINARY_OP_BUILDER" not in builders_c
    assert "loom_op_operands(*out_op)[0] = table;" in builders_c
    assert "loom_op_operands(*out_op)[1] = indices;" in builders_c


def test_generate_builders_emit_successor_fields() -> None:
    op = Op(
        "test.br",
        group=Dialect("test"),
        successors=[Successor("dest")],
        format=[BlockRef("dest")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_SUCCESSOR(loom_test_br_dest, 0)" in ops_h
    assert "loom_block_t* dest" in ops_h
    assert "loom_builder_allocate_op_with_successors" in builders_c
    assert "loom_op_successors(*out_op)[0] = dest;" in builders_c
    assert ".fixed_successor_count" not in tables_c
    assert "{LOOM_FORMAT_KIND_SUCCESSOR_REF, 0, 0}," in tables_c


def test_generate_tables_preserves_operand_and_result_descriptor_names() -> None:
    op = Op(
        "test.reduce",
        group=Dialect("test"),
        operands=[
            Operand("input", INTEGER),
            Operand("partials", INTEGER, variadic=True),
        ],
        results=[Result("results", INTEGER, variadic=True)],
        format=[Ref("input"), Ref("partials"), ResultTypeList("results")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert '{_BSTRING(5, "input"), LOOM_TYPE_CONSTRAINT_INTEGER, 0}' in tables_c
    assert ('{_BSTRING(8, "partials"), LOOM_TYPE_CONSTRAINT_INTEGER, LOOM_OPERAND_VARIADIC}') in tables_c
    assert ('{_BSTRING(7, "results"), LOOM_TYPE_CONSTRAINT_INTEGER, LOOM_RESULT_VARIADIC}') in tables_c


def test_generate_tables_emits_operand_roles_when_declared() -> None:
    op = Op(
        "test.select",
        group=Dialect("test"),
        operands=[
            Operand("condition", INTEGER, role=OperandRole.SELECT_CONDITION),
            Operand("true_value", INTEGER, role=OperandRole.SELECT_PAYLOAD),
            Operand("false_value", INTEGER, role=OperandRole.SELECT_PAYLOAD),
        ],
        results=[Result("result", INTEGER)],
        format=[
            Ref("condition"),
            Ref("true_value"),
            Ref("false_value"),
            COLON,
            ResultType("result"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ('{_BSTRING(9, "condition"), LOOM_TYPE_CONSTRAINT_INTEGER, 0, LOOM_OPERAND_OWNERSHIP_NONE, LOOM_OWNERSHIP_CARRIER_NONE, LOOM_OPERAND_ROLE_SELECT_CONDITION}') in tables_c
    assert ('{_BSTRING(10, "true_value"), LOOM_TYPE_CONSTRAINT_INTEGER, 0, LOOM_OPERAND_OWNERSHIP_NONE, LOOM_OWNERSHIP_CARRIER_NONE, LOOM_OPERAND_ROLE_SELECT_PAYLOAD}') in tables_c
    assert (".operand_role_mask = LOOM_OPERAND_ROLE_MASK_SELECT_CONDITION | LOOM_OPERAND_ROLE_MASK_SELECT_PAYLOAD,") in tables_c


def test_generate_tables_emits_ownership_descriptors_only_when_needed() -> None:
    op = Op(
        "test.resource.retain",
        group=Dialect("test"),
        operands=[Operand("resource", POOL)],
        results=[Result("result", POOL)],
        ownership_effects=[
            Retain("resource"),
            RetainedResult("result"),
        ],
    )
    alias = Op(
        "test.resource.alias",
        group=Dialect("test"),
        operands=[Operand("resource", POOL)],
        results=[Result("result", POOL)],
        ownership_effects=[
            Borrow("resource"),
            AliasResult("result", "resource"),
        ],
    )
    move = Op(
        "test.resource.move",
        group=Dialect("test"),
        operands=[Operand("resource", POOL)],
        results=[Result("result", POOL)],
        ownership_effects=[MovedResult("result", "resource")],
    )

    tables_c = generate_tables_c("test", 0, [op, alias, move])

    assert ('{_BSTRING(8, "resource"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_OPERAND_OWNERSHIP_RETAIN, LOOM_OWNERSHIP_CARRIER_BY_VALUE, LOOM_OPERAND_ROLE_NONE}') in tables_c
    assert ('{_BSTRING(6, "result"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_RESULT_OWNERSHIP_RETAINED, LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE}') in tables_c
    assert ('{_BSTRING(6, "result"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_RESULT_OWNERSHIP_ALIAS, 0}') in tables_c
    assert ('{_BSTRING(6, "result"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_RESULT_OWNERSHIP_MOVED, 0}') in tables_c


def test_generate_tables_keeps_repeated_descriptor_names_local() -> None:
    dialect = Dialect("test")
    first = Op("test.first", group=dialect, results=[Result("result", INTEGER)])
    second = Op("test.second", group=dialect, results=[Result("result", INTEGER)])

    tables_c = generate_tables_c("test", 0, [first, second])

    assert tables_c.count('_BSTRING(6, "result")') == 2
    result_rows = [line.strip() for line in tables_c.splitlines() if '_BSTRING(6, "result")' in line]
    assert len(result_rows) == 2
    assert result_rows[0] == result_rows[1]


def test_generate_tables_groups_op_metadata_and_uses_arraysize_counts() -> None:
    op = Op(
        "test.project",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("label", "string")],
        constraints=[SameType("input", "result")],
        format=[Ref("input"), Attr("label"), ResultType("result")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    format_index = tables_c.index("static const loom_format_element_t loom_test_project_format[]")
    operand_index = tables_c.index("static const loom_operand_descriptor_t loom_test_project_operand_desc[]")
    result_index = tables_c.index("static const loom_result_descriptor_t loom_test_project_result_desc[]")
    attr_index = tables_c.index("static const loom_attr_descriptor_t loom_test_project_attr_desc[]")
    constraint_index = tables_c.index("static const loom_constraint_t loom_test_project_constraints[]")
    vtable_index = tables_c.index("static const loom_op_vtable_t loom_test_project_vtable")

    assert format_index < operand_index < result_index < attr_index < constraint_index < vtable_index
    assert ".attribute_count = IREE_ARRAYSIZE(loom_test_project_attr_desc)," in tables_c
    assert ".constraint_count = IREE_ARRAYSIZE(loom_test_project_constraints)," in tables_c
    assert ".format_element_count = IREE_ARRAYSIZE(loom_test_project_format)," in tables_c
    assert ".attribute_count = 1," not in tables_c
    assert ".constraint_count = 1," not in tables_c
    assert ".format_element_count = 3," not in tables_c


def test_generate_tables_emits_call_like_interface() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        results=[Result("results", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results="results",
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "static const loom_call_like_vtable_t loom_test_call_call_like" in tables_c
    assert ".callee_attr_index = 0," in tables_c
    assert ".purity_attr_index = 255," in tables_c
    assert ".operand_field_index = 0," in tables_c
    assert ".operand_segment_count = 0," in tables_c
    assert ".result_offset = 0," in tables_c
    assert ".kind = LOOM_CALL_LIKE_KIND_SEMANTIC," in tables_c


def test_generate_tables_emits_command_program_call_kind() -> None:
    op = Op(
        "test.command_program_call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results=None,
                kind=CallLikeKind.COMMAND_PROGRAM,
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".kind = LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM," in tables_c


def test_generate_tables_emits_no_result_call_like_interface() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results=None,
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".operand_field_index = 0," in tables_c
    assert ".operand_segment_count = 0," in tables_c
    assert ".result_offset = 0," in tables_c


def test_generate_tables_emits_func_like_representation_contract() -> None:
    op = Op(
        "test.func",
        group=Dialect("test"),
        operands=[Operand("args", ANY, variadic=True)],
        attrs=[
            AttrDef("callee", ATTR_TYPE_SYMBOL),
            AttrDef("representation", ATTR_TYPE_STRING, optional=True),
        ],
        interfaces=[
            FuncLikeInterface(
                callee="callee",
                repr_contract="representation",
                args="args",
            )
        ],
        format=[FuncArgs("args")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".callee_attr_index = 0," in tables_c
    assert ".repr_contract_attr_index = 1," in tables_c


def test_generate_tables_emits_func_like_flags() -> None:
    op = Op(
        "test.kernel",
        group=Dialect("test"),
        traits=[SYMBOL_DEFINE],
        attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
        operands=[Operand("args", ANY, variadic=True)],
        symbol_def=SymbolDefinition(
            field="callee",
            name="kernel entry",
            interfaces=["func_like", "kernel_entry"],
        ),
        interfaces=[
            FuncLikeInterface(
                callee="callee",
                args="args",
            )
        ],
        format=[FuncArgs("args")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".flags = LOOM_FUNC_LIKE_FLAG_KERNEL_ENTRY," in tables_c


def test_generate_tables_rejects_non_string_representation_contract() -> None:
    op = Op(
        "test.func",
        group=Dialect("test"),
        operands=[Operand("args", ANY, variadic=True)],
        attrs=[
            AttrDef("callee", ATTR_TYPE_SYMBOL),
            AttrDef("representation", ATTR_TYPE_I64, optional=True),
        ],
        interfaces=[
            FuncLikeInterface(
                callee="callee",
                repr_contract="representation",
                args="args",
            )
        ],
        format=[FuncArgs("args")],
    )

    with _raises_value_error(
        r"FuncLikeInterface on 'test\.func': attr 'representation' referenced "
        r"by 'repr_contract' must have type 'string', got 'i64'"
    ):
        generate_tables_c("test", 0, [op])


def _make_counted_loop_op(
    *,
    body_arg_source: str | None = "iter_args",
    step: str | None = "step",
    iv_type: str = "type_of:lower_bound",
    constraints: list[Constraint] | None = None,
) -> Op:
    if constraints is None:
        constraints = [
            IterArgsMatchResults("iter_args", "results"),
            YieldCountMatchesResults("body", "results"),
            YieldTypesMatchResults("body", "results"),
        ]
    return Op(
        "test.for",
        group=Dialect("test"),
        operands=[
            Operand("lower_bound", INTEGER),
            Operand("upper_bound", INTEGER),
            Operand("step", INTEGER),
            Operand("iter_args", ANY, variadic=True),
        ],
        results=[Result("results", ANY, variadic=True)],
        regions=[
            RegionDef(
                "body",
                single_block=True,
                terminator="test.yield",
                implicit_args=(("iv", iv_type),),
                arg_source=body_arg_source,
            )
        ],
        interfaces=[
            LoopLikeInterface(
                body="body",
                iter_args="iter_args",
                iv="iv",
                lower_bound="lower_bound",
                upper_bound="upper_bound",
                step=step,
            )
        ],
        constraints=constraints,
    )


def _generate_counted_loop_tables(op: Op) -> str:
    yield_op = Op(
        "test.yield",
        group=Dialect("test"),
        operands=[Operand("values", ANY, variadic=True)],
        traits=[TERMINATOR],
    )
    return generate_tables_c("test", 0, [yield_op, op])


def test_generate_tables_emits_counted_loop_like_interface() -> None:
    tables_c = _generate_counted_loop_tables(_make_counted_loop_op())

    assert "static const loom_loop_like_vtable_t loom_test_for_loop_like" in tables_c
    assert ".body_region_index = 0," in tables_c
    assert ".condition_region_index = 255," in tables_c
    assert ".iv_block_arg_index = 0," in tables_c
    assert ".iter_args_operand_field_index = 3," in tables_c
    assert ".lower_bound_operand_index = 0," in tables_c
    assert ".upper_bound_operand_index = 1," in tables_c
    assert ".step_operand_index = 2," in tables_c


def test_generate_tables_rejects_loop_like_missing_yield_constraint() -> None:
    op = _make_counted_loop_op(
        constraints=[
            IterArgsMatchResults("iter_args", "results"),
            YieldCountMatchesResults("body", "results"),
        ]
    )

    with _raises_value_error(r"LoopLikeInterface on 'test\.for': requires YieldTypesMatchResults"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_loop_like_partial_counted_range() -> None:
    op = _make_counted_loop_op(step=None)

    with _raises_value_error(r"LoopLikeInterface on 'test\.for': lower_bound, upper_bound, and step must be declared together"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_loop_like_unprojected_body_state() -> None:
    op = _make_counted_loop_op(body_arg_source=None)

    with _raises_value_error(r"LoopLikeInterface on 'test\.for': body 'body' must source carried arguments from 'iter_args'"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_counted_loop_iv_type_mismatch() -> None:
    op = _make_counted_loop_op(iv_type="index")

    with _raises_value_error(
        r"LoopLikeInterface on 'test\.for': induction variable 'iv' "
        r"must use 'type_of:lower_bound'"
    ):
        _generate_counted_loop_tables(op)


def test_generate_tables_memory_access_defaults_use_matching_fields() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[
            Operand("view", ANY),
            Operand("indices", ANY, variadic=True),
        ],
        results=[Result("result", ANY)],
        attrs=[AttrDef("static_indices", ATTR_TYPE_I64_ARRAY)],
        effects=[Reads("view")],
        interfaces=[MemoryAccessInterface()],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "static const loom_memory_access_vtable_t loom_test_load_memory_access" in tables_c
    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_LOAD," in tables_c
    assert ".view_operand_index = 0," in tables_c
    assert ".byte_offset_operand_index = 255," in tables_c
    assert ".value_operand_index = 255," in tables_c
    assert ".indices_operand_field_index = 1," in tables_c
    assert ".static_indices_attr_index = 0," in tables_c
    assert ".cache_scope_attr_index = 255," in tables_c


def test_generate_tables_memory_access_operation_kind_rows() -> None:
    dialect = Dialect("test")
    atomic_kind = EnumDef("AtomicKind", [EnumCase("addi", 0)])
    atomic_ordering = EnumDef("AtomicOrdering", [EnumCase("relaxed", 0)])
    atomic_scope = EnumDef("AtomicScope", [EnumCase("workgroup", 0)])

    def atomic_attrs() -> list[AttrDef]:
        return [
            AttrDef("kind", ATTR_TYPE_ENUM, enum_def=atomic_kind),
            AttrDef("ordering", ATTR_TYPE_ENUM, enum_def=atomic_ordering),
            AttrDef("scope", ATTR_TYPE_ENUM, enum_def=atomic_scope),
        ]

    ops = [
        Op(
            "test.store",
            group=dialect,
            operands=[
                Operand("value", ANY),
                Operand("view", ANY),
            ],
            effects=[Writes("view")],
            interfaces=[MemoryAccessInterface(value="value")],
        ),
        Op(
            "test.prefetch",
            group=dialect,
            operands=[
                Operand("view", ANY),
            ],
            traits=[HINT],
            interfaces=[MemoryAccessInterface()],
        ),
        Op(
            "test.atomic.reduce",
            group=dialect,
            operands=[
                Operand("value", ANY),
                Operand("view", ANY),
            ],
            attrs=atomic_attrs(),
            effects=[ReadWrites("view")],
            interfaces=[
                MemoryAccessInterface(
                    value="value",
                    atomic_kind="kind",
                    atomic_ordering="ordering",
                    atomic_scope="scope",
                )
            ],
        ),
        Op(
            "test.atomic.rmw",
            group=dialect,
            operands=[
                Operand("value", ANY),
                Operand("view", ANY),
            ],
            results=[Result("old", ANY)],
            attrs=atomic_attrs(),
            effects=[ReadWrites("view")],
            interfaces=[
                MemoryAccessInterface(
                    value="value",
                    atomic_kind="kind",
                    atomic_ordering="ordering",
                    atomic_scope="scope",
                )
            ],
        ),
        Op(
            "test.atomic.cmpxchg",
            group=dialect,
            operands=[
                Operand("expected", ANY),
                Operand("replacement", ANY),
                Operand("view", ANY),
            ],
            results=[Result("old", ANY)],
            attrs=[
                AttrDef("success_ordering", ATTR_TYPE_ENUM, enum_def=atomic_ordering),
                AttrDef("failure_ordering", ATTR_TYPE_ENUM, enum_def=atomic_ordering),
                AttrDef("scope", ATTR_TYPE_ENUM, enum_def=atomic_scope),
            ],
            effects=[ReadWrites("view")],
            interfaces=[
                MemoryAccessInterface(
                    expected="expected",
                    replacement="replacement",
                    atomic_success_ordering="success_ordering",
                    atomic_failure_ordering="failure_ordering",
                    atomic_scope="scope",
                )
            ],
        ),
    ]

    tables_c = generate_tables_c("test", 0, ops)

    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_STORE," in tables_c
    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_PREFETCH," in tables_c
    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_REDUCE," in tables_c
    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_RMW," in tables_c
    assert ".operation_kind = LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG," in tables_c


def test_generate_tables_memory_access_rejects_missing_effects() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[Operand("view", ANY)],
        results=[Result("result", ANY)],
        interfaces=[MemoryAccessInterface()],
    )

    with _raises_value_error(r"MemoryAccessInterface on 'test\.load': unable to infer memory access operation kind"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_memory_access_accepts_segmented_indices_field() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[
            Operand("view", ANY),
            Operand("indices", ANY, variadic=True),
            Operand("auxiliary", ANY, variadic=True),
        ],
        results=[Result("result", ANY)],
        effects=[Reads("view")],
        interfaces=[MemoryAccessInterface()],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_OP_VTABLE_SEGMENTED_OPERANDS" in tables_c
    assert ".view_operand_index = 0," in tables_c
    assert ".indices_operand_field_index = 1," in tables_c


def test_generate_tables_memory_access_accepts_physical_byte_offset() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[
            Operand("source", ANY),
            Operand("byte_offset", ANY),
        ],
        results=[Result("result", ANY)],
        effects=[Reads("source")],
        interfaces=[MemoryAccessInterface(view="source")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".view_operand_index = 0," in tables_c
    assert ".byte_offset_operand_index = 1," in tables_c
    assert ".indices_operand_field_index = 255," in tables_c


def test_generate_tables_memory_access_rejects_mixed_byte_and_logical_offsets() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[
            Operand("view", ANY),
            Operand("byte_offset", ANY),
            Operand("indices", ANY, variadic=True),
        ],
        results=[Result("result", ANY)],
        effects=[Reads("view")],
        interfaces=[MemoryAccessInterface()],
    )

    with _raises_value_error(r"MemoryAccessInterface on 'test\.load': byte_offset is mutually exclusive"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_memory_access_rejects_explicit_missing_field() -> None:
    op = Op(
        "test.store",
        group=Dialect("test"),
        operands=[
            Operand("view", ANY),
            Operand("value", ANY),
        ],
        interfaces=[MemoryAccessInterface(value="payload")],
    )

    with _raises_value_error(r"MemoryAccessInterface on 'test\.store': operand 'payload' not found"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_call_like_non_variadic_operand() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operand", ANY)],
        results=[Result("results", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operand",
                results="results",
            ),
        ],
    )

    with _raises_value_error(r"CallLikeInterface on 'test\.call': operand 'operand' must be variadic"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_accepts_call_like_trailing_operand_groups() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[
            Operand("specializations", ANY, variadic=True),
            Operand("bindings", ANY, variadic=True),
        ],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="specializations",
                results=None,
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".operand_field_index = 0," in tables_c
    assert ".operand_segment_count = 2," in tables_c


def test_generate_tables_resolves_call_like_after_variable_operand_group() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[
            Operand("metadata", ANY, variadic=True),
            Operand("arguments", ANY, variadic=True),
        ],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="arguments",
                results=None,
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert ".operand_field_index = 1," in tables_c
    assert ".operand_segment_count = 2," in tables_c


def test_generate_tables_rejects_call_like_non_variadic_result() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        results=[Result("result", ANY)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results="result",
            ),
        ],
    )

    with _raises_value_error(r"CallLikeInterface on 'test\.call': result 'result' must be variadic"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_no_result_call_like_with_results() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        results=[Result("result", ANY)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results=None,
            ),
        ],
    )

    with _raises_value_error(r"CallLikeInterface on 'test\.call': results=None requires the operation to declare no results"):
        generate_tables_c("test", 0, [op])


def test_types_of_result_field_generates_result_type_list_format() -> None:
    op = Op(
        "test.results",
        group=Dialect("test"),
        results=[Result("results", INTEGER, variadic=True)],
        format=[COLON, TypesOf("results")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_RESULT_TYPE_LIST" in tables_c
    assert "LOOM_FORMAT_KIND_OPERAND_TYPES" not in tables_c


def test_scoped_enum_generates_domain_aware_format_metadata() -> None:
    op = Op(
        "test.packet",
        group=Dialect("test"),
        attrs=[AttrDef("descriptor", "scoped_enum")],
        format=[ScopedEnumRef("descriptor")],
        generate_c_builder=False,
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_ATTR_SCOPED_ENUM(loom_test_packet_descriptor, 0)" in ops_h
    assert "loom_test_packet_build(" not in ops_h
    assert "loom_test_packet_build(" not in builders_c
    assert '.name = _BSTRING(10, "descriptor")' in tables_c
    assert ".attr_kind = LOOM_ATTR_SCOPED_ENUM" in tables_c
    assert "{LOOM_FORMAT_KIND_SCOPED_ENUM_REF, 0, 0}," in tables_c


def test_scoped_enum_rejects_context_free_c_builder() -> None:
    op = Op(
        "test.packet",
        group=Dialect("test"),
        attrs=[AttrDef("descriptor", "scoped_enum")],
        format=[ScopedEnumRef("descriptor")],
    )

    with _raises_value_error("requires a domain-aware handwritten C builder"):
        generate_ops_h("test", 0, [op])


def test_region_syntax_generates_format_selector() -> None:
    op = Op(
        "test.region_syntax",
        group=Dialect("test"),
        regions=[RegionDef("body")],
        format=[Region("body", syntax="test.do")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "{LOOM_FORMAT_KIND_REGION, 0, LOOM_REGION_SYNTAX_TEST_DO}," in tables_c


def test_unknown_region_syntax_is_rejected() -> None:
    op = Op(
        "test.region_syntax",
        group=Dialect("test"),
        regions=[RegionDef("body")],
        format=[Region("body", syntax="missing.syntax")],
    )

    with _raises_value_error(r"Op 'test\.region_syntax': unknown region syntax 'missing\.syntax'"):
        generate_tables_c("test", 0, [op])


def test_inline_attr_dict_uses_declared_attrs() -> None:
    ordering = EnumDef("Ordering", [EnumCase("relaxed", 0)])
    scope = EnumDef("Scope", [EnumCase("workgroup", 0)])
    op = Op(
        "test.atomic",
        group=Dialect("test"),
        attrs=[
            AttrDef("ordering", "enum", enum_def=ordering),
            AttrDef("scope", "enum", enum_def=scope),
        ],
        format=[AttrDict()],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS" in tables_c
    assert "loom_test_atomic_ordering_t ordering" in ops_h
    assert "loom_test_atomic_scope_t scope" in ops_h
    assert "loom_op_attrs(*out_op)[0] = loom_attr_enum(ordering);" in builders_c
    assert "loom_op_attrs(*out_op)[1] = loom_attr_enum(scope);" in builders_c


def test_open_enum_attr_uses_byte_typedef_with_enum_constants() -> None:
    mode = EnumDef("Mode", [EnumCase("known", 1)])
    op = Op(
        "test.open_enum",
        group=Dialect("test"),
        attrs=[AttrDef("mode", "enum", enum_def=mode, open_enum=True)],
        format=[Attr("mode")],
    )

    ops_h = generate_ops_h("test", 0, [op])

    assert "typedef uint8_t loom_test_open_enum_mode_t;" in ops_h
    assert "typedef enum loom_test_open_enum_mode_e {" in ops_h
    assert "  LOOM_TEST_OPEN_ENUM_MODE_KNOWN = 1," in ops_h
    assert "  LOOM_TEST_OPEN_ENUM_MODE_COUNT_ = 2," in ops_h
    assert "} loom_test_open_enum_mode_e;" in ops_h
    assert "#define LOOM_TEST_OPEN_ENUM_MODE_KNOWN" not in ops_h
    assert "LOOM_TEST_OPEN_ENUM_MODE_BYTE_MAX_" not in ops_h
    assert ("LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_open_enum_mode, 0, loom_test_open_enum_mode_t)") in ops_h


def test_external_enum_alias_uses_shared_c_type_without_typedef() -> None:
    mode = EnumDef(
        "Mode",
        [EnumCase("fast", 0), EnumCase("slow", 1)],
        c_type="loom_shared_mode_t",
        c_const_prefix="LOOM_SHARED_MODE",
        c_include="loom/shared/mode.h",
    )
    first = Op(
        "test.first",
        group=Dialect("test"),
        attrs=[AttrDef("mode", "enum", enum_def=mode)],
        format=[Attr("mode")],
    )
    second = Op(
        "test.second",
        group=Dialect("test"),
        attrs=[AttrDef("secondary_mode", "enum", enum_def=mode)],
        format=[Attr("secondary_mode")],
    )

    ops_h = generate_ops_h("test", 0, [first, second])
    builders_c = generate_builders_c("test", [first, second])
    tables_c = generate_tables_c("test", 0, [first, second])

    assert '#include "loom/shared/mode.h"' in ops_h
    assert "typedef enum loom_shared_mode_e" not in ops_h
    assert "LOOM_SHARED_MODE_FAST" not in ops_h
    assert ("LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_first_mode, 0, loom_shared_mode_t)") in ops_h
    assert ("LOOM_DEFINE_ATTR_ENUM_TYPED(loom_test_second_secondary_mode, 0, loom_shared_mode_t)") in ops_h
    assert "loom_shared_mode_t mode" in ops_h
    assert "loom_shared_mode_t secondary_mode" in ops_h
    assert "loom_shared_mode_t mode" in builders_c
    assert "loom_shared_mode_t secondary_mode" in builders_c
    assert tables_c.count("static const loom_bstring_t loom_shared_mode_names[]") == 1
    assert tables_c.count("loom_shared_mode_names") == 5


def test_flags_attrs_do_not_shift_regular_attr_indices() -> None:
    flags = EnumDef("Flags", [EnumCase("hot", 1)])
    op = Op(
        "test.flagged",
        group=Dialect("test"),
        attrs=[
            AttrDef("flags", "flags", optional=True, enum_def=flags),
            AttrDef("name", "string"),
            AttrDef("constraints", "string"),
        ],
        format=[Flags("flags"), Attr("name"), Attr("constraints")],
    )

    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "loom_op_attrs(*out_op)[0] = loom_attr_string(name);" in builders_c
    assert "loom_op_attrs(*out_op)[1] = loom_attr_string(constraints);" in builders_c
    assert "loom_op_attrs(*out_op)[2]" not in builders_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 0, 0}" in tables_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 1, 0}" in tables_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 2, 0}" not in tables_c


def test_enum_keywords_with_punctuation_generate_valid_c_constants() -> None:
    intrinsic_kind = EnumDef(
        "Kind",
        [
            EnumCase("llvm.x86.rdtsc", 0),
            EnumCase("gfx11-generic", 1),
        ],
    )
    op = Op(
        "test.intrinsic",
        group=Dialect("test"),
        attrs=[AttrDef("kind", "enum", enum_def=intrinsic_kind)],
        format=[TemplateParam("kind")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_TEST_INTRINSIC_KIND_LLVM_X86_RDTSC = 0," in ops_h
    assert "LOOM_TEST_INTRINSIC_KIND_GFX11_GENERIC = 1," in ops_h
    assert "LOOM_TEST_INTRINSIC_KIND_LLVM.X86.RDTSC" not in ops_h
    assert "LOOM_TEST_INTRINSIC_KIND_GFX11-GENERIC" not in ops_h
    assert '"llvm.x86.rdtsc"' in tables_c
    assert '"gfx11-generic"' in tables_c


def test_template_param_flags_uses_template_attr_and_instance_flags() -> None:
    kind = EnumDef("Kind", [EnumCase("addf", 0), EnumCase("maxnumf", 1)])
    flags = EnumDef("Flags", [EnumCase("nnan", 1), EnumCase("nsz", 2)])
    op = Op(
        "test.reduce",
        group=Dialect("test"),
        attrs=[
            AttrDef("kind", "enum", enum_def=kind),
            AttrDef("assumptions", "flags", optional=True, enum_def=flags),
        ],
        format=[TemplateParamFlags("kind", "assumptions")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS" in tables_c
    assert "{LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS, 0, 0}" in tables_c
    assert "loom_test_reduce_kind_t kind" in ops_h
    assert "uint8_t instance_flags" in ops_h
    assert "loom_op_attrs(*out_op)[0] = loom_attr_enum(kind);" in builders_c
    assert "(*out_op)->instance_flags = instance_flags;" in builders_c


def test_attr_params_uses_exact_parameterized_family() -> None:
    dialect = Dialect("test")
    options = ParameterizedAttrDef(
        "test.options",
        group=dialect,
        parameters=[AttrDef("width", ATTR_TYPE_I64, optional=True)],
    )
    op = Op(
        "test.query",
        group=dialect,
        attrs=[
            AttrDef(
                "requirements",
                ATTR_TYPE_PARAMETERIZED,
                parameterized_attr=options,
            )
        ],
        format=[AttrParams("requirements")],
    )

    tables_c = generate_tables_c("test", 0, [op], parameterized_attrs=[options])
    assert "{LOOM_FORMAT_KIND_ATTR_PARAMS, 0, 0}" in tables_c


def test_operand_dict_generates_format_and_builder_support() -> None:
    op = Op(
        "test.operand_dict",
        group=Dialect("test"),
        operands=[
            Operand("input", INTEGER),
            Operand("params", INTEGER, variadic=True),
        ],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("param_names", "dict", optional=True)],
        constraints=[SameType("input", "result")],
        format=[
            Ref("input"),
            OperandDict("params", "param_names"),
            ResultType("result"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_OPERAND_DICT" in tables_c
    assert "LOOM_OP_VTABLE_HAS_OPERAND_DICT" in tables_c
    assert "const loom_named_value_t* params" in ops_h
    assert "iree_host_size_t params_count" in ops_h
    assert "loom_make_named_value_slice(params, params_count)" in builders_c
    assert "&loom_op_attrs(*out_op)[0]" in builders_c


def test_attr_table_generates_format_and_builder_support() -> None:
    op = Op(
        "test.attr_table",
        group=Dialect("test"),
        operands=[
            Operand("selector", INTEGER),
            Operand("values", ANY, variadic=True),
        ],
        results=[Result("results", ANY, variadic=True)],
        attrs=[AttrDef("case_keys", ATTR_TYPE_I64_ARRAY)],
        format=[
            Ref("selector"),
            AttrTable("case_keys", "values"),
            COLON,
            ResultTypeList("results", parens=False),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_ATTR_TABLE" in tables_c
    assert "LOOM_OP_VTABLE_HAS_OPERAND_DICT" not in tables_c
    assert "const int64_t* case_keys" in ops_h
    assert "iree_host_size_t case_keys_count" in ops_h
    assert "const loom_value_id_t* values" in ops_h
    assert "iree_host_size_t values_count" in ops_h
    assert "loom_attr_i64_array(" in builders_c


def test_aligned_refs_generate_paired_format_and_uniform_results() -> None:
    op = Op(
        "test.aligned_refs",
        group=Dialect("test"),
        operands=[Operand("byte_lengths", INTEGER, variadic=True)],
        results=[
            Result("total_byte_length", INTEGER),
            Result("byte_offsets", INTEGER, variadic=True),
        ],
        attrs=[AttrDef("minimum_alignments", ATTR_TYPE_I64_ARRAY)],
        format=[
            AlignedRefs("byte_lengths", "minimum_alignments"),
            COLON,
            ResultTypeList("total_byte_length", parens=False, uniform=True),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "const loom_value_id_t* byte_lengths" in ops_h
    assert "const int64_t* minimum_alignments" in ops_h
    assert "loom_attr_i64_array(" in builders_c
    assert "LOOM_FORMAT_KIND_ALIGNED_REFS" in tables_c
    assert "LOOM_RESULT_TYPE_LIST_UNIFORM" in tables_c
