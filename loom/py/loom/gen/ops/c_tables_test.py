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
    Attr,
    AttrDict,
    AttrTable,
    BlockRef,
    Clause,
    Flags,
    OperandDict,
    OptionalGroup,
    PredicateList,
    Ref,
    Region,
    ResultType,
    ResultTypeList,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
    TypesOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_PREDICATE_LIST,
    ATTR_TYPE_SYMBOL,
    INTEGER,
    POOL,
    SYMBOL_DEFINE,
    VECTOR,
    AliasResult,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    Borrow,
    CallLikeInterface,
    ContractFamily,
    Dialect,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    EnumCase,
    EnumDef,
    HasParent,
    LiteralMatchesElementType,
    MemoryAccessInterface,
    Op,
    OpCategory,
    Operand,
    OpPhase,
    PackedPayloadBitCountMatchesStorage,
    PositiveBitWidthAttr,
    RegionDef,
    Result,
    Retain,
    RetainedResult,
    SameType,
    Successor,
    SymbolDefinition,
    TotalBitCountEqual,
    TypeConstraint,
    TypeDef,
    TypeSemantic,
    UnpackedPayloadBitCountMatchesStorage,
)
from loom.gen.ops import model as c_table_model
from loom.gen.ops.c_builders import generate_builders_c
from loom.gen.ops.c_enums import TYPE_CONSTRAINT_MAP
from loom.gen.ops.c_registry import generate_op_registry
from loom.gen.ops.c_tables import (
    generate_ops_h,
    generate_sharded_tables_c,
    generate_tables_aggregator_c,
    generate_tables_c,
    generate_type_registry,
)


@contextmanager
def _raises_value_error(pattern: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!s} did not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected ValueError matching {pattern!r}")


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


def test_type_constraint_map_covers_every_constraint() -> None:
    assert set(TYPE_CONSTRAINT_MAP) == set(TypeConstraint)


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

    assert "loom_type_semantics_t semantics;" in type_registry_h
    assert ".semantic = LOOM_TYPE_SEMANTIC_CONTROL_TOKEN," in type_registry_tables_c
    assert ".contract_families = LOOM_CONTRACT_KERNEL_ASYNC," in type_registry_tables_c


def test_generate_type_registry_emits_builtin_type_name_table() -> None:
    type_def = TypeDef(name="test.tensor", ir_kind="tensor")

    _, _, type_registry_tables_c = generate_type_registry([type_def])

    assert "loom_type_registry_builtin_names[LOOM_TYPE_COUNT_]" in type_registry_tables_c
    assert '[LOOM_TYPE_TENSOR] = IREE_SVL("test.tensor"),' in type_registry_tables_c


def test_generate_type_registry_rejects_duplicate_builtin_type_names() -> None:
    type_defs = [
        TypeDef(name="test.tensor", ir_kind="tensor"),
        TypeDef(name="test.other_tensor", ir_kind="tensor"),
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

    op_registry_h, op_registry_tables_h, op_registry_tables_c = generate_op_registry([(dialect, [])])

    assert "loom_op_registry_register_all_dialects" in op_registry_h
    assert "loom_op_registry_dialects[]" in op_registry_tables_h
    assert "loom_test_dialect_vtables" in op_registry_tables_c
    assert "loom_op_registry_register_dialect" not in op_registry_tables_c
    assert "iree_make_status" not in op_registry_tables_c


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


def test_generate_builders_keep_count_guard_for_optional_aggregate_attrs() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[AttrDef("dict", "dict", optional=True)],
        format=[AttrDict("dict")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT" not in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" not in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" not in builders_c
    assert "iree_any_bit_set(build_flags" not in builders_c
    assert "dict.count > 0" in builders_c


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

    tables_c = generate_tables_c("test", 0, [op, alias])

    assert ('{_BSTRING(8, "resource"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_OPERAND_OWNERSHIP_RETAIN, LOOM_OWNERSHIP_CARRIER_BY_VALUE}') in tables_c
    assert ('{_BSTRING(6, "result"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_RESULT_OWNERSHIP_RETAINED, LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE}') in tables_c
    assert ('{_BSTRING(6, "result"), LOOM_TYPE_CONSTRAINT_POOL, 0, LOOM_RESULT_OWNERSHIP_ALIAS, 0}') in tables_c


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
    assert ".operand_offset = 0," in tables_c
    assert ".result_offset = 0," in tables_c
    assert ".kind = LOOM_CALL_LIKE_KIND_SEMANTIC," in tables_c


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
        interfaces=[MemoryAccessInterface()],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "static const loom_memory_access_vtable_t loom_test_load_memory_access" in tables_c
    assert ".view_operand_index = 0," in tables_c
    assert ".value_operand_index = 255," in tables_c
    assert ".indices_operand_offset = 1," in tables_c
    assert ".static_indices_attr_index = 0," in tables_c
    assert ".cache_scope_attr_index = 255," in tables_c


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
    assert "const int64_t* case_keys" in ops_h
    assert "iree_host_size_t case_keys_count" in ops_h
    assert "const loom_value_id_t* values" in ops_h
    assert "iree_host_size_t values_count" in ops_h
    assert "loom_attr_i64_array(" in builders_c
