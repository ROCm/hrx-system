# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Representation limits and cross-field contracts for C metadata tables."""

import pytest

from loom.assembly import AssemblyFormat, BlockArgs, Region
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64,
    ATTR_TYPE_PREDICATE_LIST,
    ATTR_TYPE_SYMBOL,
    INTEGER,
    ISOLATED_FROM_ABOVE,
    PURE,
    SAFE_TO_SPECULATE,
    SYMBOL_DEFINE,
    TERMINATOR,
    AttrDef,
    Dialect,
    Effect,
    EnumCase,
    EnumDef,
    ExecutionCondition,
    HasAncestor,
    HasAnyAncestor,
    HasParent,
    ImplicitTerminator,
    NoAncestor,
    Op,
    OpCategory,
    Operand,
    OperandRole,
    Reads,
    ReadWrites,
    RegionDef,
    RegionExecution,
    Result,
    SameType,
    SymbolDefinition,
    SymbolReference,
    SymbolValueContract,
    Writes,
)
from loom.gen.ops.c_dialect import generate_dialect_contents
from loom.gen.ops.c_metadata_tables import generate_tables_c
from loom.gen.ops.model import DialectGeneration


@pytest.mark.parametrize(
    ("effect", "traits"),
    [
        (Reads(), "LOOM_TRAIT_READS_MEMORY"),
        (Writes(), "LOOM_TRAIT_WRITES_MEMORY"),
        (ReadWrites(), "LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY"),
    ],
)
def test_ambient_effects_do_not_access_operand_payloads(effect: Effect, traits: str) -> None:
    op = Op(
        "test.ambient",
        operands=[Operand("value", INTEGER)],
        effects=[effect],
    )
    source = generate_tables_c("test", 0, [op])
    assert f".traits = {traits}," in source
    assert "LOOM_OPERAND_READS" not in source
    assert "LOOM_OPERAND_WRITES" not in source


def test_generate_tables_emits_execution_condition() -> None:
    op = Op(
        "test.shift",
        group=Dialect("test"),
        operands=[Operand("lhs", INTEGER), Operand("rhs", INTEGER)],
        results=[Result("result", INTEGER)],
        constraints=[SameType("lhs", "rhs", "result")],
        traits=[PURE],
        execution_condition=ExecutionCondition.INTEGER_SHIFT_RIGHT,
        eager_execution_cost=1,
    )

    source = generate_tables_c("test", 0, [op])

    assert ".execution_condition = LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_RIGHT," in source
    assert ".eager_execution_cost = 1," in source


def test_generate_tables_emits_operand_role_and_summary_mask() -> None:
    op = Op(
        "test.project",
        group=Dialect("test"),
        operands=[
            Operand("source", ANY, role=OperandRole.ACCESS_PATH),
            Operand("indices", INTEGER, variadic=True, role=OperandRole.ACCESS_PATH),
        ],
        results=[Result("result", ANY)],
        traits=[PURE],
    )

    source = generate_tables_c("test", 0, [op])

    assert source.count("LOOM_OPERAND_ROLE_ACCESS_PATH") == 2
    assert ".operand_role_mask = LOOM_OPERAND_ROLE_MASK_ACCESS_PATH," in source


def test_generate_tables_requires_authored_eager_execution_policy() -> None:
    op = Op(
        "test.safe",
        group=Dialect("test"),
        traits=[PURE, SAFE_TO_SPECULATE],
    )

    with pytest.raises(ValueError, match="explicit eager_execution_cost"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_emits_dynamic_speculation_cost() -> None:
    op = Op(
        "test.dynamic",
        group=Dialect("test"),
        effective_traits="loom_test_dynamic_effective_traits",
        eager_execution_cost=1,
    )

    source = generate_tables_c("test", 0, [op])

    assert ".effective_traits = loom_test_dynamic_effective_traits," in source
    assert ".eager_execution_cost = 1," in source


@pytest.mark.parametrize("execution", [None, *RegionExecution])
@pytest.mark.parametrize("independent_target", [False, True])
def test_region_execution_metadata(execution: RegionExecution | None, independent_target: bool) -> None:
    op = Op(
        "test.region",
        group=Dialect("test"),
        attrs=[AttrDef("target", ATTR_TYPE_SYMBOL, symbol_ref=SymbolReference("target", ["target"]))] if independent_target else [],
        regions=[RegionDef("body", execution=execution, execution_target="target" if independent_target else None)],
        format=[Region("body")],
    )
    expected = execution if execution is not None else RegionExecution.ONCE
    source = generate_tables_c("test", 0, [op])
    assert f"{{LOOM_OP_KIND_UNKNOWN, LOOM_OP_KIND_UNKNOWN, 0, {expected.c_name}, {int(independent_target)}}}" in source


def test_capturing_multiple_regions_requires_declared_control_flow() -> None:
    op = Op(
        "test.regions",
        group=Dialect("test"),
        regions=[RegionDef("first"), RegionDef("second")],
        format=[Region("first"), Region("second")],
    )
    with pytest.raises(ValueError, match="capturing multiple regions requires a control-flow interface"):
        generate_tables_c("test", 0, [op])


def test_isolated_regions_have_independent_dynamic_owners() -> None:
    op = Op(
        "test.regions",
        group=Dialect("test"),
        regions=[RegionDef("first"), RegionDef("second")],
        traits=[ISOLATED_FROM_ABOVE],
        format=[Region("first"), Region("second")],
    )
    source = generate_tables_c("test", 0, [op])
    assert source.count("LOOM_REGION_EXECUTION_ONCE") == 2


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

    with pytest.raises(ValueError, match="value contract result 'types' must not be variadic"):
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

    with pytest.raises(ValueError, match="predicates 'predicates' must name a predicate_list"):
        generate_tables_c("test", 0x01, [op])


def test_generate_tables_marks_executable_predicates() -> None:
    op = Op(
        "test.assert",
        group=Dialect("test"),
        attrs=[
            AttrDef(
                "predicates",
                ATTR_TYPE_PREDICATE_LIST,
                executable_predicates=True,
            )
        ],
    )

    source = generate_tables_c("test", 0x01, [op])
    assert ".flags = LOOM_ATTR_EXECUTABLE_PREDICATES," in source
    assert ".vtable_flags = LOOM_OP_VTABLE_HAS_PREDICATE_LIST," in source


def test_generate_tables_marks_structural_materializations() -> None:
    op = Op(
        "test.materialize",
        group=Dialect("test"),
        results=[Result("result", ANY)],
        structural_materialization=True,
    )

    source = generate_tables_c("test", 0x01, [op])
    assert ".vtable_flags = LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION," in source


def test_rejects_duplicate_assembly_mnemonics() -> None:
    dialect = Dialect("test")
    ops = [Op(f"test.{name}", group=dialect, assembly=AssemblyFormat("copy")) for name in ("first", "second")]
    with pytest.raises(ValueError, match="duplicate assembly mnemonic"):
        generate_tables_c("test", 0x01, ops)


def test_assembly_format_preserves_region_signature_ownership() -> None:
    op = Op(
        "test.region",
        group=Dialect("test"),
        regions=[RegionDef("body")],
        format=[BlockArgs("body"), Region("body")],
        assembly=AssemblyFormat("region", [Region("body")]),
    )
    with pytest.raises(ValueError, match="must preserve region argument ownership"):
        generate_tables_c("test", 0x01, [op])


def test_generate_tables_rejects_constraint_field_index_above_6_bit_max() -> None:
    op = Op(
        "test.wide",
        group=Dialect("test"),
        operands=[Operand(f"input_{i}", INTEGER) for i in range(65)],
        constraints=[SameType("input_0", "input_64")],
    )

    with pytest.raises(
        ValueError,
        match=r"Op 'test\.wide' constraint SameType: field 'input_64' "
        r"index 64 exceeds LOOM_FIELD_REF 6-bit max 63",
    ):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_unknown_region_argument_uniform_scope() -> None:
    op = Op(
        "test.bad_region_scope",
        group=Dialect("test"),
        regions=[RegionDef("body", arg_uniform_scope="device")],
        format=[Region("body")],
    )

    with pytest.raises(
        ValueError,
        match=r"Op 'test\.bad_region_scope' region 'body' has unsupported "
        r"arg_uniform_scope 'device'",
    ):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_missing_region_execution_target() -> None:
    op = Op(
        "test.worker",
        group=Dialect("test"),
        regions=[RegionDef("body", execution_target="target")],
        format=[Region("body")],
    )
    with pytest.raises(ValueError, match="execution_target 'target' does not name an attribute"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_non_target_region_execution_target() -> None:
    op = Op(
        "test.worker",
        group=Dialect("test"),
        attrs=[AttrDef("target", ATTR_TYPE_I64)],
        regions=[RegionDef("body", execution_target="target")],
        format=[Region("body")],
    )
    with pytest.raises(ValueError, match="execution_target 'target' must reference a target symbol"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_emits_alternative_required_ancestors() -> None:
    dialect = Dialect("test")
    first = Op("test.first", group=dialect)
    second = Op("test.second", group=dialect)
    nested = Op(
        "test.nested",
        group=dialect,
        traits=[HasAnyAncestor("test.first", "test.second")],
    )

    source = generate_tables_c("test", 0, [first, second, nested])

    assert "loom_test_nested_required_any_ancestors[]" in source
    assert "LOOM_OP_TEST_FIRST" in source
    assert "LOOM_OP_TEST_SECOND" in source
    assert '.required_any_ancestor_names = "test.first or test.second"' in source
    assert ".required_any_ancestor_count = IREE_ARRAYSIZE(" in source


def test_generate_tables_rejects_duplicate_alternative_ancestors() -> None:
    dialect = Dialect("test")
    context = Op("test.context", group=dialect)
    nested = Op(
        "test.nested",
        group=dialect,
        traits=[HasAnyAncestor("test.context", "test.context")],
    )

    with pytest.raises(ValueError, match="contains duplicate op names"):
        generate_tables_c("test", 0, [context, nested])


@pytest.mark.parametrize(
    ("placement", "field"),
    [
        (HasParent, "required_parents"),
        (HasAncestor, "required_ancestors"),
        (HasAnyAncestor, "required_any_ancestors"),
        (NoAncestor, "forbidden_ancestors"),
    ],
)
def test_placement_references_another_declared_dialect(placement, field: str) -> None:
    context = Op("execution.context", group=Dialect("execution", c_path="ops/execution"))
    nested = Op("test.nested", group=Dialect("test"), traits=[placement(context.name)])

    source = generate_tables_c("test", 0, [nested], referenced_ops=[context])

    assert '#include "loom/ops/execution/ops.h"' in source
    assert f"loom_test_nested_{field}[] = {{\n    LOOM_OP_EXECUTION_CONTEXT,\n}};" in source
    assert "loom_execution_context_vtable" not in source


def test_region_terminators_reference_another_declared_dialect() -> None:
    terminator = Op("execution.end", group=Dialect("execution"), traits=[TERMINATOR])
    nested = Op(
        "test.region",
        group=Dialect("test"),
        regions=[RegionDef("body", terminator=terminator.name)],
        traits=[ImplicitTerminator(terminator.name)],
        format=[Region("body")],
    )

    source = generate_tables_c("test", 0, [nested], referenced_ops=[terminator])

    assert '#include "loom/ops/execution/ops.h"' in source
    assert "{LOOM_OP_EXECUTION_END, LOOM_OP_EXECUTION_END," in source
    assert "loom_execution_end_vtable" not in source


def test_placement_requires_the_referenced_declaration() -> None:
    nested = Op("test.nested", group=Dialect("test"), traits=[HasAnyAncestor("execution.context")])

    with pytest.raises(ValueError, match="must name a registered op"):
        generate_tables_c("test", 0, [nested])


@pytest.mark.parametrize("sharded", [False, True])
def test_dialect_generation_retains_placement_declarations(sharded: bool) -> None:
    dialect = Dialect("test")
    context = Op("test.context", group=dialect)
    external = Op("execution.context", group=Dialect("execution"))
    nested = Op(
        "test.nested",
        group=dialect,
        traits=[HasAnyAncestor(context.name, external.name)],
    )
    generation = DialectGeneration(
        dialect=dialect,
        ops=[context, nested],
        table_shards=[(OpCategory("context"), [context]), (OpCategory("nested"), [nested])] if sharded else None,
        referenced_ops=[external],
    )

    files = generate_dialect_contents(generation)
    source = files["tables/nested.c" if sharded else "tables.c"]

    assert '#include "loom/ops/execution/ops.h"' in source
    assert "loom_test_nested_required_any_ancestors[] = {\n    LOOM_OP_TEST_CONTEXT,\n    LOOM_OP_EXECUTION_CONTEXT,\n};" in source
    assert "loom_execution_context_vtable" not in source
    assert "LOOM_OP_EXECUTION_CONTEXT" not in files["ops.h"]


def test_constraint_count_fits_vtable_storage() -> None:
    for count in (255, 256):
        op = Op(
            "test.constraints",
            group=Dialect("test"),
            operands=[Operand("input", ANY)],
            constraints=[SameType("input")] * count,
        )
        if count == 255:
            generate_tables_c("test", 0, [op])
        else:
            with pytest.raises(ValueError, match="constraint count exceeds uint8_t capacity"):
                generate_tables_c("test", 0, [op])


def test_external_enum_conflicting_names_rejected() -> None:
    dialect = Dialect("test")
    ops = [
        Op(
            f"test.mode{value}",
            group=dialect,
            attrs=[
                AttrDef(
                    "mode",
                    "enum",
                    enum_def=EnumDef(
                        "Mode",
                        [EnumCase(keyword, value)],
                        c_type="loom_shared_mode_t",
                        c_const_prefix="LOOM_SHARED_MODE",
                    ),
                )
            ],
        )
        for value, keyword in enumerate(("fast", "slow"))
    ]
    with pytest.raises(ValueError, match="conflicting case sets"):
        generate_tables_c("test", 0, ops)
