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
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    HasAnyAncestor,
    Op,
    Operand,
    RegionDef,
    RegionExecution,
    Result,
    SameType,
    SymbolDefinition,
    SymbolValueContract,
)
from loom.gen.ops.c_metadata_tables import generate_tables_c


@pytest.mark.parametrize("execution", [None, *RegionExecution])
def test_region_execution_metadata(execution: RegionExecution | None) -> None:
    op = Op(
        "test.region",
        group=Dialect("test"),
        regions=[RegionDef("body", execution=execution)],
        format=[Region("body")],
    )
    expected = execution if execution is not None else RegionExecution.ONCE
    source = generate_tables_c("test", 0, [op])
    assert f"{{LOOM_OP_KIND_UNKNOWN, LOOM_OP_KIND_UNKNOWN, 0, {expected.c_name}}}" in source


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
