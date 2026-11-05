# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declaration contracts required by generated interface consumers."""

from dataclasses import replace

import pytest

from loom.assembly import AttrDict, Flags, FuncArgs, Ref
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_SYMBOL,
    INTEGER,
    PAYLOAD_SCALAR,
    STORAGE_RELATION,
    TERMINATOR,
    UNIQUE_IDENTITY,
    AttrDef,
    CachePolicyInterface,
    CallLikeInterface,
    ConditionForwardedCountMatchesBlockArgs,
    ConditionForwardedTypesMatchBlockArgs,
    Constraint,
    Dialect,
    DirectBranchInterface,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    IterArgsMatchResults,
    LoopLikeInterface,
    MemoryAccessInterface,
    Op,
    Operand,
    Reads,
    RegionBranchInterface,
    RegionDef,
    RegionExecution,
    Result,
    Successor,
    TargetFactSpecialization,
    TargetLikeInterface,
    Trait,
    ValueDuplicationInterface,
    VariadicValuesMatch,
    YieldCountMatches,
    YieldTypesMatch,
)
from loom.gen.ops.c_metadata_tables import generate_tables_c


def _direct_branch_test_op(
    *,
    interface: DirectBranchInterface | None = None,
    operands: list[Operand] | None = None,
    successors: list[Successor] | None = None,
    traits: list[Trait] | None = None,
) -> Op:
    return Op(
        "test.br",
        group=Dialect("test"),
        operands=([Operand("args", ANY, variadic=True)] if operands is None else operands),
        successors=([Successor("dest")] if successors is None else successors),
        traits=[TERMINATOR] if traits is None else traits,
        interfaces=[interface or DirectBranchInterface(args="args", dest="dest")],
    )


def test_generate_tables_marks_direct_branch_interface() -> None:
    source = generate_tables_c("test", 0, [_direct_branch_test_op()])

    assert "LOOM_OP_VTABLE_DIRECT_BRANCH" in source


def test_generate_tables_rejects_direct_branch_without_terminator() -> None:
    with pytest.raises(
        ValueError,
        match=r"DirectBranchInterface on 'test\.br': operation must be a terminator",
    ):
        generate_tables_c("test", 0, [_direct_branch_test_op(traits=[])])


def test_generate_tables_rejects_direct_branch_without_complete_payload() -> None:
    with pytest.raises(
        ValueError,
        match=r"args 'args' must name the sole variadic operand field",
    ):
        generate_tables_c(
            "test",
            0,
            [_direct_branch_test_op(operands=[Operand("condition", INTEGER)])],
        )


def test_generate_tables_rejects_direct_branch_with_multiple_successors() -> None:
    with pytest.raises(
        ValueError,
        match=r"dest 'dest' must name the sole successor",
    ):
        generate_tables_c(
            "test",
            0,
            [_direct_branch_test_op(successors=[Successor("dest"), Successor("other")])],
        )


def _value_duplication_test_op(
    *,
    interface: ValueDuplicationInterface | None = None,
    operands: list[Operand] | None = None,
    results: list[Result] | None = None,
    constraints: list[Constraint] | None = None,
    traits: list[Trait] | None = None,
    facts: str | None = "loom_test_duplicate_facts",
) -> Op:
    return Op(
        "test.duplicate",
        group=Dialect("test"),
        operands=([Operand("sources", PAYLOAD_SCALAR, variadic=True)] if operands is None else operands),
        results=([Result("duplicates", PAYLOAD_SCALAR, variadic=True)] if results is None else results),
        constraints=([VariadicValuesMatch("duplicates", "sources")] if constraints is None else constraints),
        traits=[UNIQUE_IDENTITY, STORAGE_RELATION] if traits is None else traits,
        interfaces=[interface or ValueDuplicationInterface(sources="sources", duplicates="duplicates")],
        facts=facts,
    )


def test_generate_tables_marks_value_duplication_interface() -> None:
    source = generate_tables_c("test", 0, [_value_duplication_test_op()])

    assert "LOOM_OP_VTABLE_VALUE_DUPLICATION" in source


def test_generate_tables_rejects_value_duplication_without_unique_identity() -> None:
    with pytest.raises(
        ValueError,
        match=r"ValueDuplicationInterface on 'test\.duplicate': operation must declare UniqueIdentity",
    ):
        generate_tables_c(
            "test",
            0,
            [_value_duplication_test_op(traits=[STORAGE_RELATION])],
        )


def test_generate_tables_rejects_value_duplication_without_positional_remap() -> None:
    with pytest.raises(
        ValueError,
        match=r"operation must declare VariadicValuesMatch",
    ):
        generate_tables_c(
            "test",
            0,
            [_value_duplication_test_op(constraints=[])],
        )


def test_generate_tables_rejects_value_duplication_without_fact_transfer() -> None:
    with pytest.raises(
        ValueError,
        match=r"operation must provide fact inference",
    ):
        generate_tables_c(
            "test",
            0,
            [_value_duplication_test_op(facts=None)],
        )


def _target_projection_test_op(
    *,
    fact_type: str | None = None,
    fact_projector: str | None = None,
    fact_specialization: TargetFactSpecialization = TargetFactSpecialization.EXACT,
) -> Op:
    kind = EnumDef("TargetKind", [EnumCase("generic", 0)])
    return Op(
        "test.target",
        group=Dialect("test"),
        attrs=[
            AttrDef("symbol", ATTR_TYPE_SYMBOL),
            AttrDef("kind", ATTR_TYPE_ENUM, enum_def=kind),
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


def test_generate_target_projection_rejects_projector_without_family_facts() -> None:
    with pytest.raises(ValueError, match=r"family fact projector requires an external fact type"):
        generate_tables_c(
            "test",
            0,
            [_target_projection_test_op(fact_projector="loom_test_fact_projector")],
        )


def test_generate_target_projection_rejects_split_fact_ownership() -> None:
    with pytest.raises(ValueError, match=r"external fact type owns its specialization"):
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

    with pytest.raises(
        ValueError,
        match=r"FuncLikeInterface on 'test\.func': attr 'representation' referenced "
        r"by 'repr_contract' must have type 'string', got 'i64'",
    ):
        generate_tables_c("test", 0, [op])


def _make_func_with_unreachable(unreachable: str) -> Op:
    return Op(
        "test.func",
        group=Dialect("test"),
        operands=[Operand("args", ANY, variadic=True)],
        attrs=[AttrDef("callee", ATTR_TYPE_SYMBOL)],
        interfaces=[
            FuncLikeInterface(
                callee="callee",
                args="args",
                unreachable=unreachable,
            )
        ],
        format=[FuncArgs("args")],
    )


def test_generate_tables_resolves_func_unreachable_terminator() -> None:
    terminator = Op("test.unreachable", group=Dialect("test"), traits=[TERMINATOR])

    source = generate_tables_c(
        "test",
        0,
        [_make_func_with_unreachable(terminator.name), terminator],
    )

    assert ".unreachable_terminator = LOOM_OP_TEST_UNREACHABLE" in source


def test_generate_tables_rejects_unregistered_func_unreachable_terminator() -> None:
    with pytest.raises(ValueError, match=r"op kind 'test\.missing' must name a registered op"):
        generate_tables_c("test", 0, [_make_func_with_unreachable("test.missing")])


def test_generate_tables_rejects_invalid_func_unreachable_terminator() -> None:
    non_terminator = Op("test.unreachable", group=Dialect("test"))
    with pytest.raises(ValueError, match=r"op kind 'test\.unreachable' must be a terminator"):
        generate_tables_c(
            "test",
            0,
            [_make_func_with_unreachable(non_terminator.name), non_terminator],
        )

    variadic_terminator = Op(
        "test.unreachable",
        group=Dialect("test"),
        operands=[Operand("values", ANY, variadic=True)],
        traits=[TERMINATOR],
    )
    with pytest.raises(ValueError, match=r"must have no operands, results, attrs, regions, or successors"):
        generate_tables_c(
            "test",
            0,
            [_make_func_with_unreachable(variadic_terminator.name), variadic_terminator],
        )


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
            YieldCountMatches("body", "results"),
            YieldTypesMatch("body", "results"),
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
                results="results",
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


def test_loop_like_retains_repeated_region_execution() -> None:
    source = _generate_counted_loop_tables(_make_counted_loop_op())
    assert "LOOM_REGION_EXECUTION_REPEATED" in source


def test_loop_like_rejects_region_execution_override() -> None:
    op = _make_counted_loop_op()
    op = replace(op, regions=(replace(op.regions[0], execution=RegionExecution.ONCE),))
    with pytest.raises(ValueError, match="LoopLike owns region execution"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_loop_like_missing_yield_constraint() -> None:
    op = _make_counted_loop_op(
        constraints=[
            IterArgsMatchResults("iter_args", "results"),
            YieldCountMatches("body", "results"),
        ]
    )

    with pytest.raises(ValueError, match=r"LoopLikeInterface on 'test\.for': requires YieldTypesMatch"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_loop_like_partial_counted_range() -> None:
    op = _make_counted_loop_op(step=None)

    with pytest.raises(ValueError, match=r"LoopLikeInterface on 'test\.for': lower_bound, upper_bound, and step must be declared together"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_loop_like_unprojected_body_state() -> None:
    op = _make_counted_loop_op(body_arg_source=None)

    with pytest.raises(ValueError, match=r"LoopLikeInterface on 'test\.for': counted body 'body' must source carried arguments from 'iter_args'"):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_counted_loop_iv_type_mismatch() -> None:
    op = _make_counted_loop_op(iv_type="index")

    with pytest.raises(
        ValueError,
        match=r"LoopLikeInterface on 'test\.for': induction variable 'iv' "
        r"must use 'type_of:lower_bound'",
    ):
        _generate_counted_loop_tables(op)


def test_generate_tables_rejects_counted_loop_with_hidden_region() -> None:
    op = _make_counted_loop_op()
    op = replace(
        op,
        regions=(*op.regions, RegionDef("hidden", single_block=True, terminator="test.yield")),
    )

    with pytest.raises(
        ValueError,
        match=r"LoopLikeInterface on 'test\.for': counted loops require exactly 1 region\(s\), got 2",
    ):
        _generate_counted_loop_tables(op)


def _make_condition_loop_op(*, constraints: list[Constraint]) -> Op:
    return Op(
        "test.while",
        group=Dialect("test"),
        operands=[Operand("iter_args", ANY, variadic=True)],
        results=[Result("results", ANY, variadic=True)],
        regions=[
            RegionDef(
                "before",
                single_block=True,
                terminator="test.condition",
            ),
            RegionDef(
                "after",
                single_block=True,
                terminator="test.yield",
                arg_source="results",
            ),
        ],
        interfaces=[
            LoopLikeInterface(
                body="after",
                condition_region="before",
                iter_args="iter_args",
                results="results",
            )
        ],
        constraints=constraints,
    )


def _condition_loop_constraints() -> list[Constraint]:
    return [
        ConditionForwardedCountMatchesBlockArgs("before", "after", "results"),
        ConditionForwardedTypesMatchBlockArgs("before", "after", "results"),
        YieldCountMatches("after", "before"),
        YieldTypesMatch("after", "before"),
    ]


def _generate_condition_loop_tables(op: Op) -> None:
    condition_op = Op(
        "test.condition",
        group=Dialect("test"),
        operands=[
            Operand("condition", INTEGER),
            Operand("forwarded", ANY, variadic=True),
        ],
        traits=[TERMINATOR],
    )
    yield_op = Op(
        "test.yield",
        group=Dialect("test"),
        operands=[Operand("values", ANY, variadic=True)],
        traits=[TERMINATOR],
    )
    generate_tables_c("test", 0, [condition_op, yield_op, op])


def test_generate_tables_rejects_incomplete_condition_loop_contract() -> None:
    constraints = _condition_loop_constraints()
    constraints.pop(0)
    op = _make_condition_loop_op(constraints=constraints)

    with pytest.raises(
        ValueError,
        match=r"LoopLikeInterface on 'test\.while': requires "
        r"ConditionForwardedCountMatchesBlockArgs",
    ):
        _generate_condition_loop_tables(op)


def test_generate_tables_rejects_condition_loop_with_hidden_region() -> None:
    op = _make_condition_loop_op(constraints=_condition_loop_constraints())
    op = replace(
        op,
        regions=(*op.regions, RegionDef("hidden", single_block=True, terminator="test.yield")),
    )

    with pytest.raises(
        ValueError,
        match=r"LoopLikeInterface on 'test\.while': condition-controlled loops "
        r"require exactly 2 region\(s\), got 3",
    ):
        _generate_condition_loop_tables(op)


def test_generate_tables_rejects_condition_loop_using_body_as_condition() -> None:
    op = _make_condition_loop_op(constraints=_condition_loop_constraints())
    op = replace(
        op,
        interfaces=(
            LoopLikeInterface(
                body="after",
                condition_region="after",
                iter_args="iter_args",
                results="results",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"LoopLikeInterface on 'test\.while': condition and body must be distinct regions",
    ):
        _generate_condition_loop_tables(op)


def _region_branch_test_op(interface: RegionBranchInterface) -> Op:
    return Op(
        "test.branch",
        group=Dialect("test"),
        operands=[Operand("condition", INTEGER)],
        regions=[RegionDef("then_region"), RegionDef("else_region")],
        interfaces=[interface],
    )


def test_generate_tables_rejects_partial_boolean_region_branch_contract() -> None:
    op = _region_branch_test_op(RegionBranchInterface(selector="condition", true_region="then_region"))

    with pytest.raises(
        ValueError,
        match=r"RegionBranchInterface on 'test\.branch': true_region and false_region must be declared together",
    ):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_aliased_boolean_region_branch_contract() -> None:
    op = _region_branch_test_op(
        RegionBranchInterface(
            selector="condition",
            true_region="then_region",
            false_region="then_region",
        )
    )

    with pytest.raises(
        ValueError,
        match=r"RegionBranchInterface on 'test\.branch': true_region and false_region must be distinct",
    ):
        generate_tables_c("test", 0, [op])


def _cache_policy_test_op(
    names: tuple[str, ...],
    interface: CachePolicyInterface | None = None,
) -> Op:
    if interface is None:
        interface = CachePolicyInterface(cache_scope="scope", cache_temporal="temporal")
    cache_enums = {
        name: EnumDef(
            name,
            [EnumCase("default", 0)],
            c_type=f"loom_cache_{name}_t",
            c_const_prefix=f"LOOM_CACHE_{name.upper()}",
            c_include="loom/ops/cache.h",
        )
        for name in ("scope", "temporal")
    }
    return Op(
        "test.policy",
        group=Dialect("test"),
        attrs=[AttrDef(name, "enum", enum_def=cache_enums[name]) if name in cache_enums else AttrDef(name, "i64") for name in names],
        interfaces=[interface],
        format=[AttrDict()],
    )


def test_cache_policy_interface_rejects_missing_or_wrong_fields() -> None:
    with pytest.raises(ValueError, match="attr 'scope' not found"):
        generate_tables_c("test", 0, [_cache_policy_test_op(("renamed", "temporal"))])
    op = _cache_policy_test_op(("scope", "temporal"), CachePolicyInterface(cache_scope="temporal", cache_temporal="scope"))
    with pytest.raises(ValueError, match="must use the shared loom_cache_scope_t enum"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_memory_access_rejects_other_instance_flags() -> None:
    arithmetic_flags = EnumDef("ArithmeticFlags", [EnumCase("wrap", 1)])
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[Operand("view", ANY)],
        results=[Result("result", ANY)],
        attrs=[AttrDef("flags", ATTR_TYPE_FLAGS, enum_def=arithmetic_flags)],
        effects=[Reads("view")],
        interfaces=[MemoryAccessInterface()],
        format=[Flags("flags"), Ref("view")],
    )
    with pytest.raises(ValueError, match=r"MemoryAccessInterface on 'test\.load': instance flags must use the shared memory-access flag vocabulary"):
        generate_tables_c("test", 0, [op])


def test_generate_tables_memory_access_rejects_missing_effects() -> None:
    op = Op(
        "test.load",
        group=Dialect("test"),
        operands=[Operand("view", ANY)],
        results=[Result("result", ANY)],
        interfaces=[MemoryAccessInterface()],
    )

    with pytest.raises(ValueError, match=r"MemoryAccessInterface on 'test\.load': unable to infer memory access operation kind"):
        generate_tables_c("test", 0, [op])


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

    with pytest.raises(ValueError, match=r"MemoryAccessInterface on 'test\.load': byte_offset is mutually exclusive"):
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

    with pytest.raises(ValueError, match=r"MemoryAccessInterface on 'test\.store': operand 'payload' not found"):
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

    with pytest.raises(ValueError, match=r"CallLikeInterface on 'test\.call': operand 'operand' must be variadic"):
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

    with pytest.raises(ValueError, match=r"CallLikeInterface on 'test\.call': result 'result' must be variadic"):
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

    with pytest.raises(ValueError, match=r"CallLikeInterface on 'test\.call': results=None requires the operation to declare no results"):
        generate_tables_c("test", 0, [op])


def test_accepts_complete_loop_contracts() -> None:
    _generate_counted_loop_tables(_make_counted_loop_op())
    _generate_condition_loop_tables(_make_condition_loop_op(constraints=_condition_loop_constraints()))


def test_accepts_attribute_backed_and_fixed_cache_policies() -> None:
    for names in (("scope", "temporal"), ("temporal", "inserted", "scope")):
        generate_tables_c("test", 0, [_cache_policy_test_op(names)])
    generate_tables_c("test", 0, [Op("test.fixed", group=Dialect("test"), interfaces=[CachePolicyInterface(None, None)])])
