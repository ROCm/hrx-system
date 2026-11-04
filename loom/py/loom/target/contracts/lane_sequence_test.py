# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generation-time invariants for state carried by repeated emit programs."""

from dataclasses import replace

import pytest

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
    LowerEmitKind,
    SourceValueKind,
    ValueRef,
    compile_lower_rule_set,
)
from loom.target.contracts.lower_rule_tables import (
    LOWER_EMIT_FLAG_SEQUENCE_SKIP_LAST_LANE,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_IOTA_PAIR_I32_DESCRIPTOR,
)


def _pair_rule() -> DescriptorRule:
    coordinate = ValueRef.temporary("coordinate")
    first = ValueRef.temporary("first")
    second = ValueRef.temporary("second")
    return DescriptorRule(
        source_op=vector.vector_iota,
        descriptor=TEST_LOW_IOTA_PAIR_I32_DESCRIPTOR,
        sequence_carried=("coordinate",),
        emit=(
            EmitRegisterCopy(
                source=ValueRef.operand("base"),
                result=coordinate,
                result_type=ValueRef.operand("base"),
            ),
            EmitDescriptorOp(
                descriptor=TEST_LOW_IOTA_PAIR_I32_DESCRIPTOR,
                operands={"base": coordinate, "step": ValueRef.operand("step")},
                results={"first": first, "second": second},
                result_types={"first": coordinate, "second": coordinate},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                operands={"lhs": second, "rhs": ValueRef.operand("step")},
                results={"dst": coordinate},
                result_types={"dst": coordinate},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                skip_last_lane=True,
            ),
            EmitRegisterConcat(
                sources=(first, second),
                result=ValueRef.result("result"),
                unit_count=2,
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )


def test_sequence_carried_slots_survive_multi_result_steps_and_skipped_updates() -> (
    None
):
    fragment = ContractFragment(
        name="test.sequence",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=(_pair_rule(),),
    )
    table = compile_lower_rule_set(fragment, dialect_ops={"vector": ALL_VECTOR_OPS})
    setup, step, update, collect = table.emits
    assert table.rules[0].temporary_count == 3
    assert step.result_ref_count == 2
    assert update.flags & LOWER_EMIT_FLAG_SEQUENCE_SKIP_LAST_LANE
    assert collect.kind is LowerEmitKind.REGISTER_CONCAT_PER_LANE_SEQUENCE
    assert collect.structural_unit_count == 2
    # The loop edge is represented entirely by the same slot in setup,
    # update, and the first read. Neither live collected result aliases it.
    setup_ref = table.value_refs[setup.result_bind_ref_start]
    carried = table.value_refs[step.operand_ref_start]
    update_ref = table.value_refs[update.result_ref_start]
    assert setup_ref.index == carried.index == update_ref.index
    result_slots = {
        table.value_refs[step.result_bind_ref_start + index].index for index in range(2)
    }
    assert len(result_slots) == 2
    assert carried.index not in result_slots


@pytest.mark.parametrize("names", [("coordinate", "coordinate"), ("",)])
def test_sequence_rejects_invalid_carried_names(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="unique and nonempty"):
        replace(_pair_rule(), sequence_carried=names).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


def test_sequence_rejects_missing_initial_state() -> None:
    rule = _pair_rule()
    with pytest.raises(
        ValueError, match="carried temporaries require a setup definition"
    ):
        replace(rule, emit=rule.emit[1:]).validate(TEST_LOW_CORE_DESCRIPTOR_SET)


def test_sequence_rejects_undeclared_carried_state() -> None:
    with pytest.raises(
        ValueError, match="skipped sequence results must update carried"
    ):
        replace(_pair_rule(), sequence_carried=()).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


def test_sequence_rejects_carrier_changes_across_the_iteration_edge() -> None:
    rule = _pair_rule()
    update = replace(rule.emit[2], result_types={"dst": ValueRef.operand("step")})
    with pytest.raises(ValueError, match="preserve its incoming result type"):
        replace(rule, emit=(*rule.emit[:2], update, rule.emit[3])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


@pytest.mark.parametrize(
    ("register_class", "unit_count"), [("test.ptr", 1), ("test.i32", 2)]
)
def test_sequence_checks_native_carrier_class_and_width(
    register_class: str, unit_count: int
) -> None:
    rule = _pair_rule()
    setup = EmitRegisterSlice(
        source=ValueRef.operand("base"),
        result=ValueRef.temporary("coordinate"),
        result_register_class=register_class,
        unit_count=unit_count,
    )
    update = replace(rule.emit[2], result_types={"dst": DescriptorResultType()})
    with pytest.raises(ValueError, match="preserve its incoming result type"):
        replace(rule, emit=(setup, rule.emit[1], update, rule.emit[3])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )
    setup = replace(setup, result_register_class="test.i32", unit_count=1)
    replace(rule, emit=(setup, rule.emit[1], update, rule.emit[3])).validate(
        TEST_LOW_CORE_DESCRIPTOR_SET
    )


def test_sequence_rejects_skipped_lane_local_definitions() -> None:
    rule = _pair_rule()
    step = replace(rule.emit[1], skip_last_lane=True)
    with pytest.raises(
        ValueError, match="skipped sequence results must update carried"
    ):
        replace(rule, emit=(rule.emit[0], step, *rule.emit[2:])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


def test_sequence_requires_update_for_each_carried_binding() -> None:
    rule = _pair_rule()
    with pytest.raises(
        ValueError, match="carried temporaries require a sequence update"
    ):
        replace(rule, emit=(*rule.emit[:2], rule.emit[3])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


def test_sequence_requires_final_collector() -> None:
    rule = _pair_rule()
    with pytest.raises(ValueError, match="concat must be the final emit"):
        replace(rule, emit=(*rule.emit[:2], rule.emit[3], rule.emit[2])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )


def test_sequence_concat_requires_packet_width() -> None:
    with pytest.raises(ValueError, match="unit count"):
        replace(_pair_rule().emit[3], unit_count=None)


def test_skip_last_requires_sequence_form() -> None:
    with pytest.raises(ValueError, match=r"skip-last.*per-lane-sequence"):
        replace(
            _pair_rule().emit[2],
            form=DescriptorEmitForm.OP,
            results={"dst": ValueRef.temporary("next")},
        ).validate(
            vector.vector_iota, TEST_LOW_CORE_DESCRIPTOR_SET, {"coordinate", "second"}
        )


@pytest.mark.parametrize(
    "kind",
    [
        SourceValueKind.SOURCE_MEMORY_ROOT,
        SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
        SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
        SourceValueKind.SOURCE_MEMORY_BYTE_OFFSET,
        SourceValueKind.SOURCE_MEMORY_ADDRESS,
        SourceValueKind.SOURCE_MEMORY_ELEMENT_STRIDE,
    ],
)
def test_sequence_keeps_address_materialization_in_shared_setup(
    kind: SourceValueKind,
) -> None:
    rule = _pair_rule()
    step = replace(
        rule.emit[1],
        operands={
            "base": ValueRef(kind=kind, field=""),
            "step": ValueRef.operand("step"),
        },
    )
    with pytest.raises(
        ValueError, match="address geometry must be materialized in shared setup"
    ):
        replace(rule, emit=(rule.emit[0], step, *rule.emit[2:])).validate(
            TEST_LOW_CORE_DESCRIPTOR_SET
        )
