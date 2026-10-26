# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution and predication obligations at descriptor construction."""

from dataclasses import replace

import pytest

from loom.gen.target.low import validation
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    DescriptorFlag,
    Effect,
    EffectKind,
    MemorySpace,
    Operand,
    OperandFlag,
    OperandRole,
    RegClassAlt,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_STATE_ASSIGN_I32_DESCRIPTOR,
    TEST_LOW_STATE_ASSIGN_I32_IMMEDIATE_DESCRIPTOR,
    TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
)

_TOTAL_ADD = replace(
    TEST_LOW_ADD_I32_DESCRIPTOR,
    flags=(DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.SAFE_TO_SPECULATE),
)
_MASK = Operand(
    "active",
    OperandRole.IMPLICIT,
    (RegClassAlt("test.mask"),),
    flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ, OperandFlag.SCHEDULE_ONLY_STATE, OperandFlag.EXECUTION_MASK),
)


def test_total_arithmetic_can_read_a_lane_mask_or_unchanged_state() -> None:
    validation.validate_descriptor_speculation(_TOTAL_ADD)
    for state in (_MASK, replace(_MASK, flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ))):
        descriptor = replace(_TOTAL_ADD, operands=(*_TOTAL_ADD.operands, state))
        validation.validate_descriptor_operands(descriptor)
        validation.validate_descriptor_speculation(descriptor)


@pytest.mark.parametrize("kind", [EffectKind.READ, EffectKind.WRITE, EffectKind.CONVERGENT])
def test_speculation_rejects_observable_or_collective_effects(kind: EffectKind) -> None:
    descriptor = replace(_TOTAL_ADD, effects=(Effect(kind, MemorySpace.GLOBAL),))
    with pytest.raises(ValueError, match="effect-free"):
        validation.validate_descriptor_speculation(descriptor)


def test_speculation_rejects_hidden_flag_writes() -> None:
    state = replace(_MASK, flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE))
    descriptor = replace(_TOTAL_ADD, operands=(*_TOTAL_ADD.operands, state))
    with pytest.raises(ValueError, match="cannot write architectural state"):
        validation.validate_descriptor_speculation(descriptor)


def test_mask_narrowing_requires_the_incoming_mask() -> None:
    output = replace(_MASK, field_name="next_active", flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE, OperandFlag.NARROWS_EXECUTION_MASK))
    descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, operands=(*TEST_LOW_ADD_I32_DESCRIPTOR.operands, output))
    with pytest.raises(ValueError, match="must read the same execution mask"):
        validation.validate_descriptor_operands(descriptor)
    validation.validate_descriptor_operands(replace(descriptor, operands=(*descriptor.operands, _MASK)))


def test_execution_mask_is_not_an_arbitrary_state_annotation() -> None:
    state = replace(_MASK, flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE, OperandFlag.EXECUTION_MASK))
    descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, operands=(*TEST_LOW_ADD_I32_DESCRIPTOR.operands, state))
    with pytest.raises(ValueError, match="implicit schedule-only state read"):
        validation.validate_descriptor_operands(descriptor)


_STATE_CLASSES = {register_class.name: register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes}


def test_state_assignment_accepts_explicit_values_and_immediates() -> None:
    for descriptor in (TEST_LOW_STATE_ASSIGN_I32_DESCRIPTOR, TEST_LOW_STATE_ASSIGN_I32_IMMEDIATE_DESCRIPTOR):
        validation.validate_descriptor_operands(descriptor)
        validation.validate_descriptor_state_assignment(descriptor, _STATE_CLASSES)
    validation.validate_descriptor_state_assignment(TEST_LOW_ADD_I32_DESCRIPTOR, _STATE_CLASSES)


@pytest.mark.parametrize("kind", [EffectKind.READ, EffectKind.WRITE, EffectKind.CALL, EffectKind.CONVERGENT, EffectKind.COUNTER])
def test_state_assignment_rejects_additional_effects(kind: EffectKind) -> None:
    descriptor = replace(TEST_LOW_STATE_ASSIGN_I32_DESCRIPTOR, effects=(Effect(kind),))
    with pytest.raises(ValueError, match="no other effects"):
        validation.validate_descriptor_state_assignment(descriptor, _STATE_CLASSES)


@pytest.mark.parametrize("flag", [DescriptorFlag.BARRIER, DescriptorFlag.TERMINATOR, DescriptorFlag.UNIQUE_IDENTITY, DescriptorFlag.SAFE_TO_SPECULATE])
def test_state_assignment_rejects_conflicting_semantics(flag: DescriptorFlag) -> None:
    descriptor = replace(TEST_LOW_STATE_ASSIGN_I32_DESCRIPTOR, flags=(DescriptorFlag.STATE_ASSIGNMENT, flag))
    with pytest.raises(ValueError, match="no other effects"):
        validation.validate_descriptor_state_assignment(descriptor, _STATE_CLASSES)


@pytest.mark.parametrize("count", [0, 2])
def test_state_assignment_requires_one_write(count: int) -> None:
    assignment = TEST_LOW_STATE_ASSIGN_I32_IMMEDIATE_DESCRIPTOR
    descriptor = replace(assignment, operands=assignment.operands * count)
    with pytest.raises(ValueError, match="exactly one state register"):
        validation.validate_descriptor_state_assignment(descriptor, _STATE_CLASSES)


def test_state_assignment_rejects_state_reads_and_unrelated_results() -> None:
    assignment = TEST_LOW_STATE_ASSIGN_I32_DESCRIPTOR
    read = replace(_MASK, reg_alts=assignment.operands[-1].reg_alts)
    with pytest.raises(ValueError, match="cannot depend on architectural state"):
        validation.validate_descriptor_state_assignment(replace(assignment, operands=(*assignment.operands, read)), _STATE_CLASSES)
    with pytest.raises(ValueError, match="only have explicit inputs"):
        validation.validate_descriptor_state_assignment(replace(assignment, operands=(TEST_LOW_ADD_I32_DESCRIPTOR.operands[0], *assignment.operands)), _STATE_CLASSES)


def test_state_assignment_requires_whole_singleton_state() -> None:
    assignment = TEST_LOW_STATE_ASSIGN_I32_IMMEDIATE_DESCRIPTOR
    partial_write = replace(
        assignment.operands[0],
        reg_alts=(replace(assignment.operands[0].reg_alts[0], register_part="low_half"),),
    )
    for write in (replace(assignment.operands[0], unit_count=2), partial_write):
        with pytest.raises(ValueError, match="whole state register"):
            validation.validate_descriptor_state_assignment(replace(assignment, operands=(write,)), _STATE_CLASSES)
    state_class = _STATE_CLASSES[assignment.operands[0].reg_alts[0].reg_class]
    with pytest.raises(ValueError, match="singleton architectural state"):
        validation.validate_descriptor_state_assignment(assignment, {**_STATE_CLASSES, state_class.name: replace(state_class, allocatable_count=2)})


def test_implicit_state_assignment_preserves_generic_effects() -> None:
    assignment = replace(TEST_LOW_STATE_ASSIGN_I32_IMMEDIATE_DESCRIPTOR, flags=(DescriptorFlag.STATE_ASSIGNMENT,))
    with pytest.raises(ValueError, match="retain its side effect"):
        validation.validate_descriptor_state_assignment(assignment, _STATE_CLASSES)


_COMMUTATIVE_WRITE = replace(
    _MASK,
    flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE, OperandFlag.COMMUTATIVE_STATE_UPDATE),
)
_COMMUTATIVE_ADD = replace(TEST_LOW_ADD_I32_DESCRIPTOR, operands=(*TEST_LOW_ADD_I32_DESCRIPTOR.operands, _COMMUTATIVE_WRITE))


def test_commutative_update_retains_the_native_state_write() -> None:
    validation.validate_descriptor_operands(_COMMUTATIVE_ADD)


@pytest.mark.parametrize(
    "operand",
    [
        replace(_COMMUTATIVE_WRITE, role=OperandRole.OPERAND),
        replace(_COMMUTATIVE_WRITE, flags=(OperandFlag.IMPLICIT, OperandFlag.COMMUTATIVE_STATE_UPDATE)),
        replace(_COMMUTATIVE_WRITE, flags=(*_COMMUTATIVE_WRITE.flags, OperandFlag.STATE_READ)),
    ],
)
def test_commutative_update_requires_an_implicit_write(operand: Operand) -> None:
    descriptor = replace(_COMMUTATIVE_ADD, operands=(*TEST_LOW_ADD_I32_DESCRIPTOR.operands, operand))
    with pytest.raises(ValueError, match="must be an implicit state write without a state read"):
        validation.validate_descriptor_operands(descriptor)


def test_commutative_update_cannot_produce_an_explicit_state_snapshot() -> None:
    snapshot = replace(_COMMUTATIVE_WRITE, role=OperandRole.RESULT)
    descriptor = replace(_COMMUTATIVE_ADD, operands=(snapshot, *TEST_LOW_ADD_I32_DESCRIPTOR.operands))
    with pytest.raises(ValueError, match="must be an implicit state write"):
        validation.validate_descriptor_operands(descriptor)


@pytest.mark.parametrize(
    "operand",
    [
        replace(_COMMUTATIVE_WRITE, unit_count=2),
        replace(
            _COMMUTATIVE_WRITE,
            reg_alts=(replace(_COMMUTATIVE_WRITE.reg_alts[0], register_part="low_half"),),
        ),
    ],
)
def test_commutative_update_requires_whole_state(operand: Operand) -> None:
    descriptor = replace(_COMMUTATIVE_ADD, operands=(*TEST_LOW_ADD_I32_DESCRIPTOR.operands, operand))
    with pytest.raises(ValueError, match="must update a whole state register"):
        validation.validate_descriptor_operands(descriptor)


def test_replacement_is_not_a_commutative_update() -> None:
    descriptor = replace(_COMMUTATIVE_ADD, flags=(DescriptorFlag.STATE_ASSIGNMENT,))
    with pytest.raises(ValueError, match="state assignment cannot promise commutative updates"):
        validation.validate_descriptor_operands(descriptor)


def test_idempotent_update_accepts_result_coupled_commutative_state() -> None:
    validation.validate_descriptor_idempotent_state_update(TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR, _STATE_CLASSES)
    validation.validate_descriptor_idempotent_state_update(TEST_LOW_ADD_I32_DESCRIPTOR, _STATE_CLASSES)


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        (
            replace(TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR, flags=(DescriptorFlag.IDEMPOTENT_STATE_UPDATE,)),
            "couple its updates",
        ),
        (
            replace(TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR, operands=TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands[1:]),
            "produce an SSA result",
        ),
        (
            replace(
                TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
                operands=TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands[:2],
            ),
            "must update architectural state",
        ),
        (
            replace(
                TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
                operands=(
                    *TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands[:2],
                    replace(
                        TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands[2],
                        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
                    ),
                ),
            ),
            "implicit commutative updates",
        ),
        (
            replace(
                TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
                constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
            ),
            "cannot transfer input storage ownership",
        ),
    ],
)
def test_idempotent_update_rejects_incomplete_contracts(descriptor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validation.validate_descriptor_idempotent_state_update(descriptor, _STATE_CLASSES)


def test_idempotent_update_rejects_independent_effects_and_state_feedback() -> None:
    descriptor = replace(
        TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
        effects=(Effect(EffectKind.READ, MemorySpace.GLOBAL),),
    )
    with pytest.raises(ValueError, match="no independent effects"):
        validation.validate_descriptor_idempotent_state_update(descriptor, _STATE_CLASSES)

    write = TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands[2]
    same_state_read = replace(
        write,
        field_name="state_in",
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
    )
    descriptor = replace(
        TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR,
        operands=(*TEST_LOW_STATE_UPDATE_I32_DESCRIPTOR.operands, same_state_read),
    )
    with pytest.raises(ValueError, match="cannot read a state register it updates"):
        validation.validate_descriptor_idempotent_state_update(descriptor, _STATE_CLASSES)
