# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Managed object handle operations."""

from loom.assembly import ARROW, COLON, COMMA, Ref, ResultType, TypeOf
from loom.dsl import (
    I1,
    MANAGED_REFERENCE,
    OBSERVABLE_EFFECT,
    PURE,
    SAFE_TO_SPECULATE,
    UNIQUE_IDENTITY,
    AliasResult,
    Borrow,
    Dialect,
    Move,
    Op,
    Operand,
    OpPhase,
    Result,
    value_duplication_op,
)

ref_ops = Dialect(
    "ref",
    dialect_id=0x26,
    doc="Lifetime and identity operations on managed object handles.",
)

ref_duplicate = value_duplication_op(
    "ref.duplicate",
    group=ref_ops,
    carrier=MANAGED_REFERENCE,
    traits=[OBSERVABLE_EFFECT],
    lifetime_effects_only=True,
    doc=(
        "Create an independent handle to the same managed payload, preserving "
        "its type and null state. Duplicating ref adds an owner that must be "
        "released or transferred independently. Duplicating ptr preserves its "
        "borrowed mode and any protection needed through its final use. The "
        "source remains valid; live source and result lifetimes are independent."
    ),
    examples=[
        "%copy = ref.duplicate %owner : ref<@Object>",
        "%saved = ref.duplicate %pointer : ptr<@Object>",
    ],
)

ref_borrow = Op(
    "ref.borrow",
    group=ref_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Create a pointer to an owner's exact managed payload without consuming "
        "the owner. Null produces null. The pointer remains live through its "
        "final reachable use, even when the original owner is released first. "
        "The compiler acquires protection when it cannot prove a covering owner "
        "and retires that protection after the final use. The source owner's "
        "explicit lifetime is unchanged."
    ),
    operands=[Operand("owner", MANAGED_REFERENCE)],
    results=[Result("pointer", MANAGED_REFERENCE)],
    traits=[UNIQUE_IDENTITY, OBSERVABLE_EFFECT],
    lifetime_effects_only=True,
    ownership_effects=[Borrow("owner"), AliasResult("pointer", "owner")],
    verify="loom_ref_borrow_verify",
    format=[Ref("owner"), COLON, TypeOf("owner"), ARROW, ResultType("pointer")],
    examples=["%pointer = ref.borrow %owner : ref<@Frame> -> ptr<@Frame>"],
)

ref_release = Op(
    "ref.release",
    group=ref_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "End this managed handle's lifetime. The source identity is invalid "
        "afterward. An owning ref releases its owner and runs the native "
        "finalizer when it holds the last reference. A ptr retires any "
        "compiler-acquired protection; an unprotected borrow or null has no "
        "payload effect. Independent handles remain valid."
    ),
    operands=[Operand("value", MANAGED_REFERENCE)],
    traits=[OBSERVABLE_EFFECT],
    lifetime_effects_only=True,
    ownership_effects=[Move("value")],
    format=[Ref("value"), COLON, TypeOf("value")],
    examples=["ref.release %owner : ref<@Frame>"],
)

ref_null = Op(
    "ref.null",
    group=ref_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Create an independent null handle of the declared ref or ptr type. "
        "No payload is allocated or retained. An owning ref still has its own "
        "explicit lifetime and must be released or transferred on every normal "
        "path; releasing null has no payload effect."
    ),
    results=[Result("result", MANAGED_REFERENCE)],
    traits=[UNIQUE_IDENTITY],
    lifetime_effects_only=True,
    format=[COLON, ResultType("result")],
    examples=["%empty = ref.null : ref<@Frame>", "%missing = ref.null : ptr<@Frame>"],
)

ref_compare_null = Op(
    "ref.compare.null",
    group=ref_ops,
    builder_name="compare_null",
    phase=OpPhase.EXECUTABLE,
    doc=("Return true when the handle is null and false otherwise. The handle is borrowed without retaining, releasing, or inspecting its payload."),
    operands=[Operand("value", MANAGED_REFERENCE)],
    results=[Result("result", I1)],
    traits=[PURE, SAFE_TO_SPECULATE],
    ownership_effects=[Borrow("value")],
    eager_execution_cost=1,
    format=[Ref("value"), COLON, TypeOf("value")],
    examples=["%missing = ref.compare.null %frame : ptr<@Frame>"],
)

ref_compare_eq = Op(
    "ref.compare.eq",
    group=ref_ops,
    builder_name="compare_eq",
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Return true for two null handles or two handles to the same managed "
        "payload, and false otherwise. Both operands must name the same "
        "external payload type, but may have different ownership modes. "
        "Neither operand is retained, released, or dereferenced. Equality "
        "compares object identity, not payload contents."
    ),
    operands=[Operand("left", MANAGED_REFERENCE), Operand("right", MANAGED_REFERENCE)],
    results=[Result("result", I1)],
    traits=[PURE, SAFE_TO_SPECULATE],
    ownership_effects=[Borrow("left"), Borrow("right")],
    eager_execution_cost=1,
    verify="loom_ref_compare_eq_verify",
    format=[Ref("left"), COMMA, Ref("right"), COLON, TypeOf("left"), COMMA, TypeOf("right")],
    examples=["%same = ref.compare.eq %owner, %frame : ref<@Frame>, ptr<@Frame>"],
)

ALL_REF_OPS: tuple[Op, ...] = (
    ref_duplicate,
    ref_borrow,
    ref_release,
    ref_null,
    ref_compare_null,
    ref_compare_eq,
)
