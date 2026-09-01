# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Representation helpers shared by HAL 0.0 instruction declarations."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    InstructionFieldRole,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
)
from model.isa.declarations import instruction_field, ref_register
from model.isa.validation import ZERO
from model.schema import U16, RuleUse


def hal_ref(
    name: str,
    offset: int,
    type_contract: str,
    *,
    null_policy: RefNullPolicy = RefNullPolicy.REQUIRED,
    ownership: RefOwnership = RefOwnership.BORROW,
    role: InstructionFieldRole = InstructionFieldRole.OPERAND,
    description: str | None = None,
):
    """Declares one exact typed HAL-page ref field."""

    return ref_register(
        name,
        offset,
        role,
        description or f"{null_policy.value} exact {type_contract} ref.",
        RuntimeRefPolicy(type_contract, null_policy, ownership),
    )


def hal_result_ref(name: str, offset: int, type_contract: str):
    """Declares one non-null owned result replacing a ref register."""

    return hal_ref(
        name,
        offset,
        type_contract,
        null_policy=RefNullPolicy.RESULT_NONNULL,
        ownership=RefOwnership.REPLACE_OWNER,
        role=InstructionFieldRole.RESULT,
    )


def zero_u16(name: str, offset: int):
    """Declares one naturally aligned canonical-zero u16 field."""

    return instruction_field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        (RuleUse(ZERO.entity_id),),
    )


def required_ref_failures(type_contract: str, atomicity: str):
    """Returns the universal null/type failures for a required exact ref."""

    return (
        FailureCase(
            "failed_precondition",
            f"A required {type_contract} ref is canonical null.",
            atomicity,
        ),
        FailureCase(
            "invalid_argument",
            f"A non-null ref does not have exact {type_contract} type.",
            atomicity,
        ),
    )
