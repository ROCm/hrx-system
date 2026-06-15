# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-low legality policy for source operation kinds.

Dialect declarations carry target-independent semantic phase and contract
metadata. This module derives the target-low lowering-stage policy from those
facts: target-low accepts executable ops directly, requires a target provider
for declared target contracts, and rejects source structure that must lower away
before executable code reaches this stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, unique

from loom.dsl import Dialect, Op, OpPhase


@unique
class TargetLowLegality(Enum):
    """Generated C class for target-low legality after provider verification."""

    UNSUPPORTED = "LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED"
    CORE = "LOOM_TARGET_LOW_LEGALITY_CORE"
    PROVIDER = "LOOM_TARGET_LOW_LEGALITY_PROVIDER"
    SOURCE_ONLY = "LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY"
    MODULE_METADATA = "LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA"


def target_low_legality_for_op(dialect: Dialect, op: Op) -> TargetLowLegality:
    """Derives the target-low legality class for one op declaration."""

    phase = op.phase if op.phase is not None else dialect.default_phase
    if op.contracts:
        if phase in (OpPhase.SOURCE_STRUCTURE, OpPhase.MODULE_METADATA):
            raise ValueError(
                f"op '{op.name}' cannot combine target contracts with "
                f"{phase.name.lower()} phase"
            )
        return TargetLowLegality.PROVIDER
    if phase == OpPhase.EXECUTABLE:
        return TargetLowLegality.CORE
    if phase == OpPhase.SOURCE_STRUCTURE:
        return TargetLowLegality.SOURCE_ONLY
    if phase == OpPhase.MODULE_METADATA:
        return TargetLowLegality.MODULE_METADATA
    return TargetLowLegality.UNSUPPORTED


def target_low_legality_by_name(
    dialects: Sequence[tuple[Dialect, Sequence[Op]]],
) -> dict[str, TargetLowLegality]:
    """Returns the derived target-low legality class by op name."""

    result: dict[str, TargetLowLegality] = {}
    for dialect, ops in dialects:
        for op in ops:
            if op.name in result:
                raise ValueError(f"op '{op.name}' has duplicate target-low legality")
            legality = target_low_legality_for_op(dialect, op)
            if legality != TargetLowLegality.UNSUPPORTED:
                result[op.name] = legality
    return result
