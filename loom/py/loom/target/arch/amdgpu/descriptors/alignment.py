# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Instruction register alignment, independent of allocation packing policy."""

from dataclasses import replace
from functools import cache

from loom.target.low_descriptors import Descriptor, DescriptorSet, Operand, RegClassAlt

from .common import _REG_SGPR


@cache
def _with_operand_register_alignment(
    alternative: RegClassAlt, unit_count: int
) -> RegClassAlt:
    if alternative.reg_class != _REG_SGPR or unit_count <= 1:
        return alternative
    unit_alignment = max(
        alternative.unit_alignment,
        2 if unit_count == 2 else 4,
    )
    if unit_alignment == alternative.unit_alignment:
        return alternative
    return replace(alternative, unit_alignment=unit_alignment)


@cache
def _with_operand_alignment_row(operand: Operand) -> Operand:
    alternatives = tuple(
        _with_operand_register_alignment(alternative, operand.unit_count)
        for alternative in operand.reg_alts
    )
    if all(
        alternative is existing
        for alternative, existing in zip(
            alternatives,
            operand.reg_alts,
            strict=True,
        )
    ):
        return operand
    return replace(operand, reg_alts=alternatives)


def _with_descriptor_operand_alignment(descriptor: Descriptor) -> Descriptor:
    operands = tuple(
        _with_operand_alignment_row(operand) for operand in descriptor.operands
    )
    if all(
        operand is existing
        for operand, existing in zip(operands, descriptor.operands, strict=True)
    ):
        return descriptor
    return replace(descriptor, operands=operands)


def _with_operand_alignment(descriptor_set: DescriptorSet) -> DescriptorSet:
    # Scalar instruction pairs require even bases; wider operands require
    # four-SGPR alignment. This applies to the SGPR alternative, not to a VGPR
    # alternative of the same source field. SBASE and SRSRC encoding padding
    # also imposes these requirements. Structural tuples have no instruction
    # requirement until a packet reads or writes them. CDNA vector alignment
    # is declared independently by its VGPR/AGPR register classes.
    descriptors = tuple(
        _with_descriptor_operand_alignment(descriptor)
        for descriptor in descriptor_set.descriptors
    )
    if all(
        descriptor is existing
        for descriptor, existing in zip(
            descriptors,
            descriptor_set.descriptors,
            strict=True,
        )
    ):
        return descriptor_set
    return replace(descriptor_set, descriptors=descriptors)
