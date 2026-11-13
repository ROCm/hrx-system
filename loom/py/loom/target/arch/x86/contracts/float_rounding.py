# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact rounding modes over native scalar and packed floating carriers."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
    full_vector_type,
    value_type_guards,
)
from loom.target.arch.x86.vector_families import (
    FLOAT_ROUNDING_MODES,
    FloatRoundingFamily,
)
from loom.target.contracts import DescriptorRule, Guard, Scalar, ValueRef

_REGISTER_SUFFIXES = {
    16: "xmm",
    32: "xmm",
    64: "xmm",
    128: "xmm",
    256: "ymm",
    512: "zmm",
}


def float_rounding_rules(
    descriptor_lookup: DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    families: Sequence[FloatRoundingFamily],
    vector_bit_widths: Sequence[int],
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    rules = []
    for family in families:
        for scalar, widths in (
            (True, (family.element.bit_width,)),
            (False, vector_bit_widths),
        ):
            for width in widths:
                if width < family.element.bit_width:
                    continue
                value_type = (
                    Scalar(family.element.name)
                    if scalar
                    else full_vector_type(family.element, width)
                )
                # A one-lane vector uses the same instruction as a scalar.
                scalar_instruction = width == family.element.bit_width
                mnemonic = (
                    family.scalar_mnemonic
                    if scalar_instruction
                    else family.vector_mnemonic
                )
                suffix = _REGISTER_SUFFIXES[width]
                descriptor = descriptor_lookup(
                    f"{descriptor_key_prefix}.{mnemonic}.{suffix}"
                )
                operands = {"input": ValueRef.operand("input")}
                if scalar_instruction:
                    operands["passthrough"] = ValueRef.operand("input")
                for operation, control in FLOAT_ROUNDING_MODES:
                    source_op = (
                        getattr(scalar_math, f"scalar_{operation}")
                        if scalar
                        else getattr(vector, f"vector_{operation}")
                    )
                    rules.append(
                        DescriptorRule(
                            source_op=source_op,
                            descriptor=descriptor,
                            guards=(
                                *value_type_guards(("input", "result"), value_type),
                                Guard.descriptor_available(descriptor),
                            ),
                            emit=(
                                emit_descriptor_op(
                                    descriptor=descriptor,
                                    operands=operands,
                                    results={"dst": ValueRef.result("result")},
                                    immediates={"rounding": control},
                                ),
                            ),
                            priority=priority,
                        )
                    )
    return tuple(rules)
