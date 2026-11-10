# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar integer widening through Wasm's i32 and i64 carriers."""

from collections.abc import Callable

from loom.dialect.scalar import conversion
from loom.dsl import Op
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
)
from loom.target.low_descriptors import Descriptor


def _extension_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    source_bit_count: int,
    result_bit_count: int,
    source_op: Op,
    signedness: str,
    guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    current = ValueRef.operand("input")
    emits: list[EmitDescriptorOp] = []
    if source_bit_count < 32:
        # All destination widths share this same normalization. Keeping it in
        # Low also keeps the semantic extension available to source analyses.
        normalized = (
            ValueRef.temporary("normalized")
            if result_bit_count == 64
            else ValueRef.result("result")
        )
        normalized_type = {"dst": Scalar("i32")} if result_bit_count == 64 else None
        constant = ValueRef.temporary("normalization")
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_lookup("wasm.i32.const"),
                results={"dst": constant},
                result_types={"dst": Scalar("i32")},
                immediates={
                    "i32_value": (
                        32 - source_bit_count
                        if signedness == "s"
                        else (1 << source_bit_count) - 1
                    )
                },
                form=DescriptorEmitForm.CONST,
            )
        )
        if signedness == "s":
            positioned = ValueRef.temporary("positioned")
            emits.append(
                EmitDescriptorOp(
                    descriptor=descriptor_lookup("wasm.i32.shl"),
                    operands={"lhs": current, "rhs": constant},
                    results={"dst": positioned},
                    result_types={"dst": Scalar("i32")},
                )
            )
            current = positioned
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_lookup(
                    "wasm.i32.shr_s" if signedness == "s" else "wasm.i32.and"
                ),
                operands={"lhs": current, "rhs": constant},
                results={"dst": normalized},
                result_types=normalized_type,
            )
        )
        current = normalized
    if result_bit_count == 64:
        emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_lookup(f"wasm.i64.extend_i32_{signedness}"),
                operands={"input": current},
                results={"dst": ValueRef.result("result")},
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", Scalar(f"i{source_bit_count}")),
            Guard.value_type("result", Scalar(f"i{result_bit_count}")),
            *guards,
        ),
        emit=tuple(emits),
        priority=1 if guards else 0,
    )


def scalar_integer_extension_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[DescriptorRule, ...]:
    """Keeps every non-predicate integer extension semantic until Low emission."""

    rules: list[DescriptorRule] = []
    for source_bit_count in (8, 16, 32):
        for result_bit_count in (16, 32, 64):
            if source_bit_count >= result_bit_count:
                continue
            for source_op, signedness in (
                (conversion.scalar_extsi, "s"),
                (conversion.scalar_extui, "u"),
            ):
                rules.append(
                    _extension_rule(
                        descriptor_lookup,
                        source_bit_count,
                        result_bit_count,
                        source_op,
                        signedness,
                    )
                )
            if source_bit_count < 32:
                # A proved zero sign bit needs only unsigned normalization;
                # physical carrier bits above the source width remain arbitrary.
                rules.append(
                    _extension_rule(
                        descriptor_lookup,
                        source_bit_count,
                        result_bit_count,
                        conversion.scalar_extsi,
                        "u",
                        (
                            Guard.value_i64_range(
                                "input", 0, (1 << (source_bit_count - 1)) - 1
                            ),
                        ),
                    )
                )
    return tuple(rules)
