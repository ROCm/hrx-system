# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact scalar descriptor-program construction for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    ValueRef,
    descriptor_by_key,
)

_SHORT_MIN = -1024
_SHORT_MAX = 1023


class ScalarProgram:
    """Builds table-serializable scalar programs from AIE2P descriptors."""

    def __init__(self, temporary_prefix: str = "") -> None:
        self.emits: list[EmitDescriptorOp] = []
        self.temporary_prefix = temporary_prefix

    def temporary(self, name: str) -> ValueRef:
        return ValueRef.temporary(f"{self.temporary_prefix}{name}")

    def constant(
        self,
        result_name: str,
        value: int,
        *,
        descriptor_key: str | None = None,
    ) -> ValueRef:
        if descriptor_key is None:
            descriptor_key = (
                "amd.xdna.aie2p.constant.i32.short"
                if _SHORT_MIN <= value <= _SHORT_MAX
                else "amd.xdna.aie2p.constant.i32"
            )
        result = self.temporary(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, descriptor_key),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def unary(
        self, result_name: str | None, operation: str, operand: ValueRef
    ) -> ValueRef:
        return self.operation(
            result_name,
            operation,
            operands={"s0": operand},
        )

    def binary(
        self,
        result_name: str | None,
        operation: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            operation,
            operands={"s0": lhs, "s1": rhs},
        )

    def add_immediate(
        self, result_name: str, operand: ValueRef, immediate: int
    ) -> ValueRef:
        if not -64 <= immediate <= 63:
            immediate_value = self.constant(f"{result_name}_constant", immediate)
            return self.binary(result_name, "add.i32", operand, immediate_value)
        return self.operation(
            result_name,
            "add.i32.immediate",
            operands={"s0": operand},
            immediates={"imm": immediate},
        )

    def multiply_add(
        self,
        result_name: str,
        accumulator: ValueRef,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "madd.i32",
            operands={"a0": accumulator, "s0": lhs, "s1": rhs},
        )

    def select(
        self,
        result_name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
        *,
        when_zero: bool = False,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "select.zero.i32" if when_zero else "select.nonzero.i32",
            operands={"s0": true_value, "s1": false_value, "s2": condition},
            copy_operands=("s2",),
        )

    def operation(
        self,
        result_name: str | None,
        operation: str,
        *,
        operands: Mapping[str, ValueRef],
        immediates: Mapping[str, int] | None = None,
        copy_operands: Sequence[str] = (),
    ) -> ValueRef:
        result = (
            ValueRef.result("result")
            if result_name is None
            else self.temporary(result_name)
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor_by_key(
                    AIE2P_CORE_DESCRIPTOR_SET, f"amd.xdna.aie2p.{operation}"
                ),
                operands=operands,
                results={"d0": result},
                result_types=(
                    None if result_name is None else {"d0": DescriptorResultType()}
                ),
                immediates={} if immediates is None else immediates,
                form=DescriptorEmitForm.OP,
                copy_operands=copy_operands,
            )
        )
        return result
