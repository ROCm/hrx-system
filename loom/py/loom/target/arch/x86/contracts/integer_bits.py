# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integer bit-count recipes for baseline x86 scalar carriers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise as scalar
from loom.dsl import Op
from loom.target.arch.x86.descriptors import X86_INTEGER_CONDITION_CODES
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    DirectDescriptorCase,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    unary_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_I16 = Scalar("i16")
_I64 = Scalar("i64")
_INDEX = Scalar("index")
_EQUAL_CONDITION_CODE = dict(X86_INTEGER_CONDITION_CODES)["eq"]

_NATIVE_COUNT_OPERATIONS = (
    ("lzcnt", "ctlz", scalar.scalar_ctlzi, index.index_ctlzi),
    ("tzcnt", "cttz", scalar.scalar_cttzi, index.index_cttzi),
    ("popcnt", "ctpop", scalar.scalar_ctpopi, index.index_ctpopi),
)


class _ScalarIntegerRecipe:
    """Builds one closed count recipe over a scalar register carrier."""

    def __init__(
        self,
        bit_count: int,
        type_pattern: TypePattern,
        descriptor_lookup: Callable[[str], Descriptor],
    ) -> None:
        self.bit_count = bit_count
        self.type_pattern = type_pattern
        self.descriptor_lookup = descriptor_lookup
        self.emits: list[EmitDescriptorOp] = []

    @property
    def register_suffix(self) -> str:
        return f"gpr{self.bit_count}"

    def _result(self, name: str, result: ValueRef | None) -> ValueRef:
        return ValueRef.temporary(name) if result is None else result

    def constant(self, value: int, name: str) -> ValueRef:
        result = ValueRef.temporary(name)
        descriptor = self.descriptor_lookup(f"x86.scalar.movimm.{self.register_suffix}")
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": result},
                result_types={"dst": self.type_pattern},
                immediates={f"imm{self.bit_count}": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def binary(
        self,
        operation: str,
        lhs: ValueRef,
        rhs: ValueRef,
        name: str,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        output = self._result(name, result)
        descriptor = self.descriptor_lookup(
            f"x86.scalar.{operation}.{self.register_suffix}"
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": lhs, "rhs": rhs},
                results={"dst": output},
                result_types={"dst": self.type_pattern},
            )
        )
        return output

    def binary_immediate(
        self,
        operation: str,
        lhs: ValueRef,
        immediate: int,
        name: str,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        output = self._result(name, result)
        descriptor = self.descriptor_lookup(
            f"x86.scalar.{operation}.imm.{self.register_suffix}"
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": lhs},
                results={"dst": output},
                result_types={"dst": self.type_pattern},
                immediates={"imm32": immediate},
            )
        )
        return output

    def shift_right(
        self,
        value: ValueRef,
        amount: int,
        name: str,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        output = self._result(name, result)
        descriptor = self.descriptor_lookup(
            f"x86.scalar.shr.imm.{self.register_suffix}"
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": value},
                results={"dst": output},
                result_types={"dst": self.type_pattern},
                immediates={"shift": amount},
            )
        )
        return output

    def scan(self, operation: str, value: ValueRef, name: str) -> ValueRef:
        result = ValueRef.temporary(name)
        descriptor = self.descriptor_lookup(
            f"x86.scalar.{operation}.{self.register_suffix}"
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": value},
                results={"dst": result},
                result_types={"dst": self.type_pattern},
            )
        )
        return result

    def repair_zero(self, value: ValueRef, nonzero_count: ValueRef) -> None:
        width = self.constant(self.bit_count, "width")
        descriptor = self.descriptor_lookup(
            f"x86.scalar.cmp.select.imm.{self.register_suffix}.{self.register_suffix}"
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": value,
                    "true_value": width,
                    "false_value": nonzero_count,
                },
                results={"dst": ValueRef.result("result")},
                result_types={"dst": self.type_pattern},
                immediates={
                    "imm32": 0,
                    "predicate": _EQUAL_CONDITION_CODE,
                },
            )
        )


def _repeat_byte(byte: int, bit_count: int) -> int:
    return int.from_bytes(bytes((byte,)) * (bit_count // 8), "little")


def _population_count(recipe: _ScalarIntegerRecipe, value: ValueRef) -> None:
    bit_count = recipe.bit_count
    mask1_value = _repeat_byte(0x55, bit_count)
    mask2_value = _repeat_byte(0x33, bit_count)
    mask4_value = _repeat_byte(0x0F, bit_count)
    multiplier_value = _repeat_byte(0x01, bit_count)

    if bit_count == 64:
        mask1 = recipe.constant(mask1_value, "mask1")
        mask2 = recipe.constant(mask2_value, "mask2")
        mask4 = recipe.constant(mask4_value, "mask4")
    shifted1 = recipe.shift_right(value, 1, "shifted1")
    masked1 = (
        recipe.binary("and", shifted1, mask1, "masked1")
        if bit_count == 64
        else recipe.binary_immediate("and", shifted1, mask1_value, "masked1")
    )
    reduced1 = recipe.binary("sub", value, masked1, "reduced1")

    low2 = (
        recipe.binary("and", reduced1, mask2, "low2")
        if bit_count == 64
        else recipe.binary_immediate("and", reduced1, mask2_value, "low2")
    )
    shifted2 = recipe.shift_right(reduced1, 2, "shifted2")
    high2 = (
        recipe.binary("and", shifted2, mask2, "high2")
        if bit_count == 64
        else recipe.binary_immediate("and", shifted2, mask2_value, "high2")
    )
    reduced2 = recipe.binary("add", low2, high2, "reduced2")

    shifted4 = recipe.shift_right(reduced2, 4, "shifted4")
    summed4 = recipe.binary("add", reduced2, shifted4, "summed4")
    nibbles = (
        recipe.binary("and", summed4, mask4, "nibbles")
        if bit_count == 64
        else recipe.binary_immediate("and", summed4, mask4_value, "nibbles")
    )
    if bit_count == 64:
        multiplier = recipe.constant(multiplier_value, "multiplier")
        product = recipe.binary("imul", nibbles, multiplier, "product")
    else:
        product = recipe.binary_immediate("imul", nibbles, multiplier_value, "product")
    recipe.shift_right(
        product,
        bit_count - 8,
        "count",
        result=ValueRef.result("result"),
    )


def _bit_count_rule(
    source_op: Op,
    type_pattern: TypePattern,
    bit_count: int,
    descriptor_lookup: Callable[[str], Descriptor],
) -> DescriptorRule:
    recipe = _ScalarIntegerRecipe(bit_count, type_pattern, descriptor_lookup)
    value = ValueRef.operand("input")
    if source_op.name.endswith("ctlzi"):
        highest_bit = recipe.scan("bsr", value, "highest_bit")
        nonzero_count = recipe.binary_immediate(
            "xor", highest_bit, bit_count - 1, "nonzero_count"
        )
        recipe.repair_zero(value, nonzero_count)
    elif source_op.name.endswith("cttzi"):
        recipe.repair_zero(value, recipe.scan("bsf", value, "nonzero_count"))
    else:
        _population_count(recipe, value)
    return DescriptorRule(
        source_op=source_op,
        descriptor=recipe.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", type_pattern),
        ),
        emit=tuple(recipe.emits),
        report_key="integer_bit_count",
    )


def scalar_integer_bit_count_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[DescriptorRule, ...]:
    """Returns native and baseline recipes for scalar and index count ops."""
    native_cases = tuple(
        DirectDescriptorCase(
            index_op if type_pattern == _INDEX else scalar_op,
            descriptor_lookup(f"x86.scalar.{operation}.gpr{bit_count}"),
            type_pattern,
            semantic_tag=f"integer.bit_count.{semantic}.i{bit_count}",
            priority=1,
        )
        for type_pattern, bit_count in (
            (_I16, 16),
            (_I32, 32),
            (_I64, 64),
            (_INDEX, 64),
        )
        for operation, semantic, scalar_op, index_op in _NATIVE_COUNT_OPERATIONS
    )
    native_rules = unary_descriptor_rules(
        native_cases,
        form=DescriptorEmitForm.OP,
    )
    # Native counts are optional realizations. Guarding the primary descriptor
    # lets target legalization use the shared width legalization when its ISA
    # feature is unavailable; unguarded primaries deliberately survive target
    # queries so source-to-Low can diagnose a missing required feature.
    native_rules = tuple(
        replace(
            rule,
            guards=(
                *rule.guards,
                Guard.descriptor_available(case.descriptor),
            ),
        )
        for case, rule in zip(native_cases, native_rules, strict=True)
    )
    fallback_rules = tuple(
        _bit_count_rule(source_op, type_pattern, bit_count, descriptor_lookup)
        for type_pattern, bit_count, source_ops in (
            (
                _I32,
                32,
                (scalar.scalar_ctlzi, scalar.scalar_cttzi, scalar.scalar_ctpopi),
            ),
            (
                _I64,
                64,
                (scalar.scalar_ctlzi, scalar.scalar_cttzi, scalar.scalar_ctpopi),
            ),
            (
                _INDEX,
                64,
                (index.index_ctlzi, index.index_cttzi, index.index_ctpopi),
            ),
        )
        for source_op in source_ops
    )
    return (*native_rules, *fallback_rules)
