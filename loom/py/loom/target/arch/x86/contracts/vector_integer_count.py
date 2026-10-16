# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packed integer bit-count recipes for baseline x86 vector carriers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.x86.contracts.predicate import (
    avx2_predicate_register_class,
    avx2_predicate_representation_bit_widths,
    avx2_predicate_representation_id,
)
from loom.target.arch.x86.contracts.rule_builders import (
    VectorCarrier,
    vector_carriers,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX512_VECTOR_BIT_WIDTHS,
    INTEGER_ELEMENTS,
    VectorElement,
)
from loom.target.contracts import (
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ReadOnlyDataLiteral,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

type _CountKind = Literal["leading", "trailing", "population"]


@dataclass(frozen=True, slots=True)
class _CountOperation:
    source_op: Op
    kind: _CountKind


_COUNT_OPERATIONS = (
    _CountOperation(vector.vector_ctlzi, "leading"),
    _CountOperation(vector.vector_cttzi, "trailing"),
    _CountOperation(vector.vector_ctpopi, "population"),
)
_ELEMENT_SUFFIXES = {8: "b", 16: "w", 32: "d", 64: "q"}
_SHIFT_MNEMONICS = {16: "vpsrlw", 32: "vpsrld", 64: "vpsrlq"}
_SHIFT_LEFT_MNEMONICS = {16: "vpsllw", 32: "vpslld", 64: "vpsllq"}
_NIBBLE_COUNTS = bytes((0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4))


@dataclass(frozen=True, slots=True)
class _Carrier:
    descriptor_prefix: str
    vector: VectorCarrier
    and_mnemonic: str
    or_mnemonic: str
    xor_mnemonic: str
    read_only_load: str
    compare_all_ones: str | None

    @property
    def register_suffix(self) -> str:
        return self.vector.descriptor_suffix

    @property
    def register_class(self) -> str:
        return self.vector.register_class

    @property
    def vector_byte_count(self) -> int:
        return self.vector.register_bit_width // 8


class _PackedCountRecipe:
    """Builds one closed count recipe in a physical SIMD carrier."""

    def __init__(
        self,
        carrier: _Carrier,
        element_bit_width: int,
        descriptor_lookup: Callable[[str], Descriptor],
    ) -> None:
        self.carrier = carrier
        self.element_bit_width = element_bit_width
        self.descriptor_lookup = descriptor_lookup
        self.emits: list[EmitDescriptorOp] = []
        self.loaded_literals: dict[bytes, ValueRef] = {}

    def _descriptor(self, mnemonic: str) -> Descriptor:
        return self.descriptor_lookup(
            f"{self.carrier.descriptor_prefix}.{mnemonic}."
            f"{self.carrier.register_suffix}"
        )

    def binary(
        self,
        mnemonic: str,
        lhs: ValueRef,
        rhs: ValueRef,
        name: str,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        if result is None:
            output = ValueRef.temporary(name)
            result_types = {"dst": DescriptorResultType()}
        else:
            output = result
            result_types = None
        self.emits.append(
            _op_emit(
                descriptor=self._descriptor(mnemonic),
                operands={"lhs": lhs, "rhs": rhs},
                results={"dst": output},
                result_types=result_types,
            )
        )
        return output

    def shift(
        self,
        mnemonic: str,
        value: ValueRef,
        amount: int,
        name: str,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        if result is None:
            output = ValueRef.temporary(name)
            result_types = {"dst": DescriptorResultType()}
        else:
            output = result
            result_types = None
        self.emits.append(
            _op_emit(
                descriptor=self._descriptor(mnemonic),
                operands={"source": value},
                results={"dst": output},
                result_types=result_types,
                immediates={"shift": amount},
            )
        )
        return output

    def load_literal(self, data: bytes, name: str) -> ValueRef:
        previous = self.loaded_literals.get(data)
        if previous is not None:
            return previous
        result = ValueRef.temporary(name)
        self.emits.append(
            _op_emit(
                descriptor=self.descriptor_lookup(self.carrier.read_only_load),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"data": ReadOnlyDataLiteral(data)},
            )
        )
        self.loaded_literals[data] = result
        return result

    def byte_mask(self, value: int, name: str) -> ValueRef:
        return self.load_literal(
            bytes((value,)) * self.carrier.vector_byte_count,
            name,
        )

    def all_ones(self, value: ValueRef) -> ValueRef:
        if self.carrier.compare_all_ones is None:
            return self.byte_mask(0xFF, "all_ones")
        result = ValueRef.temporary("all_ones")
        self.emits.append(
            _op_emit(
                descriptor=self.descriptor_lookup(self.carrier.compare_all_ones),
                operands={"lhs": value, "rhs": value},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
            )
        )
        return result

    def shift_right_elements(self, value: ValueRef, amount: int, name: str) -> ValueRef:
        if self.element_bit_width == 8:
            shifted = self.shift("vpsrlw", value, amount, f"{name}_words")
            return self.binary(
                self.carrier.and_mnemonic,
                shifted,
                self.byte_mask(0xFF >> amount, f"mask_shift{amount}"),
                name,
            )
        return self.shift(_SHIFT_MNEMONICS[self.element_bit_width], value, amount, name)

    def population_count(self, value: ValueRef, prefix: str) -> None:
        table = self.load_literal(
            _NIBBLE_COUNTS * (self.carrier.vector_byte_count // 16),
            f"{prefix}_nibble_counts",
        )
        nibble_mask = self.byte_mask(0x0F, f"{prefix}_nibble_mask")
        low_nibbles = self.binary(
            self.carrier.and_mnemonic,
            value,
            nibble_mask,
            f"{prefix}_low_nibbles",
        )
        shifted = self.shift("vpsrlw", value, 4, f"{prefix}_shifted_nibbles")
        high_nibbles = self.binary(
            self.carrier.and_mnemonic,
            shifted,
            nibble_mask,
            f"{prefix}_high_nibbles",
        )
        low_counts = self.binary("vpshufb", table, low_nibbles, f"{prefix}_low_counts")
        high_counts = self.binary(
            "vpshufb", table, high_nibbles, f"{prefix}_high_counts"
        )
        byte_result = ValueRef.result("result") if self.element_bit_width == 8 else None
        current = self.binary(
            "vpaddb",
            low_counts,
            high_counts,
            f"{prefix}_byte_counts",
            result=byte_result,
        )
        if self.element_bit_width == 8:
            return

        for amount in (8, 16, 32):
            if amount >= self.element_bit_width:
                break
            shifted = self.shift(
                _SHIFT_MNEMONICS[self.element_bit_width],
                current,
                amount,
                f"{prefix}_shifted{amount}",
            )
            current = self.binary(
                f"vpadd{_ELEMENT_SUFFIXES[self.element_bit_width]}",
                current,
                shifted,
                f"{prefix}_summed{amount}",
            )

        result_bits = self.element_bit_width.bit_length()
        mask_shift = self.element_bit_width - result_bits
        masked = self.shift(
            _SHIFT_LEFT_MNEMONICS[self.element_bit_width],
            current,
            mask_shift,
            f"{prefix}_masked",
        )
        self.shift(
            _SHIFT_MNEMONICS[self.element_bit_width],
            masked,
            mask_shift,
            f"{prefix}_count",
            result=ValueRef.result("result"),
        )

    def count(self, kind: _CountKind) -> None:
        value = ValueRef.operand("input")
        if kind == "population":
            self.population_count(value, "population")
            return

        all_ones = self.all_ones(value)
        if kind == "trailing":
            minus_one = self.binary(
                f"vpadd{_ELEMENT_SUFFIXES[self.element_bit_width]}",
                value,
                all_ones,
                "minus_one",
            )
            inverted = self.binary(
                self.carrier.xor_mnemonic,
                value,
                all_ones,
                "inverted",
            )
            below = self.binary(
                self.carrier.and_mnemonic,
                minus_one,
                inverted,
                "below_lowest_bit",
            )
            self.population_count(below, "trailing")
            return

        propagated = value
        amount = 1
        while amount < self.element_bit_width:
            shifted = self.shift_right_elements(
                propagated, amount, f"leading_shifted{amount}"
            )
            propagated = self.binary(
                self.carrier.or_mnemonic,
                propagated,
                shifted,
                f"leading_propagated{amount}",
            )
            amount *= 2
        inverted = self.binary(
            self.carrier.xor_mnemonic,
            propagated,
            all_ones,
            "leading_inverted",
        )
        self.population_count(inverted, "leading_population")


def _integer_count_rule(
    operation: _CountOperation,
    carrier: _Carrier,
    element: VectorElement,
    descriptor_lookup: Callable[[str], Descriptor],
) -> DescriptorRule:
    recipe = _PackedCountRecipe(carrier, element.bit_width, descriptor_lookup)
    recipe.count(operation.kind)
    return DescriptorRule(
        source_op=operation.source_op,
        descriptor=recipe.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", carrier.vector.value_type),
            Guard.value_type("result", carrier.vector.value_type),
            Guard.low_value_register_class("input", carrier.register_class),
            Guard.low_value_register_class("result", carrier.register_class),
        ),
        emit=tuple(recipe.emits),
        report_key="integer_bit_count.vector.composed",
    )


def _predicate_population_alias_rule() -> ValueAliasRule:
    type_pattern = Vector("i1", minimum_static_elements=2, maximum_static_elements=64)
    return ValueAliasRule(
        source_op=vector.vector_ctpopi,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", type_pattern),
            Guard.value_type("result", type_pattern),
        ),
    )


def _predicate_guards(
    type_pattern: TypePattern,
    register_class: str,
    representation: int,
) -> tuple[Guard, ...]:
    return (
        Guard.value_type("input", type_pattern),
        Guard.value_type("result", type_pattern),
        Guard.low_value_register_class("input", register_class),
        Guard.low_value_representation("input", representation),
        Guard.low_value_register_class("result", register_class),
        Guard.low_value_representation("result", representation),
    )


def _avx2_predicate_complement_rule(
    source_op: Op,
    lane_count: int,
    representation_bit_width: int,
    descriptor_lookup: Callable[[str], Descriptor],
) -> DescriptorRule:
    register_class = avx2_predicate_register_class(representation_bit_width)
    register_suffix = register_class.removeprefix("x86.")
    ones = ValueRef.temporary("all_ones")
    compare = descriptor_lookup(f"x86.avx2.vpcmpeqd.{register_suffix}")
    xor = descriptor_lookup(f"x86.avx2.vpxor.{register_suffix}")
    emits = (
        _op_emit(
            descriptor=compare,
            operands={
                "lhs": ValueRef.operand("input"),
                "rhs": ValueRef.operand("input"),
            },
            results={"dst": ones},
            result_types={"dst": DescriptorResultType()},
        ),
        _op_emit(
            descriptor=xor,
            operands={"lhs": ValueRef.operand("input"), "rhs": ones},
            results={"dst": ValueRef.result("result")},
        ),
    )
    type_pattern = Vector(
        "i1",
        minimum_static_elements=lane_count,
        maximum_static_elements=lane_count,
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=xor,
        guards=_predicate_guards(
            type_pattern,
            register_class,
            avx2_predicate_representation_id(lane_count, representation_bit_width),
        ),
        emit=emits,
        report_key="integer_bit_count.predicate.complement",
    )


def _avx512_mask_complement_rule(
    source_op: Op, descriptor_lookup: Callable[[str], Descriptor]
) -> DescriptorRule:
    immediate = ValueRef.temporary("all_ones_bits")
    ones = ValueRef.temporary("all_ones")
    move_immediate = descriptor_lookup("x86.scalar.movimm.gpr64")
    move_mask = descriptor_lookup("x86.avx512.kmovq.k.gpr64")
    xor = descriptor_lookup("x86.avx512.kxorq")
    emits = (
        EmitDescriptorOp(
            descriptor=move_immediate,
            results={"dst": immediate},
            result_types={"dst": Scalar("i64")},
            immediates={"imm64": -1},
            form=DescriptorEmitForm.CONST,
        ),
        _op_emit(
            descriptor=move_mask,
            operands={"source": immediate},
            results={"dst": ones},
            result_types={"dst": DescriptorResultType()},
        ),
        _op_emit(
            descriptor=xor,
            operands={"lhs": ValueRef.operand("input"), "rhs": ones},
            results={"dst": ValueRef.result("result")},
        ),
    )
    type_pattern = Vector("i1", minimum_static_elements=2, maximum_static_elements=64)
    return DescriptorRule(
        source_op=source_op,
        descriptor=xor,
        guards=_predicate_guards(type_pattern, "x86.k", 1),
        emit=emits,
        report_key="integer_bit_count.predicate.complement",
    )


def _avx512_zmm_predicate_complement_rule(
    source_op: Op, descriptor_lookup: Callable[[str], Descriptor]
) -> DescriptorRule:
    load = descriptor_lookup("x86.avx512.vmovdqu64.rodata.zmm")
    xor = descriptor_lookup("x86.avx512.vpxord.zmm")
    ones = ValueRef.temporary("all_ones")
    emits = (
        _op_emit(
            descriptor=load,
            results={"dst": ones},
            result_types={"dst": DescriptorResultType()},
            immediates={"data": ReadOnlyDataLiteral(bytes((0xFF,)) * 64)},
        ),
        _op_emit(
            descriptor=xor,
            operands={"lhs": ValueRef.operand("input"), "rhs": ones},
            results={"dst": ValueRef.result("result")},
        ),
    )
    type_pattern = Vector("i1", minimum_static_elements=64, maximum_static_elements=64)
    return DescriptorRule(
        source_op=source_op,
        descriptor=xor,
        guards=_predicate_guards(type_pattern, "x86.zmm", 8),
        emit=emits,
        report_key="integer_bit_count.predicate.complement",
    )


def avx2_vector_integer_count_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[ContractCase, ...]:
    """Returns complete AVX2 count recipes for supported packed carriers."""
    predicate_complements = tuple(
        _avx2_predicate_complement_rule(
            operation.source_op,
            lane_count,
            representation_bit_width,
            descriptor_lookup,
        )
        for operation in _COUNT_OPERATIONS[:2]
        for lane_count in (2, 4, 8, 16, 32)
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        )
    )
    return (
        _predicate_population_alias_rule(),
        *predicate_complements,
        *(
            _integer_count_rule(
                operation,
                _Carrier(
                    descriptor_prefix="x86.avx2",
                    vector=vector,
                    and_mnemonic="vpand",
                    or_mnemonic="vpor",
                    xor_mnemonic="vpxor",
                    read_only_load=(
                        f"x86.avx2.vmovdqu.rodata.{vector.descriptor_suffix}"
                    ),
                    compare_all_ones=(f"x86.avx2.vpcmpeqd.{vector.descriptor_suffix}"),
                ),
                element,
                descriptor_lookup,
            )
            for operation in _COUNT_OPERATIONS
            for element in INTEGER_ELEMENTS
            for vector in vector_carriers(element, AVX2_CARRIER_VECTOR_BIT_WIDTHS)
        ),
    )


def avx512_vector_integer_count_rules(
    descriptor_lookup: Callable[[str], Descriptor],
) -> tuple[ContractCase, ...]:
    """Returns the AVX-512 delta over the inherited AVX2 count family."""
    return (
        *(
            rule_builder(operation.source_op, descriptor_lookup)
            for operation in _COUNT_OPERATIONS[:2]
            for rule_builder in (
                _avx512_mask_complement_rule,
                _avx512_zmm_predicate_complement_rule,
            )
        ),
        *(
            _integer_count_rule(
                operation,
                _Carrier(
                    descriptor_prefix="x86.avx512",
                    vector=vector,
                    and_mnemonic="vpandd",
                    or_mnemonic="vpord",
                    xor_mnemonic="vpxord",
                    read_only_load="x86.avx512.vmovdqu64.rodata.zmm",
                    compare_all_ones=None,
                ),
                element,
                descriptor_lookup,
            )
            for operation in _COUNT_OPERATIONS
            for element in INTEGER_ELEMENTS
            for vector in vector_carriers(element, AVX512_VECTOR_BIT_WIDTHS)
        ),
    )
