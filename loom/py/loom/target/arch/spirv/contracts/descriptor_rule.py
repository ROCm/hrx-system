# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builders for rules using the SPIR-V logical core descriptor set."""

from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    Guard,
    ResultTypeBinding,
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
    SourceOpProject,
    SourceValueKind,
    ValueRef,
    descriptor_by_key,
)
from loom.target.emit.float_narrowing import (
    FloatNarrowingDescriptors,
    IntegerNarrowingDescriptors,
)
from loom.target.low_descriptors import Descriptor


class IntegerRecipe:
    """Builds scalar or native-vector packets with shared literal operands."""

    def __init__(self, integer_width: int, lanes: int) -> None:
        self.integer_width = integer_width
        self.lanes = lanes
        self.integer_suffix = (
            f"i{integer_width}" if lanes == 1 else f"v{lanes}i{integer_width}"
        )
        self.emits: list[EmitDescriptorOp] = []
        self.constants: dict[tuple[int, int], ValueRef] = {}

    def emit(
        self,
        key: str,
        operands: dict[str, ValueRef],
        *,
        immediates: dict[str, int] | None = None,
        result: ValueRef | None = None,
        form: DescriptorEmitForm = DescriptorEmitForm.OP,
    ) -> ValueRef:
        result = result or ValueRef.temporary(f"bit_pattern_{len(self.emits)}")
        self.emits.append(
            EmitDescriptorOp(
                descriptor=logical_core_descriptor(f"spirv.op_{key}"),
                operands=operands,
                results={"dst": result},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result.kind is SourceValueKind.TEMPORARY
                    else None
                ),
                immediates=immediates or {},
                form=form,
            )
        )
        return result

    def constant(self, value: int, *, bit_width: int | None = None) -> ValueRef:
        bit_width = self.integer_width if bit_width is None else bit_width
        key = (bit_width, value)
        if key not in self.constants:
            modulus = 1 << bit_width
            encoded = value & (modulus - 1)
            if encoded >= modulus // 2:
                encoded -= modulus
            scalar_suffix = f"i{bit_width}"
            literal = self.emit(
                f"constant.{scalar_suffix}",
                {},
                immediates={f"{scalar_suffix}_value": encoded},
                form=DescriptorEmitForm.CONST,
            )
            if self.lanes != 1:
                literal = self.emit(
                    f"composite_construct.v{self.lanes}{scalar_suffix}",
                    {f"component{lane}": literal for lane in range(self.lanes)},
                )
            self.constants[key] = literal
        return self.constants[key]

    def binary(self, operation: str, lhs: ValueRef, rhs: ValueRef) -> ValueRef:
        return self.emit(f"{operation}.{self.integer_suffix}", {"lhs": lhs, "rhs": rhs})

    def select(
        self,
        condition: ValueRef,
        true_value: ValueRef,
        false_value: ValueRef,
        *,
        result: ValueRef | None = None,
    ) -> ValueRef:
        return self.emit(
            f"select.{self.integer_suffix}",
            {
                "condition": condition,
                "true_value": true_value,
                "false_value": false_value,
            },
            result=result,
        )

    def import_float_bits(
        self, value: ValueRef, element: str, bit_width: int
    ) -> ValueRef:
        """Reinterprets native float lanes, packing a scalar half without Int16."""
        suffix = element if self.lanes == 1 else f"v{self.lanes}{element}"
        if bit_width != self.integer_width:
            # Float16/BFloat16 do not imply Int16. The scalar packet's low half
            # carries the payload; consumers mask or shift away its high half.
            suffix = f"v2{element}"
            value = self.emit(
                f"composite_construct.{suffix}",
                {"component0": value, "component1": value},
            )
        return self.emit(f"bitcast.{suffix}.{self.integer_suffix}", {"input": value})

    def export_float_bits(self, bits: ValueRef, element: str, bit_width: int) -> None:
        """Binds a native float result from same-width lanes or a packed half."""
        result = ValueRef.result("result")
        if bit_width == self.integer_width:
            suffix = element if self.lanes == 1 else f"v{self.lanes}{element}"
            self.emit(
                f"bitcast.{self.integer_suffix}.{suffix}",
                {"input": bits},
                result=result,
            )
        else:
            suffix = f"v2{element}"
            packed = self.emit(f"bitcast.i32.{suffix}", {"input": bits})
            self.emit(
                f"composite_extract.{suffix}.{element}",
                {"composite": packed},
                immediates={"component_index": 0},
                result=result,
            )


def logical_core_descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def descriptor_feature_guards(*descriptors: Descriptor) -> tuple[Guard, ...]:
    guards: list[Guard] = []
    seen_keys: set[str] = set()
    for descriptor in descriptors:
        if not descriptor.feature_mask_words or descriptor.key in seen_keys:
            continue
        seen_keys.add(descriptor.key)
        guards.append(Guard.descriptor_available(descriptor))
    return tuple(guards)


def float_narrowing_descriptors(
    integer_suffix: str, float_suffix: str
) -> FloatNarrowingDescriptors:
    """Returns the same-width SPIR-V descriptors for exact float narrowing."""
    # Every comparison operand is nonnegative, so signed comparisons avoid
    # introducing integer-view bitcasts while preserving numeric ordering.
    integer_descriptors = IntegerNarrowingDescriptors(
        integer_bit_width=int(integer_suffix.removeprefix("i")),
        integer_constant=logical_core_descriptor(f"spirv.op_constant.{integer_suffix}"),
        integer_add=logical_core_descriptor(f"spirv.op_iadd.{integer_suffix}"),
        integer_subtract=logical_core_descriptor(f"spirv.op_isub.{integer_suffix}"),
        integer_shift_left=logical_core_descriptor(
            f"spirv.op_shift_left_logical.{integer_suffix}"
        ),
        integer_shift_right_logical=logical_core_descriptor(
            f"spirv.op_shift_right_logical.{integer_suffix}"
        ),
        integer_bitwise_and=logical_core_descriptor(
            f"spirv.op_bitwise_and.{integer_suffix}"
        ),
        integer_bitwise_or=logical_core_descriptor(
            f"spirv.op_bitwise_or.{integer_suffix}"
        ),
        integer_less_than_nonnegative=logical_core_descriptor(
            f"spirv.op_s_less_than.{integer_suffix}"
        ),
        integer_greater_than_equal_nonnegative=logical_core_descriptor(
            f"spirv.op_s_greater_than_equal.{integer_suffix}"
        ),
        integer_greater_than_nonnegative=logical_core_descriptor(
            f"spirv.op_s_greater_than.{integer_suffix}"
        ),
        integer_select=logical_core_descriptor(f"spirv.op_select.{integer_suffix}"),
    )
    return FloatNarrowingDescriptors(
        integer=integer_descriptors,
        float_constant=logical_core_descriptor(f"spirv.op_constant.{float_suffix}"),
        float_add=logical_core_descriptor(f"spirv.op_fadd.{float_suffix}"),
        reinterpret_float_as_integer=logical_core_descriptor(
            f"spirv.op_bitcast.{float_suffix}.{integer_suffix}"
        ),
        reinterpret_integer_as_float=logical_core_descriptor(
            f"spirv.op_bitcast.{integer_suffix}.{float_suffix}"
        ),
    )


def emit_descriptor_op(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject | SourceOpProject | int] | None = None,
    source_memory: SourceMemoryConstraint | None = None,
    source_memory_address_materializer: SourceMemoryAddressMaterializer | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_address_materializer=source_memory_address_materializer,
    )
