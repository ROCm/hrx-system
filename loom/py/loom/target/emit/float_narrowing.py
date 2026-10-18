# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral exact binary-float narrowing descriptor recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, unique

from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    SourceValueKind,
    ValueRef,
)
from loom.target.low_descriptors import Descriptor


@unique
class NarrowFloatOverflow(Enum):
    """Destination encoding used when a finite source value overflows."""

    INFINITY = "infinity"
    SATURATE = "saturate"


@unique
class NarrowFloatNan(Enum):
    """Destination encoding used for a source NaN."""

    PRESERVE_PAYLOAD = "preserve_payload"
    CANONICAL = "canonical"


@unique
class NarrowFloatSubnormalRounding(Enum):
    """Mechanism used to round destination subnormal values."""

    RNE_FLOAT_ADD = "rne_float_add"
    INTEGER = "integer"
    INTEGER_SOURCE_SUBNORMALS = "integer_source_subnormals"


@dataclass(frozen=True, slots=True)
class BinaryFloatFormat:
    """The bit layout of one IEEE-like source format."""

    bit_width: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int

    def __post_init__(self) -> None:
        if self.exponent_bits < 2 or self.mantissa_bits < 1:
            raise ValueError("binary float fields must have positive precision")
        if self.bit_width != 1 + self.exponent_bits + self.mantissa_bits:
            raise ValueError("binary float fields do not fill its bit width")
        if not 0 < self.exponent_bias < (1 << self.exponent_bits) - 1:
            raise ValueError("binary float exponent bias is not representable")


F32_FORMAT = BinaryFloatFormat(
    bit_width=32,
    exponent_bits=8,
    mantissa_bits=23,
    exponent_bias=127,
)

F64_FORMAT = BinaryFloatFormat(
    bit_width=64,
    exponent_bits=11,
    mantissa_bits=52,
    exponent_bias=1023,
)


@dataclass(frozen=True, slots=True)
class NarrowFloatFormat:
    """IEEE-like destination format and its exceptional-value policies."""

    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    overflow: NarrowFloatOverflow
    nan: NarrowFloatNan

    def __post_init__(self) -> None:
        if self.exponent_bits < 2 or self.mantissa_bits < 1:
            raise ValueError("narrow float fields must have positive precision")
        if not 0 < self.exponent_bias < (1 << self.exponent_bits) - 1:
            raise ValueError("narrow float exponent bias is not representable")

    @property
    def bit_width(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits


F16_FORMAT = NarrowFloatFormat(
    exponent_bits=5,
    mantissa_bits=10,
    exponent_bias=15,
    overflow=NarrowFloatOverflow.INFINITY,
    nan=NarrowFloatNan.PRESERVE_PAYLOAD,
)

F32_NARROW_FORMAT = NarrowFloatFormat(
    exponent_bits=F32_FORMAT.exponent_bits,
    mantissa_bits=F32_FORMAT.mantissa_bits,
    exponent_bias=F32_FORMAT.exponent_bias,
    overflow=NarrowFloatOverflow.INFINITY,
    nan=NarrowFloatNan.PRESERVE_PAYLOAD,
)

BF16_FORMAT = NarrowFloatFormat(
    exponent_bits=8,
    mantissa_bits=7,
    exponent_bias=127,
    overflow=NarrowFloatOverflow.INFINITY,
    nan=NarrowFloatNan.PRESERVE_PAYLOAD,
)

F8E4M3_FORMAT = NarrowFloatFormat(
    exponent_bits=4,
    mantissa_bits=3,
    exponent_bias=7,
    overflow=NarrowFloatOverflow.SATURATE,
    nan=NarrowFloatNan.CANONICAL,
)

F8E5M2_FORMAT = NarrowFloatFormat(
    exponent_bits=5,
    mantissa_bits=2,
    exponent_bias=15,
    overflow=NarrowFloatOverflow.INFINITY,
    nan=NarrowFloatNan.CANONICAL,
)

F16_SOURCE_FORMAT = BinaryFloatFormat(
    bit_width=F16_FORMAT.bit_width,
    exponent_bits=F16_FORMAT.exponent_bits,
    mantissa_bits=F16_FORMAT.mantissa_bits,
    exponent_bias=F16_FORMAT.exponent_bias,
)

BF16_SOURCE_FORMAT = BinaryFloatFormat(
    bit_width=BF16_FORMAT.bit_width,
    exponent_bits=BF16_FORMAT.exponent_bits,
    mantissa_bits=BF16_FORMAT.mantissa_bits,
    exponent_bias=BF16_FORMAT.exponent_bias,
)


@dataclass(frozen=True, slots=True)
class IntegerNarrowingImmediateForms:
    """Optional immediate descriptors for literal recipe operands."""

    add: Descriptor | None = None
    subtract: Descriptor | None = None
    shift_left: Descriptor | None = None
    shift_right_logical: Descriptor | None = None
    bitwise_and: Descriptor | None = None
    bitwise_or: Descriptor | None = None
    less_than_nonnegative: Descriptor | None = None
    greater_than_equal_nonnegative: Descriptor | None = None
    greater_than_nonnegative: Descriptor | None = None


@dataclass(frozen=True, slots=True)
class IntegerNarrowingDescriptors:
    """Integer-carrier target descriptors required by exact narrowing."""

    integer_bit_width: int
    integer_constant: Descriptor
    integer_add: Descriptor
    integer_subtract: Descriptor
    integer_shift_left: Descriptor
    integer_shift_right_logical: Descriptor
    integer_bitwise_and: Descriptor
    integer_bitwise_or: Descriptor
    integer_less_than_nonnegative: Descriptor
    integer_greater_than_equal_nonnegative: Descriptor
    integer_greater_than_nonnegative: Descriptor
    integer_select: Descriptor
    immediate_forms: IntegerNarrowingImmediateForms | None = None

    def __post_init__(self) -> None:
        if self.integer_bit_width < 2:
            raise ValueError("integer narrowing carrier must have at least two bits")


@dataclass(frozen=True, slots=True)
class FloatNarrowingDescriptors:
    """Floating-source target descriptors required by exact narrowing."""

    integer: IntegerNarrowingDescriptors
    float_constant: Descriptor
    float_add: Descriptor
    reinterpret_float_as_integer: Descriptor
    reinterpret_integer_as_float: Descriptor


class _ScalarRecipe:
    """Builds one compact straight-line scalar descriptor recipe."""

    def __init__(self, descriptors: IntegerNarrowingDescriptors) -> None:
        self.descriptors = descriptors
        self.emits: list[EmitDescriptorOp] = []

    def _operation(
        self,
        result: ValueRef,
        descriptor: Descriptor,
        operands: dict[str, ValueRef],
    ) -> ValueRef:
        result_types = None
        if result.kind is SourceValueKind.TEMPORARY:
            result_types = {"dst": DescriptorResultType()}
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": result},
                result_types=result_types,
            )
        )
        return result

    @staticmethod
    def _try_constant_immediate(
        descriptor: Descriptor, value: int
    ) -> dict[str, int] | None:
        (immediate,) = descriptor.immediates
        modulus = 1 << immediate.bit_width
        encoded_value = value & (modulus - 1)
        if encoded_value > immediate.unsigned_max:
            encoded_value -= modulus
        if not immediate.signed_min <= encoded_value <= immediate.unsigned_max:
            return None
        if encoded_value % immediate.value_step:
            return None
        return {immediate.field_name: encoded_value}

    @classmethod
    def _constant_immediate(cls, descriptor: Descriptor, value: int) -> dict[str, int]:
        immediates = cls._try_constant_immediate(descriptor, value)
        if immediates is None:
            raise ValueError(
                f"value {value} is not representable by descriptor '{descriptor.key}'"
            )
        return immediates

    def finish(self) -> tuple[EmitDescriptorOp, ...]:
        """Selects declared literal forms and returns the completed recipe."""

        immediate_forms = self.descriptors.immediate_forms
        if immediate_forms is None:
            return tuple(self.emits)

        immediate_by_descriptor_key = {
            descriptor.key: (immediate, commutative)
            for descriptor, immediate, commutative in (
                (self.descriptors.integer_add, immediate_forms.add, True),
                (
                    self.descriptors.integer_subtract,
                    immediate_forms.subtract,
                    False,
                ),
                (
                    self.descriptors.integer_shift_left,
                    immediate_forms.shift_left,
                    False,
                ),
                (
                    self.descriptors.integer_shift_right_logical,
                    immediate_forms.shift_right_logical,
                    False,
                ),
                (
                    self.descriptors.integer_bitwise_and,
                    immediate_forms.bitwise_and,
                    True,
                ),
                (
                    self.descriptors.integer_bitwise_or,
                    immediate_forms.bitwise_or,
                    True,
                ),
                (
                    self.descriptors.integer_less_than_nonnegative,
                    immediate_forms.less_than_nonnegative,
                    False,
                ),
                (
                    self.descriptors.integer_greater_than_equal_nonnegative,
                    immediate_forms.greater_than_equal_nonnegative,
                    False,
                ),
                (
                    self.descriptors.integer_greater_than_nonnegative,
                    immediate_forms.greater_than_nonnegative,
                    False,
                ),
            )
            if immediate is not None
        }
        if not immediate_by_descriptor_key:
            return tuple(self.emits)

        constant_values: dict[str, int] = {}
        for emit in self.emits:
            if (
                emit.descriptor.key != self.descriptors.integer_constant.key
                or emit.form is not DescriptorEmitForm.CONST
            ):
                continue
            (result,) = emit.results.values()
            (value,) = emit.immediates.values()
            if result.kind is not SourceValueKind.TEMPORARY or not isinstance(
                value, int
            ):
                raise ValueError(
                    "integer recipe constants must define literal temporaries"
                )
            if result.field in constant_values:
                raise ValueError(
                    f"integer recipe temporary '{result.field}' is defined twice"
                )
            constant_values[result.field] = value

        rewritten_emits: list[EmitDescriptorOp] = []
        for emit in self.emits:
            immediate_form = immediate_by_descriptor_key.get(emit.descriptor.key)
            lhs = emit.operands.get("lhs")
            rhs = emit.operands.get("rhs")
            if immediate_form is None or lhs is None or rhs is None:
                rewritten_emits.append(emit)
                continue
            immediate_descriptor, commutative = immediate_form
            literal = rhs
            value = lhs
            if (
                rhs.kind is not SourceValueKind.TEMPORARY
                or rhs.field not in constant_values
            ):
                if (
                    not commutative
                    or lhs.kind is not SourceValueKind.TEMPORARY
                    or lhs.field not in constant_values
                ):
                    rewritten_emits.append(emit)
                    continue
                literal = lhs
                value = rhs
            immediates = self._try_constant_immediate(
                immediate_descriptor, constant_values[literal.field]
            )
            if immediates is None:
                rewritten_emits.append(emit)
                continue
            operands = dict(emit.operands)
            operands["lhs"] = value
            del operands["rhs"]
            rewritten_emits.append(
                replace(
                    emit,
                    descriptor=immediate_descriptor,
                    operands=operands,
                    immediates=immediates,
                )
            )

        used_temporaries = {
            operand.field
            for emit in rewritten_emits
            for operand in emit.operands.values()
            if operand.kind is SourceValueKind.TEMPORARY
        }
        return tuple(
            emit
            for emit in rewritten_emits
            if not (
                emit.descriptor.key == self.descriptors.integer_constant.key
                and emit.form is DescriptorEmitForm.CONST
                and next(iter(emit.results.values())).field not in used_temporaries
            )
        )

    def integer_constant(self, result_name: str, value: int) -> ValueRef:
        result = ValueRef.temporary(result_name)
        descriptor = self.descriptors.integer_constant
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates=self._constant_immediate(descriptor, value),
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def float_constant(
        self, result_name: str, bits: int, descriptor: Descriptor
    ) -> ValueRef:
        result = ValueRef.temporary(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates=self._constant_immediate(descriptor, bits),
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def integer_binary(
        self,
        result_name: str,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self._operation(
            ValueRef.temporary(result_name),
            descriptor,
            {"lhs": lhs, "rhs": rhs},
        )

    def integer_binary_to(
        self,
        result: ValueRef,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self._operation(result, descriptor, {"lhs": lhs, "rhs": rhs})

    def float_add(
        self,
        result_name: str,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self._operation(
            ValueRef.temporary(result_name),
            descriptor,
            {"lhs": lhs, "rhs": rhs},
        )

    def reinterpret_float_as_integer(
        self, result_name: str, descriptor: Descriptor, value: ValueRef
    ) -> ValueRef:
        return self._operation(
            ValueRef.temporary(result_name),
            descriptor,
            {"input": value},
        )

    def reinterpret_integer_as_float(
        self, result_name: str, descriptor: Descriptor, value: ValueRef
    ) -> ValueRef:
        return self._operation(
            ValueRef.temporary(result_name),
            descriptor,
            {"input": value},
        )

    def integer_select(
        self,
        result_name: str,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self._operation(
            ValueRef.temporary(result_name),
            self.descriptors.integer_select,
            {
                "true_value": true_value,
                "false_value": false_value,
                "condition": condition,
            },
        )

    def integer_select_to(
        self,
        result: ValueRef,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self._operation(
            result,
            self.descriptors.integer_select,
            {
                "true_value": true_value,
                "false_value": false_value,
                "condition": condition,
            },
        )


def build_f32_to_bf16_emits(
    descriptors: FloatNarrowingDescriptors,
    input_ref: ValueRef,
    result_ref: ValueRef,
    *,
    preserve_nan: bool = True,
) -> tuple[EmitDescriptorOp, ...]:
    """Rounds F32 to BF16 while preserving every source NaN as a NaN."""

    integer_descriptors = descriptors.integer
    recipe = _ScalarRecipe(integer_descriptors)
    input_bits = recipe.reinterpret_float_as_integer(
        "input_bits", descriptors.reinterpret_float_as_integer, input_ref
    )
    return _build_f32_bits_to_bf16_emits(
        recipe,
        input_bits,
        result_ref,
        exact_finite_values=False,
        preserve_nan=preserve_nan,
    )


def build_representable_f32_to_bf16_emits(
    descriptors: FloatNarrowingDescriptors,
    input_ref: ValueRef,
    result_ref: ValueRef,
    *,
    preserve_nan: bool,
) -> tuple[EmitDescriptorOp, ...]:
    """Narrows F32 whose finite values are exact in BF16, optionally quieting NaNs."""

    recipe = _ScalarRecipe(descriptors.integer)
    input_bits = recipe.reinterpret_float_as_integer(
        "input_bits", descriptors.reinterpret_float_as_integer, input_ref
    )
    return _build_f32_bits_to_bf16_emits(
        recipe,
        input_bits,
        result_ref,
        exact_finite_values=True,
        preserve_nan=preserve_nan,
    )


def build_f32_bits_to_bf16_emits(
    descriptors: IntegerNarrowingDescriptors,
    input_bits_ref: ValueRef,
    result_ref: ValueRef,
    *,
    preserve_nan: bool = True,
) -> tuple[EmitDescriptorOp, ...]:
    """Rounds integer-carried F32 bits to BF16."""

    return _build_f32_bits_to_bf16_emits(
        _ScalarRecipe(descriptors),
        input_bits_ref,
        result_ref,
        exact_finite_values=False,
        preserve_nan=preserve_nan,
    )


def build_representable_f32_bits_to_bf16_emits(
    descriptors: IntegerNarrowingDescriptors,
    input_bits_ref: ValueRef,
    result_ref: ValueRef,
    *,
    preserve_nan: bool,
) -> tuple[EmitDescriptorOp, ...]:
    """Narrows integer-carried F32 bits with exact finite values in BF16."""

    return _build_f32_bits_to_bf16_emits(
        _ScalarRecipe(descriptors),
        input_bits_ref,
        result_ref,
        exact_finite_values=True,
        preserve_nan=preserve_nan,
    )


def _build_f32_bits_to_bf16_emits(
    recipe: _ScalarRecipe,
    input_bits: ValueRef,
    result_ref: ValueRef,
    *,
    exact_finite_values: bool,
    preserve_nan: bool,
) -> tuple[EmitDescriptorOp, ...]:
    """Narrows F32 bits using independent finite-format and NaN guarantees."""

    integer_descriptors = recipe.descriptors
    shift = recipe.integer_constant("shift", 16)
    upper = recipe.integer_binary_to(
        result_ref
        if exact_finite_values and not preserve_nan
        else ValueRef.temporary("upper"),
        integer_descriptors.integer_shift_right_logical,
        input_bits,
        shift,
    )
    finite = upper
    if not exact_finite_values:
        one = recipe.integer_constant("one", 1)
        retained_lsb = recipe.integer_binary(
            "retained_lsb", integer_descriptors.integer_bitwise_and, upper, one
        )
        rounding_bias = recipe.integer_constant("rounding_bias", 0x7FFF)
        bias = recipe.integer_binary(
            "bias", integer_descriptors.integer_add, rounding_bias, retained_lsb
        )
        rounded = recipe.integer_binary(
            "rounded", integer_descriptors.integer_add, input_bits, bias
        )
        finite = recipe.integer_binary_to(
            ValueRef.temporary("finite") if preserve_nan else result_ref,
            integer_descriptors.integer_shift_right_logical,
            rounded,
            shift,
        )

    if not preserve_nan:
        return recipe.finish()

    nonsign_mask = recipe.integer_constant("nonsign_mask", 0x7FFFFFFF)
    magnitude = recipe.integer_binary(
        "magnitude",
        integer_descriptors.integer_bitwise_and,
        input_bits,
        nonsign_mask,
    )
    infinity_bits = recipe.integer_constant("infinity_bits", 0x7F800000)
    is_nan = recipe.integer_binary(
        "is_nan",
        integer_descriptors.integer_greater_than_nonnegative,
        magnitude,
        infinity_bits,
    )
    quiet_nan_bit = recipe.integer_constant("quiet_nan_bit", 0x0040)
    nan = recipe.integer_binary(
        "nan", integer_descriptors.integer_bitwise_or, upper, quiet_nan_bit
    )
    recipe.integer_select_to(result_ref, nan, finite, is_nan)
    return recipe.finish()


def build_float_to_narrow_float_emits(
    descriptors: FloatNarrowingDescriptors,
    source_format: BinaryFloatFormat,
    narrow_format: NarrowFloatFormat,
    input_ref: ValueRef,
    result_ref: ValueRef,
    *,
    subnormal_rounding: NarrowFloatSubnormalRounding,
    preserve_nan: bool = True,
) -> tuple[EmitDescriptorOp, ...]:
    """Rounds a binary float using same-width arithmetic and integer carriers."""

    recipe = _ScalarRecipe(descriptors.integer)
    input_bits = recipe.reinterpret_float_as_integer(
        "input_bits", descriptors.reinterpret_float_as_integer, input_ref
    )
    return _build_float_bits_to_narrow_float_emits(
        recipe,
        source_format,
        narrow_format,
        input_bits,
        result_ref,
        subnormal_rounding=subnormal_rounding,
        preserve_nan=preserve_nan,
        float_descriptors=descriptors,
    )


def build_integer_bits_to_narrow_float_emits(
    descriptors: IntegerNarrowingDescriptors,
    source_format: BinaryFloatFormat,
    narrow_format: NarrowFloatFormat,
    input_bits_ref: ValueRef,
    result_ref: ValueRef,
    *,
    subnormal_rounding: NarrowFloatSubnormalRounding,
    preserve_nan: bool = True,
) -> tuple[EmitDescriptorOp, ...]:
    """Rounds integer-carried source bits without materializing a wider float."""

    if subnormal_rounding is NarrowFloatSubnormalRounding.RNE_FLOAT_ADD:
        raise ValueError("integer-carried narrowing requires integer rounding")
    recipe = _ScalarRecipe(descriptors)
    return _build_float_bits_to_narrow_float_emits(
        recipe,
        source_format,
        narrow_format,
        input_bits_ref,
        result_ref,
        subnormal_rounding=subnormal_rounding,
        preserve_nan=preserve_nan,
        float_descriptors=None,
    )


def _build_float_bits_to_narrow_float_emits(
    recipe: _ScalarRecipe,
    source_format: BinaryFloatFormat,
    narrow_format: NarrowFloatFormat,
    input_bits: ValueRef,
    result_ref: ValueRef,
    *,
    subnormal_rounding: NarrowFloatSubnormalRounding,
    preserve_nan: bool,
    float_descriptors: FloatNarrowingDescriptors | None,
) -> tuple[EmitDescriptorOp, ...]:
    """Builds exact narrowing after the source bits have been materialized."""

    descriptors = recipe.descriptors

    if source_format.bit_width <= narrow_format.bit_width:
        raise ValueError("float narrowing requires a smaller destination bit width")
    if source_format.mantissa_bits <= narrow_format.mantissa_bits:
        raise ValueError("float narrowing requires lower destination precision")
    if source_format.exponent_bias < narrow_format.exponent_bias:
        raise ValueError("float narrowing cannot increase the destination range")
    if source_format.bit_width > descriptors.integer_bit_width:
        raise ValueError("source format does not fit the integer carrier")
    if (
        subnormal_rounding is NarrowFloatSubnormalRounding.INTEGER
        and source_format.exponent_bias
        < narrow_format.exponent_bias + narrow_format.mantissa_bits + 1
    ):
        raise ValueError(
            "integer subnormal rounding requires every source subnormal to "
            "round to destination zero"
        )

    sign_mask_value = 1 << (source_format.bit_width - 1)
    sign_mask = recipe.integer_constant("sign_mask", sign_mask_value)
    sign = recipe.integer_binary(
        "sign", descriptors.integer_bitwise_and, input_bits, sign_mask
    )
    sign_shift = recipe.integer_constant(
        "sign_shift",
        source_format.bit_width - narrow_format.bit_width,
    )
    sign = recipe.integer_binary(
        "positioned_sign",
        descriptors.integer_shift_right_logical,
        sign,
        sign_shift,
    )

    nonsign_mask = recipe.integer_constant("nonsign_mask", sign_mask_value - 1)
    magnitude = recipe.integer_binary(
        "magnitude_bits",
        descriptors.integer_bitwise_and,
        input_bits,
        nonsign_mask,
    )

    magic_exponent = (
        source_format.exponent_bias
        - narrow_format.exponent_bias
        + (source_format.mantissa_bits - narrow_format.mantissa_bits)
        + 1
    )
    if subnormal_rounding is NarrowFloatSubnormalRounding.RNE_FLOAT_ADD:
        if float_descriptors is None:
            raise ValueError(
                "floating-point subnormal rounding requires float descriptors"
            )
        # The ULP of this power of two equals the destination minimum
        # subnormal. A guaranteed-RNE same-width addition performs the complete
        # rounding operation; subtracting the magic encoding exposes its
        # narrow payload.
        magic_bits_value = magic_exponent << source_format.mantissa_bits
        magic = recipe.float_constant(
            "subnormal_magic", magic_bits_value, float_descriptors.float_constant
        )
        magnitude_float = recipe.reinterpret_integer_as_float(
            "magnitude",
            float_descriptors.reinterpret_integer_as_float,
            magnitude,
        )
        biased_subnormal = recipe.float_add(
            "biased_subnormal",
            float_descriptors.float_add,
            magnitude_float,
            magic,
        )
        biased_subnormal_bits = recipe.reinterpret_float_as_integer(
            "biased_subnormal_bits",
            float_descriptors.reinterpret_float_as_integer,
            biased_subnormal,
        )
        magic_bits = recipe.integer_constant("subnormal_magic_bits", magic_bits_value)
        subnormal = recipe.integer_binary(
            "subnormal",
            descriptors.integer_subtract,
            biased_subnormal_bits,
            magic_bits,
        )
    else:
        # Compute the destination subnormal payload without inheriting a
        # target's ambient floating-point rounding mode. The compact mode uses
        # an implicit source leading bit because its precondition proves every
        # source subnormal rounds to zero. The general mode selects the source
        # subnormal significand and effective exponent explicitly.
        fraction_mask = recipe.integer_constant(
            "source_fraction_mask", (1 << source_format.mantissa_bits) - 1
        )
        source_fraction = recipe.integer_binary(
            "source_fraction",
            descriptors.integer_bitwise_and,
            magnitude,
            fraction_mask,
        )
        hidden_bit = recipe.integer_constant(
            "source_hidden_bit", 1 << source_format.mantissa_bits
        )
        normal_significand = recipe.integer_binary(
            "source_normal_significand",
            descriptors.integer_bitwise_or,
            source_fraction,
            hidden_bit,
        )
        exponent_shift = recipe.integer_constant(
            "source_exponent_shift", source_format.mantissa_bits
        )
        source_exponent = recipe.integer_binary(
            "source_exponent",
            descriptors.integer_shift_right_logical,
            magnitude,
            exponent_shift,
        )
        if subnormal_rounding is NarrowFloatSubnormalRounding.INTEGER_SOURCE_SUBNORMALS:
            source_exponent_one = recipe.integer_constant("source_exponent_one", 1)
            source_is_subnormal = recipe.integer_binary(
                "source_is_subnormal",
                descriptors.integer_less_than_nonnegative,
                source_exponent,
                source_exponent_one,
            )
            significand = recipe.integer_select(
                "source_significand",
                source_fraction,
                normal_significand,
                source_is_subnormal,
            )
            effective_source_exponent = recipe.integer_select(
                "effective_source_exponent",
                source_exponent_one,
                source_exponent,
                source_is_subnormal,
            )
            maximum_subnormal_exponent_value = max(
                source_format.exponent_bias - narrow_format.exponent_bias,
                1,
            )
        else:
            significand = normal_significand
            effective_source_exponent = source_exponent
            maximum_subnormal_exponent_value = (
                source_format.exponent_bias - narrow_format.exponent_bias
            )
        maximum_subnormal_exponent = recipe.integer_constant(
            "maximum_subnormal_exponent",
            maximum_subnormal_exponent_value,
        )
        exponent_above_subnormal = recipe.integer_binary(
            "exponent_above_subnormal",
            descriptors.integer_greater_than_nonnegative,
            effective_source_exponent,
            maximum_subnormal_exponent,
        )
        bounded_exponent = recipe.integer_select(
            "bounded_subnormal_exponent",
            maximum_subnormal_exponent,
            effective_source_exponent,
            exponent_above_subnormal,
        )
        magic_exponent_value = recipe.integer_constant(
            "subnormal_magic_exponent", magic_exponent
        )
        subnormal_shift = recipe.integer_binary(
            "subnormal_shift",
            descriptors.integer_subtract,
            magic_exponent_value,
            bounded_exponent,
        )
        maximum_shift = recipe.integer_constant(
            "maximum_subnormal_shift", descriptors.integer_bit_width - 1
        )
        shift_too_large = recipe.integer_binary(
            "subnormal_shift_too_large",
            descriptors.integer_greater_than_nonnegative,
            subnormal_shift,
            maximum_shift,
        )
        safe_shift = recipe.integer_select(
            "safe_subnormal_shift",
            maximum_shift,
            subnormal_shift,
            shift_too_large,
        )
        subnormal_truncated = recipe.integer_binary(
            "subnormal_truncated",
            descriptors.integer_shift_right_logical,
            significand,
            safe_shift,
        )
        one = recipe.integer_constant("subnormal_one", 1)
        rounding_shift = recipe.integer_binary(
            "subnormal_rounding_shift",
            descriptors.integer_subtract,
            safe_shift,
            one,
        )
        halfway = recipe.integer_binary(
            "subnormal_halfway",
            descriptors.integer_shift_left,
            one,
            rounding_shift,
        )
        rounding_bias = recipe.integer_binary(
            "subnormal_rounding_bias",
            descriptors.integer_subtract,
            halfway,
            one,
        )
        retained_lsb = recipe.integer_binary(
            "subnormal_retained_lsb",
            descriptors.integer_bitwise_and,
            subnormal_truncated,
            one,
        )
        rounded_significand = recipe.integer_binary(
            "subnormal_biased",
            descriptors.integer_add,
            significand,
            rounding_bias,
        )
        rounded_significand = recipe.integer_binary(
            "subnormal_rounded_significand",
            descriptors.integer_add,
            rounded_significand,
            retained_lsb,
        )
        subnormal = recipe.integer_binary(
            "subnormal",
            descriptors.integer_shift_right_logical,
            rounded_significand,
            safe_shift,
        )

    normal_shift_value = source_format.mantissa_bits - narrow_format.mantissa_bits
    normal_shift = recipe.integer_constant("normal_shift", normal_shift_value)
    normal_truncated = recipe.integer_binary(
        "normal_truncated",
        descriptors.integer_shift_right_logical,
        magnitude,
        normal_shift,
    )
    one = recipe.integer_constant("one", 1)
    retained_lsb = recipe.integer_binary(
        "normal_retained_lsb",
        descriptors.integer_bitwise_and,
        normal_truncated,
        one,
    )
    rounding_bias = (1 << (normal_shift_value - 1)) - 1
    exponent_rebias = (
        narrow_format.exponent_bias - source_format.exponent_bias
    ) << source_format.mantissa_bits
    normal_bias = recipe.integer_constant(
        "normal_bias", exponent_rebias + rounding_bias
    )
    normal_rounded = recipe.integer_binary(
        "normal_biased", descriptors.integer_add, magnitude, normal_bias
    )
    normal_rounded = recipe.integer_binary(
        "normal_rounded", descriptors.integer_add, normal_rounded, retained_lsb
    )
    normal = recipe.integer_binary(
        "normal",
        descriptors.integer_shift_right_logical,
        normal_rounded,
        normal_shift,
    )

    minimum_normal_bits = recipe.integer_constant(
        "minimum_normal_bits",
        (source_format.exponent_bias - narrow_format.exponent_bias + 1)
        << source_format.mantissa_bits,
    )
    is_subnormal = recipe.integer_binary(
        "is_subnormal",
        descriptors.integer_less_than_nonnegative,
        magnitude,
        minimum_normal_bits,
    )
    finite = recipe.integer_select("finite_unclamped", subnormal, normal, is_subnormal)

    special_payload_value = (
        (1 << narrow_format.exponent_bits) - 1
    ) << narrow_format.mantissa_bits
    nan_payload_value = special_payload_value | ((1 << narrow_format.mantissa_bits) - 1)
    special_payload = recipe.integer_constant("special_payload", special_payload_value)
    if narrow_format.overflow is NarrowFloatOverflow.INFINITY:
        needs_clamp = recipe.integer_binary(
            "needs_clamp",
            descriptors.integer_greater_than_equal_nonnegative,
            finite,
            special_payload,
        )
        finite = recipe.integer_select("finite", special_payload, finite, needs_clamp)
    else:
        nan_payload = recipe.integer_constant("nan_payload", nan_payload_value)
        maximum_finite = recipe.integer_constant(
            "maximum_finite", nan_payload_value - 1
        )
        needs_clamp = recipe.integer_binary(
            "needs_clamp",
            descriptors.integer_greater_than_equal_nonnegative,
            finite,
            nan_payload,
        )
        finite = recipe.integer_select("finite", maximum_finite, finite, needs_clamp)

    if preserve_nan and narrow_format.nan is NarrowFloatNan.PRESERVE_PAYLOAD:
        fraction_mask = recipe.integer_constant(
            "fraction_mask", (1 << source_format.mantissa_bits) - 1
        )
        fraction = recipe.integer_binary(
            "fraction", descriptors.integer_bitwise_and, magnitude, fraction_mask
        )
        nan_payload = recipe.integer_binary(
            "source_nan_payload",
            descriptors.integer_shift_right_logical,
            fraction,
            normal_shift,
        )
        nan = recipe.integer_binary(
            "nan_payload",
            descriptors.integer_bitwise_or,
            special_payload,
            nan_payload,
        )
        quiet_nan_bit = recipe.integer_constant(
            "quiet_nan_bit", 1 << (narrow_format.mantissa_bits - 1)
        )
        nan = recipe.integer_binary(
            "nan", descriptors.integer_bitwise_or, nan, quiet_nan_bit
        )
    elif preserve_nan:
        nan = recipe.integer_constant("nan", nan_payload_value)

    if preserve_nan:
        source_infinity_bits_value = (
            (1 << source_format.exponent_bits) - 1
        ) << source_format.mantissa_bits
        infinity_bits = recipe.integer_constant(
            "source_infinity_bits", source_infinity_bits_value
        )
        is_nan = recipe.integer_binary(
            "is_nan",
            descriptors.integer_greater_than_nonnegative,
            magnitude,
            infinity_bits,
        )
        unsigned_result = recipe.integer_select("unsigned_result", nan, finite, is_nan)
    else:
        unsigned_result = finite
    recipe.integer_binary_to(
        result_ref, descriptors.integer_bitwise_or, sign, unsigned_result
    )
    return recipe.finish()
