# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.contracts import (
    DescriptorEmitForm,
    EmitDescriptorOp,
    SourceValueKind,
    ValueRef,
)
from loom.target.emit.float_narrowing import (
    F8E4M3_FORMAT,
    F16_FORMAT,
    F16_SOURCE_FORMAT,
    F64_FORMAT,
    FloatNarrowingDescriptors,
    IntegerNarrowingDescriptors,
    IntegerNarrowingImmediateForms,
    NarrowFloatSubnormalRounding,
    build_f32_bits_to_bf16_emits,
    build_f32_to_bf16_emits,
    build_integer_bits_to_narrow_float_emits,
    build_representable_f32_bits_to_bf16_emits,
    build_representable_f32_to_bf16_emits,
)
from loom.target.low_descriptors import Descriptor, Immediate, ImmediateKind


def _descriptor(key: str, immediate: Immediate | None = None) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=key,
        semantic_tag=key,
        operands=(),
        immediates=() if immediate is None else (immediate,),
        schedule_class="test",
    )


_I32 = Immediate(
    "value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)
_I64 = Immediate(
    "value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)
_SHIFT = Immediate(
    "shift",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=31,
)


def _integer_descriptors(
    *,
    integer_bit_width: int = 32,
    immediate_forms: IntegerNarrowingImmediateForms | None = None,
) -> IntegerNarrowingDescriptors:
    return IntegerNarrowingDescriptors(
        integer_bit_width=integer_bit_width,
        integer_constant=_descriptor(
            "test.constant", _I64 if integer_bit_width == 64 else _I32
        ),
        integer_add=_descriptor("test.add"),
        integer_subtract=_descriptor("test.subtract"),
        integer_shift_left=_descriptor("test.shift_left"),
        integer_shift_right_logical=_descriptor("test.shift_right_logical"),
        integer_bitwise_and=_descriptor("test.bitwise_and"),
        integer_bitwise_or=_descriptor("test.bitwise_or"),
        integer_less_than_nonnegative=_descriptor("test.less_than"),
        integer_greater_than_equal_nonnegative=_descriptor("test.greater_than_equal"),
        integer_greater_than_nonnegative=_descriptor("test.greater_than"),
        integer_select=_descriptor("test.select"),
        immediate_forms=immediate_forms,
    )


def _immediate_forms(*, shift: Immediate = _SHIFT) -> IntegerNarrowingImmediateForms:
    return IntegerNarrowingImmediateForms(
        add=_descriptor("test.add.immediate", _I32),
        subtract=_descriptor("test.subtract.immediate", _I32),
        shift_left=_descriptor("test.shift_left.immediate", shift),
        shift_right_logical=_descriptor("test.shift_right_logical.immediate", shift),
        bitwise_and=_descriptor("test.bitwise_and.immediate", _I32),
        bitwise_or=_descriptor("test.bitwise_or.immediate", _I32),
        less_than_nonnegative=_descriptor("test.less_than.immediate", _I32),
        greater_than_equal_nonnegative=_descriptor(
            "test.greater_than_equal.immediate", _I32
        ),
        greater_than_nonnegative=_descriptor("test.greater_than.immediate", _I32),
    )


def _f16_to_f8e4m3_emits(
    descriptors: IntegerNarrowingDescriptors,
) -> tuple[EmitDescriptorOp, ...]:
    return build_integer_bits_to_narrow_float_emits(
        descriptors,
        F16_SOURCE_FORMAT,
        F8E4M3_FORMAT,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER_SOURCE_SUBNORMALS,
    )


def _constant_names(emits: tuple[EmitDescriptorOp, ...]) -> set[str]:
    return {
        result.field
        for emit in emits
        if emit.form is DescriptorEmitForm.CONST
        for result in emit.results.values()
        if result.kind is SourceValueKind.TEMPORARY
    }


def test_empty_immediate_forms_preserve_the_recipe() -> None:
    baseline = _f16_to_f8e4m3_emits(_integer_descriptors())
    empty = _f16_to_f8e4m3_emits(
        _integer_descriptors(immediate_forms=IntegerNarrowingImmediateForms())
    )

    assert empty == baseline


def test_immediate_folding_preserves_wide_carrier_values() -> None:
    emits = build_integer_bits_to_narrow_float_emits(
        _integer_descriptors(integer_bit_width=64, immediate_forms=_immediate_forms()),
        F64_FORMAT,
        F16_FORMAT,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER,
    )

    is_nan = next(
        emit for emit in emits if emit.results == {"dst": ValueRef.temporary("is_nan")}
    )
    assert is_nan.descriptor.key == "test.greater_than"
    assert is_nan.operands["rhs"] == ValueRef.temporary("source_infinity_bits")


def test_f32_bits_entry_point_matches_the_float_recipe_tail() -> None:
    integer_descriptors = _integer_descriptors()
    reinterpret = _descriptor("test.reinterpret_float_as_integer")
    float_emits = build_f32_to_bf16_emits(
        FloatNarrowingDescriptors(
            integer=integer_descriptors,
            float_constant=_descriptor("test.float_constant"),
            float_add=_descriptor("test.float_add"),
            reinterpret_float_as_integer=reinterpret,
            reinterpret_integer_as_float=_descriptor(
                "test.reinterpret_integer_as_float"
            ),
        ),
        ValueRef.operand("input"),
        ValueRef.result("result"),
    )
    bits_emits = build_f32_bits_to_bf16_emits(
        integer_descriptors,
        ValueRef.temporary("input_bits"),
        ValueRef.result("result"),
    )

    assert float_emits[0].descriptor is reinterpret
    assert float_emits[1:] == bits_emits


def test_representable_f32_to_bf16_uses_only_reinterpret_and_shift() -> None:
    integer_descriptors = _integer_descriptors(immediate_forms=_immediate_forms())
    reinterpret = _descriptor("test.reinterpret_float_as_integer")
    float_emits = build_representable_f32_to_bf16_emits(
        FloatNarrowingDescriptors(
            integer=integer_descriptors,
            float_constant=_descriptor("test.float_constant"),
            float_add=_descriptor("test.float_add"),
            reinterpret_float_as_integer=reinterpret,
            reinterpret_integer_as_float=_descriptor(
                "test.reinterpret_integer_as_float"
            ),
        ),
        ValueRef.operand("input"),
        ValueRef.result("result"),
        preserve_nan=False,
    )
    bits_emits = build_representable_f32_bits_to_bf16_emits(
        integer_descriptors,
        ValueRef.temporary("input_bits"),
        ValueRef.result("result"),
        preserve_nan=False,
    )

    assert float_emits[0].descriptor is reinterpret
    assert float_emits[1:] == bits_emits
    assert len(float_emits) == 2
    assert bits_emits[0].descriptor.key == "test.shift_right_logical.immediate"
    assert bits_emits[0].immediates == {"shift": 16}
    assert bits_emits[0].results == {"dst": ValueRef.result("result")}


def test_representable_f32_to_bf16_keeps_nan_handling_without_rounding() -> None:
    emits = build_representable_f32_bits_to_bf16_emits(
        _integer_descriptors(immediate_forms=_immediate_forms()),
        ValueRef.operand("input"),
        ValueRef.result("result"),
        preserve_nan=True,
    )

    assert tuple(emit.descriptor.key for emit in emits) == (
        "test.shift_right_logical.immediate",
        "test.bitwise_and.immediate",
        "test.greater_than.immediate",
        "test.bitwise_or.immediate",
        "test.select",
    )
    assert emits[0].immediates == {"shift": 16}
    assert emits[1].immediates == {"value": 0x7FFFFFFF}
    assert emits[2].immediates == {"value": 0x7F800000}
    assert emits[3].immediates == {"value": 0x40}
    assert emits[-1].operands == {
        "true_value": ValueRef.temporary("nan"),
        "false_value": ValueRef.temporary("upper"),
        "condition": ValueRef.temporary("is_nan"),
    }


def test_immediate_forms_fold_literals_and_preserve_live_constants() -> None:
    baseline = _f16_to_f8e4m3_emits(_integer_descriptors())
    folded = _f16_to_f8e4m3_emits(
        _integer_descriptors(immediate_forms=_immediate_forms())
    )

    baseline_constants = _constant_names(baseline)
    folded_constants = _constant_names(folded)
    assert "sign_shift" in baseline_constants
    assert "sign_shift" not in folded_constants
    # This literal feeds both an immediate comparison and two select values.
    assert "source_exponent_one" in folded_constants
    assert len(folded) < len(baseline)
    assert any(emit.descriptor.key.endswith(".immediate") for emit in folded)


def test_out_of_range_literals_keep_the_register_form() -> None:
    narrow_shift = Immediate(
        "shift",
        ImmediateKind.UNSIGNED,
        bit_width=8,
        unsigned_max=3,
    )
    emits = _f16_to_f8e4m3_emits(
        _integer_descriptors(immediate_forms=_immediate_forms(shift=narrow_shift))
    )

    assert "sign_shift" in _constant_names(emits)
    assert any(
        emit.descriptor.key == "test.shift_right_logical"
        and emit.operands.get("rhs") == ValueRef.temporary("sign_shift")
        for emit in emits
    )
