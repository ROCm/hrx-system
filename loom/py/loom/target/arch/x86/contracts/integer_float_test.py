# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import struct

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.integer_float import (
    avx2_integer_float_rules,
    avx2_vector_integer_float_rules,
    avx512_integer_float_rules,
    avx512_vector_integer_float_rules,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_INTEGER_FLOAT_I32_FORMS,
    AVX512_INTEGER_FLOAT_I32_FORMS,
    AVX512_INTEGER_FLOAT_I64_FORMS,
    IntegerFloatVectorForm,
)
from loom.target.contracts import (
    DescriptorRule,
    GuardKind,
    ReadOnlyDataLiteral,
    Scalar,
    ValueRef,
    Vector,
    descriptor_by_key,
)


def _avx2_descriptor(key: str):
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


def _avx512_descriptor(key: str):
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _value_type(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def _cell(rule: DescriptorRule):
    return (
        rule.source_op,
        _value_type(rule, "input"),
        _value_type(rule, "result"),
    )


def _vector_cell(source_op, form: IntegerFloatVectorForm):
    return (
        source_op,
        Vector(form.source.name, lanes=form.lane_count),
        Vector(form.result.name, lanes=form.lane_count),
    )


def test_scalar_rules_cover_signed_unsigned_types_and_native_overlays() -> None:
    base_rules = avx2_integer_float_rules(_avx2_descriptor)
    native_rules = avx512_integer_float_rules(_avx512_descriptor)
    for rule in base_rules:
        rule.validate(X86_AVX2_DESCRIPTOR_SET)
    for rule in native_rules:
        rule.validate(X86_AVX512_CORE_DESCRIPTOR_SET)

    operations = (scalar_conversion.scalar_sitofp, scalar_conversion.scalar_uitofp)
    assert {_cell(rule) for rule in base_rules} == {
        (source_op, Scalar(f"i{source_width}"), Scalar(f"f{result_width}"))
        for source_op in operations
        for source_width in (1, 8, 16, 32, 64)
        for result_width in (32, 64)
    }
    assert {_cell(rule) for rule in native_rules} == {
        (
            scalar_conversion.scalar_uitofp,
            Scalar(f"i{source_width}"),
            Scalar(f"f{result_width}"),
        )
        for source_width in (32, 64)
        for result_width in (32, 64)
    }
    base_by_cell = {_cell(rule): rule for rule in base_rules}
    for rule in native_rules:
        assert rule.priority > base_by_cell[_cell(rule)].priority
        assert [emit.descriptor.mnemonic for emit in rule.emit] == [
            "vxorps",
            "vcvtusi2ss"
            if _value_type(rule, "result") == Scalar("f32")
            else "vcvtusi2sd",
        ]
    for result_width in (32, 64):
        unsigned64 = base_by_cell[
            (
                scalar_conversion.scalar_uitofp,
                Scalar("i64"),
                Scalar(f"f{result_width}"),
            )
        ]
        assert [emit.descriptor.mnemonic for emit in unsigned64.emit[2:]] == [
            "shr",
            "and",
            "or",
            "vcvtsi2ss" if result_width == 32 else "vcvtsi2sd",
            "vaddss" if result_width == 32 else "vaddsd",
            "vmovq",
            "vblendvpd",
        ]


def test_vector_rules_cover_every_native_or_composed_packet_cell() -> None:
    avx2_rules = avx2_vector_integer_float_rules(_avx2_descriptor)
    avx512_rules = avx512_vector_integer_float_rules(_avx512_descriptor)
    for rule in avx2_rules:
        rule.validate(X86_AVX2_DESCRIPTOR_SET)
    for rule in avx512_rules:
        rule.validate(X86_AVX512_CORE_DESCRIPTOR_SET)

    operations = (vector.vector_sitofp, vector.vector_uitofp)
    assert {_cell(rule) for rule in avx2_rules} == {
        _vector_cell(source_op, form)
        for form in AVX2_INTEGER_FLOAT_I32_FORMS
        for source_op in operations
    }
    assert {_cell(rule) for rule in avx512_rules} == {
        *(
            _vector_cell(vector.vector_sitofp, form)
            for form in AVX512_INTEGER_FLOAT_I32_FORMS[-2:]
        ),
        *(
            _vector_cell(vector.vector_uitofp, form)
            for form in AVX512_INTEGER_FLOAT_I32_FORMS
        ),
        *(
            _vector_cell(source_op, form)
            for form in AVX512_INTEGER_FLOAT_I64_FORMS
            for source_op in operations
        ),
    }
    assert all(rule.priority == 1 for rule in avx512_rules)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_avx2_unsigned_i32_f32_jams_before_rounding() -> None:
    values = {
        0,
        1,
        3,
        (1 << 24) - 1,
        1 << 24,
        (1 << 31) - 1,
        1 << 31,
        (1 << 31) + 1,
        2_336_232_577,
        (1 << 32) - 2,
        (1 << 32) - 1,
    }
    for value in values:
        direct = _f32(float(value))
        if value < (1 << 31):
            composed = direct
        else:
            jammed = (value >> 1) | (value & 1)
            composed = _f32(_f32(float(jammed)) * 2.0)
        assert struct.pack("<f", composed) == struct.pack("<f", direct)

    for rule in avx2_vector_integer_float_rules(_avx2_descriptor):
        if rule.source_op is not vector.vector_uitofp:
            continue
        result_type = _value_type(rule, "result")
        suffix = "xmm" if result_type.lanes <= 4 else "ymm"
        if result_type.element == "f32":
            assert [emit.descriptor.mnemonic for emit in rule.emit] == [
                "vpsrld",
                "vpslld",
                "vpsrld",
                "vpor",
                "vcvtdq2ps",
                "vcvtdq2ps",
                "vaddps",
                "vblendvps",
            ]
            assert rule.emit[-1].descriptor.key == f"x86.avx2.vblendvps.{suffix}"


def test_avx2_unsigned_i32_f64_uses_exact_mantissa_injection() -> None:
    for rule in avx2_vector_integer_float_rules(_avx2_descriptor):
        if (
            rule.source_op is not vector.vector_uitofp
            or _value_type(rule, "result").element != "f64"
        ):
            continue
        result_type = _value_type(rule, "result")
        suffix = "xmm" if result_type.lanes == 2 else "ymm"
        assert [emit.descriptor.mnemonic for emit in rule.emit] == [
            "vpmovzxdq",
            "vmovdqu",
            "vpor",
            "vsubpd",
        ]
        literal = rule.emit[1].immediates["data"]
        assert isinstance(literal, ReadOnlyDataLiteral)
        assert literal.data == (0x4330000000000000).to_bytes(8, "little") * (
            2 if suffix == "xmm" else 4
        )
        assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
