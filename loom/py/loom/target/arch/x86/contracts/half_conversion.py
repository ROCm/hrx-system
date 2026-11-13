# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Legacy half conversions preserving scalar GPR and packed SIMD carriers."""

from loom.dialect.scalar import conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.contracts import DescriptorRule, Guard, Scalar, ValueRef, Vector

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}


def half_conversion_rules(
    descriptor_lookup: DescriptorLookup, *, evex: bool
) -> tuple[DescriptorRule, ...]:
    feature = "avx512" if evex else "f16c"
    rules = []
    for narrow in (False, True):
        mnemonic = "vcvtps2ph" if narrow else "vcvtph2ps"
        input_element, result_element = ("f32", "f16") if narrow else ("f16", "f32")
        descriptor = descriptor_lookup(f"x86.{feature}.{mnemonic}.xmm.xmm")
        move = descriptor_lookup(
            "x86.avx2.vmovd.gpr32.xmm" if narrow else "x86.avx2.vmovd.xmm.gpr32"
        )
        conversion_emit = emit_descriptor_op(
            descriptor=descriptor,
            operands={
                "input": ValueRef.operand("input")
                if narrow
                else ValueRef.temporary("bits")
            },
            results={
                "dst": ValueRef.temporary("bits")
                if narrow
                else ValueRef.result("result")
            },
            result_types={"dst": Vector("f16", lanes=2)} if narrow else None,
            immediates={"rounding": 0} if narrow else None,
        )
        move_emit = emit_descriptor_op(
            descriptor=move,
            operands={
                "input": ValueRef.temporary("bits")
                if narrow
                else ValueRef.operand("input")
            },
            results={
                "dst": ValueRef.result("result")
                if narrow
                else ValueRef.temporary("bits")
            },
            result_types=None if narrow else {"dst": Vector("f16", lanes=2)},
        )
        rules.append(
            DescriptorRule(
                source_op=conversion.scalar_fptrunc
                if narrow
                else conversion.scalar_extf,
                descriptor=descriptor,
                guards=(
                    Guard.value_type("input", Scalar(input_element)),
                    Guard.value_type("result", Scalar(result_element)),
                    Guard.low_value_register_class(
                        "result" if narrow else "input", "x86.gpr32"
                    ),
                ),
                emit=(conversion_emit, move_emit)
                if narrow
                else (move_emit, conversion_emit),
                priority=1,
            )
        )
        for lane_count in (2, 4, 8, 16) if evex else (2, 4, 8):
            input_suffix = _REGISTER_SUFFIXES[
                max(128, lane_count * (32 if narrow else 16))
            ]
            result_suffix = _REGISTER_SUFFIXES[
                max(128, lane_count * (16 if narrow else 32))
            ]
            descriptor = descriptor_lookup(
                f"x86.{feature}.{mnemonic}.{result_suffix}.{input_suffix}"
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_fptrunc if narrow else vector.vector_extf,
                    descriptor=descriptor,
                    guards=(
                        Guard.value_type(
                            "input", Vector(input_element, lanes=lane_count)
                        ),
                        Guard.value_type(
                            "result", Vector(result_element, lanes=lane_count)
                        ),
                        # Missing native features leave the exact scalar
                        # conversion available to packet legalization.
                        Guard.descriptor_available(descriptor),
                    ),
                    emit=(
                        emit_descriptor_op(
                            descriptor=descriptor,
                            operands={"input": ValueRef.operand("input")},
                            results={"dst": ValueRef.result("result")},
                            immediates={"rounding": 0} if narrow else None,
                        ),
                    ),
                )
            )
    for lane_count in (2, 4, 8) if evex else (2, 4):
        input_suffix = _REGISTER_SUFFIXES[max(128, lane_count * 16)]
        float_suffix = _REGISTER_SUFFIXES[max(128, lane_count * 32)]
        double_suffix = _REGISTER_SUFFIXES[lane_count * 64]
        half = descriptor_lookup(
            f"x86.{feature}.vcvtph2ps.{float_suffix}.{input_suffix}"
        )
        double_feature = "avx512" if evex else "avx2"
        double = descriptor_lookup(
            f"x86.{double_feature}.vcvtps2pd.{double_suffix}.{float_suffix}"
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_extf,
                descriptor=half,
                guards=(
                    Guard.value_type("input", Vector("f16", lanes=lane_count)),
                    Guard.value_type("result", Vector("f64", lanes=lane_count)),
                    Guard.descriptor_available(half),
                ),
                emit=(
                    emit_descriptor_op(
                        descriptor=half,
                        operands={"input": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("wide")},
                        result_types={"dst": Vector("f32", lanes=lane_count)},
                    ),
                    emit_descriptor_op(
                        descriptor=double,
                        operands={"input": ValueRef.temporary("wide")},
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        )
    return tuple(rules)
