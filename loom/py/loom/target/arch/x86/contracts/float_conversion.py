# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar and packed float-width conversion rules."""

from loom.dialect.scalar import conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.contracts import DescriptorRule, Guard, Scalar, ValueRef, Vector

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}


def float_conversion_rules(
    descriptor_lookup: DescriptorLookup, *, evex: bool
) -> tuple[DescriptorRule, ...]:
    feature = "avx512" if evex else "avx2"
    priority = int(evex)
    rules = []
    lane_counts = (2, 4, 8) if evex else (2, 4)
    for narrow in (False, True):
        source_element, result_element = ("f64", "f32") if narrow else ("f32", "f64")
        scalar_mnemonic = "vcvtsd2ss" if narrow else "vcvtss2sd"
        scalar = descriptor_lookup(f"x86.{feature}.{scalar_mnemonic}.xmm")
        rules.append(
            DescriptorRule(
                source_op=(
                    conversion.scalar_fptrunc if narrow else conversion.scalar_extf
                ),
                descriptor=scalar,
                guards=(
                    Guard.value_type("input", Scalar(source_element)),
                    Guard.value_type("result", Scalar(result_element)),
                ),
                emit=(
                    emit_descriptor_op(
                        descriptor=scalar,
                        operands={
                            "passthrough": ValueRef.operand("input"),
                            "input": ValueRef.operand("input"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
                priority=priority,
            )
        )
        mnemonic = "vcvtpd2ps" if narrow else "vcvtps2pd"
        for lane_count in lane_counts:
            source_width = lane_count * (64 if narrow else 32)
            result_width = lane_count * (32 if narrow else 64)
            source_suffix = _REGISTER_SUFFIXES[max(128, source_width)]
            result_suffix = _REGISTER_SUFFIXES[max(128, result_width)]
            descriptor = descriptor_lookup(
                f"x86.{feature}.{mnemonic}.{result_suffix}.{source_suffix}"
            )
            rules.append(
                DescriptorRule(
                    source_op=(vector.vector_fptrunc if narrow else vector.vector_extf),
                    descriptor=descriptor,
                    guards=(
                        Guard.value_type(
                            "input", Vector(source_element, lanes=lane_count)
                        ),
                        Guard.value_type(
                            "result", Vector(result_element, lanes=lane_count)
                        ),
                    ),
                    emit=(
                        emit_descriptor_op(
                            descriptor=descriptor,
                            operands={"input": ValueRef.operand("input")},
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                    priority=priority,
                )
            )
    for result_element, lane_counts in (
        ("f32", (16,) if evex else (2, 4, 8)),
        ("f64", (2, 4, 8) if evex else (2, 4)),
    ):
        for lane_count in lane_counts:
            # BF16 is the high half of an F32 word. Integer expansion and shift
            # preserve all payload bits, including signed zeros and subnormals.
            word_width = max(128, lane_count * 32)
            word_feature = "avx512" if word_width == 512 else "avx2"
            word_suffix = _REGISTER_SUFFIXES[word_width]
            input_suffix = _REGISTER_SUFFIXES[max(128, lane_count * 16)]
            extend = descriptor_lookup(
                f"x86.{word_feature}.vpmovzxwd.{word_suffix}.{input_suffix}"
            )
            shift = descriptor_lookup(f"x86.{word_feature}.vpslld.{word_suffix}")
            emits = [
                emit_descriptor_op(
                    descriptor=extend,
                    operands={"source": ValueRef.operand("input")},
                    results={"dst": ValueRef.temporary("words")},
                    result_types={"dst": Vector("i32", lanes=lane_count)},
                ),
                emit_descriptor_op(
                    descriptor=shift,
                    operands={"source": ValueRef.temporary("words")},
                    results={
                        "dst": ValueRef.result("result")
                        if result_element == "f32"
                        else ValueRef.temporary("wide")
                    },
                    result_types={"dst": Vector("f32", lanes=lane_count)},
                    immediates={"shift": 16},
                ),
            ]
            descriptor = shift
            if result_element == "f64":
                result_suffix = _REGISTER_SUFFIXES[lane_count * 64]
                descriptor = descriptor_lookup(
                    f"x86.{feature}.vcvtps2pd.{result_suffix}.{word_suffix}"
                )
                emits.append(
                    emit_descriptor_op(
                        descriptor=descriptor,
                        operands={"input": ValueRef.temporary("wide")},
                        results={"dst": ValueRef.result("result")},
                    )
                )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_extf,
                    descriptor=descriptor,
                    guards=(
                        Guard.value_type("input", Vector("bf16", lanes=lane_count)),
                        Guard.value_type(
                            "result", Vector(result_element, lanes=lane_count)
                        ),
                    ),
                    emit=tuple(emits),
                    priority=priority,
                )
            )
    return tuple(rules)
