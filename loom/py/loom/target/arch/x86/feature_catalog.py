# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical x86 optional-feature vocabulary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class X86FeatureInfo:
    spelling: str
    c_enum_suffix: str
    c_feature_macro: str
    runtime_cpu_suffix: str | None
    doc: str


X86_FEATURE_INFOS: tuple[X86FeatureInfo, ...] = (
    X86FeatureInfo(
        "avx512vnni",
        "AVX512VNNI",
        "LOOM_X86_FEATURE_AVX512_VNNI",
        "AVX512VNNI",
        "AVX-512 byte and word dot products.",
    ),
    X86FeatureInfo(
        "avx512vl",
        "AVX512VL",
        "LOOM_X86_FEATURE_AVX512_VL",
        "AVX512VL",
        "AVX-512 128-bit and 256-bit encodings.",
    ),
    X86FeatureInfo(
        "avxvnni",
        "AVXVNNI",
        "LOOM_X86_FEATURE_AVX_VNNI",
        "AVXVNNI",
        "VEX AVX-VNNI byte and word dot products.",
    ),
    X86FeatureInfo(
        "avxvnniint8",
        "AVXVNNIINT8",
        "LOOM_X86_FEATURE_AVX_VNNI_INT8",
        "AVXVNNIINT8",
        "AVX-VNNI-INT8 byte dot products.",
    ),
    X86FeatureInfo(
        "avxvnniint16",
        "AVXVNNIINT16",
        "LOOM_X86_FEATURE_AVX_VNNI_INT16",
        "AVXVNNIINT16",
        "AVX-VNNI-INT16 word dot products.",
    ),
    X86FeatureInfo(
        "avx10.2",
        "AVX10_2",
        "LOOM_X86_FEATURE_AVX10_2",
        None,
        "AVX10.2 extended vector instructions.",
    ),
    X86FeatureInfo(
        "avx512bf16",
        "AVX512BF16",
        "LOOM_X86_FEATURE_AVX512_BF16",
        "AVX512BF16",
        "AVX-512 bfloat16 conversions and dot products.",
    ),
    X86FeatureInfo(
        "avx512fp16",
        "AVX512FP16",
        "LOOM_X86_FEATURE_AVX512_FP16",
        "AVX512FP16",
        "AVX-512 half-precision arithmetic and conversions.",
    ),
    X86FeatureInfo(
        "avxneconvert",
        "AVXNECONVERT",
        "LOOM_X86_FEATURE_AVX_NE_CONVERT",
        "AVXNECONVERT",
        "AVX-NE-CONVERT packed floating-point conversions.",
    ),
    X86FeatureInfo(
        "f16c",
        "F16C",
        "LOOM_X86_FEATURE_F16C",
        "F16C",
        "VEX half and single-precision conversions.",
    ),
    X86FeatureInfo(
        "bmi2",
        "BMI2",
        "LOOM_X86_FEATURE_BMI2",
        "BMI2",
        "BMI2 scalar bit manipulation instructions.",
    ),
    X86FeatureInfo(
        "bmi",
        "BMI",
        "LOOM_X86_FEATURE_BMI",
        "BMI",
        "BMI1 scalar bit manipulation instructions.",
    ),
    X86FeatureInfo(
        "popcnt",
        "POPCNT",
        "LOOM_X86_FEATURE_POPCNT",
        "POPCNT",
        "Scalar population-count instructions.",
    ),
    X86FeatureInfo(
        "lzcnt",
        "LZCNT",
        "LOOM_X86_FEATURE_LZCNT",
        "LZCNT",
        "Scalar leading-zero-count instructions.",
    ),
)
