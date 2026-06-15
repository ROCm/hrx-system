# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 LLVMIR debug projection rows.

LLVMIR is not the production x86 target path. These rows intentionally live
under the LLVMIR target package so native x86 target records can consume the
same profile catalog without carrying LLVM profile names or `llc` feature
strings in native-only binaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from loom.target.arch.x86.target_info import (
    X86_TARGET_PROFILE_INFOS,
    X86TargetProfileInfo,
    x86_target_profile_info_by_key,
)

X86_LLVMIR_FEATURES_SIMD128 = ("+sse2",)
X86_LLVMIR_FEATURES_AVX2 = ("+avx", "+avx2", "+fma")
X86_LLVMIR_FEATURES_AVX512 = (
    "+avx512f",
    "+avx512bw",
    "+avx512dq",
    "+avx512vl",
    "+fma",
)
X86_LLVMIR_FEATURES_PACKED_DOT = (
    "+avx512bf16",
    "+avx512vl",
    "+avxvnni",
    "+avxvnniint8",
)
X86_LLVMIR_FEATURES_AVX512_PACKED_DOT = (
    "+avx512f",
    "+avx512bw",
    "+avx512dq",
    "+avx512vl",
    "+fma",
    "+avx512bf16",
    "+avx512vnni",
    "+avxvnni",
    "+avxvnniint8",
)


@dataclass(frozen=True, slots=True)
class X86LlvmirProjectionInfo:
    profile_key: str
    debug_profile_key: str
    target_features: tuple[str, ...] = ()

    @property
    def target_profile(self) -> X86TargetProfileInfo:
        return x86_target_profile_info_by_key(self.profile_key)


X86_LLVMIR_PROJECTION_INFOS: tuple[X86LlvmirProjectionInfo, ...] = (
    X86LlvmirProjectionInfo(
        profile_key="x86.scalar",
        debug_profile_key="x86_64-object",
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.simd128",
        debug_profile_key="x86_64-simd128-object",
        target_features=X86_LLVMIR_FEATURES_SIMD128,
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx2",
        debug_profile_key="x86_64-avx2-object",
        target_features=X86_LLVMIR_FEATURES_AVX2,
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx512",
        debug_profile_key="x86_64-avx512-object",
        target_features=X86_LLVMIR_FEATURES_AVX512,
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.packed_dot",
        debug_profile_key="x86_64-packed-dot-object",
        target_features=X86_LLVMIR_FEATURES_PACKED_DOT,
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx512_packed_dot",
        debug_profile_key="x86_64-avx512-packed-dot-object",
        target_features=X86_LLVMIR_FEATURES_AVX512_PACKED_DOT,
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx512_vnni",
        debug_profile_key="x86_64-avx512-vnni-object",
        target_features=(
            "+avx512f",
            "+avx512bw",
            "+avx512vl",
            "+avx512vnni",
        ),
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx512_bf16",
        debug_profile_key="x86_64-avx512-bf16-object",
        target_features=("+avx512bf16", "+avx512vl"),
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx_vnni",
        debug_profile_key="x86_64-avx-vnni-object",
        target_features=("+avxvnni",),
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx_vnni_int8",
        debug_profile_key="x86_64-avx-vnni-int8-object",
        target_features=("+avxvnniint8",),
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx_vnni_int16",
        debug_profile_key="x86_64-avx-vnni-int16-object",
        target_features=("+avxvnniint16",),
    ),
    X86LlvmirProjectionInfo(
        profile_key="x86.avx10_2",
        debug_profile_key="x86_64-avx10-2-object",
        target_features=("+avx10.2",),
    ),
)


def _validate_unique(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate x86 LLVMIR {description} '{value}'")
        seen.add(value)


def validate_x86_llvmir_projection_infos(
    projections: tuple[X86LlvmirProjectionInfo, ...] = X86_LLVMIR_PROJECTION_INFOS,
) -> None:
    _validate_unique((info.profile_key for info in projections), "profile key")
    _validate_unique(
        (info.debug_profile_key for info in projections),
        "debug profile key",
    )
    target_profile_keys = {info.profile_key for info in X86_TARGET_PROFILE_INFOS}
    for info in projections:
        if info.profile_key not in target_profile_keys:
            raise ValueError(
                f"x86 LLVMIR projection '{info.debug_profile_key}' references "
                f"unknown x86 profile '{info.profile_key}'"
            )


def sorted_x86_llvmir_projection_infos() -> tuple[X86LlvmirProjectionInfo, ...]:
    return tuple(
        sorted(X86_LLVMIR_PROJECTION_INFOS, key=lambda info: info.debug_profile_key)
    )


def x86_llvmir_projection_info_by_profile_key(
    profile_key: str,
) -> X86LlvmirProjectionInfo:
    for info in X86_LLVMIR_PROJECTION_INFOS:
        if info.profile_key == profile_key:
            return info
    raise ValueError(f"unknown x86 LLVMIR projection profile '{profile_key}'")


def x86_llvmir_target_feature_string(info: X86LlvmirProjectionInfo) -> str:
    return ",".join(info.target_features)


validate_x86_llvmir_projection_infos()
