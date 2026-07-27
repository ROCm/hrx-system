# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 target profile and descriptor-set row data.

This module owns the Python input rows that generators consume for x86
descriptor views and native target records. LLVMIR is a debug/inspection
projection and keeps its feature-string metadata under target/emit/llvmir so
native x86 binaries do not carry LLVMIR-only names or feature strings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from loom.target.arch.x86.packed_dot_data import (
    FEATURE_AVX10_2,
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
    FEATURE_AVX512_VNNI,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
    FEATURE_AVX_VNNI_INT16,
)

X86_DESCRIPTOR_SET_ORDINAL_NONE = (2**16) - 1

X86_REG_CLASS_GPR32 = "x86.gpr32"
X86_REG_CLASS_GPR64 = "x86.gpr64"
X86_REG_CLASS_XMM = "x86.xmm"
X86_REG_CLASS_YMM = "x86.ymm"
X86_REG_CLASS_ZMM = "x86.zmm"
X86_REG_CLASS_K = "x86.k"

X86_ISA_TIER_SCALAR = "scalar"
X86_ISA_TIER_SIMD128 = "simd128"
X86_ISA_TIER_AVX2 = "avx2"
X86_ISA_TIER_AVX512 = "avx512"
X86_ISA_TIER_PACKED_DOT = "packed_dot"

X86_FEATURE_PROFILE_NONE = "none"
X86_FEATURE_PROFILE_AVX512_VNNI = "avx512_vnni"
X86_FEATURE_PROFILE_AVX512_BF16 = "avx512_bf16"
X86_FEATURE_PROFILE_AVX_VNNI = "avx_vnni"
X86_FEATURE_PROFILE_AVX_VNNI_INT8 = "avx_vnni_int8"
X86_FEATURE_PROFILE_AVX_VNNI_INT16 = "avx_vnni_int16"
X86_FEATURE_PROFILE_AVX10_2 = "avx10_2"


@dataclass(frozen=True, slots=True)
class X86DescriptorSetInfo:
    generator_target: str
    key: str
    isa_tier: str
    register_classes: tuple[str, ...]
    storage_generator_target: str | None = None
    feature_profile: str = X86_FEATURE_PROFILE_NONE
    required_feature_bits: int = 0


@dataclass(frozen=True, slots=True)
class X86TargetProfileInfo:
    profile_key: str
    descriptor_generator_target: str
    descriptor_set_key: str
    register_classes: tuple[str, ...]
    contract_feature_bits: int = 0
    native_bundle_key: str | None = None


def _with_base_registers(*register_classes: str) -> tuple[str, ...]:
    return (X86_REG_CLASS_GPR32, X86_REG_CLASS_GPR64, *register_classes)


X86_DESCRIPTOR_SET_INFOS: tuple[X86DescriptorSetInfo, ...] = (
    X86DescriptorSetInfo(
        generator_target="scalar",
        key="x86.scalar.core",
        isa_tier=X86_ISA_TIER_SCALAR,
        register_classes=_with_base_registers(),
        storage_generator_target="avx512_packed_dot",
    ),
    X86DescriptorSetInfo(
        generator_target="simd128",
        key="x86.simd128.core",
        isa_tier=X86_ISA_TIER_SIMD128,
        register_classes=_with_base_registers(X86_REG_CLASS_XMM),
        storage_generator_target="avx512_packed_dot",
    ),
    X86DescriptorSetInfo(
        generator_target="avx2",
        key="x86.avx2.core",
        isa_tier=X86_ISA_TIER_AVX2,
        register_classes=_with_base_registers(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_packed_dot",
    ),
    X86DescriptorSetInfo(
        generator_target="avx512",
        key="x86.avx512.core",
        isa_tier=X86_ISA_TIER_AVX512,
        register_classes=_with_base_registers(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        storage_generator_target="avx512_packed_dot",
    ),
    X86DescriptorSetInfo(
        generator_target="packed_dot",
        key="x86.packed_dot.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_packed_dot",
        required_feature_bits=(
            FEATURE_AVX512_BF16
            | FEATURE_AVX512_VL
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
        ),
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_packed_dot",
        key="x86.avx512_packed_dot.core",
        isa_tier=X86_ISA_TIER_AVX512,
        register_classes=_with_base_registers(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        required_feature_bits=(
            FEATURE_AVX512_VNNI
            | FEATURE_AVX512_BF16
            | FEATURE_AVX512_VL
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
        ),
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_vnni",
        key="x86.avx512_vnni.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX512_VNNI,
        required_feature_bits=FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_bf16",
        key="x86.avx512_bf16.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX512_BF16,
        required_feature_bits=FEATURE_AVX512_BF16 | FEATURE_AVX512_VL,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni",
        key="x86.avx_vnni.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI,
        required_feature_bits=FEATURE_AVX_VNNI,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni_int8",
        key="x86.avx_vnni_int8.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI_INT8,
        required_feature_bits=FEATURE_AVX_VNNI_INT8,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni_int16",
        key="x86.avx_vnni_int16.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI_INT16,
        required_feature_bits=FEATURE_AVX_VNNI_INT16,
    ),
    X86DescriptorSetInfo(
        generator_target="avx10_2",
        key="x86.avx10_2.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_packed_dot",
        feature_profile=X86_FEATURE_PROFILE_AVX10_2,
        required_feature_bits=FEATURE_AVX10_2,
    ),
)

X86_TARGET_PROFILE_INFOS: tuple[X86TargetProfileInfo, ...] = (
    X86TargetProfileInfo(
        profile_key="x86.scalar",
        descriptor_generator_target="scalar",
        descriptor_set_key="x86.scalar.core",
        register_classes=_with_base_registers(),
        native_bundle_key="x86-scalar",
    ),
    X86TargetProfileInfo(
        profile_key="x86.simd128",
        descriptor_generator_target="simd128",
        descriptor_set_key="x86.simd128.core",
        register_classes=_with_base_registers(X86_REG_CLASS_XMM),
        native_bundle_key="x86-simd128",
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx2",
        descriptor_generator_target="avx2",
        descriptor_set_key="x86.avx2.core",
        register_classes=_with_base_registers(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        native_bundle_key="x86-avx2",
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx512",
        descriptor_generator_target="avx512",
        descriptor_set_key="x86.avx512.core",
        register_classes=_with_base_registers(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        native_bundle_key="x86-avx512",
    ),
    X86TargetProfileInfo(
        profile_key="x86.packed_dot",
        descriptor_generator_target="packed_dot",
        descriptor_set_key="x86.packed_dot.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        contract_feature_bits=(
            FEATURE_AVX512_BF16
            | FEATURE_AVX512_VL
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
        ),
        native_bundle_key="x86-packed-dot",
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx512_packed_dot",
        descriptor_generator_target="avx512_packed_dot",
        descriptor_set_key="x86.avx512_packed_dot.core",
        register_classes=_with_base_registers(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        contract_feature_bits=(
            FEATURE_AVX512_VNNI
            | FEATURE_AVX512_BF16
            | FEATURE_AVX512_VL
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
        ),
        native_bundle_key="x86-avx512-packed-dot",
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx512_vnni",
        descriptor_generator_target="avx512_vnni",
        descriptor_set_key="x86.avx512_vnni.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        contract_feature_bits=FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx512_bf16",
        descriptor_generator_target="avx512_bf16",
        descriptor_set_key="x86.avx512_bf16.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        contract_feature_bits=FEATURE_AVX512_BF16 | FEATURE_AVX512_VL,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx_vnni",
        descriptor_generator_target="avx_vnni",
        descriptor_set_key="x86.avx_vnni.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        contract_feature_bits=FEATURE_AVX_VNNI,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx_vnni_int8",
        descriptor_generator_target="avx_vnni_int8",
        descriptor_set_key="x86.avx_vnni_int8.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        contract_feature_bits=FEATURE_AVX_VNNI_INT8,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx_vnni_int16",
        descriptor_generator_target="avx_vnni_int16",
        descriptor_set_key="x86.avx_vnni_int16.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        contract_feature_bits=FEATURE_AVX_VNNI_INT16,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx10_2",
        descriptor_generator_target="avx10_2",
        descriptor_set_key="x86.avx10_2.core",
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        contract_feature_bits=FEATURE_AVX10_2,
    ),
)

X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS: tuple[str | None, ...] = (
    None,
    "x86.avx512",
    "x86.packed_dot",
    "x86.avx512_packed_dot",
    "x86.scalar",
    "x86.simd128",
    "x86.avx2",
)


def _validate_unique(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate x86 {description} '{value}'")
        seen.add(value)


def validate_x86_target_info_tables(
    descriptor_sets: tuple[X86DescriptorSetInfo, ...] = X86_DESCRIPTOR_SET_INFOS,
    target_profiles: tuple[X86TargetProfileInfo, ...] = X86_TARGET_PROFILE_INFOS,
) -> None:
    _validate_unique(
        (info.generator_target for info in descriptor_sets),
        "descriptor generator target",
    )
    _validate_unique((info.key for info in descriptor_sets), "descriptor set key")
    _validate_unique((info.profile_key for info in target_profiles), "profile key")
    _validate_unique(
        (
            info.native_bundle_key
            for info in target_profiles
            if info.native_bundle_key is not None
        ),
        "native bundle key",
    )

    descriptor_sets_by_generator_target = {
        info.generator_target: info for info in descriptor_sets
    }
    descriptor_sets_by_key = {info.key: info for info in descriptor_sets}
    for info in descriptor_sets:
        if info.storage_generator_target is None:
            continue
        storage_info = descriptor_sets_by_generator_target.get(
            info.storage_generator_target
        )
        if storage_info is None:
            raise ValueError(
                f"x86 descriptor generator target '{info.generator_target}' "
                f"references unknown storage target '{info.storage_generator_target}'"
            )
        if storage_info.storage_generator_target is not None:
            raise ValueError(
                f"x86 descriptor generator target '{info.generator_target}' uses "
                f"view-only target '{storage_info.generator_target}' as storage"
            )
        if (
            info.feature_profile != X86_FEATURE_PROFILE_NONE
            and info.required_feature_bits == 0
        ):
            raise ValueError(
                f"x86 descriptor generator target '{info.generator_target}' "
                "names a feature profile without required feature bits"
            )

    for profile_info in target_profiles:
        descriptor_set = descriptor_sets_by_generator_target.get(
            profile_info.descriptor_generator_target
        )
        if descriptor_set is None:
            raise ValueError(
                f"x86 profile '{profile_info.profile_key}' references unknown "
                "descriptor generator target "
                f"'{profile_info.descriptor_generator_target}'"
            )
        if descriptor_set.key != profile_info.descriptor_set_key:
            raise ValueError(
                f"x86 profile '{profile_info.profile_key}' expects descriptor set "
                f"'{profile_info.descriptor_set_key}', found '{descriptor_set.key}'"
            )
        if profile_info.descriptor_set_key not in descriptor_sets_by_key:
            raise ValueError(
                f"x86 profile '{profile_info.profile_key}' references unknown "
                f"descriptor set '{profile_info.descriptor_set_key}'"
            )

    target_profiles_by_key = {info.profile_key: info for info in target_profiles}
    for profile_key in X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS:
        if profile_key is None:
            continue
        selector_profile_info = target_profiles_by_key.get(profile_key)
        if selector_profile_info is None:
            raise ValueError(
                f"x86 native selector references unknown profile '{profile_key}'"
            )
        if selector_profile_info.native_bundle_key is None:
            raise ValueError(
                f"x86 native selector profile '{profile_key}' has no native bundle key"
            )


def sorted_descriptor_set_infos() -> tuple[X86DescriptorSetInfo, ...]:
    return tuple(sorted(X86_DESCRIPTOR_SET_INFOS, key=lambda info: info.key))


def x86_descriptor_set_ordinal(key: str) -> int:
    for ordinal, info in enumerate(sorted_descriptor_set_infos()):
        if info.key == key:
            return ordinal
    raise ValueError(f"unknown x86 descriptor set '{key}'")


def x86_descriptor_set_info_by_generator_target(
    generator_target: str,
) -> X86DescriptorSetInfo:
    for info in X86_DESCRIPTOR_SET_INFOS:
        if info.generator_target == generator_target:
            return info
    raise ValueError(f"unknown x86 descriptor generator target '{generator_target}'")


def x86_descriptor_set_info_by_key(key: str) -> X86DescriptorSetInfo:
    for info in X86_DESCRIPTOR_SET_INFOS:
        if info.key == key:
            return info
    raise ValueError(f"unknown x86 descriptor set '{key}'")


def x86_descriptor_set_storage_info_by_generator_target(
    generator_target: str,
) -> X86DescriptorSetInfo:
    info = x86_descriptor_set_info_by_generator_target(generator_target)
    if info.storage_generator_target is None:
        return info
    storage_info = x86_descriptor_set_info_by_generator_target(
        info.storage_generator_target
    )
    if storage_info.storage_generator_target is not None:
        raise ValueError(
            f"x86 descriptor generator target '{generator_target}' uses "
            f"view-only target '{storage_info.generator_target}' as storage"
        )
    return storage_info


def x86_descriptor_set_view_infos_by_storage_generator_target(
    storage_generator_target: str,
) -> tuple[X86DescriptorSetInfo, ...]:
    storage_info = x86_descriptor_set_info_by_generator_target(storage_generator_target)
    if storage_info.storage_generator_target is not None:
        raise ValueError(
            f"x86 descriptor generator target '{storage_generator_target}' "
            "is a view target, not a storage target"
        )
    return tuple(
        sorted(
            (
                info
                for info in X86_DESCRIPTOR_SET_INFOS
                if info.storage_generator_target == storage_generator_target
            ),
            key=lambda info: info.key,
        )
    )


def sorted_target_profile_infos() -> tuple[X86TargetProfileInfo, ...]:
    return tuple(sorted(X86_TARGET_PROFILE_INFOS, key=lambda info: info.profile_key))


def x86_target_profile_info_by_key(profile_key: str) -> X86TargetProfileInfo:
    for info in X86_TARGET_PROFILE_INFOS:
        if info.profile_key == profile_key:
            return info
    raise ValueError(f"unknown x86 target profile '{profile_key}'")


validate_x86_target_info_tables()
