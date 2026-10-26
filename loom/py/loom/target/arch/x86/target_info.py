# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 target profile and descriptor-set row data.

This module owns the Python input rows that generators consume for x86
descriptor views and native target records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX10_2,
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_FP16,
    FEATURE_AVX512_VL,
    FEATURE_AVX512_VNNI,
    FEATURE_AVX_NE_CONVERT,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
    FEATURE_AVX_VNNI_INT16,
    FEATURE_BMI,
    FEATURE_BMI2,
    FEATURE_F16C,
    FEATURE_LZCNT,
    FEATURE_POPCNT,
    X86_KNOWN_FEATURE_BITS,
)

X86_DESCRIPTOR_SET_ORDINAL_NONE = (2**16) - 1

X86_REG_CLASS_GPR32 = "x86.gpr32"
X86_REG_CLASS_GPR64 = "x86.gpr64"
X86_REG_CLASS_XMM = "x86.xmm"
X86_REG_CLASS_YMM = "x86.ymm"
X86_REG_CLASS_ZMM = "x86.zmm"
X86_REG_CLASS_K = "x86.k"

X86_ISA_TIER_SCALAR = "scalar"
X86_ISA_TIER_AVX2 = "avx2"
X86_ISA_TIER_AVX512 = "avx512"
X86_ISA_TIER_PACKED_DOT = "packed_dot"

X86_FEATURE_PROFILE_NONE = "none"
X86_FEATURE_PROFILE_AVX512_VNNI = "avx512_vnni"
X86_FEATURE_PROFILE_AVX512_BF16 = "avx512_bf16"
X86_FEATURE_PROFILE_AVX512_FP16 = "avx512_fp16"
X86_FEATURE_PROFILE_AVX_VNNI = "avx_vnni"
X86_FEATURE_PROFILE_AVX_VNNI_INT8 = "avx_vnni_int8"
X86_FEATURE_PROFILE_AVX_VNNI_INT16 = "avx_vnni_int16"
X86_FEATURE_PROFILE_AVX10_2 = "avx10_2"

_X86_NATIVE_SCALAR_FEATURE_BITS = (
    FEATURE_BMI | FEATURE_BMI2 | FEATURE_POPCNT | FEATURE_LZCNT
)


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
    feature_contract_set_key: str | None = None
    supported_feature_bits: int = 0
    required_cpu_features: tuple[str, ...] = ()
    automatic_priority: int = 0


@dataclass(frozen=True, slots=True)
class X86ProcessorMatchInfo:
    vendor: str
    family: int
    first_model: int
    last_model: int


@dataclass(frozen=True, slots=True)
class X86TargetTuneInfo:
    spelling: str
    stable_value: int
    schedule_model_key: str


@dataclass(frozen=True, slots=True)
class X86DeploymentProfileInfo:
    spelling: str
    tune_spelling: str
    cpu_features: tuple[str, ...]
    processor_matches: tuple[X86ProcessorMatchInfo, ...]


X86_TARGET_TUNE_INFOS: tuple[X86TargetTuneInfo, ...] = (
    X86TargetTuneInfo(
        spelling="znver5",
        stable_value=1,
        schedule_model_key="x86.znver5",
    ),
    X86TargetTuneInfo(
        spelling="znver6",
        stable_value=2,
        schedule_model_key="x86.znver6",
    ),
)

_X86_ZNVER5_CPU_FEATURES = (
    "SSE3",
    "SSSE3",
    "SSE41",
    "SSE42",
    "SSE4A",
    "BMI",
    "BMI2",
    "POPCNT",
    "LZCNT",
    "AVX",
    "FMA",
    "F16C",
    "AVX2",
    "AVXVNNI",
    "AVX512F",
    "AVX512CD",
    "AVX512VL",
    "AVX512DQ",
    "AVX512BW",
    "AVX512IFMA",
    "AVX512VBMI",
    "AVX512VPOPCNTDQ",
    "AVX512VNNI",
    "AVX512VBMI2",
    "AVX512BITALG",
    "AVX512BF16",
)

X86_DEPLOYMENT_PROFILE_INFOS: tuple[X86DeploymentProfileInfo, ...] = (
    X86DeploymentProfileInfo(
        spelling="znver5",
        tune_spelling="znver5",
        cpu_features=_X86_ZNVER5_CPU_FEATURES,
        processor_matches=(
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0x00, 0x4F),
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0x60, 0x77),
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0xD0, 0xD7),
        ),
    ),
    X86DeploymentProfileInfo(
        spelling="znver6",
        tune_spelling="znver6",
        cpu_features=(
            *_X86_ZNVER5_CPU_FEATURES,
            "AVXVNNIINT8",
            "AVXNECONVERT",
            "AVX512FP16",
        ),
        processor_matches=(
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0x50, 0x5F),
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0x80, 0xCF),
            X86ProcessorMatchInfo("AuthenticAMD", 0x1A, 0xD8, 0xE7),
        ),
    ),
)


def _with_base_registers(*register_classes: str) -> tuple[str, ...]:
    return (X86_REG_CLASS_GPR32, X86_REG_CLASS_GPR64, *register_classes)


X86_DESCRIPTOR_SET_INFOS: tuple[X86DescriptorSetInfo, ...] = (
    X86DescriptorSetInfo(
        generator_target="scalar",
        key="x86.scalar.core",
        isa_tier=X86_ISA_TIER_SCALAR,
        register_classes=_with_base_registers(),
        storage_generator_target="avx512_features",
    ),
    X86DescriptorSetInfo(
        generator_target="avx2",
        key="x86.avx2.core",
        isa_tier=X86_ISA_TIER_AVX2,
        register_classes=_with_base_registers(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_features",
    ),
    X86DescriptorSetInfo(
        generator_target="avx2_features",
        key="x86.avx2_features.core",
        isa_tier=X86_ISA_TIER_AVX2,
        register_classes=_with_base_registers(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_features",
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
        storage_generator_target="avx512_features",
    ),
    X86DescriptorSetInfo(
        generator_target="packed_dot",
        key="x86.packed_dot.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_features",
        required_feature_bits=(
            FEATURE_AVX512_BF16
            | FEATURE_AVX512_VL
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
        ),
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_features",
        key="x86.avx512_features.core",
        isa_tier=X86_ISA_TIER_AVX512,
        register_classes=_with_base_registers(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_vnni",
        key="x86.avx512_vnni.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX512_VNNI,
        required_feature_bits=FEATURE_AVX512_VNNI | FEATURE_AVX512_VL,
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_bf16",
        key="x86.avx512_bf16.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(
            X86_REG_CLASS_GPR64,
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX512_BF16,
        required_feature_bits=FEATURE_AVX512_BF16 | FEATURE_AVX512_VL,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni",
        key="x86.avx_vnni.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI,
        required_feature_bits=FEATURE_AVX_VNNI,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni_int8",
        key="x86.avx_vnni_int8.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI_INT8,
        required_feature_bits=FEATURE_AVX_VNNI_INT8,
    ),
    X86DescriptorSetInfo(
        generator_target="avx_vnni_int16",
        key="x86.avx_vnni_int16.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX_VNNI_INT16,
        required_feature_bits=FEATURE_AVX_VNNI_INT16,
    ),
    X86DescriptorSetInfo(
        generator_target="avx10_2",
        key="x86.avx10_2.core",
        isa_tier=X86_ISA_TIER_PACKED_DOT,
        register_classes=(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM, X86_REG_CLASS_ZMM),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX10_2,
        required_feature_bits=FEATURE_AVX10_2,
    ),
    X86DescriptorSetInfo(
        generator_target="avx512_fp16",
        key="x86.avx512_fp16.core",
        isa_tier=X86_ISA_TIER_AVX512,
        register_classes=(
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
        storage_generator_target="avx512_features",
        feature_profile=X86_FEATURE_PROFILE_AVX512_FP16,
        required_feature_bits=FEATURE_AVX512_FP16 | FEATURE_AVX512_VL,
    ),
)

X86_TARGET_PROFILE_INFOS: tuple[X86TargetProfileInfo, ...] = (
    X86TargetProfileInfo(
        profile_key="x86.scalar",
        descriptor_generator_target="scalar",
        descriptor_set_key="x86.scalar.core",
        register_classes=_with_base_registers(),
        native_bundle_key="x86-scalar",
        feature_contract_set_key="x86.scalar.core",
        supported_feature_bits=_X86_NATIVE_SCALAR_FEATURE_BITS,
        automatic_priority=1,
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx2",
        descriptor_generator_target="avx2",
        descriptor_set_key="x86.avx2.core",
        register_classes=_with_base_registers(X86_REG_CLASS_XMM, X86_REG_CLASS_YMM),
        native_bundle_key="x86-avx2",
        feature_contract_set_key="x86.avx2_features.core",
        supported_feature_bits=(
            FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
            | FEATURE_AVX_VNNI_INT16
            | FEATURE_AVX_NE_CONVERT
            | FEATURE_F16C
            | _X86_NATIVE_SCALAR_FEATURE_BITS
        ),
        required_cpu_features=("AVX", "FMA", "AVX2"),
        automatic_priority=3,
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
        contract_feature_bits=FEATURE_AVX512_VL,
        native_bundle_key="x86-avx512",
        feature_contract_set_key="x86.avx512_features.core",
        supported_feature_bits=X86_KNOWN_FEATURE_BITS,
        required_cpu_features=(
            "AVX",
            "FMA",
            "AVX2",
            "AVX512F",
            "AVX512DQ",
            "AVX512BW",
        ),
        automatic_priority=4,
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
        feature_contract_set_key="x86.packed_dot.core",
        supported_feature_bits=(
            FEATURE_AVX512_VNNI
            | FEATURE_AVX512_VL
            | FEATURE_AVX512_BF16
            | FEATURE_AVX_VNNI
            | FEATURE_AVX_VNNI_INT8
            | FEATURE_AVX_VNNI_INT16
            | FEATURE_AVX10_2
        ),
    ),
    X86TargetProfileInfo(
        profile_key="x86.avx512_packed_dot",
        descriptor_generator_target="avx512_features",
        descriptor_set_key="x86.avx512_features.core",
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
        feature_contract_set_key="x86.avx512_features.core",
        supported_feature_bits=X86_KNOWN_FEATURE_BITS,
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
        register_classes=(
            X86_REG_CLASS_GPR64,
            X86_REG_CLASS_XMM,
            X86_REG_CLASS_YMM,
            X86_REG_CLASS_ZMM,
            X86_REG_CLASS_K,
        ),
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
    "x86.avx2",
)


def x86_deployment_profile_executable_target_profiles(
    deployment_profile: X86DeploymentProfileInfo,
    target_profiles: tuple[X86TargetProfileInfo, ...] = X86_TARGET_PROFILE_INFOS,
) -> tuple[X86TargetProfileInfo, ...]:
    target_profiles_by_key = {info.profile_key: info for info in target_profiles}
    available_cpu_features = frozenset(deployment_profile.cpu_features)
    return tuple(
        profile
        for profile_key in X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS
        if profile_key is not None
        for profile in (target_profiles_by_key[profile_key],)
        if profile.automatic_priority > 0
        and frozenset(profile.required_cpu_features) <= available_cpu_features
    )


def x86_deployment_profile_preferred_target_profile(
    deployment_profile: X86DeploymentProfileInfo,
    target_profiles: tuple[X86TargetProfileInfo, ...] = X86_TARGET_PROFILE_INFOS,
) -> X86TargetProfileInfo:
    executable_profiles = x86_deployment_profile_executable_target_profiles(
        deployment_profile, target_profiles
    )
    if not executable_profiles:
        raise ValueError(
            f"x86 deployment profile '{deployment_profile.spelling}' "
            "cannot execute any native target profile"
        )
    return max(executable_profiles, key=lambda info: info.automatic_priority)


def _validate_unique(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate x86 {description} '{value}'")
        seen.add(value)


def _validate_bstring_spelling(value: str, description: str) -> None:
    try:
        encoded_value = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"x86 {description} '{value}' must be ASCII") from exc
    if not encoded_value or len(encoded_value) > 255:
        raise ValueError(f"x86 {description} '{value}' must contain 1-255 ASCII bytes")


def validate_x86_target_info_tables(
    descriptor_sets: tuple[X86DescriptorSetInfo, ...] = X86_DESCRIPTOR_SET_INFOS,
    target_profiles: tuple[X86TargetProfileInfo, ...] = X86_TARGET_PROFILE_INFOS,
    target_tunes: tuple[X86TargetTuneInfo, ...] = X86_TARGET_TUNE_INFOS,
    deployment_profiles: tuple[X86DeploymentProfileInfo, ...] = (
        X86_DEPLOYMENT_PROFILE_INFOS
    ),
) -> None:
    _validate_unique(
        (info.generator_target for info in descriptor_sets),
        "descriptor generator target",
    )
    _validate_unique((info.key for info in descriptor_sets), "descriptor set key")
    _validate_unique((info.profile_key for info in target_profiles), "profile key")
    _validate_unique((info.spelling for info in target_tunes), "target tune")
    _validate_unique(
        (info.spelling for info in deployment_profiles), "deployment profile"
    )
    _validate_unique(
        (info.schedule_model_key for info in target_tunes),
        "schedule model key",
    )
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
        if selector_profile_info.feature_contract_set_key is None:
            raise ValueError(
                f"x86 native selector profile '{profile_key}' has no feature contract"
            )
        if selector_profile_info.feature_contract_set_key not in descriptor_sets_by_key:
            raise ValueError(
                f"x86 native selector profile '{profile_key}' references unknown "
                f"feature contract '{selector_profile_info.feature_contract_set_key}'"
            )
        if selector_profile_info.supported_feature_bits & ~X86_KNOWN_FEATURE_BITS:
            raise ValueError(
                f"x86 native selector profile '{profile_key}' supports unknown features"
            )
        if selector_profile_info.contract_feature_bits & ~(
            selector_profile_info.supported_feature_bits
        ):
            raise ValueError(
                f"x86 native selector profile '{profile_key}' defaults "
                "unsupported features"
            )

    execution_selectors = {
        profile_key.removeprefix("x86.")
        for profile_key in X86_NATIVE_TARGET_SELECTOR_PROFILE_KEYS
        if profile_key is not None
    }
    tune_spellings = {info.spelling for info in target_tunes}
    for deployment_profile in deployment_profiles:
        _validate_bstring_spelling(
            deployment_profile.spelling, "deployment profile spelling"
        )
        if deployment_profile.spelling in execution_selectors:
            raise ValueError(
                f"x86 deployment profile '{deployment_profile.spelling}' "
                "collides with an execution profile"
            )
        if deployment_profile.tune_spelling not in tune_spellings:
            raise ValueError(
                f"x86 deployment profile '{deployment_profile.spelling}' "
                f"references unknown tune '{deployment_profile.tune_spelling}'"
            )
        _validate_unique(
            deployment_profile.cpu_features,
            f"{deployment_profile.spelling} CPU feature",
        )
        if not x86_deployment_profile_executable_target_profiles(
            deployment_profile, target_profiles
        ):
            raise ValueError(
                f"x86 deployment profile '{deployment_profile.spelling}' "
                "cannot execute any native target profile"
            )

    stable_values = tuple(info.stable_value for info in target_tunes)
    if stable_values != tuple(range(1, len(target_tunes) + 1)):
        raise ValueError("x86 target tune values must be dense and start at one")
    processor_matches: dict[tuple[str, int, int], str] = {}
    for deployment_profile in deployment_profiles:
        for match in deployment_profile.processor_matches:
            if len(match.vendor.encode("ascii")) != 12:
                raise ValueError(
                    f"x86 processor vendor '{match.vendor}' must be 12 ASCII bytes"
                )
            if not (0 <= match.family <= 0x10E):
                raise ValueError(
                    f"x86 processor family 0x{match.family:x} is out of range"
                )
            if not (0 <= match.first_model <= match.last_model <= 0xFF):
                raise ValueError(
                    f"x86 processor model range 0x{match.first_model:x}-"
                    f"0x{match.last_model:x} is invalid"
                )
            for model in range(match.first_model, match.last_model + 1):
                key = (match.vendor, match.family, model)
                prior_profile = processor_matches.get(key)
                if prior_profile is not None:
                    raise ValueError(
                        f"x86 processor {match.vendor} family 0x{match.family:x} "
                        f"model 0x{model:x} matches both '{prior_profile}' and "
                        f"'{deployment_profile.spelling}'"
                    )
                processor_matches[key] = deployment_profile.spelling


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


def sorted_deployment_profile_infos() -> tuple[X86DeploymentProfileInfo, ...]:
    return tuple(sorted(X86_DEPLOYMENT_PROFILE_INFOS, key=lambda info: info.spelling))


def x86_target_profile_info_by_key(profile_key: str) -> X86TargetProfileInfo:
    for info in X86_TARGET_PROFILE_INFOS:
        if info.profile_key == profile_key:
            return info
    raise ValueError(f"unknown x86 target profile '{profile_key}'")


def x86_target_tune_info_by_spelling(spelling: str) -> X86TargetTuneInfo:
    for info in X86_TARGET_TUNE_INFOS:
        if info.spelling == spelling:
            return info
    raise ValueError(f"unknown x86 target tune '{spelling}'")


validate_x86_target_info_tables()
