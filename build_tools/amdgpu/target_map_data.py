# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Canonical AMDGPU target identity and code-object compatibility facts.

LLVM generic code objects carry a version in the high byte of ELF ``e_flags``.
Each exact processor in a generic family records the first generic version that
can execute on it. A generic code object is compatible with that processor when
its emitted version is greater than or equal to the processor's introduction
version. Target-ID feature support and ASIC revisions are distinct physical and
artifact identity facts shared by build tools, HAL, and compilers.

This module is deliberately independent of Loom and the runtime generators so
all consumers share the same identity facts without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AmdgpuGenericCodeObjectInfo:
    """Identity and qualification facts for one generic code-object target."""

    processor: str
    current_version: int
    target_id_features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuAsicRevisionInfo:
    """One finite physical ASIC revision supported by an exact processor."""

    value: int
    name: str


@dataclass(frozen=True, slots=True)
class AmdgpuExactTargetInfo:
    """Identity, compatibility, and qualification facts for an exact target."""

    exact_processor: str
    code_object_processor: str
    generic_introduction_version: int
    target_id_features: tuple[str, ...] = ()
    asic_revisions: tuple[AmdgpuAsicRevisionInfo, ...] = ()
    default_asic_revision: int | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuDeviceBinaryTargetMatch:
    """One exact physical target selecting a device-library artifact."""

    processor: str
    asic_revision: int


@dataclass(frozen=True, slots=True)
class AmdgpuDeviceBinaryTarget:
    """One independently named builtin device-library artifact."""

    name: str
    architecture: str
    target_matches: tuple[AmdgpuDeviceBinaryTargetMatch, ...] = ()
    compile_options: tuple[str, ...] = ()
    link_options: tuple[str, ...] = ()


TARGET_ID_FEATURE_SRAMECC = "sramecc"
TARGET_ID_FEATURE_XNACK = "xnack"
AMDGPU_TARGET_ID_FEATURE_ORDER = (
    TARGET_ID_FEATURE_SRAMECC,
    TARGET_ID_FEATURE_XNACK,
)


AMDGPU_GENERIC_CODE_OBJECT_INFOS = (
    AmdgpuGenericCodeObjectInfo(
        "gfx9-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuGenericCodeObjectInfo(
        "gfx9-4-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuGenericCodeObjectInfo(
        "gfx10-1-generic",
        1,
        (TARGET_ID_FEATURE_XNACK,),
    ),
    AmdgpuGenericCodeObjectInfo("gfx10-3-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx11-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx12-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx12-5-generic", 1),
)


# Each exact processor must match an HSA ISA architecture suffix. Each
# code-object processor must be accepted by LLVM clang/lld as an AMDGPU -march
# value. Generic membership and introduction versions follow LLVM's generic
# processor definitions for the toolchain version supported by this tree.
AMDGPU_EXACT_TARGET_INFOS = (
    AmdgpuExactTargetInfo("gfx900", "gfx9-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx902", "gfx9-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx904", "gfx9-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx90c", "gfx9-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo(
        "gfx906",
        "gfx9-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo(
        "gfx908",
        "gfx908",
        0,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo("gfx909", "gfx9-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo(
        "gfx90a",
        "gfx90a",
        0,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo(
        "gfx940",
        "gfx9-4-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo(
        "gfx941",
        "gfx9-4-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo(
        "gfx942",
        "gfx9-4-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo(
        "gfx950",
        "gfx9-4-generic",
        1,
        (TARGET_ID_FEATURE_SRAMECC, TARGET_ID_FEATURE_XNACK),
    ),
    AmdgpuExactTargetInfo("gfx1010", "gfx10-1-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx1011", "gfx10-1-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx1012", "gfx10-1-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx1013", "gfx10-1-generic", 1, (TARGET_ID_FEATURE_XNACK,)),
    AmdgpuExactTargetInfo("gfx1030", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1031", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1032", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1033", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1034", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1035", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1036", "gfx10-3-generic", 1),
    AmdgpuExactTargetInfo("gfx1100", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1101", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1102", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1103", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1150", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1151", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1152", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1153", "gfx11-generic", 1),
    AmdgpuExactTargetInfo("gfx1170", "gfx1170", 0),
    AmdgpuExactTargetInfo("gfx1171", "gfx1171", 0),
    AmdgpuExactTargetInfo("gfx1172", "gfx1172", 0),
    AmdgpuExactTargetInfo("gfx1200", "gfx12-generic", 1),
    AmdgpuExactTargetInfo("gfx1201", "gfx12-generic", 1),
    AmdgpuExactTargetInfo(
        "gfx1250",
        "gfx12-5-generic",
        1,
        asic_revisions=(
            AmdgpuAsicRevisionInfo(value=0, name="a0"),
            AmdgpuAsicRevisionInfo(value=1, name="b0"),
        ),
        default_asic_revision=1,
    ),
    AmdgpuExactTargetInfo("gfx1251", "gfx12-5-generic", 1),
)


# Device-binary variants are artifact identities, not public processor names.
# Each row binds one artifact to the exact physical targets that cannot safely
# consume their normal code-object-family artifact.
AMDGPU_DEVICE_BINARY_VARIANTS = (
    AmdgpuDeviceBinaryTarget(
        name="gfx1250-a0",
        architecture="gfx1250",
        target_matches=(
            AmdgpuDeviceBinaryTargetMatch(processor="gfx1250", asic_revision=0),
        ),
        compile_options=("-mllvm", "-amdgpu-gfx1250-b0-specific=false"),
        link_options=("-plugin-opt=-amdgpu-gfx1250-b0-specific=false",),
    ),
)


def generic_code_object_current_version(processor: str) -> int:
    for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        if info.processor == processor:
            return info.current_version
    raise ValueError(f"unknown AMDGPU generic code-object processor: {processor}")


def exact_target_info(processor: str) -> AmdgpuExactTargetInfo | None:
    for info in AMDGPU_EXACT_TARGET_INFOS:
        if info.exact_processor == processor:
            return info
    return None


def generic_code_object_info(
    processor: str,
) -> AmdgpuGenericCodeObjectInfo | None:
    for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        if info.processor == processor:
            return info
    return None


def target_id_features_for_processor(processor: str) -> tuple[str, ...]:
    exact_info = exact_target_info(processor)
    if exact_info is not None:
        return exact_info.target_id_features
    generic_info = generic_code_object_info(processor)
    if generic_info is not None:
        return generic_info.target_id_features
    return ()


def _validate_target_id_features(
    processor: str, target_id_features: tuple[str, ...]
) -> None:
    if len(set(target_id_features)) != len(target_id_features):
        raise ValueError(f"AMDGPU processor {processor} repeats a target-ID feature")
    unknown_features = sorted(
        set(target_id_features) - set(AMDGPU_TARGET_ID_FEATURE_ORDER)
    )
    if unknown_features:
        raise ValueError(
            f"AMDGPU processor {processor} references unknown target-ID "
            f"features: {', '.join(unknown_features)}"
        )
    canonical_features = tuple(
        feature
        for feature in AMDGPU_TARGET_ID_FEATURE_ORDER
        if feature in target_id_features
    )
    if target_id_features != canonical_features:
        raise ValueError(
            f"AMDGPU processor {processor} target-ID features are not in "
            f"canonical order: {', '.join(canonical_features)}"
        )


def validate_target_map_data(
    generic_infos: Sequence[AmdgpuGenericCodeObjectInfo] = (
        AMDGPU_GENERIC_CODE_OBJECT_INFOS
    ),
    exact_infos: Sequence[AmdgpuExactTargetInfo] = AMDGPU_EXACT_TARGET_INFOS,
    device_binary_variants: Sequence[AmdgpuDeviceBinaryTarget] = (
        AMDGPU_DEVICE_BINARY_VARIANTS
    ),
) -> None:
    generic_versions: dict[str, int] = {}
    generic_processors: set[str] = set()
    for info in generic_infos:
        if not info.processor.endswith("-generic"):
            raise ValueError(
                f"generic code-object processor is not generic: {info.processor}"
            )
        if info.processor in generic_versions:
            raise ValueError(
                f"duplicate AMDGPU generic code-object processor: {info.processor}"
            )
        if info.current_version < 1 or info.current_version > 255:
            raise ValueError(
                f"AMDGPU generic code-object processor {info.processor} has "
                f"invalid current version {info.current_version}"
            )
        generic_versions[info.processor] = info.current_version
        generic_processors.add(info.processor)
        _validate_target_id_features(info.processor, info.target_id_features)

    exact_processors: set[str] = set()
    referenced_generic_processors: set[str] = set()
    for info in exact_infos:
        if info.exact_processor.endswith("-generic"):
            raise ValueError(
                f"exact AMDGPU processor is generic: {info.exact_processor}"
            )
        if info.exact_processor in exact_processors:
            raise ValueError(
                f"duplicate exact AMDGPU processor: {info.exact_processor}"
            )
        exact_processors.add(info.exact_processor)
        _validate_target_id_features(info.exact_processor, info.target_id_features)

        if info.code_object_processor.endswith("-generic"):
            current_version = generic_versions.get(info.code_object_processor)
            if current_version is None:
                raise ValueError(
                    f"exact AMDGPU processor {info.exact_processor} references "
                    "unknown generic code-object processor "
                    f"{info.code_object_processor}"
                )
            if (
                info.generic_introduction_version < 1
                or info.generic_introduction_version > current_version
            ):
                raise ValueError(
                    f"exact AMDGPU processor {info.exact_processor} has generic "
                    f"introduction version {info.generic_introduction_version} "
                    f"outside {info.code_object_processor}'s supported range "
                    f"1..{current_version}"
                )
            referenced_generic_processors.add(info.code_object_processor)
        elif (
            info.code_object_processor != info.exact_processor
            or info.generic_introduction_version != 0
        ):
            raise ValueError(
                f"exact AMDGPU processor {info.exact_processor} must map to "
                "itself with generic introduction version 0 or map to a "
                "declared generic processor"
            )

        revision_values: set[int] = set()
        revision_names: set[str] = set()
        previous_revision_value: int | None = None
        for revision in info.asic_revisions:
            if revision.value < 0 or revision.value > (2**32) - 1:
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} ASIC revision "
                    f"{revision.value} is outside the uint32 range"
                )
            if revision.value in revision_values:
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} repeats ASIC "
                    f"revision {revision.value}"
                )
            if (
                previous_revision_value is not None
                and revision.value < previous_revision_value
            ):
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} ASIC revisions "
                    "are not in ascending value order"
                )
            if not revision.name:
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} ASIC revision "
                    f"{revision.value} has no canonical name"
                )
            if not revision.name[0].isalnum() or any(
                not (
                    "a" <= character <= "z"
                    or "0" <= character <= "9"
                    or character in "-._"
                )
                for character in revision.name
            ):
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} ASIC revision "
                    f"name {revision.name!r} is not a canonical artifact "
                    "coordinate"
                )
            if revision.name in revision_names:
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} repeats ASIC "
                    f"revision name {revision.name}"
                )
            revision_values.add(revision.value)
            revision_names.add(revision.name)
            previous_revision_value = revision.value
        if info.asic_revisions:
            if info.default_asic_revision not in revision_values:
                raise ValueError(
                    f"AMDGPU processor {info.exact_processor} default ASIC "
                    f"revision {info.default_asic_revision} is not declared"
                )
        elif info.default_asic_revision is not None:
            raise ValueError(
                f"AMDGPU processor {info.exact_processor} has a default ASIC "
                "revision but declares no revisions"
            )

    unreferenced_generic_processors = sorted(
        set(generic_versions) - referenced_generic_processors
    )
    if unreferenced_generic_processors:
        raise ValueError(
            "AMDGPU generic code-object processors have no exact members: "
            + ", ".join(unreferenced_generic_processors)
        )

    variant_names: set[str] = set()
    claimed_targets: set[AmdgpuDeviceBinaryTargetMatch] = set()
    for variant in device_binary_variants:
        if variant.name in variant_names:
            raise ValueError(f"duplicate AMDGPU device binary variant: {variant.name}")
        variant_names.add(variant.name)
        if variant.architecture not in exact_processors | generic_processors:
            raise ValueError(
                f"device binary variant {variant.name} references unknown "
                f"architecture {variant.architecture}"
            )
        if not variant.target_matches:
            raise ValueError(
                f"device binary variant {variant.name} has no physical target matches"
            )
        for target_match in variant.target_matches:
            if target_match.processor not in exact_processors:
                raise ValueError(
                    f"device binary variant {variant.name} references unknown "
                    f"exact target {target_match.processor}"
                )
            exact_info = next(
                info
                for info in exact_infos
                if info.exact_processor == target_match.processor
            )
            supported_revisions = {
                revision.value for revision in exact_info.asic_revisions
            }
            if target_match.asic_revision not in supported_revisions:
                raise ValueError(
                    f"device binary variant {variant.name} selects unsupported "
                    f"ASIC revision {target_match.asic_revision} for "
                    f"{target_match.processor}"
                )
            if target_match in claimed_targets:
                raise ValueError(
                    f"AMDGPU physical target {target_match.processor} ASIC "
                    f"revision {target_match.asic_revision} is claimed by more "
                    "than one device binary variant"
                )
            claimed_targets.add(target_match)


validate_target_map_data()
