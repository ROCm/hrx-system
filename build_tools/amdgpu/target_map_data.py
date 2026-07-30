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
version. Target-ID feature support, target overlays, and physical
target resolution are shared by build tools, HAL, and compilers.

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
class AmdgpuExactTargetInfo:
    """Identity, compatibility, and qualification facts for an exact target."""

    exact_processor: str
    code_object_processor: str
    generic_introduction_version: int
    target_id_features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuTargetOverlayInfo:
    """One canonical target that overlays a backend processor."""

    target: str
    processor: str
    compile_options: tuple[str, ...] = ()
    link_options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuPhysicalTargetInfo:
    """Maps one physical processor revision to its canonical target."""

    processor: str
    asic_revision: int
    target: str


@dataclass(frozen=True, slots=True)
class AmdgpuDeviceBinaryTarget:
    """One derived builtin device-library artifact target."""

    name: str
    processor: str
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
    AmdgpuExactTargetInfo("gfx1250", "gfx12-5-generic", 1),
    AmdgpuExactTargetInfo("gfx1251", "gfx12-5-generic", 1),
)


# Target overlays bind a canonical target identity to one exact backend
# processor and any final backend-invocation options it requires. Compiler
# semantics remain structured facts on the canonical target row.
AMDGPU_TARGET_OVERLAY_INFOS = (
    AmdgpuTargetOverlayInfo(
        target="gfx1250-a0",
        processor="gfx1250",
        compile_options=("-mllvm", "-amdgpu-gfx1250-b0-specific=false"),
        link_options=("-plugin-opt=-amdgpu-gfx1250-b0-specific=false",),
    ),
)

# Physical discovery resolves directly to a canonical target. Processors absent
# from this table ignore their reported ASIC revision because the revision does
# not participate in their target identity.
AMDGPU_PHYSICAL_TARGET_INFOS = (
    AmdgpuPhysicalTargetInfo(
        processor="gfx1250",
        asic_revision=0,
        target="gfx1250-a0",
    ),
    AmdgpuPhysicalTargetInfo(
        processor="gfx1250",
        asic_revision=1,
        target="gfx1250",
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


def target_overlay_info(target: str) -> AmdgpuTargetOverlayInfo | None:
    for info in AMDGPU_TARGET_OVERLAY_INFOS:
        if info.target == target:
            return info
    return None


def target_processor(target: str) -> str | None:
    """Returns the backend processor selected by a canonical target."""
    overlay = target_overlay_info(target)
    if overlay is not None:
        return overlay.processor
    if exact_target_info(target) is not None or generic_code_object_info(target):
        return target
    return None


def physical_target_info(
    processor: str, asic_revision: int
) -> AmdgpuPhysicalTargetInfo | None:
    for info in AMDGPU_PHYSICAL_TARGET_INFOS:
        if info.processor == processor and info.asic_revision == asic_revision:
            return info
    return None


def processor_has_physical_target_infos(processor: str) -> bool:
    return any(info.processor == processor for info in AMDGPU_PHYSICAL_TARGET_INFOS)


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
    target_overlays: Sequence[AmdgpuTargetOverlayInfo] = AMDGPU_TARGET_OVERLAY_INFOS,
    physical_targets: Sequence[AmdgpuPhysicalTargetInfo] = (
        AMDGPU_PHYSICAL_TARGET_INFOS
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

    unreferenced_generic_processors = sorted(
        set(generic_versions) - referenced_generic_processors
    )
    if unreferenced_generic_processors:
        raise ValueError(
            "AMDGPU generic code-object processors have no exact members: "
            + ", ".join(unreferenced_generic_processors)
        )

    base_targets = exact_processors | generic_processors
    overlay_targets: set[str] = set()
    for overlay in target_overlays:
        if overlay.target in base_targets or overlay.target in overlay_targets:
            raise ValueError(f"duplicate AMDGPU target overlay: {overlay.target}")
        if (
            not overlay.target
            or not overlay.target[0].isalnum()
            or any(
                not (
                    "a" <= character <= "z"
                    or "0" <= character <= "9"
                    or character in "-._"
                )
                for character in overlay.target
            )
        ):
            raise ValueError(
                f"AMDGPU target overlay {overlay.target!r} is not a canonical "
                "target coordinate"
            )
        overlay_targets.add(overlay.target)

    canonical_targets = base_targets | overlay_targets

    for overlay in target_overlays:
        if overlay.processor not in exact_processors:
            raise ValueError(
                f"AMDGPU target overlay {overlay.target} references unknown "
                f"exact processor {overlay.processor}"
            )

    target_processors = {
        **{target: target for target in base_targets},
        **{overlay.target: overlay.processor for overlay in target_overlays},
    }
    previous_physical_key: tuple[str, int] | None = None
    physical_keys: set[tuple[str, int]] = set()
    physical_processors: set[str] = set()
    physical_base_targets: set[str] = set()
    for physical_target in physical_targets:
        key = (physical_target.processor, physical_target.asic_revision)
        if physical_target.processor not in exact_processors:
            raise ValueError(
                "AMDGPU physical target references unknown exact processor "
                f"{physical_target.processor}"
            )
        if (
            physical_target.asic_revision < 0
            or physical_target.asic_revision > (2**32) - 1
        ):
            raise ValueError(
                f"AMDGPU processor {physical_target.processor} ASIC revision "
                f"{physical_target.asic_revision} is outside the uint32 range"
            )
        if previous_physical_key is not None and key < previous_physical_key:
            raise ValueError(
                "AMDGPU physical targets are not in canonical processor and "
                "ASIC revision order"
            )
        if key in physical_keys:
            raise ValueError(
                f"AMDGPU processor {physical_target.processor} repeats ASIC "
                f"revision {physical_target.asic_revision}"
            )
        if physical_target.target not in canonical_targets:
            raise ValueError(
                f"AMDGPU physical target {physical_target.processor} ASIC "
                f"revision {physical_target.asic_revision} selects unknown "
                f"target {physical_target.target}"
            )
        if target_processors[physical_target.target] != physical_target.processor:
            raise ValueError(
                f"AMDGPU physical target {physical_target.processor} ASIC "
                f"revision {physical_target.asic_revision} selects target "
                f"{physical_target.target} from another processor"
            )
        physical_keys.add(key)
        physical_processors.add(physical_target.processor)
        if physical_target.target == physical_target.processor:
            physical_base_targets.add(physical_target.processor)
        previous_physical_key = key

    missing_physical_base_targets = sorted(physical_processors - physical_base_targets)
    if missing_physical_base_targets:
        raise ValueError(
            "AMDGPU processors with physical target mappings have no revision "
            "selecting their canonical base target: "
            + ", ".join(missing_physical_base_targets)
        )


validate_target_map_data()
