# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Canonical AMDGPU exact and generic code-object compatibility facts.

LLVM generic code objects carry a version in the high byte of ELF ``e_flags``.
Each exact processor in a generic family records the first generic version that
can execute on it. A generic code object is compatible with that processor when
its emitted version is greater than or equal to the processor's introduction
version.

This module is deliberately independent of Loom and the runtime generators so
all consumers share the same membership and version facts without an import
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AmdgpuGenericCodeObjectInfo:
    processor: str
    current_version: int


@dataclass(frozen=True, slots=True)
class AmdgpuCodeObjectCompatibilityInfo:
    exact_processor: str
    code_object_processor: str
    generic_introduction_version: int


AMDGPU_GENERIC_CODE_OBJECT_INFOS = (
    AmdgpuGenericCodeObjectInfo("gfx9-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx9-4-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx10-1-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx10-3-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx11-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx12-generic", 1),
    AmdgpuGenericCodeObjectInfo("gfx12-5-generic", 1),
)


# Each exact processor must match an HSA ISA architecture suffix. Each
# code-object processor must be accepted by LLVM clang/lld as an AMDGPU -march
# value. Generic membership and introduction versions follow LLVM's generic
# processor definitions for the toolchain version supported by this tree.
AMDGPU_CODE_OBJECT_COMPATIBILITY_INFOS = (
    AmdgpuCodeObjectCompatibilityInfo("gfx900", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx902", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx904", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx90c", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx906", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx908", "gfx908", 0),
    AmdgpuCodeObjectCompatibilityInfo("gfx909", "gfx9-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx90a", "gfx90a", 0),
    AmdgpuCodeObjectCompatibilityInfo("gfx940", "gfx9-4-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx941", "gfx9-4-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx942", "gfx9-4-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx950", "gfx9-4-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1010", "gfx10-1-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1011", "gfx10-1-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1012", "gfx10-1-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1013", "gfx10-1-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1030", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1031", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1032", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1033", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1034", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1035", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1036", "gfx10-3-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1100", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1101", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1102", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1103", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1150", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1151", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1152", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1153", "gfx11-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1170", "gfx1170", 0),
    AmdgpuCodeObjectCompatibilityInfo("gfx1171", "gfx1171", 0),
    AmdgpuCodeObjectCompatibilityInfo("gfx1172", "gfx1172", 0),
    AmdgpuCodeObjectCompatibilityInfo("gfx1200", "gfx12-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1201", "gfx12-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1250", "gfx12-5-generic", 1),
    AmdgpuCodeObjectCompatibilityInfo("gfx1251", "gfx12-5-generic", 1),
)


def generic_code_object_current_version(processor: str) -> int:
    for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        if info.processor == processor:
            return info.current_version
    raise ValueError(f"unknown AMDGPU generic code-object processor: {processor}")


def validate_code_object_compatibility(
    generic_infos: Sequence[AmdgpuGenericCodeObjectInfo] = (
        AMDGPU_GENERIC_CODE_OBJECT_INFOS
    ),
    compatibility_infos: Sequence[AmdgpuCodeObjectCompatibilityInfo] = (
        AMDGPU_CODE_OBJECT_COMPATIBILITY_INFOS
    ),
) -> None:
    generic_versions: dict[str, int] = {}
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

    exact_processors: set[str] = set()
    referenced_generic_processors: set[str] = set()
    for info in compatibility_infos:
        if info.exact_processor.endswith("-generic"):
            raise ValueError(
                f"exact AMDGPU processor is generic: {info.exact_processor}"
            )
        if info.exact_processor in exact_processors:
            raise ValueError(
                f"duplicate exact AMDGPU processor: {info.exact_processor}"
            )
        exact_processors.add(info.exact_processor)

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


validate_code_object_compatibility()
