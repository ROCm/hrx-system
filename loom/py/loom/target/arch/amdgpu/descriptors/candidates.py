# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Derived AMDGPU source-to-low descriptor candidate tables."""

from __future__ import annotations

from .common import *
from .sets import *

_ATOMIC_MEMORY_SPACE = {
    "workgroup": AmdgpuAtomicMemorySpace.WORKGROUP,
    "global": AmdgpuAtomicMemorySpace.GLOBAL,
    "generic": AmdgpuAtomicMemorySpace.GENERIC,
}

_ATOMIC_KIND = {
    "add.u32": (AmdgpuAtomicKind.ADDI, AmdgpuAtomicValueKind.I32),
    "sub.u32": (AmdgpuAtomicKind.SUBI, AmdgpuAtomicValueKind.I32),
    "min.i32": (AmdgpuAtomicKind.MINSI, AmdgpuAtomicValueKind.I32),
    "max.i32": (AmdgpuAtomicKind.MAXSI, AmdgpuAtomicValueKind.I32),
    "min.u32": (AmdgpuAtomicKind.MINUI, AmdgpuAtomicValueKind.I32),
    "max.u32": (AmdgpuAtomicKind.MAXUI, AmdgpuAtomicValueKind.I32),
    "and.b32": (AmdgpuAtomicKind.ANDI, AmdgpuAtomicValueKind.I32),
    "or.b32": (AmdgpuAtomicKind.ORI, AmdgpuAtomicValueKind.I32),
    "xor.b32": (AmdgpuAtomicKind.XORI, AmdgpuAtomicValueKind.I32),
    "exchange.b32": (AmdgpuAtomicKind.XCHGI, AmdgpuAtomicValueKind.I32),
    "add.f32": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.F32),
    "minnum.f32": (AmdgpuAtomicKind.MINNUMF, AmdgpuAtomicValueKind.F32),
    "maxnum.f32": (AmdgpuAtomicKind.MAXNUMF, AmdgpuAtomicValueKind.F32),
}

_ATOMIC_ADDRESS_FORM = (
    ("amdgpu.ds_", AmdgpuMemoryAddressForm.DEFAULT),
    ("amdgpu.buffer_atomic_", AmdgpuMemoryAddressForm.DEFAULT),
    ("amdgpu.global_atomic_", AmdgpuMemoryAddressForm.GLOBAL_SADDR),
    ("amdgpu.flat_atomic_", AmdgpuMemoryAddressForm.FLAT),
)

_ATOMIC_CANDIDATE_MEMORY_ORDER = {
    (
        AmdgpuAtomicMemorySpace.WORKGROUP,
        AmdgpuMemoryAddressForm.DEFAULT,
    ): 0,
    (
        AmdgpuAtomicMemorySpace.GLOBAL,
        AmdgpuMemoryAddressForm.DEFAULT,
    ): 1,
    (
        AmdgpuAtomicMemorySpace.GLOBAL,
        AmdgpuMemoryAddressForm.GLOBAL_SADDR,
    ): 2,
    (
        AmdgpuAtomicMemorySpace.GENERIC,
        AmdgpuMemoryAddressForm.FLAT,
    ): 3,
}

_ATOMIC_CANDIDATE_OPERATION_ORDER = {
    AmdgpuAtomicOperationKind.REDUCE: 0,
    AmdgpuAtomicOperationKind.RMW: 1,
    AmdgpuAtomicOperationKind.CMPXCHG: 2,
}

_ATOMIC_CANDIDATE_KIND_ORDER = {
    AmdgpuAtomicKind.ADDI: 0,
    AmdgpuAtomicKind.SUBI: 1,
    AmdgpuAtomicKind.MINSI: 2,
    AmdgpuAtomicKind.MAXSI: 3,
    AmdgpuAtomicKind.MINUI: 4,
    AmdgpuAtomicKind.MAXUI: 5,
    AmdgpuAtomicKind.ANDI: 6,
    AmdgpuAtomicKind.ORI: 7,
    AmdgpuAtomicKind.XORI: 8,
    AmdgpuAtomicKind.XCHGI: 9,
    AmdgpuAtomicKind.ADDF: 10,
    AmdgpuAtomicKind.MINNUMF: 11,
    AmdgpuAtomicKind.MAXNUMF: 12,
    AmdgpuAtomicKind.NONE: 13,
}


def _atomic_address_form(descriptor_key: str) -> AmdgpuMemoryAddressForm | None:
    for prefix, address_form in _ATOMIC_ADDRESS_FORM:
        if descriptor_key.startswith(prefix):
            return address_form
    return None


def _amdgpu_atomic_candidate_from_overlay(
    overlay: AmdgpuDescriptorOverlay,
) -> AmdgpuAtomicDescriptorCandidate | None:
    if ".atomic." not in overlay.semantic_tag:
        return None
    address_form = _atomic_address_form(overlay.descriptor_key)
    if address_form is None:
        return None
    if overlay.descriptor_key.startswith("amdgpu.global_atomic_") and not (
        overlay.descriptor_key.endswith("_saddr")
    ):
        return None

    tag_parts = overlay.semantic_tag.split(".")
    if len(tag_parts) < 4 or tag_parts[0] != "memory" or tag_parts[2] != "atomic":
        return None
    memory_space = _ATOMIC_MEMORY_SPACE.get(tag_parts[1])
    if memory_space is None:
        return None

    semantic_parts = tuple(part for part in tag_parts[3:] if part != "return")
    returns_old_value = "return" in tag_parts[3:]
    if semantic_parts == ("compare_exchange", "b32"):
        return AmdgpuAtomicDescriptorCandidate(
            memory_space=memory_space,
            address_form=address_form,
            operation_kind=AmdgpuAtomicOperationKind.CMPXCHG,
            atomic_kind=AmdgpuAtomicKind.NONE,
            value_kind=AmdgpuAtomicValueKind.I32,
            descriptor_key=overlay.descriptor_key,
        )

    atomic_kind = ".".join(semantic_parts)
    try:
        atomic_kind_enum, value_kind = _ATOMIC_KIND[atomic_kind]
    except KeyError:
        return None
    return AmdgpuAtomicDescriptorCandidate(
        memory_space=memory_space,
        address_form=address_form,
        operation_kind=(
            AmdgpuAtomicOperationKind.RMW
            if returns_old_value
            else AmdgpuAtomicOperationKind.REDUCE
        ),
        atomic_kind=atomic_kind_enum,
        value_kind=value_kind,
        descriptor_key=overlay.descriptor_key,
    )


def _amdgpu_atomic_candidate_sort_key(
    candidate: AmdgpuAtomicDescriptorCandidate,
) -> tuple[int, int, int, str]:
    return (
        _ATOMIC_CANDIDATE_MEMORY_ORDER[
            (candidate.memory_space, candidate.address_form)
        ],
        _ATOMIC_CANDIDATE_OPERATION_ORDER[candidate.operation_kind],
        _ATOMIC_CANDIDATE_KIND_ORDER[candidate.atomic_kind],
        candidate.descriptor_key,
    )


def amdgpu_atomic_descriptor_candidates() -> tuple[
    AmdgpuAtomicDescriptorCandidate, ...
]:
    """Returns source-to-low atomic descriptor candidates from AMDGPU metadata."""

    candidates_by_key: dict[str, AmdgpuAtomicDescriptorCandidate] = {}
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        for overlay in overlays:
            candidate = _amdgpu_atomic_candidate_from_overlay(overlay)
            if candidate is None:
                continue
            candidates_by_key.setdefault(candidate.descriptor_key, candidate)
    return tuple(
        sorted(candidates_by_key.values(), key=_amdgpu_atomic_candidate_sort_key)
    )


__all__ = (
    "_ATOMIC_ADDRESS_FORM",
    "_ATOMIC_CANDIDATE_KIND_ORDER",
    "_ATOMIC_CANDIDATE_MEMORY_ORDER",
    "_ATOMIC_CANDIDATE_OPERATION_ORDER",
    "_ATOMIC_KIND",
    "_ATOMIC_MEMORY_SPACE",
    "_amdgpu_atomic_candidate_from_overlay",
    "_amdgpu_atomic_candidate_sort_key",
    "_atomic_address_form",
    "amdgpu_atomic_descriptor_candidates",
)
