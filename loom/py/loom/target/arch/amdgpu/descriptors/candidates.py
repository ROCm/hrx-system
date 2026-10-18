# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Derived AMDGPU source-to-low descriptor candidate tables."""

from __future__ import annotations

from .api import amdgpu_descriptor_ref_keys
from .common import *
from .memory_candidates import _MEMORY_DESCRIPTOR_CANDIDATES
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
    "exchange.b32": (AmdgpuAtomicKind.XCHGI, AmdgpuAtomicValueKind.B32),
    "add.u64": (AmdgpuAtomicKind.ADDI, AmdgpuAtomicValueKind.I64),
    "sub.u64": (AmdgpuAtomicKind.SUBI, AmdgpuAtomicValueKind.I64),
    "min.i64": (AmdgpuAtomicKind.MINSI, AmdgpuAtomicValueKind.I64),
    "max.i64": (AmdgpuAtomicKind.MAXSI, AmdgpuAtomicValueKind.I64),
    "min.u64": (AmdgpuAtomicKind.MINUI, AmdgpuAtomicValueKind.I64),
    "max.u64": (AmdgpuAtomicKind.MAXUI, AmdgpuAtomicValueKind.I64),
    "and.b64": (AmdgpuAtomicKind.ANDI, AmdgpuAtomicValueKind.I64),
    "or.b64": (AmdgpuAtomicKind.ORI, AmdgpuAtomicValueKind.I64),
    "xor.b64": (AmdgpuAtomicKind.XORI, AmdgpuAtomicValueKind.I64),
    "exchange.b64": (AmdgpuAtomicKind.XCHGI, AmdgpuAtomicValueKind.B64),
    "add.f32": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.F32),
    "minnum.f32": (AmdgpuAtomicKind.MINNUMF, AmdgpuAtomicValueKind.F32),
    "maxnum.f32": (AmdgpuAtomicKind.MAXNUMF, AmdgpuAtomicValueKind.F32),
    "add.f64": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.F64),
    "minnum.f64": (AmdgpuAtomicKind.MINNUMF, AmdgpuAtomicValueKind.F64),
    "maxnum.f64": (AmdgpuAtomicKind.MAXNUMF, AmdgpuAtomicValueKind.F64),
    "add.pk2.f16": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.PACKED_F16),
    "add.pk2.bf16": (
        AmdgpuAtomicKind.ADDF,
        AmdgpuAtomicValueKind.PACKED_BF16,
    ),
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


def _global_to_workgroup_widths(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[int, int] | None:
    if not overlay.semantic_tag.startswith("memory.global_to_workgroup."):
        return None
    global_width_bits: int | None = None
    workgroup_width_bits: int | None = None
    for effect in overlay.effects:
        if effect.kind is EffectKind.READ and effect.memory_space is MemorySpace.GLOBAL:
            global_width_bits = effect.width_bits
        elif (
            effect.kind is EffectKind.WRITE
            and effect.memory_space is MemorySpace.WORKGROUP
        ):
            workgroup_width_bits = effect.width_bits
    if global_width_bits is None or workgroup_width_bits is None:
        return None
    return (global_width_bits, workgroup_width_bits)


def _amdgpu_async_gather_candidate_from_overlay(
    overlay: AmdgpuDescriptorOverlay,
) -> AmdgpuAsyncGatherDescriptorCandidate | None:
    if not overlay.descriptor_key.startswith("amdgpu.global_load_lds_"):
        return None
    if not overlay.descriptor_key.endswith("_saddr"):
        return None
    widths = _global_to_workgroup_widths(overlay)
    if widths is None:
        return None
    global_width_bits, workgroup_width_bits = widths
    if global_width_bits != workgroup_width_bits:
        return None
    if global_width_bits % 8 != 0:
        raise ValueError(
            "AMDGPU async gather candidate "
            f"'{overlay.descriptor_key}' has non-byte global width "
            f"{global_width_bits}"
        )
    return AmdgpuAsyncGatherDescriptorCandidate(
        packet_byte_count=global_width_bits // 8,
        dest_lane_byte_stride=4 if global_width_bits <= 32 else 16,
        descriptor_key=overlay.descriptor_key,
    )


def _record_amdgpu_async_gather_candidate(
    candidates_by_packet_byte_count: dict[int, AmdgpuAsyncGatherDescriptorCandidate],
    candidate: AmdgpuAsyncGatherDescriptorCandidate,
) -> None:
    existing = candidates_by_packet_byte_count.get(candidate.packet_byte_count)
    if existing is None:
        candidates_by_packet_byte_count[candidate.packet_byte_count] = candidate
        return
    if existing != candidate:
        raise ValueError(
            "AMDGPU async gather descriptor candidate for packet byte count "
            f"{candidate.packet_byte_count} has conflicting metadata: "
            f"{existing.descriptor_key}, {candidate.descriptor_key}"
        )


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
            value_kind=AmdgpuAtomicValueKind.B32,
            descriptor_key=overlay.descriptor_key,
        )
    if semantic_parts == ("compare_exchange", "b64"):
        return AmdgpuAtomicDescriptorCandidate(
            memory_space=memory_space,
            address_form=address_form,
            operation_kind=AmdgpuAtomicOperationKind.CMPXCHG,
            atomic_kind=AmdgpuAtomicKind.NONE,
            value_kind=AmdgpuAtomicValueKind.B64,
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


def _record_amdgpu_atomic_candidate(
    candidates_by_key: dict[str, AmdgpuAtomicDescriptorCandidate],
    candidate: AmdgpuAtomicDescriptorCandidate,
) -> None:
    existing = candidates_by_key.get(candidate.descriptor_key)
    if existing is None:
        candidates_by_key[candidate.descriptor_key] = candidate
        return
    if existing != candidate:
        raise ValueError(
            "AMDGPU atomic descriptor candidate "
            f"'{candidate.descriptor_key}' has conflicting metadata"
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
        _gfx125x_core_overlays(),
    ):
        for overlay in overlays:
            candidate = _amdgpu_atomic_candidate_from_overlay(overlay)
            if candidate is None:
                continue
            _record_amdgpu_atomic_candidate(candidates_by_key, candidate)
    descriptor_ref_keys = set(amdgpu_descriptor_ref_keys())
    missing_descriptor_refs = sorted(
        descriptor_key
        for descriptor_key in candidates_by_key
        if descriptor_key not in descriptor_ref_keys
    )
    if missing_descriptor_refs:
        raise ValueError(
            "AMDGPU atomic descriptor candidates require missing descriptor "
            f"refs: {', '.join(missing_descriptor_refs)}"
        )
    return tuple(
        sorted(candidates_by_key.values(), key=_amdgpu_atomic_candidate_sort_key)
    )


def amdgpu_async_gather_descriptor_candidates() -> tuple[
    AmdgpuAsyncGatherDescriptorCandidate, ...
]:
    """Returns source-to-low async gather descriptor candidates from metadata."""

    candidates_by_packet_byte_count: dict[
        int, AmdgpuAsyncGatherDescriptorCandidate
    ] = {}
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        for overlay in overlays:
            candidate = _amdgpu_async_gather_candidate_from_overlay(overlay)
            if candidate is None:
                continue
            _record_amdgpu_async_gather_candidate(
                candidates_by_packet_byte_count, candidate
            )
    descriptor_ref_keys = set(amdgpu_descriptor_ref_keys())
    missing_descriptor_refs = sorted(
        candidate.descriptor_key
        for candidate in candidates_by_packet_byte_count.values()
        if candidate.descriptor_key not in descriptor_ref_keys
    )
    if missing_descriptor_refs:
        raise ValueError(
            "AMDGPU async gather descriptor candidates require missing "
            f"descriptor refs: {', '.join(missing_descriptor_refs)}"
        )
    return tuple(
        sorted(
            candidates_by_packet_byte_count.values(),
            key=lambda candidate: candidate.packet_byte_count,
        )
    )


def amdgpu_memory_descriptor_candidates() -> tuple[
    AmdgpuMemoryDescriptorCandidate, ...
]:
    """Returns source-to-low memory descriptor candidates from AMDGPU metadata."""

    descriptor_ref_keys = set(amdgpu_descriptor_ref_keys())
    missing_descriptor_refs = sorted(
        key
        for candidate in _MEMORY_DESCRIPTOR_CANDIDATES
        for key in (candidate.descriptor_key, candidate.preserving_descriptor_key)
        if key is not None and key not in descriptor_ref_keys
    )
    if missing_descriptor_refs:
        raise ValueError(
            "AMDGPU memory descriptor candidates require missing descriptor "
            f"refs: {', '.join(missing_descriptor_refs)}"
        )
    return _MEMORY_DESCRIPTOR_CANDIDATES


__all__ = (
    "_ATOMIC_ADDRESS_FORM",
    "_ATOMIC_CANDIDATE_KIND_ORDER",
    "_ATOMIC_CANDIDATE_MEMORY_ORDER",
    "_ATOMIC_CANDIDATE_OPERATION_ORDER",
    "_ATOMIC_KIND",
    "_ATOMIC_MEMORY_SPACE",
    "_amdgpu_async_gather_candidate_from_overlay",
    "_amdgpu_atomic_candidate_from_overlay",
    "_amdgpu_atomic_candidate_sort_key",
    "_atomic_address_form",
    "_global_to_workgroup_widths",
    "_record_amdgpu_async_gather_candidate",
    "_record_amdgpu_atomic_candidate",
    "amdgpu_async_gather_descriptor_candidates",
    "amdgpu_atomic_descriptor_candidates",
    "amdgpu_memory_descriptor_candidates",
)
