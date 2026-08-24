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
    "add.u64": (AmdgpuAtomicKind.ADDI, AmdgpuAtomicValueKind.I64),
    "exchange.u64": (AmdgpuAtomicKind.XCHGI, AmdgpuAtomicValueKind.I64),
    "add.f32": (AmdgpuAtomicKind.ADDF, AmdgpuAtomicValueKind.F32),
    "minnum.f32": (AmdgpuAtomicKind.MINNUMF, AmdgpuAtomicValueKind.F32),
    "maxnum.f32": (AmdgpuAtomicKind.MAXNUMF, AmdgpuAtomicValueKind.F32),
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

_B = AmdgpuMemoryDescriptorDomain.BUFFER_RESOURCE
_GS = AmdgpuMemoryDescriptorDomain.GLOBAL_SADDR
_L = AmdgpuMemoryDescriptorDomain.LDS
_GF = AmdgpuMemoryDescriptorDomain.GLOBAL_FLAT
_SM = AmdgpuMemoryDescriptorDomain.GLOBAL_SMEM
_SC = AmdgpuMemoryDescriptorDomain.SCRATCH

_AD = AmdgpuMemoryAddressForm.DEFAULT
_BOZ = AmdgpuMemoryAddressForm.BUFFER_OFF_ZERO
_DS2 = AmdgpuMemoryAddressForm.DS_2ADDR
_SA = AmdgpuMemoryAddressForm.GLOBAL_SADDR
_ADT = AmdgpuMemoryAddressForm.DS_ADDTID
_FL = AmdgpuMemoryAddressForm.FLAT
_SMEM = AmdgpuMemoryAddressForm.GLOBAL_SMEM
_SV = AmdgpuMemoryAddressForm.SCRATCH_VADDR

_LD = AmdgpuMemoryOperationKind.LOAD
_ST = AmdgpuMemoryOperationKind.STORE

_V = AmdgpuMemoryPayloadRegisterClass.VGPR
_S = AmdgpuMemoryPayloadRegisterClass.SGPR

_G = AmdgpuMemoryPayloadFormat.GENERIC
_F16 = AmdgpuMemoryPayloadFormat.LOW_16BIT_FLOAT
_I16 = AmdgpuMemoryPayloadFormat.SIGNED_16BIT_INTEGER
_U8 = AmdgpuMemoryPayloadFormat.UNSIGNED_8BIT_INTEGER


def _memory_candidate(
    domain: AmdgpuMemoryDescriptorDomain,
    address_form: AmdgpuMemoryAddressForm,
    operation_kind: AmdgpuMemoryOperationKind,
    packet_byte_count: int,
    payload_register_class: AmdgpuMemoryPayloadRegisterClass,
    payload_format: AmdgpuMemoryPayloadFormat,
    payload_register_count: int,
    descriptor_ref_name: str,
) -> AmdgpuMemoryDescriptorCandidate:
    return AmdgpuMemoryDescriptorCandidate(
        domain=domain,
        address_form=address_form,
        operation_kind=operation_kind,
        packet_byte_count=packet_byte_count,
        payload_register_class=payload_register_class,
        payload_format=payload_format,
        payload_register_count=payload_register_count,
        descriptor_key=f"amdgpu.{descriptor_ref_name.lower()}",
    )


# Columns: domain, address form, operation, packet bytes, payload register class,
# payload format, payload register count, descriptor ref suffix.
_MEMORY_DESCRIPTOR_CANDIDATE_ROWS = (
    (_B, _AD, _LD, 1, _V, _G, 1, "BUFFER_LOAD_I8"),
    (_B, _AD, _LD, 1, _V, _U8, 1, "BUFFER_LOAD_U8"),
    (_B, _AD, _ST, 1, _V, _G, 1, "BUFFER_STORE_B8"),
    (_B, _AD, _LD, 2, _V, _F16, 1, "BUFFER_LOAD_B16_D16"),
    (_B, _AD, _LD, 2, _V, _I16, 1, "BUFFER_LOAD_I16"),
    (_B, _BOZ, _LD, 2, _V, _I16, 1, "BUFFER_LOAD_I16_OFF_ZERO"),
    (_B, _AD, _LD, 2, _V, _G, 1, "BUFFER_LOAD_U16"),
    (_B, _BOZ, _LD, 2, _V, _G, 1, "BUFFER_LOAD_U16_OFF_ZERO"),
    (_B, _AD, _ST, 2, _V, _G, 1, "BUFFER_STORE_B16"),
    (_B, _AD, _LD, 4, _V, _G, 1, "BUFFER_LOAD_DWORD"),
    (_B, _BOZ, _LD, 4, _V, _G, 1, "BUFFER_LOAD_DWORD_OFF_ZERO"),
    (_B, _AD, _LD, 8, _V, _G, 2, "BUFFER_LOAD_B64"),
    (_B, _BOZ, _LD, 8, _V, _G, 2, "BUFFER_LOAD_B64_OFF_ZERO"),
    (_B, _AD, _LD, 8, _V, _G, 2, "BUFFER_LOAD_DWORDX2"),
    (_B, _BOZ, _LD, 8, _V, _G, 2, "BUFFER_LOAD_DWORDX2_OFF_ZERO"),
    (_B, _AD, _LD, 12, _V, _G, 3, "BUFFER_LOAD_B96"),
    (_B, _BOZ, _LD, 12, _V, _G, 3, "BUFFER_LOAD_B96_OFF_ZERO"),
    (_B, _AD, _LD, 16, _V, _G, 4, "BUFFER_LOAD_B128"),
    (_B, _BOZ, _LD, 16, _V, _G, 4, "BUFFER_LOAD_B128_OFF_ZERO"),
    (_B, _AD, _LD, 16, _V, _G, 4, "BUFFER_LOAD_DWORDX4"),
    (_B, _BOZ, _LD, 16, _V, _G, 4, "BUFFER_LOAD_DWORDX4_OFF_ZERO"),
    (_B, _AD, _ST, 4, _V, _G, 1, "BUFFER_STORE_DWORD"),
    (_B, _BOZ, _ST, 4, _V, _G, 1, "BUFFER_STORE_DWORD_OFF_ZERO"),
    (_B, _AD, _ST, 8, _V, _G, 2, "BUFFER_STORE_B64"),
    (_B, _BOZ, _ST, 8, _V, _G, 2, "BUFFER_STORE_B64_OFF_ZERO"),
    (_B, _AD, _ST, 8, _V, _G, 2, "BUFFER_STORE_DWORDX2"),
    (_B, _BOZ, _ST, 8, _V, _G, 2, "BUFFER_STORE_DWORDX2_OFF_ZERO"),
    (_B, _AD, _ST, 12, _V, _G, 3, "BUFFER_STORE_B96"),
    (_B, _BOZ, _ST, 12, _V, _G, 3, "BUFFER_STORE_B96_OFF_ZERO"),
    (_B, _AD, _ST, 16, _V, _G, 4, "BUFFER_STORE_B128"),
    (_B, _BOZ, _ST, 16, _V, _G, 4, "BUFFER_STORE_B128_OFF_ZERO"),
    (_B, _AD, _ST, 16, _V, _G, 4, "BUFFER_STORE_DWORDX4"),
    (_B, _BOZ, _ST, 16, _V, _G, 4, "BUFFER_STORE_DWORDX4_OFF_ZERO"),
    (_SM, _SMEM, _LD, 4, _S, _G, 1, "S_LOAD_DWORD"),
    (_SM, _SMEM, _LD, 8, _S, _G, 2, "S_LOAD_DWORDX2"),
    (_SM, _SMEM, _LD, 16, _S, _G, 4, "S_LOAD_DWORDX4"),
    (_GS, _SA, _LD, 1, _V, _G, 1, "GLOBAL_LOAD_I8_SADDR"),
    (_GS, _SA, _LD, 1, _V, _U8, 1, "GLOBAL_LOAD_U8_SADDR"),
    (_GS, _SA, _ST, 1, _V, _G, 1, "GLOBAL_STORE_B8_SADDR"),
    (_GS, _SA, _LD, 2, _V, _F16, 1, "GLOBAL_LOAD_B16_D16_SADDR"),
    (_GS, _SA, _LD, 2, _V, _I16, 1, "GLOBAL_LOAD_I16_SADDR"),
    (_GS, _SA, _LD, 2, _V, _G, 1, "GLOBAL_LOAD_U16_SADDR"),
    (_GS, _SA, _ST, 2, _V, _G, 1, "GLOBAL_STORE_B16_SADDR"),
    (_GS, _SA, _LD, 4, _V, _G, 1, "GLOBAL_LOAD_B32_SADDR"),
    (_GS, _SA, _LD, 8, _V, _G, 2, "GLOBAL_LOAD_B64_SADDR"),
    (_GS, _SA, _LD, 12, _V, _G, 3, "GLOBAL_LOAD_B96_SADDR"),
    (_GS, _SA, _LD, 16, _V, _G, 4, "GLOBAL_LOAD_B128_SADDR"),
    (_GS, _SA, _ST, 4, _V, _G, 1, "GLOBAL_STORE_B32_SADDR"),
    (_GS, _SA, _ST, 8, _V, _G, 2, "GLOBAL_STORE_B64_SADDR"),
    (_GS, _SA, _ST, 12, _V, _G, 3, "GLOBAL_STORE_B96_SADDR"),
    (_GS, _SA, _ST, 16, _V, _G, 4, "GLOBAL_STORE_B128_SADDR"),
    (_GF, _FL, _LD, 1, _V, _G, 1, "GLOBAL_LOAD_I8"),
    (_GF, _FL, _LD, 1, _V, _U8, 1, "GLOBAL_LOAD_U8"),
    (_GF, _FL, _ST, 1, _V, _G, 1, "GLOBAL_STORE_B8"),
    (_GF, _FL, _LD, 2, _V, _F16, 1, "GLOBAL_LOAD_B16_D16"),
    (_GF, _FL, _LD, 2, _V, _I16, 1, "GLOBAL_LOAD_I16"),
    (_GF, _FL, _LD, 2, _V, _G, 1, "GLOBAL_LOAD_U16"),
    (_GF, _FL, _ST, 2, _V, _G, 1, "GLOBAL_STORE_B16"),
    (_GF, _FL, _LD, 4, _V, _G, 1, "GLOBAL_LOAD_B32"),
    (_GF, _FL, _LD, 8, _V, _G, 2, "GLOBAL_LOAD_B64"),
    (_GF, _FL, _LD, 12, _V, _G, 3, "GLOBAL_LOAD_B96"),
    (_GF, _FL, _LD, 16, _V, _G, 4, "GLOBAL_LOAD_B128"),
    (_GF, _FL, _ST, 4, _V, _G, 1, "GLOBAL_STORE_B32"),
    (_GF, _FL, _ST, 8, _V, _G, 2, "GLOBAL_STORE_B64"),
    (_GF, _FL, _ST, 12, _V, _G, 3, "GLOBAL_STORE_B96"),
    (_GF, _FL, _ST, 16, _V, _G, 4, "GLOBAL_STORE_B128"),
    (_L, _AD, _LD, 1, _V, _G, 1, "DS_READ_U8"),
    (_L, _AD, _LD, 1, _V, _U8, 1, "DS_READ_U8"),
    (_L, _AD, _ST, 1, _V, _G, 1, "DS_WRITE_B8"),
    (_L, _AD, _LD, 2, _V, _I16, 1, "DS_READ_U16"),
    (_L, _AD, _LD, 2, _V, _G, 1, "DS_READ_U16"),
    (_L, _AD, _LD, 4, _V, _G, 1, "DS_READ_B32"),
    (_L, _AD, _LD, 8, _V, _G, 2, "DS_READ_B64"),
    (_L, _AD, _LD, 12, _V, _G, 3, "DS_READ_B96"),
    (_L, _AD, _LD, 16, _V, _G, 4, "DS_READ_B128"),
    (_L, _AD, _ST, 2, _V, _G, 1, "DS_WRITE_B16"),
    (_L, _AD, _ST, 4, _V, _G, 1, "DS_WRITE_B32"),
    (_L, _AD, _ST, 8, _V, _G, 2, "DS_WRITE_B64"),
    (_L, _AD, _ST, 12, _V, _G, 3, "DS_WRITE_B96"),
    (_L, _AD, _ST, 16, _V, _G, 4, "DS_WRITE_B128"),
    (_L, _DS2, _LD, 8, _V, _G, 2, "DS_READ2_B32"),
    (_L, _DS2, _ST, 8, _V, _G, 2, "DS_WRITE2_B32"),
    (_L, _DS2, _LD, 8, _V, _G, 2, "DS_READ2ST64_B32"),
    (_L, _DS2, _ST, 8, _V, _G, 2, "DS_WRITE2ST64_B32"),
    (_L, _DS2, _LD, 16, _V, _G, 4, "DS_READ2_B64"),
    (_L, _DS2, _ST, 16, _V, _G, 4, "DS_WRITE2_B64"),
    (_L, _DS2, _LD, 16, _V, _G, 4, "DS_READ2ST64_B64"),
    (_L, _DS2, _ST, 16, _V, _G, 4, "DS_WRITE2ST64_B64"),
    (_L, _ADT, _LD, 4, _V, _G, 1, "DS_READ_ADDTID_B32"),
    (_L, _ADT, _ST, 4, _V, _G, 1, "DS_WRITE_ADDTID_B32"),
    (_SC, _SV, _LD, 1, _V, _G, 1, "SCRATCH_LOAD_I8_VADDR"),
    (_SC, _SV, _LD, 1, _V, _U8, 1, "SCRATCH_LOAD_U8_VADDR"),
    (_SC, _SV, _LD, 2, _V, _F16, 1, "SCRATCH_LOAD_U16_VADDR"),
    (_SC, _SV, _LD, 2, _V, _G, 1, "SCRATCH_LOAD_U16_VADDR"),
    (_SC, _SV, _LD, 2, _V, _I16, 1, "SCRATCH_LOAD_I16_VADDR"),
    (_SC, _SV, _LD, 4, _V, _G, 1, "SCRATCH_LOAD_B32_VADDR"),
    (_SC, _SV, _LD, 8, _V, _G, 2, "SCRATCH_LOAD_B64_VADDR"),
    (_SC, _SV, _LD, 12, _V, _G, 3, "SCRATCH_LOAD_B96_VADDR"),
    (_SC, _SV, _LD, 16, _V, _G, 4, "SCRATCH_LOAD_B128_VADDR"),
    (_SC, _SV, _ST, 1, _V, _G, 1, "SCRATCH_STORE_B8_VADDR"),
    (_SC, _SV, _ST, 2, _V, _G, 1, "SCRATCH_STORE_B16_VADDR"),
    (_SC, _SV, _ST, 4, _V, _G, 1, "SCRATCH_STORE_B32_VADDR"),
    (_SC, _SV, _ST, 8, _V, _G, 2, "SCRATCH_STORE_B64_VADDR"),
    (_SC, _SV, _ST, 12, _V, _G, 3, "SCRATCH_STORE_B96_VADDR"),
    (_SC, _SV, _ST, 16, _V, _G, 4, "SCRATCH_STORE_B128_VADDR"),
)

_MEMORY_DESCRIPTOR_CANDIDATES = tuple(
    _memory_candidate(*row) for row in _MEMORY_DESCRIPTOR_CANDIDATE_ROWS
)


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
        candidate.descriptor_key
        for candidate in _MEMORY_DESCRIPTOR_CANDIDATES
        if candidate.descriptor_key not in descriptor_ref_keys
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
    "_MEMORY_DESCRIPTOR_CANDIDATES",
    "_amdgpu_async_gather_candidate_from_overlay",
    "_amdgpu_atomic_candidate_from_overlay",
    "_amdgpu_atomic_candidate_sort_key",
    "_atomic_address_form",
    "_global_to_workgroup_widths",
    "_memory_candidate",
    "_record_amdgpu_async_gather_candidate",
    "_record_amdgpu_atomic_candidate",
    "amdgpu_async_gather_descriptor_candidates",
    "amdgpu_atomic_descriptor_candidates",
    "amdgpu_memory_descriptor_candidates",
)
