# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Canonical AMDGPU memory packet selection rows and variants."""

from __future__ import annotations

from collections.abc import Mapping

from .common import *

_B = AmdgpuMemoryDescriptorDomain.BUFFER_RESOURCE
_GS = AmdgpuMemoryDescriptorDomain.GLOBAL_SADDR
_L = AmdgpuMemoryDescriptorDomain.LDS
_GF = AmdgpuMemoryDescriptorDomain.GLOBAL_FLAT
_SM = AmdgpuMemoryDescriptorDomain.GLOBAL_SMEM
_SC = AmdgpuMemoryDescriptorDomain.SCRATCH
_F = AmdgpuMemoryDescriptorDomain.GENERIC_FLAT

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
_I8 = AmdgpuMemoryPayloadFormat.SIGNED_8BIT_INTEGER
_I16 = AmdgpuMemoryPayloadFormat.SIGNED_16BIT_INTEGER
_U8 = AmdgpuMemoryPayloadFormat.UNSIGNED_8BIT_INTEGER


def preserved_memory_descriptor_key(key: str) -> str:
    return f"{key}_preserve"


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
    key = f"amdgpu.{descriptor_ref_name.lower()}"
    return AmdgpuMemoryDescriptorCandidate(
        domain=domain,
        address_form=address_form,
        operation_kind=operation_kind,
        packet_byte_count=packet_byte_count,
        payload_register_class=payload_register_class,
        payload_format=payload_format,
        payload_register_count=payload_register_count,
        descriptor_key=key,
        preserving_descriptor_key=(
            preserved_memory_descriptor_key(key)
            if operation_kind is _LD and payload_register_class is _V
            else None
        ),
    )


# Columns: domain, address form, operation, packet bytes, payload register class,
# payload format, payload register count, descriptor ref suffix.
_MEMORY_DESCRIPTOR_CANDIDATE_ROWS = (
    (_B, _AD, _LD, 1, _V, _G, 1, "BUFFER_LOAD_I8"),
    (_B, _AD, _LD, 1, _V, _I8, 1, "BUFFER_LOAD_I8"),
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
    (_SM, _SMEM, _LD, 12, _S, _G, 3, "S_LOAD_B96"),
    (_SM, _SMEM, _LD, 16, _S, _G, 4, "S_LOAD_DWORDX4"),
    (_GS, _SA, _LD, 1, _V, _G, 1, "GLOBAL_LOAD_I8_SADDR"),
    (_GS, _SA, _LD, 1, _V, _I8, 1, "GLOBAL_LOAD_I8_SADDR"),
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
    (_GF, _FL, _LD, 1, _V, _I8, 1, "GLOBAL_LOAD_I8"),
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
    (_F, _FL, _LD, 1, _V, _G, 1, "FLAT_LOAD_I8"),
    (_F, _FL, _LD, 1, _V, _I8, 1, "FLAT_LOAD_I8"),
    (_F, _FL, _LD, 1, _V, _U8, 1, "FLAT_LOAD_U8"),
    (_F, _FL, _ST, 1, _V, _G, 1, "FLAT_STORE_B8"),
    (_F, _FL, _LD, 2, _V, _F16, 1, "FLAT_LOAD_U16"),
    (_F, _FL, _LD, 2, _V, _I16, 1, "FLAT_LOAD_I16"),
    (_F, _FL, _LD, 2, _V, _G, 1, "FLAT_LOAD_U16"),
    (_F, _FL, _ST, 2, _V, _G, 1, "FLAT_STORE_B16"),
    (_F, _FL, _LD, 4, _V, _G, 1, "FLAT_LOAD_B32"),
    (_F, _FL, _LD, 8, _V, _G, 2, "FLAT_LOAD_B64"),
    (_F, _FL, _LD, 12, _V, _G, 3, "FLAT_LOAD_B96"),
    (_F, _FL, _LD, 16, _V, _G, 4, "FLAT_LOAD_B128"),
    (_F, _FL, _ST, 4, _V, _G, 1, "FLAT_STORE_B32"),
    (_F, _FL, _ST, 8, _V, _G, 2, "FLAT_STORE_B64"),
    (_F, _FL, _ST, 12, _V, _G, 3, "FLAT_STORE_B96"),
    (_F, _FL, _ST, 16, _V, _G, 4, "FLAT_STORE_B128"),
    (_L, _AD, _LD, 1, _V, _G, 1, "DS_READ_U8"),
    (_L, _AD, _LD, 1, _V, _I8, 1, "DS_READ_U8"),
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
    (_SC, _SV, _LD, 1, _V, _I8, 1, "SCRATCH_LOAD_I8_VADDR"),
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


def preserving_memory_descriptor_keys(
    operand_forms: Mapping[str, tuple[OperandForm, ...]],
) -> frozenset[str]:
    """Selects native load candidates and their address-form replacements."""

    pending = [
        candidate.descriptor_key
        for candidate in _MEMORY_DESCRIPTOR_CANDIDATES
        if candidate.preserving_descriptor_key is not None
    ]
    selected = set()
    while pending:
        key = pending.pop()
        if key in selected or key not in operand_forms:
            continue
        selected.add(key)
        pending.extend(form.replacement_descriptor for form in operand_forms[key])
    return frozenset(selected)
