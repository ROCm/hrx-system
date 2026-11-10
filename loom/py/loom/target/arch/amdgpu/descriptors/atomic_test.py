# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.target.arch.amdgpu.descriptors.api import (
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X,
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
    _with_storage_lease_rows,
)
from loom.target.arch.amdgpu.descriptors.candidates import (
    _amdgpu_atomic_candidate_from_overlay,
)
from loom.target.arch.amdgpu.descriptors.cdna import (
    _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _COUNTER_LDS,
    _COUNTER_VMEM_LOAD,
    _COUNTER_VMEM_STORE,
    _COUNTER_X,
    AmdgpuAtomicMemorySpace,
    AmdgpuMemoryAddressForm,
)
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.arch.amdgpu.descriptors.rdna3 import (
    _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.rdna4 import (
    _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx9_4_generic_core_overlays,
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx125x_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
)
from loom.target.low_descriptors import MemorySpace, OperandRole, StorageLeaseKind


def test_global_atomic_address_forms_cover_every_operation_and_target() -> None:
    for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items():
        overlays = {
            row.descriptor_key: row
            for row in builder.overlay_rows()
            if row.descriptor_key.startswith("amdgpu.global_atomic_")
        }
        full_address_keys = {key for key in overlays if not key.endswith("_saddr")}
        scalar_base_keys = {
            key.removesuffix("_saddr") for key in overlays if key.endswith("_saddr")
        }
        assert full_address_keys, target
        assert full_address_keys == scalar_base_keys, target
        for key in sorted(full_address_keys):
            full = overlays[key]
            scalar_base = overlays[f"{key}_saddr"]
            assert full.instruction_name == scalar_base.instruction_name
            assert full.semantic_tag == scalar_base.semantic_tag
            assert full.effects == scalar_base.effects
            assert full.immediates == scalar_base.immediates
            assert all(
                effect.memory_space is MemorySpace.GLOBAL for effect in full.effects
            )
            full_operands = {
                row.descriptor_operand.field_name: row.descriptor_operand
                for row in full.operands
            }
            scalar_operands = {
                row.descriptor_operand.field_name: row.descriptor_operand
                for row in scalar_base.operands
            }
            assert full_operands["addr"].unit_count == 2
            assert scalar_operands["addr"].unit_count == 1
            assert "saddr" not in full_operands
            assert scalar_operands["saddr"].unit_count == 2
            assert "SADDR" in dict(full.fixed_encoding_fields)
            assert "SADDR" not in dict(scalar_base.fixed_encoding_fields)
            full_candidate = _amdgpu_atomic_candidate_from_overlay(full)
            scalar_candidate = _amdgpu_atomic_candidate_from_overlay(scalar_base)
            assert full_candidate is not None
            assert scalar_candidate is not None
            assert full_candidate.memory_space is AmdgpuAtomicMemorySpace.GLOBAL
            assert full_candidate.address_form is AmdgpuMemoryAddressForm.FLAT
            assert full_candidate.operation_kind == scalar_candidate.operation_kind
            assert full_candidate.atomic_kind == scalar_candidate.atomic_kind
            assert full_candidate.value_kind == scalar_candidate.value_kind


def test_cdna_global_integer_atomics_preserve_return_and_native_spelling() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx9_4_generic_core_overlays(),
    ):
        descriptors = {row.descriptor_key: row for row in overlays}
        for descriptor_suffix, mnemonic_suffix, semantic_suffix in (
            ("add_u32", "add", "add.u32"),
            ("sub_u32", "sub", "sub.u32"),
            ("min_i32", "smin", "min.i32"),
            ("max_i32", "smax", "max.i32"),
            ("min_u32", "umin", "min.u32"),
            ("max_u32", "umax", "max.u32"),
            ("and_b32", "and", "and.b32"),
            ("or_b32", "or", "or.b32"),
            ("xor_b32", "xor", "xor.b32"),
            ("swap_b32", "swap", "exchange.b32"),
        ):
            return_forms = (True,) if mnemonic_suffix == "swap" else (False, True)
            for returns_old_value in return_forms:
                return_suffix = "_rtn" if returns_old_value else ""
                descriptor = descriptors[
                    f"amdgpu.global_atomic_{descriptor_suffix}{return_suffix}_saddr"
                ]
                assert descriptor.instruction_name == (
                    f"GLOBAL_ATOMIC_{mnemonic_suffix.upper()}"
                )
                assert descriptor.mnemonic == f"global_atomic_{mnemonic_suffix}"
                assert descriptor.semantic_tag == (
                    f"memory.global.atomic.{semantic_suffix}"
                    + (".return" if returns_old_value else "")
                )
                assert dict(descriptor.fixed_encoding_fields)["SC0"] == int(
                    returns_old_value
                )
                results = tuple(
                    operand.descriptor_operand
                    for operand in descriptor.operands
                    if operand.descriptor_operand.role is OperandRole.RESULT
                )
                assert len(results) == int(returns_old_value)
                assert all(result.unit_count == 1 for result in results)
                assert tuple(effect.memory_space for effect in descriptor.effects) == (
                    MemorySpace.GLOBAL,
                    MemorySpace.GLOBAL,
                )
                assert tuple(effect.width_bits for effect in descriptor.effects) == (
                    32,
                    32,
                )
                assert descriptor.asm_forms[0].mnemonic == (
                    f"global_atomic_{mnemonic_suffix}{return_suffix}_saddr"
                )


def test_cdna_atomic_scope_is_independent_of_return_control() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx9_4_generic_core_overlays(),
    ):
        descriptors = tuple(
            row
            for row in overlays
            if row.descriptor_key.startswith(
                ("amdgpu.global_atomic_", "amdgpu.flat_atomic_")
            )
        )
        assert descriptors
        for descriptor in descriptors:
            fixed = dict(descriptor.fixed_encoding_fields)
            returns_old_value = any(
                operand.descriptor_operand.role is OperandRole.RESULT
                for operand in descriptor.operands
            )
            assert fixed["SC0"] == int(returns_old_value)
            assert "SC1" not in fixed
            immediates = {row.field_name: row for row in descriptor.immediates}
            assert "sc0" not in immediates
            scope = immediates["sc1"]
            assert (scope.bit_width, scope.unsigned_max, scope.default_value) == (
                1,
                1,
                0,
            )


def test_flat_atomics_complete_both_domains_without_duplicate_accesses() -> None:
    for base, overlays, enable_xcnt in (
        (_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, _gfx940_core_overlays(), False),
        (_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx950_core_overlays(), False),
        (_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, _gfx11_core_overlays(), False),
        (_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx12_core_overlays(), False),
        (_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx125x_core_overlays(), True),
    ):
        descriptors = tuple(
            _amdgpu_contract_descriptor_from_overlay(overlay)
            for overlay in overlays
            if overlay.descriptor_key.startswith("amdgpu.flat_atomic_")
        )
        descriptor_set = _with_storage_lease_rows(
            replace(base, descriptors=descriptors),
            builder_flags=(
                _AMDGPU_CORE_DESCRIPTOR_SET_BUILDER_FLAG_GFX125X if enable_xcnt else 0
            ),
        )
        schedule_classes = {row.name: row for row in descriptor_set.schedule_classes}
        assert descriptors
        for descriptor in descriptor_set.descriptors:
            results = [
                operand
                for operand in descriptor.operands
                if operand.role is OperandRole.RESULT
            ]
            completion_counters = {
                _COUNTER_LDS,
                _COUNTER_VMEM_LOAD if results else _COUNTER_VMEM_STORE,
            }
            assert {
                hazard.counter_id
                for hazard in schedule_classes[descriptor.schedule_class].hazards
            } == completion_counters
            result_leases = [
                lease
                for lease in descriptor.storage_leases
                if lease.kind is StorageLeaseKind.RESULT_WRITE
            ]
            assert {lease.release_class_id for lease in result_leases} == (
                completion_counters if results else set()
            )
            assert all(
                lease.unit_count == results[0].unit_count for lease in result_leases
            )
            source_counters = {
                lease.release_class_id
                for lease in descriptor.storage_leases
                if lease.kind is StorageLeaseKind.SOURCE_READ
            }
            assert source_counters == ({_COUNTER_X} if enable_xcnt else set())
            # Counter completion does not duplicate the single read/write access
            # reported for an atomic, including compare-and-swap and wide forms.
            assert len(descriptor.effects) == 2
            assert all(effect.counter_id == 0 for effect in descriptor.effects)
            assert descriptor.effects[0].width_bits == descriptor.effects[1].width_bits
