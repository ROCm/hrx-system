# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuOperandPredefinedValueRef,
)
from loom.target.arch.amdgpu.descriptors import (
    _ADDRESS_OFFSET_DS16_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
    _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS,
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
    _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_GFX9_4_GENERIC_MATRIX_TIMINGS,
    _AMDGPU_GFX11_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_GFX942_MATRIX_TIMINGS,
    _AMDGPU_GFX950_MATRIX_TIMINGS,
    _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_TRANS_DESCRIPTOR_KEYS,
    _AMDGPU_TRANS_PROXY_LATENCY_CYCLES,
    _COUNTER_ASYNC,
    _COUNTER_LDS,
    _COUNTER_TENSOR,
    _COUNTER_VMEM_LOAD,
    _COUNTER_VMEM_STORE,
    _D16_PARTIAL_REGISTER_ADDRESSABLE_UNIT_COUNT,
    _EARLY_CLOBBER_RESULT_CONSTRAINTS,
    _GFX12_TH_ATOMIC_RETURN_VALUE,
    _MUBUF_SOFFSET_INLINE_ZERO,
    _REG_EXEC,
    _REG_M0,
    _REG_MODE,
    _REG_PART_SGPR_HIGH16,
    _REG_PART_SGPR_LOW16,
    _REG_PART_VGPR_HIGH16,
    _REG_PART_VGPR_LOW16,
    _REG_SGPR,
    _REG_VCC,
    _REG_VGPR,
    _RESOURCE_MFMA,
    _RESOURCE_SALU,
    _RESOURCE_SWMMAC,
    _RESOURCE_VALU,
    _RESOURCE_WMMA,
    _SCHEDULE_LDS_STORE,
    _SCHEDULE_MATRIX,
    _SCHEDULE_MFMA_QUALIFIED_PREFIX,
    _SCHEDULE_MODE_CONTROL,
    _SCHEDULE_PACKED_DOT,
    _SCHEDULE_SALU,
    _SCHEDULE_SALU_COMPARE,
    _SCHEDULE_SMEM_LOAD,
    _SCHEDULE_SMEM_STORE,
    _SCHEDULE_SWMMAC,
    _SCHEDULE_VALU,
    _SCHEDULE_VMEM_LOAD,
    _SCHEDULE_VMEM_LOAD_LDS,
    _SCHEDULE_VMEM_STORE,
    _SCHEDULE_WAIT_ALU,
    _SCHEDULE_WMMA,
    _SOURCE_INLINE_F32_ENCODING_ID,
    _SOURCE_INLINE_U32_ENCODING_ID,
    _VBUFFER_SOFFSET_NULL,
    _WAIT_COUNTER_ASYNC_ENCODING_ID,
    _WAIT_COUNTER_TENSOR_ENCODING_ID,
    AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
    AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    AMDGPU_DESCRIPTOR_CATEGORIES,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP1,
    AMDGPU_ENCODING_FORMAT_VOP2,
    AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP3PX2,
    AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DELAY_ALU,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST,
    AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG,
    AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
    AmdgpuAtomicDescriptorCandidate,
    AmdgpuAtomicKind,
    AmdgpuAtomicMemorySpace,
    AmdgpuAtomicOperationKind,
    AmdgpuAtomicValueKind,
    AmdgpuMemoryAddressForm,
    _amdgpu_contract_descriptor_from_overlay,
    _amdgpu_core_descriptor_set,
    _amdgpu_core_descriptor_set_bases,
    _amdgpu_core_descriptor_set_intersection,
    _amdgpu_mfma_schedule_class_name,
    _amdgpu_trans_schedule_class_name,
    _amdgpu_trans_schedule_classes,
    _categorize_amdgpu_descriptors,
    _gfx9_4_generic_core_overlays,
    _gfx11_core_overlays,
    _gfx11_generic_core_overlays,
    _gfx12_5_generic_core_overlays,
    _gfx12_core_overlays,
    _gfx12_generic_core_overlays,
    _gfx115x_core_overlays,
    _gfx125x_core_overlays,
    _gfx125x_reg_classes,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
    _predefined,
    _record_amdgpu_atomic_candidate,
    _validate_address_immediate_units,
    _validate_descriptor_encoding_formats,
    _validate_dpp_control_fields,
    _with_execution_mask_state_read,
    _with_gfx125x_vgpr_msb_address_state,
    _with_gfx125x_vgpr_msb_address_states,
    _with_mode_state_read,
    amdgpu_atomic_descriptor_candidates,
    amdgpu_descriptor_category_groups,
    amdgpu_descriptor_ref_keys,
    amdgpu_encoding_field_id,
)
from loom.target.arch.amdgpu.descriptors.api import (
    _with_instruction_classes,
    _with_storage_lease_rows,
)
from loom.target.arch.amdgpu.descriptors.cluster import (
    _cluster_load_async_to_lds_descriptor,
    _s_wait_asynccnt_descriptor,
)
from loom.target.arch.amdgpu.descriptors.control import (
    _s_delay_alu_descriptor,
    _s_set_vgpr_msb_descriptor,
)
from loom.target.arch.amdgpu.descriptors.tensor import (
    _s_wait_tensorcnt_descriptor,
    _tensor_load_to_lds_descriptor,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_SOPP,
    AMDGPU_ENCODING_FORMAT_VGLOBAL,
    AMDGPU_ENCODING_FORMAT_VIMAGE,
    AMDGPU_ENCODING_FORMAT_VOP1_DPP16,
    AMDGPU_GFX125X_VOP_VGPR_MSB_FORMAT_NAMES,
    AmdgpuVgprMsbSlot,
    amdgpu_dpp_control_is_valid,
    amdgpu_gfx125x_vgpr_msb_slot,
)
from loom.target.arch.amdgpu.isa_xml import AmdgpuIsaEncoding, AmdgpuIsaSpec
from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES,
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EncodingFieldValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    NativeAsmValue,
    NativeAsmValueKind,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
    StorageLeaseAttachment,
    StorageLeaseFlag,
    StorageLeaseKind,
)


def test_contract_descriptor_projection_preserves_operation_kind() -> None:
    for overlay_builder in _AMDGPU_CONTRACT_DESCRIPTOR_OVERLAY_BUILDERS.values():
        overlay = overlay_builder()
        descriptor = _amdgpu_contract_descriptor_from_overlay(overlay)
        assert descriptor.op_kind is overlay.op_kind


def test_generic_descriptor_contracts_are_member_intersections() -> None:
    base_cases = (
        (
            _AMDGPU_GFX11_GENERIC_CORE_DESCRIPTOR_SET_BASE,
            (
                _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
                _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
            ),
        ),
        (
            _AMDGPU_GFX12_GENERIC_CORE_DESCRIPTOR_SET_BASE,
            (_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,),
        ),
    )
    for generic, members in base_cases:
        expected = _amdgpu_core_descriptor_set_intersection(
            key=generic.key,
            members=members,
        )
        assert generic == expected

    gfx9_4_base_intersection = _amdgpu_core_descriptor_set_intersection(
        key=_AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE.key,
        members=(
            _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
            _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        ),
    )
    assert (
        replace(
            _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
            schedule_classes=gfx9_4_base_intersection.schedule_classes,
        )
        == gfx9_4_base_intersection
    )

    gfx115x_overlays = {
        overlay.descriptor_key: overlay for overlay in _gfx115x_core_overlays()
    }
    gfx950_overlays = {
        overlay.descriptor_key: overlay for overlay in _gfx950_core_overlays()
    }
    gfx9_4_common_overlays = tuple(
        overlay
        for overlay in _gfx940_core_overlays()
        if gfx950_overlays.get(overlay.descriptor_key) == overlay
    )
    gfx9_4_packed8_matrix_keys = {
        overlay.descriptor_key
        for overlay in gfx9_4_common_overlays
        if (
            overlay.semantic_tag is not None
            and overlay.semantic_tag.startswith("matrix.")
            and (".fp8" in overlay.semantic_tag or ".bf8" in overlay.semantic_tag)
        )
    }
    assert len(gfx9_4_packed8_matrix_keys) == 16
    assert _gfx9_4_generic_core_overlays() == tuple(
        overlay
        for overlay in gfx9_4_common_overlays
        if overlay.descriptor_key not in gfx9_4_packed8_matrix_keys
    )
    assert _gfx11_generic_core_overlays() == tuple(
        overlay
        for overlay in _gfx11_core_overlays()
        if gfx115x_overlays.get(overlay.descriptor_key) == overlay
    )
    assert _gfx12_generic_core_overlays() == _gfx12_core_overlays()
    gfx125x_overlays = _gfx125x_core_overlays()
    assert all(
        not (overlay.semantic_tag or "").startswith("matrix.wmma.")
        for overlay in gfx125x_overlays
    )
    assert _gfx12_5_generic_core_overlays() == tuple(
        overlay
        for overlay in gfx125x_overlays
        if not (overlay.semantic_tag or "").startswith("matrix.swmmac.")
    )


def test_v_readlane_b32_reserves_unused_vop3_source() -> None:
    overlays = {overlay.descriptor_key: overlay for overlay in _gfx11_core_overlays()}
    for descriptor_key in (
        "amdgpu.v_readlane_b32.src1_inline",
        "amdgpu.v_readlane_b32.src1_sgpr",
    ):
        assert overlays[descriptor_key].fixed_encoding_fields == (("SRC2", 0),)


def _descriptor_set(*descriptors: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key="amdgpu.test.core",
        target_key="amdgpu.test",
        feature_key="amdgpu.test",
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test_descriptor_set",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=1,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=descriptors,
    )


def _memory_descriptor(*, immediates: tuple[Immediate, ...]) -> Descriptor:
    return Descriptor(
        key="amdgpu.memory",
        mnemonic="memory",
        semantic_tag="memory.load.u32",
        operands=(),
        schedule_class="amdgpu.vmem.load",
        immediates=immediates,
        effects=(Effect(EffectKind.READ, memory_space=MemorySpace.GLOBAL),),
    )


def _storage_lease_signature(
    descriptor: Descriptor,
) -> tuple[
    tuple[
        StorageLeaseKind,
        StorageLeaseAttachment,
        int,
        int,
        int,
        str,
        tuple[StorageLeaseFlag, ...],
    ],
    ...,
]:
    return tuple(
        (
            lease.kind,
            lease.attachment,
            lease.attachment_index,
            lease.unit_count,
            lease.release_class_id,
            lease.release_reason_name,
            lease.flags,
        )
        for lease in descriptor.storage_leases
    )


def test_storage_lease_rows_project_memory_dependencies() -> None:
    schedule_class = ScheduleClass(
        name="amdgpu.test.memory",
        latency_kind=LatencyKind.VARIABLE,
        model_quality=ModelQuality.EXACT,
        hazards=(
            Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_LOAD),
            Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_STORE),
        ),
    )
    descriptor = Descriptor(
        key="amdgpu.test.memory",
        mnemonic="test_memory",
        semantic_tag="memory.global.atomic.u32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt(_REG_VGPR),), unit_count=2),
            Operand("vaddr", OperandRole.OPERAND, (RegClassAlt(_REG_VGPR),)),
            Operand(
                "saddr",
                OperandRole.RESOURCE,
                (RegClassAlt(_REG_SGPR),),
                unit_count=4,
                latency_sensitive_resource=True,
            ),
            Operand("soffset", OperandRole.OPERAND, (RegClassAlt(_REG_SGPR),)),
            Operand(
                "exec",
                OperandRole.IMPLICIT,
                (RegClassAlt(_REG_SGPR),),
                unit_count=2,
            ),
        ),
        schedule_class=schedule_class.name,
        effects=(
            Effect(
                EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                flags=(EffectFlag.DEPENDENCY,),
            ),
            Effect(
                EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
                flags=(EffectFlag.DEPENDENCY,),
            ),
        ),
    )
    descriptor_set = replace(
        _descriptor_set(descriptor), schedule_classes=(schedule_class,)
    )

    descriptor = _with_storage_lease_rows(descriptor_set).descriptors[0]

    pressure_flags = (
        StorageLeaseFlag.STARTS_AT_ISSUE,
        StorageLeaseFlag.RELEASE_BEFORE_BOUNDARY,
        StorageLeaseFlag.RELEASE_FOR_PRESSURE,
    )
    source_flags = (
        StorageLeaseFlag.STARTS_AT_ISSUE,
        StorageLeaseFlag.MAY_CARRY_ACROSS_BOUNDARY,
    )
    latency_source_flags = (
        *source_flags,
        StorageLeaseFlag.PRESERVE_FOR_LATENCY,
    )
    assert _storage_lease_signature(descriptor) == (
        (
            StorageLeaseKind.RESULT_WRITE,
            StorageLeaseAttachment.RESULT,
            0,
            2,
            _COUNTER_VMEM_LOAD,
            "amdgpu.read_result_reuse",
            pressure_flags,
        ),
        (
            StorageLeaseKind.SOURCE_READ,
            StorageLeaseAttachment.OPERAND,
            0,
            1,
            _COUNTER_VMEM_STORE,
            "amdgpu.store_source_reuse",
            source_flags,
        ),
        (
            StorageLeaseKind.SOURCE_READ,
            StorageLeaseAttachment.OPERAND,
            1,
            4,
            _COUNTER_VMEM_LOAD,
            "amdgpu.memory_source_reuse",
            latency_source_flags,
        ),
        (
            StorageLeaseKind.SOURCE_READ,
            StorageLeaseAttachment.OPERAND,
            1,
            4,
            _COUNTER_VMEM_STORE,
            "amdgpu.memory_source_reuse",
            latency_source_flags,
        ),
        (
            StorageLeaseKind.SOURCE_READ,
            StorageLeaseAttachment.OPERAND,
            2,
            1,
            _COUNTER_VMEM_LOAD,
            "amdgpu.memory_source_reuse",
            latency_source_flags,
        ),
        (
            StorageLeaseKind.SOURCE_READ,
            StorageLeaseAttachment.OPERAND,
            2,
            1,
            _COUNTER_VMEM_STORE,
            "amdgpu.memory_source_reuse",
            latency_source_flags,
        ),
    )


def test_storage_lease_rows_project_xcnt_over_packet_inputs() -> None:
    schedule_class = ScheduleClass(
        name=_SCHEDULE_VMEM_LOAD,
        latency_kind=LatencyKind.VARIABLE,
        model_quality=ModelQuality.EXACT,
    )
    descriptor = Descriptor(
        key="amdgpu.test.xcnt",
        mnemonic="test_xcnt",
        semantic_tag="memory.global.load.u32",
        operands=(
            Operand("vaddr", OperandRole.OPERAND, (RegClassAlt(_REG_VGPR),)),
            Operand(
                "saddr",
                OperandRole.RESOURCE,
                (RegClassAlt(_REG_SGPR),),
                unit_count=2,
            ),
            Operand(
                "exec",
                OperandRole.IMPLICIT,
                (RegClassAlt(_REG_SGPR),),
                unit_count=2,
            ),
        ),
        schedule_class=schedule_class.name,
    )
    descriptor_set = replace(
        _descriptor_set(descriptor), schedule_classes=(schedule_class,)
    )

    descriptor = _with_storage_lease_rows(
        descriptor_set, enable_gfx125x_xcnt=True
    ).descriptors[0]

    assert tuple(
        (
            lease.attachment_index,
            lease.unit_count,
            lease.release_class_name,
            lease.release_reason_name,
        )
        for lease in descriptor.storage_leases
    ) == (
        (0, 1, "amdgpu.x", "amdgpu.memory_source_reuse"),
        (1, 2, "amdgpu.x", "amdgpu.memory_source_reuse"),
    )


def test_storage_lease_rows_ignore_synchronous_lds_sources() -> None:
    schedule_class = ScheduleClass(
        name=_SCHEDULE_LDS_STORE,
        latency_kind=LatencyKind.VARIABLE,
        model_quality=ModelQuality.EXACT,
        hazards=(Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_LDS),),
    )
    descriptor = Descriptor(
        key="amdgpu.test.lds",
        mnemonic="test_lds",
        semantic_tag="memory.workgroup.store.u32",
        operands=(
            Operand("addr", OperandRole.OPERAND, (RegClassAlt(_REG_VGPR),)),
            Operand("value", OperandRole.OPERAND, (RegClassAlt(_REG_VGPR),)),
        ),
        schedule_class=schedule_class.name,
        effects=(
            Effect(
                EffectKind.WRITE,
                memory_space=MemorySpace.WORKGROUP,
                flags=(EffectFlag.DEPENDENCY,),
            ),
        ),
    )
    descriptor_set = replace(
        _descriptor_set(descriptor), schedule_classes=(schedule_class,)
    )

    descriptor = _with_storage_lease_rows(descriptor_set).descriptors[0]

    assert descriptor.storage_leases == ()


def test_instruction_classes_project_target_packet_families() -> None:
    global_load = Descriptor(
        key="amdgpu.global_load_b32",
        mnemonic="global_load_b32",
        semantic_tag="memory.global.load.u32",
        operands=(),
        schedule_class=_SCHEDULE_VMEM_LOAD,
        effects=(Effect(EffectKind.READ, memory_space=MemorySpace.GLOBAL),),
    )
    private_load = Descriptor(
        key="amdgpu.scratch_load_b32",
        mnemonic="scratch_load_b32",
        semantic_tag="memory.stack.load.u32",
        operands=(),
        schedule_class=_SCHEDULE_VMEM_LOAD,
        effects=(Effect(EffectKind.READ, memory_space=MemorySpace.GLOBAL),),
    )
    staged_load = Descriptor(
        key="amdgpu.global_load_lds_b32",
        mnemonic="global_load_lds_b32",
        semantic_tag="memory.global.load.u32",
        operands=(),
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        effects=(
            Effect(EffectKind.READ, memory_space=MemorySpace.GLOBAL),
            Effect(EffectKind.WRITE, memory_space=MemorySpace.WORKGROUP),
        ),
    )
    swmmac = Descriptor(
        key="amdgpu.v_swmmac_f32_16x16x64_f16",
        mnemonic="v_swmmac_f32_16x16x64_f16",
        semantic_tag="matrix.swmmac.f32.16x16x64.f16",
        operands=(),
        schedule_class=_SCHEDULE_SWMMAC,
    )
    gfx125x_wmma = Descriptor(
        key="amdgpu.v_wmma_f32_16x16x32_f16",
        mnemonic="v_wmma_f32_16x16x32_f16",
        semantic_tag="matrix.wmma.f32.16x16x32.f16",
        operands=(),
        schedule_class=f"{_SCHEDULE_MATRIX}.gfx125x.xdl",
    )
    gfx125x_swmmac = Descriptor(
        key="amdgpu.v_swmmac_f32_16x16x64_f16",
        mnemonic="v_swmmac_f32_16x16x64_f16",
        semantic_tag="matrix.swmmac.f32.16x16x64.f16",
        operands=(),
        schedule_class=f"{_SCHEDULE_MATRIX}.gfx125x.xdl",
    )
    descriptor_set = _with_instruction_classes(
        _descriptor_set(
            global_load,
            private_load,
            staged_load,
            swmmac,
            gfx125x_wmma,
            gfx125x_swmmac,
        )
    )

    assert descriptor_set.descriptors[0].instruction_classes == (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.GLOBAL_LOAD,
    )
    assert descriptor_set.descriptors[1].instruction_classes == ()
    assert descriptor_set.descriptors[2].instruction_classes == (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.GLOBAL_LOAD,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    )
    assert descriptor_set.descriptors[3].instruction_classes == (
        InstructionClass.SWMMAC,
    )
    assert descriptor_set.descriptors[4].instruction_classes == (InstructionClass.WMMA,)
    assert descriptor_set.descriptors[5].instruction_classes == (
        InstructionClass.SWMMAC,
    )


def _immediate_default(immediates: tuple[Immediate, ...], name: str) -> int:
    for immediate in immediates:
        if immediate.field_name == name:
            return immediate.default_value
    raise AssertionError(f"missing immediate '{name}'")


def _descriptor(key: str, semantic_tag: str) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=None,
        semantic_tag=semantic_tag,
        operands=(),
        schedule_class="amdgpu.test",
    )


def _dpp_descriptor(
    *,
    immediates: tuple[Immediate, ...] = (),
    encoding_field_values: tuple[EncodingFieldValue, ...] = (),
    encoding_format_id: int = AMDGPU_ENCODING_FORMAT_VOP1_DPP16,
) -> Descriptor:
    return Descriptor(
        key="amdgpu.test.dpp",
        mnemonic="v_test_dpp",
        semantic_tag="test.dpp",
        operands=(),
        immediates=immediates,
        encoding_field_values=encoding_field_values,
        encoding_format_id=encoding_format_id,
        schedule_class="amdgpu.test",
    )


def _isa_spec(*encoding_names: str) -> AmdgpuIsaSpec:
    return AmdgpuIsaSpec(
        source_name="test.xml",
        architecture_name="test",
        architecture_id=0,
        encodings=tuple(
            AmdgpuIsaEncoding(
                name=name,
                order=0,
                bit_count=64,
                identifier_mask=0,
                identifier_values=(0,),
                fields=(),
            )
            for name in encoding_names
        ),
        instructions=(),
        operand_types=(),
    )


def _expect_value_error_contains(
    expected_message: str, thunk: Callable[[], object]
) -> None:
    actual_message: str | None = None
    try:
        thunk()
    except ValueError as exc:
        actual_message = str(exc)
    assert actual_message is not None, "expected ValueError"
    assert expected_message in actual_message


def test_amdgpu_core_descriptor_set_rejects_lds_spill_slots() -> None:
    _expect_value_error_contains(
        "lane-private LDS storage contract",
        lambda: _amdgpu_core_descriptor_set(
            key="amdgpu.test.core",
            reg_classes=(RegClass("amdgpu.test", 32, SpillSlotSpace.LDS),),
            resources=(),
            schedule_classes=(),
        ),
    )


def test_descriptor_encoding_formats_must_exist_for_target() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.literal",
        mnemonic="test_literal",
        semantic_tag="test.literal",
        operands=(),
        schedule_class="amdgpu.test",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
    )
    _expect_value_error_contains(
        "uses unavailable encoding format 'VOP3_INST_LITERAL'",
        lambda: _validate_descriptor_encoding_formats(
            "cdna3", _isa_spec("ENC_VOP3"), _descriptor_set(descriptor)
        ),
    )


def test_descriptor_encoding_formats_accept_target_supplements() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.vop3px2",
        mnemonic="test_vop3px2",
        semantic_tag="test.vop3px2",
        operands=(),
        schedule_class="amdgpu.test",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3PX2,
    )
    _validate_descriptor_encoding_formats(
        "rdna4", _isa_spec("ENC_VOP3"), _descriptor_set(descriptor)
    )


def test_execution_masked_descriptors_read_exec_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_add_u32",
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("lhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("rhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
        ),
        schedule_class=_SCHEDULE_VALU,
    )

    masked_descriptor = _with_execution_mask_state_read(descriptor)
    exec_operand = masked_descriptor.operands[-1]

    assert exec_operand.field_name == "exec_in"
    assert exec_operand.role is OperandRole.IMPLICIT
    assert exec_operand.reg_alts == (
        RegClassAlt(_REG_EXEC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in exec_operand.flags
    assert OperandFlag.STATE_READ in exec_operand.flags
    assert OperandFlag.SCHEDULE_ONLY_STATE in exec_operand.flags
    assert (
        _with_execution_mask_state_read(masked_descriptor).operands
        == masked_descriptor.operands
    )


def test_trans_descriptors_use_descriptor_specific_schedule_classes() -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _gfx11_core_overlays()
        if overlay.descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS
    }

    assert tuple(overlays) == _AMDGPU_TRANS_DESCRIPTOR_KEYS
    for descriptor_key, overlay in overlays.items():
        assert overlay.schedule_class == _amdgpu_trans_schedule_class_name(
            descriptor_key
        )
        assert overlay.schedule_class != _SCHEDULE_VALU

    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }
        for descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS:
            schedule_class = schedule_classes[
                _amdgpu_trans_schedule_class_name(descriptor_key)
            ]
            assert schedule_class.latency_kind is LatencyKind.ESTIMATE
            assert schedule_class.latency_cycles == _AMDGPU_TRANS_PROXY_LATENCY_CYCLES


def test_gfx9_4_matrix_schedule_classes_match_member_timings() -> None:
    schedule_cases = (
        (
            _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
            _AMDGPU_GFX942_MATRIX_TIMINGS,
            _gfx940_core_overlays(),
        ),
        (
            _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
            _AMDGPU_GFX950_MATRIX_TIMINGS,
            _gfx950_core_overlays(),
        ),
        (
            _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
            _AMDGPU_GFX9_4_GENERIC_MATRIX_TIMINGS,
            _gfx9_4_generic_core_overlays(),
        ),
    )
    for descriptor_set, timings, overlays in schedule_cases:
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }
        matrix_overlays = {
            overlay.descriptor_key: overlay
            for overlay in overlays
            if overlay.semantic_tag is not None
            and overlay.semantic_tag.startswith("matrix.")
        }
        assert set(matrix_overlays) == set(timings)
        for descriptor_key, timing in timings.items():
            schedule_class = schedule_classes[
                _amdgpu_mfma_schedule_class_name(descriptor_key)
            ]
            assert schedule_class.latency_kind is LatencyKind.ESTIMATE
            assert schedule_class.latency_cycles == timing.latency_cycles
            assert schedule_class.issue_uses == (
                IssueUse(
                    _RESOURCE_MFMA,
                    cycles=timing.reciprocal_throughput_cycles,
                    units=1,
                ),
            )
            overlay = matrix_overlays[descriptor_key]
            assert overlay.schedule_class == _amdgpu_mfma_schedule_class_name(
                descriptor_key
            )
            assert overlay.schedule_class.startswith(_SCHEDULE_MFMA_QUALIFIED_PREFIX)

    differing_member_timings = {
        descriptor_key
        for descriptor_key in (
            _AMDGPU_GFX942_MATRIX_TIMINGS.keys() & _AMDGPU_GFX950_MATRIX_TIMINGS.keys()
        )
        if _AMDGPU_GFX942_MATRIX_TIMINGS[descriptor_key]
        != _AMDGPU_GFX950_MATRIX_TIMINGS[descriptor_key]
    }
    assert differing_member_timings == {"amdgpu.v_mfma_f64_16x16x4_f64"}
    for descriptor_key, generic_timing in _AMDGPU_GFX9_4_GENERIC_MATRIX_TIMINGS.items():
        gfx942_timing = _AMDGPU_GFX942_MATRIX_TIMINGS[descriptor_key]
        gfx950_timing = _AMDGPU_GFX950_MATRIX_TIMINGS[descriptor_key]
        assert generic_timing.latency_cycles == max(
            gfx942_timing.latency_cycles,
            gfx950_timing.latency_cycles,
        )
        assert generic_timing.reciprocal_throughput_cycles == max(
            gfx942_timing.reciprocal_throughput_cycles,
            gfx950_timing.reciprocal_throughput_cycles,
        )


def test_gfx950_f8f6f4_mfma_descriptors_model_physical_abis() -> None:
    overlays = {overlay.descriptor_key: overlay for overlay in _gfx950_core_overlays()}
    rows = (
        ("v_mfma_f32_16x16x128_f8f6f4", 4, False),
        ("v_mfma_f32_32x32x64_f8f6f4", 16, False),
        ("v_mfma_scale_f32_16x16x128_f8f6f4", 4, True),
        ("v_mfma_scale_f32_32x32x64_f8f6f4", 16, True),
    )
    for mnemonic, accumulator_units, has_scale_operands in rows:
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
            for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
                key = f"amdgpu.{mnemonic}_{lhs_format.token}_{rhs_format.token}"
                overlay = overlays[key]
                assert tuple(
                    operand.descriptor_operand.unit_count
                    for operand in overlay.operands
                ) == (
                    accumulator_units,
                    lhs_format.register_count_for(32),
                    rhs_format.register_count_for(32),
                    accumulator_units,
                    *((1, 1) if has_scale_operands else ()),
                )
                assert overlay.immediate_fields == ("CBSZ", "BLGP")
                assert tuple(
                    immediate.enum_domain for immediate in overlay.immediates
                ) == (
                    AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES[lhs_format.token],
                    AMDGPU_CDNA4_MATRIX_FORMAT_ENUM_DOMAIN_NAMES[rhs_format.token],
                )
                form = overlay.asm_forms[0]
                assert form.native_assembly_mnemonic == mnemonic
                assert tuple(
                    value.literal for value in form.native_assembly_values[-2:]
                ) == ("cbsz", "blgp")
                if has_scale_operands:
                    assert overlay.fixed_encoding_fields == (
                        ("ABID", 1),
                        ("ENCODING", 0x1A7),
                        ("X2ENCODING", 0xD3AC),
                    )


def test_gfx11_wmma_separates_hardware_latency_from_schedule_distance() -> None:
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in (_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE.schedule_classes)
    }
    schedule_class = schedule_classes[_SCHEDULE_WMMA]
    assert schedule_class.latency_kind is LatencyKind.ESTIMATE
    assert schedule_class.latency_cycles == 5
    assert schedule_class.schedule_distance_cycles == 32


def test_gfx12_matrix_schedule_classes_match_processor_model() -> None:
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes
    }
    for schedule_class_name, resource_name in (
        (_SCHEDULE_WMMA, _RESOURCE_WMMA),
        (_SCHEDULE_SWMMAC, _RESOURCE_SWMMAC),
    ):
        schedule_class = schedule_classes[schedule_class_name]
        assert schedule_class.latency_kind is LatencyKind.EXACT
        assert schedule_class.latency_cycles == 5
        assert schedule_class.issue_uses == (
            IssueUse(resource_name, cycles=1, units=1),
        )
        assert schedule_class.model_quality is ModelQuality.EXACT

    matrix_overlays = tuple(
        overlay
        for overlay in _gfx12_core_overlays()
        if (overlay.semantic_tag or "").startswith("matrix.")
    )
    assert {overlay.schedule_class for overlay in matrix_overlays} == {
        _SCHEDULE_WMMA,
        _SCHEDULE_SWMMAC,
    }


def test_trans_schedule_classes_accept_descriptor_latency_overrides() -> None:
    descriptor_key = "amdgpu.v_rcp_f32"
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in _amdgpu_trans_schedule_classes(
            latency_cycles_by_descriptor_key={descriptor_key: 5}
        )
    }

    override_schedule_class = schedule_classes[
        _amdgpu_trans_schedule_class_name(descriptor_key)
    ]
    assert override_schedule_class.latency_cycles == 5
    assert (
        schedule_classes[
            _amdgpu_trans_schedule_class_name("amdgpu.v_exp_f32")
        ].latency_cycles
        == _AMDGPU_TRANS_PROXY_LATENCY_CYCLES
    )
    _expect_value_error_contains(
        "unknown descriptor",
        lambda: _amdgpu_trans_schedule_classes(
            latency_cycles_by_descriptor_key={"amdgpu.v_bad_f32": 4}
        ),
    )


def test_packed_dot_schedule_class_models_valu_latency() -> None:
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }

        schedule_class = schedule_classes[_SCHEDULE_PACKED_DOT]
        assert schedule_class.latency_kind is LatencyKind.ESTIMATE
        assert schedule_class.latency_cycles == 5
        assert tuple(schedule_class.issue_uses) == (
            IssueUse(_RESOURCE_VALU, cycles=1, units=1),
        )


def test_scalar_compare_schedule_class_models_scc_branch_latency() -> None:
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }

        schedule_class = schedule_classes[_SCHEDULE_SALU_COMPARE]
        assert schedule_class.latency_kind is LatencyKind.ESTIMATE
        assert schedule_class.latency_cycles == 2
        assert tuple(schedule_class.issue_uses) == (
            IssueUse(_RESOURCE_SALU, cycles=1, units=1),
        )


def test_scalar_compare_descriptors_use_scc_branch_schedule_class() -> None:
    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key, descriptor in descriptors.items():
            if descriptor_key.startswith("amdgpu.s_cmp_"):
                assert descriptor.schedule_class == _SCHEDULE_SALU_COMPARE


def test_packed_dot_descriptors_use_packed_dot_schedule_class() -> None:
    overlay_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    )
    dot_descriptor_count = 0
    for overlays in overlay_sets:
        dot_descriptors = tuple(
            overlay
            for overlay in overlays
            if overlay.semantic_tag is not None
            and overlay.semantic_tag.startswith("dot.")
        )

        dot_descriptor_count += len(dot_descriptors)
        for overlay in dot_descriptors:
            assert overlay.schedule_class == _SCHEDULE_PACKED_DOT
    assert dot_descriptor_count != 0


def test_rdna4_packed_fp8_dots_pin_no_op_modifiers() -> None:
    descriptors = {
        overlay.descriptor_key: overlay for overlay in _gfx12_core_overlays()
    }
    for lhs_type in ("fp8", "bf8"):
        for rhs_type in ("fp8", "bf8"):
            descriptor_key = f"amdgpu.v_dot4_f32_{lhs_type}_{rhs_type}"
            assert descriptors[descriptor_key].fixed_encoding_fields == (
                ("OPSEL_HI", 0x7),
            )


def test_mode_control_schedule_class_covers_generated_descriptors() -> None:
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }
        schedule_class = schedule_classes[_SCHEDULE_MODE_CONTROL]

        assert ScheduleClassFlag.CONTROL in schedule_class.flags


def test_trans_descriptors_read_exec_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_exp_f32",
        mnemonic="v_exp_f32",
        semantic_tag="float.exp2.f32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("input", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
        ),
        schedule_class=_amdgpu_trans_schedule_class_name("amdgpu.v_exp_f32"),
    )

    masked_descriptor = _with_execution_mask_state_read(descriptor)
    exec_operand = masked_descriptor.operands[-1]

    assert exec_operand.field_name == "exec_in"
    assert exec_operand.role is OperandRole.IMPLICIT
    assert exec_operand.reg_alts == (
        RegClassAlt(_REG_EXEC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in exec_operand.flags
    assert OperandFlag.STATE_READ in exec_operand.flags
    assert OperandFlag.SCHEDULE_ONLY_STATE in exec_operand.flags


def test_div_fmas_low_asm_preserves_vcc_scale_mask_operand() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_div_fmas_f32"]
        assert len(descriptor.implicit_operands) == 1
        scale_mask_operand = descriptor.implicit_operands[0].descriptor_operand
        assert scale_mask_operand is not None
        assert scale_mask_operand.field_name == "scale_mask"
        assert scale_mask_operand.role is OperandRole.PREDICATE
        assert scale_mask_operand.reg_alts == (
            RegClassAlt(_REG_VCC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
        )
        assert OperandFlag.IMPLICIT in scale_mask_operand.flags
        assert OperandFlag.STATE_READ in scale_mask_operand.flags
        assert scale_mask_operand.unit_count == 1

        form = descriptor.asm_forms[0]
        assert form.results == ("dst",)
        assert form.operands == ("a", "b", "c", "scale_mask")


def test_div_scale_low_asm_writes_architectural_vcc_scale_mask() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_div_scale_f32"]

        assert len(descriptor.ignored_operands) == 1
        ignored_operand = descriptor.ignored_operands[0]
        assert ignored_operand.xml_field_name == "SDST"
        assert ignored_operand.ignore_reason == "fixed-architectural-vcc-scale-mask"
        assert ignored_operand.fixed_encoding_value == _predefined("VCC_LO", "OPR_SDST")

        assert len(descriptor.implicit_operands) == 1
        scale_mask_operand = descriptor.implicit_operands[0].descriptor_operand
        assert scale_mask_operand is not None
        assert scale_mask_operand.field_name == "mask"
        assert scale_mask_operand.role is OperandRole.RESULT
        assert scale_mask_operand.reg_alts == (
            RegClassAlt(_REG_VCC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
        )
        assert OperandFlag.IMPLICIT in scale_mask_operand.flags
        assert OperandFlag.STATE_WRITE in scale_mask_operand.flags
        assert scale_mask_operand.unit_count == 1

        form = descriptor.asm_forms[0]
        assert form.results == ("dst", "mask")
        assert form.operands == ("value", "denominator", "numerator")


def test_scalar_scc_compare_results_are_rematerializable() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        compare_keys = tuple(
            descriptor_key
            for descriptor_key in descriptors
            if descriptor_key.startswith("amdgpu.s_cmp_")
        )
        assert compare_keys
        for descriptor_key in compare_keys:
            descriptor = descriptors[descriptor_key]
            assert (
                Constraint(ConstraintKind.REMATERIALIZABLE, lhs_operand_index=0)
                in descriptor.constraints
            )
            assert len(descriptor.implicit_operands) == 1
            scc_operand = descriptor.implicit_operands[0].descriptor_operand
            assert scc_operand is not None
            assert scc_operand.field_name == "scc"
            assert scc_operand.role is OperandRole.RESULT
            assert OperandFlag.IMPLICIT in scc_operand.flags
            assert OperandFlag.STATE_WRITE in scc_operand.flags
            assert OperandFlag.STATE_READ not in scc_operand.flags
            assert OperandFlag.SCHEDULE_ONLY_STATE not in scc_operand.flags

        assert not any(
            constraint.kind is ConstraintKind.REMATERIALIZABLE
            for constraint in descriptors["amdgpu.s_and_saveexec_b64"].constraints
        )


def test_v_mov_b32_literal_results_are_rematerializable() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_mov_b32"]
        assert (
            Constraint(ConstraintKind.REMATERIALIZABLE, lhs_operand_index=0)
            in descriptor.constraints
        )


def test_pure_vop2_f32_results_are_rematerializable() -> None:
    operations = ("add", "sub", "subrev", "mul", "min", "max")
    suffixes = ("", ".lit", ".src0_inline")
    rematerializable_result = Constraint(
        ConstraintKind.REMATERIALIZABLE,
        lhs_operand_index=0,
    )
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        matching_keys = []
        for operation in operations:
            for suffix in suffixes:
                descriptor_key = f"amdgpu.v_{operation}_f32{suffix}"
                descriptor = descriptors.get(descriptor_key)
                if descriptor is None:
                    continue
                matching_keys.append(descriptor_key)
                assert rematerializable_result in descriptor.constraints
        assert matching_keys

        for descriptor_key in descriptors:
            if ".dpp" not in descriptor_key:
                continue
            descriptor = descriptors[descriptor_key]
            assert not any(
                constraint.kind is ConstraintKind.REMATERIALIZABLE
                for constraint in descriptor.constraints
            )


def test_pure_integer_valu_results_are_rematerializable() -> None:
    descriptor_keys = (
        "amdgpu.v_add_u32",
        "amdgpu.v_add_u32.src0_inline",
        "amdgpu.v_add_u32.lit",
        "amdgpu.v_add3_u32",
        "amdgpu.v_sub_u32",
        "amdgpu.v_mul_lo_u32",
        "amdgpu.v_mul_hi_u32",
        "amdgpu.v_mul_u32_u24",
        "amdgpu.v_mul_u32_u24.src0_inline",
        "amdgpu.v_mul_u32_u24.lit",
        "amdgpu.v_mad_u32_u24",
        "amdgpu.v_min_i32",
        "amdgpu.v_max_i32",
        "amdgpu.v_min_u32",
        "amdgpu.v_max_u32",
        "amdgpu.v_and_b32",
        "amdgpu.v_and_b32.src0_inline",
        "amdgpu.v_and_b32.lit",
        "amdgpu.v_or_b32",
        "amdgpu.v_or_b32.src0_inline",
        "amdgpu.v_or_b32.lit",
        "amdgpu.v_xor_b32",
        "amdgpu.v_xor_b32.src0_inline",
        "amdgpu.v_xor_b32.lit",
        "amdgpu.v_lshlrev_b32",
        "amdgpu.v_lshlrev_b32.src0_inline",
        "amdgpu.v_lshlrev_b32.lit",
        "amdgpu.v_lshlrev_b32.vop3_imm",
        "amdgpu.v_lshl_add_u32.shift_imm",
        "amdgpu.v_bfe_u32.offset_width_inline",
        "amdgpu.v_bfe_i32.offset_width_inline",
        "amdgpu.v_bfi_b32",
        "amdgpu.v_lshrrev_b32",
        "amdgpu.v_lshrrev_b32.src0_inline",
        "amdgpu.v_lshrrev_b32.lit",
        "amdgpu.v_ashrrev_i32",
        "amdgpu.v_ashrrev_i32.src0_inline",
        "amdgpu.v_ashrrev_i32.lit",
    )
    optional_descriptor_keys = (
        "amdgpu.v_add3_u32.src0_lit",
        "amdgpu.v_add3_u32.src1_lit",
        "amdgpu.v_add3_u32.src2_lit",
        "amdgpu.v_mad_u32_u24.src0_lit",
        "amdgpu.v_mad_u32_u24.src1_lit",
        "amdgpu.v_mad_u32_u24.src2_lit",
        "amdgpu.v_lshl_add_u32.shift_imm.src2_lit",
        "amdgpu.v_bfi_b32.src0_lit",
    )
    excluded_descriptor_keys = (
        "amdgpu.v_add_co_u32",
        "amdgpu.v_add_co_ci_u32",
        "amdgpu.v_sub_co_u32",
        "amdgpu.v_sub_co_ci_u32",
        "amdgpu.v_lshlrev_b32.src0_16_low16",
        "amdgpu.v_lshlrev_b64",
        "amdgpu.v_bfe_u32.offset_0_width_16_low16",
        "amdgpu.v_permlanex16_b32.src12_inline",
    )
    rematerializable_result = Constraint(
        ConstraintKind.REMATERIALIZABLE,
        lhs_operand_index=0,
    )
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            assert descriptor is not None, descriptor_key
            assert rematerializable_result in descriptor.constraints, descriptor_key
        for descriptor_key in optional_descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            if descriptor is None:
                continue
            assert rematerializable_result in descriptor.constraints, descriptor_key

        for descriptor_key in excluded_descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            if descriptor is None:
                continue
            assert not any(
                constraint.kind is ConstraintKind.REMATERIALIZABLE
                for constraint in descriptor.constraints
            )


def test_integer_binary_src0_accepts_scalar_or_vector_registers() -> None:
    descriptor_keys = (
        "amdgpu.v_mul_lo_u32",
        "amdgpu.v_mul_hi_u32",
        "amdgpu.v_mul_u32_u24",
        "amdgpu.v_min_i32",
        "amdgpu.v_max_i32",
        "amdgpu.v_min_u32",
        "amdgpu.v_max_u32",
        "amdgpu.v_and_b32",
        "amdgpu.v_or_b32",
        "amdgpu.v_xor_b32",
        "amdgpu.v_lshlrev_b32",
        "amdgpu.v_lshrrev_b32",
        "amdgpu.v_ashrrev_i32",
    )
    expected_alternatives = (RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR))
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in descriptor_keys:
            lhs = descriptors[descriptor_key].operands[1].descriptor_operand
            assert lhs.reg_alts == expected_alternatives


def test_full_width_conversion_results_are_rematerializable() -> None:
    descriptor_keys = (
        "amdgpu.v_cvt_f32_i32",
        "amdgpu.v_cvt_i32_f32",
        "amdgpu.v_cvt_f32_f16",
        "amdgpu.v_cvt_f32_u32",
        "amdgpu.v_cvt_f32_ubyte0",
        "amdgpu.v_cvt_f32_ubyte1",
        "amdgpu.v_cvt_f32_ubyte2",
        "amdgpu.v_cvt_f32_ubyte3",
        "amdgpu.v_cvt_u32_f32",
    )
    excluded_descriptor_keys = (
        "amdgpu.v_cvt_f16_f32",
        "amdgpu.v_cvt_f32_fp8",
        "amdgpu.v_cvt_f32_bf8",
        "amdgpu.v_cvt_pk_f32_fp8",
        "amdgpu.v_cvt_pk_f32_bf8",
        "amdgpu.v_cvt_pk_f16_fp8",
        "amdgpu.v_cvt_pk_f16_bf8",
        "amdgpu.v_cvt_pk_u16_u32",
        "amdgpu.v_cvt_pk_bf16_f32",
    )
    rematerializable_result = Constraint(
        ConstraintKind.REMATERIALIZABLE,
        lhs_operand_index=0,
    )
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            assert descriptor is not None, descriptor_key
            assert rematerializable_result in descriptor.constraints, descriptor_key

        for descriptor_key in excluded_descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            if descriptor is None:
                continue
            assert not any(
                constraint.kind is ConstraintKind.REMATERIALIZABLE
                for constraint in descriptor.constraints
            ), descriptor_key


def test_unsigned_byte_conversion_family_is_available_on_all_targets() -> None:
    descriptor_keys = {
        "amdgpu.v_cvt_f32_ubyte0",
        "amdgpu.v_cvt_f32_ubyte1",
        "amdgpu.v_cvt_f32_ubyte2",
        "amdgpu.v_cvt_f32_ubyte3",
    }
    for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items():
        available_keys = {
            descriptor.descriptor_key for descriptor in builder.overlay_rows()
        }
        assert descriptor_keys <= available_keys, (
            target,
            descriptor_keys - available_keys,
        )


def test_float_to_packed_byte_primitives_are_available_on_all_targets() -> None:
    descriptor_keys = {
        "amdgpu.v_cvt_i32_f32",
        "amdgpu.v_cvt_u32_f32",
        "amdgpu.v_perm_b32",
    }
    for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items():
        available_keys = {
            descriptor.descriptor_key for descriptor in builder.overlay_rows()
        }
        assert descriptor_keys <= available_keys, (
            target,
            descriptor_keys - available_keys,
        )


def test_scc_free_scalar_integer_results_are_rematerializable() -> None:
    descriptor_keys = (
        "amdgpu.s_mul_i32",
        "amdgpu.s_mul_i32.rhs_inline",
        "amdgpu.s_mul_hi_u32",
    )
    excluded_descriptor_keys = (
        "amdgpu.s_add_u32",
        "amdgpu.s_add_u32.rhs_inline",
        "amdgpu.s_addk_i32",
        "amdgpu.s_sub_u32",
        "amdgpu.s_mulk_i32",
        "amdgpu.s_and_b32",
        "amdgpu.s_and_b64",
        "amdgpu.s_lshl_b32",
        "amdgpu.s_lshl_b32.rhs_inline",
        "amdgpu.s_lshl_b64",
        "amdgpu.s_bfe_u32",
        "amdgpu.s_bfe_u32.lit",
        "amdgpu.s_cselect_b32",
    )
    rematerializable_result = Constraint(
        ConstraintKind.REMATERIALIZABLE,
        lhs_operand_index=0,
    )
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            assert descriptor is not None, descriptor_key
            assert rematerializable_result in descriptor.constraints, descriptor_key

        for descriptor_key in excluded_descriptor_keys:
            descriptor = descriptors.get(descriptor_key)
            if descriptor is None:
                continue
            assert not any(
                constraint.kind is ConstraintKind.REMATERIALIZABLE
                for constraint in descriptor.constraints
            ), descriptor_key


def test_rdna_f16_to_f32_convert_uses_wide_encoding() -> None:
    for overlays in (
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_cvt_f32_f16"]
        assert descriptor.encoding_name == "ENC_VOP3"


def test_cdna_f16_to_f32_convert_keeps_compact_encoding() -> None:
    for overlays in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_cvt_f32_f16"]
        assert descriptor.encoding_name == "ENC_VOP1"


def test_f32_to_f16_convert_results_use_d16_low_window() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        descriptor = descriptors["amdgpu.v_cvt_f16_f32"]
        result = descriptor.operands[0].descriptor_operand
        assert result.register_part == _REG_PART_VGPR_LOW16
        assert result.address_map_kind is OperandAddressMapKind.LOW_SUBSET
        assert result.addressable_unit_count == (
            _D16_PARTIAL_REGISTER_ADDRESSABLE_UNIT_COUNT
        )


def test_gfx11_wmma_wave64_asm_forms_keep_native_mnemonics_unsuffixed() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    cases = (
        (
            "amdgpu.v_wmma_f32_16x16x16_f16.w64",
            "v_wmma_f32_16x16x16_f16_w64",
            "v_wmma_f32_16x16x16_f16",
        ),
        (
            "amdgpu.v_wmma_f32_16x16x16_bf16.w64",
            "v_wmma_f32_16x16x16_bf16_w64",
            "v_wmma_f32_16x16x16_bf16",
        ),
        (
            "amdgpu.v_wmma_f16_16x16x16_f16.w64",
            "v_wmma_f16_16x16x16_f16_w64",
            "v_wmma_f16_16x16x16_f16",
        ),
        (
            "amdgpu.v_wmma_bf16_16x16x16_bf16.w64",
            "v_wmma_bf16_16x16x16_bf16_w64",
            "v_wmma_bf16_16x16x16_bf16",
        ),
        (
            "amdgpu.v_wmma_i32_16x16x16_iu8.w64",
            "v_wmma_i32_16x16x16_iu8_w64",
            "v_wmma_i32_16x16x16_iu8",
        ),
        (
            "amdgpu.v_wmma_i32_16x16x16_iu4.w64",
            "v_wmma_i32_16x16x16_iu4_w64",
            "v_wmma_i32_16x16x16_iu4",
        ),
    )

    for descriptor_key, low_mnemonic, native_mnemonic in cases:
        descriptor = descriptors[descriptor_key]
        assert descriptor.mnemonic == low_mnemonic
        assert descriptor.asm_forms is not None
        assert len(descriptor.asm_forms) == 1
        form = descriptor.asm_forms[0]
        assert form.mnemonic is None
        assert form.native_assembly_mnemonic == native_mnemonic
        assert form.results == ("dst",)
        assert form.operands == ("a", "b", "acc")

        zero_descriptor = descriptors[f"{descriptor_key}.acc_zero"]
        assert tuple(constraint.kind for constraint in zero_descriptor.constraints) == (
            ConstraintKind.EARLY_CLOBBER,
        )
        assert tuple(
            constraint.lhs_operand_index for constraint in zero_descriptor.constraints
        ) == (0,)
        zero_form = zero_descriptor.asm_forms[0]
        assert zero_form.mnemonic == f"{low_mnemonic}_acc_zero"
        assert zero_form.native_assembly_mnemonic == native_mnemonic
        expected_native_values = (
            NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
            NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="a"),
            NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="b"),
            NativeAsmValue(NativeAsmValueKind.LITERAL, literal="0"),
        )
        if ".v_wmma_i32_" in descriptor_key:
            expected_native_values = (
                *expected_native_values,
                NativeAsmValue(
                    NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                    field_name="neg_lo",
                    literal="neg_lo",
                    bit_width=3,
                    target_format_id=AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST,
                ),
                NativeAsmValue(
                    NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                    field_name="neg_hi",
                    literal="neg_hi",
                    bit_width=3,
                    target_format_id=AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_BIT_LIST,
                ),
                NativeAsmValue(
                    NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                    field_name="clamp",
                    literal="clamp",
                    target_format_id=AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_NAMED_FLAG,
                ),
            )
        assert zero_form.native_assembly_values == expected_native_values


def test_rdna_matrix_descriptors_pin_canonical_high_half_selectors() -> None:
    for overlays, field_name in (
        (_gfx11_core_overlays(), "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL_HI"),
    ):
        matrix_descriptors = tuple(
            descriptor
            for descriptor in overlays
            if descriptor.semantic_tag.startswith("matrix.wmma.")
            or descriptor.semantic_tag.startswith("matrix.swmmac.")
        )
        assert matrix_descriptors
        for descriptor in matrix_descriptors:
            assert (field_name, 0x7) in descriptor.fixed_encoding_fields


def test_wmma_zero_accumulator_asm_forms_print_native_base_mnemonic() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    zero_descriptor = descriptors["amdgpu.v_wmma_f32_16x16x16_f16.acc_zero"]
    zero_form = zero_descriptor.asm_forms[0]

    assert zero_form.mnemonic == "v_wmma_f32_16x16x16_f16_acc_zero"
    assert zero_form.native_assembly_mnemonic == "v_wmma_f32_16x16x16_f16"
    assert zero_form.native_assembly_values == (
        NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
        NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="a"),
        NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="b"),
        NativeAsmValue(NativeAsmValueKind.LITERAL, literal="0"),
    )


def test_scalar_descriptors_do_not_get_execution_mask_state_read() -> None:
    descriptor = Descriptor(
        key="amdgpu.s_add_u32",
        mnemonic="s_add_u32",
        semantic_tag="integer.add.u32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.sgpr"),)),
            Operand("lhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.sgpr"),)),
            Operand("rhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.sgpr"),)),
        ),
        schedule_class=_SCHEDULE_SALU,
    )

    assert _with_execution_mask_state_read(descriptor) is descriptor


def test_gfx125x_register_classes_expose_mode_and_high_vgprs() -> None:
    reg_classes = {reg_class.name: reg_class for reg_class in _gfx125x_reg_classes()}

    assert reg_classes["amdgpu.vgpr"].allocatable_count == 1024
    mode = reg_classes[_REG_MODE]
    assert mode.alloc_unit_bits == 32
    assert mode.allocatable_count == 1
    assert RegClassFlag.PHYSICAL in mode.flags
    assert RegClassFlag.UNSPILLABLE in mode.flags


def test_s_set_vgpr_msb_writes_mode_state() -> None:
    descriptor = _s_set_vgpr_msb_descriptor()

    assert descriptor.key == "amdgpu.s_set_vgpr_msb"
    assert descriptor.mnemonic == "s_set_vgpr_msb"
    assert descriptor.schedule_class == _SCHEDULE_MODE_CONTROL
    assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
        "mode",
    )
    mode_operand = descriptor.operands[0]
    assert mode_operand.field_name == "mode"
    assert mode_operand.role is OperandRole.IMPLICIT
    assert mode_operand.reg_alts == (
        RegClassAlt(_REG_MODE, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in mode_operand.flags
    assert OperandFlag.STATE_WRITE in mode_operand.flags


def test_s_delay_alu_descriptor_uses_native_packed_immediate() -> None:
    descriptor = _s_delay_alu_descriptor()

    assert descriptor.key == "amdgpu.s_delay_alu"
    assert descriptor.mnemonic == "s_delay_alu"
    assert descriptor.schedule_class == _SCHEDULE_WAIT_ALU
    assert descriptor.effects == ()
    assert DescriptorFlag.SIDE_EFFECTING in descriptor.flags
    assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
        "delay",
    )
    delay = descriptor.immediates[0]
    assert delay.kind is ImmediateKind.UNSIGNED
    assert delay.bit_width == 16
    assert delay.unsigned_max == 0x07FF
    assert descriptor.asm_forms[0].native_assembly_values == (
        NativeAsmValue(
            NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
            field_name="delay",
            target_format_id=AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_DELAY_ALU,
        ),
    )


def test_s_delay_alu_descriptor_is_exposed_on_rdna_families() -> None:
    targets_with_delay_alu = {
        target
        for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items()
        if any(
            descriptor.key == "amdgpu.s_delay_alu"
            for descriptor in builder.extra_descriptors
        )
    }

    assert targets_with_delay_alu == {
        "gfx11_generic",
        "gfx12_generic",
        "gfx12_5_generic",
        "rdna3",
        "rdna3_5",
        "rdna4m",
        "rdna4",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
    }
    assert "amdgpu.s_delay_alu" in amdgpu_descriptor_ref_keys()


def test_gfx125x_cluster_flat_rank_descriptor_matches_compiler_abi() -> None:
    overlays = {overlay.descriptor_key: overlay for overlay in _gfx125x_core_overlays()}
    descriptor = overlays["amdgpu.s_getreg_b32.cluster_workgroup_flat_id"]

    assert descriptor.instruction_name == "S_GETREG_B32"
    assert descriptor.encoding_name == "ENC_SOPK"
    assert descriptor.schedule_class == _SCHEDULE_SALU
    assert descriptor.ignored_operands[0].xml_field_name == "SIMM16"
    assert descriptor.ignored_operands[0].fixed_encoding_value == 0x1D5C
    assert descriptor.asm_forms[0].native_assembly_values == (
        NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
        NativeAsmValue(
            NativeAsmValueKind.LITERAL,
            literal="hwreg(HW_REG_IB_STS2, 21, 4)",
        ),
    )
    assert (
        "amdgpu.s_getreg_b32.cluster_workgroup_flat_id" in amdgpu_descriptor_ref_keys()
    )


@pytest.mark.parametrize(
    ("dgroup_count", "expected_units", "expected_fields"),
    [
        (2, (4, 8), ("VADDR0", "VADDR1")),
        (4, (4, 8, 4, 4), ("VADDR0", "VADDR1", "VADDR2", "VADDR3")),
    ],
)
def test_gfx125x_tensor_load_descriptors_encode_dgroup_forms(
    dgroup_count: int,
    expected_units: tuple[int, ...],
    expected_fields: tuple[str, ...],
) -> None:
    descriptor = _tensor_load_to_lds_descriptor(dgroup_count)

    assert descriptor.key == f"amdgpu.tensor_load_to_lds.d{dgroup_count}"
    assert descriptor.mnemonic == f"tensor_load_to_lds_d{dgroup_count}"
    assert descriptor.semantic_tag == "memory.tensor.load.to_lds"
    assert descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_VIMAGE
    assert descriptor.encoding_id == 0xC4
    assert descriptor.schedule_class == "amdgpu.tensor.load.lds"
    assert tuple(operand.unit_count for operand in descriptor.operands) == (
        expected_units
    )
    assert tuple(operand.encoding_field_id for operand in descriptor.operands) == tuple(
        amdgpu_encoding_field_id(name) for name in expected_fields
    )
    assert tuple(
        (effect.kind, effect.memory_space, effect.counter_id)
        for effect in descriptor.effects
    ) == (
        (EffectKind.READ, MemorySpace.GLOBAL, _COUNTER_TENSOR),
        (EffectKind.WRITE, MemorySpace.WORKGROUP, _COUNTER_TENSOR),
    )
    assert descriptor.asm_forms[0].native_assembly_mnemonic == ("tensor_load_to_lds")


def test_gfx125x_tensor_wait_descriptor_uses_independent_counter() -> None:
    descriptor = _s_wait_tensorcnt_descriptor()

    assert descriptor.key == "amdgpu.s_wait_tensorcnt"
    assert descriptor.mnemonic == "s_wait_tensorcnt"
    assert descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_SOPP
    assert descriptor.encoding_id == 0x4B
    assert descriptor.schedule_class == "amdgpu.wait.tensor"
    assert len(descriptor.immediates) == 1
    tensorcnt = descriptor.immediates[0]
    assert tensorcnt.field_name == "tensorcnt"
    assert tensorcnt.encoding_id == _WAIT_COUNTER_TENSOR_ENCODING_ID
    assert tensorcnt.bit_width == 16
    assert tensorcnt.unsigned_max == 0xFFFF
    assert descriptor.effects[0].counter_id == _COUNTER_TENSOR


def test_tensor_memory_descriptors_are_gfx125x_scoped() -> None:
    target_descriptor_keys = {
        target: {descriptor.key for descriptor in builder.extra_descriptors}
        for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items()
    }
    tensor_keys = {
        "amdgpu.tensor_load_to_lds.d2",
        "amdgpu.tensor_load_to_lds.d4",
        "amdgpu.s_wait_tensorcnt",
    }

    gfx125x_targets = {
        "gfx12_5_generic",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
    }
    for target in gfx125x_targets:
        assert tensor_keys <= target_descriptor_keys[target]
    for target, descriptor_keys in target_descriptor_keys.items():
        if target not in gfx125x_targets:
            assert tensor_keys.isdisjoint(descriptor_keys)


@pytest.mark.parametrize(
    ("width_bits", "encoding_id"),
    [(8, 0x6A), (32, 0x6B), (64, 0x6C), (128, 0x6D)],
)
def test_gfx125x_cluster_load_descriptors_encode_transfer_widths(
    width_bits: int, encoding_id: int
) -> None:
    descriptor = _cluster_load_async_to_lds_descriptor(width_bits, encoding_id)

    assert descriptor.key == f"amdgpu.cluster_load_async_to_lds_b{width_bits}"
    assert descriptor.mnemonic == f"cluster_load_async_to_lds_b{width_bits}"
    assert descriptor.semantic_tag == f"memory.cluster.load.to_lds.u{width_bits}"
    assert descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_VGLOBAL
    assert descriptor.encoding_id == encoding_id
    assert descriptor.schedule_class == "amdgpu.cluster.load.lds"
    assert tuple(operand.unit_count for operand in descriptor.operands) == (1, 1, 2, 1)
    assert tuple(
        operand.encoding_field_id for operand in descriptor.operands[:3]
    ) == tuple(
        amdgpu_encoding_field_id(field_name)
        for field_name in ("VDST", "VADDR", "SADDR")
    )
    assert descriptor.operands[3].field_name == "m0"
    assert OperandFlag.IMPLICIT in descriptor.operands[3].flags
    assert OperandFlag.STATE_READ in descriptor.operands[3].flags
    assert tuple(
        (effect.kind, effect.memory_space, effect.counter_id, effect.width_bits)
        for effect in descriptor.effects
    ) == (
        (EffectKind.READ, MemorySpace.GLOBAL, _COUNTER_ASYNC, width_bits),
        (EffectKind.WRITE, MemorySpace.WORKGROUP, _COUNTER_ASYNC, width_bits),
    )
    native_values = descriptor.asm_forms[0].native_assembly_values
    assert tuple(value.field_name for value in native_values[:3]) == (
        "lds_addr",
        "addr",
        "saddr",
    )
    assert native_values[5].target_format_id == (
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_SCOPE
    )
    assert native_values[6].target_format_id == (
        AMDGPU_NATIVE_ASM_IMMEDIATE_FORMAT_GFX12_LOAD_TEMPORAL
    )


def test_gfx125x_cluster_wait_descriptor_uses_independent_counter() -> None:
    descriptor = _s_wait_asynccnt_descriptor()

    assert descriptor.key == "amdgpu.s_wait_asynccnt"
    assert descriptor.mnemonic == "s_wait_asynccnt"
    assert descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_SOPP
    assert descriptor.encoding_id == 0x4A
    assert descriptor.schedule_class == "amdgpu.wait.async"
    assert len(descriptor.immediates) == 1
    asynccnt = descriptor.immediates[0]
    assert asynccnt.field_name == "asynccnt"
    assert asynccnt.encoding_id == _WAIT_COUNTER_ASYNC_ENCODING_ID
    assert asynccnt.bit_width == 16
    assert asynccnt.unsigned_max == 0xFFFF
    assert descriptor.effects[0].counter_id == _COUNTER_ASYNC


def test_cluster_memory_descriptors_are_gfx125x_scoped() -> None:
    target_descriptor_keys = {
        target: {descriptor.key for descriptor in builder.extra_descriptors}
        for target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items()
    }
    cluster_keys = {
        "amdgpu.cluster_load_async_to_lds_b8",
        "amdgpu.cluster_load_async_to_lds_b32",
        "amdgpu.cluster_load_async_to_lds_b64",
        "amdgpu.cluster_load_async_to_lds_b128",
        "amdgpu.s_wait_asynccnt",
    }

    gfx125x_targets = {
        "gfx12_5_generic",
        "rdna4_gfx1250_a0",
        "rdna4_gfx1251",
        "rdna4_gfx125x",
    }
    for target in gfx125x_targets:
        assert cluster_keys <= target_descriptor_keys[target]
    for target, descriptor_keys in target_descriptor_keys.items():
        if target not in gfx125x_targets:
            assert cluster_keys.isdisjoint(descriptor_keys)


def test_gfx125x_vop_operands_use_mode_address_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.v_add_u32",
        mnemonic="v_add_u32",
        semantic_tag="test.v_add_u32",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VDST"),
            ),
            Operand(
                "lhs",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("SRC0"),
            ),
            Operand(
                "rhs",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VSRC1"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP2,
    )
    descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
    operands = {operand.field_name: operand for operand in descriptor.operands}

    expected_slots = {
        "dst": AmdgpuVgprMsbSlot.DST,
        "lhs": AmdgpuVgprMsbSlot.SRC0,
        "rhs": AmdgpuVgprMsbSlot.SRC1,
    }
    for field_name, expected_slot in expected_slots.items():
        operand = operands[field_name]
        assert operand.address_map_kind is OperandAddressMapKind.TARGET_STATE
        assert operand.addressable_unit_count == 256
        assert operand.address_state_slot == int(expected_slot)

    mode_operand = operands["mode_in"]
    assert mode_operand.role is OperandRole.IMPLICIT
    assert mode_operand.reg_alts == (
        RegClassAlt(_REG_MODE, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in mode_operand.flags
    assert OperandFlag.STATE_READ in mode_operand.flags
    assert OperandFlag.SCHEDULE_ONLY_STATE in mode_operand.flags


def test_gfx125x_uncontrolled_vgpr_operands_use_low_subset() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.uncontrolled_vgpr",
        mnemonic="uncontrolled_vgpr",
        semantic_tag="test.uncontrolled_vgpr",
        operands=(
            Operand(
                "scale_src0",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("SDST"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
    )
    descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
    operands = {operand.field_name: operand for operand in descriptor.operands}

    for field_name in ("scale_src0",):
        operand = operands[field_name]
        assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
        assert operand.addressable_unit_count == 256
        assert operand.address_state_slot == 0


def test_gfx125x_vop_format_family_uses_mode_address_state() -> None:
    for format_name in AMDGPU_GFX125X_VOP_VGPR_MSB_FORMAT_NAMES:
        assert (
            amdgpu_gfx125x_vgpr_msb_slot(
                "amdgpu.test.vop",
                AMDGPU_ENCODING_FORMAT_IDS[format_name],
                "VDST",
            )
            is AmdgpuVgprMsbSlot.DST
        )


def test_gfx125x_fmamk_routes_second_source_through_src2_mode_slot() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_fmamk_f32",
        mnemonic="v_fmamk_f32",
        semantic_tag="float.fmamk.f32",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VDST"),
            ),
            Operand(
                "a",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("SRC0"),
            ),
            Operand(
                "c",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VSRC1"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS["VOP2_INST_LITERAL"],
    )

    descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
    operands = {operand.field_name: operand for operand in descriptor.operands}

    assert operands["dst"].address_state_slot == int(AmdgpuVgprMsbSlot.DST)
    assert operands["a"].address_state_slot == int(AmdgpuVgprMsbSlot.SRC0)
    assert operands["c"].address_state_slot == int(AmdgpuVgprMsbSlot.SRC2)


def test_gfx125x_target_state_validation_requires_mode_control_descriptor() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.v_mov",
        mnemonic="v_mov_b32",
        semantic_tag="test.v_mov",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VDST"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
    )

    _expect_value_error_contains(
        "amdgpu.s_set_vgpr_msb",
        lambda: _with_gfx125x_vgpr_msb_address_states(_descriptor_set(descriptor)),
    )


def test_gfx125x_target_state_validation_requires_encoding_slot() -> None:
    descriptor = _with_mode_state_read(
        Descriptor(
            key="amdgpu.test.bad_vgpr_msb_slot",
            mnemonic="bad_vgpr_msb_slot",
            semantic_tag="test.bad_vgpr_msb_slot",
            operands=(
                Operand(
                    "dst",
                    OperandRole.RESULT,
                    (RegClassAlt("amdgpu.vgpr"),),
                    encoding_field_id=amdgpu_encoding_field_id("SDST"),
                    address_map_kind=OperandAddressMapKind.TARGET_STATE,
                    addressable_unit_count=256,
                ),
            ),
            schedule_class=_SCHEDULE_VALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
        )
    )

    _expect_value_error_contains(
        "has no S_SET_VGPR_MSB slot",
        lambda: _with_gfx125x_vgpr_msb_address_states(
            _descriptor_set(_s_set_vgpr_msb_descriptor(), descriptor)
        ),
    )


def test_gfx125x_target_state_validation_requires_window_size() -> None:
    descriptor = _with_mode_state_read(
        Descriptor(
            key="amdgpu.test.bad_vgpr_msb_window",
            mnemonic="bad_vgpr_msb_window",
            semantic_tag="test.bad_vgpr_msb_window",
            operands=(
                Operand(
                    "dst",
                    OperandRole.RESULT,
                    (RegClassAlt("amdgpu.vgpr"),),
                    encoding_field_id=amdgpu_encoding_field_id("VDST"),
                    address_map_kind=OperandAddressMapKind.TARGET_STATE,
                    addressable_unit_count=128,
                    address_state_slot=int(AmdgpuVgprMsbSlot.DST),
                ),
            ),
            schedule_class=_SCHEDULE_VALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
        )
    )

    _expect_value_error_contains(
        "expected 256",
        lambda: _with_gfx125x_vgpr_msb_address_states(
            _descriptor_set(_s_set_vgpr_msb_descriptor(), descriptor)
        ),
    )


def test_gfx125x_target_state_validation_requires_encodable_vgpr_capacity() -> None:
    descriptor_set = replace(
        _descriptor_set(_s_set_vgpr_msb_descriptor()),
        reg_classes=(
            RegClass(
                "amdgpu.vgpr",
                32,
                SpillSlotSpace.SCRATCH,
                allocatable_count=1025,
            ),
        ),
    )

    _expect_value_error_contains(
        "4-bank S_SET_VGPR_MSB capacity of 1024 registers",
        lambda: _with_gfx125x_vgpr_msb_address_states(descriptor_set),
    )


def test_amdgpu_descriptor_categories_are_stable() -> None:
    assert tuple(category.key for category in AMDGPU_DESCRIPTOR_CATEGORIES) == (
        "scalar",
        "vector",
        "convert",
        "compare_select",
        "memory",
        "atomic",
        "matrix",
        "control",
        "cache",
        "misc",
    )


def test_amdgpu_descriptor_categorization_uses_semantics() -> None:
    descriptors = _categorize_amdgpu_descriptors(
        (
            _descriptor("amdgpu.v_add_u32", "integer.add.u32"),
            _descriptor("amdgpu.v_cmp_eq_i32", "cmp.i32.eq"),
            _descriptor("amdgpu.buffer_load_dword", "memory.load.u32"),
            _descriptor("amdgpu.global_atomic_add_u32", "memory.global.atomic.add.u32"),
            _descriptor("amdgpu.s_waitcnt", "control.waitcnt"),
        )
    )

    assert tuple(descriptor.category for descriptor in descriptors) == (
        AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
        AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
        AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
        AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
        AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    )


def test_amdgpu_descriptor_category_groups_preserve_category_and_descriptor_order() -> (
    None
):
    descriptors = _categorize_amdgpu_descriptors(
        (
            _descriptor("amdgpu.buffer_load_dword", "memory.load.u32"),
            _descriptor("amdgpu.v_add_u32", "integer.add.u32"),
            _descriptor("amdgpu.global_atomic_add_u32", "memory.global.atomic.add.u32"),
            _descriptor("amdgpu.buffer_store_dword", "memory.store.u32"),
        )
    )

    groups = amdgpu_descriptor_category_groups(descriptors)

    assert [category for category, _ in groups] == [
        AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
        AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
        AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    ]
    assert [descriptor.key for _, group in groups for descriptor in group] == [
        "amdgpu.v_add_u32",
        "amdgpu.buffer_load_dword",
        "amdgpu.buffer_store_dword",
        "amdgpu.global_atomic_add_u32",
    ]


def test_atomic_descriptor_candidates_are_derived_from_overlay_metadata() -> None:
    candidates = amdgpu_atomic_descriptor_candidates()

    candidate_keys = {candidate.descriptor_key for candidate in candidates}
    assert len(candidates) == len(candidate_keys)
    assert candidates[0].descriptor_key == "amdgpu.ds_add_u32"
    assert candidates[0].memory_space == AmdgpuAtomicMemorySpace.WORKGROUP
    assert candidates[0].address_form == AmdgpuMemoryAddressForm.DEFAULT
    assert candidates[0].operation_kind == AmdgpuAtomicOperationKind.REDUCE
    assert candidates[0].atomic_kind == AmdgpuAtomicKind.ADDI
    assert candidates[0].value_kind == AmdgpuAtomicValueKind.I32

    global_saddr_add = next(
        candidate
        for candidate in candidates
        if candidate.descriptor_key == "amdgpu.global_atomic_add_u32_saddr"
    )
    assert global_saddr_add.memory_space == AmdgpuAtomicMemorySpace.GLOBAL
    assert global_saddr_add.address_form == AmdgpuMemoryAddressForm.GLOBAL_SADDR

    flat_cmpxchg = next(
        candidate
        for candidate in candidates
        if candidate.descriptor_key == "amdgpu.flat_atomic_cmpswap_b32_rtn"
    )
    assert flat_cmpxchg.memory_space == AmdgpuAtomicMemorySpace.GENERIC
    assert flat_cmpxchg.operation_kind == AmdgpuAtomicOperationKind.CMPXCHG
    assert flat_cmpxchg.atomic_kind == AmdgpuAtomicKind.NONE
    assert flat_cmpxchg.value_kind == AmdgpuAtomicValueKind.B32


def test_atomic_descriptor_candidates_model_packed_half_rows() -> None:
    candidates = {
        candidate.descriptor_key: candidate
        for candidate in amdgpu_atomic_descriptor_candidates()
    }

    assert (
        candidates["amdgpu.buffer_atomic_pk_add_f16"].value_kind
        == AmdgpuAtomicValueKind.PACKED_F16
    )
    assert (
        candidates["amdgpu.flat_atomic_pk_add_bf16_rtn"].value_kind
        == AmdgpuAtomicValueKind.PACKED_BF16
    )
    assert (
        candidates["amdgpu.ds_pk_add_rtn_f16"].value_kind
        == AmdgpuAtomicValueKind.PACKED_F16
    )


def test_atomic_descriptor_candidates_reject_conflicting_duplicate_metadata() -> None:
    candidates_by_key: dict[str, AmdgpuAtomicDescriptorCandidate] = {}
    candidate = AmdgpuAtomicDescriptorCandidate(
        memory_space=AmdgpuAtomicMemorySpace.GLOBAL,
        address_form=AmdgpuMemoryAddressForm.DEFAULT,
        operation_kind=AmdgpuAtomicOperationKind.REDUCE,
        atomic_kind=AmdgpuAtomicKind.ADDI,
        value_kind=AmdgpuAtomicValueKind.I32,
        descriptor_key="amdgpu.buffer_atomic_add_u32",
    )
    _record_amdgpu_atomic_candidate(candidates_by_key, candidate)
    _record_amdgpu_atomic_candidate(candidates_by_key, candidate)

    conflicting_candidate = AmdgpuAtomicDescriptorCandidate(
        memory_space=AmdgpuAtomicMemorySpace.GLOBAL,
        address_form=AmdgpuMemoryAddressForm.GLOBAL_SADDR,
        operation_kind=AmdgpuAtomicOperationKind.REDUCE,
        atomic_kind=AmdgpuAtomicKind.ADDI,
        value_kind=AmdgpuAtomicValueKind.I32,
        descriptor_key="amdgpu.buffer_atomic_add_u32",
    )
    with pytest.raises(ValueError, match="conflicting metadata"):
        _record_amdgpu_atomic_candidate(candidates_by_key, conflicting_candidate)


def test_atomic_descriptor_candidates_have_descriptor_refs() -> None:
    descriptor_ref_keys = set(amdgpu_descriptor_ref_keys())

    assert {
        candidate.descriptor_key for candidate in amdgpu_atomic_descriptor_candidates()
    }.issubset(descriptor_ref_keys)


def _assert_s_sendmsg_low_asm_form(descriptor: AmdgpuDescriptorOverlay) -> None:
    assert descriptor.asm_forms is not None
    assert len(descriptor.asm_forms) == 1
    asm_form = descriptor.asm_forms[0]
    assert asm_form.mnemonic == "s_sendmsg"
    assert asm_form.operands == ("m0",)
    assert tuple(immediate.field_name for immediate in asm_form.immediates) == (
        "message",
    )
    assert tuple(immediate.name for immediate in asm_form.immediates) == ("message",)


def test_feedback_control_descriptors_cover_execution_families() -> None:
    for overlays in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in (
            "amdgpu.s_sendmsg",
            "amdgpu.s_sethalt",
            "amdgpu.s_trap",
        ):
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_MODE_CONTROL
            assert descriptor.semantic_tag.startswith("control.")
            if descriptor_key == "amdgpu.s_sendmsg":
                _assert_s_sendmsg_low_asm_form(descriptor)

        assert "amdgpu.s_sendmsg_rtn_b32" not in descriptors

    for overlays in (
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for descriptor_key in (
            "amdgpu.s_sendmsg",
            "amdgpu.s_sendmsg_rtn_b32",
            "amdgpu.s_sethalt",
            "amdgpu.s_trap",
        ):
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_MODE_CONTROL
            assert descriptor.semantic_tag.startswith("control.")
            if descriptor_key == "amdgpu.s_sendmsg":
                _assert_s_sendmsg_low_asm_form(descriptor)

        assert descriptors["amdgpu.s_sendmsg_rtn_b32"].immediate_fields == ("SSRC0",)
        message_immediate = descriptors["amdgpu.s_sendmsg_rtn_b32"].immediates[0]
        assert message_immediate.field_name == "message"
        assert message_immediate.bit_width == 8
        assert message_immediate.unsigned_max >= 128


def test_symbol_relative_salu_descriptors_have_lossless_low_asm_forms() -> None:
    pc_relative_effect = (Effect(EffectKind.CONVERGENT, flags=(EffectFlag.ORDERED,)),)

    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}

        add_lo = descriptors["amdgpu.s_add_u32.rhs_symbol_rel32_lo"]
        assert add_lo.effects == pc_relative_effect
        assert add_lo.asm_forms is not None
        assert add_lo.asm_forms[0].mnemonic == "s_add_u32_rhs_symbol_rel32_lo"
        assert add_lo.asm_forms[0].results == ("dst",)
        assert add_lo.asm_forms[0].operands == ("lhs",)
        assert tuple(
            immediate.field_name for immediate in add_lo.asm_forms[0].immediates
        ) == ("symbol", "byte_offset")

        addc_hi = descriptors["amdgpu.s_addc_u32.rhs_symbol_rel32_hi"]
        assert addc_hi.effects == pc_relative_effect
        assert addc_hi.asm_forms is not None
        assert addc_hi.asm_forms[0].mnemonic == "s_addc_u32_rhs_symbol_rel32_hi"
        assert addc_hi.asm_forms[0].results == ("sum",)
        assert addc_hi.asm_forms[0].operands == ("lhs",)
        assert tuple(
            immediate.field_name for immediate in addc_hi.asm_forms[0].immediates
        ) == ("symbol", "byte_offset")


def _assert_feedback_atomic64_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    mnemonic: str,
    semantic_tag: str,
    memory_space: MemorySpace,
    payload_field_name: str,
    payload_units: int,
    implicit_data_format: str = "FMT_NUM_U64",
) -> None:
    assert descriptor.mnemonic == mnemonic
    assert descriptor.semantic_tag == semantic_tag
    assert tuple(effect.memory_space for effect in descriptor.effects) == (
        memory_space,
        memory_space,
    )
    assert tuple(effect.width_bits for effect in descriptor.effects) == (64, 64)
    payload_operand = next(
        operand.descriptor_operand
        for operand in descriptor.operands
        if operand.descriptor_operand.field_name == payload_field_name
    )
    assert payload_operand.unit_count == payload_units
    assert any(
        operand.operand_type == "OPR_GPUMEM"
        and operand.data_format_name == implicit_data_format
        and operand.size_bits == 64
        for operand in descriptor.implicit_operands
    )


def test_feedback_atomic64_descriptors_cover_execution_families() -> None:
    for overlays, wide_mnemonic_suffix in (
        (_gfx940_core_overlays(), "x2"),
        (_gfx950_core_overlays(), "x2"),
        (_gfx11_core_overlays(), "u64"),
        (_gfx12_core_overlays(), "u64"),
        (_gfx125x_core_overlays(), "u64"),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}

        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.flat_atomic_add_u64"],
            mnemonic=f"flat_atomic_add_{wide_mnemonic_suffix}",
            semantic_tag="memory.generic.atomic.add.u64",
            memory_space=MemorySpace.GENERIC,
            payload_field_name="value",
            payload_units=2,
        )
        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.flat_atomic_add_u64_rtn"],
            mnemonic=f"flat_atomic_add_{wide_mnemonic_suffix}",
            semantic_tag="memory.generic.atomic.add.u64.return",
            memory_space=MemorySpace.GENERIC,
            payload_field_name="value",
            payload_units=2,
        )
        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.flat_atomic_cmpswap_b64_rtn"],
            mnemonic=f"flat_atomic_cmpswap_{'x2' if wide_mnemonic_suffix == 'x2' else 'b64'}",
            semantic_tag="memory.generic.atomic.compare_exchange.b64.return",
            memory_space=MemorySpace.GENERIC,
            payload_field_name="value",
            payload_units=4,
        )

        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.global_atomic_add_u64_saddr"],
            mnemonic=f"global_atomic_add_{wide_mnemonic_suffix}",
            semantic_tag="memory.global.atomic.add.u64",
            memory_space=MemorySpace.GLOBAL,
            payload_field_name="value",
            payload_units=2,
        )
        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.global_atomic_add_u64_rtn_saddr"],
            mnemonic=f"global_atomic_add_{wide_mnemonic_suffix}",
            semantic_tag="memory.global.atomic.add.u64.return",
            memory_space=MemorySpace.GLOBAL,
            payload_field_name="value",
            payload_units=2,
        )
        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.global_atomic_swap_u64_rtn_saddr"],
            mnemonic=f"global_atomic_swap_{wide_mnemonic_suffix}",
            semantic_tag="memory.global.atomic.exchange.u64.return",
            memory_space=MemorySpace.GLOBAL,
            payload_field_name="value",
            payload_units=2,
            implicit_data_format="FMT_NUM_B64",
        )
        _assert_feedback_atomic64_overlay(
            descriptors["amdgpu.global_atomic_cmpswap_b64_rtn_saddr"],
            mnemonic=(
                f"global_atomic_cmpswap_{'x2' if wide_mnemonic_suffix == 'x2' else 'b64'}"
            ),
            semantic_tag="memory.global.atomic.compare_exchange.b64.return",
            memory_space=MemorySpace.GLOBAL,
            payload_field_name="value",
            payload_units=4,
        )


def test_feedback_atomic64_descriptors_expand_source_atomic_candidates() -> None:
    keys = {
        candidate.descriptor_key for candidate in amdgpu_atomic_descriptor_candidates()
    }

    assert "amdgpu.flat_atomic_add_u64" in keys
    assert "amdgpu.flat_atomic_add_u64_rtn" in keys
    assert "amdgpu.flat_atomic_cmpswap_b64_rtn" in keys
    assert "amdgpu.global_atomic_add_u64_saddr" in keys
    assert "amdgpu.global_atomic_add_u64_rtn_saddr" in keys
    assert "amdgpu.global_atomic_swap_u64_rtn_saddr" in keys
    assert "amdgpu.global_atomic_cmpswap_b64_rtn_saddr" in keys


def test_flat_memory_descriptors_cover_execution_families() -> None:
    cdna_load_mnemonics = (
        "flat_load_ubyte",
        "flat_load_sbyte",
        "flat_load_ushort",
        "flat_load_sshort",
        "flat_load_dword",
        "flat_load_dwordx2",
        "flat_load_dwordx3",
        "flat_load_dwordx4",
    )
    rdna_load_mnemonics = (
        "flat_load_u8",
        "flat_load_i8",
        "flat_load_u16",
        "flat_load_i16",
        "flat_load_dword",
        "flat_load_dwordx2",
        "flat_load_dwordx3",
        "flat_load_dwordx4",
    )
    cdna_store_mnemonics = (
        "flat_store_byte",
        "flat_store_short",
        "flat_store_dword",
        "flat_store_dwordx2",
        "flat_store_dwordx3",
        "flat_store_dwordx4",
    )
    rdna_store_mnemonics = (
        "flat_store_b8",
        "flat_store_b16",
        "flat_store_b32",
        "flat_store_b64",
        "flat_store_b96",
        "flat_store_b128",
    )
    for (
        overlays,
        load_mnemonics,
        store_mnemonics,
        uses_flat_scratch,
        uses_m0,
        expected_saddr_fields,
        expected_asm_immediates,
    ) in (
        (
            _gfx940_core_overlays(),
            cdna_load_mnemonics,
            cdna_store_mnemonics,
            True,
            True,
            (),
            ("offset", "nt", "sc0", "sc1"),
        ),
        (
            _gfx950_core_overlays(),
            cdna_load_mnemonics,
            cdna_store_mnemonics,
            True,
            True,
            (),
            ("offset", "nt", "sc0", "sc1"),
        ),
        (
            _gfx11_core_overlays(),
            rdna_load_mnemonics,
            rdna_store_mnemonics,
            True,
            False,
            (("SADDR", _predefined("NULL", "OPR_SREG")),),
            ("offset", "glc", "slc", "dlc"),
        ),
        (
            _gfx12_core_overlays(),
            rdna_load_mnemonics,
            rdna_store_mnemonics,
            False,
            False,
            (),
            ("offset", "nv", "scope", "th"),
        ),
        (
            _gfx125x_core_overlays(),
            rdna_load_mnemonics,
            rdna_store_mnemonics,
            False,
            False,
            (),
            ("offset", "nv", "scope", "th"),
        ),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        for (
            descriptor_key,
            width_bits,
            semantic_tag,
            operand_units,
            implicit_data_format,
        ), mnemonic in zip(
            (
                (
                    "amdgpu.flat_load_u8",
                    8,
                    "memory.generic.load.u8.zero_extend",
                    1,
                    "FMT_NUM_U8",
                ),
                (
                    "amdgpu.flat_load_i8",
                    8,
                    "memory.generic.load.i8.sign_extend",
                    1,
                    "FMT_NUM_I8",
                ),
                (
                    "amdgpu.flat_load_u16",
                    16,
                    "memory.generic.load.u16.zero_extend",
                    1,
                    "FMT_NUM_U16",
                ),
                (
                    "amdgpu.flat_load_i16",
                    16,
                    "memory.generic.load.i16.sign_extend",
                    1,
                    "FMT_NUM_I16",
                ),
                (
                    "amdgpu.flat_load_b32",
                    32,
                    "memory.generic.load.u32",
                    1,
                    "FMT_NUM_B32",
                ),
                (
                    "amdgpu.flat_load_b64",
                    64,
                    "memory.generic.load.u64",
                    2,
                    "FMT_NUM_B64",
                ),
                (
                    "amdgpu.flat_load_b96",
                    96,
                    "memory.generic.load.u96",
                    3,
                    "FMT_NUM_B96",
                ),
                (
                    "amdgpu.flat_load_b128",
                    128,
                    "memory.generic.load.u128",
                    4,
                    "FMT_NUM_B128",
                ),
            ),
            load_mnemonics,
            strict=True,
        ):
            descriptor = descriptors[descriptor_key]
            _assert_memory_width_overlay(
                descriptor,
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=mnemonic,
                operand_units=operand_units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GENERIC,
                implicit_data_format=implicit_data_format,
                implicit_ignore_reason="modeled-by-generic-read-effect",
            )
            assert descriptor.schedule_class == _SCHEDULE_VMEM_LOAD
            assert descriptor.asm_forms is not None
            assert len(descriptor.asm_forms) == 1
            asm_form = descriptor.asm_forms[0]
            assert asm_form.mnemonic == mnemonic
            assert asm_form.results == ("dst",)
            expected_asm_operands = ("addr", "m0") if uses_m0 else ("addr",)
            assert asm_form.operands == expected_asm_operands
            assert (
                tuple(immediate.field_name for immediate in asm_form.immediates)
                == expected_asm_immediates
            )
            assert (
                tuple(immediate.name for immediate in asm_form.immediates)
                == expected_asm_immediates
            )
            assert (
                any(
                    operand.operand_type == "OPR_FLAT_SCRATCH"
                    for operand in descriptor.implicit_operands
                )
                == uses_flat_scratch
            )
            assert descriptor.fixed_encoding_fields == expected_saddr_fields
            assert (
                any(
                    operand.operand_type == "OPR_SDST_M0"
                    for operand in descriptor.implicit_operands
                )
                == uses_m0
            )

        for (width_bits, operand_units), mnemonic in zip(
            ((8, 1), (16, 1), (32, 1), (64, 2), (96, 3), (128, 4)),
            store_mnemonics,
            strict=True,
        ):
            descriptor = descriptors[f"amdgpu.flat_store_b{width_bits}"]
            _assert_memory_width_overlay(
                descriptor,
                width_bits=width_bits,
                semantic_tag=f"memory.generic.store.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=operand_units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GENERIC,
                implicit_data_format=f"FMT_NUM_B{width_bits}",
                implicit_ignore_reason="modeled-by-generic-write-effect",
            )
            assert descriptor.schedule_class == _SCHEDULE_VMEM_STORE
            assert descriptor.asm_forms is not None
            assert len(descriptor.asm_forms) == 1
            asm_form = descriptor.asm_forms[0]
            assert asm_form.mnemonic == mnemonic
            assert asm_form.results == ()
            expected_asm_operands = (
                ("addr", "value", "m0") if uses_m0 else ("addr", "value")
            )
            assert asm_form.operands == expected_asm_operands
            assert (
                tuple(immediate.name for immediate in asm_form.immediates)
                == expected_asm_immediates
            )
            assert (
                any(
                    operand.operand_type == "OPR_FLAT_SCRATCH"
                    for operand in descriptor.implicit_operands
                )
                == uses_flat_scratch
            )
            assert descriptor.fixed_encoding_fields == expected_saddr_fields
            assert (
                any(
                    operand.operand_type == "OPR_SDST_M0"
                    for operand in descriptor.implicit_operands
                )
                == uses_m0
            )


def test_gfx12_global_atomic_return_uses_temporal_hint_return_bit() -> None:
    for overlays in (_gfx12_core_overlays(), _gfx125x_core_overlays()):
        for descriptor_prefix, descriptor_suffix in (
            ("amdgpu.global_atomic", "_saddr"),
            ("amdgpu.flat_atomic", ""),
        ):
            no_return = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key
                == f"{descriptor_prefix}_add_u32{descriptor_suffix}"
            )
            with_return = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key
                == f"{descriptor_prefix}_add_u32_rtn{descriptor_suffix}"
            )

            assert _immediate_default(no_return.immediates, "th") == 0
            assert _immediate_default(with_return.immediates, "th") == (
                _GFX12_TH_ATOMIC_RETURN_VALUE
            )


def test_gfx12_global_cache_controls_expose_scope_immediate() -> None:
    for overlays in (_gfx12_core_overlays(), _gfx125x_core_overlays()):
        for descriptor_key in (
            "amdgpu.global_inv",
            "amdgpu.global_wb",
            "amdgpu.global_wbinv",
        ):
            descriptor = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key == descriptor_key
            )
            assert _immediate_default(descriptor.immediates, "scope") == 0


def test_cdna_scoped_cache_controls_expose_sc_immediates() -> None:
    for overlays in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        for descriptor_key in (
            "amdgpu.buffer_inv",
            "amdgpu.buffer_wbl2",
        ):
            descriptor = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key == descriptor_key
            )
            assert _immediate_default(descriptor.immediates, "sc0") == 0
            assert _immediate_default(descriptor.immediates, "sc1") == 0


def test_vop3_shift_immediate_is_constrained_to_inline_source_selector() -> None:
    descriptor = next(
        overlay
        for overlay in _gfx12_core_overlays()
        if overlay.descriptor_key == "amdgpu.v_lshlrev_b32.vop3_imm"
    )
    assert len(descriptor.immediates) == 1
    immediate = descriptor.immediates[0]
    assert immediate.field_name == "imm32"
    assert immediate.encoding_id == _SOURCE_INLINE_U32_ENCODING_ID
    assert immediate.unsigned_max == 64


def test_vop3_mixed_inline_literal_immediates_name_both_encoding_fields() -> None:
    for overlays in (
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        shift_add = descriptors["amdgpu.v_lshl_add_u32.shift_imm.src2_lit"]
        assert shift_add.immediate_fields == ("SRC1", "LITERAL")

        src0_literal = descriptors["amdgpu.v_cndmask_b32.src0_lit_src1_inline"]
        assert src0_literal.immediate_fields == ("LITERAL", "SRC1")

        src1_literal = descriptors["amdgpu.v_cndmask_b32.src1_lit_src0_inline"]
        assert src1_literal.immediate_fields == ("LITERAL", "SRC0")


def test_vop2_f32_inline_immediate_uses_enum_domain() -> None:
    descriptor = next(
        overlay
        for overlay in _gfx12_core_overlays()
        if overlay.descriptor_key == "amdgpu.v_add_f32.src0_inline"
    )
    assert len(descriptor.immediates) == 1
    immediate = descriptor.immediates[0]
    assert immediate.field_name == "imm32"
    assert immediate.kind == ImmediateKind.ENUM
    assert immediate.encoding_id == _SOURCE_INLINE_F32_ENCODING_ID
    assert immediate.enum_domain == "amdgpu.source_inline_f32"


def test_vop2_f32_uses_inline_then_literal_operand_forms() -> None:
    overlays_by_key = {
        overlay.descriptor_key: overlay for overlay in _gfx12_core_overlays()
    }
    for descriptor_key in (
        "amdgpu.v_add_f32",
        "amdgpu.v_sub_f32",
        "amdgpu.v_mul_f32",
        "amdgpu.v_min_f32",
        "amdgpu.v_max_f32",
    ):
        descriptor = overlays_by_key[descriptor_key]
        assert tuple(
            form.replacement_descriptor for form in descriptor.operand_forms
        ) == (
            f"{descriptor_key}.src0_inline",
            f"{descriptor_key}.lit",
        )


def test_v_perm_b32_literal_forms_cover_selector_and_zero_source() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }

    descriptor = descriptors["amdgpu.v_perm_b32"]
    assert tuple(form.replacement_descriptor for form in descriptor.operand_forms) == (
        "amdgpu.v_perm_b32.src2_lit",
    )

    selector_literal = descriptors["amdgpu.v_perm_b32.src2_lit"]
    assert selector_literal.encoding_name == "ENC_VOP3"
    assert selector_literal.encoding_format_id == AMDGPU_ENCODING_FORMAT_VOP3_LITERAL
    assert tuple(operand.xml_field_name for operand in selector_literal.operands) == (
        "VDST",
        "SRC0",
        "SRC1",
    )
    assert tuple(immediate.field_name for immediate in selector_literal.immediates) == (
        "imm32",
    )
    src2_field, src2_value = selector_literal.fixed_encoding_fields[0]
    assert src2_field == "SRC2"
    assert isinstance(src2_value, AmdgpuOperandPredefinedValueRef)
    assert src2_value.value_name == "SRC_LITERAL"

    zero_selector_literal = descriptors["amdgpu.v_perm_b32.src1_zero_src2_lit"]
    assert zero_selector_literal.encoding_name == "ENC_VOP3"
    assert (
        zero_selector_literal.encoding_format_id == AMDGPU_ENCODING_FORMAT_VOP3_LITERAL
    )
    assert tuple(
        operand.xml_field_name for operand in zero_selector_literal.operands
    ) == ("VDST", "SRC0")
    assert tuple(
        immediate.field_name for immediate in zero_selector_literal.immediates
    ) == ("imm32",)
    src1_field, src1_value = zero_selector_literal.fixed_encoding_fields[0]
    assert src1_field == "SRC1"
    assert isinstance(src1_value, AmdgpuOperandPredefinedValueRef)
    assert src1_value.value_name == "0"
    assert src1_value.operand_type == "OPR_SRC"
    src2_field, src2_value = zero_selector_literal.fixed_encoding_fields[1]
    assert src2_field == "SRC2"
    assert isinstance(src2_value, AmdgpuOperandPredefinedValueRef)
    assert src2_value.value_name == "SRC_LITERAL"
    assert src2_value.operand_type == "OPR_SRC"


def test_cdna_excludes_unsupported_vop3_literal_integer_forms() -> None:
    unsupported_keys = {
        "amdgpu.v_lshl_add_u32.shift_imm.src2_lit",
        "amdgpu.v_bfi_b32.src0_lit",
        "amdgpu.v_perm_b32.src2_lit",
        "amdgpu.v_perm_b32.src1_zero_src2_lit",
    }
    for overlays in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        assert unsupported_keys.isdisjoint(
            overlay.descriptor_key for overlay in overlays
        )
    for overlays in (
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
    ):
        assert unsupported_keys.issubset(overlay.descriptor_key for overlay in overlays)


def test_sop2_bfe_literal_forms_fix_control_to_literal_source() -> None:
    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in ("amdgpu.s_bfe_i32", "amdgpu.s_bfe_u32"):
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_SALU
            assert tuple(
                form.replacement_descriptor for form in descriptor.operand_forms
            ) == (f"{descriptor_key}.lit",)

            literal_descriptor = descriptors[f"{descriptor_key}.lit"]
            assert literal_descriptor.encoding_name == "ENC_SOP2"
            assert (
                literal_descriptor.encoding_format_id
                == AMDGPU_ENCODING_FORMAT_SOP2_LITERAL
            )
            assert tuple(
                operand.xml_field_name for operand in literal_descriptor.operands
            ) == ("SDST", "SSRC0")
            assert literal_descriptor.immediate_fields == ("LITERAL",)
            assert tuple(
                immediate.field_name for immediate in literal_descriptor.immediates
            ) == ("imm32",)
            fixed_field, fixed_value = literal_descriptor.fixed_encoding_fields[0]
            assert fixed_field == "SSRC1"
            assert isinstance(fixed_value, AmdgpuOperandPredefinedValueRef)
            assert fixed_value.operand_type == "OPR_SSRC"
            assert fixed_value.value_name == "SRC_LITERAL"


def test_fmamk_f32_descriptor_pins_literal_multiply_slot() -> None:
    descriptor_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    )
    for descriptor_set in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_fmamk_f32"]
        assert descriptor.instruction_name == "V_FMAMK_F32"
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "VDST",
            "SRC0",
            "VSRC1",
        )
        assert tuple(
            operand.descriptor_operand.field_name for operand in descriptor.operands
        ) == ("dst", "a", "c")
        assert descriptor.immediate_fields == ("LITERAL",)
        assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
            "imm32",
        )


def test_scalar_f16_fma_descriptor_families_are_arch_specific() -> None:
    cdna_keys = {
        "amdgpu.v_mad_f16",
        "amdgpu.v_mac_f16",
        "amdgpu.v_madak_f16",
        "amdgpu.v_madmk_f16",
        "amdgpu.v_fma_f16",
        "amdgpu.v_fma_f64",
        "amdgpu.v_fmac_f64",
    }
    rdna_keys = {
        "amdgpu.v_fma_f16",
        "amdgpu.v_fmac_f16",
        "amdgpu.v_fmaak_f16",
        "amdgpu.v_fmamk_f16",
        "amdgpu.v_fma_f64",
    }
    for descriptor_set in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert cdna_keys <= descriptors.keys()
        assert (
            not {
                "amdgpu.v_fmac_f16",
                "amdgpu.v_fmaak_f16",
                "amdgpu.v_fmamk_f16",
            }
            & descriptors.keys()
        )

    for descriptor_set in (
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert rdna_keys <= descriptors.keys()
        assert (
            not {
                "amdgpu.v_mad_f16",
                "amdgpu.v_mac_f16",
                "amdgpu.v_madak_f16",
                "amdgpu.v_madmk_f16",
                "amdgpu.v_fmac_f64",
            }
            & descriptors.keys()
        )


def test_scalar_f16_fma_descriptors_pin_low16_and_literal_width() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor
        for descriptor in (
            *_gfx940_core_overlays(),
            *_gfx11_core_overlays(),
        )
    }
    for descriptor_key in (
        "amdgpu.v_fma_f16",
        "amdgpu.v_mad_f16",
    ):
        descriptor = descriptors[descriptor_key]
        assert tuple(
            operand.descriptor_operand.register_part for operand in descriptor.operands
        ) == (
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
        )

    for descriptor_key in (
        "amdgpu.v_fmac_f16",
        "amdgpu.v_mac_f16",
    ):
        descriptor = descriptors[descriptor_key]
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "VDST",
            "VDST",
            "SRC0",
            "VSRC1",
        )
        assert tuple(
            operand.descriptor_operand.register_part for operand in descriptor.operands
        ) == (
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
            _REG_PART_VGPR_LOW16,
        )
        assert tuple(constraint.kind for constraint in descriptor.constraints) == (
            ConstraintKind.TIED,
            ConstraintKind.DESTRUCTIVE,
        )

    for descriptor_key in (
        "amdgpu.v_fmaak_f16",
        "amdgpu.v_fmamk_f16",
        "amdgpu.v_madak_f16",
        "amdgpu.v_madmk_f16",
    ):
        descriptor = descriptors[descriptor_key]
        assert descriptor.encoding_name == "VOP2_INST_LITERAL"
        assert descriptor.immediate_fields == ("LITERAL",)
        assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
            "imm16",
        )
        assert tuple(immediate.bit_width for immediate in descriptor.immediates) == (
            16,
        )
        assert tuple(immediate.unsigned_max for immediate in descriptor.immediates) == (
            0xFFFF,
        )


def test_scalar_f64_fma_descriptors_pin_register_pair_widths() -> None:
    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_fma_f64"]
        assert tuple(
            operand.descriptor_operand.unit_count for operand in descriptor.operands
        ) == (2, 2, 2, 2)

    for descriptor_set in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_fmac_f64"]
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "VDST",
            "VDST",
            "SRC0",
            "VSRC1",
        )
        assert tuple(
            operand.descriptor_operand.unit_count for operand in descriptor.operands
        ) == (2, 2, 2, 2)
        assert tuple(constraint.kind for constraint in descriptor.constraints) == (
            ConstraintKind.TIED,
            ConstraintKind.DESTRUCTIVE,
        )


def test_scalar_domain_fma_descriptors_are_arch_specific() -> None:
    scalar_domain_keys = {
        "amdgpu.s_fmaak_f32",
        "amdgpu.s_fmamk_f32",
        "amdgpu.s_fmac_f32",
        "amdgpu.s_fmac_f16",
    }

    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert not scalar_domain_keys & descriptors.keys()

    for descriptor_set in (
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert scalar_domain_keys <= descriptors.keys()


def test_scalar_float_arithmetic_descriptors_are_arch_specific() -> None:
    scalar_float_keys = {
        f"amdgpu.s_{operation}_f{bit_width}"
        for bit_width in (16, 32)
        for operation in (
            "add",
            "sub",
            "mul",
            "min",
            "max",
            "ceil",
            "floor",
            "rndne",
            "trunc",
        )
    }

    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx11_generic_core_overlays(),
    ):
        descriptor_keys = {descriptor.descriptor_key for descriptor in descriptor_set}
        assert not scalar_float_keys & descriptor_keys

    for descriptor_set in (
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert scalar_float_keys <= descriptors.keys()
        for descriptor_key in scalar_float_keys:
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_SALU
            expected_register_part = (
                _REG_PART_SGPR_LOW16 if descriptor_key.endswith("f16") else None
            )
            assert all(
                operand.descriptor_operand.register_part == expected_register_part
                for operand in descriptor.operands
            )


def test_scalar_float_conversion_descriptors_are_arch_specific() -> None:
    expected_register_parts = {
        "amdgpu.s_cvt_f32_i32": (None, None),
        "amdgpu.s_cvt_f32_u32": (None, None),
        "amdgpu.s_cvt_i32_f32": (None, None),
        "amdgpu.s_cvt_u32_f32": (None, None),
        "amdgpu.s_cvt_f16_f32": (_REG_PART_SGPR_LOW16, None),
        "amdgpu.s_cvt_f32_f16": (None, _REG_PART_SGPR_LOW16),
        "amdgpu.s_cvt_hi_f32_f16": (None, _REG_PART_SGPR_HIGH16),
        "amdgpu.s_cvt_pk_rtz_f16_f32": (None, None, None),
    }

    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx11_generic_core_overlays(),
    ):
        descriptor_keys = {descriptor.descriptor_key for descriptor in descriptor_set}
        assert not expected_register_parts.keys() & descriptor_keys

    for descriptor_set in (
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert expected_register_parts.keys() <= descriptors.keys()
        for descriptor_key, register_parts in expected_register_parts.items():
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_SALU
            assert descriptor.encoding_name == (
                "ENC_SOP2"
                if descriptor_key == "amdgpu.s_cvt_pk_rtz_f16_f32"
                else "ENC_SOP1"
            )
            assert (
                tuple(
                    operand.descriptor_operand.register_part
                    for operand in descriptor.operands
                )
                == register_parts
            )


def test_scalar_float_compare_descriptors_are_arch_specific() -> None:
    scalar_float_compare_keys = {
        f"amdgpu.s_cmp_{predicate}_f{bit_width}"
        for bit_width in (16, 32)
        for predicate in (
            "oeq",
            "ogt",
            "oge",
            "olt",
            "ole",
            "one",
            "ord",
            "ueq",
            "ugt",
            "uge",
            "ult",
            "ule",
            "une",
            "uno",
        )
    }

    for descriptor_set in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx11_generic_core_overlays(),
    ):
        descriptor_keys = {descriptor.descriptor_key for descriptor in descriptor_set}
        assert not scalar_float_compare_keys & descriptor_keys

    for descriptor_set in (
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert scalar_float_compare_keys <= descriptors.keys()
        for descriptor_key in scalar_float_compare_keys:
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_SALU_COMPARE
            assert descriptor.encoding_name == "ENC_SOPC"
            expected_register_part = (
                _REG_PART_SGPR_LOW16 if descriptor_key.endswith("f16") else None
            )
            assert all(
                operand.descriptor_operand.register_part == expected_register_part
                for operand in descriptor.operands
            )
            assert tuple(
                operand.descriptor_operand.field_name
                for operand in descriptor.implicit_operands
            ) == ("scc",)


def test_scalar_domain_fma_descriptors_pin_sgpr_contracts() -> None:
    for descriptor_set in (
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }

        for descriptor_key, expected_operands in (
            ("amdgpu.s_fmaak_f32", ("dst", "a", "b")),
            ("amdgpu.s_fmamk_f32", ("dst", "a", "c")),
        ):
            descriptor = descriptors[descriptor_key]
            assert descriptor.schedule_class == _SCHEDULE_SALU
            assert descriptor.encoding_name == "SOP2_INST_LITERAL"
            assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
                "SDST",
                "SSRC0",
                "SSRC1",
            )
            assert (
                tuple(
                    operand.descriptor_operand.field_name
                    for operand in descriptor.operands
                )
                == expected_operands
            )
            assert descriptor.immediate_fields == ("LITERAL",)
            assert len(descriptor.immediates) == 1
            immediate = descriptor.immediates[0]
            assert immediate.kind == ImmediateKind.UNSIGNED
            assert immediate.field_name == "imm32"
            assert immediate.bit_width == 32
            assert immediate.unsigned_max == 0xFFFFFFFF

        descriptor = descriptors["amdgpu.s_fmac_f32"]
        assert descriptor.schedule_class == _SCHEDULE_SALU
        assert descriptor.encoding_name == "ENC_SOP2"
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "SDST",
            "SDST",
            "SSRC0",
            "SSRC1",
        )
        assert tuple(
            operand.descriptor_operand.field_name for operand in descriptor.operands
        ) == ("dst", "acc", "a", "b")
        assert tuple(
            operand.descriptor_operand.register_part for operand in descriptor.operands
        ) == (None, None, None, None)
        assert descriptor.operands[1].descriptor_operand.role is OperandRole.OPERAND
        assert OperandFlag.IMPLICIT in descriptor.operands[1].descriptor_operand.flags
        constraint_contracts = {
            (
                constraint.kind,
                constraint.lhs_operand_index,
                constraint.rhs_operand_index,
            )
            for constraint in descriptor.constraints
        }
        assert (ConstraintKind.TIED, 0, 1) in constraint_contracts
        assert (ConstraintKind.DESTRUCTIVE, 0, 1) in constraint_contracts

        descriptor = descriptors["amdgpu.s_fmac_f16"]
        assert descriptor.schedule_class == _SCHEDULE_SALU
        assert descriptor.encoding_name == "ENC_SOP2"
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "SDST",
            "SDST",
            "SSRC0",
            "SSRC1",
        )
        assert tuple(
            operand.descriptor_operand.field_name for operand in descriptor.operands
        ) == ("dst", "acc", "a", "b")
        assert tuple(
            operand.descriptor_operand.register_part for operand in descriptor.operands
        ) == (
            _REG_PART_SGPR_LOW16,
            _REG_PART_SGPR_LOW16,
            _REG_PART_SGPR_LOW16,
            _REG_PART_SGPR_LOW16,
        )
        assert descriptor.operands[1].descriptor_operand.role is OperandRole.OPERAND
        assert OperandFlag.IMPLICIT in descriptor.operands[1].descriptor_operand.flags
        assert tuple(constraint.kind for constraint in descriptor.constraints) == (
            ConstraintKind.TIED,
            ConstraintKind.DESTRUCTIVE,
        )
        assert tuple(
            (constraint.lhs_operand_index, constraint.rhs_operand_index)
            for constraint in descriptor.constraints
        ) == ((0, 1), (0, 1))


def test_packed_fma_mad_descriptors_pin_lane_container_widths() -> None:
    descriptor_sets = (
        (_gfx940_core_overlays(), "OP_SEL_HI"),
        (_gfx950_core_overlays(), "OP_SEL_HI"),
        (_gfx11_core_overlays(), "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL_HI"),
    )
    expected_32_bit_keys = (
        "amdgpu.v_pk_fma_f16",
        "amdgpu.v_pk_mad_i16",
        "amdgpu.v_pk_mad_u16",
    )
    for descriptor_set, op_sel_hi_field in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in expected_32_bit_keys:
            descriptor = descriptors[descriptor_key]
            assert descriptor.encoding_name == "ENC_VOP3P"
            assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
                "VDST",
                "SRC0",
                "SRC1",
                "SRC2",
            )
            assert tuple(
                operand.descriptor_operand.unit_count for operand in descriptor.operands
            ) == (1, 1, 1, 1)
            assert descriptor.fixed_encoding_fields == ((op_sel_hi_field, 0x7),)

    cdna_descriptor_sets = (_gfx940_core_overlays(), _gfx950_core_overlays())
    for descriptor_set in cdna_descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_pk_fma_f32"]
        assert descriptor.encoding_name == "ENC_VOP3P"
        assert tuple(
            operand.descriptor_operand.unit_count for operand in descriptor.operands
        ) == (2, 2, 2, 2)
        assert descriptor.fixed_encoding_fields == (("OP_SEL_HI", 0x7),)

    rdna_descriptor_sets = (
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    )
    for descriptor_set in rdna_descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert "amdgpu.v_pk_fma_f32" not in descriptors


def test_gfx125x_packed_bf16_descriptors_are_arch_scoped() -> None:
    unsupported_descriptor_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
    )
    for descriptor_set in unsupported_descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert "amdgpu.v_pk_add_bf16" not in descriptors
        assert "amdgpu.v_pk_mul_bf16" not in descriptors
        assert "amdgpu.v_pk_fma_bf16" not in descriptors

    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx125x_core_overlays()
    }
    for descriptor_key in ("amdgpu.v_pk_add_bf16", "amdgpu.v_pk_mul_bf16"):
        binary_descriptor = descriptors[descriptor_key]
        assert binary_descriptor.encoding_name == "ENC_VOP3P"
        assert tuple(
            operand.xml_field_name for operand in binary_descriptor.operands
        ) == (
            "VDST",
            "SRC0",
            "SRC1",
        )
        assert tuple(
            operand.descriptor_operand.unit_count
            for operand in binary_descriptor.operands
        ) == (1, 1, 1)
        assert binary_descriptor.fixed_encoding_fields == (("OPSEL_HI", 0x7),)

    fma_descriptor = descriptors["amdgpu.v_pk_fma_bf16"]
    assert fma_descriptor.encoding_name == "ENC_VOP3P"
    assert tuple(operand.xml_field_name for operand in fma_descriptor.operands) == (
        "VDST",
        "SRC0",
        "SRC1",
        "SRC2",
    )
    assert tuple(
        operand.descriptor_operand.unit_count for operand in fma_descriptor.operands
    ) == (1, 1, 1, 1)
    assert fma_descriptor.fixed_encoding_fields == (("OPSEL_HI", 0x7),)


def test_gfx125x_packed_fp8_to_f16_sources_use_low_half_window() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx125x_core_overlays()
    }
    for descriptor_key in (
        "amdgpu.v_cvt_pk_f16_fp8.ocp",
        "amdgpu.v_cvt_pk_f16_bf8.ocp",
    ):
        descriptor = descriptors[descriptor_key]
        assert descriptor.encoding_name == "ENC_VOP1_VGPR"
        source = descriptor.operands[1]
        assert source.xml_field_name == "VSRC0"
        assert source.descriptor_operand.register_part == _REG_PART_VGPR_LOW16
        assert (
            source.descriptor_operand.address_map_kind
            is OperandAddressMapKind.LOW_SUBSET
        )
        assert source.descriptor_operand.addressable_unit_count == 128


def test_packed8_encode_descriptors_own_numeric_and_partial_result_semantics() -> None:
    for descriptor_set, target_semantics, op_sel_field in (
        (_gfx940_core_overlays(), "fnuz", "OP_SEL"),
        (_gfx950_core_overlays(), "ocp", "OP_SEL"),
        (_gfx12_core_overlays(), "ocp", "OPSEL"),
        (_gfx125x_core_overlays(), "ocp", "OPSEL"),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for target_type in ("fp8", "bf8"):
            key_prefix = f"amdgpu.v_cvt_pk_{target_type}_f32.{target_semantics}"
            low_descriptor = descriptors[f"{key_prefix}.low"]
            high_descriptor = descriptors[f"{key_prefix}.high"]

            assert low_descriptor.operands[0].descriptor_operand.register_part == (
                _REG_PART_VGPR_LOW16
            )
            assert low_descriptor.fixed_encoding_fields == ((op_sel_field, 0),)
            assert low_descriptor.constraints == ()

            assert high_descriptor.operands[0].descriptor_operand.register_part == (
                _REG_PART_VGPR_HIGH16
            )
            accumulator = high_descriptor.operands[1].descriptor_operand
            assert accumulator.register_part == _REG_PART_VGPR_LOW16
            assert OperandFlag.IMPLICIT in accumulator.flags
            assert OperandFlag.STORAGE_CONTINUATION in accumulator.flags
            assert high_descriptor.fixed_encoding_fields == ((op_sel_field, 0b1000),)
            assert tuple(
                constraint.kind for constraint in high_descriptor.constraints
            ) == (ConstraintKind.TIED,)
            assert (
                high_descriptor.asm_forms[0].native_assembly_values[-1].literal
                == "op_sel:[0,0,1]"
            )

    unsupported_f16_descriptor_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx12_core_overlays(),
    )
    for descriptor_set in unsupported_f16_descriptor_sets:
        descriptor_keys = {descriptor.descriptor_key for descriptor in descriptor_set}
        assert "amdgpu.v_cvt_pk_fp8_f16.ocp.low" not in descriptor_keys
        assert "amdgpu.v_cvt_pk_bf8_f16.ocp.low" not in descriptor_keys

    gfx125x_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx125x_core_overlays()
    }
    for target_type in ("fp8", "bf8"):
        key_prefix = f"amdgpu.v_cvt_pk_{target_type}_f16.ocp"
        low_descriptor = gfx125x_descriptors[f"{key_prefix}.low"]
        high_descriptor = gfx125x_descriptors[f"{key_prefix}.high"]
        assert low_descriptor.fixed_encoding_fields == (("OPSEL", 0),)
        assert high_descriptor.fixed_encoding_fields == (("OPSEL", 0b1000),)
        assert (
            high_descriptor.asm_forms[0].native_assembly_values[-1].literal
            == "op_sel:[0,1]"
        )


def test_packed_binary_descriptors_pin_lane_container_widths() -> None:
    packed_keys = (
        "amdgpu.v_pk_add_f16",
        "amdgpu.v_pk_mul_f16",
        "amdgpu.v_pk_minnum_f16",
        "amdgpu.v_pk_maxnum_f16",
        "amdgpu.v_pk_add_u16",
        "amdgpu.v_pk_sub_i16",
        "amdgpu.v_pk_mul_lo_u16",
        "amdgpu.v_pk_min_i16",
        "amdgpu.v_pk_max_i16",
        "amdgpu.v_pk_min_u16",
        "amdgpu.v_pk_max_u16",
        "amdgpu.v_pk_lshlrev_b16",
        "amdgpu.v_pk_lshrrev_b16",
        "amdgpu.v_pk_ashrrev_i16",
    )
    for descriptor_set, op_sel_hi_field in (
        (_gfx940_core_overlays(), "OP_SEL_HI"),
        (_gfx950_core_overlays(), "OP_SEL_HI"),
        (_gfx11_core_overlays(), "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL_HI"),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in packed_keys:
            descriptor = descriptors[descriptor_key]
            assert descriptor.encoding_name == "ENC_VOP3P"
            assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
                "VDST",
                "SRC0",
                "SRC1",
            )
            assert tuple(
                operand.descriptor_operand.unit_count for operand in descriptor.operands
            ) == (1, 1, 1)
            assert descriptor.fixed_encoding_fields == ((op_sel_hi_field, 0x7),)


def test_packed_float_descriptors_follow_target_numeric_semantics() -> None:
    for descriptor_set, minimum_name, maximum_name in (
        (_gfx940_core_overlays(), "V_PK_MIN_F16", "V_PK_MAX_F16"),
        (_gfx950_core_overlays(), "V_PK_MIN_F16", "V_PK_MAX_F16"),
        (_gfx11_core_overlays(), "V_PK_MIN_F16", "V_PK_MAX_F16"),
        (_gfx12_core_overlays(), "V_PK_MIN_NUM_F16", "V_PK_MAX_NUM_F16"),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        assert descriptors["amdgpu.v_pk_minnum_f16"].instruction_name == minimum_name
        assert descriptors["amdgpu.v_pk_maxnum_f16"].instruction_name == maximum_name

    rdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx12_core_overlays()
    }
    for descriptor_key, instruction_name in (
        ("amdgpu.v_pk_minimum_f16", "V_PK_MINIMUM_F16"),
        ("amdgpu.v_pk_maximum_f16", "V_PK_MAXIMUM_F16"),
    ):
        descriptor = rdna4_descriptors[descriptor_key]
        assert descriptor.instruction_name == instruction_name
        assert tuple(
            operand.descriptor_operand.unit_count for operand in descriptor.operands
        ) == (1, 1, 1)
        assert descriptor.fixed_encoding_fields == (("OPSEL_HI", 0x7),)

    for descriptor_set in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in (
            "amdgpu.v_pk_add_f32",
            "amdgpu.v_pk_mul_f32",
        ):
            descriptor = descriptors[descriptor_key]
            assert tuple(
                operand.descriptor_operand.unit_count for operand in descriptor.operands
            ) == (2, 2, 2)
            assert descriptor.fixed_encoding_fields == (("OP_SEL_HI", 0x7),)


def test_f32_med3_descriptors_follow_target_numeric_semantics() -> None:
    for descriptor_set, instruction_name, mnemonic in (
        (_gfx940_core_overlays(), "V_MED3_F32", "v_med3_f32"),
        (_gfx950_core_overlays(), "V_MED3_F32", "v_med3_f32"),
        (_gfx11_core_overlays(), "V_MED3_F32", "v_med3_f32"),
        (_gfx12_core_overlays(), "V_MED3_NUM_F32", "v_med3_num_f32"),
        (_gfx125x_core_overlays(), "V_MED3_NUM_F32", "v_med3_num_f32"),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_med3_num_f32"]
        assert descriptor.instruction_name == instruction_name
        assert descriptor.mnemonic == mnemonic
        assert descriptor.semantic_tag == "float.med3_num.f32"
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "VDST",
            "SRC0",
            "SRC1",
            "SRC2",
        )


def test_packed_fma_mad_rdna_literal_forms_cover_source_positions() -> None:
    source_fields = {
        "src0": ("SRC0", "a", ("VDST", "SRC1", "SRC2")),
        "src1": ("SRC1", "b", ("VDST", "SRC0", "SRC2")),
        "src2": ("SRC2", "c", ("VDST", "SRC0", "SRC1")),
    }
    packed_keys = (
        "amdgpu.v_pk_fma_f16",
        "amdgpu.v_pk_mad_i16",
        "amdgpu.v_pk_mad_u16",
    )
    for descriptor_set, op_sel_hi_field in (
        (_gfx11_core_overlays(), "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL_HI"),
    ):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in packed_keys:
            descriptor = descriptors[descriptor_key]
            assert tuple(
                form.replacement_descriptor for form in descriptor.operand_forms
            ) == tuple(
                f"{descriptor_key}.{source_name}_lit" for source_name in source_fields
            )
            assert tuple(
                form.matches[0].source_operand for form in descriptor.operand_forms
            ) == tuple(
                source_operand for _, source_operand, _ in source_fields.values()
            )
            for source_name, (
                literal_field,
                _,
                expected_operand_fields,
            ) in source_fields.items():
                literal_descriptor = descriptors[f"{descriptor_key}.{source_name}_lit"]
                assert literal_descriptor.encoding_name == "ENC_VOP3P"
                assert (
                    literal_descriptor.encoding_format_id
                    == AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL
                )
                assert (
                    tuple(
                        operand.xml_field_name
                        for operand in literal_descriptor.operands
                    )
                    == expected_operand_fields
                )
                assert tuple(
                    immediate.field_name for immediate in literal_descriptor.immediates
                ) == ("imm32",)
                assert literal_descriptor.fixed_encoding_fields[0] == (
                    op_sel_hi_field,
                    0x7,
                )
                fixed_field, fixed_value = literal_descriptor.fixed_encoding_fields[1]
                assert fixed_field == literal_field
                assert isinstance(fixed_value, AmdgpuOperandPredefinedValueRef)
                assert fixed_value.value_name == "SRC_LITERAL"

    for descriptor_set in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in packed_keys:
            assert descriptors[descriptor_key].operand_forms == ()
            assert not any(
                key.startswith(f"{descriptor_key}.") and key.endswith("_lit")
                for key in descriptors
            )


def test_packed_fmac_f16_descriptor_pins_destructive_accumulator() -> None:
    descriptor_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    )
    for descriptor_set in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        descriptor = descriptors["amdgpu.v_pk_fmac_f16"]
        assert descriptor.encoding_name == "ENC_VOP2"
        assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
            "VDST",
            "VDST",
            "SRC0",
            "VSRC1",
        )
        assert tuple(
            operand.descriptor_operand.field_name for operand in descriptor.operands
        ) == ("dst", "acc", "a", "b")
        assert tuple(constraint.kind for constraint in descriptor.constraints) == (
            ConstraintKind.TIED,
            ConstraintKind.DESTRUCTIVE,
        )
        assert tuple(
            (constraint.lhs_operand_index, constraint.rhs_operand_index)
            for constraint in descriptor.constraints
        ) == ((0, 1), (0, 1))


def test_packed_dot2_descriptors_pin_destructive_accumulator() -> None:
    descriptor_sets = (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    )
    for descriptor_set in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key in (
            "amdgpu.v_dot2_f32_f16",
            "amdgpu.v_dot2_f32_bf16",
        ):
            if descriptor_key not in descriptors:
                continue
            descriptor = descriptors[descriptor_key]
            assert descriptor.encoding_name == "ENC_VOP3P"
            assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
                "VDST",
                "SRC0",
                "SRC1",
                "SRC2",
            )
            assert tuple(
                operand.descriptor_operand.field_name for operand in descriptor.operands
            ) == ("dst", "lhs", "rhs", "acc")
            constraint_contracts = {
                (
                    constraint.kind,
                    constraint.lhs_operand_index,
                    constraint.rhs_operand_index,
                )
                for constraint in descriptor.constraints
            }
            assert (ConstraintKind.TIED, 0, 3) in constraint_contracts
            assert (ConstraintKind.DESTRUCTIVE, 0, 3) in constraint_contracts
            assert (ConstraintKind.COMMUTABLE, 1, 2) in constraint_contracts


def _expected_mix_descriptor_keys(
    descriptor_key_prefix: str, *, include_all_f32: bool
) -> set[str]:
    return {
        f"{descriptor_key_prefix}.{source0}_{source1}_{source2}"
        for source0 in ("f32", "f16lo", "f16hi")
        for source1 in ("f32", "f16lo", "f16hi")
        for source2 in ("f32", "f16lo", "f16hi")
        if include_all_f32 or (source0, source1, source2) != ("f32", "f32", "f32")
    }


def _assert_mix_descriptor_sources(
    descriptor: AmdgpuDescriptorOverlay,
    source_parts: list[str],
    *,
    op_sel_field: str,
    op_sel_hi_field: str,
    source_operand_start: int = 1,
) -> None:
    expected_op_sel = 0
    expected_op_sel_hi = 0
    for source_index, source_part in enumerate(source_parts):
        operand = descriptor.operands[
            source_operand_start + source_index
        ].descriptor_operand
        if source_part == "f16lo":
            expected_op_sel_hi |= 1 << source_index
            assert operand.register_part == _REG_PART_VGPR_LOW16
        elif source_part == "f16hi":
            expected_op_sel |= 1 << source_index
            expected_op_sel_hi |= 1 << source_index
            assert operand.register_part == _REG_PART_VGPR_HIGH16
        else:
            assert source_part == "f32"
            assert operand.register_part is None
    assert descriptor.fixed_encoding_fields == (
        (op_sel_field, expected_op_sel),
        (op_sel_hi_field, expected_op_sel_hi),
    )


def _assert_mix_descriptor_family(
    descriptors: dict[str, AmdgpuDescriptorOverlay],
    descriptor_key_prefix: str,
    *,
    include_all_f32: bool,
    op_sel_field: str,
    op_sel_hi_field: str,
    result_register_part: str | None,
    tied_half_result: bool = False,
) -> None:
    expected_keys = _expected_mix_descriptor_keys(
        descriptor_key_prefix, include_all_f32=include_all_f32
    )
    actual_keys = {
        key
        for key in descriptors
        if key.startswith(f"{descriptor_key_prefix}.") and not key.endswith(".src2_lit")
    }
    assert actual_keys == expected_keys
    for descriptor_key in expected_keys:
        descriptor = descriptors[descriptor_key]
        source_parts = descriptor_key.removeprefix(f"{descriptor_key_prefix}.").split(
            "_"
        )
        assert descriptor.operands[0].descriptor_operand.register_part == (
            result_register_part
        )
        source_operand_start = 1
        if tied_half_result:
            assert tuple(operand.xml_field_name for operand in descriptor.operands) == (
                "VDST",
                "VDST",
                "SRC0",
                "SRC1",
                "SRC2",
            )
            acc = descriptor.operands[1].descriptor_operand
            assert acc.field_name == "acc"
            assert acc.role is OperandRole.OPERAND
            assert OperandFlag.IMPLICIT in acc.flags
            assert descriptor.operands[1].size_exception_reason is not None
            assert tuple(constraint.kind for constraint in descriptor.constraints) == (
                ConstraintKind.TIED,
                ConstraintKind.DESTRUCTIVE,
            )
            assert tuple(
                (constraint.lhs_operand_index, constraint.rhs_operand_index)
                for constraint in descriptor.constraints
            ) == ((0, 1), (0, 1))
            assert descriptor.asm_forms is not None
            assert descriptor.asm_forms[0].operands == ("acc", "a", "b", "c")
            source_operand_start = 2
        else:
            assert tuple(constraint.kind for constraint in descriptor.constraints) == ()
        _assert_mix_descriptor_sources(
            descriptor,
            source_parts,
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
            source_operand_start=source_operand_start,
        )


def test_fma_mix_f32_half_lane_descriptors_pin_modifier_fields() -> None:
    rdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    rdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx12_core_overlays()
    }
    gfx125x_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx125x_core_overlays()
    }
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }

    for descriptors, op_sel_field, op_sel_hi_field in (
        (rdna3_descriptors, "OP_SEL", "OP_SEL_HI"),
        (rdna4_descriptors, "OPSEL", "OPSEL_HI"),
        (gfx125x_descriptors, "OPSEL", "OPSEL_HI"),
    ):
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_fma_mix_f32",
            include_all_f32=False,
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
            result_register_part=None,
        )
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_fma_mixlo_f16",
            include_all_f32=True,
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
            result_register_part=_REG_PART_VGPR_LOW16,
            tied_half_result=True,
        )
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_fma_mixhi_f16",
            include_all_f32=True,
            op_sel_field=op_sel_field,
            op_sel_hi_field=op_sel_hi_field,
            result_register_part=_REG_PART_VGPR_HIGH16,
            tied_half_result=True,
        )

    for descriptor_key_prefix in (
        "amdgpu.v_fma_mix_f32",
        "amdgpu.v_fma_mixlo_f16",
        "amdgpu.v_fma_mixhi_f16",
    ):
        assert not any(
            key.startswith(f"{descriptor_key_prefix}.") for key in cdna3_descriptors
        )
        assert not any(
            key.startswith(f"{descriptor_key_prefix}.") for key in cdna4_descriptors
        )


def test_vinterp_descriptors_cover_architectural_half_register_forms() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    for phase in ("p10", "p2"):
        descriptor = descriptors[f"amdgpu.v_interp_{phase}_f32"]
        assert descriptor.instruction_name == f"V_INTERP_{phase.upper()}_F32"
        assert dict(descriptor.fixed_encoding_fields) == {
            "WAIT_EXP": 7,
            "OP_SEL": 0,
            "CLAMP": 0,
            "NEG": 0,
        }

    for operation, source_name, source_shift in (
        ("p10", "p0", 2),
        ("p2", "result", 3),
    ):
        for rounding in ("", "rtz_"):
            mnemonic = f"v_interp_{operation}_{rounding}f16_f32"
            for first_part, first_register_part, first_op_sel in (
                ("lo", _REG_PART_VGPR_LOW16, 0),
                ("hi", _REG_PART_VGPR_HIGH16, 1),
            ):
                for second_part, second_register_part, second_op_sel in (
                    ("lo", _REG_PART_VGPR_LOW16, 0),
                    ("hi", _REG_PART_VGPR_HIGH16, 1),
                ):
                    descriptor = descriptors[
                        f"amdgpu.{mnemonic}."
                        f"{operation if operation == 'p10' else 'p20'}_{first_part}."
                        f"{source_name}_{second_part}"
                    ]
                    if operation == "p10":
                        assert descriptor.operands[
                            1
                        ].descriptor_operand.register_part == (first_register_part)
                        assert descriptor.operands[
                            3
                        ].descriptor_operand.register_part == (second_register_part)
                    else:
                        assert descriptor.operands[
                            0
                        ].descriptor_operand.register_part == (second_register_part)
                        assert descriptor.operands[
                            1
                        ].descriptor_operand.register_part == (first_register_part)
                    assert dict(descriptor.fixed_encoding_fields)["OP_SEL"] == (
                        first_op_sel | (second_op_sel << source_shift)
                    )
                    assert len(descriptor.implicit_operands) == 1
                    m0_operand = descriptor.implicit_operands[0].descriptor_operand
                    assert m0_operand is not None
                    assert m0_operand.reg_alts[0].reg_class == _REG_M0
                    assert OperandFlag.IMPLICIT in m0_operand.flags
                    assert OperandFlag.STATE_READ in m0_operand.flags
                    assert descriptor.asm_forms is not None
                    assert descriptor.asm_forms[0].native_assembly_mnemonic == mnemonic
                    native_values = descriptor.asm_forms[0].native_assembly_values
                    assert native_values[-1] == NativeAsmValue(
                        NativeAsmValueKind.MODIFIER_LITERAL,
                        literal="wait_exp:7",
                    )
                    register_part_fields = (
                        ("p10", "p0") if operation == "p10" else ("dst", "p20")
                    )
                    assert {
                        value.field_name
                        for value in native_values
                        if value.kind is NativeAsmValueKind.REGISTER_PART
                    } == set(register_part_fields)


def test_vinterp_descriptors_follow_hardware_architecture_coverage() -> None:
    expected_keys = {
        "amdgpu.v_interp_p10_f32",
        "amdgpu.v_interp_p2_f32",
    }
    for mnemonic, first_source_name, second_source_name in (
        ("v_interp_p10_f16_f32", "p10", "p0"),
        ("v_interp_p2_f16_f32", "p20", "result"),
        ("v_interp_p10_rtz_f16_f32", "p10", "p0"),
        ("v_interp_p2_rtz_f16_f32", "p20", "result"),
    ):
        expected_keys.update(
            f"amdgpu.{mnemonic}.{first_source_name}_{first_part}."
            f"{second_source_name}_{second_part}"
            for first_part in ("lo", "hi")
            for second_part in ("lo", "hi")
        )

    for overlays in (
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx11_generic_core_overlays(),
        _gfx12_core_overlays(),
        _gfx12_generic_core_overlays(),
    ):
        assert {
            overlay.descriptor_key
            for overlay in overlays
            if overlay.descriptor_key.startswith("amdgpu.v_interp_")
        } == expected_keys

    for overlays in (
        _gfx125x_core_overlays(),
        _gfx12_5_generic_core_overlays(),
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
    ):
        assert not any(
            overlay.descriptor_key.startswith("amdgpu.v_interp_")
            for overlay in overlays
        )


def test_fma_mix_f32_source2_literal_forms_cover_full_f32_addends() -> None:
    descriptor_sets = (
        (_gfx11_core_overlays(), "OP_SEL", "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL", "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL", "OPSEL_HI"),
    )
    for descriptor_set, op_sel_field, op_sel_hi_field in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for source0 in ("f32", "f16lo", "f16hi"):
            for source1 in ("f32", "f16lo", "f16hi"):
                if (source0, source1) == ("f32", "f32"):
                    continue
                descriptor_key = f"amdgpu.v_fma_mix_f32.{source0}_{source1}_f32"
                literal_key = f"{descriptor_key}.src2_lit"
                descriptor = descriptors[descriptor_key]
                literal_descriptor = descriptors[literal_key]
                assert tuple(
                    form.replacement_descriptor for form in descriptor.operand_forms
                ) == (literal_key,)
                assert literal_descriptor.encoding_name == "ENC_VOP3P"
                assert (
                    literal_descriptor.encoding_format_id
                    == AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL
                )
                assert tuple(
                    operand.xml_field_name for operand in literal_descriptor.operands
                ) == ("VDST", "SRC0", "SRC1")
                assert tuple(
                    immediate.field_name for immediate in literal_descriptor.immediates
                ) == ("imm32",)
                src2_field, src2_value = literal_descriptor.fixed_encoding_fields[2]
                assert literal_descriptor.fixed_encoding_fields[0][0] == op_sel_field
                assert literal_descriptor.fixed_encoding_fields[1][0] == op_sel_hi_field
                assert src2_field == "SRC2"
                assert isinstance(src2_value, AmdgpuOperandPredefinedValueRef)
                assert src2_value.value_name == "SRC_LITERAL"

        assert not any(
            key.startswith("amdgpu.v_fma_mix_f32.") and key.endswith("_f16lo.src2_lit")
            for key in descriptors
        )
        assert not any(
            key.startswith("amdgpu.v_fma_mix_f32.") and key.endswith("_f16hi.src2_lit")
            for key in descriptors
        )


def test_fma_mix_half_result_source2_literal_forms_cover_zero_addends() -> None:
    descriptor_sets = (
        (_gfx11_core_overlays(), "OP_SEL", "OP_SEL_HI"),
        (_gfx12_core_overlays(), "OPSEL", "OPSEL_HI"),
        (_gfx125x_core_overlays(), "OPSEL", "OPSEL_HI"),
    )
    for descriptor_set, op_sel_field, op_sel_hi_field in descriptor_sets:
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key_prefix, result_register_part in (
            ("amdgpu.v_fma_mixlo_f16", _REG_PART_VGPR_LOW16),
            ("amdgpu.v_fma_mixhi_f16", _REG_PART_VGPR_HIGH16),
        ):
            for source0 in ("f32", "f16lo", "f16hi"):
                for source1 in ("f32", "f16lo", "f16hi"):
                    descriptor_key = f"{descriptor_key_prefix}.{source0}_{source1}_f32"
                    literal_key = f"{descriptor_key}.src2_lit"
                    descriptor = descriptors[descriptor_key]
                    literal_descriptor = descriptors[literal_key]
                    assert tuple(
                        form.replacement_descriptor for form in descriptor.operand_forms
                    ) == (literal_key,)
                    assert literal_descriptor.encoding_name == "ENC_VOP3P"
                    assert (
                        literal_descriptor.encoding_format_id
                        == AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL
                    )
                    assert tuple(
                        operand.xml_field_name
                        for operand in literal_descriptor.operands
                    ) == ("VDST", "VDST", "SRC0", "SRC1")
                    assert (
                        literal_descriptor.operands[0].descriptor_operand.register_part
                        == result_register_part
                    )
                    assert tuple(
                        constraint.kind for constraint in literal_descriptor.constraints
                    ) == (ConstraintKind.TIED, ConstraintKind.DESTRUCTIVE)
                    assert literal_descriptor.asm_forms is not None
                    assert literal_descriptor.asm_forms[0].operands == (
                        "acc",
                        "a",
                        "b",
                    )
                    assert tuple(
                        immediate.field_name
                        for immediate in literal_descriptor.asm_forms[0].immediates
                    ) == ("imm32",)
                    assert tuple(
                        immediate.field_name
                        for immediate in literal_descriptor.immediates
                    ) == ("imm32",)
                    src2_field, src2_value = literal_descriptor.fixed_encoding_fields[2]
                    assert (
                        literal_descriptor.fixed_encoding_fields[0][0] == op_sel_field
                    )
                    assert (
                        literal_descriptor.fixed_encoding_fields[1][0]
                        == op_sel_hi_field
                    )
                    assert src2_field == "SRC2"
                    assert isinstance(src2_value, AmdgpuOperandPredefinedValueRef)
                    assert src2_value.value_name == "SRC_LITERAL"

    for descriptor_set in (_gfx940_core_overlays(), _gfx950_core_overlays()):
        descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in descriptor_set
        }
        for descriptor_key_prefix in (
            "amdgpu.v_mad_mixlo_f16",
            "amdgpu.v_mad_mixhi_f16",
        ):
            assert f"{descriptor_key_prefix}.f32_f32_f32" in descriptors
            assert not any(
                key.startswith(f"{descriptor_key_prefix}.")
                and key.endswith(".src2_lit")
                for key in descriptors
            )


def test_mad_mix_descriptors_cover_cdna_half_lane_forms() -> None:
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }
    rdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    rdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx12_core_overlays()
    }
    gfx125x_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx125x_core_overlays()
    }

    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_mad_mix_f32",
            include_all_f32=False,
            op_sel_field="OP_SEL",
            op_sel_hi_field="OP_SEL_HI",
            result_register_part=None,
        )
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_mad_mixlo_f16",
            include_all_f32=True,
            op_sel_field="OP_SEL",
            op_sel_hi_field="OP_SEL_HI",
            result_register_part=_REG_PART_VGPR_LOW16,
            tied_half_result=True,
        )
        _assert_mix_descriptor_family(
            descriptors,
            "amdgpu.v_mad_mixhi_f16",
            include_all_f32=True,
            op_sel_field="OP_SEL",
            op_sel_hi_field="OP_SEL_HI",
            result_register_part=_REG_PART_VGPR_HIGH16,
            tied_half_result=True,
        )

    for descriptors in (rdna3_descriptors, rdna4_descriptors, gfx125x_descriptors):
        for descriptor_key_prefix in (
            "amdgpu.v_mad_mix_f32",
            "amdgpu.v_mad_mixlo_f16",
            "amdgpu.v_mad_mixhi_f16",
        ):
            assert not any(
                key.startswith(f"{descriptor_key_prefix}.") for key in descriptors
            )


def test_dpp_control_domain_covers_only_architectural_encodings() -> None:
    expected_values = {
        *range(0x100),
        *range(0x101, 0x110),
        *range(0x111, 0x120),
        *range(0x121, 0x130),
        0x130,
        0x134,
        0x138,
        0x13C,
        *range(0x140, 0x144),
        *range(0x150, 0x160),
        *range(0x160, 0x170),
    }
    for value in range(-1, 0x201):
        assert amdgpu_dpp_control_is_valid(value) == (value in expected_values)


def test_dpp_control_validation_accepts_direct_immediate() -> None:
    dpp_control_field_id = amdgpu_encoding_field_id("DPP_CTRL")
    descriptor = _dpp_descriptor(
        immediates=(
            Immediate(
                "dpp_ctrl",
                ImmediateKind.UNSIGNED,
                bit_width=9,
                encoding_field_id=dpp_control_field_id,
                unsigned_max=0x1FF,
            ),
        )
    )

    _validate_dpp_control_fields(_descriptor_set(descriptor))


def test_dpp_control_validation_rejects_missing_source() -> None:
    _expect_value_error_contains(
        "exactly one DPP_CTRL source; found 0",
        lambda: _validate_dpp_control_fields(_descriptor_set(_dpp_descriptor())),
    )


def test_dpp_control_validation_rejects_reserved_fixed_value() -> None:
    dpp_control_field_id = amdgpu_encoding_field_id("DPP_CTRL")
    descriptor = _dpp_descriptor(
        encoding_field_values=(EncodingFieldValue(dpp_control_field_id, 0x100),)
    )

    _expect_value_error_contains(
        "reserved value 256",
        lambda: _validate_dpp_control_fields(_descriptor_set(descriptor)),
    )


def test_dpp_control_validation_rejects_reserved_default_value() -> None:
    dpp_control_field_id = amdgpu_encoding_field_id("DPP_CTRL")
    descriptor = _dpp_descriptor(
        immediates=(
            Immediate(
                "dpp_ctrl",
                ImmediateKind.UNSIGNED,
                flags=(ImmediateFlag.DEFAULT_VALUE,),
                bit_width=9,
                encoding_field_id=dpp_control_field_id,
                unsigned_max=0x1FF,
                default_value=0x110,
            ),
        )
    )

    _expect_value_error_contains(
        "reserved default value 272",
        lambda: _validate_dpp_control_fields(_descriptor_set(descriptor)),
    )


def test_address_immediate_validation_rejects_missing_unit_metadata() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                unsigned_max=255,
            ),
        )
    )

    _expect_value_error_contains(
        "no address-unit encoding",
        lambda: _validate_address_immediate_units(_descriptor_set(descriptor)),
    )


def test_address_immediate_validation_rejects_inconsistent_split_units() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset0",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                encoding_id=_ADDRESS_OFFSET_DWORD_ENCODING_ID,
                unsigned_max=255,
            ),
            Immediate(
                "offset1",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                encoding_id=_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
                unsigned_max=255,
            ),
        )
    )

    _expect_value_error_contains(
        "inconsistent split address offset units",
        lambda: _validate_address_immediate_units(_descriptor_set(descriptor)),
    )


def test_address_immediate_validation_accepts_split_ds16_offset() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                encoding_id=_ADDRESS_OFFSET_DS16_ENCODING_ID,
                unsigned_max=65535,
            ),
        )
    )

    _validate_address_immediate_units(_descriptor_set(descriptor))


def test_plain_ds_memory_offsets_use_split_16_bit_byte_immediates() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }

    for descriptor_key in (
        "amdgpu.ds_read_b128",
        "amdgpu.ds_write_b128",
        "amdgpu.ds_add_u32",
        "amdgpu.ds_cmpst_rtn_b32",
        "amdgpu.ds_read_addtid_b32",
        "amdgpu.ds_write_addtid_b32",
    ):
        descriptor = descriptors[descriptor_key]
        assert len(descriptor.immediates) == 1
        immediate = descriptor.immediates[0]
        assert immediate.field_name == "offset"
        assert immediate.bit_width == 16
        assert immediate.encoding_id == _ADDRESS_OFFSET_DS16_ENCODING_ID
        assert immediate.unsigned_max == 65535
        assert tuple(
            (encoding_slice.source_bit_offset, encoding_slice.bit_count)
            for encoding_slice in immediate.encoding_slices
        ) == ((0, 8), (8, 8))


def test_global_vaddr_memory_forms_have_unique_low_asm_mnemonics() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }

    load_forms = descriptors["amdgpu.global_load_b128"].asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "global_load_b128_vaddr"
    assert load_form.results == ("dst",)
    assert load_form.operands == ("addr",)
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    store_forms = descriptors["amdgpu.global_store_b128"].asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "global_store_b128_vaddr"
    assert store_form.results == ()
    assert store_form.operands == ("addr", "value")
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    saddr_load_forms = descriptors["amdgpu.global_load_b128_saddr"].asm_forms
    assert saddr_load_forms is not None
    saddr_load_form = saddr_load_forms[0]
    assert saddr_load_form.mnemonic == "global_load_b128_saddr"
    assert saddr_load_form.results == ("dst",)
    assert saddr_load_form.operands == ("addr", "saddr")
    assert tuple(immediate.name for immediate in saddr_load_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    saddr_store_forms = descriptors["amdgpu.global_store_b128_saddr"].asm_forms
    assert saddr_store_forms is not None
    saddr_store_form = saddr_store_forms[0]
    assert saddr_store_form.mnemonic == "global_store_b128_saddr"
    assert saddr_store_form.results == ()
    assert saddr_store_form.operands == ("addr", "value", "saddr")
    assert tuple(immediate.name for immediate in saddr_store_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )


def test_gfx950_global_saddr_memory_asm_forms_include_m0() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }

    load_forms = descriptors["amdgpu.global_load_b128_saddr"].asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "global_load_dwordx4_saddr"
    assert load_form.results == ("dst",)
    assert load_form.operands == ("addr", "saddr", "m0")
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )

    store_forms = descriptors["amdgpu.global_store_b128_saddr"].asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "global_store_dwordx4_saddr"
    assert store_form.results == ()
    assert store_form.operands == ("addr", "value", "saddr", "m0")
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )


def _assert_memory_width_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    width_bits: int,
    semantic_tag: str,
    mnemonic: str,
    operand_units: int,
    payload_field_name: str,
    effect_kind: EffectKind,
    memory_space: MemorySpace,
    implicit_data_format: str | None = None,
    implicit_ignore_reason: str | None = None,
) -> None:
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == mnemonic
    payload_operand = next(
        operand.descriptor_operand
        for operand in descriptor.operands
        if operand.descriptor_operand.field_name == payload_field_name
    )
    assert payload_operand.unit_count == operand_units
    assert descriptor.effects == (
        Effect(
            effect_kind,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            width_bits=width_bits,
        ),
    )
    assert any(
        operand.operand_type == "OPR_GPUMEM"
        and operand.data_format_name
        == (implicit_data_format or f"FMT_NUM_B{width_bits}")
        and operand.size_bits == width_bits
        and (
            implicit_ignore_reason is None
            or operand.ignore_reason == implicit_ignore_reason
        )
        for operand in descriptor.implicit_operands
    )


def _assert_memory_vaddr_offset_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    width_bits: int,
    semantic_tag: str,
    mnemonic: str,
    operand_units: int,
    payload_field_name: str,
    effect_kind: EffectKind,
    memory_space: MemorySpace,
    implicit_data_format: str | None = None,
    implicit_ignore_reason: str | None = None,
    offset_field_name: str = "OFFSET",
    fixed_soffset: AmdgpuOperandPredefinedValueRef = _MUBUF_SOFFSET_INLINE_ZERO,
    fixed_soffset_native_spelling: str = "0",
) -> None:
    _assert_memory_width_overlay(
        descriptor,
        width_bits=width_bits,
        semantic_tag=semantic_tag,
        mnemonic=mnemonic,
        operand_units=operand_units,
        payload_field_name=payload_field_name,
        effect_kind=effect_kind,
        memory_space=memory_space,
        implicit_data_format=implicit_data_format,
        implicit_ignore_reason=implicit_ignore_reason,
    )
    assert (
        "SOFFSET",
        fixed_soffset,
    ) in descriptor.fixed_encoding_fields
    assert ("IDXEN", 0) in descriptor.fixed_encoding_fields
    assert ("OFFEN", 1) in descriptor.fixed_encoding_fields
    assert descriptor.immediate_fields[0] == offset_field_name
    forms = descriptor.asm_forms
    assert forms is not None
    form = forms[0]
    assert form.mnemonic == f"{mnemonic}_vaddr_offset"
    assert form.native_assembly_mnemonic == mnemonic
    payload_value_kind = (
        NativeAsmValueKind.RESULT
        if effect_kind == EffectKind.READ
        else NativeAsmValueKind.OPERAND
    )
    assert form.native_assembly_values == (
        NativeAsmValue(payload_value_kind, field_name=payload_field_name),
        NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="vaddr"),
        NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="resource"),
        NativeAsmValue(
            NativeAsmValueKind.LITERAL,
            literal=f"{fixed_soffset_native_spelling} offen",
        ),
    )
    assert form.results == (("dst",) if effect_kind == EffectKind.READ else ())
    assert form.operands == (
        ("resource", "vaddr")
        if effect_kind == EffectKind.READ
        else ("value", "resource", "vaddr")
    )
    assert form.immediates[0].name == "offset"


def _assert_global_load_lds_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    global_width_bits: int,
    workgroup_width_bits: int,
    semantic_tag: str,
    mnemonic: str,
) -> None:
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == mnemonic
    assert descriptor.schedule_class == _SCHEDULE_VMEM_LOAD_LDS
    assert descriptor.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=global_width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=workgroup_width_bits,
        ),
    )
    assert any(
        operand.xml_field_name == "VDST"
        and operand.ignore_reason == "legacy-lds-dma-has-no-vgpr-result"
        for operand in descriptor.ignored_operands
    )
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and OperandFlag.IMPLICIT in operand.descriptor_operand.flags
        and OperandFlag.STATE_READ in operand.descriptor_operand.flags
        for operand in descriptor.implicit_operands
    )


def _assert_buffer_load_lds_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    global_width_bits: int,
    workgroup_width_bits: int,
    semantic_tag: str,
    native_mnemonic: str,
    asm_mnemonic: str,
    implicit_data_format: str,
    address_form: str,
) -> None:
    expected_operands_by_form = {
        "full": (3, asm_mnemonic, ("resource", "vaddr", "soffset", "m0")),
        "off_zero": (1, f"{asm_mnemonic}_off_zero", ("resource", "m0")),
        "vaddr_offset": (
            2,
            f"{asm_mnemonic}_vaddr_offset",
            ("resource", "vaddr", "m0"),
        ),
    }
    expected_operand_count, expected_form_mnemonic, expected_asm_operands = (
        expected_operands_by_form[address_form]
    )
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == native_mnemonic
    assert descriptor.schedule_class == _SCHEDULE_VMEM_LOAD_LDS
    assert descriptor.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=global_width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=workgroup_width_bits,
        ),
    )
    assert ("LDS", 1) in descriptor.fixed_encoding_fields
    assert ("IDXEN", 0) in descriptor.fixed_encoding_fields
    assert (
        "OFFEN",
        0 if address_form == "off_zero" else 1,
    ) in descriptor.fixed_encoding_fields
    if address_form != "full":
        assert (
            "SOFFSET",
            _MUBUF_SOFFSET_INLINE_ZERO,
        ) in descriptor.fixed_encoding_fields
    assert any(
        operand.xml_field_name == "VDATA"
        and operand.ignore_reason == "lds-bit-has-no-vgpr-result"
        for operand in descriptor.ignored_operands
    )
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and OperandFlag.IMPLICIT in operand.descriptor_operand.flags
        and OperandFlag.STATE_READ in operand.descriptor_operand.flags
        and not operand.xml_operand_required
        for operand in descriptor.implicit_operands
    )
    assert any(
        operand.operand_type == "OPR_GPUMEM"
        and operand.data_format_name == implicit_data_format
        and operand.size_bits == global_width_bits
        and operand.ignore_reason == "modeled-by-global-read-effect"
        and operand.xml_operand_required
        for operand in descriptor.implicit_operands
    )
    assert len(descriptor.operands) == expected_operand_count
    forms = descriptor.asm_forms
    assert forms is not None
    form = forms[0]
    assert form.mnemonic == expected_form_mnemonic
    assert form.native_assembly_mnemonic == native_mnemonic
    assert form.results == ()
    assert form.operands == expected_asm_operands
    assert form.immediates[0].name == "offset"


def test_dwordx3_memory_descriptors_cover_cdna_and_rdna_families() -> None:
    for (
        descriptors,
        buffer_load_key,
        buffer_store_key,
        buffer_mnemonic,
        global_mnemonic,
        scratch_store_b16_mnemonic,
        offset_field_name,
    ) in (
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx940_core_overlays()
            },
            "amdgpu.buffer_load_dwordx3",
            "amdgpu.buffer_store_dwordx3",
            "buffer_load_dwordx3",
            "global_load_dwordx3_saddr",
            "scratch_store_short",
            "OFFSET",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx950_core_overlays()
            },
            "amdgpu.buffer_load_dwordx3",
            "amdgpu.buffer_store_dwordx3",
            "buffer_load_dwordx3",
            "global_load_dwordx3_saddr",
            "scratch_store_short",
            "OFFSET",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx11_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
            "scratch_store_b16",
            "OFFSET",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx12_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
            "scratch_store_b16",
            "IOFFSET",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx125x_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
            "scratch_store_b16",
            "IOFFSET",
        ),
    ):
        _assert_memory_width_overlay(
            descriptors[buffer_load_key],
            width_bits=96,
            semantic_tag="memory.load.u96",
            mnemonic=buffer_mnemonic,
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
        )
        _assert_memory_width_overlay(
            descriptors[buffer_store_key],
            width_bits=96,
            semantic_tag="memory.store.u96",
            mnemonic=buffer_mnemonic.replace("load", "store"),
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.GLOBAL,
        )
        buffer_load_vaddr_offset_key = f"{buffer_load_key}_vaddr_offset"
        buffer_store_vaddr_offset_key = f"{buffer_store_key}_vaddr_offset"
        fixed_soffset = (
            _VBUFFER_SOFFSET_NULL
            if offset_field_name == "IOFFSET"
            else _MUBUF_SOFFSET_INLINE_ZERO
        )
        fixed_soffset_native_spelling = (
            "null" if offset_field_name == "IOFFSET" else "0"
        )
        if buffer_load_vaddr_offset_key in descriptors:
            expected_load_forms = tuple(
                key
                for key in (f"{buffer_load_key}_off_zero", buffer_load_vaddr_offset_key)
                if key in descriptors
            )
            assert (
                tuple(
                    form.replacement_descriptor
                    for form in descriptors[buffer_load_key].operand_forms
                )
                == expected_load_forms
            )
            _assert_memory_vaddr_offset_overlay(
                descriptors[buffer_load_vaddr_offset_key],
                width_bits=96,
                semantic_tag="memory.load.u96",
                mnemonic=buffer_mnemonic,
                operand_units=3,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                offset_field_name=offset_field_name,
                fixed_soffset=fixed_soffset,
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            )
            expected_store_forms = tuple(
                key
                for key in (
                    f"{buffer_store_key}_off_zero",
                    buffer_store_vaddr_offset_key,
                )
                if key in descriptors
            )
            assert (
                tuple(
                    form.replacement_descriptor
                    for form in descriptors[buffer_store_key].operand_forms
                )
                == expected_store_forms
            )
            _assert_memory_vaddr_offset_overlay(
                descriptors[buffer_store_vaddr_offset_key],
                width_bits=96,
                semantic_tag="memory.store.u96",
                mnemonic=buffer_mnemonic.replace("load", "store"),
                operand_units=3,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
                offset_field_name=offset_field_name,
                fixed_soffset=fixed_soffset,
                fixed_soffset_native_spelling=fixed_soffset_native_spelling,
            )
        else:
            assert descriptors[buffer_load_key].operand_forms == ()
            assert descriptors[buffer_store_key].operand_forms == ()
            assert buffer_store_vaddr_offset_key not in descriptors

        global_load = descriptors["amdgpu.global_load_b96_saddr"]
        _assert_memory_width_overlay(
            global_load,
            width_bits=96,
            semantic_tag="memory.load.u96",
            mnemonic=global_mnemonic.removesuffix("_saddr"),
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
        )
        assert global_load.asm_forms is not None
        assert global_load.asm_forms[0].mnemonic == global_mnemonic

        global_store = descriptors["amdgpu.global_store_b96_saddr"]
        _assert_memory_width_overlay(
            global_store,
            width_bits=96,
            semantic_tag="memory.store.u96",
            mnemonic=global_mnemonic.removesuffix("_saddr").replace("load", "store"),
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.GLOBAL,
        )
        assert global_store.asm_forms is not None
        assert global_store.asm_forms[0].mnemonic == global_mnemonic.replace(
            "load", "store"
        )

        scratch_load = descriptors["amdgpu.scratch_load_b96_vaddr"]
        _assert_memory_width_overlay(
            scratch_load,
            width_bits=96,
            semantic_tag="memory.stack.load.u96",
            mnemonic="scratch_load_b96",
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.STACK,
        )
        assert scratch_load.asm_forms is not None
        assert scratch_load.asm_forms[0].mnemonic == "scratch_load_b96_vaddr"

        scratch_store = descriptors["amdgpu.scratch_store_b96_vaddr"]
        _assert_memory_width_overlay(
            scratch_store,
            width_bits=96,
            semantic_tag="memory.stack.store.u96",
            mnemonic="scratch_store_b96",
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.STACK,
        )
        assert scratch_store.asm_forms is not None
        assert scratch_store.asm_forms[0].mnemonic == "scratch_store_b96_vaddr"

        scratch_store_b16 = descriptors["amdgpu.scratch_store_b16_vaddr"]
        _assert_memory_width_overlay(
            scratch_store_b16,
            width_bits=16,
            semantic_tag="memory.stack.store.u16",
            mnemonic=scratch_store_b16_mnemonic,
            operand_units=1,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.STACK,
        )
        assert scratch_store_b16.asm_forms is not None
        assert (
            scratch_store_b16.asm_forms[0].mnemonic
            == f"{scratch_store_b16_mnemonic}_vaddr"
        )


def test_cdna_global_load_lds_descriptors_cover_extension_rows() -> None:
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }
    base_rows = (
        (
            "ubyte",
            "memory.global_to_workgroup.u8.zero_extend",
            8,
            32,
        ),
        (
            "sbyte",
            "memory.global_to_workgroup.i8.sign_extend",
            8,
            32,
        ),
        (
            "ushort",
            "memory.global_to_workgroup.u16.zero_extend",
            16,
            32,
        ),
        (
            "sshort",
            "memory.global_to_workgroup.i16.sign_extend",
            16,
            32,
        ),
        ("dword", "memory.global_to_workgroup.u32", 32, 32),
    )
    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        for suffix, semantic_tag, global_width_bits, workgroup_width_bits in base_rows:
            for descriptor_key_suffix in ("", "_saddr"):
                descriptor_key = f"amdgpu.global_load_lds_{suffix}"
                _assert_global_load_lds_overlay(
                    descriptors[f"{descriptor_key}{descriptor_key_suffix}"],
                    global_width_bits=global_width_bits,
                    workgroup_width_bits=workgroup_width_bits,
                    semantic_tag=semantic_tag,
                    mnemonic=f"global_load_lds_{suffix}",
                )

    for suffix, width_bits in (("dwordx3", 96), ("dwordx4", 128)):
        for descriptor_key_suffix in ("", "_saddr"):
            descriptor_key = f"amdgpu.global_load_lds_{suffix}"
            assert f"{descriptor_key}{descriptor_key_suffix}" not in cdna3_descriptors
            _assert_global_load_lds_overlay(
                cdna4_descriptors[f"{descriptor_key}{descriptor_key_suffix}"],
                global_width_bits=width_bits,
                workgroup_width_bits=width_bits,
                semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
                mnemonic=f"global_load_lds_{suffix}",
            )


def test_cdna_buffer_load_lds_descriptors_cover_fixed_lds_rows() -> None:
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }
    base_rows = (
        (
            "ubyte",
            "memory.global_to_workgroup.u8.zero_extend",
            8,
            8,
            "FMT_NUM_U8",
        ),
        (
            "sbyte",
            "memory.global_to_workgroup.i8.sign_extend",
            8,
            8,
            "FMT_NUM_I8",
        ),
        (
            "ushort",
            "memory.global_to_workgroup.u16.zero_extend",
            16,
            16,
            "FMT_NUM_U16",
        ),
        (
            "sshort",
            "memory.global_to_workgroup.i16.sign_extend",
            16,
            16,
            "FMT_NUM_I16",
        ),
        ("dword", "memory.global_to_workgroup.u32", 32, 32, "FMT_NUM_B32"),
    )
    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        for (
            suffix,
            semantic_tag,
            global_width_bits,
            workgroup_width_bits,
            implicit_data_format,
        ) in base_rows:
            descriptor_key = f"amdgpu.buffer_load_lds_{suffix}"
            _assert_buffer_load_lds_overlay(
                descriptors[descriptor_key],
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                semantic_tag=semantic_tag,
                native_mnemonic=f"buffer_load_{suffix}",
                asm_mnemonic=f"buffer_load_lds_{suffix}",
                implicit_data_format=implicit_data_format,
                address_form="full",
            )
            _assert_buffer_load_lds_overlay(
                descriptors[f"{descriptor_key}_off_zero"],
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                semantic_tag=semantic_tag,
                native_mnemonic=f"buffer_load_{suffix}",
                asm_mnemonic=f"buffer_load_lds_{suffix}",
                implicit_data_format=implicit_data_format,
                address_form="off_zero",
            )
            _assert_buffer_load_lds_overlay(
                descriptors[f"{descriptor_key}_vaddr_offset"],
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                semantic_tag=semantic_tag,
                native_mnemonic=f"buffer_load_{suffix}",
                asm_mnemonic=f"buffer_load_lds_{suffix}",
                implicit_data_format=implicit_data_format,
                address_form="vaddr_offset",
            )

    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        assert "amdgpu.buffer_load_lds_dwordx2" not in descriptors
        assert "amdgpu.buffer_load_lds_dwordx2_off_zero" not in descriptors
        assert "amdgpu.buffer_load_lds_dwordx2_vaddr_offset" not in descriptors

    for suffix, width_bits in (("dwordx3", 96), ("dwordx4", 128)):
        descriptor_key = f"amdgpu.buffer_load_lds_{suffix}"
        assert descriptor_key not in cdna3_descriptors
        assert f"{descriptor_key}_off_zero" not in cdna3_descriptors
        assert f"{descriptor_key}_vaddr_offset" not in cdna3_descriptors
        _assert_buffer_load_lds_overlay(
            cdna4_descriptors[descriptor_key],
            global_width_bits=width_bits,
            workgroup_width_bits=width_bits,
            semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
            native_mnemonic=f"buffer_load_{suffix}",
            asm_mnemonic=f"buffer_load_lds_{suffix}",
            implicit_data_format=f"FMT_NUM_B{width_bits}",
            address_form="full",
        )
        _assert_buffer_load_lds_overlay(
            cdna4_descriptors[f"{descriptor_key}_off_zero"],
            global_width_bits=width_bits,
            workgroup_width_bits=width_bits,
            semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
            native_mnemonic=f"buffer_load_{suffix}",
            asm_mnemonic=f"buffer_load_lds_{suffix}",
            implicit_data_format=f"FMT_NUM_B{width_bits}",
            address_form="off_zero",
        )
        _assert_buffer_load_lds_overlay(
            cdna4_descriptors[f"{descriptor_key}_vaddr_offset"],
            global_width_bits=width_bits,
            workgroup_width_bits=width_bits,
            semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
            native_mnemonic=f"buffer_load_{suffix}",
            asm_mnemonic=f"buffer_load_lds_{suffix}",
            implicit_data_format=f"FMT_NUM_B{width_bits}",
            address_form="vaddr_offset",
        )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx125x_core_overlays()
        },
    ):
        assert "amdgpu.buffer_load_lds_dword" not in descriptors


def test_smem_dword_width_descriptors_cover_active_xml_families() -> None:
    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx125x_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (256, 8, "amdgpu.s_load_dwordx8", "s_load_dwordx8"),
            (512, 16, "amdgpu.s_load_dwordx16", "s_load_dwordx16"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )
            assert f"{descriptor_key}_offset_only" in descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (128, 4, "amdgpu.s_buffer_load_dwordx4", "s_buffer_load_dwordx4"),
            (256, 8, "amdgpu.s_buffer_load_dwordx8", "s_buffer_load_dwordx8"),
            (512, 16, "amdgpu.s_buffer_load_dwordx16", "s_buffer_load_dwordx16"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx125x_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (128, 4, "amdgpu.s_buffer_load_b128", "s_buffer_load_b128"),
            (256, 8, "amdgpu.s_buffer_load_b256", "s_buffer_load_b256"),
            (512, 16, "amdgpu.s_buffer_load_b512", "s_buffer_load_b512"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx125x_core_overlays()
        },
    ):
        for descriptor_key, mnemonic in (
            ("amdgpu.s_load_b96", "s_load_b96"),
            ("amdgpu.s_buffer_load_b96", "s_buffer_load_b96"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=96,
                semantic_tag="memory.load.u96",
                mnemonic=mnemonic,
                operand_units=3,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )


def test_smem_load_results_are_early_clobber() -> None:
    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        load_descriptors = (
            descriptor
            for descriptor in overlays
            if descriptor.schedule_class == _SCHEDULE_SMEM_LOAD
            and any(
                operand.descriptor_operand.role is OperandRole.RESULT
                for operand in descriptor.operands
            )
        )
        for descriptor in load_descriptors:
            assert descriptor.constraints == _EARLY_CLOBBER_RESULT_CONSTRAINTS, (
                descriptor.descriptor_key
            )


def test_rdna4_smem_narrow_load_descriptors_have_extension_semantics() -> None:
    gfx11_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    rows = (
        ("i8", 8, "memory.load.i8.sign_extend", "FMT_NUM_I8"),
        ("u8", 8, "memory.load.u8.zero_extend", "FMT_NUM_U8"),
        ("i16", 16, "memory.load.i16.sign_extend", "FMT_NUM_I16"),
        ("u16", 16, "memory.load.u16.zero_extend", "FMT_NUM_U16"),
    )
    for suffix, _width_bits, _semantic_tag, _implicit_data_format in rows:
        assert f"amdgpu.s_load_{suffix}" not in gfx11_descriptors
        assert f"amdgpu.s_buffer_load_{suffix}" not in gfx11_descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx125x_core_overlays()
        },
    ):
        for suffix, width_bits, semantic_tag, implicit_data_format in rows:
            scalar_load_key = f"amdgpu.s_load_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scalar_load_key],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )
            _assert_memory_width_overlay(
                descriptors[f"{scalar_load_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )
            _assert_memory_width_overlay(
                descriptors[f"amdgpu.s_buffer_load_{suffix}"],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_buffer_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )


def test_vmem_narrow_load_descriptors_cover_active_xml_families() -> None:
    rows = (
        (
            "u8",
            8,
            "memory.load.u8.zero_extend",
            "memory.stack.load.u8.zero_extend",
            "FMT_NUM_U8",
        ),
        (
            "i8",
            8,
            "memory.load.i8.sign_extend",
            "memory.stack.load.i8.sign_extend",
            "FMT_NUM_I8",
        ),
        (
            "u16",
            16,
            "memory.load.u16.zero_extend",
            "memory.stack.load.u16.zero_extend",
            "FMT_NUM_U16",
        ),
        (
            "i16",
            16,
            "memory.load.i16.sign_extend",
            "memory.stack.load.i16.sign_extend",
            "FMT_NUM_I16",
        ),
    )
    cdna_mnemonic_suffixes = {
        "u8": "ubyte",
        "i8": "sbyte",
        "u16": "ushort",
        "i16": "sshort",
    }
    rdna_mnemonic_suffixes = {suffix: suffix for suffix, *_ in rows}

    for (
        overlays,
        global_mnemonic_suffixes,
        scratch_mnemonic_suffixes,
        buffer_has_off_zero,
        buffer_has_vaddr_offset,
        buffer_offset_field_name,
    ) in (
        (
            _gfx940_core_overlays(),
            cdna_mnemonic_suffixes,
            cdna_mnemonic_suffixes,
            True,
            True,
            "OFFSET",
        ),
        (
            _gfx950_core_overlays(),
            cdna_mnemonic_suffixes,
            cdna_mnemonic_suffixes,
            True,
            True,
            "OFFSET",
        ),
        (
            _gfx11_core_overlays(),
            rdna_mnemonic_suffixes,
            rdna_mnemonic_suffixes,
            True,
            True,
            "OFFSET",
        ),
        (
            _gfx12_core_overlays(),
            rdna_mnemonic_suffixes,
            rdna_mnemonic_suffixes,
            False,
            True,
            "IOFFSET",
        ),
        (
            _gfx125x_core_overlays(),
            rdna_mnemonic_suffixes,
            rdna_mnemonic_suffixes,
            False,
            True,
            "IOFFSET",
        ),
    ):
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in overlays}
        buffer_fixed_soffset = (
            _VBUFFER_SOFFSET_NULL
            if buffer_offset_field_name == "IOFFSET"
            else _MUBUF_SOFFSET_INLINE_ZERO
        )
        buffer_fixed_soffset_native_spelling = (
            "null" if buffer_offset_field_name == "IOFFSET" else "0"
        )
        for (
            suffix,
            width_bits,
            global_semantic_tag,
            scratch_semantic_tag,
            implicit_data_format,
        ) in rows:
            buffer_load_key = f"amdgpu.buffer_load_{suffix}"
            _assert_memory_width_overlay(
                descriptors[buffer_load_key],
                width_bits=width_bits,
                semantic_tag=global_semantic_tag,
                mnemonic=f"buffer_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )
            buffer_load_off_zero_key = f"{buffer_load_key}_off_zero"
            buffer_load_vaddr_offset_key = f"{buffer_load_key}_vaddr_offset"
            expected_load_forms = tuple(
                key
                for key, include in (
                    (buffer_load_off_zero_key, buffer_has_off_zero),
                    (buffer_load_vaddr_offset_key, buffer_has_vaddr_offset),
                )
                if include
            )
            assert (
                tuple(
                    form.replacement_descriptor
                    for form in descriptors[buffer_load_key].operand_forms
                )
                == expected_load_forms
            )
            if buffer_has_off_zero:
                _assert_memory_width_overlay(
                    descriptors[buffer_load_off_zero_key],
                    width_bits=width_bits,
                    semantic_tag=global_semantic_tag,
                    mnemonic=f"buffer_load_{suffix}",
                    operand_units=1,
                    payload_field_name="dst",
                    effect_kind=EffectKind.READ,
                    memory_space=MemorySpace.GLOBAL,
                    implicit_data_format=implicit_data_format,
                )
                off_zero_forms = descriptors[buffer_load_off_zero_key].asm_forms
                assert off_zero_forms is not None
                off_zero_form = off_zero_forms[0]
                assert off_zero_form.mnemonic == f"buffer_load_{suffix}_off_zero"
                assert off_zero_form.results == ("dst",)
                assert off_zero_form.operands == ("resource",)
                assert off_zero_form.immediates[0].name == "offset"
            else:
                assert buffer_load_off_zero_key not in descriptors
            if buffer_has_vaddr_offset:
                _assert_memory_vaddr_offset_overlay(
                    descriptors[buffer_load_vaddr_offset_key],
                    width_bits=width_bits,
                    semantic_tag=global_semantic_tag,
                    mnemonic=f"buffer_load_{suffix}",
                    operand_units=1,
                    payload_field_name="dst",
                    effect_kind=EffectKind.READ,
                    memory_space=MemorySpace.GLOBAL,
                    implicit_data_format=implicit_data_format,
                    offset_field_name=buffer_offset_field_name,
                    fixed_soffset=buffer_fixed_soffset,
                    fixed_soffset_native_spelling=(
                        buffer_fixed_soffset_native_spelling
                    ),
                )
            else:
                assert buffer_load_vaddr_offset_key not in descriptors

            for descriptor_key_suffix, asm_suffix in (
                ("", "_vaddr"),
                ("_saddr", "_saddr"),
            ):
                global_load = descriptors[
                    f"amdgpu.global_load_{suffix}{descriptor_key_suffix}"
                ]
                global_mnemonic = f"global_load_{global_mnemonic_suffixes[suffix]}"
                _assert_memory_width_overlay(
                    global_load,
                    width_bits=width_bits,
                    semantic_tag=global_semantic_tag,
                    mnemonic=global_mnemonic,
                    operand_units=1,
                    payload_field_name="dst",
                    effect_kind=EffectKind.READ,
                    memory_space=MemorySpace.GLOBAL,
                    implicit_data_format=implicit_data_format,
                )
                assert global_load.asm_forms is not None
                assert global_load.asm_forms[0].mnemonic == (
                    f"{global_mnemonic}{asm_suffix}"
                ), (
                    global_load.descriptor_key,
                    global_load.asm_forms[0].mnemonic,
                    f"{global_mnemonic}{asm_suffix}",
                )

            scratch_mnemonic = f"scratch_load_{scratch_mnemonic_suffixes[suffix]}"
            for descriptor_key_suffix, asm_suffix in (
                ("_vaddr", "_vaddr"),
                ("_offset_only", "_offset_only"),
            ):
                scratch_load = descriptors[
                    f"amdgpu.scratch_load_{suffix}{descriptor_key_suffix}"
                ]
                _assert_memory_width_overlay(
                    scratch_load,
                    width_bits=width_bits,
                    semantic_tag=scratch_semantic_tag,
                    mnemonic=scratch_mnemonic,
                    operand_units=1,
                    payload_field_name="dst",
                    effect_kind=EffectKind.READ,
                    memory_space=MemorySpace.STACK,
                    implicit_data_format=implicit_data_format,
                    implicit_ignore_reason="modeled-by-stack-read-effect",
                )
                assert scratch_load.asm_forms is not None
                assert scratch_load.asm_forms[0].mnemonic == (
                    f"{scratch_mnemonic}{asm_suffix}"
                )

        for store_key, width_bits, semantic_tag, mnemonic in (
            ("amdgpu.buffer_store_b8", 8, "memory.store.u8", "buffer_store_b8"),
            (
                "amdgpu.buffer_store_b16",
                16,
                "memory.store.u16.low",
                "buffer_store_short",
            ),
        ):
            vaddr_offset_key = f"{store_key}_vaddr_offset"
            if buffer_has_vaddr_offset:
                assert tuple(
                    form.replacement_descriptor
                    for form in descriptors[store_key].operand_forms
                ) == (vaddr_offset_key,)
                _assert_memory_vaddr_offset_overlay(
                    descriptors[vaddr_offset_key],
                    width_bits=width_bits,
                    semantic_tag=semantic_tag,
                    mnemonic=mnemonic,
                    operand_units=1,
                    payload_field_name="value",
                    effect_kind=EffectKind.WRITE,
                    memory_space=MemorySpace.GLOBAL,
                    offset_field_name=buffer_offset_field_name,
                    fixed_soffset=buffer_fixed_soffset,
                    fixed_soffset_native_spelling=(
                        buffer_fixed_soffset_native_spelling
                    ),
                )
            else:
                assert descriptors[store_key].operand_forms == ()
                assert vaddr_offset_key not in descriptors

        b16_d16_load_key = "amdgpu.buffer_load_b16_d16"
        b16_d16_vaddr_offset_key = f"{b16_d16_load_key}_vaddr_offset"
        if buffer_has_vaddr_offset:
            assert tuple(
                form.replacement_descriptor
                for form in descriptors[b16_d16_load_key].operand_forms
            ) == (b16_d16_vaddr_offset_key,)
            _assert_memory_vaddr_offset_overlay(
                descriptors[b16_d16_vaddr_offset_key],
                width_bits=16,
                semantic_tag="memory.load.u16.d16.low",
                mnemonic="buffer_load_d16_b16",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                offset_field_name=buffer_offset_field_name,
                fixed_soffset=buffer_fixed_soffset,
                fixed_soffset_native_spelling=buffer_fixed_soffset_native_spelling,
            )
        else:
            assert descriptors[b16_d16_load_key].operand_forms == ()
            assert b16_d16_vaddr_offset_key not in descriptors


def test_d16_high_loads_preserve_tied_low_storage_without_consuming_it() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    for descriptor_key in (
        "amdgpu.ds_load_u16_d16_hi",
        "amdgpu.buffer_load_b16_d16_hi",
        "amdgpu.buffer_load_b16_d16_hi_vaddr_offset",
        "amdgpu.global_load_b16_d16_hi",
        "amdgpu.global_load_b16_d16_hi_saddr",
    ):
        descriptor = descriptors[descriptor_key]
        result = descriptor.operands[0].descriptor_operand
        source = descriptor.operands[1].descriptor_operand
        assert result.register_part == _REG_PART_VGPR_HIGH16
        assert source.register_part == _REG_PART_VGPR_LOW16
        assert OperandFlag.IMPLICIT in source.flags
        assert OperandFlag.STORAGE_CONTINUATION in source.flags
        assert descriptor.constraints == (Constraint(ConstraintKind.TIED, 0, 1),)


def test_cdna_smem_dwordx4_store_and_scratch_descriptors_cover_xml() -> None:
    rows = (
        (32, 1, "dword"),
        (64, 2, "dwordx2"),
        (128, 4, "dwordx4"),
    )
    for rdna_overlays in (
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx125x_core_overlays(),
    ):
        rdna_descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in rdna_overlays
        }
        for _width_bits, _units, suffix in rows:
            assert f"amdgpu.s_store_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_scratch_load_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_scratch_store_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_buffer_store_{suffix}" not in rdna_descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
    ):
        for width_bits, units, suffix in rows:
            store_key = f"amdgpu.s_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )
            assert descriptors[store_key].schedule_class == _SCHEDULE_SMEM_STORE
            _assert_memory_width_overlay(
                descriptors[f"{store_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )

            scratch_load_key = f"amdgpu.s_scratch_load_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scratch_load_key],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.load.u{width_bits}",
                mnemonic=f"s_scratch_load_{suffix}",
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-read-effect",
            )
            _assert_memory_width_overlay(
                descriptors[f"{scratch_load_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.load.u{width_bits}",
                mnemonic=f"s_scratch_load_{suffix}",
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-read-effect",
            )

            scratch_store_key = f"amdgpu.s_scratch_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scratch_store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.store.u{width_bits}",
                mnemonic=f"s_scratch_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-write-effect",
            )
            assert descriptors[scratch_store_key].schedule_class == _SCHEDULE_SMEM_STORE
            _assert_memory_width_overlay(
                descriptors[f"{scratch_store_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.store.u{width_bits}",
                mnemonic=f"s_scratch_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-write-effect",
            )

            buffer_store_key = f"amdgpu.s_buffer_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[buffer_store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_buffer_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )
            assert descriptors[buffer_store_key].schedule_class == _SCHEDULE_SMEM_STORE


def test_cdna_scratch_m0_operands_are_state_reads() -> None:
    for overlay_builder in (_gfx940_core_overlays, _gfx950_core_overlays):
        m0_rows = [
            (descriptor, implicit_operand)
            for descriptor in overlay_builder()
            if descriptor.semantic_tag is not None
            and descriptor.semantic_tag.startswith("memory.stack.")
            for implicit_operand in descriptor.implicit_operands
            if implicit_operand.operand_type == "OPR_SDST_M0"
        ]

        assert m0_rows
        for _, implicit_operand in m0_rows:
            assert implicit_operand.is_input
            assert not implicit_operand.is_output
            operand = implicit_operand.descriptor_operand
            assert operand is not None
            assert operand.field_name == "m0"
            assert operand.role is OperandRole.RESOURCE
            assert OperandFlag.IMPLICIT in operand.flags
            assert OperandFlag.STATE_READ in operand.flags
            assert OperandFlag.STATE_WRITE not in operand.flags


def test_gfx940_scratch_memory_forms_cover_spill_packets() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }

    load_descriptor = descriptors["amdgpu.scratch_load_b32_offset_only"]
    load_forms = load_descriptor.asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "scratch_load_b32_offset_only"
    assert load_form.results == ("dst",)
    assert load_form.operands == ()
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )

    store_descriptor = descriptors["amdgpu.scratch_store_b32_offset_only"]
    store_forms = store_descriptor.asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "scratch_store_b32_offset_only"
    assert store_form.results == ()
    assert store_form.operands == ("value",)
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )
