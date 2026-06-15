# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared AMDGPU descriptor vocabulary and builder helpers."""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuEncodingFieldAllOnes,
    AmdgpuFixedEncodingValue,
    AmdgpuIgnoredOperandOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    AmdgpuOperandPredefinedValueRef,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_DS,
    AMDGPU_ENCODING_FORMAT_FLAT,
    AMDGPU_ENCODING_FORMAT_MUBUF,
    AMDGPU_ENCODING_FORMAT_SOP1,
    AMDGPU_ENCODING_FORMAT_SOP2,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_SOPP,
    AMDGPU_ENCODING_FORMAT_VBUFFER,
    AMDGPU_ENCODING_FORMAT_VDS,
    AMDGPU_ENCODING_FORMAT_VFLAT,
    AMDGPU_ENCODING_FORMAT_VGLOBAL,
    AMDGPU_ENCODING_FORMAT_VOP1,
    AMDGPU_ENCODING_FORMAT_VOP1_DPP,
    AMDGPU_ENCODING_FORMAT_VOP1_DPP16,
    AMDGPU_ENCODING_FORMAT_VOP1_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
    AMDGPU_ENCODING_FORMAT_VOP2,
    AMDGPU_ENCODING_FORMAT_VOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP3,
    AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP3_SDST,
    AMDGPU_ENCODING_FORMAT_VOP3P,
    AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP3PX2,
    AMDGPU_ENCODING_FORMAT_VSCRATCH,
    amdgpu_encoding_field_id,
    amdgpu_encoding_field_name,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_ordinal,
    validate_amdgpu_descriptor_set_isa_xml,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    AsmImmediate,
    CEnum,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorCategory,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    NativeAsmValue,
    NativeAsmValueKind,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandForm,
    OperandFormImmediateAction,
    OperandFormMatch,
    OperandFormMatchKind,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    RegisterPart,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
    StorageLease,
    StorageLeaseAttachment,
    StorageLeaseFlag,
    StorageLeaseKind,
    StorageLeaseReleaseScope,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"
_REG_AGPR = "amdgpu.agpr"
_REG_M0 = "amdgpu.m0"
_REG_SCC = "amdgpu.scc"
_REG_EXEC = "amdgpu.exec"
_REG_MODE = "amdgpu.mode"

_REG_PART_SGPR_LOW16 = "amdgpu.sgpr.low16"
_REG_PART_SGPR_HIGH16 = "amdgpu.sgpr.high16"
_REG_PART_SGPR_FULL32_MASK = 0x3
_REG_PART_VGPR_LOW16 = "amdgpu.vgpr.low16"
_REG_PART_VGPR_HIGH16 = "amdgpu.vgpr.high16"
_REG_PART_VGPR_FULL32_MASK = 0x3

_RESOURCE_SALU = "amdgpu.salu"
_RESOURCE_VALU = "amdgpu.valu"
_RESOURCE_SMEM = "amdgpu.smem"
_RESOURCE_VMEM_LOAD = "amdgpu.vmem.load"
_RESOURCE_VMEM_STORE = "amdgpu.vmem.store"
_RESOURCE_LDS_LOAD = "amdgpu.lds.load"
_RESOURCE_LDS_STORE = "amdgpu.lds.store"
_RESOURCE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_WMMA = "amdgpu.wmma"
_RESOURCE_SWMMAC = "amdgpu.swmmac"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_TRANS = "amdgpu.trans"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_SMEM_STORE = "amdgpu.smem.store"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_LOAD_LDS = "amdgpu.vmem.load.lds"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_VMEM_ATOMIC_RETURN = "amdgpu.vmem.atomic.return"
_SCHEDULE_VMEM_ATOMIC_NO_RETURN = "amdgpu.vmem.atomic.no_return"
_SCHEDULE_LDS_LOAD = "amdgpu.lds.load"
_SCHEDULE_LDS_STORE = "amdgpu.lds.store"
_SCHEDULE_LDS_ATOMIC = "amdgpu.lds.atomic"
_SCHEDULE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_SCHEDULE_BARRIER = "amdgpu.barrier"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WMMA = "amdgpu.wmma"
_SCHEDULE_WMMA_SCALE = "amdgpu.wmma.scale"
_SCHEDULE_SWMMAC = "amdgpu.swmmac"
_SCHEDULE_CACHE_CONTROL = "amdgpu.cache.control"
_SCHEDULE_MODE_CONTROL = "amdgpu.mode.control"
_SCHEDULE_WAIT_MEMORY = "amdgpu.wait.memory"
_SCHEDULE_WAIT_VMEM_STORE = "amdgpu.wait.vmem.store"
_SCHEDULE_WAIT_LDS = "amdgpu.wait.lds"
_SCHEDULE_WAIT_SMEM = "amdgpu.wait.smem"
_SCHEDULE_WAIT_LOAD = "amdgpu.wait.load"
_SCHEDULE_WAIT_STORE = "amdgpu.wait.store"
_SCHEDULE_WAIT_ALU = "amdgpu.wait.alu"
_SCHEDULE_WAIT_IDLE = "amdgpu.wait.idle"

_AMDGPU_TRANS_DESCRIPTOR_KEYS = (
    "amdgpu.v_exp_f32",
    "amdgpu.v_log_f32",
    "amdgpu.v_sin_f32",
    "amdgpu.v_cos_f32",
    "amdgpu.v_sqrt_f32",
    "amdgpu.v_rsq_f32",
    "amdgpu.v_rcp_f32",
)

_AMDGPU_TRANS_PROXY_LATENCY_CYCLES = 8
# These proxy values keep TRANS op scheduling conservative until per-processor
# latency tables are available. The schedule classes remain descriptor-specific
# so a future table can tune individual operations without changing descriptors.
_AMDGPU_TRANS_DESCRIPTOR_LATENCY_CYCLES = {
    descriptor_key: _AMDGPU_TRANS_PROXY_LATENCY_CYCLES
    for descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS
}

# AMDGPU vector, memory, and matrix packets observe EXEC even when the vendor XML
# does not expose EXEC as an implicit operand. Model that state read in Loom so
# scheduling cannot move the packet across divergent-control EXEC writes.
_EXECUTION_MASKED_SCHEDULE_CLASSES = frozenset(
    (
        _SCHEDULE_VALU,
        _SCHEDULE_TRANS,
        _SCHEDULE_VMEM_LOAD,
        _SCHEDULE_VMEM_LOAD_LDS,
        _SCHEDULE_VMEM_STORE,
        _SCHEDULE_VMEM_ATOMIC_RETURN,
        _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
        _SCHEDULE_LDS_LOAD,
        _SCHEDULE_LDS_STORE,
        _SCHEDULE_LDS_ATOMIC,
        _SCHEDULE_LDS_CROSSLANE,
        _SCHEDULE_MFMA,
        _SCHEDULE_WMMA,
        _SCHEDULE_WMMA_SCALE,
        _SCHEDULE_SWMMAC,
    )
)


def _amdgpu_trans_schedule_class_name(descriptor_key: str) -> str:
    if descriptor_key not in _AMDGPU_TRANS_DESCRIPTOR_KEYS:
        raise ValueError(f"AMDGPU descriptor '{descriptor_key}' is not a TRANS op")
    return f"{_SCHEDULE_TRANS}.{descriptor_key.removeprefix('amdgpu.')}"


def _amdgpu_schedule_class_reads_exec_state(schedule_class: str) -> bool:
    return (
        schedule_class in _EXECUTION_MASKED_SCHEDULE_CLASSES
        or schedule_class.startswith(f"{_SCHEDULE_TRANS}.")
    )


AMDGPU_SCALAR_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "scalar",
    doc="Scalar ALU, scalar data movement, and SGPR descriptors.",
)
AMDGPU_VECTOR_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "vector",
    doc="Vector ALU, vector data movement, and VGPR descriptors.",
)
AMDGPU_CONVERT_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "convert",
    doc="Numeric conversion descriptors.",
)
AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "compare_select",
    doc="Comparison and predicate-controlled selection descriptors.",
)
AMDGPU_MEMORY_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "memory",
    doc="Non-atomic memory load, store, and transfer descriptors.",
)
AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "atomic",
    doc="Atomic memory operation descriptors.",
)
AMDGPU_MATRIX_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "matrix",
    doc="Matrix, dot, WMMA, MFMA, and SWMMAC descriptors.",
)
AMDGPU_CONTROL_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "control",
    doc="Control, wait, barrier, and special state descriptors.",
)
AMDGPU_CACHE_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "cache",
    doc="Cache control and prefetch descriptors.",
)
AMDGPU_MISC_DESCRIPTOR_CATEGORY = DescriptorCategory(
    "misc",
    doc="Descriptors that do not fit another stable AMDGPU category.",
)

AMDGPU_DESCRIPTOR_CATEGORIES = (
    AMDGPU_SCALAR_DESCRIPTOR_CATEGORY,
    AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
    AMDGPU_CONVERT_DESCRIPTOR_CATEGORY,
    AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
    AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
    AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    AMDGPU_MATRIX_DESCRIPTOR_CATEGORY,
    AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    AMDGPU_CACHE_DESCRIPTOR_CATEGORY,
    AMDGPU_MISC_DESCRIPTOR_CATEGORY,
)

_AMDGPU_DESCRIPTOR_SOURCE_DIR = Path("loom/src/loom/target/arch/amdgpu/descriptors")
_AMDGPU_DESCRIPTOR_PUBLIC_HEADER_DIR = "loom/target/arch/amdgpu/descriptors"
_AMDGPU_INLINE_F32_ENUM_DOMAIN_NAME = "amdgpu.source_inline_f32"
_AMDGPU_INLINE_U32_16_ENUM_DOMAIN_NAME = "amdgpu.source_inline_u32_16"


def _f32_bits(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", value))[0])


_AMDGPU_SOURCE_INLINE_F32_ENUM_DOMAIN = EnumDomain(
    _AMDGPU_INLINE_F32_ENUM_DOMAIN_NAME,
    values=(
        EnumValue("f32_0_5", _f32_bits(0.5)),
        EnumValue("f32_n0_5", _f32_bits(-0.5)),
        EnumValue("f32_1_0", _f32_bits(1.0)),
        EnumValue("f32_n1_0", _f32_bits(-1.0)),
        EnumValue("f32_2_0", _f32_bits(2.0)),
        EnumValue("f32_n2_0", _f32_bits(-2.0)),
        EnumValue("f32_4_0", _f32_bits(4.0)),
        EnumValue("f32_n4_0", _f32_bits(-4.0)),
        EnumValue("f32_inv_2pi", _f32_bits(0.15915494)),
    ),
)
_AMDGPU_SOURCE_INLINE_U32_16_ENUM_DOMAIN = EnumDomain(
    _AMDGPU_INLINE_U32_16_ENUM_DOMAIN_NAME,
    values=(EnumValue("u32_16", 16),),
)


def _amdgpu_descriptor_set_file_stem(key: str) -> str:
    key_prefix = "amdgpu."
    key_suffix = ".core"
    if not key.startswith(key_prefix) or not key.endswith(key_suffix):
        raise ValueError(
            f"AMDGPU core descriptor set key '{key}' must have form "
            "'amdgpu.<family>.core'"
        )
    return key.removeprefix(key_prefix).removesuffix(key_suffix).replace(".", "_")


def _amdgpu_camel_case(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in value.split("_") if part)


def _amdgpu_core_descriptor_set(
    *,
    key: str,
    reg_classes: tuple[RegClass, ...],
    resources: tuple[Resource, ...],
    schedule_classes: tuple[ScheduleClass, ...],
    descriptors: tuple[Descriptor, ...] = (),
    register_parts: tuple[RegisterPart, ...] = (),
    enum_domains: tuple[EnumDomain, ...] = (
        _AMDGPU_SOURCE_INLINE_F32_ENUM_DOMAIN,
        _AMDGPU_SOURCE_INLINE_U32_16_ENUM_DOMAIN,
    ),
    categories: tuple[DescriptorCategory, ...] = AMDGPU_DESCRIPTOR_CATEGORIES,
) -> DescriptorSet:
    file_stem = _amdgpu_descriptor_set_file_stem(key)
    c_suffix = _amdgpu_camel_case(file_stem)
    c_enum_stem = file_stem.upper()
    feature_stem = key.removesuffix(".core")
    return DescriptorSet(
        key=key,
        target_key="amdgpu",
        feature_key=f"{feature_stem}.v1",
        c_header_path=_AMDGPU_DESCRIPTOR_SOURCE_DIR / f"{file_stem}_descriptors.h",
        c_source_path=_AMDGPU_DESCRIPTOR_SOURCE_DIR / f"{file_stem}_descriptors.c",
        header_guard=f"LOOM_TARGET_ARCH_AMDGPU_{c_enum_stem}_DESCRIPTORS_H_",
        public_header=(
            f"{_AMDGPU_DESCRIPTOR_PUBLIC_HEADER_DIR}/{file_stem}_descriptors.h"
        ),
        function_name=f"loom_amdgpu_{file_stem}_core_descriptor_set",
        c_table_prefix=f"Amdgpu{c_suffix}Core",
        c_enum_prefix=f"AMDGPU_{c_enum_stem}_CORE",
        generator_version=1,
        reg_classes=reg_classes,
        register_parts=register_parts,
        enum_domains=enum_domains,
        resources=resources,
        schedule_classes=schedule_classes,
        descriptors=descriptors,
        descriptor_set_ordinal=amdgpu_descriptor_set_ordinal(key),
        categories=categories,
        requires_explicit_asm_surface=True,
    )


_COUNTER_VMEM_LOAD = 1
_COUNTER_VMEM_STORE = 2
_COUNTER_LDS = 3
_COUNTER_SMEM = 4
_COUNTER_ALU = 5


def _predefined(
    value_name: str, operand_type: str | None = None
) -> AmdgpuOperandPredefinedValueRef:
    return AmdgpuOperandPredefinedValueRef(value_name, operand_type)


def _instruction_encoding_opcode(
    spec: AmdgpuIsaFactSource, instruction_name: str, encoding_name: str
) -> int:
    opcodes = set()
    for summary in spec.instruction_encoding_summaries(
        (instruction_name,), include_aliases=False
    ):
        if summary.encoding_name == encoding_name:
            opcodes.add(summary.opcode)
    if len(opcodes) != 1:
        raise ValueError(
            f"{spec.source_name}: expected one {encoding_name} opcode for "
            f"{instruction_name}, found {len(opcodes)}"
        )
    return next(iter(opcodes))


_MUBUF_SOFFSET_INLINE_ZERO = _predefined("0")
_VBUFFER_SOFFSET_NULL = _predefined("NULL", "OPR_SREG_M0")

_VMEM_LOAD_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_LOAD
)
_VMEM_STORE_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_STORE
)
_LDS_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_LDS)
_SMEM_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_SMEM)
_ALU_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_ALU)
_GFX950_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_GFX11_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_VMEM_LOAD_WAIT_HAZARDS = (_VMEM_LOAD_COUNTER_HAZARD,)
_VMEM_STORE_WAIT_HAZARDS = (_VMEM_STORE_COUNTER_HAZARD,)
_LDS_WAIT_HAZARDS = (_LDS_COUNTER_HAZARD,)
_SMEM_WAIT_HAZARDS = (_SMEM_COUNTER_HAZARD,)
_ALU_WAIT_HAZARDS = (_ALU_COUNTER_HAZARD,)
_IDLE_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
    _ALU_COUNTER_HAZARD,
)

_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS = (
    ("IMM", 1),
    ("SOFFSET_EN", 1),
)
_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS = (
    ("IMM", 1),
    ("SOFFSET_EN", 0),
)

_ADDRESS_OFFSET_BYTE_ENCODING_ID = 1
_ADDRESS_OFFSET_DWORD_ENCODING_ID = 2
_ADDRESS_OFFSET_QWORD_ENCODING_ID = 3
_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID = 4
_ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID = 5
_ADDRESS_OFFSET_DS16_ENCODING_ID = 6
_SOURCE_INLINE_U32_ENCODING_ID = 7
_SOURCE_INLINE_F32_ENCODING_ID = 8
_WAIT_COUNTER_VMEM_ENCODING_ID = 16
_WAIT_COUNTER_LGKM_ENCODING_ID = 17
_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID = 18
_WAIT_COUNTER_VMEM_STORE_ENCODING_ID = 19
_WAIT_COUNTER_LDS_ENCODING_ID = 20
_WAIT_COUNTER_SMEM_ENCODING_ID = 21
_WAIT_COUNTER_ALU_ENCODING_ID = 22
_GFX9_11_VECTOR_CACHE_FIELDS = (("GLC", 1), ("SLC", 1), ("DLC", 1))
_GFX950_VECTOR_CACHE_FIELDS = (("NT", 1), ("SC0", 1), ("SC1", 1))
_GFX12_VECTOR_CACHE_FIELDS = (("NV", 1), ("SCOPE", 2), ("TH", 3))
_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS = ("SCOPE", "TH")
# VGLOBAL atomic instructions use TH bit 0 to request returning the old value.
_GFX12_TH_ATOMIC_RETURN_VALUE = 0x1
_ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS = frozenset(
    (
        _ADDRESS_OFFSET_BYTE_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_DS16_ENCODING_ID,
    )
)
_ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES = frozenset(("offset", "offset0", "offset1"))
_GLOBAL_SADDR_OFF = _predefined("NULL")
_GLOBAL_GFX950_SADDR_OFF = AmdgpuEncodingFieldAllOnes()
_SCRATCH_CDNA_SADDR_OFF = AmdgpuEncodingFieldAllOnes()
_MUBUF_VADDR_OFFSET_ONLY_SIZE_REASON = "idxen-disabled-mubuf-vaddr-uses-one-offset-vgpr"
_GLOBAL_SADDR_OFFSET_ONLY_SIZE_REASON = (
    "saddr-enabled-global-address-uses-one-offset-vgpr"
)
_D16_PARTIAL_REGISTER_SIZE_REASON = "d16-instruction-uses-half-vgpr-lane"
_U24_SOURCE_SIZE_REASON = "u24-instruction-reads-low-24-bits-of-b32-source"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)
_SGPR_VGPR_ALT = (RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR))
_VGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_VGPR_AGPR_ALT = (RegClassAlt(_REG_VGPR), RegClassAlt(_REG_AGPR))
_VGPR_AGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(_REG_AGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_M0_ALT = (RegClassAlt(_REG_M0, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)
_SCC_ALT = (RegClassAlt(_REG_SCC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)
_EXEC_ALT = (RegClassAlt(_REG_EXEC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)
_MODE_ALT = (RegClassAlt(_REG_MODE, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)

_VGPR_REGISTER_PARTS = (
    RegisterPart(_REG_PART_VGPR_LOW16, _REG_VGPR, 0x1),
    RegisterPart(_REG_PART_VGPR_HIGH16, _REG_VGPR, 0x2),
)
_SGPR_REGISTER_PARTS = (
    RegisterPart(_REG_PART_SGPR_LOW16, _REG_SGPR, 0x1),
    RegisterPart(_REG_PART_SGPR_HIGH16, _REG_SGPR, 0x2),
)
_AMDGPU_REGISTER_PARTS = (*_VGPR_REGISTER_PARTS, *_SGPR_REGISTER_PARTS)


class AmdgpuAtomicMemorySpace(CEnum):
    WORKGROUP = "LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP"
    GLOBAL = "LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL"
    GENERIC = "LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC"


class AmdgpuMemoryAddressForm(CEnum):
    DEFAULT = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT"
    GLOBAL_SADDR = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR"
    FLAT = "LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT"


class AmdgpuAtomicOperationKind(CEnum):
    REDUCE = "LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE"
    RMW = "LOOM_AMDGPU_ATOMIC_OPERATION_RMW"
    CMPXCHG = "LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG"


class AmdgpuAtomicKind(CEnum):
    ADDI = "LOOM_ATOMIC_KIND_ADDI"
    SUBI = "LOOM_ATOMIC_KIND_SUBI"
    MINSI = "LOOM_ATOMIC_KIND_MINSI"
    MAXSI = "LOOM_ATOMIC_KIND_MAXSI"
    MINUI = "LOOM_ATOMIC_KIND_MINUI"
    MAXUI = "LOOM_ATOMIC_KIND_MAXUI"
    ANDI = "LOOM_ATOMIC_KIND_ANDI"
    ORI = "LOOM_ATOMIC_KIND_ORI"
    XORI = "LOOM_ATOMIC_KIND_XORI"
    XCHGI = "LOOM_ATOMIC_KIND_XCHGI"
    ADDF = "LOOM_ATOMIC_KIND_ADDF"
    MINNUMF = "LOOM_ATOMIC_KIND_MINNUMF"
    MAXNUMF = "LOOM_ATOMIC_KIND_MAXNUMF"
    NONE = "LOOM_AMDGPU_ATOMIC_KIND_NONE"


class AmdgpuAtomicValueKind(CEnum):
    I32 = "LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32"
    F32 = "LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32"


@dataclass(frozen=True, slots=True)
class AmdgpuAtomicDescriptorCandidate:
    memory_space: AmdgpuAtomicMemorySpace
    address_form: AmdgpuMemoryAddressForm
    operation_kind: AmdgpuAtomicOperationKind
    atomic_kind: AmdgpuAtomicKind
    value_kind: AmdgpuAtomicValueKind
    descriptor_key: str


def _matrix_hazards(resource: str) -> tuple[Hazard, ...]:
    return (
        _ALU_COUNTER_HAZARD,
        Hazard(
            HazardKind.MIN_DISTANCE,
            resource=resource,
            producer_stage=0,
            consumer_stage=0,
            distance=2,
        ),
    )


def _common_scalar_vector_memory_resources() -> tuple[Resource, ...]:
    return (
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_LDS_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_CROSSLANE, capacity_per_cycle=1, kind=ResourceKind.LOAD),
    )


def _amdgpu_trans_schedule_classes(
    *,
    latency_cycles_by_descriptor_key: Mapping[str, int] | None = None,
) -> tuple[ScheduleClass, ...]:
    latency_cycles = dict(_AMDGPU_TRANS_DESCRIPTOR_LATENCY_CYCLES)
    if latency_cycles_by_descriptor_key is not None:
        unknown_descriptor_keys = set(latency_cycles_by_descriptor_key).difference(
            _AMDGPU_TRANS_DESCRIPTOR_KEYS
        )
        if unknown_descriptor_keys:
            unknown_keys = ", ".join(sorted(unknown_descriptor_keys))
            raise ValueError(
                f"AMDGPU TRANS latency table references unknown descriptor(s): "
                f"{unknown_keys}"
            )
        latency_cycles.update(latency_cycles_by_descriptor_key)
    return tuple(
        ScheduleClass(
            _amdgpu_trans_schedule_class_name(descriptor_key),
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=latency_cycles[descriptor_key],
            issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            model_quality=ModelQuality.ESTIMATED,
        )
        for descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS
    )


def _common_scalar_vector_memory_schedule_classes(
    *,
    smem_load_hazards: tuple[Hazard, ...],
    smem_store_hazards: tuple[Hazard, ...],
    vmem_load_hazards: tuple[Hazard, ...],
    vmem_store_hazards: tuple[Hazard, ...],
    lds_load_hazards: tuple[Hazard, ...],
    lds_store_hazards: tuple[Hazard, ...],
    lds_atomic_hazards: tuple[Hazard, ...],
    lds_crosslane_hazards: tuple[Hazard, ...],
    trans_latency_cycles_by_descriptor_key: Mapping[str, int] | None = None,
) -> tuple[ScheduleClass, ...]:
    return (
        ScheduleClass(
            _SCHEDULE_SALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SALU, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            model_quality=ModelQuality.ESTIMATED,
        ),
        *_amdgpu_trans_schedule_classes(
            latency_cycles_by_descriptor_key=trans_latency_cycles_by_descriptor_key
        ),
        ScheduleClass(
            _SCHEDULE_SMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_SMEM, cycles=1, units=1),),
            hazards=smem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_SMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_SMEM, cycles=1, units=1),),
            hazards=smem_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),),
            hazards=lds_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),),
            hazards=lds_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_ATOMIC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(
                IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=lds_atomic_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_CROSSLANE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_CROSSLANE, cycles=1, units=1),),
            hazards=lds_crosslane_hazards,
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_BARRIER,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CACHE_CONTROL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    )


def _asm(
    *,
    mnemonic: str | None = None,
    native_assembly_mnemonic: str | None = None,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    named_immediates: bool = False,
    native_assembly_values: tuple[NativeAsmValue, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            native_assembly_mnemonic=native_assembly_mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(
                AsmImmediate(field_name, name=field_name if named_immediates else None)
                for field_name in immediates
            ),
            native_assembly_values=native_assembly_values,
        ),
    )


def _native_result(field_name: str) -> NativeAsmValue:
    return NativeAsmValue(NativeAsmValueKind.RESULT, field_name=field_name)


def _native_operand(field_name: str) -> NativeAsmValue:
    return NativeAsmValue(NativeAsmValueKind.OPERAND, field_name=field_name)


def _native_literal(spelling: str) -> NativeAsmValue:
    return NativeAsmValue(NativeAsmValueKind.LITERAL, literal=spelling)


def _native_i64_immediate(field_name: str) -> NativeAsmValue:
    return NativeAsmValue(NativeAsmValueKind.IMMEDIATE_I64, field_name=field_name)


def _native_unsigned_hex_immediate(field_name: str, bit_width: int) -> NativeAsmValue:
    return NativeAsmValue(
        NativeAsmValueKind.IMMEDIATE_UNSIGNED_HEX,
        field_name=field_name,
        bit_width=bit_width,
    )


def _global_vaddr_asm(
    *,
    mnemonic: str,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...],
    immediates: tuple[str, ...],
) -> tuple[AsmForm, ...]:
    return _asm(
        mnemonic=f"{mnemonic}_vaddr",
        results=results,
        operands=operands,
        immediates=immediates,
        named_immediates=True,
    )


def _global_saddr_asm(
    *,
    mnemonic: str,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...],
    implicit_m0: bool = False,
    immediates: tuple[str, ...],
) -> tuple[AsmForm, ...]:
    if implicit_m0:
        operands = (*operands, "m0")
    return _asm(
        mnemonic=f"{mnemonic}_saddr",
        results=results,
        operands=operands,
        immediates=immediates,
        named_immediates=True,
    )


def _literal_operand_form(
    *,
    replacement_descriptor: str,
    source_operand: str,
    immediate_field: str = "imm32",
) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand=source_operand,
                match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64,
            ),
        ),
        immediate_action=OperandFormImmediateAction.SET_MATCHED_I64,
        immediate_field=immediate_field,
        immediate_source_operand=source_operand,
    )


def _soffset_zero_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
        ),
    )


def _soffset_offset_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64,
            ),
        ),
        immediate_action=OperandFormImmediateAction.ADD_MATCHED_I64,
        immediate_field="offset",
        immediate_source_operand="soffset",
    )


def _buffer_off_zero_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="vaddr",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                match_i64=0,
            ),
        ),
    )


def _buffer_soffset_offset_operand_form(*, replacement_descriptor: str) -> OperandForm:
    return OperandForm(
        replacement_descriptor=replacement_descriptor,
        matches=(
            OperandFormMatch(
                source_operand="soffset",
                match_kind=OperandFormMatchKind.ALL_EQUAL_EXACT_I64,
            ),
        ),
        immediate_action=OperandFormImmediateAction.ADD_MATCHED_I64,
        immediate_field="offset",
        immediate_source_operand="soffset",
    )


def _sgpr_result(
    field_name: str = "dst", *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _SGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _sgpr_operand(
    field_name: str, *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.OPERAND,
        _SGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _sgpr_predicate(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _SGPR_ALT, unit_count=units)


def _sgpr_resource(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _SGPR_ALT, unit_count=units)


def _m0_implicit_resource(field_name: str = "m0") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESOURCE,
        _M0_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _m0_clobber(field_name: str = "m0") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _M0_ALT,
        flags=(OperandFlag.IMPLICIT,),
        unit_count=1,
    )


def _m0_result(field_name: str = "dst") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _M0_ALT,
        flags=(OperandFlag.STATE_WRITE,),
        unit_count=1,
    )


def _scc_result(field_name: str = "scc") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _scc_clobber(field_name: str = "scc") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _scc_state_read(field_name: str = "scc_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _scc_predicate(field_name: str) -> Operand:
    return Operand(
        field_name,
        OperandRole.PREDICATE,
        _SCC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _exec_clobber(field_name: str = "exec") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _EXEC_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _exec_state_read(field_name: str = "exec_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _EXEC_ALT,
        flags=(
            OperandFlag.IMPLICIT,
            OperandFlag.STATE_READ,
            OperandFlag.SCHEDULE_ONLY_STATE,
        ),
        unit_count=1,
    )


def _mode_state_read(field_name: str = "mode_in") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _MODE_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
        unit_count=1,
    )


def _mode_state_write(field_name: str = "mode") -> Operand:
    return Operand(
        field_name,
        OperandRole.IMPLICIT,
        _MODE_ALT,
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        unit_count=1,
    )


def _is_exec_state_read(operand: Operand) -> bool:
    return (
        OperandFlag.STATE_READ in operand.flags
        and len(operand.reg_alts) == 1
        and operand.reg_alts[0].reg_class == _REG_EXEC
    )


def _is_mode_state_read(operand: Operand) -> bool:
    return (
        OperandFlag.STATE_READ in operand.flags
        and len(operand.reg_alts) == 1
        and operand.reg_alts[0].reg_class == _REG_MODE
    )


def _with_execution_mask_state_read(descriptor: Descriptor) -> Descriptor:
    if not _amdgpu_schedule_class_reads_exec_state(descriptor.schedule_class):
        return descriptor
    if any(_is_exec_state_read(operand) for operand in descriptor.operands):
        return descriptor
    return replace(descriptor, operands=(*descriptor.operands, _exec_state_read()))


def _with_execution_mask_state_reads(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        _with_execution_mask_state_read(descriptor) for descriptor in descriptors
    )


def _with_mode_state_read(descriptor: Descriptor) -> Descriptor:
    if any(_is_mode_state_read(operand) for operand in descriptor.operands):
        return descriptor
    return replace(descriptor, operands=(*descriptor.operands, _mode_state_read()))


def _vgpr_result(
    field_name: str = "dst", *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        _VGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _vgpr_operand(
    field_name: str, *, units: int = 1, register_part: str | None = None
) -> Operand:
    return Operand(
        field_name,
        OperandRole.OPERAND,
        _VGPR_ALT,
        unit_count=units,
        register_part=register_part,
    )


def _mubuf_vaddr_operand(field_name: str = "vaddr") -> AmdgpuOperandOverlay:
    return AmdgpuOperandOverlay(
        "VADDR",
        _vgpr_operand(field_name),
        size_exception_reason=_MUBUF_VADDR_OFFSET_ONLY_SIZE_REASON,
    )


def _global_addr_operand(
    xml_field_name: str, *, units: int, has_saddr: bool
) -> AmdgpuOperandOverlay:
    return AmdgpuOperandOverlay(
        xml_field_name,
        _vgpr_operand("addr", units=units),
        size_exception_reason=(
            _GLOBAL_SADDR_OFFSET_ONLY_SIZE_REASON if has_saddr and units == 1 else None
        ),
    )


def _sgpr_vgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SGPR_VGPR_ALT, unit_count=units)


def _vgpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_CONST_ALT, unit_count=units)


def _vgpr_agpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(
        field_name, OperandRole.OPERAND, _VGPR_AGPR_CONST_ALT, unit_count=units
    )


def _u32_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=(2**32) - 1,
    )


def _source_inline_u32_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=64,
        encoding_id=_SOURCE_INLINE_U32_ENCODING_ID,
    )


def _source_inline_u32_16_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.ENUM,
        bit_width=32,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        enum_domain=_AMDGPU_INLINE_U32_16_ENUM_DOMAIN_NAME,
        encoding_id=_SOURCE_INLINE_U32_ENCODING_ID,
        default_value=16,
    )


def _source_inline_f32_immediate(field_name: str = "imm32") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.ENUM,
        bit_width=32,
        enum_domain=_AMDGPU_INLINE_F32_ENUM_DOMAIN_NAME,
        encoding_id=_SOURCE_INLINE_F32_ENCODING_ID,
    )


_U32_IMMEDIATE = _u32_immediate()

_SOURCE_INLINE_U32_IMMEDIATE = _source_inline_u32_immediate()
_SOURCE_INLINE_U32_16_IMMEDIATE = _source_inline_u32_16_immediate()
_SOURCE_INLINE_F32_IMMEDIATE = _source_inline_f32_immediate()

_LITERAL_U32_IMMEDIATE = replace(
    _U32_IMMEDIATE, encoding_field_id=amdgpu_encoding_field_id("LITERAL")
)

_HAL_BUFFER_DESCRIPTOR_EXTENT_IMMEDIATE = Immediate(
    "extent",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE = Immediate(
    "cache_swizzle_stride",
    ImmediateKind.UNSIGNED,
    bit_width=14,
    unsigned_max=(2**14) - 1,
)


_MANUAL_SCALAR_DESCRIPTOR_KEYS = (
    "amdgpu.s_mov_b32",
    "amdgpu.s_mov_b32_m0",
    "amdgpu.s_mov_b32_m0.imm",
    "amdgpu.s_mov_b64_exec",
    "amdgpu.s_mov_b64_exec.full",
    "amdgpu.s_mov_b64_exec_read",
    "amdgpu.s_xor_b64_exec",
)


def _s_mov_b32_contract_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mov_b32",
        instruction_name="S_MOV_B32",
        mnemonic="s_mov_b32",
        encoding_name="ENC_SOP1",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(AmdgpuOperandOverlay("SDST", _sgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediates=(_U32_IMMEDIATE,),
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _manual_scalar_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    s_mov_b32_opcode = _instruction_encoding_opcode(spec, "S_MOV_B32", "ENC_SOP1")
    s_mov_b64_opcode = _instruction_encoding_opcode(spec, "S_MOV_B64", "ENC_SOP1")
    s_xor_b64_opcode = _instruction_encoding_opcode(spec, "S_XOR_B64", "ENC_SOP2")
    return (
        Descriptor(
            key="amdgpu.s_mov_b32",
            mnemonic="s_mov_b32",
            semantic_tag="integer.const.u32",
            operands=(_sgpr_result(),),
            immediates=(_U32_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b32_m0",
            mnemonic="s_mov_b32",
            semantic_tag="special.m0.move.u32",
            operands=(
                _m0_result(),
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                ),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_M0", "M0"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b32_m0", results=("dst",), operands=("src",)
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b32_m0.imm",
            mnemonic="s_mov_b32",
            semantic_tag="special.m0.const.u32",
            operands=(_m0_result(),),
            immediates=(_U32_IMMEDIATE,),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_M0", "M0"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b32_m0_imm", results=("dst",), immediates=("imm32",)
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b32_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b64_exec",
            mnemonic="s_mov_b64",
            semantic_tag="control.exec.restore",
            operands=(
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                    unit_count=2,
                ),
                _exec_clobber(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_EXEC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b64_exec",
                native_assembly_mnemonic="s_mov_b64",
                operands=("src",),
                native_assembly_values=(
                    _native_literal("exec"),
                    _native_operand("src"),
                ),
            ),
            effects=(_CONVERGENT_EFFECT,),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b64_opcode,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b64_exec.full",
            mnemonic="s_mov_b64",
            semantic_tag="control.exec.full",
            operands=(_exec_clobber(),),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_EXEC", "EXEC_LO"),
                ),
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SSRC0"),
                    spec.operand_predefined_value("OPR_SSRC", "-1"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b64_exec_full",
                native_assembly_mnemonic="s_mov_b64",
                native_assembly_values=(
                    _native_literal("exec"),
                    _native_literal("-1"),
                ),
            ),
            effects=(_CONVERGENT_EFFECT,),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b64_opcode,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b64_exec_read",
            mnemonic="s_mov_b64",
            semantic_tag="control.exec.read",
            operands=(
                Operand(
                    "dst",
                    OperandRole.RESULT,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SDST"),
                    unit_count=2,
                ),
                _exec_state_read(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SSRC0"),
                    spec.operand_predefined_value("OPR_SSRC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(mnemonic="s_mov_b64_exec_read", results=("dst",)),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            encoding_id=s_mov_b64_opcode,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_xor_b64_exec",
            mnemonic="s_xor_b64",
            semantic_tag="control.exec.xor",
            operands=(
                _scc_result("active"),
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                    unit_count=2,
                ),
                _exec_clobber("exec_out"),
                _exec_state_read(),
            ),
            encoding_field_values=(
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SDST"),
                    spec.operand_predefined_value("OPR_SDST_EXEC", "EXEC_LO"),
                ),
                EncodingFieldValue(
                    amdgpu_encoding_field_id("SSRC1"),
                    spec.operand_predefined_value("OPR_SSRC", "EXEC_LO"),
                ),
            ),
            asm_forms=_asm(
                mnemonic="s_xor_b64_exec", results=("active",), operands=("src",)
            ),
            effects=(_CONVERGENT_EFFECT,),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2,
            encoding_id=s_xor_b64_opcode,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    )


_VMCNT_IMMEDIATE = Immediate(
    "vmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
    default_value=(2**6) - 1,
)

_LGKMCNT_4BIT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=4,
    encoding_id=_WAIT_COUNTER_LGKM_ENCODING_ID,
    unsigned_max=(2**4) - 1,
    default_value=(2**4) - 1,
)

_LGKMCNT_6BIT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LGKM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
    default_value=(2**6) - 1,
)

_LOADCNT_IMMEDIATE = Immediate(
    "loadcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_STORECNT_IMMEDIATE = Immediate(
    "storecnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_VSCNT_IMMEDIATE = Immediate(
    "vscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_DSCNT_IMMEDIATE = Immediate(
    "dscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LDS_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_KMCNT_IMMEDIATE = Immediate(
    "kmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    encoding_id=_WAIT_COUNTER_SMEM_ENCODING_ID,
    unsigned_max=(2**5) - 1,
)

_DEPCTR_IMMEDIATE = Immediate(
    "depctr",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    encoding_id=_WAIT_COUNTER_ALU_ENCODING_ID,
    unsigned_max=(2**16) - 1,
)

_PREFETCH_COUNT_IMMEDIATE = Immediate(
    "count",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    unsigned_max=(2**5) - 1,
)

_PREFETCH_DISTANCE_IMMEDIATE = Immediate(
    "distance",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    unsigned_max=(2**16) - 1,
)

_MATRIX_A_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_IMMEDIATE = Immediate(
    "matrix_a_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_IMMEDIATE = Immediate(
    "matrix_b_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_REUSE_IMMEDIATE = Immediate(
    "matrix_a_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_MATRIX_B_REUSE_IMMEDIATE = Immediate(
    "matrix_b_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_GLOBAL_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_LOAD_B16_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=16,
)

_GLOBAL_LOAD_B8_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=8,
)

_GLOBAL_LOAD_B64_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_LOAD_B96_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=96,
)

_GLOBAL_LOAD_B128_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_GLOBAL_LOAD_B256_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=256,
)

_GLOBAL_LOAD_B512_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=512,
)

_GLOBAL_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_STORE_B16_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=16,
)

_GLOBAL_STORE_B8_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=8,
)

_GLOBAL_STORE_B64_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_STORE_B96_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=96,
)

_GLOBAL_STORE_B128_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)


def _stack_memory_effect(kind: EffectKind, width_bits: int) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.STACK,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _workgroup_memory_effect(kind: EffectKind, width_bits: int) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.WORKGROUP,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


_WORKGROUP_BARRIER_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_CACHE_CONTROL_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_GLOBAL_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
)

_INSTRUCTION_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
)

_CONVERGENT_EFFECT = Effect(
    EffectKind.CONVERGENT,
    flags=(EffectFlag.ORDERED,),
)


def _ds_crosslane_effects(width_bits: int) -> tuple[Effect, Effect]:
    # DS cross-lane instructions retire through LGKM even though they do not
    # access LDS memory. Their SSA results must still wait on the LDS counter.
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GENERIC,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_LDS,
            width_bits=width_bits,
        ),
        _CONVERGENT_EFFECT,
    )


_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_VMEM_LOAD_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_LOAD,
)

_VMEM_STORE_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_STORE,
)

_LDS_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_LDS,
)

_SMEM_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_SMEM,
)

_ALU_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_ALU,
)


_EARLY_CLOBBER_RESULT_CONSTRAINTS = (Constraint(ConstraintKind.EARLY_CLOBBER, 0),)


def _destructive_accumulator_constraints(
    accumulator_operand_index: int,
) -> tuple[Constraint, ...]:
    return (
        Constraint(ConstraintKind.TIED, 0, accumulator_operand_index),
        Constraint(ConstraintKind.DESTRUCTIVE, 0, accumulator_operand_index),
        Constraint(ConstraintKind.EARLY_CLOBBER, 0),
    )


_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS = _destructive_accumulator_constraints(1)
_BUFFER_ATOMIC_VDATA_INPUT_REASON = "xml-models-buffer-atomic-vdata-as-output-only"
_SMFMAC_VDST_ACCUMULATOR_REASON = "xml-models-smfmac-accumulator-as-vdst"
_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)
_PSEUDO_DEAD_REMOVABLE_FLAGS = (DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.PSEUDO)


def _hal_buffer_descriptor_pseudos() -> tuple[Descriptor, ...]:
    return (
        Descriptor(
            key="amdgpu.hal.buffer_descriptor",
            mnemonic=None,
            semantic_tag="memory.hal.buffer_descriptor",
            operands=(
                _sgpr_result("descriptor", units=4),
                _sgpr_operand("binding", units=2),
            ),
            immediates=(
                _HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE,
                _HAL_BUFFER_DESCRIPTOR_EXTENT_IMMEDIATE,
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
            asm_surface=DescriptorAsmSurface.GENERATED_ONLY,
            asm_surface_reason="expanded by AMDGPU HAL buffer descriptor lowering",
        ),
        Descriptor(
            key="amdgpu.hal.buffer_descriptor.extent",
            mnemonic=None,
            semantic_tag="memory.hal.buffer_descriptor",
            operands=(
                _sgpr_result("descriptor", units=4),
                _sgpr_operand("binding", units=2),
                _sgpr_operand("extent"),
            ),
            immediates=(_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE,),
            schedule_class=_SCHEDULE_SALU,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
            asm_surface=DescriptorAsmSurface.GENERATED_ONLY,
            asm_surface_reason="expanded by AMDGPU HAL buffer descriptor lowering",
        ),
    )


def _offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return _named_offset_immediate("offset", bit_width, encoding_id=encoding_id)


def _named_offset_immediate(
    field_name: str,
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        encoding_id=encoding_id,
        unsigned_max=(2**bit_width) - 1,
    )


def _ds_offset_immediate() -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=16,
        encoding_id=_ADDRESS_OFFSET_DS16_ENCODING_ID,
        encoding_slices=(
            ImmediateEncodingSlice(amdgpu_encoding_field_id("OFFSET0"), 0, 8),
            ImmediateEncodingSlice(amdgpu_encoding_field_id("OFFSET1"), 8, 8),
        ),
        unsigned_max=(2**16) - 1,
    )


def _ds_fixed_fields_without_offset1(
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...],
) -> tuple[tuple[str, AmdgpuFixedEncodingValue], ...]:
    return tuple(
        fixed_field
        for fixed_field in fixed_encoding_fields
        if fixed_field[0] != "OFFSET1"
    )


def _signed_offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.SIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        encoding_id=encoding_id,
        signed_min=-(2 ** (bit_width - 1)),
        unsigned_max=(2 ** (bit_width - 1)) - 1,
    )


def _cache_immediate(
    field_name: str, bit_width: int, *, default_value: int = 0
) -> Immediate:
    return Immediate(
        field_name.lower(),
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        unsigned_max=(2**bit_width) - 1,
        default_value=default_value,
    )


def _cache_field_names(cache_fields: tuple[tuple[str, int], ...]) -> tuple[str, ...]:
    return tuple(field_name for field_name, _bit_width in cache_fields)


def _memory_asm_immediate_names(
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[str, ...]:
    return ("offset", *(field_name.lower() for field_name, _bit_width in cache_fields))


def _cache_immediates(
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(field_name, bit_width)
        for field_name, bit_width in cache_fields
    )


def _cache_immediates_with_defaults(
    cache_fields: tuple[tuple[str, int], ...],
    defaults: dict[str, int],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(
            field_name, bit_width, default_value=defaults.get(field_name, 0)
        )
        for field_name, bit_width in cache_fields
    )


def _scc_output(descriptor_operand: Operand) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SSRC_SPECIAL_SCC",
        descriptor_operand=descriptor_operand,
        data_format_name="FMT_NUM_B1",
        size_bits=1,
        is_input=False,
        is_output=True,
    )


def _scc_input(descriptor_operand: Operand) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SSRC_SPECIAL_SCC",
        descriptor_operand=descriptor_operand,
        data_format_name="FMT_NUM_B1",
        size_bits=1,
        is_input=True,
        is_output=False,
    )


_SCC_CLOBBER_OUTPUT = _scc_output(_scc_clobber())

_IGNORE_GLOBAL_READ_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B16",
    size_bits=16,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B8 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B8",
    size_bits=8,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_I8 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_I8",
    size_bits=8,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_U8 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_U8",
    size_bits=8,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_I16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_I16",
    size_bits=16,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_U16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_U16",
    size_bits=16,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B96 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B96",
    size_bits=96,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B256 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B256",
    size_bits=256,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B512 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B512",
    size_bits=512,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B16 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B16",
    size_bits=16,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B8 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B8",
    size_bits=8,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B96 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B96",
    size_bits=96,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)


def _ignore_workgroup_memory(
    *, width_bits: int, is_input: bool, data_format_name: str | None = None
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_DSMEM",
        data_format_name=data_format_name or f"FMT_NUM_B{width_bits}",
        size_bits=width_bits,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-workgroup-read-effect"
            if is_input
            else "modeled-by-workgroup-write-effect"
        ),
    )


def _ignore_global_read_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 8:
            return _IGNORE_GLOBAL_READ_MEMORY_B8
        case 16:
            return _IGNORE_GLOBAL_READ_MEMORY_B16
        case 32:
            return _IGNORE_GLOBAL_READ_MEMORY
        case 64:
            return _IGNORE_GLOBAL_READ_MEMORY_B64
        case 96:
            return _IGNORE_GLOBAL_READ_MEMORY_B96
        case 128:
            return _IGNORE_GLOBAL_READ_MEMORY_B128
        case 256:
            return _IGNORE_GLOBAL_READ_MEMORY_B256
        case 512:
            return _IGNORE_GLOBAL_READ_MEMORY_B512
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _ignore_global_write_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 8:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B8
        case 16:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B16
        case 32:
            return _IGNORE_GLOBAL_WRITE_MEMORY
        case 64:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B64
        case 96:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B96
        case 128:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B128
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _global_read_effect(width_bits: int) -> Effect:
    match width_bits:
        case 8:
            return _GLOBAL_LOAD_B8_EFFECT
        case 16:
            return _GLOBAL_LOAD_B16_EFFECT
        case 32:
            return _GLOBAL_LOAD_EFFECT
        case 64:
            return _GLOBAL_LOAD_B64_EFFECT
        case 96:
            return _GLOBAL_LOAD_B96_EFFECT
        case 128:
            return _GLOBAL_LOAD_B128_EFFECT
        case 256:
            return _GLOBAL_LOAD_B256_EFFECT
        case 512:
            return _GLOBAL_LOAD_B512_EFFECT
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _global_write_effect(width_bits: int) -> Effect:
    match width_bits:
        case 8:
            return _GLOBAL_STORE_B8_EFFECT
        case 16:
            return _GLOBAL_STORE_B16_EFFECT
        case 32:
            return _GLOBAL_STORE_EFFECT
        case 64:
            return _GLOBAL_STORE_B64_EFFECT
        case 96:
            return _GLOBAL_STORE_B96_EFFECT
        case 128:
            return _GLOBAL_STORE_B128_EFFECT
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _ignore_global_atomic_memory(
    *, data_format_name: str, is_input: bool
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name,
        size_bits=32,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-global-atomic-read-effect"
            if is_input
            else "modeled-by-global-atomic-write-effect"
        ),
    )


def _ignore_generic_atomic_memory(
    *, data_format_name: str, is_input: bool
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name,
        size_bits=32,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-generic-atomic-read-effect"
            if is_input
            else "modeled-by-generic-atomic-write-effect"
        ),
    )


_IGNORE_FLAT_SCRATCH_INPUT = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_FLAT_SCRATCH",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-flat-address-state",
)


def _ignore_scratch_memory(
    *, width_bits: int, is_input: bool, data_format_name: str | None = None
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name or f"FMT_NUM_B{width_bits}",
        size_bits=width_bits,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-stack-read-effect"
            if is_input
            else "modeled-by-stack-write-effect"
        ),
    )


def _atomic_effects(
    memory_space: MemorySpace, width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
    )


def _global_atomic_effects(
    width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return _atomic_effects(MemorySpace.GLOBAL, width_bits, counter_id=counter_id)


def _generic_atomic_effects(
    width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return _atomic_effects(MemorySpace.GENERIC, width_bits, counter_id=counter_id)


def _global_to_lds_effects(
    global_width_bits: int,
    *,
    workgroup_width_bits: int | None = None,
) -> tuple[Effect, Effect]:
    workgroup_width_bits = workgroup_width_bits or global_width_bits
    return (
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


def _implicit_m0_input(
    *, xml_operand_required: bool = True
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SDST_M0",
        descriptor_operand=_m0_implicit_resource(),
        data_format_name="FMT_NUM_B32",
        size_bits=32,
        is_input=True,
        is_output=False,
        xml_operand_required=xml_operand_required,
    )


def _implicit_m0_clobber() -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SDST_M0",
        descriptor_operand=_m0_clobber(),
        data_format_name="FMT_NUM_B32",
        size_bits=32,
        is_input=True,
        is_output=False,
    )


__all__ = (
    "AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY",
    "AMDGPU_CACHE_DESCRIPTOR_CATEGORY",
    "AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY",
    "AMDGPU_CONTROL_DESCRIPTOR_CATEGORY",
    "AMDGPU_CONVERT_DESCRIPTOR_CATEGORY",
    "AMDGPU_DESCRIPTOR_CATEGORIES",
    "AMDGPU_ENCODING_FORMAT_DS",
    "AMDGPU_ENCODING_FORMAT_FLAT",
    "AMDGPU_ENCODING_FORMAT_MUBUF",
    "AMDGPU_ENCODING_FORMAT_SOP1",
    "AMDGPU_ENCODING_FORMAT_SOP2",
    "AMDGPU_ENCODING_FORMAT_SOP2_LITERAL",
    "AMDGPU_ENCODING_FORMAT_SOPP",
    "AMDGPU_ENCODING_FORMAT_VBUFFER",
    "AMDGPU_ENCODING_FORMAT_VDS",
    "AMDGPU_ENCODING_FORMAT_VFLAT",
    "AMDGPU_ENCODING_FORMAT_VGLOBAL",
    "AMDGPU_ENCODING_FORMAT_VOP1",
    "AMDGPU_ENCODING_FORMAT_VOP1_DPP",
    "AMDGPU_ENCODING_FORMAT_VOP1_DPP16",
    "AMDGPU_ENCODING_FORMAT_VOP1_LITERAL",
    "AMDGPU_ENCODING_FORMAT_VOP1_SDWA",
    "AMDGPU_ENCODING_FORMAT_VOP2",
    "AMDGPU_ENCODING_FORMAT_VOP2_LITERAL",
    "AMDGPU_ENCODING_FORMAT_VOP3",
    "AMDGPU_ENCODING_FORMAT_VOP3_LITERAL",
    "AMDGPU_ENCODING_FORMAT_VOP3P",
    "AMDGPU_ENCODING_FORMAT_VOP3P_LITERAL",
    "AMDGPU_ENCODING_FORMAT_VOP3PX2",
    "AMDGPU_ENCODING_FORMAT_VOP3_SDST",
    "AMDGPU_ENCODING_FORMAT_VSCRATCH",
    "AMDGPU_MATRIX_DESCRIPTOR_CATEGORY",
    "AMDGPU_MEMORY_DESCRIPTOR_CATEGORY",
    "AMDGPU_MISC_DESCRIPTOR_CATEGORY",
    "AMDGPU_SCALAR_DESCRIPTOR_CATEGORY",
    "AMDGPU_VECTOR_DESCRIPTOR_CATEGORY",
    "AmdgpuAtomicDescriptorCandidate",
    "AmdgpuAtomicKind",
    "AmdgpuAtomicMemorySpace",
    "AmdgpuAtomicOperationKind",
    "AmdgpuAtomicValueKind",
    "AmdgpuDescriptorOverlay",
    "AmdgpuEncodingFieldAllOnes",
    "AmdgpuFixedEncodingValue",
    "AmdgpuIgnoredOperandOverlay",
    "AmdgpuImplicitOperandOverlay",
    "AmdgpuIsaFactSource",
    "AmdgpuMemoryAddressForm",
    "AmdgpuOperandOverlay",
    "AmdgpuOperandPredefinedValueRef",
    "AsmForm",
    "AsmImmediate",
    "CEnum",
    "Callable",
    "Constraint",
    "ConstraintKind",
    "Descriptor",
    "DescriptorAsmSurface",
    "DescriptorCategory",
    "DescriptorFlag",
    "DescriptorSet",
    "Effect",
    "EffectFlag",
    "EffectKind",
    "EncodingFieldValue",
    "EnumDomain",
    "EnumValue",
    "Hazard",
    "HazardKind",
    "Immediate",
    "ImmediateEncodingSlice",
    "ImmediateFlag",
    "ImmediateKind",
    "IssueUse",
    "LOW_DESCRIPTOR_ENCODING_ID_NONE",
    "LatencyKind",
    "MemorySpace",
    "ModelQuality",
    "NativeAsmValue",
    "NativeAsmValueKind",
    "Operand",
    "OperandAddressMapKind",
    "OperandFlag",
    "OperandForm",
    "OperandFormImmediateAction",
    "OperandFormMatch",
    "OperandFormMatchKind",
    "OperandRole",
    "Path",
    "RegClass",
    "RegClassAlt",
    "RegClassAltFlag",
    "RegClassFlag",
    "RegisterPart",
    "Resource",
    "ResourceKind",
    "ScheduleClass",
    "ScheduleClassFlag",
    "Sequence",
    "SpillSlotSpace",
    "StorageLease",
    "StorageLeaseAttachment",
    "StorageLeaseFlag",
    "StorageLeaseKind",
    "StorageLeaseReleaseScope",
    "_ADDRESS_OFFSET_BYTE_ENCODING_ID",
    "_ADDRESS_OFFSET_DS16_ENCODING_ID",
    "_ADDRESS_OFFSET_DWORD_ENCODING_ID",
    "_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID",
    "_ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS",
    "_ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES",
    "_ADDRESS_OFFSET_QWORD_ENCODING_ID",
    "_ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID",
    "_ALU_COUNTER_HAZARD",
    "_ALU_WAIT_EFFECT",
    "_ALU_WAIT_HAZARDS",
    "_AMDGPU_DESCRIPTOR_PUBLIC_HEADER_DIR",
    "_AMDGPU_DESCRIPTOR_SOURCE_DIR",
    "_AMDGPU_INLINE_F32_ENUM_DOMAIN_NAME",
    "_AMDGPU_REGISTER_PARTS",
    "_AMDGPU_SOURCE_INLINE_F32_ENUM_DOMAIN",
    "_AMDGPU_TRANS_DESCRIPTOR_KEYS",
    "_AMDGPU_TRANS_DESCRIPTOR_LATENCY_CYCLES",
    "_AMDGPU_TRANS_PROXY_LATENCY_CYCLES",
    "_BUFFER_ATOMIC_VDATA_INPUT_REASON",
    "_CACHE_CONTROL_EFFECT",
    "_CDNA_SMEM_OFFSET_ONLY_FIXED_FIELDS",
    "_CDNA_SMEM_SGPR_IMM_FIXED_FIELDS",
    "_CONVERGENT_EFFECT",
    "_COUNTER_ALU",
    "_COUNTER_LDS",
    "_COUNTER_SMEM",
    "_COUNTER_VMEM_LOAD",
    "_COUNTER_VMEM_STORE",
    "_D16_PARTIAL_REGISTER_SIZE_REASON",
    "_DEPCTR_IMMEDIATE",
    "_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS",
    "_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS",
    "_DSCNT_IMMEDIATE",
    "_EXECUTION_MASKED_SCHEDULE_CLASSES",
    "_EXEC_ALT",
    "_GFX11_MEMORY_WAIT_HAZARDS",
    "_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS",
    "_GFX12_TH_ATOMIC_RETURN_VALUE",
    "_GFX12_VECTOR_CACHE_FIELDS",
    "_GFX950_MEMORY_WAIT_HAZARDS",
    "_GFX950_VECTOR_CACHE_FIELDS",
    "_GFX9_11_VECTOR_CACHE_FIELDS",
    "_GLOBAL_GFX950_SADDR_OFF",
    "_GLOBAL_LOAD_B16_EFFECT",
    "_GLOBAL_LOAD_B128_EFFECT",
    "_GLOBAL_LOAD_B256_EFFECT",
    "_GLOBAL_LOAD_B512_EFFECT",
    "_GLOBAL_LOAD_B8_EFFECT",
    "_GLOBAL_LOAD_B64_EFFECT",
    "_GLOBAL_LOAD_B96_EFFECT",
    "_GLOBAL_LOAD_EFFECT",
    "_GLOBAL_PREFETCH_EFFECT",
    "_GLOBAL_SADDR_OFF",
    "_GLOBAL_SADDR_OFFSET_ONLY_SIZE_REASON",
    "_SCRATCH_CDNA_SADDR_OFF",
    "_GLOBAL_STORE_B128_EFFECT",
    "_GLOBAL_STORE_B16_EFFECT",
    "_GLOBAL_STORE_B8_EFFECT",
    "_GLOBAL_STORE_B64_EFFECT",
    "_GLOBAL_STORE_B96_EFFECT",
    "_GLOBAL_STORE_EFFECT",
    "_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_STRIDE_IMMEDIATE",
    "_HAL_BUFFER_DESCRIPTOR_EXTENT_IMMEDIATE",
    "_IDLE_WAIT_HAZARDS",
    "_IGNORE_FLAT_SCRATCH_INPUT",
    "_IGNORE_GLOBAL_READ_MEMORY",
    "_IGNORE_GLOBAL_READ_MEMORY_B16",
    "_IGNORE_GLOBAL_READ_MEMORY_B128",
    "_IGNORE_GLOBAL_READ_MEMORY_B256",
    "_IGNORE_GLOBAL_READ_MEMORY_B512",
    "_IGNORE_GLOBAL_READ_MEMORY_B8",
    "_IGNORE_GLOBAL_READ_MEMORY_B64",
    "_IGNORE_GLOBAL_READ_MEMORY_B96",
    "_IGNORE_GLOBAL_READ_MEMORY_I16",
    "_IGNORE_GLOBAL_READ_MEMORY_I8",
    "_IGNORE_GLOBAL_READ_MEMORY_U8",
    "_IGNORE_GLOBAL_READ_MEMORY_U16",
    "_IGNORE_GLOBAL_WRITE_MEMORY",
    "_IGNORE_GLOBAL_WRITE_MEMORY_B128",
    "_IGNORE_GLOBAL_WRITE_MEMORY_B16",
    "_IGNORE_GLOBAL_WRITE_MEMORY_B8",
    "_IGNORE_GLOBAL_WRITE_MEMORY_B64",
    "_IGNORE_GLOBAL_WRITE_MEMORY_B96",
    "_INSTRUCTION_PREFETCH_EFFECT",
    "_KMCNT_IMMEDIATE",
    "_LDS_COUNTER_HAZARD",
    "_LDS_WAIT_EFFECT",
    "_LDS_WAIT_HAZARDS",
    "_LGKMCNT_4BIT_IMMEDIATE",
    "_LGKMCNT_6BIT_IMMEDIATE",
    "_LITERAL_U32_IMMEDIATE",
    "_LOADCNT_IMMEDIATE",
    "_M0_ALT",
    "_MODE_ALT",
    "_MANUAL_SCALAR_DESCRIPTOR_KEYS",
    "_MATRIX_A_FORMAT_IMMEDIATE",
    "_MATRIX_A_REUSE_IMMEDIATE",
    "_MATRIX_A_SCALE_FORMAT_IMMEDIATE",
    "_MATRIX_A_SCALE_IMMEDIATE",
    "_MATRIX_B_FORMAT_IMMEDIATE",
    "_MATRIX_B_REUSE_IMMEDIATE",
    "_MATRIX_B_SCALE_FORMAT_IMMEDIATE",
    "_MATRIX_B_SCALE_IMMEDIATE",
    "_MUBUF_SOFFSET_INLINE_ZERO",
    "_MUBUF_VADDR_OFFSET_ONLY_SIZE_REASON",
    "_VBUFFER_SOFFSET_NULL",
    "_PREFETCH_COUNT_IMMEDIATE",
    "_PREFETCH_DISTANCE_IMMEDIATE",
    "_PSEUDO_DEAD_REMOVABLE_FLAGS",
    "_REG_AGPR",
    "_REG_EXEC",
    "_REG_M0",
    "_REG_MODE",
    "_REG_PART_SGPR_FULL32_MASK",
    "_REG_PART_SGPR_HIGH16",
    "_REG_PART_SGPR_LOW16",
    "_REG_PART_VGPR_FULL32_MASK",
    "_REG_PART_VGPR_HIGH16",
    "_REG_PART_VGPR_LOW16",
    "_REG_SCC",
    "_REG_SGPR",
    "_REG_VGPR",
    "_RESOURCE_CONTROL",
    "_RESOURCE_LDS_CROSSLANE",
    "_RESOURCE_LDS_LOAD",
    "_RESOURCE_LDS_STORE",
    "_RESOURCE_MFMA",
    "_RESOURCE_SALU",
    "_RESOURCE_SMEM",
    "_RESOURCE_SWMMAC",
    "_RESOURCE_VALU",
    "_RESOURCE_VMEM_LOAD",
    "_RESOURCE_VMEM_STORE",
    "_RESOURCE_WMMA",
    "_SCC_ALT",
    "_SCC_CLOBBER_OUTPUT",
    "_SCHEDULE_BARRIER",
    "_SCHEDULE_CACHE_CONTROL",
    "_SCHEDULE_LDS_ATOMIC",
    "_SCHEDULE_LDS_CROSSLANE",
    "_SCHEDULE_LDS_LOAD",
    "_SCHEDULE_LDS_STORE",
    "_SCHEDULE_MFMA",
    "_SCHEDULE_MODE_CONTROL",
    "_SCHEDULE_SALU",
    "_SCHEDULE_SMEM_LOAD",
    "_SCHEDULE_SMEM_STORE",
    "_SCHEDULE_SWMMAC",
    "_SCHEDULE_TRANS",
    "_SCHEDULE_VALU",
    "_SCHEDULE_VMEM_ATOMIC_NO_RETURN",
    "_SCHEDULE_VMEM_ATOMIC_RETURN",
    "_SCHEDULE_VMEM_LOAD",
    "_SCHEDULE_VMEM_LOAD_LDS",
    "_SCHEDULE_VMEM_STORE",
    "_SCHEDULE_WAIT_ALU",
    "_SCHEDULE_WAIT_IDLE",
    "_SCHEDULE_WAIT_LDS",
    "_SCHEDULE_WAIT_LOAD",
    "_SCHEDULE_WAIT_MEMORY",
    "_SCHEDULE_WAIT_SMEM",
    "_SCHEDULE_WAIT_STORE",
    "_SCHEDULE_WAIT_VMEM_STORE",
    "_SCHEDULE_WMMA",
    "_SCHEDULE_WMMA_SCALE",
    "_SGPR_ALT",
    "_SGPR_REGISTER_PARTS",
    "_SGPR_VGPR_ALT",
    "_SMEM_COUNTER_HAZARD",
    "_SMEM_WAIT_EFFECT",
    "_SMEM_WAIT_HAZARDS",
    "_SMFMAC_VDST_ACCUMULATOR_REASON",
    "_SOURCE_INLINE_F32_ENCODING_ID",
    "_SOURCE_INLINE_F32_IMMEDIATE",
    "_SOURCE_INLINE_U32_16_IMMEDIATE",
    "_SOURCE_INLINE_U32_ENCODING_ID",
    "_SOURCE_INLINE_U32_IMMEDIATE",
    "_STORECNT_IMMEDIATE",
    "_U24_SOURCE_SIZE_REASON",
    "_U32_IMMEDIATE",
    "_VGPR_AGPR_ALT",
    "_VGPR_AGPR_CONST_ALT",
    "_VGPR_ALT",
    "_VGPR_CONST_ALT",
    "_VGPR_REGISTER_PARTS",
    "_VMCNT_IMMEDIATE",
    "_VMEM_LOAD_COUNTER_HAZARD",
    "_VMEM_LOAD_WAIT_EFFECT",
    "_VMEM_LOAD_WAIT_HAZARDS",
    "_VMEM_STORE_COUNTER_HAZARD",
    "_VMEM_STORE_WAIT_EFFECT",
    "_VMEM_STORE_WAIT_HAZARDS",
    "_VSCNT_IMMEDIATE",
    "_WAIT_COUNTER_ALU_ENCODING_ID",
    "_WAIT_COUNTER_LDS_ENCODING_ID",
    "_WAIT_COUNTER_LGKM_ENCODING_ID",
    "_WAIT_COUNTER_SMEM_ENCODING_ID",
    "_WAIT_COUNTER_VMEM_ENCODING_ID",
    "_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID",
    "_WAIT_COUNTER_VMEM_STORE_ENCODING_ID",
    "_WAIT_EFFECT",
    "_WORKGROUP_BARRIER_EFFECT",
    "_amdgpu_camel_case",
    "_amdgpu_core_descriptor_set",
    "_amdgpu_descriptor_set_file_stem",
    "_amdgpu_schedule_class_reads_exec_state",
    "_amdgpu_trans_schedule_class_name",
    "_amdgpu_trans_schedule_classes",
    "_asm",
    "_atomic_effects",
    "_buffer_off_zero_operand_form",
    "_buffer_soffset_offset_operand_form",
    "_cache_field_names",
    "_cache_immediate",
    "_cache_immediates",
    "_cache_immediates_with_defaults",
    "_common_scalar_vector_memory_resources",
    "_common_scalar_vector_memory_schedule_classes",
    "_ds_crosslane_effects",
    "_ds_fixed_fields_without_offset1",
    "_ds_offset_immediate",
    "_EARLY_CLOBBER_RESULT_CONSTRAINTS",
    "_destructive_accumulator_constraints",
    "_exec_clobber",
    "_exec_state_read",
    "_f32_bits",
    "_generic_atomic_effects",
    "_global_addr_operand",
    "_global_atomic_effects",
    "_global_read_effect",
    "_global_to_lds_effects",
    "_global_saddr_asm",
    "_global_vaddr_asm",
    "_global_write_effect",
    "_hal_buffer_descriptor_pseudos",
    "_ignore_generic_atomic_memory",
    "_ignore_global_atomic_memory",
    "_ignore_global_read_memory",
    "_ignore_global_write_memory",
    "_ignore_scratch_memory",
    "_ignore_workgroup_memory",
    "_implicit_m0_clobber",
    "_implicit_m0_input",
    "_instruction_encoding_opcode",
    "_is_exec_state_read",
    "_is_mode_state_read",
    "_literal_operand_form",
    "_m0_clobber",
    "_m0_implicit_resource",
    "_m0_result",
    "_mode_state_read",
    "_mode_state_write",
    "_manual_scalar_descriptors",
    "_matrix_hazards",
    "_memory_asm_immediate_names",
    "_mubuf_vaddr_operand",
    "_native_i64_immediate",
    "_native_literal",
    "_native_operand",
    "_native_result",
    "_native_unsigned_hex_immediate",
    "_named_offset_immediate",
    "_offset_immediate",
    "_predefined",
    "_s_mov_b32_contract_overlay",
    "_scc_clobber",
    "_scc_input",
    "_scc_output",
    "_scc_predicate",
    "_scc_result",
    "_scc_state_read",
    "_sgpr_operand",
    "_sgpr_predicate",
    "_sgpr_resource",
    "_sgpr_result",
    "_sgpr_vgpr_operand",
    "_signed_offset_immediate",
    "_soffset_offset_operand_form",
    "_soffset_zero_operand_form",
    "_source_inline_f32_immediate",
    "_source_inline_u32_16_immediate",
    "_source_inline_u32_immediate",
    "_stack_memory_effect",
    "_u32_immediate",
    "_vgpr_agpr_const_operand",
    "_vgpr_agpr_operand",
    "_vgpr_agpr_result",
    "_vgpr_const_operand",
    "_vgpr_operand",
    "_vgpr_result",
    "_with_execution_mask_state_read",
    "_with_execution_mask_state_reads",
    "_with_mode_state_read",
    "_workgroup_memory_effect",
    "amdgpu_descriptor_set_info_by_generator_target",
    "amdgpu_descriptor_set_ordinal",
    "amdgpu_encoding_field_id",
    "amdgpu_encoding_field_name",
    "dataclass",
    "materialize_amdgpu_descriptor_overlays",
    "parse_amdgpu_isa_xml_path",
    "replace",
    "struct",
    "validate_amdgpu_descriptor_set_isa_xml",
)
