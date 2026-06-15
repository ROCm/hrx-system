# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Wait, barrier, cache-control, and prefetch descriptor overlays."""

from __future__ import annotations

from .common import *


def _s_waitcnt_overlay(
    *,
    effects: tuple[Effect, ...],
    lgkmcnt_immediate: Immediate = _LGKMCNT_6BIT_IMMEDIATE,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt",
        instruction_name="S_WAITCNT",
        mnemonic="s_waitcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt",
        schedule_class=_SCHEDULE_WAIT_MEMORY,
        operands=(),
        immediate_fields=("VM", "LGKM"),
        immediates=(_VMCNT_IMMEDIATE, lgkmcnt_immediate),
        fixed_encoding_fields=(("EXP", AmdgpuEncodingFieldAllOnes()),),
        effects=effects,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_depctr_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_depctr",
        instruction_name="S_WAITCNT_DEPCTR",
        mnemonic="s_waitcnt_depctr",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_vscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_vscnt",
        instruction_name="S_WAITCNT_VSCNT",
        mnemonic="s_waitcnt_vscnt",
        encoding_name="ENC_SOPK",
        semantic_tag="control.waitcnt.vmem_store",
        schedule_class=_SCHEDULE_WAIT_VMEM_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_VSCNT_IMMEDIATE,),
        fixed_encoding_fields=(("SDST", _predefined("NULL", "OPR_SDST")),),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_loadcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_loadcnt",
        instruction_name="S_WAIT_LOADCNT",
        mnemonic="s_wait_loadcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.load",
        schedule_class=_SCHEDULE_WAIT_LOAD,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_LOADCNT_IMMEDIATE,),
        effects=(_VMEM_LOAD_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_storecnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_storecnt",
        instruction_name="S_WAIT_STORECNT",
        mnemonic="s_wait_storecnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.store",
        schedule_class=_SCHEDULE_WAIT_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_STORECNT_IMMEDIATE,),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_dscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_dscnt",
        instruction_name="S_WAIT_DSCNT",
        mnemonic="s_wait_dscnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.lds",
        schedule_class=_SCHEDULE_WAIT_LDS,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DSCNT_IMMEDIATE,),
        effects=(_LDS_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_kmcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_kmcnt",
        instruction_name="S_WAIT_KMCNT",
        mnemonic="s_wait_kmcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.smem",
        schedule_class=_SCHEDULE_WAIT_SMEM,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_KMCNT_IMMEDIATE,),
        effects=(_SMEM_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_alu_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_alu",
        instruction_name="S_WAIT_ALU",
        mnemonic="s_wait_alu",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_idle_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_idle",
        instruction_name="S_WAIT_IDLE",
        mnemonic="s_wait_idle",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.idle",
        schedule_class=_SCHEDULE_WAIT_IDLE,
        operands=(),
        effects=(_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_set_vgpr_msb_descriptor() -> Descriptor:
    return Descriptor(
        key="amdgpu.s_set_vgpr_msb",
        mnemonic="s_set_vgpr_msb",
        semantic_tag="control.mode.vgpr_msb",
        schedule_class=_SCHEDULE_MODE_CONTROL,
        operands=(_mode_state_write(),),
        immediates=(
            Immediate(
                "mode",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                encoding_field_id=amdgpu_encoding_field_id("SIMM16"),
                unsigned_max=0xFFFF,
            ),
        ),
        asm_forms=_asm(immediates=("mode",)),
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOPP,
        encoding_id=0x006,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_barrier_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_barrier",
        instruction_name="S_BARRIER",
        mnemonic="s_barrier",
        encoding_name="ENC_SOPP",
        semantic_tag="control.barrier.workgroup",
        schedule_class=_SCHEDULE_BARRIER,
        operands=(),
        effects=(_WORKGROUP_BARRIER_EFFECT, _CONVERGENT_EFFECT),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _cache_control_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    semantic_tag: str,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=_cache_field_names(cache_fields),
        immediates=_cache_immediates(cache_fields),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_set_inst_prefetch_distance_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_set_inst_prefetch_distance",
        instruction_name="S_SET_INST_PREFETCH_DISTANCE",
        mnemonic="s_set_inst_prefetch_distance",
        encoding_name="ENC_SOPP",
        semantic_tag="memory.cache.prefetch.instruction.distance",
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_PREFETCH_DISTANCE_IMMEDIATE,),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_dcache_discard_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(
            AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_prefetch_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    base_operand: Operand | None,
    effect: Effect,
) -> AmdgpuDescriptorOverlay:
    operands = []
    if base_operand is not None:
        operands.append(AmdgpuOperandOverlay("SBASE", base_operand))
    operands.append(AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")))
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=tuple(operands),
        immediate_fields=("IOFFSET", "SDATA"),
        immediates=(
            _offset_immediate(24),
            _PREFETCH_COUNT_IMMEDIATE,
        ),
        effects=(effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx12_prefetch_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst",
            instruction_name="S_PREFETCH_INST",
            mnemonic="s_prefetch_inst",
            semantic_tag="memory.cache.prefetch.instruction",
            base_operand=_sgpr_operand("base", units=2),
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst_pc_rel",
            instruction_name="S_PREFETCH_INST_PC_REL",
            mnemonic="s_prefetch_inst_pc_rel",
            semantic_tag="memory.cache.prefetch.instruction.pc_relative",
            base_operand=None,
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data",
            instruction_name="S_PREFETCH_DATA",
            mnemonic="s_prefetch_data",
            semantic_tag="memory.cache.prefetch.data",
            base_operand=_sgpr_operand("base", units=2),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_buffer_prefetch_data",
            instruction_name="S_BUFFER_PREFETCH_DATA",
            mnemonic="s_buffer_prefetch_data",
            semantic_tag="memory.cache.prefetch.data.buffer",
            base_operand=_sgpr_resource("resource", units=4),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data_pc_rel",
            instruction_name="S_PREFETCH_DATA_PC_REL",
            mnemonic="s_prefetch_data_pc_rel",
            semantic_tag="memory.cache.prefetch.data.pc_relative",
            base_operand=None,
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
    )


def _gfx950_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_inv",
            instruction_name="BUFFER_INV",
            mnemonic="buffer_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_wbl2",
            instruction_name="BUFFER_WBL2",
            mnemonic="buffer_wbl2",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.writeback.buffer.l2",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb",
            instruction_name="S_DCACHE_WB",
            mnemonic="s_dcache_wb",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv_vol",
            instruction_name="S_DCACHE_INV_VOL",
            mnemonic="s_dcache_inv_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb_vol",
            instruction_name="S_DCACHE_WB_VOL",
            mnemonic="s_dcache_wb_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard",
            instruction_name="S_DCACHE_DISCARD",
            mnemonic="s_dcache_discard",
            semantic_tag="memory.cache.discard.data",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard_x2",
            instruction_name="S_DCACHE_DISCARD_X2",
            mnemonic="s_dcache_discard_x2",
            semantic_tag="memory.cache.discard.data.x2",
        ),
    )


def _gfx11_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl0_inv",
            instruction_name="BUFFER_GL0_INV",
            mnemonic="buffer_gl0_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl0",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl1_inv",
            instruction_name="BUFFER_GL1_INV",
            mnemonic="buffer_gl1_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_gl1_inv",
            instruction_name="S_GL1_INV",
            mnemonic="s_gl1_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.global.l1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_set_inst_prefetch_distance_overlay(),
    )


def _gfx12_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.global_inv",
            instruction_name="GLOBAL_INV",
            mnemonic="global_inv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wb",
            instruction_name="GLOBAL_WB",
            mnemonic="global_wb",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wbinv",
            instruction_name="GLOBAL_WBINV",
            mnemonic="global_wbinv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback_invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
    )


__all__ = (
    "_cache_control_overlay",
    "_gfx11_cache_control_overlays",
    "_gfx12_cache_control_overlays",
    "_gfx12_prefetch_overlays",
    "_gfx950_cache_control_overlays",
    "_s_barrier_overlay",
    "_s_dcache_discard_overlay",
    "_s_prefetch_overlay",
    "_s_set_inst_prefetch_distance_overlay",
    "_s_set_vgpr_msb_descriptor",
    "_s_wait_alu_overlay",
    "_s_wait_dscnt_overlay",
    "_s_wait_idle_overlay",
    "_s_wait_kmcnt_overlay",
    "_s_wait_loadcnt_overlay",
    "_s_wait_storecnt_overlay",
    "_s_waitcnt_depctr_overlay",
    "_s_waitcnt_overlay",
    "_s_waitcnt_vscnt_overlay",
)
