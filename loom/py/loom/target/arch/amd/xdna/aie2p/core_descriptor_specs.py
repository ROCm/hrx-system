# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P semantic descriptor specifications derived from owned machine tables."""

from __future__ import annotations

from dataclasses import dataclass, replace

from loom.target.arch.amd.xdna.aie.machine import (
    has_property,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.low_descriptors import (
    DescriptorOpKind,
    Effect,
    EffectKind,
    MemorySpace,
    RegisterPart,
)

_TARGET_KEY = "amd.xdna.aie2p"
_EL_LOW32_PART = "aie2p.elpredicate.low32"
_EL_HIGH32_PART = "aie2p.elpredicate.high32"
_VEC256_LOW128_PART = "aie2p.vec256.low128"
_VEC256_HIGH128_PART = "aie2p.vec256.high128"
_EWL_LOW128_PART = "aie2p.ewl.low128"
_REGISTER_PARTS = (
    RegisterPart(_EL_LOW32_PART, "aie2p.elpredicate", 0x1),
    RegisterPart(_EL_HIGH32_PART, "aie2p.elpredicate", 0x2),
    RegisterPart(_VEC256_LOW128_PART, "aie2p.vec256", 0x1),
    RegisterPart(_VEC256_HIGH128_PART, "aie2p.vec256", 0x2),
    RegisterPart(_EWL_LOW128_PART, "aie2p.ewl", 0x1),
)
_REGISTER_PARTS_BY_NAME = {part.name: part for part in _REGISTER_PARTS}


@dataclass(frozen=True, slots=True)
class _DescriptorSpec:
    """Semantic selection of one physical form and its exact itinerary."""

    form_name: str
    key: str
    semantic_tag: str
    itinerary: str
    storage_overrides: tuple[tuple[str, str], ...] = ()
    op_kind: DescriptorOpKind = DescriptorOpKind.OP
    implicit_outputs: tuple[str, ...] = ()
    asm_mnemonic: str | None = None
    operand_register_parts: tuple[tuple[str, str], ...] = ()
    encoding_adapter_overrides: tuple[tuple[str, str], ...] = ()
    storage_continuation_part: str | None = None
    schedule_alternatives: tuple[str, ...] = ()
    memory_width_bits: int | None = None
    ordered_memory: bool = False
    effects: tuple[Effect, ...] = ()
    allocation_move: bool = False


_VECTOR_MEMORY_FORM_FAMILIES = (
    (
        128,
        (
            "VLDA_128_dmv_lda_w_idx",
            "VLDA_128_dmv_lda_w_idx_imm",
            "OP_mWa",
        ),
        ("VLDB_128_idx", "VLDB_128_idx_imm", "OP_mWb"),
        (
            "VST_128_dmv_sts_w_idx",
            "VST_128_dmv_sts_w_idx_imm",
            "OP_mWs",
        ),
    ),
    (
        256,
        ("VLDA_dmw_lda_w_idx", "VLDA_dmw_lda_w_idx_imm", "OP_mWa"),
        ("VLDB_dmw_ldb_idx", "VLDB_dmw_ldb_idx_imm", "OP_mWb"),
        ("VST_dmw_sts_w_idx", "VST_dmw_sts_w_idx_imm", "OP_mWs"),
    ),
    (
        512,
        ("VLDA_dmx_lda_x_idx", "VLDA_dmx_lda_x_idx_imm", None),
        ("VLDB_dmx_ldb_x_idx", "VLDB_dmx_ldb_x_idx_imm", None),
        ("VST_dmx_sts_x_idx", "VST_dmx_sts_x_idx_imm", None),
    ),
)

# Value types with native register layouts for the bit-preserving vector
# load/store forms below.
AIE2P_VECTOR_MEMORY_ELEMENT_TYPES = (
    ("i8", 8),
    ("i16", 16),
    ("bf16", 16),
    ("i32", 32),
    ("f32", 32),
)


def _vector_memory_operand_overrides(
    width_bits: int,
    element_type: str,
    operand_name: str,
    native_adapter: str | None,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Returns storage, part, and encoding overrides for one memory operand."""

    if width_bits != 128:
        return (), (), ()
    if native_adapter is None:
        raise ValueError("128-bit vector memory forms need an encoding adapter")
    if element_type == "bf16":
        return (
            ((operand_name, "eWL"),),
            ((operand_name, _EWL_LOW128_PART),),
            ((operand_name, f"LOOM_eWL_{native_adapter}"),),
        )
    return (
        (),
        ((operand_name, _VEC256_LOW128_PART),),
        ((operand_name, native_adapter),),
    )


def _vector_memory_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects every exact-width native vector memory form."""

    result = []
    for width_bits, load_a, load_b, store in _VECTOR_MEMORY_FORM_FAMILIES:
        for element_type, element_bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES:
            shape = f"{element_type}x{width_bits // element_bits}"
            load_a_register_key = f"{_TARGET_KEY}.load.a.{shape}.indexed.register"
            load_a_immediate_key = f"{_TARGET_KEY}.load.a.{shape}.indexed.immediate"
            load_b_register_key = f"{_TARGET_KEY}.load.b.{shape}.indexed.register"
            load_b_immediate_key = f"{_TARGET_KEY}.load.b.{shape}.indexed.immediate"
            store_register_key = f"{_TARGET_KEY}.store.{shape}.indexed.register"
            store_immediate_key = f"{_TARGET_KEY}.store.{shape}.indexed.immediate"
            load_a_overrides = _vector_memory_operand_overrides(
                width_bits, element_type, "dst", load_a[2]
            )
            load_b_overrides = _vector_memory_operand_overrides(
                width_bits, element_type, "dst", load_b[2]
            )
            store_overrides = _vector_memory_operand_overrides(
                width_bits, element_type, "src", store[2]
            )
            load_storage_continuation_part = (
                _VEC256_HIGH128_PART
                if width_bits == 128 and element_type != "bf16"
                else None
            )
            result.extend(
                (
                    _DescriptorSpec(
                        load_a[1],
                        load_a_immediate_key,
                        f"memory.load.indexed.{shape}",
                        f"II_{load_a[1]}",
                        storage_overrides=load_a_overrides[0],
                        asm_mnemonic=f"vlda.{width_bits}.{shape}",
                        operand_register_parts=load_a_overrides[1],
                        encoding_adapter_overrides=load_a_overrides[2],
                        storage_continuation_part=load_storage_continuation_part,
                        schedule_alternatives=(load_b_immediate_key,),
                        memory_width_bits=width_bits,
                    ),
                    _DescriptorSpec(
                        load_a[0],
                        load_a_register_key,
                        f"memory.load.indexed.{shape}",
                        f"II_{load_a[0]}",
                        storage_overrides=load_a_overrides[0],
                        asm_mnemonic=f"vlda.{width_bits}.{shape}.index",
                        operand_register_parts=load_a_overrides[1],
                        encoding_adapter_overrides=load_a_overrides[2],
                        storage_continuation_part=load_storage_continuation_part,
                        schedule_alternatives=(load_b_register_key,),
                        memory_width_bits=width_bits,
                    ),
                    _DescriptorSpec(
                        load_b[1],
                        load_b_immediate_key,
                        f"memory.load.indexed.{shape}",
                        f"II_{load_b[1]}",
                        storage_overrides=load_b_overrides[0],
                        asm_mnemonic=f"vldb.{width_bits}.{shape}",
                        operand_register_parts=load_b_overrides[1],
                        encoding_adapter_overrides=load_b_overrides[2],
                        storage_continuation_part=load_storage_continuation_part,
                        memory_width_bits=width_bits,
                    ),
                    _DescriptorSpec(
                        load_b[0],
                        load_b_register_key,
                        f"memory.load.indexed.{shape}",
                        f"II_{load_b[0]}",
                        storage_overrides=load_b_overrides[0],
                        asm_mnemonic=f"vldb.{width_bits}.{shape}.index",
                        operand_register_parts=load_b_overrides[1],
                        encoding_adapter_overrides=load_b_overrides[2],
                        storage_continuation_part=load_storage_continuation_part,
                        memory_width_bits=width_bits,
                    ),
                    _DescriptorSpec(
                        store[1],
                        store_immediate_key,
                        f"memory.store.indexed.{shape}",
                        f"II_{store[1]}",
                        storage_overrides=store_overrides[0],
                        asm_mnemonic=f"vst.{width_bits}.{shape}",
                        operand_register_parts=store_overrides[1],
                        encoding_adapter_overrides=store_overrides[2],
                        memory_width_bits=width_bits,
                    ),
                    _DescriptorSpec(
                        store[0],
                        store_register_key,
                        f"memory.store.indexed.{shape}",
                        f"II_{store[0]}",
                        storage_overrides=store_overrides[0],
                        asm_mnemonic=f"vst.{width_bits}.{shape}.index",
                        operand_register_parts=store_overrides[1],
                        encoding_adapter_overrides=store_overrides[2],
                        memory_width_bits=width_bits,
                    ),
                )
            )
    return tuple(result)


_INTEGER_MATRIX_NUMERIC_KINDS = ("s8s8", "u8s8", "s8u8", "u8u8")


def _integer_matrix_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects configured 8x8x8 integer multiply and accumulate forms."""

    operations = (
        (
            "multiply",
            "VMUL_vmul_cm_core_X_X",
            "II_VMUL_vmul_cm_core_X_X",
            "mmul",
            (("dst", "mBMs"),),
        ),
        (
            "accumulate",
            "VMAC_vmul_cm_core_X_X",
            "II_VMAC_vmul_cm_core_X_X",
            "mma",
            (("dst", "mBMs"), ("acc1", "mBMs")),
        ),
    )
    return tuple(
        _DescriptorSpec(
            form_name,
            f"{_TARGET_KEY}.matrix.{operation}.{numeric_kind}.m8n8k8.configured",
            f"matrix.{operation}.{numeric_kind}.m8n8k8.configured",
            itinerary,
            storage_overrides=storage_overrides,
            asm_mnemonic=f"{mnemonic}.{numeric_kind}.m8n8k8",
        )
        for operation, form_name, itinerary, mnemonic, storage_overrides in operations
        for numeric_kind in _INTEGER_MATRIX_NUMERIC_KINDS
    )


def _packed_dot_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects the configured 8-bit channel multiply used by dot4i."""

    return (
        _DescriptorSpec(
            "VMUL_vmul_cm_core_Y_X",
            f"{_TARGET_KEY}.dot4i.i8x64.configured",
            "integer.dot4i.i8x64.configured",
            "II_VMUL_vmul_cm_core_Y_X",
            storage_overrides=(("dst", "mBMs"), ("s1", "VEC256")),
            asm_mnemonic="dot4i.i8x64",
        ),
    )


def _packed_i4_unpack_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native 64-lane signed and unsigned 4-to-8-bit unpack forms."""

    return tuple(
        _DescriptorSpec(
            f"VUNPACK_mv_unpack_w_unpackSign{sign_bit}",
            f"{_TARGET_KEY}.unpack.{source_kind}4x64.to.{source_kind}8x64.configured",
            f"integer.unpack.{source_kind}4x64.to.{source_kind}8x64.configured",
            f"II_VUNPACK_mv_unpack_w_unpackSign{sign_bit}",
            asm_mnemonic=f"vunpack.{source_kind}4.to.{source_kind}8x64",
        )
        for source_kind, sign_bit in (("u", 0), ("s", 1))
    )


def _scalar_memory_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Builds exact-width scalar load and store descriptors."""

    result = []
    for (
        element_type,
        memory_width_bits,
        load_immediate_form,
        load_immediate_itinerary,
        load_register_form,
        load_register_itinerary,
        store_immediate_form,
        store_immediate_itinerary,
        store_register_form,
        store_register_itinerary,
    ) in (
        (
            "i8",
            8,
            "LDA_u8_idx_imm",
            "II_LDA_u8_idx_imm",
            "LDA_u8_idx",
            "II_LDA_u8_idx",
            "ST_s8_idx_imm",
            "II_ST_s8_idx_imm",
            "ST_s8_idx",
            "II_ST_s8_idx",
        ),
        (
            "i16",
            16,
            "LDA_u16_idx_imm",
            "II_LDA_u16_idx_imm",
            "LDA_u16_idx",
            "II_LDA_u16_idx",
            "ST_s16_idx_imm",
            "II_ST_s16_idx_imm",
            "ST_s16_idx",
            "II_ST_s16_idx",
        ),
        (
            "i32",
            32,
            "LDA_dms_lda_idx_imm",
            "II_LDA_dms_lda_idx_imm_eR",
            "LDA_dms_lda_idx",
            "II_LDA_dms_lda_idx_eR",
            "ST_dms_sts_idx_imm",
            "II_ST_dms_sts_idx_imm_eR",
            "ST_dms_sts_idx",
            "II_ST_dms_sts_idx_eR",
        ),
    ):
        result.extend(
            (
                _DescriptorSpec(
                    load_immediate_form,
                    f"{_TARGET_KEY}.load.scalar.{element_type}.indexed.immediate",
                    f"memory.load.indexed.{element_type}",
                    load_immediate_itinerary,
                    (("dst", "eR"),),
                    memory_width_bits=memory_width_bits,
                ),
                _DescriptorSpec(
                    load_register_form,
                    f"{_TARGET_KEY}.load.scalar.{element_type}.indexed.register",
                    f"memory.load.indexed.{element_type}",
                    load_register_itinerary,
                    (("dst", "eR"),),
                    asm_mnemonic=f"lda.{element_type}.index",
                    memory_width_bits=memory_width_bits,
                ),
                _DescriptorSpec(
                    store_immediate_form,
                    f"{_TARGET_KEY}.store.scalar.{element_type}.indexed.immediate",
                    f"memory.store.indexed.{element_type}",
                    store_immediate_itinerary,
                    (("src", "eR"),),
                    memory_width_bits=memory_width_bits,
                ),
                _DescriptorSpec(
                    store_register_form,
                    f"{_TARGET_KEY}.store.scalar.{element_type}.indexed.register",
                    f"memory.store.indexed.{element_type}",
                    store_register_itinerary,
                    (("src", "eR"),),
                    asm_mnemonic=f"st.{element_type}.index",
                    memory_width_bits=memory_width_bits,
                ),
            )
        )
    return tuple(result)


_BASE_DESCRIPTOR_SPECS = (
    _DescriptorSpec(
        "ADD_add_r_ri",
        f"{_TARGET_KEY}.add.i32.immediate",
        "integer.add.i32",
        "II_ADD_add_r_ri",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_alu_fx2flt",
        f"{_TARGET_KEY}.convert.signed.i32.to.f32",
        "conversion.signed.i32.to.f32",
        "II_MOV_alu_mv_alu_fx2flt",
        asm_mnemonic="convert.signed.i32.to.f32",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_alu_flt2fx",
        f"{_TARGET_KEY}.convert.round-nearest.f32.to.signed.i32",
        "conversion.round-nearest.f32.to.signed.i32",
        "II_MOV_alu_mv_alu_flt2fx",
        asm_mnemonic="convert.round-nearest.f32.to.signed.i32",
    ),
    _DescriptorSpec(
        "INV_mRx",
        f"{_TARGET_KEY}.reciprocal.f32",
        "floating.reciprocal.f32",
        "II_INV_mRx",
        asm_mnemonic="inv.f32",
    ),
    _DescriptorSpec(
        "INVSQRT_mRx",
        f"{_TARGET_KEY}.reciprocal-sqrt.f32",
        "floating.reciprocal-sqrt.f32",
        "II_INVSQRT_mRx",
        asm_mnemonic="invsqrt.f32",
    ),
    _DescriptorSpec(
        "SQRT_mRx",
        f"{_TARGET_KEY}.sqrt.f32",
        "floating.sqrt.f32",
        "II_SQRT_mRx",
        asm_mnemonic="sqrt.f32",
    ),
    _DescriptorSpec(
        "ADD_add_r_ri",
        f"{_TARGET_KEY}.select.mask.i32",
        "integer.select.mask.i32",
        "II_ADD_add_r_ri",
        (("d0", "eRS16"),),
        asm_mnemonic="select.mask",
    ),
    *_scalar_memory_descriptor_specs(),
    _DescriptorSpec(
        "MOVS",
        f"{_TARGET_KEY}.move.to.address-index",
        "register.move.to.address-index",
        "II_MOVS_eDJ_eR",
        (("dst", "eDJ"), ("src", "eR")),
        asm_mnemonic="mov.address-index",
    ),
    _DescriptorSpec(
        "NOP",
        f"{_TARGET_KEY}.nop",
        "control.nop",
        "NoItinerary",
    ),
    _DescriptorSpec(
        "ACQ_mLockId_imm",
        f"{_TARGET_KEY}.lock.acquire.immediate",
        "synchronization.lock.acquire",
        "II_ACQ_mLockId_imm",
        asm_mnemonic="acq",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "ACQ_mLockId_reg",
        f"{_TARGET_KEY}.lock.acquire.register",
        "synchronization.lock.acquire",
        "II_ACQ_mLockId_reg",
        asm_mnemonic="acq.reg",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "ACQ_COND_mLockId_imm",
        f"{_TARGET_KEY}.lock.acquire.conditional.immediate",
        "synchronization.lock.acquire.conditional",
        "II_ACQ_COND_mLockId_imm",
        asm_mnemonic="acq.cond",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "ACQ_COND_mLockId_reg",
        f"{_TARGET_KEY}.lock.acquire.conditional.register",
        "synchronization.lock.acquire.conditional",
        "II_ACQ_COND_mLockId_reg",
        asm_mnemonic="acq.cond.reg",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "REL_mLockId_imm",
        f"{_TARGET_KEY}.lock.release.immediate",
        "synchronization.lock.release",
        "II_REL_mLockId_imm",
        asm_mnemonic="rel",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "REL_mLockId_reg",
        f"{_TARGET_KEY}.lock.release.register",
        "synchronization.lock.release",
        "II_REL_mLockId_reg",
        asm_mnemonic="rel.reg",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "REL_COND_mLockId_imm",
        f"{_TARGET_KEY}.lock.release.conditional.immediate",
        "synchronization.lock.release.conditional",
        "II_REL_COND_mLockId_imm",
        asm_mnemonic="rel.cond",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "REL_COND_mLockId_reg",
        f"{_TARGET_KEY}.lock.release.conditional.register",
        "synchronization.lock.release.conditional",
        "II_REL_COND_mLockId_reg",
        asm_mnemonic="rel.cond.reg",
        effects=(Effect(EffectKind.BARRIER, MemorySpace.WORKGROUP),),
    ),
    _DescriptorSpec(
        "J_lng",
        f"{_TARGET_KEY}.branch.direct",
        "control.branch.direct",
        "II_J_lng",
        asm_mnemonic="j",
    ),
    _DescriptorSpec(
        "JNZ",
        f"{_TARGET_KEY}.branch.nonzero",
        "control.branch.nonzero",
        "II_JNZ",
        asm_mnemonic="jnz",
    ),
    _DescriptorSpec(
        "JZ",
        f"{_TARGET_KEY}.branch.zero",
        "control.branch.zero",
        "II_JZ",
        asm_mnemonic="jz",
    ),
    _DescriptorSpec(
        "RET",
        f"{_TARGET_KEY}.return",
        "control.return",
        "II_RET",
    ),
    _DescriptorSpec(
        "VADD_8",
        f"{_TARGET_KEY}.add.i8x64",
        "integer.add.i8x64",
        "II_VADD_8",
    ),
    _DescriptorSpec(
        "VADD_16",
        f"{_TARGET_KEY}.add.i16x32",
        "integer.add.i16x32",
        "II_VADD_16",
    ),
    _DescriptorSpec(
        "VADD_32",
        f"{_TARGET_KEY}.add.i32x16",
        "integer.add.i32x16",
        "II_VADD_32",
    ),
    _DescriptorSpec(
        "VSUB_8",
        f"{_TARGET_KEY}.sub.i8x64",
        "integer.sub.i8x64",
        "II_VSUB_8",
    ),
    _DescriptorSpec(
        "VSUB_16",
        f"{_TARGET_KEY}.sub.i16x32",
        "integer.sub.i16x32",
        "II_VSUB_16",
    ),
    _DescriptorSpec(
        "VSUB_32",
        f"{_TARGET_KEY}.sub.i32x16",
        "integer.sub.i32x16",
        "II_VSUB_32",
    ),
    *(
        _DescriptorSpec(
            f"V{operation.upper()}_{comparison}_{width}_vaddSign{sign_bit}",
            f"{_TARGET_KEY}.{operation}.{signedness}.i{width}x{512 // width}",
            f"integer.{operation}.{signedness}.i{width}x{512 // width}",
            f"II_V{operation.upper()}_{comparison}_{width}_vaddSign{sign_bit}",
            implicit_outputs=("cmp",),
            asm_mnemonic=(
                f"{operation}.{'s' if signedness == 'signed' else 'u'}"
                f"{width}x{512 // width}"
            ),
        )
        for width in (8, 16, 32)
        for operation, comparison in (("min", "GE"), ("max", "LT"))
        for signedness, sign_bit in (("signed", 1), ("unsigned", 0))
    ),
    _DescriptorSpec(
        "VMUL_vmul_cm_core_X_X",
        f"{_TARGET_KEY}.multiply.i16x32.configured",
        "integer.multiply.i16x32.configured",
        "II_VMUL_vmul_cm_core_X_X",
        asm_mnemonic="vmul.i16x32",
    ),
    _DescriptorSpec(
        "VMUL_f_vmul_bf_vmul_bf_core_X_X",
        f"{_TARGET_KEY}.multiply.bf16x32.configured",
        "floating.multiply.bf16x32.configured",
        "II_VMUL_f_vmul_bf_vmul_bf_core_X_X",
        storage_overrides=(("dst", "mBMs"),),
        asm_mnemonic="vmul.bf16x32",
    ),
    _DescriptorSpec(
        "VMAC_f_vmac_bf_vmul_bf_core_X_X",
        f"{_TARGET_KEY}.accumulate.bf16x32.configured",
        "floating.accumulate.bf16x32.configured",
        "II_VMAC_f_vmac_bf_vmul_bf_core_X_X",
        storage_overrides=(("dst", "mBMs"), ("acc1", "mBMs")),
        asm_mnemonic="vmac.bf16x32",
    ),
    _DescriptorSpec(
        "VEXTBCST_128_vec_extract_broadcast_imm",
        f"{_TARGET_KEY}.broadcast.bf16x8.to.bf16x32",
        "floating.broadcast.bf16x8.to.bf16x32",
        "II_VEXTBCST_128_vec_extract_broadcast_imm",
        storage_overrides=(("s1", "eWL"),),
        asm_mnemonic="vbroadcast.bf16x8.to.bf16x32",
        operand_register_parts=(("s1", _EWL_LOW128_PART),),
        encoding_adapter_overrides=(("s1", "LOOM_eWL_OP_mXm"),),
    ),
    _DescriptorSpec(
        "VSHUFFLE_vec_shuffle_x",
        f"{_TARGET_KEY}.shuffle.x.configured",
        "register.shuffle.x.configured",
        "II_VSHUFFLE_vec_shuffle_x",
        storage_overrides=(("dst", "VEC256"),),
        asm_mnemonic="vshuffle",
    ),
    _DescriptorSpec(
        "VSHIFT",
        f"{_TARGET_KEY}.shift.bytes.x.configured",
        "register.shift.bytes.x.configured",
        "II_VSHIFT",
        asm_mnemonic="vshift",
    ),
    _DescriptorSpec(
        "VMOV_alu_mv_mv_x",
        f"{_TARGET_KEY}.move.vector512",
        "register.move.vector512",
        "II_VMOV_alu_mv_mv_x",
        storage_overrides=(("dst", "VEC256"), ("src", "VEC256")),
        asm_mnemonic="vmov.512",
        encoding_adapter_overrides=(
            ("dst", "LOOM_mXm_OP_mMvBMXDst"),
            ("src", "LOOM_mXm_OP_mMvBMXSrc"),
        ),
    ),
    _DescriptorSpec(
        "VMOV_alu_mv_mv_w",
        f"{_TARGET_KEY}.move.vec256",
        "register.move.vec256",
        "II_VMOV_alu_mv_mv_w",
        storage_overrides=(("dst", "VEC256"), ("src", "VEC256")),
        asm_mnemonic="vmov.256",
        allocation_move=True,
    ),
    _DescriptorSpec(
        "VMOV_alu_mv_mv_x",
        f"{_TARGET_KEY}.move.vector512.to.accumulator512",
        "register.move.vector512.to.accumulator512",
        "II_VMOV_alu_mv_mv_x",
        storage_overrides=(("dst", "mBMs"), ("src", "VEC256")),
        asm_mnemonic="vmov.vector512.to.accumulator512",
        encoding_adapter_overrides=(
            ("dst", "LOOM_mBMs_OP_mMvBMXDst"),
            ("src", "LOOM_mXm_OP_mMvBMXSrc"),
        ),
    ),
    _DescriptorSpec(
        "VMOV_alu_mv_mv_x",
        f"{_TARGET_KEY}.move.accumulator512.to.vector512",
        "register.move.accumulator512.to.vector512",
        "II_VMOV_alu_mv_mv_x",
        storage_overrides=(("dst", "VEC256"), ("src", "mBMs")),
        asm_mnemonic="vmov.accumulator512.to.vector512",
        encoding_adapter_overrides=(
            ("dst", "LOOM_mXm_OP_mMvBMXDst"),
            ("src", "LOOM_mBMs_OP_mMvBMXSrc"),
        ),
    ),
    _DescriptorSpec(
        "VMOV_alu_mv_mv_x",
        f"{_TARGET_KEY}.move.accumulator512",
        "register.move.accumulator512",
        "II_VMOV_alu_mv_mv_x",
        storage_overrides=(("dst", "mBMs"), ("src", "mBMs")),
        asm_mnemonic="vmov.accumulator512",
        encoding_adapter_overrides=(
            ("dst", "LOOM_mBMs_OP_mMvBMXDst"),
            ("src", "LOOM_mBMs_OP_mMvBMXSrc"),
        ),
        allocation_move=True,
    ),
    _DescriptorSpec(
        "VMUL_f_vmul_bf_vmul_bf_core_Y_Y",
        f"{_TARGET_KEY}.matrix.multiply.bf16bf16.m8n8k1.configured",
        "matrix.multiply.bf16bf16.m8n8k1.configured",
        "II_VMUL_f_vmul_bf_vmul_bf_core_Y_Y",
        storage_overrides=(("dst", "mBMs"), ("s1", "VEC256"), ("s2", "VEC256")),
        asm_mnemonic="mmul.bf16bf16.m8n8k1",
    ),
    _DescriptorSpec(
        "VMAC_f_vmac_bf_vmul_bf_core_Y_Y",
        f"{_TARGET_KEY}.matrix.accumulate.bf16bf16.m8n8k1.configured",
        "matrix.accumulate.bf16bf16.m8n8k1.configured",
        "II_VMAC_f_vmac_bf_vmul_bf_core_Y_Y",
        storage_overrides=(
            ("dst", "mBMs"),
            ("acc1", "mBMs"),
            ("s1", "VEC256"),
            ("s2", "VEC256"),
        ),
        asm_mnemonic="mma.bf16bf16.m8n8k1",
    ),
    _DescriptorSpec(
        "VADD_f_vmac_cm2_add_reg",
        f"{_TARGET_KEY}.add.f32x64.configured",
        "floating.add.f32x64.configured",
        "II_VADD_f_vmac_cm2_add_reg",
        storage_overrides=(
            ("dst", "mBMs"),
            ("acc1", "mBMs"),
            ("acc2", "mBMs"),
        ),
        asm_mnemonic="vadd.f32x64",
    ),
    _DescriptorSpec(
        "VSUB_f_vmac_cm2_add_reg",
        f"{_TARGET_KEY}.sub.f32x64.configured",
        "floating.sub.f32x64.configured",
        "II_VSUB_f_vmac_cm2_add_reg",
        storage_overrides=(
            ("dst", "mBMs"),
            ("acc1", "mBMs"),
            ("acc2", "mBMs"),
        ),
        asm_mnemonic="vsub.f32x64",
    ),
    _DescriptorSpec(
        "VCONV_bf16_fp32_mv_w_srs_bf",
        f"{_TARGET_KEY}.convert.f32x16.to.bf16x16",
        "floating.convert.f32x16.to.bf16x16",
        "II_VCONV_bf16_fp32_mv_w_srs_bf",
        storage_overrides=(("dst", "VEC256"), ("src", "mBMs")),
        asm_mnemonic="vconv.bf16.fp32x16",
    ),
    _DescriptorSpec(
        "VCONV_bf16_fp32_mv_x_srs_bf",
        f"{_TARGET_KEY}.convert.f32x32.to.bf16x32",
        "floating.convert.f32x32.to.bf16x32",
        "II_VCONV_bf16_fp32_mv_x_srs_bf",
        storage_overrides=(("src", "mBMs"),),
        asm_mnemonic="vconv.bf16.fp32",
    ),
    _DescriptorSpec(
        "VCONV_fp32_bf16_mv_ups_xbf",
        f"{_TARGET_KEY}.convert.bf16x32.to.f32x32",
        "floating.convert.bf16x32.to.f32x32",
        "II_VCONV_fp32_bf16_mv_ups_xbf",
        storage_overrides=(("dst", "mBMs"), ("src", "VEC256")),
        asm_mnemonic="vconv.fp32.bf16",
    ),
    _DescriptorSpec(
        "VCONV_fp32_bf16_mv_ups_wbf",
        f"{_TARGET_KEY}.convert.bf16x16.to.f32x16",
        "floating.convert.bf16x16.to.f32x16",
        "II_VCONV_fp32_bf16_mv_ups_wbf",
        storage_overrides=(("dst", "mBMs"), ("src", "VEC256")),
        asm_mnemonic="vconv.fp32.bf16x16",
    ),
    _DescriptorSpec(
        "VEXP2",
        f"{_TARGET_KEY}.exp2.f32x16.to.bf16x16",
        "floating.exp2.f32x16.to.bf16x16",
        "II_VEXP2",
        storage_overrides=(("dst", "VEC256"), ("src", "mBMs")),
        asm_mnemonic="vexp2.bf16x16",
    ),
    _DescriptorSpec(
        "VTANH",
        f"{_TARGET_KEY}.tanh.f32x16.to.bf16x16",
        "floating.tanh.f32x16.to.bf16x16",
        "II_VTANH",
        storage_overrides=(("dst", "VEC256"), ("src", "mBMs")),
        asm_mnemonic="vtanh.bf16x16",
    ),
    _DescriptorSpec(
        "VCLR",
        f"{_TARGET_KEY}.accumulator.clear.i32x64",
        "matrix.accumulator.clear.i32x64",
        "II_VCLR",
        storage_overrides=(("dst", "mBMs"),),
        asm_mnemonic="acc.clear.i32x64",
    ),
    _DescriptorSpec(
        "VCLR",
        f"{_TARGET_KEY}.accumulator.clear.f32x64",
        "matrix.accumulator.clear.f32x64",
        "II_VCLR",
        storage_overrides=(("dst", "mBMs"),),
        asm_mnemonic="acc.clear.f32x64",
    ),
    *_integer_matrix_descriptor_specs(),
    *_packed_dot_descriptor_specs(),
    *_packed_i4_unpack_descriptor_specs(),
    _DescriptorSpec(
        "VLDA_dmx_lda_bm_idx",
        f"{_TARGET_KEY}.load.accumulator.f32x16.indexed.register",
        "memory.load.accumulator.indexed.f32x16",
        "II_VLDA_dmx_lda_bm_idx",
        storage_overrides=(("dst", "mBMs"),),
        asm_mnemonic="vlda.acc.f32x16.index",
        memory_width_bits=512,
    ),
    _DescriptorSpec(
        "VLDA_dmx_lda_bm_idx_imm",
        f"{_TARGET_KEY}.load.accumulator.f32x16.indexed.immediate",
        "memory.load.accumulator.indexed.f32x16",
        "II_VLDA_dmx_lda_bm_idx_imm",
        storage_overrides=(("dst", "mBMs"),),
        asm_mnemonic="vlda.acc.f32x16",
        memory_width_bits=512,
    ),
    _DescriptorSpec(
        "VST_dmx_sts_bm_idx",
        f"{_TARGET_KEY}.store.accumulator.f32x16.indexed.register",
        "memory.store.accumulator.indexed.f32x16",
        "II_VST_dmx_sts_bm_idx",
        asm_mnemonic="vst.acc.f32x16.index",
        memory_width_bits=512,
    ),
    _DescriptorSpec(
        "VST_dmx_sts_bm_idx_imm",
        f"{_TARGET_KEY}.store.accumulator.f32x16.indexed.immediate",
        "memory.store.accumulator.indexed.f32x16",
        "II_VST_dmx_sts_bm_idx_imm",
        asm_mnemonic="vst.acc.f32x16",
        memory_width_bits=512,
    ),
    _DescriptorSpec(
        "VST_dmx_sts_bm_idx_imm",
        f"{_TARGET_KEY}.store.accumulator.i32x16.indexed.immediate",
        "memory.store.accumulator.indexed.i32x16",
        "II_VST_dmx_sts_bm_idx_imm",
        asm_mnemonic="vst.acc.i32x16",
        memory_width_bits=512,
    ),
    _DescriptorSpec(
        "VSRS_4x_mv_x_srs_dm_srsSign1",
        f"{_TARGET_KEY}.narrow.trunc.signed.i16x32",
        "integer.narrow.trunc.signed.i16x32",
        "II_VSRS_4x_mv_x_srs_dm_srsSign1",
        asm_mnemonic="vsrs.trunc.s16x32",
    ),
    _DescriptorSpec(
        "VBAND",
        f"{_TARGET_KEY}.and.bits512",
        "integer.and.bits512",
        "II_VBAND",
    ),
    _DescriptorSpec(
        "VBOR",
        f"{_TARGET_KEY}.or.bits512",
        "integer.or.bits512",
        "II_VBOR",
    ),
    _DescriptorSpec(
        "VBCST_8",
        f"{_TARGET_KEY}.splat.i8x64",
        "integer.splat.i8x64",
        "II_VBCST_8",
    ),
    _DescriptorSpec(
        "VBCST_16",
        f"{_TARGET_KEY}.splat.i16x32",
        "integer.splat.i16x32",
        "II_VBCST_16",
    ),
    _DescriptorSpec(
        "VBCST_32",
        f"{_TARGET_KEY}.splat.i32x16",
        "integer.splat.i32x16",
        "II_VBCST_32",
    ),
    _DescriptorSpec(
        "VBCST_64",
        f"{_TARGET_KEY}.splat.i64x8",
        "integer.splat.i64x8",
        "II_VBCST_64",
    ),
    _DescriptorSpec(
        "VEQZ_8",
        f"{_TARGET_KEY}.cmp.eqz.i8x64",
        "integer.cmp.eq.i8x64",
        "II_VEQZ_8",
        storage_overrides=(("cmp", "eLPredicate"),),
    ),
    *(
        _DescriptorSpec(
            f"VEQZ_{width}",
            f"{_TARGET_KEY}.cmp.eqz.i{width}x{512 // width}.el.low32",
            f"integer.cmp.eq.i{width}x{512 // width}.low32",
            f"II_VEQZ_{width}",
            storage_overrides=(("cmp", "eLPredicate"),),
            asm_mnemonic=f"veqz.{width}.el.low32",
            operand_register_parts=(("cmp", _EL_LOW32_PART),),
            encoding_adapter_overrides=(("cmp", "LOOM_eL_low32"),),
        )
        for width in (16, 32)
    ),
    *(
        _DescriptorSpec(
            f"V{relation.upper()}_{width}_vaddSign{sign_bit}",
            (
                f"{_TARGET_KEY}.cmp.{relation}.{signedness}."
                f"i{width}x{512 // width}"
                f"{'.el.low32' if width != 8 else ''}"
            ),
            (
                f"integer.cmp.{relation}.{signedness}."
                f"i{width}x{512 // width}"
                f"{'.low32' if width != 8 else ''}"
            ),
            f"II_V{relation.upper()}_{width}_vaddSign{sign_bit}",
            storage_overrides=(("cmp", "eLPredicate"),),
            asm_mnemonic=(
                f"v{relation}.{'s' if signedness == 'signed' else 'u'}"
                f"{width}x{512 // width}"
                f"{'.el.low32' if width != 8 else ''}"
            ),
            operand_register_parts=((("cmp", _EL_LOW32_PART),) if width != 8 else ()),
            encoding_adapter_overrides=(
                (("cmp", "LOOM_eL_low32"),) if width != 8 else ()
            ),
        )
        for width in (8, 16, 32)
        for relation in ("lt", "ge")
        for signedness, sign_bit in (("signed", 1), ("unsigned", 0))
    ),
    _DescriptorSpec(
        "VSEL_8",
        f"{_TARGET_KEY}.select.i8x64",
        "integer.select.i8x64",
        "II_VSEL_8",
        storage_overrides=(("sel", "eLPredicate"),),
    ),
    _DescriptorSpec(
        "VSEL_32",
        f"{_TARGET_KEY}.select.i32x16",
        "integer.select.i32x16",
        "II_VSEL_32",
    ),
    *(
        _DescriptorSpec(
            f"VSEL_{width}",
            f"{_TARGET_KEY}.select.i{width}x{512 // width}.mask64",
            f"integer.select.i{width}x{512 // width}.mask64",
            f"II_VSEL_{width}",
            storage_overrides=(("sel", "eLPredicate"),),
            asm_mnemonic=f"vsel.{width}.mask64",
            operand_register_parts=(("sel", _EL_LOW32_PART),),
            encoding_adapter_overrides=(("sel", "LOOM_eL_low32"),),
        )
        for width in (16, 32)
    ),
    *(
        _DescriptorSpec(
            operation.upper(),
            f"{_TARGET_KEY}.predicate.{operation}.low32",
            f"integer.predicate.{operation}.low32",
            f"II_{operation.upper()}",
            storage_overrides=(
                ("d0", "eLPredicate"),
                ("s0", "eLPredicate"),
                ("s1", "eLPredicate"),
            ),
            asm_mnemonic=f"predicate.{operation}.low32",
            operand_register_parts=(
                ("d0", _EL_LOW32_PART),
                ("s0", _EL_LOW32_PART),
                ("s1", _EL_LOW32_PART),
            ),
            encoding_adapter_overrides=(
                ("d0", "LOOM_eL_low32"),
                ("s0", "LOOM_eL_low32"),
                ("s1", "LOOM_eL_low32"),
            ),
        )
        for operation in ("and", "or", "xor")
    ),
    *(
        _DescriptorSpec(
            operation.upper(),
            f"{_TARGET_KEY}.predicate.{operation}.high32",
            f"integer.predicate.{operation}.high32",
            f"II_{operation.upper()}",
            storage_overrides=(
                ("d0", "eLPredicate"),
                ("s0", "eLPredicate"),
                ("s1", "eLPredicate"),
            ),
            asm_mnemonic=f"predicate.{operation}.high32",
            operand_register_parts=(
                ("d0", _EL_HIGH32_PART),
                ("s0", _EL_HIGH32_PART),
                ("s1", _EL_HIGH32_PART),
            ),
            encoding_adapter_overrides=(
                ("d0", "LOOM_eL_high32"),
                ("s0", "LOOM_eL_high32"),
                ("s1", "LOOM_eL_high32"),
            ),
            storage_continuation_part=_EL_LOW32_PART,
        )
        for operation in ("and", "or", "xor")
    ),
    _DescriptorSpec(
        "MOVA",
        f"{_TARGET_KEY}.predicate.complete.zero.high32",
        "integer.predicate.complete.zero.high32",
        "II_MOVA_eR",
        storage_overrides=(("dst", "eLPredicate"),),
        asm_mnemonic="predicate.complete.zero.high32",
        operand_register_parts=(("dst", _EL_HIGH32_PART),),
        encoding_adapter_overrides=(("dst", "LOOM_eL_high32_OP_mLdaCg"),),
        storage_continuation_part=_EL_LOW32_PART,
    ),
    _DescriptorSpec(
        "VEXTBCST_8_vec_extract_broadcast_imm",
        f"{_TARGET_KEY}.broadcast.i8x64.from-vector",
        "integer.broadcast.i8x64.from-vector",
        "II_VEXTBCST_8_vec_extract_broadcast_imm",
    ),
    _DescriptorSpec(
        "VEXTBCST_16_vec_extract_broadcast_imm",
        f"{_TARGET_KEY}.broadcast.i16x32.from-vector",
        "integer.broadcast.i16x32.from-vector",
        "II_VEXTBCST_16_vec_extract_broadcast_imm",
    ),
    _DescriptorSpec(
        "VEXTBCST_32_vec_extract_broadcast_imm",
        f"{_TARGET_KEY}.broadcast.i32x16.from-vector",
        "integer.broadcast.i32x16.from-vector",
        "II_VEXTBCST_32_vec_extract_broadcast_imm",
    ),
    _DescriptorSpec(
        "VEXTRACT_8_vec_extract_imm_vaddSign0",
        f"{_TARGET_KEY}.extract.i8.immediate",
        "integer.extract.i8",
        "II_VEXTRACT_8_vec_extract_imm_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_8_vec_extract_r_vaddSign0",
        f"{_TARGET_KEY}.extract.i8.register",
        "integer.extract.i8",
        "II_VEXTRACT_8_vec_extract_r_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_16_vec_extract_imm_vaddSign0",
        f"{_TARGET_KEY}.extract.i16.immediate",
        "integer.extract.i16",
        "II_VEXTRACT_16_vec_extract_imm_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_16_vec_extract_r_vaddSign0",
        f"{_TARGET_KEY}.extract.i16.register",
        "integer.extract.i16",
        "II_VEXTRACT_16_vec_extract_r_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_32_vec_extract_imm_vaddSign0",
        f"{_TARGET_KEY}.extract.i32.immediate",
        "integer.extract.i32",
        "II_VEXTRACT_32_vec_extract_imm_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_32_vec_extract_r_vaddSign0",
        f"{_TARGET_KEY}.extract.i32.register",
        "integer.extract.i32",
        "II_VEXTRACT_32_vec_extract_r_vaddSign0",
    ),
    _DescriptorSpec(
        "VEXTRACT_64_vec_extract_imm_vaddSign1",
        f"{_TARGET_KEY}.extract.predicate64.immediate",
        "integer.extract.predicate64",
        "II_VEXTRACT_64_vec_extract_imm_vaddSign1",
        storage_overrides=(("dst", "eLPredicate"),),
        asm_mnemonic="vextract.predicate64",
    ),
    _DescriptorSpec(
        "VEXTRACT_64_vec_extract_imm_vaddSign1",
        f"{_TARGET_KEY}.extract.i64.immediate",
        "integer.extract.i64",
        "II_VEXTRACT_64_vec_extract_imm_vaddSign1",
        asm_mnemonic="vextract.64.imm",
    ),
    _DescriptorSpec(
        "VEXTRACT_64_vec_extract_r_vaddSign1",
        f"{_TARGET_KEY}.extract.i64.register",
        "integer.extract.i64",
        "II_VEXTRACT_64_vec_extract_r_vaddSign1",
        asm_mnemonic="vextract.64.reg",
    ),
    _DescriptorSpec(
        "VINSERT_8_mIdxImm0",
        f"{_TARGET_KEY}.insert.i8.zero",
        "integer.insert.i8",
        "II_VINSERT_8_mIdxImm0",
    ),
    _DescriptorSpec(
        "VINSERT_8_mR29_insert",
        f"{_TARGET_KEY}.insert.i8.register",
        "integer.insert.i8",
        "II_VINSERT_8_mR29_insert",
    ),
    _DescriptorSpec(
        "VINSERT_16_mIdxImm0",
        f"{_TARGET_KEY}.insert.i16.zero",
        "integer.insert.i16",
        "II_VINSERT_16_mIdxImm0",
    ),
    _DescriptorSpec(
        "VINSERT_16_mIdxImm0",
        f"{_TARGET_KEY}.insert.bf16x8.zero",
        "integer.insert.bf16x8",
        "II_VINSERT_16_mIdxImm0",
        storage_overrides=(("dst", "eWL"), ("s1", "eWL")),
        asm_mnemonic="vinsert.16.ewl.zero",
        encoding_adapter_overrides=(
            ("dst", "LOOM_eWL_OP_mXm"),
            ("s1", "LOOM_eWL_OP_mXm"),
        ),
    ),
    _DescriptorSpec(
        "VINSERT_16_mR29_insert",
        f"{_TARGET_KEY}.insert.i16.register",
        "integer.insert.i16",
        "II_VINSERT_16_mR29_insert",
    ),
    _DescriptorSpec(
        "VINSERT_16_mR29_insert",
        f"{_TARGET_KEY}.insert.bf16x8.register",
        "integer.insert.bf16x8",
        "II_VINSERT_16_mR29_insert",
        storage_overrides=(("dst", "eWL"), ("s1", "eWL")),
        asm_mnemonic="vinsert.16.ewl.reg",
        encoding_adapter_overrides=(
            ("dst", "LOOM_eWL_OP_mXm"),
            ("s1", "LOOM_eWL_OP_mXm"),
        ),
    ),
    _DescriptorSpec(
        "VINSERT_32_mIdxImm0",
        f"{_TARGET_KEY}.insert.i32.zero",
        "integer.insert.i32",
        "II_VINSERT_32_mIdxImm0",
    ),
    _DescriptorSpec(
        "VINSERT_32_mR29_insert",
        f"{_TARGET_KEY}.insert.i32.register",
        "integer.insert.i32",
        "II_VINSERT_32_mR29_insert",
    ),
    _DescriptorSpec(
        "VINSERT_64_mIdxImm0",
        f"{_TARGET_KEY}.insert.i64.zero",
        "integer.insert.i64",
        "II_VINSERT_64_mIdxImm0",
        asm_mnemonic="vinsert.64.zero",
    ),
    _DescriptorSpec(
        "VINSERT_64_mR29_insert",
        f"{_TARGET_KEY}.insert.i64.register",
        "integer.insert.i64",
        "II_VINSERT_64_mR29_insert",
        asm_mnemonic="vinsert.64.reg",
    ),
    *_vector_memory_descriptor_specs(),
    _DescriptorSpec(
        "MOVA",
        f"{_TARGET_KEY}.constant.i32.mova",
        "integer.const.i32",
        "II_MOVA_eR",
        (("dst", "eR"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mova.i32",
    ),
    _DescriptorSpec(
        "MOVA",
        f"{_TARGET_KEY}.constant.i32.select",
        "integer.const.i32",
        "II_MOVA_eR",
        (("dst", "eRS16"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mov.select",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.constant.i32.short",
        "integer.const.i32",
        "II_MOV_alu_mv_mv_mv_cg_eR",
        (("dst", "eR"),),
        DescriptorOpKind.CONST,
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.constant.i32.shift",
        "integer.const.i32",
        "II_MOV_alu_mv_mv_mv_cg_eS",
        (("dst", "eS"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mov.shift",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.constant.i32.fx2flt-scale",
        "conversion.scale.signed.i32.to.f32",
        "II_MOV_alu_mv_mv_mv_cg_eS",
        (("dst", "mS2"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mov.fx2flt-scale",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.constant.i32.flt2fx-scale",
        "conversion.scale.round-nearest.f32.to.signed.i32",
        "II_MOV_alu_mv_mv_mv_cg_eS",
        (("dst", "mS3"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mov.flt2fx-scale",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.rounding.immediate",
        "state.write.rounding",
        "II_MOV_alu_mv_mv_mv_cg_mCRRnd",
        (("dst", "mCRRnd"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.rounding",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.srs-mode.immediate",
        "state.write.srs-mode",
        "II_MOV_alu_mv_mv_mv_cg_mCRSRSMode",
        (("dst", "mCRSRSMode"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.srs-mode",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.saturation.immediate",
        "state.write.saturation",
        "II_MOV_alu_mv_mv_mv_cg_mCRSat",
        (("dst", "mCRSat"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.saturation",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.unpack-size.immediate",
        "state.write.unpack-size",
        "II_MOV_alu_mv_mv_mv_cg_mCRUnpackSize",
        (("dst", "mCRUnpackSize"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.unpack-size",
    ),
    _DescriptorSpec(
        "MOVXM",
        f"{_TARGET_KEY}.constant.i32",
        "integer.const.i32",
        "II_MOVXM_eR",
        (("dst", "eR"),),
        DescriptorOpKind.CONST,
    ),
    _DescriptorSpec(
        "MOVXM",
        f"{_TARGET_KEY}.materialize.static-byte-offset.i32",
        "integer.materialize.static-byte-offset.i32",
        "II_MOVXM_eR",
        (("dst", "eR"),),
        asm_mnemonic="mov.static-byte-offset",
    ),
    _DescriptorSpec(
        "MOVXM",
        f"{_TARGET_KEY}.materialize.local-address.i32",
        "memory.materialize.local-address.i32",
        "II_MOVXM_eP",
        (("dst", "eP"),),
        asm_mnemonic="mov.local-address",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.move.scalar",
        "register.move.scalar",
        "II_MOV_alu_mv_mv_mv_scl_eR_eR",
        (("dst", "eR"), ("src", "eR")),
        allocation_move=True,
    ),
    _DescriptorSpec(
        "MOVS",
        f"{_TARGET_KEY}.move.local-address",
        "register.move.local-address",
        "II_MOVS_eP_eP",
        (("dst", "eP"), ("src", "eP")),
        allocation_move=True,
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.move.local-address-to-scalar",
        "register.move.local-address-to-scalar",
        "II_MOV_alu_mv_mv_mv_scl_eR_eP",
        (("dst", "eR"), ("src", "eP")),
        asm_mnemonic="mov.address-to-scalar",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.move.scalar-to-local-address",
        "register.move.scalar-to-local-address",
        "II_MOV_alu_mv_mv_mv_scl_eP_eR",
        (("dst", "eP"), ("src", "eR")),
        asm_mnemonic="mov.scalar-to-address",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.move.to.division-state",
        "register.move.to.division-state",
        "II_MOV_alu_mv_mv_mv_scl",
        (("dst", "mR31_divs"), ("src", "eR")),
        asm_mnemonic="mov.dividend",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.move.from.division-state",
        "register.move.from.division-state",
        "II_MOV_alu_mv_mv_mv_scl",
        (("dst", "eR"), ("src", "mR31_divs")),
        asm_mnemonic="mov.quotient",
    ),
    _DescriptorSpec(
        "ADD_alu_r_rr",
        f"{_TARGET_KEY}.add.i32",
        "integer.add.i32",
        "II_ADD_alu_r_rr",
    ),
    _DescriptorSpec(
        "ADC",
        f"{_TARGET_KEY}.add.carry.i32",
        "integer.add.carry.i32",
        "II_ADC",
    ),
    _DescriptorSpec("SUB", f"{_TARGET_KEY}.sub.i32", "integer.sub.i32", "II_SUB"),
    _DescriptorSpec(
        "SBC",
        f"{_TARGET_KEY}.sub.borrow.i32",
        "integer.sub.borrow.i32",
        "II_SBC",
    ),
    _DescriptorSpec("MUL", f"{_TARGET_KEY}.mul.i32", "integer.mul.i32", "II_MUL"),
    _DescriptorSpec("AND", f"{_TARGET_KEY}.and.i32", "integer.and.i32", "II_AND"),
    _DescriptorSpec("OR", f"{_TARGET_KEY}.or.i32", "integer.or.i32", "II_OR"),
    _DescriptorSpec("XOR", f"{_TARGET_KEY}.xor.i32", "integer.xor.i32", "II_XOR"),
    _DescriptorSpec("ASHL", f"{_TARGET_KEY}.ashl.i32", "integer.ashl.i32", "II_ASHL"),
    _DescriptorSpec("LSHL", f"{_TARGET_KEY}.lshl.i32", "integer.lshl.i32", "II_LSHL"),
    _DescriptorSpec("ABS", f"{_TARGET_KEY}.abs.i32", "integer.abs.i32", "II_ABS"),
    _DescriptorSpec("CLZ", f"{_TARGET_KEY}.clz.i32", "integer.clz.i32", "II_CLZ"),
    _DescriptorSpec(
        "POPCOUNT",
        f"{_TARGET_KEY}.popcount.i32",
        "integer.popcount.i32",
        "II_POPCOUNT",
    ),
    _DescriptorSpec("MAC", f"{_TARGET_KEY}.madd.i32", "integer.madd.i32", "II_MAC"),
    _DescriptorSpec(
        "DIVS",
        f"{_TARGET_KEY}.divide.step.unsigned.i32",
        "integer.divide.step.unsigned.i32",
        "II_DIVS",
    ),
    _DescriptorSpec(
        "SEL_EQZ",
        f"{_TARGET_KEY}.select.zero.i32",
        "integer.select.zero.i32",
        "II_SEL_EQZ",
    ),
    _DescriptorSpec(
        "SEL_NEZ",
        f"{_TARGET_KEY}.select.nonzero.i32",
        "integer.select.nonzero.i32",
        "II_SEL_NEZ",
    ),
    _DescriptorSpec(
        "EXTEND_s8",
        f"{_TARGET_KEY}.extend.signed.i8",
        "integer.extend.signed.i8",
        "II_EXTEND_s8",
    ),
    _DescriptorSpec(
        "EXTEND_s16",
        f"{_TARGET_KEY}.extend.signed.i16",
        "integer.extend.signed.i16",
        "II_EXTEND_s16",
    ),
    _DescriptorSpec(
        "EXTEND_u8",
        f"{_TARGET_KEY}.extend.unsigned.i8",
        "integer.extend.unsigned.i8",
        "II_EXTEND_u8",
    ),
    _DescriptorSpec(
        "EXTEND_u16",
        f"{_TARGET_KEY}.extend.unsigned.i16",
        "integer.extend.unsigned.i16",
        "II_EXTEND_u16",
    ),
    _DescriptorSpec("EQ", f"{_TARGET_KEY}.cmp.eq.i32", "integer.cmp.eq.i32", "II_EQ"),
    _DescriptorSpec("NE", f"{_TARGET_KEY}.cmp.ne.i32", "integer.cmp.ne.i32", "II_NE"),
    _DescriptorSpec(
        "EQZ",
        f"{_TARGET_KEY}.cmp.eqz.i32",
        "integer.cmp.eq.i32",
        "II_EQZ",
    ),
    _DescriptorSpec(
        "NEZ",
        f"{_TARGET_KEY}.cmp.nez.i32",
        "integer.cmp.ne.i32",
        "II_NEZ",
    ),
    _DescriptorSpec(
        "LT",
        f"{_TARGET_KEY}.cmp.slt.i32",
        "integer.cmp.slt.i32",
        "II_LT",
    ),
    _DescriptorSpec(
        "GE",
        f"{_TARGET_KEY}.cmp.sge.i32",
        "integer.cmp.sge.i32",
        "II_GE",
    ),
    _DescriptorSpec(
        "LTU",
        f"{_TARGET_KEY}.cmp.ult.i32",
        "integer.cmp.ult.i32",
        "II_LTU",
    ),
    _DescriptorSpec(
        "LT",
        f"{_TARGET_KEY}.cmp.slt.i32.select",
        "integer.cmp.slt.i32.select",
        "II_LT",
        (("d0", "mR27_select"),),
        asm_mnemonic="lt.select",
    ),
    _DescriptorSpec(
        "LTU",
        f"{_TARGET_KEY}.cmp.ult.i32.select",
        "integer.cmp.ult.i32.select",
        "II_LTU",
        (("d0", "mR27_select"),),
        asm_mnemonic="ltu.select",
    ),
    _DescriptorSpec(
        "GEU",
        f"{_TARGET_KEY}.cmp.uge.i32",
        "integer.cmp.uge.i32",
        "II_GEU",
    ),
)

_ASM_MNEMONIC_BY_FORM = {
    "MOV_alu_mv_mv_mv_cg": "mov.short",
    "MOV_alu_mv_mv_mv_scl": "mov.scalar",
    "MOVXM": "mov.i32",
    "ADD_alu_r_rr": "add.rr",
    "VEXTRACT_8_vec_extract_imm_vaddSign0": "vextract.8.imm",
    "VEXTRACT_8_vec_extract_r_vaddSign0": "vextract.8.reg",
    "VEXTRACT_16_vec_extract_imm_vaddSign0": "vextract.16.imm",
    "VEXTRACT_16_vec_extract_r_vaddSign0": "vextract.16.reg",
    "VEXTRACT_32_vec_extract_imm_vaddSign0": "vextract.32.imm",
    "VEXTRACT_32_vec_extract_r_vaddSign0": "vextract.32.reg",
    "VINSERT_8_mIdxImm0": "vinsert.8.zero",
    "VINSERT_8_mR29_insert": "vinsert.8.reg",
    "VINSERT_16_mIdxImm0": "vinsert.16.zero",
    "VINSERT_16_mR29_insert": "vinsert.16.reg",
    "VINSERT_32_mIdxImm0": "vinsert.32.zero",
    "VINSERT_32_mR29_insert": "vinsert.32.reg",
}

_MACHINE_FORMS = {form.name: form for form in CORE_MACHINE_TABLE.forms}


def _with_ordered_memory_variants(
    specifications: tuple[_DescriptorSpec, ...],
) -> tuple[_DescriptorSpec, ...]:
    """Adds semantic aliases for independently observable memory accesses."""

    result: list[_DescriptorSpec] = []
    for spec in specifications:
        if spec.ordered_memory:
            raise ValueError(f"{spec.key}: base descriptor is already ordered")
        result.append(spec)
        form = _MACHINE_FORMS[spec.form_name]
        if not (has_property(form, "mayLoad") or has_property(form, "mayStore")):
            continue
        result.append(
            replace(
                spec,
                key=f"{spec.key}.volatile",
                semantic_tag=f"{spec.semantic_tag}.volatile",
                schedule_alternatives=tuple(
                    f"{key}.volatile" for key in spec.schedule_alternatives
                ),
                ordered_memory=True,
            )
        )
    return tuple(result)


_DESCRIPTOR_SPECS = _with_ordered_memory_variants(_BASE_DESCRIPTOR_SPECS)
