# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P semantic descriptor specifications derived from owned machine tables."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amd.xdna.aie.machine import (
    has_property,
)
from loom.target.arch.amd.xdna.aie2p.core_address_descriptors import (
    DIMENSION_REGISTER_PARTS,
    _address_descriptor_specs,
    _dimension_update,
)
from loom.target.arch.amd.xdna.aie2p.core_comparison_descriptors import (
    BF16_COMPARISON_DESCRIPTOR_SPECS,
    INTEGER_COMPARISON_DESCRIPTOR_SPECS,
    INTEGER_EXTREMA_DESCRIPTOR_SPECS,
    PREDICATE_DESCRIPTOR_SPECS,
    PREDICATE_REGISTER_PARTS,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.arch.amd.xdna.aie2p.core_fifo_descriptors import (
    FIFO_REGISTER_PARTS,
    _fifo_load_descriptor_specs,
    _fifo_storage_descriptor_specs,
    _fifo_store_descriptor_specs,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_stream_descriptors import (
    _cascade_descriptor_specs,
    _cascade_matrix_descriptor_specs,
    _scalar_stream_descriptor_specs,
)
from loom.target.low_descriptors import (
    DescriptorOpKind,
    Effect,
    EffectKind,
    MemorySpace,
    RegisterPart,
)

_TARGET_KEY = "amd.xdna.aie2p"
# AIEBaseInstrInfo's conservative lock model at LLVM_AIE_SCHEDULE_SOURCE_COMMIT
# in core_schedule_data.py. AIE2P inherits these core stall/resume cycles and
# uses the itinerary memory cycles for lock ordering; these are not calibration
# measurements or additional resource occupancy from the itinerary tables.
_LOCK_CORE_STALL_CYCLE = 2
_LOCK_CORE_RESUME_CYCLE = 8
_LOCK_EFFECT = Effect(
    EffectKind.BARRIER,
    MemorySpace.WORKGROUP,
    producer_event=f"{_TARGET_KEY}.lock.resume.c{_LOCK_CORE_RESUME_CYCLE}",
    consumer_event=f"{_TARGET_KEY}.lock.stall.c{_LOCK_CORE_STALL_CYCLE}",
)
_VEC256_LOW128_PART = "aie2p.vec256.low128"
_VEC256_HIGH128_PART = "aie2p.vec256.high128"
_REGISTER_PARTS = (
    *PREDICATE_REGISTER_PARTS,
    RegisterPart(_VEC256_LOW128_PART, "aie2p.vec256", 0x1),
    RegisterPart(_VEC256_HIGH128_PART, "aie2p.vec256", 0x2),
    *FIFO_REGISTER_PARTS,
    *DIMENSION_REGISTER_PARTS,
)
_REGISTER_PARTS_BY_NAME = {part.name: part for part in _REGISTER_PARTS}


# Each lane names its linear address stem, dimension template, and adapter.
_VECTOR_MEMORY_FORM_FAMILIES = (
    (
        128,
        ("VLDA_128_dmv_lda_w", "VLDA_{dimension}D_128", "OP_mWa"),
        ("VLDB_128", "VLDB_{dimension}D_128", "OP_mWb"),
        ("VST_128_dmv_sts_w", "VST_{dimension}D_128", "OP_mWs"),
    ),
    (
        256,
        ("VLDA_dmw_lda_w", "VLDA_{dimension}D_dmw_lda_w", "OP_mWa"),
        ("VLDB_dmw_ldb", "VLDB_{dimension}D_dmw_ldb", "OP_mWb"),
        ("VST_dmw_sts_w", "VST_{dimension}D_dmw_sts_w", "OP_mWs"),
    ),
    (
        512,
        ("VLDA_dmx_lda_x", "VLDA_{dimension}D_dmx_lda_x", None),
        ("VLDB_dmx_ldb_x", "VLDB_{dimension}D_dmx_ldb_x", None),
        ("VST_dmx_sts_x", "VST_{dimension}D_dmx_sts_x", None),
    ),
)

_MEMORY_ADDRESS_FORMS = (
    ("indexed.immediate", "idx_imm", "", 0),
    ("indexed.register", "idx", ".index", 0),
    ("postincrement.immediate", "pstm_nrm_imm", ".post", 0),
    ("postincrement.register", "pstm_nrm", ".post.modifier", 0),
    ("2d", None, ".2d", 2),
    ("3d", None, ".3d", 3),
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
    # Native loads define fresh W storage without preserving a destination
    # input. Stores consume only the low 128 bits of their source register.
    parts = ((operand_name, _VEC256_LOW128_PART),) if operand_name == "src" else ()
    return (
        (),
        parts,
        ((operand_name, native_adapter),),
    )


def _vector_memory_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native widths and address updates for both load lanes and stores."""

    result = []
    for width_bits, load_a, load_b, store in _VECTOR_MEMORY_FORM_FAMILIES:
        for element_type, element_bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES:
            shape = f"{element_type}x{width_bits // element_bits}"
            for family, mnemonic, forms in (
                ("load.a", "vlda", load_a),
                ("load.b", "vldb", load_b),
                ("store", "vst", store),
            ):
                operation = "store" if family == "store" else "load"
                overrides = _vector_memory_operand_overrides(
                    width_bits,
                    "src" if family == "store" else "dst",
                    forms[2],
                )
                for (
                    addressing,
                    native_suffix,
                    asm_suffix,
                    dimension,
                ) in _MEMORY_ADDRESS_FORMS:
                    form = (
                        forms[1].format(dimension=dimension)
                        if dimension
                        else f"{forms[0]}_{native_suffix}"
                    )
                    spec = _DescriptorSpec(
                        form,
                        f"{_TARGET_KEY}.{family}.{shape}.{addressing}",
                        f"memory.{operation}.{addressing.split('.')[0]}.{shape}",
                        f"II_{form}",
                        storage_overrides=overrides[0],
                        asm_mnemonic=f"{mnemonic}.{width_bits}.{shape}{asm_suffix}",
                        operand_register_parts=overrides[1],
                        encoding_adapter_overrides=overrides[2],
                        schedule_alternatives=(
                            (f"{_TARGET_KEY}.load.b.{shape}.{addressing}",)
                            if family == "load.a"
                            else ()
                        ),
                        memory_width_bits=width_bits,
                    )
                    result.append(
                        _dimension_update(spec, dimension) if dimension else spec
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
            "dot.integer.i8x64.configured",
            "II_VMUL_vmul_cm_core_Y_X",
            storage_overrides=(("dst", "mBMs"), ("s1", "VEC256")),
            asm_mnemonic="dot4i.i8x64",
        ),
    )


def _dense_integer_matrix_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Exposes full integer matrix operands with native scalar mode controls."""

    return tuple(
        _DescriptorSpec(
            f"{native}_vmul_cm_core_{left}_{right}",
            f"{_TARGET_KEY}.matrix.{operation}.integer.{shape}.configured",
            f"matrix.{operation}.integer.{shape}.configured",
            f"II_{native}_vmul_cm_core_{left}_{right}",
            storage_overrides=(
                ("dst", "mBMs"),
                *accumulator_storage,
                ("s1", "VEC256"),
                ("s2", "VEC256"),
            ),
            asm_mnemonic=f"{mnemonic}.integer.{shape}",
        )
        for operation, native, mnemonic, accumulator_storage in (
            ("multiply", "VMUL", "mmul", ()),
            ("negative-multiply", "VNEGMUL", "mnegmul", ()),
            ("accumulate", "VMAC", "mma", (("acc1", "mBMs"),)),
            ("subtract-product", "VMSC", "mms", (("acc1", "mBMs"),)),
            (
                "add-accumulate",
                "VADDMAC_vmac_cm2_add_reg",
                "maddmac",
                (("acc1", "mBMs"), ("acc2", "mBMs")),
            ),
            (
                "add-subtract-product",
                "VADDMSC_vmac_cm2_add_reg",
                "maddmsc",
                (("acc1", "mBMs"), ("acc2", "mBMs")),
            ),
        )
        for left in ("X", "Y")
        for right in ("X", "Y")
        for shape in (f"{left.lower()}-{right.lower()}",)
    )


def _dense_floating_matrix_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native floating matrix variants with explicit datapath controls."""

    result = []
    for kind, native_kind, left, right, storage in (
        ("bf16", "bf", "X", "X", "VEC256"),
        ("bf16", "bf", "Y", "Y", "VEC256"),
        ("bfp", "bfp", "EX", "EX", "mEXa"),
        ("bfp", "bfp", "EX", "EY", "mEXa"),
    ):
        shape = f"{left.lower()}-{right.lower()}"
        for operation, opcode, unit, mnemonic, accumulator_count in (
            ("multiply", "VMUL", "vmul", "mmul", 0),
            ("negative-multiply", "VNEGMUL", "vmul", "mnegmul", 0),
            ("accumulate", "VMAC", "vmac", "mma", 1),
            ("subtract-product", "VMSC", "vmac", "mms", 1),
            ("add-accumulate", "VADDMAC", "vaddmac", "maddmac", 2),
            ("add-subtract-product", "VADDMSC", "vaddmac", "maddmsc", 2),
        ):
            # Shaped BF16 and BFP aliases own the other multiply/accumulate forms.
            if operation in ("multiply", "accumulate") and right != "EY":
                continue
            native = f"{opcode}_f_{unit}_{native_kind}"
            if accumulator_count == 2:
                native += "_vmac_cm2_add_reg"
            native += f"_vmul_{native_kind}_core_{left}_{right}"
            result.append(
                _DescriptorSpec(
                    native,
                    f"{_TARGET_KEY}.matrix.{operation}.{kind}.{shape}.configured",
                    f"matrix.{operation}.{kind}.{shape}.configured",
                    f"II_{native}",
                    storage_overrides=(
                        ("dst", "mBMs"),
                        *(
                            (f"acc{index + 1}", "mBMs")
                            for index in range(accumulator_count)
                        ),
                        ("s1", storage),
                        ("s2", storage),
                    ),
                    asm_mnemonic=f"{mnemonic}.{kind}.{shape}",
                )
            )
    return tuple(result)


def _packed_i4_unpack_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native signed and unsigned 4-to-8-bit unpack forms."""

    return tuple(
        _DescriptorSpec(
            f"VUNPACK_mv_unpack_{carrier}_unpackSign{sign_bit}",
            (
                f"{_TARGET_KEY}.unpack.{source_kind}4x{result_lane_count}.to."
                f"{source_kind}8x{result_lane_count}.configured"
            ),
            (
                f"convert.integer.unpack.{source_kind}4x{result_lane_count}.to."
                f"{source_kind}8x{result_lane_count}.configured"
            ),
            f"II_VUNPACK_mv_unpack_{carrier}_unpackSign{sign_bit}",
            asm_mnemonic=(
                f"vunpack.{source_kind}4.to.{source_kind}8x{result_lane_count}"
            ),
        )
        for source_lane_count, carrier in ((32, "w"), (64, "x"))
        for result_lane_count in (source_lane_count * 2,)
        for source_kind, sign_bit in (("u", 0), ("s", 1))
    )


def _integer_conversion_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects exact native integer widening and configured pack forms."""

    widen_specs = tuple(
        _DescriptorSpec(
            f"{form_name}_upsSign{sign_bit}",
            f"{_TARGET_KEY}.widen.{shape}.{signedness}.configured",
            f"convert.integer.widen.{shape}.{signedness}.configured",
            f"II_{form_name}_upsSign{sign_bit}",
            storage_overrides=storage_overrides,
            asm_mnemonic=f"vups.{shape}.{signedness}",
        )
        for shape, form_name, storage_overrides in (
            (
                "2x.w-to-b",
                "VUPS_2x_mv_ups_w2b",
                (("dst", "mBMs"), ("src", "VEC256")),
            ),
            (
                "4x.w-to-c",
                "VUPS_4x_mv_ups_w2c",
                (("dst", "mBMs"), ("src", "VEC256")),
            ),
            (
                "2x.x-to-c",
                "VUPS_2x_mv_ups_x2c",
                (("dst", "mBMs"),),
            ),
            (
                "4x.x-to-d",
                "VUPS_4x_mv_ups_x2d",
                (("dst", "mBMs"),),
            ),
        )
        for signedness, sign_bit in (("unsigned", 0), ("signed", 1))
    )
    pack_specs = tuple(
        _DescriptorSpec(
            f"VPACK_mv_pack_{width}_packSign{sign_bit}",
            f"{_TARGET_KEY}.pack.{width}.{signedness}.configured",
            f"convert.integer.pack.{width}.{signedness}.configured",
            f"II_VPACK_mv_pack_{width}_packSign{sign_bit}",
            storage_overrides=storage_overrides,
            asm_mnemonic=f"vpack.{width}.{signedness}",
        )
        for width, storage_overrides in (
            ("w", ()),
            ("x", (("src", "VEC256"),)),
        )
        for signedness, sign_bit in (("trunc", 0), ("signed", 1))
    )
    return (*widen_specs, *pack_specs)


def _fused_memory_descriptor_family(
    *,
    prefix: str,
    stem: str,
    suffix: str = "",
    key: str,
    tag: str,
    mnemonic: str,
    width: int,
    storage: tuple[tuple[str, str], ...] = (),
) -> tuple[_DescriptorSpec, ...]:
    """Preserves packet conversion while selecting native address recurrence."""

    result = []
    for addressing, native_suffix, asm_suffix, dimension in _MEMORY_ADDRESS_FORMS:
        form = (
            f"{prefix}_{dimension}D_{stem}{suffix}"
            if dimension
            else f"{prefix}_{stem}_{native_suffix}{suffix}"
        )
        spec = _DescriptorSpec(
            form,
            f"{_TARGET_KEY}.{key}.{addressing}",
            tag,
            f"II_{form}",
            storage_overrides=storage,
            asm_mnemonic=f"{mnemonic}{asm_suffix}",
            memory_width_bits=width,
        )
        result.append(_dimension_update(spec, dimension) if dimension else spec)
    return tuple(result)


def _integer_narrowing_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native SRS conversion to registers or exact-width memory."""

    result = []
    for factor, accumulator, vector, width, memory_stem in (
        (2, "b", "w", 256, "SRS_2x_dmw_sts_srs_bm"),
        (2, "c", "x", 512, "SRS_2x_dm_sts_srs_cm"),
        (4, "c", "w", 256, "SRS_4x_dm_sts_srs_cm"),
        (4, "d", "x", 512, "SRS_4x_dmx_sts_srs_dm"),
    ):
        shape = f"{factor}x.{accumulator}-to-{vector}"
        for signedness, sign_bit in (("unsigned", 0), ("signed", 1)):
            form = f"VSRS_{factor}x_mv_{vector}_srs_{accumulator}m_srsSign{sign_bit}"
            key = f"narrow.{shape}.{signedness}.configured"
            tag = f"convert.integer.{key}"
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.{key}",
                    tag,
                    f"II_{form}",
                    storage_overrides=(("src", "mBMs"),),
                    asm_mnemonic=f"vsrs.{shape}.{signedness}",
                )
            )
            result.extend(
                _fused_memory_descriptor_family(
                    prefix="VST",
                    stem=memory_stem,
                    suffix=f"_srsSign{sign_bit}",
                    key=f"store.{key}",
                    tag=f"{tag}.memory.store",
                    mnemonic=f"vst.srs.{shape}.{signedness}",
                    width=width,
                    storage=(("src", "mBMs"),),
                )
            )
    return tuple(result)


def _fused_vector_memory_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects vector memory forms with native packet conversion."""

    load_unpack_specs = tuple(
        spec
        for source_lane_count, carrier in ((32, "w"), (64, "x"))
        for result_lane_count in (source_lane_count * 2,)
        for source_kind, sign_bit in (("u", 0), ("s", 1))
        for spec in _fused_memory_descriptor_family(
            prefix="VLDB",
            stem=f"UNPACK_dm{carrier}_ldb_unpack",
            suffix=f"_unpackSign{sign_bit}",
            key=(
                f"load.unpack.{source_kind}4x{result_lane_count}."
                f"to.{source_kind}8x{result_lane_count}.configured"
            ),
            tag=(
                f"convert.integer.unpack.{source_kind}4x{result_lane_count}."
                f"to.{source_kind}8x{result_lane_count}.configured.memory.load"
            ),
            mnemonic=(
                f"vldb.unpack.{source_kind}4.to.{source_kind}8x{result_lane_count}"
            ),
            width=source_lane_count * 8,
        )
    )
    load_convert_specs = tuple(
        spec
        for source_shape, result_shape, memory_width_bits, form_stem in (
            ("bf16x16", "f32x16", 256, "CONV_fp32_bf16_dmw_lda_ups_bf"),
            ("bf16x32", "f32x32", 512, "CONV_fp32_bf16_dmx_lda_ups_bf"),
        )
        for spec in _fused_memory_descriptor_family(
            prefix="VLDA",
            stem=form_stem,
            key=f"load.convert.{source_shape}.to.{result_shape}",
            tag=f"convert.floating.{source_shape}.to.{result_shape}.memory.load",
            storage=(("op", "mBMs"),),
            mnemonic=f"vlda.convert.{source_shape}.to.{result_shape}",
            width=memory_width_bits,
        )
    )
    load_widen_specs = tuple(
        spec
        for shape, memory_width_bits, form_stem in (
            ("2x.w-to-b", 256, "UPS_2x_dmw_lda_ups_w2b"),
            ("4x.w-to-c", 256, "UPS_4x_dmw_lda_ups_w2c"),
            ("2x.x-to-c", 512, "UPS_2x_dmx_lda_ups_x2c"),
            ("4x.x-to-d", 512, "UPS_4x_dmx_lda_ups_x2d"),
        )
        for signedness, sign_bit in (("unsigned", 0), ("signed", 1))
        for spec in _fused_memory_descriptor_family(
            prefix="VLDA",
            stem=form_stem,
            suffix=f"_upsSign{sign_bit}",
            key=f"load.widen.{shape}.{signedness}.configured",
            tag=f"convert.integer.widen.{shape}.{signedness}.configured.memory.load",
            storage=(("dst", "mBMs"),),
            mnemonic=f"vlda.ups.{shape}.{signedness}",
            width=memory_width_bits,
        )
    )
    store_convert_specs = tuple(
        spec
        for source_shape, result_shape, memory_width_bits, form_stem in (
            ("f32x16", "bf16x16", 256, "CONV_bf16_fp32_dmw_sts_srs_bf"),
            ("f32x32", "bf16x32", 512, "CONV_bf16_fp32_dmx_sts_srs_bf"),
        )
        for spec in _fused_memory_descriptor_family(
            prefix="VST",
            stem=form_stem,
            key=f"store.convert.{source_shape}.to.{result_shape}",
            tag=f"convert.floating.{source_shape}.to.{result_shape}.memory.store",
            storage=(("src", "mBMs"),),
            mnemonic=f"vst.convert.{source_shape}.to.{result_shape}",
            width=memory_width_bits,
        )
    )
    store_pack_specs = tuple(
        spec
        for width, memory_width_bits, form_stem, storage in (
            ("w", 256, "PACK_dmw_sts_pack", ()),
            ("x", 512, "PACK_dmx_sts_pack", (("src", "VEC256"),)),
        )
        for signedness, sign_bit in (("trunc", 0), ("signed", 1))
        for spec in _fused_memory_descriptor_family(
            prefix="VST",
            stem=form_stem,
            suffix=f"_packSign{sign_bit}",
            key=f"store.pack.{width}.{signedness}.configured",
            tag=f"convert.integer.pack.{width}.{signedness}.configured.memory.store",
            storage=storage,
            mnemonic=f"vst.pack.{width}.{signedness}",
            width=memory_width_bits,
        )
    )
    return (
        *load_unpack_specs,
        *load_convert_specs,
        *load_widen_specs,
        *store_convert_specs,
        *store_pack_specs,
    )


def _accumulator_memory_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects raw 512-bit accumulator transfers and address updates."""

    result = []
    for operation, prefix, memory_stem, mnemonic in (
        ("load", "VLDA", "dmx_lda_bm", "vlda"),
        ("store", "VST", "dmx_sts_bm", "vst"),
    ):
        for addressing, native_suffix, asm_suffix, dimension in _MEMORY_ADDRESS_FORMS:
            form = (
                f"{prefix}_{dimension}D_{memory_stem}"
                if dimension
                else f"{prefix}_{memory_stem}_{native_suffix}"
            )
            spec = _DescriptorSpec(
                form,
                f"{_TARGET_KEY}.{operation}.accumulator.{addressing}",
                f"memory.{operation}.accumulator.{addressing.split('.')[0]}",
                f"II_{form}",
                storage_overrides=((("dst", "mBMs"),) if operation == "load" else ()),
                asm_mnemonic=f"{mnemonic}.acc{asm_suffix}",
                memory_width_bits=512,
            )
            result.append(_dimension_update(spec, dimension) if dimension else spec)
    return tuple(result)


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


def _scalar_nonlinear_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects native nonlinear math and fused fixed-point conversions."""

    return tuple(
        _DescriptorSpec(
            f"{stem}_{form}",
            f"{_TARGET_KEY}.{operation}.{conversion}",
            f"floating.{operation}.{conversion}",
            f"II_{stem}_{form}",
            asm_mnemonic=f"{mnemonic}.{conversion}",
        )
        for stem, operation, mnemonic in (
            ("INV", "reciprocal", "inv"),
            ("INVSQRT", "reciprocal-sqrt", "invsqrt"),
            ("SQRT", "sqrt", "sqrt"),
        )
        for conversion, form in (
            ("f32", "mRx"),
            ("fx2flt", "mRx_mOptConv"),
            ("flt2fx", "mOptConvDel_mRx"),
            ("fx2flt.flt2fx", "mOptConvDel_mRx_mOptConv"),
        )
    )


_DENSE_MATRIX_DESCRIPTOR_SPECS = (
    *_dense_integer_matrix_descriptor_specs(),
    *_dense_floating_matrix_descriptor_specs(),
)


_BASE_DESCRIPTOR_SPECS = (
    *_address_descriptor_specs(),
    _DescriptorSpec(
        "ADD_add_r_ri",
        f"{_TARGET_KEY}.add.i32.immediate",
        "integer.add.i32",
        "II_ADD_add_r_ri",
    ),
    _DescriptorSpec(
        "ADD_add_r_ri",
        f"{_TARGET_KEY}.add.carry_out.i32.immediate",
        "integer.add.carry_out.i32",
        "II_ADD_add_r_ri",
        asm_mnemonic="add.carry.immediate",
        expose_carry=True,
    ),
    _DescriptorSpec(
        "MOV_alu_mv_alu_fx2flt",
        f"{_TARGET_KEY}.convert.signed.i32.to.f32",
        "convert.signed.i32.to.f32",
        "II_MOV_alu_mv_alu_fx2flt",
        asm_mnemonic="convert.signed.i32.to.f32",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_alu_flt2fx",
        f"{_TARGET_KEY}.convert.round-nearest.f32.to.signed.i32",
        "convert.round-nearest.f32.to.signed.i32",
        "II_MOV_alu_mv_alu_flt2fx",
        asm_mnemonic="convert.round-nearest.f32.to.signed.i32",
    ),
    *_scalar_nonlinear_descriptor_specs(),
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
        rematerializable=True,
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
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "ACQ_mLockId_reg",
        f"{_TARGET_KEY}.lock.acquire.register",
        "synchronization.lock.acquire",
        "II_ACQ_mLockId_reg",
        asm_mnemonic="acq.reg",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "ACQ_COND_mLockId_imm",
        f"{_TARGET_KEY}.lock.acquire.conditional.immediate",
        "synchronization.lock.acquire.conditional",
        "II_ACQ_COND_mLockId_imm",
        asm_mnemonic="acq.cond",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "ACQ_COND_mLockId_reg",
        f"{_TARGET_KEY}.lock.acquire.conditional.register",
        "synchronization.lock.acquire.conditional",
        "II_ACQ_COND_mLockId_reg",
        asm_mnemonic="acq.cond.reg",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "REL_mLockId_imm",
        f"{_TARGET_KEY}.lock.release.immediate",
        "synchronization.lock.release",
        "II_REL_mLockId_imm",
        asm_mnemonic="rel",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "REL_mLockId_reg",
        f"{_TARGET_KEY}.lock.release.register",
        "synchronization.lock.release",
        "II_REL_mLockId_reg",
        asm_mnemonic="rel.reg",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "REL_COND_mLockId_imm",
        f"{_TARGET_KEY}.lock.release.conditional.immediate",
        "synchronization.lock.release.conditional",
        "II_REL_COND_mLockId_imm",
        asm_mnemonic="rel.cond",
        effects=(_LOCK_EFFECT,),
    ),
    _DescriptorSpec(
        "REL_COND_mLockId_reg",
        f"{_TARGET_KEY}.lock.release.conditional.register",
        "synchronization.lock.release.conditional",
        "II_REL_COND_mLockId_reg",
        asm_mnemonic="rel.cond.reg",
        effects=(_LOCK_EFFECT,),
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
    *INTEGER_EXTREMA_DESCRIPTOR_SPECS,
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
        asm_mnemonic="vbroadcast.bf16x8.to.bf16x32",
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
        "VMOV_alu_mv_mv_ex",
        f"{_TARGET_KEY}.move.bfp576",
        "register.move.bfp576",
        "II_VMOV_alu_mv_mv_ex",
        storage_overrides=(("dst", "mEXa"), ("src", "mEXa")),
        asm_mnemonic="vmov.bfp576",
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
        "VNEG_f",
        f"{_TARGET_KEY}.neg.f32x64.configured",
        "floating.neg.f32x64.configured",
        "II_VNEG_f",
        storage_overrides=(("dst", "mBMs"), ("acc1", "mBMs")),
        asm_mnemonic="vneg.f32x64",
    ),
    _DescriptorSpec(
        "VFLOOR_s32_bf16_mv_float_to_int_w",
        f"{_TARGET_KEY}.convert.floor.bf16x16.to.i32x16",
        "convert.floor.bf16x16.to.i32x16",
        "II_VFLOOR_s32_bf16_mv_float_to_int_w",
        storage_overrides=(("src", "VEC256"),),
        asm_mnemonic="vfloor.s32.bf16",
    ),
    _DescriptorSpec(
        "VCONV_bf16_fp32_mv_w_srs_bf",
        f"{_TARGET_KEY}.convert.f32x16.to.bf16x16",
        "convert.floating.f32x16.to.bf16x16",
        "II_VCONV_bf16_fp32_mv_w_srs_bf",
        storage_overrides=(("dst", "VEC256"), ("src", "mBMs")),
        asm_mnemonic="vconv.bf16.fp32x16",
    ),
    _DescriptorSpec(
        "VCONV_bf16_fp32_mv_x_srs_bf",
        f"{_TARGET_KEY}.convert.f32x32.to.bf16x32",
        "convert.floating.f32x32.to.bf16x32",
        "II_VCONV_bf16_fp32_mv_x_srs_bf",
        storage_overrides=(("src", "mBMs"),),
        asm_mnemonic="vconv.bf16.fp32",
    ),
    _DescriptorSpec(
        "VCONV_fp32_bf16_mv_ups_xbf",
        f"{_TARGET_KEY}.convert.bf16x32.to.f32x32",
        "convert.floating.bf16x32.to.f32x32",
        "II_VCONV_fp32_bf16_mv_ups_xbf",
        storage_overrides=(("dst", "mBMs"), ("src", "VEC256")),
        asm_mnemonic="vconv.fp32.bf16",
    ),
    _DescriptorSpec(
        "VCONV_fp32_bf16_mv_ups_wbf",
        f"{_TARGET_KEY}.convert.bf16x16.to.f32x16",
        "convert.floating.bf16x16.to.f32x16",
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
    *_DENSE_MATRIX_DESCRIPTOR_SPECS,
    *(
        _DescriptorSpec(
            native,
            f"{_TARGET_KEY}.accumulator.{operation}.integer.configured",
            f"integer.accumulator.{operation}.configured",
            f"II_{native}",
            storage_overrides=tuple(
                (name, "mBMs")
                for name in ("dst", "acc1", *(("acc2",) if operation != "neg" else ()))
            ),
            asm_mnemonic=f"v{operation}.acc.integer",
        )
        for operation, native in (
            ("add", "VADD_vmac_cm2_add_reg"),
            ("sub", "VSUB_vmac_cm2_add_reg"),
            ("neg", "VNEG"),
        )
    ),
    _DescriptorSpec(
        "VCONV_bfp16ebs8_fp32",
        f"{_TARGET_KEY}.convert.f32x64.bfp16ebs8",
        "convert.floating.f32x64.bfp16ebs8",
        "II_VCONV_bfp16ebs8_fp32",
        storage_overrides=(("src", "mBMs"),),
        asm_mnemonic="vconv.bfp16ebs8.fp32",
    ),
    _DescriptorSpec(
        "VCONV_bfp16ebs16_fp32",
        f"{_TARGET_KEY}.convert.f32x64.bfp16ebs16",
        "convert.floating.f32x64.bfp16ebs16",
        "II_VCONV_bfp16ebs16_fp32",
        storage_overrides=(("src", "mBMs"),),
        asm_mnemonic="vconv.bfp16ebs16.fp32",
    ),
    _DescriptorSpec(
        "VCONV_bfp16ebs16_ebs8",
        f"{_TARGET_KEY}.convert.bfp16ebs8.bfp16ebs16",
        "convert.floating.bfp16ebs8.bfp16ebs16",
        "II_VCONV_bfp16ebs16_ebs8",
        storage_overrides=(("src", "mEXa"),),
        asm_mnemonic="vconv.bfp16ebs16.ebs8",
    ),
    _DescriptorSpec(
        "VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX",
        f"{_TARGET_KEY}.matrix.multiply.bfp16ebs8.m8n8k8",
        "matrix.multiply.bfp16ebs8.m8n8k8",
        "II_VMUL_f_vmul_bfp_vmul_bfp_core_EX_EX",
        storage_overrides=(("dst", "mBMs"), ("s1", "mEXa"), ("s2", "mEXa")),
        asm_mnemonic="mmul.bfp16ebs8.m8n8k8",
    ),
    _DescriptorSpec(
        "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX",
        f"{_TARGET_KEY}.matrix.accumulate.bfp16ebs8.m8n8k8",
        "matrix.accumulate.bfp16ebs8.m8n8k8",
        "II_VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX",
        storage_overrides=(
            ("dst", "mBMs"),
            ("acc1", "mBMs"),
            ("s1", "mEXa"),
            ("s2", "mEXa"),
        ),
        asm_mnemonic="mma.bfp16ebs8.m8n8k8",
    ),
    *_packed_dot_descriptor_specs(),
    *_packed_i4_unpack_descriptor_specs(),
    *_integer_conversion_descriptor_specs(),
    *_integer_narrowing_descriptor_specs(),
    *_fused_vector_memory_descriptor_specs(),
    *_accumulator_memory_descriptor_specs(),
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
    *PREDICATE_DESCRIPTOR_SPECS,
    *BF16_COMPARISON_DESCRIPTOR_SPECS,
    *INTEGER_COMPARISON_DESCRIPTOR_SPECS,
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
        "VINSERT_16_mR29_insert",
        f"{_TARGET_KEY}.insert.i16.register",
        "integer.insert.i16",
        "II_VINSERT_16_mR29_insert",
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
    # Each gather selects four pointers from one 256-bit vector and returns
    # four 64-bit windows. The mode controls subword selection, not result
    # width. The itinerary accounts for all load interfaces used by slot B.
    *(
        _DescriptorSpec(
            f"VLDB_4x{width}_{half}",
            f"{_TARGET_KEY}.load.b.lookup.4x{width}.{half}",
            f"memory.load.lookup.4x{width}.{half}",
            f"II_VLDB_4x{width}_{half}",
            asm_mnemonic=f"vldb.4x{width}.{half}",
            operand_register_parts=(
                ("src", _VEC256_LOW128_PART if half == "lo" else _VEC256_HIGH128_PART),
            ),
            encoding_adapter_overrides=(("src", "OP_mWs"),),
            memory_width_bits=256,
        )
        for width in (16, 32, 64)
        for half in ("lo", "hi")
    ),
    *_fifo_load_descriptor_specs(AIE2P_VECTOR_MEMORY_ELEMENT_TYPES),
    *_fifo_store_descriptor_specs(AIE2P_VECTOR_MEMORY_ELEMENT_TYPES),
    *_fifo_storage_descriptor_specs(),
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
        f"{_TARGET_KEY}.constant.i32.fifo-position",
        "integer.const.i32",
        "II_MOVA_eR",
        (("dst", "eRF2"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mova.fifo.position",
    ),
    _DescriptorSpec(
        "MOVA",
        f"{_TARGET_KEY}.constant.i32.store-fifo-position",
        "integer.const.i32",
        "II_MOVA_eR",
        (("dst", "mR26_fifo_st"),),
        DescriptorOpKind.CONST,
        asm_mnemonic="mova.fifo.store.position",
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
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.ups-mode.immediate",
        "state.write.ups-mode",
        "II_MOV_alu_mv_mv_mv_cg_mCRUPSMode",
        (("dst", "mCRUPSMode"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.ups-mode",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.state.pack-size.immediate",
        "state.write.pack-size",
        "II_MOV_alu_mv_mv_mv_cg_mCRPackSize",
        (("dst", "mCRPackSize"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.pack-size",
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
        rematerializable=True,
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
        "ADD_alu_r_rr",
        f"{_TARGET_KEY}.add.carry_out.i32",
        "integer.add.carry_out.i32",
        "II_ADD_alu_r_rr",
        asm_mnemonic="add.carry",
        expose_carry=True,
    ),
    _DescriptorSpec(
        "ADC",
        f"{_TARGET_KEY}.add.carry.i32",
        "integer.add.carry_in_out.i32",
        "II_ADC",
        expose_carry=True,
    ),
    _DescriptorSpec("SUB", f"{_TARGET_KEY}.sub.i32", "integer.sub.i32", "II_SUB"),
    _DescriptorSpec(
        "SUB",
        f"{_TARGET_KEY}.sub.borrow_out.i32",
        "integer.sub.borrow_out.i32",
        "II_SUB",
        asm_mnemonic="sub.borrow",
        expose_carry=True,
    ),
    _DescriptorSpec(
        "SBC",
        f"{_TARGET_KEY}.sub.borrow.i32",
        "integer.sub.borrow_in_out.i32",
        "II_SBC",
        expose_carry=True,
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


_DESCRIPTOR_SPECS = (
    *_with_ordered_memory_variants(_BASE_DESCRIPTOR_SPECS),
    *_scalar_stream_descriptor_specs(),
    *_cascade_descriptor_specs(),
    *_cascade_matrix_descriptor_specs(_DENSE_MATRIX_DESCRIPTOR_SPECS),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        f"{_TARGET_KEY}.read.core.id",
        "read.core.id",
        # The oracle has no CORE_ID-specific itinerary. Keep its scalar-move
        # resource model instead of substituting a general-register read.
        "II_MOV_alu_mv_mv_mv_scl",
        storage_overrides=(("dst", "eR"), ("src", "mCoreID")),
        implicit_inputs=("src",),
        asm_mnemonic="mov.coreid",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cntr2l",
        f"{_TARGET_KEY}.read.tile.cycles",
        "read.tile.cycles",
        "II_MOV_alu_mv_mv_mv_cntr2l",
        asm_mnemonic="mov.cycles",
        # Each observation is distinct and ordered with memory/protocol
        # effects. This does not impose a hardware completion fence.
        effects=(Effect(EffectKind.BARRIER),),
    ),
)
