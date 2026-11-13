# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P FIFO storage, transfer widths, and architectural state updates."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amd.xdna.aie2p.core_address_descriptors import _dimension_update
from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import RegisterPart

_TARGET_KEY = "amd.xdna.aie2p"
_FIFO_STORAGE_CLASSES = (
    ("eLdFifoReg", "load", ("eLdFifoLReg", "eLdFifoHReg")),
    ("mStFifo", "store", ("mStFifol", "mStFifoh")),
)
FIFO_REGISTER_PARTS = tuple(
    RegisterPart(
        f"aie2p.{register_class.lower()}.{half}512",
        f"aie2p.{register_class.lower()}",
        1 << ordinal,
    )
    for register_class, _, _ in _FIFO_STORAGE_CLASSES
    for ordinal, half in enumerate(("low", "high"))
)


def _fifo_address_variants(
    spec: _DescriptorSpec, operation: str
) -> tuple[_DescriptorSpec, ...]:
    """Preserves the FIFO tuple while updating its owning dimension state."""

    result = [spec]
    for suffix, native_suffix, mnemonic_suffix, dimension in (
        (".postincrement.register", f"fifo_1d_{operation}", ".post.modifier", 0),
        (".2d", "2D", ".2d", 2),
        (".3d", "3D", ".3d", 3),
    ):
        form = spec.form_name.removesuffix(f"normal_{operation}") + native_suffix
        variant = replace(
            spec,
            form_name=form,
            key=spec.key + suffix,
            semantic_tag=spec.semantic_tag + suffix,
            itinerary=f"II_{form}",
            asm_mnemonic=spec.asm_mnemonic + mnemonic_suffix,
            schedule_alternatives=tuple(
                key + suffix for key in spec.schedule_alternatives
            ),
        )
        result.append(_dimension_update(variant, dimension) if dimension else variant)
    return tuple(result)


def _fifo_load_descriptor_specs(
    element_types: tuple[tuple[str, int], ...],
) -> tuple[_DescriptorSpec, ...]:
    """Selects vector streaming loads with complete FIFO state."""

    fill_keys = {
        lane: f"{_TARGET_KEY}.load.{lane}.fifo.fill.512" for lane in ("a", "b")
    }
    result = [
        _DescriptorSpec(
            f"VLD{lane.upper()}_FILL_512",
            fill_keys[lane],
            "memory.load.fifo.fill.512",
            f"II_VLD{lane.upper()}_FILL_512",
            asm_mnemonic=f"vld{lane}.fill.512",
            schedule_alternatives=(fill_keys["b"],) if lane == "a" else (),
            memory_width_bits=512,
        )
        for lane in ("a", "b")
    ]
    # Pops fetch 512 bits; packed payloads consume additional buffered bytes.
    for shape, width_bits in (
        *((f"{element}x{512 // bits}", 512) for element, bits in element_types),
        ("bfp16ebs16", 544),
        ("bfp16ebs8", 576),
    ):
        mnemonic_shape = f"512.{shape}" if width_bits == 512 else shape
        pop_keys = {
            lane: f"{_TARGET_KEY}.load.{lane}.{shape}.fifo.pop" for lane in ("a", "b")
        }
        for lane in ("a", "b"):
            spec = _DescriptorSpec(
                f"VLD{lane.upper()}_POP_{width_bits}_normal_pop",
                pop_keys[lane],
                f"memory.load.fifo.pop.{shape}",
                f"II_VLD{lane.upper()}_POP_{width_bits}_normal_pop",
                storage_overrides=(("dst", "mEXa"),) if width_bits != 512 else (),
                asm_mnemonic=f"vld{lane}.pop.{mnemonic_shape}",
                schedule_alternatives=(pop_keys["b"],) if lane == "a" else (),
                memory_width_bits=512,
            )
            result.extend(_fifo_address_variants(spec, "pop"))
            if shape == "bfp16ebs8":
                # At position 64 the pop consumes only the initialized low
                # FIFO half and its next 64-byte fetch. Later pops carry the
                # complete state produced here. Other positions require the
                # ordinary full-state descriptor.
                result.append(
                    replace(
                        spec,
                        key=spec.key + ".buffered64",
                        semantic_tag=spec.semantic_tag + ".buffered64",
                        asm_mnemonic=spec.asm_mnemonic + ".buffered64",
                        operand_register_parts=(
                            ("fifo_reg", "aie2p.eldfiforeg.low512"),
                        ),
                        schedule_alternatives=tuple(
                            key + ".buffered64" for key in spec.schedule_alternatives
                        ),
                    )
                )
    return tuple(result)


def _fifo_store_descriptor_specs(
    element_types: tuple[tuple[str, int], ...],
) -> tuple[_DescriptorSpec, ...]:
    """Selects state-preserving streaming stores and their final partial flush."""

    flushes = (
        _DescriptorSpec(
            "VST_FLUSH_512_normal_flush",
            f"{_TARGET_KEY}.store.fifo.flush.512",
            "memory.store.fifo.flush.512",
            "II_VST_FLUSH_512_normal_flush",
            asm_mnemonic="vst.flush.512",
            memory_width_bits=512,
        ),
        _DescriptorSpec(
            "VST_FLUSH_512_CONV_normal_flush",
            f"{_TARGET_KEY}.store.fifo.flush.convert.512",
            "memory.store.fifo.flush.convert.512",
            "II_VST_FLUSH_512_CONV_normal_flush",
            asm_mnemonic="vst.flush.convert.512",
            memory_width_bits=512,
        ),
    )
    result = [
        variant
        for flush in flushes
        for variant in _fifo_address_variants(flush, "flush")
    ]
    for element_type, element_bits in element_types:
        shape = f"{element_type}x{512 // element_bits}"
        result.append(
            _DescriptorSpec(
                "VST_PUSH_512",
                f"{_TARGET_KEY}.store.{shape}.fifo.push",
                f"memory.store.fifo.push.{shape}",
                "II_VST_PUSH_512",
                asm_mnemonic=f"vst.push.512.{shape}",
                memory_width_bits=512,
            )
        )
    # Packed pushes write 512 bits and retain the remainder until a flush.
    for width_bits, shape, conversion, source_class in (
        (544, "bfp16ebs16", "", "mEXa"),
        (576, "bfp16ebs8", "", "mEXa"),
        (544, "bfp16ebs16", "ebs8", "mEXa"),
        (544, "bfp16ebs16", "fp32", "mBMs"),
        (576, "bfp16ebs8", "fp32", "mBMs"),
    ):
        form_suffix = f"_CONV_{shape}_{conversion}" if conversion else ""
        conversion_suffix = f".from.{conversion}" if conversion else ""
        form = f"VST_PUSH_{width_bits}{form_suffix}"
        result.append(
            _DescriptorSpec(
                form,
                f"{_TARGET_KEY}.store.{shape}.fifo.push{conversion_suffix}",
                f"memory.store.fifo.push.{shape}{conversion_suffix}",
                f"II_{form}",
                storage_overrides=(("src", source_class),),
                asm_mnemonic=f"vst.push.{shape}{conversion_suffix}",
                memory_width_bits=512,
            )
        )
    # At available=0 no incoming FIFO byte participates in the transfer. The
    # reset form defines its storage directly; callers establish the empty
    # position before issuing it. No destination load initializes this state.
    result.extend(
        replace(
            spec,
            key=spec.key + ".empty",
            semantic_tag=spec.semantic_tag + ".empty",
            asm_mnemonic=spec.asm_mnemonic + ".empty",
            overwritten_inputs=("fifo_reg",),
        )
        for spec in tuple(result)
        if spec.semantic_tag.startswith("memory.store.fifo.push.")
    )
    return tuple(result)


def _fifo_storage_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Exposes partial FIFO storage and moves into its constrained state banks."""

    result = []
    for register_class, fifo_kind, half_classes in _FIFO_STORAGE_CLASSES:
        for half, half_class in zip(("low", "high"), half_classes, strict=True):
            part = f"aie2p.{register_class.lower()}.{half}512"
            for operation, prefix, mnemonic, operand in (
                ("load", "VLDA_dmx_lda", "vlda", "dst"),
                ("store", "VST_dmx_sts", "vst", "src"),
            ):
                for form_suffix, suffix in (("_imm", ""), ("", ".index")):
                    form = f"{prefix}_fifohl_idx{form_suffix}"
                    result.append(
                        _DescriptorSpec(
                            form,
                            f"{_TARGET_KEY}.{operation}.{fifo_kind}-fifo.{half}512{suffix}",
                            f"memory.{operation}.{fifo_kind}-fifo.{half}512",
                            f"II_{form}_{half_class}",
                            storage_overrides=((operand, register_class),),
                            asm_mnemonic=(
                                f"{mnemonic}.{fifo_kind}-fifo.{half}512{suffix}"
                            ),
                            operand_register_parts=((operand, part),),
                            encoding_adapter_overrides=(
                                (operand, f"LOOM_{register_class}_{half}512"),
                            ),
                            storage_continuation_part=(
                                f"aie2p.{register_class.lower()}.low512"
                                if operation == "load" and half == "high"
                                else None
                            ),
                            memory_width_bits=512,
                        )
                    )
    result.append(
        _DescriptorSpec(
            "MOVS",
            f"{_TARGET_KEY}.move.store-fifo.pointer",
            "register.move.store-fifo.pointer",
            "II_MOVS_eP_eP",
            storage_overrides=(("dst", "mPfs"), ("src", "eP")),
            asm_mnemonic="mov.store-fifo.pointer",
        )
    )
    result.append(
        _DescriptorSpec(
            "MOVS",
            f"{_TARGET_KEY}.move.load-fifo.pointer",
            "register.move.load-fifo.pointer",
            "II_MOVS_eP_eP",
            storage_overrides=(("dst", "ePS"), ("src", "eP")),
            asm_mnemonic="mov.load-fifo.pointer",
        )
    )
    return tuple(result)
