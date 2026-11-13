# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P address setup and memory operations with explicit pointer recurrence."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.arch.amd.xdna.aie2p.core_machine_data import DIMENSION_FIELDS
from loom.target.low_descriptors import DescriptorFlag, DescriptorOpKind, RegisterPart

_TARGET_KEY = "amd.xdna.aie2p"
_SCALAR_MEMORY_FORMS = (
    ("load", "LDA", "dms_lda", "i32", 32),
    ("load", "LDA", "s8", "s8", 8),
    ("load", "LDA", "u8", "u8", 8),
    ("load", "LDA", "s16", "s16", 16),
    ("load", "LDA", "u16", "u16", 16),
    ("store", "ST", "dms_sts", "i32", 32),
    ("store", "ST", "s8", "i8", 8),
    ("store", "ST", "s16", "i16", 16),
)
DIMENSION_REGISTER_PARTS = tuple(
    part
    for register_class, fields in DIMENSION_FIELDS.items()
    for prefix in (f"aie2p.{register_class.lower()}",)
    for part in (
        *(
            RegisterPart(f"{prefix}.{name}", prefix, 1 << index)
            for index, (name, _, _) in enumerate(fields)
        ),
        *(
            RegisterPart(f"{prefix}.before.{name}", prefix, (1 << index) - 1)
            for index, (name, _, _) in enumerate(fields)
            if index
        ),
        RegisterPart(f"{prefix}.after.count", prefix, (1 << len(fields)) - 2),
        RegisterPart(f"{prefix}.state", prefix, (1 << len(fields)) - 1),
        RegisterPart(
            f"{prefix}.counts",
            prefix,
            sum(
                1 << index
                for index, (name, _, _) in enumerate(fields)
                if name.endswith("count")
            ),
        ),
    )
)


def _dimension_update(spec: _DescriptorSpec, dimension: int) -> _DescriptorSpec:
    """Ties native count writes to the owning modifier's next aggregate value."""

    register_class = "eD" if dimension == 2 else "eDS"
    prefix = f"aie2p.{register_class.lower()}"
    outputs = ("dc",) if dimension == 2 else ("dcl", "dch")
    return replace(
        spec,
        storage_overrides=(*spec.storage_overrides, (outputs[0], register_class)),
        operand_register_parts=(
            *spec.operand_register_parts,
            (outputs[0], f"{prefix}.counts"),
            ("mod", f"{prefix}.state"),
        ),
        encoding_adapter_overrides=(
            *spec.encoding_adapter_overrides,
            ("mod", f"LOOM_{register_class}"),
        ),
        aggregate_updates=(("mod", outputs),),
    )


def _dimension_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Builds and updates one native aggregate without separate count storage."""

    result = []
    for dimension, register_class in ((2, "eD"), (3, "eDS")):
        fields = DIMENSION_FIELDS[register_class]
        part_prefix = f"aie2p.{register_class.lower()}"
        for index, (component, _, machine_class) in enumerate(fields):
            # Prefix continuations support construction and later field updates:
            # ties preserve every already-defined part, including later fields.
            continuations = [
                ("", f"{part_prefix}.before.{component}" if index else None)
            ]
            if not index:
                continuations.append((".update", f"{part_prefix}.after.count"))
            for suffix, continuation in continuations:
                for form, mnemonic, adapter, inputs in (
                    ("MOVA", "mova", "OP_mLdaCg", ()),
                    ("MOVXM", "movxm", "OP_mMvSclDstCg", ()),
                    ("MOV_alu_mv_mv_mv_scl", "mov", "OP_mMvSclDst", (("src", "eR"),)),
                ):
                    result.append(
                        _DescriptorSpec(
                            form,
                            f"{_TARGET_KEY}.dimension.{dimension}d.{mnemonic}.{component}{suffix}",
                            f"dimension.{dimension}d.set.{component}{suffix}",
                            f"II_{form}_{machine_class}{'_eR' if inputs else ''}",
                            storage_overrides=(("dst", register_class), *inputs),
                            operand_register_parts=(
                                ("dst", f"{part_prefix}.{component}"),
                            ),
                            encoding_adapter_overrides=(
                                ("dst", f"LOOM_{register_class}_{component}_{adapter}"),
                            ),
                            storage_continuation_part=continuation,
                            asm_mnemonic=(
                                f"mov.{dimension}d.{component}.immediate{suffix}"
                                if form == "MOVXM"
                                else f"{mnemonic}.{dimension}d.{component}{suffix}"
                            ),
                        )
                    )
            result.append(
                _DescriptorSpec(
                    "MOV_alu_mv_mv_mv_scl",
                    f"{_TARGET_KEY}.dimension.{dimension}d.read.{component}",
                    f"dimension.{dimension}d.read.{component}",
                    f"II_MOV_alu_mv_mv_mv_scl_eR_{machine_class}",
                    storage_overrides=(("dst", "eR"), ("src", register_class)),
                    operand_register_parts=(("src", f"{part_prefix}.{component}"),),
                    encoding_adapter_overrides=(
                        ("src", f"LOOM_{register_class}_{component}_OP_mMvSclSrc"),
                    ),
                    asm_mnemonic=f"mov.{dimension}d.{component}-to-scalar",
                )
            )
        for operation, prefix, suffix, shape, width in _SCALAR_MEMORY_FORMS:
            form = f"{prefix}_{dimension}D_{suffix}"
            mnemonic, operand = ("lda", "dst") if operation == "load" else ("st", "src")
            result.append(
                _dimension_update(
                    _DescriptorSpec(
                        form,
                        f"{_TARGET_KEY}.{operation}.scalar.{shape}.{dimension}d",
                        f"memory.{operation}.{dimension}d.{shape}",
                        f"II_{form}{'_eR' if width == 32 else ''}",
                        storage_overrides=((operand, "eR"),),
                        asm_mnemonic=f"{mnemonic}.{shape}.{dimension}d",
                        memory_width_bits=width,
                    ),
                    dimension,
                )
            )
        # B and S have identical completion timing. A retains a longer AGU
        # itinerary and cannot participate in their schedule alternatives.
        for lane in ("a", "b", "s"):
            form = f"PADD{lane.upper()}_{dimension}D"
            result.append(
                _dimension_update(
                    _DescriptorSpec(
                        form,
                        f"{_TARGET_KEY}.address.add.{lane}.{dimension}d",
                        f"address.add.{dimension}d.i20",
                        f"II_{form}",
                        asm_mnemonic=f"padd{lane}.{dimension}d",
                        schedule_alternatives=tuple(
                            f"{_TARGET_KEY}.address.add.{alternate}.{dimension}d"
                            for alternate in ("b", "s")
                            if lane != "a" and alternate != lane
                        ),
                    ),
                    dimension,
                )
            )
    return tuple(result)


def _address_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects modifier setup and native post-increment pointer updates."""

    result = list(_dimension_descriptor_specs())
    for form, addressing, mnemonic in (
        ("ADD_NC_mv_add_rr", "register", "add.address-index"),
        ("ADD_NC_mv_add_ri", "immediate", "add.address-index.immediate"),
    ):
        result.append(
            _DescriptorSpec(
                form,
                f"{_TARGET_KEY}.address.index.add.{addressing}",
                "address.index.add.i20",
                f"II_{form}_eDJ",
                storage_overrides=(("dst", "eDJ"),),
                asm_mnemonic=mnemonic,
                flags=(DescriptorFlag.SAFE_TO_SPECULATE,),
                rematerializable=True,
            )
        )
    for form, mnemonic in (
        ("MOVA", "mova.modifier"),
        ("MOVXM", "mov.modifier.immediate"),
    ):
        result.append(
            _DescriptorSpec(
                form,
                f"{_TARGET_KEY}.constant.modifier.{form.lower()}",
                "integer.const.i20",
                f"II_{form}_eM",
                storage_overrides=(("dst", "eM"),),
                op_kind=DescriptorOpKind.CONST,
                asm_mnemonic=mnemonic,
            )
        )
    for destination, source, mnemonic in (
        ("eM", "eR", "mov.modifier"),
        ("eR", "eM", "mov.modifier-to-scalar"),
        ("eM", "eM", "mov.modifier.copy"),
    ):
        result.append(
            _DescriptorSpec(
                "MOV_alu_mv_mv_mv_scl",
                f"{_TARGET_KEY}.move.{source.lower()}.to.{destination.lower()}",
                f"register.move.{source.lower()}.to.{destination.lower()}",
                f"II_MOV_alu_mv_mv_mv_scl_{destination}_{source}",
                storage_overrides=(("dst", destination), ("src", source)),
                asm_mnemonic=mnemonic,
                allocation_move=destination == source,
                rematerializable=destination != source,
                flags=(DescriptorFlag.SAFE_TO_SPECULATE,),
            )
        )
    for immediate, suffix in ((False, ""), (True, "_imm")):
        addressing = "immediate" if immediate else "register"
        mnemonic_suffix = "" if immediate else ".modifier"
        for operation, prefix, form_suffix, shape, width in _SCALAR_MEMORY_FORMS:
            form = f"{prefix}_{form_suffix}_pstm_nrm{suffix}"
            mnemonic, operand = ("lda", "dst") if operation == "load" else ("st", "src")
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.{operation}.scalar.{shape}.postincrement.{addressing}",
                    f"memory.{operation}.postincrement.{shape}",
                    f"II_{form}{'_eR' if width == 32 else ''}",
                    storage_overrides=((operand, "eR"),),
                    asm_mnemonic=f"{mnemonic}.{shape}.post{mnemonic_suffix}",
                    memory_width_bits=width,
                )
            )
        for lane in ("a", "b", "s"):
            form = f"PADD{lane.upper()}_pstm_nrm{suffix}"
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.address.add.{lane}.{addressing}",
                    "address.add.i20",
                    f"II_{form}",
                    asm_mnemonic=f"padd{lane}{mnemonic_suffix}",
                    flags=(DescriptorFlag.SAFE_TO_SPECULATE,),
                    rematerializable=True,
                    schedule_alternatives=tuple(
                        f"{_TARGET_KEY}.address.add.{alternate}.{addressing}"
                        for alternate in ("b", "s")
                        if lane != "a" and alternate != lane
                    ),
                )
            )
    return tuple(result)
