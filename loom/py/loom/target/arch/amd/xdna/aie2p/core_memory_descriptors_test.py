# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P memory descriptor widths, state, and observable effects."""

from __future__ import annotations

from loom.target.arch.amd.xdna.aie.machine import has_property
from loom.target.arch.amd.xdna.aie2p.core_descriptor_specs import (
    _DESCRIPTOR_SPECS,
    _MACHINE_FORMS,
    AIE2P_VECTOR_MEMORY_ELEMENT_TYPES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    EffectFlag,
    EffectKind,
    MemorySpace,
    OperandFlag,
)


def test_lookup_loads_preserve_address_vectors_and_memory_effects() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for width in (16, 32, 64):
        for half in ("lo", "hi"):
            descriptor = descriptors[f"amd.xdna.aie2p.load.b.lookup.4x{width}.{half}"]
            assert descriptor.asm_forms[0].results == ("dst",)
            assert descriptor.asm_forms[0].operands == ("src",)
            assert [
                operand.reg_alts[0].reg_class for operand in descriptor.operands
            ] == [
                "aie2p.vec256",
                "aie2p.vec256",
            ]
            assert all(operand.unit_count == 1 for operand in descriptor.operands)
            assert descriptor.operands[0].register_part is None
            assert descriptor.operands[1].register_part == (
                f"aie2p.vec256.{'low' if half == 'lo' else 'high'}128"
            )
            assert len(descriptor.effects) == 1
            read = descriptor.effects[0]
            assert read.kind is EffectKind.READ
            assert read.memory_space is MemorySpace.WORKGROUP
            assert read.width_bits == 256
            assert read.flags == (EffectFlag.DEPENDENCY,)
            assert read.producer_event == read.consumer_event
            assert read.producer_event


def test_fifo_load_descriptors_preserve_fifo_state_and_recurrence() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }

    for lane, alternate_lane in (("a", "b"), ("b", None)):
        fill = descriptors[f"amd.xdna.aie2p.load.{lane}.fifo.fill.512"]
        assert fill.mnemonic == f"vld{lane}.fill.512"
        assert [operand.field_name for operand in fill.operands] == [
            "ptr_out",
            "fifo_reg_out",
            "pos_out",
            "ptr",
            "fifo_reg",
            "pos",
        ]
        assert [operand.reg_alts[0].reg_class for operand in fill.operands] == [
            "aie2p.eps",
            "aie2p.eldfiforeg",
            "aie2p.erf2",
            "aie2p.eps",
            "aie2p.eldfiforeg",
            "aie2p.erf2",
        ]
        assert fill.asm_forms[0].results == (
            "ptr_out",
            "fifo_reg_out",
            "pos_out",
        )
        assert fill.asm_forms[0].operands == ("ptr", "fifo_reg", "pos")
        assert fill.effects[0].kind is EffectKind.READ
        assert fill.effects[0].width_bits == 512
        assert fill.schedule_alternatives == (
            (f"amd.xdna.aie2p.load.{alternate_lane}.fifo.fill.512",)
            if alternate_lane is not None
            else ()
        )

        tied_names = {
            (
                fill.operands[constraint.lhs_operand_index].field_name,
                fill.operands[constraint.rhs_operand_index].field_name,
            )
            for constraint in fill.constraints
            if constraint.kind is ConstraintKind.TIED
        }
        assert tied_names == {
            ("ptr_out", "ptr"),
            ("fifo_reg_out", "fifo_reg"),
            ("pos_out", "pos"),
        }
        coindexed_names = {
            frozenset(
                (
                    fill.operands[constraint.lhs_operand_index].field_name,
                    fill.operands[constraint.rhs_operand_index].field_name,
                )
            )
            for constraint in fill.constraints
            if constraint.kind is ConstraintKind.SAME_REGISTER_ORDINAL
        }
        assert coindexed_names == {
            frozenset(("ptr", "fifo_reg")),
            frozenset(("ptr", "pos")),
            frozenset(("fifo_reg", "pos")),
        }

        for shape, width in (
            *(
                (f"{element}x{512 // bits}", 512)
                for element, bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES
            ),
            ("bfp16ebs16", 544),
            ("bfp16ebs8", 576),
        ):
            pop = descriptors[f"amd.xdna.aie2p.load.{lane}.{shape}.fifo.pop"]
            mnemonic_shape = f"512.{shape}" if width == 512 else shape
            assert pop.mnemonic == f"vld{lane}.pop.{mnemonic_shape}"
            assert [operand.field_name for operand in pop.operands] == [
                "dst",
                "ptr_out",
                "fifo_reg_out",
                "pos_out",
                "ptr",
                "fifo_reg",
                "pos",
                "implicit_def_srfifo_uf",
            ]
            assert pop.asm_forms[0].results == (
                "dst",
                "ptr_out",
                "fifo_reg_out",
                "pos_out",
            )
            assert pop.asm_forms[0].operands == ("ptr", "fifo_reg", "pos")
            assert pop.effects[0].kind is EffectKind.READ
            assert pop.effects[0].width_bits == 512
            assert pop.operands[0].reg_alts[0].reg_class == (
                "aie2p.vec256" if width == 512 else "aie2p.mexa"
            )
            assert pop.operands[0].unit_count == (2 if width == 512 else 1)
            assert pop.operands[-1].reg_alts[0].reg_class == "aie2p.state.srfifo_uf"
            assert OperandFlag.STATE_WRITE in pop.operands[-1].flags
            assert pop.schedule_alternatives == (
                (f"amd.xdna.aie2p.load.{alternate_lane}.{shape}.fifo.pop",)
                if alternate_lane is not None
                else ()
            )

            tied_names = {
                (
                    pop.operands[constraint.lhs_operand_index].field_name,
                    pop.operands[constraint.rhs_operand_index].field_name,
                )
                for constraint in pop.constraints
                if constraint.kind is ConstraintKind.TIED
            }
            assert tied_names == {
                ("ptr_out", "ptr"),
                ("fifo_reg_out", "fifo_reg"),
                ("pos_out", "pos"),
            }
            coindexed_names = {
                frozenset(
                    (
                        pop.operands[constraint.lhs_operand_index].field_name,
                        pop.operands[constraint.rhs_operand_index].field_name,
                    )
                )
                for constraint in pop.constraints
                if constraint.kind is ConstraintKind.SAME_REGISTER_ORDINAL
            }
            assert coindexed_names == {
                frozenset(("ptr", "fifo_reg")),
                frozenset(("ptr", "pos")),
                frozenset(("fifo_reg", "pos")),
            }


def test_fifo_seeks_retain_tuple_and_dimension_ownership() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    shapes = (
        *(
            f"{element}x{512 // bits}"
            for element, bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES
        ),
        "bfp16ebs16",
        "bfp16ebs8",
    )
    keys = (
        *(f"load.{lane}.{shape}.fifo.pop" for lane in ("a", "b") for shape in shapes),
        "store.fifo.flush.512",
        "store.fifo.flush.convert.512",
    )
    for key in keys:
        base = descriptors[f"amd.xdna.aie2p.{key}"]
        for addressing in ("postincrement.register", "2d", "3d"):
            descriptor = descriptors[f"{base.key}.{addressing}"]
            operands = {operand.field_name: operand for operand in descriptor.operands}
            names = tuple(operands)
            ties = {
                (names[tie.lhs_operand_index], names[tie.rhs_operand_index])
                for tie in descriptor.constraints
                if tie.kind is ConstraintKind.TIED
            }
            position = "pos" if key.startswith("load.") else "avail"
            expected = {
                ("ptr_out", "ptr"),
                ("fifo_reg_out", "fifo_reg"),
                (f"{position}_out", position),
            }
            if addressing in ("2d", "3d"):
                count = "dc" if addressing == "2d" else "dcl"
                expected.add((count, "mod"))
                register_class = "aie2p.ed" if addressing == "2d" else "aie2p.eds"
                assert operands[count].reg_alts == operands["mod"].reg_alts
                assert operands[count].reg_alts[0].reg_class == register_class
                assert operands[count].register_part == f"{register_class}.counts"
                assert operands["mod"].register_part == f"{register_class}.state"
                assert operands[count].encoding_field_id == 0
                assert OperandFlag.STORAGE_CONTINUATION not in operands["mod"].flags
                assert "dch" not in operands
            assert ties == expected
            assert descriptor.effects == base.effects
            assert descriptor.schedule_alternatives == tuple(
                f"{alternate}.{addressing}" for alternate in base.schedule_alternatives
            )
            for name in ("ptr", "fifo_reg", position):
                assert operands[name].reg_alts == next(
                    operand.reg_alts
                    for operand in base.operands
                    if operand.field_name == name
                )
            assert {
                frozenset((names[tie.lhs_operand_index], names[tie.rhs_operand_index]))
                for tie in descriptor.constraints
                if tie.kind is ConstraintKind.SAME_REGISTER_ORDINAL
            } == (
                {
                    frozenset(pair)
                    for pair in (
                        ("ptr", "fifo_reg"),
                        ("ptr", "pos"),
                        ("fifo_reg", "pos"),
                    )
                }
                if key.startswith("load.")
                else set()
            )


def test_fifo_stores_preserve_the_fixed_tuple_and_overflow_state() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    flushes = (
        descriptors["amd.xdna.aie2p.store.fifo.flush.512"],
        descriptors["amd.xdna.aie2p.store.fifo.flush.convert.512"],
    )
    pushes = [
        descriptor
        for descriptor in descriptors.values()
        if descriptor.key.startswith("amd.xdna.aie2p.store.")
        and ".fifo.push" in descriptor.key
        and not descriptor.key.endswith(".volatile")
    ]
    for descriptor in (*flushes, *pushes):
        assert descriptor.asm_forms[0].results == (
            "fifo_reg_out",
            "ptr_out",
            "avail_out",
        )
        assert descriptor.asm_forms[0].operands == (
            ("fifo_reg", "ptr", "avail")
            if descriptor in flushes
            else ("fifo_reg", "src", "ptr", "avail")
        )
        assert descriptor.effects[0].kind is EffectKind.WRITE
        assert descriptor.effects[0].width_bits == 512
        assert {
            (
                descriptor.operands[constraint.lhs_operand_index].field_name,
                descriptor.operands[constraint.rhs_operand_index].field_name,
            )
            for constraint in descriptor.constraints
            if constraint.kind is ConstraintKind.TIED
        } == {
            ("fifo_reg_out", "fifo_reg"),
            ("ptr_out", "ptr"),
            ("avail_out", "avail"),
        }
        overflow = next(
            operand
            for operand in descriptor.operands
            if operand.field_name == "implicit_def_srfifo_of"
        )
        assert OperandFlag.STATE_WRITE in overflow.flags


def test_fused_vector_memory_descriptors_preserve_conversion_and_memory_contracts() -> (
    None
):
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }

    for source_lane_count in (32, 64):
        result_lane_count = source_lane_count * 2
        for source_kind, sign_bit in (("u", 0), ("s", 1)):
            for address_form, address_field in (
                ("register", "dj"),
                ("immediate", None),
            ):
                key = (
                    f"amd.xdna.aie2p.load.unpack.{source_kind}4x"
                    f"{result_lane_count}.to.{source_kind}8x{result_lane_count}."
                    f"configured.indexed.{address_form}"
                )
                descriptor = descriptors[key]
                assert descriptor.mnemonic == (
                    f"vldb.unpack.{source_kind}4.to.{source_kind}8x"
                    f"{result_lane_count}"
                    f"{'.index' if address_form == 'register' else ''}"
                )
                assert [operand.field_name for operand in descriptor.operands] == [
                    "dst",
                    "ptr",
                    *([address_field] if address_field is not None else []),
                    "implicit_use_crunpacksize",
                    f"implicit_use_unpacksign{sign_bit}",
                ]
                assert descriptor.operands[0].reg_alts[0].reg_class == ("aie2p.vec256")
                assert descriptor.operands[0].unit_count == result_lane_count // 32
                assert descriptor.effects[0].kind is EffectKind.READ
                assert descriptor.effects[0].width_bits == source_lane_count * 8

    load_convert_shapes = (
        ("bf16x16", "f32x16", 256, 1),
        ("bf16x32", "f32x32", 512, 2),
    )
    for (
        source_shape,
        result_shape,
        memory_width_bits,
        result_units,
    ) in load_convert_shapes:
        for address_form, address_field in (
            ("register", "dj"),
            ("immediate", None),
        ):
            key = (
                f"amd.xdna.aie2p.load.convert.{source_shape}.to.{result_shape}"
                f".indexed.{address_form}"
            )
            descriptor = descriptors[key]
            assert descriptor.mnemonic == (
                f"vlda.convert.{source_shape}.to.{result_shape}"
                f"{'.index' if address_form == 'register' else ''}"
            )
            assert descriptor.semantic_tag == (
                f"convert.floating.{source_shape}.to.{result_shape}.memory.load"
            )
            assert [operand.field_name for operand in descriptor.operands] == [
                "op",
                "ptr",
                *([address_field] if address_field is not None else []),
            ]
            assert descriptor.operands[0].reg_alts[0].reg_class == "aie2p.mbms"
            assert descriptor.operands[0].unit_count == result_units
            assert len(descriptor.effects) == 1
            assert descriptor.effects[0].kind is EffectKind.READ
            assert descriptor.effects[0].width_bits == memory_width_bits

    for shape, memory_width_bits, result_units in (
        ("2x.w-to-b", 256, 1),
        ("4x.w-to-c", 256, 2),
        ("2x.x-to-c", 512, 2),
        ("4x.x-to-d", 512, 4),
    ):
        for signedness, sign_bit in (("unsigned", 0), ("signed", 1)):
            key = (
                f"amd.xdna.aie2p.load.widen.{shape}.{signedness}.configured"
                ".indexed.immediate"
            )
            descriptor = descriptors[key]
            assert [operand.field_name for operand in descriptor.operands] == [
                "dst",
                "su",
                "ptr",
                "implicit_def_srups_of",
                "implicit_use_crsat",
                "implicit_use_crupsmode",
                f"implicit_use_upssign{sign_bit}",
            ]
            assert descriptor.operands[0].reg_alts[0].reg_class == "aie2p.mbms"
            assert descriptor.operands[0].unit_count == result_units
            assert descriptor.effects[0].kind is EffectKind.READ
            assert descriptor.effects[0].width_bits == memory_width_bits

    for source_shape, result_shape, memory_width_bits, source_units in (
        ("f32x16", "bf16x16", 256, 1),
        ("f32x32", "bf16x32", 512, 2),
    ):
        key = (
            f"amd.xdna.aie2p.store.convert.{source_shape}.to.{result_shape}"
            ".indexed.immediate"
        )
        descriptor = descriptors[key]
        assert [operand.field_name for operand in descriptor.operands] == [
            "src",
            "ptr",
            "implicit_def_srf2fflags",
            "implicit_use_crf2fmask",
            "implicit_use_crrnd",
        ]
        assert descriptor.operands[0].reg_alts[0].reg_class == "aie2p.mbms"
        assert descriptor.operands[0].unit_count == source_units
        assert descriptor.effects[0].kind is EffectKind.WRITE
        assert descriptor.effects[0].width_bits == memory_width_bits

    for width, memory_width_bits, source_units in (
        ("w", 256, 2),
        ("x", 512, 4),
    ):
        key = f"amd.xdna.aie2p.store.pack.{width}.trunc.configured.indexed.immediate"
        descriptor = descriptors[key]
        assert [operand.field_name for operand in descriptor.operands] == [
            "src",
            "ptr",
            "implicit_use_crpacksize",
            "implicit_use_crsat",
            "implicit_use_packsign0",
        ]
        assert descriptor.operands[0].reg_alts[0].reg_class == "aie2p.vec256"
        assert descriptor.operands[0].unit_count == source_units
        assert descriptor.effects[0].kind is EffectKind.WRITE
        assert descriptor.effects[0].width_bits == memory_width_bits

    for key in tuple(descriptors):
        if not key.startswith(
            (
                "amd.xdna.aie2p.load.convert.",
                "amd.xdna.aie2p.load.unpack.",
                "amd.xdna.aie2p.load.widen.",
                "amd.xdna.aie2p.store.convert.",
                "amd.xdna.aie2p.store.narrow.",
                "amd.xdna.aie2p.store.pack.",
            )
        ) or key.endswith(".volatile"):
            continue
        ordinary = descriptors[key]
        ordered = descriptors[f"{key}.volatile"]
        assert ordered.mnemonic == f"{ordinary.mnemonic}.volatile"
        assert ordered.effects[0].kind is ordinary.effects[0].kind
        assert ordered.effects[0].width_bits == ordinary.effects[0].width_bits
        assert EffectFlag.ORDERED in ordered.effects[0].flags


def test_vector_memory_descriptors_cover_each_native_width_and_value_shape() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for width_bits in (128, 256, 512):
        unit_count = 2 if width_bits == 512 else 1
        immediate_step = width_bits // 8
        for element_type, element_bits in (
            ("i8", 8),
            ("i16", 16),
            ("bf16", 16),
            ("i32", 32),
            ("f32", 32),
        ):
            shape = f"{element_type}x{width_bits // element_bits}"
            for load_pipe in ("a", "b"):
                for address_form in ("immediate", "register"):
                    descriptor = descriptors[
                        f"amd.xdna.aie2p.load.{load_pipe}.{shape}.indexed."
                        f"{address_form}"
                    ]
                    payload = descriptor.operands[0]
                    assert payload.reg_alts[0].reg_class == "aie2p.vec256"
                    assert payload.unit_count == unit_count
                    assert payload.register_part is None
                    assert all(
                        operand.field_name != "storage"
                        for operand in descriptor.operands
                    )
                    assert all(
                        constraint.kind is not ConstraintKind.TIED
                        for constraint in descriptor.constraints
                    )
                    assert descriptor.effects[0].width_bits == width_bits
                    if address_form == "immediate":
                        assert descriptor.immediates[0].value_step == immediate_step
            for address_form in ("immediate", "register"):
                descriptor = descriptors[
                    f"amd.xdna.aie2p.store.{shape}.indexed.{address_form}"
                ]
                payload = descriptor.operands[0]
                assert payload.reg_alts[0].reg_class == "aie2p.vec256"
                assert payload.unit_count == unit_count
                assert descriptor.effects[0].width_bits == width_bits
                if address_form == "immediate":
                    assert descriptor.immediates[0].value_step == immediate_step

            load = descriptors[f"amd.xdna.aie2p.load.a.{shape}.indexed.immediate"]
            store = descriptors[f"amd.xdna.aie2p.store.{shape}.indexed.immediate"]
            if width_bits == 128:
                assert store.operands[0].register_part == "aie2p.vec256.low128"
            else:
                assert load.operands[0].register_part is None
                assert store.operands[0].register_part is None
                assert all(operand.field_name != "storage" for operand in load.operands)


def test_float_vector_memory_descriptors_reuse_bit_exact_physical_forms() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    for width_bits in (128, 256, 512):
        for value_type, storage_type, element_bits in (
            ("bf16", "i16", 16),
            ("f32", "i32", 32),
        ):
            value_shape = f"{value_type}x{width_bits // element_bits}"
            storage_shape = f"{storage_type}x{width_bits // element_bits}"
            for descriptor_family in ("load.a", "load.b", "store"):
                for address_form in ("immediate", "register"):
                    value_descriptor = descriptors[
                        f"amd.xdna.aie2p.{descriptor_family}.{value_shape}.indexed."
                        f"{address_form}"
                    ]
                    storage_descriptor = descriptors[
                        f"amd.xdna.aie2p.{descriptor_family}.{storage_shape}.indexed."
                        f"{address_form}"
                    ]
                    assert value_descriptor.encoding_id == (
                        storage_descriptor.encoding_id
                    )
                    assert value_descriptor.encoding_field_values == (
                        storage_descriptor.encoding_field_values
                    )
                    assert value_descriptor.operands == storage_descriptor.operands
                    assert value_descriptor.immediates == storage_descriptor.immediates


def test_ordered_memory_descriptors_are_semantic_physical_aliases() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    ordinary_specs = {
        spec.key: spec for spec in _DESCRIPTOR_SPECS if not spec.ordered_memory
    }
    ordered_specs = tuple(spec for spec in _DESCRIPTOR_SPECS if spec.ordered_memory)
    assert ordered_specs
    assert {spec.key.removesuffix(".volatile") for spec in ordered_specs} == {
        spec.key
        for spec in ordinary_specs.values()
        if has_property(_MACHINE_FORMS[spec.form_name], "mayLoad")
        or has_property(_MACHINE_FORMS[spec.form_name], "mayStore")
    }

    for ordered_spec in ordered_specs:
        ordinary_key = ordered_spec.key.removesuffix(".volatile")
        ordinary_spec = ordinary_specs[ordinary_key]
        assert ordered_spec.form_name == ordinary_spec.form_name
        assert ordered_spec.itinerary == ordinary_spec.itinerary
        assert ordered_spec.schedule_alternatives == tuple(
            f"{key}.volatile" for key in ordinary_spec.schedule_alternatives
        )

        ordinary = descriptors[ordinary_key]
        ordered = descriptors[ordered_spec.key]
        assert ordered.encoding_id == ordinary.encoding_id
        assert ordered.schedule_class == ordinary.schedule_class
        assert ordered.operands == ordinary.operands
        assert ordered.immediates == ordinary.immediates
        assert ordered.encoding_field_values == ordinary.encoding_field_values
        assert ordered.constraints == ordinary.constraints
        assert ordered.flags == ordinary.flags
        assert ordered.instruction_classes == ordinary.instruction_classes
        assert ordered.mnemonic == f"{ordinary.mnemonic}.volatile"
        assert ordered.semantic_tag == f"{ordinary.semantic_tag}.volatile"
        assert ordered.schedule_alternatives == tuple(
            f"{key}.volatile" for key in ordinary.schedule_alternatives
        )
        assert len(ordered.asm_forms) == len(ordinary.asm_forms) == 1
        assert (
            ordered.asm_forms[0].mnemonic
            == f"{ordinary.asm_forms[0].mnemonic}.volatile"
        )
        assert (
            ordered.asm_forms[0].native_assembly_mnemonic
            == ordinary.asm_forms[0].mnemonic
        )
        assert len(ordered.effects) == len(ordinary.effects)
        for ordered_effect, ordinary_effect in zip(
            ordered.effects, ordinary.effects, strict=True
        ):
            assert ordered_effect.kind == ordinary_effect.kind
            assert ordered_effect.memory_space == ordinary_effect.memory_space
            assert ordered_effect.scope_id == ordinary_effect.scope_id
            assert ordered_effect.counter_id == ordinary_effect.counter_id
            assert ordered_effect.width_bits == ordinary_effect.width_bits
            assert ordered_effect.producer_event == ordinary_effect.producer_event
            assert ordered_effect.consumer_event == ordinary_effect.consumer_event
            assert ordered_effect.flags == (
                EffectFlag.ORDERED,
                *ordinary_effect.flags,
            )


def test_dimension_updates_retain_real_read_dependencies() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    for dimension, register_class in ((2, "ed"), (3, "eds")):
        descriptor = descriptors[f"amd.xdna.aie2p.load.scalar.i32.{dimension}d"]
        # Both 3D native counters belong to the one returned aggregate; neither
        # can acquire an independent physical index or lose its read dependency.
        assert len(descriptor.asm_forms[0].results) == 3
        result, source = descriptor.operands[2], descriptor.operands[4]
        assert result.reg_alts == source.reg_alts
        assert result.reg_alts[0].reg_class == f"aie2p.{register_class}"
        assert result.encoding_field_id == 0
        assert result.ready_stage == source.read_stage == 1
        assert OperandFlag.STORAGE_CONTINUATION not in source.flags
        assert Constraint(ConstraintKind.TIED, 2, 4) in descriptor.constraints


def test_vector_address_updates_preserve_pointer_and_dimension_ownership() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    for width in (128, 256, 512):
        for element_type, element_bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES:
            shape = f"{element_type}x{width // element_bits}"
            for family in ("load.a", "load.b", "store"):
                for addressing in (
                    "postincrement.immediate",
                    "postincrement.register",
                    "2d",
                    "3d",
                ):
                    descriptor = descriptors[
                        f"amd.xdna.aie2p.{family}.{shape}.{addressing}"
                    ]
                    operands = {
                        operand.field_name: operand for operand in descriptor.operands
                    }
                    tied_names = {
                        (
                            descriptor.operands[tie.lhs_operand_index].field_name,
                            descriptor.operands[tie.rhs_operand_index].field_name,
                        )
                        for tie in descriptor.constraints
                        if tie.kind is ConstraintKind.TIED
                    }
                    expected_ties = {("ptr_out", "ptr")}
                    if addressing in ("2d", "3d"):
                        count = "dc" if addressing == "2d" else "dcl"
                        expected_ties.add((count, "mod"))
                        assert operands[count].reg_alts == operands["mod"].reg_alts
                        assert operands[count].encoding_field_id == 0
                        assert (
                            operands[count].ready_stage
                            == operands["mod"].read_stage
                            == 1
                        )
                        assert (
                            OperandFlag.STORAGE_CONTINUATION
                            not in operands["mod"].flags
                        )
                        assert "dch" not in operands
                    assert tied_names == expected_ties
                    assert "storage" not in operands
                    if family != "store":
                        assert operands["dst"].register_part is None
                    assert descriptor.effects[0].width_bits == width
                    assert descriptor.schedule_alternatives == (
                        (f"amd.xdna.aie2p.load.b.{shape}.{addressing}",)
                        if family == "load.a"
                        else ()
                    )


def test_accumulator_address_updates_retain_raw_storage() -> None:
    for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors:
        if not descriptor.key.startswith(
            ("amd.xdna.aie2p.load.accumulator.", "amd.xdna.aie2p.store.accumulator.")
        ):
            continue
        value = next(
            operand
            for operand in descriptor.operands
            if operand.field_name in ("src", "dst")
        )
        assert value.reg_alts[0].reg_class == "aie2p.mbms"
        assert value.unit_count == 1
        assert value.register_part is None
        assert descriptor.effects[0].width_bits == 512
        assert all(
            not operand.field_name.startswith("implicit_")
            for operand in descriptor.operands
        )


def test_fused_address_updates_preserve_numeric_storage_and_state() -> None:
    descriptors = {row.key: row for row in AIE2P_CORE_DESCRIPTOR_SET.descriptors}
    for key, indexed in descriptors.items():
        if not key.endswith(".indexed.register") or not key.startswith(
            (
                "amd.xdna.aie2p.load.convert.",
                "amd.xdna.aie2p.load.unpack.",
                "amd.xdna.aie2p.load.widen.",
                "amd.xdna.aie2p.store.convert.",
                "amd.xdna.aie2p.store.narrow.",
                "amd.xdna.aie2p.store.pack.",
            )
        ):
            continue
        numeric_operands = {
            operand.field_name: operand
            for operand in indexed.operands
            if operand.field_name not in ("ptr", "dj")
        }
        for addressing in (
            "postincrement.immediate",
            "postincrement.register",
            "2d",
            "3d",
        ):
            updated = descriptors[
                f"{key.removesuffix('.indexed.register')}.{addressing}"
            ]
            assert updated.semantic_tag == indexed.semantic_tag
            assert updated.effects == indexed.effects
            numeric_fields = set()
            for operand in updated.operands:
                if operand.field_name in ("ptr", "ptr_out", "mod", "dc", "dcl"):
                    continue
                original = numeric_operands[operand.field_name]
                assert operand.reg_alts == original.reg_alts
                assert operand.unit_count == original.unit_count
                assert operand.register_part == original.register_part
                assert operand.flags == original.flags
                numeric_fields.add(operand.field_name)
            assert numeric_fields == numeric_operands.keys()
