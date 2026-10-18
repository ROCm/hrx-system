# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.descriptor_overlay import _asm_forms_for_overlay
from loom.target.arch.amdgpu.descriptors.api import (
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
    _validate_native_asm_values,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _REG_AGPR,
    _REG_PART_AGPR_LOW16,
    _REG_PART_VGPR_LOW16,
    _REG_VGPR,
)
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.arch.amdgpu.descriptors.memory_candidates import (
    _MEMORY_DESCRIPTOR_CANDIDATES,
    preserved_memory_descriptor_key,
)
from loom.target.arch.amdgpu.descriptors.memory_predication import (
    with_preserving_memory_loads,
)
from loom.target.arch.amdgpu.descriptors.rdna4 import (
    _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx125x_core_overlays,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    EffectKind,
    NativeAsmValueKind,
    OperandFlag,
    OperandRole,
)


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_preserving_loads_retain_native_contract_and_address_forms(target: str) -> None:
    originals = tuple(
        replace(
            _amdgpu_contract_descriptor_from_overlay(overlay),
            asm_forms=_asm_forms_for_overlay(overlay),
        )
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    )
    expanded = with_preserving_memory_loads(originals)
    assert expanded[: len(originals)] == originals
    by_key = {descriptor.key: descriptor for descriptor in expanded}
    native_by_key = {descriptor.key: descriptor for descriptor in originals}
    for candidate in _MEMORY_DESCRIPTOR_CANDIDATES:
        if candidate.descriptor_key not in native_by_key:
            continue
        key = preserved_memory_descriptor_key(candidate.descriptor_key)
        assert (key in by_key) == (candidate.preserving_descriptor_key is not None)
    for preserved in expanded[len(originals) :]:
        original = native_by_key[preserved.key.removesuffix("_preserve")]
        result, fallback = preserved.operands[:2]
        assert result == original.operands[0]
        assert fallback == replace(
            result,
            field_name="fallback",
            role=OperandRole.OPERAND,
            flags=(OperandFlag.IMPLICIT,),
        )
        assert preserved.operands[2:] == original.operands[1:]
        assert Constraint(ConstraintKind.TIED, 0, 1) in preserved.constraints
        assert preserved.effects == original.effects
        assert preserved.schedule_class == original.schedule_class
        assert preserved.feature_mask_words == original.feature_mask_words
        assert preserved.immediates == original.immediates
        for source_form, form in zip(
            original.operand_forms, preserved.operand_forms, strict=True
        ):
            assert form.replacement_descriptor == preserved_memory_descriptor_key(
                source_form.replacement_descriptor
            )
            assert form.replacement_descriptor in by_key
            assert form.matches == source_form.matches
            assert form.immediate_action == source_form.immediate_action
        for source_form, form in zip(
            original.asm_forms, preserved.asm_forms, strict=True
        ):
            assert form.operands == ("fallback", *source_form.operands)
            assert all(
                value.field_name != "fallback" for value in form.native_assembly_values
            )
            if source_form.native_assembly_values:
                assert form.native_assembly_values == source_form.native_assembly_values


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_native_memory_forms_preserve_every_authored_immediate(target: str) -> None:
    prefixes = tuple(
        f"amdgpu.{space}_{operation}_"
        for space in ("buffer", "global", "flat")
        for operation in ("load", "store", "atomic")
    )
    for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows():
        if not overlay.descriptor_key.startswith(prefixes):
            continue
        assert overlay.asm_forms, overlay.descriptor_key
        for form in overlay.asm_forms:
            printed = {
                value.field_name
                for value in form.native_assembly_values
                if value.kind is NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT
            }
            assert printed == {field.field_name for field in overlay.immediates}, (
                overlay.descriptor_key
            )


def test_native_temporal_hint_requires_indexed_scope() -> None:
    overlay = next(
        row
        for row in _gfx12_core_overlays()
        if row.descriptor_key == "amdgpu.flat_load_b32"
    )
    descriptor = replace(
        _amdgpu_contract_descriptor_from_overlay(overlay), asm_forms=overlay.asm_forms
    )
    descriptor_set = replace(
        _AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE, descriptors=(descriptor,)
    )
    _validate_native_asm_values(descriptor_set)

    # Reordering named fields preserves parsing but invalidates the formatter's
    # direct predecessor reference. Reject this at descriptor construction.
    scope = next(
        field for field in descriptor.immediates if field.field_name == "scope"
    )
    reordered = replace(
        descriptor,
        immediates=(
            scope,
            *(field for field in descriptor.immediates if field is not scope),
        ),
    )
    with pytest.raises(ValueError, match="preceding two-bit scope"):
        _validate_native_asm_values(replace(descriptor_set, descriptors=(reordered,)))


def test_gfx12_vector_store_family_exposes_cache_scope() -> None:
    prefixes = (
        "amdgpu.global_store_",
        "amdgpu.buffer_store_",
        "amdgpu.flat_store_",
    )
    stores = tuple(
        overlay
        for overlay in _gfx12_core_overlays()
        if overlay.descriptor_key.startswith(prefixes)
    )
    assert stores
    assert {
        prefix
        for prefix in prefixes
        if any(store.descriptor_key.startswith(prefix) for store in stores)
    } == set(prefixes)
    for store in stores:
        assert "scope" in {immediate.field_name for immediate in store.immediates}, (
            store.descriptor_key
        )


def test_rdna4_global_transpose_load_family_covers_widths_and_address_forms() -> None:
    expected = {
        "amdgpu.global_load_tr_b64": (
            64,
            2,
            2,
            "memory.load.transpose.b8x8",
            32,
        ),
        "amdgpu.global_load_tr_b64_saddr": (
            64,
            2,
            1,
            "memory.load.transpose.b8x8",
            32,
        ),
        "amdgpu.global_load_tr_b64.w64": (
            64,
            1,
            2,
            "memory.load.transpose.b8x8",
            64,
        ),
        "amdgpu.global_load_tr_b64_saddr.w64": (
            64,
            1,
            1,
            "memory.load.transpose.b8x8",
            64,
        ),
        "amdgpu.global_load_tr_b128": (
            128,
            4,
            2,
            "memory.load.transpose.b16x8",
            32,
        ),
        "amdgpu.global_load_tr_b128_saddr": (
            128,
            4,
            1,
            "memory.load.transpose.b16x8",
            32,
        ),
        "amdgpu.global_load_tr_b128.w64": (
            128,
            2,
            2,
            "memory.load.transpose.b16x8",
            64,
        ),
        "amdgpu.global_load_tr_b128_saddr.w64": (
            128,
            2,
            1,
            "memory.load.transpose.b16x8",
            64,
        ),
    }
    for overlay_rows in (_gfx12_core_overlays(), _gfx125x_core_overlays()):
        overlays = {overlay.descriptor_key: overlay for overlay in overlay_rows}
        for descriptor_key, (
            width_bits,
            result_units,
            address_units,
            semantic_tag,
            wave_size,
        ) in expected.items():
            overlay = overlays[descriptor_key]
            assert overlay.instruction_name == f"GLOBAL_LOAD_TR_B{width_bits}"
            low_mnemonic_suffix = "_w64" if wave_size == 64 else ""
            assert overlay.mnemonic == (
                f"global_load_tr_b{width_bits}{low_mnemonic_suffix}"
            )
            assert overlay.semantic_tag == semantic_tag
            assert tuple(
                operand.descriptor_operand.unit_count for operand in overlay.operands
            )[:2] == (result_units, address_units)
            assert any(
                effect.kind is EffectKind.CONVERGENT for effect in overlay.effects
            )
            assert (overlay.operands[0].size_exception_reason is not None) == (
                wave_size == 64
            )
            if wave_size == 64:
                assert overlay.asm_forms is not None
                assert overlay.asm_forms[0].native_assembly_mnemonic == (
                    f"global_load_tr_b{width_bits}"
                )

    gfx11_keys = {overlay.descriptor_key for overlay in _gfx11_core_overlays()}
    assert gfx11_keys.isdisjoint(expected)


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_narrow_stores_read_only_the_low_register_part(target: str) -> None:
    descriptors = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    for width in (8, 16, 32):
        keys = (
            f"amdgpu.ds_write_b{width}",
            f"amdgpu.global_store_b{width}",
            f"amdgpu.global_store_b{width}_saddr",
            f"amdgpu.scratch_store_b{width}_vaddr",
            f"amdgpu.scratch_store_b{width}_saddr"
            if target == "cdna2"
            else f"amdgpu.scratch_store_b{width}_offset_only",
            "amdgpu.buffer_store_dword"
            if width == 32
            else f"amdgpu.buffer_store_b{width}",
        )
        for key in keys:
            descriptor = descriptors[key]
            variants = [descriptor]
            if key.startswith("amdgpu.buffer_store"):
                variants.extend(
                    descriptors[form.replacement_descriptor]
                    for form in descriptor.operand_forms
                )
            for variant in variants:
                value = next(
                    operand.descriptor_operand
                    for operand in variant.operands
                    if operand.descriptor_operand.field_name == "value"
                )
                assert value.reg_alts[0].register_part == (
                    _REG_PART_VGPR_LOW16 if width < 32 else None
                ), variant.descriptor_key


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_flat_store_read_parts_follow_the_selected_register_class(target: str) -> None:
    descriptors = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    for width in (8, 16, 32, 64, 96, 128):
        descriptor = descriptors[f"amdgpu.flat_store_b{width}"]
        value = next(
            operand.descriptor_operand
            for operand in descriptor.operands
            if operand.descriptor_operand.field_name == "value"
        )
        parts_by_class = {
            alternative.reg_class: alternative.register_part
            for alternative in value.reg_alts
        }
        assert parts_by_class[_REG_VGPR] == (
            _REG_PART_VGPR_LOW16 if width < 32 else None
        )
        if _REG_AGPR in parts_by_class:
            assert parts_by_class[_REG_AGPR] == (
                _REG_PART_AGPR_LOW16 if width < 32 else None
            )


@pytest.mark.parametrize("target", ["cdna2", "cdna3", "cdna4", "gfx9_4_generic"])
def test_cdna_memory_payloads_share_accumulator_parts_and_address_banks(
    target: str,
) -> None:
    families = set()
    for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows():
        family = overlay.instruction_name.partition("_")[0]
        if family not in ("FLAT", "GLOBAL", "SCRATCH", "BUFFER"):
            continue
        for row in overlay.operands:
            operand = row.descriptor_operand
            alternatives = {alt.reg_class: alt for alt in operand.reg_alts}
            if row.xml_field_name in ("VDST", "DATA", "VDATA"):
                assert alternatives.keys() == {_REG_VGPR, _REG_AGPR}, (
                    overlay.descriptor_key
                )
                vector = alternatives[_REG_VGPR]
                accumulator = alternatives[_REG_AGPR]
                assert (
                    replace(
                        accumulator,
                        reg_class=_REG_VGPR,
                        register_part=vector.register_part,
                    )
                    == vector
                )
                assert accumulator.register_part == (
                    _REG_PART_AGPR_LOW16
                    if vector.register_part == _REG_PART_VGPR_LOW16
                    else None
                ), overlay.descriptor_key
                families.add(family)
            else:
                assert _REG_AGPR not in alternatives, overlay.descriptor_key
    assert families == {"FLAT", "GLOBAL", "SCRATCH", "BUFFER"}


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_scratch_native_spelling_comes_from_the_selected_instruction(
    target: str,
) -> None:
    for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows():
        if not overlay.instruction_name.startswith("SCRATCH_"):
            continue
        assert overlay.asm_forms
        for form in overlay.asm_forms:
            assert form.native_assembly_mnemonic == overlay.instruction_name.lower()
