# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.descriptors.alignment import _with_operand_alignment
from loom.target.arch.amdgpu.descriptors.api import _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS
from loom.target.arch.amdgpu.descriptors.common import _REG_SGPR, _REG_VGPR
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.low_descriptors import RegClassAlt, RegClassFlag


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_instruction_alignment_across_native_families(target) -> None:
    builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    source = replace(
        builder.base,
        descriptors=(
            *(
                _amdgpu_contract_descriptor_from_overlay(overlay)
                for overlay in builder.overlay_rows()
            ),
            *builder.extra_descriptors,
            *builder.base.descriptors,
        ),
    )
    aligned = _with_operand_alignment(source)
    assert aligned.reg_classes == source.reg_classes
    seen_pairs = seen_wide = seen_vectors = False
    for original, descriptor in zip(
        source.descriptors, aligned.descriptors, strict=True
    ):
        for original_operand, operand in zip(
            original.operands, descriptor.operands, strict=True
        ):
            for original_alternative, alternative in zip(
                original_operand.reg_alts, operand.reg_alts, strict=True
            ):
                expected = original_alternative.unit_alignment
                if alternative.reg_class == _REG_SGPR:
                    if operand.unit_count == 2:
                        expected = max(expected, 2)
                        seen_pairs = True
                    elif operand.unit_count > 2:
                        expected = max(expected, 4)
                        seen_wide = True
                elif alternative.reg_class == _REG_VGPR and operand.unit_count > 1:
                    seen_vectors = True
                assert alternative.unit_alignment == expected, (
                    descriptor.key,
                    operand.field_name,
                )
    assert seen_pairs and seen_wide and seen_vectors
    vgpr = next(row for row in aligned.reg_classes if row.name == _REG_VGPR)
    assert (RegClassFlag.EVEN_ALIGNED_TUPLES in vgpr.flags) == target.startswith(
        ("cdna", "gfx9_4")
    )


def test_alignment_belongs_to_selected_register_alternative() -> None:
    builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS["rdna3_5"]
    descriptor = _amdgpu_contract_descriptor_from_overlay(builder.overlay_rows()[0])
    operand = replace(
        descriptor.operands[0],
        unit_count=4,
        reg_alts=(RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR)),
    )
    source = replace(
        builder.base, descriptors=(replace(descriptor, operands=(operand,)),)
    )
    aligned = _with_operand_alignment(source)
    assert tuple(
        row.unit_alignment for row in aligned.descriptors[0].operands[0].reg_alts
    ) == (4, 1)


@pytest.mark.parametrize("target", ["cdna2", "cdna3", "cdna4", "gfx9_4_generic"])
@pytest.mark.parametrize(
    ("operation", "names"),
    [("add", ("lhs", "rhs")), ("mul", ("lhs", "rhs")), ("fma", ("a", "b", "c"))],
)
def test_packed_f32_broadcast_alignment_preserves_single_register_reads(
    target, operation, names
) -> None:
    builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    key = f"amdgpu.v_pk_{operation}_f32"
    overlays = {
        row.descriptor_key: row
        for row in builder.overlay_rows()
        if row.descriptor_key.startswith(f"{key}.broadcast_")
    }
    assert len(overlays) == (1 << len(names)) - 1
    for mask in range(1, 1 << len(names)):
        suffix = "_".join(
            name for index, name in enumerate(names) if mask & (1 << index)
        )
        overlay = overlays[f"{key}.broadcast_{suffix}"]
        assert overlay.fixed_encoding_fields == (("OP_SEL_HI", 7 ^ mask),)
        descriptor = _amdgpu_contract_descriptor_from_overlay(overlay)
        assert descriptor.operands[0].unit_count == 2
        for index, name in enumerate(names):
            operand = next(
                operand for operand in descriptor.operands if operand.field_name == name
            )
            assert operand.unit_count == (1 if mask & (1 << index) else 2)
            assert {row.reg_class for row in operand.reg_alts} == {
                _REG_SGPR,
                _REG_VGPR,
            }
            if mask & (1 << index):
                assert all(row.unit_alignment == 2 for row in operand.reg_alts)
