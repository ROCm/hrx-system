# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.descriptors.alu import (
    _s_and_b64_overlay,
    _s_or_b64_overlay,
    _v_cmp_overlays,
    _v_mov_b32_copy_overlay,
    _v_readfirstlane_b32_overlay,
    _v_readlane_b32_src1_inline_overlay,
)
from loom.target.arch.amdgpu.descriptors.common import _REG_VGPR
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.arch.amdgpu.descriptors.memory import _s_load_dword_overlay
from loom.target.arch.amdgpu.descriptors.rdna3 import (
    _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.timing import (
    _with_lds_service_timing,
    _with_valu_sgpr_timing,
)
from loom.target.arch.amdgpu.descriptors.workgroup import (
    _ds_atomic_overlays,
    _ds_bpermute_b32_overlay,
    _ds_read2_overlay,
    _ds_read_narrow_overlay,
    _ds_read_overlay,
    _ds_write_overlay,
)
from loom.target.low_descriptors import (
    EventSeparation,
    IssueUse,
    ModelQuality,
    RegClassAlt,
)


def test_lds_packets_share_width_dependent_service() -> None:
    rows = (
        (_ds_read_narrow_overlay(width_bits=16), 1),
        (_ds_read_overlay(width_bits=32, units=1), 1),
        (_ds_read_overlay(width_bits=64, units=2), 2),
        (_ds_read_overlay(width_bits=128, units=4), 4),
        (_ds_read2_overlay(element_width_bits=64, value_units=2), 4),
        (_ds_write_overlay(width_bits=32, units=1), 1),
        (_ds_write_overlay(width_bits=128, units=4), 4),
        (_ds_bpermute_b32_overlay(), 1),
        *(
            (overlay, 2 if overlay.descriptor_key.endswith("b64") else 1)
            for overlay in _ds_atomic_overlays(
                cmpxchg_expected_field="DATA0", cmpxchg_replacement_field="DATA1"
            )
        ),
    )
    descriptors = tuple(
        _amdgpu_contract_descriptor_from_overlay(overlay) for overlay, _ in rows
    )
    original = replace(_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=descriptors)
    timed = _with_lds_service_timing(original, 32)
    original_classes = {row.name: row for row in original.schedule_classes}
    timed_classes = {row.name: row for row in timed.schedule_classes}
    resources = {row.name: row for row in timed.resources}
    assert resources["amdgpu.lds.service"].capacity_per_cycle == 1
    for before, after, (_, cycles) in zip(
        descriptors, timed.descriptors, rows, strict=True
    ):
        parent = original_classes[before.schedule_class]
        variant = timed_classes[after.schedule_class]
        assert variant.issue_uses == (
            *parent.issue_uses,
            IssueUse("amdgpu.lds.service", cycles=cycles, units=1),
        )
        # Bandwidth does not change completion, hazards, or packet semantics.
        assert (
            replace(variant, name=parent.name, issue_uses=parent.issue_uses) == parent
        )
        assert replace(after, schedule_class=before.schedule_class) == before
    # Both encodings transfer four dwords per lane and use the same class.
    assert timed.descriptors[3].schedule_class == timed.descriptors[4].schedule_class


def test_non_lds_packets_keep_their_schedule_classes() -> None:
    descriptors = tuple(
        _amdgpu_contract_descriptor_from_overlay(overlay)
        for overlay in (_v_mov_b32_copy_overlay(), _s_load_dword_overlay())
    )
    original = replace(_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=descriptors)
    timed = _with_lds_service_timing(original, 32)
    assert timed.descriptors == original.descriptors
    assert timed.schedule_classes == original.schedule_classes


def test_compare_and_lane_results_have_scalar_read_separation() -> None:
    producer_overlays = (
        *_v_cmp_overlays(),
        _v_readfirstlane_b32_overlay(),
        _v_readlane_b32_src1_inline_overlay(),
    )
    descriptors = tuple(
        _amdgpu_contract_descriptor_from_overlay(overlay)
        for overlay in (
            *producer_overlays,
            _s_and_b64_overlay(),
            _s_or_b64_overlay(),
        )
    )
    original = replace(_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=descriptors)
    timed = _with_valu_sgpr_timing(original, 5)
    assert timed.event_separations == (
        EventSeparation(
            "amdgpu.valu.sgpr.write", "amdgpu.sgpr.read", 5, ModelQuality.ESTIMATED
        ),
    )
    # Dependency readiness does not change class priority or completion latency.
    assert timed.schedule_classes == original.schedule_classes
    for descriptor in timed.descriptors[: len(producer_overlays)]:
        assert descriptor.operands[0].write_event == "amdgpu.valu.sgpr.write"
        assert descriptor.operands[0].read_event is None
        assert all(operand.write_event is None for operand in descriptor.operands[1:])
    for descriptor in timed.descriptors[len(producer_overlays) :]:
        assert descriptor.operands[0].write_event is None
        assert descriptor.operands[1].read_event == "amdgpu.sgpr.read"
        assert descriptor.operands[2].read_event == "amdgpu.sgpr.read"


def test_valu_scalar_results_require_unambiguous_register_classes() -> None:
    descriptor = _amdgpu_contract_descriptor_from_overlay(
        _v_readfirstlane_b32_overlay()
    )
    result = descriptor.operands[0]
    descriptor = replace(
        descriptor,
        operands=(
            replace(result, reg_alts=(*result.reg_alts, RegClassAlt(_REG_VGPR))),
            *descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=(descriptor,)
    )
    with pytest.raises(ValueError, match="requires distinct SGPR and VGPR forms"):
        _with_valu_sgpr_timing(descriptor_set, 5)


def test_other_producers_preserve_their_result_timing() -> None:
    descriptors = tuple(
        _amdgpu_contract_descriptor_from_overlay(overlay)
        for overlay in (_v_mov_b32_copy_overlay(), _s_load_dword_overlay())
    )
    original = replace(_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=descriptors)
    timed = _with_valu_sgpr_timing(original, 5)
    vector_move, scalar_load = timed.descriptors
    assert vector_move.operands[0].write_event is None
    assert scalar_load.operands[0].write_event is None
    # The selected producer fixes the class of a mixed SGPR/VGPR source.
    assert vector_move.operands[1].read_event == "amdgpu.sgpr.read"
    assert scalar_load.operands[1].read_event == "amdgpu.sgpr.read"
