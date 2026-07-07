# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amdgpu.planning import amdgpu_vopd_component_tables
from loom.target.arch.amdgpu.target_info import sorted_descriptor_set_infos


def test_vopd_descriptor_set_groups_follow_target_info() -> None:
    descriptor_set_infos = sorted_descriptor_set_infos()

    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_RDNA_VOPD,
        descriptor_set_infos,
    ) == (
        "amdgpu.rdna3.core",
        "amdgpu.rdna4.core",
        "amdgpu.rdna4.gfx125x.core",
    )
    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_GFX11_GFX12,
        descriptor_set_infos,
    ) == ("amdgpu.rdna3.core", "amdgpu.rdna4.core")
    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
        descriptor_set_infos,
    ) == ("amdgpu.rdna4.gfx125x.core",)


def test_vopd_component_rows_have_unique_descriptor_keys() -> None:
    components = amdgpu_vopd_component_tables._component_definitions()
    descriptor_keys = [component.descriptor_key for component in components]
    assert len(descriptor_keys) == len(set(descriptor_keys))


def test_vopd_component_rows_have_unique_opcodes() -> None:
    components = amdgpu_vopd_component_tables._component_definitions()
    op_values = [component.op_value for component in components]
    assert len(op_values) == len(set(op_values))


def test_vopd_component_fragment_is_data_only() -> None:
    component = amdgpu_vopd_component_tables._component_definitions()[0]
    rule = amdgpu_vopd_component_tables._VopdComponentRule(
        component=component,
        descriptor_ref="LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAC_F32",
        descriptor_set_keys=("amdgpu.rdna3.core",),
        descriptor_set_mask=("LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3)"),
    )
    fragment = amdgpu_vopd_component_tables._emit_component_rules(amdgpu_vopd_component_tables._VopdComponentTables(rules=(rule,)))

    assert "LOOM_AMDGPU_VOPD_COMPONENT_RULE(" in fragment
    assert "typedef " not in fragment
    assert "struct " not in fragment
    assert "static " not in fragment
