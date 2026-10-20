# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.gen.target.arch.amdgpu.descriptors import amdgpu_vopd_component_tables
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_INTERLOCK_HAZARD,
    sorted_descriptor_set_infos,
)

# Each row identifies the encoded register fields read through SRC0/SRC1/SRC2.
# A literal does not consume a VGPR cache port. A tied accumulator reads the
# destination through SRC2, even though it has no encoded source field.
_CACHE_OPERANDS = {
    "amdgpu.v_fmac_f32": (("operand", 1), ("operand", 2), ("result", 0)),
    "amdgpu.v_fmaak_f32": (("operand", 0), ("operand", 1), None),
    "amdgpu.v_fmamk_f32": (("operand", 0), None, ("operand", 1)),
    "amdgpu.v_dot2_f32_f16": (("operand", 0), ("operand", 1), ("result", 0)),
    "amdgpu.v_dot2_f32_bf16": (("operand", 0), ("operand", 1), ("result", 0)),
    "amdgpu.v_sub_f32": (("operand", 0), ("operand", 1), None),
    "amdgpu.v_mov_b32_copy": (("operand", 0), None, None),
}


@pytest.mark.parametrize("first_key", _CACHE_OPERANDS)
@pytest.mark.parametrize("second_key", _CACHE_OPERANDS)
@pytest.mark.parametrize("source_mask", [1, 3])
def test_vopd_placement_matches_read_ports(first_key: str, second_key: str, source_mask: int) -> None:
    tables = amdgpu_vopd_component_tables
    components = {component.descriptor_key: component for component in tables._component_definitions()}
    flags = frozenset({tables._COMPONENT_FLAG_SOURCE_PARITY}) if source_mask == 1 else frozenset()
    recipe = tables._pair_placement_recipe(components[first_key], components[second_key], flags, flags)
    assert len(recipe.alternatives) == 1

    def location(ref):
        kind = "result" if ref.kind == tables._PLACEMENT_VALUE_RESULT else "operand"
        return kind, ref.index

    actual = {(location(relation.result), location(relation.source), relation.location_mask) for relation in recipe.alternatives[0] if relation.kind == tables._PLACEMENT_DIFFERENT_MASKED}
    expected = {(("result", 0), ("result", 0), 1)}
    for first, second, mask in zip(_CACHE_OPERANDS[first_key], _CACHE_OPERANDS[second_key], (source_mask, source_mask, 1), strict=True):
        if first is not None and second is not None:
            expected.add((first, second, mask))
    assert actual == expected


def test_vopd_interlock_profiles() -> None:
    assert {info.key for info in sorted_descriptor_set_infos() if info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_INTERLOCK_HAZARD} == {
        "amdgpu.gfx11.generic.core",
        "amdgpu.rdna3.core",
        "amdgpu.rdna3_5.core",
    }


def test_vopd_descriptor_set_groups_follow_target_info() -> None:
    descriptor_set_infos = sorted_descriptor_set_infos()

    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_RDNA_VOPD,
        descriptor_set_infos,
    ) == (
        "amdgpu.gfx11.generic.core",
        "amdgpu.gfx12.generic.core",
        "amdgpu.gfx12_5.generic.core",
        "amdgpu.rdna3.core",
        "amdgpu.rdna3_5.core",
        "amdgpu.rdna4.core",
        "amdgpu.rdna4.gfx1250_a0.core",
        "amdgpu.rdna4.gfx1251.core",
        "amdgpu.rdna4.gfx125x.core",
        "amdgpu.rdna4m.core",
    )
    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_GFX11_GFX12,
        descriptor_set_infos,
    ) == (
        "amdgpu.gfx11.generic.core",
        "amdgpu.gfx12.generic.core",
        "amdgpu.rdna3.core",
        "amdgpu.rdna3_5.core",
        "amdgpu.rdna4.core",
        "amdgpu.rdna4m.core",
    )
    assert amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
        amdgpu_vopd_component_tables._DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
        descriptor_set_infos,
    ) == (
        "amdgpu.gfx12_5.generic.core",
        "amdgpu.rdna4.gfx1250_a0.core",
        "amdgpu.rdna4.gfx1251.core",
        "amdgpu.rdna4.gfx125x.core",
    )


def test_vopd_component_rows_have_unique_descriptor_keys() -> None:
    components = amdgpu_vopd_component_tables._component_definitions()
    descriptor_keys = [component.descriptor_key for component in components]
    assert len(descriptor_keys) == len(set(descriptor_keys))


def test_packed_float_dot_components_exclude_gfx125x_profiles() -> None:
    descriptor_set_infos = sorted_descriptor_set_infos()
    components = {component.descriptor_key: component for component in amdgpu_vopd_component_tables._component_definitions() if component.descriptor_key.startswith("amdgpu.v_dot")}
    assert set(components) == {
        "amdgpu.v_dot2_f32_f16",
        "amdgpu.v_dot2_f32_bf16",
    }
    expected_descriptor_sets = (
        "amdgpu.gfx11.generic.core",
        "amdgpu.gfx12.generic.core",
        "amdgpu.rdna3.core",
        "amdgpu.rdna3_5.core",
        "amdgpu.rdna4.core",
        "amdgpu.rdna4m.core",
    )
    for component in components.values():
        assert (
            amdgpu_vopd_component_tables._descriptor_set_keys_for_group(
                component.descriptor_set_group,
                descriptor_set_infos,
            )
            == expected_descriptor_sets
        )


def test_vopd_component_source_forms_share_canonical_native_info() -> None:
    components = amdgpu_vopd_component_tables._component_definitions()
    move_components = tuple(component for component in components if component.op_value == 8)
    assert tuple(component.descriptor_key for component in move_components) == (
        "amdgpu.v_mov_b32",
        "amdgpu.v_mov_b32_copy",
    )
    assert len({amdgpu_vopd_component_tables._component_info_key(component) for component in move_components}) == 1


def test_vopd_descriptor_lookup_rows_use_descriptor_order() -> None:
    component = amdgpu_vopd_component_tables._component_definitions()[0]
    rule = amdgpu_vopd_component_tables._VopdComponentRule(
        component=component,
        descriptor_set_keys=("amdgpu.rdna3.core",),
    )

    lookup = amdgpu_vopd_component_tables._descriptor_lookup_for_set(
        "amdgpu.rdna3.core",
        ("amdgpu.s_nop", component.descriptor_key),
        (rule,),
    )

    assert lookup.rows == (0, 1)
    assert lookup.pair_affinity_candidate_ordinals == (1,)


def test_vopd_pair_affinities_use_descriptor_ordinals_and_priority() -> None:
    components = amdgpu_vopd_component_tables._component_definitions()
    same_opcode_component = components[0]
    mixed_second_component = next(component for component in components if component.op_name == "add_u32")
    rules = (
        amdgpu_vopd_component_tables._VopdComponentRule(
            component=same_opcode_component,
            descriptor_set_keys=("amdgpu.rdna3.core",),
        ),
        amdgpu_vopd_component_tables._VopdComponentRule(
            component=mixed_second_component,
            descriptor_set_keys=("amdgpu.rdna3.core",),
        ),
    )

    lookup = amdgpu_vopd_component_tables._descriptor_lookup_for_set(
        "amdgpu.rdna3.core",
        tuple(rule.component.descriptor_key for rule in rules),
        rules,
    )
    placement_recipe_catalog = amdgpu_vopd_component_tables._VopdPairPlacementRecipeCatalog(
        recipes=[],
        indices_by_recipe={},
        indices_by_rule_pair={},
    )
    rows = amdgpu_vopd_component_tables._pair_affinity_rows_for_set(
        rules,
        lookup,
        placement_recipe_catalog,
    )

    assert (
        amdgpu_vopd_component_tables._VopdPairAffinityRow(
            first_descriptor_ordinal=0,
            second_descriptor_ordinal=0,
            priority=2,
            placement_recipe_index=0,
        )
        in rows
    )
    assert (
        amdgpu_vopd_component_tables._VopdPairAffinityRow(
            first_descriptor_ordinal=1,
            second_descriptor_ordinal=0,
            priority=1,
            placement_recipe_index=0,
        )
        not in rows
    )
    assert (
        amdgpu_vopd_component_tables._VopdPairAffinityRow(
            first_descriptor_ordinal=0,
            second_descriptor_ordinal=1,
            priority=1,
            placement_recipe_index=1,
        )
        in rows
    )


def test_vopd_pair_placement_guides_one_commutable_orientation() -> None:
    component = next(component for component in amdgpu_vopd_component_tables._component_definitions() if component.op_name == "add_u32")

    recipe = amdgpu_vopd_component_tables._pair_placement_recipe(
        component,
        component,
        frozenset({amdgpu_vopd_component_tables._COMPONENT_FLAG_COMMUTABLE_SOURCES}),
        frozenset({amdgpu_vopd_component_tables._COMPONENT_FLAG_COMMUTABLE_SOURCES}),
    )

    assert len(recipe.alternatives) == 2
    source_bank_relations = tuple(relation for relation in recipe.alternatives[1] if relation.kind == amdgpu_vopd_component_tables._PLACEMENT_DIFFERENT_MASKED and relation.location_mask == 0x3)
    assert tuple(
        (
            relation.result.component,
            relation.result.index,
            relation.source.component,
            relation.source.index,
        )
        for relation in source_bank_relations
    ) == (
        (
            amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_FIRST,
            1,
            amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_SECOND,
            0,
        ),
        (
            amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_FIRST,
            0,
            amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_SECOND,
            1,
        ),
    )


def test_vopd_polymorphic_move_is_not_a_descriptor_affinity() -> None:
    component = next(component for component in amdgpu_vopd_component_tables._component_definitions() if component.descriptor_key == "amdgpu.v_mov_b32_copy")
    rule = amdgpu_vopd_component_tables._VopdComponentRule(
        component=component,
        descriptor_set_keys=("amdgpu.rdna3.core",),
    )
    lookup = amdgpu_vopd_component_tables._descriptor_lookup_for_set(
        "amdgpu.rdna3.core",
        (component.descriptor_key,),
        (rule,),
    )
    placement_recipe_catalog = amdgpu_vopd_component_tables._VopdPairPlacementRecipeCatalog(
        recipes=[],
        indices_by_recipe={},
        indices_by_rule_pair={},
    )

    rows = amdgpu_vopd_component_tables._pair_affinity_rows_for_set(
        (rule,),
        lookup,
        placement_recipe_catalog,
    )

    assert rows == ()
    assert placement_recipe_catalog.recipes == []


def test_vopd_pair_placement_rejects_component_value_out_of_bounds() -> None:
    component = amdgpu_vopd_component_tables._component_definitions()[0]
    invalid_ref = amdgpu_vopd_component_tables._VopdPairPlacementValueRef(
        component=amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_FIRST,
        kind=amdgpu_vopd_component_tables._PLACEMENT_VALUE_RESULT,
        index=1,
    )

    with pytest.raises(ValueError, match="outside count 1"):
        amdgpu_vopd_component_tables._validate_pair_placement_ref(
            "test recipe",
            "result",
            invalid_ref,
            component,
            component,
        )


def test_vopd_descriptor_lookup_validation_rejects_out_of_bounds_rule() -> None:
    tables = amdgpu_vopd_component_tables._VopdComponentTables(
        rules=(),
        descriptor_lookup_ranges=(
            amdgpu_vopd_component_tables._VopdComponentDescriptorLookupRange(
                descriptor_set_key="amdgpu.rdna3.core",
                descriptor_set_ordinal=0,
                first_descriptor_lookup=0,
                descriptor_lookup_count=1,
            ),
        ),
        descriptor_lookup_rows=(1,),
        pair_affinity_ranges=(),
        pair_affinity_rows=(),
        pair_placement_recipes=(),
    )

    with pytest.raises(ValueError, match="out-of-bounds rule"):
        amdgpu_vopd_component_tables._validate_vopd_component_tables(tables)


def test_vopd_pair_affinity_validation_rejects_out_of_bounds_descriptor() -> None:
    tables = amdgpu_vopd_component_tables._VopdComponentTables(
        rules=(),
        descriptor_lookup_ranges=(
            amdgpu_vopd_component_tables._VopdComponentDescriptorLookupRange(
                descriptor_set_key="amdgpu.rdna3.core",
                descriptor_set_ordinal=0,
                first_descriptor_lookup=0,
                descriptor_lookup_count=1,
            ),
        ),
        descriptor_lookup_rows=(0,),
        pair_placement_recipes=(
            amdgpu_vopd_component_tables._VopdPairPlacementRecipe(
                alternatives=(
                    (
                        amdgpu_vopd_component_tables._VopdPairPlacementRelation(
                            result=amdgpu_vopd_component_tables._placement_result_ref(amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_FIRST),
                            source=amdgpu_vopd_component_tables._placement_result_ref(amdgpu_vopd_component_tables._PLACEMENT_COMPONENT_SECOND),
                            kind=amdgpu_vopd_component_tables._PLACEMENT_DIFFERENT_MASKED,
                            location_mask=1,
                        ),
                    ),
                )
            ),
        ),
        pair_affinity_ranges=(
            amdgpu_vopd_component_tables._VopdPairAffinityRange(
                descriptor_set_key="amdgpu.rdna3.core",
                descriptor_set_ordinal=0,
                first_pair_affinity=0,
                pair_affinity_count=1,
            ),
        ),
        pair_affinity_rows=(
            amdgpu_vopd_component_tables._VopdPairAffinityRow(
                first_descriptor_ordinal=1,
                second_descriptor_ordinal=0,
                priority=1,
                placement_recipe_index=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="out-of-bounds descriptor ordinal"):
        amdgpu_vopd_component_tables._validate_vopd_component_tables(tables)
