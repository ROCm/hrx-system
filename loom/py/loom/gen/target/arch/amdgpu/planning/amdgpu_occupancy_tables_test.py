# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import dataclasses

import pytest

from loom.gen.target.arch.amdgpu.planning import amdgpu_occupancy_tables
from loom.target.arch.amdgpu.target_info import (
    AmdgpuOccupancyModelInfo,
    sorted_occupancy_model_infos,
)

_OCCUPANCY_HEADER = "loom/target/arch/amdgpu/planning/occupancy_model.h"


def test_occupancy_generator_emits_data_source_only() -> None:
    source = amdgpu_occupancy_tables._emit_source(sorted_occupancy_model_infos())

    assert f'#include "{_OCCUPANCY_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "loom_amdgpu_occupancy_model_for_descriptor_set_ordinal" not in source
    assert "loom_low_pressure_cliff_t" in source
    assert "loom_low_pressure_cliff_range_t" in source
    assert "loom_low_pressure_resource_t" in source
    assert "loom_low_pressure_resource_member_t" in source
    assert ".pressure_model =" in source
    assert ".register_class_cliffs =" in source
    assert ".ranges =" in source
    assert "PressureResourceMemberIndicesByRegClass" in source
    assert ".member_indices_by_reg_class =" in source
    assert ".member_ranges_by_reg_class =" in source
    assert "RegisterClassIndexByDescriptorRegClassId" in source
    assert ".register_class_indices_by_descriptor_reg_class_id =" in source
    assert ".descriptor_reg_class_count =" in source
    assert "kLoomAmdgpuOccupancyModels[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT]" in source


def test_occupancy_models_cover_all_descriptor_sets() -> None:
    amdgpu_occupancy_tables._validate_models(sorted_occupancy_model_infos())


def test_occupancy_models_reject_missing_descriptor_set() -> None:
    models = sorted_occupancy_model_infos()[:-1]
    with pytest.raises(ValueError, match="missing descriptor sets"):
        amdgpu_occupancy_tables._validate_models(models)


def test_occupancy_models_reject_missing_base_register_class() -> None:
    models = list(sorted_occupancy_model_infos())
    model = models[0]
    models[0] = dataclasses.replace(
        model,
        register_classes=tuple(row for row in model.register_classes if row.register_class != "amdgpu.vgpr"),
    )
    with pytest.raises(ValueError, match="missing base register classes"):
        amdgpu_occupancy_tables._validate_models(models)


def test_occupancy_models_reject_missing_spillable_register_class() -> None:
    models = list(sorted_occupancy_model_infos())
    model = next(info for info in models if info.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model)
    models[model_index] = AmdgpuOccupancyModelInfo(
        descriptor_set_key=model.descriptor_set_key,
        wave_size=model.wave_size,
        max_waves_per_simd=model.max_waves_per_simd,
        register_classes=tuple(row for row in model.register_classes if row.register_class != "amdgpu.agpr"),
        resources=(),
    )
    with pytest.raises(
        ValueError,
        match=r"missing spillable descriptor register classes: amdgpu\.agpr",
    ):
        amdgpu_occupancy_tables._validate_models(models)


def test_occupancy_models_reject_zero_residency_resource_capacity() -> None:
    models = list(sorted_occupancy_model_infos())
    model = next(info for info in models if info.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model)
    resource = model.resources[0]
    models[model_index] = dataclasses.replace(
        model,
        resources=(dataclasses.replace(resource, pool_units=511),),
    )

    with pytest.raises(ValueError, match="zero residency"):
        amdgpu_occupancy_tables._validate_models(models)


def test_occupancy_models_reject_zero_residency_register_capacity() -> None:
    models = list(sorted_occupancy_model_infos())
    model = next(info for info in models if info.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model)
    register_classes = tuple(dataclasses.replace(row, pool_units=255) if row.register_class == "amdgpu.vgpr" else row for row in model.register_classes)
    models[model_index] = dataclasses.replace(
        model,
        register_classes=register_classes,
    )

    with pytest.raises(ValueError, match="zero residency"):
        amdgpu_occupancy_tables._validate_models(models)


def test_occupancy_models_reject_unrepresentable_terminal_cliff() -> None:
    models = list(sorted_occupancy_model_infos())
    model = models[0]
    register_class = model.register_classes[0]
    models[0] = dataclasses.replace(
        model,
        register_classes=(
            dataclasses.replace(
                register_class,
                pool_units=0xFFFFFFFF,
                allocation_granularity=1,
            ),
            *model.register_classes[1:],
        ),
    )

    with pytest.raises(ValueError, match="pressure cliff does not fit uint32"):
        amdgpu_occupancy_tables._validate_models(models)


def test_pressure_cliffs_jump_directly_between_reachable_tiers() -> None:
    cliffs = amdgpu_occupancy_tables._pressure_cliffs(
        pool_units=800,
        allocation_granularity=16,
        max_waves_per_simd=16,
    )

    assert cliffs[0] == (49, 16, 12)
    assert cliffs[-1] == (801, 1, 0)
    assert all(cliff_units > 0 and tier_before > tier_after for cliff_units, tier_before, tier_after in cliffs)


def test_pressure_cliffs_match_exhaustive_current_models() -> None:
    for model in sorted_occupancy_model_infos():
        for source in (*model.register_classes, *model.resources):
            expected: list[tuple[int, int, int]] = []
            previous_wave_limit = model.max_waves_per_simd
            stop_candidate = source.pool_units + source.allocation_granularity
            for candidate in range(1, stop_candidate + 1):
                wave_limit = amdgpu_occupancy_tables._wave_limit(
                    source.pool_units,
                    source.allocation_granularity,
                    model.max_waves_per_simd,
                    candidate,
                )
                if wave_limit < previous_wave_limit:
                    expected.append((candidate, previous_wave_limit, wave_limit))
                    previous_wave_limit = wave_limit
                if wave_limit == 0:
                    break

            assert amdgpu_occupancy_tables._pressure_cliffs(
                source.pool_units,
                source.allocation_granularity,
                model.max_waves_per_simd,
            ) == tuple(expected)
