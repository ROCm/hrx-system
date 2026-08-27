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
    sorted_descriptor_set_infos,
    sorted_processor_infos,
)

_OCCUPANCY_HEADER = "loom/target/arch/amdgpu/planning/occupancy_model.h"


def _current_models() -> tuple[amdgpu_occupancy_tables._AmdgpuOccupancyModelRow, ...]:
    return amdgpu_occupancy_tables._materialize_models(sorted_processor_infos())


def _current_descriptor_schemas_by_key() -> dict[str, amdgpu_occupancy_tables._AmdgpuOccupancyDescriptorSchema]:
    return amdgpu_occupancy_tables._descriptor_schemas_by_key(
        sorted_descriptor_set_infos(),
    )


def _replace_row_model(
    row: amdgpu_occupancy_tables._AmdgpuOccupancyModelRow,
    model: AmdgpuOccupancyModelInfo,
) -> amdgpu_occupancy_tables._AmdgpuOccupancyModelRow:
    processors = tuple(
        dataclasses.replace(
            processor,
            occupancy=dataclasses.replace(
                processor.occupancy,
                wave32=model if row.wave_size == 32 else processor.occupancy.wave32,
                wave64=model if row.wave_size == 64 else processor.occupancy.wave64,
            ),
        )
        for processor in row.processors
    )
    return dataclasses.replace(row, model=model, processors=processors)


def test_occupancy_generator_emits_data_source_only() -> None:
    descriptor_sets = sorted_descriptor_set_infos()
    processors = sorted_processor_infos()
    source = amdgpu_occupancy_tables.generate_occupancy_tables(
        descriptor_sets,
        processors,
    )

    assert f'#include "{_OCCUPANCY_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "loom_amdgpu_occupancy_model_for_descriptor_set_ordinal" not in source
    assert "loom_target_residency_cliff_t" in source
    assert "loom_target_residency_cliff_range_t" in source
    assert "loom_target_residency_derived_resource_t" in source
    assert "loom_target_residency_derived_member_t" in source
    assert ".residency_model =" in source
    assert ".best_tier =" in source
    assert ".direct_resources =" in source
    assert ".names =" in source
    assert ".cliff_ranges =" in source
    assert ".derived_resources =" in source
    assert ".member_count =" in source
    assert ".cliff_count =" in source
    assert "PressureResourceMemberIndicesByRegClass" in source
    assert ".member_indices_by_direct_resource =" in source
    assert ".member_ranges_by_direct_resource =" in source
    assert "RegisterClassIndexByDescriptorRegClassId" in source
    assert ".register_class_indices_by_descriptor_reg_class_id =" in source
    assert ".descriptor_reg_class_count =" in source
    assert "kLoomAmdgpuOccupancyModelsByProcessor" in source
    assert "LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_32" in source
    assert "LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_64" in source


def test_occupancy_models_cover_all_processor_wave_modes() -> None:
    processors = sorted_processor_infos()
    amdgpu_occupancy_tables._validate_models(
        amdgpu_occupancy_tables._materialize_models(processors),
        processors,
        _current_descriptor_schemas_by_key(),
    )


def test_occupancy_models_reject_missing_processor_wave_mode() -> None:
    processors = sorted_processor_infos()
    models = _current_models()[:-1]
    with pytest.raises(ValueError, match="do not cover processor wave modes"):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


def test_occupancy_models_reject_missing_base_register_class() -> None:
    processors = sorted_processor_infos()
    models = list(_current_models())
    model = models[0].model
    models[0] = _replace_row_model(
        models[0],
        dataclasses.replace(
            model,
            register_classes=tuple(row for row in model.register_classes if row.register_class != "amdgpu.vgpr"),
        ),
    )
    with pytest.raises(ValueError, match="missing base register classes"):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


def test_occupancy_models_reject_missing_spillable_register_class() -> None:
    processors = sorted_processor_infos()
    models = list(_current_models())
    model_row = next(row for row in models if row.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model_row)
    model = model_row.model
    replacement = AmdgpuOccupancyModelInfo(
        max_waves_per_simd=model.max_waves_per_simd,
        domain=model.domain,
        register_classes=tuple(row for row in model.register_classes if row.register_class != "amdgpu.agpr"),
        resources=(),
    )
    models[model_index] = _replace_row_model(model_row, replacement)
    with pytest.raises(
        ValueError,
        match=r"missing spillable descriptor register classes: amdgpu\.agpr",
    ):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


def test_occupancy_models_reject_zero_residency_resource_capacity() -> None:
    processors = sorted_processor_infos()
    models = list(_current_models())
    model_row = next(row for row in models if row.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model_row)
    model = model_row.model
    resource = model.resources[0]
    models[model_index] = _replace_row_model(
        model_row,
        dataclasses.replace(
            model,
            resources=(dataclasses.replace(resource, pool_units=511),),
        ),
    )

    with pytest.raises(ValueError, match="zero residency"):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


def test_occupancy_models_reject_zero_residency_register_capacity() -> None:
    processors = sorted_processor_infos()
    models = list(_current_models())
    model_row = next(row for row in models if row.descriptor_set_key == "amdgpu.cdna3.core")
    model_index = models.index(model_row)
    model = model_row.model
    register_classes = tuple(dataclasses.replace(row, pool_units=255) if row.register_class == "amdgpu.vgpr" else row for row in model.register_classes)
    models[model_index] = _replace_row_model(
        model_row,
        dataclasses.replace(model, register_classes=register_classes),
    )

    with pytest.raises(ValueError, match="zero residency"):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


def test_occupancy_models_reject_unrepresentable_terminal_cliff() -> None:
    processors = sorted_processor_infos()
    models = list(_current_models())
    model_row = models[0]
    model = model_row.model
    register_class = model.register_classes[0]
    models[0] = _replace_row_model(
        model_row,
        dataclasses.replace(
            model,
            register_classes=(
                dataclasses.replace(
                    register_class,
                    pool_units=0xFFFFFFFF,
                    allocation_granularity=1,
                    limits_occupancy=True,
                ),
                *model.register_classes[1:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="pressure cliff does not fit uint32"):
        amdgpu_occupancy_tables._validate_models(
            models,
            processors,
            _current_descriptor_schemas_by_key(),
        )


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
    for model_row in _current_models():
        model = model_row.model
        sources = (
            *(row for row in model.register_classes if row.limits_occupancy),
            *model.resources,
        )
        for source in sources:
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
