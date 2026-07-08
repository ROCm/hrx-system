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
    assert "loom_amdgpu_occupancy_pressure_cliff_model_t" in source
    assert ".pressure_cliffs =" in source
    assert ".pressure_cliff_count =" in source
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
