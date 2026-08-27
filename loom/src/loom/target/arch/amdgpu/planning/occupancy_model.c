// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/occupancy_model.h"

extern const loom_amdgpu_occupancy_model_t* const
    kLoomAmdgpuOccupancyModelsByProcessor
        [][LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_COUNT];

const loom_amdgpu_occupancy_model_t* loom_amdgpu_occupancy_model_for_properties(
    const loom_amdgpu_processor_properties_t* properties, uint32_t wave_size) {
  IREE_ASSERT(properties != NULL);
  IREE_ASSERT(wave_size == 32 || wave_size == 64);
  const loom_amdgpu_occupancy_wave_slot_t wave_slot =
      wave_size == 32 ? LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_32
                      : LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_64;
  const loom_amdgpu_occupancy_model_t* model =
      kLoomAmdgpuOccupancyModelsByProcessor[properties->occupancy_model_ordinal]
                                           [wave_slot];
  IREE_ASSERT(model != NULL);
  return model;
}
