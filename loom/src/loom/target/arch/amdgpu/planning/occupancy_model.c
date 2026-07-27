// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/occupancy_model.h"

#include <stddef.h>

#include "loom/target/arch/amdgpu/target_info.h"

extern const loom_amdgpu_occupancy_model_t* const
    kLoomAmdgpuOccupancyModels[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT];

const loom_amdgpu_occupancy_model_t*
loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal) {
  if (descriptor_set_ordinal >= LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT) {
    return NULL;
  }
  return kLoomAmdgpuOccupancyModels[descriptor_set_ordinal];
}
