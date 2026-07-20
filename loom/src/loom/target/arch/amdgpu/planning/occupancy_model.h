// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU occupancy model rows consumed by register-pressure diagnostics.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_MODEL_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_MODEL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_occupancy_register_class_model_t {
  // Stable target-low register-class name.
  iree_string_view_t register_class;
  // Descriptor-set-local register class ID for |register_class|.
  uint16_t descriptor_reg_class_id;
  // Occupancy register-file pool shared by resident waves.
  uint32_t pool_units;
  // Allocation granularity used by occupancy calculations.
  uint32_t allocation_granularity;
} loom_amdgpu_occupancy_register_class_model_t;

typedef struct loom_amdgpu_occupancy_model_t {
  // Dense generated AMDGPU descriptor-set ordinal.
  uint16_t descriptor_set_ordinal;
  // AMDGPU wave size used by this model.
  uint32_t wave_size;
  // Maximum resident waves per SIMD.
  uint32_t max_waves_per_simd;
  // Target residency policy shared by scheduling and final occupancy.
  loom_target_residency_model_t residency_model;
  // Register-class occupancy models in diagnostic order.
  const loom_amdgpu_occupancy_register_class_model_t* register_classes;
  // Number of entries in register_classes.
  iree_host_size_t register_class_count;
  // Model index by descriptor-set-local register class ID, or UINT16_MAX for
  // descriptor classes that do not contribute to occupancy.
  const uint16_t* register_class_indices_by_descriptor_reg_class_id;
  // Number of entries in register_class_indices_by_descriptor_reg_class_id.
  iree_host_size_t descriptor_reg_class_count;
} loom_amdgpu_occupancy_model_t;

// Returns the occupancy model for |descriptor_set_ordinal|, or NULL when the
// descriptor set does not define one.
const loom_amdgpu_occupancy_model_t*
loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_MODEL_H_
