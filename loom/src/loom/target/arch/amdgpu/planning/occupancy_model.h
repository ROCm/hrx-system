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
#include "loom/codegen/low/pressure.h"
#include "loom/target/arch/amdgpu/target_info.h"

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

typedef struct loom_amdgpu_occupancy_resource_member_model_t {
  // Index into loom_amdgpu_occupancy_model_t::register_classes.
  uint16_t register_class_index;
  // Member contribution granularity applied before summing pressure.
  uint32_t contribution_granularity;
} loom_amdgpu_occupancy_resource_member_model_t;

typedef struct loom_amdgpu_occupancy_resource_model_t {
  // Stable target-low resource name.
  iree_string_view_t resource;
  // Occupancy resource pool shared by resident waves.
  uint32_t pool_units;
  // Allocation granularity used by occupancy calculations.
  uint32_t allocation_granularity;
  // Register-class members contributing to this resource.
  const loom_amdgpu_occupancy_resource_member_model_t* members;
  // Number of entries in members.
  iree_host_size_t member_count;
} loom_amdgpu_occupancy_resource_model_t;

typedef struct loom_amdgpu_occupancy_model_t {
  // Dense generated AMDGPU descriptor-set ordinal.
  uint16_t descriptor_set_ordinal;
  // AMDGPU wave size used by this model.
  uint32_t wave_size;
  // Maximum resident waves per SIMD.
  uint32_t max_waves_per_simd;
  // Target throughput cliffs shared by scheduling and allocation.
  loom_low_pressure_cliff_table_t pressure_cliffs;
  // Register-class occupancy models in diagnostic order.
  const loom_amdgpu_occupancy_register_class_model_t* register_classes;
  // Number of entries in register_classes.
  iree_host_size_t register_class_count;
  // Model index by descriptor-set-local register class ID, or UINT16_MAX for
  // descriptor classes that do not contribute to occupancy.
  const uint16_t* register_class_indices_by_descriptor_reg_class_id;
  // Number of entries in register_class_indices_by_descriptor_reg_class_id.
  iree_host_size_t descriptor_reg_class_count;
  // Derived occupancy resources in diagnostic order.
  const loom_amdgpu_occupancy_resource_model_t* resources;
  // Number of entries in resources.
  iree_host_size_t resource_count;
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
