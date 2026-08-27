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
  // Whether this register class directly limits resident waves.
  bool limits_occupancy;
} loom_amdgpu_occupancy_register_class_model_t;

typedef enum loom_amdgpu_occupancy_wave_slot_e {
  // Generated occupancy model slot for wave32 execution.
  LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_32 = 0,
  // Generated occupancy model slot for wave64 execution.
  LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_64 = 1,
  // Number of generated wave-mode slots per processor.
  LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_COUNT = 2,
} loom_amdgpu_occupancy_wave_slot_t;

// Hardware resources shared by waves in one occupancy calculation domain.
//
// On current AMDGPU processors the domain is either a CU or a WGP. Naming it
// by its scheduling role keeps the units stable across both hardware modes.
typedef struct loom_amdgpu_occupancy_domain_model_t {
  // Number of SIMD execution units in the occupancy domain.
  uint32_t simd_count;
  // Local-memory bytes shared by workgroups in the occupancy domain.
  uint32_t local_memory_bytes;
  // Local-memory allocation granularity in bytes per workgroup.
  uint32_t local_memory_allocation_granularity;
  // Barrier-using workgroups available in the occupancy domain.
  uint32_t max_barrier_workgroup_count;
} loom_amdgpu_occupancy_domain_model_t;

typedef struct loom_amdgpu_occupancy_model_t {
  // Dense generated AMDGPU descriptor-set ordinal.
  uint16_t descriptor_set_ordinal;
  // AMDGPU wave size used by this model.
  uint32_t wave_size;
  // Maximum resident waves per SIMD.
  uint32_t max_waves_per_simd;
  // Workgroup and local-memory resources in one occupancy domain.
  loom_amdgpu_occupancy_domain_model_t domain;
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

// Returns the generated occupancy model for |properties| and |wave_size|.
//
// The target properties and wave mode must have passed target verification.
const loom_amdgpu_occupancy_model_t* loom_amdgpu_occupancy_model_for_properties(
    const loom_amdgpu_processor_properties_t* properties, uint32_t wave_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_MODEL_H_
