// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_DISPATCH_CONCURRENCY_H_
#define IREE_HAL_DRIVER_AMDGPU_DISPATCH_CONCURRENCY_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/driver/amdgpu/queue_execution_resources.h"
#include "iree/hal/driver/amdgpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable physical-device facts needed for exact dispatch concurrency.
typedef struct iree_hal_amdgpu_dispatch_concurrency_capabilities_t {
  // Class of the physical device target identity.
  iree_hal_amdgpu_target_kind_t target_kind;
  // Parsed gfx IP version of the physical device.
  iree_hal_amdgpu_gfxip_version_t gfxip_version;
  // Maximum resident wave count per compute unit reported by HSA.
  uint32_t maximum_waves_per_compute_unit;
  // SIMD count in each compute unit reported by HSA, or zero if unavailable.
  uint32_t simd_count_per_compute_unit;
  // Compute units reliably available to cooperative dispatches.
  uint32_t cooperative_compute_unit_count;
  // True when |cooperative_compute_unit_count| was derived exactly.
  uint32_t has_cooperative_compute_unit_count : 1;
} iree_hal_amdgpu_dispatch_concurrency_capabilities_t;

// Exact queue and executable facts consumed by the concurrency calculator.
typedef struct iree_hal_amdgpu_dispatch_concurrency_inputs_t {
  // Physical-device concurrency capabilities. Borrowed for the query.
  const iree_hal_amdgpu_dispatch_concurrency_capabilities_t* capabilities;
  // Physical-device execution-resource topology. Borrowed for the query.
  const iree_hal_amdgpu_queue_execution_resource_topology_t*
      execution_resource_topology;
  // Exact achieved execution resources of the queue.
  iree_hal_queue_execution_resource_list_t execution_resources;
  // Exact achieved queue feature bits.
  iree_hal_queue_feature_flags_t queue_features;
  // Exact loaded AMDHSA kernel descriptor. Borrowed for the query.
  const iree_hal_amdgpu_kernel_descriptor_t* kernel_descriptor;
  // Workgroup cluster dimensions declared by executable metadata.
  uint8_t workgroup_cluster_size[3];
  // Maximum additional workgroup-local memory accepted by the function.
  uint32_t maximum_dynamic_workgroup_local_memory_size;
} iree_hal_amdgpu_dispatch_concurrency_inputs_t;

// Calculates exact architectural concurrency for one loaded function on one
// achieved queue realization.
//
// This is a synchronous, allocation-free calculation over immutable state. It
// does not account for live device load or dynamically backed resources such as
// scratch memory. |out_concurrency| is unchanged on failure.
iree_status_t iree_hal_amdgpu_calculate_dispatch_concurrency(
    const iree_hal_amdgpu_dispatch_concurrency_inputs_t* inputs,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_DISPATCH_CONCURRENCY_H_
