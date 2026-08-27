// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/drivers/vulkan/device_plan.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters for creating a Vulkan HAL device spec.
typedef struct iree_hal_vulkan_device_spec_params_t {
  // Stable logical device identifier.
  iree_string_view_t logical_device_id;
  // Human-readable logical device name.
  iree_string_view_t display_name;
  // Immutable physical-device snapshot selected for this logical device.
  const iree_hal_vulkan_physical_device_snapshot_t* physical_device;
  // Planned logical-device features, extensions, and queue assignment.
  const iree_hal_vulkan_device_plan_t* device_plan;
  // Device allocator used to query stable allocation classes.
  iree_hal_allocator_t* device_allocator;
} iree_hal_vulkan_device_spec_params_t;

// Encodes |spec| and adds it as a Vulkan driver-local facet to |builder|.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_vulkan_device_spec_t* spec,
    iree_host_size_t cooperative_matrix_property_count,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties);

// Creates an immutable spec for a Vulkan HAL device.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_create(
    const iree_hal_vulkan_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_
