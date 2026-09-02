// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/drivers/vulkan/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID "iree.hal.drivers.vulkan.device"
#define IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION 4u

// Stable Vulkan device spec flags.
typedef uint32_t iree_hal_vulkan_device_spec_flags_t;
typedef enum iree_hal_vulkan_device_spec_flag_bits_e {
  // No Vulkan device spec flags are present.
  IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE = 0u,
} iree_hal_vulkan_device_spec_flag_bits_t;

// Borrowed view of encoded Vulkan cooperative matrix property rows.
typedef struct iree_hal_vulkan_cooperative_matrix_property_table_t {
  // Number of encoded property rows.
  iree_host_size_t count;
  // Canonical little-endian property row data borrowed from the facet payload.
  iree_const_byte_span_t encoded_data;
} iree_hal_vulkan_cooperative_matrix_property_table_t;

// Vulkan device facts decoded from a driver-local facet.
typedef struct iree_hal_vulkan_device_spec_t {
  // Vulkan API version supported by the physical device.
  uint32_t api_version;
  // Vulkan driver version reported by the physical device.
  uint32_t driver_version;
  // VkPhysicalDeviceType value reported by the physical device.
  uint32_t physical_device_type;
  // IREE Vulkan features enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;
  // Stable Vulkan device spec flags.
  iree_hal_vulkan_device_spec_flags_t flags;
  // Cooperative matrix properties enabled on the logical device.
  iree_hal_vulkan_cooperative_matrix_property_table_t cooperative_matrix;
} iree_hal_vulkan_device_spec_t;

// Calculates the canonical payload size for |property_count| rows.
IREE_API_EXPORT iree_status_t
iree_hal_vulkan_device_spec_calculate_payload_size(
    iree_host_size_t property_count, iree_host_size_t* out_payload_size);

// Encodes |spec| and cooperative matrix |properties| into |payload|.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_encode(
    const iree_hal_vulkan_device_spec_t* spec, iree_host_size_t property_count,
    const iree_hal_vulkan_cooperative_matrix_property_t* properties,
    iree_byte_span_t payload);

// Decodes the fixed-width little-endian |payload| into |out_spec| after
// validating its schema envelope.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_decode(
    iree_const_byte_span_t payload, iree_hal_vulkan_device_spec_t* out_spec);

// Reads the cooperative matrix property at |ordinal| into |out_property|.
//
// Returns false when |ordinal| is out of range.
IREE_API_EXPORT bool
iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
    const iree_hal_vulkan_device_spec_t* spec, iree_host_size_t ordinal,
    iree_hal_vulkan_cooperative_matrix_property_t* out_property);

// Finds the Vulkan device spec facet in |device_spec| or NULL.
IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_vulkan_device_spec_find_facet(
    const iree_hal_device_spec_t* device_spec);

// Decodes a Vulkan device spec |facet| into |out_spec|.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_decode_facet(
    const iree_hal_device_spec_facet_t* facet,
    iree_hal_vulkan_device_spec_t* out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_H_
