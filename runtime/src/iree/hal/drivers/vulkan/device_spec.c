// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec.h"

#include "iree/base/alignment.h"

#define IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_MAGIC UINT32_C(0x53445656)  // VVDS
#define IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_SIZE 28u

IREE_API_EXPORT iree_host_size_t
iree_hal_vulkan_device_spec_payload_size(void) {
  return IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_SIZE;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_encode(
    const iree_hal_vulkan_device_spec_t* spec, iree_byte_span_t payload) {
  IREE_ASSERT_ARGUMENT(spec);
  if (payload.data_length != iree_hal_vulkan_device_spec_payload_size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec payload must be exactly %" PRIhsz
        " bytes; got %" PRIhsz,
        iree_hal_vulkan_device_spec_payload_size(), payload.data_length);
  }
  if (IREE_UNLIKELY(!payload.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec payload storage is NULL");
  }
  iree_unaligned_store_le_u32(payload.data + 0,
                              IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_MAGIC);
  iree_unaligned_store_le_u32(payload.data + 4,
                              IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION);
  iree_unaligned_store_le_u32(payload.data + 8, spec->api_version);
  iree_unaligned_store_le_u32(payload.data + 12, spec->driver_version);
  iree_unaligned_store_le_u32(payload.data + 16, spec->physical_device_type);
  iree_unaligned_store_le_u32(payload.data + 20, spec->enabled_features);
  iree_unaligned_store_le_u32(payload.data + 24, spec->flags);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_decode(
    iree_const_byte_span_t payload, iree_hal_vulkan_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  memset(out_spec, 0, sizeof(*out_spec));
  if (payload.data_length != iree_hal_vulkan_device_spec_payload_size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec payload must be exactly %" PRIhsz
        " bytes; got %" PRIhsz,
        iree_hal_vulkan_device_spec_payload_size(), payload.data_length);
  }
  if (IREE_UNLIKELY(!payload.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec payload storage is NULL");
  }
  const uint32_t magic = iree_unaligned_load_le_u32(payload.data + 0);
  if (magic != IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec payload has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(payload.data + 4);
  if (version != IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec payload version %u is not supported", version);
  }
  out_spec->api_version = iree_unaligned_load_le_u32(payload.data + 8);
  out_spec->driver_version = iree_unaligned_load_le_u32(payload.data + 12);
  out_spec->physical_device_type =
      iree_unaligned_load_le_u32(payload.data + 16);
  out_spec->enabled_features = iree_unaligned_load_le_u32(payload.data + 20);
  out_spec->flags = iree_unaligned_load_le_u32(payload.data + 24);
  return iree_ok_status();
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_vulkan_device_spec_find_facet(
    const iree_hal_device_spec_t* device_spec) {
  IREE_ASSERT_ARGUMENT(device_spec);
  return iree_hal_device_spec_find_facet(
      device_spec,
      iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID));
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_decode_facet(
    const iree_hal_device_spec_facet_t* facet,
    iree_hal_vulkan_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(facet);
  IREE_ASSERT_ARGUMENT(out_spec);
  if (!iree_string_view_equal(
          facet->schema_id,
          iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "facet is not a Vulkan device spec");
  }
  if (facet->schema_version != IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec facet version %u is not supported",
        facet->schema_version);
  }
  return iree_hal_vulkan_device_spec_decode(facet->payload, out_spec);
}
