// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec.h"

#include "iree/base/alignment.h"

#define IREE_HAL_VULKAN_DEVICE_SPEC_PAYLOAD_MAGIC UINT32_C(0x53445656)  // VVDS
#define IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE 36u
#define IREE_HAL_VULKAN_COOPERATIVE_MATRIX_PROPERTY_SIZE 36u

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_device_spec_calculate_payload_size(
    iree_host_size_t property_count, iree_host_size_t* out_payload_size) {
  IREE_ASSERT_ARGUMENT(out_payload_size);
  *out_payload_size = 0;
  if (IREE_UNLIKELY(property_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan cooperative matrix property count %" PRIhsz
                            " exceeds the uint32_t wire range",
                            property_count);
  }
  iree_host_size_t property_data_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          property_count, IREE_HAL_VULKAN_COOPERATIVE_MATRIX_PROPERTY_SIZE,
          &property_data_size)) ||
      IREE_UNLIKELY(
          !iree_host_size_checked_add(IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE,
                                      property_data_size, out_payload_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan device spec payload size overflow");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_encode(
    const iree_hal_vulkan_device_spec_t* spec, iree_host_size_t property_count,
    const iree_hal_vulkan_cooperative_matrix_property_t* properties,
    iree_byte_span_t payload) {
  IREE_ASSERT_ARGUMENT(spec);
  if (IREE_UNLIKELY(property_count != 0 && !properties)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan cooperative matrix property storage is NULL");
  }
  iree_host_size_t expected_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      property_count, &expected_payload_size));
  if (payload.data_length != expected_payload_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec payload must be exactly %" PRIhsz
        " bytes; got %" PRIhsz,
        expected_payload_size, payload.data_length);
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
  iree_unaligned_store_le_u64(payload.data + 20, spec->enabled_features);
  iree_unaligned_store_le_u32(payload.data + 28, spec->flags);
  iree_unaligned_store_le_u32(payload.data + 32, (uint32_t)property_count);
  for (iree_host_size_t i = 0; i < property_count; ++i) {
    const iree_hal_vulkan_cooperative_matrix_property_t* property =
        &properties[i];
    uint8_t* row = payload.data + IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE +
                   i * IREE_HAL_VULKAN_COOPERATIVE_MATRIX_PROPERTY_SIZE;
    iree_unaligned_store_le_u32(row + 0, property->m_size);
    iree_unaligned_store_le_u32(row + 4, property->n_size);
    iree_unaligned_store_le_u32(row + 8, property->k_size);
    iree_unaligned_store_le_u32(row + 12, property->a_type);
    iree_unaligned_store_le_u32(row + 16, property->b_type);
    iree_unaligned_store_le_u32(row + 20, property->c_type);
    iree_unaligned_store_le_u32(row + 24, property->result_type);
    iree_unaligned_store_le_u32(row + 28, property->saturating_accumulation);
    iree_unaligned_store_le_u32(row + 32, property->scope);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_decode(
    iree_const_byte_span_t payload, iree_hal_vulkan_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  memset(out_spec, 0, sizeof(*out_spec));
  if (payload.data_length < IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan device spec payload must contain at least "
                            "%u bytes; got %" PRIhsz,
                            IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE,
                            payload.data_length);
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
  const iree_host_size_t property_count =
      iree_unaligned_load_le_u32(payload.data + 32);
  iree_host_size_t expected_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      property_count, &expected_payload_size));
  if (payload.data_length != expected_payload_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan device spec payload with %" PRIhsz
        " cooperative matrix properties must be exactly %" PRIhsz
        " bytes; got %" PRIhsz,
        property_count, expected_payload_size, payload.data_length);
  }
  out_spec->api_version = iree_unaligned_load_le_u32(payload.data + 8);
  out_spec->driver_version = iree_unaligned_load_le_u32(payload.data + 12);
  out_spec->physical_device_type =
      iree_unaligned_load_le_u32(payload.data + 16);
  out_spec->enabled_features = iree_unaligned_load_le_u64(payload.data + 20);
  out_spec->flags = iree_unaligned_load_le_u32(payload.data + 28);
  out_spec->cooperative_matrix.count = property_count;
  out_spec->cooperative_matrix.encoded_data =
      property_count == 0
          ? iree_const_byte_span_empty()
          : iree_make_const_byte_span(
                payload.data + IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE,
                expected_payload_size -
                    IREE_HAL_VULKAN_DEVICE_SPEC_HEADER_SIZE);
  return iree_ok_status();
}

IREE_API_EXPORT bool
iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
    const iree_hal_vulkan_device_spec_t* spec, iree_host_size_t ordinal,
    iree_hal_vulkan_cooperative_matrix_property_t* out_property) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(out_property);
  memset(out_property, 0, sizeof(*out_property));
  if (ordinal >= spec->cooperative_matrix.count ||
      spec->cooperative_matrix.encoded_data.data == NULL ||
      spec->cooperative_matrix.encoded_data.data_length /
              IREE_HAL_VULKAN_COOPERATIVE_MATRIX_PROPERTY_SIZE <=
          ordinal) {
    return false;
  }
  const uint8_t* row =
      spec->cooperative_matrix.encoded_data.data +
      ordinal * IREE_HAL_VULKAN_COOPERATIVE_MATRIX_PROPERTY_SIZE;
  out_property->m_size = iree_unaligned_load_le_u32(row + 0);
  out_property->n_size = iree_unaligned_load_le_u32(row + 4);
  out_property->k_size = iree_unaligned_load_le_u32(row + 8);
  out_property->a_type = iree_unaligned_load_le_u32(row + 12);
  out_property->b_type = iree_unaligned_load_le_u32(row + 16);
  out_property->c_type = iree_unaligned_load_le_u32(row + 20);
  out_property->result_type = iree_unaligned_load_le_u32(row + 24);
  out_property->saturating_accumulation = iree_unaligned_load_le_u32(row + 28);
  out_property->scope = iree_unaligned_load_le_u32(row + 32);
  return true;
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
