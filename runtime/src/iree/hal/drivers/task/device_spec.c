// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/device_spec.h"

#include "iree/base/alignment.h"

#define IREE_HAL_CPU_DEVICE_SPEC_PAYLOAD_MAGIC UINT32_C(0x53445043)  // CPDS
#define IREE_HAL_CPU_DEVICE_SPEC_PAYLOAD_SIZE 80u

static_assert(IREE_CPU_DATA_FIELD_COUNT == 8,
              "CPU device spec v1 encodes eight data fields");

static iree_status_t iree_hal_cpu_device_spec_validate(
    const iree_hal_cpu_device_spec_t* spec) {
  if (iree_string_view_is_empty(
          iree_cpu_architecture_name(spec->cpu_data.architecture))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec has invalid architecture %" PRIu32,
                            spec->cpu_data.architecture);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t iree_hal_cpu_device_spec_payload_size(void) {
  return IREE_HAL_CPU_DEVICE_SPEC_PAYLOAD_SIZE;
}

IREE_API_EXPORT iree_status_t iree_hal_cpu_device_spec_encode(
    const iree_hal_cpu_device_spec_t* spec, iree_byte_span_t payload) {
  IREE_ASSERT_ARGUMENT(spec);
  if (payload.data_length != iree_hal_cpu_device_spec_payload_size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec payload must be exactly %" PRIhsz
                            " bytes; got %" PRIhsz,
                            iree_hal_cpu_device_spec_payload_size(),
                            payload.data_length);
  }
  if (IREE_UNLIKELY(!payload.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec payload storage is NULL");
  }
  IREE_RETURN_IF_ERROR(iree_hal_cpu_device_spec_validate(spec));

  iree_unaligned_store_le_u32(payload.data + 0,
                              IREE_HAL_CPU_DEVICE_SPEC_PAYLOAD_MAGIC);
  iree_unaligned_store_le_u32(payload.data + 4,
                              IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION);
  iree_unaligned_store_le_u32(payload.data + 8, spec->cpu_data.architecture);
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    iree_unaligned_store_le_u64(payload.data + 12 + i * sizeof(uint64_t),
                                spec->cpu_data.fields[i]);
  }
  iree_unaligned_store_le_u32(payload.data + 76, spec->flags);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_cpu_device_spec_decode(
    iree_const_byte_span_t payload, iree_hal_cpu_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  memset(out_spec, 0, sizeof(*out_spec));
  if (payload.data_length != iree_hal_cpu_device_spec_payload_size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec payload must be exactly %" PRIhsz
                            " bytes; got %" PRIhsz,
                            iree_hal_cpu_device_spec_payload_size(),
                            payload.data_length);
  }
  if (IREE_UNLIKELY(!payload.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec payload storage is NULL");
  }
  const uint32_t magic = iree_unaligned_load_le_u32(payload.data + 0);
  if (magic != IREE_HAL_CPU_DEVICE_SPEC_PAYLOAD_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec payload has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(payload.data + 4);
  if (version != IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CPU device spec payload version %u is not supported", version);
  }
  out_spec->cpu_data.architecture =
      iree_unaligned_load_le_u32(payload.data + 8);
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    out_spec->cpu_data.fields[i] =
        iree_unaligned_load_le_u64(payload.data + 12 + i * sizeof(uint64_t));
  }
  out_spec->flags = iree_unaligned_load_le_u32(payload.data + 76);
  return iree_hal_cpu_device_spec_validate(out_spec);
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_cpu_device_spec_find_facet(const iree_hal_device_spec_t* device_spec) {
  IREE_ASSERT_ARGUMENT(device_spec);
  return iree_hal_device_spec_find_facet(
      device_spec, iree_make_cstring_view(IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID));
}

IREE_API_EXPORT iree_status_t
iree_hal_cpu_device_spec_decode_facet(const iree_hal_device_spec_facet_t* facet,
                                      iree_hal_cpu_device_spec_t* out_spec) {
  IREE_ASSERT_ARGUMENT(facet);
  IREE_ASSERT_ARGUMENT(out_spec);
  if (!iree_string_view_equal(
          facet->schema_id,
          iree_make_cstring_view(IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "facet is not a CPU device spec");
  }
  if (facet->schema_version != IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU device spec facet version %u is not supported",
                            facet->schema_version);
  }
  return iree_hal_cpu_device_spec_decode(facet->payload, out_spec);
}
