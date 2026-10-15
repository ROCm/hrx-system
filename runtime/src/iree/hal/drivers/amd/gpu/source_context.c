// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/source_context.h"

#include <string.h>

// LSIT v1 is little-endian with a 32-byte header and dense 48-byte records.
enum {
  IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH = 32,
  IREE_HAL_AMD_GPU_SITE_RECORD_LENGTH = 48,
  IREE_HAL_AMD_GPU_SITE_HAS_PAYLOAD = 1u << 0,
  IREE_HAL_AMD_GPU_SITE_HAS_LOCATION = 1u << 1,
};

struct iree_hal_amd_gpu_source_context_t {
  // Next context owned by the device's registry.
  iree_hal_amd_gpu_source_context_t* next;
  // Stable device-local identity assigned during registration.
  uint64_t executable_id;
  // Number of validated dense site records.
  uint32_t row_count;
  // Byte offset of source strings within data.
  uint32_t string_offset;
  // Byte offset of producer payloads within data.
  uint32_t payload_offset;
  // Immutable owned table, including all referenced strings and payloads.
  uint8_t data[];
};

static bool iree_hal_amd_gpu_source_context_contains(uint64_t length,
                                                     uint64_t offset,
                                                     uint64_t extent) {
  return offset <= length && extent <= length - offset;
}

static iree_status_t iree_hal_amd_gpu_source_context_validate(
    iree_const_byte_span_t table) {
  if (!table.data_length) {
    return iree_ok_status();
  }
  const uint8_t* data = table.data;
  if (table.data_length < IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH ||
      iree_unaligned_load_le_u32(data) != 0x5449534Cu || data[4] != 1 ||
      data[5] != IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH ||
      iree_unaligned_load_le_u16(data + 6) !=
          IREE_HAL_AMD_GPU_SITE_RECORD_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported GPU sanitizer source table header");
  }
  const uint32_t row_count = iree_unaligned_load_le_u32(data + 8);
  const uint32_t string_offset = iree_unaligned_load_le_u32(data + 12);
  const uint32_t string_length = iree_unaligned_load_le_u32(data + 16);
  const uint32_t payload_offset = iree_unaligned_load_le_u32(data + 20);
  const uint32_t payload_length = iree_unaligned_load_le_u32(data + 24);
  if (!iree_hal_amd_gpu_source_context_contains(
          table.data_length, IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH,
          (uint64_t)row_count * IREE_HAL_AMD_GPU_SITE_RECORD_LENGTH) ||
      !iree_hal_amd_gpu_source_context_contains(table.data_length,
                                                string_offset, string_length) ||
      !iree_hal_amd_gpu_source_context_contains(
          table.data_length, payload_offset, payload_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "GPU sanitizer source table sections exceed storage");
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < row_count && iree_status_is_ok(status); ++i) {
    const uint8_t* row =
        data + IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH +
        (iree_host_size_t)i * IREE_HAL_AMD_GPU_SITE_RECORD_LENGTH;
    const uint32_t flags = iree_unaligned_load_le_u32(row + 8);
    const uint32_t row_payload_offset = iree_unaligned_load_le_u32(row + 12);
    const uint32_t row_payload_length = iree_unaligned_load_le_u32(row + 16);
    const uint32_t name_offset = iree_unaligned_load_le_u32(row + 20);
    const uint32_t name_length = iree_unaligned_load_le_u32(row + 24);
    const uint16_t source_kind = iree_unaligned_load_le_u16(row + 44);
    if (iree_unaligned_load_le_u32(row) != i || (flags & ~3u) ||
        ((flags & IREE_HAL_AMD_GPU_SITE_HAS_PAYLOAD)
             ? !iree_hal_amd_gpu_source_context_contains(
                   payload_length, row_payload_offset, row_payload_length)
             : row_payload_length != 0) ||
        ((flags & IREE_HAL_AMD_GPU_SITE_HAS_LOCATION)
             ? (source_kind != 1 ||
                !iree_hal_amd_gpu_source_context_contains(
                    string_length, name_offset, (uint64_t)name_length + 1) ||
                data[(iree_host_size_t)string_offset + name_offset +
                     name_length])
             : source_kind != 0)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid GPU sanitizer source table row %u", i);
    }
  }
  return status;
}

void iree_hal_amd_gpu_source_context_registry_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_source_context_registry_t* out_registry) {
  *out_registry = (iree_hal_amd_gpu_source_context_registry_t){
      .host_allocator = host_allocator};
  iree_slim_mutex_initialize(&out_registry->mutex);
}

void iree_hal_amd_gpu_source_context_registry_deinitialize(
    iree_hal_amd_gpu_source_context_registry_t* registry) {
  iree_hal_amd_gpu_source_context_t* context = registry->first;
  while (context) {
    iree_hal_amd_gpu_source_context_t* next = context->next;
    iree_allocator_free(registry->host_allocator, context);
    context = next;
  }
  iree_slim_mutex_deinitialize(&registry->mutex);
}

iree_status_t iree_hal_amd_gpu_source_context_registry_register(
    iree_hal_amd_gpu_source_context_registry_t* registry,
    iree_const_byte_span_t site_table,
    const iree_hal_amd_gpu_source_context_t** out_context) {
  *out_context = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_source_context_validate(site_table));
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_amd_gpu_source_context_t),
                                  site_table.data_length, &length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "GPU source context allocation overflows");
  }
  iree_hal_amd_gpu_source_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(registry->host_allocator, length,
                                             (void**)&context));
  if (site_table.data_length) {
    memcpy(context->data, site_table.data, site_table.data_length);
    context->row_count = iree_unaligned_load_le_u32(context->data + 8);
    context->string_offset = iree_unaligned_load_le_u32(context->data + 12);
    context->payload_offset = iree_unaligned_load_le_u32(context->data + 20);
  }
  iree_slim_mutex_lock(&registry->mutex);
  context->executable_id = ++registry->last_executable_id;
  context->next = registry->first;
  registry->first = context;
  iree_slim_mutex_unlock(&registry->mutex);
  *out_context = context;
  return iree_ok_status();
}

uint64_t iree_hal_amd_gpu_source_context_executable_id(
    const iree_hal_amd_gpu_source_context_t* context) {
  return context ? context->executable_id : 0;
}

static iree_string_view_t iree_hal_amd_gpu_source_context_operation(
    uint32_t kind) {
  switch (kind) {
    case 0x1029:
      return IREE_SV("kernel.assert");
    case 0x1D00:
      return IREE_SV("sanitizer.assert.access");
    case 0x1D01:
      return IREE_SV("sanitizer.assert.value");
    case 0x1D02:
      return IREE_SV("sanitizer.assert.op");
    case 0x1D03:
      return IREE_SV("sanitizer.assert.layout");
    case 0x1D04:
      return IREE_SV("sanitizer.race.access");
    case 0x1D05:
      return IREE_SV("sanitizer.race.sync");
    default:
      return iree_string_view_empty();
  }
}

bool iree_hal_amd_gpu_source_context_resolve(
    const iree_hal_amd_gpu_source_context_t* context, uint64_t site_id,
    iree_hal_device_event_site_t* out_site) {
  *out_site = iree_hal_device_event_site_default();
  if (!context || site_id >= context->row_count) {
    return false;
  }
  const uint8_t* row =
      context->data + IREE_HAL_AMD_GPU_SITE_HEADER_LENGTH +
      (iree_host_size_t)site_id * IREE_HAL_AMD_GPU_SITE_RECORD_LENGTH;
  const uint32_t flags = iree_unaligned_load_le_u32(row + 8);
  out_site->site_id = site_id;
  out_site->operation_name = iree_hal_amd_gpu_source_context_operation(
      iree_unaligned_load_le_u32(row + 4));
  if (flags & IREE_HAL_AMD_GPU_SITE_HAS_LOCATION) {
    out_site->source_file = iree_make_string_view(
        (const char*)context->data + context->string_offset +
            iree_unaligned_load_le_u32(row + 20),
        iree_unaligned_load_le_u32(row + 24));
    out_site->start_line = iree_unaligned_load_le_u32(row + 28);
    out_site->start_column = iree_unaligned_load_le_u32(row + 32);
    out_site->end_line = iree_unaligned_load_le_u32(row + 36);
    out_site->end_column = iree_unaligned_load_le_u32(row + 40);
  }
  if (flags & IREE_HAL_AMD_GPU_SITE_HAS_PAYLOAD) {
    out_site->producer_payload =
        iree_make_const_byte_span(context->data + context->payload_offset +
                                      iree_unaligned_load_le_u32(row + 12),
                                  iree_unaligned_load_le_u32(row + 16));
  }
  return true;
}
