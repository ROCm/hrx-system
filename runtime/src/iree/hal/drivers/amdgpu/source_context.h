// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SOURCE_CONTEXT_H_
#define IREE_HAL_DRIVERS_AMDGPU_SOURCE_CONTEXT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/device_event.h"
#include "iree/hal/drivers/amdgpu/util/loaded_code_object.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executable global name for Loom sanitizer site-table metadata.
#define IREE_HAL_AMDGPU_LOOM_SANITIZER_SITE_TABLE_GLOBAL_NAME \
  "loom_sanitizer_sites"

typedef struct iree_hal_amdgpu_source_context_site_table_t {
  // Borrowed host pointer to the full site-table blob.
  const uint8_t* data;
  // Byte length of the full site-table blob.
  iree_host_size_t data_length;
  // Number of dense site records in the table.
  uint32_t row_count;
  // Byte length of each site record.
  uint16_t record_length;
  // Byte offset from |data| to the source string table.
  uint32_t string_table_offset;
  // Byte length of the source string table.
  uint32_t string_table_length;
  // Byte offset from |data| to the producer payload data.
  uint32_t payload_data_offset;
  // Byte length of the producer payload data.
  uint32_t payload_data_length;
} iree_hal_amdgpu_source_context_site_table_t;

typedef struct iree_hal_amdgpu_source_context_t {
  // Logical-device-local executable identifier assigned at creation.
  uint64_t executable_id;
  // Stable content hash for the exact loaded HSACO/code-object bytes.
  uint64_t code_object_hash[2];
  // Number of physical device ranges in |loaded_code_object_ranges|.
  iree_host_size_t physical_device_count;
  // Bitmask of physical device ordinals with loaded code objects.
  uint64_t loaded_physical_device_mask;
  // Borrowed executable-owned loaded code-object ranges indexed by device.
  iree_hal_amdgpu_loaded_code_object_range_t* loaded_code_object_ranges;
  // Optional parsed Loom sanitizer site-table descriptor.
  iree_hal_amdgpu_source_context_site_table_t sanitizer_site_table;
  // Non-zero after |sanitizer_site_table| has been release-published.
  iree_atomic_int32_t sanitizer_site_table_published;
} iree_hal_amdgpu_source_context_t;

// Initializes |out_context| with executable-owned storage.
void iree_hal_amdgpu_source_context_initialize(
    uint64_t executable_id, const uint64_t code_object_hash[2],
    iree_host_size_t physical_device_count,
    uint64_t loaded_physical_device_mask,
    iree_hal_amdgpu_loaded_code_object_range_t* loaded_code_object_ranges,
    iree_hal_amdgpu_source_context_t* out_context);

// Returns the executable id associated with |context|.
static inline uint64_t iree_hal_amdgpu_source_context_executable_id(
    const iree_hal_amdgpu_source_context_t* context) {
  return context ? context->executable_id : 0;
}

// Records the loaded code-object range for one physical device ordinal.
iree_status_t iree_hal_amdgpu_source_context_set_loaded_code_object_range(
    iree_hal_amdgpu_source_context_t* context,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_loaded_code_object_range_t range);

// Translates a device-address span within one loaded code object to host bytes.
bool iree_hal_amdgpu_source_context_try_translate_device_span(
    const iree_hal_amdgpu_source_context_t* context,
    iree_host_size_t physical_device_ordinal, uint64_t device_pointer,
    uint64_t byte_length, iree_const_byte_span_t* out_host_span);

// Validates and release-publishes a borrowed Loom sanitizer site table.
// May be called at most once for a source context.
iree_status_t iree_hal_amdgpu_source_context_set_sanitizer_site_table(
    iree_hal_amdgpu_source_context_t* context, iree_const_byte_span_t table);

// Acquire-observes the published table and tries to resolve |site_id|.
bool iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
    const iree_hal_amdgpu_source_context_t* context, uint64_t site_id,
    iree_hal_device_event_site_t* out_site);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SOURCE_CONTEXT_H_
