// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_builder.h"

#include <inttypes.h>

#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

void iree_hal_amdgpu_pm4_dword_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_dword_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

void iree_hal_amdgpu_pm4_dword_builder_deinitialize(
    iree_hal_amdgpu_pm4_dword_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->dwords);
  }
  memset(builder, 0, sizeof(*builder));
}

void iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t* dwords,
    uint32_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->dwords);
  }
  builder->dwords = dwords;
  builder->dword_count = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

iree_status_t iree_hal_amdgpu_pm4_dword_builder_reserve(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident program storage is undersized");
  }

  uint32_t new_capacity = builder->capacity ? builder->capacity : 256u;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity >
                      IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT / 2u)) {
      new_capacity = IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires %u dwords, exceeding PM4-IB maximum %u",
        required_capacity, IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }

  uint32_t* new_dwords = builder->dwords;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc_array(builder->host_allocator, new_capacity,
                                   sizeof(*new_dwords), (void**)&new_dwords));
  builder->dwords = new_dwords;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dword_builder_append(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t dword_count,
    uint32_t** out_dwords) {
  *out_dwords = NULL;
  if (IREE_UNLIKELY(dword_count > IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                                      builder->dword_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  const uint32_t required_capacity = builder->dword_count + dword_count;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_dword_builder_reserve(builder, required_capacity));
  *out_dwords = &builder->dwords[builder->dword_count];
  builder->dword_count = required_capacity;
  return iree_ok_status();
}

void iree_hal_amdgpu_pm4_byte_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_byte_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

void iree_hal_amdgpu_pm4_byte_builder_deinitialize(
    iree_hal_amdgpu_pm4_byte_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->bytes);
  }
  memset(builder, 0, sizeof(*builder));
}

void iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, uint8_t* bytes,
    iree_host_size_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->bytes);
  }
  builder->bytes = bytes;
  builder->length = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

iree_status_t iree_hal_amdgpu_pm4_byte_builder_reserve(
    iree_hal_amdgpu_pm4_byte_builder_t* builder,
    iree_host_size_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident template storage is undersized");
  }

  iree_host_size_t new_capacity = builder->capacity ? builder->capacity : 4096;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity > UINT32_MAX / 2u)) {
      new_capacity = UINT32_MAX;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic template storage requires %" PRIhsz
        " bytes, exceeding uint32_t fixup offsets",
        required_capacity);
  }

  uint8_t* new_bytes = builder->bytes;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      builder->host_allocator, new_capacity, (void**)&new_bytes));
  builder->bytes = new_bytes;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_aligned(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t alignment,
    iree_host_size_t byte_length, uint32_t* out_offset, uint8_t** out_bytes) {
  *out_offset = 0;
  *out_bytes = NULL;
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 template alignment must be a power-of-two");
  }
  const iree_host_size_t aligned_length =
      iree_host_align(builder->length, alignment);
  if (IREE_UNLIKELY(aligned_length > UINT32_MAX ||
                    byte_length > UINT32_MAX - aligned_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dynamic template offset exceeds uint32_t storage");
  }
  const iree_host_size_t required_capacity = aligned_length + byte_length;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_byte_builder_reserve(builder, required_capacity));
  if (aligned_length > builder->length) {
    memset(builder->bytes + builder->length, 0,
           aligned_length - builder->length);
  }
  uint8_t* bytes = builder->bytes + aligned_length;
  memset(bytes, 0, byte_length);
  builder->length = required_capacity;
  *out_offset = (uint32_t)aligned_length;
  *out_bytes = bytes;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_record(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t byte_length,
    uint8_t** out_bytes) {
  *out_bytes = NULL;
  if (IREE_UNLIKELY(byte_length > UINT32_MAX - builder->length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer host record storage exceeds uint32_t offsets");
  }
  const iree_host_size_t required_capacity = builder->length + byte_length;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pm4_byte_builder_reserve(builder, required_capacity));
  *out_bytes = builder->bytes + builder->length;
  builder->length = required_capacity;
  return iree_ok_status();
}

void iree_hal_amdgpu_pm4_fixup_entry_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->owns_storage = true;
}

void iree_hal_amdgpu_pm4_fixup_entry_builder_deinitialize(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->entries);
  }
  memset(builder, 0, sizeof(*builder));
}

void iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries,
    uint32_t capacity) {
  if (builder->owns_storage) {
    iree_allocator_free(builder->host_allocator, builder->entries);
  }
  builder->entries = entries;
  builder->count = 0;
  builder->capacity = capacity;
  builder->owns_storage = false;
}

iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    uint32_t required_capacity) {
  if (required_capacity <= builder->capacity) return iree_ok_status();
  if (IREE_UNLIKELY(!builder->owns_storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer resident fixup storage is undersized");
  }

  uint32_t new_capacity = builder->capacity ? builder->capacity : 256u;
  while (new_capacity < required_capacity) {
    if (IREE_UNLIKELY(new_capacity > UINT32_MAX / 2u)) {
      new_capacity = UINT32_MAX;
      break;
    }
    new_capacity *= 2u;
  }
  if (IREE_UNLIKELY(new_capacity < required_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer fixup entry count exceeds uint32_t storage");
  }

  iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* new_entries =
      builder->entries;
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc_array(builder->host_allocator, new_capacity,
                                   sizeof(*new_entries), (void**)&new_entries));
  builder->entries = new_entries;
  builder->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_append(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t entry) {
  if (IREE_UNLIKELY(builder->count == UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer fixup entry count exceeds uint32_t storage");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
      builder, builder->count + 1u));
  builder->entries[builder->count++] = entry;
  return iree_ok_status();
}
