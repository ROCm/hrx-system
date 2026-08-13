// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_builder.h"

#include <inttypes.h>

#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
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

iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  const uint32_t barrier_dword_count = iree_hal_amdgpu_pm4_barrier_dword_count(
      capabilities, barrier_flags, acquire_scope, release_scope);
  if (IREE_UNLIKELY(barrier_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 command-buffer barrier cannot be emitted with capabilities "
        "0x%08" PRIx32 ", flags 0x%08" PRIx32,
        capabilities, barrier_flags);
  }

  uint32_t* barrier_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, barrier_dword_count, &barrier_dwords));
  uint32_t emitted_dword_count = 0;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_pm4_barrier_emit(
                        capabilities, barrier_flags, acquire_scope,
                        release_scope, barrier_dword_count, barrier_dwords,
                        &emitted_dword_count) ||
                    emitted_dword_count != barrier_dword_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 command-buffer barrier emission changed size");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t
        source_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT],
    uint32_t setup_dword_count) {
  if (IREE_UNLIKELY(setup_dword_count !=
                    IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 dispatch setup dword count is invalid");
  }
  uint32_t* target_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, setup_dword_count, &target_dwords));
  memcpy(target_dwords, source_dwords,
         setup_dword_count * sizeof(source_dwords[0]));
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    uint64_t kernarg_address, const void* kernarg_preload_data) {
  if (launch_state->user_data_dword_count == 0) return iree_ok_status();

  const uint32_t user_data_dword_count =
      2u + launch_state->user_data_dword_count;
  uint32_t* user_data_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, user_data_dword_count, &user_data_dwords));
  uint32_t emitted_dword_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      launch_state, kernarg_address, kernarg_preload_data,
      user_data_dword_count, user_data_dwords, &emitted_dword_count));
  if (IREE_UNLIKELY(emitted_dword_count != user_data_dword_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 dispatch user-data emission changed size");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t dispatch_thread_count[3], uint32_t dispatch_initiator) {
  uint32_t* dispatch_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
      &dispatch_dwords));
  dispatch_dwords[0] = iree_hal_amdgpu_pm4_make_compute_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT,
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT);
  dispatch_dwords[1] = dispatch_thread_count[0];
  dispatch_dwords[2] = dispatch_thread_count[1];
  dispatch_dwords[3] = dispatch_thread_count[2];
  dispatch_dwords[4] = dispatch_initiator;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    uint32_t kernarg_byte_offset, uint32_t kernarg_byte_length,
    bool* out_is_preloaded, uint32_t* out_preload_dword_offset) {
  *out_is_preloaded = false;
  *out_preload_dword_offset = 0;
  if (launch_state->kernarg_preload_dword_count == 0) return iree_ok_status();

  if (IREE_UNLIKELY(kernarg_byte_offset > UINT32_MAX - kernarg_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dynamic kernarg byte range overflows");
  }
  const uint32_t preload_start =
      launch_state->kernarg_preload_dword_offset * sizeof(uint32_t);
  const uint32_t preload_length =
      launch_state->kernarg_preload_dword_count * sizeof(uint32_t);
  const uint32_t preload_end = preload_start + preload_length;
  const uint32_t kernarg_end = kernarg_byte_offset + kernarg_byte_length;
  const bool overlaps =
      kernarg_byte_offset < preload_end && kernarg_end > preload_start;
  if (!overlaps) return iree_ok_status();
  if (IREE_UNLIKELY(
          kernarg_byte_offset < preload_start || kernarg_end > preload_end ||
          (kernarg_byte_offset - preload_start) % sizeof(uint32_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 dynamic binding fixup cannot patch partial kernarg preload "
        "range [%u, %u) inside preload range [%u, %u)",
        kernarg_byte_offset, kernarg_end, preload_start, preload_end);
  }
  *out_is_preloaded = true;
  *out_preload_dword_offset =
      (kernarg_byte_offset - preload_start) / sizeof(uint32_t);
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
