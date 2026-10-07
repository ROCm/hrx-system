// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/storage.h"

#include <string.h>

void loom_serve_text_storage_release_payloads(
    loom_serve_text_storage_t* storage) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT; ++i) {
    iree_vm_buffer_release(storage->buffers[i]);
  }
  memset(storage->buffers, 0, sizeof(storage->buffers));
  memset(storage->bytes, 0, sizeof(storage->bytes));
}

void loom_serve_text_storage_deinitialize(loom_serve_text_storage_t* storage) {
  loom_serve_text_storage_release_payloads(storage);
  iree_allocator_free(storage->allocator, storage->regions);
  memset(storage, 0, sizeof(*storage));
}

// Source records cross the external model boundary once. Native consumers use
// the established sizes directly; HAL owns allocation and subspan validation.
iree_status_t loom_serve_text_storage_initialize(
    const iree_vm_ref_types_t* types, iree_vm_variant_t* results,
    iree_host_size_t row_count, iree_host_size_t context_capacity,
    iree_host_size_t pool_capacity, loom_serve_text_storage_t* storage,
    iree_allocator_t allocator) {
  *storage = (loom_serve_text_storage_t){.allocator = allocator};
  const iree_host_size_t lengths[] = {
      LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT * 3 * sizeof(int64_t),
      row_count * 5 * 2 * sizeof(int64_t),
      0,
      0,
      4 * sizeof(int64_t),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT && iree_status_is_ok(status);
       ++i) {
    status = iree_vm_buffer_ptr_from_variant_move(types, &results[i],
                                                  &storage->buffers[i]);
    const iree_host_size_t length =
        storage->buffers[i] ? iree_vm_buffer_length(storage->buffers[i]) : 0;
    const bool geometry = i == LOOM_SERVE_TEXT_STORAGE_GEOMETRY;
    const bool origins = i == LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS ||
                         i == LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS;
    if (iree_status_is_ok(status) &&
        (!storage->buffers[i] ||
         (!origins &&
          (geometry ? (length < lengths[i] || (length - lengths[i]) % 40)
                    : length != lengths[i])))) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "source storage result %zu has invalid size", i);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_read(storage->buffers[i], 0, length,
                                       &storage->bytes[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const uint8_t* geometry =
      storage->bytes[LOOM_SERVE_TEXT_STORAGE_GEOMETRY].data;
  const uint64_t block_size = iree_unaligned_load_le_u64(geometry);
  const uint64_t map_origin = iree_unaligned_load_le_u64(geometry + 8);
  const uint64_t carry_stride = iree_unaligned_load_le_u64(geometry + 16);
  const uint64_t feedback_split = iree_unaligned_load_le_u64(geometry + 24);
  if (!block_size || block_size > UINT32_MAX || pool_capacity % block_size ||
      map_origin > INT64_MAX || !carry_stride ||
      carry_stride > INT64_MAX / row_count || !feedback_split ||
      feedback_split > INT64_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source cache geometry cannot represent this pool");
  }
  storage->block_size = (iree_host_size_t)block_size;
  storage->map_origin = map_origin;
  storage->blocks_per_row = (context_capacity + block_size - 1) / block_size;
  storage->carry_stride = carry_stride;
  storage->feedback_split = feedback_split;
  // The native owner interprets page maps, not the model's preceding origin
  // records. Optional unallocated tables have no upload or map consumers.
  const iree_host_size_t table_allocations[] = {3, 10};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(table_allocations); ++i) {
    const uint64_t length = iree_unaligned_load_le_u64(
        storage->bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data +
        table_allocations[i] * 24);
    if (!length) {
      continue;
    }
    const iree_host_size_t payload_length =
        storage->bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS + i].data_length;
    if (payload_length > length ||
        (pool_capacity &&
         (map_origin < payload_length || map_origin > length ||
          storage->blocks_per_row >
              (length - map_origin) / (row_count * sizeof(uint32_t))))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source table %zu cannot fit its origin "
                              "payload and page maps",
                              i);
    }
  }
  storage->region_count =
      (storage->bytes[LOOM_SERVE_TEXT_STORAGE_GEOMETRY].data_length - 32) / 40;
  if ((storage->region_count != 0) != (pool_capacity != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pooled state requires source cache regions");
  }
  if (!storage->region_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, storage->region_count, sizeof(*storage->regions),
      (void**)&storage->regions));
  for (iree_host_size_t i = 0; i < storage->region_count; ++i) {
    const uint8_t* record = geometry + 32 + i * 40;
    const uint64_t allocation = iree_unaligned_load_le_u64(record);
    const uint64_t origin = iree_unaligned_load_le_u64(record + 8);
    const uint64_t count = iree_unaligned_load_le_u64(record + 16);
    const uint64_t stride = iree_unaligned_load_le_u64(record + 24);
    const uint64_t block_bytes = iree_unaligned_load_le_u64(record + 32);
    if (allocation >= LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT || !count ||
        !stride || !block_bytes) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid cache region %zu", i);
    }
    const uint64_t length = iree_unaligned_load_le_u64(
        storage->bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data +
        allocation * 24);
    const uint64_t private_length = iree_unaligned_load_le_u64(
        storage->bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data +
        allocation * 24 + 16);
    const uint64_t blocks = pool_capacity / block_size;
    if (origin < private_length || origin > length ||
        count - 1 > (length - origin) / stride ||
        block_bytes > (length - origin - (count - 1) * stride) / blocks ||
        block_bytes > stride / blocks) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cache region %zu exceeds its allocation", i);
    }
    storage->regions[i] = (loom_serve_text_cache_region_t){
        .allocation = allocation,
        .blocks = {origin, count, stride, block_bytes}};
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_storage_plan_snapshot(
    const loom_serve_text_storage_t* storage, iree_host_size_t row_index,
    iree_hal_buffer_t* const* private_views, bool enable_mtp,
    uint32_t block_count, const uint32_t* blocks, iree_host_size_t* out_count,
    loom_serve_snapshot_range_t** out_ranges, iree_allocator_t allocator) {
  *out_count = 0;
  *out_ranges = NULL;
  iree_host_size_t capacity = 5 + (enable_mtp ? 1 : 0);
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    if (block_count && storage->regions[r].blocks.count >
                           (IREE_HOST_SIZE_MAX - capacity) / block_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "snapshot transfer plan exceeds host size");
    }
    capacity += storage->regions[r].blocks.count * block_count;
  }
  loom_serve_snapshot_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, capacity, sizeof(*ranges), (void**)&ranges));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < 5; ++i) {
    if (private_views[i]) {
      ranges[count++] = (loom_serve_snapshot_range_t){
          1, iree_hal_buffer_byte_offset(private_views[i]),
          iree_hal_buffer_byte_length(private_views[i])};
    }
  }
  if (enable_mtp) {
    ranges[count++] = (loom_serve_snapshot_range_t){
        6, row_index * storage->carry_stride, storage->carry_stride};
  }
  for (iree_host_size_t r = 0; r < storage->region_count; ++r) {
    const loom_serve_text_cache_region_t* region = &storage->regions[r];
    for (iree_host_size_t plane = 0; plane < region->blocks.count; ++plane) {
      for (uint32_t block = 0; block < block_count; ++block) {
        ranges[count++] = (loom_serve_snapshot_range_t){
            region->allocation,
            region->blocks.origin + plane * region->blocks.stride +
                blocks[block] * region->blocks.block_bytes,
            region->blocks.block_bytes};
      }
    }
  }
  *out_count = count;
  *out_ranges = ranges;
  return iree_ok_status();
}
