// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/serve/service/text/state.h"

// Cold source boundary: capacity may append state, not reinterpret existing
// histories or change the compiled epoch's workspace and feedback geometry.
static iree_status_t loom_serve_text_state_validate_resize(
    loom_serve_text_state_t* state, iree_host_size_t row_count,
    const loom_serve_text_storage_t* storage) {
  const loom_serve_text_storage_t* previous = &state->storage;
  const uint64_t required_slots =
      state->checkpoints.capacity
          ? 2 * (uint64_t)row_count + state->checkpoints.capacity
          : row_count;
  if (storage->block_size != previous->block_size ||
      storage->blocks_per_row != previous->blocks_per_row ||
      storage->carry_stride != previous->carry_stride ||
      storage->feedback_split != previous->feedback_split ||
      storage->recurrent.origin != previous->recurrent.origin ||
      storage->recurrent.stride != previous->recurrent.stride ||
      storage->recurrent.length != previous->recurrent.length ||
      storage->recurrent.capacity < previous->recurrent.capacity ||
      storage->recurrent.capacity < required_slots ||
      storage->region_count != previous->region_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source resize must preserve retained geometry");
  }
  for (iree_host_size_t i = 0; i < storage->region_count; ++i) {
    const loom_serve_text_cache_region_t* a = &previous->regions[i];
    const loom_serve_text_cache_region_t* b = &storage->regions[i];
    if (a->allocation != b->allocation ||
        a->blocks.origin != b->blocks.origin ||
        a->blocks.count != b->blocks.count ||
        a->blocks.stride != b->blocks.stride ||
        a->blocks.block_bytes != b->blocks.block_bytes) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source resize changes cache plane %zu", i);
    }
  }
  const uint32_t bindings[] = {TEXT_CONTROL, TEXT_ATTENTION, TEXT_TOKENS,
                               TEXT_PROGRESS};
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(bindings); ++j) {
      iree_hal_buffer_t* view = state->rows[i]->buffers[bindings[j]];
      const uint8_t* record =
          storage->bytes[LOOM_SERVE_TEXT_STORAGE_VIEWS].data +
          (i * IREE_ARRAYSIZE(bindings) + j) * 16;
      const uint64_t origin = iree_unaligned_load_le_u64(record);
      const uint64_t length = iree_unaligned_load_le_u64(record + 8);
      if (length != (view ? iree_hal_buffer_byte_length(view) : 0) ||
          (view && origin != iree_hal_buffer_byte_offset(view))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source resize moves private view %zu:%zu", i,
                                j);
      }
    }
  }
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    iree_hal_buffer_t* root = *loom_serve_text_state_allocation(state, i);
    const iree_device_size_t length = storage->allocations[i].length;
    const bool resizable = i == 1 || i == 3 || i == 6 || i >= 9;
    if ((length != 0) != (root != NULL) ||
        (root && !resizable && length != iree_hal_buffer_byte_length(root)) ||
        (root && resizable && !state->memory.buffers[i] &&
         length < iree_hal_buffer_byte_length(root))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source resize changes allocation role %zu", i);
    }
    if (i == 6 && storage->carry_stride &&
        row_count + state->checkpoints.capacity >
            length / storage->carry_stride) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source resize lacks checkpoint carry space");
    }
  }
  return iree_ok_status();
}

// Metadata follows amortized cold capacity growth, never a token or epoch.
static iree_status_t loom_serve_text_resize_directory(
    iree_allocator_t host_allocator, iree_host_size_t old_count,
    iree_host_size_t new_count, iree_host_size_t element_size,
    void** directory) {
  void* replacement = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, new_count,
                                                   element_size, &replacement));
  if (old_count) {
    memcpy(replacement, *directory, old_count * element_size);
  }
  iree_allocator_free(host_allocator, *directory);
  *directory = replacement;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_state_resize_directories(
    loom_serve_text_state_t* state, iree_host_size_t row_count,
    uint32_t slot_count) {
  IREE_RETURN_IF_ERROR(loom_serve_text_resize_directory(
      state->host_allocator, state->row_count, row_count, sizeof(*state->rows),
      (void**)&state->rows));
  if (state->cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_text_resize_directory(
        state->host_allocator, state->row_count, row_count,
        state->storage.blocks_per_row * sizeof(*state->cache.maps),
        (void**)&state->cache.maps));
    IREE_RETURN_IF_ERROR(loom_serve_text_resize_directory(
        state->host_allocator, 0, row_count, 2 * sizeof(*state->cache.uploads),
        (void**)&state->cache.uploads));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_resize_directory(
      state->host_allocator, state->recurrent.pool.capacity, slot_count,
      sizeof(*state->recurrent.buffers), (void**)&state->recurrent.buffers));
  IREE_RETURN_IF_ERROR(
      loom_serve_block_pool_grow(&state->recurrent.pool, slot_count));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = state->row_count;
       i < row_count && iree_status_is_ok(status); ++i) {
    status = iree_arena_allocate(&state->row_records, sizeof(**state->rows),
                                 (void**)&state->rows[i]);
    if (iree_status_is_ok(status)) {
      *state->rows[i] = (loom_serve_text_state_row_t){
          .owner = state,
          .index = i,
          .recurrent = {.slot = UINT32_MAX, .anchor = UINT32_MAX},
          .retiring_tail = UINT32_MAX};
      ++state->row_count;
    }
  }
  return status;
}

static iree_status_t loom_serve_text_state_resize_views(
    loom_serve_text_state_t* state) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < state->recurrent.pool.capacity && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_t* replacement = NULL;
    status = iree_hal_buffer_subspan(
        state->row_arena,
        state->storage.recurrent.origin + i * state->storage.recurrent.stride,
        state->storage.recurrent.length, state->host_allocator, &replacement);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_release(state->recurrent.buffers[i]);
      state->recurrent.buffers[i] = replacement;
    }
  }
  const uint32_t bindings[] = {TEXT_CONTROL, TEXT_ATTENTION, TEXT_TOKENS,
                               TEXT_PROGRESS};
  for (iree_host_size_t i = 0;
       i < state->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_state_row_t* row = state->rows[i];
    row->buffers[TEXT_RESIDUAL] = state->residual;
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(bindings) && iree_status_is_ok(status); ++j) {
      const uint8_t* view =
          state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_VIEWS].data +
          (i * IREE_ARRAYSIZE(bindings) + j) * 16;
      const uint64_t origin = iree_unaligned_load_le_u64(view);
      const uint64_t length = iree_unaligned_load_le_u64(view + 8);
      iree_hal_buffer_t* replacement = NULL;
      if (length) {
        status = iree_hal_buffer_subspan(state->row_arena, origin, length,
                                         state->host_allocator, &replacement);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_buffer_release(row->buffers[bindings[j]]);
        row->buffers[bindings[j]] = replacement;
      }
    }
    if (row->recurrent.slot != UINT32_MAX) {
      row->buffers[TEXT_RECURRENT] =
          state->recurrent.buffers[row->recurrent.slot];
      state->recurrent.dirty_count += !row->recurrent.dirty;
      row->recurrent.dirty = true;
    }
  }
  state->epoch.buffers[3] = state->row_arena;
  return status;
}

iree_status_t loom_serve_text_state_resize(loom_serve_text_state_t* state,
                                           iree_host_size_t row_count,
                                           loom_serve_text_storage_t* storage) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_text_state_validate_resize(state, row_count, storage));
  const bool backed =
      loom_serve_memory_operation_is_complete(state->memory.backing.operation);
  IREE_ASSERT(backed && !state->prepared_reservations);
  loom_serve_memory_operation_free(state->memory.backing.operation);
  state->memory.backing.operation = NULL;
  const iree_host_size_t previous_count = state->row_count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_text_state_resize_directories(
              state, row_count, storage->recurrent.capacity));
  loom_serve_text_storage_deinitialize(&state->storage);
  state->storage = *storage;
  memset(storage, 0, sizeof(*storage));

  iree_hal_transfer_operation_t
      transfers[2 * LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 4];
  iree_host_size_t transfer_count = 0;
  iree_hal_buffer_t* previous[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(previous) && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_t** root = loom_serve_text_state_allocation(state, i);
    const loom_serve_text_allocation_t* record = &state->storage.allocations[i];
    const iree_device_size_t length = record->length;
    // Carry endpoints change index even when the source reserved spare bytes.
    // Give that bank a fresh target so overlapping old/new ranges never alias.
    if (!*root || (i != 6 && length <= iree_hal_buffer_byte_length(*root))) {
      continue;
    }
    iree_hal_buffer_t* replacement = NULL;
    if (state->memory.buffers[i]) {
      status = iree_hal_semaphore_create(
          state->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state->memory.growth.semaphores[i]);
      if (iree_status_is_ok(status)) {
        status = loom_serve_virtual_buffer_grow_async(
            state->memory.buffers[i], length,
            state->memory.growth.semaphores[i], 1,
            &state->memory.growth.operations[i]);
      }
      if (iree_status_is_ok(status)) {
        uint64_t value = 1;
        status = loom_serve_execution_barrier(
            state->execution,
            (iree_hal_semaphore_list_t){1, &state->memory.growth.semaphores[i],
                                        &value});
        replacement =
            loom_serve_virtual_buffer_handle(state->memory.buffers[i]);
        iree_hal_buffer_retain(replacement);
      }
    } else {
      const iree_hal_pool_reservation_request_t request =
          iree_hal_make_pool_reservation_request(length, record->alignment);
      status = loom_serve_execution_alloca_from_pool(
          state->execution, loom_serve_pool_storage(state->pool), 1, &request,
          &replacement);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_retirement_track(&state->retirement, &replacement,
                                           state->host_allocator);
    }
    if (iree_status_is_ok(status)) {
      previous[i] = *root;
      *root = replacement;
      replacement = NULL;
      if (i == 6) {
        // Endpoint carry follows the resident bank, so only that small tail
        // changes offset. The large recurrent arena remains append-only.
        const uint64_t offset = previous_count * state->storage.carry_stride;
        transfers[transfer_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy = {.source_buffer = previous[i],
                     .target_buffer = *root,
                     .length = offset}};
        if (state->checkpoints.capacity) {
          transfers[transfer_count++] = (iree_hal_transfer_operation_t){
              .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
              .copy = {.source_buffer = previous[i],
                       .source_offset = offset,
                       .target_buffer = *root,
                       .target_offset = row_count * state->storage.carry_stride,
                       .length = state->checkpoints.capacity *
                                 state->storage.carry_stride}};
        }
      } else if (!state->memory.buffers[i] && i != 3 && i != 10) {
        transfers[transfer_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy = {.source_buffer = previous[i],
                     .target_buffer = *root,
                     .length = iree_hal_buffer_byte_length(previous[i])}};
      }
    }
    iree_hal_buffer_release(replacement);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t slots[] = {3, 10};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
      iree_hal_buffer_t* table =
          *loom_serve_text_state_allocation(state, slots[i]);
      const iree_const_byte_span_t origins =
          state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS + i];
      if (table && origins.data_length) {
        transfers[transfer_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
            .upload = {.source = origins.data,
                       .target_buffer = table,
                       .length = origins.data_length}};
      }
      if (table && state->cache.capacity) {
        transfers[transfer_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
            .upload = {.source = state->cache.maps,
                       .target_buffer = table,
                       .target_offset = state->storage.map_origin,
                       .length = previous_count *
                                 state->storage.blocks_per_row *
                                 sizeof(*state->cache.maps)}};
      }
    }
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(state->execution, transfer_count,
                                           transfers, &completion);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(previous); ++i) {
    iree_hal_buffer_release(previous[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_state_resize_views(state);
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

void loom_serve_text_state_complete_resize(loom_serve_text_state_t* state) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    const bool complete = loom_serve_memory_operation_is_complete(
        state->memory.growth.operations[i]);
    IREE_ASSERT(complete);
    loom_serve_memory_operation_free(state->memory.growth.operations[i]);
    state->memory.growth.operations[i] = NULL;
    iree_hal_semaphore_release(state->memory.growth.semaphores[i]);
    state->memory.growth.semaphores[i] = NULL;
  }
  loom_serve_text_storage_release_payloads(&state->storage);
}
