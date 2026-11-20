// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/execution.h"
#include "loom/runtime/host/host.h"
#include "loom/runtime/storage/snapshot.h"
#include "loom/serve/service/text/state.h"

void loom_serve_text_state_initialize(const loom_serve_text_options_t* options,
                                      iree_allocator_t host_allocator,
                                      loom_serve_text_state_t* out_state) {
  *out_state = (loom_serve_text_state_t){
      .host_allocator = host_allocator,
      .flags =
          (options->span_capacity ? LOOM_SERVE_TEXT_STATE_FLAG_PACKED : 0) |
          (options->draft_depth ? LOOM_SERVE_TEXT_STATE_FLAG_DRAFT : 0),
      .row_count = options->row_count,
      .cache.capacity = options->pool_capacity,
      .checkpoints.capacity = (uint32_t)options->checkpoint_capacity,
  };
  loom_runtime_retirement_initialize(&out_state->retirement);
}

// Source allocation slots name the same exported roots during creation,
// maintenance and execution. Every root carries state retirement tracking.
iree_hal_buffer_t** loom_serve_text_state_allocation(
    loom_serve_text_state_t* state, iree_host_size_t index) {
  iree_hal_buffer_t** allocations[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {
      &state->residual,
      &state->row_arena,
      &state->epoch.buffers[1],
      &state->epoch.buffers[2],
      &state->epoch.buffers[4],
      &state->epoch.buffers[5],
      &state->carry,
      &state->draft.committed,
      &state->draft.results,
      &state->draft.cache,
      &state->draft.row_table,
  };
  return allocations[index];
}

iree_status_t loom_serve_text_state_allocate(
    loom_serve_text_state_t* state, loom_runtime_device_t* device,
    loom_runtime_pool_t* pool, loom_runtime_execution_t* execution) {
  if (state->storage.carry_stride &&
      !iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source carry requires packed epochs");
  }
  state->device = loom_runtime_device_handle(device);
  state->pool = pool;
  iree_arena_initialize(
      loom_runtime_host_block_pool(loom_runtime_device_host(device)),
      &state->row_records);
  state->execution = execution;
  state->memory.pool = loom_runtime_pool_memory(pool);
  if (state->memory.pool) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
        state->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state->memory.backing.semaphore));
  }
  if (state->cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_runtime_block_pool_initialize(
        0, state->host_allocator, &state->cache.pool));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        state->host_allocator, state->row_count,
        state->storage.blocks_per_row * sizeof(*state->cache.maps),
        (void**)&state->cache.maps));
  }
  // Binding roles are the private adapter contract. Sizes, views, initial
  // contents and zero extents are produced by the state's source bootstrap.
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t
      transfers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 2] = {0};
  union {
    // Geometry for initialization fills.
    iree_hal_fill_region_t fill;
    // Geometry for initialization uploads.
    iree_hal_transfer_region_t transfer;
  } transfers_regions[IREE_ARRAYSIZE(transfers)];
  iree_host_size_t transfer_count = 0;
  iree_hal_pool_reservation_request_t
      requests[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {0};
  iree_hal_buffer_t** targets[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_device_size_t clear_lengths[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {
      0};
  loom_runtime_memory_range_t
      commitments[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_host_size_t allocation_count = 0;
  iree_host_size_t commitment_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_t** allocation = loom_serve_text_state_allocation(state, i);
    const loom_serve_text_allocation_t* record = &state->storage.allocations[i];
    const iree_device_size_t length = record->length;
    const iree_device_size_t alignment = record->alignment;
    const iree_device_size_t clear_length = record->clear_length;
    // Carry belongs to the source's retained state, not the drafting policy.
    // Speculative feedback and optional draft cache roots still require
    // drafting.
    const bool required =
        i < 2 ||
        (i < 6
             ? iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)
         : i == 6
             ? state->storage.carry_stride != 0
             : i < 9 && iree_any_bit_set(state->flags,
                                         LOOM_SERVE_TEXT_STATE_FLAG_DRAFT));
    const bool optional =
        i >= 9 &&
        iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT);
    if ((length != 0) != required && !optional) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid source allocation role %zu", i);
    } else if (length) {
      if (state->memory.pool && record->cache) {
        const loom_runtime_memory_geometry_t geometry =
            loom_runtime_pool_state_geometry(pool);
        status = loom_runtime_virtual_buffer_allocate(
            state->memory.pool, geometry,
            iree_device_align(length, geometry.mapping_size), alignment,
            &state->memory.statistics, &state->memory.buffers[i]);
        if (iree_status_is_ok(status)) {
          *allocation =
              loom_runtime_virtual_buffer_handle(state->memory.buffers[i]);
          iree_hal_buffer_retain(*allocation);
          status = loom_runtime_retirement_track(&state->retirement, allocation,
                                                 state->host_allocator);
          if (iree_status_is_ok(status) && clear_length && i != 1) {
            state->memory.private_lengths[i] = clear_length;
            commitments[commitment_count++] = (loom_runtime_memory_range_t){
                state->memory.buffers[i], 0, clear_length};
          }
        }
      } else {
        requests[allocation_count] =
            iree_hal_make_pool_reservation_request(length, alignment);
        targets[allocation_count++] = allocation;
      }
    }
    if (iree_status_is_ok(status) && clear_length &&
        !(i == 1 && state->memory.buffers[i])) {
      clear_lengths[i] = clear_length;
    }
  }
  if (iree_status_is_ok(status) && allocation_count) {
    status = loom_runtime_execution_alloca_from_pool(
        execution, loom_runtime_pool_storage(pool), allocation_count, requests,
        buffers);
    for (iree_host_size_t i = 0; i < allocation_count; ++i) {
      *targets[i] = buffers[i];
      if (iree_status_is_ok(status)) {
        status = loom_runtime_retirement_track(&state->retirement, targets[i],
                                               state->host_allocator);
      }
    }
  }
  if (iree_status_is_ok(status) && commitment_count) {
    loom_runtime_memory_admission_t admission;
    status = loom_runtime_memory_pool_commit_async(
        state->memory.pool, commitment_count, commitments,
        state->memory.backing.semaphore, state->memory.backing.value + 1,
        &admission, &state->memory.backing.operation);
    if (iree_status_is_ok(status) &&
        admission != LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "physical pool cannot back private text state");
    }
    if (iree_status_is_ok(status)) {
      ++state->memory.backing.value;
      status = loom_runtime_execution_barrier(
          execution,
          (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                      &state->memory.backing.value});
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
         ++i) {
      if (clear_lengths[i]) {
        transfers_regions[transfer_count].fill = (iree_hal_fill_region_t){
            .row_length = clear_lengths[i], .row_count = 1, .slice_count = 1};
        transfers[transfer_count] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill = {.target_buffer =
                         *loom_serve_text_state_allocation(state, i),
                     .region = &transfers_regions[transfer_count].fill,
                     .pattern = &zero,
                     .pattern_length = sizeof(zero)},
        };
        ++transfer_count;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_block_pool_initialize(
        state->storage.recurrent.capacity, state->host_allocator,
        &state->recurrent.pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        state->host_allocator, state->recurrent.pool.capacity,
        sizeof(*state->recurrent.buffers), (void**)&state->recurrent.buffers);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(state->host_allocator, state->row_count,
                                    sizeof(*state->rows), (void**)&state->rows);
  }
  for (iree_host_size_t i = 0;
       i < state->row_count && iree_status_is_ok(status); ++i) {
    status = iree_arena_allocate(&state->row_records, sizeof(**state->rows),
                                 (void**)&state->rows[i]);
    if (iree_status_is_ok(status)) {
      *state->rows[i] = (loom_serve_text_state_row_t){
          .owner = state,
          .index = i,
          .recurrent = {.slot = UINT32_MAX, .anchor = UINT32_MAX},
          .retiring_tail = UINT32_MAX};
    }
  }
  if (iree_status_is_ok(status) && state->cache.capacity) {
    status =
        iree_allocator_malloc_array(state->host_allocator, state->row_count,
                                    2 * (sizeof(*state->cache.uploads) +
                                         sizeof(iree_hal_transfer_region_t)),
                                    (void**)&state->cache.uploads);
  }
  if (iree_status_is_ok(status) && state->checkpoints.capacity) {
    status = loom_runtime_block_pool_initialize(state->checkpoints.capacity,
                                                state->host_allocator,
                                                &state->checkpoints.pool);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(state->host_allocator,
                                           state->checkpoints.capacity,
                                           sizeof(*state->checkpoints.values),
                                           (void**)&state->checkpoints.values);
    }
    for (uint32_t i = 0;
         i < state->checkpoints.capacity && iree_status_is_ok(status); ++i) {
      state->checkpoints.values[i].owner = state;
      state->checkpoints.values[i].recurrent_slot = UINT32_MAX;
    }
  }
  for (iree_host_size_t i = 0;
       i < state->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_state_row_t* row = state->rows[i];
    row->buffers[TEXT_RESIDUAL] = state->residual;
    const uint32_t bindings[] = {TEXT_CONTROL, TEXT_ATTENTION, TEXT_TOKENS,
                                 TEXT_PROGRESS};
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(bindings) && iree_status_is_ok(status); ++j) {
      const uint8_t* view =
          state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_VIEWS].data +
          (i * IREE_ARRAYSIZE(bindings) + j) * 16;
      const uint64_t offset = iree_unaligned_load_le_u64(view);
      const uint64_t length = iree_unaligned_load_le_u64(view + 8);
      if (length) {
        status = iree_hal_buffer_subspan(state->row_arena, offset, length,
                                         state->host_allocator,
                                         &row->buffers[bindings[j]]);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)) {
    state->epoch.buffers[0] = state->residual;
    state->epoch.buffers[3] = state->row_arena;
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS];
    if (origins.data_length) {
      transfers_regions[transfer_count].transfer = (iree_hal_transfer_region_t){
          .row_length = origins.data_length, .row_count = 1, .slice_count = 1};
      transfers[transfer_count] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->epoch.buffers[2],
                     .region = &transfers_regions[transfer_count].transfer},
      };
      ++transfer_count;
    }
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)) {
    status = iree_hal_buffer_subspan(
        state->draft.results, state->storage.feedback_split,
        IREE_HAL_WHOLE_BUFFER, state->host_allocator,
        &state->draft.next_results);
  }
  if (iree_status_is_ok(status) && state->draft.row_table) {
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS];
    if (origins.data_length) {
      transfers_regions[transfer_count].transfer = (iree_hal_transfer_region_t){
          .row_length = origins.data_length, .row_count = 1, .slice_count = 1};
      transfers[transfer_count] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->draft.row_table,
                     .region = &transfers_regions[transfer_count].transfer},
      };
      ++transfer_count;
    }
  }
  if (iree_status_is_ok(status)) {
    uint64_t completion = 0;
    status = loom_runtime_execution_transfer(state->execution, transfer_count,
                                             transfers, &completion);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_state_backing_initialize(state);
  }
  return status;
}

void loom_serve_text_state_retire(loom_serve_text_state_t* state,
                                  void (*wake)(void* user_data),
                                  void* user_data) {
  loom_runtime_retirement_set_observer(&state->retirement, wake, user_data);
  loom_runtime_memory_operation_set_observer(state->memory.backing.operation,
                                             wake, user_data);
  loom_runtime_memory_operation_set_observer(state->memory.reclaim, wake,
                                             user_data);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    loom_runtime_memory_operation_set_observer(
        state->memory.growth.operations[i], wake, user_data);
  }
  if (state->rows) {
    for (iree_host_size_t i = 0; i < state->row_count; ++i) {
      if (!state->rows[i]) {
        continue;
      }
      loom_runtime_snapshot_release(state->rows[i]->snapshot);
      state->rows[i]->snapshot = NULL;
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        if (binding != TEXT_RECURRENT) {
          iree_hal_buffer_release(state->rows[i]->buffers[binding]);
        }
        state->rows[i]->buffers[binding] = NULL;
      }
    }
  }
  if (state->recurrent.buffers) {
    for (uint32_t i = 0; i < state->recurrent.pool.capacity; ++i) {
      iree_hal_buffer_release(state->recurrent.buffers[i]);
      state->recurrent.buffers[i] = NULL;
    }
  }
  if (state->checkpoints.values) {
    for (uint32_t i = 0; i < state->checkpoints.capacity; ++i) {
      loom_runtime_snapshot_release(state->checkpoints.values[i].snapshot);
      state->checkpoints.values[i].snapshot = NULL;
    }
  }
  iree_hal_buffer_release(state->draft.next_results);
  state->draft.next_results = NULL;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    iree_hal_buffer_release(*loom_serve_text_state_allocation(state, i));
    *loom_serve_text_state_allocation(state, i) = NULL;
  }
  loom_runtime_storage_release(&state->declared_storage);
}

bool loom_serve_text_state_is_retired(loom_serve_text_state_t* state) {
  bool complete = loom_runtime_memory_operation_is_complete(
      state->memory.backing.operation);
  complete &= loom_runtime_memory_operation_is_complete(state->memory.reclaim);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    complete &= loom_runtime_memory_operation_is_complete(
        state->memory.growth.operations[i]);
  }
  return complete && loom_runtime_retirement_is_idle(&state->retirement);
}

void loom_serve_text_state_await_retirement(loom_serve_text_state_t* state) {
  loom_runtime_memory_operation_await(state->memory.backing.operation);
  loom_runtime_memory_operation_await(state->memory.reclaim);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    loom_runtime_memory_operation_await(state->memory.growth.operations[i]);
  }
  loom_runtime_retirement_await(&state->retirement);
}

iree_status_t loom_serve_text_state_deinitialize(
    loom_serve_text_state_t* state) {
  IREE_ASSERT(loom_serve_text_state_is_retired(state));
  IREE_RETURN_IF_ERROR(loom_serve_text_state_collect_reclaim(state));
  loom_runtime_memory_operation_free(state->memory.backing.operation);
  state->memory.backing.operation = NULL;
  loom_serve_text_state_complete_resize(state);
  iree_hal_semaphore_release(state->memory.backing.semaphore);
  state->memory.backing.semaphore = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    status = loom_runtime_virtual_buffer_free(state->memory.buffers[i]);
    if (iree_status_is_ok(status)) {
      state->memory.buffers[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_runtime_retirement_deinitialize(&state->retirement);
    loom_runtime_storage_deinitialize(&state->declared_storage,
                                      state->host_allocator);
    loom_serve_text_state_backing_deinitialize(state);
    loom_serve_text_storage_deinitialize(&state->storage);
    iree_allocator_free(state->host_allocator, state->recurrent.buffers);
    loom_runtime_block_pool_deinitialize(&state->recurrent.pool);
    loom_runtime_block_pool_deinitialize(&state->checkpoints.pool);
    for (uint32_t i = 0;
         state->checkpoints.values && i < state->checkpoints.capacity; ++i) {
      iree_allocator_free(state->host_allocator,
                          state->checkpoints.values[i].blocks);
    }
    iree_allocator_free(state->host_allocator, state->checkpoints.values);
    iree_allocator_free(state->host_allocator, state->cache.destinations);
    loom_runtime_block_pool_deinitialize(&state->cache.pool);
    iree_allocator_free(state->host_allocator, state->rows);
    iree_arena_deinitialize(&state->row_records);
    iree_allocator_free(state->host_allocator, state->cache.maps);
    iree_allocator_free(state->host_allocator, state->cache.uploads);
  }
  return status;
}
