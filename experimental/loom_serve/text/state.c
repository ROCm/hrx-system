// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/state.h"

#include <string.h>

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/storage/relocation.h"
#include "experimental/loom_serve/storage/snapshot.h"

void loom_serve_text_state_initialize(const loom_serve_text_options_t* options,
                                      loom_serve_text_state_t* out_state,
                                      iree_allocator_t allocator) {
  *out_state = (loom_serve_text_state_t){
      .allocator = allocator,
      .flags = (options->epoch_count ? LOOM_SERVE_TEXT_STATE_FLAG_PACKED : 0) |
               (options->enable_mtp ? LOOM_SERVE_TEXT_STATE_FLAG_MTP : 0),
      .row_count = options->row_count,
      .cache.capacity = options->pool_capacity,
      .checkpoints.capacity = (uint32_t)options->checkpoint_capacity,
  };
  loom_serve_retirement_initialize(&out_state->retirement);
}

static iree_status_t text_allocate_buffer(loom_serve_text_state_t* state,
                                          iree_device_size_t length,
                                          iree_device_size_t minimum_alignment,
                                          iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = minimum_alignment;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(state->device), params, length, out_buffer));
  return loom_serve_retirement_track(&state->retirement, out_buffer,
                                     state->allocator);
}

// Source allocation slots name the same exported roots during creation,
// maintenance and execution. Every root carries state retirement tracking.
static iree_hal_buffer_t** text_allocation(loom_serve_text_state_t* state,
                                           iree_host_size_t index) {
  iree_hal_buffer_t** allocations[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = {
      &state->residual,         &state->row_arena,
      &state->epoch.buffers[1], &state->epoch.buffers[2],
      &state->epoch.buffers[4], &state->epoch.buffers[5],
      &state->mtp.carry,        &state->mtp.committed,
      &state->mtp.results,      &state->mtp.cache,
      &state->mtp.row_table,
  };
  return allocations[index];
}

iree_status_t loom_serve_text_state_allocate(
    loom_serve_text_state_t* state, loom_serve_device_t* device,
    iree_device_size_t workspace_length,
    iree_device_size_t workspace_alignment) {
  state->device = loom_serve_device_handle(device);
  state->execution = loom_serve_device_execution(device);
  state->memory.pool = loom_serve_device_memory_pool(device);
  if (state->cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_block_pool_initialize(
        (uint32_t)(state->cache.capacity / state->storage.block_size),
        state->allocator, &state->cache.pool));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        state->allocator, state->row_count * state->storage.blocks_per_row,
        sizeof(*state->cache.maps), (void**)&state->cache.maps));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        state->allocator, state->cache.pool.capacity,
        sizeof(*state->cache.destinations),
        (void**)&state->cache.destinations));
  }
  // Workspace contents are undefined on entry. Retained state belongs to the
  // source-declared allocations below, never to command transient storage.
  IREE_RETURN_IF_ERROR(text_allocate_buffer(
      state, workspace_length, workspace_alignment, &state->workspace));
  // Binding roles are the private adapter contract. Sizes, views, initial
  // contents and zero extents are produced by the state's source bootstrap.
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t
      transfers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 2] = {0};
  iree_host_size_t transfer_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_t** allocation = text_allocation(state, i);
    const uint8_t* record =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data + i * 24;
    const uint64_t length = iree_unaligned_load_le_u64(record);
    const uint64_t alignment = iree_unaligned_load_le_u64(record + 8);
    const uint64_t clear_length = iree_unaligned_load_le_u64(record + 16);
    const bool required =
        i < 2 ||
        (i < 6
             ? iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)
             : iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP));
    if ((length != 0) != required || clear_length > length ||
        length > INT64_MAX || !alignment ||
        !iree_is_power_of_two_uint64(alignment)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid source allocation record %zu", i);
    } else if (length) {
      bool elastic = false;
      for (iree_host_size_t j = 0; j < state->storage.region_count; ++j) {
        elastic |= state->storage.regions[j].allocation == i;
      }
      if (state->memory.pool && state->cache.capacity && elastic) {
        status = loom_serve_virtual_buffer_create(
            state->memory.pool, length, alignment, &state->memory.statistics,
            &state->memory.buffers[i]);
        if (iree_status_is_ok(status)) {
          *allocation =
              loom_serve_virtual_buffer_handle(state->memory.buffers[i]);
          iree_hal_buffer_retain(*allocation);
          status = loom_serve_retirement_track(&state->retirement, allocation,
                                               state->allocator);
          if (iree_status_is_ok(status) && clear_length && i != 1) {
            state->memory.private_lengths[i] = clear_length;
            status = loom_serve_virtual_buffer_commit(state->memory.buffers[i],
                                                      0, clear_length);
          }
        }
      } else {
        status = text_allocate_buffer(state, length, alignment, allocation);
      }
    }
    if (iree_status_is_ok(status) && clear_length &&
        !(i == 1 && state->memory.buffers[i])) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {.target_buffer = *allocation,
                   .length = clear_length,
                   .pattern = &zero,
                   .pattern_length = sizeof(zero)},
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_block_pool_initialize(state->storage.recurrent.capacity,
                                              state->allocator,
                                              &state->recurrent.pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        state->allocator, state->recurrent.pool.capacity,
        sizeof(*state->recurrent.buffers), (void**)&state->recurrent.buffers);
  }
  for (uint32_t i = 0;
       i < state->recurrent.pool.capacity && iree_status_is_ok(status); ++i) {
    status = iree_hal_buffer_subspan(
        state->row_arena,
        state->storage.recurrent.origin + i * state->storage.recurrent.stride,
        state->storage.recurrent.length, state->allocator,
        &state->recurrent.buffers[i]);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(state->allocator, state->row_count,
                                    sizeof(*state->rows), (void**)&state->rows);
  }
  if (iree_status_is_ok(status) && state->checkpoints.capacity) {
    status = loom_serve_block_pool_initialize(state->checkpoints.capacity,
                                              state->allocator,
                                              &state->checkpoints.pool);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(state->allocator,
                                           state->checkpoints.capacity,
                                           sizeof(*state->checkpoints.values),
                                           (void**)&state->checkpoints.values);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          state->allocator,
          state->checkpoints.capacity * state->storage.blocks_per_row,
          sizeof(*state->checkpoints.maps), (void**)&state->checkpoints.maps);
    }
    for (uint32_t i = 0;
         i < state->checkpoints.capacity && iree_status_is_ok(status); ++i) {
      state->checkpoints.values[i].owner = state;
      state->checkpoints.values[i].blocks =
          state->checkpoints.maps + i * state->storage.blocks_per_row;
    }
  }
  for (iree_host_size_t i = 0;
       i < state->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_state_row_t* row = &state->rows[i];
    row->owner = state;
    row->recurrent.slot = UINT32_MAX;
    row->recurrent.anchor = UINT32_MAX;
    row->retiring_tail = UINT32_MAX;
    row->buffers[TEXT_RESIDUAL] = state->residual;
    row->buffers[TEXT_WORKSPACE] = state->workspace;
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
                                         state->allocator,
                                         &row->buffers[bindings[j]]);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_PACKED)) {
    state->epoch.buffers[0] = state->residual;
    state->epoch.buffers[3] = state->row_arena;
    state->epoch.buffers[6] = state->workspace;
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS];
    if (origins.data_length) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->epoch.buffers[2],
                     .length = origins.data_length},
      };
    }
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP)) {
    status = iree_hal_buffer_subspan(
        state->mtp.results, state->storage.feedback_split,
        IREE_HAL_WHOLE_BUFFER, state->allocator, &state->mtp.next_results);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP)) {
    const iree_const_byte_span_t origins =
        state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS];
    if (origins.data_length) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = origins.data,
                     .target_buffer = state->mtp.row_table,
                     .length = origins.data_length},
      };
    }
  }
  if (iree_status_is_ok(status)) {
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(state->execution, transfer_count,
                                           transfers, &completion);
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(state->execution, completion);
    }
  }
  return status;
}

iree_status_t loom_serve_text_state_deinitialize(
    loom_serve_text_state_t* state) {
  if (state->rows) {
    for (iree_host_size_t i = 0; i < state->row_count; ++i) {
      loom_serve_snapshot_destroy(state->rows[i].snapshot);
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        if (binding != TEXT_RECURRENT) {
          iree_hal_buffer_release(state->rows[i].buffers[binding]);
        }
      }
    }
  }
  if (state->recurrent.buffers) {
    for (uint32_t i = 0; i < state->recurrent.pool.capacity; ++i) {
      iree_hal_buffer_release(state->recurrent.buffers[i]);
    }
  }
  iree_allocator_free(state->allocator, state->recurrent.buffers);
  loom_serve_block_pool_deinitialize(&state->recurrent.pool);
  loom_serve_block_pool_deinitialize(&state->checkpoints.pool);
  iree_allocator_free(state->allocator, state->checkpoints.values);
  iree_allocator_free(state->allocator, state->checkpoints.maps);
  iree_allocator_free(state->allocator, state->cache.destinations);
  loom_serve_block_pool_deinitialize(&state->cache.pool);
  iree_hal_buffer_release(state->mtp.next_results);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    iree_hal_buffer_release(*text_allocation(state, i));
  }
  iree_hal_buffer_release(state->workspace);
  // Failed readiness does not retire borrowed payloads. Final queue ownership
  // of their device counterparts does, including nested row views.
  loom_serve_retirement_deinitialize(&state->retirement);
  loom_serve_text_storage_deinitialize(&state->storage);
  iree_allocator_free(state->allocator, state->rows);
  iree_allocator_free(state->allocator, state->cache.maps);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    status = iree_status_join(
        status, loom_serve_virtual_buffer_destroy(state->memory.buffers[i]));
  }
  return status;
}

iree_status_t loom_serve_text_state_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result) {
  *out_result = (loom_serve_text_trim_result_t){0};
  if (!state->memory.pool) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(state->execution));
  const uint64_t released_before =
      loom_serve_memory_pool_statistics(state->memory.pool).released_bytes;
  out_result->moved_blocks = loom_serve_block_pool_plan_compaction(
      &state->cache.pool, state->cache.destinations);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       out_result->moved_blocks && i < state->storage.region_count &&
       iree_status_is_ok(status);
       ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    uint64_t copied_bytes = 0;
    status = loom_serve_block_region_relocate(
        state->execution, state->memory.buffers[region->allocation],
        *text_allocation(state, region->allocation), &region->blocks,
        state->cache.pool.capacity, state->cache.destinations, &copied_bytes);
    out_result->copied_bytes += copied_bytes;
  }
  // Even a rejected later copy batch leaves earlier submissions owning their
  // sources/destinations. Observe retirement before publishing or returning.
  status =
      iree_status_join(status, loom_serve_execution_drain(state->execution));
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    for (iree_host_size_t i = 0; i < state->row_count; ++i) {
      uint32_t* blocks = state->cache.maps + i * state->storage.blocks_per_row;
      for (uint32_t j = 0; j < state->rows[i].block_count; ++j) {
        blocks[j] = state->cache.destinations[blocks[j]];
      }
    }
    for (uint32_t i = 0; i < state->checkpoints.pool.capacity; ++i) {
      if (!state->checkpoints.pool.references[i]) {
        continue;
      }
      loom_serve_text_state_checkpoint_t* checkpoint =
          &state->checkpoints.values[i];
      for (uint32_t j = 0; j < checkpoint->block_count; ++j) {
        checkpoint->blocks[j] =
            state->cache.destinations[checkpoint->blocks[j]];
      }
    }
    const iree_device_size_t length = state->row_count *
                                      state->storage.blocks_per_row *
                                      sizeof(*state->cache.maps);
    const iree_hal_transfer_operation_t uploads[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = state->cache.maps,
                    .target_buffer = state->epoch.buffers[2],
                    .target_offset = state->storage.map_origin,
                    .length = length}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = state->cache.maps,
                    .target_buffer = state->mtp.row_table,
                    .target_offset = state->storage.map_origin,
                    .length = length}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        state->execution,
        iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP) ? 2 : 1,
        uploads, &completion);
    status =
        iree_status_join(status, loom_serve_execution_drain(state->execution));
  }
  if (iree_status_is_ok(status)) {
    loom_serve_block_pool_commit_compaction(&state->cache.pool,
                                            state->cache.destinations);
    for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
         ++i) {
      if (!state->memory.buffers[i]) {
        continue;
      }
      loom_serve_virtual_buffer_begin_trim(state->memory.buffers[i]);
      if (state->memory.private_lengths[i]) {
        loom_serve_virtual_buffer_keep(state->memory.buffers[i], 0,
                                       state->memory.private_lengths[i]);
      }
    }
    for (iree_host_size_t i = 0;
         state->memory.buffers[1] && i < state->row_count; ++i) {
      if (!state->rows[i].position || state->rows[i].snapshot) {
        continue;
      }
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        iree_hal_buffer_t* view = state->rows[i].buffers[binding];
        if (view) {
          loom_serve_virtual_buffer_keep(state->memory.buffers[1],
                                         iree_hal_buffer_byte_offset(view),
                                         iree_hal_buffer_byte_length(view));
        }
      }
    }
    const uint32_t live =
        state->cache.pool.capacity - state->cache.pool.available;
    for (uint32_t i = 0;
         state->memory.buffers[1] && i < state->checkpoints.pool.capacity;
         ++i) {
      if (!state->checkpoints.pool.references[i]) {
        continue;
      }
      iree_hal_buffer_t* view =
          state->recurrent.buffers[state->checkpoints.values[i].recurrent_slot];
      loom_serve_virtual_buffer_keep(state->memory.buffers[1],
                                     iree_hal_buffer_byte_offset(view),
                                     iree_hal_buffer_byte_length(view));
    }
    for (iree_host_size_t i = 0; live && i < state->storage.region_count; ++i) {
      const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
      for (iree_host_size_t plane = 0; plane < region->blocks.count; ++plane) {
        loom_serve_virtual_buffer_keep(
            state->memory.buffers[region->allocation],
            region->blocks.origin + plane * region->blocks.stride,
            live * region->blocks.block_bytes);
      }
    }
  }
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    if (state->memory.buffers[i]) {
      status = loom_serve_virtual_buffer_trim(state->memory.buffers[i]);
    }
  }
  out_result->released_bytes =
      loom_serve_memory_pool_statistics(state->memory.pool).released_bytes -
      released_before;
  return status;
}

// Only a retired frontier may release pages. A rejected suffix within the
// retained last page stays private; subsequent appends overwrite it before use.
void loom_serve_text_state_row_trim(loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  if (row->recurrent.anchor != UINT32_MAX) {
    loom_serve_block_pool_release(&state->recurrent.pool, 1,
                                  &row->recurrent.anchor);
    row->recurrent.anchor = UINT32_MAX;
  }
  if (row->retiring_tail != UINT32_MAX) {
    loom_serve_block_pool_release(&state->cache.pool, 1, &row->retiring_tail);
    row->retiring_tail = UINT32_MAX;
  }
  if (!state->cache.capacity) {
    return;
  }
  const uint32_t keep =
      (uint32_t)((row->position + state->storage.block_size - 1) /
                 state->storage.block_size);
  const iree_host_size_t row_index = (iree_host_size_t)(row - state->rows);
  loom_serve_block_pool_release(
      &state->cache.pool, row->block_count - keep,
      state->cache.maps + row_index * state->storage.blocks_per_row + keep);
  row->block_count = keep;
}

static void text_row_acquire_recurrent(loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  loom_serve_block_pool_acquire(&state->recurrent.pool, 1,
                                &row->recurrent.slot);
  row->buffers[TEXT_RECURRENT] = state->recurrent.buffers[row->recurrent.slot];
  state->recurrent.dirty_rows |= 1u << (row - state->rows);
}

static void text_row_release_recurrent(loom_serve_text_state_row_t* row) {
  if (row->recurrent.slot == UINT32_MAX) {
    return;
  }
  loom_serve_text_state_t* state = row->owner;
  loom_serve_block_pool_release(&state->recurrent.pool, 1,
                                &row->recurrent.slot);
  row->recurrent.slot = UINT32_MAX;
  row->buffers[TEXT_RECURRENT] = NULL;
  state->recurrent.dirty_rows &= ~(1u << (row - state->rows));
}

iree_status_t loom_serve_text_state_row_activate(
    loom_serve_text_state_row_t* row) {
  if (row->recurrent.slot != UINT32_MAX) {
    return iree_ok_status();
  }
  loom_serve_text_state_t* state = row->owner;
  text_row_acquire_recurrent(row);
  if (state->memory.buffers[1]) {
    for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
         ++binding) {
      iree_hal_buffer_t* view = row->buffers[binding];
      if (view) {
        IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_commit(
            state->memory.buffers[1], iree_hal_buffer_byte_offset(view),
            iree_hal_buffer_byte_length(view)));
      }
    }
  }
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t fills[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_CONTROL],
                .length =
                    iree_hal_buffer_byte_length(row->buffers[TEXT_CONTROL]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_RECURRENT],
                .length =
                    iree_hal_buffer_byte_length(row->buffers[TEXT_RECURRENT]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_TOKENS],
                .length =
                    iree_hal_buffer_byte_length(row->buffers[TEXT_TOKENS]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_PROGRESS],
                .length = 8 * sizeof(int32_t),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      state->execution, IREE_ARRAYSIZE(fills), fills, &completion);
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(row->owner->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP)) {
    const iree_hal_transfer_operation_t clear_carry = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = row->owner->mtp.carry,
                 .target_offset = (uint64_t)(row - row->owner->rows) *
                                  row->owner->storage.carry_stride,
                 .length = row->owner->storage.carry_stride,
                 .pattern = &zero,
                 .pattern_length = sizeof(zero)},
    };
    status = loom_serve_execution_transfer(row->owner->execution, 1,
                                           &clear_carry, &completion);
  }
  return status;
}

iree_status_t loom_serve_text_state_row_reset(
    loom_serve_text_state_row_t* row) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(row->owner->execution));
  loom_serve_snapshot_destroy(row->snapshot);
  row->snapshot = NULL;
  row->position = 0;
  loom_serve_text_state_row_trim(row);
  text_row_release_recurrent(row);
  return iree_ok_status();
}

iree_status_t loom_serve_text_state_row_suspend(
    loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  if (!state->memory.buffers[1]) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session suspension requires elastic pooled state");
  }
  if (row->snapshot || !row->position) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(state->execution));
  const iree_host_size_t index = (iree_host_size_t)(row - state->rows);
  uint32_t* blocks = state->cache.maps + index * state->storage.blocks_per_row;
  iree_host_size_t range_count = 0;
  loom_serve_snapshot_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(
      &state->storage, index, &row->buffers[TEXT_CONTROL],
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP),
      row->block_count, blocks, &range_count, &ranges, state->allocator));
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = *text_allocation(state, i);
  }
  iree_status_t status = loom_serve_snapshot_capture(
      state->execution, IREE_ARRAYSIZE(buffers), buffers, range_count, ranges,
      &row->snapshot, state->allocator);
  if (iree_status_is_ok(status)) {
    loom_serve_block_pool_release(&state->cache.pool, row->block_count, blocks);
    row->block_count = 0;
    text_row_release_recurrent(row);
  }
  iree_allocator_free(state->allocator, ranges);
  return status;
}

iree_status_t loom_serve_text_state_row_try_resume(
    loom_serve_text_state_row_t* row, bool* out_resumed) {
  *out_resumed = false;
  if (!row->snapshot) {
    *out_resumed = true;
    return iree_ok_status();
  }
  loom_serve_text_state_t* state = row->owner;
  const uint32_t needed =
      (uint32_t)((row->position + state->storage.block_size - 1) /
                 state->storage.block_size);
  if (needed > state->cache.pool.available) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(state->execution));
  const iree_host_size_t index = (iree_host_size_t)(row - state->rows);
  const iree_host_size_t map_index = index * state->storage.blocks_per_row;
  uint32_t* blocks = state->cache.maps + map_index;
  loom_serve_block_pool_acquire(&state->cache.pool, needed, blocks);
  row->block_count = needed;
  text_row_acquire_recurrent(row);
  iree_host_size_t range_count = 0;
  loom_serve_snapshot_range_t* ranges = NULL;
  iree_status_t status = loom_serve_text_storage_plan_snapshot(
      &state->storage, index, &row->buffers[TEXT_CONTROL],
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP), needed,
      blocks, &range_count, &ranges, state->allocator);
  loom_serve_memory_range_t* commitments = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(state->allocator, range_count,
                                    sizeof(*commitments), (void**)&commitments);
  }
  iree_host_size_t commitment_count = 0;
  bool admitted = false;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      const loom_serve_snapshot_range_t* range = &ranges[i];
      loom_serve_virtual_buffer_t* buffer =
          state->memory.buffers[range->buffer_index];
      if (buffer) {
        commitments[commitment_count++] =
            (loom_serve_memory_range_t){buffer, range->offset, range->length};
      }
    }
    status = loom_serve_memory_pool_try_commit(
        state->memory.pool, commitment_count, commitments, &admitted);
  }
  if (iree_status_is_ok(status) && admitted) {
    iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
      buffers[i] = *text_allocation(state, i);
    }
    status = loom_serve_snapshot_restore(row->snapshot, state->execution,
                                         IREE_ARRAYSIZE(buffers), buffers,
                                         range_count, ranges);
  }
  if (iree_status_is_ok(status) && admitted) {
    const iree_hal_transfer_operation_t uploads[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = blocks,
                    .target_buffer = state->epoch.buffers[2],
                    .target_offset = state->storage.map_origin + map_index * 4,
                    .length = needed * sizeof(*blocks)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = blocks,
                    .target_buffer = state->mtp.row_table,
                    .target_offset = state->storage.map_origin + map_index * 4,
                    .length = needed * sizeof(*blocks)}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        state->execution,
        iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP) ? 2 : 1,
        uploads, &completion);
    status =
        iree_status_join(status, loom_serve_execution_drain(state->execution));
  }
  if (iree_status_is_ok(status) && admitted) {
    loom_serve_snapshot_destroy(row->snapshot);
    row->snapshot = NULL;
    *out_resumed = true;
  } else if (iree_status_is_ok(status)) {
    loom_serve_block_pool_release(&state->cache.pool, needed, blocks);
    row->block_count = 0;
    text_row_release_recurrent(row);
  }
  iree_allocator_free(state->allocator, commitments);
  iree_allocator_free(state->allocator, ranges);
  return status;
}

static iree_status_t text_copy_carry(loom_serve_text_state_t* state,
                                     iree_host_size_t source,
                                     iree_host_size_t target) {
  if (!iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP)) {
    return iree_ok_status();
  }
  const iree_hal_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy = {.source_buffer = state->mtp.carry,
               .source_offset = source * state->storage.carry_stride,
               .target_buffer = state->mtp.carry,
               .target_offset = target * state->storage.carry_stride,
               .length = state->storage.carry_stride}};
  uint64_t completion = 0;
  return loom_serve_execution_transfer(state->execution, 1, &copy, &completion);
}

iree_status_t loom_serve_text_state_row_try_pin(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  loom_serve_text_state_t* state = row->owner;
  if (!state->checkpoints.pool.available) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(state->execution));
  uint32_t index = 0;
  loom_serve_block_pool_acquire(&state->checkpoints.pool, 1, &index);
  loom_serve_text_state_checkpoint_t* checkpoint =
      &state->checkpoints.values[index];
  checkpoint->position = row->position;
  checkpoint->recurrent_slot = row->recurrent.slot;
  checkpoint->block_count = row->block_count;
  const iree_host_size_t row_index = (iree_host_size_t)(row - state->rows);
  memcpy(checkpoint->blocks,
         state->cache.maps + row_index * state->storage.blocks_per_row,
         row->block_count * sizeof(*checkpoint->blocks));
  loom_serve_block_pool_retain(&state->cache.pool, checkpoint->block_count,
                               checkpoint->blocks);
  loom_serve_block_pool_retain(&state->recurrent.pool, 1,
                               &checkpoint->recurrent_slot);
  iree_status_t status =
      text_copy_carry(state, row_index, state->row_count + index);
  status =
      iree_status_join(status, loom_serve_execution_drain(state->execution));
  if (iree_status_is_ok(status)) {
    *out_checkpoint = checkpoint;
  }
  return status;
}

void loom_serve_text_state_checkpoint_release(
    loom_serve_text_state_checkpoint_t* checkpoint) {
  loom_serve_text_state_t* state = checkpoint->owner;
  loom_serve_block_pool_release(&state->cache.pool, checkpoint->block_count,
                                checkpoint->blocks);
  loom_serve_block_pool_release(&state->recurrent.pool, 1,
                                &checkpoint->recurrent_slot);
  const uint32_t index = (uint32_t)(checkpoint - state->checkpoints.values);
  loom_serve_block_pool_release(&state->checkpoints.pool, 1, &index);
}

iree_status_t loom_serve_text_state_row_try_restore(
    loom_serve_text_state_row_t* row,
    const loom_serve_text_state_checkpoint_t* checkpoint, bool* out_restored) {
  *out_restored = false;
  loom_serve_text_state_t* state = row->owner;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(state->execution));
  if (state->memory.buffers[1]) {
    loom_serve_memory_range_t ranges[4];
    iree_host_size_t count = 0;
    for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
         ++binding) {
      iree_hal_buffer_t* view = row->buffers[binding];
      if (binding != TEXT_RECURRENT && view) {
        ranges[count++] = (loom_serve_memory_range_t){
            state->memory.buffers[1], iree_hal_buffer_byte_offset(view),
            iree_hal_buffer_byte_length(view)};
      }
    }
    bool admitted = false;
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_try_commit(
        state->memory.pool, count, ranges, &admitted));
    if (!admitted) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_reset(row));
  row->position = checkpoint->position;
  row->recurrent.slot = checkpoint->recurrent_slot;
  row->buffers[TEXT_RECURRENT] = state->recurrent.buffers[row->recurrent.slot];
  loom_serve_block_pool_retain(&state->recurrent.pool, 1, &row->recurrent.slot);
  const iree_host_size_t row_index = (iree_host_size_t)(row - state->rows);
  state->recurrent.dirty_rows |= 1u << row_index;
  row->block_count = checkpoint->block_count;
  const iree_host_size_t map_index = row_index * state->storage.blocks_per_row;
  uint32_t* blocks = state->cache.maps + map_index;
  memcpy(blocks, checkpoint->blocks, row->block_count * sizeof(*blocks));
  loom_serve_block_pool_retain(&state->cache.pool, row->block_count, blocks);
  const iree_host_size_t checkpoint_index =
      (iree_host_size_t)(checkpoint - state->checkpoints.values);
  iree_status_t status =
      text_copy_carry(state, state->row_count + checkpoint_index, row_index);
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = blocks,
                  .target_buffer = state->epoch.buffers[2],
                  .target_offset = state->storage.map_origin + map_index * 4,
                  .length = row->block_count * sizeof(*blocks)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = blocks,
                  .target_buffer = state->mtp.row_table,
                  .target_offset = state->storage.map_origin + map_index * 4,
                  .length = row->block_count * sizeof(*blocks)}},
  };
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_transfer(
        state->execution,
        iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP) ? 2 : 1,
        uploads, &completion);
  }
  status =
      iree_status_join(status, loom_serve_execution_drain(state->execution));
  if (iree_status_is_ok(status)) {
    *out_restored = true;
  }
  return status;
}

static bool text_row_has_shared_tail(const loom_serve_text_state_row_t* row) {
  const loom_serve_text_state_t* state = row->owner;
  if (!row->position || !(row->position % state->storage.block_size)) {
    return false;
  }
  const iree_host_size_t row_index = (iree_host_size_t)(row - state->rows);
  const uint32_t tail =
      state->cache.maps[row_index * state->storage.blocks_per_row +
                        row->block_count - 1];
  return loom_serve_block_pool_is_shared(&state->cache.pool, tail);
}

uint32_t loom_serve_text_state_row_growth(
    const loom_serve_text_state_row_t* row, uint32_t extent) {
  const loom_serve_text_state_t* state = row->owner;
  return (uint32_t)((row->position + extent + state->storage.block_size - 1) /
                    state->storage.block_size) -
         row->block_count + text_row_has_shared_tail(row);
}

// COW copies only one partial logical page, across every source-declared plane.
// The row retains the old page through the existing cohort completion.
static iree_status_t text_fork_cache_block(loom_serve_text_state_t* state,
                                           uint32_t source, uint32_t target) {
  iree_hal_transfer_operation_t copies[64];
  iree_host_size_t count = 0;
  uint64_t completion = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0;
       r < state->storage.region_count && iree_status_is_ok(status); ++r) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[r];
    iree_hal_buffer_t* buffer = *text_allocation(state, region->allocation);
    for (iree_host_size_t plane = 0;
         plane < region->blocks.count && iree_status_is_ok(status); ++plane) {
      const iree_device_size_t origin =
          region->blocks.origin + plane * region->blocks.stride;
      const iree_device_size_t target_offset =
          origin + target * region->blocks.block_bytes;
      if (state->memory.buffers[region->allocation]) {
        status = loom_serve_virtual_buffer_commit(
            state->memory.buffers[region->allocation], target_offset,
            region->blocks.block_bytes);
      }
      if (iree_status_is_ok(status)) {
        copies[count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy = {
                .source_buffer = buffer,
                .source_offset = origin + source * region->blocks.block_bytes,
                .target_buffer = buffer,
                .target_offset = target_offset,
                .length = region->blocks.block_bytes}};
      }
      if (iree_status_is_ok(status) && count == IREE_ARRAYSIZE(copies)) {
        status = loom_serve_execution_transfer(state->execution, count, copies,
                                               &completion);
        count = 0;
      }
    }
  }
  if (iree_status_is_ok(status) && count) {
    status = loom_serve_execution_transfer(state->execution, count, copies,
                                           &completion);
  }
  return status;
}

// The whole epoch has passed its capacity check. Publish only newly assigned
// map entries on the existing ordered transfer path. The fixed host maps stay
// alive until completion, including partial submission failure and destruction.
iree_status_t loom_serve_text_state_grow(loom_serve_text_state_t* state,
                                         iree_host_size_t span_count,
                                         const loom_serve_text_span_t* spans,
                                         const uint32_t* extents) {
  iree_hal_transfer_operation_t uploads[2 * LOOM_SERVE_TEXT_ROW_CAPACITY] = {0};
  iree_host_size_t upload_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    loom_serve_text_state_row_t* row = &state->rows[span->row_index];
    IREE_RETURN_IF_ERROR(loom_serve_text_state_row_activate(row));
    if (loom_serve_block_pool_is_shared(&state->recurrent.pool,
                                        row->recurrent.slot)) {
      row->recurrent.anchor = row->recurrent.slot;
      text_row_acquire_recurrent(row);
      if (state->memory.buffers[1]) {
        IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_commit(
            state->memory.buffers[1],
            iree_hal_buffer_byte_offset(row->buffers[TEXT_RECURRENT]),
            state->storage.recurrent.length));
      }
    }
    if (!state->cache.capacity) {
      continue;
    }
    const uint32_t needed = (uint32_t)((row->position + extents[i] +
                                        state->storage.block_size - 1) /
                                       state->storage.block_size);
    uint32_t first = row->block_count;
    const iree_host_size_t row_map =
        span->row_index * state->storage.blocks_per_row;
    if (text_row_has_shared_tail(row)) {
      --first;
      uint32_t* tail = &state->cache.maps[row_map + first];
      row->retiring_tail = *tail;
      loom_serve_block_pool_acquire(&state->cache.pool, 1, tail);
      IREE_RETURN_IF_ERROR(
          text_fork_cache_block(state, row->retiring_tail, *tail));
    }
    const uint32_t new_count = needed - row->block_count;
    loom_serve_block_pool_acquire(
        &state->cache.pool, new_count,
        state->cache.maps + row_map + row->block_count);
    const uint32_t count = needed - first;
    if (!count) {
      continue;
    }
    const iree_host_size_t map_index = row_map + first;
    uint32_t* blocks = state->cache.maps + map_index;
    row->block_count = needed;
    for (iree_host_size_t j = 0;
         state->memory.pool && j < state->storage.region_count; ++j) {
      const loom_serve_text_cache_region_t* cache_region =
          &state->storage.regions[j];
      const loom_serve_block_region_t* region = &cache_region->blocks;
      for (iree_host_size_t plane = 0; plane < region->count; ++plane) {
        for (iree_host_size_t block = 0; block < count; ++block) {
          IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_commit(
              state->memory.buffers[cache_region->allocation],
              region->origin + plane * region->stride +
                  blocks[block] * region->block_bytes,
              region->block_bytes));
        }
      }
    }
    uploads[upload_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = blocks,
                   .target_buffer = state->epoch.buffers[2],
                   .target_offset =
                       state->storage.map_origin + map_index * sizeof(*blocks),
                   .length = count * sizeof(*blocks)},
    };
    if (iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_MTP)) {
      uploads[upload_count] = uploads[upload_count - 1];
      uploads[upload_count++].upload.target_buffer = state->mtp.row_table;
    }
  }
  if (!upload_count) {
    return iree_ok_status();
  }
  uint64_t completion = 0;
  return loom_serve_execution_transfer(state->execution, upload_count, uploads,
                                       &completion);
}
