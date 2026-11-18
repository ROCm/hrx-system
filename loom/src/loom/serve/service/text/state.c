// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/state.h"

#include <string.h>

#include "loom/runtime/execution/execution.h"
#include "loom/runtime/storage/relocation.h"
#include "loom/runtime/storage/snapshot.h"
#include "loom/serve/service/text/reservation_plan.h"

// Successful model readiness includes the native signal; join its final access
// before reusing the state-owned operation slot or editing its slabs again.
// A prefix pin can leave completed backing here until the next capacity change.
static void loom_serve_text_complete_backing(loom_serve_text_state_t* state) {
  const bool complete = loom_runtime_memory_operation_is_complete(
      state->memory.backing.operation);
  IREE_ASSERT(complete);
  loom_runtime_memory_operation_free(state->memory.backing.operation);
  state->memory.backing.operation = NULL;
}

// Successful copy readiness retires this source independently of the recurrent
// reader, which remains authoritative until the next epoch or cancellation.
static void loom_serve_text_row_release_tail(loom_serve_text_state_row_t* row) {
  if (row->retiring_tail != UINT32_MAX) {
    loom_runtime_block_pool_release(&row->owner->cache.pool, 1,
                                    &row->retiring_tail);
    row->retiring_tail = UINT32_MAX;
  }
}

// Grows logical page-ID ownership at a serialized admission boundary. The
// source-defined virtual roots already cover the complete addressable range;
// this metadata follows live demand and never changes an existing ID.
static iree_status_t loom_serve_text_cache_ensure_available(
    loom_serve_text_state_t* state, uint32_t required, bool* out_available) {
  *out_available = false;
  loom_runtime_block_pool_t* pool = &state->cache.pool;
  if (required <= pool->available) {
    *out_available = true;
    return iree_ok_status();
  }
  const uint32_t maximum =
      (uint32_t)(state->cache.capacity / state->storage.block_size);
  const uint32_t live = pool->capacity - pool->available;
  if (required > maximum - live) {
    return iree_ok_status();
  }
  const uint32_t minimum = live + required;
  uint64_t capacity = pool->capacity ? 2ull * pool->capacity : 16;
  capacity = iree_max(capacity, (uint64_t)minimum);
  capacity = iree_min(capacity, (uint64_t)maximum);
  IREE_RETURN_IF_ERROR(loom_runtime_block_pool_grow(pool, (uint32_t)capacity));
  *out_available = true;
  return iree_ok_status();
}

// Marks the exact slab union for either the current sparse layout or the
// compact prefix, including private state sharing slabs with cache planes.
static iree_status_t loom_serve_text_keep_range(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, uint64_t* inout_bytes) {
  uint64_t added_bytes = 0;
  IREE_RETURN_IF_ERROR(
      loom_runtime_virtual_buffer_keep(buffer, offset, length, &added_bytes));
  *inout_bytes += added_bytes;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_keep_backing(
    loom_serve_text_state_t* state, bool compact, uint64_t* out_bytes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, state->cache.pool.capacity);
  uint64_t bytes = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    if (state->memory.buffers[i]) {
      loom_runtime_virtual_buffer_begin_trim(state->memory.buffers[i]);
      if (state->memory.private_lengths[i]) {
        status = loom_serve_text_keep_range(state->memory.buffers[i], 0,
                                            state->memory.private_lengths[i],
                                            &bytes);
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < state->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_state_row_t* row = state->rows[i];
    // An active prefix pin can leave a completed COW source until the next
    // epoch. Retire it before moving IDs so no deferred release names a block
    // that compaction has returned to the pool for another owner.
    loom_serve_text_row_release_tail(row);
    if (row->recurrent.slot == UINT32_MAX) {
      continue;
    }
    for (iree_host_size_t binding = TEXT_CONTROL;
         binding <= TEXT_PROGRESS && iree_status_is_ok(status); ++binding) {
      iree_hal_buffer_t* view = row->buffers[binding];
      if (binding != TEXT_RECURRENT && view) {
        status = loom_serve_text_keep_range(
            state->memory.buffers[1], iree_hal_buffer_byte_offset(view),
            iree_hal_buffer_byte_length(view), &bytes);
      }
    }
  }
  for (uint32_t i = 0;
       i < state->recurrent.pool.capacity && iree_status_is_ok(status); ++i) {
    if (state->recurrent.pool.references[i]) {
      iree_hal_buffer_t* view = state->recurrent.buffers[i];
      status = loom_serve_text_keep_range(
          state->memory.buffers[1], iree_hal_buffer_byte_offset(view),
          iree_hal_buffer_byte_length(view), &bytes);
    }
  }
  const uint32_t block_count =
      compact ? state->cache.pool.capacity - state->cache.pool.available
              : state->cache.pool.capacity;
  const uint32_t range_blocks = compact ? block_count : 1;
  for (iree_host_size_t i = 0;
       i < state->storage.region_count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    for (uint32_t block = 0; block < block_count && iree_status_is_ok(status);
         block += range_blocks) {
      if (!compact && !state->cache.pool.references[block]) {
        continue;
      }
      for (iree_host_size_t plane = 0;
           plane < region->blocks.count && iree_status_is_ok(status); ++plane) {
        status = loom_serve_text_keep_range(
            state->memory.buffers[region->allocation],
            region->blocks.origin + plane * region->blocks.stride +
                block * region->blocks.block_bytes,
            range_blocks * region->blocks.block_bytes, &bytes);
      }
    }
  }
  if (out_bytes) {
    *out_bytes = bytes;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Reclaims only unowned physical slabs. The map need not be compact: a denied
// relocation must still be able to release dead backing under pressure.
static iree_status_t loom_serve_text_trim_backing(
    loom_serve_text_state_t* state) {
  IREE_RETURN_IF_ERROR(
      loom_serve_text_keep_backing(state, /*compact=*/false, NULL));
  loom_runtime_virtual_buffer_t*
      buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    if (state->memory.buffers[i]) {
      buffers[count++] = state->memory.buffers[i];
    }
  }
  IREE_ASSERT(!state->memory.backing.operation);
  IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_trim_async(
      state->memory.pool, count, buffers, state->memory.backing.semaphore,
      state->memory.backing.value + 1, &state->memory.backing.operation));
  ++state->memory.backing.value;
  if (!state->memory.backing.operation) {
    return iree_ok_status();
  }
  return loom_runtime_execution_barrier(
      state->execution,
      (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                  &state->memory.backing.value});
}

static iree_status_t loom_serve_text_try_back_compaction(
    loom_serve_text_state_t* state, uint32_t count, bool* out_admitted) {
  uint32_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      state->host_allocator, count, sizeof(*blocks), (void**)&blocks));
  uint32_t index = 0;
  for (uint32_t i = 0; i < state->cache.pool.capacity; ++i) {
    const uint32_t target = state->cache.destinations[i];
    if (target != UINT32_MAX && target != i) {
      blocks[index++] = target;
    }
  }
  iree_hal_buffer_t* views[5] = {0};
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  iree_status_t status = loom_serve_text_storage_plan_snapshot(
      &state->storage, 0, views, false, 1,
      &(loom_serve_text_storage_blocks_t){count, blocks}, state->host_allocator,
      &range_count, &ranges);
  loom_runtime_memory_range_t* commitments = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(state->host_allocator, range_count,
                                    sizeof(*commitments), (void**)&commitments);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      commitments[i] = (loom_runtime_memory_range_t){
          state->memory.buffers[ranges[i].buffer_index], ranges[i].offset,
          ranges[i].length};
    }
    IREE_ASSERT(!state->memory.backing.operation);
    loom_runtime_memory_admission_t admission;
    status = loom_runtime_memory_pool_commit_async(
        state->memory.pool, range_count, commitments,
        state->memory.backing.semaphore, state->memory.backing.value + 1,
        &admission, &state->memory.backing.operation);
    *out_admitted = admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
    if (*out_admitted) {
      ++state->memory.backing.value;
    }
    if (iree_status_is_ok(status) && state->memory.backing.operation) {
      status = loom_runtime_execution_barrier(
          state->execution,
          (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                      &state->memory.backing.value});
    }
  }
  iree_allocator_free(state->host_allocator, commitments);
  iree_allocator_free(state->host_allocator, ranges);
  iree_allocator_free(state->host_allocator, blocks);
  return status;
}

// Completion credit owns backing but has no payload beyond the consumed
// frontier. Only resident rows and immutable endpoints preserve KV contents;
// shared physical pages enter this copy map once, regardless of reader count.
static void loom_serve_text_plan_compaction_copies(
    const loom_serve_text_state_t* state, uint32_t* destinations) {
  memset(destinations, 0xFF,
         state->cache.pool.capacity * sizeof(*destinations));
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    const loom_serve_text_state_row_t* row = state->rows[i];
    if (row->snapshot) {
      continue;
    }
    const uint32_t count = (uint32_t)iree_host_size_ceil_div(
        row->position, state->storage.block_size);
    const uint32_t* blocks =
        state->cache.maps + i * state->storage.blocks_per_row;
    for (uint32_t j = 0; j < count; ++j) {
      destinations[blocks[j]] = state->cache.destinations[blocks[j]];
    }
  }
  for (uint32_t i = 0; i < state->checkpoints.pool.capacity; ++i) {
    if (!state->checkpoints.pool.references[i]) {
      continue;
    }
    const loom_serve_text_state_checkpoint_t* checkpoint =
        &state->checkpoints.values[i];
    for (uint32_t j = 0; j < checkpoint->block_count; ++j) {
      const uint32_t block = checkpoint->blocks[j];
      destinations[block] = state->cache.destinations[block];
    }
  }
}

iree_status_t loom_serve_text_state_submit_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result) {
  *out_result = (loom_serve_text_trim_result_t){0};
  loom_serve_text_complete_backing(state);
  state->memory.trim.released_before = state->memory.statistics.released_bytes;
  state->memory.trim.phase = LOOM_SERVICE_TEXT_TRIM_COMPLETE;
  if (!state->memory.buffers[1]) {
    return iree_ok_status();
  }
  // Prepared continuations retain exact logical IDs, not relocation entries.
  // Dead backing can still retire; moving IDs waits until their plans retire.
  state->memory.trim.phase = state->prepared_reservations
                                 ? LOOM_SERVICE_TEXT_TRIM_COMPLETE
                                 : LOOM_SERVICE_TEXT_TRIM_RELOCATE;
  return loom_serve_text_trim_backing(state);
}

static iree_status_t loom_serve_text_submit_compaction(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result) {
  const uint32_t live =
      state->cache.pool.capacity - state->cache.pool.available;
  out_result->moved_blocks = 0;
  if (!loom_runtime_block_pool_any_owned(&state->cache.pool, live,
                                         state->cache.pool.available)) {
    return iree_ok_status();
  }
  if (state->cache.destination_capacity < state->cache.pool.capacity) {
    uint32_t* destinations = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        state->host_allocator, state->cache.pool.capacity,
        sizeof(*destinations), (void**)&destinations));
    iree_allocator_free(state->host_allocator, state->cache.destinations);
    state->cache.destinations = destinations;
    state->cache.destination_capacity = state->cache.pool.capacity;
  }
  out_result->moved_blocks = loom_runtime_block_pool_plan_compaction(
      &state->cache.pool, state->cache.destinations);
  iree_status_t status = iree_ok_status();
  uint64_t compact_bytes = 0;
  if (out_result->moved_blocks && iree_status_is_ok(status)) {
    status =
        loom_serve_text_keep_backing(state, /*compact=*/true, &compact_bytes);
    // Preview marks cannot escape into admission or asynchronous trimming:
    // every source stays live until relocation and map publication complete.
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_keep_backing(state, /*compact=*/false, NULL);
    }
    if (iree_status_is_ok(status) &&
        compact_bytes >= state->memory.statistics.committed_bytes) {
      out_result->moved_blocks = 0;
    }
  }
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    bool admitted = false;
    status = loom_serve_text_try_back_compaction(
        state, out_result->moved_blocks, &admitted);
    if (!admitted) {
      out_result->moved_blocks = 0;
    }
  }
  uint32_t* copy_destinations = NULL;
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    status = iree_allocator_malloc_array(
        state->host_allocator, state->cache.pool.capacity,
        sizeof(*copy_destinations), (void**)&copy_destinations);
    if (iree_status_is_ok(status)) {
      loom_serve_text_plan_compaction_copies(state, copy_destinations);
    }
  }
  for (iree_host_size_t i = 0;
       out_result->moved_blocks && i < state->storage.region_count &&
       iree_status_is_ok(status);
       ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    uint64_t copied_bytes = 0;
    status = loom_runtime_block_region_relocate(
        state->execution,
        *loom_serve_text_state_allocation(state, region->allocation),
        &region->blocks, state->cache.pool.capacity, copy_destinations,
        &copied_bytes);
    out_result->copied_bytes += copied_bytes;
  }
  // Map uploads follow every accepted copy on the model timeline. Keep both
  // source and destination backing until successful completion; the owner
  // excludes any other mutation of these host maps in the meantime.
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    for (iree_host_size_t i = 0; i < state->row_count; ++i) {
      uint32_t* blocks = state->cache.maps + i * state->storage.blocks_per_row;
      for (uint32_t j = 0; j < state->rows[i]->block_count; ++j) {
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
                    .target_buffer = state->draft.row_table,
                    .target_offset = state->storage.map_origin,
                    .length = length}},
    };
    uint64_t completion = 0;
    status = loom_runtime_execution_transfer(
        state->execution, state->draft.row_table ? 2 : 1, uploads, &completion);
  }
  iree_allocator_free(state->host_allocator, copy_destinations);
  return status;
}

static bool loom_serve_text_recurrent_is_backed(void* user_data,
                                                uint32_t slot) {
  const loom_serve_text_state_t* state = user_data;
  return loom_runtime_virtual_buffer_is_backed(
      state->memory.buffers[1],
      state->storage.recurrent.origin + slot * state->storage.recurrent.stride,
      state->storage.recurrent.length);
}

static bool loom_serve_text_block_is_backed(void* user_data, uint32_t block) {
  const loom_serve_text_state_t* state = user_data;
  for (iree_host_size_t i = 0; i < state->storage.region_count; ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    loom_runtime_virtual_buffer_t* buffer =
        state->memory.buffers[region->allocation];
    if (!buffer) {
      continue;
    }
    for (iree_host_size_t plane = 0; plane < region->blocks.count; ++plane) {
      if (!loom_runtime_virtual_buffer_is_backed(
              buffer,
              region->blocks.origin + plane * region->blocks.stride +
                  block * region->blocks.block_bytes,
              region->blocks.block_bytes)) {
        return false;
      }
    }
  }
  return true;
}

iree_status_t loom_serve_text_state_complete_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* result,
    bool* out_complete) {
  *out_complete = false;
  loom_serve_text_complete_backing(state);
  switch (state->memory.trim.phase) {
    case LOOM_SERVICE_TEXT_TRIM_RELOCATE: {
      IREE_RETURN_IF_ERROR(loom_serve_text_submit_compaction(state, result));
      state->memory.trim.phase = result->moved_blocks
                                     ? LOOM_SERVICE_TEXT_TRIM_RELEASE
                                     : LOOM_SERVICE_TEXT_TRIM_COMPLETE;
      break;
    }
    case LOOM_SERVICE_TEXT_TRIM_RELEASE: {
      loom_runtime_block_pool_commit_compaction(&state->cache.pool,
                                                state->cache.destinations);
      IREE_RETURN_IF_ERROR(loom_serve_text_trim_backing(state));
      state->memory.trim.phase = LOOM_SERVICE_TEXT_TRIM_COMPLETE;
      break;
    }
    case LOOM_SERVICE_TEXT_TRIM_COMPLETE:
      break;
  }
  *out_complete = state->memory.trim.phase == LOOM_SERVICE_TEXT_TRIM_COMPLETE &&
                  !state->memory.backing.operation;
  if (*out_complete) {
    if (state->memory.buffers[1]) {
      // Releasing a tentative plan restores its selected IDs to the stack.
      // Maintenance must prefer holes in surviving backing over that failed
      // placement; otherwise retries can request the same unavailable slab
      // forever despite eviction having made a backed slot reusable.
      loom_runtime_block_pool_prioritize(
          &state->recurrent.pool, loom_serve_text_recurrent_is_backed, state);
      loom_runtime_block_pool_prioritize(
          &state->cache.pool, loom_serve_text_block_is_backed, state);
    }
    result->released_bytes = state->memory.statistics.released_bytes -
                             state->memory.trim.released_before;
  }
  return iree_ok_status();
}

// Only a retired frontier may release pages. A rejected suffix within the
// retained last page stays private; subsequent appends overwrite it before use.
void loom_serve_text_state_row_trim(loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  if (row->recurrent.anchor != UINT32_MAX) {
    loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                    &row->recurrent.anchor);
    row->recurrent.anchor = UINT32_MAX;
  }
  loom_serve_text_row_release_tail(row);
  if (!state->cache.capacity) {
    return;
  }
  const uint32_t keep = (uint32_t)iree_host_size_ceil_div(
      iree_max(row->position, row->reserved_extent), state->storage.block_size);
  const iree_host_size_t row_index = row->index;
  loom_runtime_block_pool_release(
      &state->cache.pool, row->block_count - keep,
      state->cache.maps + row_index * state->storage.blocks_per_row + keep);
  row->block_count = keep;
}

static void loom_serve_text_row_dirty_recurrent(
    loom_serve_text_state_row_t* row) {
  row->owner->recurrent.dirty_count += !row->recurrent.dirty;
  row->recurrent.dirty = true;
}

// Slot identity and source geometry exist before a slot is used. A native
// view is needed only when a row or retained endpoint acquires that identity.
static iree_status_t loom_serve_text_prepare_recurrent(
    loom_serve_text_state_t* state, uint32_t slot) {
  if (state->recurrent.buffers[slot]) {
    return iree_ok_status();
  }
  return iree_hal_buffer_subspan(
      state->row_arena,
      state->storage.recurrent.origin + slot * state->storage.recurrent.stride,
      state->storage.recurrent.length, state->host_allocator,
      &state->recurrent.buffers[slot]);
}

static iree_status_t loom_serve_text_row_acquire_recurrent(
    loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  uint32_t slot = 0;
  loom_runtime_block_pool_acquire(&state->recurrent.pool, 1, &slot);
  iree_status_t status = loom_serve_text_prepare_recurrent(state, slot);
  if (!iree_status_is_ok(status)) {
    loom_runtime_block_pool_release(&state->recurrent.pool, 1, &slot);
    return status;
  }
  row->recurrent.slot = slot;
  row->buffers[TEXT_RECURRENT] = state->recurrent.buffers[slot];
  loom_serve_text_row_dirty_recurrent(row);
  return iree_ok_status();
}

static void loom_serve_text_row_release_recurrent(
    loom_serve_text_state_row_t* row) {
  if (row->recurrent.slot == UINT32_MAX) {
    return;
  }
  loom_serve_text_state_t* state = row->owner;
  loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                  &row->recurrent.slot);
  row->recurrent.slot = UINT32_MAX;
  row->buffers[TEXT_RECURRENT] = NULL;
  state->recurrent.dirty_count -= row->recurrent.dirty;
  row->recurrent.dirty = false;
}

void loom_serve_text_state_row_release_reservation(
    loom_serve_text_state_row_t* row) {
  if (row->recurrent.anchor != UINT32_MAX) {
    // A request cancelled before its first epoch never initialized its writer.
    // Reattach the retained reader when relinquishing unused completion credit.
    loom_serve_text_state_t* state = row->owner;
    loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                    &row->recurrent.slot);
    row->recurrent.slot = row->recurrent.anchor;
    row->recurrent.anchor = UINT32_MAX;
    row->buffers[TEXT_RECURRENT] =
        state->recurrent.buffers[row->recurrent.slot];
    loom_serve_text_row_dirty_recurrent(row);
  }
  row->reserved_extent = 0;
  loom_serve_text_state_row_trim(row);
}

static iree_status_t loom_serve_text_row_initialize(
    loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
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
  iree_status_t status = loom_runtime_execution_transfer(
      state->execution, IREE_ARRAYSIZE(fills), fills, &completion);
  if (iree_status_is_ok(status) && row->owner->storage.carry_stride) {
    const iree_hal_transfer_operation_t clear_carry = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = row->owner->carry,
                 .target_offset =
                     (uint64_t)row->index * row->owner->storage.carry_stride,
                 .length = row->owner->storage.carry_stride,
                 .pattern = &zero,
                 .pattern_length = sizeof(zero)},
    };
    status = loom_runtime_execution_transfer(row->owner->execution, 1,
                                             &clear_carry, &completion);
  }
  return status;
}

// Converts one source-owned storage plan into the virtual reservations that
// actually require backing. Source allocations may mix elastic and fixed
// roots, so fixed ranges are intentionally omitted from the pool transaction.
static iree_status_t loom_serve_text_try_commit_plan(
    loom_serve_text_state_t* state, iree_host_size_t range_count,
    const loom_runtime_snapshot_range_t* ranges, bool* out_admitted) {
  *out_admitted = false;
  loom_runtime_memory_range_t* commitments = NULL;
  iree_status_t status = iree_ok_status();
  if (range_count) {
    status =
        iree_allocator_malloc_array(state->host_allocator, range_count,
                                    sizeof(*commitments), (void**)&commitments);
  }
  iree_host_size_t commitment_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      const loom_runtime_snapshot_range_t* range = &ranges[i];
      loom_runtime_virtual_buffer_t* buffer =
          state->memory.buffers[range->buffer_index];
      if (buffer) {
        commitments[commitment_count++] =
            (loom_runtime_memory_range_t){buffer, range->offset, range->length};
      }
    }
    if (commitment_count) {
      status = loom_runtime_memory_pool_try_commit(
          state->memory.pool, commitment_count, commitments, out_admitted);
    } else {
      *out_admitted = true;
    }
  }
  iree_allocator_free(state->host_allocator, commitments);
  return status;
}

// Direct state users do not have the reservation planner's precommitted range
// union. Reuse its coalesced storage plan and commit one row transaction rather
// than synchronously committing every block in every cache plane.
static iree_status_t loom_serve_text_commit_direct_growth(
    loom_serve_text_state_t* state, loom_serve_text_state_row_t* row,
    iree_hal_buffer_t* const* private_views, uint32_t block_count,
    const uint32_t* blocks) {
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  iree_status_t status = loom_serve_text_storage_plan_snapshot(
      &state->storage, row->index, private_views, false, 1,
      &(loom_serve_text_storage_blocks_t){block_count, blocks},
      state->host_allocator, &range_count, &ranges);
  bool admitted = false;
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_text_try_commit_plan(state, range_count, ranges, &admitted);
  }
  if (iree_status_is_ok(status) && !admitted) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "physical pool cannot back text state growth");
  }
  iree_allocator_free(state->host_allocator, ranges);
  return status;
}

iree_status_t loom_serve_text_state_row_activate(
    loom_serve_text_state_row_t* row) {
  if (row->recurrent.slot != UINT32_MAX) {
    return iree_ok_status();
  }
  loom_serve_text_state_t* state = row->owner;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_acquire_recurrent(row));
  if (state->memory.buffers[1]) {
    IREE_RETURN_IF_ERROR(loom_serve_text_commit_direct_growth(
        state, row, &row->buffers[TEXT_CONTROL], 0, NULL));
  }
  return loom_serve_text_row_initialize(row);
}

void loom_serve_text_state_row_reset(loom_serve_text_state_row_t* row) {
  loom_runtime_snapshot_release(row->snapshot);
  row->snapshot = NULL;
  row->position = 0;
  row->source_position = 0;
  row->reserved_extent = 0;
  loom_serve_text_state_row_trim(row);
  loom_serve_text_row_release_recurrent(row);
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
  IREE_RETURN_IF_ERROR(loom_runtime_execution_drain(state->execution));
  const iree_host_size_t index = row->index;
  uint32_t* blocks = state->cache.maps + index * state->storage.blocks_per_row;
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(
      &state->storage, index, &row->buffers[TEXT_CONTROL], true, 1,
      &(loom_serve_text_storage_blocks_t){row->block_count, blocks},
      state->host_allocator, &range_count, &ranges));
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = *loom_serve_text_state_allocation(state, i);
  }
  iree_status_t status = loom_runtime_snapshot_capture(
      state->execution, IREE_ARRAYSIZE(buffers), buffers, range_count, ranges,
      state->host_allocator, &row->snapshot);
  status =
      iree_status_join(status, loom_runtime_execution_drain(state->execution));
  if (iree_status_is_ok(status)) {
    loom_runtime_block_pool_release(&state->cache.pool, row->block_count,
                                    blocks);
    row->block_count = 0;
    loom_serve_text_row_release_recurrent(row);
  }
  iree_allocator_free(state->host_allocator, ranges);
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
  const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
      row->position, state->storage.block_size);
  bool available = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_cache_ensure_available(state, needed, &available));
  if (!available) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_runtime_execution_drain(state->execution));
  const iree_host_size_t index = row->index;
  const iree_host_size_t map_index = index * state->storage.blocks_per_row;
  uint32_t* blocks = state->cache.maps + map_index;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_acquire_recurrent(row));
  loom_runtime_block_pool_acquire(&state->cache.pool, needed, blocks);
  row->block_count = needed;
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  iree_status_t status = loom_serve_text_storage_plan_snapshot(
      &state->storage, index, &row->buffers[TEXT_CONTROL], true, 1,
      &(loom_serve_text_storage_blocks_t){needed, blocks},
      state->host_allocator, &range_count, &ranges);
  bool admitted = false;
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_text_try_commit_plan(state, range_count, ranges, &admitted);
  }
  if (iree_status_is_ok(status) && admitted) {
    iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
      buffers[i] = *loom_serve_text_state_allocation(state, i);
    }
    status = loom_runtime_snapshot_restore(row->snapshot, state->execution,
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
                    .target_buffer = state->draft.row_table,
                    .target_offset = state->storage.map_origin + map_index * 4,
                    .length = needed * sizeof(*blocks)}},
    };
    uint64_t completion = 0;
    status = loom_runtime_execution_transfer(
        state->execution, state->draft.row_table ? 2 : 1, uploads, &completion);
    status = iree_status_join(status,
                              loom_runtime_execution_drain(state->execution));
  }
  if (iree_status_is_ok(status) && admitted) {
    loom_runtime_snapshot_release(row->snapshot);
    row->snapshot = NULL;
    *out_resumed = true;
  } else if (iree_status_is_ok(status)) {
    loom_runtime_block_pool_release(&state->cache.pool, needed, blocks);
    row->block_count = 0;
    loom_serve_text_row_release_recurrent(row);
  }
  iree_allocator_free(state->host_allocator, ranges);
  return status;
}

static iree_status_t loom_serve_text_copy_carry(loom_serve_text_state_t* state,
                                                iree_host_size_t source,
                                                iree_host_size_t target) {
  if (!state->storage.carry_stride) {
    return iree_ok_status();
  }
  const iree_hal_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy = {.source_buffer = state->carry,
               .source_offset = source * state->storage.carry_stride,
               .target_buffer = state->carry,
               .target_offset = target * state->storage.carry_stride,
               .length = state->storage.carry_stride}};
  uint64_t completion = 0;
  return loom_runtime_execution_transfer(state->execution, 1, &copy,
                                         &completion);
}

iree_status_t loom_serve_text_state_row_submit_pin(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  loom_serve_text_state_t* state = row->owner;
  if (!state->checkpoints.pool.available) {
    return iree_ok_status();
  }
  uint32_t index = 0;
  loom_runtime_block_pool_acquire(&state->checkpoints.pool, 1, &index);
  loom_serve_text_state_checkpoint_t* checkpoint =
      &state->checkpoints.values[index];
  const uint32_t block_count = (uint32_t)iree_host_size_ceil_div(
      row->position, state->storage.block_size);
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      state->host_allocator, block_count, sizeof(*checkpoint->blocks),
      (void**)&checkpoint->blocks);
  if (!iree_status_is_ok(status)) {
    loom_runtime_block_pool_release(&state->checkpoints.pool, 1, &index);
    return status;
  }
  checkpoint->position = row->position;
  checkpoint->source_position = row->source_position;
  checkpoint->recurrent_slot = row->recurrent.anchor != UINT32_MAX
                                   ? row->recurrent.anchor
                                   : row->recurrent.slot;
  checkpoint->block_count = block_count;
  const iree_host_size_t row_index = row->index;
  memcpy(checkpoint->blocks,
         state->cache.maps + row_index * state->storage.blocks_per_row,
         checkpoint->block_count * sizeof(*checkpoint->blocks));
  loom_runtime_block_pool_retain(&state->cache.pool, checkpoint->block_count,
                                 checkpoint->blocks);
  loom_runtime_block_pool_retain(&state->recurrent.pool, 1,
                                 &checkpoint->recurrent_slot);
  // Sharing a reserved writer cannot defer COW admission to the next epoch.
  // The normal reservation planner preserves its unused private pages and
  // either backs the complete continuation or leaves the original row intact.
  loom_runtime_memory_admission_t admission =
      LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  if (row->reserved_extent) {
    status = loom_serve_text_state_row_submit_reserve(
        row, LOOM_SERVE_TEXT_RESERVE_CONTINUE, NULL, row->reserved_extent, NULL,
        NULL, &admission);
  }
  const bool reserved = admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  if (iree_status_is_ok(status) && !reserved) {
    loom_serve_text_state_checkpoint_release(checkpoint);
  } else if (iree_status_is_ok(status)) {
    status =
        loom_serve_text_copy_carry(state, row_index, state->row_count + index);
  }
  if (iree_status_is_ok(status) && reserved) {
    *out_checkpoint = checkpoint;
  }
  return status;
}

void loom_serve_text_state_checkpoint_release(
    loom_serve_text_state_checkpoint_t* checkpoint) {
  loom_serve_text_state_t* state = checkpoint->owner;
  loom_runtime_block_pool_release(&state->cache.pool, checkpoint->block_count,
                                  checkpoint->blocks);
  if (checkpoint->recurrent_slot != UINT32_MAX) {
    loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                    &checkpoint->recurrent_slot);
    checkpoint->recurrent_slot = UINT32_MAX;
  }
  loom_runtime_snapshot_release(checkpoint->snapshot);
  checkpoint->snapshot = NULL;
  iree_allocator_free(state->host_allocator, checkpoint->blocks);
  checkpoint->blocks = NULL;
  checkpoint->block_count = 0;
  const uint32_t index = (uint32_t)(checkpoint - state->checkpoints.values);
  loom_runtime_block_pool_release(&state->checkpoints.pool, 1, &index);
}

static iree_status_t loom_serve_text_checkpoint_plan_snapshot(
    loom_serve_text_state_checkpoint_t* checkpoint, iree_host_size_t* out_count,
    loom_runtime_snapshot_range_t** out_ranges) {
  loom_serve_text_state_t* state = checkpoint->owner;
  iree_hal_buffer_t* views[5] = {0};
  views[TEXT_RECURRENT - TEXT_CONTROL] =
      state->recurrent.buffers[checkpoint->recurrent_slot];
  const iree_host_size_t carry_index =
      state->row_count + (checkpoint - state->checkpoints.values);
  return loom_serve_text_storage_plan_snapshot(
      &state->storage, carry_index, views, true, 1,
      &(loom_serve_text_storage_blocks_t){checkpoint->block_count,
                                          checkpoint->blocks},
      state->host_allocator, out_count, out_ranges);
}

iree_status_t loom_serve_text_state_checkpoint_submit_suspend(
    loom_serve_text_state_checkpoint_t* checkpoint) {
  loom_serve_text_state_t* state = checkpoint->owner;
  if (!state->memory.buffers[1]) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "checkpoint suspension requires elastic pooled state");
  }
  if (checkpoint->snapshot) {
    return iree_ok_status();
  }
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_plan_snapshot(
      checkpoint, &range_count, &ranges));
  iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = *loom_serve_text_state_allocation(state, i);
  }
  iree_status_t status = loom_runtime_snapshot_capture(
      state->execution, IREE_ARRAYSIZE(buffers), buffers, range_count, ranges,
      state->host_allocator, &checkpoint->snapshot);
  iree_allocator_free(state->host_allocator, ranges);
  return status;
}

void loom_serve_text_state_checkpoint_complete_suspend(
    loom_serve_text_state_checkpoint_t* checkpoint) {
  if (checkpoint->recurrent_slot == UINT32_MAX) {
    return;
  }
  loom_serve_text_state_t* state = checkpoint->owner;
  loom_runtime_block_pool_release(&state->cache.pool, checkpoint->block_count,
                                  checkpoint->blocks);
  checkpoint->block_count = 0;
  loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                  &checkpoint->recurrent_slot);
  checkpoint->recurrent_slot = UINT32_MAX;
}

static iree_host_size_t loom_serve_text_row_private_ranges(
    const loom_serve_text_state_row_t* row,
    loom_runtime_memory_range_t* ranges) {
  iree_host_size_t count = 0;
  for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
       ++binding) {
    iree_hal_buffer_t* view = row->buffers[binding];
    if (binding != TEXT_RECURRENT && view) {
      ranges[count++] = (loom_runtime_memory_range_t){
          row->owner->memory.buffers[1], iree_hal_buffer_byte_offset(view),
          iree_hal_buffer_byte_length(view)};
    }
  }
  return count;
}

// Admission covers both owners before replacing either. The checkpoint's
// image has no row-private execution state, so warming it never mutates the
// selected row. Ordinary denial returns the tentative IDs and keeps the image.
static iree_status_t loom_serve_text_checkpoint_try_resume(
    loom_serve_text_state_checkpoint_t* checkpoint,
    const loom_serve_text_state_row_t* row, bool* out_resumed) {
  *out_resumed = false;
  loom_serve_text_state_t* state = checkpoint->owner;
  const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
      checkpoint->position, state->storage.block_size);
  bool available = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_cache_ensure_available(state, needed, &available));
  if (!available) {
    return iree_ok_status();
  }
  loom_runtime_block_pool_acquire(&state->cache.pool, needed,
                                  checkpoint->blocks);
  checkpoint->block_count = needed;
  loom_runtime_block_pool_acquire(&state->recurrent.pool, 1,
                                  &checkpoint->recurrent_slot);
  iree_host_size_t range_count = 0;
  loom_runtime_snapshot_range_t* ranges = NULL;
  iree_status_t status =
      loom_serve_text_prepare_recurrent(state, checkpoint->recurrent_slot);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_checkpoint_plan_snapshot(checkpoint, &range_count,
                                                      &ranges);
  }
  loom_runtime_memory_range_t* commitments = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(state->host_allocator, range_count + 4,
                                    sizeof(*commitments), (void**)&commitments);
  }
  bool admitted = false;
  if (iree_status_is_ok(status)) {
    iree_host_size_t count =
        loom_serve_text_row_private_ranges(row, commitments);
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      const loom_runtime_snapshot_range_t* range = &ranges[i];
      loom_runtime_virtual_buffer_t* buffer =
          state->memory.buffers[range->buffer_index];
      if (buffer) {
        commitments[count++] =
            (loom_runtime_memory_range_t){buffer, range->offset, range->length};
      }
    }
    status = loom_runtime_memory_pool_try_commit(state->memory.pool, count,
                                                 commitments, &admitted);
  }
  if (iree_status_is_ok(status) && admitted) {
    iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
      buffers[i] = *loom_serve_text_state_allocation(state, i);
    }
    status = loom_runtime_snapshot_restore(
        checkpoint->snapshot, state->execution, IREE_ARRAYSIZE(buffers),
        buffers, range_count, ranges);
    status = iree_status_join(status,
                              loom_runtime_execution_drain(state->execution));
  }
  if (iree_status_is_ok(status) && admitted) {
    loom_runtime_snapshot_release(checkpoint->snapshot);
    checkpoint->snapshot = NULL;
    *out_resumed = true;
  } else if (!admitted) {
    loom_runtime_block_pool_release(&state->cache.pool, needed,
                                    checkpoint->blocks);
    checkpoint->block_count = 0;
    loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                    &checkpoint->recurrent_slot);
    checkpoint->recurrent_slot = UINT32_MAX;
  }
  iree_allocator_free(state->host_allocator, commitments);
  iree_allocator_free(state->host_allocator, ranges);
  return status;
}

iree_status_t loom_serve_text_state_row_try_restore(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t* checkpoint, bool* out_restored) {
  *out_restored = false;
  loom_serve_text_state_t* state = row->owner;
  IREE_RETURN_IF_ERROR(loom_runtime_execution_drain(state->execution));
  if (checkpoint->snapshot) {
    bool resumed = false;
    IREE_RETURN_IF_ERROR(
        loom_serve_text_checkpoint_try_resume(checkpoint, row, &resumed));
    if (!resumed) {
      return iree_ok_status();
    }
  } else if (state->memory.buffers[1]) {
    loom_runtime_memory_range_t ranges[4];
    const iree_host_size_t count =
        loom_serve_text_row_private_ranges(row, ranges);
    bool admitted = false;
    IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_try_commit(
        state->memory.pool, count, ranges, &admitted));
    if (!admitted) {
      return iree_ok_status();
    }
  }
  loom_serve_text_state_row_reset(row);
  row->position = checkpoint->position;
  row->source_position = checkpoint->source_position;
  row->recurrent.slot = checkpoint->recurrent_slot;
  row->buffers[TEXT_RECURRENT] = state->recurrent.buffers[row->recurrent.slot];
  loom_runtime_block_pool_retain(&state->recurrent.pool, 1,
                                 &row->recurrent.slot);
  loom_serve_text_row_dirty_recurrent(row);
  row->block_count = checkpoint->block_count;
  const iree_host_size_t row_index = row->index;
  const iree_host_size_t map_index = row_index * state->storage.blocks_per_row;
  uint32_t* blocks = state->cache.maps + map_index;
  memcpy(blocks, checkpoint->blocks, row->block_count * sizeof(*blocks));
  loom_runtime_block_pool_retain(&state->cache.pool, row->block_count, blocks);
  const iree_host_size_t checkpoint_index =
      (iree_host_size_t)(checkpoint - state->checkpoints.values);
  iree_status_t status = loom_serve_text_copy_carry(
      state, state->row_count + checkpoint_index, row_index);
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = blocks,
                  .target_buffer = state->epoch.buffers[2],
                  .target_offset = state->storage.map_origin + map_index * 4,
                  .length = row->block_count * sizeof(*blocks)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = blocks,
                  .target_buffer = state->draft.row_table,
                  .target_offset = state->storage.map_origin + map_index * 4,
                  .length = row->block_count * sizeof(*blocks)}},
  };
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    status = loom_runtime_execution_transfer(
        state->execution, state->draft.row_table ? 2 : 1, uploads, &completion);
  }
  status =
      iree_status_join(status, loom_runtime_execution_drain(state->execution));
  if (iree_status_is_ok(status)) {
    *out_restored = true;
  }
  return status;
}

static bool loom_serve_text_row_has_shared_tail(
    const loom_serve_text_state_row_t* row) {
  const loom_serve_text_state_t* state = row->owner;
  if (!row->position || !(row->position % state->storage.block_size)) {
    return false;
  }
  const iree_host_size_t row_index = row->index;
  const uint32_t tail =
      state->cache.maps[row_index * state->storage.blocks_per_row +
                        row->position / state->storage.block_size];
  return loom_runtime_block_pool_is_shared(&state->cache.pool, tail);
}

uint32_t loom_serve_text_state_row_growth(
    const loom_serve_text_state_row_t* row, uint32_t extent) {
  const loom_serve_text_state_t* state = row->owner;
  const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
      row->position + extent, state->storage.block_size);
  return iree_max(needed, row->block_count) - row->block_count +
         loom_serve_text_row_has_shared_tail(row);
}

// COW copies only one partial logical page, across every source-declared plane.
// Destinations are already backed or ordered behind mapping readiness. The row
// retains the old page through the existing cohort completion.
static iree_status_t loom_serve_text_fork_cache_block(
    loom_serve_text_state_t* state, uint32_t source, uint32_t target) {
  iree_hal_transfer_operation_t copies[64];
  iree_host_size_t count = 0;
  uint64_t completion = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0;
       r < state->storage.region_count && iree_status_is_ok(status); ++r) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[r];
    iree_hal_buffer_t* buffer =
        *loom_serve_text_state_allocation(state, region->allocation);
    for (iree_host_size_t plane = 0;
         plane < region->blocks.count && iree_status_is_ok(status); ++plane) {
      const iree_device_size_t origin =
          region->blocks.origin + plane * region->blocks.stride;
      const iree_device_size_t target_offset =
          origin + target * region->blocks.block_bytes;
      copies[count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy = {
              .source_buffer = buffer,
              .source_offset = origin + source * region->blocks.block_bytes,
              .target_buffer = buffer,
              .target_offset = target_offset,
              .length = region->blocks.block_bytes}};
      if (count == IREE_ARRAYSIZE(copies)) {
        status = loom_runtime_execution_transfer(state->execution, count,
                                                 copies, &completion);
        count = 0;
      }
    }
  }
  if (iree_status_is_ok(status) && count) {
    status = loom_runtime_execution_transfer(state->execution, count, copies,
                                             &completion);
  }
  return status;
}

// Admission may reuse the selected row's private IDs, but only borrows them
// until the physical union fits. Its existing references preserve denial.
typedef struct loom_serve_text_replacement_t {
  // Selected row whose private ownership can fund a replacement.
  loom_serve_text_state_row_t* row;
  // Next selected map entry to consider exactly once.
  uint32_t cursor;
  // Number of IDs acquired from free storage, following all reused row IDs.
  uint32_t acquired;
  // Whether the selected recurrent slot has already been borrowed.
  bool recurrent_used;
} loom_serve_text_replacement_t;

static uint32_t loom_serve_text_reserve_block(
    loom_serve_text_state_t* state,
    loom_serve_text_replacement_t* replacement) {
  const loom_serve_text_state_row_t* row = replacement->row;
  const uint32_t* blocks =
      state->cache.maps + row->index * state->storage.blocks_per_row;
  while (replacement->cursor < row->block_count) {
    const uint32_t block = blocks[replacement->cursor++];
    if (!loom_runtime_block_pool_is_shared(&state->cache.pool, block)) {
      loom_runtime_block_pool_retain(&state->cache.pool, 1, &block);
      return block;
    }
  }
  uint32_t block = 0;
  loom_runtime_block_pool_acquire(&state->cache.pool, 1, &block);
  ++replacement->acquired;
  return block;
}

static uint32_t loom_serve_text_reserve_recurrent(
    loom_serve_text_state_t* state, loom_serve_text_replacement_t* replacement,
    bool* out_acquired) {
  *out_acquired = false;
  if (!replacement->recurrent_used) {
    replacement->recurrent_used = true;
    const uint32_t slot = replacement->row->recurrent.slot;
    if (slot != UINT32_MAX &&
        !loom_runtime_block_pool_is_shared(&state->recurrent.pool, slot)) {
      loom_runtime_block_pool_retain(&state->recurrent.pool, 1, &slot);
      return slot;
    }
  }
  uint32_t slot = 0;
  loom_runtime_block_pool_acquire(&state->recurrent.pool, 1, &slot);
  *out_acquired = true;
  return slot;
}

// Reverses tentative acquisition order, preserving the free-ID stack on a
// denied request as well as the selected owners and their logical maps.
static void loom_serve_text_release_reverse(loom_runtime_block_pool_t* pool,
                                            uint32_t count,
                                            const uint32_t* blocks) {
  while (count) {
    loom_runtime_block_pool_release(pool, 1, &blocks[--count]);
  }
}

iree_status_t loom_serve_text_state_row_prepare_reserve(
    loom_serve_text_state_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_state_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_state_reservation_t** out_reservation) {
  *out_reservation = NULL;
  loom_serve_text_state_t* state = row->owner;
  loom_serve_text_complete_backing(state);
  const iree_host_size_t index = row->index;
  const iree_host_size_t position = checkpoint ? checkpoint->position
                                    : mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE
                                        ? row->position
                                        : 0;
  const uint32_t source_position = checkpoint ? checkpoint->source_position
                                   : mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE
                                       ? row->source_position
                                       : 0;
  const bool cold = checkpoint && checkpoint->snapshot;
  const bool pooled = state->cache.capacity != 0;
  const uint32_t prefix = pooled ? (uint32_t)iree_host_size_ceil_div(
                                       position, state->storage.block_size)
                                 : 0;
  const uint32_t count = pooled ? (uint32_t)iree_host_size_ceil_div(
                                      extent, state->storage.block_size)
                                : 0;
  const bool fork_tail =
      prefix && position % state->storage.block_size &&
      (checkpoint || loom_serve_text_row_has_shared_tail(row));
  const uint32_t source_slot = checkpoint  ? checkpoint->recurrent_slot
                               : !position ? UINT32_MAX
                               : row->recurrent.anchor != UINT32_MAX
                                   ? row->recurrent.anchor
                                   : row->recurrent.slot;
  const bool fork_recurrent =
      position && (checkpoint || loom_runtime_block_pool_is_shared(
                                     &state->recurrent.pool, source_slot));
  const uint32_t* old_blocks =
      state->cache.maps
          ? state->cache.maps + index * state->storage.blocks_per_row
          : NULL;
  uint32_t preserved = cold ? 0 : prefix - fork_tail;
  if (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE && !fork_tail) {
    // Preserve private completion credit in place, as well as the consumed
    // prefix. Shared unused storage must still be replaced before writing.
    while (preserved < count && preserved < row->block_count &&
           !loom_runtime_block_pool_is_shared(&state->cache.pool,
                                              old_blocks[preserved])) {
      ++preserved;
    }
  }
  loom_serve_text_replacement_t replacement = {
      .row = row,
      .cursor = mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE
                    ? iree_max(prefix, preserved)
                    : 0};
  uint32_t reusable = 0;
  for (uint32_t i = replacement.cursor; i < row->block_count; ++i) {
    reusable +=
        !loom_runtime_block_pool_is_shared(&state->cache.pool, old_blocks[i]);
  }
  const uint32_t needed =
      count - iree_max(prefix, preserved) + fork_tail + (cold ? prefix : 0);
  const uint32_t recurrent_needed = cold + (!position || fork_recurrent);
  const uint32_t recurrent_reusable =
      row->recurrent.slot != UINT32_MAX &&
      !loom_runtime_block_pool_is_shared(&state->recurrent.pool,
                                         row->recurrent.slot);
  if (recurrent_needed > state->recurrent.pool.available + recurrent_reusable) {
    return iree_ok_status();
  }
  const uint32_t free_needed = needed - iree_min(needed, reusable);
  bool available = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_cache_ensure_available(state, free_needed, &available));
  if (!available) {
    return iree_ok_status();
  }
  loom_serve_text_state_reservation_t* reservation = NULL;
  iree_host_size_t allocation_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*reservation), &allocation_length,
      IREE_STRUCT_FIELD_FAM((uint64_t)count - preserved + (cold ? prefix : 0),
                            uint32_t)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, allocation_length, (void**)&reservation));
  uint32_t* blocks = reservation->blocks;
  loom_serve_text_state_checkpoint_t restored = {0};
  bool restore_recurrent_acquired = false;
  if (cold) {
    restored = *checkpoint;
    restored.blocks = blocks + count - preserved;
    restored.block_count = prefix;
    for (uint32_t i = 0; i < prefix; ++i) {
      restored.blocks[i] = loom_serve_text_reserve_block(state, &replacement);
    }
    restored.recurrent_slot = loom_serve_text_reserve_recurrent(
        state, &replacement, &restore_recurrent_acquired);
  }
  const uint32_t* source_blocks = cold         ? restored.blocks
                                  : checkpoint ? checkpoint->blocks
                                               : old_blocks;
  const uint32_t restore_acquired = replacement.acquired;
  for (uint32_t i = preserved; i < prefix; ++i) {
    blocks[i - preserved] = source_blocks[i];
    loom_runtime_block_pool_retain(&state->cache.pool, 1,
                                   &blocks[i - preserved]);
  }
  uint32_t tail = UINT32_MAX;
  if (fork_tail) {
    tail = blocks[prefix - preserved - 1];
    blocks[prefix - preserved - 1] =
        loom_serve_text_reserve_block(state, &replacement);
  }
  for (uint32_t i = iree_max(prefix, preserved); i < count; ++i) {
    blocks[i - preserved] = loom_serve_text_reserve_block(state, &replacement);
  }
  const uint32_t acquired = replacement.acquired - restore_acquired;
  uint32_t reader = UINT32_MAX;
  if (position) {
    reader = cold ? restored.recurrent_slot : source_slot;
    loom_runtime_block_pool_retain(&state->recurrent.pool, 1, &reader);
  }
  uint32_t writer = reader;
  bool writer_acquired = false;
  if (!position || fork_recurrent) {
    writer = loom_serve_text_reserve_recurrent(state, &replacement,
                                               &writer_acquired);
  } else {
    reader = UINT32_MAX;
  }
  *reservation = (loom_serve_text_state_reservation_t){
      .row = row,
      .checkpoint = checkpoint,
      .restored = restored,
      .extent = (uint32_t)extent,
      .position = (uint32_t)position,
      .source_position = source_position,
      .prefix = prefix,
      .count = count,
      .preserved = preserved,
      .reader = reader,
      .writer = writer,
      .tail = tail,
      .flags = (cold ? LOOM_SERVE_TEXT_RESERVATION_COLD : 0) |
               (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE
                    ? LOOM_SERVE_TEXT_RESERVATION_CONTINUE
                    : 0),
  };
  reservation->next = state->prepared_reservations;
  reservation->previous_link = &state->prepared_reservations;
  if (reservation->next) {
    reservation->next->previous_link = &reservation->next;
  }
  state->prepared_reservations = reservation;
  iree_status_t status = loom_serve_text_prepare_recurrent(state, writer);
  if (iree_status_is_ok(status) && cold) {
    status = loom_serve_text_prepare_recurrent(state, restored.recurrent_slot);
  }
  if (iree_status_is_ok(status) && cold) {
    iree_hal_buffer_t* views[5] = {0};
    views[TEXT_RECURRENT - TEXT_CONTROL] =
        state->recurrent.buffers[restored.recurrent_slot];
    status = loom_serve_text_storage_plan_snapshot(
        &state->storage,
        state->row_count + (checkpoint - state->checkpoints.values), views,
        true, 1, &(loom_serve_text_storage_blocks_t){prefix, restored.blocks},
        state->host_allocator, &reservation->restore_count,
        &reservation->restore_ranges);
  }
  // Installed rows and borrowed endpoints already own their backing. Only
  // newly acquired IDs and an inactive destination's private views are
  // uncertain. Global epoch/proposal storage was backed at state creation.
  if (iree_status_is_ok(status) && state->memory.pool) {
    const bool activate_private = row->recurrent.slot == UINT32_MAX;
    iree_hal_buffer_t* views[5] = {0};
    if (activate_private) {
      memcpy(views, &row->buffers[TEXT_CONTROL], sizeof(views));
    }
    views[TEXT_RECURRENT - TEXT_CONTROL] =
        writer_acquired ? state->recurrent.buffers[writer] : NULL;
    const loom_serve_text_storage_blocks_t spans[] = {
        {acquired, blocks + count - preserved - acquired},
        {restore_acquired,
         cold ? restored.blocks + prefix - restore_acquired : NULL},
    };
    iree_host_size_t range_count = 0;
    loom_runtime_snapshot_range_t* ranges = NULL;
    status = loom_serve_text_storage_plan_snapshot(
        &state->storage, index, views, activate_private, IREE_ARRAYSIZE(spans),
        spans, state->host_allocator, &range_count, &ranges);
    const iree_host_size_t capacity = range_count + (cold ? 2 : 0);
    if (iree_status_is_ok(status) && capacity) {
      status = iree_allocator_malloc_array(state->host_allocator, capacity,
                                           sizeof(*reservation->commitments),
                                           (void**)&reservation->commitments);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < range_count; ++i) {
        const loom_runtime_snapshot_range_t* range = &ranges[i];
        loom_runtime_virtual_buffer_t* buffer =
            state->memory.buffers[range->buffer_index];
        if (buffer) {
          reservation->commitments[reservation->commitment_count++] =
              (loom_runtime_memory_range_t){buffer, range->offset,
                                            range->length};
        }
      }
      if (cold && state->storage.carry_stride && state->memory.buffers[6]) {
        reservation->commitments[reservation->commitment_count++] =
            (loom_runtime_memory_range_t){
                state->memory.buffers[6],
                (state->row_count + (checkpoint - state->checkpoints.values)) *
                    state->storage.carry_stride,
                state->storage.carry_stride};
      }
      if (restore_recurrent_acquired && state->memory.buffers[1]) {
        iree_hal_buffer_t* view =
            state->recurrent.buffers[restored.recurrent_slot];
        reservation->commitments[reservation->commitment_count++] =
            (loom_runtime_memory_range_t){state->memory.buffers[1],
                                          iree_hal_buffer_byte_offset(view),
                                          iree_hal_buffer_byte_length(view)};
      }
    }
    iree_allocator_free(state->host_allocator, ranges);
  }
  if (iree_status_is_ok(status)) {
    *out_reservation = reservation;
  } else {
    loom_serve_text_state_reservation_free(reservation);
  }
  return status;
}

iree_status_t loom_serve_text_state_reservation_submit(
    loom_serve_text_state_reservation_t* reservation,
    loom_serve_text_state_admit_fn_t admit, void* user_data,
    loom_runtime_memory_admission_t* out_admission) {
  IREE_ASSERT(!iree_any_bit_set(reservation->flags,
                                LOOM_SERVE_TEXT_RESERVATION_INSTALLED));
  *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
  loom_serve_text_state_row_t* row = reservation->row;
  loom_serve_text_state_t* state = row->owner;
  loom_serve_text_complete_backing(state);
  const iree_host_size_t index = row->index;
  loom_serve_text_state_checkpoint_t* checkpoint = reservation->checkpoint;
  const uint32_t position = reservation->position;
  const uint32_t count = reservation->count;
  const uint32_t preserved = reservation->preserved;
  const uint32_t unchanged =
      iree_any_bit_set(reservation->flags, LOOM_SERVE_TEXT_RESERVATION_CONTINUE)
          ? preserved
          : 0;
  const uint32_t* blocks = reservation->blocks;
  loom_runtime_memory_admission_t admission =
      LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  iree_status_t status = iree_ok_status();
  if (admit) {
    status = admit(user_data, reservation->commitment_count,
                   reservation->commitments, state->memory.backing.semaphore,
                   state->memory.backing.value + 1, &admission,
                   &state->memory.backing.operation);
  } else if (state->memory.pool) {
    status = loom_runtime_memory_pool_commit_async(
        state->memory.pool, reservation->commitment_count,
        reservation->commitments, state->memory.backing.semaphore,
        state->memory.backing.value + 1, &admission,
        &state->memory.backing.operation);
  }
  if (state->memory.pool &&
      admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE) {
    ++state->memory.backing.value;
  }
  // From here ordinary denial is impossible. Transfer/platform failure is
  // terminal and the installed owners retain every accepted-work resource.
  if (iree_status_is_ok(status) &&
      admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE) {
    uint32_t* row_blocks =
        state->cache.maps
            ? state->cache.maps + index * state->storage.blocks_per_row
            : NULL;
    loom_runtime_snapshot_release(row->snapshot);
    row->snapshot = NULL;
    if (row->recurrent.anchor != UINT32_MAX) {
      loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                      &row->recurrent.anchor);
    }
    loom_serve_text_row_release_tail(row);
    if (row->recurrent.slot != reservation->writer) {
      loom_serve_text_row_release_recurrent(row);
      row->recurrent.slot = reservation->writer;
      row->buffers[TEXT_RECURRENT] =
          state->recurrent.buffers[reservation->writer];
      loom_serve_text_row_dirty_recurrent(row);
    } else {
      // The destination already owns this slot. Drop the planner's temporary
      // reference. A fresh or restored frontier must replace the source's
      // encoded reader and fork position even when its writer is unchanged.
      loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                      &reservation->writer);
      if (!iree_any_bit_set(reservation->flags,
                            LOOM_SERVE_TEXT_RESERVATION_CONTINUE) ||
          row->recurrent.anchor != reservation->reader) {
        loom_serve_text_row_dirty_recurrent(row);
      }
    }
    if (row->block_count != unchanged) {
      loom_runtime_block_pool_release(&state->cache.pool,
                                      row->block_count - unchanged,
                                      row_blocks + unchanged);
    }
    if (preserved != unchanged) {
      const uint32_t* prefix_blocks =
          loom_serve_text_state_reservation_prefix(reservation);
      loom_runtime_block_pool_retain(&state->cache.pool, preserved,
                                     prefix_blocks);
      memcpy(row_blocks, prefix_blocks, preserved * sizeof(*blocks));
    }
    if (iree_any_bit_set(reservation->flags,
                         LOOM_SERVE_TEXT_RESERVATION_COLD)) {
      checkpoint->recurrent_slot = reservation->restored.recurrent_slot;
      checkpoint->block_count = reservation->prefix;
      memcpy(checkpoint->blocks, reservation->restored.blocks,
             reservation->prefix * sizeof(*blocks));
    }
    row->position = position;
    row->source_position = reservation->source_position;
    row->block_count = count;
    row->reserved_extent = reservation->extent;
    if (count != preserved) {
      memcpy(row_blocks + preserved, blocks,
             (count - preserved) * sizeof(*blocks));
    }
    row->recurrent.anchor = reservation->reader;
    row->retiring_tail = reservation->tail;
    reservation->flags |= LOOM_SERVE_TEXT_RESERVATION_INSTALLED;
    if (state->memory.backing.operation) {
      status = loom_runtime_execution_barrier(
          state->execution,
          (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                      &state->memory.backing.value});
    }
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(reservation->flags,
                         LOOM_SERVE_TEXT_RESERVATION_COLD)) {
      iree_hal_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
        buffers[i] = *loom_serve_text_state_allocation(state, i);
      }
      status = loom_runtime_snapshot_restore(
          checkpoint->snapshot, state->execution, IREE_ARRAYSIZE(buffers),
          buffers, reservation->restore_count, reservation->restore_ranges);
      if (iree_status_is_ok(status)) {
        loom_runtime_snapshot_release(checkpoint->snapshot);
        checkpoint->snapshot = NULL;
      }
    }
    if (iree_status_is_ok(status) && !position) {
      status = loom_serve_text_row_initialize(row);
    } else if (iree_status_is_ok(status) && checkpoint) {
      status = loom_serve_text_copy_carry(
          state, state->row_count + (checkpoint - state->checkpoints.values),
          index);
    }
    if (iree_status_is_ok(status) && reservation->tail != UINT32_MAX) {
      status = loom_serve_text_fork_cache_block(
          state, reservation->tail,
          blocks[reservation->prefix - reservation->preserved - 1]);
    }
    if (iree_status_is_ok(status) && count != unchanged) {
      const iree_hal_transfer_operation_t uploads[] = {
          {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
           .upload = {.source = state->cache.maps +
                                index * state->storage.blocks_per_row +
                                unchanged,
                      .target_buffer = state->epoch.buffers[2],
                      .target_offset =
                          state->storage.map_origin +
                          (index * state->storage.blocks_per_row + unchanged) *
                              4,
                      .length = (count - unchanged) * sizeof(*blocks)}},
          {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
           .upload = {.source = state->cache.maps +
                                index * state->storage.blocks_per_row +
                                unchanged,
                      .target_buffer = state->draft.row_table,
                      .target_offset =
                          state->storage.map_origin +
                          (index * state->storage.blocks_per_row + unchanged) *
                              4,
                      .length = (count - unchanged) * sizeof(*blocks)}},
      };
      uint64_t completion = 0;
      status = loom_runtime_execution_transfer(state->execution,
                                               state->draft.row_table ? 2 : 1,
                                               uploads, &completion);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_admission = admission;
  }
  return status;
}

void loom_serve_text_state_reservation_free(
    loom_serve_text_state_reservation_t* reservation) {
  if (!reservation) {
    return;
  }
  loom_serve_text_state_t* state = reservation->row->owner;
  *reservation->previous_link = reservation->next;
  if (reservation->next) {
    reservation->next->previous_link = reservation->previous_link;
  }
  if (!iree_any_bit_set(reservation->flags,
                        LOOM_SERVE_TEXT_RESERVATION_INSTALLED)) {
    const uint32_t owned_count = reservation->count - reservation->preserved;
    const uint32_t owned_prefix =
        reservation->prefix -
        iree_min(reservation->prefix, reservation->preserved);
    loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                    &reservation->writer);
    if (reservation->reader != UINT32_MAX) {
      loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                      &reservation->reader);
    }
    loom_serve_text_release_reverse(&state->cache.pool,
                                    owned_count - owned_prefix,
                                    reservation->blocks + owned_prefix);
    if (reservation->tail != UINT32_MAX) {
      loom_runtime_block_pool_release(&state->cache.pool, 1,
                                      &reservation->blocks[owned_prefix - 1]);
      reservation->blocks[owned_prefix - 1] = reservation->tail;
    }
    loom_serve_text_release_reverse(&state->cache.pool, owned_prefix,
                                    reservation->blocks);
    if (iree_any_bit_set(reservation->flags,
                         LOOM_SERVE_TEXT_RESERVATION_COLD)) {
      loom_runtime_block_pool_release(&state->recurrent.pool, 1,
                                      &reservation->restored.recurrent_slot);
      loom_serve_text_release_reverse(&state->cache.pool, reservation->prefix,
                                      reservation->restored.blocks);
    }
  }
  iree_allocator_free(state->host_allocator, reservation->requirements);
  iree_allocator_free(state->host_allocator, reservation->commitments);
  iree_allocator_free(state->host_allocator, reservation->restore_ranges);
  iree_allocator_free(state->host_allocator, reservation);
}

iree_status_t loom_serve_text_state_row_submit_reserve(
    loom_serve_text_state_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_state_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_state_admit_fn_t admit, void* user_data,
    loom_runtime_memory_admission_t* out_admission) {
  *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
  loom_serve_text_state_reservation_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_prepare_reserve(
      row, mode, checkpoint, extent, &reservation));
  if (!reservation) {
    return iree_ok_status();
  }
  loom_runtime_memory_admission_t admission;
  iree_status_t status = loom_serve_text_state_reservation_submit(
      reservation, admit, user_data, &admission);
  loom_serve_text_state_reservation_free(reservation);
  if (iree_status_is_ok(status)) {
    *out_admission = admission;
  }
  return status;
}

void loom_serve_text_state_row_complete_reserve(
    loom_serve_text_state_row_t* row) {
  loom_serve_text_state_t* state = row->owner;
  loom_serve_text_complete_backing(state);
  loom_serve_text_row_release_tail(row);
}

// The whole epoch has passed its capacity check. Publish only newly assigned
// map entries on the existing ordered transfer path. The fixed host maps stay
// alive until completion, including partial submission failure and destruction.
iree_status_t loom_serve_text_state_grow(loom_serve_text_state_t* state,
                                         iree_host_size_t span_count,
                                         const loom_serve_text_span_t* spans,
                                         const uint32_t* extents) {
  uint64_t required = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    required += loom_serve_text_state_row_growth(
        state->rows[spans[i].row_index], extents[i]);
  }
  bool available = false;
  if (required <= UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_serve_text_cache_ensure_available(
        state, (uint32_t)required, &available));
  }
  if (!available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "text state exceeds logical page addressability");
  }
  iree_hal_transfer_operation_t* uploads = state->cache.uploads;
  iree_host_size_t upload_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    loom_serve_text_state_row_t* row = state->rows[span->row_index];
    status = loom_serve_text_state_row_activate(row);
    if (!iree_status_is_ok(status)) {
      break;
    }
    iree_hal_buffer_t* private_views[5] = {0};
    if (loom_runtime_block_pool_is_shared(&state->recurrent.pool,
                                          row->recurrent.slot)) {
      const uint32_t previous = row->recurrent.slot;
      status = loom_serve_text_row_acquire_recurrent(row);
      if (!iree_status_is_ok(status)) {
        break;
      }
      row->recurrent.anchor = previous;
      if (state->memory.buffers[1]) {
        private_views[TEXT_RECURRENT - TEXT_CONTROL] =
            row->buffers[TEXT_RECURRENT];
      }
    }

    uint32_t count = 0;
    iree_host_size_t map_index = 0;
    uint32_t* blocks = NULL;
    if (state->cache.capacity) {
      const uint32_t needed = (uint32_t)iree_host_size_ceil_div(
          row->position + extents[i], state->storage.block_size);
      uint32_t first = row->block_count;
      const iree_host_size_t row_map =
          span->row_index * state->storage.blocks_per_row;
      if (loom_serve_text_row_has_shared_tail(row)) {
        --first;
        uint32_t* tail = &state->cache.maps[row_map + first];
        row->retiring_tail = *tail;
        loom_runtime_block_pool_acquire(&state->cache.pool, 1, tail);
      }
      const uint32_t new_count =
          iree_max(needed, row->block_count) - row->block_count;
      loom_runtime_block_pool_acquire(
          &state->cache.pool, new_count,
          state->cache.maps + row_map + row->block_count);
      count = row->block_count + new_count - first;
      if (count) {
        map_index = row_map + first;
        blocks = state->cache.maps + map_index;
        row->block_count += new_count;
      }
    }
    if (state->memory.pool &&
        (private_views[TEXT_RECURRENT - TEXT_CONTROL] || count)) {
      status = loom_serve_text_commit_direct_growth(state, row, private_views,
                                                    count, blocks);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    if (!count) {
      continue;
    }
    if (row->retiring_tail != UINT32_MAX) {
      status = loom_serve_text_fork_cache_block(state, row->retiring_tail,
                                                blocks[0]);
      if (!iree_status_is_ok(status)) {
        break;
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
    if (state->draft.row_table) {
      uploads[upload_count] = uploads[upload_count - 1];
      uploads[upload_count++].upload.target_buffer = state->draft.row_table;
    }
  }
  if (iree_status_is_ok(status) && upload_count) {
    uint64_t completion = 0;
    status = loom_runtime_execution_transfer(state->execution, upload_count,
                                             uploads, &completion);
  }
  return status;
}
