// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/retention.h"

#include <string.h>

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/reservation_plan.h"
#include "loom/serve/service/text/storage_bounds.h"

void loom_serve_text_retention_add_row(loom_serve_text_row_t* row) {
  row->reclaimable = true;
}

void loom_serve_text_retention_add_checkpoint(
    loom_serve_text_checkpoint_t* checkpoint) {
  ++checkpoint->reclaimable_references;
}

void loom_serve_text_retention_clear(loom_serve_text_model_t* model) {
  for (iree_host_size_t i = 0; i < model->row_count; ++i) {
    model->rows[i]->reclaimable = false;
  }
  for (iree_host_size_t i = 0; i < model->state.checkpoints.capacity; ++i) {
    model->checkpoints[i].reclaimable_references = 0;
  }
}

typedef struct loom_serve_text_retention_t {
  // Borrowed serialized storage owner.
  loom_serve_text_state_t* state;
  // Mapping granularity shared by this owner's elastic roots.
  uint64_t mapping_size;
  // Current root supplied to the backing visitor.
  iree_host_size_t allocation;
  // Owned cold constraint array; growth frees the previous allocation.
  loom_serve_text_storage_bound_t* bounds;
  // Number of initialized constraints.
  iree_host_size_t count;
  // Allocated constraint capacity.
  iree_host_size_t capacity;
} loom_serve_text_retention_t;

static iree_status_t loom_serve_text_retention_append(
    loom_serve_text_retention_t* query, iree_host_size_t allocation,
    uint64_t offset, uint64_t length, uint64_t minimum) {
  if (!length || !minimum || !query->state->memory.buffers[allocation]) {
    return iree_ok_status();
  }
  if (query->count == query->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        query->state->host_allocator, iree_max(32, query->count + 1),
        sizeof(*query->bounds), &query->capacity, (void**)&query->bounds));
  }
  const uint64_t begin = offset / query->mapping_size;
  const uint64_t end = (offset + length - 1) / query->mapping_size + 1;
  query->bounds[query->count++] = (loom_serve_text_storage_bound_t){
      .allocation = allocation,
      .begin = begin,
      .end = end,
      .minimum = minimum == length ? end - begin
                                   : (minimum - 1) / query->mapping_size + 1,
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_text_retention_backing(
    void* user_data, iree_device_size_t offset, iree_device_size_t length) {
  loom_serve_text_retention_t* query = user_data;
  return loom_serve_text_retention_append(query, query->allocation, offset,
                                          length, length);
}

static void loom_serve_text_retention_select(loom_serve_text_state_t* state,
                                             iree_host_size_t allocation,
                                             uint64_t offset, uint64_t length) {
  if (state->memory.buffers[allocation]) {
    loom_runtime_virtual_buffer_select_backing(
        state->memory.buffers[allocation], offset, length);
  }
}

static void loom_serve_text_retention_select_view(
    loom_serve_text_state_t* state, iree_hal_buffer_t* view) {
  if (view) {
    loom_serve_text_retention_select(state, 1,
                                     iree_hal_buffer_byte_offset(view),
                                     iree_hal_buffer_byte_length(view));
  }
}

static void loom_serve_text_retention_remove_blocks(uint32_t* references,
                                                    uint32_t count,
                                                    const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    --references[blocks[i]];
  }
}

static void loom_serve_text_retention_remove_plan(
    const loom_serve_text_state_reservation_t* plan, uint32_t* blocks,
    uint32_t* recurrent) {
  --recurrent[plan->writer];
  if (plan->reader != UINT32_MAX) {
    --recurrent[plan->reader];
  }
  loom_serve_text_retention_remove_blocks(blocks, plan->count, plan->blocks);
  if (plan->tail != UINT32_MAX) {
    --blocks[plan->tail];
  }
  if (plan->cold) {
    --recurrent[plan->restored.recurrent_slot];
    loom_serve_text_retention_remove_blocks(blocks, plan->prefix,
                                            plan->restored.blocks);
  }
}

static iree_status_t loom_serve_text_retention_required_view(
    loom_serve_text_retention_t* query, iree_hal_buffer_t* view) {
  return view ? loom_serve_text_retention_append(
                    query, 1, iree_hal_buffer_byte_offset(view),
                    iree_hal_buffer_byte_length(view),
                    iree_hal_buffer_byte_length(view))
              : iree_ok_status();
}

static iree_status_t loom_serve_text_retention_request(
    loom_serve_text_retention_t* query,
    const loom_serve_text_state_reservation_t* request, bool replan) {
  loom_serve_text_state_t* state = query->state;
  iree_status_t status = iree_ok_status();
  if (!replan) {
    for (iree_host_size_t i = 0;
         i < request->commitment_count && iree_status_is_ok(status); ++i) {
      const loom_runtime_memory_range_t* range = &request->commitments[i];
      status = loom_serve_text_retention_append(
          query, request->commitment_allocations[i], range->offset,
          range->length, range->length);
    }
    return status;
  }
  // Row identity and source-private geometry are fixed across retries. KV and
  // recurrent IDs may change; their logical counts supply movable constraints.
  for (iree_host_size_t i = TEXT_CONTROL;
       i <= TEXT_PROGRESS && iree_status_is_ok(status); ++i) {
    if (i != TEXT_RECURRENT) {
      status = loom_serve_text_retention_required_view(
          query, request->row->buffers[i]);
    }
  }
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT &&
                               iree_status_is_ok(status);
       ++i) {
    status = loom_serve_text_retention_append(query, i, 0,
                                              state->memory.private_lengths[i],
                                              state->memory.private_lengths[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_retention_append(
        query, 6, request->row->index * state->storage.carry_stride,
        state->storage.carry_stride, state->storage.carry_stride);
  }
  if (iree_status_is_ok(status) && request->cold) {
    const iree_host_size_t index =
        state->row_count + (request->checkpoint - state->checkpoints.values);
    status = loom_serve_text_retention_append(
        query, 6, index * state->storage.carry_stride,
        state->storage.carry_stride, state->storage.carry_stride);
  }
  return status;
}

iree_status_t loom_serve_text_retention_forecast(
    loom_serve_text_model_t* model,
    const loom_serve_text_state_reservation_t* request, bool replan,
    loom_runtime_residency_projection_t* out_projection) {
  loom_serve_text_state_t* state = &model->state;
  *out_projection = (loom_runtime_residency_projection_t){
      .resident_bytes = state->memory.statistics.committed_bytes +
                        state->memory.statistics.pending_bytes};
  if (!state->memory.pool || (!request && !out_projection->resident_bytes)) {
    return iree_ok_status();
  }
  // Copy only demand-grown logical metadata, never virtual address capacity.
  const uint32_t block_capacity = state->cache.pool.capacity;
  const uint32_t recurrent_capacity = state->recurrent.pool.capacity;
  uint32_t* references = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      state->host_allocator, (uint64_t)block_capacity + recurrent_capacity,
      sizeof(*references), (void**)&references));
  uint32_t* recurrent = references + block_capacity;
  if (block_capacity) {
    memcpy(references, state->cache.pool.references,
           block_capacity * sizeof(*references));
  }
  memcpy(recurrent, state->recurrent.pool.references,
         recurrent_capacity * sizeof(*recurrent));
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    const loom_serve_text_state_row_t* row = model->rows[i]->state;
    if (model->rows[i]->reclaimable) {
      if (row->block_count) {
        loom_serve_text_retention_remove_blocks(
            references, row->block_count,
            state->cache.maps + i * state->storage.blocks_per_row);
      }
      if (row->recurrent.slot != UINT32_MAX) {
        --recurrent[row->recurrent.slot];
      }
      if (row->recurrent.anchor != UINT32_MAX) {
        --recurrent[row->recurrent.anchor];
      }
      if (row->retiring_tail != UINT32_MAX) {
        --references[row->retiring_tail];
      }
    } else if (row->recurrent.slot != UINT32_MAX) {
      for (iree_host_size_t j = TEXT_CONTROL; j <= TEXT_PROGRESS; ++j) {
        if (j != TEXT_RECURRENT) {
          loom_serve_text_retention_select_view(state, row->buffers[j]);
        }
      }
      loom_serve_text_retention_select(state, 6,
                                       i * state->storage.carry_stride,
                                       state->storage.carry_stride);
    }
  }
  for (iree_host_size_t i = 0; i < state->checkpoints.capacity; ++i) {
    const loom_serve_text_checkpoint_t* handle = &model->checkpoints[i];
    if (!handle->references) {
      continue;
    }
    const loom_serve_text_state_checkpoint_t* checkpoint = handle->state;
    if (handle->references == handle->reclaimable_references) {
      loom_serve_text_retention_remove_blocks(
          references, checkpoint->block_count, checkpoint->blocks);
      if (checkpoint->recurrent_slot != UINT32_MAX) {
        --recurrent[checkpoint->recurrent_slot];
      }
    } else if (checkpoint->recurrent_slot != UINT32_MAX) {
      loom_serve_text_retention_select(
          state, 6, (state->row_count + i) * state->storage.carry_stride,
          state->storage.carry_stride);
    }
  }
  for (const loom_serve_text_state_reservation_t* plan =
           state->prepared_reservations;
       plan; plan = plan->next) {
    if (plan == request || plan->installed) {
      continue;
    }
    // A tentative plan does not charge its unbacked declarations. Existing
    // backing borrowed by its immutable ranges still cannot be discarded.
    loom_serve_text_retention_remove_plan(plan, references, recurrent);
    for (iree_host_size_t i = 0; i < plan->commitment_count; ++i) {
      const loom_runtime_memory_range_t* range = &plan->commitments[i];
      loom_runtime_virtual_buffer_select_backing(range->buffer, range->offset,
                                                 range->length);
    }
  }
  uint64_t block_count = 0, recurrent_count = 0;
  for (uint32_t i = 0; i < block_capacity; ++i) {
    block_count += references[i] != 0;
  }
  for (uint32_t i = 0; i < recurrent_capacity; ++i) {
    recurrent_count += recurrent[i] != 0;
  }
  if (request && replan) {
    // Only this attempt's new writer/reader may be reselected after denial.
    // Other references to the same slot continue to protect its exact backing.
    loom_serve_text_retention_remove_plan(request, references, recurrent);
  }
  for (uint32_t i = 0; i < recurrent_capacity; ++i) {
    if (recurrent[i]) {
      loom_serve_text_retention_select_view(state, state->recurrent.buffers[i]);
    }
  }
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    loom_serve_text_retention_select(state, i, 0,
                                     state->memory.private_lengths[i]);
  }
  iree_allocator_free(state->host_allocator, references);

  loom_serve_text_retention_t query = {
      .state = state,
      .mapping_size =
          loom_runtime_pool_state_geometry(state->pool).mapping_size};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    if (!state->memory.buffers[i]) {
      continue;
    }
    query.allocation = i;
    status =
        iree_status_join(status, loom_runtime_virtual_buffer_visit_backing(
                                     state->memory.buffers[i],
                                     iree_status_is_ok(status)
                                         ? loom_serve_text_retention_backing
                                         : NULL,
                                     &query));
  }
  for (iree_host_size_t i = 0;
       i < state->storage.region_count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    for (iree_host_size_t plane = 0;
         plane < region->blocks.count && iree_status_is_ok(status); ++plane) {
      status = loom_serve_text_retention_append(
          &query, region->allocation,
          region->blocks.origin + plane * region->blocks.stride,
          (state->cache.capacity / state->storage.block_size) *
              region->blocks.block_bytes,
          block_count * region->blocks.block_bytes);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_retention_append(
        &query, 1, state->storage.recurrent.origin,
        (state->storage.recurrent.capacity - 1ull) *
                state->storage.recurrent.stride +
            state->storage.recurrent.length,
        recurrent_count * state->storage.recurrent.length);
  }
  if (iree_status_is_ok(status) && request) {
    status = loom_serve_text_retention_request(&query, request, replan);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t units =
        loom_serve_text_storage_bound(query.count, query.bounds);
    out_projection->minimum_bytes = units > UINT64_MAX / query.mapping_size
                                        ? UINT64_MAX
                                        : units * query.mapping_size;
  }
  iree_allocator_free(state->host_allocator, query.bounds);
  return status;
}
