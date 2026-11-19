// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>
#include <string.h>

#include "loom/runtime/execution/execution.h"
#include "loom/runtime/storage/relocation.h"
#include "loom/runtime/storage/snapshot.h"
#include "loom/serve/service/text/state.h"
#include "loom/util/index_set.h"

typedef enum loom_serve_text_backing_domain_e {
  LOOM_TEXT_BACKING_PREFIX,
  LOOM_TEXT_BACKING_PRIVATE,
  LOOM_TEXT_BACKING_RECURRENT,
  LOOM_TEXT_BACKING_CACHE,
  LOOM_TEXT_BACKING_ROW_CARRY,
  LOOM_TEXT_BACKING_CHECKPOINT_CARRY,
} loom_serve_text_backing_domain_t;

// An immutable index entry refers to source geometry instead of copying its
// layout or expanding it into physical pages. Source ordering stays intact.
typedef struct loom_serve_text_backing_node_t {
  // Source byte origin, retained as the index ordering key.
  uint64_t begin;
  // Largest exclusive byte end in this balanced interval subtree.
  uint64_t maximum_end;
  // Source region or row ordinal, depending on domain.
  iree_host_size_t ordinal;
  // Source allocation slot shared by this interval subtree.
  uint8_t allocation;
  // Geometry and canonical ownership query for this descriptor.
  uint8_t domain;
  // Row-private binding; repeated cache planes retain their byte origin above.
  uint8_t binding;
} loom_serve_text_backing_node_t;

struct loom_serve_text_backing_t {
  // Allocation policy for this index and its reusable release-plan scratch.
  iree_allocator_t host_allocator;
  // Offsets delimiting each source allocation's sorted interval subtree.
  iree_host_size_t starts[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 1];
  // Source-descriptor index coallocated with this object.
  loom_serve_text_backing_node_t* nodes;
  // Domain for row releases and completed COW sources awaiting maintenance.
  loom_index_set_layout_t row_layout;
  // Pending row work, coallocated and changed only by the row owner.
  uint64_t* rows;
  // Domain for elastic checkpoint carry; empty when that root is fixed.
  loom_index_set_layout_t checkpoint_layout;
  // Resident carry membership derived from checkpoint recurrent ownership.
  uint64_t* checkpoints;
  // Released checkpoint carry awaiting successful maintenance.
  uint64_t* released_checkpoints;
  // Number of release descriptors currently prepared for native submission.
  iree_host_size_t range_count;
  // High-water descriptor capacity, reused across maintenance cuts.
  iree_host_size_t range_capacity;
  // Owned exact release coverage; contains no independent lifetime references.
  loom_runtime_memory_range_t* ranges;
};

static int loom_serve_text_backing_compare(const void* lhs, const void* rhs) {
  const loom_serve_text_backing_node_t* a = lhs;
  const loom_serve_text_backing_node_t* b = rhs;
  if (a->allocation != b->allocation) {
    return (a->allocation > b->allocation) - (a->allocation < b->allocation);
  }
  return (a->begin > b->begin) - (a->begin < b->begin);
}

static uint64_t loom_serve_text_backing_build(
    loom_serve_text_backing_node_t* nodes, iree_host_size_t count) {
  if (!count) {
    return 0;
  }
  const iree_host_size_t middle = count / 2;
  const uint64_t left = loom_serve_text_backing_build(nodes, middle);
  const uint64_t right =
      loom_serve_text_backing_build(nodes + middle + 1, count - middle - 1);
  nodes[middle].maximum_end =
      iree_max(nodes[middle].maximum_end, iree_max(left, right));
  return nodes[middle].maximum_end;
}

static void loom_serve_text_backing_move_membership(
    const loom_index_set_layout_t* old_layout, uint64_t* old_words,
    const loom_index_set_layout_t* layout, uint64_t* words) {
  uint32_t member = loom_index_set_select(old_layout, old_words, 0);
  while (member != LOOM_INDEX_SET_NONE) {
    loom_index_set_insert(layout, words, member);
    // Remove from the old work set only after the replacement allocation has
    // succeeded. The enclosing serialized resize owns both versions here.
    loom_index_set_erase(old_layout, old_words, member);
    member = loom_index_set_select(old_layout, old_words, 0);
  }
}

iree_status_t loom_serve_text_state_backing_initialize(
    loom_serve_text_state_t* state) {
  iree_host_size_t elastic_count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    elastic_count += state->memory.buffers[i] != NULL;
  }
  if (!elastic_count) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 2;
  if (!iree_host_size_checked_mul_add(capacity, state->row_count, 5,
                                      &capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source backing index exceeds host size");
  }
  for (iree_host_size_t i = 0; i < state->storage.region_count; ++i) {
    if (!iree_host_size_checked_add(
            capacity, state->storage.regions[i].blocks.count, &capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source backing index exceeds host size");
    }
  }
  const loom_index_set_layout_t row_layout =
      loom_index_set_calculate_layout((uint32_t)state->row_count);
  const loom_index_set_layout_t checkpoint_layout =
      loom_index_set_calculate_layout(state->memory.buffers[6] &&
                                              state->storage.carry_stride
                                          ? state->checkpoints.pool.capacity
                                          : 0);
  iree_host_size_t size = 0, nodes_offset = 0, rows_offset = 0,
                   checkpoints_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_backing_t), &size,
      IREE_STRUCT_FIELD(capacity, loom_serve_text_backing_node_t,
                        &nodes_offset),
      IREE_STRUCT_FIELD(row_layout.word_count, uint64_t, &rows_offset),
      IREE_STRUCT_FIELD(2 * checkpoint_layout.word_count, uint64_t,
                        &checkpoints_offset)));
  loom_serve_text_backing_t* backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(state->host_allocator, size, (void**)&backing));
  backing->host_allocator = state->host_allocator;
  backing->nodes =
      (loom_serve_text_backing_node_t*)((uint8_t*)backing + nodes_offset);
  backing->row_layout = row_layout;
  backing->rows = (uint64_t*)((uint8_t*)backing + rows_offset);
  backing->checkpoint_layout = checkpoint_layout;
  backing->checkpoints = (uint64_t*)((uint8_t*)backing + checkpoints_offset);
  backing->released_checkpoints =
      backing->checkpoints + checkpoint_layout.word_count;
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    if (state->memory.buffers[i] && state->memory.private_lengths[i]) {
      backing->nodes[count++] = (loom_serve_text_backing_node_t){
          .maximum_end = state->memory.private_lengths[i],
          .allocation = (uint8_t)i,
          .domain = LOOM_TEXT_BACKING_PREFIX};
    }
  }
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
         ++binding) {
      iree_hal_buffer_t* view = state->rows[i]->buffers[binding];
      if (state->memory.buffers[1] && binding != TEXT_RECURRENT && view) {
        const uint64_t begin = iree_hal_buffer_byte_offset(view);
        backing->nodes[count++] = (loom_serve_text_backing_node_t){
            .begin = begin,
            .maximum_end = begin + iree_hal_buffer_byte_length(view),
            .ordinal = i,
            .allocation = 1,
            .domain = LOOM_TEXT_BACKING_PRIVATE,
            .binding = (uint8_t)binding};
      }
    }
    if (state->memory.buffers[6] && state->storage.carry_stride) {
      const uint64_t begin = i * state->storage.carry_stride;
      backing->nodes[count++] = (loom_serve_text_backing_node_t){
          .begin = begin,
          .maximum_end = begin + state->storage.carry_stride,
          .ordinal = i,
          .allocation = 6,
          .domain = LOOM_TEXT_BACKING_ROW_CARRY};
    }
  }
  if (state->memory.buffers[1]) {
    backing->nodes[count++] = (loom_serve_text_backing_node_t){
        .begin = state->storage.recurrent.origin,
        .maximum_end = state->storage.recurrent.origin +
                       (state->storage.recurrent.capacity - 1) *
                           state->storage.recurrent.stride +
                       state->storage.recurrent.length,
        .allocation = 1,
        .domain = LOOM_TEXT_BACKING_RECURRENT};
  }
  if (checkpoint_layout.word_count) {
    const uint64_t begin = state->row_count * state->storage.carry_stride;
    backing->nodes[count++] = (loom_serve_text_backing_node_t){
        .begin = begin,
        .maximum_end = begin + state->checkpoints.pool.capacity *
                                   state->storage.carry_stride,
        .allocation = 6,
        .domain = LOOM_TEXT_BACKING_CHECKPOINT_CARRY};
  }
  for (iree_host_size_t i = 0; i < state->storage.region_count; ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    if (!state->memory.buffers[region->allocation]) {
      continue;
    }
    for (iree_host_size_t plane = 0; plane < region->blocks.count; ++plane) {
      const uint64_t begin =
          region->blocks.origin + plane * region->blocks.stride;
      backing->nodes[count++] = (loom_serve_text_backing_node_t){
          .begin = begin,
          .maximum_end = begin + state->cache.capacity /
                                     state->storage.block_size *
                                     region->blocks.block_bytes,
          .ordinal = i,
          .allocation = (uint8_t)region->allocation,
          .domain = LOOM_TEXT_BACKING_CACHE};
    }
  }
  qsort(backing->nodes, count, sizeof(*backing->nodes),
        loom_serve_text_backing_compare);
  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    backing->starts[i] = cursor;
    while (cursor < count && backing->nodes[cursor].allocation == i) {
      ++cursor;
    }
    loom_serve_text_backing_build(backing->nodes + backing->starts[i],
                                  cursor - backing->starts[i]);
  }
  backing->starts[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT] = count;
  loom_serve_text_backing_t* previous = state->memory.index;
  if (previous) {
    loom_serve_text_backing_move_membership(
        &previous->row_layout, previous->rows, &row_layout, backing->rows);
    loom_serve_text_backing_move_membership(
        &previous->checkpoint_layout, previous->checkpoints, &checkpoint_layout,
        backing->checkpoints);
    loom_serve_text_backing_move_membership(
        &previous->checkpoint_layout, previous->released_checkpoints,
        &checkpoint_layout, backing->released_checkpoints);
    backing->ranges = previous->ranges;
    backing->range_capacity = previous->range_capacity;
    iree_allocator_free(state->host_allocator, previous);
  }
  state->memory.index = backing;
  if (state->memory.statistics.committed_bytes ||
      state->memory.statistics.pending_bytes) {
    state->flags |= LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
  }
  return iree_ok_status();
}

void loom_serve_text_state_backing_deinitialize(
    loom_serve_text_state_t* state) {
  loom_serve_text_backing_t* backing = state->memory.index;
  if (backing) {
    iree_allocator_free(state->host_allocator, backing->ranges);
    iree_allocator_free(state->host_allocator, backing);
    state->memory.index = NULL;
  }
}

void loom_serve_text_state_backing_row_changed(
    loom_serve_text_state_row_t* row) {
  loom_serve_text_backing_t* backing = row->owner->memory.index;
  if (!backing) {
    return;
  }
  if (row->recurrent.slot == UINT32_MAX || row->retiring_tail != UINT32_MAX) {
    loom_index_set_insert(&backing->row_layout, backing->rows,
                          (uint32_t)row->index);
  } else {
    loom_index_set_erase(&backing->row_layout, backing->rows,
                         (uint32_t)row->index);
  }
}

void loom_serve_text_state_backing_checkpoint_changed(
    loom_serve_text_state_checkpoint_t* checkpoint) {
  loom_serve_text_state_t* state = checkpoint->owner;
  loom_serve_text_backing_t* backing = state->memory.index;
  if (!backing || !backing->checkpoint_layout.word_count) {
    return;
  }
  const uint32_t index = (uint32_t)(checkpoint - state->checkpoints.values);
  if (checkpoint->recurrent_slot != UINT32_MAX) {
    loom_index_set_insert(&backing->checkpoint_layout, backing->checkpoints,
                          index);
    loom_index_set_erase(&backing->checkpoint_layout,
                         backing->released_checkpoints, index);
  } else {
    loom_index_set_erase(&backing->checkpoint_layout, backing->checkpoints,
                         index);
    loom_index_set_insert(&backing->checkpoint_layout,
                          backing->released_checkpoints, index);
  }
}

typedef struct loom_serve_text_backing_interval_t {
  // First byte covered by a surviving owner, or the query end when empty.
  uint64_t begin;
  // Exclusive byte end of that owner's contribution.
  uint64_t end;
} loom_serve_text_backing_interval_t;

static loom_serve_text_backing_interval_t loom_serve_text_backing_owned(
    const loom_serve_text_state_t* state,
    const loom_serve_text_backing_node_t* node, uint64_t begin, uint64_t end,
    bool compact) {
  uint64_t stride = 1, length = 0;
  uint32_t capacity = 1;
  const loom_runtime_block_pool_t* pool = NULL;
  switch (node->domain) {
    case LOOM_TEXT_BACKING_PREFIX:
      length = state->memory.private_lengths[node->allocation];
      break;
    case LOOM_TEXT_BACKING_PRIVATE:
      if (state->rows[node->ordinal]->recurrent.slot != UINT32_MAX) {
        length = iree_hal_buffer_byte_length(
            state->rows[node->ordinal]->buffers[node->binding]);
      }
      break;
    case LOOM_TEXT_BACKING_ROW_CARRY:
      if (state->rows[node->ordinal]->recurrent.slot != UINT32_MAX) {
        length = state->storage.carry_stride;
      }
      break;
    case LOOM_TEXT_BACKING_RECURRENT:
      stride = state->storage.recurrent.stride;
      length = state->storage.recurrent.length;
      pool = &state->recurrent.pool;
      capacity = pool->capacity;
      break;
    case LOOM_TEXT_BACKING_CACHE:
      stride = length =
          state->storage.regions[node->ordinal].blocks.block_bytes;
      pool = &state->cache.pool;
      capacity = compact ? pool->capacity - pool->available : pool->capacity;
      break;
    case LOOM_TEXT_BACKING_CHECKPOINT_CARRY:
      stride = length = state->storage.carry_stride;
      capacity = state->checkpoints.pool.capacity;
      break;
  }
  loom_serve_text_backing_interval_t result = {end, end};
  if (!length || !capacity || end <= node->begin) {
    return result;
  }
  const uint64_t first_unbounded =
      begin < node->begin + length
          ? 0
          : (begin - node->begin - length) / stride + 1;
  const uint64_t end_unbounded = (end - node->begin - 1) / stride + 1;
  const uint32_t first = (uint32_t)iree_min(first_unbounded, capacity);
  const uint32_t last = (uint32_t)iree_min(end_unbounded, capacity);
  if (first >= last) {
    return result;
  }
  uint32_t member = first;
  if (pool && !(compact && node->domain == LOOM_TEXT_BACKING_CACHE)) {
    member = loom_runtime_block_pool_find_owned(pool, first, last - first);
  } else if (node->domain == LOOM_TEXT_BACKING_CHECKPOINT_CARRY) {
    const loom_serve_text_backing_t* backing = state->memory.index;
    member = loom_index_set_select(&backing->checkpoint_layout,
                                   backing->checkpoints, first);
  }
  if (member < first || member >= last) {
    return result;
  }
  result.begin = node->begin + member * stride;
  result.end = result.begin + length;
  // A dense compaction prefix is one continuous contribution per plane.
  if (compact && node->domain == LOOM_TEXT_BACKING_CACHE) {
    result.end = node->begin + capacity * stride;
  }
  return result;
}

static void loom_serve_text_backing_find(
    const loom_serve_text_state_t* state,
    const loom_serve_text_backing_node_t* nodes, iree_host_size_t count,
    uint64_t begin, bool compact, loom_serve_text_backing_interval_t* match) {
  if (!count) {
    return;
  }
  const iree_host_size_t middle = count / 2;
  if (nodes[middle].maximum_end <= begin) {
    return;
  }
  loom_serve_text_backing_find(state, nodes, middle, begin, compact, match);
  if (nodes[middle].begin >= match->begin) {
    return;
  }
  const loom_serve_text_backing_interval_t candidate =
      loom_serve_text_backing_owned(state, &nodes[middle], begin, match->begin,
                                    compact);
  if (candidate.begin < match->begin) {
    *match = candidate;
  }
  loom_serve_text_backing_find(state, nodes + middle + 1, count - middle - 1,
                               begin, compact, match);
}

static iree_status_t loom_serve_text_backing_append(
    loom_serve_text_state_t* state, iree_host_size_t allocation, uint64_t begin,
    uint64_t end) {
  loom_serve_text_backing_t* backing = state->memory.index;
  loom_runtime_virtual_buffer_t* buffer = state->memory.buffers[allocation];
  if (backing->range_count) {
    loom_runtime_memory_range_t* last =
        &backing->ranges[backing->range_count - 1];
    if (last->buffer == buffer && begin <= last->offset + last->length &&
        end >= last->offset) {
      const uint64_t new_end = iree_max(end, last->offset + last->length);
      last->offset = iree_min(begin, last->offset);
      last->length = new_end - last->offset;
      return iree_ok_status();
    }
  }
  if (backing->range_count == backing->range_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        state->host_allocator, backing->range_count + 1,
        sizeof(*backing->ranges), &backing->range_capacity,
        (void**)&backing->ranges));
  }
  backing->ranges[backing->range_count++] =
      (loom_runtime_memory_range_t){buffer, begin, end - begin};
  return iree_ok_status();
}

// A released logical interval may share its boundary mappings with other
// source domains. Jump between actual surviving contributions, preserving each
// native page they touch; empty spans need one geometry query, not a page walk.
static iree_status_t loom_serve_text_backing_release(
    loom_serve_text_state_t* state, iree_host_size_t allocation,
    uint64_t offset, uint64_t length) {
  loom_runtime_virtual_buffer_t* buffer = state->memory.buffers[allocation];
  if (!buffer || !length) {
    return iree_ok_status();
  }
  loom_serve_text_backing_t* backing = state->memory.index;
  const uint64_t size = loom_runtime_virtual_buffer_mapping_size(buffer);
  const uint64_t limit =
      iree_hal_buffer_byte_length(loom_runtime_virtual_buffer_handle(buffer));
  const uint64_t exact_end = offset + length;
  const uint64_t end =
      exact_end + iree_min((size - exact_end % size) % size, limit - exact_end);
  uint64_t begin = offset - offset % size;
  const iree_host_size_t first = backing->starts[allocation];
  const iree_host_size_t count = backing->starts[allocation + 1] - first;
  iree_status_t status = iree_ok_status();
  while (begin < end && iree_status_is_ok(status)) {
    loom_serve_text_backing_interval_t live = {end, end};
    loom_serve_text_backing_find(state, backing->nodes + first, count, begin,
                                 false, &live);
    const uint64_t free_end =
        live.begin == end ? end : live.begin - live.begin % size;
    if (begin < free_end) {
      status =
          loom_serve_text_backing_append(state, allocation, begin, free_end);
    }
    begin = live.end >= end
                ? end
                : live.end +
                      iree_min((size - live.end % size) % size, end - live.end);
  }
  return status;
}

// Successful model readiness includes the native signal; join its final access
// before reusing the state-owned operation slot or editing its slabs again.
// A prefix pin can leave completed backing here until the next capacity change.
void loom_serve_text_state_complete_backing(loom_serve_text_state_t* state) {
  const bool complete = loom_runtime_memory_operation_is_complete(
      state->memory.backing.operation);
  IREE_ASSERT(complete);
  loom_serve_text_state_backing_update(state, state->memory.backing.operation);
  loom_runtime_memory_operation_free(state->memory.backing.operation);
  state->memory.backing.operation = NULL;
}

static bool loom_serve_text_backing_has_releases(
    const loom_serve_text_state_t* state) {
  const loom_serve_text_backing_t* backing = state->memory.index;
  return backing && (loom_runtime_block_pool_find_released(&state->cache.pool,
                                                           0) != UINT32_MAX ||
                     loom_runtime_block_pool_find_released(
                         &state->recurrent.pool, 0) != UINT32_MAX ||
                     loom_index_set_select(&backing->row_layout, backing->rows,
                                           0) != LOOM_INDEX_SET_NONE ||
                     loom_index_set_select(&backing->checkpoint_layout,
                                           backing->released_checkpoints,
                                           0) != LOOM_INDEX_SET_NONE);
}

bool loom_serve_text_state_needs_compaction(
    const loom_serve_text_state_t* state) {
  return state->memory.index && !state->prepared_reservations &&
         (iree_any_bit_set(state->flags,
                           LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING) ||
          loom_serve_text_backing_has_releases(state));
}

// Accepted native work owns its captured edits. Acknowledge now so later
// logical releases accumulate independently and cannot be lost at completion.
static void loom_serve_text_backing_acknowledge(
    loom_serve_text_state_t* state) {
  loom_serve_text_backing_t* backing = state->memory.index;
  if (loom_serve_text_backing_has_releases(state)) {
    state->flags |= LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
  }
  loom_runtime_block_pool_acknowledge_releases(&state->cache.pool);
  loom_runtime_block_pool_acknowledge_releases(&state->recurrent.pool);
  loom_index_set_clear(&backing->row_layout, backing->rows);
  loom_index_set_clear(&backing->checkpoint_layout,
                       backing->released_checkpoints);
}

// Selection wraps for normal allocation. Maintenance consumes each work item
// once in increasing order and terminates at that wrap.
static uint32_t loom_serve_text_backing_next(
    const loom_index_set_layout_t* layout, const uint64_t* words,
    uint32_t capacity, uint32_t start) {
  if (start >= capacity) {
    return LOOM_INDEX_SET_NONE;
  }
  const uint32_t next = loom_index_set_select(layout, words, start);
  return next >= start ? next : LOOM_INDEX_SET_NONE;
}

static void loom_serve_text_backing_update_range(
    loom_serve_text_state_t* state, iree_hal_buffer_binding_t range,
    bool binding);

bool loom_serve_text_state_prepare_reclaim(loom_serve_text_state_t* state) {
  loom_serve_text_state_complete_backing(state);
  const loom_serve_text_backing_t* backing = state->memory.index;
  if (!backing) {
    return false;
  }
  uint32_t index = loom_serve_text_backing_next(
      &backing->row_layout, backing->rows, (uint32_t)state->row_count, 0);
  while (index != LOOM_INDEX_SET_NONE) {
    // Completed prefix copies can retire their old COW sources here. A later
    // epoch may establish new borrowers before physical release is captured.
    loom_serve_text_state_row_complete_reserve(state->rows[index]);
    index = loom_serve_text_backing_next(&backing->row_layout, backing->rows,
                                         (uint32_t)state->row_count, index + 1);
  }
  return loom_serve_text_backing_has_releases(state);
}

static iree_status_t loom_serve_text_release_backing(
    loom_serve_text_state_t* state, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_runtime_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_text_backing_t* backing = state->memory.index;
  backing->range_count = 0;
  iree_status_t status = iree_ok_status();
  uint32_t row_index = loom_serve_text_backing_next(
      &backing->row_layout, backing->rows, (uint32_t)state->row_count, 0);
  while (row_index != LOOM_INDEX_SET_NONE && iree_status_is_ok(status)) {
    loom_serve_text_state_row_t* row = state->rows[row_index];
    if (row->recurrent.slot == UINT32_MAX) {
      for (iree_host_size_t binding = TEXT_CONTROL;
           binding <= TEXT_PROGRESS && iree_status_is_ok(status); ++binding) {
        iree_hal_buffer_t* view = row->buffers[binding];
        if (binding != TEXT_RECURRENT && view) {
          status = loom_serve_text_backing_release(
              state, 1, iree_hal_buffer_byte_offset(view),
              iree_hal_buffer_byte_length(view));
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_backing_release(
            state, 6, row_index * state->storage.carry_stride,
            state->storage.carry_stride);
      }
    }
    row_index =
        loom_serve_text_backing_next(&backing->row_layout, backing->rows,
                                     (uint32_t)state->row_count, row_index + 1);
  }
  for (uint32_t block =
           loom_runtime_block_pool_find_released(&state->recurrent.pool, 0);
       block != UINT32_MAX && iree_status_is_ok(status);
       block = loom_runtime_block_pool_find_released(&state->recurrent.pool,
                                                     block + 1)) {
    status = loom_serve_text_backing_release(
        state, 1,
        state->storage.recurrent.origin +
            block * state->storage.recurrent.stride,
        state->storage.recurrent.length);
  }
  const uint32_t first =
      loom_runtime_block_pool_find_released(&state->cache.pool, 0);
  for (iree_host_size_t i = 0;
       first != UINT32_MAX && i < state->storage.region_count &&
       iree_status_is_ok(status);
       ++i) {
    const loom_serve_text_cache_region_t* region = &state->storage.regions[i];
    for (iree_host_size_t plane = 0;
         plane < region->blocks.count && iree_status_is_ok(status); ++plane) {
      for (uint32_t block = first;
           block != UINT32_MAX && iree_status_is_ok(status);
           block = loom_runtime_block_pool_find_released(&state->cache.pool,
                                                         block + 1)) {
        status = loom_serve_text_backing_release(
            state, region->allocation,
            region->blocks.origin + plane * region->blocks.stride +
                block * region->blocks.block_bytes,
            region->blocks.block_bytes);
      }
    }
  }
  uint32_t checkpoint = loom_serve_text_backing_next(
      &backing->checkpoint_layout, backing->released_checkpoints,
      backing->checkpoint_layout.word_count ? state->checkpoints.pool.capacity
                                            : 0,
      0);
  while (checkpoint != LOOM_INDEX_SET_NONE && iree_status_is_ok(status)) {
    status = loom_serve_text_backing_release(
        state, 6, (state->row_count + checkpoint) * state->storage.carry_stride,
        state->storage.carry_stride);
    checkpoint = loom_serve_text_backing_next(
        &backing->checkpoint_layout, backing->released_checkpoints,
        state->checkpoints.pool.capacity, checkpoint + 1);
  }
  if (iree_status_is_ok(status) && backing->range_count) {
    status = loom_runtime_memory_pool_trim_async(
        state->memory.pool, backing->range_count, backing->ranges, readiness,
        value, out_operation);
  }
  if (iree_status_is_ok(status)) {
    // Invalidate at capture, not completion: collection may happen after a new
    // admission has already reused these IDs and established fresh backing.
    for (iree_host_size_t i = 0; i < backing->range_count; ++i) {
      const loom_runtime_memory_range_t range = backing->ranges[i];
      loom_serve_text_backing_update_range(
          state,
          (iree_hal_buffer_binding_t){
              loom_runtime_virtual_buffer_handle(range.buffer), range.offset,
              range.length},
          false);
    }
    loom_serve_text_backing_acknowledge(state);
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

static iree_status_t loom_serve_text_trim_backing(
    loom_serve_text_state_t* state) {
  IREE_ASSERT(!state->memory.backing.operation);
  IREE_RETURN_IF_ERROR(loom_serve_text_release_backing(
      state, state->memory.backing.semaphore, state->memory.backing.value + 1,
      &state->memory.backing.operation));
  if (state->memory.index->range_count) {
    ++state->memory.backing.value;
  }
  if (!state->memory.backing.operation) {
    return iree_ok_status();
  }
  return loom_runtime_execution_barrier(
      state->execution,
      (iree_hal_semaphore_list_t){1, &state->memory.backing.semaphore,
                                  &state->memory.backing.value});
}

iree_status_t loom_serve_text_state_submit_reclaim(
    loom_serve_text_state_t* state) {
  IREE_ASSERT(!state->memory.reclaim);
  if (!state->memory.index) {
    return iree_ok_status();
  }
  return loom_serve_text_release_backing(state, NULL, 0,
                                         &state->memory.reclaim);
}

iree_status_t loom_serve_text_state_collect_reclaim(
    loom_serve_text_state_t* state) {
  if (!state->memory.reclaim ||
      !loom_runtime_memory_operation_is_complete(state->memory.reclaim)) {
    return iree_ok_status();
  }
  iree_status_t status =
      loom_runtime_memory_operation_query_status(state->memory.reclaim);
  loom_runtime_memory_operation_free(state->memory.reclaim);
  state->memory.reclaim = NULL;
  return status;
}

// Compaction profitability is a geometry query. It never marks live mappings
// or destroys the sparse ownership facts needed for admission and retirement.
static uint64_t loom_serve_text_backing_compact_bytes(
    const loom_serve_text_state_t* state) {
  const loom_serve_text_backing_t* backing = state->memory.index;
  uint64_t bytes = 0;
  for (iree_host_size_t allocation = 0;
       allocation < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT; ++allocation) {
    const loom_runtime_virtual_buffer_t* buffer =
        state->memory.buffers[allocation];
    if (!buffer) {
      continue;
    }
    const uint64_t limit =
        iree_hal_buffer_byte_length(loom_runtime_virtual_buffer_handle(buffer));
    const uint64_t size = loom_runtime_virtual_buffer_mapping_size(buffer);
    const iree_host_size_t first = backing->starts[allocation];
    const iree_host_size_t count = backing->starts[allocation + 1] - first;
    uint64_t cursor = 0;
    while (cursor < limit) {
      loom_serve_text_backing_interval_t live = {limit, limit};
      loom_serve_text_backing_find(state, backing->nodes + first, count, cursor,
                                   true, &live);
      if (live.begin == limit) {
        break;
      }
      const uint64_t begin = iree_max(cursor, live.begin - live.begin % size);
      const uint64_t end = live.end + iree_min((size - live.end % size) % size,
                                               limit - live.end);
      bytes += end - begin;
      cursor = end;
    }
  }
  return bytes;
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
    iree_host_size_t commitment_count = 0;
    for (iree_host_size_t i = 0; i < range_count; ++i) {
      loom_runtime_virtual_buffer_t* buffer =
          state->memory.buffers[ranges[i].buffer_index];
      if (buffer) {
        commitments[commitment_count++] = (loom_runtime_memory_range_t){
            buffer, ranges[i].offset, ranges[i].length};
      }
    }
    IREE_ASSERT(!state->memory.backing.operation);
    loom_runtime_memory_admission_t admission;
    status = loom_runtime_memory_pool_commit_async(
        state->memory.pool, commitment_count, commitments,
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
  loom_serve_text_state_prepare_reclaim(state);
  state->memory.trim.released_before = state->memory.statistics.released_bytes;
  state->memory.trim.phase = LOOM_SERVICE_TEXT_TRIM_COMPLETE;
  if (!state->memory.index) {
    return iree_ok_status();
  }
  // Prepared continuations retain exact logical IDs, not relocation entries.
  // Dead backing can still retire; moving IDs waits until their plans retire.
  IREE_ASSERT(!state->memory.reclaim);
  state->memory.trim.phase = loom_serve_text_state_needs_compaction(state)
                                 ? LOOM_SERVICE_TEXT_TRIM_RELOCATE
                                 : LOOM_SERVICE_TEXT_TRIM_COMPLETE;
  return loom_serve_text_trim_backing(state);
}

static iree_status_t loom_serve_text_submit_compaction(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result) {
  state->flags &= ~LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
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
  if (out_result->moved_blocks &&
      loom_serve_text_backing_compact_bytes(state) >=
          state->memory.statistics.committed_bytes) {
    out_result->moved_blocks = 0;
  }
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    bool admitted = false;
    status = loom_serve_text_try_back_compaction(
        state, out_result->moved_blocks, &admitted);
    if (!admitted) {
      out_result->moved_blocks = 0;
      // This attempt proved benefit, not that shared destination capacity will
      // remain unavailable. A later pressure request may retry after peers
      // release backing; the failed attempt itself supplies no progress wake.
      state->flags |= LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
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

static bool loom_serve_text_recurrent_is_backed(
    const loom_serve_text_state_t* state, uint32_t slot) {
  return loom_runtime_virtual_buffer_is_backed(
      state->memory.buffers[1],
      state->storage.recurrent.origin + slot * state->storage.recurrent.stride,
      state->storage.recurrent.length);
}

static bool loom_serve_text_block_is_backed(
    const loom_serve_text_state_t* state, uint32_t block) {
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

// Only a changed native range can change an established logical preference.
// The interval index carries source geometry; native runs carry exact edits.
static void loom_serve_text_backing_update_nodes(
    loom_serve_text_state_t* state, const loom_serve_text_backing_node_t* nodes,
    iree_host_size_t count, uint64_t begin, uint64_t end, bool binding) {
  if (!count) {
    return;
  }
  const iree_host_size_t middle = count / 2;
  const loom_serve_text_backing_node_t* node = &nodes[middle];
  if (node->maximum_end <= begin) {
    return;
  }
  loom_serve_text_backing_update_nodes(state, nodes, middle, begin, end,
                                       binding);
  if (node->begin >= end) {
    return;
  }
  loom_runtime_block_pool_t* pool = NULL;
  uint64_t stride = 0, length = 0;
  if (node->domain == LOOM_TEXT_BACKING_RECURRENT) {
    pool = &state->recurrent.pool;
    stride = state->storage.recurrent.stride;
    length = state->storage.recurrent.length;
  } else if (node->domain == LOOM_TEXT_BACKING_CACHE) {
    pool = &state->cache.pool;
    stride = length = state->storage.regions[node->ordinal].blocks.block_bytes;
  }
  if (pool && length) {
    const uint64_t first_unbounded =
        begin < node->begin + length
            ? 0
            : (begin - node->begin - length) / stride + 1;
    const uint64_t last_unbounded = (end - node->begin - 1) / stride + 1;
    const uint32_t first = (uint32_t)iree_min(first_unbounded, pool->capacity);
    const uint32_t last = (uint32_t)iree_min(last_unbounded, pool->capacity);
    for (uint32_t block = first; block < last; ++block) {
      if (loom_runtime_block_pool_is_preferred(pool, block) == binding) {
        continue;
      }
      const bool preferred =
          binding && (node->domain == LOOM_TEXT_BACKING_RECURRENT
                          ? loom_serve_text_recurrent_is_backed(state, block)
                          : loom_serve_text_block_is_backed(state, block));
      loom_runtime_block_pool_set_preferred(pool, block, preferred);
    }
  }
  loom_serve_text_backing_update_nodes(state, nodes + middle + 1,
                                       count - middle - 1, begin, end, binding);
}

static void loom_serve_text_backing_update_range(
    loom_serve_text_state_t* state, iree_hal_buffer_binding_t range,
    bool binding) {
  const loom_serve_text_backing_t* backing = state->memory.index;
  for (iree_host_size_t allocation = 0;
       backing && allocation < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++allocation) {
    if (!state->memory.buffers[allocation] ||
        loom_runtime_virtual_buffer_handle(state->memory.buffers[allocation]) !=
            range.buffer) {
      continue;
    }
    const iree_host_size_t first = backing->starts[allocation];
    loom_serve_text_backing_update_nodes(
        state, backing->nodes + first, backing->starts[allocation + 1] - first,
        range.offset, range.offset + range.length, binding);
    break;
  }
}

void loom_serve_text_state_backing_update(
    loom_serve_text_state_t* state,
    const loom_runtime_memory_operation_t* operation) {
  const iree_host_size_t count =
      loom_runtime_memory_operation_mapping_count(operation);
  if (count) {
    state->flags |= LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_buffer_binding_t range;
    const iree_hal_memory_mapping_operation_t kind =
        loom_runtime_memory_operation_mapping(operation, i, &range);
    loom_serve_text_backing_update_range(
        state, range, kind == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND);
  }
}

void loom_serve_text_state_backing_extend(
    loom_serve_text_state_t* state, uint32_t previous_cache_capacity,
    uint32_t previous_recurrent_capacity) {
  if (!state->memory.index) {
    return;
  }
  for (uint32_t block = previous_cache_capacity;
       block < state->cache.pool.capacity; ++block) {
    loom_runtime_block_pool_set_preferred(
        &state->cache.pool, block,
        loom_serve_text_block_is_backed(state, block));
  }
  if (state->memory.buffers[1]) {
    for (uint32_t slot = previous_recurrent_capacity;
         slot < state->recurrent.pool.capacity; ++slot) {
      loom_runtime_block_pool_set_preferred(
          &state->recurrent.pool, slot,
          loom_serve_text_recurrent_is_backed(state, slot));
    }
  }
}

iree_status_t loom_serve_text_state_complete_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* result,
    bool* out_complete) {
  *out_complete = false;
  loom_serve_text_state_complete_backing(state);
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
  if (*out_complete && result->moved_blocks) {
    state->flags &= ~LOOM_SERVE_TEXT_STATE_FLAG_COMPACTION_PENDING;
  }
  if (*out_complete) {
    result->released_bytes = state->memory.statistics.released_bytes -
                             state->memory.trim.released_before;
  }
  return iree_ok_status();
}
