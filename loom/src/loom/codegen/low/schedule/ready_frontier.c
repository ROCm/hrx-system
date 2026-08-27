// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/ready_frontier.h"

#include <string.h>

enum {
  LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_SHIFT = 9,
  LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_CAPACITY =
      1u << LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_SHIFT,
  LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_MASK =
      LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_CAPACITY - 1u,
  LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_SHIFT = 12,
  LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_CAPACITY =
      1u << LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_SHIFT,
  LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_MASK =
      LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_CAPACITY - 1u,
};

typedef iree_alignas(64) struct loom_low_schedule_ready_node_state_t {
  // Nomination keys indexed by ready view.
  uint64_t keys[LOOM_LOW_SCHEDULE_READY_VIEW_COUNT];
  // Indexed heap positions, or READY_NODE_NONE when not ready.
  uint32_t heap_positions[LOOM_LOW_SCHEDULE_READY_VIEW_COUNT];
  // Descriptor ordinal, or READY_NODE_NONE for descriptor-less nodes.
  uint32_t descriptor_ordinal;
  // Previous ready node with the same descriptor ordinal.
  uint32_t descriptor_previous_node;
  // Next ready node with the same descriptor ordinal.
  uint32_t descriptor_next_node;
} loom_low_schedule_ready_node_state_t;

typedef iree_alignas(64) struct loom_low_schedule_ready_node_segment_t {
  // Node states indexed by the low bits of a node index.
  loom_low_schedule_ready_node_state_t
      values[LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_CAPACITY];
} loom_low_schedule_ready_node_segment_t;

typedef iree_alignas(64) struct loom_low_schedule_ready_heap_segment_t {
  // Heap node indices indexed by the low bits of a heap position.
  uint32_t values[LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_CAPACITY];
} loom_low_schedule_ready_heap_segment_t;

static_assert(sizeof(loom_low_schedule_ready_node_state_t) == 64,
              "ready node state must remain one cache line");
static_assert(sizeof(loom_low_schedule_ready_node_segment_t) == 32 * 1024,
              "ready node segment must fit in a workspace block");
static_assert(sizeof(loom_low_schedule_ready_heap_segment_t) == 16 * 1024,
              "ready heap segment must fit in a workspace block");

static loom_low_schedule_ready_node_state_t*
loom_low_schedule_ready_frontier_node_state(
    loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index) {
  IREE_ASSERT(node_index < frontier->node_capacity);
  loom_low_schedule_ready_node_segment_t* segment =
      (loom_low_schedule_ready_node_segment_t*)loom_segmented_storage_segment(
          &frontier->node_states,
          node_index >> LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_SHIFT);
  return &segment
              ->values[node_index & LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_MASK];
}

static const loom_low_schedule_ready_node_state_t*
loom_low_schedule_ready_frontier_const_node_state(
    const loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index) {
  IREE_ASSERT(node_index < frontier->node_capacity);
  const loom_low_schedule_ready_node_segment_t* segment =
      (const loom_low_schedule_ready_node_segment_t*)
          loom_segmented_storage_const_segment(
              &frontier->node_states,
              node_index >> LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_SHIFT);
  return &segment
              ->values[node_index & LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_MASK];
}

static uint32_t loom_low_schedule_ready_heap_get(
    const loom_low_schedule_ready_heap_t* heap, uint32_t position) {
  IREE_ASSERT(position < heap->count);
  const loom_low_schedule_ready_heap_segment_t* segment =
      (const loom_low_schedule_ready_heap_segment_t*)
          loom_segmented_storage_const_segment(
              &heap->nodes,
              position >> LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_SHIFT);
  return segment->values[position & LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_MASK];
}

static void loom_low_schedule_ready_heap_set(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t position,
    uint32_t node_index) {
  loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  IREE_ASSERT(position < heap->count);
  loom_low_schedule_ready_heap_segment_t* segment =
      (loom_low_schedule_ready_heap_segment_t*)loom_segmented_storage_segment(
          &heap->nodes, position >> LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_SHIFT);
  segment->values[position & LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_MASK] =
      node_index;
  loom_low_schedule_ready_frontier_node_state(frontier, node_index)
      ->heap_positions[view] = position;
}

static bool loom_low_schedule_ready_node_less(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t left_node,
    uint32_t right_node) {
  const uint64_t left_key =
      loom_low_schedule_ready_frontier_const_node_state(frontier, left_node)
          ->keys[view];
  const uint64_t right_key =
      loom_low_schedule_ready_frontier_const_node_state(frontier, right_node)
          ->keys[view];
  return left_key != right_key ? left_key < right_key : left_node < right_node;
}

static void loom_low_schedule_ready_heap_sift_up(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t position,
    uint32_t node_index) {
  loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  while (position != 0) {
    const uint32_t parent_position = (position - 1u) / 2u;
    const uint32_t parent_node =
        loom_low_schedule_ready_heap_get(heap, parent_position);
    if (loom_low_schedule_ready_node_less(frontier, view, parent_node,
                                          node_index)) {
      break;
    }
    loom_low_schedule_ready_heap_set(frontier, view, position, parent_node);
    position = parent_position;
  }
  loom_low_schedule_ready_heap_set(frontier, view, position, node_index);
}

static void loom_low_schedule_ready_heap_sift_down(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t position,
    uint32_t node_index) {
  loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  while (true) {
    if (heap->count < 2u || position > (heap->count - 2u) / 2u) break;
    const uint32_t left_position = position * 2u + 1u;
    const uint32_t right_position = left_position + 1u;
    uint32_t child_position = left_position;
    if (right_position < heap->count &&
        loom_low_schedule_ready_node_less(
            frontier, view,
            loom_low_schedule_ready_heap_get(heap, right_position),
            loom_low_schedule_ready_heap_get(heap, left_position))) {
      child_position = right_position;
    }
    const uint32_t child_node =
        loom_low_schedule_ready_heap_get(heap, child_position);
    if (loom_low_schedule_ready_node_less(frontier, view, node_index,
                                          child_node)) {
      break;
    }
    loom_low_schedule_ready_heap_set(frontier, view, position, child_node);
    position = child_position;
  }
  loom_low_schedule_ready_heap_set(frontier, view, position, node_index);
}

static void loom_low_schedule_ready_heap_insert(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t node_index) {
  loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  const uint32_t position = heap->count++;
  loom_low_schedule_ready_heap_sift_up(frontier, view, position, node_index);
}

static void loom_low_schedule_ready_heap_remove(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t node_index) {
  loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  loom_low_schedule_ready_node_state_t* state =
      loom_low_schedule_ready_frontier_node_state(frontier, node_index);
  uint32_t position = state->heap_positions[view];
  IREE_ASSERT(position < heap->count);
  const uint32_t replacement_node =
      loom_low_schedule_ready_heap_get(heap, heap->count - 1u);
  --heap->count;
  state->heap_positions[view] = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  if (position == heap->count) return;
  if (position != 0) {
    const uint32_t parent_position = (position - 1u) / 2u;
    const uint32_t parent_node =
        loom_low_schedule_ready_heap_get(heap, parent_position);
    if (loom_low_schedule_ready_node_less(frontier, view, replacement_node,
                                          parent_node)) {
      loom_low_schedule_ready_heap_sift_up(frontier, view, position,
                                           replacement_node);
      return;
    }
  }
  loom_low_schedule_ready_heap_sift_down(frontier, view, position,
                                         replacement_node);
}

iree_status_t loom_low_schedule_ready_frontier_initialize(
    uint32_t node_capacity, uint32_t descriptor_count, uint8_t view_count,
    iree_arena_allocator_t* arena,
    loom_low_schedule_ready_frontier_t* out_frontier) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontier);
  IREE_ASSERT(view_count > 0 &&
              view_count <= LOOM_LOW_SCHEDULE_READY_VIEW_COUNT);
  *out_frontier = (loom_low_schedule_ready_frontier_t){
      .node_capacity = node_capacity,
      .descriptor_count = descriptor_count,
      .view_count = view_count,
  };
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_ready_node_segment_t),
      iree_alignof(loom_low_schedule_ready_node_segment_t),
      &out_frontier->node_states);
  const uint32_t node_segment_count =
      (node_capacity >> LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_SHIFT) +
      ((node_capacity & LOOM_LOW_SCHEDULE_READY_NODE_SEGMENT_MASK) != 0);
  for (uint32_t i = 0; i < node_segment_count; ++i) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &out_frontier->node_states, arena, &segment));
    memset(segment, 0xFF, sizeof(loom_low_schedule_ready_node_segment_t));
  }

  const uint32_t heap_segment_count =
      (node_capacity >> LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_SHIFT) +
      ((node_capacity & LOOM_LOW_SCHEDULE_READY_HEAP_SEGMENT_MASK) != 0);
  for (uint8_t view = 0; view < view_count; ++view) {
    loom_segmented_storage_initialize(
        sizeof(loom_low_schedule_ready_heap_segment_t),
        iree_alignof(loom_low_schedule_ready_heap_segment_t),
        &out_frontier->views[view].nodes);
    for (uint32_t i = 0; i < heap_segment_count; ++i) {
      void* segment = NULL;
      IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
          &out_frontier->views[view].nodes, arena, &segment));
    }
  }

  if (descriptor_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, descriptor_count, sizeof(*out_frontier->descriptor_heads),
        (void**)&out_frontier->descriptor_heads));
    memset(out_frontier->descriptor_heads, 0xFF,
           descriptor_count * sizeof(*out_frontier->descriptor_heads));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, descriptor_count, sizeof(*out_frontier->descriptor_counts),
        (void**)&out_frontier->descriptor_counts));
    memset(out_frontier->descriptor_counts, 0,
           descriptor_count * sizeof(*out_frontier->descriptor_counts));
  }
  return iree_ok_status();
}

bool loom_low_schedule_ready_frontier_contains(
    const loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  return node_index < frontier->node_capacity &&
         loom_low_schedule_ready_frontier_const_node_state(frontier, node_index)
                 ->heap_positions[LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] !=
             LOOM_LOW_SCHEDULE_READY_NODE_NONE;
}

uint32_t loom_low_schedule_ready_frontier_count(
    const loom_low_schedule_ready_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(frontier);
  return frontier->views[LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE].count;
}

void loom_low_schedule_ready_frontier_insert(
    loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index,
    const loom_low_schedule_ready_keys_t* keys, uint32_t descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_ARGUMENT(keys);
  IREE_ASSERT(node_index < frontier->node_capacity);
  IREE_ASSERT(descriptor_ordinal == LOOM_LOW_SCHEDULE_READY_NODE_NONE ||
              descriptor_ordinal < frontier->descriptor_count);
  loom_low_schedule_ready_node_state_t* state =
      loom_low_schedule_ready_frontier_node_state(frontier, node_index);
  IREE_ASSERT_EQ(state->heap_positions[LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE],
                 LOOM_LOW_SCHEDULE_READY_NODE_NONE);
  memcpy(state->keys, keys->values,
         frontier->view_count * sizeof(*state->keys));
  for (uint8_t view = 0; view < frontier->view_count; ++view) {
    loom_low_schedule_ready_heap_insert(
        frontier, (loom_low_schedule_ready_view_t)view, node_index);
  }

  state->descriptor_ordinal = descriptor_ordinal;
  if (descriptor_ordinal == LOOM_LOW_SCHEDULE_READY_NODE_NONE) return;
  const uint32_t previous_head = frontier->descriptor_heads[descriptor_ordinal];
  state->descriptor_previous_node = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  state->descriptor_next_node = previous_head;
  if (previous_head != LOOM_LOW_SCHEDULE_READY_NODE_NONE) {
    loom_low_schedule_ready_frontier_node_state(frontier, previous_head)
        ->descriptor_previous_node = node_index;
  }
  frontier->descriptor_heads[descriptor_ordinal] = node_index;
  ++frontier->descriptor_counts[descriptor_ordinal];
}

void loom_low_schedule_ready_frontier_remove(
    loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(loom_low_schedule_ready_frontier_contains(frontier, node_index));
  loom_low_schedule_ready_node_state_t* state =
      loom_low_schedule_ready_frontier_node_state(frontier, node_index);
  for (uint8_t view = 0; view < frontier->view_count; ++view) {
    loom_low_schedule_ready_heap_remove(
        frontier, (loom_low_schedule_ready_view_t)view, node_index);
  }

  const uint32_t descriptor_ordinal = state->descriptor_ordinal;
  if (descriptor_ordinal == LOOM_LOW_SCHEDULE_READY_NODE_NONE) return;
  const uint32_t previous_node = state->descriptor_previous_node;
  const uint32_t next_node = state->descriptor_next_node;
  if (previous_node == LOOM_LOW_SCHEDULE_READY_NODE_NONE) {
    frontier->descriptor_heads[descriptor_ordinal] = next_node;
  } else {
    loom_low_schedule_ready_frontier_node_state(frontier, previous_node)
        ->descriptor_next_node = next_node;
  }
  if (next_node != LOOM_LOW_SCHEDULE_READY_NODE_NONE) {
    loom_low_schedule_ready_frontier_node_state(frontier, next_node)
        ->descriptor_previous_node = previous_node;
  }
  IREE_ASSERT_NE(frontier->descriptor_counts[descriptor_ordinal], 0u);
  --frontier->descriptor_counts[descriptor_ordinal];
  state->descriptor_ordinal = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  state->descriptor_previous_node = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  state->descriptor_next_node = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
}

static bool loom_low_schedule_ready_heap_position_less(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view,
    const loom_low_schedule_ready_heap_t* heap, uint32_t left_position,
    uint32_t right_position) {
  return loom_low_schedule_ready_node_less(
      frontier, view, loom_low_schedule_ready_heap_get(heap, left_position),
      loom_low_schedule_ready_heap_get(heap, right_position));
}

static void loom_low_schedule_ready_position_heap_insert(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view,
    const loom_low_schedule_ready_heap_t* heap, uint32_t position,
    uint32_t* position_heap, uint8_t* position_count) {
  IREE_ASSERT_LT(*position_count, LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY);
  uint8_t insertion_index = (*position_count)++;
  while (insertion_index != 0) {
    const uint8_t parent_index = (insertion_index - 1u) / 2u;
    if (loom_low_schedule_ready_heap_position_less(
            frontier, view, heap, position_heap[parent_index], position)) {
      break;
    }
    position_heap[insertion_index] = position_heap[parent_index];
    insertion_index = parent_index;
  }
  position_heap[insertion_index] = position;
}

static uint32_t loom_low_schedule_ready_position_heap_pop(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view,
    const loom_low_schedule_ready_heap_t* heap, uint32_t* position_heap,
    uint8_t* position_count) {
  const uint32_t result = position_heap[0];
  const uint32_t replacement = position_heap[--*position_count];
  if (*position_count == 0) return result;
  uint8_t insertion_index = 0;
  while (true) {
    const uint8_t left_index = insertion_index * 2u + 1u;
    if (left_index >= *position_count) break;
    const uint8_t right_index = left_index + 1u;
    uint8_t child_index = left_index;
    if (right_index < *position_count &&
        loom_low_schedule_ready_heap_position_less(frontier, view, heap,
                                                   position_heap[right_index],
                                                   position_heap[left_index])) {
      child_index = right_index;
    }
    if (loom_low_schedule_ready_heap_position_less(
            frontier, view, heap, replacement, position_heap[child_index])) {
      break;
    }
    position_heap[insertion_index] = position_heap[child_index];
    insertion_index = child_index;
  }
  position_heap[insertion_index] = replacement;
  return result;
}

uint8_t loom_low_schedule_ready_frontier_copy_best(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint8_t capacity,
    uint32_t* out_node_indices) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_ARGUMENT(out_node_indices);
  IREE_ASSERT((uint8_t)view < frontier->view_count);
  IREE_ASSERT(capacity > 0 &&
              capacity <= LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY);
  const loom_low_schedule_ready_heap_t* heap = &frontier->views[view];
  if (heap->count == 0) return 0;
  capacity = (uint8_t)iree_min((uint32_t)capacity, heap->count);
  uint32_t position_heap[LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY] = {0};
  uint8_t position_count = 1;
  uint8_t output_count = 0;
  while (output_count < capacity) {
    const uint32_t position = loom_low_schedule_ready_position_heap_pop(
        frontier, view, heap, position_heap, &position_count);
    out_node_indices[output_count++] =
        loom_low_schedule_ready_heap_get(&frontier->views[view], position);
    if (output_count == capacity) break;
    const uint32_t left_position = position * 2u + 1u;
    if (left_position < heap->count) {
      loom_low_schedule_ready_position_heap_insert(
          frontier, view, heap, left_position, position_heap, &position_count);
    }
    const uint32_t right_position = left_position + 1u;
    if (right_position < heap->count) {
      loom_low_schedule_ready_position_heap_insert(
          frontier, view, heap, right_position, position_heap, &position_count);
    }
  }
  return output_count;
}

void loom_low_schedule_ready_frontier_update_key(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t node_index, uint64_t key) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT((uint8_t)view < frontier->view_count);
  IREE_ASSERT(loom_low_schedule_ready_frontier_contains(frontier, node_index));
  loom_low_schedule_ready_node_state_t* state =
      loom_low_schedule_ready_frontier_node_state(frontier, node_index);
  const uint64_t old_key = state->keys[view];
  if (old_key == key) return;
  state->keys[view] = key;
  const uint32_t position = state->heap_positions[view];
  if (position != 0) {
    const uint32_t parent_position = (position - 1u) / 2u;
    const uint32_t parent_node = loom_low_schedule_ready_heap_get(
        &frontier->views[view], parent_position);
    if (loom_low_schedule_ready_node_less(frontier, view, node_index,
                                          parent_node)) {
      loom_low_schedule_ready_heap_sift_up(frontier, view, position,
                                           node_index);
      return;
    }
  }
  loom_low_schedule_ready_heap_sift_down(frontier, view, position, node_index);
}

uint32_t loom_low_schedule_ready_frontier_descriptor_head(
    const loom_low_schedule_ready_frontier_t* frontier,
    uint32_t descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(descriptor_ordinal < frontier->descriptor_count);
  return frontier->descriptor_heads[descriptor_ordinal];
}

uint32_t loom_low_schedule_ready_frontier_descriptor_count(
    const loom_low_schedule_ready_frontier_t* frontier,
    uint32_t descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(descriptor_ordinal < frontier->descriptor_count);
  return frontier->descriptor_counts[descriptor_ordinal];
}
