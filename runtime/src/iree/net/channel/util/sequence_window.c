// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/sequence_window.h"

//===----------------------------------------------------------------------===//
// Storage
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_sequence_window_storage_layout(
    iree_host_size_t capacity, iree_host_size_t* out_total_size,
    iree_host_size_t* out_observed_sequences_offset,
    iree_host_size_t* out_observed_slots_offset,
    iree_host_size_t* out_pending_lists_offset) {
  return IREE_STRUCT_LAYOUT(
      0, out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(capacity, uint64_t, iree_alignof(uint64_t),
                                out_observed_sequences_offset),
      IREE_STRUCT_FIELD(capacity, bool, out_observed_slots_offset),
      IREE_STRUCT_FIELD_ALIGNED(capacity, iree_net_sequence_node_t*,
                                iree_alignof(iree_net_sequence_node_t*),
                                out_pending_lists_offset));
}

static iree_status_t iree_net_sequence_window_allocate_storage(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    uint64_t** out_observed_sequences, bool** out_observed_slots,
    iree_net_sequence_node_t*** out_pending_lists, void** out_storage) {
  *out_observed_sequences = NULL;
  *out_observed_slots = NULL;
  *out_pending_lists = NULL;
  *out_storage = NULL;

  iree_host_size_t total_size = 0;
  iree_host_size_t observed_sequences_offset = 0;
  iree_host_size_t observed_slots_offset = 0;
  iree_host_size_t pending_lists_offset = 0;
  IREE_RETURN_IF_ERROR(iree_net_sequence_window_storage_layout(
      capacity, &total_size, &observed_sequences_offset, &observed_slots_offset,
      &pending_lists_offset));

  void* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, &storage));
  memset(storage, 0, total_size);

  uint8_t* storage_bytes = (uint8_t*)storage;
  *out_observed_sequences =
      (uint64_t*)(storage_bytes + observed_sequences_offset);
  *out_observed_slots = (bool*)(storage_bytes + observed_slots_offset);
  *out_pending_lists =
      (iree_net_sequence_node_t**)(storage_bytes + pending_lists_offset);
  *out_storage = storage;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Ring Helpers
//===----------------------------------------------------------------------===//

static void iree_net_sequence_window_append_ready_list(
    iree_net_sequence_node_t** ready_list,
    iree_net_sequence_node_t*** ready_tail, iree_net_sequence_node_t* head,
    iree_net_sequence_node_t* tail) {
  if (!head) return;
  **ready_tail = head;
  *ready_tail = &tail->next;
  **ready_tail = NULL;
}

static void iree_net_sequence_window_drain_pending(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t** ready_list,
    iree_net_sequence_node_t*** ready_tail) {
  iree_host_size_t slot = (iree_host_size_t)sequence & (window->capacity - 1);
  iree_net_sequence_node_t* remaining_head = NULL;
  iree_net_sequence_node_t* ready_head = NULL;
  iree_net_sequence_node_t** ready_sequence_tail = &ready_head;
  iree_net_sequence_node_t* ready_sequence_last = NULL;
  iree_net_sequence_node_t* pending = window->pending_lists[slot];
  window->pending_lists[slot] = NULL;
  while (pending) {
    iree_net_sequence_node_t* next = pending->next;
    if (pending->sequence == sequence) {
      *ready_sequence_tail = pending;
      ready_sequence_tail = &pending->next;
      ready_sequence_last = pending;
    } else {
      pending->next = remaining_head;
      remaining_head = pending;
    }
    pending = next;
  }
  *ready_sequence_tail = NULL;
  window->pending_lists[slot] = remaining_head;
  iree_net_sequence_window_append_ready_list(ready_list, ready_tail, ready_head,
                                             ready_sequence_last);
}

iree_status_t iree_net_sequence_window_reserve(
    iree_net_sequence_window_t* window, uint64_t sequence) {
  IREE_ASSERT_ARGUMENT(window);
  if (sequence <= window->observed_sequence) return iree_ok_status();
  uint64_t distance = sequence - window->observed_sequence;
  if (distance <= window->capacity) return iree_ok_status();

  if (distance > (uint64_t)IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence gap %" PRIu64
                            " exceeds addressable window capacity",
                            distance);
  }

  iree_host_size_t new_capacity =
      iree_host_size_next_power_of_two((iree_host_size_t)distance);
  if (new_capacity <= window->capacity) {
    new_capacity = window->capacity * 2;
  }
  if (new_capacity == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence window capacity overflow");
  }

  uint64_t* new_observed_sequences = NULL;
  bool* new_observed_slots = NULL;
  iree_net_sequence_node_t** new_pending_lists = NULL;
  void* new_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_net_sequence_window_allocate_storage(
      new_capacity, window->host_allocator, &new_observed_sequences,
      &new_observed_slots, &new_pending_lists, &new_storage));

  for (iree_host_size_t i = 0; i < window->capacity; ++i) {
    if (window->observed_slots[i]) {
      uint64_t observed_sequence = window->observed_sequences[i];
      iree_host_size_t new_slot =
          (iree_host_size_t)observed_sequence & (new_capacity - 1);
      new_observed_sequences[new_slot] = observed_sequence;
      new_observed_slots[new_slot] = true;
    }
    iree_net_sequence_node_t* pending = window->pending_lists[i];
    while (pending) {
      iree_net_sequence_node_t* next = pending->next;
      iree_host_size_t new_slot =
          (iree_host_size_t)pending->sequence & (new_capacity - 1);
      pending->next = new_pending_lists[new_slot];
      new_pending_lists[new_slot] = pending;
      pending = next;
    }
  }

  iree_allocator_free(window->host_allocator, window->storage);
  window->capacity = new_capacity;
  window->observed_sequences = new_observed_sequences;
  window->observed_slots = new_observed_slots;
  window->pending_lists = new_pending_lists;
  window->storage = new_storage;
  return iree_ok_status();
}

static void iree_net_sequence_window_advance(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t** ready_list) {
  iree_net_sequence_node_t** ready_tail = ready_list;
  while (*ready_tail) {
    ready_tail = &(*ready_tail)->next;
  }

  window->observed_sequence = sequence;
  iree_net_sequence_window_drain_pending(window, sequence, ready_list,
                                         &ready_tail);

  bool advancing = true;
  while (advancing) {
    uint64_t next_sequence = window->observed_sequence + 1;
    iree_host_size_t slot =
        (iree_host_size_t)next_sequence & (window->capacity - 1);
    advancing = window->observed_slots[slot] &&
                window->observed_sequences[slot] == next_sequence;
    if (advancing) {
      window->observed_slots[slot] = false;
      window->observed_sequences[slot] = 0;
      window->observed_sequence = next_sequence;
      iree_net_sequence_window_drain_pending(window, next_sequence, ready_list,
                                             &ready_tail);
    }
  }
}

//===----------------------------------------------------------------------===//
// iree_net_sequence_window_t
//===----------------------------------------------------------------------===//

iree_status_t iree_net_sequence_window_initialize(
    uint64_t initial_observed_sequence, iree_host_size_t initial_capacity,
    iree_allocator_t host_allocator, iree_net_sequence_window_t* out_window) {
  IREE_ASSERT_ARGUMENT(out_window);
  memset(out_window, 0, sizeof(*out_window));

  iree_host_size_t capacity =
      iree_host_size_next_power_of_two(iree_max(initial_capacity, 1));
  if (capacity == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence window initial capacity overflow");
  }

  iree_status_t status = iree_net_sequence_window_allocate_storage(
      capacity, host_allocator, &out_window->observed_sequences,
      &out_window->observed_slots, &out_window->pending_lists,
      &out_window->storage);
  if (iree_status_is_ok(status)) {
    out_window->host_allocator = host_allocator;
    out_window->observed_sequence = initial_observed_sequence;
    out_window->capacity = capacity;
  }
  return status;
}

void iree_net_sequence_window_deinitialize(iree_net_sequence_window_t* window) {
  if (!window || !window->storage) return;
  iree_allocator_t host_allocator = window->host_allocator;
  iree_allocator_free(host_allocator, window->storage);
  memset(window, 0, sizeof(*window));
}

void iree_net_sequence_window_take_pending(
    iree_net_sequence_window_t* window,
    iree_net_sequence_node_t** out_pending_list) {
  IREE_ASSERT_ARGUMENT(out_pending_list);
  *out_pending_list = NULL;
  if (!window || !window->pending_lists) return;

  iree_net_sequence_node_t** pending_tail = out_pending_list;
  for (iree_host_size_t i = 0; i < window->capacity; ++i) {
    iree_net_sequence_node_t* pending_list = window->pending_lists[i];
    window->pending_lists[i] = NULL;
    if (!pending_list) continue;

    *pending_tail = pending_list;
    while (*pending_tail) {
      pending_tail = &(*pending_tail)->next;
    }
  }
}

bool iree_net_sequence_window_has_observed(
    const iree_net_sequence_window_t* window, uint64_t sequence) {
  if (sequence <= window->observed_sequence) return true;
  uint64_t distance = sequence - window->observed_sequence;
  if (distance > window->capacity) return false;
  iree_host_size_t slot = (iree_host_size_t)sequence & (window->capacity - 1);
  return window->observed_slots[slot] &&
         window->observed_sequences[slot] == sequence;
}

iree_status_t iree_net_sequence_window_observe(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t** out_ready_list) {
  IREE_ASSERT_ARGUMENT(window);
  IREE_ASSERT_ARGUMENT(out_ready_list);
  *out_ready_list = NULL;

  if (sequence <= window->observed_sequence) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "sequence %" PRIu64
                            " is already covered by observed prefix %" PRIu64,
                            sequence, window->observed_sequence);
  }

  IREE_RETURN_IF_ERROR(iree_net_sequence_window_reserve(window, sequence));

  if (sequence == window->observed_sequence + 1) {
    iree_net_sequence_window_advance(window, sequence, out_ready_list);
    return iree_ok_status();
  }

  iree_host_size_t slot = (iree_host_size_t)sequence & (window->capacity - 1);
  if (window->observed_slots[slot]) {
    if (window->observed_sequences[slot] == sequence) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "sequence %" PRIu64 " was already observed",
                              sequence);
    }
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "sequence window slot collision for %" PRIu64
                            " and %" PRIu64,
                            sequence, window->observed_sequences[slot]);
  }

  window->observed_sequences[slot] = sequence;
  window->observed_slots[slot] = true;
  return iree_ok_status();
}

iree_status_t iree_net_sequence_window_defer_until(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t* node, iree_net_sequence_node_t** out_ready_list) {
  IREE_ASSERT_ARGUMENT(window);
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(out_ready_list);
  *out_ready_list = NULL;

  node->next = NULL;
  node->sequence = sequence;
  if (sequence <= window->observed_sequence) {
    *out_ready_list = node;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_net_sequence_window_reserve(window, sequence));

  iree_host_size_t slot = (iree_host_size_t)sequence & (window->capacity - 1);
  node->next = window->pending_lists[slot];
  window->pending_lists[slot] = node;
  return iree_ok_status();
}
