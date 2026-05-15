// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar sequence reconstruction window.
//
// Tracks a monotonically increasing sequence stream where observations may
// arrive out of order. The window reconstructs the greatest contiguous
// observed prefix and can hold intrusive nodes that become ready once that
// prefix reaches their required sequence.

#ifndef IREE_NET_CHANNEL_UTIL_SEQUENCE_WINDOW_H_
#define IREE_NET_CHANNEL_UTIL_SEQUENCE_WINDOW_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Intrusive node waiting for a sequence window prefix.
typedef struct iree_net_sequence_node_t {
  // Next node in an intrusive list owned by the sequence window.
  struct iree_net_sequence_node_t* next;
  // Required sequence for this node to become ready.
  uint64_t sequence;
} iree_net_sequence_node_t;

// Reconstructs a contiguous observed prefix from out-of-order observations.
typedef struct iree_net_sequence_window_t {
  // Host allocator used for rare ring growth.
  iree_allocator_t host_allocator;
  // Greatest contiguous observed sequence.
  uint64_t observed_sequence;
  // Power-of-two capacity of the ring arrays.
  iree_host_size_t capacity;
  // Exact sequence tags for out-of-order observed slots.
  uint64_t* observed_sequences;
  // Occupancy state for |observed_sequences|.
  bool* observed_slots;
  // Intrusive pending lists keyed by required sequence.
  iree_net_sequence_node_t** pending_lists;
  // Single allocation backing all ring arrays.
  void* storage;
} iree_net_sequence_window_t;

// Initializes |out_window| with |initial_observed_sequence| as the contiguous
// prefix. |initial_capacity| is rounded up to a power of two and may grow on
// rare out-of-order bursts that exceed the current ring.
iree_status_t iree_net_sequence_window_initialize(
    uint64_t initial_observed_sequence, iree_host_size_t initial_capacity,
    iree_allocator_t host_allocator, iree_net_sequence_window_t* out_window);

// Deinitializes |window| storage. Pending nodes must have been drained by the
// owner before deinitialization.
void iree_net_sequence_window_deinitialize(iree_net_sequence_window_t* window);

// Takes all pending nodes still held by |window| into |out_pending_list|.
//
// The returned intrusive list order is unspecified. The caller owns the nodes
// and may release, fail, or requeue them before deinitializing |window|.
void iree_net_sequence_window_take_pending(
    iree_net_sequence_window_t* window,
    iree_net_sequence_node_t** out_pending_list);

// Ensures |window| can represent |sequence| without further allocation.
//
// Call before side effects that cannot be rolled back when a later observe or
// defer must succeed without allocating.
iree_status_t iree_net_sequence_window_reserve(
    iree_net_sequence_window_t* window, uint64_t sequence);

// Returns the greatest contiguous observed sequence.
static inline uint64_t iree_net_sequence_window_observed(
    const iree_net_sequence_window_t* window) {
  return window->observed_sequence;
}

// Returns true if |sequence| has been observed, even if earlier holes prevent
// it from being part of the contiguous prefix yet.
bool iree_net_sequence_window_has_observed(
    const iree_net_sequence_window_t* window, uint64_t sequence);

// Records |sequence| as observed.
//
// If this closes one or more holes, |out_ready_list| receives all pending nodes
// whose required sequence is now covered by the contiguous prefix. The returned
// list is intrusive and owned by the caller; sequence groups are linked in
// ascending order. On success, |*out_ready_list| is NULL when no nodes became
// ready.
iree_status_t iree_net_sequence_window_observe(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t** out_ready_list);

// Defers |node| until the contiguous observed prefix reaches |sequence|.
//
// If |sequence| is already covered, |out_ready_list| receives |node|
// immediately. Otherwise the node is retained in the window until a future
// observe advances the prefix far enough.
iree_status_t iree_net_sequence_window_defer_until(
    iree_net_sequence_window_t* window, uint64_t sequence,
    iree_net_sequence_node_t* node, iree_net_sequence_node_t** out_ready_list);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_SEQUENCE_WINDOW_H_
