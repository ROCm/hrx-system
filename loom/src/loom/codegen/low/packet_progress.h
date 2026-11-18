// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-defined progress accounting over scheduled low packets.
//
// Progress classes are target-owned counters such as issue slots, scoreboard
// domains, latency buckets, stack-depth changes, or barrier epochs. The common
// packet-progress table records the ordered facts; target overlays decide how
// to interpret those facts for waits, diagnostics, or legality checks.

#ifndef LOOM_CODEGEN_LOW_PACKET_PROGRESS_H_
#define LOOM_CODEGEN_LOW_PACKET_PROGRESS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/packet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent progress-class identifiers.
#define LOOM_LOW_PACKET_PROGRESS_CLASS_NONE UINT16_MAX

typedef uint8_t loom_low_packet_progress_action_t;

enum loom_low_packet_progress_action_e {
  // Unknown or uninitialized progress action.
  LOOM_LOW_PACKET_PROGRESS_ACTION_UNKNOWN = 0,
  // Packet advances the progress class by |units|.
  LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE = 1,
  // Packet resets the progress class to its target-defined origin.
  LOOM_LOW_PACKET_PROGRESS_ACTION_RESET = 2,
  // Packet bounds outstanding work in the progress class by |units|. This
  // does not identify completed producers or advance elapsed-work progress.
  LOOM_LOW_PACKET_PROGRESS_ACTION_BOUND = 3,
};

// Maps a target-owned progress-class identifier to its stable diagnostic name.
// This stateless target vocabulary is independent of per-function provider
// state. The progress table retains the function and calls it only when names
// are requested, after provider construction state may have been released.
typedef iree_string_view_t (*loom_low_packet_progress_class_name_fn_t)(
    uint16_t progress_class_id);

// Target-emitted progress fact for the current packet.
typedef struct loom_low_packet_progress_event_t {
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Progress operation performed by the packet.
  loom_low_packet_progress_action_t action;
  // Units advanced for ADVANCE, zero for RESET, or nonzero outstanding-work
  // upper bound for BOUND.
  uint32_t units;
} loom_low_packet_progress_event_t;

// Emits one target progress event for the packet currently being queried.
// Provider construction has already reserved exact storage for every event.
typedef void (*loom_low_packet_progress_emit_fn_t)(
    void* user_data, const loom_low_packet_progress_event_t* event);

// Queries target progress events for one scheduled packet.
//
// The builder calls this function exactly once for each scheduled packet in
// increasing packet-index order. Implementations may advance monotonic state
// in |user_data| across calls. Fallible target analysis must complete before
// provider construction; this callback only projects compiler-owned facts.
typedef void (*loom_low_packet_progress_query_fn_t)(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data);

// Target progress provider used to populate a packet-progress table.
typedef struct loom_low_packet_progress_provider_t {
  // Target-owned context passed to |query|.
  void* user_data;
  // Exact number of progress events emitted across all scheduled packets.
  // Providers establish this without replaying |query|.
  iree_host_size_t event_count;
  // Progress query callback.
  loom_low_packet_progress_query_fn_t query;
  // Stable target vocabulary, required when |event_count| is nonzero.
  loom_low_packet_progress_class_name_fn_t class_name;
} loom_low_packet_progress_provider_t;

// One progress event attached to a scheduled packet.
typedef struct loom_low_packet_progress_record_t {
  // Packet ordinal in final scheduled order.
  iree_host_size_t packet_index;
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Progress operation performed by the packet.
  loom_low_packet_progress_action_t action;
  // Units advanced for ADVANCE, zero for RESET, or nonzero outstanding-work
  // upper bound for BOUND.
  uint32_t units;
} loom_low_packet_progress_record_t;

static_assert(sizeof(loom_low_packet_progress_record_t) ==
                  sizeof(iree_host_size_t) + 8,
              "packet-progress records retain only packet and event facts");

// Ordered target-progress sidecar for one scheduled and allocated low function.
typedef struct loom_low_packet_progress_table_t {
  // Borrowed immutable schedule, which outlives this table. Packet identities
  // and block/ordinal metadata remain indexed in the schedule.
  const loom_low_schedule_table_t* schedule;
  // Allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Progress records in scheduled packet order.
  const loom_low_packet_progress_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
  // Stable target vocabulary retained independently of provider state.
  loom_low_packet_progress_class_name_fn_t class_name;
} loom_low_packet_progress_table_t;

// Builds target progress records for |schedule| using |provider|. Provider
// projection is infallible.
iree_status_t loom_low_packet_progress_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_progress_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_PROGRESS_H_
