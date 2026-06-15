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

typedef enum loom_low_packet_progress_action_e {
  // Unknown or uninitialized progress action.
  LOOM_LOW_PACKET_PROGRESS_ACTION_UNKNOWN = 0,
  // Packet advances the progress class by |units|.
  LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE = 1,
  // Packet resets the progress class to its target-defined origin.
  LOOM_LOW_PACKET_PROGRESS_ACTION_RESET = 2,
} loom_low_packet_progress_action_t;

// Target-emitted progress fact for the current packet.
typedef struct loom_low_packet_progress_event_t {
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Borrowed stable progress-class name for diagnostics and traces.
  iree_string_view_t progress_class_name;
  // Progress operation performed by the packet.
  loom_low_packet_progress_action_t action;
  // Units advanced. Must be non-zero for ADVANCE and zero for RESET.
  uint32_t units;
} loom_low_packet_progress_event_t;

// Emits one target progress event for the packet currently being queried.
typedef iree_status_t (*loom_low_packet_progress_emit_fn_t)(
    void* user_data, const loom_low_packet_progress_event_t* event);

// Queries target progress events for one scheduled packet.
//
// The builder may call this function more than once for the same packet while
// sizing and populating the output table. Implementations must be pure for a
// given packet and target state.
typedef iree_status_t (*loom_low_packet_progress_query_fn_t)(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data);

// Target progress provider used to populate a packet-progress table.
typedef struct loom_low_packet_progress_provider_t {
  // Target-owned context passed to |query|.
  void* user_data;
  // Progress query callback.
  loom_low_packet_progress_query_fn_t query;
} loom_low_packet_progress_provider_t;

// One progress event attached to a scheduled packet.
typedef struct loom_low_packet_progress_record_t {
  // Packet ordinal in final scheduled order.
  iree_host_size_t packet_index;
  // Schedule node represented by |packet_index|.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Borrowed stable progress-class name.
  iree_string_view_t progress_class_name;
  // Progress operation performed by the packet.
  loom_low_packet_progress_action_t action;
  // Units advanced. Zero for RESET.
  uint32_t units;
} loom_low_packet_progress_record_t;

// Ordered target-progress sidecar for one scheduled and allocated low function.
typedef struct loom_low_packet_progress_table_t {
  // Schedule table walked to build this progress table.
  const loom_low_schedule_table_t* schedule;
  // Allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Progress records in scheduled packet order.
  const loom_low_packet_progress_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
} loom_low_packet_progress_table_t;

// Builds target progress records for |schedule| using |provider|.
iree_status_t loom_low_packet_progress_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_progress_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_PROGRESS_H_
