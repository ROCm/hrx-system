// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned residual hazard planning over scheduled low packets.
//
// The common plan owns row shape, ordered packet identity, progress accounting
// fields, and generic diagnostic categories. Target policies own the predicates
// that decide whether a descriptor, operand, physical assignment, counter, or
// special register alias creates a residual hazard.

#ifndef LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_H_
#define LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/packet_progress.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent target hazard reason identifiers.
#define LOOM_LOW_PACKET_HAZARD_PLAN_REASON_NONE UINT16_MAX
// Sentinel for absent target residual action identifiers.
#define LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE 0
// Sentinel for absent packet indices in hazard-plan records.
#define LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE IREE_HOST_SIZE_MAX
// Sentinel for absent scheduled ordinals in hazard-plan records.
#define LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE UINT32_MAX

typedef enum loom_low_packet_hazard_plan_record_kind_e {
  // Unknown or uninitialized hazard-plan record kind.
  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNKNOWN = 0,
  // Valid residual action that must be materialized by a target overlay.
  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION = 1,
  // Target policy could not evaluate because required target data is missing.
  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA = 2,
  // Target policy requires allocation but was run before allocation.
  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION = 3,
  // Target policy proved that no finite residual action can satisfy the hazard.
  LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION = 4,
} loom_low_packet_hazard_plan_record_kind_t;

// Target-emitted hazard-plan event for the current packet.
typedef struct loom_low_packet_hazard_plan_event_t {
  // Kind of hazard-plan record to emit.
  loom_low_packet_hazard_plan_record_kind_t kind;
  // Target-owned residual action identifier, or ACTION_NONE for diagnostics.
  uint16_t action_id;
  // Borrowed stable target residual action name. Empty for diagnostics.
  iree_string_view_t action_name;
  // Target-owned reason identifier.
  uint16_t reason_id;
  // Borrowed stable target reason name.
  iree_string_view_t reason_name;
  // Producer schedule-node index, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t producer_node_index;
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Borrowed stable progress-class name.
  iree_string_view_t progress_class_name;
  // Required progress in target-defined units.
  uint32_t required_progress;
  // Observed progress in target-defined units.
  uint32_t observed_progress;
  // Remaining progress in target-defined units.
  uint32_t residual_progress;
} loom_low_packet_hazard_plan_event_t;

// Emits one target hazard-plan event for the packet currently being queried.
typedef iree_status_t (*loom_low_packet_hazard_plan_emit_fn_t)(
    void* user_data, const loom_low_packet_hazard_plan_event_t* event);

// Queries target residual hazard events for one scheduled packet.
//
// |allocation| may be NULL for schedule-only policies. |progress| may be NULL
// when the target policy does not need packet-progress facts. The builder may
// call this function more than once for the same packet while sizing and
// populating the output table. Implementations must be pure for a given packet
// and target state.
typedef iree_status_t (*loom_low_packet_hazard_plan_query_fn_t)(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data);

// Target residual hazard provider used to populate a hazard-plan table.
typedef struct loom_low_packet_hazard_plan_provider_t {
  // Target-owned context passed to |query|.
  void* user_data;
  // Hazard-plan query callback.
  loom_low_packet_hazard_plan_query_fn_t query;
} loom_low_packet_hazard_plan_provider_t;

// One target hazard-plan action or diagnostic attached to a scheduled packet.
typedef struct loom_low_packet_hazard_plan_record_t {
  // Kind of hazard-plan record.
  loom_low_packet_hazard_plan_record_kind_t kind;
  // Target-owned residual action identifier, or ACTION_NONE for diagnostics.
  uint16_t action_id;
  // Borrowed stable target residual action name. Empty for diagnostics.
  iree_string_view_t action_name;
  // Target-owned reason identifier.
  uint16_t reason_id;
  // Borrowed stable target reason name.
  iree_string_view_t reason_name;
  // Producer schedule-node index, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t producer_node_index;
  // Producer packet ordinal, or LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE.
  iree_host_size_t producer_packet_index;
  // Scheduled ordinal of |producer_node_index|, or
  // LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE.
  uint32_t producer_scheduled_ordinal;
  // Consumer or insertion schedule-node index.
  uint32_t consumer_node_index;
  // Consumer or insertion packet ordinal.
  iree_host_size_t insertion_packet_index;
  // Region block containing the insertion point.
  uint32_t block_index;
  // Scheduled ordinal of the insertion point within |block_index|.
  uint32_t scheduled_ordinal;
  // Target-owned progress-class identifier.
  uint16_t progress_class_id;
  // Borrowed stable progress-class name.
  iree_string_view_t progress_class_name;
  // Required progress in target-defined units.
  uint32_t required_progress;
  // Observed progress in target-defined units.
  uint32_t observed_progress;
  // Remaining progress in target-defined units.
  uint32_t residual_progress;
} loom_low_packet_hazard_plan_record_t;

// Ordered target hazard-plan sidecar for one scheduled low function.
typedef struct loom_low_packet_hazard_plan_t {
  // Schedule table walked to build this hazard plan.
  const loom_low_schedule_table_t* schedule;
  // Optional allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Optional progress table paired with |schedule|.
  const loom_low_packet_progress_table_t* progress;
  // Hazard-plan records in scheduled packet order.
  const loom_low_packet_hazard_plan_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
} loom_low_packet_hazard_plan_t;

// Builds target residual hazard records for |schedule| using |provider|.
iree_status_t loom_low_packet_hazard_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_hazard_plan_provider_t* provider,
    iree_arena_allocator_t* arena, loom_low_packet_hazard_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_H_
