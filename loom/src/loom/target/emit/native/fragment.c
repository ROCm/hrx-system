// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/fragment.h"

#include <inttypes.h>

#include "loom/ops/low/ops.h"

static iree_status_t loom_native_fragment_validate_void_returns(
    const loom_low_schedule_table_t* schedule) {
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_op_t* op = schedule->nodes[i].op;
    if (!loom_low_return_isa(op)) {
      continue;
    }
    loom_value_slice_t values = loom_low_return_values(op);
    if (values.count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "native fragment emission requires ABI-lowered void low.return "
          "terminators; return at schedule node %" PRIhsz " has %" PRIhsz
          " value(s)",
          i, (iree_host_size_t)values.count);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_native_fragment_validate_physical_allocations(
    const loom_low_allocation_table_t* allocation) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "native fragment emission requires physical allocation for register "
          "value %" PRIu32,
          assignment->value_id);
    }
    if (assignment->location_count == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "native fragment emission requires non-empty physical allocation for "
          "register value %" PRIu32,
          assignment->value_id);
    }
  }
  for (iree_host_size_t i = 0; i < allocation->edge_copy_temporary_count; ++i) {
    const loom_low_allocation_edge_copy_temporary_t* temporary =
        &allocation->edge_copy_temporaries[i];
    if (temporary->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "native fragment emission requires physical edge-copy temporary "
          "locations");
    }
  }
  for (iree_host_size_t i = 0; i < allocation->packet_move_temporary_count;
       ++i) {
    const loom_low_allocation_packet_move_temporary_t* temporary =
        &allocation->packet_move_temporaries[i];
    if (temporary->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "native fragment emission requires physical packet-local move "
          "temporary locations");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_native_fragment_validate_emission_inputs(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_low_packet_sequence_t* out_packets) {
  if (out_packets == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native fragment packet sequence output is required");
  }
  *out_packets = (loom_low_packet_sequence_t){0};
  loom_low_packet_sequence_t packets = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocated_packet_sequence_initialize(
      schedule, allocation, &packets));
  if (allocation->spill_plan_count != 0 || allocation->spill_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "native fragment emission requires spill materialization before "
        "spilled allocations reach the emitter; allocation has %" PRIhsz
        " spill plan(s) and %" PRIhsz " spill slot assignment(s)",
        allocation->spill_plan_count, allocation->spill_count);
  }
  IREE_RETURN_IF_ERROR(
      loom_native_fragment_validate_physical_allocations(allocation));
  IREE_RETURN_IF_ERROR(loom_native_fragment_validate_void_returns(schedule));
  *out_packets = packets;
  return iree_ok_status();
}
