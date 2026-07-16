// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/compile_report_planning_format.h"

#include <inttypes.h>

iree_status_t loom_target_compile_report_append_low_planning_text_fields(
    const loom_low_planning_statistics_t* statistics,
    iree_string_builder_t* builder) {
  const loom_low_planning_repair_statistics_t* repair = &statistics->repair;
  const loom_low_planning_memory_statistics_t* memory = &statistics->memory;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " frame_builds=%" PRIu64 " allocation_runs=%" PRIu64
      " repair_iterations=%" PRIu64 " diagnostic_replays=%" PRIu64
      " spill_traffic_lowerings=%" PRIu64 " rematerialized_operands=%" PRIu64
      " live_range_split_operands=%" PRIu64
      " pair_replication_attempts=%" PRIu64 " pair_replication_edits=%" PRIu64
      " pair_replication_rejections=%" PRIu64
      " spill_materialization_batches=%" PRIu64
      " address_state_materializations=%" PRIu64
      " address_state_changes=%" PRIu64 " frame_arena_used_peak=%" PRIu64
      " frame_arena_owned_peak=%" PRIu64 " repair_arena_used_peak=%" PRIu64
      " repair_arena_owned_peak=%" PRIu64 " scratch_arena_used_peak=%" PRIu64
      " scratch_arena_owned_peak=%" PRIu64,
      statistics->frame_build_count, statistics->allocation_run_count,
      repair->iteration_count, repair->diagnostic_replay_count,
      repair->spill_traffic_lowering_count,
      repair->rematerialized_operand_count,
      repair->live_range_split_operand_count,
      repair->pair_replication_attempt_count,
      repair->pair_replication_edit_count,
      repair->pair_replication_rejection_count,
      repair->spill_materialization_batch_count,
      repair->address_state_materialization_count,
      repair->address_state_change_count,
      memory->frame_arena.used_bytes_high_water,
      memory->frame_arena.owned_bytes_high_water,
      memory->repair_arena.used_bytes_high_water,
      memory->repair_arena.owned_bytes_high_water,
      memory->scratch_arena.used_bytes_high_water,
      memory->scratch_arena.owned_bytes_high_water));
  if (iree_any_bit_set(statistics->flags,
                       LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " block_system_allocations=%" PRIu64
        " block_system_allocation_bytes=%" PRIu64
        " oversized_allocations=%" PRIu64
        " oversized_allocation_bytes=%" PRIu64,
        memory->block_system_allocation_count,
        memory->block_system_allocation_bytes,
        memory->oversized_allocation_count,
        memory->oversized_allocation_bytes));
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_low_planning_json(
    const loom_low_planning_statistics_t* statistics,
    loom_output_stream_t* stream) {
  const loom_low_planning_repair_statistics_t* repair = &statistics->repair;
  const loom_low_planning_memory_statistics_t* memory = &statistics->memory;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(
          stream,
          "{\"frame_build_count\":%" PRIu64 ",\"allocation_run_count\":%" PRIu64
          ",\"repair\":{\"iteration_count\":%" PRIu64
          ",\"diagnostic_replay_count\":%" PRIu64
          ",\"spill_traffic_lowering_count\":%" PRIu64
          ",\"rematerialized_operand_count\":%" PRIu64
          ",\"live_range_split_operand_count\":%" PRIu64
          ",\"pair_replication_attempt_count\":%" PRIu64
          ",\"pair_replication_edit_count\":%" PRIu64
          ",\"pair_replication_rejection_count\":%" PRIu64
          ",\"spill_materialization_batch_count\":%" PRIu64
          ",\"address_state_materialization_count\":%" PRIu64
          ",\"address_state_change_count\":%" PRIu64 "}"
          ",\"memory\":{\"frame_arena\":{\"used_bytes_high_water\":%" PRIu64
          ",\"owned_bytes_high_water\":%" PRIu64 "}"
          ",\"repair_arena\":{\"used_bytes_high_water\":%" PRIu64
          ",\"owned_bytes_high_water\":%" PRIu64 "}"
          ",\"scratch_arena\":{\"used_bytes_high_water\":%" PRIu64
          ",\"owned_bytes_high_water\":%" PRIu64 "}",
          statistics->frame_build_count, statistics->allocation_run_count,
          repair->iteration_count, repair->diagnostic_replay_count,
          repair->spill_traffic_lowering_count,
          repair->rematerialized_operand_count,
          repair->live_range_split_operand_count,
          repair->pair_replication_attempt_count,
          repair->pair_replication_edit_count,
          repair->pair_replication_rejection_count,
          repair->spill_materialization_batch_count,
          repair->address_state_materialization_count,
          repair->address_state_change_count,
          memory->frame_arena.used_bytes_high_water,
          memory->frame_arena.owned_bytes_high_water,
          memory->repair_arena.used_bytes_high_water,
          memory->repair_arena.owned_bytes_high_water,
          memory->scratch_arena.used_bytes_high_water,
          memory->scratch_arena.owned_bytes_high_water));
  if (iree_any_bit_set(statistics->flags,
                       LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"system_allocations\":{\"block_count\":%" PRIu64
        ",\"block_bytes\":%" PRIu64 ",\"oversized_count\":%" PRIu64
        ",\"oversized_bytes\":%" PRIu64 "}",
        memory->block_system_allocation_count,
        memory->block_system_allocation_bytes,
        memory->oversized_allocation_count,
        memory->oversized_allocation_bytes));
  }
  return loom_output_stream_write_cstring(stream, "}}");
}
