// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format_planning.h"

#include <inttypes.h>

#include "loom/util/json.h"

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
      " frame_arena_used_peak=%" PRIu64 " frame_arena_owned_peak=%" PRIu64
      " repair_arena_used_peak=%" PRIu64 " repair_arena_owned_peak=%" PRIu64
      " scratch_arena_used_peak=%" PRIu64 " scratch_arena_owned_peak=%" PRIu64,
      statistics->frame_build_count, statistics->allocation_run_count,
      repair->iteration_count, repair->diagnostic_replay_count,
      repair->spill_traffic_lowering_count,
      repair->rematerialized_operand_count,
      repair->live_range_split_operand_count,
      repair->pair_replication_attempt_count,
      repair->pair_replication_edit_count,
      repair->pair_replication_rejection_count,
      repair->spill_materialization_batch_count,
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

static iree_status_t loom_target_compile_report_write_arena_statistics_json(
    iree_string_view_t name,
    const loom_low_planning_arena_statistics_t* statistics,
    loom_json_object_writer_t* memory_object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(memory_object, name));
  loom_json_object_writer_t arena_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(memory_object->stream, &arena_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &arena_object, IREE_SV("used_bytes_high_water"),
      statistics->used_bytes_high_water));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &arena_object, IREE_SV("owned_bytes_high_water"),
      statistics->owned_bytes_high_water));
  return loom_json_object_end(&arena_object);
}

iree_status_t loom_target_compile_report_format_low_planning_json(
    const loom_low_planning_statistics_t* statistics,
    loom_output_stream_t* stream) {
  const loom_low_planning_repair_statistics_t* repair = &statistics->repair;
  const loom_low_planning_memory_statistics_t* memory = &statistics->memory;
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &root, IREE_SV("frame_build_count"), statistics->frame_build_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &root, IREE_SV("allocation_run_count"),
      statistics->allocation_run_count));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("repair")));
  loom_json_object_writer_t repair_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &repair_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("iteration_count"), repair->iteration_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("diagnostic_replay_count"),
      repair->diagnostic_replay_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("spill_traffic_lowering_count"),
      repair->spill_traffic_lowering_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("rematerialized_operand_count"),
      repair->rematerialized_operand_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("live_range_split_operand_count"),
      repair->live_range_split_operand_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("pair_replication_attempt_count"),
      repair->pair_replication_attempt_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("pair_replication_edit_count"),
      repair->pair_replication_edit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("pair_replication_rejection_count"),
      repair->pair_replication_rejection_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &repair_object, IREE_SV("spill_materialization_batch_count"),
      repair->spill_materialization_batch_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&repair_object));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("memory")));
  loom_json_object_writer_t memory_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memory_object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_write_arena_statistics_json(
      IREE_SV("frame_arena"), &memory->frame_arena, &memory_object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_write_arena_statistics_json(
      IREE_SV("repair_arena"), &memory->repair_arena, &memory_object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_write_arena_statistics_json(
      IREE_SV("scratch_arena"), &memory->scratch_arena, &memory_object));
  if (iree_any_bit_set(statistics->flags,
                       LOOM_LOW_PLANNING_STATISTICS_FLAG_SYSTEM_ALLOCATIONS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &memory_object, IREE_SV("system_allocations")));
    loom_json_object_writer_t system_allocations;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &system_allocations));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &system_allocations, IREE_SV("block_count"),
        memory->block_system_allocation_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &system_allocations, IREE_SV("block_bytes"),
        memory->block_system_allocation_bytes));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &system_allocations, IREE_SV("oversized_count"),
        memory->oversized_allocation_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &system_allocations, IREE_SV("oversized_bytes"),
        memory->oversized_allocation_bytes));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&system_allocations));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&memory_object));
  return loom_json_object_end(&root);
}
