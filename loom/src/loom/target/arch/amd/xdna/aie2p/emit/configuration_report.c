// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/configuration_report.h"

#include <stdlib.h>
#include <string.h>

typedef struct loom_aie2p_configuration_report_memory_t {
  // Physical usage accumulated exactly once at reservation and link sites.
  loom_target_compile_report_pipeline_memory_row_t row;
  // Occupied bytes per bank, allocated lazily for nonempty stores.
  uint32_t* bank_bytes;
} loom_aie2p_configuration_report_memory_t;

struct loom_aie2p_configuration_report_t {
  // Destination owns copied rows; names remain borrowed from its module.
  loom_target_compile_report_t* destination;
  // Device facts determining capacities and bank boundaries.
  const loom_xdna_array_family_t* family;
  // Optional report-only storage, retired with configuration emission.
  iree_arena_allocator_t* arena;
  // Aggregate physical inventory, independent of source channel provenance.
  loom_target_compile_report_pipeline_plan_summary_t summary;
  // Dense physical tile table indexed by column * row_count + row.
  loom_aie2p_configuration_report_memory_t* memories;
  // Complete partition tile count, including unused stores.
  iree_host_size_t memory_count;
  // One row per program load, in configuration order.
  loom_target_compile_report_pipeline_worker_row_t* workers;
  // Allocated worker row capacity.
  iree_host_size_t worker_capacity;
};

iree_status_t loom_aie2p_configuration_report_allocate(
    loom_target_compile_report_t* destination,
    const loom_xdna_array_family_t* family, uint32_t column_count,
    iree_string_view_t root_name, iree_arena_allocator_t* arena,
    loom_aie2p_configuration_report_t** out_report) {
  loom_aie2p_configuration_report_t* report = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*report), (void**)&report));
  *report = (loom_aie2p_configuration_report_t){
      .destination = destination,
      .family = family,
      .arena = arena,
      .summary =
          {
              .root_name = root_name,
              .realization = IREE_SV("resident-configuration"),
              .available_facts =
                  LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_PROGRAMS |
                  LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_MEMORY,
          },
      .memory_count = (iree_host_size_t)column_count * family->row_count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, report->memory_count,
                                                 sizeof(*report->memories),
                                                 (void**)&report->memories));
  memset(report->memories, 0, report->memory_count * sizeof(*report->memories));
  *out_report = report;
  return iree_ok_status();
}

void loom_aie2p_configuration_report_program(
    loom_aie2p_configuration_report_t* report, uint32_t code_byte_count) {
  ++report->summary.program_count;
  report->summary.program_code_byte_count += code_byte_count;
}

static iree_status_t loom_aie2p_configuration_report_memory(
    loom_aie2p_configuration_report_t* report,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_configuration_report_memory_t** out_memory) {
  loom_aie2p_configuration_report_memory_t* memory =
      &report->memories[coordinate.column * report->family->row_count +
                        coordinate.row];
  if (!memory->bank_bytes) {
    const loom_xdna_tile_facts_t* tile =
        loom_xdna_array_tile_facts(report->family, coordinate);
    memory->row.placement = (loom_target_compile_report_pipeline_placement_t){
        .rank = 2, .x = coordinate.column, .y = coordinate.row};
    memory->row.capacity_byte_count = tile->memory.local_capacity;
    memory->row.bank_storage_capacity_byte_count =
        tile->memory.local_capacity / tile->memory.bank_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        report->arena, tile->memory.bank_count, sizeof(*memory->bank_bytes),
        (void**)&memory->bank_bytes));
    memset(memory->bank_bytes, 0,
           tile->memory.bank_count * sizeof(*memory->bank_bytes));
  }
  *out_memory = memory;
  return iree_ok_status();
}

// The caller has already formed disjoint occupied ranges. Accounting walks
// only the banks touched by this range and never revisits other allocations.
static void loom_aie2p_configuration_report_range(
    loom_aie2p_configuration_report_memory_t* memory, uint32_t offset,
    uint32_t length) {
  if (!length) {
    return;
  }
  const uint32_t end = offset + length;
  memory->row.occupied_byte_count += length;
  memory->row.high_water_byte_count =
      iree_max(memory->row.high_water_byte_count, end);
  const uint32_t bank_length = memory->row.bank_storage_capacity_byte_count;
  for (uint32_t position = offset; position < end;) {
    const uint32_t bank = position / bank_length;
    const uint32_t bank_end = iree_min(end, (bank + 1) * bank_length);
    memory->bank_bytes[bank] += bank_end - position;
    memory->row.maximum_bank_storage_byte_count = iree_max(
        memory->row.maximum_bank_storage_byte_count, memory->bank_bytes[bank]);
    position = bank_end;
  }
}

static int loom_aie2p_configuration_report_range_compare(const void* lhs,
                                                         const void* rhs) {
  const uint64_t a =
      ((const loom_source_storage_packing_range_t*)lhs)->byte_offset;
  const uint64_t b =
      ((const loom_source_storage_packing_range_t*)rhs)->byte_offset;
  return (a > b) - (a < b);
}

iree_status_t loom_aie2p_configuration_report_reservations(
    loom_aie2p_configuration_report_t* report,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_source_storage_packing_range_t* ranges,
    iree_host_size_t range_count) {
  if (!range_count) {
    return iree_ok_status();
  }
  loom_source_storage_packing_range_t* sorted = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      report->arena, range_count, sizeof(*sorted), (void**)&sorted));
  memcpy(sorted, ranges, range_count * sizeof(*sorted));
  qsort(sorted, range_count, sizeof(*sorted),
        loom_aie2p_configuration_report_range_compare);
  loom_aie2p_configuration_report_memory_t* memory = NULL;
  uint32_t end = 0;
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    if (!sorted[i].byte_length) {
      continue;
    }
    if (!memory) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_configuration_report_memory(report, coordinate, &memory));
    }
    const uint32_t begin = iree_max(end, sorted[i].byte_offset);
    end = iree_max(end, sorted[i].byte_offset + sorted[i].byte_length);
    const uint32_t length = end - begin;
    memory->row.reserved_byte_count += length;
    loom_aie2p_configuration_report_range(memory, begin, length);
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_configuration_report_worker(
    loom_aie2p_configuration_report_t* report, iree_string_view_t entry_name,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_aie2p_leaf_realization_t* realization,
    const loom_aie2p_tile_link_layout_t* layout) {
  loom_target_compile_report_pipeline_plan_summary_t* summary =
      &report->summary;
  if (summary->worker_count == report->worker_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        report->arena, summary->worker_count, summary->worker_count + 1,
        sizeof(*report->workers), &report->worker_capacity,
        (void**)&report->workers));
  }
  loom_aie2p_configuration_report_memory_t* memory = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_configuration_report_memory(report, coordinate, &memory));
  uint32_t data_bytes = 0;
  for (iree_host_size_t i = 0; i < layout->storage_placement_count; ++i) {
    const loom_aie2p_tile_storage_placement_t* storage =
        &layout->storage_placements[i];
    const uint32_t length =
        loom_aie2p_leaf_storage_requirement(realization, storage->storage_space)
            ->byte_length;
    data_bytes += length;
    loom_aie2p_configuration_report_range(memory, storage->owner_offset,
                                          length);
  }
  for (iree_host_size_t i = 0; i < layout->read_only_data_placement_count;
       ++i) {
    const loom_aie2p_tile_read_only_data_placement_t* data =
        &layout->read_only_data_placements[i];
    data_bytes += data->byte_length;
    loom_aie2p_configuration_report_range(memory, data->owner_offset,
                                          data->byte_length);
  }
  memory->row.program_data_byte_count += data_bytes;
  const uint32_t code_bytes = realization->code.byte_length;
  const uint32_t code_headroom = layout->program_byte_capacity - code_bytes;
  summary->minimum_worker_code_headroom_byte_count =
      summary->worker_count
          ? iree_min(summary->minimum_worker_code_headroom_byte_count,
                     code_headroom)
          : code_headroom;
  summary->maximum_worker_code_byte_count =
      iree_max(summary->maximum_worker_code_byte_count, code_bytes);
  summary->worker_code_byte_count += code_bytes;
  summary->worker_storage_byte_count += data_bytes;
  report->workers[summary->worker_count] =
      (loom_target_compile_report_pipeline_worker_row_t){
          .worker_index = summary->worker_count,
          .flags = LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_PLACED,
          .entry_name = entry_name,
          .placement = memory->row.placement,
          .code_byte_count = code_bytes,
          .code_capacity_byte_count = layout->program_byte_capacity,
          .worker_storage_byte_count = data_bytes,
      };
  ++summary->worker_count;
  return iree_ok_status();
}

iree_status_t loom_aie2p_configuration_report_finish(
    loom_aie2p_configuration_report_t* report, uint32_t binding_count) {
  loom_target_compile_report_pipeline_plan_summary_t* summary =
      &report->summary;
  summary->binding_count = binding_count;
  loom_target_compile_report_pipeline_memory_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      report->arena, report->memory_count, sizeof(*rows), (void**)&rows));
  for (iree_host_size_t i = 0; i < report->memory_count; ++i) {
    loom_target_compile_report_pipeline_memory_row_t row =
        report->memories[i].row;
    if (!report->memories[i].bank_bytes) {
      continue;
    }
    row.memory_index = summary->memory_count;
    const uint32_t headroom =
        row.capacity_byte_count - row.high_water_byte_count;
    summary->minimum_tile_local_memory_headroom_byte_count =
        summary->memory_count
            ? iree_min(summary->minimum_tile_local_memory_headroom_byte_count,
                       headroom)
            : headroom;
    rows[summary->memory_count++] = row;
    summary->reserved_storage_byte_count += row.reserved_byte_count;
    summary->maximum_tile_local_memory_byte_count =
        iree_max(summary->maximum_tile_local_memory_byte_count,
                 row.high_water_byte_count);
    if (row.maximum_bank_storage_byte_count >=
        summary->maximum_bank_storage_byte_count) {
      summary->maximum_bank_storage_byte_count =
          row.maximum_bank_storage_byte_count;
      summary->bank_storage_capacity_byte_count =
          row.bank_storage_capacity_byte_count;
    }
  }
  for (uint32_t i = 0; i < summary->worker_count; ++i) {
    loom_target_compile_report_pipeline_worker_row_t* worker =
        &report->workers[i];
    const loom_target_compile_report_pipeline_memory_row_t* memory =
        &report
             ->memories[worker->placement.x * report->family->row_count +
                        worker->placement.y]
             .row;
    worker->local_memory_byte_count = memory->high_water_byte_count;
    worker->local_memory_capacity_byte_count = memory->capacity_byte_count;
    worker->maximum_bank_storage_byte_count =
        memory->maximum_bank_storage_byte_count;
    worker->bank_storage_capacity_byte_count =
        memory->bank_storage_capacity_byte_count;
  }
  const loom_target_compile_report_pipeline_plan_t plan = {
      .summary = *summary,
      .worker_rows = report->workers,
      .worker_row_count = summary->worker_count,
      .memory_rows = rows,
      .memory_row_count = summary->memory_count,
  };
  return loom_target_compile_report_record_pipeline_plan(report->destination,
                                                         &plan);
}
