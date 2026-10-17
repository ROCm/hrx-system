// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/stage_table.h"

#include <string.h>

iree_status_t loom_serve_stage_record_check(
    const loom_serve_stage_record_options_t* options,
    const loom_serve_jit_stage_t* stage) {
  const loom_cmd_program_t* program = loom_serve_jit_stage_program(stage);
  if (!program) {
    if (options->workspace == LOOM_SERVE_STAGE_WORKSPACE_INVOCATION ||
        loom_serve_dispatch_binding_count(
            loom_serve_jit_stage_dispatch(stage)) != options->binding_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct stage does not implement model bindings");
    }
    return iree_ok_status();
  }
  const bool has_workspace =
      options->workspace == LOOM_SERVE_STAGE_WORKSPACE_INVOCATION ||
      program->requirements.transient.required_byte_length != 0;
  if (program->requirements.rebindable_binding_count !=
          options->binding_count + has_workspace ||
      (has_workspace && program->requirements.transient.binding_index !=
                            options->binding_count) ||
      program->requirements.launch_counts.binding_index != UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command stage does not implement model bindings");
  }
  return iree_ok_status();
}

void loom_serve_stage_table_free(loom_serve_stage_table_t* table) {
  if (!table) {
    return;
  }
  for (iree_host_size_t i = table->owned_part_begin; i < table->part_count;
       ++i) {
    iree_hal_command_buffer_release(table->parts[i].command_buffer);
    loom_serve_dispatch_release(table->parts[i].dispatch);
  }
  iree_allocator_free(table->host_allocator, table->parts);
  iree_allocator_free(table->host_allocator, table->launches);
  iree_allocator_free(table->host_allocator, table);
}

iree_host_size_t loom_serve_stage_table_storage_size(
    const loom_serve_stage_table_t* table) {
  return sizeof(*table) +
         (table->launches ? table->stage_count * sizeof(*table->launches) : 0) +
         table->part_capacity * sizeof(*table->parts);
}

// Parts grow with actual barrier cuts, not with the root's dispatch count.
// Every published slot owns its reference immediately, including on failure.
static iree_status_t loom_serve_stage_table_append(
    loom_serve_stage_table_t* table,
    const loom_serve_stage_record_options_t* options,
    const loom_serve_jit_stage_t* stage,
    const loom_cmd_program_command_range_t* range) {
  const iree_host_size_t maximum =
      iree_min(INT32_MAX, IREE_HOST_SIZE_MAX / sizeof(*table->parts));
  if (table->part_count == maximum) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parts exceed VM stage capacity");
  }
  if (table->part_count == table->part_capacity) {
    const iree_host_size_t capacity =
        iree_min(iree_max(table->part_capacity * 2, 1), maximum);
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        table->host_allocator, capacity * sizeof(*table->parts),
        (void**)&table->parts));
    table->part_capacity = capacity;
  }
  loom_serve_stage_t* part = &table->parts[table->part_count];
  *part = (loom_serve_stage_t){.binding_count = options->binding_count};
  const loom_cmd_program_t* program = loom_serve_jit_stage_program(stage);
  if (program) {
    IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
        stage, options->queue_family, options->command_mode, range,
        options->parameters, &part->command_buffer));
    if (options->workspace == LOOM_SERVE_STAGE_WORKSPACE_INVOCATION) {
      ++part->binding_count;
    } else {
      part->transient = program->requirements.transient;
    }
  } else {
    part->dispatch = loom_serve_jit_stage_dispatch(stage);
    loom_serve_dispatch_retain(part->dispatch);
  }
  ++table->part_count;
  return iree_ok_status();
}

static iree_status_t loom_serve_stage_table_record(
    loom_serve_stage_table_t* table,
    const loom_serve_stage_record_options_t* options,
    const loom_serve_jit_stage_t* stage) {
  if (!options->command_quantum) {
    return loom_serve_stage_table_append(table, options, stage, NULL);
  }
  const loom_cmd_program_t* program = loom_serve_jit_stage_program(stage);
  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(program, &iterator);
  loom_cmd_program_barrier_wave_t wave;
  loom_cmd_program_command_range_t range = {0};
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave)) {
    if (range.command_count &&
        (uint64_t)range.command_count + wave.commands.command_count >
            options->command_quantum) {
      status = loom_serve_stage_table_append(table, options, stage, &range);
      range.first_command = wave.commands.first_command;
      range.command_count = 0;
    }
    range.command_count += wave.commands.command_count;
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_table_append(table, options, stage, &range);
  }
  return status;
}

iree_status_t loom_serve_stage_table_allocate(
    const loom_serve_stage_record_options_t* options,
    const loom_serve_stage_table_t* prefix, iree_host_size_t stage_count,
    loom_serve_jit_stage_t* const* stages, iree_allocator_t host_allocator,
    loom_serve_stage_table_t** out_table) {
  *out_table = NULL;
  const iree_host_size_t prefix_count = prefix ? prefix->stage_count : 0;
  loom_serve_stage_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*table), (void**)&table));
  table->host_allocator = host_allocator;
  table->stage_count = prefix_count + stage_count;
  iree_status_t status = iree_ok_status();
  if (options->workspace == LOOM_SERVE_STAGE_WORKSPACE_INVOCATION) {
    status = iree_allocator_malloc_array(host_allocator, table->stage_count,
                                         sizeof(*table->launches),
                                         (void**)&table->launches);
  }
  if (iree_status_is_ok(status)) {
    table->part_capacity = (prefix ? prefix->part_count : 0) + stage_count;
    status = iree_allocator_malloc_array(host_allocator, table->part_capacity,
                                         sizeof(*table->parts),
                                         (void**)&table->parts);
    if (iree_status_is_ok(status) && prefix) {
      if (table->launches) {
        memcpy(table->launches, prefix->launches,
               prefix_count * sizeof(*table->launches));
      }
      memcpy(table->parts, prefix->parts,
             prefix->part_count * sizeof(*table->parts));
      table->part_count = prefix->part_count;
      table->owned_part_begin = prefix->part_count;
    }
  }
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    const uint32_t first_part = (uint32_t)table->part_count;
    status = loom_serve_stage_table_record(table, options, stages[i]);
    if (table->launches) {
      table->launches[prefix_count + i] = (loom_serve_stage_launch_t){
          .first_part = first_part,
          .part_count = (uint32_t)table->part_count - first_part,
          .transient =
              loom_serve_jit_stage_program(stages[i])->requirements.transient};
    }
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  } else {
    loom_serve_stage_table_free(table);
  }
  return status;
}
