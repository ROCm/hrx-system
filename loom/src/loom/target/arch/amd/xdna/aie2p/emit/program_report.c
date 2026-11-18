// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_report.h"

iree_status_t loom_aie2p_program_report_worker(
    loom_aie2p_program_report_t* report, iree_arena_allocator_t* arena,
    uint32_t worker_index,
    loom_target_compile_report_pipeline_placement_t placement,
    const loom_aie2p_program_object_t* program,
    const loom_aie2p_program_layout_t* layout,
    const loom_aie2p_linked_tile_t* linked) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, report->function_count,
      report->function_count + program->function_count,
      sizeof(*report->functions), &report->function_capacity,
      (void**)&report->functions));
  for (iree_host_size_t i = 0; i < program->function_count; ++i) {
    const loom_aie2p_program_function_t* function = &program->functions[i];
    const loom_aie2p_function_issue_t* issue = function->contribution->issue;
    const loom_native_object_symbol_layout_t* symbol =
        &linked->symbol_layouts[function->entry_symbol_index];
    loom_target_compile_report_placed_function_t* row =
        &report->functions[report->function_count++];
    *row = (loom_target_compile_report_placed_function_t){
        .name = issue->name,
        .worker_index = worker_index,
        .placement = placement,
        .group = layout ? (uint32_t)layout->function_groups[i] : 0,
        .address = (uint32_t)(linked->assembly.sections[symbol->section_index]
                                  .address +
                              symbol->section_offset),
        .code_byte_count =
            (uint32_t)function->contribution->realization.code.byte_length,
        .stored_packet_count = issue->stored_packet_count,
        .packets_per_call = issue->stored_packet_count,
        .has_packets_per_call = !issue->has_unaccounted_control,
        .calls_per_entry =
            i ? program->helper_invocations[i - 1].invocation_count : 1,
        .has_calls_per_entry =
            !i || program->helper_invocations[i - 1].has_exact_invocation_count,
        .first_loop = report->loop_count,
        .loop_count = (uint32_t)issue->loop_count,
    };
    if (issue->loop_count) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, report->loop_count, report->loop_count + issue->loop_count,
          sizeof(*report->loops), &report->loop_capacity,
          (void**)&report->loops));
      for (iree_host_size_t j = 0; j < issue->loop_count; ++j) {
        const loom_aie2p_issue_loop_t loop = issue->loops[j];
        report->loops[report->loop_count++] =
            (loom_target_compile_report_packet_loop_t){
                .byte_offset = loop.byte_offset,
                .packet_count = loop.packet_count,
                .trip_count = loop.trip_count,
            };
        row->packets_per_call +=
            (uint64_t)loop.packet_count * (loop.trip_count - 1u);
      }
    }
  }
  return iree_ok_status();
}
