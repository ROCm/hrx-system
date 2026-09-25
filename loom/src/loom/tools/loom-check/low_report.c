// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/low_report.h"

#include "loom/codegen/low/frame.h"
#include "loom/target/reporting/format.h"
#include "loom/target/reporting/low.h"
#include "loom/tools/loom-check/low_emit.h"

iree_status_t loom_check_emit_low_report(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_test_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_check_result_t* result) {
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      module, symbol_name, test_case, filename, diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }

  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = descriptor_registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .emitter = emitter,
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      module, low_function, &frame_options, arena, &frame, &frame_accepted));
  if (!frame_accepted) {
    return iree_ok_status();
  }

  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_arena_allocator(arena));
  iree_status_t status =
      loom_target_compile_report_record_low_emission_frame(&report, &frame);
  if (iree_status_is_ok(status)) {
    loom_target_compile_report_format_options_t options;
    loom_target_compile_report_format_options_initialize(&options);
    options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    status = loom_target_compile_report_format_text(&report, &options,
                                                    &result->actual_output);
  }
  loom_target_compile_report_deinitialize(&report);
  return status;
}
