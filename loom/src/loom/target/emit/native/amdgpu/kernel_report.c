// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_report.h"

#include "loom/target/emit/native/amdgpu/function_report.h"

iree_status_t loom_amdgpu_kernel_report_record_emission(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context,
    const loom_amdgpu_kernel_plan_t* plan,
    const loom_amdgpu_kernel_emission_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_function_report_record_emission(
      report, frame, dynamic_context, &plan->function, &result->body));

  uint64_t instruction_count = 0;
  if (!iree_checked_add_u64(result->body.instruction_count,
                            result->entry_instruction_count,
                            &instruction_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel instruction count overflowed");
  }
  const uint64_t text_length = result->kernel.text.data_length;
  loom_target_compile_report_record_emission(report, instruction_count,
                                             text_length, text_length);
  loom_target_compile_report_emission_breakdown_t emission_breakdown =
      report->emission_breakdown;
  emission_breakdown.entry_instruction_count = result->entry_instruction_count;
  loom_target_compile_report_record_emission_breakdown(report,
                                                       &emission_breakdown);
  loom_target_compile_report_record_memory(
      report, plan->record.metadata.private_segment_fixed_size,
      plan->record.metadata.group_segment_fixed_size);
  return iree_ok_status();
}
