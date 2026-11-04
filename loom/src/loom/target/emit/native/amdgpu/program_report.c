// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_report.h"

#include "loom/ops/low/ops.h"

static iree_status_t loom_amdgpu_program_report_add(uint64_t source,
                                                    uint64_t* target) {
  if (!iree_checked_add_u64(*target, source, target)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU retained-program report overflowed");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_report_compose_entry(
    const loom_amdgpu_program_artifact_root_t* root,
    loom_target_compile_report_t* report) {
  if (root == NULL || root->helper_count == 0) {
    return iree_ok_status();
  }

  uint64_t instruction_count = report->emitted_instruction_count;
  uint64_t code_byte_count = report->emitted_code_byte_count;
  uint64_t code_storage_byte_count = report->emitted_code_storage_byte_count;
  loom_target_compile_report_emission_breakdown_t emission_breakdown =
      report->emission_breakdown;
  const loom_symbol_ref_t entry_ref =
      loom_low_func_def_callee(root->function_op);
  for (iree_host_size_t i = 0; i < root->helper_count; ++i) {
    const loom_symbol_id_t helper_symbol_id = root->helper_symbols[i];
    IREE_ASSERT_LT(helper_symbol_id, root->call_table->symbol_count);
    const loom_amdgpu_program_callable_t* callable =
        root->call_table->callables_by_symbol[helper_symbol_id];
    IREE_ASSERT_NE(callable, NULL);
    IREE_ASSERT_NE(callable->report, NULL);
    const loom_target_compile_report_call_edge_t* call_edge =
        loom_target_compile_report_find_call_edge(report, entry_ref,
                                                  callable->function);
    IREE_ASSERT_NE(call_edge, NULL);
    const loom_target_compile_report_function_merge_options_t merge_options = {
        .invocation_count = call_edge->dynamic_call_count,
        .has_exact_invocation_count = iree_all_bits_set(
            call_edge->flags,
            LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT),
    };
    IREE_RETURN_IF_ERROR(loom_target_compile_report_merge_function_evidence(
        report, callable->report, &merge_options));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emitted_instruction_count, &instruction_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emitted_code_byte_count, &code_byte_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emitted_code_storage_byte_count,
        &code_storage_byte_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emission_breakdown.body_instruction_count,
        &emission_breakdown.body_instruction_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emission_breakdown.entry_instruction_count,
        &emission_breakdown.entry_instruction_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emission_breakdown.coissued_instruction_count,
        &emission_breakdown.coissued_instruction_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_report_add(
        callable->report->emission_breakdown.coissued_component_count,
        &emission_breakdown.coissued_component_count));
  }
  loom_target_compile_report_record_emission(
      report, instruction_count, code_byte_count, code_storage_byte_count);
  loom_target_compile_report_record_emission_breakdown(report,
                                                       &emission_breakdown);
  return iree_ok_status();
}
