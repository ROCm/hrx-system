// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"

#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/reporting/low.h"

iree_status_t loom_aie2p_leaf_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_contribution) {
  *out_contribution = (loom_aie2p_leaf_contribution_t){0};

  loom_low_planning_statistics_t planning_statistics = {0};
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = options->descriptor_registry,
      .function_target_facts = options->function_target_facts,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .schedule_structural_models = loom_aie2p_low_structural_schedule_models(),
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .emitter = options->diagnostic_emitter,
      .statistics =
          options->compile_report != NULL ? &planning_statistics : NULL,
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_NONE,
              .record_materialized_spills = true,
              .emitter = options->diagnostic_emitter,
          },
  };
  loom_low_emission_frame_t frame = {0};
  iree_status_t status = loom_low_emission_frame_build_spill_free(
      module, function_op, &frame_options, &spill_free_options, arena, &frame);
  if (options->compile_report != NULL) {
    loom_target_compile_report_record_low_planning(options->compile_report,
                                                   &planning_statistics);
  }
  if (iree_status_is_ok(status) && options->compile_report != NULL) {
    const loom_target_bundle_t* bundle =
        loom_low_resolved_target_bundle(&frame.target);
    if (bundle != NULL) {
      loom_target_compile_report_record_target_bundle(options->compile_report,
                                                      bundle);
    }
    status = loom_target_compile_report_record_low_emission_frame(
        options->compile_report, &frame);
  }

  loom_aie2p_bundle_plan_t bundle_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_bundle_plan_build(&frame, arena, &bundle_plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_leaf_object_emit(&bundle_plan, arena, out_contribution);
  }
  if (iree_status_is_ok(status) && options->compile_report != NULL) {
    loom_target_compile_report_record_emission(
        options->compile_report, bundle_plan.slot_count,
        bundle_plan.encoded_byte_length, bundle_plan.encoded_byte_length);
  }
  return status;
}
