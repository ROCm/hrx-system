// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pipeline.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pass/ops.h"
#include "loom/sanitizer/options.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/walk.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct PipelineRunCounts {
  // Number of final template-selection pass runs.
  int final_template_selection = 0;
  // Lexical pass-run ordinal of final template selection.
  int final_template_selection_ordinal = 0;
  // Rewrite mode completing the selected providers before source lowering.
  iree_string_view_t final_template_rewrite = iree_string_view_empty();
  // Number of retained call-graph specialization pass runs.
  int target_callgraph_specialization = 0;
  // Lexical pass-run ordinal of retained call-graph specialization.
  int target_callgraph_specialization_ordinal = 0;
  // Lexical pass-run ordinal of the first target-required call inlining.
  int first_target_inlining_ordinal = 0;
  // Lexical pass-run ordinal of the last target-required call inlining.
  int last_target_inlining_ordinal = 0;
  // Last source combination before target legalization.
  int last_source_combination_ordinal = 0;
  // First target legalization pass that selects physical representations.
  int first_target_legalization_ordinal = 0;
  // Number of sanitizer-driven vector-memory scalarization pass runs.
  int vector_memory_to_scalar = 0;
  // Lexical pass-run ordinal of vector-memory scalarization.
  int vector_memory_to_scalar_ordinal = 0;
  // Lexical pass position of pipeline-only source preparation.
  int pipeline_preparation_ordinal = 0;
  // Lexical pass position of worker outlining.
  int worker_outlining_ordinal = 0;
  // Number of source-loop unrolling pass runs.
  int source_loop_unrolling = 0;
  // Lexical pass-run ordinal of source-loop unrolling.
  int source_loop_unrolling_ordinal = 0;
  // Number of vector-bank scalar replacement pass runs.
  int vector_bank_sroa = 0;
  // Lexical pass-run ordinal of vector-bank scalar replacement.
  int vector_bank_sroa_ordinal = 0;
  // Number of single-use read sinking pass runs.
  int sink_single_use_reads = 0;
  // Lexical pass-run ordinal of single-use read sinking.
  int sink_single_use_reads_ordinal = 0;
  // Number of structured-to-CFG lowering pass runs.
  int scf_to_cfg = 0;
  // Lexical pass-run ordinal of the first structured-to-CFG lowering.
  int first_scf_to_cfg_ordinal = 0;
  // Number of composed boundary projection pass runs.
  int boundary_projection = 0;
  // Lexical pass-run ordinal of composed boundary projection.
  int boundary_projection_ordinal = 0;
  // Number of source-to-low pass runs.
  int source_to_low = 0;
  // Lexical pass-run ordinal of source-to-low.
  int source_to_low_ordinal = 0;
  // Number of symbol-DCE runs after source-to-Low lowering.
  int symbol_dce = 0;
  // Lexical pass-run ordinal of symbol DCE.
  int symbol_dce_ordinal = 0;
  // Diagnostics option on the source-to-low pass.
  iree_string_view_t source_to_low_diagnostics = iree_string_view_empty();
  // Sanitizer reporting option on the source-to-low pass.
  iree_string_view_t source_to_low_sanitizer_reporting =
      iree_string_view_empty();
  // Number of source assertion-insertion pass runs.
  int sanitizer_insert_assertions = 0;
  // Lexical pass-run ordinal of source assertion insertion.
  int sanitizer_insert_assertions_ordinal = 0;
  // Checks option on the source assertion-insertion pass.
  iree_string_view_t sanitizer_insert_checks = iree_string_view_empty();
  // Number of source race-observation pass runs.
  int sanitizer_insert_race_observations = 0;
  // Checks option on the source race-observation pass.
  iree_string_view_t sanitizer_insert_race_checks = iree_string_view_empty();
  // Number of semantic assertion-materialization pass runs.
  int sanitizer_materialize_assertions = 0;
  // Lexical pass-run ordinal of semantic assertion materialization.
  int sanitizer_materialize_assertions_ordinal = 0;
  // Number of other sanitizer pass runs.
  int other_sanitizer_runs = 0;
} PipelineRunCounts;

typedef struct PipelineRunCountContext {
  // Module owning the pass IR being inspected.
  loom_module_t* module;
  // Counts updated while walking pass.run operations.
  PipelineRunCounts counts;
  // One-based lexical ordinal of the current pass.run operation.
  int current_run_ordinal;
} PipelineRunCountContext;

iree_string_view_t FindStringOption(loom_module_t* module,
                                    loom_named_attr_slice_t options,
                                    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < options.count; ++i) {
    const loom_named_attr_t* option = &options.entries[i];
    iree_string_view_t option_name =
        loom_string_table_get(&module->strings, option->name_id);
    if (!iree_string_view_equal(option_name, name)) {
      continue;
    }
    if (option->value.kind != LOOM_ATTR_STRING) {
      return iree_string_view_empty();
    }
    return loom_string_table_get(&module->strings,
                                 loom_attr_as_string_id(option->value));
  }
  return iree_string_view_empty();
}

iree_status_t InspectPipelineRun(void* user_data, loom_op_t* op,
                                 const loom_walk_context_t* context,
                                 loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  PipelineRunCountContext* count_context =
      static_cast<PipelineRunCountContext*>(user_data);
  if (loom_pass_where_isa(op) &&
      iree_string_view_equal(
          loom_string_table_get(&count_context->module->strings,
                                loom_pass_where_predicate(op)),
          IREE_SV("op")) &&
      iree_string_view_equal(
          FindStringOption(count_context->module, loom_pass_where_attrs(op),
                           IREE_SV("name")),
          IREE_SV("pipeline.def"))) {
    count_context->counts.pipeline_preparation_ordinal =
        ++count_context->current_run_ordinal;
    *out_result = LOOM_WALK_SKIP;
    return iree_ok_status();
  }
  if (!loom_pass_run_isa(op)) {
    return iree_ok_status();
  }
  ++count_context->current_run_ordinal;
  PipelineRunCounts* counts = &count_context->counts;
  iree_string_view_t key = loom_string_table_get(
      &count_context->module->strings, loom_pass_run_key(op));
  if (iree_string_view_equal(key, IREE_SV("outline-pipeline-strands"))) {
    counts->worker_outlining_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("select-templates")) &&
             iree_string_view_equal(
                 FindStringOption(count_context->module,
                                  loom_pass_run_options(op), IREE_SV("mode")),
                 IREE_SV("final"))) {
    ++counts->final_template_selection;
    counts->final_template_selection_ordinal =
        count_context->current_run_ordinal;
    counts->final_template_rewrite = FindStringOption(
        count_context->module, loom_pass_run_options(op), IREE_SV("rewrite"));
  } else if (iree_string_view_equal(key,
                                    IREE_SV("specialize-target-callgraph"))) {
    ++counts->target_callgraph_specialization;
    counts->target_callgraph_specialization_ordinal =
        count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("inline-callables")) &&
             iree_string_view_equal(
                 FindStringOption(count_context->module,
                                  loom_pass_run_options(op), IREE_SV("policy")),
                 IREE_SV("target"))) {
    if (counts->first_target_inlining_ordinal == 0) {
      counts->first_target_inlining_ordinal =
          count_context->current_run_ordinal;
    }
    counts->last_target_inlining_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("combine"))) {
    counts->last_source_combination_ordinal =
        count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("target-legalize"))) {
    if (counts->first_target_legalization_ordinal == 0) {
      counts->first_target_legalization_ordinal =
          count_context->current_run_ordinal;
    }
  } else if (iree_string_view_equal(key, IREE_SV("vector-memory-to-scalar"))) {
    ++counts->vector_memory_to_scalar;
    counts->vector_memory_to_scalar_ordinal =
        count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("unroll-scf-for"))) {
    ++counts->source_loop_unrolling;
    counts->source_loop_unrolling_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("sroa-vector-banks"))) {
    ++counts->vector_bank_sroa;
    counts->vector_bank_sroa_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("sink-single-use-reads"))) {
    ++counts->sink_single_use_reads;
    counts->sink_single_use_reads_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("scf-to-cfg"))) {
    ++counts->scf_to_cfg;
    if (counts->first_scf_to_cfg_ordinal == 0) {
      counts->first_scf_to_cfg_ordinal = count_context->current_run_ordinal;
    }
  } else if (iree_string_view_equal(
                 key, IREE_SV("project-boundary-representations"))) {
    ++counts->boundary_projection;
    counts->boundary_projection_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key, IREE_SV("source-to-low"))) {
    ++counts->source_to_low;
    counts->source_to_low_ordinal = count_context->current_run_ordinal;
    counts->source_to_low_diagnostics =
        FindStringOption(count_context->module, loom_pass_run_options(op),
                         IREE_SV("diagnostics"));
    counts->source_to_low_sanitizer_reporting =
        FindStringOption(count_context->module, loom_pass_run_options(op),
                         IREE_SV("sanitizer-reporting"));
  } else if (iree_string_view_equal(key, IREE_SV("symbol-dce")) &&
             counts->source_to_low != 0) {
    ++counts->symbol_dce;
    counts->symbol_dce_ordinal = count_context->current_run_ordinal;
  } else if (iree_string_view_equal(key,
                                    IREE_SV("sanitizer-insert-assertions"))) {
    ++counts->sanitizer_insert_assertions;
    counts->sanitizer_insert_assertions_ordinal =
        count_context->current_run_ordinal;
    counts->sanitizer_insert_checks = FindStringOption(
        count_context->module, loom_pass_run_options(op), IREE_SV("checks"));
  } else if (iree_string_view_equal(
                 key, IREE_SV("sanitizer-insert-race-observations"))) {
    ++counts->sanitizer_insert_race_observations;
    counts->sanitizer_insert_race_checks = FindStringOption(
        count_context->module, loom_pass_run_options(op), IREE_SV("checks"));
  } else if (iree_string_view_equal(
                 key, IREE_SV("sanitizer-materialize-assertions"))) {
    ++counts->sanitizer_materialize_assertions;
    counts->sanitizer_materialize_assertions_ordinal =
        count_context->current_run_ordinal;
  } else if (iree_string_view_starts_with(key, IREE_SV("sanitizer-"))) {
    ++counts->other_sanitizer_runs;
  }
  return iree_ok_status();
}

class TargetPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    provider_set_ = loom_target_provider_set_make(nullptr, 0);
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&provider_set_, &environment_));
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr AllocateModule(iree_string_view_t name) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_module_allocate(&context_, name, &block_pool_, nullptr,
                                        iree_allocator_system(), &module));
    return ModulePtr(module);
  }

  PipelineRunCounts CountPipelineRuns(loom_module_t* module,
                                      loom_op_t* pipeline_op) {
    PipelineRunCounts counts = {};
    PipelineRunCountContext count_context = {
        .module = module,
    };
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_EXPECT_OK(loom_walk_region(
        module, loom_pass_pipeline_body(pipeline_op), LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){InspectPipelineRun, &count_context},
        &walk_result));
    EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
    counts = count_context.counts;
    return counts;
  }

  // Block pool backing test modules.
  iree_arena_block_pool_t block_pool_;
  // IR context with all dialects registered.
  loom_context_t context_;
  // Provider set borrowed by the target environment for its lifetime.
  loom_target_provider_set_t provider_set_;
  // Target environment used by pipeline construction.
  loom_target_environment_t environment_;
};

TEST_F(TargetPipelineTest, ZeroChecksStillMaterializesAuthoredAssertions) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {0};

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.final_template_selection, 1);
  EXPECT_TRUE(
      iree_string_view_equal(counts.final_template_rewrite, IREE_SV("inline")));
  EXPECT_EQ(counts.target_callgraph_specialization, 1);
  EXPECT_EQ(counts.boundary_projection, 1);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_EQ(counts.symbol_dce, 1);
  EXPECT_LT(counts.final_template_selection_ordinal,
            counts.target_callgraph_specialization_ordinal);
  EXPECT_LT(counts.target_callgraph_specialization_ordinal,
            counts.first_target_inlining_ordinal);
  EXPECT_LT(counts.first_target_inlining_ordinal,
            counts.last_source_combination_ordinal);
  EXPECT_LT(counts.last_source_combination_ordinal,
            counts.first_target_legalization_ordinal);
  EXPECT_LT(counts.first_target_inlining_ordinal,
            counts.pipeline_preparation_ordinal);
  EXPECT_LT(counts.pipeline_preparation_ordinal,
            counts.worker_outlining_ordinal);
  EXPECT_LT(counts.worker_outlining_ordinal,
            counts.first_target_legalization_ordinal);
  EXPECT_EQ(counts.source_loop_unrolling, 1);
  EXPECT_EQ(counts.vector_bank_sroa, 1);
  EXPECT_EQ(counts.sink_single_use_reads, 1);
  EXPECT_GT(counts.scf_to_cfg, 0);
  EXPECT_LT(counts.source_loop_unrolling_ordinal,
            counts.vector_bank_sroa_ordinal);
  EXPECT_LT(counts.vector_bank_sroa_ordinal,
            counts.sink_single_use_reads_ordinal);
  EXPECT_LT(counts.sink_single_use_reads_ordinal,
            counts.first_scf_to_cfg_ordinal);
  EXPECT_LT(counts.first_target_legalization_ordinal,
            counts.boundary_projection_ordinal);
  EXPECT_LT(counts.boundary_projection_ordinal, counts.source_to_low_ordinal);
  EXPECT_LT(counts.source_to_low_ordinal, counts.last_target_inlining_ordinal);
  EXPECT_LT(counts.source_to_low_ordinal, counts.symbol_dce_ordinal);
  EXPECT_TRUE(iree_string_view_is_empty(counts.source_to_low_diagnostics));
  EXPECT_TRUE(
      iree_string_view_is_empty(counts.source_to_low_sanitizer_reporting));
  EXPECT_EQ(counts.vector_memory_to_scalar, 0);
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_LT(counts.first_scf_to_cfg_ordinal,
            counts.sanitizer_materialize_assertions_ordinal);
  EXPECT_LT(counts.sanitizer_materialize_assertions_ordinal,
            counts.boundary_projection_ordinal);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, DiagnosticArtifactsPreserveRawSourceBoundary) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_source_low_diagnostic_artifacts(
      module.get(), IREE_SV("compile"), /*options=*/nullptr, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.final_template_selection, 0);
  EXPECT_EQ(counts.target_callgraph_specialization, 1);
  EXPECT_EQ(counts.boundary_projection, 1);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_EQ(counts.symbol_dce, 0);
  EXPECT_LT(counts.target_callgraph_specialization_ordinal,
            counts.first_target_inlining_ordinal);
  EXPECT_LT(counts.first_target_inlining_ordinal,
            counts.boundary_projection_ordinal);
  EXPECT_LT(counts.boundary_projection_ordinal, counts.source_to_low_ordinal);
  EXPECT_LT(counts.source_to_low_ordinal, counts.last_target_inlining_ordinal);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
}

TEST_F(TargetPipelineTest, OperandFormDiagnosticsBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags =
          LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM,
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.source_to_low_diagnostics,
                                     IREE_SV("operand-forms")));
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, TrapReportingBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = 0,
              .flags = 0,
              .reporting_mode = LOOM_SANITIZER_REPORTING_MODE_TRAP,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.source_to_low_sanitizer_reporting,
                                     IREE_SV("trap")));
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, ReportOnlyBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = 0,
              .flags = 0,
              .reporting_mode = LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.source_to_low_sanitizer_reporting,
                                     IREE_SV("report-only")));
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, EnabledChecksBuildSanitizerPassSlots) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = LOOM_SANITIZER_CHECK_ACCESS |
                        LOOM_SANITIZER_CHECK_VALUE |
                        LOOM_SANITIZER_CHECK_OPERATION,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_TRUE(
      iree_string_view_is_empty(counts.source_to_low_sanitizer_reporting));
  EXPECT_EQ(counts.vector_memory_to_scalar, 1);
  EXPECT_EQ(counts.sanitizer_insert_assertions, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.sanitizer_insert_checks,
                                     IREE_SV("access|value|operation")));
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_LT(counts.sanitizer_insert_assertions_ordinal,
            counts.sanitizer_materialize_assertions_ordinal);
  EXPECT_LT(counts.vector_memory_to_scalar_ordinal,
            counts.sanitizer_insert_assertions_ordinal);
  EXPECT_LT(counts.sanitizer_materialize_assertions_ordinal,
            counts.source_to_low_ordinal);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, RaceChecksBuildRaceObservationPassSlot) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = LOOM_SANITIZER_CHECK_RACE,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_EQ(counts.vector_memory_to_scalar, 1);
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.sanitizer_insert_race_checks,
                                     IREE_SV("race")));
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, UnknownCheckBitsFailValidation) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = 1ull << 63,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_pipeline_build_to_prepared_low(
          module.get(), IREE_SV("compile"), &options, &environment_,
          loom_pass_environment_empty(), &pipeline_op));
  EXPECT_EQ(pipeline_op, nullptr);
}

TEST_F(TargetPipelineTest, UnknownReportingModeFailsValidation) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = {},
      .source_to_low_legality_diagnostic_flags = {},
      .control_flow_lowering = {},
      .sanitizer =
          {
              .checks = 0,
              .flags = 0,
              .reporting_mode = (loom_sanitizer_reporting_mode_t)99,
          },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_pipeline_build_to_prepared_low(
          module.get(), IREE_SV("compile"), &options, &environment_,
          loom_pass_environment_empty(), &pipeline_op));
  EXPECT_EQ(pipeline_op, nullptr);
}

}  // namespace
}  // namespace loom
