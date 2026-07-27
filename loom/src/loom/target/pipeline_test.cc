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
  // Number of source-to-low pass runs.
  int source_to_low = 0;
  // Diagnostics option on the source-to-low pass.
  iree_string_view_t source_to_low_diagnostics = iree_string_view_empty();
  // Sanitizer reporting option on the source-to-low pass.
  iree_string_view_t source_to_low_sanitizer_reporting =
      iree_string_view_empty();
  // Number of source assertion-insertion pass runs.
  int sanitizer_insert_assertions = 0;
  // Checks option on the source assertion-insertion pass.
  iree_string_view_t sanitizer_insert_checks = iree_string_view_empty();
  // Number of source race-observation pass runs.
  int sanitizer_insert_race_observations = 0;
  // Checks option on the source race-observation pass.
  iree_string_view_t sanitizer_insert_race_checks = iree_string_view_empty();
  // Number of target-low assertion-materialization pass runs.
  int sanitizer_materialize_assertions = 0;
  // Number of other sanitizer pass runs.
  int other_sanitizer_runs = 0;
} PipelineRunCounts;

typedef struct PipelineRunCountContext {
  // Module owning the pass IR being inspected.
  loom_module_t* module;
  // Counts updated while walking pass.run operations.
  PipelineRunCounts counts;
} PipelineRunCountContext;

iree_string_view_t FindStringOption(loom_module_t* module,
                                    loom_named_attr_slice_t options,
                                    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < options.count; ++i) {
    const loom_named_attr_t* option = &options.entries[i];
    iree_string_view_t option_name = module->strings.entries[option->name_id];
    if (!iree_string_view_equal(option_name, name)) continue;
    if (option->value.kind != LOOM_ATTR_STRING) return iree_string_view_empty();
    return module->strings.entries[loom_attr_as_string_id(option->value)];
  }
  return iree_string_view_empty();
}

iree_status_t CountSanitizerRun(void* user_data, loom_op_t* op,
                                const loom_walk_context_t* context,
                                loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_pass_run_isa(op)) {
    return iree_ok_status();
  }

  PipelineRunCountContext* count_context =
      static_cast<PipelineRunCountContext*>(user_data);
  PipelineRunCounts* counts = &count_context->counts;
  iree_string_view_t key =
      count_context->module->strings.entries[loom_pass_run_key(op)];
  if (iree_string_view_equal(key, IREE_SV("source-to-low"))) {
    ++counts->source_to_low;
    counts->source_to_low_diagnostics =
        FindStringOption(count_context->module, loom_pass_run_options(op),
                         IREE_SV("diagnostics"));
    counts->source_to_low_sanitizer_reporting =
        FindStringOption(count_context->module, loom_pass_run_options(op),
                         IREE_SV("sanitizer-reporting"));
  } else if (iree_string_view_equal(key,
                                    IREE_SV("sanitizer-insert-assertions"))) {
    ++counts->sanitizer_insert_assertions;
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
        /*.module=*/module,
    };
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_EXPECT_OK(loom_walk_region(
        module, loom_pass_pipeline_body(pipeline_op), LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){CountSanitizerRun, &count_context}, &arena,
        &walk_result));
    EXPECT_EQ(walk_result, LOOM_WALK_CONTINUE);
    counts = count_context.counts;
    iree_arena_deinitialize(&arena);
    return counts;
  }

  // Block pool backing test modules and walk scratch arenas.
  iree_arena_block_pool_t block_pool_;
  // IR context with all dialects registered.
  loom_context_t context_;
  // Provider set borrowed by the target environment for its lifetime.
  loom_target_provider_set_t provider_set_;
  // Target environment used by pipeline construction.
  loom_target_environment_t environment_;
};

TEST_F(TargetPipelineTest, ZeroChecksBuildsNoSanitizerPassSlots) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {0};

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_TRUE(iree_string_view_is_empty(counts.source_to_low_diagnostics));
  EXPECT_TRUE(
      iree_string_view_is_empty(counts.source_to_low_sanitizer_reporting));
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, OperandFormDiagnosticsBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/
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
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, TrapReportingBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/0,
          /*.flags=*/0,
          /*.reporting_mode=*/LOOM_SANITIZER_REPORTING_MODE_TRAP,
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
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, ReportOnlyBuildsSourceToLowOption) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/0,
          /*.flags=*/0,
          /*.reporting_mode=*/LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY,
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
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, EnabledChecksBuildSanitizerPassSlots) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
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
  EXPECT_EQ(counts.sanitizer_insert_assertions, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.sanitizer_insert_checks,
                                     IREE_SV("access|value|operation")));
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 0);
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 1);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, RaceChecksBuildRaceObservationPassSlot) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_RACE,
      },
  };

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("compile"), &options, &environment_,
      loom_pass_environment_empty(), &pipeline_op));

  const PipelineRunCounts counts = CountPipelineRuns(module.get(), pipeline_op);
  EXPECT_EQ(counts.source_to_low, 1);
  EXPECT_EQ(counts.sanitizer_insert_assertions, 0);
  EXPECT_EQ(counts.sanitizer_insert_race_observations, 1);
  EXPECT_TRUE(iree_string_view_equal(counts.sanitizer_insert_race_checks,
                                     IREE_SV("race")));
  EXPECT_EQ(counts.sanitizer_materialize_assertions, 0);
  EXPECT_EQ(counts.other_sanitizer_runs, 0);
}

TEST_F(TargetPipelineTest, UnknownCheckBitsFailValidation) {
  ModulePtr module = AllocateModule(IREE_SV("pipeline"));
  const loom_target_pipeline_options_t options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/1ull << 63,
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
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/0,
          /*.flags=*/0,
          /*.reporting_mode=*/(loom_sanitizer_reporting_mode_t)99,
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
