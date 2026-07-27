// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/work_plan.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

class BenchmarkWorkPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : owned_modules_) {
      loom_module_free(module);
    }
    owned_modules_.clear();
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("work_plan_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModuleAllowingIssues(const char* source) {
    loom_module_t* module = ParseModule(source);
    owned_modules_.push_back(module);
    loom_testbench_module_plan_t module_plan = {};
    IREE_EXPECT_OK(loom_testbench_plan_module(module, nullptr, &plan_arena_,
                                              &module_plan));
    return module_plan;
  }

  loom_testbench_module_plan_t PlanModule(const char* source) {
    loom_testbench_module_plan_t module_plan = PlanModuleAllowingIssues(source);
    EXPECT_EQ(module_plan.issue_count, 0u);
    return module_plan;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  std::vector<loom_module_t*> owned_modules_;
};

TEST_F(BenchmarkWorkPlanTest, DeduplicatesPinnedDispatchSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp>
check.benchmark<@mlp> @decode {rows = 2}
check.benchmark<@mlp> @full {rows = 3584}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode =
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 3u);
  EXPECT_EQ(work_plan.logical_sample_count, 4u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 2u);
  EXPECT_EQ(work_plan.work_item_count, 2u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].case_sample_ordinal, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[2].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[2].case_sample_ordinal, 0u);
  EXPECT_EQ(work_plan.logical_samples[2].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[3].selection_index, 2u);
  EXPECT_EQ(work_plan.logical_samples[3].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[3].work_item_index, 1u);

  EXPECT_EQ(work_plan.work_items[0].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].dispatch_compile_item_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].case_sample_ordinal, 0u);
  EXPECT_EQ(work_plan.work_items[1].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[1].dispatch_compile_item_index, 1u);
  EXPECT_EQ(work_plan.work_items[1].case_sample_ordinal, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, DeduplicatesEquivalentCaseEndToEndSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
check.benchmark<@sampled> @value7_a {value = 7}
check.benchmark<@sampled> @value7_b {value = 7}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("case_end_to_end");

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 4u);
  EXPECT_EQ(work_plan.logical_sample_count, 4u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 0u);
  EXPECT_EQ(work_plan.work_item_count, 2u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[2].selection_index, 2u);
  EXPECT_EQ(work_plan.logical_samples[2].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[2].work_item_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[3].selection_index, 3u);
  EXPECT_EQ(work_plan.logical_samples[3].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[3].work_item_index, 1u);

  EXPECT_EQ(work_plan.work_items[0].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.work_items[0].end_benchmark_sample, 2u);
  EXPECT_FALSE(work_plan.work_items[0].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.work_items[1].representative_selection_index, 2u);
  EXPECT_EQ(work_plan.work_items[1].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.work_items[1].end_benchmark_sample, 1u);
  EXPECT_TRUE(work_plan.work_items[1].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.work_items[1].case_sample_ordinal, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, AppliesSampleWindowBeforeDedupe) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7, 11]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode =
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;
  options.sample_ordinal = 1;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.logical_samples[0].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.logical_samples[1].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);

  EXPECT_TRUE(work_plan.dispatch_compile_items[0].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.dispatch_compile_items[0].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.work_items[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.work_items[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.work_items[0].case_sample_ordinal, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, SharesCompileOnceCandidateAcrossSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp>
check.benchmark<@mlp> @decode {rows = 2}
check.benchmark<@mlp> @full {rows = 3584}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 3u);
  EXPECT_EQ(work_plan.logical_sample_count, 4u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 2u);
  EXPECT_FALSE(work_plan.dispatch_compile_items[0].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.work_items[0].dispatch_compile_item_index, 0u);
  EXPECT_EQ(work_plan.work_items[1].dispatch_compile_item_index, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, KeepsSampleCompilationPoliciesSeparate) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp>
check.benchmark<@mlp> @decode {rows = 2}
check.benchmark<@mlp> @full {rows = 3584}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 3u);
  EXPECT_EQ(work_plan.logical_sample_count, 8u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 3u);
  EXPECT_EQ(work_plan.work_item_count, 4u);

  EXPECT_FALSE(work_plan.dispatch_compile_items[0].has_case_sample_ordinal);
  EXPECT_TRUE(work_plan.dispatch_compile_items[1].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.dispatch_compile_items[1].case_sample_ordinal, 0u);
  EXPECT_TRUE(work_plan.dispatch_compile_items[2].has_case_sample_ordinal);
  EXPECT_EQ(work_plan.dispatch_compile_items[2].case_sample_ordinal, 1u);

  EXPECT_TRUE(iree_string_view_equal(work_plan.work_items[0].sample_compilation,
                                     IREE_SV("once")));
  EXPECT_EQ(work_plan.work_items[0].dispatch_compile_item_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].case_sample_ordinal, 0u);
  EXPECT_TRUE(iree_string_view_equal(work_plan.work_items[1].sample_compilation,
                                     IREE_SV("once")));
  EXPECT_EQ(work_plan.work_items[1].dispatch_compile_item_index, 0u);
  EXPECT_EQ(work_plan.work_items[1].case_sample_ordinal, 1u);
  EXPECT_TRUE(iree_string_view_equal(work_plan.work_items[2].sample_compilation,
                                     IREE_SV("per_sample")));
  EXPECT_EQ(work_plan.work_items[2].dispatch_compile_item_index, 1u);
  EXPECT_EQ(work_plan.work_items[2].case_sample_ordinal, 0u);
  EXPECT_TRUE(iree_string_view_equal(work_plan.work_items[3].sample_compilation,
                                     IREE_SV("per_sample")));
  EXPECT_EQ(work_plan.work_items[3].dispatch_compile_item_index, 2u);
  EXPECT_EQ(work_plan.work_items[3].case_sample_ordinal, 1u);

  EXPECT_NE(work_plan.logical_samples[0].work_item_index,
            work_plan.logical_samples[2].work_item_index);
  EXPECT_NE(work_plan.logical_samples[1].work_item_index,
            work_plan.logical_samples[3].work_item_index);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, CompareSelectionsShareDuplicatePhysicalWork) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp> @decode_a {rows = 2}
check.benchmark<@mlp> @decode_b {rows = 2}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode =
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;
  options.compare = IREE_SV("@decode_a,@decode_b");

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);
  EXPECT_EQ(work_plan.dispatch_compile_items[0].case_sample_ordinal, 0u);
  EXPECT_EQ(work_plan.work_items[0].case_sample_ordinal, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, CompareSampleFlagNarrowsBeforeDedupe) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7, 11]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_compilation_mode =
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;
  options.compare = IREE_SV("@all_a,@all_b");
  options.sample_ordinal = 1;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.dispatch_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.dispatch_compile_items[0].case_sample_ordinal, 1u);
  EXPECT_EQ(work_plan.work_items[0].case_sample_ordinal, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, RejectsSelectedBenchmarkWithZeroSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModuleAllowingIssues(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @missing_value {value = 9}
)");
  ASSERT_EQ(module_plan.benchmark_count, 1u);
  ASSERT_EQ(module_plan.benchmarks[0].sample_count, 0u);

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");

  iree_benchmark_loom_work_plan_t work_plan = {};
  iree::Status status(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("zero executable samples"));
  EXPECT_THAT(status.ToString(), HasSubstr("missing_value"));
  EXPECT_THAT(status.ToString(), HasSubstr("sampled"));
}

TEST_F(BenchmarkWorkPlanTest, RejectsMissingBenchmarkSelector) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  check.return
}

check.benchmark<@sampled> @available
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.selected_benchmark = IREE_SV("missing");

  iree_benchmark_loom_work_plan_t work_plan = {};
  iree::Status status(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(status.ToString(), HasSubstr("no check.benchmark matched"));
  EXPECT_THAT(status.ToString(), HasSubstr("missing"));
}

TEST(BenchmarkComparisonExecutionTest, AbabaSampleCapacityKeepsBaselinePairs) {
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA,
                /*candidate_count=*/2, /*candidate_index=*/0,
                /*repetitions=*/3),
            4u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA,
                /*candidate_count=*/2, /*candidate_index=*/1,
                /*repetitions=*/3),
            3u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN,
                /*candidate_count=*/4, /*candidate_index=*/0,
                /*repetitions=*/3),
            3u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN,
                /*candidate_count=*/4, /*candidate_index=*/3,
                /*repetitions=*/3),
            3u);
}

}  // namespace
}  // namespace loom
