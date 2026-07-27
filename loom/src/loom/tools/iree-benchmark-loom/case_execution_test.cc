// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/case_execution.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/tools/iree-benchmark-loom/options.h"

namespace loom {
namespace {

typedef struct event_collector_t {
  iree_host_size_t event_count;
  iree_benchmark_loom_event_t events[8];
} event_collector_t;

static iree_status_t collect_event(void* user_data,
                                   const iree_benchmark_loom_event_t* event) {
  event_collector_t* collector = (event_collector_t*)user_data;
  if (collector->event_count == IREE_ARRAYSIZE(collector->events)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "benchmark event collector is full");
  }
  collector->events[collector->event_count++] = *event;
  return iree_ok_status();
}

class BenchmarkCaseExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    iree_arena_initialize(&block_pool_, &execution_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : owned_modules_) {
      loom_module_free(module);
    }
    owned_modules_.clear();
    iree_arena_deinitialize(&execution_arena_);
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

  loom_testbench_module_plan_t PlanModule(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("case_execution_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    owned_modules_.push_back(module);

    loom_testbench_module_plan_t module_plan = {};
    IREE_EXPECT_OK(loom_testbench_plan_module(module, nullptr, &plan_arena_,
                                              &module_plan));
    EXPECT_EQ(module_plan.issue_count, 0u);
    return module_plan;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  iree_arena_allocator_t execution_arena_;
  loom_context_t context_;
  std::vector<loom_module_t*> owned_modules_;
};

TEST_F(BenchmarkCaseExecutionTest, RunsCaseEndToEndWorkItemThroughEventSink) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("case_end_to_end");
  options.iterations = 1;
  options.warmup_iterations = 0;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));
  ASSERT_EQ(work_plan.work_item_count, 1u);

  loom_testbench_case_execution_options_t execution_options = {};
  loom_testbench_case_execution_options_initialize(&execution_options);
  execution_options.materializer.host_allocator = iree_allocator_system();

  event_collector_t collector = {};
  iree_benchmark_loom_event_sink_t event_sink = {};
  event_sink.emit = collect_event;
  event_sink.user_data = &collector;
  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  IREE_ASSERT_OK(iree_benchmark_loom_run_case_end_to_end_work_item(
      &run, &module_plan, &work_plan, &work_plan.work_items[0],
      &execution_options, &execution_arena_, iree_allocator_system(),
      &event_sink, &correctness_sample_count, &correctness_failed_sample_count,
      &failed_benchmark_count));

  EXPECT_EQ(correctness_sample_count, 2u);
  EXPECT_EQ(correctness_failed_sample_count, 0u);
  EXPECT_EQ(failed_benchmark_count, 0u);
  ASSERT_EQ(collector.event_count, 8u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[0].sample.case_sample_ordinal, 0u);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[0].sample.candidate->candidate_id, IREE_SV("c0")));
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[1].sample.candidate->candidate_id, IREE_SV("c1")));
  EXPECT_EQ(collector.events[2].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[2].sample.case_sample_ordinal, 1u);
  EXPECT_EQ(collector.events[4].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_EQ(collector.events[4].benchmark_result.work_item_index, 0u);
  EXPECT_EQ(collector.events[4].benchmark_result.correctness_sample_count, 2u);
  EXPECT_EQ(collector.events[5].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE);
  EXPECT_EQ(collector.events[6].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[6].benchmark_result.candidate->candidate_id,
      IREE_SV("c1")));
  EXPECT_EQ(collector.events[7].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

}  // namespace
}  // namespace loom
