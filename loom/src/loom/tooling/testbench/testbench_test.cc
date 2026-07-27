// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/testbench.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class TestbenchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
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
                                   IREE_SV("testbench_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
};

TEST_F(TestbenchTest, DiscoversCasesAndBenchmarks) {
  loom_module_t* module = ParseModule(R"(
test.func @scale(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case public @scale_case {
  %input = check.literal value(2) : i32
  %actual = test.invoke @scale(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%input) : i32
  check.return
}

check.case @private_case {
  check.return
}

check.benchmark<@scale_case> @scale_latency
check.benchmark<@private_case>
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(plan.cases[0].name, IREE_SV("scale_case")));
  EXPECT_TRUE(plan.cases[0].is_public);
  EXPECT_EQ(plan.cases[0].parameter_count, 0u);
  EXPECT_EQ(plan.cases[0].value_source_count, 1u);
  ASSERT_EQ(plan.cases[0].value_sources[0].kind,
            LOOM_TESTBENCH_VALUE_SOURCE_LITERAL);
  EXPECT_EQ(loom_attr_as_i64(plan.cases[0].value_sources[0].literal.value), 2);
  EXPECT_EQ(plan.cases[0].file_write_count, 0u);
  ASSERT_EQ(plan.cases[0].invocation_count, 1u);
  EXPECT_EQ(plan.cases[0].invocations[0].kind,
            LOOM_TESTBENCH_INVOCATION_ACTUAL);
  EXPECT_EQ(plan.cases[0].invocations[0].input_count, 1u);
  EXPECT_EQ(plan.cases[0].invocations[0].result_count, 1u);
  ASSERT_EQ(plan.cases[0].expectation_count, 1u);
  EXPECT_EQ(plan.cases[0].expectations[0].kind,
            LOOM_TESTBENCH_EXPECTATION_EQUAL);
  EXPECT_EQ(plan.cases[0].sample_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(plan.cases[1].name, IREE_SV("private_case")));
  EXPECT_FALSE(plan.cases[1].is_public);

  ASSERT_EQ(plan.benchmark_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(plan.benchmarks[0].name,
                                     IREE_SV("scale_latency")));
  EXPECT_EQ(plan.benchmarks[0].case_index, 0u);
  EXPECT_EQ(plan.benchmarks[0].attrs.count, 0u);
  EXPECT_EQ(plan.benchmarks[0].sample_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(plan.benchmarks[1].name, IREE_SV("private")));
  EXPECT_EQ(plan.benchmarks[1].symbol, nullptr);
  EXPECT_EQ(plan.benchmarks[1].case_index, 1u);
  EXPECT_EQ(plan.benchmarks[1].sample_count, 1u);
  EXPECT_EQ(plan.issue_count, 0u);

  loom_module_free(module);
}

TEST_F(TestbenchTest, PlansValueSourcesAndFileWrites) {
  loom_module_t* module = ParseModule(R"(
check.case @sources {
  %seed = check.param.seed base(7) count(2) : i64
  %scalar = check.literal value(42) : i32
  %iota = check.generate.iota offset(0) step(1) : tensor<4xi32>
  %fill = check.generate.fill value(17) : tensor<4xi32>
  %uniform = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<4xf32>
  %file = check.file.read.npy path("fixtures/input.npy") : tensor<4xf32>
  check.file.write.npy value(%uniform) path("outputs/actual.npy") mode(always) : tensor<4xf32>
  check.expect.equal actual(%scalar) expected(%scalar) : i32
  check.expect.bitwise actual(%uniform) expected(%file) : tensor<4xf32>
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.parameter_count, 1u);
  ASSERT_EQ(case_plan.value_source_count, 5u);
  EXPECT_EQ(case_plan.value_sources[0].kind,
            LOOM_TESTBENCH_VALUE_SOURCE_LITERAL);
  EXPECT_EQ(loom_attr_as_i64(case_plan.value_sources[0].literal.value), 42);
  EXPECT_EQ(case_plan.value_sources[1].kind, LOOM_TESTBENCH_VALUE_SOURCE_IOTA);
  EXPECT_EQ(loom_attr_as_i64(case_plan.value_sources[1].iota.step), 1);
  EXPECT_EQ(case_plan.value_sources[2].kind, LOOM_TESTBENCH_VALUE_SOURCE_FILL);
  EXPECT_EQ(loom_attr_as_i64(case_plan.value_sources[2].fill.value), 17);
  EXPECT_EQ(case_plan.value_sources[3].kind,
            LOOM_TESTBENCH_VALUE_SOURCE_RANDOM_UNIFORM);
  EXPECT_EQ(case_plan.value_sources[3].random_uniform.seed_value_id,
            case_plan.parameters[0].value_id);
  EXPECT_EQ(case_plan.value_sources[4].kind,
            LOOM_TESTBENCH_VALUE_SOURCE_FILE_READ_NPY);
  EXPECT_TRUE(iree_string_view_equal(case_plan.value_sources[4].file.path,
                                     IREE_SV("fixtures/input.npy")));

  ASSERT_EQ(case_plan.file_write_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(case_plan.file_writes[0].path,
                                     IREE_SV("outputs/actual.npy")));
  EXPECT_EQ(case_plan.file_writes[0].mode,
            LOOM_CHECK_FILE_WRITE_NPY_MODE_ALWAYS);
  ASSERT_EQ(case_plan.expectation_count, 2u);
  EXPECT_EQ(case_plan.expectations[0].kind, LOOM_TESTBENCH_EXPECTATION_EQUAL);
  EXPECT_EQ(case_plan.expectations[1].kind, LOOM_TESTBENCH_EXPECTATION_BITWISE);
  EXPECT_EQ(case_plan.issue_count, 0u);
  EXPECT_EQ(plan.issue_count, 0u);

  loom_module_free(module);
}

TEST_F(TestbenchTest, PlansDeterministicParameterSamples) {
  loom_module_t* module = ParseModule(R"(
check.case @sweep {
  %m = check.param.range po2 bounds(1 to 16) name("m") : index
  %n = check.param.choice values([3, 5]) name("n") : i32
  %seed = check.param.seed base(10) count(4) name("seed") : i64
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_plan_options_t options = {};
  loom_testbench_plan_options_initialize(&options);
  options.max_samples_per_case = 6;
  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, &options, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.parameter_count, 3u);
  EXPECT_EQ(case_plan.cartesian_sample_count, 40u);
  EXPECT_EQ(case_plan.sample_count, 6u);
  EXPECT_TRUE(case_plan.sample_count_truncated);
  EXPECT_EQ(case_plan.issue_count, 0u);

  EXPECT_EQ(case_plan.parameters[0].kind, LOOM_TESTBENCH_PARAMETER_RANGE);
  EXPECT_TRUE(
      iree_string_view_equal(case_plan.parameters[0].name, IREE_SV("m")));
  EXPECT_EQ(case_plan.parameters[0].sample_count, 5u);
  EXPECT_EQ(case_plan.parameters[1].kind, LOOM_TESTBENCH_PARAMETER_CHOICE);
  EXPECT_TRUE(
      iree_string_view_equal(case_plan.parameters[1].name, IREE_SV("n")));
  EXPECT_EQ(case_plan.parameters[1].sample_count, 2u);
  EXPECT_EQ(case_plan.parameters[2].kind, LOOM_TESTBENCH_PARAMETER_SEED);
  EXPECT_TRUE(
      iree_string_view_equal(case_plan.parameters[2].name, IREE_SV("seed")));
  EXPECT_EQ(case_plan.parameters[2].sample_count, 4u);

  EXPECT_EQ(loom_testbench_case_sample_parameter_ordinal(&case_plan, 5, 0), 0u);
  EXPECT_EQ(loom_testbench_case_sample_parameter_ordinal(&case_plan, 5, 1), 1u);
  EXPECT_EQ(loom_testbench_case_sample_parameter_ordinal(&case_plan, 5, 2), 0u);

  loom_attribute_t value = {};
  IREE_ASSERT_OK(loom_testbench_parameter_sample_value(&case_plan.parameters[0],
                                                       3, &value));
  EXPECT_EQ(value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(loom_attr_as_i64(value), 8);
  IREE_ASSERT_OK(loom_testbench_parameter_sample_value(&case_plan.parameters[1],
                                                       1, &value));
  EXPECT_EQ(value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(loom_attr_as_i64(value), 5);
  IREE_ASSERT_OK(loom_testbench_parameter_sample_value(&case_plan.parameters[2],
                                                       2, &value));
  EXPECT_EQ(value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(loom_attr_as_i64(value), 12);

  loom_module_free(module);
}

TEST_F(TestbenchTest, PlansBenchmarkParameterAssignments) {
  loom_module_t* module = ParseModule(R"(
check.case @sweep {
  %m = check.param.range po2 bounds(1 to 16) name("m") : index
  %n = check.param.choice values([3, 5]) name("n") : i32
  %seed = check.param.seed base(10) count(4) name("seed") : i64
  check.return
}

check.benchmark<@sweep> @only_m {m = 8}
check.benchmark<@sweep> @m_and_seed {m = 8, seed = 12}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.benchmark_count, 2u);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  const loom_testbench_benchmark_plan_t& only_m = plan.benchmarks[0];
  EXPECT_EQ(only_m.sample_count, 8u);
  EXPECT_EQ(only_m.cartesian_sample_count, 8u);
  ASSERT_EQ(only_m.parameter_sample_ordinal_count, 3u);
  EXPECT_EQ(only_m.parameter_sample_ordinals[0], 3u);
  EXPECT_EQ(only_m.parameter_sample_ordinals[1],
            LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL);
  EXPECT_EQ(only_m.parameter_sample_ordinals[2],
            LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL);
  EXPECT_EQ(
      loom_testbench_benchmark_sample_case_ordinal(&case_plan, &only_m, 0), 3u);
  EXPECT_EQ(
      loom_testbench_benchmark_sample_case_ordinal(&case_plan, &only_m, 7),
      38u);

  const loom_testbench_benchmark_plan_t& m_and_seed = plan.benchmarks[1];
  EXPECT_EQ(m_and_seed.sample_count, 2u);
  EXPECT_EQ(m_and_seed.cartesian_sample_count, 2u);
  ASSERT_EQ(m_and_seed.parameter_sample_ordinal_count, 3u);
  EXPECT_EQ(m_and_seed.parameter_sample_ordinals[0], 3u);
  EXPECT_EQ(m_and_seed.parameter_sample_ordinals[1],
            LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL);
  EXPECT_EQ(m_and_seed.parameter_sample_ordinals[2], 2u);
  EXPECT_EQ(
      loom_testbench_benchmark_sample_case_ordinal(&case_plan, &m_and_seed, 0),
      23u);
  EXPECT_EQ(
      loom_testbench_benchmark_sample_case_ordinal(&case_plan, &m_and_seed, 1),
      28u);

  loom_module_free(module);
}

TEST_F(TestbenchTest, ReportsInvalidBenchmarkAssignments) {
  loom_module_t* module = ParseModule(R"(
check.case @sweep {
  %m = check.param.range po2 bounds(1 to 16) name("m") : index
  %n = check.param.choice values([3, 5]) name("n") : i32
  check.return
}

check.benchmark<@sweep> @bad_key {missing = 8}
check.benchmark<@sweep> @bad_value {n = 7}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.benchmark_count, 2u);
  ASSERT_EQ(plan.issue_count, 2u);
  EXPECT_EQ(plan.issues[0].kind,
            LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT);
  EXPECT_EQ(plan.issues[0].benchmark_index, 0u);
  EXPECT_EQ(plan.issues[1].kind,
            LOOM_TESTBENCH_ISSUE_INVALID_BENCHMARK_ASSIGNMENT);
  EXPECT_EQ(plan.issues[1].benchmark_index, 1u);
  EXPECT_EQ(plan.benchmarks[0].sample_count, 0u);
  EXPECT_EQ(plan.benchmarks[1].sample_count, 0u);

  loom_module_free(module);
}

TEST_F(TestbenchTest, ReportsDuplicateParameterNames) {
  loom_module_t* module = ParseModule(R"(
check.case @bad {
  %m0 = check.param.range po2 bounds(1 to 16) name("m") : index
  %m1 = check.param.choice values([3, 5]) name("m") : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 1u);
  EXPECT_EQ(plan.issues[0].kind, LOOM_TESTBENCH_ISSUE_DUPLICATE_PARAMETER_NAME);
  EXPECT_EQ(plan.issues[0].case_index, 0u);
  EXPECT_EQ(plan.cases[0].sample_count, 0u);

  loom_module_free(module);
}

TEST_F(TestbenchTest, ReportsUnsupportedCaseBodyOps) {
  loom_module_t* module = ParseModule(R"(
check.case @bad {
  %x = check.literal value(1) : i32
  test.use %x : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 1u);
  EXPECT_EQ(plan.cases[0].issue_count, 1u);
  EXPECT_EQ(plan.cases[0].sample_count, 0u);
  EXPECT_EQ(plan.issues[0].kind, LOOM_TESTBENCH_ISSUE_UNSUPPORTED_CASE_BODY_OP);
  EXPECT_EQ(plan.issues[0].case_index, 0u);
  EXPECT_EQ(plan.issues[0].op->kind, LOOM_OP_TEST_USE);

  loom_module_free(module);
}

TEST_F(TestbenchTest, ReportsInvalidParameters) {
  loom_module_t* module = ParseModule(R"(
check.case @bad {
  %m = check.param.range po2 bounds(0 to 8) : index
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 1u);
  EXPECT_EQ(plan.cases[0].parameter_count, 1u);
  EXPECT_EQ(plan.cases[0].parameters[0].sample_count, 0u);
  EXPECT_EQ(plan.cases[0].sample_count, 0u);
  EXPECT_EQ(plan.issues[0].kind, LOOM_TESTBENCH_ISSUE_INVALID_PARAMETER);
  EXPECT_EQ(plan.issues[0].case_index, 0u);
  EXPECT_EQ(plan.issues[0].op->kind, LOOM_OP_CHECK_PARAM_RANGE);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
