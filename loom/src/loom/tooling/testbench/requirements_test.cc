// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/requirements.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class RequirementsTest : public ::testing::Test {
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

  ModulePtr ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("requirements_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
    return plan;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
};

static iree_status_t QueryEnabled(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  (void)user_data;
  bool present = false;
  int64_t enabled = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_optional_i64_attr(
      module, attrs, IREE_SV("enabled"), &present, &enabled));
  const bool enabled_predicate = present && enabled != 0;
  *out_result = (loom_testbench_requirement_provider_result_t){
      /*.state=*/
      enabled_predicate ? LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED
                        : LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED,
      /*.provider_code=*/
      enabled_predicate ? iree_string_view_empty() : IREE_SV("fake_disabled"),
      /*.display_message=*/
      enabled_predicate ? iree_string_view_empty()
                        : IREE_SV("fake requirement was disabled"),
  };
  return iree_ok_status();
}

static iree_status_t QueryUnavailable(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  (void)user_data;
  (void)module;
  (void)attrs;
  *out_result = (loom_testbench_requirement_provider_result_t){
      /*.state=*/LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE,
      /*.provider_code=*/IREE_SV("fake_runtime_unavailable"),
      /*.display_message=*/IREE_SV("fake runtime unavailable"),
  };
  return iree_ok_status();
}

TEST_F(RequirementsTest, EvaluatesRequiresAndSkipPredicates) {
  ModulePtr module = ParseModule(R"(
check.case @runs {
  check.requires<fake.enabled> {enabled = 1}
  check.return
}

check.case @missing_requirement {
  check.requires<fake.enabled> {enabled = 0}
  check.return
}

check.case @explicit_skip {
  check.skip_if<fake.enabled> {enabled = 1} reason("case is intentionally skipped")
  check.return
}

check.case @unavailable_requirement {
  check.requires<fake.unavailable> {}
  check.return
}
)");
  ASSERT_NE(module.get(), nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module.get());
  ASSERT_EQ(plan.issue_count, 0u);

  const loom_testbench_requirement_provider_t providers[] = {
      {
          /*.name=*/IREE_SV("fake.enabled"),
          /*.user_data=*/nullptr,
          /*.query=*/QueryEnabled,
      },
      {
          /*.name=*/IREE_SV("fake.unavailable"),
          /*.user_data=*/nullptr,
          /*.query=*/QueryUnavailable,
      },
  };
  loom_testbench_requirement_provider_registry_t registry = {};
  loom_testbench_requirement_provider_registry_initialize(
      providers, IREE_ARRAYSIZE(providers), &registry);

  loom_testbench_requirement_result_t result = {};
  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[0], &registry, &result));
  EXPECT_FALSE(result.skipped);

  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[1], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_EQ(result.op_kind, LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES);
  EXPECT_EQ(result.code,
            LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_REQUIREMENT_NOT_SATISFIED);
  EXPECT_TRUE(iree_string_view_equal(result.provider, IREE_SV("fake.enabled")));
  EXPECT_TRUE(
      iree_string_view_equal(result.provider_code, IREE_SV("fake_disabled")));
  EXPECT_TRUE(iree_string_view_equal(result.display_message,
                                     IREE_SV("fake requirement was disabled")));

  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[2], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_EQ(result.op_kind, LOOM_TESTBENCH_REQUIREMENT_OP_KIND_SKIP_IF);
  EXPECT_EQ(result.code,
            LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_SKIP_PREDICATE_MATCHED);
  EXPECT_TRUE(iree_string_view_is_empty(result.provider_code));
  EXPECT_TRUE(iree_string_view_equal(result.display_message,
                                     IREE_SV("case is intentionally skipped")));

  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[3], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_EQ(result.op_kind, LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES);
  EXPECT_EQ(result.code,
            LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_UNAVAILABLE);
  EXPECT_TRUE(
      iree_string_view_equal(result.provider, IREE_SV("fake.unavailable")));
  EXPECT_TRUE(iree_string_view_equal(result.provider_code,
                                     IREE_SV("fake_runtime_unavailable")));
}

TEST_F(RequirementsTest, ReportsUnknownProvidersAsSkippedEvidence) {
  ModulePtr module = ParseModule(R"(
check.case @unknown_requires {
  check.requires<missing.provider> {}
  check.return
}

check.case @unknown_skip_if {
  check.skip_if<missing.provider> {} reason("human text")
  check.return
}
)");
  ASSERT_NE(module.get(), nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module.get());
  ASSERT_EQ(plan.issue_count, 0u);

  loom_testbench_requirement_provider_registry_t registry = {};
  loom_testbench_requirement_provider_registry_initialize(nullptr, 0,
                                                          &registry);
  loom_testbench_requirement_result_t result = {};
  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[0], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_EQ(result.op_kind, LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES);
  EXPECT_EQ(result.code,
            LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_NOT_REGISTERED);
  EXPECT_TRUE(
      iree_string_view_equal(result.provider, IREE_SV("missing.provider")));
  EXPECT_TRUE(iree_string_view_is_empty(result.provider_code));
  EXPECT_TRUE(iree_string_view_is_empty(result.display_message));

  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[1], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_EQ(result.op_kind, LOOM_TESTBENCH_REQUIREMENT_OP_KIND_SKIP_IF);
  EXPECT_EQ(result.code,
            LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_NOT_REGISTERED);
  EXPECT_TRUE(
      iree_string_view_equal(result.display_message, IREE_SV("human text")));
}

}  // namespace
}  // namespace loom
