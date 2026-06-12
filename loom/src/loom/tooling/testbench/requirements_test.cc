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

static iree_status_t QueryEnabled(void* user_data, const loom_module_t* module,
                                  loom_named_attr_slice_t attrs,
                                  bool* out_satisfied,
                                  iree_string_view_t* out_reason) {
  (void)user_data;
  bool present = false;
  int64_t enabled = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_optional_i64_attr(
      module, attrs, IREE_SV("enabled"), &present, &enabled));
  *out_satisfied = present && enabled != 0;
  *out_reason = IREE_SV("fake requirement was disabled");
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
  EXPECT_TRUE(iree_string_view_equal(result.provider, IREE_SV("fake.enabled")));
  EXPECT_TRUE(iree_string_view_equal(result.reason,
                                     IREE_SV("fake requirement was disabled")));

  IREE_ASSERT_OK(loom_testbench_evaluate_case_requirements(
      module.get(), &plan.cases[2], &registry, &result));
  EXPECT_TRUE(result.skipped);
  EXPECT_TRUE(iree_string_view_equal(result.reason,
                                     IREE_SV("case is intentionally skipped")));
}

TEST_F(RequirementsTest, ReportsUnknownProviders) {
  ModulePtr module = ParseModule(R"(
check.case @unknown {
  check.requires<missing.provider> {}
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_testbench_evaluate_case_requirements(
                            module.get(), &plan.cases[0], &registry, &result));
}

}  // namespace
}  // namespace loom
