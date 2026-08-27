// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_projection.h"

#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

struct ModuleIndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using ModuleIndexPtr =
    std::unique_ptr<loom_link_module_index_t, ModuleIndexDeleter>;

struct LinkPlanDeleter {
  void operator()(loom_link_plan_t* plan) const { loom_link_plan_free(plan); }
};
using LinkPlanPtr = std::unique_ptr<loom_link_plan_t, LinkPlanDeleter>;

class LinkPlanProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, vtables, (uint16_t)vtable_count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        loom_func_dialect_op_semantics(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_FUNC, semantics, (uint16_t)semantics_count));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("projection_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  ModuleIndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return ModuleIndexPtr(index);
  }

  void AddModule(loom_link_module_index_t* index, const loom_module_t* module,
                 iree_string_view_t provider_name,
                 loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/provider_name,
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_materialized(
        index, module, &options, /*out_provider_ordinal=*/nullptr));
  }

  LinkPlanPtr BuildPlan(const loom_link_module_index_t* index,
                        const loom_link_plan_options_t* options) {
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        loom_link_plan_build(index, options, iree_allocator_system(), &plan));
    return LinkPlanPtr(plan);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkPlanProjectionTest, GroupsExactOrdinalsBySourceModule) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl public @callee(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @callee(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public @callee(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  ModuleIndexPtr index = CreateIndex();
  AddModule(index.get(), harness, IREE_SV("harness"),
            LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddModule(index.get(), library, IREE_SV("library"),
            LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const iree_string_view_t roots[] = {IREE_SV("@entry")};
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  LinkPlanPtr plan = BuildPlan(index.get(), &options);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  ASSERT_EQ(projection.modules.count, 2u);
  ASSERT_EQ(projection.symbols.count, 4u);

  for (iree_host_size_t module_index = 0;
       module_index < projection.modules.count; ++module_index) {
    const loom_link_plan_module_selection_t* module =
        &projection.modules.values[module_index];
    ASSERT_NE(module->source_module, nullptr);
    EXPECT_EQ(module->source_module->ordinal, module_index);
    ASSERT_EQ(module->symbols.count, 2u);
    EXPECT_EQ(module->symbols.values[0].source_symbol->module_symbol_ordinal,
              0u);
    EXPECT_EQ(module->symbols.values[1].source_symbol->module_symbol_ordinal,
              1u);
    for (iree_host_size_t symbol_index = 0;
         symbol_index < module->symbols.count; ++symbol_index) {
      const loom_link_plan_module_symbol_t* projected_symbol =
          &module->symbols.values[symbol_index];
      EXPECT_EQ(projected_symbol->source_symbol->module_ordinal,
                module->source_module->ordinal);
      const loom_link_plan_symbol_t* planned_symbol =
          projected_symbol->plan_symbol;
      ASSERT_NE(planned_symbol, nullptr);
      const loom_link_module_index_symbol_t* source_symbol =
          loom_link_module_index_symbol_at(index.get(),
                                           planned_symbol->symbol_ordinal);
      ASSERT_NE(source_symbol, nullptr);
      EXPECT_EQ(source_symbol, projected_symbol->source_symbol);
    }
  }
  EXPECT_EQ(projection.modules.values[0].symbols.values[0].plan_symbol->ordinal,
            1u);
  EXPECT_EQ(projection.modules.values[0].symbols.values[1].plan_symbol->ordinal,
            0u);
}

TEST_F(LinkPlanProjectionTest, EmptyArchiveHasNoProjectionStorage) {
  ModuleIndexPtr index = CreateIndex();
  LinkPlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  EXPECT_EQ(projection.modules.count, 0u);
  EXPECT_EQ(projection.modules.values, nullptr);
  EXPECT_EQ(projection.symbols.count, 0u);
  EXPECT_EQ(projection.symbols.values, nullptr);
}

TEST_F(LinkPlanProjectionTest, ArchiveProjectsSymbolEmptyModules) {
  loom_module_t* metadata = Parse(IREE_SV("test.module_metadata\n"));
  ModuleIndexPtr index = CreateIndex();
  AddModule(index.get(), metadata, IREE_SV("metadata"),
            LOOM_LINK_PROVIDER_ROLE_INPUT);
  LinkPlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  ASSERT_EQ(projection.modules.count, 1u);
  EXPECT_EQ(projection.modules.values[0].source_module,
            loom_link_module_index_module_at(index.get(), 0));
  EXPECT_EQ(projection.modules.values[0].symbols.count, 0u);
  EXPECT_EQ(projection.modules.values[0].symbols.values, nullptr);
  EXPECT_EQ(projection.symbols.count, 0u);
  EXPECT_EQ(projection.symbols.values, nullptr);
}

}  // namespace
}  // namespace loom
