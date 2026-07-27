// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_liveness.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct ApplyEdgeTestState {
  loom_string_id_t contract_id;
  loom_symbol_id_t provider_symbol_id;
} ApplyEdgeTestState;

static bool RootPublicFunc(void* user_data, const loom_module_t* module,
                           loom_symbol_id_t symbol_id,
                           const loom_symbol_t* symbol) {
  (void)user_data;
  (void)symbol_id;
  if (!symbol->defining_op) return false;
  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  return loom_func_like_isa(function) &&
         loom_func_like_visibility(function) == LOOM_FUNC_VISIBILITY_PUBLIC;
}

static iree_status_t MarkProviderForApply(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* op) {
  ApplyEdgeTestState* state = (ApplyEdgeTestState*)user_data;
  if (!loom_func_apply_isa(op)) return iree_ok_status();
  if (loom_func_apply_contract(op) != state->contract_id) {
    return iree_ok_status();
  }
  return loom_symbol_liveness_mark_symbol_id(context,
                                             state->provider_symbol_id);
}

class SymbolLivenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("symbol_liveness_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_symbol_liveness_t ComputeLiveness(
      const loom_module_t* module,
      const loom_symbol_liveness_options_t* options) {
    loom_symbol_dependency_table_t dependencies = {};
    IREE_CHECK_OK(loom_symbol_dependency_table_build(module, &analysis_arena_,
                                                     &dependencies));
    loom_symbol_liveness_t liveness = {};
    IREE_CHECK_OK(loom_symbol_liveness_compute(module, &dependencies, options,
                                               &analysis_arena_, &liveness));
    return liveness;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(SymbolLivenessTest, ConcreteEdgesReachCalledPrivateFunctions) {
  ModulePtr module = ParseModule(R"(
func.def public @entry() {
  func.call @helper() : ()
  func.return
}

func.def @helper() {
  func.return
}

func.def @dead() {
  func.return
}
)");

  loom_symbol_liveness_options_t options = {
      /*.flags=*/{},
      /*.root_query=*/RootPublicFunc,
  };
  loom_symbol_liveness_t liveness = ComputeLiveness(module.get(), &options);

  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("entry"))));
  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("helper"))));
  EXPECT_FALSE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("dead"))));
  EXPECT_EQ(liveness.contributed_edge_count, 0u);
}

TEST_F(SymbolLivenessTest, ContributorsScanReachableBodiesOnly) {
  ModulePtr module = ParseModule(R"(
func.template<demo.contract> @provider(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}

func.template<dead.contract> @dead_provider(%arg0: i32) -> (i32) {
  func.return %arg0 : i32
}

func.def public @entry(%arg0: i32) -> (i32) {
  %result = func.call @live_user(%arg0) : (i32) -> (i32)
  func.return %result : i32
}

func.def @live_user(%arg0: i32) -> (i32) {
  %result = func.apply<demo.contract>(%arg0) : (i32) -> (i32)
  func.return %result : i32
}

func.def @dead_user(%arg0: i32) -> (i32) {
  %result = func.apply<dead.contract>(%arg0) : (i32) -> (i32)
  func.return %result : i32
}
)");

  ApplyEdgeTestState apply_state = {
      /*.contract_id=*/
      loom_module_lookup_string(module.get(), IREE_SV("demo.contract")),
      /*.provider_symbol_id=*/FindSymbol(module.get(), IREE_SV("provider")),
  };
  ASSERT_NE(apply_state.contract_id, LOOM_STRING_ID_INVALID);
  loom_symbol_liveness_contributor_t contributor = {
      /*.visit_op=*/MarkProviderForApply,
      /*.user_data=*/&apply_state,
  };
  loom_symbol_liveness_options_t options = {
      /*.flags=*/{},
      /*.root_query=*/RootPublicFunc,
      /*.root_query_user_data=*/{},
      /*.contributors=*/&contributor,
      /*.contributor_count=*/1,
  };
  loom_symbol_liveness_t liveness = ComputeLiveness(module.get(), &options);

  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("entry"))));
  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("live_user"))));
  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("provider"))));
  EXPECT_FALSE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("dead_user"))));
  EXPECT_FALSE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("dead_provider"))));
  EXPECT_EQ(liveness.contributed_edge_count, 1u);
}

}  // namespace
}  // namespace loom
