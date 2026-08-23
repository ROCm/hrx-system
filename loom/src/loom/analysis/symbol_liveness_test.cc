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
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct ApplyEdgeTestState {
  loom_symbol_id_t family_symbol_id;
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

static iree_status_t MarkProviderForDemand(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  ApplyEdgeTestState* state = (ApplyEdgeTestState*)user_data;
  if (demand->family_symbol_id != state->family_symbol_id) {
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
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
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
    loom_symbol_reference_table_t references = {};
    IREE_CHECK_OK(loom_symbol_reference_table_build(module, &analysis_arena_,
                                                    &references));
    loom_symbol_liveness_t liveness = {};
    IREE_CHECK_OK(loom_symbol_liveness_compute(module, &references, options,
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

TEST_F(SymbolLivenessTest, ExplicitRootReachesDependencies) {
  ModulePtr module = ParseModule(R"(
func.def @entry() {
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

  const loom_symbol_id_t root_symbol_ids[] = {
      FindSymbol(module.get(), IREE_SV("entry")),
  };
  loom_symbol_liveness_options_t options = {
      /*.flags=*/{},
      /*.root_query=*/{},
      /*.root_query_user_data=*/{},
      /*.contributors=*/{},
      /*.contributor_count=*/{},
      /*.root_symbol_ids=*/
      {
          /*.values=*/root_symbol_ids,
          /*.count=*/IREE_ARRAYSIZE(root_symbol_ids),
      },
  };
  loom_symbol_liveness_t liveness = ComputeLiveness(module.get(), &options);

  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("entry"))));
  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("helper"))));
  EXPECT_FALSE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("dead"))));
}

TEST_F(SymbolLivenessTest, ModuleAvailabilityDoesNotRootSymbols) {
  ModulePtr module = ParseModule(R"(
test.record @available
test.template_param_symbol<@available>

func.def public @entry() {
  func.return
}
)");

  loom_symbol_liveness_options_t options = {
      /*.flags=*/LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      /*.root_query=*/RootPublicFunc,
  };
  loom_symbol_liveness_t liveness = ComputeLiveness(module.get(), &options);

  EXPECT_TRUE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("entry"))));
  EXPECT_FALSE(loom_symbol_liveness_is_live(
      &liveness, FindSymbol(module.get(), IREE_SV("available"))));
  EXPECT_EQ(liveness.concrete_edge_count, 0u);
}

TEST_F(SymbolLivenessTest, ContributorsScanReachableBodiesOnly) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.contract(%arg0: i32) -> (i32)
template.decl @dead.contract(%arg0: i32) -> (i32)

template.def<@demo.contract> @provider(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}

template.def<@dead.contract> @dead_provider(%arg0: i32) -> (i32) {
  template.return %arg0 : i32
}

func.def public @entry(%arg0: i32) -> (i32) {
  %result = func.call @live_user(%arg0) : (i32) -> (i32)
  func.return %result : i32
}

func.def @live_user(%arg0: i32) -> (i32) {
  %result = template.apply<@demo.contract>(%arg0) : (i32) -> (i32)
  func.return %result : i32
}

func.def @dead_user(%arg0: i32) -> (i32) {
  %result = template.apply<@dead.contract>(%arg0) : (i32) -> (i32)
  func.return %result : i32
}
)");

  ApplyEdgeTestState apply_state = {
      /*.family_symbol_id=*/
      FindSymbol(module.get(), IREE_SV("demo.contract")),
      /*.provider_symbol_id=*/FindSymbol(module.get(), IREE_SV("provider")),
  };
  loom_symbol_liveness_contributor_t contributor = {
      /*.visit_template_demand=*/MarkProviderForDemand,
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
