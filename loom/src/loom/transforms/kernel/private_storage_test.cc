// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/private_storage.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class PrivateStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    iree_host_size_t count = 0;
    const auto* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("storage"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_pass_value_fact_owner_initialize(&pool_, &facts_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_private_storage_plan_release(&plan_);
    loom_pass_value_fact_owner_deinitialize(&facts_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Constant(int64_t value, loom_scalar_type_t type) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(&builder_, loom_attr_i64(value),
                                            loom_type_scalar(type),
                                            LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  // Independent read-only loops exercise construction space without requiring
  // a large kernel or any target lowering. All cells have one reaching value.
  void CreateReadOnlyLoops(int64_t cell_count, int loop_count) {
    loom_string_id_t name;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("read_only"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t types[] = {loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                 loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
    IREE_ASSERT_OK(loom_func_def_build(
        &builder_, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), {0, symbol},
        types, 2, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function_));
    auto* body = loom_func_def_body(function_);
    loom_builder_enter_region(&builder_, function_, body);
    const auto initial = loom_region_entry_arg_id(body, 0);
    const auto limit = loom_region_entry_arg_id(body, 1);
    const auto bytes = Constant(cell_count * 4, LOOM_SCALAR_TYPE_OFFSET);
    const auto origin = Constant(0, LOOM_SCALAR_TYPE_OFFSET);
    const auto begin = Constant(0, LOOM_SCALAR_TYPE_INDEX);
    const auto step = Constant(1, LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* allocation = nullptr;
    IREE_ASSERT_OK(loom_buffer_alloca_build(
        &builder_, 0, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
        LOOM_VALUE_ID_INVALID, 4, bytes, loom_type_buffer(),
        LOOM_LOCATION_UNKNOWN, &allocation));
    loom_op_t* view = nullptr;
    IREE_ASSERT_OK(loom_buffer_view_build(
        &builder_, loom_buffer_alloca_result(allocation), origin,
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                            loom_dim_pack_static(cell_count), 0),
        LOOM_LOCATION_UNKNOWN, &view));
    const auto storage = loom_buffer_view_result(view);
    loom_op_t* op = nullptr;
    for (int64_t cell = 0; cell < cell_count; ++cell) {
      IREE_ASSERT_OK(loom_view_store_build(&builder_, 0, 0, initial, storage,
                                           nullptr, 0, &cell, 1, 0, 0,
                                           LOOM_LOCATION_UNKNOWN, &op));
    }
    for (int i = 0; i < loop_count; ++i) {
      loom_op_t* loop = nullptr;
      IREE_ASSERT_OK(loom_scf_for_build(
          &builder_, 0, begin, limit, step, nullptr, 0, nullptr, nullptr, 0,
          LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
          LOOM_LOCATION_UNKNOWN, &loop));
      const auto saved =
          loom_builder_enter_region(&builder_, loop, loom_scf_for_body(loop));
      const int64_t cell = i % cell_count;
      IREE_ASSERT_OK(loom_view_load_build(&builder_, 0, 0, storage, nullptr, 0,
                                          &cell, 1, 0, 0, types[0],
                                          LOOM_LOCATION_UNKNOWN, &op));
      const auto value = loom_view_load_result(op);
      IREE_ASSERT_OK(loom_test_use_build(&builder_, &value, 1,
                                         LOOM_LOCATION_UNKNOWN, &op));
      IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_restore(&builder_, saved);
    }
    IREE_ASSERT_OK(loom_func_return_build(&builder_, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &op));
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  // Shared pool, with separate module, fact, and plan lifetimes.
  iree_arena_block_pool_t pool_;
  // Retained promotion plan storage.
  iree_arena_allocator_t arena_;
  // Registered production operation semantics.
  loom_context_t context_;
  // Verified IR supplied to the planner.
  loom_module_t* module_ = nullptr;
  // Builder positioned in the current function or loop body.
  loom_builder_t builder_;
  // Function whose cells are analyzed.
  loom_op_t* function_ = nullptr;
  // Function-scoped facts shared with planning.
  loom_pass_value_fact_owner_t facts_;
  // Result whose local value domain is released at teardown.
  loom_private_storage_plan_t plan_ = {};
};

TEST_F(PrivateStorageTest, ReadOnlyLoopsDoNotRetainCellCrossProduct) {
  ASSERT_NO_FATAL_FAILURE(CreateReadOnlyLoops(128, 256));
  loom_pass_t pass = {.arena = &arena_, .value_facts = &facts_};
  IREE_ASSERT_OK(loom_private_storage_plan_build(
      &pass, module_, loom_func_like_cast(module_, function_), &plan_));
  ASSERT_EQ(plan_.cell_count, 128u);
  ASSERT_NE(plan_.allocations, nullptr);
  EXPECT_TRUE(plan_.allocations->selected);
  iree_host_size_t transport_count = 0;
  for (auto* signature = plan_.signatures; signature;
       signature = signature->next) {
    for (auto* transport = signature->transports; transport;
         transport = transport->next) {
      ++transport_count;
    }
  }
  EXPECT_EQ(transport_count, 0u);
  for (auto* access = plan_.accesses; access; access = access->next) {
    ASSERT_NE(access->value, nullptr);
    EXPECT_EQ(loom_private_storage_value_resolve(access->value)->source,
              loom_region_entry_arg_id(loom_func_def_body(function_), 0));
  }
  // The input has only 128 cells and 256 independent control scopes. Even
  // retaining one full state copy per loop exceeds this entire-plan budget.
  EXPECT_LT(arena_.total_allocation_size, 256u * 1024u);
}

}  // namespace
}  // namespace loom
