// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/program_clone.h"

#include <initializer_list>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/function_requirements.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

namespace loom::aie2p::testing {
namespace {

class ProgramCloneTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("private"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &target_));
    IREE_ASSERT_OK(loom_module_intern_string(
        source_, IREE_SV("amd.xdna.aie2p.core"), &contract_));
  }

  void TearDown() override {
    loom_module_free(target_);
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* Function(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(source_, name, &name_id));
    loom_symbol_ref_t symbol = {.module_id = 0};
    IREE_CHECK_OK(loom_module_add_symbol(source_, name_id, &symbol.symbol_id));
    loom_builder_t builder;
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &builder);
    loom_op_t* function = nullptr;
    IREE_CHECK_OK(loom_low_func_def_build(
        &builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, contract_, loom_symbol_ref_null(),
        0, loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, nullptr,
        0, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function));
    return function;
  }

  loom_builder_t Body(loom_op_t* function) {
    loom_builder_t builder;
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &builder);
    loom_builder_enter_region(&builder, function,
                              loom_low_func_def_body(function));
    return builder;
  }

  void Call(loom_builder_t* builder, const loom_op_t* callee) {
    loom_op_t* call = nullptr;
    IREE_ASSERT_OK(loom_low_func_call_build(
        builder, 0, 0, 0, 0, loom_low_func_def_callee(callee), nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &call));
  }

  void Return(loom_builder_t* builder) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(
        loom_low_return_build(builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_aie2p_program_source_function_t Inventory(loom_op_t* function) {
    loom_aie2p_program_source_function_t source = {
        .symbol = loom_low_func_def_callee(function),
        .function_op = function,
    };
    IREE_CHECK_OK(loom_low_function_requirements_build(
        source_, loom_aie2p_core_descriptor_set(),
        loom_low_func_def_body(function), &arena_, &source.requirements));
    return source;
  }

  // Shared blocks support modules and retained compiler products.
  iree_arena_block_pool_t pool_;
  // Owns the source inventories and private projection records.
  iree_arena_allocator_t arena_;
  // Low operation vocabulary shared by both modules.
  loom_context_t context_;
  // Immutable source during projection; may be released before checking output.
  loom_module_t* source_ = nullptr;
  // Independent module owning every projected call.
  loom_module_t* target_ = nullptr;
  // Source representation contract for the empty-signature core functions.
  loom_string_id_t contract_ = LOOM_STRING_ID_INVALID;
};

TEST_F(ProgramCloneTest, RetainsRepeatedCallsAcrossBlocksAndBufferReuse) {
  auto* first = Function(IREE_SV("first"));
  auto first_body = Body(first);
  Return(&first_body);
  auto* second = Function(IREE_SV("second"));
  auto second_body = Body(second);
  Call(&second_body, first);
  Return(&second_body);
  auto* entry = Function(IREE_SV("entry"));
  auto entry_body = Body(entry);
  Call(&entry_body, first);
  Call(&entry_body, second);
  loom_block_t* finish = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(
      source_, loom_low_func_def_body(entry), &finish));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&entry_body, finish, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&entry_body, finish);
  Call(&entry_body, first);
  Return(&entry_body);
  const auto first_source = Inventory(first);
  const auto second_source = Inventory(second);
  const auto entry_source = Inventory(entry);

  loom_aie2p_program_clone_t clone;
  IREE_ASSERT_OK(
      loom_aie2p_program_clone_initialize(source_, target_, &arena_, &clone));
  // Reserve in a different order so correspondence cannot rely on symbol IDs.
  for (const auto* source : {&entry_source, &second_source, &first_source}) {
    IREE_ASSERT_OK(loom_aie2p_program_clone_reserve_function(&clone, source));
  }
  IREE_ASSERT_OK(loom_aie2p_program_clone_function(&clone, &entry_source));
  const auto* projected =
      clone.functions_by_source[entry_source.symbol.symbol_id];
  ASSERT_EQ(projected->call_boundaries.count, 3u);
  const auto* projection_buffer = clone.call_projection;
  IREE_ASSERT_OK(loom_aie2p_program_clone_function(&clone, &second_source));
  EXPECT_EQ(clone.call_projection, projection_buffer);
  IREE_ASSERT_OK(loom_aie2p_program_clone_function(&clone, &first_source));
  const auto* empty = clone.functions_by_source[first_source.symbol.symbol_id];
  EXPECT_EQ(empty->call_boundaries.count, 0u);
  EXPECT_EQ(empty->call_boundaries.calls, nullptr);
  const size_t allocation_size = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_aie2p_program_clone_function(&clone, &entry_source));
  EXPECT_EQ(arena_.used_allocation_size, allocation_size);

  const auto first_target =
      clone.symbols_by_source[first_source.symbol.symbol_id];
  const auto second_target =
      clone.symbols_by_source[second_source.symbol.symbol_id];
  EXPECT_NE(first_target.symbol_id, first_source.symbol.symbol_id);
  for (size_t i = 0; i < projected->call_boundaries.count; ++i) {
    EXPECT_NE(projected->call_boundaries.calls[i],
              entry_source.requirements.call_boundaries.calls[i]);
  }
  loom_module_free(source_);
  source_ = nullptr;

  const auto* calls = projected->call_boundaries.calls;
  const loom_symbol_ref_t expected[] = {first_target, second_target,
                                        first_target};
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    ASSERT_TRUE(loom_low_func_call_isa(calls[i]));
    EXPECT_EQ(loom_low_func_call_callee(calls[i]).symbol_id,
              expected[i].symbol_id);
  }
  EXPECT_EQ(calls[0]->parent_block, calls[1]->parent_block);
  EXPECT_NE(calls[1]->parent_block, calls[2]->parent_block);
  ASSERT_EQ(projected->direct_callees.count, 2u);
  EXPECT_EQ(projected->direct_callees.values[0].symbol_id,
            first_target.symbol_id);
  EXPECT_EQ(projected->direct_callees.values[1].symbol_id,
            second_target.symbol_id);
}

}  // namespace
}  // namespace loom::aie2p::testing
