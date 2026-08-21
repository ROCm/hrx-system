// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/module_projection.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class ModuleProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC, vtables,
                                      static_cast<uint16_t>(vtable_count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(const char* source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("module_projection_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return module;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_CHECK_OK(loom_text_print_module_to_builder(module, &builder,
                                                    LOOM_TEXT_PRINT_DEFAULT));
    std::string text(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return text;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t scratch_arena_;
};

TEST_F(ModuleProjectionTest, ClonesThroughDirectSymbolAndValueCorrespondence) {
  loom_module_t* source = Parse(R"(
func.def public @entry(%x: i32) -> (i32) {
  // Preserve authored grouping with the cloned operation.
  %dead = func.call @helper(%x) : (i32) -> (i32)
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  ASSERT_NE(source, nullptr);

  loom_op_t* entry_function = source->symbols.entries[0].defining_op;
  ASSERT_TRUE(loom_func_def_isa(entry_function));
  loom_block_t* entry_block =
      loom_region_entry_block(loom_func_def_body(entry_function));
  loom_op_t* dead_call = entry_block->first_op;
  ASSERT_TRUE(loom_func_call_isa(dead_call));
  const loom_value_id_t dead_value =
      loom_func_call_results(dead_call).values[0];
  IREE_ASSERT_OK(loom_op_erase(source, dead_call));

  const loom_module_size_hints_t hints = {
      /*.value_count=*/0,
      /*.string_count=*/source->strings.count,
      /*.type_count=*/source->types.count,
      /*.encoding_count=*/source->encodings.count,
      /*.symbol_count=*/source->symbols.count,
  };
  loom_module_t* target = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(
      &context_, source->strings.entries[source->name_id], &block_pool_, &hints,
      iree_allocator_system(), &target));
  ASSERT_NE(target, nullptr);

  std::vector<loom_symbol_ref_t> target_symbols(source->symbols.count);
  for (loom_symbol_id_t source_symbol_id = 0;
       source_symbol_id < source->symbols.count; ++source_symbol_id) {
    const loom_symbol_t* source_symbol =
        &source->symbols.entries[source_symbol_id];
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        target, source->strings.entries[source_symbol->name_id],
        &target_name_id));
    loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_add_symbol(target, target_name_id, &target_symbol_id));
    target->symbols.entries[target_symbol_id].flags = source_symbol->flags;
    target_symbols[source_symbol_id] = {
        /*.module_id=*/0,
        /*.symbol_id=*/target_symbol_id,
    };
  }

  loom_ir_module_projection_t projection = {};
  IREE_ASSERT_OK(loom_ir_module_projection_initialize(
      source, target, target_symbols.data(), target_symbols.size(),
      &projection));
  IREE_ASSERT_OK(loom_ir_module_projection_clone(&projection, &scratch_arena_));
  for (loom_value_id_t source_value_id = 0;
       source_value_id < source->values.count; ++source_value_id) {
    loom_value_id_t target_value_id = LOOM_VALUE_ID_INVALID;
    if (source_value_id == dead_value) {
      EXPECT_FALSE(loom_ir_module_projection_try_target_value(
          &projection, source_value_id, &target_value_id));
      EXPECT_EQ(target_value_id, LOOM_VALUE_ID_INVALID);
      continue;
    }
    ASSERT_TRUE(loom_ir_module_projection_try_target_value(
        &projection, source_value_id, &target_value_id));
    EXPECT_LT(target_value_id, target->values.count);
  }
  EXPECT_EQ(target->values.count + 1, source->values.count);
  EXPECT_NE(projection.remap.target_values_by_source, nullptr);
  EXPECT_EQ(projection.remap.value_map_entries, nullptr);
  EXPECT_EQ(projection.remap.value_map_entry_capacity, 0u);
  ASSERT_EQ(target->symbols.count, source->symbols.count);
  for (loom_symbol_id_t source_symbol_id = 0;
       source_symbol_id < source->symbols.count; ++source_symbol_id) {
    const loom_symbol_ref_t target_ref =
        loom_ir_module_projection_target_symbol(&projection, source_symbol_id);
    ASSERT_EQ(target_ref.symbol_id, source_symbol_id);
    EXPECT_NE(target->symbols.entries[target_ref.symbol_id].defining_op,
              source->symbols.entries[source_symbol_id].defining_op);
  }

  const loom_verify_options_t verify_options = {};
  loom_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_verify_module(target, &verify_options, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_EQ(Print(target), Print(source));

  loom_module_free(target);
  loom_module_free(source);
}

}  // namespace
}  // namespace loom
