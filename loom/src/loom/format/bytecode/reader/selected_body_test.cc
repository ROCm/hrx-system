// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_body.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class BytecodeSelectedBodyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &table_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_body_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&table_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Number of malformed-input diagnostics accepted by the sink.
  uint32_t error_count_ = 0;
  // Shared source for module and transient arena allocations.
  iree_arena_block_pool_t block_pool_;
  // Scratch storage used by selected table materialization.
  iree_arena_allocator_t table_arena_;
  // Finalized registry context used by the output module.
  loom_context_t context_;
  // Compact selected module under construction.
  loom_module_t* module_ = nullptr;
  // Bounded decoder under test.
  loom_bytecode_reader_decoder_t decoder_ = {};
};

TEST_F(BytecodeSelectedBodyTest, ProjectsHighValueReferencesToCompactIds) {
  constexpr uint32_t kSourceOrdinal = 63;
  std::vector<iree_string_view_t> strings(kSourceOrdinal + 1,
                                          IREE_SV("rejected"));
  strings[kSourceOrdinal] = IREE_SV("selected_value");

  std::vector<uint8_t> bytecode(kSourceOrdinal, 0xFF);
  std::vector<loom_bytecode_table_entry_metadata_t> types(kSourceOrdinal + 1);
  for (uint32_t i = 0; i < kSourceOrdinal; ++i) {
    types[i] = {
        /*.entry_offset=*/i,
        /*.entry_length=*/1,
    };
  }
  types[kSourceOrdinal].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_SCALAR);
  bytecode.push_back(LOOM_SCALAR_TYPE_F32);
  types[kSourceOrdinal].entry_length = 2;

  loom_bytecode_module_metadata_t metadata = {};
  metadata.strings = {strings.size(), strings.data()};
  metadata.types = {types.size(), types.data()};
  loom_bytecode_selected_table_materializer_t tables;
  loom_bytecode_selected_table_materializer_initialize(
      &decoder_, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      &context_, &metadata, &table_arena_, module_, iree_allocator_system(),
      &tables);
  loom_bytecode_selected_body_materializer_t materializer = {
      /*.tables=*/&tables,
      /*.block_pool=*/&block_pool_,
  };

  iree_arena_allocator_t body_arena;
  iree_arena_initialize(&block_pool_, &body_arena);
  loom_value_id_t value_map[1] = {LOOM_VALUE_ID_INVALID};
  loom_bytecode_selected_value_scope_t value_scope;
  IREE_ASSERT_OK(loom_bytecode_selected_value_scope_initialize_fresh(
      &materializer, &body_arena, IREE_SV("@selected"),
      /*payload_offset=*/4096, value_map, IREE_ARRAYSIZE(value_map),
      &value_scope));

  const uint8_t value_bytes[] = {
      kSourceOrdinal,  // Value name STRINGS ordinal.
      kSourceOrdinal,  // Value type TYPES ordinal.
      0x00,            // Dynamic dimension binding count.
      0x00,            // Encoding binding.
  };
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(value_bytes, sizeof(value_bytes),
                                         /*absolute_offset=*/4096,
                                         IREE_SV("IR"), &cursor);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_value_scope_materialize_definition(
      &value_scope, &cursor, &value_id));
  IREE_ASSERT_OK(
      loom_bytecode_reader_expect_empty(&decoder_, &cursor, IREE_SV("IR")));

  ASSERT_EQ(module_->strings.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(module_->strings.entries[1],
                                     IREE_SV("selected_value")));
  ASSERT_EQ(module_->types.count, 1u);
  EXPECT_EQ(loom_type_kind(module_->types.entries[0]), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_element_type(module_->types.entries[0]),
            LOOM_SCALAR_TYPE_F32);
  ASSERT_EQ(module_->values.count, 1u);
  EXPECT_EQ(value_id, 0u);
  EXPECT_EQ(loom_module_value(module_, value_id)->name_id, 1u);
  EXPECT_TRUE(loom_type_equal(loom_module_value(module_, value_id)->type,
                              module_->types.entries[0]));
  EXPECT_EQ(tables.projection.slots.count, 1u);
  EXPECT_EQ(error_count_, 0u);

  iree_arena_deinitialize(&body_arena);
  loom_bytecode_selected_table_materializer_deinitialize(&tables);
}

}  // namespace
}  // namespace loom
