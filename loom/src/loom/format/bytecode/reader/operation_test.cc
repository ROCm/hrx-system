// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/operation.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/ops/module/ops.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class BytecodeOperationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    iree_arena_initialize(&block_pool_, &retained_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_module_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_MODULE, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("operation_test.loombc"), &error_count_, &decoder_);

    strings_[0] = iree_string_view_empty();
    strings_[1] = IREE_SV("module.import");
    module_view_.strings.values = strings_;
    module_view_.strings.count = IREE_ARRAYSIZE(strings_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_reader_section_t MakeSection(const uint8_t* data,
                                             iree_host_size_t length) {
    return loom_bytecode_reader_section_t{
        /*.kind=*/LOOM_BYTECODE_SECTION_OPS,
        /*.flags=*/{},
        /*.offset=*/0,
        /*.length=*/length,
        /*.absolute_offset=*/41,
        /*.bytes=*/iree_make_const_byte_span(data, length),
    };
  }

  // Validated module strings addressed by operation entries.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Source string table containing one registered operation name.
  iree_string_view_t strings_[2];
  // Block source shared by short-lived and retained arenas.
  iree_arena_block_pool_t block_pool_;
  // Storage for dense resolved operation facts.
  iree_arena_allocator_t scratch_arena_;
  // Storage retaining operation names beyond validation.
  iree_arena_allocator_t retained_arena_;
  // Finalized module-dialect registry.
  loom_context_t context_;
};

TEST_F(BytecodeOperationTest, ValidatesAndResolvesDenseTable) {
  const uint8_t data[] = {
      0x01,  // Operation count.
      0x01,  // module.import string ordinal.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(data, sizeof(data));

  IREE_ASSERT_OK(loom_bytecode_operation_table_validate(
      &decoder_, &context_, &module_view_, &scratch_arena_, &section));

  ASSERT_EQ(module_view_.ops.count, 1u);
  ASSERT_NE(module_view_.ops.values, nullptr);
  ASSERT_NE(module_view_.ops.kinds, nullptr);
  EXPECT_NE(module_view_.ops.values[0], nullptr);
  EXPECT_EQ(module_view_.ops.kinds[0], LOOM_OP_MODULE_IMPORT);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeOperationTest, RetainsRegisteredNames) {
  const uint8_t data[] = {
      0x01,  // Operation count.
      0x01,  // module.import string ordinal.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(data, sizeof(data));

  loom_bytecode_op_metadata_t* entries = nullptr;
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(loom_bytecode_operation_table_index(
      &decoder_, &context_, &module_view_, &scratch_arena_, &section,
      &retained_arena_, &entries, &count));

  ASSERT_EQ(count, 1u);
  ASSERT_NE(entries, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(entries[0].name, IREE_SV("module.import")));
  EXPECT_EQ(module_view_.ops.kinds[0], LOOM_OP_MODULE_IMPORT);
  EXPECT_EQ(error_count_, 0u);
}

}  // namespace
}  // namespace loom
