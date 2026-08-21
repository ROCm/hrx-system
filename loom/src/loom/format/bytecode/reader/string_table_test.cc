// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/string_table.h"

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

class BytecodeStringTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    iree_arena_initialize(&block_pool_, &storage_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("string_table_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&storage_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_reader_section_t MakeSection(uint16_t kind, const uint8_t* data,
                                             iree_host_size_t length) {
    return loom_bytecode_reader_section_t{
        /*.kind=*/kind,
        /*.flags=*/{},
        /*.offset=*/0,
        /*.length=*/length,
        /*.absolute_offset=*/41,
        /*.bytes=*/iree_make_const_byte_span(data, length),
    };
  }

  // Validated table facts retained by the storage arena.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Block source shared by scratch, retained, and module arenas.
  iree_arena_block_pool_t block_pool_;
  // Temporary uniqueness-validation storage.
  iree_arena_allocator_t scratch_arena_;
  // Storage retaining borrowed table views.
  iree_arena_allocator_t storage_arena_;
  // Finalized context for the output module.
  loom_context_t context_;
  // Module receiving string and source projections.
  loom_module_t* module_ = nullptr;
};

TEST_F(BytecodeStringTableTest, ReadsCanonicalStringsAndReclaimsScratch) {
  const uint8_t data[] = {
      0x03,  // String count.
      0x00,  // Empty string zero.
      0x05,  // String one length.
      'a',  'l',  'p', 'h', 'a',
      0x02,        // String two length.
      0xCE, 0xB2,  // UTF-8 beta.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));
  const iree_host_size_t scratch_size_before =
      scratch_arena_.used_allocation_size;

  IREE_ASSERT_OK(loom_bytecode_string_table_read(
      &decoder_, &section, &scratch_arena_, &storage_arena_, &module_view_));

  ASSERT_EQ(module_view_.strings.count, 3u);
  EXPECT_TRUE(iree_string_view_is_empty(module_view_.strings.values[0]));
  EXPECT_TRUE(
      iree_string_view_equal(module_view_.strings.values[1], IREE_SV("alpha")));
  EXPECT_EQ(module_view_.strings.values[2].size, 2u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, scratch_size_before);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeStringTableTest, RejectsMissingEmptyStringZero) {
  const uint8_t data[] = {0x00};
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_string_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeStringTableTest, RejectsNonemptyStringZero) {
  const uint8_t data[] = {0x01, 0x01, 'x'};
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_string_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeStringTableTest, RejectsDuplicateStrings) {
  const uint8_t data[] = {
      0x03,       // String count.
      0x00,       // Empty string zero.
      0x01, 'x',  // String one.
      0x01, 'x',  // Duplicate string two.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_string_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
}

TEST_F(BytecodeStringTableTest, RejectsInvalidUtf8) {
  const uint8_t data[] = {0x02, 0x00, 0x01, 0xFF};
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_string_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeStringTableTest, MaterializesCanonicalStringIds) {
  const uint8_t data[] = {
      0x03,  // String count.
      0x00,  // Empty string zero.
      0x05,  // String one length.
      'a',  'l', 'p', 'h', 'a',
      0x04,  // String two length.
      'b',  'e', 't', 'a',
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_STRINGS, data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_string_table_read(
      &decoder_, &section, &scratch_arena_, &storage_arena_, &module_view_));

  IREE_ASSERT_OK(
      loom_bytecode_string_table_materialize(&module_view_, module_));

  ASSERT_EQ(module_->strings.count, 3u);
  EXPECT_TRUE(
      iree_string_view_equal(module_->strings.entries[1], IREE_SV("alpha")));
  EXPECT_TRUE(
      iree_string_view_equal(module_->strings.entries[2], IREE_SV("beta")));
}

TEST_F(BytecodeStringTableTest, RejectsDuplicateSources) {
  const uint8_t data[] = {
      0x02,       // Source count.
      0x01, 'a',  // Source zero.
      0x01, 'a',  // Duplicate source one.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_SOURCES, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_source_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
}

TEST_F(BytecodeStringTableTest, MaterializesCanonicalSourceIds) {
  const uint8_t data[] = {
      0x02,       // Source count.
      0x01, 'a',  // Source zero.
      0x01, 'b',  // Source one.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_SOURCES, data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_source_table_read(
      &decoder_, &section, &scratch_arena_, &storage_arena_, &module_view_));

  IREE_ASSERT_OK(
      loom_bytecode_source_table_materialize(&module_view_, module_));

  ASSERT_EQ(module_->sources.count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(module_->sources.entries[0], IREE_SV("a")));
  EXPECT_TRUE(
      iree_string_view_equal(module_->sources.entries[1], IREE_SV("b")));
}

TEST_F(BytecodeStringTableTest, RejectsSourceCountOutsideRuntimeIdSpace) {
  const uint8_t data[] = {0x80, 0x80, 0x04};  // 65536 sources.
  const loom_bytecode_reader_section_t section =
      MakeSection(LOOM_BYTECODE_SECTION_SOURCES, data, sizeof(data));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_source_table_read(&decoder_, &section, &scratch_arena_,
                                      &storage_arena_, &module_view_));

  EXPECT_EQ(error_count_, 1u);
}

}  // namespace
}  // namespace loom
