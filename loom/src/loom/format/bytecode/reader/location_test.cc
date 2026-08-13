// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/location.h"

#include <cstring>

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

class BytecodeLocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &retained_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("location_test.loombc"), &error_count_, &decoder_);

    source_names_[0] = IREE_SV("model.loom");
    module_view_.sources.values = source_names_;
    module_view_.sources.count = IREE_ARRAYSIZE(source_names_);
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_append_source(module_, source_names_[0], &source_id));
    IREE_ASSERT_EQ(source_id, 0u);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_reader_section_t MakeSection(const uint8_t* data,
                                             iree_host_size_t length) {
    return loom_bytecode_reader_section_t{
        /*.kind=*/LOOM_BYTECODE_SECTION_LOCATIONS,
        /*.flags=*/{},
        /*.offset=*/0,
        /*.length=*/length,
        /*.absolute_offset=*/41,
        /*.bytes=*/iree_make_const_byte_span(data, length),
    };
  }

  // Canonical table covering every recursive location variant.
  static constexpr uint8_t kLocationTable[] = {
      0x05,                // Location count.
      LOOM_LOCATION_NONE,  // Location 0: unknown sentinel.
      0x00,                // Flags.
      LOOM_LOCATION_FILE,  // Location 1: source range.
      0x00,                // Flags.
      0x00,                // Source ordinal.
      0x01,
      0x02,
      0x03,
      0x04,                 // Source coordinates.
      LOOM_LOCATION_FUSED,  // Location 2: one prior child.
      0x00,                 // Flags.
      0x01,
      0x01,                  // Child count and location 1.
      LOOM_LOCATION_OPAQUE,  // Location 3: source-owned bytes.
      0x00,                  // Flags.
      0x00,                  // Source ordinal.
      0x03,
      'a',
      'b',
      'c',                   // Payload.
      LOOM_LOCATION_TAGGED,  // Location 4: tagged prior child.
      0x00,                  // Flags.
      0x01,
      0x02,  // Tag and location 2.
      0x02,
      'd',
      'e',  // Payload.
  };

  // Validated module facts referenced by location entries.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Source table addressed by file and opaque locations.
  iree_string_view_t source_names_[1];
  // Block source shared by retained and output-module arenas.
  iree_arena_block_pool_t block_pool_;
  // Storage retaining exact entry metadata beyond validation.
  iree_arena_allocator_t retained_arena_;
  // Finalized registry context for the output module.
  loom_context_t context_;
  // Module receiving canonical location instances.
  loom_module_t* module_ = nullptr;
};

constexpr uint8_t BytecodeLocationTest::kLocationTable[];

TEST_F(BytecodeLocationTest, ValidatesWithoutRetainingEntries) {
  const loom_bytecode_reader_section_t section =
      MakeSection(kLocationTable, sizeof(kLocationTable));

  IREE_ASSERT_OK(loom_bytecode_location_table_validate(&decoder_, &module_view_,
                                                       &section));

  EXPECT_EQ(module_view_.locations.count, 5u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeLocationTest, IndexesExactEntryRanges) {
  const loom_bytecode_reader_section_t section =
      MakeSection(kLocationTable, sizeof(kLocationTable));

  loom_bytecode_table_entry_metadata_t* entries = nullptr;
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(loom_bytecode_location_table_index(
      &decoder_, &module_view_, &section, &retained_arena_, &entries, &count));

  ASSERT_EQ(count, 5u);
  ASSERT_NE(entries, nullptr);
  EXPECT_EQ(entries[0].entry_offset, 42u);
  EXPECT_EQ(entries[0].entry_length, 2u);
  EXPECT_EQ(entries[1].entry_offset, 44u);
  EXPECT_EQ(entries[1].entry_length, 7u);
  EXPECT_EQ(entries[2].entry_offset, 51u);
  EXPECT_EQ(entries[2].entry_length, 4u);
  EXPECT_EQ(entries[3].entry_offset, 55u);
  EXPECT_EQ(entries[3].entry_length, 7u);
  EXPECT_EQ(entries[4].entry_offset, 62u);
  EXPECT_EQ(entries[4].entry_length, 7u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeLocationTest, RejectsCoordinateOutsideRuntimeWidth) {
  const uint8_t data[] = {
      0x02,                // Location count.
      LOOM_LOCATION_NONE,  // Location 0: unknown sentinel.
      0x00,                // Flags.
      LOOM_LOCATION_FILE,  // Location 1: source range.
      0x00,                // Flags.
      0x00,                // Source ordinal.
      0x80,
      0x80,
      0x04,  // Start line 65536.
      0x00,
      0x00,
      0x00,  // Remaining coordinates.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(data, sizeof(data));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        loom_bytecode_location_table_validate(
                            &decoder_, &module_view_, &section));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeLocationTest, MaterializesCanonicalTable) {
  const loom_bytecode_reader_section_t section =
      MakeSection(kLocationTable, sizeof(kLocationTable));
  module_view_.locations.count = 5;
  loom_bytecode_location_materializer_t materializer = {
      /*.decoder=*/&decoder_,
      /*.module_view=*/&module_view_,
      /*.output_module=*/module_,
  };

  IREE_ASSERT_OK(
      loom_bytecode_location_table_materialize(&materializer, &section));

  ASSERT_EQ(module_->locations.count, 5u);
  const loom_location_entry_t& file = module_->locations.entries[1];
  EXPECT_EQ(file.kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(file.file.source_id, 0u);
  EXPECT_EQ(file.file.start_line, 1u);
  EXPECT_EQ(file.file.start_col, 2u);
  EXPECT_EQ(file.file.end_line, 3u);
  EXPECT_EQ(file.file.end_col, 4u);

  const loom_location_entry_t& fused = module_->locations.entries[2];
  EXPECT_EQ(fused.kind, LOOM_LOCATION_FUSED);
  ASSERT_EQ(fused.fused.count, 1u);
  EXPECT_EQ(fused.fused.children[0], 1u);

  const loom_location_entry_t& opaque = module_->locations.entries[3];
  EXPECT_EQ(opaque.kind, LOOM_LOCATION_OPAQUE);
  EXPECT_EQ(opaque.opaque.source_id, 0u);
  ASSERT_EQ(opaque.opaque.data_length, 3u);
  EXPECT_EQ(std::memcmp(opaque.opaque.data, "abc", 3), 0);

  const loom_location_entry_t& tagged = module_->locations.entries[4];
  EXPECT_EQ(tagged.kind, LOOM_LOCATION_TAGGED);
  EXPECT_EQ(tagged.tagged.tag, 1u);
  EXPECT_EQ(tagged.tagged.child, 2u);
  ASSERT_EQ(tagged.tagged.data_length, 2u);
  EXPECT_EQ(std::memcmp(tagged.tagged.data, "de", 2), 0);
  EXPECT_EQ(error_count_, 0u);
}

}  // namespace
}  // namespace loom
