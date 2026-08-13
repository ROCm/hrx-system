// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_tables.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kEncodingParameters[] = {
    {
        /*.name=*/LOOM_BSTRING_REF(5, "block"),
        /*.attr_kind=*/LOOM_ATTR_I64,
        /*.flags=*/LOOM_ATTR_OPTIONAL,
    },
    {
        /*.name=*/LOOM_BSTRING_REF(4, "base"),
        /*.attr_kind=*/LOOM_ATTR_ENCODING,
        /*.flags=*/LOOM_ATTR_OPTIONAL,
    },
};
static const loom_encoding_family_descriptor_t kEncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(4, "q8_0"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/IREE_ARRAYSIZE(kEncodingParameters),
    /*.parameter_descriptors=*/kEncodingParameters,
};
static const loom_encoding_vtable_t kEncodingVtable = {
    /*.descriptor=*/&kEncodingDescriptor,
};

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

static void AppendUVarint(uint64_t value, std::vector<uint8_t>* bytes) {
  do {
    uint8_t byte = value & 0x7Fu;
    value >>= 7;
    if (value != 0) {
      byte |= 0x80u;
    }
    bytes->push_back(byte);
  } while (value != 0);
}

class BytecodeSelectedTablesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_tables_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_selected_table_materializer_t MakeMaterializer(
      const std::vector<uint8_t>& bytecode,
      const loom_bytecode_module_metadata_t* metadata) {
    loom_bytecode_selected_table_materializer_t materializer;
    loom_bytecode_selected_table_materializer_initialize(
        &decoder_, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        &context_, metadata, &scratch_arena_, module_, iree_allocator_system(),
        &materializer);
    return materializer;
  }

  uint32_t error_count_ = 0;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_bytecode_reader_decoder_t decoder_ = {};
};

TEST_F(BytecodeSelectedTablesTest, MaterializesOnlyReachedMixedTableFacts) {
  iree_string_view_t strings[] = {
      iree_string_view_empty(),
      IREE_SV("q8_0"),
      IREE_SV("block"),
      IREE_SV("base"),
  };
  iree_string_view_t sources[] = {IREE_SV("model.loom")};
  std::vector<uint8_t> bytecode;

  loom_bytecode_encoding_metadata_t encodings[2] = {};
  encodings[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {
                      0x00,  // Family ordinal.
                      0x00,  // No alias.
                      0x01,  // Parameter count.
                      0x02,  // "block".
                      LOOM_BYTECODE_ATTR_I64,
                      0x0E,  // Signed value 7.
                  });
  encodings[0].entry_length = bytecode.size() - encodings[0].entry_offset;
  encodings[0].name_string_index = 1;
  encodings[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {
                      0x00,  // Family ordinal.
                      0x00,  // No alias.
                      0x01,  // Parameter count.
                      0x03,  // "base".
                      LOOM_BYTECODE_ATTR_ENCODING,
                      0x01,  // Prior encoding ID.
                  });
  encodings[1].entry_length = bytecode.size() - encodings[1].entry_offset;
  encodings[1].name_string_index = 1;

  loom_bytecode_table_entry_metadata_t types[2] = {};
  types[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(),
                  {LOOM_BYTECODE_TYPE_SCALAR, LOOM_SCALAR_TYPE_F32});
  types[0].entry_length = bytecode.size() - types[0].entry_offset;
  types[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {
                                      LOOM_BYTECODE_TYPE_TENSOR,
                                      LOOM_SCALAR_TYPE_F32,
                                      0x01,  // Rank.
                                      LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC,
                                      0x02,  // Encoding ID.
                                      0x00,  // Static dimension.
                                      0x04,  // Dimension size.
                                  });
  types[1].entry_length = bytecode.size() - types[1].entry_offset;

  loom_bytecode_table_entry_metadata_t locations[2] = {};
  locations[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  locations[0].entry_length = bytecode.size() - locations[0].entry_offset;
  locations[1].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {
                                      LOOM_LOCATION_FILE,
                                      0x00,  // Flags.
                                      0x00,  // Source ordinal.
                                      0x01,
                                      0x02,
                                      0x03,
                                      0x04,  // Coordinates.
                                  });
  locations[1].entry_length = bytecode.size() - locations[1].entry_offset;

  loom_bytecode_module_metadata_t metadata = {};
  metadata.strings = {IREE_ARRAYSIZE(strings), strings};
  metadata.sources = {IREE_ARRAYSIZE(sources), sources};
  metadata.types = {IREE_ARRAYSIZE(types), types};
  metadata.encodings = {IREE_ARRAYSIZE(encodings), encodings};
  metadata.locations = {IREE_ARRAYSIZE(locations), locations};
  loom_bytecode_selected_table_materializer_t materializer =
      MakeMaterializer(bytecode, &metadata);

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/1, &target_type_id));
  EXPECT_EQ(target_type_id, 0u);
  ASSERT_EQ(module_->types.count, 1u);
  EXPECT_EQ(loom_type_kind(module_->types.entries[0]), LOOM_TYPE_TENSOR);
  EXPECT_EQ(module_->types.entries[0].encoding_id, 2u);
  ASSERT_EQ(module_->encodings.count, 2u);
  ASSERT_EQ(module_->encodings.entries[0].attribute_count, 1u);
  EXPECT_EQ(module_->encodings.entries[0].attributes[0].value.i64, 7);
  ASSERT_EQ(module_->encodings.entries[1].attribute_count, 1u);
  EXPECT_EQ(module_->encodings.entries[1].attributes[0].value.encoding_id, 1u);

  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
      &materializer, /*source_location_id=*/1, &target_location_id));
  EXPECT_EQ(target_location_id, 1u);
  ASSERT_EQ(module_->locations.count, 2u);
  EXPECT_EQ(module_->locations.entries[1].file.source_id, 0u);
  ASSERT_EQ(module_->sources.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_->sources.entries[0], sources[0]));

  EXPECT_EQ(materializer.projection.slots.count, 5u);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, MaterializesDeepTypeChainIteratively) {
  constexpr uint32_t kTypeCount = 4096;
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(kTypeCount);
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < kTypeCount; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
    AppendUVarint(/*argument_count=*/1, &bytecode);
    AppendUVarint(/*result_count=*/0, &bytecode);
    AppendUVarint(/*prior_type_id=*/i - 1, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {};
  metadata.types = {entries.size(), entries.data()};
  loom_bytecode_selected_table_materializer_t materializer =
      MakeMaterializer(bytecode, &metadata);

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, kTypeCount - 1, &target_type_id));
  EXPECT_EQ(target_type_id, kTypeCount - 1);
  EXPECT_EQ(module_->types.count, kTypeCount);
  EXPECT_EQ(materializer.projection.slots.count, kTypeCount);
  EXPECT_GE(materializer.worklist.capacity, kTypeCount);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, MaterializesWideTypeReferencesInOneRetry) {
  constexpr uint32_t kArgumentCount = 4096;
  std::vector<uint8_t> bytecode;
  loom_bytecode_table_entry_metadata_t entries[2] = {};
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  entries[1].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
  AppendUVarint(kArgumentCount, &bytecode);
  AppendUVarint(/*result_count=*/0, &bytecode);
  for (uint32_t i = 0; i < kArgumentCount; ++i) {
    AppendUVarint(/*type_id=*/0, &bytecode);
  }
  entries[1].entry_length = bytecode.size() - entries[1].entry_offset;
  loom_bytecode_module_metadata_t metadata = {};
  metadata.types = {IREE_ARRAYSIZE(entries), entries};
  loom_bytecode_selected_table_materializer_t materializer =
      MakeMaterializer(bytecode, &metadata);

  loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_type(
      &materializer, /*source_type_id=*/1, &target_type_id));
  EXPECT_EQ(target_type_id, 1u);
  ASSERT_EQ(module_->types.count, 2u);
  EXPECT_EQ(loom_type_func_arg_count(module_->types.entries[1]),
            kArgumentCount);
  EXPECT_EQ(materializer.projection.slots.count, 2u);
  EXPECT_GE(materializer.worklist.capacity, kArgumentCount);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

TEST_F(BytecodeSelectedTablesTest, MaterializesDeepLocationChainIteratively) {
  constexpr uint32_t kLocationCount = 4096;
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(kLocationCount);
  entries[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < kLocationCount; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.insert(bytecode.end(), {
                                        LOOM_LOCATION_TAGGED,
                                        0x00,  // Flags.
                                        0x01,  // Tag.
                                    });
    AppendUVarint(/*prior_location_id=*/i - 1, &bytecode);
    AppendUVarint(/*data_length=*/0, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {};
  metadata.locations = {entries.size(), entries.data()};
  loom_bytecode_selected_table_materializer_t materializer =
      MakeMaterializer(bytecode, &metadata);

  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_bytecode_selected_table_materialize_location(
      &materializer, kLocationCount - 1, &target_location_id));
  EXPECT_EQ(target_location_id, kLocationCount - 1);
  EXPECT_EQ(module_->locations.count, kLocationCount);
  EXPECT_EQ(materializer.projection.slots.count, kLocationCount - 1);
  EXPECT_GE(materializer.worklist.capacity, kLocationCount - 1);
  EXPECT_EQ(error_count_, 0u);
  loom_bytecode_selected_table_materializer_deinitialize(&materializer);
}

}  // namespace
}  // namespace loom
