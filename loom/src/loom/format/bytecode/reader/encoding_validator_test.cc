// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/encoding.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kEncodingParameters[] = {{
    /*.name=*/LOOM_BSTRING_REF(5, "block"),
    /*.attr_kind=*/LOOM_ATTR_I64,
    /*.flags=*/LOOM_ATTR_OPTIONAL,
}};
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

class BytecodeEncodingValidatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    iree_arena_initialize(&block_pool_, &retained_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("encoding_validator_test.loombc"), &error_count_, &decoder_);

    strings_[0] = iree_string_view_empty();
    strings_[1] = IREE_SV("q8_0");
    strings_[2] = IREE_SV("block");
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
        /*.kind=*/LOOM_BYTECODE_SECTION_ENCODINGS,
        /*.flags=*/{},
        /*.offset=*/0,
        /*.length=*/length,
        /*.absolute_offset=*/41,
        /*.bytes=*/iree_make_const_byte_span(data, length),
    };
  }

  // Registered family and string facts required by encoding validation.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // String table addressed by the encoded family and parameter names.
  iree_string_view_t strings_[3];
  // Block source shared by the short-lived and retained arenas.
  iree_arena_block_pool_t block_pool_;
  // Storage for validation facts consumed by later tables.
  iree_arena_allocator_t scratch_arena_;
  // Storage retaining exact entry metadata beyond validation.
  iree_arena_allocator_t retained_arena_;
  // Finalized encoding-family registry.
  loom_context_t context_;
};

TEST_F(BytecodeEncodingValidatorTest, ValidatesWithoutRetainingEntries) {
  const uint8_t data[] = {
      0x01,  // Family count.
      0x01,  // Family name string ordinal.
      0x01,  // Instance count.
      0x00,  // Instance family ordinal.
      0x00,  // No alias.
      0x01,  // Parameter count.
      0x02,  // Parameter name string ordinal.
      LOOM_BYTECODE_ATTR_I64,
      0x0E,  // Signed value 7.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(data, sizeof(data));

  IREE_ASSERT_OK(loom_bytecode_encoding_table_validate(
      &decoder_, &context_, &module_view_, &scratch_arena_, &section));

  ASSERT_EQ(module_view_.encodings.family_count, 1u);
  EXPECT_EQ(module_view_.encodings.family_name_ids[0], 1u);
  EXPECT_EQ(module_view_.encodings.count, 1u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeEncodingValidatorTest, IndexesExactInstanceRange) {
  const uint8_t data[] = {
      0x01,  // Family count.
      0x01,  // Family name string ordinal.
      0x01,  // Instance count.
      0x00,  // Instance family ordinal.
      0x00,  // No alias.
      0x01,  // Parameter count.
      0x02,  // Parameter name string ordinal.
      LOOM_BYTECODE_ATTR_I64,
      0x0E,  // Signed value 7.
  };
  const loom_bytecode_reader_section_t section =
      MakeSection(data, sizeof(data));

  loom_bytecode_encoding_metadata_t* entries = nullptr;
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(loom_bytecode_encoding_table_index(
      &decoder_, &context_, &module_view_, &scratch_arena_, &section,
      &retained_arena_, &entries, &count));

  ASSERT_EQ(count, 1u);
  ASSERT_NE(entries, nullptr);
  EXPECT_EQ(entries[0].entry_offset, 44u);
  EXPECT_EQ(entries[0].entry_length, 6u);
  EXPECT_EQ(entries[0].name_string_index, 1u);
  EXPECT_EQ(error_count_, 0u);
}

}  // namespace
}  // namespace loom
