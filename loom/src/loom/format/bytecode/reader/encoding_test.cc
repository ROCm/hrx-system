// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/encoding.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"

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

class BytecodeEncodingTest : public ::testing::Test {
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
        IREE_SV("encoding_test.loombc"), &error_count_, &decoder_);

    strings_[0] = iree_string_view_empty();
    strings_[1] = IREE_SV("q8_0");
    strings_[2] = IREE_SV("block");
    module_view_.strings.values = strings_;
    module_view_.strings.count = IREE_ARRAYSIZE(strings_);
    family_name_ids_[0] = 1;
    module_view_.encodings.family_name_ids = family_name_ids_;
    module_view_.encodings.family_count = IREE_ARRAYSIZE(family_name_ids_);
    module_view_.encodings.count = 1;

    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(strings_); ++i) {
      loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
      IREE_ASSERT_OK(
          loom_module_intern_string(module_, strings_[i], &string_id));
      ASSERT_EQ(string_id, i);
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Validated module facts referenced by encoding parameters.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Source string table preserving direct module string IDs.
  iree_string_view_t strings_[3];
  // Encoding family identities in wire order.
  loom_string_id_t family_name_ids_[1];
  // Block source shared by scratch and output-module arenas.
  iree_arena_block_pool_t block_pool_;
  // Resettable parameter construction storage.
  iree_arena_allocator_t scratch_arena_;
  // Finalized encoding-family registry.
  loom_context_t context_;
  // Module receiving canonical encoding instances.
  loom_module_t* module_ = nullptr;
};

TEST_F(BytecodeEncodingTest, MaterializesCanonicalTable) {
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
  const loom_bytecode_reader_section_t section = {
      /*.kind=*/LOOM_BYTECODE_SECTION_ENCODINGS,
      /*.flags=*/{},
      /*.offset=*/0,
      /*.length=*/sizeof(data),
      /*.absolute_offset=*/41,
      /*.bytes=*/iree_make_const_byte_span(data, sizeof(data)),
  };
  loom_bytecode_encoding_materializer_t materializer = {
      /*.decoder=*/&decoder_,
      /*.context=*/&context_,
      /*.module_view=*/&module_view_,
      /*.scratch_arena=*/&scratch_arena_,
      /*.output_module=*/module_,
  };

  IREE_ASSERT_OK(
      loom_bytecode_encoding_table_materialize(&materializer, &section));

  ASSERT_EQ(module_->encodings.count, 1u);
  const loom_encoding_t* encoding = loom_module_encoding(module_, 1);
  ASSERT_NE(encoding, nullptr);
  EXPECT_EQ(encoding->name_id, 1u);
  EXPECT_EQ(encoding->alias_id, LOOM_STRING_ID_INVALID);
  ASSERT_EQ(encoding->attribute_count, 1u);
  EXPECT_EQ(encoding->attributes[0].name_id, 2u);
  EXPECT_EQ(encoding->attributes[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(encoding->attributes[0].value.i64, 7);
  EXPECT_EQ(error_count_, 0u);
}

}  // namespace
}  // namespace loom
