// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kElementParameter = {
    .name = LOOM_BSTRING_REF(7, "element"),
    .attr_kind = LOOM_ATTR_TYPE,
};
static const loom_encoding_family_descriptor_t kTypedEncodingDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(5, "typed"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/1,
    /*.parameter_descriptors=*/&kElementParameter,
};
static const loom_encoding_vtable_t kTypedEncodingVtable = {
    .descriptor = &kTypedEncodingDescriptor,
};

class MixedTablesTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &source_pool_);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &output_pool_);
    iree_arena_initialize(&source_pool_, &metadata_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kTypedEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("mixed_tables"),
                                        &source_pool_, nullptr,
                                        iree_allocator_system(), &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("selected"),
                                        &output_pool_, nullptr,
                                        iree_allocator_system(), &selected_));
  }

  void TearDown() override {
    loom_module_free(source_);
    loom_module_free(full_);
    loom_module_free(selected_);
    iree_arena_deinitialize(&metadata_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&source_pool_);
    iree_arena_block_pool_deinitialize(&output_pool_);
  }

  uint16_t AddEncoding(loom_module_t* module, loom_type_id_t type) {
    loom_named_attr_t parameter = {};
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("element"),
                                            &parameter.name_id));
    parameter.value = loom_attr_type(type);
    loom_encoding_t encoding = {};
    IREE_CHECK_OK(
        loom_module_intern_string(module, IREE_SV("typed"), &encoding.name_id));
    encoding.alias_id = LOOM_STRING_ID_INVALID;
    encoding.attribute_count = 1;
    encoding.attributes = &parameter;
    uint16_t encoding_id = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module, &encoding, &encoding_id));
    return encoding_id;
  }

  void BuildChain(int count) {
    loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        source_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &type_id));
    for (int i = 0; i < count; ++i) {
      const uint16_t encoding_id = AddEncoding(source_, type_id);
      ASSERT_EQ(encoding_id, i + 1);
      IREE_ASSERT_OK(loom_module_intern_type_id(
          source_,
          loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                              loom_dim_pack_static(4), encoding_id),
          &type_id));
    }
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("owner"), &name));
    loom_symbol_ref_t symbol = {};
    IREE_ASSERT_OK(loom_module_add_symbol(source_, name, &symbol.symbol_id));
    loom_named_attr_t attribute = {};
    IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("type"),
                                             &attribute.name_id));
    attribute.value = loom_attr_type(type_id);
    loom_builder_t builder;
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &builder);
    loom_op_t* record = nullptr;
    IREE_ASSERT_OK(loom_test_record_build(
        &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0, symbol,
        loom_make_named_attr_slice(&attribute, 1), LOOM_LOCATION_NONE,
        &record));
  }

  std::vector<uint8_t> WriteSource() {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(
        loom_bytecode_write_module(source_, stream, nullptr, &source_pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  void CheckChain(loom_module_t* module, int count, uint16_t encoding_offset) {
    ASSERT_EQ(module->symbols.count, 1u);
    const loom_op_t* record = module->symbols.entries[0].defining_op;
    ASSERT_NE(record, nullptr);
    ASSERT_TRUE(loom_test_record_isa(record));
    const auto dict = loom_test_record_dict(record);
    ASSERT_EQ(dict.count, 1u);
    loom_type_id_t type_id = loom_attr_as_type_id(dict.entries[0].value);
    for (int i = count; i > 0; --i) {
      const auto type = loom_type_table_get(&module->types, type_id);
      ASSERT_EQ(loom_type_kind(type), LOOM_TYPE_TILE);
      EXPECT_EQ(type.dims[0], loom_dim_pack_static(4));
      ASSERT_EQ(type.encoding_id, i + encoding_offset);
      const auto* encoding = loom_module_encoding(module, type.encoding_id);
      ASSERT_NE(encoding, nullptr);
      ASSERT_EQ(encoding->attribute_count, 1u);
      type_id = loom_attr_as_type_id(encoding->attributes[0].value);
    }
    EXPECT_TRUE(loom_type_equal(loom_type_table_get(&module->types, type_id),
                                loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
    EXPECT_EQ(module->types.count, count + encoding_offset + 1);
    EXPECT_EQ(module->encodings.count, count + encoding_offset);
    const loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  // Registry shared by independent source and output ownership domains.
  loom_context_t context_;
  // Source IR, serialized index, and selective-reader scratch block owner.
  iree_arena_block_pool_t source_pool_;
  // Returned IR block owner, kept live after source teardown.
  iree_arena_block_pool_t output_pool_;
  // Index metadata released before inspecting returned IR.
  iree_arena_allocator_t metadata_arena_;
  // Source table chain, explicitly freed before output inspection.
  loom_module_t* source_ = nullptr;
  // Standalone full-reader output.
  loom_module_t* full_ = nullptr;
  // Existing module receiving selectively projected table identities.
  loom_module_t* selected_ = nullptr;
};

TEST_P(MixedTablesTest, FullAndSelectedOutputSurviveSourceTeardown) {
  const int count = GetParam();
  ASSERT_NO_FATAL_FAILURE(BuildChain(count));
  ASSERT_NO_FATAL_FAILURE(CheckChain(source_, count, 0));
  loom_type_id_t existing_type = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      selected_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &existing_type));
  const uint16_t existing_encoding = AddEncoding(selected_, existing_type);
  ASSERT_EQ(existing_encoding, 1u);
  {
    const auto bytes = WriteSource();
    const auto span = iree_make_const_byte_span(bytes.data(), bytes.size());
    const auto filename = IREE_SV("mixed_tables.loombc");
    const loom_bytecode_read_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_bytecode_read_result_t result = {};
    IREE_ASSERT_OK(loom_bytecode_read_module(span, filename, &context_,
                                             &output_pool_, &options, &result,
                                             &full_, iree_allocator_system()));
    ASSERT_EQ(result.error_count, 0u);
    ASSERT_NE(full_, nullptr);
    const loom_bytecode_index_options_t index_options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_bytecode_file_metadata_t metadata = {};
    IREE_ASSERT_OK(loom_bytecode_read_index(
        span, filename, &context_, &source_pool_, &metadata_arena_,
        &index_options, &result, &metadata));
    ASSERT_EQ(result.error_count, 0u);
    const iree_host_size_t ordinal = 0;
    IREE_ASSERT_OK(loom_bytecode_materialize_module_symbols_into(
        span, filename, &source_pool_, &metadata, 0, {1, &ordinal}, {},
        &options, &result, selected_, iree_allocator_system()));
    ASSERT_EQ(result.error_count, 0u);
  }
  loom_module_free(source_);
  source_ = nullptr;
  iree_arena_reset(&metadata_arena_);
  iree_arena_block_pool_trim(&source_pool_);
  ASSERT_NO_FATAL_FAILURE(CheckChain(full_, count, 0));
  ASSERT_NO_FATAL_FAILURE(CheckChain(selected_, count, 1));
  EXPECT_TRUE(
      loom_type_equal(loom_type_table_get(&selected_->types, existing_type),
                      loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  EXPECT_EQ(
      loom_attr_as_type_id(loom_module_encoding(selected_, existing_encoding)
                               ->attributes[0]
                               .value),
      existing_type);
}

INSTANTIATE_TEST_SUITE_P(Depth, MixedTablesTest,
                         ::testing::Values(1, 4, 129, 2048));

}  // namespace
}  // namespace loom
