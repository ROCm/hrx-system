// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/type/ops.h"
#include "loom/ops/type_builders.h"
#include "loom/verify/verify.h"

namespace {

static iree_status_t CaptureFailure(void* user_data,
                                    const loom_diagnostic_t* diagnostic) {
  auto* failure = static_cast<std::string*>(user_data);
  *failure = loom_error_def_id(diagnostic->error);
  if (diagnostic->param_count == 6 &&
      diagnostic->params[5].kind == LOOM_PARAM_STRING) {
    const auto value = diagnostic->params[5].string;
    failure->assign(value.data, value.size);
  }
  return iree_ok_status();
}

class ManagedTypeBytecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &index_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    iree_host_size_t count = 0;
    const auto* vtables = loom_type_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TYPE,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    source_ = Allocate();
    Declare(source_, "Other", "other");
    const auto decoder = Declare(source_, "Decoder", "input");
    Declare(source_, "Alias", "input");
    loom_type_t types[2];
    IREE_ASSERT_OK(loom_ref_type_make(source_, decoder, &types[0]));
    IREE_ASSERT_OK(loom_ptr_type_make(source_, decoder, &types[1]));
    const auto consumer = AddSymbol(source_, "consume");
    loom_builder_t builder;
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &builder);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_decl_build(&builder, 0, 0, 0, consumer, types, 2,
                                        nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &op));
    bytes_ = Write(source_);
    loom_bytecode_read_result_t result = {};
    IREE_ASSERT_OK(loom_bytecode_read_index(Bytes(), IREE_SV("objects.loombc"),
                                            &context_, &pool_, &index_arena_,
                                            nullptr, &result, &metadata_));
    ASSERT_EQ(result.error_count, 0u);
    for (iree_host_size_t i = 0; i < metadata_.modules[0].types.count; ++i) {
      const auto& entry = metadata_.modules[0].types.entries[i];
      if (bytes_[entry.entry_offset] == LOOM_BYTECODE_TYPE_REFERENCE) {
        ASSERT_EQ(entry.entry_length, 5u);
        reference_offsets_.push_back(entry.entry_offset);
      }
    }
    ASSERT_EQ(reference_offsets_.size(), 2u);
    ASSERT_LT(metadata_.modules[0].strings.count, 127u);
  }

  void TearDown() override {
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&index_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_module_t* Allocate() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                       &pool_, nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  loom_symbol_ref_t AddSymbol(loom_module_t* module, const char* name) {
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(
        module, iree_make_cstring_view(name), &name_id));
    loom_symbol_ref_t result = {};
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &result.symbol_id));
    return result;
  }

  loom_symbol_ref_t Declare(loom_module_t* module, const char* name,
                            const char* provider) {
    const auto symbol = AddSymbol(module, name);
    loom_string_id_t namespace_id, payload_id;
    IREE_CHECK_OK(loom_module_intern_string(
        module, iree_make_cstring_view(provider), &namespace_id));
    IREE_CHECK_OK(
        loom_module_intern_string(module, IREE_SV("decoder"), &payload_id));
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_type_decl_build(&builder, namespace_id, payload_id,
                                       symbol, LOOM_LOCATION_UNKNOWN, &op));
    return symbol;
  }

  std::vector<uint8_t> Write(loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_READABLE |
            IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream, nullptr, &pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_const_byte_span_t Bytes() const {
    return iree_make_const_byte_span(bytes_.data(), bytes_.size());
  }

  uint32_t StringIndex(const char* text) {
    const auto& strings = metadata_.modules[0].strings;
    uint32_t index = UINT32_MAX;
    for (uint32_t i = 0; i < strings.count; ++i) {
      if (iree_string_view_equal(strings.values[i],
                                 iree_make_cstring_view(text))) {
        index = i;
        break;
      }
    }
    EXPECT_LT(index, strings.count);
    return index;
  }

  void ExpectRejected(const char* expected_failure) {
    std::string failure;
    loom_bytecode_read_result_t result = {};
    const loom_bytecode_index_options_t index_options = {
        .diagnostic_sink = {CaptureFailure, &failure},
    };
    IREE_ASSERT_OK(loom_bytecode_read_metadata(
        Bytes(), IREE_SV("objects.loombc"), &context_, &pool_, &index_options,
        &result));
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(failure, expected_failure);
    loom_bytecode_file_metadata_t rejected_index = {};
    IREE_ASSERT_OK(loom_bytecode_read_index(
        Bytes(), IREE_SV("objects.loombc"), &context_, &pool_, &index_arena_,
        &index_options, &result, &rejected_index));
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(failure, expected_failure);
    const loom_bytecode_read_options_t read_options = {
        .diagnostic_sink = {CaptureFailure, &failure},
    };
    loom_module_t* rejected_module = nullptr;
    IREE_ASSERT_OK(loom_bytecode_read_module(
        Bytes(), IREE_SV("objects.loombc"), &context_, &pool_, &read_options,
        &result, &rejected_module, iree_allocator_system()));
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(failure, expected_failure);
    EXPECT_EQ(rejected_module, nullptr);
    loom_module_free(rejected_module);
  }

  // Shared storage outlives source, index, and materialized modules.
  iree_arena_block_pool_t pool_;
  // Retains index metadata while its source bytes are alive.
  iree_arena_allocator_t index_arena_;
  // Only declaration and test function schemas are registered; no provider.
  loom_context_t context_;
  // Source declarations and two independently typed function arguments.
  loom_module_t* source_ = nullptr;
  // Valid writer output, mutated only by malformed-wire tests.
  std::vector<uint8_t> bytes_;
  // Exact public index over the initial valid bytes.
  loom_bytecode_file_metadata_t metadata_ = {};
  // Offsets of the two compact wire records under test.
  std::vector<size_t> reference_offsets_;
};

TEST_F(ManagedTypeBytecodeTest, RejectsKeyAndBindingMismatchBeforeInterning) {
  const auto original = bytes_;
  const auto offset = reference_offsets_[1];
  for (const char* declaration : {"Other", "consume"}) {
    SCOPED_TRACE(declaration);
    bytes_ = original;
    // Make the second record semantically equal to the first, but bind it to
    // a different key or a non-type symbol. Canonicalization must not hide it.
    bytes_[offset + 1] = LOOM_REFERENCE_OWNERSHIP_OWNED;
    bytes_[offset + 4] = StringIndex(declaration);
    ExpectRejected("managed_type_key_disagrees_with_its_declaration");
  }
  for (size_t field : {2u, 3u}) {
    bytes_ = original;
    bytes_[offset + field] = StringIndex("other");
    ExpectRejected("managed_type_key_disagrees_with_its_declaration");
  }
}

TEST_F(ManagedTypeBytecodeTest, RejectsMissingDeclarationAndInvalidMode) {
  const auto offset = reference_offsets_[0];
  const auto original = bytes_;
  bytes_[offset + 4] = StringIndex("input");
  ExpectRejected("managed_type_declaration_name_is_not_defined");
  bytes_ = original;
  bytes_[offset + 1] = 2;
  ExpectRejected("ERR_BYTECODE_011");
  bytes_ = original;
  bytes_[offset + 2] = 127;
  ExpectRejected("ERR_BYTECODE_010");
}

TEST_F(ManagedTypeBytecodeTest, DeclarationKeyFieldsAreNonemptyAndNulFree) {
  auto* declaration = source_->symbols.entries[1].defining_op;
  const auto namespace_id = loom_type_decl_namespace(declaration);
  const auto name_id = loom_type_decl_type_name(declaration);
  loom_string_id_t empty_id, nul_id;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, iree_string_view_empty(), &empty_id));
  IREE_ASSERT_OK(loom_module_intern_string(
      source_, iree_make_string_view("a\0b", 3), &nul_id));
  for (bool change_namespace : {true, false}) {
    for (auto value : {loom_attr_string(empty_id), loom_attr_string(nul_id)}) {
      IREE_ASSERT_OK(loom_type_decl_set_namespace(
          source_, declaration, loom_attr_string(namespace_id)));
      IREE_ASSERT_OK(loom_type_decl_set_type_name(source_, declaration,
                                                  loom_attr_string(name_id)));
      if (change_namespace) {
        IREE_ASSERT_OK(
            loom_type_decl_set_namespace(source_, declaration, value));
      } else {
        IREE_ASSERT_OK(
            loom_type_decl_set_type_name(source_, declaration, value));
      }
      bytes_ = Write(source_);
      ExpectRejected("managed_type_key_requires_nonempty_nul_free_strings");
    }
  }
}

TEST_F(ManagedTypeBytecodeTest, DeclarationKeyFieldsAreRequiredStrings) {
  const auto& metadata = metadata_.modules[0];
  const auto ordinal =
      metadata.symbol_ordinal_by_string_index[StringIndex("Decoder")];
  const size_t start = metadata.symbols[ordinal].entry_offset;
  // This small record has one-byte name, location, op reference, and trivia
  // fields, with four bytes of kind/visibility/flags between name and location.
  const size_t attribute_count = start + 8;
  ASSERT_EQ(bytes_[start + 1], LOOM_BYTECODE_SYMBOL_RECORD);
  ASSERT_EQ(bytes_[start + 7], 0u);
  ASSERT_EQ(bytes_[attribute_count], 2u);
  const size_t attributes = attribute_count + 1;
  ASSERT_EQ(bytes_[attributes + 1], LOOM_BYTECODE_ATTR_STRING);
  ASSERT_EQ(bytes_[attributes + 4], LOOM_BYTECODE_ATTR_STRING);
  const auto original = bytes_;
  for (bool reverse : {false, true}) {
    bytes_ = original;
    if (reverse) {
      for (size_t i = 0; i < 3; ++i) {
        std::swap(bytes_[attributes + i], bytes_[attributes + 3 + i]);
      }
    }
    bytes_[attribute_count] = 1;
    ExpectRejected("managed_type_declaration_requires_both_key_attributes");
  }
  for (size_t kind_offset : {attributes + 1, attributes + 4}) {
    bytes_ = original;
    bytes_[kind_offset] = LOOM_BYTECODE_ATTR_I64;
    ExpectRejected("managed_type_key_requires_string_attributes");
  }
}

TEST_F(ManagedTypeBytecodeTest, SelectedTypesReuseDestinationAliasAndOwnKeys) {
  loom_module_t* destination = Allocate();
  const auto existing = Declare(destination, "Existing", "input");
  const auto mapped = Declare(destination, "Mapped", "input");
  const auto target_consumer = AddSymbol(destination, "consumer");
  loom_type_t canonical[2];
  IREE_ASSERT_OK(loom_ref_type_make(destination, existing, &canonical[0]));
  IREE_ASSERT_OK(loom_ptr_type_make(destination, existing, &canonical[1]));
  const auto& module_metadata = metadata_.modules[0];
  const iree_host_size_t selected =
      module_metadata.symbol_ordinal_by_string_index[StringIndex("consume")];
  const uint32_t source_decoder =
      module_metadata.symbol_ordinal_by_string_index[StringIndex("Decoder")];
  struct Projection {
    // Source declaration referenced by the selected signature.
    uint32_t source_ordinal;
    // Defined destination alias with the same exact payload key.
    loom_symbol_ref_t target;
    // Selected function whose target identity is already reserved.
    uint32_t consumer_ordinal;
    // Unfilled target function populated by this materialization.
    loom_symbol_ref_t consumer;
  } projection = {source_decoder, mapped, (uint32_t)selected, target_consumer};
  loom_bytecode_source_symbol_resolver_t resolver = {
      .fn =
          [](void* user_data, uint32_t ordinal, loom_symbol_ref_t* out_ref) {
            auto* projection = static_cast<Projection*>(user_data);
            if (ordinal == projection->consumer_ordinal) {
              *out_ref = projection->consumer;
            } else {
              EXPECT_EQ(ordinal, projection->source_ordinal);
              *out_ref = projection->target;
            }
            return iree_ok_status();
          },
      .user_data = &projection,
  };
  loom_bytecode_read_result_t result = {};
  IREE_ASSERT_OK(loom_bytecode_materialize_module_symbols_into(
      Bytes(), IREE_SV("objects.loombc"), &pool_, &metadata_, 0, {1, &selected},
      resolver, nullptr, &result, destination, iree_allocator_system()));
  ASSERT_EQ(result.error_count, 0u);
  loom_module_free(source_);
  source_ = nullptr;
  std::vector<uint8_t>().swap(bytes_);
  iree_arena_reset(&index_arena_);
  const auto* consumer = destination->symbols.entries[2].defining_op;
  ASSERT_NE(consumer, nullptr);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const auto value = loom_op_operands(consumer)[i];
    const auto type = loom_module_value_type(destination, value);
    EXPECT_TRUE(loom_type_equal(type, canonical[i]));
    EXPECT_EQ(loom_type_reference_declaration_id(type), existing.symbol_id);
    EXPECT_TRUE(iree_string_view_equal(
        loom_string_table_get(&destination->strings,
                              loom_type_reference_namespace_id(type)),
        IREE_SV("input")));
  }
  loom_verify_result_t verification = {};
  IREE_ASSERT_OK(loom_verify_module(destination, nullptr, &verification));
  EXPECT_EQ(verification.error_count, 0u);
  loom_module_free(destination);
}

}  // namespace
