// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/module.h"
#include "loom/ops/template/ops.h"

namespace loom {
namespace {

class InputContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* vtables = loom_template_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEMPLATE, vtables, (uint16_t)count));
    const auto* semantics = loom_template_dialect_op_semantics(&count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_TEMPLATE, semantics, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  std::vector<uint8_t> WriteFamily(bool dimension_from_result,
                                   bool encoding_from_result) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("inputs"), &pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, IREE_SV("family"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name, &symbol));
    const loom_type_t view = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(1), 0);
    const loom_type_t arguments[] = {
        view, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), loom_type_encoding()};
    const loom_type_t results[] = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                   loom_type_encoding(), view};
    loom_op_t* family = nullptr;
    IREE_CHECK_OK(loom_template_decl_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*effect_contract=*/0, /*temperature=*/0, loom_symbol_ref_null(),
        loom_parameterized_attr_array_empty(), {0, symbol}, arguments,
        IREE_ARRAYSIZE(arguments), results, IREE_ARRAYSIZE(results),
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &family));
    const auto inputs = loom_template_decl_args(family);
    const auto outputs = loom_template_decl_results(family);
    loom_type_t input_type = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
        loom_dim_pack_dynamic(dimension_from_result ? outputs.values[0]
                                                    : inputs.values[1]),
        encoding_from_result ? outputs.values[1] : inputs.values[2]);
    input_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    IREE_CHECK_OK(
        loom_module_set_value_type(module, inputs.values[0], input_type));
    loom_type_t output_type = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
        loom_dim_pack_dynamic(outputs.values[0]), outputs.values[1]);
    output_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    IREE_CHECK_OK(
        loom_module_set_value_type(module, outputs.values[2], output_type));

    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_RESIZABLE | IREE_IO_STREAM_MODE_SEEKABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream, nullptr, &pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    loom_module_free(module);
    return bytes;
  }

  loom_bytecode_read_result_t Read(const std::vector<uint8_t>& bytes,
                                   loom_bytecode_file_metadata_t* metadata) {
    const loom_bytecode_index_options_t options = {
        .diagnostic_sink =
            {
                .fn =
                    [](void* user_data, const loom_diagnostic_t* diagnostic) {
                      auto* test = static_cast<InputContractTest*>(user_data);
                      test->diagnostic_id_ =
                          loom_error_def_id(diagnostic->error);
                      if (diagnostic->param_count == 6 &&
                          diagnostic->params[5].kind == LOOM_PARAM_STRING) {
                        const auto reason = diagnostic->params[5].string;
                        test->failure_reason_.assign(reason.data, reason.size);
                      }
                      return iree_ok_status();
                    },
                .user_data = this,
            },
    };
    loom_bytecode_read_result_t result = {};
    IREE_CHECK_OK(loom_bytecode_read_index(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("inputs.loombc"), &context_, &pool_, &arena_, &options, &result,
        metadata));
    return result;
  }

  // Backing for source serialization, reader scratch, and retained metadata.
  iree_arena_block_pool_t pool_;
  // Retained input contracts outlive the serialized source module.
  iree_arena_allocator_t arena_;
  // Only the template dialect is needed for the callable contract.
  loom_context_t context_;
  // Structured source diagnostic captured at admission.
  std::string diagnostic_id_;
  // Invalid-field detail accompanying the source diagnostic.
  std::string failure_reason_;
};

TEST_F(InputContractTest, RetainsForwardInputsWithoutResultRequirements) {
  const auto bytes = WriteFamily(false, false);
  loom_bytecode_file_metadata_t metadata = {};
  EXPECT_EQ(Read(bytes, &metadata).error_count, 0u);
  ASSERT_EQ(metadata.module_count, 1u);
  const auto& module = metadata.modules[0];
  ASSERT_EQ(module.symbol_count, 1u);
  const auto ordinal = module.symbols[0].input_contract_ordinal_plus_one;
  ASSERT_NE(ordinal, 0u);
  const auto& contract = *module.input_contracts[ordinal - 1];
  ASSERT_EQ(contract.count, 2u);
  EXPECT_EQ(contract.values[0].argument, 0u);
  EXPECT_EQ(contract.values[0].axis, 0u);
  EXPECT_TRUE(contract.values[0].expected_is_argument);
  EXPECT_EQ(contract.values[0].expected, 1u);
  EXPECT_EQ(contract.values[1].argument, 0u);
  EXPECT_EQ(contract.values[1].axis, UINT8_MAX);
  EXPECT_TRUE(contract.values[1].expected_is_argument);
  EXPECT_EQ(contract.values[1].expected, 2u);
}

TEST_F(InputContractTest, RejectsInputDimensionDependingOnResult) {
  const auto bytes = WriteFamily(true, false);
  loom_bytecode_file_metadata_t metadata = {};
  EXPECT_EQ(Read(bytes, &metadata).error_count, 1u);
  EXPECT_EQ(diagnostic_id_, "ERR_BYTECODE_006");
  EXPECT_EQ(failure_reason_, "dimension");
}

TEST_F(InputContractTest, RejectsInputEncodingDependingOnResult) {
  const auto bytes = WriteFamily(false, true);
  loom_bytecode_file_metadata_t metadata = {};
  EXPECT_EQ(Read(bytes, &metadata).error_count, 1u);
  EXPECT_EQ(diagnostic_id_, "ERR_BYTECODE_006");
  EXPECT_EQ(failure_reason_, "encoding_value");
}

}  // namespace
}  // namespace loom
