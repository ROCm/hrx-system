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
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// Text syntax produces aligned arrays by construction. API-created IR and
// serialized declarations must independently establish that same invariant.
struct SignatureFlagsCase {
  // Argument arity independently admitted by the source IR.
  uint16_t argument_count;
  // Result arity independently admitted by the source IR.
  uint16_t result_count;
  // Explicit flags in argument-then-result order, including malformed inputs.
  std::vector<uint8_t> flags;
  // Expected structured verification error, or zero for an admitted signature.
  uint16_t error_code;
};

class SignatureFlagsTest : public ::testing::TestWithParam<SignatureFlagsCase> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("access"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("native"), &name));
    uint16_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const auto& test = GetParam();
    const std::vector<loom_type_t> arguments(test.argument_count,
                                             loom_type_buffer());
    const std::vector<loom_type_t> results(test.result_count,
                                           loom_type_buffer());
    loom_op_t* declaration;
    IREE_ASSERT_OK(loom_func_decl_build(
        &builder, LOOM_FUNC_DECL_BUILD_FLAG_HAS_SIGNATURE_FLAGS, 0, 0,
        LOOM_STRING_ID_INVALID, LOOM_STRING_ID_INVALID, 0, 0, 0, 0,
        loom_symbol_ref_null(), loom_attr_absent(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), {0, symbol}, arguments.data(),
        arguments.size(),
        iree_make_const_byte_span(test.flags.data(), test.flags.size()),
        results.data(), results.size(), nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &declaration));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Arena backing the source module and readers.
  iree_arena_block_pool_t pool_;
  // Registered production operations and types.
  loom_context_t context_;
  // Source module constructed through the production declaration builder.
  loom_module_t* module_ = nullptr;
};

TEST_P(SignatureFlagsTest, VerifierAdmitsOnlyCompleteSupportedContracts) {
  testing::DiagnosticCapture capture;
  const loom_verify_options_t options = {.sink = capture.sink()};
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
  EXPECT_EQ(result.error_count, GetParam().error_code ? 1u : 0u);
  if (GetParam().error_code) {
    EXPECT_NE(testing::FindDiagnostic(
                  capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE,
                                                 GetParam().error_code)),
              nullptr);
  }
}

TEST_P(SignatureFlagsTest, BytecodeChecksContractsBeforeMaterialization) {
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));
  IREE_ASSERT_OK(loom_bytecode_write_module(module_, stream, nullptr, &pool_));
  std::vector<uint8_t> bytes(iree_io_stream_length(stream));
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  iree_io_stream_release(stream);
  testing::DiagnosticCapture capture;
  const loom_bytecode_read_options_t options = {
      .diagnostic_sink = capture.sink(),
  };
  loom_bytecode_read_result_t result = {};
  loom_module_t* decoded = nullptr;
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("access.loombc"), &context_, &pool_, &options,
      iree_allocator_system(), &result, &decoded));
  if (GetParam().error_code) {
    EXPECT_GT(result.error_count, 0u);
    EXPECT_EQ(decoded, nullptr);
  } else {
    EXPECT_EQ(result.error_count, 0u);
    ASSERT_NE(decoded, nullptr);
    const auto flags = loom_func_like_signature_flags(
        loom_func_like_cast(decoded, loom_module_block(decoded)->first_op));
    EXPECT_EQ(std::vector<uint8_t>(flags.data, flags.data + flags.data_length),
              GetParam().flags);
  }
  loom_module_free(decoded);
}

INSTANTIATE_TEST_SUITE_P(
    CombinedSignature, SignatureFlagsTest,
    ::testing::Values(
        SignatureFlagsCase{1, 0, {}, 68},
        SignatureFlagsCase{1, 0, {LOOM_ARGUMENT_ACCESS_READ, 0}, 68},
        SignatureFlagsCase{1, 1, {LOOM_ARGUMENT_ACCESS_READ}, 68},
        SignatureFlagsCase{1, 1, {LOOM_ARGUMENT_ACCESS_READ, 0}, 0},
        SignatureFlagsCase{1, 1, {0x80, 0}, 71},
        SignatureFlagsCase{1, 1, {5, 0}, 71},
        SignatureFlagsCase{1, 1, {0, LOOM_ARGUMENT_ACCESS_READ}, 71},
        SignatureFlagsCase{32768, 32768,
                           [] {
                             std::vector<uint8_t> flags(65536, 0);
                             flags.front() = LOOM_ARGUMENT_ACCESS_READ;
                             flags[32767] = LOOM_ARGUMENT_ACCESS_WRITE;
                             return flags;
                           }(),
                           0}));

}  // namespace
}  // namespace loom
