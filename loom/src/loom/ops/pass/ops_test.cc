// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/pass/ops.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class PassOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_pass_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_PASS,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.max_errors=*/20,
    };
    loom_module_t* module = NULL;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("passes.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  void VerifyOk(loom_module_t* module) {
    loom_verify_options_t options = {
        /*.sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.max_errors=*/20,
    };
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  std::vector<uint8_t> WriteBytecode(const loom_module_t* module) {
    iree_io_stream_t* stream = NULL;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, NULL, &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
    iree_io_stream_release(stream);
    return bytes;
  }

  loom_module_t* ReadBytecode(const std::vector<uint8_t>& bytes) {
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.verify_module=*/true,
    };
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = NULL;
    IREE_EXPECT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("passes.loombc"), &context_, &block_pool_, &options, &result,
        &module, iree_allocator_system()));
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(result.error_count, 0u);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(PassOpsTest, ParsePrintVerifyAndBytecodeRoundTrip) {
  static const char kSource[] =
      "pass.pipeline<module> @cleanup pipeline {\n"
      "  canonicalize(max_iterations = 10)\n"
      "  repeat until_converged(max_iterations = 8) {\n"
      "    cse\n"
      "    dce\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<func> @function_cleanup pipeline {\n"
      "  for func {\n"
      "    where name(value = \"matmul\") {\n"
      "      vector-memory-footprint(budget_bytes = 4096)\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<module> @debug pipeline {\n"
      "  call @cleanup\n"
      "  fail \"expected cleanup to run\"\n"
      "  halt \"inspect IR\"\n"
      "}\n";
  iree_string_view_t source =
      iree_make_string_view(kSource, IREE_ARRAYSIZE(kSource) - 1);

  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);
  VerifyOk(module);
  EXPECT_EQ(Print(module), std::string(source.data, source.size));

  std::vector<uint8_t> bytecode = WriteBytecode(module);
  loom_module_t* loaded = ReadBytecode(bytecode);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(Print(loaded), std::string(source.data, source.size));

  loom_module_free(loaded);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
