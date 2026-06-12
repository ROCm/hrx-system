// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C bytecode reader round-trip coverage over the checked-in text corpus.
//
// The text corpus is the durable source of representative Loom IR. This test
// uses it as bytecode reader seeds without checking in generated .loombc blobs:
// parse text, write bytecode, read bytecode through the C reader, and require a
// second write to produce identical bytes.

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/test/corpus/text/golden_text_corpus.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* error_ids = static_cast<std::vector<std::string>*>(user_data);
  error_ids->push_back(diagnostic->error->error_id);
  return loom_diagnostic_stderr_sink(nullptr, diagnostic);
}

class ReaderCorpusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    // This is a full text-corpus integration test: every checked-in dialect can
    // appear in corpus entries and must round-trip through bytecode.
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_target_core_test_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source, iree_string_view_t filename) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &options, &module));
    EXPECT_NE(module, nullptr)
        << "parse failed for " << std::string(filename.data, filename.size);
    return module;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_text_low_asm_environment_t low_asm_environment = {};
    loom_low_descriptor_text_asm_environment_initialize(&low_registry_.registry,
                                                        &low_asm_environment);
    const loom_text_print_options_t options = {
        /*.flags=*/LOOM_TEXT_PRINT_DEFAULT,
        /*.low_asm_environment=*/low_asm_environment,
    };
    IREE_EXPECT_OK(loom_text_print_module_to_builder_with_options(
        module, &builder, &options));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  iree_status_t WriteModule(const loom_module_t* module,
                            std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    iree_status_t status =
        loom_bytecode_write_module(module, stream, NULL, &block_pool_);

    if (iree_status_is_ok(status)) {
      iree_io_stream_pos_t length = iree_io_stream_length(stream);
      out_bytes->resize((size_t)length);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_read(stream, out_bytes->size(), out_bytes->data(),
                                   nullptr);
    }
    iree_io_stream_release(stream);
    return status;
  }

  loom_bytecode_read_result_t ReadModule(const std::vector<uint8_t>& bytes,
                                         loom_module_t** out_module,
                                         std::vector<std::string>* error_ids) {
    loom_bytecode_read_options_t options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/CaptureDiagnostic,
            /*.user_data=*/error_ids,
        },
        // The checked-in text corpus is a syntax/format corpus. Some entries
        // intentionally exercise constructs without being standalone semantic
        // programs, so this test isolates bytecode reader/writer canonicality.
        /*.verify_module=*/false,
        /*.verify_max_errors=*/20,
    };
    loom_bytecode_read_result_t result = {0};
    IREE_EXPECT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("corpus.loombc"), &context_, &block_pool_, &options, &result,
        out_module, iree_allocator_system()));
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_;
};

TEST_F(ReaderCorpusTest, CorpusIsNotEmpty) {
  EXPECT_GT(loom_test_corpus_text_size(), 0u);
}

TEST_F(ReaderCorpusTest, TextCorpusBytecodeRoundTripsCanonically) {
  const iree_file_toc_t* corpus = loom_test_corpus_text_create();
  size_t supported_count = 0;
  for (size_t i = 0; i < loom_test_corpus_text_size(); ++i) {
    const iree_file_toc_t& file = corpus[i];
    SCOPED_TRACE(file.name);
    iree_string_view_t filename = iree_make_cstring_view(file.name);
    iree_string_view_t source =
        iree_make_string_view(file.data, (iree_host_size_t)file.size);

    loom_module_t* module = Parse(source, filename);
    if (!module) continue;
    std::vector<uint8_t> first;
    iree_status_t write_status = WriteModule(module, &first);
    if (!iree_status_is_ok(write_status)) {
      IREE_EXPECT_OK(write_status);
      loom_module_free(module);
      continue;
    }
    std::string source_text = Print(module);
    loom_module_free(module);
    ++supported_count;

    loom_module_t* read_module = nullptr;
    std::vector<std::string> error_ids;
    loom_bytecode_read_result_t result =
        ReadModule(first, &read_module, &error_ids);
    EXPECT_EQ(result.error_count, 0u)
        << (error_ids.empty() ? "" : error_ids.front());
    EXPECT_TRUE(error_ids.empty())
        << (error_ids.empty() ? "" : error_ids.front());
    ASSERT_NE(read_module, nullptr);

    EXPECT_EQ(source_text, Print(read_module));

    std::vector<uint8_t> second;
    IREE_EXPECT_OK(WriteModule(read_module, &second));
    loom_module_free(read_module);

    EXPECT_EQ(first, second);
  }
  EXPECT_GT(supported_count, 0u);
}

}  // namespace
}  // namespace loom
