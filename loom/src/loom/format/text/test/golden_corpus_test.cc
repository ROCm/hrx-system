// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C parser/printer identity tests over the checked-in .loom text corpus.
//
// Keeping the checked-in corpus active in C catches drift in comments,
// locations, dialect registration, type syntax, and format-element handling
// before cross-language tests have to diagnose it indirectly.

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/test/corpus/text/golden_text_corpus.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

class GoldenCorpusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    // This is the full checked-in text corpus, not a parser unit test: entries
    // intentionally cover every production dialect available in the build.
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
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = NULL;
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_;
};

TEST_F(GoldenCorpusTest, CorpusIsNotEmpty) {
  EXPECT_GT(loom_test_corpus_text_size(), 0u);
}

TEST_F(GoldenCorpusTest, TextRoundTripsIdentically) {
  const iree_file_toc_t* corpus = loom_test_corpus_text_create();
  for (size_t i = 0; i < loom_test_corpus_text_size(); ++i) {
    const iree_file_toc_t& file = corpus[i];
    SCOPED_TRACE(file.name);
    iree_string_view_t filename = iree_make_cstring_view(file.name);
    iree_string_view_t source =
        iree_make_string_view(file.data, (iree_host_size_t)file.size);

    loom_module_t* module = Parse(source, filename);
    if (!module) continue;
    std::string printed = Print(module);
    loom_module_free(module);

    EXPECT_EQ(std::string(file.data, file.size), printed);

    loom_module_t* reparsed =
        Parse(iree_make_string_view(printed.data(), printed.size()), filename);
    if (!reparsed) continue;
    std::string reprinted = Print(reparsed);
    loom_module_free(reparsed);

    EXPECT_EQ(printed, reprinted);
  }
}

}  // namespace
}  // namespace loom
