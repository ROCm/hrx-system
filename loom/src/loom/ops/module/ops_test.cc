// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/module/ops.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class ModuleOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_module_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_MODULE,
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
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("imports.loom"), &context_,
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ModuleOpsTest, ParsePrintAndVerifyImports) {
  static const char kSource[] =
      "module.import \"motif/format/ggml.loom\" [@decode_q4, @decode_q6]\n"
      "module.import \"target/amdgpu/format/ggml.loom\" [@decode_q4]\n";
  iree_string_view_t source =
      iree_make_string_view(kSource, IREE_ARRAYSIZE(kSource) - 1);

  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);
  VerifyOk(module);
  EXPECT_EQ(Print(module), std::string(source.data, source.size));

  const loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 2u);
  const loom_op_t* import_op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_module_import_isa(import_op));
  EXPECT_EQ(
      loom_module_import_provider(import_op),
      loom_module_lookup_string(module, IREE_SV("motif/format/ggml.loom")));
  loom_symbol_ref_array_t symbols = loom_module_import_symbols(import_op);
  ASSERT_EQ(symbols.count, 2u);
  EXPECT_NE(symbols.values[0].symbol_id, symbols.values[1].symbol_id);

  const loom_op_t* target_import_op = loom_block_const_op(body, 1);
  ASSERT_TRUE(loom_module_import_isa(target_import_op));
  loom_symbol_ref_array_t target_symbols =
      loom_module_import_symbols(target_import_op);
  ASSERT_EQ(target_symbols.count, 1u);
  EXPECT_EQ(target_symbols.values[0].symbol_id, symbols.values[0].symbol_id);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
