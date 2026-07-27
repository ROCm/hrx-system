// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CfgFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id, const loom_op_vtable_t* const* (
                                               *vtable_fn)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = vtable_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_module_t* ParseOk(const char* source) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/100,
    };
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("cfg_format_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  std::string PrintModule(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  std::string RoundTrip(const char* source) {
    loom_module_t* first_module = ParseOk(source);
    if (!first_module) {
      return "";
    }
    std::string first_text = PrintModule(first_module);
    loom_module_free(first_module);

    loom_module_t* second_module = ParseOk(first_text.c_str());
    if (!second_module) {
      return "";
    }
    std::string second_text = PrintModule(second_module);
    loom_module_free(second_module);

    EXPECT_EQ(first_text, second_text) << "Round-trip mismatch";
    return first_text;
  }

  // Block pool backing parser and module arenas in each test.
  iree_arena_block_pool_t block_pool_;
  // Dialect registry used by parser calls.
  loom_context_t context_;
};

TEST_F(CfgFormatTest, BranchWithArgumentsRoundTrips) {
  std::string text = RoundTrip(
      "test.func @cfg(%arg: i32) -> (i32) {\n"
      "  cfg.br ^exit(%arg: i32)\n"
      "^exit(%value: i32):\n"
      "  test.yield %value : i32\n"
      "}\n");
  EXPECT_NE(text.find("cfg.br ^exit(%arg: i32)"), std::string::npos) << text;
  EXPECT_NE(text.find("^exit(%value: i32):"), std::string::npos) << text;
}

}  // namespace
}  // namespace loom
