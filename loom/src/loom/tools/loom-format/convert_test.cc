// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-format/convert.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

class LoomFormatConvertTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_FUNC,
                                   loom_func_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_INDEX,
                                   loom_index_dialect_vtables));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string ConvertToString(iree_string_view_t input,
                              loom_module_format_t input_format,
                              loom_module_format_t output_format) {
    loom_format_output_t output = {0};
    loom_format_convert_options_t options = {
        /*.input_format=*/input_format,
        /*.output_format=*/output_format,
        /*.diagnostic_sink=*/{0},
    };
    IREE_EXPECT_OK(loom_format_convert(
        iree_make_const_byte_span(input.data, input.size), IREE_SV("test"),
        &context_, &block_pool_, &options, &output, iree_allocator_system()));
    std::string result((const char*)output.data, output.length);
    loom_format_output_deinitialize(&output, iree_allocator_system());
    return result;
  }

  std::string ConvertToString(iree_const_byte_span_t input,
                              loom_module_format_t input_format,
                              loom_module_format_t output_format) {
    loom_format_output_t output = {0};
    loom_format_convert_options_t options = {
        /*.input_format=*/input_format,
        /*.output_format=*/output_format,
        /*.diagnostic_sink=*/{0},
    };
    IREE_EXPECT_OK(loom_format_convert(input, IREE_SV("test"), &context_,
                                       &block_pool_, &options, &output,
                                       iree_allocator_system()));
    std::string result((const char*)output.data, output.length);
    loom_format_output_deinitialize(&output, iree_allocator_system());
    return result;
  }

  std::string ConvertTextToBytecode(iree_string_view_t input) {
    return ConvertToString(input, LOOM_MODULE_FORMAT_TEXT,
                           LOOM_MODULE_FORMAT_BYTECODE);
  }

  loom_context_t context_;
  iree_arena_block_pool_t block_pool_;
};

static iree_string_view_t ModuleText() {
  return IREE_SV(
      "func.def @identity(%value: index) -> (index) {\n"
      "  func.return %value : index\n"
      "}\n");
}

TEST(FormatKind, ParsesAcceptedSpellings) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_AUTO;

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("auto"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_AUTO);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("text"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_TEXT);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("bc"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_BYTECODE);

  IREE_EXPECT_OK(loom_module_format_parse(IREE_SV("bytecode"), true, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_BYTECODE);
}

TEST(FormatKind, RejectsAutoWhenDisallowed) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_TEXT;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_format_parse(IREE_SV("auto"), false, &format));
  EXPECT_EQ(format, LOOM_MODULE_FORMAT_TEXT);
}

TEST_F(LoomFormatConvertTest, TextToTextCanonicalizes) {
  EXPECT_EQ(ConvertToString(ModuleText(), LOOM_MODULE_FORMAT_TEXT,
                            LOOM_MODULE_FORMAT_TEXT),
            std::string(ModuleText().data, ModuleText().size));
}

TEST_F(LoomFormatConvertTest, AutoDetectsText) {
  EXPECT_EQ(ConvertToString(ModuleText(), LOOM_MODULE_FORMAT_AUTO,
                            LOOM_MODULE_FORMAT_TEXT),
            std::string(ModuleText().data, ModuleText().size));
}

TEST_F(LoomFormatConvertTest, TextToBytecodeEmitsMagic) {
  std::string bytecode = ConvertTextToBytecode(ModuleText());
  ASSERT_GE(bytecode.size(), static_cast<size_t>(LOOM_BYTECODE_MAGIC_LENGTH));
  EXPECT_EQ(
      bytecode.compare(0, LOOM_BYTECODE_MAGIC_LENGTH, LOOM_BYTECODE_MAGIC), 0);
}

TEST_F(LoomFormatConvertTest, BytecodeRoundTripsToText) {
  std::string bytecode = ConvertTextToBytecode(ModuleText());
  std::string text = ConvertToString(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      LOOM_MODULE_FORMAT_AUTO, LOOM_MODULE_FORMAT_TEXT);
  EXPECT_EQ(text, std::string(ModuleText().data, ModuleText().size));
}

TEST_F(LoomFormatConvertTest, ExplicitBytecodeInputRoundTripsToText) {
  std::string bytecode = ConvertTextToBytecode(ModuleText());
  std::string text = ConvertToString(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      LOOM_MODULE_FORMAT_BYTECODE, LOOM_MODULE_FORMAT_TEXT);
  EXPECT_EQ(text, std::string(ModuleText().data, ModuleText().size));
}

TEST_F(LoomFormatConvertTest, RejectsAutoOutputFormat) {
  loom_format_output_t output = {0};
  loom_format_convert_options_t options = {
      /*.input_format=*/LOOM_MODULE_FORMAT_TEXT,
      /*.output_format=*/LOOM_MODULE_FORMAT_AUTO,
      /*.diagnostic_sink=*/{0},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_format_convert(
          iree_make_const_byte_span(ModuleText().data, ModuleText().size),
          IREE_SV("test"), &context_, &block_pool_, &options, &output,
          iree_allocator_system()));
  loom_format_output_deinitialize(&output, iree_allocator_system());
}

TEST_F(LoomFormatConvertTest, MalformedTextReturnsInvalidArgument) {
  loom_format_output_t output = {0};
  loom_format_convert_options_t options = {
      /*.input_format=*/LOOM_MODULE_FORMAT_TEXT,
      /*.output_format=*/LOOM_MODULE_FORMAT_TEXT,
      /*.diagnostic_sink=*/{0},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_format_convert(iree_make_const_byte_span("not.an.op\n", 10),
                          IREE_SV("bad.loom"), &context_, &block_pool_,
                          &options, &output, iree_allocator_system()));
  loom_format_output_deinitialize(&output, iree_allocator_system());
}

}  // namespace
}  // namespace loom
