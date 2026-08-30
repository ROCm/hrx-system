// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-format/convert.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/bytecode/format.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* error_ids = static_cast<std::vector<std::string>*>(user_data);
  error_ids->push_back(loom_error_def_id(diagnostic->error));
  return iree_ok_status();
}

class LoomFormatConvertTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    low_descriptor_registry_.descriptor_set_providers =
        kLowDescriptorSetProviders;
    low_descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kLowDescriptorSetProviders);
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &low_asm_environment_);
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
        /*.low_asm_environment=*/low_asm_environment_,
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
        /*.low_asm_environment=*/low_asm_environment_,
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

  // Dialect context shared by converter calls in each test.
  loom_context_t context_;
  // Arena block pool backing parsed and decoded modules.
  iree_arena_block_pool_t block_pool_;
  // Test descriptor registry used by text assembly parsing and printing.
  loom_low_descriptor_registry_t low_descriptor_registry_ = {};
  // Configured text assembly environment passed through the converter.
  loom_text_low_asm_environment_t low_asm_environment_ = {};
};

static iree_string_view_t ModuleText() {
  return IREE_SV(
      "func.def @identity(%value: index) -> (index) {\n"
      "  func.return %value : index\n"
      "}\n");
}

static iree_string_view_t UnresolvedCallText() {
  return IREE_SV(
      "func.def @caller(%value: index) -> (index) {\n"
      "  %result = func.call @external_identity(%value) : (index) -> "
      "(index)\n"
      "  func.return %result : index\n"
      "}\n");
}

static iree_string_view_t DeclaredCallText() {
  return IREE_SV(
      "func.decl @external_identity(%value: index) -> (index)\n"
      "\n"
      "func.def @caller(%value: index) -> (index) {\n"
      "  %result = func.call @external_identity(%value) : (index) -> "
      "(index)\n"
      "  func.return %result : index\n"
      "}\n");
}

static iree_string_view_t AuthoredLowSyntaxText() {
  return IREE_SV(
      "low.func.def target<test.low.core> @assembly("
      "%value: reg<test.i32>) -> (reg<test.i32>) asm {\n"
      "  return %value\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.core> @generic("
      "%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  low.return %value : reg<test.i32>\n"
      "}\n"
      "\n"
      "low.func.def target<test.low.core> @select("
      "%condition: reg<test.i32>, %then_value: reg<test.i32>, "
      "%else_value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %result = low.scf.if %condition -> (reg<test.i32>) asm {\n"
      "    low.scf.yield %then_value : reg<test.i32>\n"
      "  } else {\n"
      "    low.scf.yield %else_value : reg<test.i32>\n"
      "  }\n"
      "  low.return %result : reg<test.i32>\n"
      "}\n");
}

static iree_string_view_t GroupedSourceText() {
  return IREE_SV(
      "func.def @first() {\n"
      "  func.return\n"
      "}\n"
      "\n"
      "// Grouped symbol.\n"
      "func.def @grouped() {\n"
      "\n"
      "// Explicit entry block.\n"
      "^entry:\n"
      "\n"
      "  // Grouped terminator.\n"
      "  func.return\n"
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

TEST_F(LoomFormatConvertTest, TextToBytecodeAcceptsDeclaredCall) {
  std::string bytecode = ConvertTextToBytecode(DeclaredCallText());
  ASSERT_GE(bytecode.size(), static_cast<size_t>(LOOM_BYTECODE_MAGIC_LENGTH));
  EXPECT_EQ(
      bytecode.compare(0, LOOM_BYTECODE_MAGIC_LENGTH, LOOM_BYTECODE_MAGIC), 0);
}

TEST_F(LoomFormatConvertTest, RejectsUnresolvedCallBeforeWritingOutput) {
  for (loom_module_format_t output_format :
       {LOOM_MODULE_FORMAT_TEXT, LOOM_MODULE_FORMAT_BYTECODE}) {
    std::vector<std::string> error_ids;
    loom_format_output_t output = {0};
    loom_format_convert_options_t options = {
        /*.input_format=*/LOOM_MODULE_FORMAT_TEXT,
        /*.output_format=*/output_format,
        /*.diagnostic_sink=*/{CaptureDiagnostic, &error_ids},
    };
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_format_convert(
            iree_make_const_byte_span(UnresolvedCallText().data,
                                      UnresolvedCallText().size),
            IREE_SV("unresolved.loom"), &context_, &block_pool_, &options,
            &output, iree_allocator_system()));
    EXPECT_EQ(error_ids, std::vector<std::string>({"ERR_SYMBOL_002"}));
    EXPECT_EQ(output.data, nullptr);
    EXPECT_EQ(output.length, 0u);
    loom_format_output_deinitialize(&output, iree_allocator_system());
  }
}

TEST_F(LoomFormatConvertTest, BytecodeRoundTripsToText) {
  std::string bytecode = ConvertTextToBytecode(ModuleText());
  std::string text = ConvertToString(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      LOOM_MODULE_FORMAT_AUTO, LOOM_MODULE_FORMAT_TEXT);
  EXPECT_EQ(text, std::string(ModuleText().data, ModuleText().size));
}

TEST_F(LoomFormatConvertTest, BytecodePreservesAuthoredLowSyntax) {
  std::string bytecode = ConvertTextToBytecode(AuthoredLowSyntaxText());
  std::string text = ConvertToString(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      LOOM_MODULE_FORMAT_BYTECODE, LOOM_MODULE_FORMAT_TEXT);
  EXPECT_EQ(text, std::string(AuthoredLowSyntaxText().data,
                              AuthoredLowSyntaxText().size));
}

TEST_F(LoomFormatConvertTest, BytecodePreservesVerticalSourceGrouping) {
  std::string bytecode = ConvertTextToBytecode(GroupedSourceText());
  std::string text = ConvertToString(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      LOOM_MODULE_FORMAT_BYTECODE, LOOM_MODULE_FORMAT_TEXT);
  EXPECT_EQ(text,
            std::string(GroupedSourceText().data, GroupedSourceText().size));
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
