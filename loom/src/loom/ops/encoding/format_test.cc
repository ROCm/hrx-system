// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::GetStringParam;

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    /*.name=*/IREE_SV("q8_0"),
};

static const loom_encoding_vtable_t kQuantizationEncodingVtable = {
    /*.name=*/IREE_SV("quantization"),
};

class EncodingFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kQuantizationEncodingVtable));
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

  iree_status_t Parse(const char* source, loom_module_t** out_module) {
    capture_.Reset();
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = capture_.sink();
    options.max_errors = 100;
    return loom_text_parse(iree_make_cstring_view(source),
                           IREE_SV("encoding_format_test.loom"), &context_,
                           &block_pool_, &options, out_module);
  }

  loom_module_t* ParseOk(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, &module));
    EXPECT_TRUE(capture_.diagnostics.empty());
    EXPECT_NE(module, nullptr);
    return module;
  }

  const std::vector<CapturedDiagnostic>& ParseExpectErrors(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, &module));
    EXPECT_EQ(module, nullptr);
    EXPECT_GT(capture_.diagnostics.size(), 0u);
    return capture_.diagnostics;
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

  // Block pool backing parser and module arenas in each test.
  iree_arena_block_pool_t block_pool_;
  // Dialect/type/encoding registry used by parser calls.
  loom_context_t context_;
  // Diagnostic capture sink populated by parse helpers.
  DiagnosticCapture capture_;
};

static loom_op_t* GetFirstFunctionOp(const loom_module_t* module) {
  if (!module || module->symbols.count == 0) {
    return nullptr;
  }
  return module->symbols.entries[0].defining_op;
}

static loom_block_t* GetEntryBlock(loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return nullptr;
  }
  return loom_region_entry_block(region);
}

TEST_F(EncodingFormatTest, DefineInlineSpec) {
  loom_module_t* module =
      ParseOk("%enc = encoding.define #q8_0<block=32> : encoding<schema>\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* spec_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(spec_encoding, nullptr);
  ASSERT_LT(spec_encoding->name_id, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->name_id], IREE_SV("q8_0")));
  ASSERT_EQ(spec_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->attributes[0].name_id],
      IREE_SV("block")));
  EXPECT_EQ(spec_encoding->attributes[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(spec_encoding->attributes[0].value.i64, 32);

  EXPECT_EQ(PrintModule(module),
            "%enc = encoding.define #q8_0<block=32> : encoding<schema>\n");
  loom_module_free(module);
}

TEST_F(EncodingFormatTest, DefineDynamicParams) {
  loom_module_t* module = ParseOk(
      "test.func @test(%group_size: index, %scale: f32) {\n"
      "  %enc = encoding.define #q8_0<block=32> "
      "{scale = %scale : f32, group_size = %group_size : index} : "
      "encoding<schema>\n"
      "  test.yield\n"
      "}\n");
  ASSERT_NE(module, nullptr);

  loom_op_t* func_op = GetFirstFunctionOp(module);
  ASSERT_NE(func_op, nullptr);
  loom_block_t* entry = GetEntryBlock(loom_test_func_body(func_op));
  ASSERT_NE(entry, nullptr);
  ASSERT_GE(entry->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(entry, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_value_slice_t params = loom_encoding_define_params(op);
  ASSERT_EQ(params.count, 2u);
  EXPECT_EQ(params.values[0], entry->arg_ids[0]);
  EXPECT_EQ(params.values[1], entry->arg_ids[1]);

  loom_named_attr_slice_t param_names = loom_encoding_define_param_names(op);
  ASSERT_EQ(param_names.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[param_names.entries[0].name_id],
      IREE_SV("group_size")));
  EXPECT_EQ(param_names.entries[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(param_names.entries[0].value.i64, 0);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[param_names.entries[1].name_id],
      IREE_SV("scale")));
  EXPECT_EQ(param_names.entries[1].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(param_names.entries[1].value.i64, 1);

  EXPECT_EQ(PrintModule(module),
            "test.func @test(%group_size: index, %scale: f32) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{group_size = %group_size : index, scale = %scale : f32} : "
            "encoding<schema>\n"
            "  test.yield\n"
            "}\n");
  loom_module_free(module);
}

TEST_F(EncodingFormatTest, StaticEncodingRejectsSSAParameter) {
  const auto& diagnostics = ParseExpectErrors(
      "test.func @test(%group_size: index) {\n"
      "  %enc = encoding.define #q8_0<group_size=%group_size> : "
      "encoding<schema>\n"
      "  test.yield\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0],
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 28));
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "group_size");
}

TEST_F(EncodingFormatTest, StaticEncodingRejectsDuplicateParameter) {
  const auto& diagnostics = ParseExpectErrors(
      "%enc = encoding.define #q8_0<block=32, block=64> : encoding<schema>\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0],
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 29));
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "block");
}

TEST_F(EncodingFormatTest, DefineAliasSpec) {
  loom_module_t* module = ParseOk(
      "#enc = #q8_0<block=32>\n"
      "%enc = encoding.define #enc : encoding<schema>\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* spec_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(spec_encoding, nullptr);
  ASSERT_NE(spec_encoding->alias_id, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->alias_id], IREE_SV("enc")));

  EXPECT_EQ(PrintModule(module),
            "#enc = #q8_0<block=32>\n"
            "%enc = encoding.define #enc : encoding<schema>\n");
  loom_module_free(module);
}

TEST_F(EncodingFormatTest, DefineNestedInlineSpec) {
  loom_module_t* module = ParseOk(
      "%enc = encoding.define "
      "#quantization<spec=#q8_0<block=32>> : encoding<schema>\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* outer_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(outer_encoding, nullptr);
  ASSERT_EQ(outer_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[outer_encoding->attributes[0].name_id],
      IREE_SV("spec")));

  loom_attribute_t nested_spec = outer_encoding->attributes[0].value;
  ASSERT_EQ(nested_spec.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* nested_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(nested_spec));
  ASSERT_NE(nested_encoding, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[nested_encoding->name_id], IREE_SV("q8_0")));
  ASSERT_EQ(nested_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[nested_encoding->attributes[0].name_id],
      IREE_SV("block")));
  EXPECT_EQ(nested_encoding->attributes[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(nested_encoding->attributes[0].value.i64, 32);

  EXPECT_EQ(PrintModule(module),
            "%enc = encoding.define "
            "#quantization<spec=#q8_0<block=32>> : encoding<schema>\n");
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
