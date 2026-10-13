// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/format_tables.h"

#include <cstdio>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

class OperandDictionaryParseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static std::string Name(size_t ordinal) {
    char name[32];
    std::snprintf(name, sizeof(name), "parameter_%05zu", ordinal);
    return name;
  }

  // Descending names pair two distinct SSA values across every sort boundary.
  static std::string Dictionary(size_t count) {
    std::string source = "{\n";
    for (size_t i = count; i > 0; --i) {
      source += "  " + Name(i - 1) + " = " + ((i - 1) % 2 ? "%odd" : "%even") +
                " : index";
      source += i > 1 ? ",\n" : "\n";
    }
    return source + "}";
  }

  iree_status_t Parse(const std::string& dictionary,
                      loom_module_t** out_module) {
    const std::string source =
        "%even = test.constant 2 : index\n"
        "%odd = test.constant 3 : index\n"
        "%result = test.operand_dict %even " +
        dictionary + " : index\n";
    capture_.Reset();
    loom_text_parse_options_t options = {.diagnostic_sink = capture_.sink()};
    return loom_text_parse(iree_make_string_view(source.data(), source.size()),
                           IREE_SV("dictionary.loom"), &context_, &pool_,
                           &options, out_module);
  }

  // Shared production pool for persistent modules and parser scratch.
  iree_arena_block_pool_t pool_ = {};
  // Registered format and operand metadata consumed by the parser.
  loom_context_t context_ = {};
  // Copied diagnostics remain inspectable after parsing releases scratch.
  testing::DiagnosticCapture capture_;
};

TEST_F(OperandDictionaryParseTest, SortingPreservesValuesAndSourceSpans) {
  for (size_t count : {0, 1, 16, 17, 64, 65, 1024, 8192}) {
    SCOPED_TRACE(count);
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(Parse(Dictionary(count), &module));
    ASSERT_NE(module, nullptr);
    EXPECT_TRUE(capture_.diagnostics.empty());
    auto* block = loom_module_block(module);
    const loom_value_id_t values[] = {
        loom_test_constant_result(loom_block_op(block, 0)),
        loom_test_constant_result(loom_block_op(block, 1)),
    };
    const auto* op = loom_block_op(block, 2);
    const auto names = loom_test_operand_dict_param_names(op);
    const auto operands = loom_test_operand_dict_params(op);
    ASSERT_EQ(names.count, count);
    ASSERT_EQ(operands.count, count);
    for (size_t i = 0; i < count; ++i) {
      auto name =
          loom_string_table_get(&module->strings, names.entries[i].name_id);
      EXPECT_EQ(std::string(name.data, name.size), Name(i));
      EXPECT_EQ(names.entries[i].value.i64, (int64_t)i);
      EXPECT_EQ(operands.values[i], values[i % 2]);
    }
    const auto* location =
        loom_location_table_const_entry(&module->locations, op->location);
    ASSERT_EQ(location->kind, LOOM_LOCATION_FILE);
    size_t parameter_spans = 0;
    for (uint16_t i = 0; i < location->file.field_span_count; ++i) {
      const auto& span = location->file.field_spans[i];
      if (span.kind != LOOM_LOCATION_FIELD_OPERAND || span.index == 0) {
        continue;
      }
      const size_t ordinal = span.index - 1;
      EXPECT_EQ(span.start_line, count - ordinal + 3);
      EXPECT_EQ(span.start_col, 2 + Name(ordinal).size() + 4);
      EXPECT_EQ(span.end_line, span.start_line);
      EXPECT_EQ(span.end_col - span.start_col, ordinal % 2 ? 4 : 5);
      ++parameter_spans;
    }
    EXPECT_EQ(parameter_spans, count);
    loom_module_free(module);
  }
}

TEST_F(OperandDictionaryParseTest, DuplicateRetainsFirstTokenAndErrorPriority) {
  for (size_t count : {1, 16, 17, 64, 65, 1024}) {
    SCOPED_TRACE(count);
    std::string source = Dictionary(count);
    source.erase(source.size() - 2);
    // Deliberately omit the value. Duplicate detection owns this diagnostic
    // before parsing the repeated entry's equals sign, SSA name, or type.
    source += ",\n  " + Name(count - 1) + "}\n";
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(Parse(source, &module));
    EXPECT_EQ(module, nullptr);
    ASSERT_EQ(capture_.diagnostics.size(), 1u);
    const auto& diagnostic = capture_.diagnostics[0];
    EXPECT_EQ(diagnostic.error,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 27));
    EXPECT_EQ(diagnostic.origin_line, count + 4);
    EXPECT_EQ(diagnostic.origin_column, 3u);
    ASSERT_EQ(diagnostic.related_locations.size(), 1u);
    const auto& previous = diagnostic.related_locations[0].source_location;
    EXPECT_EQ(previous.start_line, 4u);
    EXPECT_EQ(previous.start_column, 3u);
  }
}

TEST_F(OperandDictionaryParseTest, DictionaryScratchEndsBeforeTheNextField) {
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(Parse("", &module));
  ASSERT_NE(module, nullptr);
  loom_parser_scope_t scope = {};
  loom_parser_t parser = {
      .module = module,
      .context = &context_,
      .scope = &scope,
  };
  parser.definition_scope.pop_at = UINT16_MAX;
  iree_arena_initialize(&pool_, &parser.parser_arena);
  const char* names[] = {"even", "odd"};
  for (size_t i = 0; i < 2; ++i) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module, iree_make_cstring_view(names[i]), &name));
    IREE_ASSERT_OK(loom_parser_scope_define(
        &scope, &parser.parser_arena, name,
        loom_test_constant_result(loom_block_op(loom_module_block(module), i)),
        nullptr));
  }
  const auto* vtable =
      loom_context_resolve_op(&context_, LOOM_OP_TEST_OPERAND_DICT);
  const loom_format_element_t* element = nullptr;
  for (uint16_t i = 0; i < vtable->format_element_count; ++i) {
    if (vtable->format_elements[i].kind == LOOM_FORMAT_KIND_OPERAND_DICT) {
      element = &vtable->format_elements[i];
    }
  }
  ASSERT_NE(element, nullptr);
  loom_parsed_op_t parsed;
  loom_parsed_op_initialize(&parsed);
  const std::string source = Dictionary(1024);
  iree_host_size_t retained_scratch = 0;
  for (int iteration = 0; iteration < 16; ++iteration) {
    loom_parsed_op_reset(&parsed);
    IREE_ASSERT_OK(
        loom_parsed_op_set_operand(&parsed, &parser.parser_arena, 0,
                                   loom_test_constant_result(loom_block_op(
                                       loom_module_block(module), 0))));
    loom_tokenizer_initialize(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("dictionary.loom"), &parser.parser_arena, &parser.tokenizer);
    IREE_ASSERT_OK(
        loom_parse_format_operand_dict(&parser, vtable, element, &parsed));
    EXPECT_EQ(parser.error_count, 0u);
    EXPECT_TRUE(loom_tokenizer_at(&parser.tokenizer, LOOM_TOKEN_EOF));
    loom_tokenizer_deinitialize(&parser.tokenizer);
    if (iteration == 0) {
      retained_scratch = parser.parser_arena.used_allocation_size;
    }
    EXPECT_EQ(parser.parser_arena.used_allocation_size, retained_scratch);
  }
  iree_arena_deinitialize(&parser.parser_arena);
  loom_module_free(module);
}

TEST_F(OperandDictionaryParseTest, DiagnosticSinkFailureAfterSpillPropagates) {
  std::string dictionary = Dictionary(1024);
  dictionary.erase(dictionary.size() - 2);
  dictionary += ",\n  " + Name(1023) + "}\n";
  const std::string source =
      "%even = test.constant 2 : index\n"
      "%odd = test.constant 3 : index\n"
      "%result = test.operand_dict %even " +
      dictionary + " : index\n";
  loom_text_parse_options_t options = {};
  options.diagnostic_sink.fn = [](void*, const loom_diagnostic_t*) {
    return iree_make_status(IREE_STATUS_CANCELLED, "diagnostic sink stopped");
  };
  loom_module_t* module = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      loom_text_parse(iree_make_string_view(source.data(), source.size()),
                      IREE_SV("dictionary.loom"), &context_, &pool_, &options,
                      &module));
  EXPECT_EQ(module, nullptr);
}

class OperandTreeParseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("tree"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    parser_.module = module_;
    parser_.context = &context_;
    parser_.definition_scope.pop_at = UINT16_MAX;
    iree_arena_initialize(&pool_, &parser_.parser_arena);
    vtable_ = loom_context_resolve_op(&context_, LOOM_OP_TEST_OPERAND_TREE);
    element_ = &vtable_->format_elements[0];
    ASSERT_EQ(element_->kind, LOOM_FORMAT_KIND_TREE_REFS);
    loom_parsed_op_initialize(&parsed_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&parser_.parser_arena);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Parse(const std::string& source) {
    loom_parsed_op_reset(&parsed_);
    loom_tokenizer_initialize(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("tree.loom"), &parser_.parser_arena, &parser_.tokenizer);
    iree_status_t status =
        loom_parse_format_tree_refs(&parser_, vtable_, element_, &parsed_);
    if (iree_status_is_ok(status)) {
      EXPECT_TRUE(loom_tokenizer_at(&parser_.tokenizer, LOOM_TOKEN_EOF));
    }
    loom_tokenizer_deinitialize(&parser_.tokenizer);
    return status;
  }

  // Shared blocks for persistent shapes and short-lived parse stacks.
  iree_arena_block_pool_t pool_ = {};
  // Canonical parser metadata.
  loom_context_t context_ = {};
  // Owner of the copied shape attributes.
  loom_module_t* module_ = nullptr;
  // Reused parser state, excluding per-tree scratch.
  loom_parser_t parser_ = {};
  // Reused operation accumulator.
  loom_parsed_op_t parsed_ = {};
  // Borrowed generated operation metadata.
  const loom_op_vtable_t* vtable_ = nullptr;
  // First tree field from the generated format.
  const loom_format_element_t* element_ = nullptr;
};

TEST_F(OperandTreeParseTest, DeepGroupScratchEndsBeforeTheNextField) {
  const std::string source =
      std::string(4096, '[') + "none" + std::string(4096, ']');
  iree_host_size_t retained_scratch = 0;
  for (int iteration = 0; iteration < 16; ++iteration) {
    IREE_ASSERT_OK(Parse(source));
    ASSERT_EQ(parsed_.attribute_count, 1u);
    const auto shape = parsed_.attributes[0];
    ASSERT_EQ(shape.count, 4097u);
    for (uint16_t i = 0; i < 4096; ++i) {
      EXPECT_EQ(shape.i64_array[i], 7);
    }
    EXPECT_EQ(shape.i64_array[4096], 1);
    if (!iteration) {
      retained_scratch = parser_.parser_arena.used_allocation_size;
    }
    EXPECT_EQ(parser_.parser_arena.used_allocation_size, retained_scratch);
  }
}

TEST_F(OperandTreeParseTest, OversizedShapeFailsAtTheFieldBoundary) {
  const std::string source =
      std::string(UINT16_MAX, '[') + "none" + std::string(UINT16_MAX, ']');
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Parse(source));
}

}  // namespace
}  // namespace loom
