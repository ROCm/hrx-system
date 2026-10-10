// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format_signatures.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/verify/verify.h"

namespace {

// Build signatures without source names: parsing authored text first would
// supply the binders whose absence this API roundtrip exercises.
class FormatSignaturesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_config_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_CONFIG,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("signatures"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(reparsed_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    uint16_t symbol_id;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {0, symbol_id};
  }

  void SetName(loom_value_id_t value, const char* name) {
    loom_string_id_t name_id;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    IREE_ASSERT_OK(loom_module_set_value_name(module_, value, name_id));
  }

  loom_op_t* Config(const char* name) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_config_decl_build(
        &builder_, 0, Symbol(name), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void ConstrainConfig(loom_op_t* op) {
    loom_predicate_t* predicate = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&module_->arena, sizeof(*predicate),
                                       (void**)&predicate));
    *predicate = {
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_CONST},
        .reserved = {},
        .args = {loom_config_decl_type(op), 1, 4},
    };
    IREE_ASSERT_OK(loom_config_decl_set_predicates(
        module_, op, loom_attr_predicate_list(predicate, 1)));
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &output,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return text;
  }

  void RoundTrip() {
    ASSERT_NO_FATAL_FAILURE(Verify(module_));
    std::string text = Print(module_);
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    IREE_ASSERT_OK(loom_text_parse(
        iree_make_string_view(text.data(), text.size()),
        IREE_SV("signatures.loom"), &context_, &pool_, &options, &reparsed_));
    ASSERT_NE(reparsed_, nullptr) << text;
    ASSERT_NO_FATAL_FAILURE(Verify(reparsed_));
    EXPECT_EQ(Print(reparsed_), text);
  }

  void CheckConfigRange(loom_op_t* op) {
    loom_attribute_t predicates = loom_config_decl_predicates(op);
    ASSERT_EQ(predicates.count, 1u);
    const loom_predicate_t& range = predicates.predicate_list[0];
    EXPECT_EQ(range.kind, LOOM_PREDICATE_RANGE);
    EXPECT_EQ(range.arg_count, 3);
    EXPECT_EQ(range.arg_tags[0], LOOM_PRED_ARG_VALUE);
    EXPECT_EQ(range.args[0], loom_config_decl_type(op));
    EXPECT_EQ(range.args[1], 1);
    EXPECT_EQ(range.args[2], 4);
  }

  // Shared arena storage for the built and reparsed modules.
  iree_arena_block_pool_t pool_;
  // Minimal dialect registry for config symbols and signature syntax.
  loom_context_t context_;
  // Builder-produced module with genuinely unnamed signature results.
  loom_module_t* module_ = nullptr;
  // Reparsed module owned until fixture teardown.
  loom_module_t* reparsed_ = nullptr;
  // Current insertion point in the builder-produced module.
  loom_builder_t builder_;
};

TEST_F(FormatSignaturesTest, AnonymousConfigPredicateRetainsRange) {
  ConstrainConfig(Config("configured.workgroup_count.x"));
  EXPECT_EQ(Print(module_),
            "config.decl @configured.workgroup_count.x : %0: index "
            "where [range(%0, 1, 4)]\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  CheckConfigRange(loom_block_op(loom_module_block(reparsed_), 0));
}

TEST_F(FormatSignaturesTest, UnreferencedConfigResultStaysUnbound) {
  Config("extent");
  EXPECT_EQ(Print(module_), "config.decl @extent : index\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
}

TEST_F(FormatSignaturesTest, ExplicitUnreferencedResultNameIsPreserved) {
  SetName(loom_config_decl_type(Config("extent")), "size");
  EXPECT_EQ(Print(module_), "config.decl @extent : %size: index\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
}

TEST_F(FormatSignaturesTest, RemovingPredicateDropsAnonymousBinder) {
  loom_op_t* config = Config("extent");
  ConstrainConfig(config);
  EXPECT_NE(Print(module_).find(": %0: index"), std::string::npos);
  IREE_ASSERT_OK(
      loom_config_decl_set_predicates(module_, config, loom_attribute_t{}));
  EXPECT_EQ(Print(module_), "config.decl @extent : index\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
}

TEST_F(FormatSignaturesTest, GeneratedBinderAvoidsExplicitNumericName) {
  ConstrainConfig(Config("first"));
  loom_op_t* second = Config("second");
  ConstrainConfig(second);
  SetName(loom_config_decl_type(second), "0");
  const std::string text = Print(module_);
  EXPECT_NE(text.find("@first : %$0: index where [range(%$0, 1, 4)]"),
            std::string::npos);
  EXPECT_NE(text.find("@second : %0: index where [range(%0, 1, 4)]"),
            std::string::npos);
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  CheckConfigRange(loom_block_op(loom_module_block(reparsed_), 0));
  CheckConfigRange(loom_block_op(loom_module_block(reparsed_), 1));
}

TEST_F(FormatSignaturesTest, AnonymousResultBindsDependentResultDimension) {
  loom_type_t types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), 0)};
  loom_op_t* declaration = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, 0, Symbol("shape"), nullptr, 0, types,
      IREE_ARRAYSIZE(types), nullptr, 0, LOOM_LOCATION_UNKNOWN, &declaration));
  const loom_value_id_t* results = loom_op_const_results(declaration);
  IREE_ASSERT_OK(loom_module_set_value_type(
      module_, results[1],
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(results[0]), 0)));
  EXPECT_EQ(Print(module_),
            "test.decl @shape() -> (%0: index, tensor<[%0]xf32>)\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  const loom_value_id_t* parsed_results =
      loom_op_const_results(loom_block_op(loom_module_block(reparsed_), 0));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(reparsed_, parsed_results[1]),
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(parsed_results[0]), 0)));
}

TEST_F(FormatSignaturesTest, TiedAndDuplicateNamedResultsKeepIdentity) {
  const loom_type_t argument_type = loom_type_buffer();
  loom_type_t types[] = {argument_type,
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* declaration = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, 0, Symbol("results"), &argument_type, 1, types,
      IREE_ARRAYSIZE(types), &tie, 1, LOOM_LOCATION_UNKNOWN, &declaration));
  SetName(loom_op_const_operands(declaration)[0], "input");
  const loom_value_id_t* results = loom_op_const_results(declaration);
  SetName(results[0], "output");
  SetName(results[1], "extent");
  SetName(results[2], "extent");
  const std::string text = Print(module_);
  EXPECT_NE(text.find("%output: %input as buffer"), std::string::npos);
  EXPECT_NE(text.find("%extent$" + std::to_string(results[1]) + ": index"),
            std::string::npos);
  EXPECT_NE(text.find("%extent$" + std::to_string(results[2]) + ": index"),
            std::string::npos);
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  loom_op_t* parsed = loom_block_op(loom_module_block(reparsed_), 0);
  ASSERT_EQ(parsed->tied_result_count, 1u);
  EXPECT_EQ(loom_op_tied_results(parsed)[0].result_index, 0u);
  EXPECT_EQ(loom_op_tied_results(parsed)[0].operand_index, 0u);
}

TEST_F(FormatSignaturesTest, AnonymousTiedResultBindsDependentDimension) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t types[] = {
      index, loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                 loom_dim_pack_static(4), 0)};
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* declaration = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, 0, Symbol("shape"), &index, 1, types,
      IREE_ARRAYSIZE(types), &tie, 1, LOOM_LOCATION_UNKNOWN, &declaration));
  SetName(loom_op_const_operands(declaration)[0], "input");
  const loom_value_id_t* results = loom_op_const_results(declaration);
  IREE_ASSERT_OK(loom_module_set_value_type(
      module_, results[1],
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(results[0]), 0)));
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  loom_op_t* parsed = loom_block_op(loom_module_block(reparsed_), 0);
  ASSERT_EQ(parsed->tied_result_count, 1u);
  EXPECT_EQ(loom_op_tied_results(parsed)[0].result_index, 0u);
  EXPECT_EQ(loom_op_tied_results(parsed)[0].operand_index, 0u);
  const loom_value_id_t* parsed_results = loom_op_const_results(parsed);
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(reparsed_, parsed_results[1]),
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(parsed_results[0]), 0)));
}

TEST_F(FormatSignaturesTest, NamedTiePreservesTypeChange) {
  const loom_type_t argument_type = loom_type_buffer();
  const loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = true,
  };
  loom_op_t* declaration = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(&builder_, 0, 0, 0, Symbol("window"),
                                      &argument_type, 1, &result_type, 1, &tie,
                                      1, LOOM_LOCATION_UNKNOWN, &declaration));
  SetName(loom_op_const_operands(declaration)[0], "input");
  SetName(loom_op_const_results(declaration)[0], "output");
  EXPECT_EQ(Print(module_),
            "test.decl @window(%input: buffer) -> "
            "(%output: %input as view<16xf32>)\n");
  ASSERT_NO_FATAL_FAILURE(RoundTrip());
  loom_op_t* parsed = loom_block_op(loom_module_block(reparsed_), 0);
  ASSERT_EQ(parsed->tied_result_count, 1u);
  EXPECT_TRUE(loom_op_tied_results(parsed)[0].has_type_change);
}

}  // namespace
