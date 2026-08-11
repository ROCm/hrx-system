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
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::GetStringParam;

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
      "module.import \"motif/format/ggml.loom\" [@decode_q6, @decode_q4]\n"
      "module.import \"target/amdgpu/format/ggml.loom\" [@decode_q4]\n";
  static const char kExpected[] =
      "module.import \"motif/format/ggml.loom\" [@decode_q4, @decode_q6]\n"
      "module.import \"target/amdgpu/format/ggml.loom\" [@decode_q4]\n";
  iree_string_view_t source =
      iree_make_string_view(kSource, IREE_ARRAYSIZE(kSource) - 1);

  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);
  VerifyOk(module);
  EXPECT_EQ(Print(module),
            std::string(kExpected, IREE_ARRAYSIZE(kExpected) - 1));

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

TEST_F(ModuleOpsTest, ParserDiagnosesDuplicateAnchorAtSecondOccurrence) {
  DiagnosticCapture capture;
  loom_text_parse_options_t options = {
      /*.diagnostic_sink=*/capture.sink(),
      /*.max_errors=*/20,
  };
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_text_parse(
      IREE_SV("module.import \"provider\" [@decode_q4, @decode_q4]\n"),
      IREE_SV("imports.loom"), &context_, &block_pool_, &options, &module));

  EXPECT_EQ(module, nullptr);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  ExpectError(capture.diagnostics[0],
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 35));
  EXPECT_EQ(GetStringParam(capture.diagnostics[0], 0), "decode_q4");
  ASSERT_EQ(capture.diagnostics[0].related_locations.size(), 1u);
  EXPECT_EQ(capture.diagnostics[0].related_locations[0].label,
            "previously listed here");
}

TEST_F(ModuleOpsTest, BuilderCanonicalizesAnchorsBeforeAllocatingImport) {
  loom_module_t* module = NULL;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                      NULL, iree_allocator_system(), &module));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t zeta_name_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("provider"), &provider_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("zeta"), &zeta_name_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("alpha"), &alpha_name_id));
  uint16_t zeta_symbol_id = LOOM_SYMBOL_ID_INVALID;
  uint16_t alpha_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, zeta_name_id, &zeta_symbol_id));
  IREE_ASSERT_OK(
      loom_module_add_symbol(module, alpha_name_id, &alpha_symbol_id));

  loom_symbol_ref_t refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/zeta_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
  };
  loom_op_t* import_op = NULL;
  IREE_ASSERT_OK(loom_module_import_build(
      &builder, provider_id,
      loom_make_symbol_ref_array(refs, IREE_ARRAYSIZE(refs)),
      LOOM_LOCATION_UNKNOWN, &import_op));
  ASSERT_NE(import_op, nullptr);
  loom_symbol_ref_array_t canonical_refs =
      loom_module_import_symbols(import_op);
  ASSERT_EQ(canonical_refs.count, 2u);
  EXPECT_EQ(canonical_refs.values[0].symbol_id, alpha_symbol_id);
  EXPECT_EQ(canonical_refs.values[1].symbol_id, zeta_symbol_id);

  loom_symbol_ref_t duplicates[] = {
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_module_import_build(
          &builder, provider_id,
          loom_make_symbol_ref_array(duplicates, IREE_ARRAYSIZE(duplicates)),
          LOOM_LOCATION_UNKNOWN, &import_op));
  EXPECT_EQ(loom_module_block(module)->op_count, 1u)
      << "duplicate rejection must happen before op allocation";

  loom_module_free(module);
}

TEST_F(ModuleOpsTest, VerifierRejectsNoncanonicalRawAnchorSet) {
  loom_module_t* module =
      Parse(IREE_SV("module.import \"provider\" [@alpha, @zeta]\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* import_op = loom_block_op(loom_module_block(module), 0);
  loom_symbol_ref_array_t canonical_refs =
      loom_module_import_symbols(import_op);
  ASSERT_EQ(canonical_refs.count, 2u);
  loom_symbol_ref_t reversed_refs[] = {
      canonical_refs.values[1],
      canonical_refs.values[0],
  };
  loom_op_attrs(import_op)[1] = loom_attr_symbol_set(
      reversed_refs, (uint16_t)IREE_ARRAYSIZE(reversed_refs));

  loom_verify_options_t options = {
      /*.sink=*/{NULL, NULL},
      /*.max_errors=*/20,
  };
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
  EXPECT_EQ(result.error_count, 1u);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
