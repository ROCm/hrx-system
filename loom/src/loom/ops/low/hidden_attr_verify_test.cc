// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectFieldRefParam;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

class LowHiddenAttrVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_target_core_test_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_module_t* ParseSource(const char* source) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &parse_options.low_asm_environment);

    loom_module_t* module = NULL;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("low_hidden_attr_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_TRUE(parse_capture.diagnostics.empty());
    return module;
  }

  loom_verify_result_t VerifyParsedModule(loom_module_t* module,
                                          DiagnosticCapture* capture) {
    capture->Reset();
    loom_verify_options_t verify_options = {};
    verify_options.sink = capture->sink();
    verify_options.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &result));
    return result;
  }

  static loom_op_t* FirstLowFuncBodyOp(loom_module_t* module,
                                       bool (*predicate)(const loom_op_t*)) {
    loom_op_t* module_op = NULL;
    loom_block_for_each_op(loom_module_block(module), module_op) {
      if (!loom_low_func_def_isa(module_op)) continue;
      loom_region_t* body = loom_low_func_def_body(module_op);
      if (!body || body->block_count == 0) return NULL;
      loom_op_t* body_op = NULL;
      loom_block_for_each_op(loom_region_entry_block(body), body_op) {
        if (predicate(body_op)) return body_op;
      }
    }
    return NULL;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_;
};

TEST_F(LowHiddenAttrVerifyTest, DescriptorOrdinalRejectsInvalidNegative) {
  loom_module_t* module = ParseSource(
      "test.target<low_core> @target\n"
      "low.func.def target(@target) @add(%lhs: reg<test.i32>, "
      "%rhs: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.return %sum : reg<test.i32>\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  loom_op_t* packet_op = FirstLowFuncBodyOp(module, loom_low_op_isa);
  ASSERT_NE(packet_op, nullptr);
  loom_op_attrs(packet_op)[loom_low_op_descriptor_ordinal_ATTR_INDEX] =
      loom_attr_i64(-2);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "descriptor_ordinal");
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_op_descriptor_ordinal_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "-1 or a non-negative uint32");
}

TEST_F(LowHiddenAttrVerifyTest, LiveInSourceIdMatchesSourceStableId) {
  loom_module_t* module = ParseSource(
      "test.target<low_core> @target\n"
      "low.func.def target(@target) @live_in() -> (reg<test.i32>) {\n"
      "  %arg0 = low.live_in<test.arg0> : reg<test.i32>\n"
      "  low.return %arg0 : reg<test.i32>\n"
      "}\n");
  ASSERT_NE(module, nullptr);
  loom_op_t* live_in_op = FirstLowFuncBodyOp(module, loom_low_live_in_isa);
  ASSERT_NE(live_in_op, nullptr);
  loom_op_attrs(live_in_op)[loom_low_live_in_source_id_ATTR_INDEX] =
      loom_attr_i64(0);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "source_id");
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_live_in_source_id_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "stable ID derived from key");
}

}  // namespace
}  // namespace loom
