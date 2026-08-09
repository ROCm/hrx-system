// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/func/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectTypeParam;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

static iree_status_t EmitEncodingParamError(iree_diagnostic_emitter_t emitter,
                                            const loom_op_t* op,
                                            const loom_error_def_t* error,
                                            iree_string_view_t encoding_name,
                                            iree_string_view_t param_name) {
  loom_diagnostic_param_t diagnostic_params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
  };
  loom_diagnostic_emission_t emission = {
      /*.op=*/op,
      /*.error=*/error,
      /*.params=*/diagnostic_params,
      /*.param_count=*/IREE_ARRAYSIZE(diagnostic_params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t VerifyRequiresLayoutDefine(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter) {
  if (!loom_encoding_define_has_dynamic_parameter(params, 0)) {
    return EmitEncodingParamError(
        emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
        IREE_SV("requires_layout"), IREE_SV("layout"));
  }
  return iree_ok_status();
}

static const loom_encoding_dynamic_parameter_descriptor_t
    kRequiresLayoutDynamicParameters[] = {{
        /*.name=*/LOOM_BSTRING_REF(6, "layout"),
        /*.type_constraint=*/LOOM_TYPE_CONSTRAINT_ANY_ENCODING,
    }};
static const loom_encoding_family_descriptor_t kRequiresLayoutDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(15, "requires_layout"),
    /*.role=*/LOOM_ENCODING_ROLE_UNKNOWN,
    /*.parameter_count=*/{},
    /*.parameter_descriptors=*/{},
    /*.dynamic_parameter_count=*/
    IREE_ARRAYSIZE(kRequiresLayoutDynamicParameters),
    /*.dynamic_parameter_descriptors=*/kRequiresLayoutDynamicParameters,
};
static const loom_encoding_vtable_t kRequiresLayoutEncodingVtable = {
    /*.descriptor=*/&kRequiresLayoutDescriptor,
    /*.is_static_valid=*/{},
    /*.diagnose_static=*/{},
    /*.verify_define=*/VerifyRequiresLayoutDefine,
};

class EncodingVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kRequiresLayoutEncodingVtable));
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

  void VerifySource(const char* source, DiagnosticCapture* capture,
                    loom_verify_result_t* result) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 100;

    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_text_parse(IREE_SV(source), IREE_SV("test.loom"),
                                   &context_, &block_pool_, &parse_options,
                                   &module));
    ASSERT_TRUE(parse_capture.diagnostics.empty());

    capture->Reset();
    loom_verify_options_t verify_options = {};
    verify_options.sink = capture->sink();
    verify_options.max_errors = 100;
    IREE_ASSERT_OK(loom_verify_module(module, &verify_options, result));
    loom_module_free(module);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(EncodingVerifyTest, CustomVerifierRejectsMissingParam) {
  DiagnosticCapture capture;
  loom_verify_result_t result;
  VerifySource(
      "func.def @missing() {\n"
      "  %enc = encoding.define #requires_layout : encoding<schema>\n"
      "  func.return\n"
      "}\n",
      &capture, &result);

  ASSERT_EQ(result.error_count, 1u);
  const auto* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "requires_layout");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "layout");
}

TEST_F(EncodingVerifyTest, CustomVerifierRejectsWrongParamType) {
  DiagnosticCapture capture;
  loom_verify_result_t result;
  VerifySource(
      "func.def @wrong_type(%x: index) {\n"
      "  %enc = encoding.define #requires_layout "
      "{layout = %x : index} : encoding<schema>\n"
      "  func.return\n"
      "}\n",
      &capture, &result);

  ASSERT_EQ(result.error_count, 1u);
  const auto* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 9));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 9),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "requires_layout");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "layout");
  ExpectTypeParam(*diagnostic, 2, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "encoding");
}

TEST_F(EncodingVerifyTest, CustomVerifierRejectsUnknownParam) {
  DiagnosticCapture capture;
  loom_verify_result_t result;
  VerifySource(
      "func.def @unknown(%layout: encoding<layout>, %x: index) {\n"
      "  %enc = encoding.define #requires_layout "
      "{bogus = %x : index, layout = %layout : encoding<layout>} : "
      "encoding<schema>\n"
      "  func.return\n"
      "}\n",
      &capture, &result);

  ASSERT_EQ(result.error_count, 1u);
  const auto* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "requires_layout");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "bogus");
}

TEST_F(EncodingVerifyTest, UnusedMalformedStaticEncodingIsDiagnosed) {
  DiagnosticCapture capture;
  loom_verify_result_t result;
  VerifySource("#bad = #q8_0<block=\"thirty_two\">\n", &capture, &result);

  ASSERT_EQ(result.error_count, 1u);
  const auto* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 10));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 10),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "q8_0");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "block");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "integer");
}

}  // namespace
}  // namespace loom
