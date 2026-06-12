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

static const loom_named_attr_t* FindDynamicParamName(
    const loom_module_t* module,
    const loom_encoding_define_param_view_t* params, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t entry_name = module->strings.entries[entry->name_id];
    if (iree_string_view_equal(entry_name, name)) return entry;
  }
  return nullptr;
}

static bool DynamicParamValue(const loom_encoding_define_param_view_t* params,
                              const loom_named_attr_t* name_entry,
                              loom_value_id_t* out_value) {
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

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

static iree_status_t EmitEncodingParamTypeError(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    loom_type_t actual_type, iree_string_view_t expected_type) {
  loom_diagnostic_param_t diagnostic_params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  loom_diagnostic_emission_t emission = {
      /*.op=*/op,
      /*.error=*/loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 9),
      /*.params=*/diagnostic_params,
      /*.param_count=*/IREE_ARRAYSIZE(diagnostic_params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t VerifyRequiresLayoutDefine(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  iree_string_view_t encoding_name = IREE_SV("requires_layout");
  iree_string_view_t layout_name = IREE_SV("layout");

  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!iree_string_view_equal(param_name, layout_name)) {
      return EmitEncodingParamError(
          emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8),
          encoding_name, param_name);
    }
  }

  const loom_named_attr_t* layout_entry =
      FindDynamicParamName(module, params, layout_name);
  if (!layout_entry) {
    return EmitEncodingParamError(
        emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
        encoding_name, layout_name);
  }

  loom_value_id_t layout_value = LOOM_VALUE_ID_INVALID;
  if (!DynamicParamValue(params, layout_entry, &layout_value)) {
    return iree_ok_status();
  }

  loom_type_t actual_type = loom_module_value_type(module, layout_value);
  if (!loom_type_is_encoding(actual_type)) {
    return EmitEncodingParamTypeError(emitter, op, encoding_name, layout_name,
                                      actual_type, IREE_SV("encoding"));
  }
  return iree_ok_status();
}

static const loom_encoding_vtable_t kRequiresLayoutEncodingVtable = {
    /*.name=*/IREE_SV("requires_layout"),
    /*.role=*/{},
    /*.verify=*/{},
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

}  // namespace
}  // namespace loom
