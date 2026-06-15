// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_defs.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ErrorDefsTest, LookupType001) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(def->code, 1);
  EXPECT_EQ(def->severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(def->summary, "SameType constraint violated.");
  EXPECT_EQ(def->param_count, 4);
}

TEST(ErrorDefsTest, LookupFold005) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_FOLD, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->domain, LOOM_ERROR_DOMAIN_FOLD);
  EXPECT_EQ(def->code, 5);
  EXPECT_EQ(def->severity, LOOM_DIAGNOSTIC_REMARK);
  EXPECT_NE(def->message_template, nullptr);
  EXPECT_EQ(def->param_count, 3);
}

TEST(ErrorDefsTest, LookupBytecode007) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->domain, LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(def->code, 7);
  EXPECT_EQ(def->severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(def->summary, "Invalid bytecode range.");
  ASSERT_EQ(def->param_count, 4);
  EXPECT_STREQ(def->param_defs[1].name, "offset");
  EXPECT_EQ(def->param_defs[1].kind, LOOM_PARAM_U64);
}

TEST(ErrorDefsTest, LookupBackendPressurePeakRemark) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(def->code, 3);
  EXPECT_EQ(def->severity, LOOM_DIAGNOSTIC_REMARK);
  EXPECT_STREQ(def->summary, "Register pressure peak observed.");
  ASSERT_EQ(def->param_count, 10);
  EXPECT_STREQ(def->param_defs[4].name, "value_class");
  EXPECT_EQ(def->param_defs[4].kind, LOOM_PARAM_STRING);
  EXPECT_STREQ(def->param_defs[5].name, "budget");
  EXPECT_EQ(def->param_defs[5].kind, LOOM_PARAM_U32);
  EXPECT_STREQ(def->param_defs[6].name, "peak");
  EXPECT_EQ(def->param_defs[6].kind, LOOM_PARAM_U32);
  EXPECT_STREQ(def->param_defs[9].name, "contributors");
  EXPECT_EQ(def->param_defs[9].kind, LOOM_PARAM_STRING_LIST);
}

TEST(ErrorDefsTest, LookupBackendAllocationError) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(def->code, 5);
  EXPECT_EQ(def->severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(def->summary, "Register allocation failed.");
  EXPECT_NE(def->fix_hint_template, nullptr);
}

TEST(ErrorDefsTest, LookupNonExistentReturnsNull) {
  EXPECT_EQ(loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 999), nullptr);
  EXPECT_EQ(loom_error_def_lookup(LOOM_ERROR_DOMAIN_FOLD, 0), nullptr);
}

TEST(ErrorDefsTest, GlobalLookupExcludesOptionalTargetCatalogs) {
  EXPECT_EQ(loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, 1), nullptr);
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_WASM, 1)),
      nullptr);
}

TEST(ErrorDefsTest, LookupRefsResolve) {
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TYPE, 1)),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_BYTECODE, 7)),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7));
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_FOLD, 5)),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_FOLD, 5));
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_PARSE, 3)),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3));
  EXPECT_EQ(
      loom_error_def_lookup_ref(LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_BACKEND, 3)),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3));
}

TEST(ErrorDefsTest, EmptyRefReturnsNull) {
  EXPECT_EQ(loom_error_def_lookup_ref(LOOM_ERROR_REF_NONE), nullptr);
}

TEST(ErrorDefsTest, DomainNames) {
  EXPECT_STREQ(loom_error_domain_name(LOOM_ERROR_DOMAIN_TYPE), "TYPE");
  EXPECT_STREQ(loom_error_domain_name(LOOM_ERROR_DOMAIN_SHAPE), "SHAPE");
  EXPECT_STREQ(loom_error_domain_name(LOOM_ERROR_DOMAIN_FOLD), "FOLD");
  EXPECT_STREQ(loom_error_domain_name(LOOM_ERROR_DOMAIN_BACKEND), "BACKEND");
}

TEST(ErrorDefsTest, EmitterNames) {
  EXPECT_STREQ(loom_emitter_name(LOOM_EMITTER_VERIFIER), "verifier");
  EXPECT_STREQ(loom_emitter_name(LOOM_EMITTER_PARSER), "parser");
  EXPECT_STREQ(loom_emitter_name(LOOM_EMITTER_BYTECODE_READER),
               "bytecode_reader");
}

TEST(ErrorDefsTest, SeverityNames) {
  EXPECT_STREQ(loom_diagnostic_severity_name(LOOM_DIAGNOSTIC_ERROR), "error");
  EXPECT_STREQ(loom_diagnostic_severity_name(LOOM_DIAGNOSTIC_WARNING),
               "warning");
  EXPECT_STREQ(loom_diagnostic_severity_name(LOOM_DIAGNOSTIC_REMARK), "remark");
}

TEST(ErrorDefsTest, ParamDefsValid) {
  // Spot-check that param_defs arrays have expected content.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(def, nullptr);
  ASSERT_EQ(def->param_count, 4);
  EXPECT_STREQ(def->param_defs[0].name, "field_a");
  EXPECT_EQ(def->param_defs[0].kind, LOOM_PARAM_STRING);
  EXPECT_STREQ(def->param_defs[1].name, "type_a");
  EXPECT_EQ(def->param_defs[1].kind, LOOM_PARAM_TYPE);
}

TEST(ErrorDefsTest, NoParamsErrorHasNullParamDefs) {
  // ERR_PARSE_005 has no params.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->param_count, 0);
  EXPECT_EQ(def->param_defs, nullptr);
}

TEST(ErrorDefsTest, FixHintNullWhenEmpty) {
  // ERR_STRUCTURE_001 has no fix hint.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->fix_hint_template, nullptr);
}

TEST(ErrorDefsTest, FixHintPresent) {
  // ERR_TYPE_001 has a fix hint.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(def, nullptr);
  ASSERT_NE(def->fix_hint_template, nullptr);
  EXPECT_TRUE(strstr(def->fix_hint_template, "same type") != nullptr);
}

}  // namespace
