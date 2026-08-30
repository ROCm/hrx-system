// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_defs.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "loom/error/error_catalog.h"

namespace {

TEST(ErrorDefsTest, LookupType001) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_domain(def), LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(loom_error_def_code(def), 1);
  EXPECT_EQ(loom_error_def_severity(def), LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(loom_error_def_summary(def), "SameType constraint violated.");
  EXPECT_EQ(def->param_count, 4);
}

TEST(ErrorDefsTest, LookupFold005) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_FOLD, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_domain(def), LOOM_ERROR_DOMAIN_FOLD);
  EXPECT_EQ(loom_error_def_code(def), 5);
  EXPECT_EQ(loom_error_def_severity(def), LOOM_DIAGNOSTIC_REMARK);
  EXPECT_NE(loom_error_def_message_template(def), nullptr);
  EXPECT_EQ(def->param_count, 3);
}

TEST(ErrorDefsTest, LookupBytecode007) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_domain(def), LOOM_ERROR_DOMAIN_BYTECODE);
  EXPECT_EQ(loom_error_def_code(def), 7);
  EXPECT_EQ(loom_error_def_severity(def), LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(loom_error_def_summary(def), "Invalid bytecode range.");
  ASSERT_EQ(def->param_count, 4);
  EXPECT_STREQ(loom_error_def_param_name(def, 1), "offset");
  EXPECT_EQ(loom_error_def_param_kind(def, 1), LOOM_PARAM_U64);
}

TEST(ErrorDefsTest, LookupBackendPressurePeakRemark) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_domain(def), LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(loom_error_def_code(def), 3);
  EXPECT_EQ(loom_error_def_severity(def), LOOM_DIAGNOSTIC_REMARK);
  EXPECT_STREQ(loom_error_def_summary(def), "Register pressure peak observed.");
  ASSERT_EQ(def->param_count, 10);
  EXPECT_STREQ(loom_error_def_param_name(def, 4), "value_class");
  EXPECT_EQ(loom_error_def_param_kind(def, 4), LOOM_PARAM_STRING);
  EXPECT_STREQ(loom_error_def_param_name(def, 5), "budget");
  EXPECT_EQ(loom_error_def_param_kind(def, 5), LOOM_PARAM_U32);
  EXPECT_STREQ(loom_error_def_param_name(def, 6), "peak");
  EXPECT_EQ(loom_error_def_param_kind(def, 6), LOOM_PARAM_U32);
  EXPECT_STREQ(loom_error_def_param_name(def, 9), "contributors");
  EXPECT_EQ(loom_error_def_param_kind(def, 9), LOOM_PARAM_STRING_LIST);
}

TEST(ErrorDefsTest, LookupBackendAllocationError) {
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_domain(def), LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(loom_error_def_code(def), 5);
  EXPECT_EQ(loom_error_def_severity(def), LOOM_DIAGNOSTIC_ERROR);
  EXPECT_STREQ(loom_error_def_summary(def), "Register allocation failed.");
  EXPECT_NE(loom_error_def_fix_hint_template(def), nullptr);
}

TEST(ErrorDefsTest, LookupNonExistentReturnsNull) {
  EXPECT_EQ(loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 999), nullptr);
  EXPECT_EQ(loom_error_def_lookup(LOOM_ERROR_DOMAIN_FOLD, 0), nullptr);
}

TEST(ErrorDefsTest, LookupComposedCatalogFallsBackAndShadows) {
  const loom_error_def_t* expected =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(expected, nullptr);

  uint16_t error_indices[] = {UINT16_MAX, UINT16_MAX};
  loom_error_catalog_t catalog = {};
  catalog.error_indices_by_code = error_indices;
  catalog.domain_spans[LOOM_ERROR_DOMAIN_TYPE].code_count =
      IREE_ARRAYSIZE(error_indices);
  catalog.fallback_catalog = &loom_error_catalog_core;

  EXPECT_EQ(loom_error_catalog_lookup(&catalog, LOOM_ERROR_DOMAIN_TYPE, 1),
            expected);

  loom_error_def_t shadow = *expected;
  shadow.catalog = &catalog;
  error_indices[1] = 0;
  catalog.string_data = expected->catalog->string_data;
  catalog.error_defs = &shadow;
  catalog.param_defs = expected->catalog->param_defs;
  EXPECT_EQ(loom_error_catalog_lookup(&catalog, LOOM_ERROR_DOMAIN_TYPE, 1),
            &shadow);
}

TEST(ErrorDefsTest, LookupDeepFallbackChain) {
  const loom_error_def_t* expected =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(expected, nullptr);

  constexpr size_t kCatalogCount = 16384;
  std::vector<loom_error_catalog_t> catalogs(kCatalogCount);
  for (size_t i = 0; i + 1 < catalogs.size(); ++i) {
    catalogs[i].fallback_catalog = &catalogs[i + 1];
  }
  catalogs.back().fallback_catalog = &loom_error_catalog_core;

  EXPECT_EQ(
      loom_error_catalog_lookup(&catalogs.front(), LOOM_ERROR_DOMAIN_TYPE, 1),
      expected);
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
  EXPECT_STREQ(loom_error_def_param_name(def, 0), "field_a");
  EXPECT_EQ(loom_error_def_param_kind(def, 0), LOOM_PARAM_STRING);
  EXPECT_STREQ(loom_error_def_param_name(def, 1), "type_a");
  EXPECT_EQ(loom_error_def_param_kind(def, 1), LOOM_PARAM_TYPE);
}

TEST(ErrorDefsTest, NoParamsErrorHasEmptyParamSpan) {
  // ERR_PARSE_005 has no params.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->param_count, 0);
}

TEST(ErrorDefsTest, FixHintNullWhenEmpty) {
  // ERR_STRUCTURE_001 has no fix hint.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1);
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(loom_error_def_fix_hint_template(def), nullptr);
}

TEST(ErrorDefsTest, FixHintPresent) {
  // ERR_TYPE_001 has a fix hint.
  const loom_error_def_t* def =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  ASSERT_NE(def, nullptr);
  ASSERT_NE(loom_error_def_fix_hint_template(def), nullptr);
  EXPECT_TRUE(strstr(loom_error_def_fix_hint_template(def), "same type") !=
              nullptr);
}

}  // namespace
