// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/template_sync.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/testing/test_file.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

class TemplateSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_CONFIG,
                                   loom_config_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_FUNC,
                                   loom_func_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_INDEX,
                                   loom_index_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_KERNEL,
                                   loom_kernel_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_PIPELINE,
                                   loom_pipeline_dialect_vtables));
    IREE_ASSERT_OK(RegisterDialect(&context_, LOOM_DIALECT_SCALAR,
                                   loom_scalar_dialect_vtables));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Build(const char* target_source, const char* template_source,
                      std::string* out_source, bool* out_changed) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);

    loom_test_file_t target_file = {};
    iree_status_t status = loom_test_file_parse(
        iree_make_cstring_view(target_source), &arena, &target_file);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    if (iree_status_is_ok(status)) {
      status = loom_check_template_sync_build_source(
          iree_make_cstring_view(target_source), &target_file,
          IREE_SV("target.loom-test"), iree_make_cstring_view(template_source),
          IREE_SV("template.loom-test"), &context_, &block_pool_, &arena,
          iree_allocator_system(), &builder, out_changed);
    }
    if (iree_status_is_ok(status)) {
      *out_source = std::string(iree_string_builder_buffer(&builder),
                                iree_string_builder_size(&builder));
    }

    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TemplateSyncTest, UsesTemplateCasesAndTargetPreamble) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "ignored expected\n"
      "\n"
      "// ====\n"
      "\n"
      "// RUN: verify\n"
      "func.def @beta() {\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("// TEMPLATE: loom/src/loom/test/corpus/vector/"
                        "arithmetic.loom-test\n"),
            std::string::npos);
  EXPECT_NE(result.find("// RUN: emit source-low "
                        "output=module\n"),
            std::string::npos);
  EXPECT_NE(result.find("func.def @alpha()"), std::string::npos);
  EXPECT_NE(result.find("func.def @beta()"), std::string::npos);
  EXPECT_EQ(result.find("ignored expected"), std::string::npos);
  EXPECT_EQ(result.find("// RUN: verify"), std::string::npos);
}

TEST_F(TemplateSyncTest, PreservesMatchingTargetEvidenceAndDirectives) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n"
      "// REQUIRES: fake-target\n"
      "// XFAIL: pending target support\n"
      "// ERROR@+1: \"unsupported\"\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old target evidence\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @stale() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "stale target evidence\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @beta() {\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("// REQUIRES: fake-target\n"), std::string::npos);
  EXPECT_NE(result.find("// XFAIL: pending target support\n"),
            std::string::npos);
  EXPECT_NE(result.find("// ERROR@+1: \"unsupported\"\n"), std::string::npos);
  EXPECT_NE(result.find("old target evidence\n"), std::string::npos);
  EXPECT_NE(result.find("func.def @beta()"), std::string::npos);
  EXPECT_EQ(result.find("func.def @stale()"), std::string::npos);
  EXPECT_EQ(result.find("stale target evidence"), std::string::npos);
}

TEST_F(TemplateSyncTest, ExcludesExactCasesAndPreservesRemainingEvidence) {
  const char* names[] = {"alpha", "beta", "gamma"};
  const char* cases[] = {
      "func.def @alpha() {\n}\n"
      "\n// ----\nalpha evidence\n\n",
      "// REQUIRES: fake-target\n"
      "// ERROR@+1: \"unsupported\"\n"
      "func.def @beta() {\n}\n"
      "\n// ----\nbeta evidence\n\n",
      "func.def @gamma() {\n}\n"
      "\n// ----\ngamma evidence\n\n",
  };
  const char* template_source =
      "func.def @alpha() {\n}\n"
      "\n// ====\n\nfunc.def @beta() {\n}\n"
      "\n// ====\n\nfunc.def @gamma() {\n}\n";
  for (size_t excluded = 0; excluded < IREE_ARRAYSIZE(names); ++excluded) {
    SCOPED_TRACE(names[excluded]);
    std::string preamble =
        "// TEMPLATE: corpus.loom-test\n"
        "// RUN: emit source-low output=low\n"
        "// TEMPLATE-EXCLUDE: @";
    preamble += names[excluded];
    preamble += " requires explicit CFG support\n\n";
    std::string target_source = preamble;
    std::string expected = preamble;
    for (size_t i = 0; i < IREE_ARRAYSIZE(cases); ++i) {
      if (i > 0) {
        target_source += "// ====\n\n";
      }
      target_source += cases[i];
      if (i == excluded) {
        continue;
      }
      if (expected != preamble) {
        expected += "// ====\n\n";
      }
      expected += cases[i];
    }
    std::string result;
    bool changed = false;
    IREE_ASSERT_OK(
        Build(target_source.c_str(), template_source, &result, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(result, expected);
    IREE_ASSERT_OK(Build(expected.c_str(), template_source, &result, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(result, expected);

    std::string extended_template = template_source;
    extended_template += "\n// ====\n\nfunc.def @delta() {\n}\n";
    IREE_ASSERT_OK(
        Build(expected.c_str(), extended_template.c_str(), &result, &changed));
    EXPECT_TRUE(changed);
    EXPECT_EQ(result, expected +
                          "// ====\n\n"
                          "func.def @delta() {\n}\n");
  }
}

TEST_F(TemplateSyncTest, MultipleExclusionsApplyWhenBootstrapping) {
  const char* preamble =
      "// TEMPLATE: corpus.loom-test\n"
      "// TEMPLATE-EXCLUDE: @first requires explicit CFG support\n"
      "// TEMPLATE-EXCLUDE: @last requires flat addressing\n\n";
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(preamble,
                       "func.def @first() {\n}\n"
                       "\n// ====\n\nfunc.def @middle() {\n}\n"
                       "\n// ====\n\nfunc.def @last() {\n}\n",
                       &result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(result, std::string(preamble) + "func.def @middle() {\n}\n");
}

TEST_F(TemplateSyncTest, ExclusionsRejectUnknownNamesAndEmptyCoverage) {
  for (const char* name : {"missing", "entry*", "entry"}) {
    SCOPED_TRACE(name);
    std::string target_source =
        "// TEMPLATE: corpus.loom-test\n// TEMPLATE-EXCLUDE: @";
    target_source += name;
    target_source += " requires explicit CFG support\n\n";
    std::string result;
    bool changed = false;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Build(target_source.c_str(),
                                "func.def @entry() {\n}\n", &result, &changed));
  }
}

TEST_F(TemplateSyncTest, UsesKernelDefAsCaseSymbol) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n",
      "// RUN: roundtrip\n"
      "\n"
      "kernel.def @entry() {\n"
      "  %c1 = index.constant 1 : index\n"
      "  kernel.launch.config workgroups(%c1, %c1, %c1) "
      "workgroup_size(%c1, %c1, %c1) : index\n"
      "} launch() {\n"
      "  %id = kernel.workgroup.id<x> : index\n"
      "  kernel.return\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("kernel.def @entry()"), std::string::npos);
}

TEST_F(TemplateSyncTest, IgnoresFuncDeclWhenSelectingCaseSymbol) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.decl @callee()\n"
      "func.def @entry() {\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("func.decl @callee()"), std::string::npos);
  EXPECT_NE(result.find("func.def @entry()"), std::string::npos);
}

TEST_F(TemplateSyncTest, SharedDeclarationsAndExpectationsDoNotAccumulate) {
  const char* template_source =
      "// RUN: verify\n\n"
      "func.decl @callee()\n"
      "func.def @entry() {\n"
      "  // ERROR@+1: \"source expectation\"\n"
      "  func.return\n"
      "}\n";
  const char* target_source =
      "// TEMPLATE: corpus.loom-test\n"
      "// RUN: verify\n\n"
      "func.decl @callee()\n"
      "func.def @entry() {\n"
      "  // ERROR@+1: \"target expectation\"\n"
      "  func.return\n"
      "}\n";
  std::string first_result;
  bool changed = true;
  IREE_ASSERT_OK(
      Build(target_source, template_source, &first_result, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(first_result, target_source);

  std::string second_result;
  IREE_ASSERT_OK(
      Build(first_result.c_str(), template_source, &second_result, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(second_result, first_result);
}

TEST_F(TemplateSyncTest, PreservesTargetAnnotationsAtAnchoredInputLines) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n"
      "func.def @alpha(%value: i32) -> (i32) {\n"
      "  // ERROR@+1: \"unsupported\"\n"
      "  %zero = scalar.constant 0 : i32\n"
      "  %result = scalar.addi %value, %zero : i32\n"
      "  func.return %result : i32\n"
      "}\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha(%value: i32) -> (i32) {\n"
      "  %zero = scalar.constant 0 : i32\n"
      "  %result = scalar.addi %value, %zero : i32\n"
      "  func.return %result : i32\n"
      "}\n",
      &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_NE(result.find("  // ERROR@+1: \"unsupported\"\n"
                        "  %zero = scalar.constant 0 : i32\n"),
            std::string::npos);
}

TEST_F(TemplateSyncTest, PreservesAnnotationsOnSourceDefinition) {
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @entry(%value: i64) {\n"
      "}\n";
  for (bool follows_definition : {false, true}) {
    std::string target_source =
        "// TEMPLATE: loom/src/loom/test/corpus/source_low/example.loom-test\n"
        "// RUN: emit source-low output=low\n"
        "\n";
    if (!follows_definition) {
      target_source += "// ERROR@+1: TARGET/033 {actual_type=\"i64\"}\n";
    }
    target_source += "func.def @entry(%value: i64) {\n";
    if (follows_definition) {
      target_source += "// ERROR@-1: TARGET/033 {actual_type=\"i64\"}\n";
    }
    target_source += "}\n";
    SCOPED_TRACE(target_source);

    std::string result;
    bool changed = true;
    IREE_ASSERT_OK(
        Build(target_source.c_str(), template_source, &result, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(result, target_source);
  }
}

TEST_F(TemplateSyncTest, RejectsAnnotationOnChangedSignature) {
  std::string result;
  bool changed = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Build("// TEMPLATE: "
            "loom/src/loom/test/corpus/source_low/example.loom-test\n"
            "// RUN: emit source-low output=low\n"
            "\n"
            "func.decl @target()\n"
            "// ERROR@+1: TARGET/033 {actual_type=\"i64\"}\n"
            "func.def target(@target) @entry(%value: i64) {\n"
            "}\n",
            "// RUN: roundtrip\n"
            "\n"
            "func.def @entry(%value: i32) {\n"
            "}\n",
            &result, &changed));
}

TEST_F(TemplateSyncTest, NewCasesHaveOnlyTemplateSource) {
  const char* preamble =
      "// TEMPLATE: corpus.loom-test\n"
      "// RUN: emit source-low target=vm:core output=low\n\n";
  std::string target_source =
      std::string(preamble) +
      "func.decl @local_target()\n"
      "func.def public retain target(@local_target) @alpha() {\n}\n"
      "\n// ----\nold target evidence\n";
  const char* template_source =
      "func.def @alpha() {\n}\n"
      "\n// ====\n\nfunc.def @beta() {\n}\n";
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(
      Build(target_source.c_str(), template_source, &result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(result, std::string(preamble) +
                        "func.def @alpha() {\n}\n"
                        "\n// ----\nold target evidence\n"
                        "\n// ====\n\nfunc.def @beta() {\n}\n");
  std::string second_result;
  IREE_ASSERT_OK(
      Build(result.c_str(), template_source, &second_result, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(second_result, result);
}

TEST_F(TemplateSyncTest, PreservesAnnotationLineOccurrence) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n"
      "func.def @alpha() {\n"
      "  func.return\n"
      "  // ERROR@+1: \"second return\"\n"
      "  func.return\n"
      "}\n";
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "  func.return\n"
      "  func.return\n"
      "}\n";

  std::string result;
  bool changed = true;
  IREE_ASSERT_OK(Build(target_source, template_source, &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_EQ(result, target_source);
}

TEST_F(TemplateSyncTest, ReportsUnchangedWhenConcreteFileIsCurrent) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";

  std::string result;
  bool changed = true;
  IREE_ASSERT_OK(Build(target_source, template_source, &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_EQ(result, target_source);
}

TEST_F(TemplateSyncTest, UpdatesHelpersAroundPublicEntry) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/source_low/callables.loom-test\n"
      "// RUN: emit source-low output=module\n"
      "\n"
      "func.decl @target()\n"
      "\n"
      "func.def @helper(%value: i32) -> (i32) {\n"
      "  func.return %value : i32\n"
      "}\n"
      "\n"
      "func.def public target(@target) @entry(%input: i32) -> (i32) {\n"
      "  %result = func.call @helper(%input) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"
      "\n"
      "// ----\n"
      "retained target expectation\n";
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @helper(%value: i32) -> (i32) {\n"
      "  %result = func.call @twice(%value) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"
      "\n"
      "func.def public @entry(%input: i32) -> (i32) {\n"
      "  %result = func.call @helper(%input) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n"
      "\n"
      "func.def @twice(%value: i32) -> (i32) {\n"
      "  %result = scalar.addi %value, %value : i32\n"
      "  func.return %result : i32\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def public @next_entry(%value: i32) -> (i32) {\n"
      "  func.return %value : i32\n"
      "}\n";

  std::string first_result;
  bool first_changed = false;
  IREE_ASSERT_OK(
      Build(target_source, template_source, &first_result, &first_changed));
  EXPECT_TRUE(first_changed);
  EXPECT_NE(first_result.find("func.def public @entry"), std::string::npos);
  EXPECT_NE(first_result.find("func.call @twice(%value)"), std::string::npos);
  EXPECT_EQ(first_result.find("func.def @helper"),
            first_result.rfind("func.def @helper"));
  EXPECT_NE(first_result.find("func.def public @next_entry"),
            std::string::npos);
  EXPECT_NE(first_result.find("retained target expectation"),
            std::string::npos);

  std::string second_result;
  bool second_changed = true;
  IREE_ASSERT_OK(Build(first_result.c_str(), template_source, &second_result,
                       &second_changed));
  EXPECT_FALSE(second_changed);
  EXPECT_EQ(second_result, first_result);
}

TEST_F(TemplateSyncTest, NewHelperCasePreservesSourceVisibility) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/source_low/callables.loom-test\n"
      "// RUN: emit source-low output=module\n\n"
      "func.decl @target()\n\n"
      "func.def target(@target) @first() {\n  func.return\n}\n";
  const char* template_source =
      "// RUN: roundtrip\n\n"
      "func.def @first() {\n  func.return\n}\n"
      "\n// ====\n\n"
      "func.def @helper() {\n  func.return\n}\n\n"
      "func.def public @entry() {\n"
      "  func.call @helper() : () -> ()\n"
      "  func.return\n}\n";
  std::string first_result;
  bool changed = false;
  IREE_ASSERT_OK(
      Build(target_source, template_source, &first_result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_NE(first_result.find("func.def public @entry"), std::string::npos);
  std::string second_result;
  IREE_ASSERT_OK(
      Build(first_result.c_str(), template_source, &second_result, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(second_result, first_result);
}

TEST_F(TemplateSyncTest, RejectsAmbiguousMultiFunctionCases) {
  for (const char* visibility : {"", "public "}) {
    const std::string template_source =
        std::string("// RUN: roundtrip\n\nfunc.def ") + visibility +
        "@alpha() {\n  func.return\n}\n\nfunc.def " + visibility +
        "@beta() {\n  func.return\n}\n";
    std::string result;
    bool changed = false;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Build("// TEMPLATE: "
              "loom/src/loom/test/corpus/source_low/callables.loom-test\n"
              "// RUN: emit source-low output=module\n\n",
              template_source.c_str(), &result, &changed));
  }
}

TEST_F(TemplateSyncTest, PreservesCaseCompileOptions) {
  const char* target_source =
      "// TEMPLATE: corpus.loom-test\n"
      "// RUN: emit source-low target=spirv:vulkan1.3+bda output=low\n\n"
      "// REQUIRES: fake-target\n"
      "// RUN: emit source-low target=spirv:vulkan1.3+bda+hal output=low\n"
      "// XFAIL: pending target support\n"
      "func.def @alpha() {\n}\n";
  const char* template_source = "// RUN: roundtrip\n\nfunc.def @alpha() {\n}\n";
  std::string result;
  bool changed = true;
  IREE_ASSERT_OK(Build(target_source, template_source, &result, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(result, target_source);
}

TEST_F(TemplateSyncTest, RejectsEmptyTemplate) {
  std::string result;
  bool changed = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Build("// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "// RUN: emit source-low output=module\n"
            "\n",
            "// RUN: roundtrip\n", &result, &changed));
}

TEST_F(TemplateSyncTest, RejectsDuplicateTemplateFunctions) {
  std::string result;
  bool changed = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Build("// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "// RUN: emit source-low output=module\n"
            "\n",
            "// RUN: roundtrip\n"
            "\n"
            "func.def @alpha() {\n"
            "}\n"
            "\n"
            "// ====\n"
            "\n"
            "func.def @alpha() {\n"
            "}\n",
            &result, &changed));
}

TEST_F(TemplateSyncTest, SourceOwnsDeclarationModifiersAndSignatures) {
  const char* preamble =
      "// TEMPLATE: corpus.loom-test\n"
      "// RUN: emit source-low target=vm:core output=low\n\n";
  const char* sources[] = {
      "func.def @entry() {\n  func.return\n}\n",
      "func.def public retain pure noinline @entry(\n"
      "    %integer: i32, %real: f32) -> (i32, f32) {\n"
      "  // Preserve both returned values and their distinct types.\n"
      "  func.return %integer, %real : i32, f32\n}\n",
      "func.def @entry() {\n  func.return\n}\n",
  };
  std::string result = std::string(preamble) + sources[0];
  for (const char* source : sources) {
    SCOPED_TRACE(source);
    std::string previous = result;
    bool changed = false;
    IREE_ASSERT_OK(Build(previous.c_str(), source, &result, &changed));
    EXPECT_EQ(result, std::string(preamble) + source);
    std::string repeated;
    IREE_ASSERT_OK(Build(result.c_str(), source, &repeated, &changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(repeated, result);
  }
}

TEST_F(TemplateSyncTest, CopiesScopedDefinitionsAndSourceComments) {
  const char* preamble = "// TEMPLATE: corpus.loom-test\n// RUN: roundtrip\n\n";
  std::string target_source =
      std::string(preamble) +
      "func.decl @local_target()\n"
      "pipeline.def retain target(@local_target) @entry() launch() {\n"
      "  pipeline.return\n}\n";
  const char* template_source =
      "// A scoped pipeline owns this source contract.\n"
      "pipeline.def <kernel> public retain @entry() launch() {\n"
      "  // The body is copied without rewriting its header.\n"
      "  pipeline.return\n}\n";
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(
      Build(target_source.c_str(), template_source, &result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(result, std::string(preamble) + template_source);
  std::string repeated;
  IREE_ASSERT_OK(Build(result.c_str(), template_source, &repeated, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(repeated, result);
}

TEST_F(TemplateSyncTest, SharedDeclarationsRemainDeclarations) {
  const char* preamble = "// TEMPLATE: corpus.loom-test\n// RUN: roundtrip\n\n";
  std::string target_source = std::string(preamble) +
                              "config.def @batch_size = 32 : index\n"
                              "func.decl @local_helper()\n"
                              "func.def @entry() {\n}\n";
  const char* template_source =
      "config.decl @batch_size : %value: index where [range(%value, 1, 16)]\n"
      "func.decl @shared_helper()\n"
      "func.def @entry() {\n}\n";
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(
      Build(target_source.c_str(), template_source, &result, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(result, std::string(preamble) + template_source);
  std::string repeated;
  IREE_ASSERT_OK(Build(result.c_str(), template_source, &repeated, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(repeated, result);
}

TEST_F(TemplateSyncTest, RejectsAnnotationOnRemovedSourceLine) {
  std::string result;
  bool changed = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Build("// TEMPLATE: corpus.loom-test\n// RUN: roundtrip\n\n"
            "// REMARK@+1: \"local declaration\"\n"
            "func.decl @local_target()\n"
            "func.def @entry() {\n}\n",
            "func.def @entry() {\n}\n", &result, &changed));
}

}  // namespace
}  // namespace loom
