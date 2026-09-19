// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/initialization.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <algorithm>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

cxx::FunctionDefinitionAST* definition(Source& source) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    if (auto* function =
            cxx::ast_cast<cxx::FunctionDefinitionAST>(declaration)) {
      return function;
    }
  }
  return nullptr;
}

TEST(ControlFlowTest, CountedAdmissionRetainsTheNonwrappingIntervalProof) {
  struct Case {
    // Source parameter type determines the comparison's promoted width.
    const char* parameter;
    // Standalone source loop whose classification is under test.
    const char* loop;
    // Expected positive counted step, or zero for ordinary while semantics.
    unsigned step;
  };
  const Case cases[] = {
      {"unsigned n", "for (unsigned i=0; i<n; ++i) {}", 1},
      {"unsigned n", "for (unsigned i=0; i<16u; i+=4u) {}", 4},
      {"unsigned n", "for (unsigned i=0; i<4294967292u; i+=4u) {}", 4},
      {"unsigned n", "for (unsigned i=0; i<4294967295u; i+=4u) {}", 0},
      {"unsigned long n", "for (unsigned i=0; i<n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { --n; }", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { ++i; }", 0},
      {"unsigned n", "for (unsigned i=0; i<n; --i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<=n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; i+=0u) {}", 0},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.loop);
    std::string text =
        std::string("void entry(") + test.parameter + ") {" + test.loop + "}";
    Source source(view(text), IREE_SV("loops.cpp"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    auto* loop =
        cxx::ast_cast<cxx::ForStatementAST>(body->statementList->value);
    ASSERT_NE(loop, nullptr);
    ControlFlow analysis(source.unit(), body);
    auto* counted = analysis.counted(loop);
    ASSERT_EQ(counted != nullptr, test.step != 0);
    if (counted) {
      EXPECT_EQ(counted->step, test.step);
      EXPECT_EQ(cxx::to_string(counted->induction->name()), "i");
      auto* condition =
          cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition);
      EXPECT_EQ(counted->upper, condition->rightExpression);
      EXPECT_EQ(analysis.counted(loop), counted);
    }
  }
}

TEST(ControlFlowTest, NestedWritesPreserveOrderAndShadowedSymbolIdentity) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    void entry(int x, int y) {
      if (x) {
        y = 1;
        x += 2;
        y++;
        { int x = 0; ++x; }
        while (y) { --y; }
      }
    }
  )"),
                IREE_SV("writes.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* branch = cxx::ast_cast<cxx::IfStatementAST>(body->statementList->value);
  ASSERT_NE(branch, nullptr);
  ControlFlow analysis(source.unit(), body);
  auto writes = analysis.written(branch);
  ASSERT_EQ(writes.size(), 3u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[1]);
  EXPECT_EQ(writes[1], function->symbol->parameters()[0]);
  EXPECT_NE(writes[2], writes[1]);
  EXPECT_EQ(cxx::to_string(writes[2]->name()), "x");
  EXPECT_TRUE(std::ranges::equal(writes, analysis.written(body)));
  EXPECT_TRUE(analysis.written(branch->condition).empty());
}

TEST(ControlFlowTest, ReturnSummariesRetainFallthroughAndNestedExits) {
  struct Case {
    // Source body whose nested outcomes are aggregated once.
    const char* body;
    // Expected function-body return summary.
    ReturnFlow flow;
  };
  const Case cases[] = {
      {"", ReturnFlow::None},
      {"return;", ReturnFlow::All},
      {"if (x) return;", ReturnFlow::Some},
      {"if (x) return; else return;", ReturnFlow::All},
      {"if (x) { if (y) return; }", ReturnFlow::Some},
      {"if (x) { if (y) return; } return;", ReturnFlow::All},
      {"while (x) { return; }", ReturnFlow::Some},
      {"do { return; } while (x);", ReturnFlow::All},
      {"for (unsigned i=0; i<4u; ++i) { if (y) return; }", ReturnFlow::Some},
      {"while (x) { --x; }", ReturnFlow::None},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.body);
    std::string text =
        std::string("void entry(int x, int y) {") + test.body + "}";
    Source source(view(text), IREE_SV("returns.cpp"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    ControlFlow analysis(source.unit(), body);
    EXPECT_EQ(analysis.returns(body), test.flow);
  }
}

TEST(ControlFlowTest, ConditionalValuesRetainOrderedBindingMutations) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    unsigned entry(unsigned x, unsigned y, unsigned choose) {
      return choose++ ? ((x)++ && ++(y)) : (x++ || y--);
    }
  )"),
                IREE_SV("conditional.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* ret =
      cxx::ast_cast<cxx::ReturnStatementAST>(body->statementList->value);
  ASSERT_NE(ret, nullptr);
  auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(
      cxx::Initializer::stripImplicitCasts(ret->expression));
  ASSERT_NE(select, nullptr);
  ControlFlow analysis(source.unit(), body);
  auto writes = analysis.written(select);
  ASSERT_EQ(writes.size(), 3u);
  auto parameters = function->symbol->parameters();
  EXPECT_EQ(writes[0], parameters[2]);
  EXPECT_EQ(writes[1], parameters[0]);
  EXPECT_EQ(writes[2], parameters[1]);
  for (auto* arm : {select->iftrueExpression, select->iffalseExpression}) {
    auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(
        cxx::Initializer::stripImplicitCasts(arm));
    ASSERT_NE(nested, nullptr);
    auto arm_writes = analysis.written(nested->expression);
    ASSERT_EQ(arm_writes.size(), 2u);
    EXPECT_EQ(arm_writes[0], parameters[0]);
    EXPECT_EQ(arm_writes[1], parameters[1]);
  }
}

}  // namespace
}  // namespace loom::cxx_import
