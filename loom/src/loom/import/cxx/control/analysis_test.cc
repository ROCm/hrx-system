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
    // Proven constant upper bound, or no value for a runtime scalar bound.
    std::optional<unsigned> upper = std::nullopt;
  };
  const Case cases[] = {
      {"unsigned n", "for (unsigned i=0; i<n; ++i) {}", 1},
      {"unsigned n", "for (unsigned i=0; i<16u; i+=4u) {}", 4, 16},
      {"unsigned n", "for (unsigned i=0; i<4294967292u; i+=4u) {}", 4,
       4294967292u},
      {"unsigned n", "for (unsigned i=n; i<(320u/16u); i+=(1u<<2)) {}", 4, 20},
      {"unsigned n", "for (unsigned i=n; i<(capacity/16u); i+=stride) {}", 4,
       20},
      {"unsigned n", "for (unsigned i=n; (i<(20u)); (i)++) {}", 1, 20},
      {"unsigned n", "for (unsigned i=n; i<(0xffffffffu+21u); ++i) {}", 1, 20},
      {"unsigned n", "for (unsigned i=n; i<static_cast<unsigned>(20); ++i) {}",
       1, 20},
      {"unsigned n", "for (unsigned i=n; i<(unsigned char)276u; ++i) {}", 1,
       20},
      {"unsigned n", "for (unsigned i=n; i<(true ? 20u : 16u); ++i) {}", 1, 20},
      {"unsigned n", "for (unsigned i=n; i<0u; i+=(1u<<2)) {}", 4, 0},
      {"unsigned n", "for (unsigned i=n; i<1u; i+=~0u) {}", ~0u, 1},
      {"unsigned n", "for (unsigned i=n; i<~0u; ++i) {}", 1, ~0u},
      {"unsigned n", "for (unsigned i=n; i<(~0u-3u); i+=(1u<<2)) {}", 4,
       4294967292u},
      {"unsigned n", "for (unsigned i=n; i<~0u; i+=(1u<<2)) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20ULL; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20u; i+=(1ULL<<32)) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20u; i+=(1u/0u)) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20u; i+=(2147483647+1)) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20u; i+=(4u-4u)) {}", 0},
      {"unsigned n", "for (unsigned i=n; i<20u; i+=(-1)) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; i+=(4u-3u)) {}", 1},
      {"unsigned n", "for (unsigned i=0; i<n; i+=(4u-2u)) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<(n+0u); ++i) {}", 0},
      {"volatile unsigned n", "for (unsigned i=0; i<n; ++i) {}", 0},
      {"unsigned n", "for (volatile unsigned i=0; i<n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<(++n); ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<(n=20u); ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<(false ? ++n : 20u); ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<opaque(); ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<20u; i+=opaque()) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<(unsigned)sizeof(++n); ++i) {}", 1,
       4},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { sizeof(++n); }", 1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { (void)&n; }", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { (void)&i; }", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { sizeof(&i); }", 1},
      {"unsigned n",
       "for (unsigned i=0; i<n; ++i) { if constexpr(false) (void)&n; }", 1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) {} (void)&n;", 0},
      {"unsigned n", "for (unsigned i=0; i<4294967295u; i+=4u) {}", 0},
      {"unsigned long n", "for (unsigned i=0; i<n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { --n; }", 0},
      {"unsigned n",
       "for (unsigned i=0; i<n; ++i) { if constexpr(false) --n; }", 1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) { if constexpr(true) --n; }",
       0},
      {"unsigned n",
       "for (unsigned i=0; i<n; ++i) { if constexpr(--n; false) {} }", 0},
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
        std::string(
            "constexpr unsigned capacity=320; const unsigned stride=4; "
            "unsigned opaque(); void entry(") +
        test.parameter + ") {" + test.loop + "}";
    Source source(view(text), IREE_SV("loops.cpp"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    auto* loop =
        cxx::ast_cast<cxx::ForStatementAST>(body->statementList->value);
    ASSERT_NE(loop, nullptr);
    Types types(source.unit(), source.diagnostics());
    ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
    auto* counted = analysis.counted(loop);
    ASSERT_EQ(counted != nullptr, test.step != 0);
    if (counted) {
      EXPECT_EQ(counted->step, test.step);
      EXPECT_EQ(cxx::to_string(counted->induction->name()), "i");
      if (test.upper) {
        EXPECT_EQ(std::get<unsigned>(counted->upper), *test.upper);
      } else {
        const auto& bound = std::get<CountedLoop::Bound>(counted->upper);
        EXPECT_EQ(bound.binding, function->symbol->parameters()[0]);
        auto* condition =
            cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition);
        EXPECT_EQ(bound.expression, condition->rightExpression);
      }
      EXPECT_EQ(analysis.counted(loop), counted);
    }
  }
}

TEST(ControlFlowTest, SourceSelectionRetainsInitializerAndSelectedWrites) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV("void entry(int selected, int discarded, int initialized) {"
              "if constexpr (++initialized; false) ++discarded;"
              "else ++selected; }"),
      IREE_SV("selected.cpp"), options);
  auto* function = definition(source);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* branch = cxx::ast_cast<cxx::IfStatementAST>(body->statementList->value);
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  auto writes = analysis.written(branch);
  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[2]);
  EXPECT_EQ(writes[1], function->symbol->parameters()[0]);
  EXPECT_TRUE(std::ranges::equal(writes, analysis.written(body)));
}

TEST(ControlFlowTest, DecisionSyntaxAndInitializerWritesStayWithTheBinding) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV("void entry(int input) { if (int value = input++) ++value; "
              "if constexpr (const int value = 3) ++input; }"),
      IREE_SV("decisions.cpp"), options);
  auto* function = definition(source);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  for (auto* statement : cxx::ListView{body->statementList}) {
    auto* branch = cxx::ast_cast<cxx::IfStatementAST>(statement);
    ASSERT_NE(branch, nullptr);
    auto* declaration =
        analysis.condition_declaration(branch->decisionVariable);
    ASSERT_NE(declaration, nullptr);
    EXPECT_EQ(declaration->symbol, branch->decisionVariable);
    EXPECT_EQ(declaration->initializer,
              branch->decisionVariable->initializer());
    auto writes = analysis.written(branch);
    ASSERT_FALSE(writes.empty());
    EXPECT_EQ(writes.front(), function->symbol->parameters()[0]);
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
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  auto writes = analysis.written(branch);
  ASSERT_EQ(writes.size(), 3u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[1]);
  EXPECT_EQ(writes[1], function->symbol->parameters()[0]);
  EXPECT_NE(writes[2], writes[1]);
  EXPECT_EQ(cxx::to_string(writes[2]->name()), "x");
  EXPECT_TRUE(std::ranges::equal(writes, analysis.written(body)));
  EXPECT_TRUE(analysis.written(branch->condition).empty());
}

TEST(ControlFlowTest, NestedMemberWritesRetainTheOwningBindingAndSlice) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Payload { int value; const int* cursor; };
    struct Envelope { int prefix; Payload payload; };
    void entry(Envelope state, bool condition) {
      if (condition) {
        ++(state.payload.value);
        state.payload.cursor += 1;
      }
    }
  )"),
                IREE_SV("member_writes.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* branch = cxx::ast_cast<cxx::IfStatementAST>(body->statementList->value);
  ASSERT_NE(branch, nullptr);
  auto* arm = cxx::ast_cast<cxx::CompoundStatementAST>(branch->statement);
  ASSERT_NE(arm, nullptr);
  auto* increment_statement =
      cxx::ast_cast<cxx::ExpressionStatementAST>(arm->statementList->value);
  auto* increment =
      cxx::ast_cast<cxx::UnaryExpressionAST>(increment_statement->expression);
  ASSERT_NE(increment, nullptr);
  auto* assignment_statement = cxx::ast_cast<cxx::ExpressionStatementAST>(
      arm->statementList->next->value);
  auto* assignment = cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(
      assignment_statement->expression);
  ASSERT_NE(assignment, nullptr);
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  auto writes = analysis.written(branch);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[0]);
  EXPECT_TRUE(std::ranges::equal(writes, analysis.written(body)));
  auto scalar = analysis.destination(increment->expression);
  ASSERT_TRUE(scalar.has_value());
  EXPECT_EQ(scalar->binding, writes[0]);
  EXPECT_EQ(scalar->component_offset, 1u);
  EXPECT_EQ(scalar->member, &kSSAPartition);
  auto pointer = analysis.destination(assignment->targetExpression);
  ASSERT_TRUE(pointer.has_value());
  EXPECT_EQ(pointer->binding, writes[0]);
  EXPECT_EQ(pointer->component_offset, 2u);
  EXPECT_EQ(pointer->member, &kPointerPartition);
}

TEST(ControlFlowTest, MemoryMembersRetainStorageIdentityAndAddressSideEffects) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Payload { volatile unsigned value; };
    struct Envelope { unsigned prefix; Payload payload; };
    void entry(Envelope* packets, unsigned index) {
      packets[index++].payload.value += 1;
      (*packets).payload.value += 2;
      (packets++)->payload.value += 3;
    }
  )"),
                IREE_SV("memory_members.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  auto writes = analysis.written(body);
  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[1]);
  EXPECT_EQ(writes[1], function->symbol->parameters()[0]);
  for (auto* statement : cxx::ListView{body->statementList}) {
    auto* expression = cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
    ASSERT_NE(expression, nullptr);
    auto* assignment = cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(
        expression->expression);
    ASSERT_NE(assignment, nullptr);
    auto* member =
        cxx::ast_cast<cxx::MemberExpressionAST>(assignment->targetExpression);
    ASSERT_NE(member, nullptr);
    EXPECT_FALSE(analysis.destination(member).has_value());
    EXPECT_TRUE(analysis.storage_backed(member));
    auto* nested =
        cxx::ast_cast<cxx::MemberExpressionAST>(member->baseExpression);
    ASSERT_NE(nested, nullptr);
    EXPECT_TRUE(analysis.storage_backed(nested));
  }
}

TEST(ControlFlowTest, ReturnSummariesRetainFallthroughAndNestedExits) {
  struct Case {
    // Source body whose nested outcomes are aggregated once.
    const char* body;
    // Expected function-body return summary.
    ExitFlow flow;
  };
  const Case cases[] = {
      {"", ExitFlow::None},
      {"return;", ExitFlow::All},
      {"if (x) return;", ExitFlow::Some},
      {"if (x) return; else return;", ExitFlow::All},
      {"if (x) { if (y) return; }", ExitFlow::Some},
      {"if (x) { if (y) return; } return;", ExitFlow::All},
      {"while (x) { return; }", ExitFlow::Some},
      {"do { return; } while (x);", ExitFlow::All},
      {"for (unsigned i=0; i<4u; ++i) { if (y) return; }", ExitFlow::Some},
      {"while (x) { --x; }", ExitFlow::None},
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
    Types types(source.unit(), source.diagnostics());
    ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
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
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
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

TEST(ControlFlowTest, IterationExitsStopAtTheirLoopAndExcludeUnreachablePaths) {
  struct Case {
    // Statements inside the outer counted loop's body.
    const char* body;
    // Paths that continue the outer loop.
    ExitFlow continues;
    // Paths that return from the enclosing function.
    ExitFlow returns;
  };
  const Case cases[] = {
      {"", ExitFlow::None, ExitFlow::None},
      {"continue;", ExitFlow::All, ExitFlow::None},
      {"if (x) continue;", ExitFlow::Some, ExitFlow::None},
      {"if (x) continue; else continue;", ExitFlow::All, ExitFlow::None},
      {"if constexpr(true) continue; else return;", ExitFlow::All,
       ExitFlow::None},
      {"if constexpr(false) continue; else return;", ExitFlow::None,
       ExitFlow::All},
      {"if constexpr(false) continue;", ExitFlow::None, ExitFlow::None},
      {"if constexpr(true) { if (x) continue; } else return;", ExitFlow::Some,
       ExitFlow::None},
      {"if (x) { if (y) continue; } else { if (y) continue; }", ExitFlow::Some,
       ExitFlow::None},
      {"if (x) continue; return;", ExitFlow::Some, ExitFlow::Some},
      {"continue; return;", ExitFlow::All, ExitFlow::None},
      {"return; continue;", ExitFlow::None, ExitFlow::All},
      {"while (x) { --x; continue; }", ExitFlow::None, ExitFlow::None},
      {"do { --x; continue; } while (x);", ExitFlow::None, ExitFlow::None},
      {"for (unsigned j=0; j<4u; ++j) { if (y) continue; }", ExitFlow::None,
       ExitFlow::None},
      {"while (x) { continue; return; }", ExitFlow::None, ExitFlow::None},
      {"do { if (x) continue; return; } while (y);", ExitFlow::None,
       ExitFlow::Some},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.body);
    std::string text = std::string("void entry(int x, int y) { ") +
                       "for (unsigned i=0; i<4u; ++i) { " + test.body + " } }";
    Source source(view(text), IREE_SV("continue.cpp"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    auto* loop =
        cxx::ast_cast<cxx::ForStatementAST>(body->statementList->value);
    ASSERT_NE(loop, nullptr);
    Types types(source.unit(), source.diagnostics());
    ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
    EXPECT_EQ(analysis.continues(loop->statement), test.continues);
    EXPECT_EQ(analysis.returns(loop->statement), test.returns);
    EXPECT_EQ(analysis.continues(loop), ExitFlow::None);
    EXPECT_EQ(analysis.continues(body), ExitFlow::None);
    EXPECT_NE(analysis.counted(loop), nullptr);
  }
}

}  // namespace
}  // namespace loom::cxx_import
