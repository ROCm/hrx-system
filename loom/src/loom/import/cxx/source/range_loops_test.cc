// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/range_loops.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/symbols.h>

#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class Ranges final : public cxx::ASTVisitor {
 public:
  void visit(cxx::ForRangeStatementAST* ast) override { loops.push_back(ast); }

  // Authored loop nodes borrowed from the source unit.
  std::vector<cxx::ForRangeStatementAST*> loops;
};

TEST(RangeLoopsTest, IndexesFrontendLoweringWithoutRebuildingIt) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    void entry(int (&values)[3]) {
      for (int increment = 2; int& value : values) value += increment;
    }
  )"),
                IREE_SV("range.cxx"), options);
  Ranges ranges;
  ranges.accept(source.unit().ast());
  ASSERT_EQ(ranges.loops.size(), 1u);
  auto* loop = ranges.loops.front();
  auto* original =
      cxx::ast_cast<cxx::SimpleDeclarationAST>(loop->rangeDeclaration);
  ASSERT_NE(original, nullptr);
  auto* original_declarator = original->initDeclaratorList->value;
  ASSERT_EQ(original_declarator->initializer, nullptr);
  auto* original_body = loop->statement;

  RangeLoops expansions;
  auto* scope = expansions.expand(loop);
  EXPECT_EQ(scope, loop->range.statement);
  EXPECT_EQ(expansions.scope(loop), scope);
  auto* children = scope->statementList;
  ASSERT_NE(children, nullptr);
  EXPECT_EQ(children->value, loop->initializer);
  children = children->next;
  for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
    ASSERT_NE(children, nullptr);
    auto* statement =
        cxx::ast_cast<cxx::DeclarationStatementAST>(children->value);
    ASSERT_NE(statement, nullptr);
    auto* declaration =
        cxx::ast_cast<cxx::SimpleDeclarationAST>(statement->declaration);
    ASSERT_NE(declaration, nullptr);
    auto* declarator = declaration->initDeclaratorList->value;
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
    ASSERT_NE(variable, nullptr);
    EXPECT_EQ(declarator->initializer, nullptr);
    EXPECT_NE(variable->initializer(), nullptr);
    EXPECT_EQ(expansions.initializes_range(declarator), ordinal == 0);
    children = children->next;
  }
  ASSERT_NE(children, nullptr);
  ASSERT_EQ(children->next, nullptr);
  auto* counted = cxx::ast_cast<cxx::ForStatementAST>(children->value);
  ASSERT_NE(counted, nullptr);
  EXPECT_NE(counted->condition, nullptr);
  EXPECT_NE(counted->expression, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementAST>(counted->statement);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body->statementList->next->value, original_body);
  EXPECT_EQ(body->statementList->next->next, nullptr);
  auto* element =
      cxx::ast_cast<cxx::DeclarationStatementAST>(body->statementList->value);
  ASSERT_NE(element, nullptr);
  auto* declaration =
      cxx::ast_cast<cxx::SimpleDeclarationAST>(element->declaration);
  ASSERT_NE(declaration, nullptr);
  EXPECT_EQ(declaration, original);
  auto* declarator = declaration->initDeclaratorList->value;
  EXPECT_EQ(declarator->symbol, original_declarator->symbol);
  EXPECT_EQ(declarator, original_declarator);
  EXPECT_NE(
      cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol)->initializer(),
      nullptr);
  EXPECT_EQ(loop->rangeDeclaration, original);
  EXPECT_EQ(loop->statement, original_body);
  EXPECT_EQ(original_declarator->initializer, nullptr);
}

TEST(RangeLoopsTest, PreservesStructuredBindingIdentityAndIterationScope) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    void entry(int (&rows)[2][2]) {
      for (auto& [first, second] : rows) first += second;
    }
  )"),
                IREE_SV("bindings.cxx"), options);
  Ranges ranges;
  ranges.accept(source.unit().ast());
  ASSERT_EQ(ranges.loops.size(), 1u);
  auto* loop = ranges.loops.front();
  auto* original = cxx::ast_cast<cxx::StructuredBindingDeclarationAST>(
      loop->rangeDeclaration);
  ASSERT_NE(original, nullptr);
  auto* original_hidden = original->binding.hiddenVariable;
  auto* original_initializer = original_hidden->initializer;
  RangeLoops expansions;
  auto* scope = expansions.expand(loop);
  auto* tail = scope->statementList->next->next->next;
  auto* iteration = cxx::ast_cast<cxx::ForStatementAST>(tail->value);
  ASSERT_NE(iteration, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementAST>(iteration->statement);
  ASSERT_NE(body, nullptr);
  auto* statement =
      cxx::ast_cast<cxx::DeclarationStatementAST>(body->statementList->value);
  ASSERT_NE(statement, nullptr);
  auto* binding = cxx::ast_cast<cxx::StructuredBindingDeclarationAST>(
      statement->declaration);
  ASSERT_NE(binding, nullptr);
  EXPECT_EQ(binding, original);
  EXPECT_EQ(binding->binding.bindingDeclaratorList,
            original->binding.bindingDeclaratorList);
  EXPECT_EQ(binding->binding.hiddenVariable, original_hidden);
  EXPECT_NE(cxx::symbol_cast<cxx::VariableSymbol>(original_hidden->symbol)
                ->initializer(),
            nullptr);
  EXPECT_EQ(original_hidden->initializer, original_initializer);
  EXPECT_EQ(loop->rangeDeclaration, original);
}

}  // namespace
}  // namespace loom::cxx_import
