// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/archive.h>
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/private/semantic_codec.h>
#include <cxx/symbols.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class BranchVisitor final : public cxx::ASTVisitor {
 public:
  void visit(cxx::IfStatementAST* ast) override { branches.push_back(ast); }

  // Source branches in syntax visitation order, borrowed from the source arena.
  std::vector<cxx::IfStatementAST*> branches;
};

class DecisionVisitor final : public cxx::ASTVisitor {
 public:
  void visit(cxx::IfStatementAST* ast) override {
    if (ast->decisionVariable) {
      variables.push_back(ast->decisionVariable);
    }
    cxx::ASTVisitor::visit(ast);
  }
  void visit(cxx::SwitchStatementAST* ast) override {
    if (ast->decisionVariable) {
      variables.push_back(ast->decisionVariable);
    }
    cxx::ASTVisitor::visit(ast);
  }
  void visit(cxx::WhileStatementAST* ast) override {
    if (ast->decisionVariable) {
      variables.push_back(ast->decisionVariable);
    }
    cxx::ASTVisitor::visit(ast);
  }
  void visit(cxx::ForStatementAST* ast) override {
    if (ast->decisionVariable) {
      variables.push_back(ast->decisionVariable);
    }
    cxx::ASTVisitor::visit(ast);
  }
  void visit(cxx::ConditionExpressionAST* ast) override {
    declarations.push_back(ast);
  }
  void visit(cxx::IdExpressionAST* ast) override {
    references.push_back(ast->symbol);
  }

  // Statements retain the same symbols as their nested declaration syntax.
  std::vector<cxx::VariableSymbol*> variables;
  // Original declaration nodes under contextual conversions, borrowed from AST.
  std::vector<cxx::ConditionExpressionAST*> declarations;
  // Name references within an initializer under inspection.
  std::vector<cxx::Symbol*> references;
};

TEST(ConstexprTest, SelectionSurvivesSourceDestructionArchivesAndClones) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  std::vector<std::uint8_t> bytes;
  {
    Source source(
        IREE_SV("void known() { if constexpr(const int chosen = 3) {} "
                "if constexpr(false) {} }"
                "void runtime(bool value) { if (value) {} }"
                "void loops(int input) { while(int n=input--) {} "
                "for(;int n=input--;--n) {} "
                "switch(int n=input) { default: break; } }"
                "template<bool Value> void deferred() { "
                "if constexpr(Value) {} }"),
        IREE_SV("branches.cpp"), options);
    cxx::ArchiveWriter writer;
    cxx::SemanticArchiveRoots roots;
    roots.globalScope = source.unit().globalScope();
    roots.ast = source.unit().ast();
    cxx::SemanticEncoder encoder(&source.unit());
    ASSERT_TRUE(encoder(roots, writer));
    bytes = writer();
  }
  Source destination(IREE_SV(""), IREE_SV("destination.cpp"), options);
  cxx::ArchiveReader reader;
  ASSERT_TRUE(reader(bytes)) << reader.error();
  cxx::SemanticArchiveRoots restored;
  cxx::SemanticDecoder decoder(&destination.unit());
  ASSERT_TRUE(decoder(reader, restored)) << decoder.error();
  for (auto* ast :
       {restored.ast, restored.ast->clone(destination.unit().arena())}) {
    BranchVisitor visitor;
    visitor.accept(ast);
    ASSERT_EQ(visitor.branches.size(), 4u);
    EXPECT_EQ(visitor.branches[0]->constexprValue, std::optional<bool>(true));
    EXPECT_EQ(visitor.branches[1]->constexprValue, std::optional<bool>(false));
    EXPECT_EQ(visitor.branches[2]->constexprValue, std::nullopt);
    EXPECT_EQ(visitor.branches[3]->constexprValue, std::nullopt);
    EXPECT_TRUE(visitor.branches[3]->constexprLoc);
    auto* selected = visitor.branches[0];
    ASSERT_NE(selected->decisionVariable, nullptr);
    DecisionVisitor decisions;
    decisions.accept(selected->condition);
    ASSERT_EQ(decisions.declarations.size(), 1u);
    EXPECT_EQ(decisions.declarations[0]->symbol, selected->decisionVariable);
    EXPECT_NE(selected->decisionVariable->initializer(), nullptr);
    EXPECT_EQ(visitor.branches[1]->decisionVariable, nullptr);
    DecisionVisitor all_decisions;
    all_decisions.accept(ast);
    ASSERT_EQ(all_decisions.variables.size(), 4u);
    ASSERT_EQ(all_decisions.declarations.size(), 4u);
    for (size_t i = 0; i < all_decisions.variables.size(); ++i) {
      EXPECT_EQ(all_decisions.variables[i],
                all_decisions.declarations[i]->symbol);
    }
  }
}

TEST(ConstexprTest, InstantiationRetainsSelectionAndOmitsTheDiscardedArm) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV(
          "template<bool Value> int select() { "
          "if constexpr(const bool chosen = sizeof(chosen) ? Value : false) "
          "return 11; "
          "else return 22; }"
          "int entry() { return select<true>() + select<false>(); }"),
      IREE_SV("specializations.cpp"), options);
  auto symbols = source.unit().globalScope()->find("select");
  ASSERT_FALSE(symbols.begin() == symbols.end());
  auto functions = cxx::views::each_function(*symbols.begin());
  ASSERT_EQ(std::ranges::distance(functions), 1);
  auto* pattern = *functions.begin();
  ASSERT_TRUE(pattern->isTemplatePattern());
  ASSERT_EQ(pattern->specializations().size(), 2u);
  BranchVisitor primary;
  primary.accept(pattern->declaration());
  ASSERT_EQ(primary.branches.size(), 1u);
  EXPECT_EQ(primary.branches.front()->constexprValue, std::nullopt);
  std::vector<bool> selections;
  for (const auto& specialization : pattern->specializations()) {
    auto* function =
        cxx::symbol_cast<cxx::FunctionSymbol>(specialization.symbol);
    ASSERT_NE(function, nullptr);
    BranchVisitor visitor;
    visitor.accept(function->declaration());
    ASSERT_EQ(visitor.branches.size(), 1u);
    auto* branch = visitor.branches.front();
    ASSERT_NE(branch->decisionVariable, nullptr);
    EXPECT_NE(branch->decisionVariable, primary.branches[0]->decisionVariable);
    DecisionVisitor decisions;
    decisions.accept(branch->condition);
    ASSERT_EQ(decisions.declarations.size(), 1u);
    EXPECT_EQ(decisions.declarations[0]->symbol, branch->decisionVariable);
    DecisionVisitor initializer;
    initializer.accept(branch->decisionVariable->initializer());
    EXPECT_NE(
        std::ranges::find(initializer.references, branch->decisionVariable),
        initializer.references.end());
    ASSERT_TRUE(branch->constexprValue.has_value());
    selections.push_back(*branch->constexprValue);
    EXPECT_EQ(branch->statement != nullptr, *branch->constexprValue);
    EXPECT_EQ(branch->elseStatement != nullptr, !*branch->constexprValue);
  }
  EXPECT_EQ(selections, (std::vector<bool>{true, false}));
}

}  // namespace
}  // namespace loom::cxx_import
