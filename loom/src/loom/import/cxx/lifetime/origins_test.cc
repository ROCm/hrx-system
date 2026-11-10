// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/lifetime/origins.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/initialization.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

// Uses the production postorder contract. Tests inspect the resulting facts,
// without asking a query to revisit an expression's descendants.
class Expressions final : public cxx::ASTVisitor {
 public:
  explicit Expressions(Source& source)
      : types(source.unit(), source.diagnostics()),
        origins(source.unit(), types) {}

  void postVisit(cxx::AST* ast) override {
    if (auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast)) {
      origins.retain(expression);
    }
  }

  void visit(cxx::InitDeclaratorAST* ast) override {
    cxx::ASTVisitor::visit(ast);
    if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol)) {
      initializers.emplace(cxx::to_string(variable->name()), ast->initializer);
    }
  }

  void visit(cxx::BracedTypeConstructionAST* ast) override {
    constructions.push_back(ast);
    cxx::ASTVisitor::visit(ast);
  }

  void visit(cxx::ReturnStatementAST* ast) override {
    returned = ast->expression;
    cxx::ASTVisitor::visit(ast);
  }

  // Canonical admitted types, shared by every retained expression fact.
  Types types;
  // Analysis under test, with the same lifetime as its source unit.
  TemporaryOrigins origins;
  // Named declarations provide stable source roles to the assertions.
  std::unordered_map<std::string, cxx::ExpressionAST*> initializers;
  // Explicit construction sites, retained in source order.
  std::vector<cxx::BracedTypeConstructionAST*> constructions;
  // Last return operand, used by the single-return construction witness.
  cxx::ExpressionAST* returned = nullptr;
};

TEST(TemporaryOriginsTest, ConstructorSyntaxSharesOneResultObject) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    Guard copy(const Guard& input) { return input; }
  )"),
                IREE_SV("copy_result.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  auto* conversion =
      cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expressions.returned);
  ASSERT_NE(conversion, nullptr);
  ASSERT_NE(conversion->conversionFunction, nullptr);
  ASSERT_TRUE(conversion->conversionFunction->isConstructor());
  auto* result = expressions.origins.result(conversion);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result, expressions.origins.result(conversion->expression));
  auto* arguments =
      cxx::ast_cast<cxx::ParenInitializerAST>(conversion->expression);
  ASSERT_NE(arguments, nullptr);
  ASSERT_NE(arguments->expressionList, nullptr);
  EXPECT_EQ(expressions.origins.result(arguments->expressionList->value),
            nullptr);
}

TEST(TemporaryOriginsTest, EqualPayloadsRemainSeparateConstructionSites) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    void entry(unsigned digit) {
      const Guard& first = Guard{digit};
      const Guard& second = Guard{digit};
      unsigned scalar = digit;
    }
  )"),
                IREE_SV("distinct_objects.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 2u);
  auto* first = expressions.origins.result(expressions.constructions[0]);
  auto* second = expressions.origins.result(expressions.constructions[1]);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(expressions.origins.result(expressions.initializers.at("scalar")),
            nullptr);
}

TEST(TemporaryOriginsTest, ReferenceExtensionFollowsTheSourceLifetimeRules) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard {
      unsigned digit;
      Guard& self() { return *this; }
      ~Guard() {}
    };
    void entry(bool choose) {
      const Guard& direct = Guard{1};
      const unsigned& field = Guard{2}.digit;
      const Guard& forwarded = Guard{3}.self();
      const unsigned& comma = (Guard{4}, Guard{5}.digit);
      const unsigned& branch = choose ? Guard{6}.digit : Guard{7}.digit;
    }
  )"),
                IREE_SV("reference_lifetimes.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 7u);
  for (const auto& [name, initializer] : expressions.initializers) {
    expressions.origins.extend(initializer);
  }
  expressions.origins.finish();
  const bool extended[] = {true, true, false, false, true, true, true};
  for (size_t index = 0; index < expressions.constructions.size(); ++index) {
    SCOPED_TRACE(index);
    auto* origin = expressions.origins.result(expressions.constructions[index]);
    ASSERT_NE(origin, nullptr);
    EXPECT_EQ(expressions.origins.extended(origin), extended[index]);
  }
}

TEST(TemporaryOriginsTest, TrivialAggregateRetainsOnlyBracedReferenceBindings) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    struct Holder { const Guard& member; };
    void entry() {
      Holder braced{Guard{1}};
      Holder parenthesized(Guard{2});
    }
  )"),
                IREE_SV("aggregate_forms.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 2u);
  for (const auto& [name, initializer] : expressions.initializers) {
    EXPECT_EQ(expressions.origins.result(initializer), nullptr);
    expressions.origins.extend_members(initializer);
  }
  expressions.origins.finish();
  auto* braced = expressions.origins.result(expressions.constructions[0]);
  auto* parenthesized =
      expressions.origins.result(expressions.constructions[1]);
  ASSERT_NE(braced, nullptr);
  ASSERT_NE(parenthesized, nullptr);
  EXPECT_TRUE(expressions.origins.extended(braced));
  EXPECT_FALSE(expressions.origins.extended(parenthesized));
}

TEST(TemporaryOriginsTest, AggregateDependenciesFollowMembersAndArrayElements) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    struct Holder { const Guard& member; };
    struct Outer { Holder members[2]; };
    void entry() {
      Outer nested{{{Guard{1}}, {(Guard{2}, Guard{3})}}};
    }
  )"),
                IREE_SV("aggregate_dependencies.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 3u);
  expressions.origins.extend_members(expressions.initializers.at("nested"));
  expressions.origins.finish();
  const bool extended[] = {true, false, true};
  for (size_t index = 0; index < expressions.constructions.size(); ++index) {
    SCOPED_TRACE(index);
    auto* origin = expressions.origins.result(expressions.constructions[index]);
    ASSERT_NE(origin, nullptr);
    EXPECT_EQ(expressions.origins.extended(origin), extended[index]);
  }
}

TEST(TemporaryOriginsTest, CopiesAndConstructorArgumentsAddNoMemberLifetime) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    struct Holder { const Guard& member; };
    struct Constructed {
      const Guard& member;
      Constructed(const Guard& source) : member(source) {}
    };
    void entry() {
      Holder original{Guard{1}};
      Holder copied = original;
      Constructed constructed{Guard{2}};
    }
  )"),
                IREE_SV("copied_reference_members.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 2u);
  expressions.origins.extend_members(expressions.initializers.at("copied"));
  expressions.origins.extend_members(
      expressions.initializers.at("constructed"));
  expressions.origins.finish();
  for (auto* construction : expressions.constructions) {
    auto* origin = expressions.origins.result(construction);
    ASSERT_NE(origin, nullptr);
    EXPECT_FALSE(expressions.origins.extended(origin));
  }
}

TEST(TemporaryOriginsTest, ReferenceToAggregateExtendsItsMemberDependencies) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned digit; ~Guard() {} };
    struct Holder { const Guard& member; };
    void entry() {
      const Holder& owner = Holder{Guard{1}};
    }
  )"),
                IREE_SV("extended_aggregate.cxx"), options);
  Expressions expressions(source);
  expressions.accept(source.unit().ast());
  ASSERT_EQ(expressions.constructions.size(), 2u);
  expressions.origins.extend(expressions.initializers.at("owner"));
  expressions.origins.finish();
  // The outer Holder is trivial, so only its referenced Guard owns cleanup.
  EXPECT_EQ(expressions.origins.result(expressions.constructions[0]), nullptr);
  auto* guard = expressions.origins.result(expressions.constructions[1]);
  ASSERT_NE(guard, nullptr);
  EXPECT_TRUE(expressions.origins.extended(guard));
}

}  // namespace
}  // namespace loom::cxx_import
