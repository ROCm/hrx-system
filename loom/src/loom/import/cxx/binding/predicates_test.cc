// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/predicates.h"

#include <cxx/ast.h>
#include <cxx/decl.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <string_view>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

cxx::ExpressionAST* where_expression(
    cxx::TranslationUnit& unit,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes) {
  cxx::ExpressionAST* result = nullptr;
  visit_loom_attributes(
      unit, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "where") {
          return;
        }
        auto* clause = attribute->attributeArgumentClause;
        auto* arguments = clause ? clause->expressionList : nullptr;
        if (arguments && !arguments->next) {
          result = arguments->value;
        }
      });
  return result;
}

cxx::ExpressionAST* function_where(Source& source, std::string_view name) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration);
    if (!simple) {
      continue;
    }
    for (auto* declarator : cxx::ListView{simple->initDeclaratorList}) {
      auto* function =
          cxx::symbol_cast<cxx::FunctionSymbol>(declarator->symbol);
      if (!function || !function->name() ||
          cxx::to_string(function->name()) != name) {
        continue;
      }
      auto* prototype = cxx::getFunctionPrototype(declarator->declarator);
      return prototype
                 ? where_expression(source.unit(), prototype->attributeList)
                 : nullptr;
    }
  }
  return nullptr;
}

cxx::ExpressionAST* variable_where(Source& source, std::string_view name) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration);
    if (!simple) {
      continue;
    }
    for (auto* declarator : cxx::ListView{simple->initDeclaratorList}) {
      auto* variable =
          cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
      if (variable && variable->name() &&
          cxx::to_string(variable->name()) == name) {
        return where_expression(source.unit(), simple->attributeList);
      }
    }
  }
  return nullptr;
}

loom_cxx_import_options_t options() {
  loom_cxx_import_options_t result;
  loom_cxx_import_options_initialize(&result);
  return result;
}

TEST(PredicatesTest, ProjectsOperatorsHelpersAndResultMembers) {
  auto import_options = options();
  Source source(
      IREE_SV(R"cpp(
#include <loomcxx/predicate.h>
        struct Selection {
          float scale;
          unsigned count;
        };
        Selection select(unsigned count, unsigned capacity)
            [[loom::where(count > 0u && count <= capacity &&
                          loom::predicate::multiple_of(count, 16u) &&
                          loom::predicate::finite(
                              loom::predicate::result<Selection>().scale))]];
      )cpp"),
      IREE_SV("predicates.cxx"), import_options);
  auto* expression = function_where(source, "select");
  ASSERT_NE(expression, nullptr);
  auto predicates =
      project_predicates(source.unit(), source.diagnostics(), expression);
  ASSERT_EQ(predicates.size(), 4u);

  EXPECT_EQ(predicates[0].predicate.kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(predicates[0].predicate.args[1], 0);
  EXPECT_EQ(cxx::to_string(predicates[0].values[0].binding->name()), "count");

  EXPECT_EQ(predicates[1].predicate.kind, LOOM_PREDICATE_ULE);
  ASSERT_EQ(predicates[1].value_count, 2u);
  EXPECT_EQ(cxx::to_string(predicates[1].values[0].binding->name()), "count");
  EXPECT_EQ(cxx::to_string(predicates[1].values[1].binding->name()),
            "capacity");

  EXPECT_EQ(predicates[2].predicate.kind, LOOM_PREDICATE_MULTIPLE_OF);
  EXPECT_EQ(predicates[2].predicate.args[1], 16);

  EXPECT_EQ(predicates[3].predicate.kind, LOOM_PREDICATE_FINITE);
  ASSERT_EQ(predicates[3].value_count, 1u);
  EXPECT_EQ(predicates[3].values[0].origin, PredicateValueOrigin::Result);
  ASSERT_EQ(predicates[3].values[0].members.size(), 1u);
  EXPECT_EQ(cxx::to_string(predicates[3].values[0].members[0]->name()),
            "scale");
}

TEST(PredicatesTest, ProjectsEveryCanonicalHelperBySemanticBinding) {
  auto import_options = options();
  Source source(
      IREE_SV(R"cpp(
#include <loomcxx/predicate.h>
        bool all(int a, int b, unsigned ua, unsigned ub, float f) [[loom::where(
            loom::predicate::eq(a, b) && loom::predicate::ne(a, b) &&
            loom::predicate::lt(a, b) && loom::predicate::le(a, b) &&
            loom::predicate::gt(a, b) && loom::predicate::ge(a, b) &&
            loom::predicate::multiple_of(a, 4) &&
            loom::predicate::power_of_two(a) &&
            loom::predicate::range(a, -8, 8) && loom::predicate::not_nan(f) &&
            loom::predicate::not_inf(f) && loom::predicate::finite(f) &&
            loom::predicate::ult(ua, ub) && loom::predicate::ule(ua, ub) &&
            loom::predicate::ugt(ua, ub) && loom::predicate::uge(ua, ub))]];
      )cpp"),
      IREE_SV("helpers.cxx"), import_options);
  auto* expression = function_where(source, "all");
  ASSERT_NE(expression, nullptr);
  auto predicates =
      project_predicates(source.unit(), source.diagnostics(), expression);
  ASSERT_EQ(predicates.size(), LOOM_PREDICATE_COUNT_);
  for (size_t i = 0; i < predicates.size(); ++i) {
    EXPECT_EQ(predicates[i].predicate.kind, i);
  }
}

TEST(PredicatesTest, RetainsQualifiedBaseSubobjectPaths) {
  auto import_options = options();
  Source source(IREE_SV(R"cpp(
#include <loomcxx/predicate.h>
                  struct Base {
                    unsigned count;
                  };
                  struct Left : Base {};
                  struct Right : Base {};
                  struct Shape : Left, Right {};
                  unsigned base_count(Shape value)
                      [[loom::where(loom::predicate::ule(value.Left::count,
                                                         value.Right::count))]];
                )cpp"),
                IREE_SV("base_paths.cxx"), import_options);
  auto* expression = function_where(source, "base_count");
  ASSERT_NE(expression, nullptr);
  auto predicates =
      project_predicates(source.unit(), source.diagnostics(), expression);
  ASSERT_EQ(predicates.size(), 1u);
  ASSERT_EQ(predicates[0].value_count, 2u);

  for (size_t index = 0; index < 2; ++index) {
    const auto& value = predicates[0].values[index];
    EXPECT_EQ(value.origin, PredicateValueOrigin::Binding);
    EXPECT_EQ(cxx::to_string(value.binding->name()), "value");
    ASSERT_EQ(value.members.size(), 3u);
    EXPECT_EQ(cxx::to_string(value.members[0]->name()),
              index == 0 ? "Left" : "Right");
    EXPECT_EQ(cxx::to_string(value.members[1]->name()), "Base");
    EXPECT_EQ(cxx::to_string(value.members[2]->name()), "count");
  }
}

TEST(PredicatesTest, InsertsConfigSubjectForImplicitHelpers) {
  auto import_options = options();
  Source source(IREE_SV(R"cpp(
#include <loomcxx/predicate.h>
                  [[loom::config("model.tile"),
                    loom::where(loom::predicate::range(16u, 256u) &&
                                loom::predicate::multiple_of(16u))]]
                  extern const unsigned tile;
                )cpp"),
                IREE_SV("config.cxx"), import_options);
  auto* expression = variable_where(source, "tile");
  ASSERT_NE(expression, nullptr);
  auto predicates = project_predicates(source.unit(), source.diagnostics(),
                                       expression, PredicateSubject::Implicit);
  ASSERT_EQ(predicates.size(), 2u);
  EXPECT_EQ(predicates[0].predicate.kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(predicates[0].predicate.args[1], 16);
  EXPECT_EQ(predicates[0].predicate.args[2], 256);
  EXPECT_EQ(predicates[1].predicate.kind, LOOM_PREDICATE_MULTIPLE_OF);
  for (const auto& predicate : predicates) {
    ASSERT_EQ(predicate.value_count, 1u);
    EXPECT_EQ(predicate.values[0].origin, PredicateValueOrigin::Subject);
    EXPECT_EQ(predicate.predicate.arg_tags[0], LOOM_PRED_ARG_VALUE);
    EXPECT_EQ(predicate.predicate.args[0], 0);
  }
}

TEST(PredicatesTest, RejectsUnboundCallsWithoutPublishingPartialResults) {
  auto import_options = options();
  Source source(IREE_SV(R"cpp(
                  bool opaque(unsigned);
                  bool invalid(unsigned value) [[loom::where(value > 0u && opaque(value))]];
                )cpp"),
                IREE_SV("invalid.cxx"), import_options);
  auto* expression = function_where(source, "invalid");
  ASSERT_NE(expression, nullptr);
  EXPECT_THROW(
      project_predicates(source.unit(), source.diagnostics(), expression),
      SourceRejected);
}

}  // namespace
}  // namespace loom::cxx_import
