// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/assumptions.h"

#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

cxx::CallExpressionAST* source_call(Source& source) {
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    if (auto* function =
            cxx::ast_cast<cxx::FunctionDefinitionAST>(declaration)) {
      auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                       function->functionBody)
                       ->statement;
      for (auto* statement : cxx::ListView{body->statementList}) {
        auto* expression =
            cxx::ast_cast<cxx::ExpressionStatementAST>(statement);
        if (expression) {
          return cxx::ast_cast<cxx::CallExpressionAST>(expression->expression);
        }
      }
    }
  }
  return nullptr;
}

TEST(AssumptionsTest, ConjunctionRetainsBindingsPromotionsAndConstantBounds) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(
      IREE_SV(R"cpp(
        [[loom::assume]] void assume(bool);
        constexpr unsigned capacity = 28672;
        enum { stride = 16 };
        void entry(unsigned count, unsigned char byte, unsigned long long wide) {
          assume(
              ((count <
                ((capacity / sizeof(unsigned) - 16u - 320u) / stride + 1u))) &&
              ((byte) < 256u && count < 256u) &&
              wide < static_cast<unsigned char>(272u));
        }
      )cpp"),
      IREE_SV("bounds.cpp"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 4u);
  auto& first = predicates[0];
  auto& second = predicates[1];
  auto& third = predicates[2];
  auto& fourth = predicates[3];
  EXPECT_EQ(first.predicate.kind, LOOM_PREDICATE_RANGE);
  EXPECT_EQ(first.predicate.args[2], 427);
  EXPECT_EQ(second.predicate.args[2], 255);
  EXPECT_EQ(third.predicate.args[2], 255);
  EXPECT_EQ(fourth.predicate.args[2], 15);
  EXPECT_EQ(first.values[0].origin, PredicateValueOrigin::Binding);
  EXPECT_EQ(second.values[0].origin, PredicateValueOrigin::Binding);
  EXPECT_EQ(third.values[0].origin, PredicateValueOrigin::Binding);
  EXPECT_EQ(fourth.values[0].origin, PredicateValueOrigin::Binding);
  ASSERT_NE(first.values[0].binding, nullptr);
  ASSERT_NE(second.values[0].binding, nullptr);
  EXPECT_EQ(cxx::to_string(first.values[0].binding->name()), "count");
  EXPECT_EQ(cxx::to_string(second.values[0].binding->name()), "byte");
  EXPECT_EQ(first.values[0].binding, third.values[0].binding);
  auto traits = source.unit().typeTraits();
  EXPECT_EQ(traits.integral_representation(second.values[0].source->type)->bits,
            8);
  EXPECT_EQ(
      traits.integral_representation(second.values[0].converted->type)->bits,
      32);
  EXPECT_EQ(
      traits.integral_representation(fourth.values[0].converted->type)->bits,
      64);
}

TEST(AssumptionsTest, RetainsRelationsNonzeroAndBothBindings) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  [[loom::assume]] void assume(bool);
                  void entry(int hidden, int limit, unsigned token,
                             unsigned capacity, bool enabled) {
                    assume(hidden > 0 && hidden <= limit && token <= capacity &&
                           token != capacity && token > 0u && enabled);
                  }
                )cpp"),
                IREE_SV("relations.cpp"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 6u);

  auto& positive = predicates[0];
  EXPECT_EQ(positive.predicate.kind, LOOM_PREDICATE_GT);
  EXPECT_EQ(positive.predicate.args[1], 0);
  EXPECT_EQ(cxx::to_string(positive.values[0].binding->name()), "hidden");

  auto& signed_relation = predicates[1];
  EXPECT_EQ(signed_relation.predicate.kind, LOOM_PREDICATE_LE);
  EXPECT_EQ(cxx::to_string(signed_relation.values[0].binding->name()),
            "hidden");
  EXPECT_EQ(cxx::to_string(signed_relation.values[1].binding->name()), "limit");

  auto& unsigned_relation = predicates[2];
  EXPECT_EQ(unsigned_relation.predicate.kind, LOOM_PREDICATE_ULE);
  EXPECT_EQ(cxx::to_string(unsigned_relation.values[0].binding->name()),
            "token");
  EXPECT_EQ(cxx::to_string(unsigned_relation.values[1].binding->name()),
            "capacity");

  auto& unsigned_equality = predicates[3];
  EXPECT_EQ(unsigned_equality.predicate.kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(cxx::to_string(unsigned_equality.values[0].binding->name()),
            "token");
  EXPECT_EQ(cxx::to_string(unsigned_equality.values[1].binding->name()),
            "capacity");

  auto& unsigned_nonzero = predicates[4];
  EXPECT_EQ(unsigned_nonzero.predicate.kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(unsigned_nonzero.predicate.args[1], 0);

  auto& boolean_truth = predicates[5];
  EXPECT_EQ(boolean_truth.predicate.kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(boolean_truth.predicate.args[1], 0);
  EXPECT_EQ(cxx::to_string(boolean_truth.values[0].binding->name()), "enabled");
}

TEST(AssumptionsTest, RetainsUnsignedRelationsAcrossCarrierSignBit) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  [[loom::assume]] void assume(bool);
                  void entry(unsigned value) {
                    assume(value < 0x90000000u && value <= 0x90000000u &&
                           value > 7u && value >= 7u);
                  }
                )cpp"),
                IREE_SV("unsigned_relations.cpp"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 4u);

  constexpr loom_predicate_kind_t kExpectedKinds[] = {
      LOOM_PREDICATE_ULT,
      LOOM_PREDICATE_ULE,
      LOOM_PREDICATE_UGT,
      LOOM_PREDICATE_UGE,
  };
  constexpr int64_t kExpectedConstants[] = {
      -1879048192,
      -1879048192,
      7,
      7,
  };
  for (size_t i = 0; i < predicates.size(); ++i) {
    EXPECT_EQ(predicates[i].predicate.kind, kExpectedKinds[i]);
    EXPECT_EQ(predicates[i].predicate.args[1], kExpectedConstants[i]);
  }
}

TEST(AssumptionsTest, UsesUsualArithmeticConversionsForOrdering) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  [[loom::assume]] void assume(bool);
                  void entry(int signed_value, unsigned unsigned_limit,
                             unsigned char narrow, int signed_limit) {
                    assume(signed_value < unsigned_limit && narrow < signed_limit);
                  }
                )cpp"),
                IREE_SV("promotions.cpp"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 2u);

  EXPECT_EQ(predicates[0].predicate.kind, LOOM_PREDICATE_ULT);
  EXPECT_EQ(predicates[1].predicate.kind, LOOM_PREDICATE_LT);
  auto traits = source.unit().typeTraits();
  for (size_t i = 0; i < predicates[0].value_count; ++i) {
    auto representation =
        traits.integral_representation(predicates[0].values[i].converted->type);
    ASSERT_TRUE(representation.has_value());
    EXPECT_FALSE(representation->isSigned);
    EXPECT_EQ(representation->bits, 32);
  }
  for (size_t i = 0; i < predicates[1].value_count; ++i) {
    auto representation =
        traits.integral_representation(predicates[1].values[i].converted->type);
    ASSERT_TRUE(representation.has_value());
    EXPECT_TRUE(representation->isSigned);
    EXPECT_EQ(representation->bits, 32);
  }
}

TEST(AssumptionsTest, RetainsMaterializedDerivedScalarExpressions) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  [[loom::assume]] void assume(bool);
                  void entry(unsigned pair_index, unsigned limit, float value,
                             float divisor, float expected) {
                    const unsigned index = 1u + 2u * pair_index;
                    assume(index + 1u < limit && (index - 1u) * 2u < limit &&
                           (index & 15u) != (limit ^ 3u) &&
                           value / divisor == expected && !(index & 1u));
                  }
                )cpp"),
                IREE_SV("derived.cxx"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 5u);

  size_t expression_count = 0;
  size_t binding_count = 0;
  for (const auto& predicate : predicates) {
    ASSERT_GT(predicate.value_count, 0u);
    for (size_t i = 0; i < predicate.value_count; ++i) {
      const auto& value = predicate.values[i];
      if (value.origin == PredicateValueOrigin::Expression) {
        ++expression_count;
        EXPECT_FALSE(value.dependencies.empty());
      } else {
        ++binding_count;
        EXPECT_EQ(value.origin, PredicateValueOrigin::Binding);
      }
    }
  }
  EXPECT_EQ(expression_count, 6u);
  EXPECT_EQ(binding_count, 3u);
  EXPECT_EQ(predicates[0].predicate.kind, LOOM_PREDICATE_ULT);
  EXPECT_EQ(
      cxx::to_string(predicates[0].values[0].dependencies[0].binding->name()),
      "index");
  EXPECT_EQ(predicates[1].predicate.kind, LOOM_PREDICATE_ULT);
  EXPECT_EQ(predicates[2].predicate.kind, LOOM_PREDICATE_NE);
  EXPECT_EQ(predicates[3].predicate.kind, LOOM_PREDICATE_EQ);
  EXPECT_EQ(predicates[4].predicate.kind, LOOM_PREDICATE_EQ);
}

TEST(AssumptionsTest, AdmitsConstexprCallsAndRecordPointerReads) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  [[loom::assume]] void assume(bool);
                  constexpr int extent(int scale) { return scale * 4; }
                  consteval int maximum() { return 16; }
                  struct Bounds {
                    int limit;
                  };
                  void entry(int index, Bounds* mutable_bounds, const Bounds* const_bounds) {
                    assume(index < extent(2) && index < mutable_bounds->limit &&
                           const_bounds->limit <= maximum());
                  }
                )cpp"),
                IREE_SV("record_bounds.cxx"), options);
  auto* call = source_call(source);
  ASSERT_NE(call, nullptr);
  auto predicates =
      assumption_predicates(source.unit(), source.diagnostics(), call);
  ASSERT_EQ(predicates.size(), 3u);

  EXPECT_EQ(predicates[0].predicate.kind, LOOM_PREDICATE_LT);
  EXPECT_EQ(predicates[0].predicate.args[1], 8);
  EXPECT_EQ(predicates[0].values[0].origin, PredicateValueOrigin::Binding);

  EXPECT_EQ(predicates[1].predicate.kind, LOOM_PREDICATE_LT);
  ASSERT_EQ(predicates[1].value_count, 2u);
  EXPECT_EQ(predicates[1].values[1].origin, PredicateValueOrigin::Expression);
  ASSERT_EQ(predicates[1].values[1].dependencies.size(), 1u);
  EXPECT_EQ(
      cxx::to_string(predicates[1].values[1].dependencies[0].binding->name()),
      "mutable_bounds");

  EXPECT_EQ(predicates[2].predicate.kind, LOOM_PREDICATE_LE);
  EXPECT_EQ(predicates[2].predicate.args[1], 16);
  EXPECT_EQ(predicates[2].values[0].origin, PredicateValueOrigin::Expression);
  ASSERT_EQ(predicates[2].values[0].dependencies.size(), 1u);
  EXPECT_EQ(
      cxx::to_string(predicates[2].values[0].dependencies[0].binding->name()),
      "const_bounds");
}

TEST(AssumptionsTest, RejectsPartialDerivedScalarExpressions) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source division(IREE_SV(R"cpp(
                    [[loom::assume]] void assume(bool);
                    void entry(unsigned value, unsigned divisor) { assume(value / divisor < 16u); }
                  )cpp"),
                  IREE_SV("division.cxx"), options);
  EXPECT_THROW(assumption_predicates(division.unit(), division.diagnostics(),
                                     source_call(division)),
               SourceRejected);

  Source shift(IREE_SV(R"cpp(
                 [[loom::assume]] void assume(bool);
                 void entry(unsigned value, unsigned amount) { assume((value << amount) < 16u); }
               )cpp"),
               IREE_SV("shift.cxx"), options);
  EXPECT_THROW(assumption_predicates(shift.unit(), shift.diagnostics(),
                                     source_call(shift)),
               SourceRejected);
}

}  // namespace
}  // namespace loom::cxx_import
