// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/lambda_captures.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/expressions.h"
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

class FirstLambda final : private cxx::ASTVisitor {
 public:
  static cxx::LambdaExpressionAST* find(cxx::AST* root) {
    FirstLambda finder;
    finder.accept(root);
    return finder.result_;
  }

 private:
  void visit(cxx::LambdaExpressionAST* ast) override {
    if (!result_) {
      result_ = ast;
    }
  }

  cxx::LambdaExpressionAST* result_ = nullptr;
};

TEST(ControlFlowTest, LambdaConstructionVisitsCapturesButNotDeferredBody) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    static unsigned initialize();
    static void deferred();
    void entry(unsigned value) {
      [&value, initialized = initialize()]() mutable {
        value += initialized;
        deferred();
      };
    }
  )"),
                IREE_SV("lambda_boundary.cxx"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* lambda = FirstLambda::find(body);
  ASSERT_NE(lambda, nullptr);
  ASSERT_NE(lambda->captureList, nullptr);
  auto* value_capture = lambda->captureList->next->value;
  auto* initializer = cxx::capture_initializer(value_capture);
  ASSERT_NE(initializer, nullptr);

  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body,
                       function->symbol);
  auto calls = analysis.calls();
  ASSERT_EQ(calls.size(), 1u);
  ASSERT_NE(calls[0].function, nullptr);
  EXPECT_EQ(cxx::to_string(calls[0].function->name()), "initialize");
  EXPECT_TRUE(analysis.written(body).empty());
}

// Comparison classification is independent of pointer comparison lowering:
// the capture analysis consumes resolved value reads, not operator spelling.
TEST(ControlFlowTest, DependentPointerComparisonsReadCapturedValues) {
  for (const char* comparison : {"==", "!=", "<", "<=", ">", ">="}) {
    SCOPED_TRACE(comparison);
    std::string text = R"(
      bool entry(const int* input, const int* other) {
        auto compare = [&input](auto value) { return input )";
    text += comparison;
    text += R"( value; };
        return compare(other);
      }
    )";
    loom_cxx_import_options_t options;
    loom_cxx_import_options_initialize(&options);
    Source source(iree_make_string_view(text.data(), text.size()),
                  IREE_SV("capture_comparisons.cxx"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    auto* lambda = FirstLambda::find(body);
    ASSERT_NE(lambda, nullptr);
    auto* field = cxx::capture_field(lambda->captureList->value);
    ASSERT_NE(field, nullptr);
    auto* primary = lambda->symbol->closureType()->functionCallOperator();
    ASSERT_EQ(primary->specializations().size(), 1u);
    auto* concrete = cxx::symbol_cast<cxx::FunctionSymbol>(
        primary->specializations().front().symbol);
    ASSERT_NE(concrete, nullptr);
    auto* concrete_body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
        concrete->declaration()->functionBody);
    ASSERT_NE(concrete_body, nullptr);

    Types types(source.unit(), source.diagnostics());
    ControlFlow construction(source.unit(), source.diagnostics(), types, body,
                             function->symbol);
    ControlFlow invocation(source.unit(), source.diagnostics(), types,
                           concrete_body->statement, concrete);
    std::unordered_set<cxx::FieldSymbol*> identities;
    invocation.collect_pointer_capture_identities(identities);
    EXPECT_TRUE(identities.empty());
    EXPECT_FALSE(construction.resolve_pointer_reference_captures(identities));
    construction.publish_pointer_reference_snapshots(types);
    EXPECT_TRUE(types.pointer_reference_is_snapshot(field));
  }
}

TEST(ControlFlowTest,
     DirectOperatorsRetainCallableOperandsAndEvaluationContracts) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Member {
      unsigned value;
      Member operator+(Member) const;
      Member operator-() const;
      Member& operator+=(Member);
      Member& operator++();
      Member operator++(int);
    };
    struct Free { unsigned value; };
    Free operator+(Free, Free);
    Free operator-(Free);
    Free& operator+=(Free&, Free);
    Free& operator++(Free&);
    Free operator++(Free&, int);
    void entry(Member lhs, Member rhs, Free free_lhs, Free free_rhs) {
      (void)(lhs + rhs);
      (void)-lhs;
      lhs += rhs;
      ++lhs;
      lhs++;
      (void)(free_lhs + free_rhs);
      (void)-free_lhs;
      free_lhs += free_rhs;
      ++free_lhs;
      free_lhs++;
    }
    Member Member::operator+(Member other) const {
      return {value + other.value};
    }
    Member Member::operator-() const { return {0u - value}; }
    Member& Member::operator+=(Member other) {
      value += other.value;
      return *this;
    }
    Member& Member::operator++() { ++value; return *this; }
    Member Member::operator++(int) { Member old = *this; ++*this; return old; }
    Free operator+(Free lhs, Free rhs) { return {lhs.value + rhs.value}; }
    Free operator-(Free value) { return {0u - value.value}; }
    Free& operator+=(Free& lhs, Free rhs) {
      lhs.value += rhs.value;
      return lhs;
    }
    Free& operator++(Free& value) { ++value.value; return value; }
    Free operator++(Free& value, int) { Free old = value; ++value; return old; }
  )"),
                IREE_SV("direct_operators.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body,
                       function->symbol);
  auto calls = analysis.calls();
  ASSERT_EQ(calls.size(), 10u);

  for (size_t index = 0; index < calls.size(); ++index) {
    auto direct = direct_operator_call(
        cxx::ast_cast<cxx::ExpressionAST>(calls[index].source));
    ASSERT_TRUE(direct.has_value());
    EXPECT_EQ(direct->function, calls[index].function);
    EXPECT_EQ(calls[index].current_receiver, false);
    EXPECT_EQ(calls[index].receiver.has_value(), index < 5);
  }

  auto member_binary =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[0].source));
  ASSERT_TRUE(member_binary.has_value());
  EXPECT_NE(member_binary->receiver, nullptr);
  EXPECT_EQ(member_binary->argument_count, 1u);
  EXPECT_EQ(member_binary->evaluation_order, OperatorEvaluationOrder::Forward);

  auto member_compound =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[2].source));
  ASSERT_TRUE(member_compound.has_value());
  EXPECT_EQ(member_compound->evaluation_order,
            OperatorEvaluationOrder::ArgumentsThenReceiver);

  auto member_postfix =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[4].source));
  ASSERT_TRUE(member_postfix.has_value());
  EXPECT_TRUE(member_postfix->has_postfix_dummy);
  EXPECT_EQ(member_postfix->argument_count, 0u);
  ASSERT_EQ(calls[4].arguments.size(), 1u);
  EXPECT_EQ(calls[4].arguments[0].source, nullptr);

  auto free_binary =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[5].source));
  ASSERT_TRUE(free_binary.has_value());
  EXPECT_EQ(free_binary->receiver, nullptr);
  EXPECT_EQ(free_binary->argument_count, 2u);

  auto free_compound =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[7].source));
  ASSERT_TRUE(free_compound.has_value());
  EXPECT_EQ(free_compound->evaluation_order,
            OperatorEvaluationOrder::ReverseArguments);

  auto free_postfix =
      direct_operator_call(cxx::ast_cast<cxx::ExpressionAST>(calls[9].source));
  ASSERT_TRUE(free_postfix.has_value());
  EXPECT_TRUE(free_postfix->has_postfix_dummy);
  EXPECT_EQ(free_postfix->argument_count, 1u);
  ASSERT_EQ(calls[9].arguments.size(), 2u);
  EXPECT_EQ(calls[9].arguments[1].source, nullptr);
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
      {"unsigned n", "for (unsigned i=0; i<n; ++i) ;", 1},
      {"unsigned n", "for (unsigned i=0; i<16u; i+=4u) n += i;", 4, 16},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) ++i;", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) --n;", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) (void)&i;", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) sizeof(++i);", 1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) unsigned value = ++i;", 0},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) if constexpr(false) ++i;",
       1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) if constexpr(true) ++i;", 0},
      {"unsigned n",
       "for (unsigned i=0; i<n; ++i) for (unsigned j=0; j<4u; ++j) ++i;", 0},
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
      const auto& range = std::get<CountedLoop::UnsignedRange>(counted->range);
      EXPECT_EQ(range.step, test.step);
      EXPECT_EQ(cxx::to_string(counted->induction->name()), "i");
      if (test.upper) {
        EXPECT_EQ(std::get<unsigned>(range.upper), *test.upper);
      } else {
        const auto& bound = std::get<CountedLoop::Bound>(range.upper);
        EXPECT_EQ(bound.binding, function->symbol->parameters()[0]);
        auto* condition =
            cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition);
        EXPECT_EQ(bound.expression, condition->rightExpression);
      }
      EXPECT_EQ(analysis.counted(loop), counted);
    }
  }
}

TEST(ControlFlowTest,
     SignedCountedAdmissionRetainsTheNonwrappingIntervalProof) {
  struct Case {
    // Source parameter type determines the comparison's promoted width.
    const char* parameter;
    // Standalone source loop whose classification is under test.
    const char* loop;
    // Expected positive counted step, or zero for ordinary while semantics.
    unsigned step;
    // Proven constant upper bound, or no value for a runtime scalar bound.
    std::optional<std::int32_t> upper = std::nullopt;
  };
  const Case cases[] = {
      {"int n", "for (int i=0; i<n; ++i) {}", 1},
      {"short n", "for (int i=0; i<n; ++i) {}", 1},
      {"unsigned short n", "for (int i=0; i<n; ++i) {}", 1},
      {"int n", "for (int i=-4; i<3; i+=2) {}", 2, 3},
      {"int n", "for (int i=n; i<-3; i++) {}", 1, -3},
      {"int n", "for (int i=n; i<2147483647; ++i) {}", 1, 2147483647},
      {"int n", "for (int i=n; i<2147483644; i+=4) {}", 4, 2147483644},
      {"int n", "for (int i=n; i<(-2147483647-1); ++i) {}", 1,
       std::numeric_limits<std::int32_t>::min()},
      {"int n", "for (int i=0; i<n; ++i) ++i;", 0},
      {"int n", "for (int i=0; i<n; ++i) --n;", 0},
      {"int n", "for (int i=0; i<n; ++i) (void)&i;", 0},
      {"int n", "for (int i=0; i<n; i+=2) {}", 0},
      {"int n", "for (int i=0; i<20; i+=2u) {}", 0},
      {"volatile int n", "for (int i=0; i<n; ++i) {}", 0},
      {"int n", "for (volatile int i=0; i<n; ++i) {}", 0},
      {"int n", "for (int i=0; i<(n+0); ++i) {}", 0},
      {"int n", "for (int i=0; i<opaque(); ++i) {}", 0},
      {"int n", "for (int i=0; i<20; i+=opaque()) {}", 0},
      {"int n", "for (int i=0; i<n; --i) {}", 0},
      {"int n", "for (int i=0; i<=n; ++i) {}", 0},
      {"int n", "for (int i=0; i<2147483645; i+=4) {}", 0},
      {"unsigned n", "for (int i=0; i<(int)n; ++i) {}", 0},
      {"long n", "for (int i=0; i<n; ++i) {}", 0},
      {"int n", "for (short i=0; i<n; ++i) {}", 0},
      {"long n", "for (long i=0; i<n; ++i) {}", 0},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.loop);
    std::string text = std::string("int opaque(); void entry(") +
                       test.parameter + ") {" + test.loop + "}";
    Source source(view(text), IREE_SV("signed_loops.cpp"), options);
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
    if (!counted) {
      continue;
    }
    const auto& range = std::get<CountedLoop::SignedRange>(counted->range);
    EXPECT_EQ(range.step, test.step);
    EXPECT_EQ(cxx::to_string(counted->induction->name()), "i");
    if (test.upper) {
      EXPECT_EQ(std::get<std::int32_t>(range.upper), *test.upper);
    } else {
      const auto& bound = std::get<CountedLoop::Bound>(range.upper);
      EXPECT_EQ(bound.binding, function->symbol->parameters()[0]);
      auto* condition =
          cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition);
      EXPECT_EQ(bound.expression, condition->rightExpression);
    }
    EXPECT_EQ(analysis.counted(loop), counted);
  }
}

TEST(ControlFlowTest, AddressAdmissionRetainsNativeUnitStepIntervals) {
  struct Case {
    // Fully qualified native address type used by the source loop.
    const char* type;
    // Standalone source loop whose classification is under test.
    const char* loop;
    // Expected native range, or no value for general while semantics.
    std::optional<CountedLoop::AddressDomain> domain;
    // Whether the admitted upper bound is an exact temporary constant.
    bool constant_bound = false;
  };
  const Case cases[] = {
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{1}) {}",
       CountedLoop::AddressDomain::Index},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=loom::type::index(1)+i) {}",
       CountedLoop::AddressDomain::Index},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=i+loom::type::offset{1}) {}",
       CountedLoop::AddressDomain::Offset},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=loom::type::offset{1}+i) {}",
       CountedLoop::AddressDomain::Offset},
      {"loom::type::index",
       "for (auto i=begin; i<loom::type::index{4}; "
       "i=i+loom::type::index{1}) {}",
       CountedLoop::AddressDomain::Index, true},
      {"loom::type::offset",
       "for (auto i=begin; i<loom::type::offset(4); "
       "i=i+loom::type::offset{1}) {}",
       CountedLoop::AddressDomain::Offset, true},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{2}) {}", std::nullopt},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=i+loom::type::offset{2}) {}", std::nullopt},
      {"loom::type::index",
       "for (auto i=begin; loom::index::ult(i,end); "
       "i=i+loom::type::index{1}) {}",
       std::nullopt},
      {"loom::type::offset",
       "for (auto i=begin; loom::offset::slt(i,end); "
       "i=i+loom::type::offset{1}) {}",
       std::nullopt},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{1}) "
       "end=loom::type::index{4};",
       std::nullopt},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=i+loom::type::offset{1}) "
       "i=loom::type::offset{4};",
       std::nullopt},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{1}) (void)&end;",
       std::nullopt},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{1}) {} (void)&end;",
       std::nullopt},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=i+loom::type::offset{1}) (void)&i;",
       std::nullopt},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.loop);
    std::string text = std::string("#include <loomcxx/index.h>\nvoid entry(") +
                       test.type + " begin, " + test.type + " end) {" +
                       test.loop + "}";
    Source source(view(text), IREE_SV("address_loops.cxx"), options);
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
    ASSERT_EQ(counted != nullptr, test.domain.has_value());
    if (!counted) {
      continue;
    }
    EXPECT_EQ(counted->induction,
              cxx::ast_cast<cxx::SimpleDeclarationAST>(
                  cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer)
                      ->declaration)
                  ->initDeclaratorList->value->symbol);
    const auto& range = std::get<CountedLoop::AddressRange>(counted->range);
    EXPECT_EQ(range.domain, *test.domain);
    EXPECT_EQ(range.upper.binding, test.constant_bound
                                       ? nullptr
                                       : function->symbol->parameters()[1]);
    EXPECT_EQ(range.upper.expression,
              cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition)
                  ->rightExpression);
    EXPECT_EQ(analysis.counted(loop), counted);
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

TEST(ControlFlowTest, AutomaticObjectStorageRemainsDistinctFromAddressDemand) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Pair { unsigned first; unsigned second; };
    struct Constructed { Constructed(); unsigned value; };
    void entry(unsigned input) {
      Pair value{input, input + 1u};
      Pair output;
      unsigned scalar;
      unsigned values[2] = {};
      volatile unsigned observed = input;
      unsigned addressed = input;
      (void)&addressed;
      Constructed constructed;
    }
    Constructed::Constructed() : value(0u) {}
  )"),
                IREE_SV("automatic_storage.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body,
                       function->symbol);

  std::vector<cxx::Symbol*> bindings;
  for (auto* statement : cxx::ListView{body->statementList}) {
    auto* declaration = cxx::ast_cast<cxx::DeclarationStatementAST>(statement);
    if (!declaration) {
      continue;
    }
    auto* simple =
        cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration);
    ASSERT_NE(simple, nullptr);
    bindings.push_back(simple->initDeclaratorList->value->symbol);
  }
  ASSERT_EQ(bindings.size(), 7u);
  EXPECT_FALSE(analysis.storage_backed(bindings[0]));
  for (auto* binding : std::span{bindings}.subspan(1)) {
    EXPECT_TRUE(analysis.storage_backed(binding));
  }
  for (size_t index = 0; index < bindings.size(); ++index) {
    EXPECT_EQ(analysis.addressed(bindings[index]), index == 5u);
  }
}

TEST(ControlFlowTest, StructuredBindingsAnalyzeTheSynthesizedDeclarationOnce) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Pair { unsigned first; unsigned second; };
    Pair make_pair(unsigned input);
    void entry(unsigned input) {
      auto [first, second] = make_pair(input);
      (void)(first + second);
    }
    Pair make_pair(unsigned input) { return {input, input + 1u}; }
  )"),
                IREE_SV("structured_bindings.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* statement =
      cxx::ast_cast<cxx::DeclarationStatementAST>(body->statementList->value);
  ASSERT_NE(statement, nullptr);
  auto* declaration = cxx::ast_cast<cxx::StructuredBindingDeclarationAST>(
      statement->declaration);
  ASSERT_NE(declaration, nullptr);
  ASSERT_NE(declaration->hiddenVariable, nullptr);

  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body,
                       function->symbol);

  auto calls = analysis.calls();
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_NE(calls.front().function, nullptr);
  EXPECT_FALSE(analysis.storage_backed(declaration->hiddenVariable->symbol));
  for (auto* binding : cxx::ListView{declaration->bindingDeclaratorList}) {
    EXPECT_FALSE(analysis.storage_backed(binding->symbol));
    EXPECT_FALSE(analysis.addressed(binding->symbol));
  }
}

TEST(ControlFlowTest, CallArgumentsRetainExactAutomaticObjectIntervals) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Pair { unsigned first; unsigned second; };
    static void update(Pair& whole, unsigned& second, const unsigned& first);
    void entry(Pair value) { update(value, value.second, value.first); }
    static void update(Pair& whole, unsigned& second, const unsigned& first) {
      whole.first += first;
      ++second;
    }
  )"),
                IREE_SV("call_intervals.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body,
                       function->symbol);

  auto calls = analysis.calls();
  ASSERT_EQ(calls.size(), 1u);
  ASSERT_EQ(calls.front().arguments.size(), 3u);
  auto* value = function->symbol->parameters().front();
  const ObjectInterval expected[] = {
      {value, 0, ObjectInterval::kWholeObjectEnd},
      {value, 1, 2},
      {value, 0, 1},
  };
  for (size_t ordinal = 0; ordinal < std::size(expected); ++ordinal) {
    const auto& argument = calls.front().arguments[ordinal];
    if (expected[ordinal].end == ObjectInterval::kWholeObjectEnd) {
      ASSERT_EQ(argument.identity_bindings.size(), 1u);
      EXPECT_EQ(argument.identity_bindings.front(), expected[ordinal].binding);
      EXPECT_EQ(argument.identity_intervals, nullptr);
      continue;
    }
    ASSERT_NE(argument.identity_intervals, nullptr);
    ASSERT_EQ(argument.identity_intervals->size(), 1u);
    EXPECT_EQ(argument.identity_intervals->front().binding,
              expected[ordinal].binding);
    EXPECT_EQ(argument.identity_intervals->front().begin,
              expected[ordinal].begin);
    EXPECT_EQ(argument.identity_intervals->front().end, expected[ordinal].end);
  }
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
    auto* declaration = condition_declaration(branch->condition);
    ASSERT_NE(declaration, nullptr);
    ASSERT_NE(declaration->symbol, nullptr);
    EXPECT_EQ(declaration->initializer, declaration->symbol->initializer());
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

TEST(ControlFlowTest, VectorLaneWritesRetainTheOwningVectorSlice) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    using Lanes = unsigned __attribute__((ext_vector_type(4)));
    struct Packet { unsigned prefix; Lanes lanes; };
    void entry(Packet packet, Packet* memory, unsigned index) {
      if (index < 4) {
        packet.lanes[index++] += 1;
        memory[index].lanes[index] = 2;
      }
    }
  )"),
                IREE_SV("vector_lane_writes.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* branch = cxx::ast_cast<cxx::IfStatementAST>(body->statementList->value);
  ASSERT_NE(branch, nullptr);
  auto* arm = cxx::ast_cast<cxx::CompoundStatementAST>(branch->statement);
  ASSERT_NE(arm, nullptr);
  auto* automatic_statement =
      cxx::ast_cast<cxx::ExpressionStatementAST>(arm->statementList->value);
  auto* automatic = cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(
      automatic_statement->expression);
  ASSERT_NE(automatic, nullptr);
  auto* memory_statement = cxx::ast_cast<cxx::ExpressionStatementAST>(
      arm->statementList->next->value);
  auto* memory =
      cxx::ast_cast<cxx::AssignmentExpressionAST>(memory_statement->expression);
  ASSERT_NE(memory, nullptr);

  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  auto writes = analysis.written(branch);
  ASSERT_EQ(writes.size(), 2u);
  EXPECT_EQ(writes[0], function->symbol->parameters()[0]);
  EXPECT_EQ(writes[1], function->symbol->parameters()[2]);
  auto lane = analysis.destination(automatic->targetExpression);
  ASSERT_TRUE(lane.has_value());
  EXPECT_EQ(lane->binding, function->symbol->parameters()[0]);
  EXPECT_EQ(lane->component_offset, 1u);
  EXPECT_EQ(lane->member, &kSSAPartition);
  EXPECT_FALSE(analysis.destination(memory->leftExpression).has_value());
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
    // Whether the function body can reach its lexical successor.
    bool falls_through;
  };
  const Case cases[] = {
      {"", ExitFlow::None, true},
      {"return;", ExitFlow::All, false},
      {"if (x) return;", ExitFlow::Some, true},
      {"if (x) return; else return;", ExitFlow::All, false},
      {"if (x) { if (y) return; }", ExitFlow::Some, true},
      {"if (x) { if (y) return; } return;", ExitFlow::All, false},
      {"while (x) { return; }", ExitFlow::Some, true},
      {"do { return; } while (x);", ExitFlow::All, false},
      {"for (unsigned i=0; i<4u; ++i) { if (y) return; }", ExitFlow::Some,
       true},
      {"while (x) { --x; }", ExitFlow::None, true},
      {"while (true) {}", ExitFlow::None, false},
      {"for (;;) { continue; }", ExitFlow::None, false},
      {"do {} while (1.0);", ExitFlow::None, false},
      {"while (true) { return; }", ExitFlow::All, false},
      {"while (true) { if (x) return; }", ExitFlow::Some, false},
      {"while (true) { if (x) break; }", ExitFlow::None, true},
      {"while (true) { break; }", ExitFlow::None, true},
      {"do { break; } while (true);", ExitFlow::None, true},
      {"while (x) { while (true) {} return; }", ExitFlow::None, true},
      {"__builtin_unreachable();", ExitFlow::None, false},
      {"(__builtin_unreachable());", ExitFlow::None, false},
      {"(void)__builtin_unreachable();", ExitFlow::None, false},
      {"(x = 3, __builtin_unreachable());", ExitFlow::None, false},
      {"x ? __builtin_unreachable() : __builtin_unreachable();", ExitFlow::None,
       false},
      {"x ? __builtin_unreachable() : void();", ExitFlow::None, true},
      {"(void)noexcept(__builtin_unreachable()); return;", ExitFlow::All,
       false},
      {"using T = decltype(__builtin_unreachable()); return;", ExitFlow::All,
       false},
      {"(void)sizeof((__builtin_unreachable(), 1)); return;", ExitFlow::All,
       false},
      {"(void)requires { __builtin_unreachable(); }; return;", ExitFlow::All,
       false},
      {"(void)[&] { __builtin_unreachable(); }; return;", ExitFlow::All, false},
      {"if (x) __builtin_unreachable(); else return;", ExitFlow::Some, false},
      {"if (x) return; else __builtin_unreachable();", ExitFlow::Some, false},
      {"__builtin_unreachable(); return;", ExitFlow::None, false},
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
    EXPECT_EQ(analysis.falls_through(body), test.falls_through);
  }
}

TEST(ControlFlowTest, ReturnSummariesRetainCallableExitRequirements) {
  struct Case {
    // Source body whose return projection is classified.
    const char* body;
    // Whether a loop or switch return requires callable-exit projection.
    bool requires_callable_exit;
    // Direct function-body children with a callable exit.
    size_t callable_exit_count;
  };
  const Case cases[] = {
      {"if (x) return; else return;", false, 0},
      {"if (x) { if (y) return; } else return;", false, 0},
      {"if (x) { while (y) return; } else return;", true, 1},
      {"if (x) { switch (y) { case 0: return; default: return; } } "
       "else return;",
       true, 1},
      {"if constexpr(false) { while (y) return; } else return;", false, 0},
      {"if (x) return; for (;;) { if (y) return; }", true, 1},
      {"while (x) return; for (;;) { if (y) return; }", true, 2},
      {"if (x) return; else __builtin_unreachable();", true, 1},
      {"if (x) return; else for (;;) {}", true, 1},
      {"if (x) return; __builtin_unreachable();", true, 1},
      {"if (x) return; for (;;) {}", true, 1},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.body);
    std::string text =
        std::string("void entry(int x, int y) {") + test.body + "}";
    Source source(view(text), IREE_SV("callable_exits.cpp"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    Types types(source.unit(), source.diagnostics());
    ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
    EXPECT_EQ(analysis.requires_callable_exit(body),
              test.requires_callable_exit);
    EXPECT_EQ(analysis.callable_exit_count(body), test.callable_exit_count);
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
      strip_implicit_casts(ret->expression));
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
    auto* nested =
        cxx::ast_cast<cxx::NestedExpressionAST>(strip_implicit_casts(arm));
    ASSERT_NE(nested, nullptr);
    auto arm_writes = analysis.written(nested->expression);
    ASSERT_EQ(arm_writes.size(), 2u);
    EXPECT_EQ(arm_writes[0], parameters[0]);
    EXPECT_EQ(arm_writes[1], parameters[1]);
  }
}

TEST(ControlFlowTest,
     SwitchClausesRetainLexicalLabelsWritesAndEscapingContinues) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    void entry(int selector, int x, int y) {
      for (unsigned i = 0; i < 4u; ++i) {
        switch (selector) {
          case -2:
          case 4:
            ++x;
            if (x) break;
            ++y;
          default:
            ++y;
            continue;
          case 7:
            break;
        }
        ++x;
      }
    }
  )"),
                IREE_SV("switch.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* loop = cxx::ast_cast<cxx::ForStatementAST>(body->statementList->value);
  ASSERT_NE(loop, nullptr);
  auto* loop_body = cxx::ast_cast<cxx::CompoundStatementAST>(loop->statement);
  ASSERT_NE(loop_body, nullptr);
  auto* selection =
      cxx::ast_cast<cxx::SwitchStatementAST>(loop_body->statementList->value);
  ASSERT_NE(selection, nullptr);

  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  const auto& layout = analysis.switch_layout(selection);
  ASSERT_EQ(layout.clauses.size(), 3u);
  EXPECT_TRUE(layout.has_default);
  EXPECT_FALSE(layout.direct);

  const auto& first = layout.clauses[0];
  ASSERT_EQ(first.labels.size(), 2u);
  ASSERT_TRUE(first.labels[0].key.has_value());
  EXPECT_EQ(*first.labels[0].key, -2);
  ASSERT_TRUE(first.labels[1].key.has_value());
  EXPECT_EQ(*first.labels[1].key, 4);
  EXPECT_EQ(first.statements.size(), 3u);
  EXPECT_TRUE(first.falls_through);

  const auto& fallback = layout.clauses[1];
  ASSERT_EQ(fallback.labels.size(), 1u);
  EXPECT_FALSE(fallback.labels[0].key.has_value());
  EXPECT_EQ(fallback.statements.size(), 2u);
  EXPECT_FALSE(fallback.falls_through);

  const auto& last = layout.clauses[2];
  ASSERT_EQ(last.labels.size(), 1u);
  ASSERT_TRUE(last.labels[0].key.has_value());
  EXPECT_EQ(*last.labels[0].key, 7);
  EXPECT_EQ(last.statements.size(), 1u);
  EXPECT_FALSE(last.falls_through);

  auto parameters = function->symbol->parameters();
  ASSERT_EQ(first.writes.size(), 2u);
  EXPECT_EQ(first.writes[0], parameters[1]);
  EXPECT_EQ(first.writes[1], parameters[2]);
  ASSERT_EQ(fallback.writes.size(), 1u);
  EXPECT_EQ(fallback.writes[0], parameters[2]);
  EXPECT_TRUE(last.writes.empty());
  auto switch_writes = analysis.written(selection);
  ASSERT_EQ(switch_writes.size(), 2u);
  EXPECT_EQ(switch_writes[0], parameters[1]);
  EXPECT_EQ(switch_writes[1], parameters[2]);

  EXPECT_EQ(analysis.breaks(selection), ExitFlow::None);
  EXPECT_EQ(analysis.continues(selection), ExitFlow::Some);
  EXPECT_EQ(analysis.iteration_exits(selection), ExitFlow::Some);
  EXPECT_EQ(analysis.returns(selection), ExitFlow::None);
  EXPECT_EQ(analysis.breaks(loop->statement), ExitFlow::None);
  EXPECT_EQ(analysis.continues(loop->statement), ExitFlow::Some);
  EXPECT_EQ(analysis.breaks(loop), ExitFlow::None);
  EXPECT_EQ(analysis.continues(loop), ExitFlow::None);
}

TEST(ControlFlowTest, DirectAndNestedSwitchesConsumeTheirOwnBreaks) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    int entry(int selector, int value) {
      switch (selector) {
        case 2:
          switch (value) {
            case 0: ++value; break;
            default: break;
          }
          value += 3;
          break;
        default:
          value += 5;
          break;
        case -4:
          value += 7;
      }
      return value;
    }
  )"),
                IREE_SV("switch.cpp"), options);
  auto* function = definition(source);
  ASSERT_NE(function, nullptr);
  auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                   function->functionBody)
                   ->statement;
  auto* outer =
      cxx::ast_cast<cxx::SwitchStatementAST>(body->statementList->value);
  ASSERT_NE(outer, nullptr);
  auto* outer_body = cxx::ast_cast<cxx::CompoundStatementAST>(outer->statement);
  ASSERT_NE(outer_body, nullptr);
  auto* nested = cxx::ast_cast<cxx::SwitchStatementAST>(
      outer_body->statementList->next->value);
  ASSERT_NE(nested, nullptr);

  Types types(source.unit(), source.diagnostics());
  ControlFlow analysis(source.unit(), source.diagnostics(), types, body);
  const auto& outer_layout = analysis.switch_layout(outer);
  ASSERT_EQ(outer_layout.clauses.size(), 3u);
  EXPECT_TRUE(outer_layout.has_default);
  EXPECT_TRUE(outer_layout.direct);
  EXPECT_EQ(analysis.breaks(outer), ExitFlow::None);
  EXPECT_EQ(analysis.continues(outer), ExitFlow::None);
  EXPECT_EQ(analysis.returns(outer), ExitFlow::None);

  const auto& nested_layout = analysis.switch_layout(nested);
  ASSERT_EQ(nested_layout.clauses.size(), 2u);
  EXPECT_TRUE(nested_layout.has_default);
  EXPECT_TRUE(nested_layout.direct);
  EXPECT_EQ(analysis.breaks(nested), ExitFlow::None);
  EXPECT_EQ(analysis.continues(nested), ExitFlow::None);
  EXPECT_EQ(analysis.returns(nested), ExitFlow::None);
}

TEST(ControlFlowTest, IterationExitsStopAtTheirLoopAndExcludeUnreachablePaths) {
  struct Case {
    // Statements inside the outer counted loop's body.
    const char* body;
    // Paths that continue the outer loop.
    ExitFlow continues;
    // Paths that break from the outer loop.
    ExitFlow breaks;
    // Paths that leave the current iteration by either transfer.
    ExitFlow iteration_exits;
    // Paths that return from the enclosing function.
    ExitFlow returns;
  };
  const Case cases[] = {
      {"", ExitFlow::None, ExitFlow::None, ExitFlow::None, ExitFlow::None},
      {"continue;", ExitFlow::All, ExitFlow::None, ExitFlow::All,
       ExitFlow::None},
      {"break;", ExitFlow::None, ExitFlow::All, ExitFlow::All, ExitFlow::None},
      {"if (x) continue;", ExitFlow::Some, ExitFlow::None, ExitFlow::Some,
       ExitFlow::None},
      {"if (x) break;", ExitFlow::None, ExitFlow::Some, ExitFlow::Some,
       ExitFlow::None},
      {"if (x) break; else continue;", ExitFlow::Some, ExitFlow::Some,
       ExitFlow::All, ExitFlow::None},
      {"if (x) break; else break;", ExitFlow::None, ExitFlow::All,
       ExitFlow::All, ExitFlow::None},
      {"if constexpr(true) break; else continue;", ExitFlow::None,
       ExitFlow::All, ExitFlow::All, ExitFlow::None},
      {"if constexpr(false) break; else return;", ExitFlow::None,
       ExitFlow::None, ExitFlow::None, ExitFlow::All},
      {"if constexpr(false) break;", ExitFlow::None, ExitFlow::None,
       ExitFlow::None, ExitFlow::None},
      {"if constexpr(true) { if (x) break; } else return;", ExitFlow::None,
       ExitFlow::Some, ExitFlow::Some, ExitFlow::None},
      {"if (x) { if (y) continue; } else { if (y) break; }", ExitFlow::Some,
       ExitFlow::Some, ExitFlow::Some, ExitFlow::None},
      {"if (x) break; return;", ExitFlow::None, ExitFlow::Some, ExitFlow::Some,
       ExitFlow::Some},
      {"break; return;", ExitFlow::None, ExitFlow::All, ExitFlow::All,
       ExitFlow::None},
      {"return; break;", ExitFlow::None, ExitFlow::None, ExitFlow::None,
       ExitFlow::All},
      {"while (x) { --x; continue; }", ExitFlow::None, ExitFlow::None,
       ExitFlow::None, ExitFlow::None},
      {"do { --x; break; } while (x);", ExitFlow::None, ExitFlow::None,
       ExitFlow::None, ExitFlow::None},
      {"for (unsigned j=0; j<4u; ++j) { if (y) break; }", ExitFlow::None,
       ExitFlow::None, ExitFlow::None, ExitFlow::None},
      {"while (x) { break; return; }", ExitFlow::None, ExitFlow::None,
       ExitFlow::None, ExitFlow::None},
      {"do { if (x) break; return; } while (y);", ExitFlow::None,
       ExitFlow::None, ExitFlow::None, ExitFlow::Some},
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
    EXPECT_EQ(analysis.breaks(loop->statement), test.breaks);
    EXPECT_EQ(analysis.iteration_exits(loop->statement), test.iteration_exits);
    EXPECT_EQ(analysis.returns(loop->statement), test.returns);
    EXPECT_EQ(analysis.continues(loop), ExitFlow::None);
    EXPECT_EQ(analysis.breaks(loop), ExitFlow::None);
    EXPECT_EQ(analysis.iteration_exits(loop), ExitFlow::None);
    EXPECT_EQ(analysis.continues(body), ExitFlow::None);
    EXPECT_EQ(analysis.breaks(body), ExitFlow::None);
    EXPECT_EQ(analysis.iteration_exits(body), ExitFlow::None);
    EXPECT_NE(analysis.counted(loop), nullptr);
  }
}

}  // namespace
}  // namespace loom::cxx_import
