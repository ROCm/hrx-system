// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/loops/counted.h"

#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <limits>
#include <span>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"

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

struct HeaderCase {
  // Source argument list, including C++ qualifiers and integer widths.
  const char* parameters;
  // Source statement with its comparison, increment, and initialization.
  const char* loop;
  // Expected step, or zero for an unrecognized interval.
  unsigned step;
  // Folded scalar upper limit; expressions retain their checked AST.
  std::optional<std::int64_t> constant = std::nullopt;
};

void check_headers(std::span<const HeaderCase> cases,
                   loom_scalar_type_t domain) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.loop);
    std::string text =
        "constexpr unsigned capacity=320; const unsigned stride=4; ";
    text += domain == LOOM_SCALAR_TYPE_OFFSET ? "unsigned opaque();"
                                              : "int opaque();";
    text +=
        std::string("void entry(") + test.parameters + ") {" + test.loop + "}";
    Source source(view(text), IREE_SV("loop_header.cxx"), options);
    auto* function = definition(source);
    ASSERT_NE(function, nullptr);
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
                     function->functionBody)
                     ->statement;
    auto* loop =
        cxx::ast_cast<cxx::ForStatementAST>(body->statementList->value);
    Types types(source.unit(), source.diagnostics());
    const auto interval = CountedLoop::recognize(source.unit(), types, loop);
    ASSERT_EQ(interval.has_value(), test.step != 0);
    if (!interval) {
      continue;
    }
    EXPECT_EQ(interval->domain, domain);
    EXPECT_FALSE(interval->native);
    EXPECT_EQ(interval->step, test.step);
    EXPECT_EQ(cxx::to_string(interval->induction->name()), "i");
    if (test.constant) {
      EXPECT_EQ(std::get<std::int64_t>(interval->upper), *test.constant);
    } else {
      auto& bound = std::get<CountedLoop::Bound>(interval->upper);
      ASSERT_NE(bound.identifier, nullptr);
      EXPECT_EQ(bound.identifier->symbol, function->symbol->parameters()[0]);
      EXPECT_EQ(bound.expression,
                cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition)
                    ->rightExpression);
    }
  }
}

TEST(CountedLoopTest, UnsignedIntervals) {
  const HeaderCase cases[] = {
      {"unsigned n", "for (unsigned i=0; i<n; ++i) {}", 1},
      {"unsigned n", "for (unsigned i=0; i<n; ++i) ;", 1},
      {"unsigned n", "for (unsigned i=0; i<16u; i+=4u) n += i;", 4, 16},
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
      {"unsigned n", "for (unsigned i=0; i<4294967295u; i+=4u) {}", 0},
      {"unsigned long n", "for (unsigned i=0; i<n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; --i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<=n; ++i) {}", 0},
      {"unsigned n", "for (unsigned i=0; i<n; i+=0u) {}", 0},
  };
  check_headers(cases, LOOM_SCALAR_TYPE_OFFSET);
}

TEST(CountedLoopTest, SignedIntervals) {
  const HeaderCase cases[] = {
      {"int n", "for (int i=0; i<n; ++i) {}", 1},
      {"short n", "for (int i=0; i<n; ++i) {}", 1},
      {"unsigned short n", "for (int i=0; i<n; ++i) {}", 1},
      {"int n", "for (int i=-4; i<3; i+=2) {}", 2, 3},
      {"int n", "for (int i=n; i<-3; i++) {}", 1, -3},
      {"int n", "for (int i=n; i<2147483647; ++i) {}", 1, 2147483647},
      {"int n", "for (int i=n; i<2147483644; i+=4) {}", 4, 2147483644},
      {"int n", "for (int i=n; i<(-2147483647-1); ++i) {}", 1,
       std::numeric_limits<std::int32_t>::min()},
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
  check_headers(cases, LOOM_SCALAR_TYPE_INDEX);
}

TEST(CountedLoopTest, NativeAddressIntervals) {
  struct Case {
    // Fully qualified native address type used by the source loop.
    const char* type;
    // Standalone source loop whose classification is under test.
    const char* loop;
    // Expected native range, or no value for general while semantics.
    std::optional<loom_scalar_type_t> domain;
    // Whether the admitted upper bound is an exact temporary constant.
    bool constant_bound = false;
  };
  const Case cases[] = {
      {"loom::type::index",
       "for (auto i=begin; i<end; i=i+loom::type::index{1}) {}",
       LOOM_SCALAR_TYPE_INDEX},
      {"loom::type::index",
       "for (auto i=begin; i<end; i=loom::type::index(1)+i) {}",
       LOOM_SCALAR_TYPE_INDEX},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=i+loom::type::offset{1}) {}",
       LOOM_SCALAR_TYPE_OFFSET},
      {"loom::type::offset",
       "for (auto i=begin; i<end; i=loom::type::offset{1}+i) {}",
       LOOM_SCALAR_TYPE_OFFSET},
      {"loom::type::index",
       "for (auto i=begin; i<loom::type::index{4}; "
       "i=i+loom::type::index{1}) {}",
       LOOM_SCALAR_TYPE_INDEX, true},
      {"loom::type::offset",
       "for (auto i=begin; i<loom::type::offset(4); "
       "i=i+loom::type::offset{1}) {}",
       LOOM_SCALAR_TYPE_OFFSET, true},
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
    auto counted = CountedLoop::recognize(source.unit(), types, loop);
    ASSERT_EQ(counted.has_value(), test.domain.has_value());
    if (!counted) {
      continue;
    }
    EXPECT_EQ(counted->induction,
              cxx::ast_cast<cxx::SimpleDeclarationAST>(
                  cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer)
                      ->declaration)
                  ->initDeclaratorList->value->symbol);
    EXPECT_EQ(counted->domain, *test.domain);
    EXPECT_TRUE(counted->native);
    EXPECT_EQ(counted->step, 1u);
    auto& bound = std::get<CountedLoop::Bound>(counted->upper);
    EXPECT_EQ(
        bound.identifier ? bound.identifier->symbol : nullptr,
        test.constant_bound ? nullptr : function->symbol->parameters()[1]);
    EXPECT_EQ(bound.expression,
              cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition)
                  ->rightExpression);
  }
}

}  // namespace
}  // namespace loom::cxx_import
