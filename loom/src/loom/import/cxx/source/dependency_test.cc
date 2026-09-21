// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/dependent_types.h>
#include <cxx/symbols.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class InitializerVisitor final : public cxx::ASTVisitor {
 public:
  void visit(cxx::InitDeclaratorAST* ast) override {
    variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol);
  }

  // Sole variable declaration in each source, borrowed from its arena.
  cxx::VariableSymbol* variable = nullptr;
};

TEST(DependencyTest, SelfAddressDoesNotHideDependenceInOtherOperands) {
  struct Case {
    // Valid source with a single automatic variable initializer.
    const char* source;
    // Whether the initializer depends on a template parameter.
    bool dependent;
  };
  const Case cases[] = {
      {"int entry() { const int value = ((void)&value, 7); return value; }",
       false},
      {"template<int Count> int entry() { "
       "const int value = ((void)&value, Count); return value; }",
       true},
      {"template<int Count> int entry() { "
       "const int value = Count + ((void)&value, 7); return value; }",
       true},
      {"template<int Count> int entry() { "
       "const int value = ((void)&value, 7); return value; }",
       false},
  };
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  for (const auto& test : cases) {
    SCOPED_TRACE(test.source);
    Source source(view(test.source), IREE_SV("dependency.cpp"), options);
    InitializerVisitor visitor;
    visitor.accept(source.unit().ast());
    ASSERT_NE(visitor.variable, nullptr);
    auto* initializer = visitor.variable->initializer();
    ASSERT_NE(initializer, nullptr);
    EXPECT_EQ(cxx::isDependent(&source.unit(), initializer), test.dependent);
    EXPECT_EQ(
        cxx::isDependentTemplateArgument(
            &source.unit(),
            cxx::TemplateArgument{static_cast<cxx::Symbol*>(visitor.variable)}),
        test.dependent);
    EXPECT_EQ(cxx::isDependent(&source.unit(), initializer), test.dependent);
  }
}

}  // namespace
}  // namespace loom::cxx_import
