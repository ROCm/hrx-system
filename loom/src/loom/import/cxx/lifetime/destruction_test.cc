// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/lifetime/destruction.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <string_view>
#include <unordered_map>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class Variables final : public cxx::ASTVisitor {
 public:
  void visit(cxx::InitDeclaratorAST* ast) override {
    if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol)) {
      variables.emplace(cxx::to_string(variable->name()), variable);
    }
    cxx::ASTVisitor::visit(ast);
  }

  std::unordered_map<std::string, cxx::VariableSymbol*> variables;
};

TEST(DestructionTest, RetainsSelectedCompleteObjectDestructorAndArrayExtent) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Guard { unsigned value; ~Guard() {} };
    struct Plain { unsigned value; };
    void entry() {
      Guard one{};
      Guard many[2][3]{};
      Plain plain{};
      Guard& borrowed = one;
      const Guard& constant_borrow = one;
    }
  )"),
                IREE_SV("destruction.cxx"), options);
  Variables variables;
  variables.accept(source.unit().ast());

  auto one =
      destruction_for(source.unit(), variables.variables.at("one")->type());
  ASSERT_TRUE(one);
  EXPECT_TRUE(one->function->isDestructor());
  EXPECT_EQ(one->element_count, 1u);

  auto many =
      destruction_for(source.unit(), variables.variables.at("many")->type());
  ASSERT_TRUE(many);
  EXPECT_EQ(many->function->canonical(), one->function->canonical());
  EXPECT_EQ(many->element_count, 6u);

  EXPECT_FALSE(
      destruction_for(source.unit(), variables.variables.at("plain")->type()));
  EXPECT_FALSE(destruction_for(source.unit(),
                               variables.variables.at("borrowed")->type()));
  EXPECT_FALSE(destruction_for(
      source.unit(), variables.variables.at("constant_borrow")->type()));
}

}  // namespace
}  // namespace loom::cxx_import
