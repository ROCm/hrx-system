// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/initializer_lists.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <vector>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class Lists final : public cxx::ASTVisitor {
 public:
  explicit Lists(cxx::TranslationUnit& unit) : unit_(unit) {}
  void visit(cxx::BracedInitListAST* ast) override {
    if (unit_.typeTraits().initializer_list_element_type(ast->type)) {
      lists.push_back(ast);
    }
    cxx::ASTVisitor::visit(ast);
  }

  // Borrowed checked list expressions in source order.
  std::vector<cxx::BracedInitListAST*> lists;

 private:
  // Owns the checked source type identities.
  cxx::TranslationUnit& unit_;
};

TEST(InitializerListsTest, RetainsConversionsAndDistinctBackingIdentity) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    #include <initializer_list>
    void consume(std::initializer_list<long>);
    void entry(int value) { consume({value, 8}); consume({value, 8}); }
  )"),
                IREE_SV("lists.cxx"), options);
  Lists syntax(source.unit());
  syntax.accept(source.unit().ast());
  ASSERT_EQ(syntax.lists.size(), 2u);
  InitializerLists plans(source.unit(), source.diagnostics());
  EXPECT_EQ(plans.get(syntax.lists[0]), nullptr);
  auto* original_type = syntax.lists[0]->type;
  auto* original_clauses = syntax.lists[0]->expressionList;
  auto& first = plans.retain(syntax.lists[0]);
  auto& second = plans.retain(syntax.lists[1]);
  EXPECT_EQ(plans.get(syntax.lists[0]), &first);
  EXPECT_NE(first.backing, second.backing);
  EXPECT_EQ(first.backing->expressionList, original_clauses);
  EXPECT_EQ(syntax.lists[0]->expressionList, original_clauses);
  EXPECT_EQ(syntax.lists[0]->type, original_type);
  auto* array = cxx::type_cast<cxx::BoundedArrayType>(first.backing->type);
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->size(), 2u);
  auto traits = source.unit().typeTraits();
  EXPECT_TRUE(traits.is_const(array->elementType()));
  EXPECT_EQ(traits.remove_const(array->elementType()),
            original_clauses->value->type);
  EXPECT_TRUE(traits.is_pointer(first.begin->type()));
  EXPECT_TRUE(traits.is_integral(first.limit->type()));
  EXPECT_EQ(first.backing->lbraceLoc, syntax.lists[0]->lbraceLoc);
}

TEST(InitializerListsTest, PreservesPointerPairLayoutAndEmptyExtent) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    namespace std {
    template<class E> class initializer_list {
      const E* first;
      const E* last;
    public:
      const E* begin() const { return first; }
      const E* end() const { return last; }
    };
    }
    void consume(std::initializer_list<int>);
    void entry(int value) { consume({value, 9}); consume({}); }
  )"),
                IREE_SV("pointer_pair.cxx"), options);
  Lists syntax(source.unit());
  syntax.accept(source.unit().ast());
  ASSERT_EQ(syntax.lists.size(), 2u);
  InitializerLists plans(source.unit(), source.diagnostics());
  auto& first = plans.retain(syntax.lists[0]);
  auto& empty = plans.retain(syntax.lists[1]);
  EXPECT_EQ(first.begin->type(), first.limit->type());
  auto* array = cxx::type_cast<cxx::BoundedArrayType>(empty.backing->type);
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->size(), 0u);
  EXPECT_EQ(empty.backing->expressionList, nullptr);
}

}  // namespace
}  // namespace loom::cxx_import
