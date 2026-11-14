// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/object_identity.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>

#include <string>
#include <unordered_map>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

// Supplies the same postorder source events as ControlFlow, without callable
// propagation or IR emission, so construction equivalence is tested directly.
class ConstructionSites final : private cxx::ASTVisitor {
 public:
  explicit ConstructionSites(ObjectIdentity& identities)
      : identities_(identities) {}

  void retain(cxx::AST* source) {
    accept(source);
    identities_.finish();
  }

  cxx::VariableSymbol* binding(const char* name) const {
    return bindings_.at(name);
  }

 private:
  void visit(cxx::InitDeclaratorAST* declaration) override {
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(declaration->symbol);
    if (!variable) {
      return;
    }
    identities_.bind(variable, nullptr, false);
    accept(variable->initializer());
    identities_.bind(variable, variable->initializer(), false);
    bindings_.emplace(cxx::to_string(variable->name()), variable);
  }

  void postVisit(cxx::AST* source) override {
    if (auto* expression = cxx::ast_cast<cxx::ExpressionAST>(source)) {
      identities_.retain(expression);
    }
  }

  // Construction groups owned by the test invocation.
  ObjectIdentity& identities_;
  // Named declarations used to select independently owned source objects.
  std::unordered_map<std::string, cxx::VariableSymbol*> bindings_;
};

TEST(ObjectIdentityTest, DirectInitializationSharesIdentityButCopyDoesNot) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Object { unsigned value; };
    void entry(unsigned input) {
      Object original{input};
      Object copied = original;
      Object direct = Object{input};
    }
  )"),
                IREE_SV("identity.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  ObjectIdentity identities(source.unit(), types);
  ConstructionSites sites(identities);
  sites.retain(source.unit().ast());
  auto* original = sites.binding("original");
  auto* copied = sites.binding("copied");
  auto* direct = sites.binding("direct");
  EXPECT_TRUE(identities.require_storage(original));
  EXPECT_FALSE(identities.requires_storage(copied));
  EXPECT_FALSE(identities.requires_storage(direct));
  auto* initializer =
      cxx::Initializer(direct->initializer()).singleExpression();
  EXPECT_TRUE(identities.require_storage(initializer));
  EXPECT_TRUE(identities.requires_storage(direct));
  EXPECT_FALSE(identities.require_storage(direct));
  EXPECT_FALSE(identities.requires_storage(copied));
}

TEST(ObjectIdentityTest, OwnedSubobjectSharesDemandButReferencedObjectDoesNot) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Object { unsigned value; };
    struct Owned { Object object; };
    struct Borrowed { const Object& object; };
    void entry(unsigned input) {
      Owned owned{Object{input}};
      Owned parenthesized(Object{input});
      Borrowed borrowed{Object{input}};
    }
  )"),
                IREE_SV("subobjects.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  ObjectIdentity identities(source.unit(), types);
  ConstructionSites sites(identities);
  sites.retain(source.unit().ast());
  auto* owned = sites.binding("owned");
  auto* borrowed = sites.binding("borrowed");
  auto* parenthesized = sites.binding("parenthesized");
  auto* owned_element =
      cxx::Initializer(owned->initializer()).arguments().front();
  auto* referenced =
      cxx::Initializer(borrowed->initializer()).arguments().front();
  EXPECT_TRUE(identities.require_storage(owned_element));
  EXPECT_TRUE(identities.requires_storage(owned));
  auto* paren_element =
      cxx::Initializer(parenthesized->initializer()).arguments().front();
  EXPECT_TRUE(identities.require_storage(paren_element));
  EXPECT_TRUE(identities.requires_storage(parenthesized));
  EXPECT_TRUE(identities.require_storage(borrowed));
  EXPECT_FALSE(identities.requires_storage(referenced));
}

}  // namespace
}  // namespace loom::cxx_import
