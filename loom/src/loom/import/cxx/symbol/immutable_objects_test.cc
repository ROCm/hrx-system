// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/immutable_objects.h"

#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include <algorithm>
#include <array>

#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/global/ops.h"

namespace loom::cxx_import {
namespace {
using ImmutableObjectsTest = ValueBuilderTest;

cxx::VariableSymbol* variable(cxx::TranslationUnit& unit, const char* name) {
  auto symbols = unit.globalScope()->find(name);
  if (symbols.begin() == symbols.end()) {
    return nullptr;
  }
  return cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
}

cxx::VariableSymbol* variable(cxx::NamespaceSymbol* scope, const char* name) {
  auto symbols = scope->find(name);
  if (symbols.begin() == symbols.end()) {
    return nullptr;
  }
  return cxx::symbol_cast<cxx::VariableSymbol>(*symbols.begin());
}

iree_string_view_t symbol_name(const loom_module_t* module,
                               loom_symbol_ref_t symbol) {
  return loom_string_table_get(
      &module->strings, module->symbols.entries[symbol.symbol_id].name_id);
}

TEST_F(ImmutableObjectsTest, RetainsEachReachedSourceObjectOnce) {
  Source source(IREE_SV(R"(
    namespace detail {
    static constexpr unsigned table[4] = {11, 22, 33, 44};
    }
  )"),
                IREE_SV("immutable_objects.cxx"), options());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);
  auto* detail = cxx::symbol_cast<cxx::NamespaceSymbol>(
      *source.unit().globalScope()->find("detail").begin());
  ASSERT_NE(detail, nullptr);
  auto* table =
      cxx::symbol_cast<cxx::VariableSymbol>(*detail->find("table").begin());
  ASSERT_NE(table, nullptr);

  auto first = objects.reference(table, table->initializer());
  auto second = objects.reference(table, table->initializer());
  EXPECT_EQ(first.symbol.symbol_id, second.symbol.symbol_id);
  EXPECT_TRUE(
      iree_string_view_equal(first.value_name, IREE_SV("detail_table")));
  EXPECT_EQ(first.type, table->type());
  EXPECT_EQ(first.alignment, 4);
  objects.build(&builder_);

  auto* definition = loom_module_block(module_)->first_op;
  ASSERT_NE(definition, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(definition));
  EXPECT_EQ(loom_global_rodata_def_symbol(definition).symbol_id,
            first.symbol.symbol_id);
  EXPECT_EQ(loom_global_rodata_def_alignment(definition), 4);
  auto contents = loom_global_rodata_def_contents(definition);
  constexpr std::array<uint8_t, 16> expected = {11, 0, 0, 0, 22, 0, 0, 0,
                                                33, 0, 0, 0, 44, 0, 0, 0};
  ASSERT_EQ(contents.data_length, expected.size());
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), contents.data));
  EXPECT_EQ(definition->next_op, nullptr);
}

TEST_F(ImmutableObjectsTest, ProjectsSourceLinkage) {
  Source source(IREE_SV(R"(
    namespace tables {
    extern const unsigned declaration[2];
    extern const unsigned definition[2] = {3, 4};
    inline constexpr unsigned inline_definition[2] = {5, 6};
    static constexpr unsigned private_definition[2] = {7, 8};
    }
  )"),
                IREE_SV("immutable_object_linkage.cxx"), options());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);
  auto* tables = cxx::symbol_cast<cxx::NamespaceSymbol>(
      *source.unit().globalScope()->find("tables").begin());
  ASSERT_NE(tables, nullptr);
  auto* declaration = variable(tables, "declaration");
  auto* definition = variable(tables, "definition");
  auto* inline_definition = variable(tables, "inline_definition");
  auto* private_definition = variable(tables, "private_definition");
  ASSERT_NE(declaration, nullptr);
  ASSERT_NE(definition, nullptr);
  ASSERT_NE(inline_definition, nullptr);
  ASSERT_NE(private_definition, nullptr);

  auto* owner = source.unit().ast();
  auto declaration_object = objects.reference(declaration, owner);
  objects.retain_definition(definition, owner);
  auto definition_object = objects.reference(definition, owner);
  objects.retain_definition(inline_definition, owner);
  auto inline_object = objects.reference(inline_definition, owner);
  objects.retain_definition(private_definition, owner);
  auto private_object = objects.reference(private_definition, owner);
  EXPECT_TRUE(
      iree_string_view_equal(symbol_name(module_, declaration_object.symbol),
                             IREE_SV("tables.declaration")));
  EXPECT_TRUE(iree_string_view_equal(declaration_object.value_name,
                                     IREE_SV("tables_declaration")));
  EXPECT_TRUE(
      iree_string_view_equal(symbol_name(module_, definition_object.symbol),
                             IREE_SV("tables.definition")));
  EXPECT_TRUE(iree_string_view_equal(definition_object.value_name,
                                     IREE_SV("tables_definition")));
  EXPECT_TRUE(iree_string_view_equal(symbol_name(module_, inline_object.symbol),
                                     IREE_SV("tables.inline_definition")));
  EXPECT_TRUE(iree_string_view_equal(inline_object.value_name,
                                     IREE_SV("tables_inline_definition")));
  EXPECT_TRUE(
      iree_string_view_equal(symbol_name(module_, private_object.symbol),
                             IREE_SV("tables_private_definition")));
  EXPECT_TRUE(iree_string_view_equal(private_object.value_name,
                                     IREE_SV("tables_private_definition")));
  objects.build(&builder_);

  auto* declaration_op = loom_module_block(module_)->first_op;
  ASSERT_NE(declaration_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_decl_isa(declaration_op));
  EXPECT_EQ(loom_global_rodata_decl_symbol(declaration_op).symbol_id,
            declaration_object.symbol.symbol_id);

  auto* definition_op = declaration_op->next_op;
  ASSERT_NE(definition_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(definition_op));
  EXPECT_EQ(loom_global_rodata_def_symbol(definition_op).symbol_id,
            definition_object.symbol.symbol_id);
  EXPECT_EQ(loom_global_rodata_def_visibility(definition_op),
            LOOM_GLOBAL_RODATA_DEF_VISIBILITY_PUBLIC);
  EXPECT_EQ(loom_global_rodata_def_linkage(definition_op), 0);

  auto* inline_op = definition_op->next_op;
  ASSERT_NE(inline_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(inline_op));
  EXPECT_EQ(loom_global_rodata_def_symbol(inline_op).symbol_id,
            inline_object.symbol.symbol_id);
  EXPECT_EQ(loom_global_rodata_def_visibility(inline_op),
            LOOM_GLOBAL_RODATA_DEF_VISIBILITY_PUBLIC);
  EXPECT_EQ(loom_global_rodata_def_linkage(inline_op),
            LOOM_SYMBOL_LINKAGE_LINK_ONCE);

  auto* private_op = inline_op->next_op;
  ASSERT_NE(private_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(private_op));
  EXPECT_EQ(loom_global_rodata_def_symbol(private_op).symbol_id,
            private_object.symbol.symbol_id);
  EXPECT_EQ(loom_global_rodata_def_visibility(private_op), 0);
  EXPECT_EQ(loom_global_rodata_def_linkage(private_op), 0);
  EXPECT_EQ(private_op->next_op, nullptr);
}

TEST_F(ImmutableObjectsTest, RejectsUnsupportedIdentityAndRepresentation) {
  Source source(IREE_SV(R"(
    struct Inner { mutable unsigned value; };
    struct Outer { Inner inner; };
    static constexpr Outer mutable_object{{2}};
    static constexpr const unsigned* address_object = nullptr;
  )"),
                IREE_SV("invalid_immutable_objects.cxx"), options());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);

  auto* mutable_object = variable(source.unit(), "mutable_object");
  auto* address_object = variable(source.unit(), "address_object");
  ASSERT_NE(mutable_object, nullptr);
  ASSERT_NE(address_object, nullptr);
  EXPECT_THROW(objects.reference(mutable_object, mutable_object->initializer()),
               SourceRejected);
  EXPECT_THROW(objects.reference(address_object, address_object->initializer()),
               SourceRejected);
}

TEST_F(ImmutableObjectsTest, RetainsLocalIdentityAndSourceLinkage) {
  Source source(IREE_SV(R"(
    inline const unsigned* shared(bool first) {
      if (first) {
        alignas(64) static constexpr unsigned table[2] = {7, 11};
        return table;
      } else {
        static constexpr unsigned table[2] = {13, 17};
        return table;
      }
    }
    static inline const unsigned* private_table() {
      static constexpr unsigned table[2] = {19, 23};
      return table;
    }
  )"),
                IREE_SV("local_immutable_objects.cxx"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  SymbolNames names(source.unit(), source.diagnostics());
  ImmutableObjects objects(source.unit(), source.diagnostics(), locations,
                           names, module_);
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  std::vector<ImmutableObject> retained;
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* definition = cxx::ast_cast<cxx::FunctionDefinitionAST>(declaration);
    if (!definition) {
      continue;
    }
    auto* body = cxx::ast_cast<cxx::CompoundStatementFunctionBodyAST>(
        definition->functionBody);
    ASSERT_NE(body, nullptr);
    ControlFlow control(source.unit(), source.diagnostics(), types,
                        body->statement, definition->symbol);
    objects.retain_locals(definition->symbol, control.static_locals());
    objects.retain_locals(definition->symbol, control.static_locals());
    for (auto* local : control.static_locals()) {
      auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(local->symbol);
      ASSERT_NE(variable, nullptr);
      auto first = objects.reference(variable, local);
      auto second = objects.reference(variable, local);
      EXPECT_EQ(first.symbol.symbol_id, second.symbol.symbol_id);
      retained.push_back(first);
    }
  }
  ASSERT_EQ(retained.size(), 3u);
  EXPECT_NE(retained[0].symbol.symbol_id, retained[1].symbol.symbol_id);
  EXPECT_NE(retained[1].symbol.symbol_id, retained[2].symbol.symbol_id);
  EXPECT_EQ(retained[0].alignment, 64);
  EXPECT_EQ(retained[1].alignment, 4);
  EXPECT_EQ(retained[2].alignment, 4);
  EXPECT_TRUE(
      iree_string_view_equal(symbol_name(module_, retained[0].symbol),
                             IREE_SV("__loom_cxx_static._Z6sharedb.0")));
  EXPECT_TRUE(
      iree_string_view_equal(symbol_name(module_, retained[1].symbol),
                             IREE_SV("__loom_cxx_static._Z6sharedb.1")));
  objects.build(&builder_);
  size_t ordinal = 0;
  constexpr std::array<uint8_t, 3> first_values = {7, 13, 19};
  for (auto* definition = loom_module_block(module_)->first_op; definition;
       definition = definition->next_op, ++ordinal) {
    ASSERT_LT(ordinal, retained.size());
    ASSERT_TRUE(loom_global_rodata_def_isa(definition));
    EXPECT_EQ(loom_global_rodata_def_symbol(definition).symbol_id,
              retained[ordinal].symbol.symbol_id);
    EXPECT_EQ(loom_global_rodata_def_linkage(definition),
              ordinal < 2 ? LOOM_SYMBOL_LINKAGE_LINK_ONCE : 0);
    EXPECT_EQ(loom_global_rodata_def_visibility(definition),
              ordinal < 2 ? LOOM_GLOBAL_RODATA_DEF_VISIBILITY_PUBLIC : 0);
    auto contents = loom_global_rodata_def_contents(definition);
    ASSERT_EQ(contents.data_length, 8u);
    EXPECT_EQ(contents.data[0], first_values[ordinal]);
  }
  EXPECT_EQ(ordinal, retained.size());
}

}  // namespace
}  // namespace loom::cxx_import
