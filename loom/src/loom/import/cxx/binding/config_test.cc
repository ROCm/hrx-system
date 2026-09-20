// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/config.h"

#include <cxx/symbols.h>

#include "loom/import/cxx/value/builder_test.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {
using ConfigTest = ValueBuilderTest;

TEST_F(ConfigTest, ReadsUseOneSymbolForRedeclarationsAndAliases) {
  Source source(IREE_SV(R"(
    [[loom::config("tuning.factor")]] extern const unsigned first;
    [[loom::config("tuning.factor")]] extern const unsigned first;
    [[loom::config("tuning.factor")]] extern const unsigned alias;
    const unsigned ordinary = 3;
  )"),
                IREE_SV("config.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  SymbolNames names(source.unit(), source.diagnostics());
  Configs configs(source.unit(), source.diagnostics(), types, scalars,
                  locations, names);
  std::vector<cxx::VariableSymbol*> variables;
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration);
    if (!simple || !simple->initDeclaratorList) {
      continue;
    }
    auto* declarator = simple->initDeclaratorList->value;
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
    if (!variable) {
      continue;
    }
    variables.push_back(variable);
    EXPECT_EQ(configs.declaration(variable, simple->attributeList, declarator),
              variables.size() != 4);
  }
  ASSERT_EQ(variables.size(), 4u);
  configs.build(&builder_);
  ASSERT_EQ(module_->symbols.count, 1u);
  ASSERT_TRUE(loom_config_decl_isa(loom_module_block(module_)->first_op));
  for (size_t i = 0; i < 3; ++i) {
    auto value = configs.read(variables[i], &builder_, LOOM_LOCATION_UNKNOWN);
    ASSERT_TRUE(value);
    auto* op = producer(*value);
    ASSERT_TRUE(loom_config_get_isa(op));
    EXPECT_EQ(loom_config_get_config(op).symbol_id, 0u);
    EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, *value),
                                loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  }
  EXPECT_FALSE(
      configs.read(variables.back(), &builder_, LOOM_LOCATION_UNKNOWN));
}

TEST_F(ConfigTest, LaterDefinitionFixesEarlierDeclarationsAndAliases) {
  Source source(IREE_SV(R"(
    [[loom::config("tuning.mask")]] extern const unsigned first;
    [[loom::config("tuning.mask")]] extern const unsigned alias;
    [[loom::config("tuning.mask")]] const unsigned first = 0x80000000u;
  )"),
                IREE_SV("config.cpp"), options());
  Types types(source.unit(), source.diagnostics());
  Locations locations(source.unit(), source.diagnostics(), module_);
  Scalars scalars(source.unit(), source.diagnostics(), types, locations,
                  builder_);
  SymbolNames names(source.unit(), source.diagnostics());
  Configs configs(source.unit(), source.diagnostics(), types, scalars,
                  locations, names);
  std::vector<cxx::VariableSymbol*> variables;
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    auto* simple = cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration);
    if (!simple || !simple->initDeclaratorList) {
      continue;
    }
    auto* declarator = simple->initDeclaratorList->value;
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(declarator->symbol);
    if (!variable) {
      continue;
    }
    variables.push_back(variable);
    EXPECT_TRUE(
        configs.declaration(variable, simple->attributeList, declarator));
  }
  ASSERT_EQ(variables.size(), 3u);
  configs.build(&builder_);
  ASSERT_EQ(module_->symbols.count, 1u);
  ASSERT_TRUE(loom_config_def_isa(loom_module_block(module_)->first_op));
  for (auto* variable : variables) {
    auto value = configs.read(variable, &builder_, LOOM_LOCATION_UNKNOWN);
    ASSERT_TRUE(value);
    auto* op = producer(*value);
    ASSERT_TRUE(loom_scalar_constant_isa(op));
    EXPECT_EQ(loom_scalar_constant_value(op).i64, INT32_MIN);
  }
}

}  // namespace
}  // namespace loom::cxx_import
