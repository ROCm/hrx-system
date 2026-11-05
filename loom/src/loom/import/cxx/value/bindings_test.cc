// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bindings.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>

#include <array>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

TEST(ValueBindingsTest, MaintainsLazyOwnersAcrossUpdatesAndSnapshots) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("void entry(unsigned first, unsigned second) {}"),
                IREE_SV("bindings.cxx"), options);
  auto* root = cxx::ast_cast<cxx::TranslationUnitAST>(source.unit().ast());
  cxx::FunctionDefinitionAST* definition = nullptr;
  for (auto* declaration : cxx::ListView{root->declarationList}) {
    if ((definition = cxx::ast_cast<cxx::FunctionDefinitionAST>(declaration))) {
      break;
    }
  }
  ASSERT_NE(definition, nullptr);
  auto parameters = definition->symbol->parameters();
  ASSERT_EQ(parameters.size(), 2u);
  auto* first = parameters[0];
  auto* second = parameters[1];
  ValueBindings bindings;
  bindings.set(first, Value(1));
  bindings.set(second, Value(1));

  auto* owners = bindings.owners(1);
  ASSERT_NE(owners, nullptr);
  EXPECT_EQ(owners->at(first), 1u);
  EXPECT_EQ(owners->at(second), 1u);

  ValueBindings snapshot = bindings;
  bindings.set(first, Value(2));
  owners = bindings.owners(1);
  ASSERT_NE(owners, nullptr);
  EXPECT_FALSE(owners->contains(first));
  EXPECT_EQ(owners->at(second), 1u);
  EXPECT_EQ(bindings.owners(2)->at(first), 1u);

  bindings.erase(second);
  EXPECT_EQ(bindings.owners(1), nullptr);
  bindings = snapshot;
  EXPECT_EQ(bindings.owners(1)->size(), 2u);
  bindings.clear();
  EXPECT_EQ(bindings.owners(1), nullptr);

  const Partition pair = {ValueKind::Record, 2};
  const std::array<loom_value_id_t, 2> repeated_components = {3, 3};
  ValueArena arena;
  bindings.set(first, arena.capture(pair, repeated_components));
  EXPECT_EQ(bindings.owners(3)->at(first), 2u);
}

}  // namespace
}  // namespace loom::cxx_import
