// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/construction.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include <array>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

TEST(ConstructionTest, BreakPathsConstrainExitWithoutConstrainingFallthrough) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV("struct Pair { unsigned first; unsigned second; }; "
                        "void entry(Pair result, unsigned temporary);"),
                IREE_SV("construction.cxx"), options);
  auto* overloads = cxx::symbol_cast<cxx::OverloadSetSymbol>(
      *source.unit().globalScope()->find("entry").begin());
  ASSERT_NE(overloads, nullptr);
  ASSERT_EQ(overloads->functions().size(), 1u);
  auto* function = overloads->functions().front();
  auto* result = function->parameters()[0];
  auto* temporary = function->parameters()[1];
  auto* owner = source.unit().ast();
  const std::array<cxx::Symbol*, 2> written = {result, temporary};

  Construction construction;
  construction.enter();
  construction.declare(result, 2);
  construction.assign(result, 0, 1);
  auto initial = construction.state();
  construction.enter();
  construction.declare(temporary, 1);
  construction.assign(temporary, 0, 1);
  construction.assign(result, 1, 1);
  auto continuing = construction.state();
  construction.restore(initial);
  construction.break_from_region();
  construction.restore(continuing);
  construction.leave(owner, written);

  EXPECT_TRUE(construction.initialized(owner, result, 0, 1));
  EXPECT_FALSE(construction.initialized(owner, result, 1, 1));
  EXPECT_FALSE(construction.tracked(temporary));
  EXPECT_FALSE(construction.state().available.contains(temporary));
  EXPECT_EQ(construction.state().available.at(result)[1], 1u);
  construction.finish_loop(construction.state());
  EXPECT_EQ(construction.state().available.at(result)[0], 1u);
  EXPECT_EQ(construction.state().available.at(result)[1], 0u);
}

TEST(ConstructionTest, ReturningAlternativeDoesNotInventAnUninitializedPath) {
  Construction::State returned{false, {}};
  Construction::State continuing{true, {}};
  EXPECT_TRUE(Construction::join(returned, continuing).reachable);
  EXPECT_TRUE(Construction::join(continuing, returned).reachable);
  EXPECT_FALSE(Construction::join(returned, returned).reachable);
}

}  // namespace
}  // namespace loom::cxx_import
