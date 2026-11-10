// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bindings.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

TEST(ValueBindingsTest, PartialSnapshotsRemainIndependentThroughCompletion) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"(
    struct Record { unsigned first, second, third; };
    void entry(Record left, Record right);
  )"),
                IREE_SV("bindings.cxx"), options);
  Types types(source.unit(), source.diagnostics());
  auto* overloads = cxx::symbol_cast<cxx::OverloadSetSymbol>(
      *source.unit().globalScope()->find("entry").begin());
  ASSERT_NE(overloads, nullptr);
  ASSERT_EQ(overloads->functions().size(), 1u);
  auto* function = overloads->functions().front();
  auto* left = function->parameters()[0];
  auto* right = function->parameters()[1];
  const auto& partition = types.partition(left->type(), source.unit().ast());
  ASSERT_EQ(partition.component_count, 3u);

  ValueBindings values;
  ValueArena arena;
  values.declare(left, partition);
  EXPECT_TRUE(values.contains(left));
  EXPECT_FALSE(values.initialized(left));
  EXPECT_FALSE(values.project(left, partition, 0, arena));
  values.replace(left, 1, Value(11), arena);
  auto original = values;
  values.replace(left, 1, Value(22), arena);
  values.replace(left, 0, Value(1), arena);
  EXPECT_FALSE(values.initialized(left));
  values.replace(left, 2, Value(3), arena);
  ASSERT_TRUE(values.initialized(left));
  EXPECT_EQ(values.at(left).components()[1], 22u);
  EXPECT_FALSE(original.initialized(left));
  auto member = original.project(left, kSSAPartition, 1, arena);
  ASSERT_TRUE(member);
  EXPECT_EQ(member->ssa(), 11u);
  EXPECT_FALSE(original.project(left, kSSAPartition, 0, arena));

  values.set(right, values.at(left));
  values.replace(left, 1, Value(33), arena);
  EXPECT_EQ(values.at(left).components()[1], 33u);
  EXPECT_EQ(values.at(right).components()[1], 22u);

  // Source scope transfers keep pending payloads and absence independent of
  // the replacements made while those bindings are hidden.
  auto pending = original.take(left);
  EXPECT_FALSE(original.contains(left));
  original.set(left, values.at(left));
  original.restore(left, std::move(pending));
  EXPECT_FALSE(original.initialized(left));
  EXPECT_EQ(original.project(left, kSSAPartition, 1, arena)->ssa(), 11u);
  EXPECT_FALSE(original.project(left, kSSAPartition, 0, arena));
  auto complete = values.take(right);
  values.declare(right, partition);
  values.restore(right, std::move(complete));
  EXPECT_EQ(values.at(right).components()[1], 22u);
  auto absent = original.take(right);
  original.declare(right, partition);
  original.restore(right, std::move(absent));
  EXPECT_FALSE(original.contains(right));
  original.erase(left);
  EXPECT_FALSE(original.contains(left));
  values.clear();
  EXPECT_FALSE(values.contains(left));
  EXPECT_FALSE(values.contains(right));
}

}  // namespace
}  // namespace loom::cxx_import
