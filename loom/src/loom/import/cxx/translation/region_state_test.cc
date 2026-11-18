// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation/region_state.h"

#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include <array>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/builder_test.h"

namespace loom::cxx_import {
namespace {

using RegionStateTest = ValueBuilderTest;

TEST_F(RegionStateTest, CarriesOneMemberWithoutRebindingItsSibling) {
  Source source(IREE_SV(R"(
    struct Row { int* data; int tag; };
    void step(Row row, int count);
  )"),
                IREE_SV("region.cxx"), options());
  Types types(source.unit(), source.diagnostics(), module_);
  auto* overloads = cxx::symbol_cast<cxx::OverloadSetSymbol>(
      *source.unit().globalScope()->find("step").begin());
  const auto& parameters = overloads->functions().front()->parameters();
  auto* row = parameters[0];
  auto* count = parameters[1];
  const auto* record = types.record(row->type(), source.unit().ast());
  const auto& data = record->members.front();
  RegionState state;
  state.append(row, {data.type(), data.partition, data.component_offset});
  state.append(count, {count->type(), &kSSAPartition, 0});
  EXPECT_EQ(state.group_count(), 2u);
  EXPECT_EQ(state.component_count(), 3u);

  const std::array component_types = {loom_type_buffer(),
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                      loom_type_scalar(LOOM_SCALAR_TYPE_I32)};
  std::array<loom_value_id_t, 4> ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    IREE_ASSERT_OK(
        loom_module_define_value(module_, component_types[i], &ids[i]));
  }
  const auto [root, displacement, length, tag] = ids;
  ValueArena arena;
  ValueBindings bindings;
  bindings.declare(row, *record);
  bindings.set(count, length);
  std::vector<loom_value_id_t> missing;
  EXPECT_FALSE(state.append_values(bindings, arena, missing));
  bindings.replace(row, data.component_offset,
                   Value(Pointer{root, displacement}), arena);
  std::vector<loom_value_id_t> edge{tag};
  ASSERT_TRUE(state.append_values(bindings, arena, edge));
  EXPECT_EQ(edge,
            (std::vector<loom_value_id_t>{tag, root, displacement, length}));

  ValueBindings destination;
  destination.declare(row, *record);
  destination.replace(row, record->members.back().component_offset, tag, arena);
  destination.declare(count, kSSAPartition);
  state.bind_values(destination, arena, module_, edge.data() + 1);
  auto components = destination.at(row).components();
  ASSERT_EQ(components.size(), 3u);
  EXPECT_EQ(components[0], root);
  EXPECT_EQ(components[1], displacement);
  EXPECT_EQ(components[2], tag);
  EXPECT_EQ(destination.at(count).ssa(), length);
  EXPECT_FALSE(bindings.initialized(row));
}

}  // namespace
}  // namespace loom::cxx_import
