// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/component_names.h"

#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/value/builder_test.h"

namespace loom::cxx_import {
namespace {

using ComponentNamesTest = ValueBuilderTest;

TEST_F(ComponentNamesTest, ArrayElementsAndRecordFieldsShareTheAdmittedNames) {
  Source source(IREE_SV("struct Row { int* values[2]; int count; };"),
                IREE_SV("row.cxx"), options());
  Types types(source.unit(), source.diagnostics(), module_);
  const auto* row =
      types.record((*source.unit().globalScope()->find("Row").begin())->type(),
                   source.unit().ast());
  ASSERT_NE(row, nullptr);
  const std::vector<std::string> expected = {"values_0", "values_0_byte_offset",
                                             "values_1", "values_1_byte_offset",
                                             "count"};
  EXPECT_EQ(row->component_names, expected);

  std::vector<loom_type_t> component_types;
  types.append(row->source->type(), source.unit().ast(), component_types);
  std::vector<loom_value_id_t> ids;
  for (auto type : component_types) {
    loom_value_id_t id;
    IREE_ASSERT_OK(loom_module_define_value(module_, type, &id));
    ids.push_back(id);
  }

  ValueArena arena;
  auto value = arena.capture(*row, ids);
  // Model a row constructed from an already named pointer and fresh results.
  name_value(module_, ids[0], "input");
  name_value(module_, value, "row");
  name_value(module_, value, "alias");
  for (size_t index = 0; index < ids.size(); ++index) {
    auto name = loom_module_value_name(module_, ids[index]);
    EXPECT_EQ(std::string_view(name.data, name.size),
              index == 0 ? "input" : "row_" + expected[index]);
  }
}

}  // namespace
}  // namespace loom::cxx_import
