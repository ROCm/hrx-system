// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/ownership.h"

#include <cxx/translation_unit.h>

#include <array>
#include <vector>

#include "loom/import/cxx/value/builder_test.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

using ValueOwnershipTest = ValueBuilderTest;

TEST_F(ValueOwnershipTest, SoleOwnerKeepsItsTiedIdentity) {
  auto* owner = source_.unit().ast();
  auto* type = source_.unit().control()->getUnsignedIntType();
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  ValueArena arena;
  ValueDuplication duplication(source_.unit(), source_.diagnostics(), types_,
                               arena, locations, builder_);
  Value value(scalars.integer(7, LOOM_SCALAR_TYPE_I32));
  std::vector<loom_value_id_t> arguments(value.components().begin(),
                                         value.components().end());
  const std::array<TiedValueConsumption, 1> consumptions = {
      TiedValueConsumption{value, type, 0}};
  ValueOwnership ownership(duplication);
  ownership.add_owner(value);

  ownership.separate(consumptions, arguments, owner);

  EXPECT_EQ(arguments[0], value.ssa());
}

TEST_F(ValueOwnershipTest, SurvivingOwnerDuplicatesTheTiedOccurrence) {
  auto* owner = source_.unit().ast();
  auto* type = source_.unit().control()->getUnsignedIntType();
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  ValueArena arena;
  ValueDuplication duplication(source_.unit(), source_.diagnostics(), types_,
                               arena, locations, builder_);
  Value value(scalars.integer(7, LOOM_SCALAR_TYPE_I32));
  std::vector<loom_value_id_t> arguments(value.components().begin(),
                                         value.components().end());
  const std::array<TiedValueConsumption, 1> consumptions = {
      TiedValueConsumption{value, type, 0}};
  ValueOwnership ownership(duplication);
  ownership.add_owner(value);
  ownership.add_owner(value);

  ownership.separate(consumptions, arguments, owner);

  EXPECT_NE(arguments[0], value.ssa());
  EXPECT_TRUE(loom_scalar_duplicate_isa(producer(arguments[0])));
}

TEST_F(ValueOwnershipTest, RepeatedTiesDuplicateAllButOneOccurrence) {
  auto* owner = source_.unit().ast();
  auto* type = source_.unit().control()->getUnsignedIntType();
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  ValueArena arena;
  ValueDuplication duplication(source_.unit(), source_.diagnostics(), types_,
                               arena, locations, builder_);
  Value value(scalars.integer(7, LOOM_SCALAR_TYPE_I32));
  std::vector<loom_value_id_t> arguments = {value.ssa(), value.ssa()};
  const std::array<TiedValueConsumption, 2> consumptions = {
      TiedValueConsumption{value, type, 0},
      TiedValueConsumption{value, type, 1},
  };
  ValueOwnership ownership(duplication);
  ownership.add_owner(value);
  ownership.add_owner(value);

  ownership.separate(consumptions, arguments, owner);

  EXPECT_TRUE(loom_scalar_duplicate_isa(producer(arguments[0])));
  EXPECT_EQ(arguments[1], value.ssa());
}

TEST_F(ValueOwnershipTest, OrdinaryCallUsePreservesTheOriginalIdentity) {
  auto* owner = source_.unit().ast();
  auto* type = source_.unit().control()->getUnsignedIntType();
  Locations locations(source_.unit(), source_.diagnostics(), module_);
  Scalars scalars(source_.unit(), source_.diagnostics(), types_, locations,
                  builder_);
  ValueArena arena;
  ValueDuplication duplication(source_.unit(), source_.diagnostics(), types_,
                               arena, locations, builder_);
  Value value(scalars.integer(7, LOOM_SCALAR_TYPE_I32));
  std::vector<loom_value_id_t> arguments = {value.ssa(), value.ssa()};
  const std::array<TiedValueConsumption, 1> consumptions = {
      TiedValueConsumption{value, type, 0}};
  ValueOwnership ownership(duplication);
  ownership.add_owner(value);

  ownership.separate(consumptions, arguments, owner);

  EXPECT_TRUE(loom_scalar_duplicate_isa(producer(arguments[0])));
  EXPECT_EQ(arguments[1], value.ssa());
}

}  // namespace
}  // namespace loom::cxx_import
