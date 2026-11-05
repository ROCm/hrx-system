// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/ownership.h"

#include <algorithm>

namespace loom::cxx_import {

void ValueOwnership::add_owner(Value value) {
  add_owner_components(value.components());
}

void ValueOwnership::add_owner_components(
    std::span<const loom_value_id_t> components) {
  for (auto component : components) {
    ++owner_counts_[component];
  }
}

void ValueOwnership::separate(
    std::span<const TiedValueConsumption> consumptions,
    std::vector<loom_value_id_t>& arguments, cxx::AST* owner) {
  if (consumptions.empty()) {
    return;
  }

  std::vector<bool> consumed_arguments(arguments.size());
  std::unordered_map<loom_value_id_t, size_t> consumed_counts;
  for (const auto& consumption : consumptions) {
    auto components = consumption.value.components();
    IREE_ASSERT_LE(consumption.argument_offset + components.size(),
                   arguments.size());
    for (size_t index = 0; index < components.size(); ++index) {
      const size_t argument_index = consumption.argument_offset + index;
      IREE_ASSERT(!consumed_arguments[argument_index]);
      IREE_ASSERT_EQ(arguments[argument_index], components[index]);
      consumed_arguments[argument_index] = true;
      ++consumed_counts[components[index]];
    }
  }

  std::unordered_map<loom_value_id_t, size_t> ordinary_use_counts;
  for (size_t index = 0; index < arguments.size(); ++index) {
    if (!consumed_arguments[index]) {
      ++ordinary_use_counts[arguments[index]];
    }
  }

  auto remaining_consumptions = consumed_counts;
  for (const auto& consumption : consumptions) {
    auto components = consumption.value.components();
    std::unordered_map<loom_value_id_t, size_t> local_counts;
    for (auto component : components) {
      ++local_counts[component];
    }

    bool requires_duplicate = false;
    for (const auto& [component, local_count] : local_counts) {
      auto owners = owner_counts_.find(component);
      IREE_ASSERT(owners != owner_counts_.end());
      IREE_ASSERT_GE(owners->second, consumed_counts.at(component));
      requires_duplicate |= owners->second > consumed_counts.at(component);
      requires_duplicate |= ordinary_use_counts.contains(component);
      requires_duplicate |= remaining_consumptions.at(component) != local_count;
      requires_duplicate |= local_count != 1;
    }

    if (requires_duplicate) {
      auto duplicate =
          duplication_.build(consumption.value, consumption.source_type, owner);
      std::ranges::copy(duplicate.components(),
                        arguments.begin() + consumption.argument_offset);
    }
    for (const auto& [component, local_count] : local_counts) {
      remaining_consumptions.at(component) -= local_count;
    }
  }
}

}  // namespace loom::cxx_import
