// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bindings.h"

#include <array>

namespace loom::cxx_import {

const Partition& ValueBindings::partition(cxx::Symbol* binding) const {
  auto found = values_.find(binding);
  return found == values_.end() ? *uninitialized_.at(binding).partition
                                : found->second.partition();
}

void ValueBindings::declare(cxx::Symbol* binding, const Partition& partition) {
  values_.erase(binding);
  if (!partition.component_count) {
    ValueArena arena;
    set(binding, arena.capture(partition, {}));
    return;
  }
  uninitialized_.insert_or_assign(
      binding, Pending{&partition,
                       std::vector<std::optional<loom_value_id_t>>(
                           partition.component_count),
                       partition.component_count});
}

bool ValueBindings::initialized(cxx::Symbol* binding, size_t offset,
                                size_t count) const {
  if (values_.contains(binding)) {
    return true;
  }
  const auto& pending = uninitialized_.at(binding);
  for (size_t index = offset; index < offset + count; ++index) {
    if (!pending.components[index]) {
      return false;
    }
  }
  return true;
}

std::optional<Value> ValueBindings::project(cxx::Symbol* binding,
                                            const Partition& part,
                                            size_t offset,
                                            ValueArena& arena) const {
  if (auto found = values_.find(binding); found != values_.end()) {
    return found->second.project(part, offset);
  }
  const auto& pending = uninitialized_.at(binding);
  std::array<loom_value_id_t, 2> small;
  std::vector<loom_value_id_t> large;
  if (part.component_count > small.size()) {
    large.resize(part.component_count);
  }
  auto* components = large.empty() ? small.data() : large.data();
  for (size_t index = 0; index < part.component_count; ++index) {
    const auto& component = pending.components[offset + index];
    if (!component) {
      return std::nullopt;
    }
    components[index] = *component;
  }
  return arena.capture(part, {components, part.component_count});
}

void ValueBindings::replace(cxx::Symbol* binding, size_t offset, Value value,
                            ValueArena& arena) {
  if (auto found = values_.find(binding); found != values_.end()) {
    found->second = arena.replace(found->second, offset, value);
    return;
  }
  auto& pending = uninitialized_.at(binding);
  for (auto component : value.components()) {
    if (!pending.components[offset]) {
      --pending.remaining;
    }
    pending.components[offset++] = component;
  }
  if (!pending.remaining) {
    set(binding, *project(binding, *pending.partition, 0, arena));
  }
}

}  // namespace loom::cxx_import
