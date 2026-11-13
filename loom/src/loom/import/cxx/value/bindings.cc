// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bindings.h"

#include <array>

namespace loom::cxx_import {

ValueBindings::Binding ValueBindings::take(cxx::Symbol* binding) {
  auto value = values_.extract(binding);
  if (!value.empty()) {
    return Binding(std::move(value));
  }
  return Binding(uninitialized_.extract(binding));
}

void ValueBindings::restore(cxx::Symbol* binding, Binding state) {
  erase(binding);
  if (auto* value = std::get_if<Map::node_type>(&state.state_)) {
    values_.insert(std::move(*value));
  } else {
    uninitialized_.insert(
        std::get<PendingMap::node_type>(std::move(state.state_)));
  }
}

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
  return project(uninitialized_.at(binding), part, offset, arena);
}

std::optional<Value> ValueBindings::project(const Pending& pending,
                                            const Partition& part,
                                            size_t offset, ValueArena& arena) {
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

std::optional<Value> ValueBindings::Updates::project(cxx::Symbol* binding,
                                                     const Partition& partition,
                                                     size_t offset) {
  auto found = pending_.find(binding);
  return found == pending_.end()
             ? bindings_.project(binding, partition, offset, arena_)
             : ValueBindings::project(found->second, partition, offset, arena_);
}

void ValueBindings::Updates::replace(cxx::Symbol* binding, size_t offset,
                                     Value value) {
  auto found = pending_.find(binding);
  if (found == pending_.end()) {
    if (auto source = bindings_.values_.find(binding);
        source != bindings_.values_.end()) {
      auto components = source->second.components();
      found =
          pending_
              .emplace(binding, Pending{&source->second.partition(),
                                        {components.begin(), components.end()},
                                        0})
              .first;
    } else {
      found =
          pending_.emplace(binding, bindings_.uninitialized_.at(binding)).first;
    }
  }
  auto& pending = found->second;
  for (auto component : value.components()) {
    if (!pending.components[offset]) {
      --pending.remaining;
    }
    pending.components[offset++] = component;
  }
}

void ValueBindings::Updates::apply() {
  for (auto& [binding, pending] : pending_) {
    if (pending.remaining) {
      bindings_.values_.erase(binding);
      bindings_.uninitialized_.insert_or_assign(binding, std::move(pending));
    } else {
      bindings_.set(binding, *ValueBindings::project(
                                 pending, *pending.partition, 0, arena_));
    }
  }
  pending_.clear();
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
