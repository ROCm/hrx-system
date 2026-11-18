// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation/region_state.h"

#include <cxx/symbols.h>

#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/component_names.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

void RegionState::append(cxx::Symbol* binding,
                         const ValueProjection& projection) {
  groups_.push_back({binding, projection.type, projection.partition,
                     projection.component_offset});
  component_count_ += projection.partition->component_count;
}

void RegionState::append_types(std::vector<const cxx::Type*>& types) const {
  types.reserve(types.size() + group_count());
  for (const auto& group : groups_) {
    types.push_back(group.type);
  }
}

std::vector<const cxx::Type*> RegionState::source_types(
    std::initializer_list<const cxx::Type*> prefix) const {
  std::vector<const cxx::Type*> types;
  types.reserve(prefix.size() + group_count());
  types.insert(types.end(), prefix.begin(), prefix.end());
  append_types(types);
  return types;
}

std::vector<const cxx::Type*> RegionState::dependent_types(
    Types& types, cxx::AST* owner) const {
  for (const auto& group : groups_) {
    if (types.requires_binding(group.type, owner)) {
      return source_types();
    }
  }
  return {};
}

bool RegionState::append_values(const ValueBindings& bindings,
                                ValueArena& arena,
                                std::vector<loom_value_id_t>& values) const {
  values.reserve(values.size() + component_count_);
  for (const auto& group : groups_) {
    auto value =
        bindings.project(group.binding, *group.partition, group.offset, arena);
    if (!value) {
      return false;
    }
    value->append_to(values);
  }
  return true;
}

void RegionState::bind_values(ValueBindings& bindings, ValueArena& arena,
                              loom_module_t* module,
                              const loom_value_id_t* values) const {
  std::span<const loom_value_id_t> remaining(values, component_count_);
  for (const auto& group : groups_) {
    const auto& partition = *group.partition;
    auto value =
        arena.capture(partition, remaining.first(partition.component_count));
    value = name_value(module, value, cxx::to_string(group.binding->name()));
    if (&partition == &bindings.partition(group.binding)) {
      bindings.set(group.binding, value);
    } else {
      bindings.replace(group.binding, group.offset, value, arena);
    }
    remaining = remaining.subspan(partition.component_count);
  }
}

}  // namespace loom::cxx_import
