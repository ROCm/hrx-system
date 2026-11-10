// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/lvalues.h"

#include <algorithm>
#include <utility>

namespace loom::cxx_import {

const SelectedLvalue* Lvalue::selection() const {
  const auto* selected =
      std::get_if<std::shared_ptr<const SelectedLvalue>>(&projection);
  return selected ? selected->get() : nullptr;
}

const ObjectLvalue& Lvalue::object() const {
  if (const auto* object = std::get_if<ObjectLvalue>(&projection)) {
    return *object;
  }
  if (const auto* lane = std::get_if<VectorLaneLvalue>(&projection)) {
    return lane->destination;
  }
  return std::get<VectorComponentLvalue>(projection).destination;
}

void Lvalue::append_bindings(std::vector<cxx::Symbol*>& bindings) const {
  auto append = [&](cxx::Symbol* binding) {
    if (std::ranges::find(bindings, binding) == bindings.end()) {
      bindings.push_back(binding);
    }
  };
  if (const auto* selected = selection()) {
    for (auto* binding : selected->bindings) {
      append(binding);
    }
    return;
  }
  const auto& owner = object();
  if (const auto* destination = std::get_if<Destination>(&owner.location)) {
    append(destination->binding);
  } else if (const auto* receiver =
                 std::get_if<CurrentReceiver>(&owner.location)) {
    for (const auto& member : receiver->record->members) {
      append(member.source);
    }
  }
}

Lvalue Lvalue::project(const SubobjectProjection& selected,
                       const cxx::Type* type, Storage& storage,
                       cxx::AST* owner) const {
  if (const auto* choices = selection()) {
    std::vector<Lvalue> alternatives;
    alternatives.reserve(choices->alternatives.size());
    for (const auto& alternative : choices->alternatives) {
      alternatives.push_back(
          alternative.project(selected, type, storage, owner));
    }
    return {std::make_shared<const SelectedLvalue>(choices->selector, type,
                                                   std::move(alternatives))};
  }
  const auto& object = std::get<ObjectLvalue>(projection);
  if (std::holds_alternative<CurrentReceiver>(object.location)) {
    return {ObjectLvalue{type, Destination{selected.root->source,
                                           selected.component_offset -
                                               selected.root->component_offset,
                                           selected.partition}}};
  }
  if (const auto* destination = std::get_if<Destination>(&object.location)) {
    return {ObjectLvalue{type, Destination{destination->binding,
                                           destination->component_offset +
                                               selected.component_offset,
                                           selected.partition}}};
  }
  const auto& address = std::get<StorageProjection>(object.location);
  return {ObjectLvalue{
      type, storage.subobject(address, selected.byte_offset, owner)}};
}

SelectedLvalue::SelectedLvalue(loom_value_id_t selector, const cxx::Type* type,
                               std::vector<Lvalue> alternatives,
                               std::optional<loom_value_id_t> lane)
    : selector(selector),
      type(type),
      alternatives(std::move(alternatives)),
      lane(lane) {
  for (const auto& alternative : this->alternatives) {
    alternative.append_bindings(bindings);
  }
}

}  // namespace loom::cxx_import
