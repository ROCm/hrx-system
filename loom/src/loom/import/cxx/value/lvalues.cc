// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/lvalues.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace loom::cxx_import {
namespace {

// Small selected-origin sets dominate source calls. Keep their unique binding
// summary inline while bounding the prefix before switching wide sets to a
// retained hash index.
constexpr size_t kLinearBindingThreshold = 8;

class BindingAccumulator {
 public:
  explicit BindingAccumulator(std::vector<cxx::Symbol*>& bindings)
      : bindings_(bindings) {}

  void append(cxx::Symbol* binding) {
    if (index_.empty()) {
      if (std::ranges::find(bindings_, binding) != bindings_.end()) {
        return;
      }
      if (bindings_.size() < kLinearBindingThreshold) {
        bindings_.push_back(binding);
        return;
      }
      index_.reserve(bindings_.size() + 1);
      index_.insert(bindings_.begin(), bindings_.end());
    }
    if (index_.insert(binding).second) {
      bindings_.push_back(binding);
    }
  }

  void append(const Lvalue& value) {
    if (const auto* selected = value.selection()) {
      for (auto* binding : selected->bindings) {
        append(binding);
      }
      return;
    }
    const auto& owner = value.object();
    if (const auto* destination = std::get_if<Destination>(&owner.location)) {
      append(destination->binding);
    } else if (const auto* receiver =
                   std::get_if<CurrentReceiver>(&owner.location)) {
      for (const auto& member : receiver->record->members) {
        append(member.source);
      }
    }
  }

 private:
  // Unique source bindings retained in stable first-origin order.
  std::vector<cxx::Symbol*>& bindings_;
  // Wide-set membership index, empty for the bounded common case.
  std::unordered_set<cxx::Symbol*> index_;
};

}  // namespace

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
  BindingAccumulator accumulator(bindings);
  for (const auto& alternative : this->alternatives) {
    accumulator.append(alternative);
  }
}

}  // namespace loom::cxx_import
