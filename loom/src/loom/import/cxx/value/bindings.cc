// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bindings.h"

namespace loom::cxx_import {

ValueBindings::ValueBindings(const ValueBindings& other)
    : values_(other.values_) {
  if (other.owner_index_) {
    owner_index_ = std::make_unique<OwnerIndex>(*other.owner_index_);
  }
}

ValueBindings& ValueBindings::operator=(const ValueBindings& other) {
  if (this == &other) {
    return *this;
  }
  values_ = other.values_;
  owner_index_ = other.owner_index_
                     ? std::make_unique<OwnerIndex>(*other.owner_index_)
                     : nullptr;
  return *this;
}

void ValueBindings::set(cxx::Symbol* binding, Value value) {
  auto found = values_.find(binding);
  if (owner_index_ && found != values_.end()) {
    remove(*owner_index_, binding, found->second);
  }
  values_.insert_or_assign(binding, value);
  if (owner_index_) {
    add(*owner_index_, binding, value);
  }
}

void ValueBindings::erase(cxx::Symbol* binding) {
  auto found = values_.find(binding);
  if (found == values_.end()) {
    return;
  }
  if (owner_index_) {
    remove(*owner_index_, binding, found->second);
  }
  values_.erase(found);
}

void ValueBindings::clear() {
  values_.clear();
  if (owner_index_) {
    owner_index_->clear();
  }
}

const ValueBindings::OwnerCounts* ValueBindings::owners(
    loom_value_id_t component) const {
  ensure_owner_index();
  auto found = owner_index_->find(component);
  return found == owner_index_->end() ? nullptr : &found->second;
}

void ValueBindings::ensure_owner_index() const {
  if (owner_index_) {
    return;
  }
  owner_index_ = std::make_unique<OwnerIndex>();
  for (const auto& [binding, value] : values_) {
    add(*owner_index_, binding, value);
  }
}

void ValueBindings::add(OwnerIndex& index, cxx::Symbol* binding, Value value) {
  for (auto component : value.components()) {
    ++index[component][binding];
  }
}

void ValueBindings::remove(OwnerIndex& index, cxx::Symbol* binding,
                           Value value) {
  for (auto component : value.components()) {
    auto identity = index.find(component);
    IREE_ASSERT(identity != index.end());
    auto owner = identity->second.find(binding);
    IREE_ASSERT(owner != identity->second.end() && owner->second != 0);
    if (--owner->second == 0) {
      identity->second.erase(owner);
    }
    if (identity->second.empty()) {
      index.erase(identity);
    }
  }
}

}  // namespace loom::cxx_import
