// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/receiver/scopes.h"

#include <cxx/symbols.h>

#include <utility>

#include "iree/base/api.h"

namespace loom::cxx_import {

void ReceiverScopes::enter(const RecordPartition& record,
                           std::optional<StorageProjection> storage,
                           const cxx::Type* object_type) {
  Frame frame{{&record, storage, object_type}, {}};
  frame.bindings.reserve(record.members.size());
  const size_t index = frames_.size();
  for (const auto& member : record.members) {
    auto owner = owners_.find(member.source);
    frame.bindings.push_back({member.source,
                              owner == owners_.end()
                                  ? std::nullopt
                                  : std::optional<Owner>{owner->second},
                              values_.take(member.source)});
    owners_[member.source] = {index, false};
  }
  frames_.push_back(std::move(frame));
}

void ReceiverScopes::leave() {
  const size_t index = frames_.size() - 1;
  auto frame = std::move(frames_.back());
  frames_.pop_back();
  for (auto& binding : frame.bindings) {
    auto found = owners_.find(binding.subobject);
    IREE_ASSERT(found != owners_.end() && found->second.index == index);
    if (binding.previous_owner) {
      found->second = *binding.previous_owner;
    } else {
      owners_.erase(found);
    }
    values_.restore(binding.subobject, std::move(binding.previous_binding));
  }
}

bool ReceiverScopes::suspend() {
  if (frames_.size() <= 1) {
    return false;
  }
  const size_t index = frames_.size() - 1;
  SuspendedFrame suspended{index, std::move(frames_.back()), {}};
  frames_.pop_back();
  suspended.bindings.reserve(suspended.frame.bindings.size());
  for (auto& binding : suspended.frame.bindings) {
    auto owner = owners_.find(binding.subobject);
    IREE_ASSERT(owner != owners_.end() && owner->second.index == index);
    suspended.bindings.push_back(
        {owner->second, values_.take(binding.subobject)});
    if (binding.previous_owner) {
      owner->second = *binding.previous_owner;
    } else {
      owners_.erase(owner);
    }
    values_.restore(binding.subobject, std::move(binding.previous_binding));
  }
  suspended_.push_back(std::move(suspended));
  return true;
}

void ReceiverScopes::resume() {
  auto suspended = std::move(suspended_.back());
  suspended_.pop_back();
  IREE_ASSERT_EQ(frames_.size(), suspended.index);
  for (size_t index = 0; index < suspended.frame.bindings.size(); ++index) {
    auto& binding = suspended.frame.bindings[index];
    auto& hidden = suspended.bindings[index];
    auto lexical_owner = owners_.find(binding.subobject);
    binding.previous_owner = lexical_owner == owners_.end()
                                 ? std::nullopt
                                 : std::optional<Owner>{lexical_owner->second};
    binding.previous_binding = values_.take(binding.subobject);
    owners_.insert_or_assign(binding.subobject, hidden.owner);
    values_.restore(binding.subobject, std::move(hidden.binding));
  }
  frames_.push_back(std::move(suspended.frame));
}

void ReceiverScopes::clear() {
  frames_.clear();
  owners_.clear();
  suspended_.clear();
}

const ReceiverState* ReceiverScopes::find(cxx::Symbol* subobject) const {
  auto found = owners_.find(subobject);
  return found == owners_.end() ? nullptr
                                : &frames_[found->second.index].receiver;
}

void ReceiverScopes::publish(cxx::Symbol* subobject, Value value) {
  auto& owner = owners_.at(subobject);
  IREE_ASSERT(!frames_[owner.index].receiver.storage);
  values_.set(subobject, value);
  owner.complete = true;
}

void ReceiverScopes::mark_complete(cxx::Symbol* subobject) {
  auto& owner = owners_.at(subobject);
  owner.complete = true;
}

void ReceiverScopes::replace_current_storage_root(loom_value_id_t root) {
  frames_.back().receiver.storage->pointer.root = root;
}

}  // namespace loom::cxx_import
