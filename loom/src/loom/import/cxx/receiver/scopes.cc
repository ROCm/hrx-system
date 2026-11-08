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
    auto value = values_.find(member.source);
    frame.bindings.push_back(
        {member.source,
         owner == owners_.end() ? std::nullopt
                                : std::optional<Owner>{owner->second},
         value == values_.end() ? std::nullopt
                                : std::optional<Value>{value->second}});
    owners_[member.source] = {index, false};
    values_.erase(member.source);
  }
  frames_.push_back(std::move(frame));
}

void ReceiverScopes::leave() {
  const size_t index = frames_.size() - 1;
  auto frame = std::move(frames_.back());
  frames_.pop_back();
  for (const auto& binding : frame.bindings) {
    auto found = owners_.find(binding.subobject);
    IREE_ASSERT(found != owners_.end() && found->second.index == index);
    if (binding.previous_owner) {
      found->second = *binding.previous_owner;
    } else {
      owners_.erase(found);
    }
    if (binding.previous_value) {
      values_.set(binding.subobject, *binding.previous_value);
    } else {
      values_.erase(binding.subobject);
    }
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
  for (const auto& binding : suspended.frame.bindings) {
    auto owner = owners_.find(binding.subobject);
    IREE_ASSERT(owner != owners_.end() && owner->second.index == index);
    auto value = values_.find(binding.subobject);
    suspended.bindings.push_back(
        {owner->second, value == values_.end()
                            ? std::nullopt
                            : std::optional<Value>{value->second}});
    if (binding.previous_owner) {
      owner->second = *binding.previous_owner;
    } else {
      owners_.erase(owner);
    }
    if (binding.previous_value) {
      values_.set(binding.subobject, *binding.previous_value);
    } else {
      values_.erase(binding.subobject);
    }
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
    const auto& hidden = suspended.bindings[index];
    auto lexical_owner = owners_.find(binding.subobject);
    binding.previous_owner = lexical_owner == owners_.end()
                                 ? std::nullopt
                                 : std::optional<Owner>{lexical_owner->second};
    auto lexical_value = values_.find(binding.subobject);
    binding.previous_value = lexical_value == values_.end()
                                 ? std::nullopt
                                 : std::optional<Value>{lexical_value->second};
    owners_.insert_or_assign(binding.subobject, hidden.owner);
    if (hidden.value) {
      values_.set(binding.subobject, *hidden.value);
    } else {
      values_.erase(binding.subobject);
    }
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
  IREE_ASSERT(frames_[owner.index].receiver.storage);
  owner.complete = true;
}

}  // namespace loom::cxx_import
