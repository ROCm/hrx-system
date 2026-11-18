// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/member_storage.h"

#include <algorithm>

namespace loom::cxx_import {
namespace {

size_t member_offset(const MemberStorage::Member& member) {
  return member.group.component_offset;
}

}  // namespace

std::span<MemberStorage::Member> MemberStorage::lookup(cxx::Symbol* binding) {
  if (!bindings_.empty()) {
    auto found = bindings_.find(binding);
    if (found != bindings_.end()) {
      return found->second;
    }
  }
  return {};
}

void MemberStorage::allocate(cxx::Symbol* binding,
                             std::span<const ValueProjection> groups,
                             cxx::AST* source) {
  if (groups.empty()) {
    return;
  }
  auto& members = bindings_[binding];
  members.clear();
  members.reserve(groups.size());
  for (const auto& group : groups) {
    auto local = storage_.allocate(
        group.type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, source);
    members.push_back(
        {group, storage_.project(local.pointer, group.type, source), false});
  }
}

void MemberStorage::bind(cxx::Symbol* binding, ValueProjection group,
                         StorageProjection address) {
  bindings_[binding].push_back({group, address, true});
}

std::optional<StorageProjection> MemberStorage::address(
    std::span<const Member> members, size_t offset, const cxx::Type* type,
    cxx::AST* source) {
  auto next = std::ranges::upper_bound(members, offset, {}, member_offset);
  if (next == members.begin()) {
    return {};
  }
  const auto& home = *std::prev(next);
  auto delta = types_.member_displacement(home.group, offset, type);
  if (!delta) {
    return {};
  }
  return *delta ? storage_.subobject(home.address, *delta, source)
                : home.address;
}

std::optional<Value> MemberStorage::read(std::span<const Member> members,
                                         ValueBindings& values,
                                         Destination dest,
                                         const cxx::Type* type,
                                         cxx::AST* source) {
  if (auto physical = address(members, dest.component_offset, type, source)) {
    return objects_.load(*physical, type, source);
  }
  const auto& part =
      dest.member ? *dest.member : values.partition(dest.binding);
  std::vector<loom_value_id_t> components;
  components.reserve(part.component_count);
  for (const auto& group :
       types_.projections(types_.unqualified_value(type), source)) {
    const size_t offset = dest.component_offset + group.component_offset;
    std::optional<Value> value;
    if (auto physical = address(members, offset, group.type, source)) {
      value = objects_.load(*physical, group.type, source);
    } else {
      value = values.project(dest.binding, *group.partition, offset, arena_);
    }
    if (!value) {
      return {};
    }
    value->append_to(components);
  }
  return arena_.capture(part, components);
}

std::optional<StorageProjection> MemberStorage::begin_construction(
    Destination dest, const cxx::Type* type) {
  auto members = lookup(dest.binding);
  auto member = std::ranges::lower_bound(members, dest.component_offset, {},
                                         member_offset);
  if (member == members.end() ||
      member_offset(*member) != dest.component_offset ||
      types_.unqualified_value(member->group.type) !=
          types_.unqualified_value(type)) {
    return {};
  }
  member->initialized = true;
  return member->address;
}

void MemberStorage::write(std::span<Member> members, ValueBindings& values,
                          Destination dest, Value value, const cxx::Type* type,
                          cxx::AST* source, bool initializing) {
  const size_t first = dest.component_offset;
  const size_t end = first + value.partition().component_count;
  if (auto physical = address(members, first, type, source)) {
    if (initializing) {
      auto member = std::ranges::lower_bound(members, first, {}, member_offset);
      if (member != members.end() && member_offset(*member) == first &&
          member->group.partition->component_count ==
              value.partition().component_count) {
        if (member->initialized) {
          return;
        }
        member->initialized = true;
      }
    }
    objects_.store(*physical, value, type, source);
    return;
  }
  auto start = std::ranges::lower_bound(members, first, {}, member_offset);
  std::span selected(start, members.end());
  size_t count = 0;
  while (count < selected.size() &&
         selected[count].group.component_offset < end) {
    auto& member = selected[count++];
    if (initializing && member.initialized) {
      continue;
    }
    auto input = value.project(*member.group.partition,
                               member.group.component_offset - first);
    objects_.store(member.address, input, member.group.type, source);
    member.initialized = true;
  }
  if (!count) {
    values.replace(dest.binding, first, value, arena_);
    return;
  }
  selected = selected.first(count);
  for (const auto& group :
       types_.projections(types_.unqualified_value(type), source)) {
    const size_t offset = first + group.component_offset;
    while (!selected.empty() &&
           selected.front().group.component_offset +
                   selected.front().group.partition->component_count <=
               offset) {
      selected = selected.subspan(1);
    }
    if (selected.empty() || offset < selected.front().group.component_offset) {
      values.replace(dest.binding, offset,
                     value.project(*group.partition, group.component_offset),
                     arena_);
    }
  }
}

}  // namespace loom::cxx_import
