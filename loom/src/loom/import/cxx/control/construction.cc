// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/construction.h"

#include <algorithm>
#include <utility>

namespace loom::cxx_import {

Construction::State Construction::join(State left, const State& right) {
  if (!left.reachable) {
    return right;
  }
  if (!right.reachable) {
    return left;
  }
  for (auto entry = left.available.begin(); entry != left.available.end();) {
    auto found = right.available.find(entry->first);
    if (found == right.available.end()) {
      entry = left.available.erase(entry);
      continue;
    }
    auto& components = entry->second;
    bool any = false;
    for (size_t index = 0; index < components.size(); ++index) {
      components[index] &= found->second[index];
      any |= components[index] != 0;
    }
    if (any) {
      ++entry;
    } else {
      entry = left.available.erase(entry);
    }
  }
  return left;
}

void Construction::declare(cxx::Symbol* binding, size_t component_count) {
  declarations_.emplace(binding, component_count);
  if (declaration_scope_ != SIZE_MAX) {
    frames_[declaration_scope_].declarations.push_back(binding);
  }
}

void Construction::assign(cxx::Symbol* binding, size_t offset, size_t count) {
  if (!current_.reachable || declarations_.empty()) {
    return;
  }
  auto found = declarations_.find(binding);
  if (found == declarations_.end()) {
    return;
  }
  auto& components = current_.available[binding];
  components.resize(found->second);
  if (count == SIZE_MAX) {
    count = components.size();
  }
  std::fill_n(components.begin() + offset, count, uint8_t{1});
}

void Construction::enter(bool declaration_scope) {
  frames_.push_back(
      {std::move(breaks_), std::move(continues_), {}, declaration_scope_});
  if (declaration_scope) {
    declaration_scope_ = frames_.size() - 1;
  }
  breaks_ = {false, {}};
  continues_ = {false, {}};
}

void Construction::leave(cxx::AST* owner,
                         std::span<cxx::Symbol* const> written) {
  auto frame = std::move(frames_.back());
  frames_.pop_back();
  declaration_scope_ = frame.enclosing_declaration_scope;
  for (auto* binding : frame.declarations) {
    declarations_.erase(binding);
    current_.available.erase(binding);
    breaks_.available.erase(binding);
    continues_.available.erase(binding);
  }
  if (!declarations_.empty() && !written.empty()) {
    // Region facts cover only written bindings. Retaining every earlier
    // declaration at every later region would grow quadratically.
    auto select = [&](const State& state) {
      State selected{state.reachable, {}};
      for (auto* binding : written) {
        if (auto found = state.available.find(binding);
            found != state.available.end()) {
          selected.available.emplace(binding, found->second);
        }
      }
      return selected;
    };
    auto result =
        join(join(select(current_), select(breaks_)), select(continues_));
    if (!result.available.empty()) {
      outgoing_.emplace(owner, std::move(result));
    }
  }
  breaks_ = join(std::move(frame.breaks), breaks_);
  continues_ = join(std::move(frame.continues), continues_);
}

void Construction::end_path() { current_ = {false, {}}; }

void Construction::break_from_region() {
  breaks_ = join(std::move(breaks_), current_);
  current_ = {false, {}};
}

void Construction::continue_iteration() {
  continues_ = join(std::move(continues_), current_);
  current_ = {false, {}};
}

void Construction::reach_latch() {
  current_ = join(std::move(current_), continues_);
  continues_ = {false, {}};
}

void Construction::finish_loop(State condition_exit) {
  current_ = join(std::move(condition_exit), breaks_);
  breaks_ = {false, {}};
  continues_ = {false, {}};
}

void Construction::finish_switch(State unmatched) {
  current_ = join(join(std::move(current_), unmatched), breaks_);
  breaks_ = {false, {}};
}

bool Construction::initialized(cxx::AST* owner, cxx::Symbol* binding,
                               size_t offset, size_t count) const {
  auto found = outgoing_.find(owner);
  if (found == outgoing_.end()) {
    return false;
  }
  auto value = found->second.available.find(binding);
  if (value == found->second.available.end()) {
    return false;
  }
  return std::all_of(value->second.begin() + offset,
                     value->second.begin() + offset + count,
                     [](uint8_t available) { return available != 0; });
}

}  // namespace loom::cxx_import
