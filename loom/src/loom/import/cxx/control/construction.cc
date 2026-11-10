// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/construction.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace loom::cxx_import {

void Construction::append(
    std::variant<Read, Assign, Invoke, Join, Forget> effect) {
  nodes_.push_back({current_.point, std::move(effect)});
  current_.point = nodes_.size();
}

Construction::State Construction::join(State left, State right) {
  if (!left.reachable) {
    return right;
  }
  if (!right.reachable || left.point == right.point) {
    return left;
  }
  nodes_.push_back({left.point, Join{right.point}});
  return {true, nodes_.size()};
}

void Construction::declare(cxx::Symbol* binding, size_t component_count) {
  declarations_.emplace(binding, Declaration{component_count});
  if (declaration_scope_ != SIZE_MAX) {
    frames_[declaration_scope_].declarations.push_back(binding);
  }
}

void Construction::declare_parameter(cxx::Symbol* binding,
                                     size_t component_count) {
  declare(binding, component_count);
  parameters_.emplace(binding, Parameter{component_count});
}

bool Construction::tracked(cxx::Symbol* binding) const {
  auto found = declarations_.find(binding);
  return found != declarations_.end() && found->second.active;
}

Construction::Parameter* Construction::parameter(cxx::Symbol* binding) {
  auto found = parameters_.find(binding);
  return found == parameters_.end() ? nullptr : &found->second;
}

const Construction::Parameter* Construction::parameter(
    cxx::Symbol* binding) const {
  auto found = parameters_.find(binding);
  return found == parameters_.end() ? nullptr : &found->second;
}

void Construction::configure_parameter(cxx::Symbol* binding, ParameterMode mode,
                                       bool written) {
  auto* value = parameter(binding);
  if (!value) {
    return;
  }
  value->mode = mode;
  if (mode != ParameterMode::Projected) {
    value->inputs.add({0, value->component_count});
    if (written) {
      value->outputs.add({0, value->component_count});
    }
  }
}

ComponentRange Construction::range(cxx::Symbol* binding, size_t offset,
                                   size_t count) const {
  return {offset, count == SIZE_MAX ? declarations_.at(binding).component_count
                                    : offset + count};
}

void Construction::read(cxx::Symbol* binding, size_t offset, size_t count) {
  if (!current_.reachable || !tracked(binding)) {
    return;
  }
  auto selected = range(binding, offset, count);
  if (selected.begin != selected.end) {
    append(Read{{binding, selected}});
  }
}

void Construction::assign(cxx::Symbol* binding, size_t offset, size_t count) {
  if (!current_.reachable || !tracked(binding)) {
    return;
  }
  auto selected = range(binding, offset, count);
  if (selected.begin == selected.end) {
    return;
  }
  append(Assign{{binding, selected}});
  if (auto* formal = parameter(binding)) {
    formal->outputs.add(selected);
  }
}

void Construction::call(size_t source_index) {
  if (!current_.reachable || declarations_.empty()) {
    return;
  }
  append(Invoke{calls_.size()});
  calls_.push_back({source_index, {}});
}

void Construction::enter(bool declaration_scope) {
  frames_.push_back({breaks_, continues_, {}, declaration_scope_});
  if (declaration_scope) {
    declaration_scope_ = frames_.size() - 1;
  }
  breaks_ = {false};
  continues_ = {false};
}

void Construction::leave(cxx::AST* owner) {
  auto frame = std::move(frames_.back());
  frames_.pop_back();
  declaration_scope_ = frame.enclosing_declaration_scope;
  auto forget = [&](State state, cxx::Symbol* binding) {
    if (state.reachable) {
      nodes_.push_back({state.point, Forget{binding}});
      state.point = nodes_.size();
    }
    return state;
  };
  for (auto* binding : frame.declarations) {
    declarations_.at(binding).active = false;
    current_ = forget(current_, binding);
    breaks_ = forget(breaks_, binding);
    continues_ = forget(continues_, binding);
  }
  if (!nodes_.empty()) {
    // Calls can add written fields after this source visit. Keep the source
    // position and select its written objects after call propagation finishes.
    checkpoints_.emplace(owner, join(join(current_, breaks_), continues_));
  }
  breaks_ = join(frame.breaks, breaks_);
  continues_ = join(frame.continues, continues_);
}

void Construction::return_from_function() {
  returns_ = join(returns_, current_);
}
void Construction::finish_function() { return_from_function(); }
void Construction::end_path() { current_ = {false}; }
void Construction::break_from_region() {
  breaks_ = join(breaks_, current_);
  end_path();
}
void Construction::continue_iteration() {
  continues_ = join(continues_, current_);
  end_path();
}
void Construction::reach_latch() {
  current_ = join(current_, continues_);
  continues_ = {false};
}
void Construction::finish_loop(State condition_exit) {
  current_ = join(condition_exit, breaks_);
  breaks_ = {false};
  continues_ = {false};
}
void Construction::finish_switch(State unmatched) {
  current_ = join(join(current_, unmatched), breaks_);
  breaks_ = {false};
}

bool Construction::propagate_outputs() {
  bool changed = false;
  for (const auto& call : calls_) {
    for (const auto& argument : call.arguments) {
      auto* destination = parameter(argument.binding);
      if (!destination) {
        continue;
      }
      if (argument.parameter) {
        changed = destination->outputs.add(argument.parameter->outputs.project(
                      argument.formal, argument.offset)) ||
                  changed;
      } else if (argument.writable) {
        changed = destination->outputs.add(
                      {argument.offset, argument.offset + argument.formal.end -
                                            argument.formal.begin}) ||
                  changed;
      }
    }
  }
  return changed;
}

void Construction::resolve(const Writes& writes) {
  if (nodes_.empty()) {
    return;
  }
  // Sequential nodes transfer their state; only source forks copy it. Counts
  // include both inputs of joins, so no per-node map copy or backward walk is
  // needed. Checkpoint facts are captured when their source position is ready.
  std::vector<size_t> uses(nodes_.size() + 1);
  for (const auto& node : nodes_) {
    ++uses[node.previous];
    if (auto* join = std::get_if<Join>(&node.effect)) {
      ++uses[join->other];
    }
  }
  std::vector<std::pair<size_t, cxx::AST*>> checkpoints;
  checkpoints.reserve(checkpoints_.size());
  for (const auto& [owner, state] : checkpoints_) {
    if (state.reachable && writes.contains(owner)) {
      checkpoints.emplace_back(state.point, owner);
    }
  }
  std::sort(checkpoints.begin(), checkpoints.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  size_t checkpoint_index = 0;
  auto capture = [&](size_t point, const Available& available) {
    while (checkpoint_index < checkpoints.size() &&
           checkpoints[checkpoint_index].first == point) {
      auto* owner = checkpoints[checkpoint_index++].second;
      Available selected;
      for (auto* binding : writes.at(owner)) {
        auto found = available.find(binding);
        if (found != available.end()) {
          selected.emplace(binding, found->second);
        }
      }
      if (!selected.empty()) {
        outgoing_.emplace(owner, std::move(selected));
      }
    }
    if (returns_.reachable && returns_.point == point) {
      for (auto& [binding, formal] : parameters_) {
        auto preserving = formal.outputs;
        if (auto found = available.find(binding); found != available.end()) {
          preserving.subtract(found->second);
        }
        formal.inputs.add(preserving);
      }
    }
  };
  std::vector<std::unique_ptr<Available>> states(nodes_.size() + 1);
  states[0] = std::make_unique<Available>();
  capture(0, *states[0]);
  auto take = [&](size_t point) {
    if (--uses[point] == 0) {
      return std::move(states[point]);
    }
    return std::make_unique<Available>(*states[point]);
  };
  for (size_t index = 0; index < nodes_.size(); ++index) {
    const auto& node = nodes_[index];
    auto state = take(node.previous);
    if (auto* read = std::get_if<Read>(&node.effect)) {
      if (auto* formal = parameter(read->access.binding)) {
        Components required;
        required.add(read->access.range);
        if (auto found = state->find(read->access.binding);
            found != state->end()) {
          required.subtract(found->second);
        }
        formal->inputs.add(required);
      }
    } else if (auto* assign = std::get_if<Assign>(&node.effect)) {
      (*state)[assign->access.binding].add(assign->access.range);
    } else if (auto* invoke = std::get_if<Invoke>(&node.effect)) {
      const auto& call = calls_[invoke->call_index];
      // All reference arguments are read before any returned field is written.
      for (const auto& argument : call.arguments) {
        auto* destination = parameter(argument.binding);
        if (!destination) {
          continue;
        }
        auto found = state->find(argument.binding);
        auto available = found == state->end() ? Components{} : found->second;
        if (argument.parameter) {
          input_dependencies_.push_back({destination, argument.parameter,
                                         argument.formal, argument.offset,
                                         std::move(available)});
        } else {
          Components required;
          required.add({argument.offset, argument.offset + argument.formal.end -
                                             argument.formal.begin});
          required.subtract(available);
          destination->inputs.add(required);
        }
      }
      for (const auto& argument : call.arguments) {
        if (argument.definite && argument.parameter &&
            argument.parameter->mode != ParameterMode::Storage) {
          (*state)[argument.binding].add(argument.parameter->outputs.project(
              argument.formal, argument.offset));
        }
      }
    } else if (auto* join = std::get_if<Join>(&node.effect)) {
      const auto& other = *states[join->other];
      for (auto entry = state->begin(); entry != state->end();) {
        auto found = other.find(entry->first);
        if (found == other.end()) {
          entry = state->erase(entry);
        } else {
          entry->second.intersect(found->second);
          if (entry->second.empty()) {
            entry = state->erase(entry);
          } else {
            ++entry;
          }
        }
      }
      if (--uses[join->other] == 0) {
        states[join->other].reset();
      }
    } else {
      state->erase(std::get<Forget>(node.effect).binding);
    }
    const size_t point = index + 1;
    capture(point, *state);
    if (uses[point]) {
      states[point] = std::move(state);
    }
  }
}

bool Construction::propagate_inputs() {
  bool changed = false;
  for (const auto& dependency : input_dependencies_) {
    auto required =
        dependency.source->inputs.project(dependency.formal, dependency.offset);
    required.subtract(dependency.available);
    changed = dependency.destination->inputs.add(required) || changed;
  }
  return changed;
}

bool Construction::initialized(cxx::AST* owner, cxx::Symbol* binding,
                               size_t offset, size_t count) const {
  const ComponentRange requested{offset, offset + count};
  if (auto* formal = parameter(binding);
      formal && formal->inputs.contains(requested)) {
    return true;
  }
  auto found = outgoing_.find(owner);
  if (found == outgoing_.end()) {
    return false;
  }
  auto value = found->second.find(binding);
  return value != found->second.end() && value->second.contains(requested);
}

}  // namespace loom::cxx_import
