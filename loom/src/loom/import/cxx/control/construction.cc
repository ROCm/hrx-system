// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/construction.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace loom::cxx_import {
namespace {

// Tiny source choices dominate ordinary calls. Bound their direct comparison
// before switching wide choices to a sorted endpoint sweep.
constexpr size_t kLinearChoiceThreshold = 8;

struct ChoiceIdentity {
  // Caller object selected by this source alternative.
  cxx::Symbol* binding;
  // First caller component participating in identity equality.
  size_t component_offset;
  // Exclusive caller component bound, equal to the offset when unused.
  size_t component_end;

  bool operator==(const ChoiceIdentity&) const = default;
};

struct ChoiceInterval {
  // Formal or source components in which alternatives compete.
  ComponentRange overlap;
  // Caller destination that must agree across overlapping alternatives.
  ChoiceIdentity identity;
  // Original order used to make endpoint sorting deterministic.
  size_t ordinal;
  // Classification result retained directly in the owning construction fact.
  bool* definite;
};

struct IdentityExtremum {
  // Source identity owning this extremum, or no identity for an empty slot.
  std::optional<ChoiceIdentity> identity;
  // Furthest endpoint observed for this identity in the active sweep.
  size_t value = 0;
};

template <typename IsBetter>
void retain_identity_extremum(std::array<IdentityExtremum, 2>& extrema,
                              ChoiceIdentity identity, size_t value,
                              IsBetter is_better) {
  for (size_t index = 0; index < extrema.size(); ++index) {
    auto& candidate = extrema[index];
    if (!candidate.identity || *candidate.identity != identity) {
      continue;
    }
    if (is_better(value, candidate.value)) {
      candidate.value = value;
      if (index == 1 && is_better(candidate.value, extrema[0].value)) {
        std::swap(extrema[0], extrema[1]);
      }
    }
    return;
  }

  IdentityExtremum candidate{identity, value};
  if (!extrema[0].identity || is_better(value, extrema[0].value)) {
    extrema[1] = std::move(extrema[0]);
    extrema[0] = std::move(candidate);
  } else if (!extrema[1].identity || is_better(value, extrema[1].value)) {
    extrema[1] = std::move(candidate);
  }
}

std::optional<size_t> other_identity_extremum(
    const std::array<IdentityExtremum, 2>& extrema,
    const ChoiceIdentity& identity) {
  for (const auto& candidate : extrema) {
    if (candidate.identity && *candidate.identity != identity) {
      return candidate.value;
    }
  }
  return std::nullopt;
}

void classify_choices(std::span<ChoiceInterval> choices) {
  if (choices.size() <= kLinearChoiceThreshold) {
    for (size_t i = 0; i < choices.size(); ++i) {
      const auto left_range = choices[i].overlap;
      for (size_t j = 0; j < i; ++j) {
        const auto right_range = choices[j].overlap;
        if (left_range.begin < right_range.end &&
            right_range.begin < left_range.end &&
            choices[i].identity != choices[j].identity) {
          *choices[i].definite = false;
          *choices[j].definite = false;
        }
      }
    }
    return;
  }

  std::sort(choices.begin(), choices.end(),
            [](const ChoiceInterval& left, const ChoiceInterval& right) {
              const auto left_range = left.overlap;
              const auto right_range = right.overlap;
              return left_range.begin != right_range.begin
                         ? left_range.begin < right_range.begin
                         : left.ordinal < right.ordinal;
            });

  std::array<IdentityExtremum, 2> furthest_ends;
  for (auto& choice : choices) {
    const auto item_range = choice.overlap;
    const auto item_identity = choice.identity;
    auto other_end = other_identity_extremum(furthest_ends, item_identity);
    if (other_end && *other_end > item_range.begin) {
      *choice.definite = false;
    }
    retain_identity_extremum(furthest_ends, item_identity, item_range.end,
                             std::greater<size_t>{});
  }

  std::array<IdentityExtremum, 2> nearest_begins;
  for (auto cursor = choices.rbegin(); cursor != choices.rend(); ++cursor) {
    const auto item_range = cursor->overlap;
    const auto item_identity = cursor->identity;
    auto other_begin = other_identity_extremum(nearest_begins, item_identity);
    if (other_begin && *other_begin < item_range.end) {
      *cursor->definite = false;
    }
    retain_identity_extremum(nearest_begins, item_identity, item_range.begin,
                             std::less<size_t>{});
  }
}

template <typename Item, typename MakeChoice>
void classify_definite(std::span<Item> items, MakeChoice make_choice) {
  if (items.empty()) {
    return;
  }
  auto first = make_choice(items.front(), 0);
  bool common_range = true;
  bool common_identity = true;
  for (size_t ordinal = 0; ordinal < items.size(); ++ordinal) {
    items[ordinal].definite = true;
    auto choice = make_choice(items[ordinal], ordinal);
    common_range &= choice.overlap == first.overlap;
    common_identity &= choice.identity == first.identity;
  }
  if (common_range) {
    if (!common_identity) {
      for (auto& item : items) {
        item.definite = false;
      }
    }
    return;
  }

  std::array<ChoiceInterval, kLinearChoiceThreshold> inline_choices;
  std::vector<ChoiceInterval> wide_choices;
  std::span<ChoiceInterval> choices;
  if (items.size() <= inline_choices.size()) {
    choices = std::span(inline_choices).first(items.size());
  } else {
    wide_choices.reserve(items.size());
  }
  for (size_t ordinal = 0; ordinal < items.size(); ++ordinal) {
    auto choice = make_choice(items[ordinal], ordinal);
    if (!choices.empty()) {
      choices[ordinal] = choice;
    } else {
      wide_choices.push_back(choice);
    }
  }
  if (!wide_choices.empty()) {
    choices = wide_choices;
  }
  classify_choices(choices);
}

}  // namespace

void Construction::classify_definite_arguments(std::span<Argument> arguments) {
  classify_definite(arguments, [](Argument& argument, size_t ordinal) {
    return ChoiceInterval{
        argument.formal,
        {argument.binding, argument.offset, argument.offset},
        ordinal,
        &argument.definite,
    };
  });
}

void Construction::classify_definite_destinations(
    std::span<Destination> destinations) {
  classify_definite(destinations, [](Destination& destination, size_t ordinal) {
    return ChoiceInterval{
        {destination.source_offset, destination.source_offset +
                                        destination.range.end -
                                        destination.range.begin},
        {destination.binding, destination.range.begin, destination.range.end},
        ordinal,
        &destination.definite,
    };
  });
}

void Construction::append(
    std::variant<Read, Assign, IndirectAssign, Invoke, Join, Forget> effect) {
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
                                     size_t component_count, bool returned) {
  declare(binding, component_count);
  parameters_.emplace(
      binding, Parameter{component_count, ParameterMode::Projected, returned});
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
    if (written && value->returned) {
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
  if (auto* formal = parameter(binding); formal && formal->returned) {
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

void Construction::assign_indirect(size_t source_index) {
  if (!current_.reachable || declarations_.empty()) {
    return;
  }
  append(IndirectAssign{assignments_.size()});
  assignments_.push_back({source_index, {}});
}

void Construction::defer_call(size_t source_index) {
  if (!current_.reachable || declarations_.empty()) {
    return;
  }
  auto& cleanups = declaration_scope_ == SIZE_MAX
                       ? function_cleanups_
                       : frames_[declaration_scope_].cleanups;
  cleanups.push_back(calls_.size());
  calls_.push_back({source_index, {}});
}

Construction::State Construction::cleanup(State state,
                                          std::span<const size_t> calls) {
  if (state.reachable) {
    for (size_t index = calls.size(); index > 0; --index) {
      nodes_.push_back({state.point, Invoke{calls[index - 1]}});
      state.point = nodes_.size();
    }
  }
  return state;
}

void Construction::enter(bool declaration_scope) {
  frames_.push_back(
      {breaks_, continues_, returns_, {}, {}, declaration_scope_});
  if (declaration_scope) {
    declaration_scope_ = frames_.size() - 1;
  }
  breaks_ = {false};
  continues_ = {false};
  returns_ = {false};
}

void Construction::leave(cxx::AST* owner) {
  auto frame = std::move(frames_.back());
  frames_.pop_back();
  declaration_scope_ = frame.enclosing_declaration_scope;
  current_ = cleanup(current_, frame.cleanups);
  breaks_ = cleanup(breaks_, frame.cleanups);
  continues_ = cleanup(continues_, frame.cleanups);
  returns_ = cleanup(returns_, frame.cleanups);
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
    returns_ = forget(returns_, binding);
  }
  if (!nodes_.empty()) {
    // Calls can add written fields after this source visit. Keep the source
    // position and select its written objects after call propagation finishes.
    checkpoints_.emplace(owner, join(join(current_, breaks_), continues_));
  }
  breaks_ = join(frame.breaks, breaks_);
  continues_ = join(frame.continues, continues_);
  returns_ = join(frame.returns, returns_);
}

void Construction::return_from_function() {
  returns_ = join(returns_, current_);
}
void Construction::finish_function() {
  return_from_function();
  returns_ = cleanup(returns_, function_cleanups_);
}
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
  for (const auto& assignment : assignments_) {
    for (const auto& destination : assignment.destinations) {
      if (auto* formal = parameter(destination.binding);
          formal && formal->returned) {
        changed = formal->outputs.add(destination.range) || changed;
      }
    }
  }
  for (const auto& call : calls_) {
    for (const auto& argument : call.arguments) {
      auto* destination = parameter(argument.binding);
      if (!destination || !destination->returned) {
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
    } else if (auto* assign = std::get_if<IndirectAssign>(&node.effect)) {
      for (const auto& destination :
           assignments_[assign->assignment_index].destinations) {
        if (destination.definite) {
          (*state)[destination.binding].add(destination.range);
        }
      }
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
