// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_CONSTRUCTION_H_
#define LOOM_IMPORT_CXX_CONTROL_CONSTRUCTION_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace loom::cxx_import {

// Definite source-value availability, propagated by ControlFlow's existing
// source walk. Only declarations without a complete initial value participate;
// ordinary initialized values and physical C++ storage need no entries here.
// Availability is source analysis state and never becomes an IR value.
class Construction {
 public:
  struct State {
    // Whether a source path reaches this point. An absent path does not
    // constrain the intersection at a control-flow join.
    bool reachable = true;
    // Initialized components of tracked source bindings. Absence means no
    // components are available; dependent subobjects are assigned as a group.
    std::unordered_map<cxx::Symbol*, std::vector<uint8_t>> available;
  };

  const State& state() const { return current_; }
  void restore(State state) { current_ = std::move(state); }
  // Intersects availability over reachable alternatives, preserving a sole
  // continuing path when the other leaves the function.
  static State join(State left, const State& right);

  void declare(cxx::Symbol* binding, size_t component_count);
  bool tracked(cxx::Symbol* binding) const {
    return declarations_.contains(binding);
  }
  void assign(cxx::Symbol* binding, size_t offset, size_t count);
  // Isolates escaping break/continue paths of one structured source owner.
  void enter(bool declaration_scope = true);
  void leave(cxx::AST* owner, std::span<cxx::Symbol* const> written);
  // Ends a path that returns or has no defined continuation.
  void end_path();
  void break_from_region();
  void continue_iteration();
  // Consumes iteration exits at their source target. The caller evaluates
  // the condition/increment only on paths reaching that phase.
  void reach_latch();
  void finish_loop(State condition_exit);
  void finish_switch(State unmatched);

  // Retained outgoing availability, queried without revisiting source code.
  bool initialized(cxx::AST* owner, cxx::Symbol* binding, size_t offset,
                   size_t count) const;

 private:
  struct Frame {
    // Escapes belonging to enclosing source owners, restored on leave.
    State breaks;
    // Continues belonging to enclosing iteration owners.
    State continues;
    // Declarations whose source lifetime ends at this owner.
    std::vector<cxx::Symbol*> declarations;
    // Lexical declaration owner restored after a nonlexical switch clause.
    size_t enclosing_declaration_scope;
  };

  // Current fallthrough path during the owning source traversal.
  State current_;
  // Intersection of paths escaping the active owner via break.
  State breaks_{false, {}};
  // Intersection of paths escaping the active owner via continue.
  State continues_{false, {}};
  // Nested source owners; empty maps allocate no component storage.
  std::vector<Frame> frames_;
  // Direct index of the lexical scope owning newly visited declarations.
  size_t declaration_scope_ = SIZE_MAX;
  // Complete component count for the sparse set of tracked declarations.
  std::unordered_map<cxx::Symbol*, size_t> declarations_;
  // Completed availability at structured joins, including iteration exits
  // but excluding paths that return from the callable.
  std::unordered_map<cxx::AST*, State> outgoing_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_CONSTRUCTION_H_
