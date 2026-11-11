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
#include <variant>
#include <vector>

#include "loom/import/cxx/control/components.h"

namespace loom::cxx_import {

// Source field effects retained by ControlFlow's single source traversal.
// Construction states name immutable graph positions. Call resolution supplies
// callee contracts before availability is published; emission only queries the
// finalized facts. Ordinary functions without tracked objects allocate no
// graph.
class Construction {
 public:
  struct State {
    // Whether this path continues to the next source expression.
    bool reachable = true;
    // Retained effect position, with zero denoting the empty entry state.
    size_t point = 0;
  };

  enum class ParameterMode { Projected, Value, Storage };
  struct Parameter {
    // Complete flattened source value extent.
    size_t component_count;
    // Selected source transport; storage calls do not define SSA fields.
    ParameterMode mode = ParameterMode::Projected;
    // Whether writes survive this call. A destructor consumes its receiver.
    bool returned = true;
    // Complete component groups required by the input contract.
    Components inputs;
    // Complete component groups returned by a value call.
    Components outputs;
  };
  struct Argument {
    // Caller object identity, retained by ControlFlow.
    cxx::Symbol* binding;
    // First caller component corresponding to the formal interval.
    size_t offset;
    // Referenced interval within the callee's formal object.
    ComponentRange formal;
    // Callee contract, or null for an external/unmodeled source call.
    const Parameter* parameter;
    // Whether this actual object is selected on every path reaching the call.
    bool definite;
    // External call permission when no callee contract is available.
    bool writable;
  };
  struct Call {
    // Index into ControlFlow's retained direct calls.
    size_t source_index;
    // Producer-resolved formal-to-actual component mappings.
    std::vector<Argument> arguments;
  };
  struct Destination {
    // Caller object selected by the resolved source lvalue.
    cxx::Symbol* binding;
    // Complete component interval assigned in that object.
    ComponentRange range;
    // First source-value component mapped into this destination interval.
    size_t source_offset;
    // Whether every alternative names this same object interval.
    bool definite;
  };
  struct Assignment {
    // Index into ControlFlow's retained assignment identities.
    size_t source_index;
    // Resolved destinations; selections can name several possible objects.
    std::vector<Destination> destinations;
  };

  // Classifies actual mappings selected on every source path. Overlapping
  // formal components remain definite only when every alternative names the
  // same caller identity and offset.
  static void classify_definite_arguments(std::span<Argument> arguments);
  // Classifies indirect assignment destinations selected on every source
  // path. Overlapping source components remain definite only when every
  // alternative names the same complete destination interval.
  static void classify_definite_destinations(
      std::span<Destination> destinations);

  const State& state() const { return current_; }
  void restore(State state) { current_ = state; }
  State join(State left, State right);

  void declare(cxx::Symbol* binding, size_t component_count);
  void declare_parameter(cxx::Symbol* binding, size_t component_count,
                         bool returned = true);
  bool tracked(cxx::Symbol* binding) const;
  bool declared(cxx::Symbol* binding) const {
    return declarations_.contains(binding);
  }
  bool has_parameters() const { return !parameters_.empty(); }
  bool has_bindings() const { return !declarations_.empty(); }
  bool has_effects() const { return !nodes_.empty(); }
  Parameter* parameter(cxx::Symbol* binding);
  const Parameter* parameter(cxx::Symbol* binding) const;
  void configure_parameter(cxx::Symbol* binding, ParameterMode mode,
                           bool written);
  void read(cxx::Symbol* binding, size_t offset, size_t count);
  void assign(cxx::Symbol* binding, size_t offset, size_t count);
  // Retains an assignment through a selected or returned reference at its
  // source position. Identity propagation binds its destinations later.
  void assign_indirect(size_t source_index);
  std::span<Assignment> assignments() { return assignments_; }
  void call(size_t source_index);
  // Registers cleanup after successful construction. Its effects occur when
  // the current declaration scope is left, in reverse registration order.
  void defer_call(size_t source_index);
  std::span<Call> calls() { return calls_; }

  void enter(bool declaration_scope = true);
  void leave(cxx::AST* owner);
  void return_from_function();
  void end_path();
  void break_from_region();
  void continue_iteration();
  void reach_latch();
  void finish_loop(State condition_exit);
  void finish_switch(State unmatched);
  void finish_function();

  // Resolution grows finite component sets. Outputs stabilize before flow is
  // evaluated once; input dependencies then consume the retained reaching
  // facts.
  bool propagate_outputs();
  using Writes = std::unordered_map<cxx::AST*, std::vector<cxx::Symbol*>>;
  void resolve(const Writes& writes);
  bool propagate_inputs();
  bool initialized(cxx::AST* owner, cxx::Symbol* binding, size_t offset,
                   size_t count) const;

 private:
  struct Access {
    // Source object owning this interval.
    cxx::Symbol* binding;
    // Component interval read or defined by the source expression.
    ComponentRange range;
  };
  struct Read {
    // Source components consumed at this position.
    Access access;
  };
  struct Assign {
    // Source components completely defined at this position.
    Access access;
  };
  struct Invoke {
    // Index into the retained formal-to-actual mappings.
    size_t call_index;
  };
  struct IndirectAssign {
    // Index into the resolved assignment destinations.
    size_t assignment_index;
  };
  struct Join {
    // Alternative predecessor whose availability must also hold.
    size_t other;
  };
  struct Forget {
    // Object whose lexical lifetime ended.
    cxx::Symbol* binding;
  };
  struct Node {
    // Prior ordered effect on this source path.
    size_t previous;
    // Source effect or control-flow merge applied at this position.
    std::variant<Read, Assign, IndirectAssign, Invoke, Join, Forget> effect;
  };
  struct Declaration {
    // Complete partition extent, retained after the lexical scope ends.
    size_t component_count;
    // Whether this binding is in the current lexical declaration scope.
    bool active = true;
  };
  struct Frame {
    // Exits of the surrounding source owner.
    State breaks;
    // Iteration continuations of the surrounding source owner.
    State continues;
    // Returns outside this scope; new returns unwind through its cleanups.
    State returns;
    // Objects whose lifetime ends when this declaration scope is left.
    std::vector<cxx::Symbol*> declarations;
    // Call indices executed before the declarations in this scope expire.
    std::vector<size_t> cleanups;
    // Enclosing declaration scope, or SIZE_MAX at function entry.
    size_t enclosing_declaration_scope;
  };
  struct InputDependency {
    // Caller formal whose entry requirement depends on this call.
    Parameter* destination;
    // Callee formal supplying the transitive required components.
    const Parameter* source;
    // Callee component interval mapped into the caller formal.
    ComponentRange formal;
    // First destination component corresponding to formal.begin.
    size_t offset;
    // Destination components already defined before this call.
    Components available;
  };
  using Available = std::unordered_map<cxx::Symbol*, Components>;
  void append(
      std::variant<Read, Assign, IndirectAssign, Invoke, Join, Forget> effect);
  State cleanup(State state, std::span<const size_t> calls);
  ComponentRange range(cxx::Symbol* binding, size_t offset, size_t count) const;

  // Current source continuation and escaping alternatives.
  State current_;
  // Break paths of the active source owner.
  State breaks_{false};
  // Continue paths of the active source iteration.
  State continues_{false};
  // Normal function returns, including implicit fallthrough.
  State returns_{false};
  // Lexical/control stack used only during the owning source traversal.
  std::vector<Frame> frames_;
  // Index of the current declaration scope in frames_.
  size_t declaration_scope_ = SIZE_MAX;
  // Participating local and formal object declarations.
  std::unordered_map<cxx::Symbol*, Declaration> declarations_;
  // Formal contracts; references into this table remain stable during solving.
  std::unordered_map<cxx::Symbol*, Parameter> parameters_;
  // Ordered source effects, with predecessor indices smaller than each node.
  std::vector<Node> nodes_;
  // Direct call mappings supplied after source identity propagation.
  std::vector<Call> calls_;
  // Assignments whose object identities arrive from the shared call analysis.
  std::vector<Assignment> assignments_;
  // Parameter and receiver subobject cleanups following body-local cleanup.
  std::vector<size_t> function_cleanups_;
  // Region exits whose finalized availability is consumed by translation.
  std::unordered_map<cxx::AST*, State> checkpoints_;
  // Sparse outgoing facts, restricted to each region's written objects.
  std::unordered_map<cxx::AST*, Available> outgoing_;
  // Input requirements with the caller's reaching definitions already removed.
  std::vector<InputDependency> input_dependencies_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_CONSTRUCTION_H_
