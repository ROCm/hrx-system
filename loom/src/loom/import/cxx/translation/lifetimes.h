// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TRANSLATION_LIFETIMES_H_
#define LOOM_IMPORT_CXX_TRANSLATION_LIFETIMES_H_

#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loom/import/cxx/lifetime/destruction.h"
#include "loom/import/cxx/value/evaluation.h"
#include "loom/import/cxx/value/partition.h"
#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

class ControlFlow;
class Objects;

// Evaluated source reference into one nontrivial temporary object. Geometry
// remains explicit when a returned reference denotes a base or subobject.
struct TemporaryReference {
  // Stable construction-instance index in the current cleanup queue.
  size_t index;
  // First SSA component within the complete temporary's current value.
  size_t component_offset = 0;
  // Source-layout displacement if the complete object becomes addressable.
  uint64_t byte_offset = 0;
};

// One completed source object awaiting destruction. Automatic bindings and
// current receiver fields resolve their latest value at the cleanup edge;
// temporary objects retain the value or storage created by their expression.
struct DestructionAction {
  // Frontend-selected complete-object destruction.
  Destruction destruction;
  // Automatic local or parameter binding, mutually exclusive with |field|,
  // |value| and |storage|.
  cxx::Symbol* binding = nullptr;
  // Current receiver subobject, mutually exclusive with |binding|, |value|
  // and |storage|.
  cxx::Symbol* field = nullptr;
  // Complete source object type, including any bounded array extents.
  const cxx::Type* object_type = nullptr;
  // Direct SSA object retained by an unmaterialized temporary.
  std::optional<Value> value;
  // One-based head of addressable members to read back at destruction.
  size_t borrowed_members = 0;
  // Direct object storage retained by an addressed temporary.
  std::optional<StorageProjection> storage;
  // Conditional temporary activation flag, stored outside its source arm.
  std::optional<StorageAccess> activity;
  // Canonical construction site; each evaluation gets its own cleanup entry.
  cxx::ExpressionAST* origin = nullptr;
  // Source declaration or expression owning the implicit call location.
  cxx::AST* source = nullptr;
};

// Addressable construction records the prospective cleanup separately from
// the returned address until its initializer finishes.
struct PendingConstruction {
  // Object storage passed to the initializer.
  StorageProjection storage;
  // Present when the completed object requires destruction.
  std::optional<DestructionAction> cleanup;
};

// Tracks construction, transfer, and destruction in the current source body.
// Source identity and lifetime decisions come from its retained control facts.
class ObjectLifetimes {
 public:
  class Emitter {
   public:
    // Resolve a named object at the cleanup edge, after its last mutation.
    virtual void destroy_object(const DestructionAction& action) = 0;

   protected:
    ~Emitter() = default;
  };

  ObjectLifetimes(cxx::TranslationUnit& unit, Scalars& scalars,
                  Storage& storage, Objects& objects, Locations& locations,
                  ValueArena& arena, loom_builder_t& builder, Emitter& emitter);
  ~ObjectLifetimes();
  void reset(const ControlFlow& control);

  // A structured arm saves its lexical objects on entry and restores them
  // when switching arms. A terminating edge has already emitted its cleanups.
  size_t scope_mark() const { return active_destructions_.size(); }
  using Scope = std::vector<DestructionAction>;
  Scope save_scope() const { return active_destructions_; }
  void restore_scope(const Scope& scope) { active_destructions_ = scope; }
  void restore_scope(Scope&& scope) { active_destructions_ = std::move(scope); }
  // Retire objects on a terminated path whose cleanups have already run.
  void discard_scope(size_t mark) { active_destructions_.resize(mark); }
  void activate(DestructionAction action) {
    active_destructions_.push_back(std::move(action));
  }
  // Buffer restrict scopes refine stored incoming roots before body entry.
  std::span<DestructionAction> automatic_objects() {
    return active_destructions_;
  }
  size_t temporary_mark() const { return temporary_destructions_.size(); }
  void enter_conditional(loom_builder_ip_t anchor) {
    conditional_cleanup_anchors_.push_back(anchor);
  }
  void leave_conditional(loom_op_t* operation = nullptr);
  // Emit the reverse-construction-order suffix. Other control-flow edges may
  // still need the same active objects. Closing a scope also retires them.
  void emit_destructions(size_t mark);
  void emit_automatic_destruction(cxx::Symbol* binding, cxx::AST* source);
  void emit_automatic_destruction_if(loom_value_id_t condition,
                                     cxx::Symbol* binding, cxx::AST* source);
  void close_scope(size_t mark);
  void close_scope_if(loom_value_id_t condition, size_t mark, cxx::AST* source);
  void activate_local(cxx::Symbol* binding, cxx::AST* source);
  void activate_subobjects(cxx::AST* source);
  void materialize_conditional_temporaries(size_t mark);
  std::optional<TemporaryReference> temporary_result(
      cxx::ExpressionAST* expression) const;
  // Attach the result to its evaluated construction site. Transparent source
  // wrappers can share that result within one evaluation.
  void retain_temporary(cxx::ExpressionAST* origin,
                        cxx::ExpressionAST* expression, Value value,
                        size_t evaluation_mark);
  FullExpression begin_full_expression(bool extend_temporaries = false);
  void finish_full_expression(FullExpression expression);
  void adopt_temporary_result(cxx::ExpressionAST* expression);
  void bind_temporary_materialization(cxx::ExpressionAST* expression,
                                      cxx::Symbol* binding, cxx::AST* source);
  void update_temporary(std::optional<TemporaryReference> temporary,
                        Value updated, const cxx::Type* type, cxx::AST* owner);
  // Materializes a numeric member of a completed object. A nontrivial object's
  // cleanup observes the latest contents through this same address.
  StorageProjection borrow_member(Value value, ValueProjection group,
                                  std::optional<TemporaryReference> temporary,
                                  cxx::AST* source);
  Value temporary_value(const DestructionAction& action);
  StorageProjection materialize_temporary(
      cxx::ExpressionAST* expression, Value value,
      std::optional<TemporaryReference> temporary = std::nullopt);
  PendingConstruction begin_construction(const cxx::Type* type,
                                         cxx::AST* owner);
  void complete_construction(PendingConstruction& object);

 private:
  void emit_destruction(const DestructionAction& active);
  void allocate_temporary_storage(DestructionAction& action);
  void activate_temporary_storage(const DestructionAction& action);
  void materialize_temporary_value(DestructionAction& action);
  void forget_temporary(size_t index);
  void extend_materialized_temporary(size_t index);

  // Canonical source types for activity flags and materializations.
  cxx::TranslationUnit& unit_;
  // Source's predicate and integer builder.
  Scalars& scalars_;
  // Physical object and activity-flag storage.
  Storage& storage_;
  // Value transport for materialized nontrivial objects.
  Objects& objects_;
  // Source ranges for all implicit cleanup operations.
  Locations& locations_;
  // Owns large SSA values across temporary writeback.
  ValueArena& value_arena_;
  // Insertion point for materialization and conditional cleanup.
  loom_builder_t& builder_;
  // Current binding and receiver resolver at each destruction edge.
  Emitter& emitter_;
  // Borrowed analysis of the active function, installed by reset().
  const ControlFlow* control_ = nullptr;
  // Completed automatic objects and destructor subobjects active at the
  // current source insertion point, in construction order.
  std::vector<DestructionAction> active_destructions_;
  // Nontrivial prvalue results retained until the innermost full expression
  // completes. Result-object transfer removes only the selected final value;
  // side temporaries retain reverse construction order.
  std::vector<std::optional<DestructionAction>> temporary_destructions_;
  struct BorrowedMember {
    // Selected member relative to the complete temporary.
    ValueProjection group;
    // Numeric member's address for this evaluation.
    StorageProjection home;
    // Previous member of this same cleanup, or zero.
    size_t previous;
  };
  // Allocated only for evaluated nontrivial prvalues with addressable members.
  std::vector<BorrowedMember> borrowed_members_;
  // Latest evaluated result at each retained construction site. Cleanup tokens
  // remain stable when another result is adopted or extended to lexical scope.
  std::unordered_map<cxx::ExpressionAST*, size_t> temporary_results_;
  // Temporary prefix owned by the current full expression. Lexical caller
  // tokens retain parent marks without a heap-backed translator stack.
  std::optional<size_t> full_expression_mark_;
  // A range initializer extends its surviving temporaries to the loop scope.
  // Nested full expressions save and suspend this policy in their caller token.
  bool extends_temporaries_ = false;
  // Structured conditional expressions hoist branch-only temporary storage
  // before the outermost conditional operation so cleanup remains dominated.
  std::vector<loom_builder_ip_t> conditional_cleanup_anchors_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_TRANSLATION_LIFETIMES_H_
