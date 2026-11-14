// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_RECEIVER_SCOPES_H_
#define LOOM_IMPORT_CXX_RECEIVER_SCOPES_H_

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// One current source object, independent of expression evaluation and emission.
struct ReceiverState {
  // Canonical flattened structure shared by value and storage receivers.
  const RecordPartition* record;
  // Addressable destination, absent for a record assembled as SSA values.
  std::optional<StorageProjection> storage;
  // Cv-qualified complete object type governing storage observations.
  const cxx::Type* object_type;
};

// Owns lexical current-object scopes and subobject completion. Nested receivers
// may reuse the same subobject declarations; entering one hides the enclosing
// bindings until it leaves. Explicit aggregate clauses suspend construction
// state while evaluating against the lexical receiver, preserving writes to
// that receiver when construction resumes. All values remain in the shared
// environment used by ordinary structured-control snapshots.
class ReceiverScopes {
 public:
  explicit ReceiverScopes(ValueBindings& values) : values_(values) {}

  void enter(const RecordPartition& record,
             std::optional<StorageProjection> storage,
             const cxx::Type* object_type);
  void leave();
  // Returns whether a construction receiver was hidden. With no enclosing
  // current object, type-checked explicit clauses cannot observe that receiver.
  bool suspend();
  void resume();
  // Discards function-local scopes after their common values are cleared.
  void clear();

  // Borrowed states remain valid until the active scope stack changes.
  const ReceiverState* current() const {
    return frames_.empty() ? nullptr : &frames_.back().receiver;
  }
  const ReceiverState* find(cxx::Symbol* subobject) const;
  // Subobject queries and updates use declarations owned by an active receiver.
  bool is_complete(cxx::Symbol* subobject) const {
    return owners_.at(subobject).complete;
  }
  // Publishes a completed value subobject into the common binding environment.
  void publish(cxx::Symbol* subobject, Value value);
  // Marks a subobject's source lifetime as begun. A value member can still
  // have pending payload components after default initialization.
  void mark_complete(cxx::Symbol* subobject);
  // Replaces the root refinement carried by the current storage receiver.
  // The source object, byte origin, layout, and completion state are retained.
  void replace_current_storage_root(loom_value_id_t root);

 private:
  struct Owner {
    // Active frame containing this direct subobject.
    size_t index;
    // Whether source initialization has completed the subobject.
    bool complete;
  };

  struct Binding {
    // Member identity owned by this source current-object scope.
    cxx::Symbol* subobject;
    // Enclosing receiver for the same declaration, when recursively active.
    std::optional<Owner> previous_owner;
    // Complete or pending enclosing binding hidden by the current receiver.
    ValueBindings::Binding previous_binding;
  };

  struct Frame {
    // Object whose members are visible in this lexical current-object scope.
    ReceiverState receiver;
    // State restored when this current-object scope ends.
    std::vector<Binding> bindings;
  };

  struct SuspendedBinding {
    // Construction ownership hidden during an explicit aggregate clause.
    Owner owner;
    // Construction binding hidden independently of lexical receiver mutations.
    ValueBindings::Binding binding;
  };

  struct SuspendedFrame {
    // Stack position restored after the explicit clause completes.
    size_t index;
    // Construction receiver whose state remains live but lexically hidden.
    Frame frame;
    // Completion and value state restored independently of lexical writes.
    std::vector<SuspendedBinding> bindings;
  };

  // Shared source environment; its identity outlives every receiver scope.
  ValueBindings& values_;
  // Nested current objects in source evaluation order.
  std::vector<Frame> frames_;
  // Direct declaration lookup avoids searching the stack on each access.
  std::unordered_map<cxx::Symbol*, Owner> owners_;
  // Temporarily hidden construction receivers, restored in reverse order.
  std::vector<SuspendedFrame> suspended_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_RECEIVER_SCOPES_H_
