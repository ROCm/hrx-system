// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
#define LOOM_IMPORT_CXX_VALUE_BINDINGS_H_

#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "loom/import/cxx/value/representation.h"

namespace loom::cxx_import {

// Retained ownership of a source assignment destination. A field or vector-lane
// write replaces a slice of its owning automatic binding; it is not a
// storage-backed store. Dynamic lane selection remains on the source expression
// and is evaluated once by translation.
struct Destination {
  // Owning local or parameter identity, borrowed from the source function.
  cxx::Symbol* binding;
  // First component of a nested member in the complete owning binding.
  size_t component_offset;
  // Member structure, or null when the destination is the entire binding.
  const Partition* member;
};

// Current immutable High values, copied at structured-control forks.
class ValueBindings {
 public:
  using Map = std::unordered_map<cxx::Symbol*, Value>;
  using const_iterator = Map::const_iterator;

 private:
  struct Pending {
    // Complete source structure admitted before any field value exists.
    const Partition* partition;
    // Availability is frontend state; absent entries never become IR values.
    std::vector<std::optional<loom_value_id_t>> components;
    // Unassigned component count, avoiding a complete-record scan on every
    // member assignment while a large aggregate is being constructed.
    size_t remaining;
  };
  using PendingMap = std::unordered_map<cxx::Symbol*, Pending>;

 public:
  // Move-owned binding hidden by a nested source scope. Retaining the map node
  // preserves partial construction without copying its components or allocating
  // a replacement node when the enclosing scope resumes. An empty node denotes
  // an absent binding.
  class Binding {
   public:
    Binding() = default;

   private:
    friend class ValueBindings;
    explicit Binding(Map::node_type value) : state_(std::move(value)) {}
    explicit Binding(PendingMap::node_type pending)
        : state_(std::move(pending)) {}
    // Exactly one complete or pending binding, owned until restored or dropped.
    std::variant<Map::node_type, PendingMap::node_type> state_;
  };

  // Accumulates functional subobject replacements in one mutable component list
  // per changed binding, then captures the completed value on apply. Reads see
  // earlier replacements; source bindings and snapshots stay unchanged until
  // then. No source expressions or externally observable effects occur in the
  // batch.
  class Updates {
   public:
    Updates(ValueBindings& bindings, ValueArena& arena)
        : bindings_(bindings), arena_(arena) {}

    std::optional<Value> project(cxx::Symbol* binding,
                                 const Partition& partition, size_t offset);
    void replace(cxx::Symbol* binding, size_t offset, Value value);
    void apply();

   private:
    // Current source bindings receiving the completed replacements.
    ValueBindings& bindings_;
    // Function-local immutable payload storage for reads and completed values.
    ValueArena& arena_;
    // Mutable component lists owned only by this batch, never by a snapshot.
    PendingMap pending_;
  };

  // Removes the current binding until a source scope restores it.
  Binding take(cxx::Symbol* binding);
  // Replaces the current binding with state previously taken for this symbol.
  void restore(cxx::Symbol* binding, Binding state);

  const_iterator find(cxx::Symbol* binding) const {
    return values_.find(binding);
  }
  const_iterator end() const { return values_.end(); }
  bool contains(cxx::Symbol* binding) const {
    return values_.contains(binding) || uninitialized_.contains(binding);
  }
  bool initialized(cxx::Symbol* binding) const {
    return values_.contains(binding);
  }
  const Value& at(cxx::Symbol* binding) const { return values_.at(binding); }

  const Partition& partition(cxx::Symbol* binding) const;
  // Starts a source binding without manufacturing any High values. An empty
  // record is already complete because it has no transported components.
  void declare(cxx::Symbol* binding, const Partition& partition);
  // Tests a complete subobject's availability without assembling its value.
  bool initialized(cxx::Symbol* binding, size_t offset, size_t count) const;
  // Reads one complete group, or reports that its source construction has not
  // supplied every component yet. Partial records never become High values.
  std::optional<Value> project(cxx::Symbol* binding, const Partition& partition,
                               size_t offset, ValueArena& arena) const;
  // Replaces an immutable subobject, completing a pending aggregate when its
  // final missing component arrives. Snapshots retain independent availability.
  void replace(cxx::Symbol* binding, size_t offset, Value value,
               ValueArena& arena);

  void set(cxx::Symbol* binding, Value value) {
    values_.insert_or_assign(binding, value);
    uninitialized_.erase(binding);
  }
  void erase(cxx::Symbol* binding) {
    values_.erase(binding);
    uninitialized_.erase(binding);
  }
  void clear() {
    values_.clear();
    uninitialized_.clear();
  }

 private:
  static std::optional<Value> project(const Pending& pending,
                                      const Partition& partition, size_t offset,
                                      ValueArena& arena);

  // Canonical current source bindings copied at structured-control forks.
  Map values_;
  // Declared source objects with no complete payload yet.
  PendingMap uninitialized_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
