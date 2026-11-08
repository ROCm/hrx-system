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
#include <vector>

#include "loom/import/cxx/value/representation.h"

namespace loom::cxx_import {

// Current immutable High values, copied at structured-control forks.
class ValueBindings {
 public:
  using Map = std::unordered_map<cxx::Symbol*, Value>;
  using const_iterator = Map::const_iterator;

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
  // Canonical current source bindings copied at structured-control forks.
  Map values_;
  struct Pending {
    // Complete source structure admitted before any field value exists.
    const Partition* partition;
    // Availability is frontend state; absent entries never become IR values.
    std::vector<std::optional<loom_value_id_t>> components;
    // Unassigned component count, avoiding a complete-record scan on every
    // member assignment while a large aggregate is being constructed.
    size_t remaining;
  };
  // Declared source objects with no complete payload yet.
  std::unordered_map<cxx::Symbol*, Pending> uninitialized_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
