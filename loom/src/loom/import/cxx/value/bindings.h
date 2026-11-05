// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
#define LOOM_IMPORT_CXX_VALUE_BINDINGS_H_

#include <cxx/symbols_fwd.h>

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "loom/import/cxx/value/representation.h"

namespace loom::cxx_import {

// Current immutable High values of source bindings. A reverse component index
// is materialized only when mutable-call ownership needs it, then maintained
// across binding updates and structured-control snapshots. Ordinary functions
// retain the source map plus one null pointer and perform no indexing work.
class ValueBindings {
 public:
  using Map = std::unordered_map<cxx::Symbol*, Value>;
  using const_iterator = Map::const_iterator;
  using OwnerCounts = std::unordered_map<cxx::Symbol*, size_t>;

  ValueBindings() = default;
  ValueBindings(const ValueBindings& other);
  ValueBindings(ValueBindings&& other) noexcept = default;
  ValueBindings& operator=(const ValueBindings& other);
  ValueBindings& operator=(ValueBindings&& other) noexcept = default;

  const_iterator find(cxx::Symbol* binding) const {
    return values_.find(binding);
  }
  const_iterator end() const { return values_.end(); }
  bool contains(cxx::Symbol* binding) const {
    return values_.contains(binding);
  }
  const Value& at(cxx::Symbol* binding) const { return values_.at(binding); }

  // Replaces one complete source binding and updates a live reverse index.
  void set(cxx::Symbol* binding, Value value);
  void erase(cxx::Symbol* binding);
  void clear();

  // Returns source bindings currently carrying |component| and each binding's
  // number of matching flattened components. The borrowed map remains valid
  // until this collection is mutated.
  const OwnerCounts* owners(loom_value_id_t component) const;

 private:
  using OwnerIndex = std::unordered_map<loom_value_id_t, OwnerCounts>;

  void ensure_owner_index() const;
  static void add(OwnerIndex& index, cxx::Symbol* binding, Value value);
  static void remove(OwnerIndex& index, cxx::Symbol* binding, Value value);

  // Canonical current source bindings copied at structured-control forks.
  Map values_;
  // Lazily materialized reverse component ownership for mutable tied calls.
  mutable std::unique_ptr<OwnerIndex> owner_index_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
