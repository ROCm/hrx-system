// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
#define LOOM_IMPORT_CXX_VALUE_BINDINGS_H_

#include <cxx/symbols_fwd.h>

#include <unordered_map>

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
    return values_.contains(binding);
  }
  const Value& at(cxx::Symbol* binding) const { return values_.at(binding); }

  void set(cxx::Symbol* binding, Value value) {
    values_.insert_or_assign(binding, value);
  }
  void erase(cxx::Symbol* binding) { values_.erase(binding); }
  void clear() { values_.clear(); }

 private:
  // Canonical current source bindings copied at structured-control forks.
  Map values_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_BINDINGS_H_
