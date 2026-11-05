// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_OWNERSHIP_H_
#define LOOM_IMPORT_CXX_VALUE_OWNERSHIP_H_

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/value/duplication.h"

namespace loom::cxx_import {

// One call operand tuple whose results are tied to and consume its identities.
struct TiedValueConsumption {
  // Complete source value transported through the call.
  Value value;
  // Source type used to bind a structurally equivalent duplicate tuple.
  const cxx::Type* source_type;
  // First component in the flattened call operand list.
  size_t argument_offset;
};

// Separates tied call operands from source-level value copies that retain the
// same SSA identities. The caller contributes exactly the live semantic owners
// at this source boundary, including anonymous prvalue receivers. Accounting
// is linear in owner, argument, and tied-component occurrences; ordinary calls
// never construct this object or scan their values.
class ValueOwnership {
 public:
  explicit ValueOwnership(ValueDuplication& duplication)
      : duplication_(duplication) {}

  // Records one live source owner. Repeated component identities are distinct
  // owner occurrences, as with two aggregate members copied from one scalar.
  void add_owner(Value value);
  // Records a projected owner slice when another part of its aggregate is the
  // tied destination and will be replaced by the call result.
  void add_owner_components(std::span<const loom_value_id_t> components);

  // Rewrites only those tied operand tuples that would otherwise consume an
  // identity also owned or used elsewhere. When every owner is consumed, the
  // final compatible occurrence keeps the original identity.
  void separate(std::span<const TiedValueConsumption> consumptions,
                std::vector<loom_value_id_t>& arguments, cxx::AST* owner);

 private:
  // Number of live semantic owner occurrences for each High identity.
  std::unordered_map<loom_value_id_t, size_t> owner_counts_;
  // Representation-aware duplicate emitter shared by every source value kind.
  ValueDuplication& duplication_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_OWNERSHIP_H_
