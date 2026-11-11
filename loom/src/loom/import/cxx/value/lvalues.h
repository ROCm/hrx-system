// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_LVALUES_H_
#define LOOM_IMPORT_CXX_VALUE_LVALUES_H_

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

// The current value receiver is partitioned into its own field bindings.
struct CurrentReceiver {
  // Retained structure whose direct fields own the receiver's current values.
  const RecordPartition* record;
};

// One complete object selected by an automatic binding or evaluated memory
// address. Reads and writes retain this identity as current values change.
struct ObjectLvalue {
  // Source object type retains qualifiers and storage access rules.
  const cxx::Type* type;
  // An automatic SSA partition or an evaluated storage projection.
  std::variant<Destination, StorageAccess, StorageProjection, CurrentReceiver>
      location;
};

// One lane whose complete vector destination and index were evaluated once.
struct VectorLaneLvalue {
  // Complete vector destination receiving the functional lane update.
  ObjectLvalue destination;
  // Snapshot for one immediate compound operation. Durable references have no
  // snapshot and observe the destination's current value on each later use.
  std::optional<Value> value;
  // Lane selector before projection to Loom's index domain.
  loom_value_id_t index;
  // Source selector type retaining its integer width and signedness.
  const cxx::Type* index_type;
};

// A frontend-validated static component projection of a complete vector.
struct VectorComponentLvalue {
  // Complete vector destination receiving functional lane updates.
  ObjectLvalue destination;
  // Vector snapshot used by one immediate compound operation.
  std::optional<Value> value;
  // Resolved static selector and source member spelling.
  cxx::MemberExpressionAST* selection;
};

struct SelectedLvalue;

// An evaluated reference retains source identity separately from the current
// immutable High value. Neither construction nor projection revisits source
// expressions. Selected alternatives share an immutable, function-local plan;
// only their ordinal and dynamic coordinates are transported in High SSA.
struct Lvalue {
  // Complete object or one evaluated projection of a complete vector object.
  std::variant<ObjectLvalue, VectorLaneLvalue, VectorComponentLvalue,
               std::shared_ptr<const SelectedLvalue>>
      projection;

  const SelectedLvalue* selection() const;
  // Complete owner of one concrete object/lane/component projection. Callers
  // recurse through selected references before this query.
  const ObjectLvalue& object() const;
  // Applies a resolved field/base projection to each possible source origin.
  Lvalue project(const SubobjectProjection& selected, const cxx::Type* type,
                 Storage& storage, cxx::AST* owner) const;
};

struct SelectedLvalue {
  SelectedLvalue(loom_value_id_t selector, const cxx::Type* type,
                 std::vector<Lvalue> alternatives,
                 std::optional<loom_value_id_t> lane = std::nullopt);

  // Ordinal of the concrete source object selected at initialization.
  loom_value_id_t selector;
  // Source object type shared by the selected alternatives.
  const cxx::Type* type;
  // Concrete destinations or independently selected nested destinations.
  // Retaining the selection tree avoids multiplying already evaluated source
  // choices merely to encode another choice around them.
  std::vector<Lvalue> alternatives;
  // Joined lane coordinate, if any alternative names an indexed vector.
  // Source-valid coordinates use an unsigned 64-bit carrier independently of
  // the written index expression's integer width and signedness.
  std::optional<loom_value_id_t> lane;
  // Unique automatic bindings whose current values a selected write can
  // replace. Captured once at construction and reused by every write and call
  // projection.
  std::vector<cxx::Symbol*> bindings;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_LVALUES_H_
