// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_OBJECTS_H_
#define LOOM_IMPORT_CXX_VALUE_OBJECTS_H_

#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

// Bridges immutable source values and addressable source-layout objects.
// Aggregate reads capture every component before a replacing store begins,
// preserving independent copies and self-assignment. Memory offsets belong to
// Storage; recursive component identities belong to the retained type schema.
// No array value contains a pointer to temporary object storage.
class Objects {
 public:
  Objects(cxx::TranslationUnit& unit, Types& types, Storage& storage,
          ValueArena& arena)
      : unit_(unit), types_(types), storage_(storage), arena_(arena) {}

  // Captures a scalar, vector, or aggregate object's current value. Array
  // snapshots are used inside enclosing record values, separately from decay.
  Value load(StorageProjection object, const cxx::Type* type, cxx::AST* owner);
  // Replaces object value components in source declaration/element order.
  // Padding is not observed or overwritten by this value projection.
  void store(StorageProjection object, Value value, const cxx::Type* type,
             cxx::AST* owner);
  // Applies the enclosing object's observation qualifier to a source member.
  const cxx::Type* member_type(const cxx::Type* object_type,
                               cxx::Symbol* field) const;

 private:
  void load_components(StorageProjection object, const cxx::Type* type,
                       const Partition& partition, cxx::AST* owner,
                       std::vector<loom_value_id_t>& components);

  // Source qualifiers propagate from enclosing objects to their members.
  cxx::TranslationUnit& unit_;
  // Retained value partitions and source type admission.
  Types& types_;
  // Typed memory operations and source-layout object projections.
  Storage& storage_;
  // Function-scoped immutable storage for captured aggregate components.
  ValueArena& arena_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_OBJECTS_H_
