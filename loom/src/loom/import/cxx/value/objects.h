// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_OBJECTS_H_
#define LOOM_IMPORT_CXX_VALUE_OBJECTS_H_

#include <memory>
#include <span>
#include <vector>

#include "loom/import/cxx/value/storage.h"

namespace loom::cxx_import {

class Locations;

// One source object whose managed-reference components live in typed cells.
// The descriptor is compiler state; the emitted views own runtime storage.
struct ReferenceStorage {
  // Static source-to-cell layout retained by Types.
  const ReferenceStorageLayout* layout;
  // Typed cell views in flattened source-component order.
  std::vector<loom_value_id_t> cells;
};

// A complete source value or subobject projected from shared reference cells.
struct ReferenceStorageProjection {
  // Shared root identity preserving aliases across source lvalues and calls.
  std::shared_ptr<const ReferenceStorage> storage;
  // Source value structure reconstructed by this projection.
  const Partition* partition;
  // First component in the root's flattened source order.
  size_t component_offset;
};

// Bridges immutable source values and addressable source-layout objects.
// Aggregate reads capture every component before a replacing store begins,
// preserving independent copies and self-assignment. Memory offsets belong to
// Storage; recursive component identities belong to the retained type schema.
// No array value contains a pointer to temporary object storage.
class Objects {
 public:
  Objects(cxx::TranslationUnit& unit, Types& types, Storage& storage,
          ValueArena& arena, Locations& locations, loom_builder_t& builder)
      : unit_(unit),
        types_(types),
        storage_(storage),
        arena_(arena),
        locations_(locations),
        builder_(builder) {}

  // Captures a scalar, vector, or aggregate object's current value. Array
  // snapshots are used inside enclosing record values, separately from decay.
  Value load(StorageProjection object, const cxx::Type* type, cxx::AST* owner);
  // Replaces object value components in source declaration/element order.
  // Padding is not observed or overwritten by this value projection.
  void store(StorageProjection object, Value value, const cxx::Type* type,
             cxx::AST* owner);
  // Allocates one private typed cell per managed-reference component and moves
  // |initial| into the resulting source object.
  ReferenceStorageProjection allocate_reference(
      const ReferenceStorageLayout& layout, Value initial, cxx::AST* owner);
  // Admits borrowed cell views supplied by a function boundary.
  ReferenceStorageProjection admit_reference(
      const ReferenceStorageLayout& layout,
      std::span<const loom_value_id_t> cells);
  // Projects a retained source subobject without changing cell identity.
  ReferenceStorageProjection project_reference(
      ReferenceStorageProjection object, const Partition& partition,
      size_t component_offset) const;
  // Loads an independent managed handle from each projected cell.
  Value load(ReferenceStorageProjection object, cxx::AST* owner);
  // Moves a complete source value into its projected cells, replacing owners.
  void store(ReferenceStorageProjection object, Value value, cxx::AST* owner);
  // Moves a complete source value out and leaves every projected cell null.
  Value take(ReferenceStorageProjection object, cxx::AST* owner);
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
  // Source locations attached to typed cell operations.
  Locations& locations_;
  // Current output insertion point for typed cell operations.
  loom_builder_t& builder_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_OBJECTS_H_
