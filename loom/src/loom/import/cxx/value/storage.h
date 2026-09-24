// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_STORAGE_H_
#define LOOM_IMPORT_CXX_VALUE_STORAGE_H_

#include <optional>
#include <unordered_map>

#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/ir/facts.h"

namespace loom::cxx_import {

// An evaluated source lvalue origin with an admitted storage representation.
// Alignment belongs to this object projection, not to the two-component
// pointer value transported by the ABI.
struct StorageProjection {
  // Allocation identity and byte origin of the projected object.
  Pointer pointer;
  // Source-layout byte alignment retained through fields and array indexing.
  uint64_t alignment;
  // Whether the emitted origin already carries the source pointer-width fact.
  bool pointer_width_constrained = false;
};

// One typed access, either a dynamic element of a retained array view or a
// scalar or vector projection of a pointer's buffer.
struct StorageAccess {
  // Typed view containing the memory access.
  loom_value_id_t view;
  // Full-rank dynamic array index, absent for a scalar pointer projection.
  std::optional<loom_value_id_t> index;
};

struct StorageAllocation {
  // Pointer to the beginning of the declared allocation.
  Pointer pointer;
  // Typed view retaining its element type and fixed extent.
  loom_value_id_t view;
};

// Owns the source memory representation contract for one output module.
// Operands are already evaluated by the caller. Fixed array views are recorded
// at allocation and reused directly, including through source aliases.
// Source, type, scalar, location and insertion-point owners outlive this
// object.
class Storage {
 public:
  Storage(cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
          Scalars& scalars, Locations& locations, loom_builder_t& builder)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        scalars_(scalars),
        locations_(locations),
        builder_(builder) {}

  // Forms a zero-origin source pointer for a kernel buffer binding. The buffer
  // already carries any admitted parameter contracts.
  Pointer root(loom_value_id_t buffer, cxx::AST* owner);
  // Refines an opaque pointer origin to the configured source pointer width.
  // The allocation root remains independent; this only publishes the range
  // every valid source pointer representation already satisfies.
  Pointer constrain_origin(Pointer pointer, cxx::AST* owner);
  // Admits the object's storage layout and starts a projection with its
  // ordinary source ABI alignment. Opaque pointer transport needs neither.
  // Nested fields use member() instead of resetting to their nominal type.
  StorageProjection project(Pointer pointer, const cxx::Type* object_type,
                            cxx::AST* owner);
  // Computes an object-relative byte origin using the source integer's width
  // and signedness. Subtraction is represented by T_MINUS; addition by T_PLUS.
  // Only the final origin enters offset, allowing negative displacements from
  // interior pointers without forming negative offset values. Its nonnegative
  // range is published from the source language's within-object precondition.
  StorageProjection advance(StorageProjection base,
                            loom_value_id_t displacement,
                            const cxx::Type* base_type,
                            const cxx::Type* index_type,
                            cxx::TokenKind operation, cxx::AST* owner);
  // Projects an admitted record field using its retained C++ byte offset.
  // The allocation identity is preserved, including for nested records.
  // Packing changes the origin, without asserting natural field alignment.
  StorageProjection member(StorageProjection base, cxx::FieldSymbol* field,
                           cxx::AST* owner);
  // Constructs addressing for an integral subscript without narrowing pointer
  // byte arithmetic through target-selected index.
  // Unsupported source types are diagnosed at owner. The C++ driver separately
  // admits builtin indexing and evaluates base before index.
  StorageAccess subscript(StorageProjection base, loom_value_id_t index,
                          const cxx::Type* base_type,
                          const cxx::Type* subscript_type, cxx::AST* owner);
  // Projects an object's scalar lane footprint at the pointer's current origin.
  StorageAccess dereference(StorageProjection base,
                            const cxx::Type* element_type, cxx::AST* owner);
  // Reads an already resolved scalar/vector element. Reusing an
  // access preserves its address across a source read/modify/write operation.
  // The source element type supplies the footprint and memory qualifiers.
  loom_value_id_t load(const StorageAccess& access,
                       const cxx::Type* element_type, cxx::AST* owner);
  // Writes to the same resolved location with the source element qualifiers.
  void store(const StorageAccess& access, loom_value_id_t value,
             const cxx::Type* element_type, cxx::AST* owner);
  // Allocates a scalar, vector or fixed scalar array using its source layout
  // and explicit alignment. Each execution creates a fresh root; initialization
  // is emitted separately. The driver owns storage-duration and scope
  // admission.
  StorageAllocation allocate(const cxx::Type* type,
                             loom_value_fact_memory_space_t memory_space,
                             int64_t explicit_alignment, cxx::AST* owner);

 private:
  // Source memory layout for element sizes and allocation alignment.
  cxx::TranslationUnit& unit_;
  // Source admission diagnostics.
  Diagnostics& diagnostics_;
  // Source-to-IR type representation contract.
  Types& types_;
  // Offset constant construction shares the normal scalar builder.
  Scalars& scalars_;
  // Retained provenance for emitted operations.
  Locations& locations_;
  // Borrowed insertion point, controlled by the AST driver.
  loom_builder_t& builder_;
  // A declared scalar array's direct-index view. Other array types or interior
  // origins sharing the allocation use ordinary object-relative addressing.
  struct ArrayView {
    // Unqualified source array type whose extent and stride formed the view.
    const cxx::Type* type;
    // Declared array origin; an interior pointer cannot reuse this view.
    loom_value_id_t byte_offset;
    // Typed view dominating all uses of the allocation.
    loom_value_id_t view;
  };
  // Declared array views retained once at the allocating producer.
  std::unordered_map<loom_value_id_t, ArrayView> array_views_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_STORAGE_H_
