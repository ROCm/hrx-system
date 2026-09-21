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

namespace loom::cxx_import {

// One typed access, either a dynamic element of a retained array view or a
// scalar or vector projection of a pointer's buffer.
struct StorageAccess {
  // Typed view containing the memory access.
  loom_value_id_t view;
  // Full-rank dynamic array index, absent for a scalar pointer projection.
  std::optional<loom_value_id_t> index;
};

struct StorageAllocation {
  // Pointer to the beginning of the declared workgroup allocation.
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

  // Forms the source pointer for a kernel buffer binding.
  Pointer root(loom_value_id_t buffer, cxx::AST* owner);
  // Computes an object-relative byte origin using the source integer's width
  // and signedness. Subtraction is represented by T_MINUS; addition by T_PLUS.
  // Only the final origin enters offset, allowing negative displacements from
  // interior pointers without forming negative offset values. Its nonnegative
  // range is published from the source language's within-object precondition.
  Pointer advance(Pointer base, loom_value_id_t displacement,
                  const cxx::Type* base_type, const cxx::Type* index_type,
                  cxx::TokenKind operation, cxx::AST* owner);
  // Constructs addressing for an integral subscript without narrowing pointer
  // byte arithmetic through target-selected index.
  // Unsupported source types are diagnosed at owner. The C++ driver separately
  // admits builtin indexing and evaluates base before index.
  StorageAccess subscript(Pointer base, loom_value_id_t index,
                          const cxx::Type* base_type,
                          const cxx::Type* subscript_type, cxx::AST* owner);
  // Projects an object's scalar lane footprint at the pointer's current origin.
  StorageAccess dereference(Pointer base, const cxx::Type* element_type,
                            cxx::AST* owner);
  // Reads an already resolved scalar/vector element. Reusing an
  // access preserves its address across a source read/modify/write operation.
  // The source element type supplies the footprint and memory qualifiers.
  loom_value_id_t load(const StorageAccess& access,
                       const cxx::Type* element_type, cxx::AST* owner);
  // Writes to the same resolved location with the source element qualifiers.
  void store(const StorageAccess& access, loom_value_id_t value,
             const cxx::Type* element_type, cxx::AST* owner);
  // Allocates a fixed workgroup scalar array using its source layout and any
  // explicit alignment. The driver admits the declaration's storage duration,
  // initialization and enclosing kernel contract before calling this method.
  StorageAllocation workgroup(const cxx::BoundedArrayType* array,
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
  // Declared array extents retained once at the allocating producer.
  std::unordered_map<loom_value_id_t, loom_value_id_t> array_views_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_STORAGE_H_
