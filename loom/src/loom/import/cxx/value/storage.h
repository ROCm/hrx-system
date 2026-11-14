// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_STORAGE_H_
#define LOOM_IMPORT_CXX_VALUE_STORAGE_H_

#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/ir/facts.h"

namespace loom::cxx_import {

struct ObjectRepresentation;

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
  // Whether this access relies on the source's defined-access promise because
  // its allocation extent is unavailable in the current function.
  bool root_bounds_promised = false;
};

struct StorageAllocation {
  // Pointer to the beginning of the declared allocation.
  Pointer pointer;
  // Allocation footprint view. Scalar objects and scalar arrays retain their
  // source element type; aggregate footprints use bytes and are projected into
  // typed field views at each access.
  loom_value_id_t view;
};

// Owns the source memory representation contract for one output module.
// Operands are already evaluated by the caller. Fixed array views are recorded
// at allocation and reused directly, including through source aliases.
// Boolean objects and Boolean-enum objects use byte views; loads and stores
// convert between those bytes and the i1 value representation.
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
  // Constructs absent storage with zero byte origin. Null owns no allocation.
  Pointer null(cxx::AST* owner);
  // Constructs the opaque null root for source-level typed buffer values.
  loom_value_id_t null_root(cxx::AST* owner);
  // Tests the root's null identity without observing its extent or contents.
  // The caller has already evaluated the source pointer expression.
  loom_value_id_t compare_null(loom_value_id_t root, bool equal,
                               cxx::AST* owner);
  // Tests opaque byte positions. Shared IR cleanup can use a common root
  // established after inlining; independent roots retain address equality.
  loom_value_id_t compare_address(Pointer left, Pointer right, bool equal,
                                  cxx::AST* owner);
  // Emits signed byte distance with a compiler-proven common-origin contract.
  // Shared cleanup establishes the proof after ordinary inlining and rejects
  // unresolved roots before target lowering; no native address is exposed.
  loom_value_id_t byte_distance(Pointer left, Pointer right,
                                loom_scalar_type_t result_type,
                                cxx::AST* owner);
  // Computes source element distance using the admitted pointee footprint,
  // then converts the signed result to the source ptrdiff_t representation.
  loom_value_id_t distance(Pointer left, Pointer right,
                           const cxx::Type* element_type,
                           const cxx::Type* result_type, cxx::AST* owner);
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
  StorageProjection member(StorageProjection base, cxx::Symbol* field,
                           cxx::AST* owner);
  // Projects a source-selected subobject from a live complete object. The
  // displacement is retained by Types from the frontend's selected path.
  StorageProjection subobject(StorageProjection base, uint64_t byte_offset,
                              cxx::AST* owner);
  // Adjusts a possibly-null class pointer to one of its nonvirtual base
  // subobjects. Null retains a zero origin; nonnull pointers retain their
  // allocation identity and add the source-layout displacement.
  Pointer subobject(Pointer base, uint64_t byte_offset, cxx::AST* owner);
  // Reverses a nonvirtual base projection. The source reference cast requires
  // an enclosing live object; this operation never adjusts a nullable pointer.
  StorageProjection enclosing(StorageProjection base, uint64_t byte_offset,
                              cxx::AST* owner);
  // Reverses a nonvirtual base-pointer projection while preserving null. A
  // valid nonnull source already denotes the selected base subobject.
  Pointer enclosing(Pointer base, uint64_t byte_offset, cxx::AST* owner);
  // Projects an in-bounds constant element of an admitted fixed array. Source
  // subobject offsets need no signed pointer-arithmetic conversion chain.
  StorageProjection element(StorageProjection base,
                            const cxx::BoundedArrayType* type, size_t index,
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
  // Reads the value representation of an already resolved scalar/vector
  // element. Reusing an access preserves its address across a read/modify/write
  // operation. The source element type supplies the footprint and memory
  // qualifiers.
  loom_value_id_t load(const StorageAccess& access,
                       const cxx::Type* element_type, cxx::AST* owner);
  // Writes to the same resolved location with the source element qualifiers.
  void store(const StorageAccess& access, loom_value_id_t value,
             const cxx::Type* element_type, cxx::AST* owner);
  // Allocates an admitted scalar, vector, record or fixed array using its
  // source layout and explicit alignment. Each execution creates a fresh root;
  // an optional exact object representation initializes the fresh storage.
  // The driver owns storage-duration and scope admission. Aggregate footprints
  // use bytes; Objects projects their value components independently of the
  // backing allocation.
  StorageAllocation allocate(const cxx::Type* type,
                             loom_value_fact_memory_space_t memory_space,
                             int64_t explicit_alignment, cxx::AST* owner,
                             const ObjectRepresentation* initializer = nullptr);
  // Materializes an anonymous immutable source object with its ABI alignment
  // and fixed-array projection. Equal objects may share executable storage.
  StorageAllocation constant(const cxx::Type* type,
                             const ObjectRepresentation& contents,
                             cxx::AST* owner);
  // Builds an exact byte payload in immutable executable storage or fresh
  // initialized frame storage. The caller establishes the source object
  // extent and alignment; the builder copies bytes into module-owned IR.
  loom_value_id_t literal(iree_const_byte_span_t contents, bool is_utf8,
                          loom_value_fact_memory_space_t memory_space,
                          int64_t alignment, cxx::AST* owner);
  // Admits an existing exact-size buffer root as one fixed source object.
  // The producer owns lifetime, access rights, and root alignment; this owner
  // retains the source layout, extent, and memory domain used by projections.
  StorageAllocation admit_fixed(loom_value_id_t root, const cxx::Type* type,
                                loom_value_fact_memory_space_t memory_space,
                                cxx::AST* owner);
  // Allocates a runtime number of admitted objects and returns its typed source
  // pointer representation. The logical element count remains distinct from
  // its physical byte stride until index.scale forms the allocation extent.
  // Dynamic allocations do not enter the fixed-declaration projection cache;
  // their root facts own the runtime extent.
  Pointer allocate_elements(const cxx::Type* element_type,
                            loom_value_id_t element_count,
                            const cxx::Type* element_count_type,
                            loom_value_fact_memory_space_t memory_space,
                            int64_t explicit_alignment, loom_value_id_t pool,
                            cxx::AST* owner);
  // Allocates an exact runtime byte extent already represented in Loom's
  // physical offset domain. No logical count conversion or scaling is added.
  Pointer allocate_bytes(loom_value_id_t byte_length,
                         loom_value_fact_memory_space_t memory_space,
                         int64_t alignment, loom_value_id_t pool,
                         cxx::AST* owner);
  // Returns the storage root of a pointer proven to have zero byte origin.
  // Interior pointers cannot cross an untyped buffer ABI boundary.
  loom_value_id_t zero_origin_root(Pointer pointer, cxx::AST* owner);
  // Refines one pointer root with a scoped no-alias identity while preserving
  // its byte origin and compiler-owned allocation facts.
  Pointer assume_noalias(Pointer pointer, cxx::AST* owner);
  // Refines a borrowed buffer root with an independent-root promise.
  loom_value_id_t assume_noalias(loom_value_id_t root, cxx::AST* owner);
  // Refines a zero-origin pointer root with a scoped minimum byte alignment.
  // Interior pointers cannot represent buffer-root alignment and are rejected.
  Pointer assume_alignment(Pointer pointer, int64_t minimum_alignment,
                           cxx::AST* owner);
  // Refines a borrowed zero-origin buffer root with a minimum byte alignment.
  loom_value_id_t assume_alignment(loom_value_id_t root,
                                   int64_t minimum_alignment, cxx::AST* owner);
  // Refines a borrowed buffer root's storage domain, preserving its identity
  // and compiler-owned allocation facts. This does not move or copy storage.
  loom_value_id_t assume_memory_space(
      loom_value_id_t root, loom_value_fact_memory_space_t memory_space,
      cxx::AST* owner);
  // Begins one source-language restrict scope over incoming buffer or view
  // paths. Restricted references carry distinct access associations; peers
  // carry the shared domain without becoming independent from one another.
  // Physical root identity and allocation facts remain unchanged.
  std::vector<loom_value_id_t> begin_restrict_scope(
      std::span<const loom_value_id_t> references, size_t restricted_count,
      cxx::AST* owner);
  // Removes restrict associations from references escaping their source scope.
  // Other physical and source-storage facts survive unchanged.
  std::vector<loom_value_id_t> end_restrict_scope(
      std::span<const loom_value_id_t> references, cxx::AST* owner);

 private:
  // Publishes the defined-source dereference range for a compiler-owned
  // allocation. Returns no value when the current function has no exact root
  // extent and the access must instead retain the source bounds promise.
  std::optional<Pointer> constrain_known_access(Pointer pointer,
                                                int64_t access_bytes,
                                                cxx::AST* owner);
  // Copies producer-owned storage facts to a root refinement.
  void inherit_root_facts(loom_value_id_t source, loom_value_id_t result);
  // Records one fixed allocation after its producer and source layout have
  // established the trusted root contract.
  StorageAllocation record_fixed(loom_value_id_t root, const cxx::Type* type,
                                 loom_value_fact_memory_space_t memory_space,
                                 int64_t byte_length,
                                 uint64_t natural_alignment, cxx::AST* owner);

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
  // Fixed allocation footprint established by its source declaration.
  struct Allocation {
    // Declared allocation extent in bytes.
    int64_t byte_length;
    // Source array type, or null for a scalar/vector or raw byte object.
    const cxx::Type* array_type;
    // Declared array origin, or invalid when no typed array view is retained.
    loom_value_id_t byte_offset;
    // Direct typed view for a scalar array; aggregate arrays use projections.
    std::optional<loom_value_id_t> array_view;
  };
  struct RootFacts {
    // Domain established by the allocating producer or a source promise.
    loom_value_fact_memory_space_t memory_space;
    // Fixed declaration footprint, absent for external or dynamic storage.
    std::optional<Allocation> allocation;
  };
  // Storage facts retained at their producer and inherited by root refinements.
  std::unordered_map<loom_value_id_t, RootFacts> roots_;
  // Offsets proven to be the zero origin of their buffer root. Refining
  // assumptions retain this identity without downstream producer inspection.
  std::unordered_set<loom_value_id_t> zero_origin_offsets_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_STORAGE_H_
