// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_TYPES_H_
#define LOOM_IMPORT_CXX_VALUE_TYPES_H_

#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <forward_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

// One source member's retained position in an admitted record value. Component
// slices describe High value transport, independently of C++ object offsets.
struct MemberPartition {
  // Declaration identity and cv-qualified source type, borrowed from Source.
  cxx::FieldSymbol* field;
  // Admitted member structure, owned by Types or a static leaf partition.
  const Partition* partition;
  // First component of this member in its containing record binding.
  size_t component_offset;
};

// Canonical source record structure, built once at admission. Members retain
// their own partitions so nested dependent values bind types from the current
// destination identities instead of caching another value's extents/layout.
struct RecordPartition final : Partition {
  // Canonical definition retaining nominal identity and source object layout.
  cxx::ClassSymbol* source;
  // Declaration-order members, including zero-component empty records.
  std::vector<MemberPartition> members;
  // Retained member paths used to name newly bound High components.
  std::vector<std::string> component_names;
  // Whether any component needs destination identities to bind its High type.
  bool requires_binding = false;
};

// A fixed source array transported as immutable element components inside an
// enclosing value. This is not array decay: borrowing an array's address uses
// its source-layout object storage instead of the value partition.
struct ArrayPartition final : Partition {
  // Canonical bounded source type, including its element layout and extent.
  const cxx::BoundedArrayType* source;
  // Retained recursive element structure, shared by all positions.
  const Partition* element;
  // Whether element types depend on destination SSA identities.
  bool requires_binding;
};

// Canonical typed source handle projected to one zero-origin High buffer. The
// scalar or vector object type constrains source binding and view construction
// but is not carried by Loom's untyped buffer value.
struct BufferPartition final : Partition {
  // Concrete source specialization retaining copy and lifetime semantics.
  cxx::ClassSymbol* source;
  // Source element type, including qualifiers used by typed views.
  const cxx::Type* element_type;
};

// Canonical source handle projected to one Loom address-domain scalar. Index
// and offset remain nominally distinct in C++ and retain their independent
// target-selected carriers through every value boundary.
struct AddressPartition final : Partition {
  // Concrete source class retaining copy and conversion semantics.
  cxx::ClassSymbol* source;
  // Native address domain represented by this source class.
  loom_scalar_type_t element_type;
};

// Canonical source encoding object projected to one first-class High encoding
// value. Rank remains a source type refinement used to reject mismatched view
// construction; the High encoding role owns the runtime semantic type.
struct EncodingPartition final : Partition {
  // Concrete source specialization retaining copy and object-layout semantics.
  cxx::ClassSymbol* source;
  // Semantic role preserved across calls, records, and structured results.
  loom_encoding_role_t role;
  // Number of layout/storage axes, or zero for a schema/transform.
  size_t rank;
};

// Canonical source handle projected to one registered opaque dialect value.
// The source class supplies C++ naming and copy semantics while the dialect
// descriptor owns the native semantic type.
struct OpaqueDialectPartition final : Partition {
  // Concrete source class retaining copy and lifetime semantics.
  cxx::ClassSymbol* source;
  // Registered opaque dialect type interned into the destination module.
  loom_type_t type;
};

// Owns temporary overflow dimensions while bound High types are consumed by
// one immediately following builder call. Rank-one and rank-two types retain
// dimensions inline and leave this storage empty.
class BoundTypeStorage {
 public:
  BoundTypeStorage() = default;
  BoundTypeStorage(const BoundTypeStorage&) = delete;
  BoundTypeStorage& operator=(const BoundTypeStorage&) = delete;
  BoundTypeStorage(BoundTypeStorage&&) = default;
  BoundTypeStorage& operator=(BoundTypeStorage&&) = default;

  // Retains one bounded dimension sequence and returns stable contiguous
  // storage. The sequence must have rank in [3, LOOM_TYPE_MAX_RANK].
  const loom_overflow_dim_t* retain_dimensions(
      std::span<const loom_overflow_dim_t> dimensions);

 private:
  // Each node keeps one complete shaped-type payload at a stable address.
  std::forward_list<std::array<loom_overflow_dim_t, LOOM_TYPE_MAX_RANK>>
      dimension_payloads_;
};

// Bounded shaped-value extents with the former rank-two representation inline.
// Higher ranks allocate one exact overflow array when their source type is
// admitted.
class ShapeExtents {
 public:
  ShapeExtents() = default;
  ShapeExtents(const ShapeExtents&) = delete;
  ShapeExtents& operator=(const ShapeExtents&) = delete;
  ShapeExtents(ShapeExtents&&) = default;
  ShapeExtents& operator=(ShapeExtents&&) = default;

  // Retains |extents| for the lifetime of this projection.
  void assign(std::span<const int64_t> extents);

  size_t size() const { return rank_; }
  const int64_t* begin() const { return data(); }
  const int64_t* end() const { return data() + rank_; }
  int64_t operator[](size_t axis) const { return data()[axis]; }

 private:
  const int64_t* data() const {
    return rank_ <= inline_extents_.size() ? inline_extents_.data()
                                           : overflow_extents_.get();
  }

  // Number of retained logical axes.
  size_t rank_ = 0;
  // Rank-one and rank-two extents requiring no independent allocation.
  std::array<int64_t, 2> inline_extents_ = {};
  // Exact extent array for ranks above two; null for inline ranks.
  std::unique_ptr<int64_t[]> overflow_extents_;
};

// Canonical rank-generic source vector. Dynamic extents are transported before
// the dependent High vector so its type references this binding's own
// component identities at every call and region boundary. Native C++ vectors
// remain independent static rank-one source values.
struct VectorPartition final : Partition {
  // Binds the High vector type to |dynamic_extents| in source-axis order.
  // Static vectors require an empty span.
  loom_type_t bind_type(std::span<const loom_value_id_t> dynamic_extents,
                        BoundTypeStorage& storage) const;
  // Returns the vector SSA value from one complete component binding.
  loom_value_id_t vector_value(
      std::span<const loom_value_id_t> components) const;
  bool is_dynamic() const { return component_count > 1; }

  // Concrete source specialization retaining nominal copy semantics.
  cxx::ClassSymbol* source;
  // Unqualified source scalar lane type.
  const cxx::Type* element_type;
  // High scalar representation of one logical lane.
  loom_scalar_type_t element;
  // Static extent, or -1 when the axis has a transported dynamic extent.
  ShapeExtents extents;
  // Component labels in transport order; an empty label names the vector.
  std::vector<std::string> component_names;
  // Stable overflow storage for an all-static vector type.
  BoundTypeStorage static_type_storage;
  // Prebound all-static High type; unused by dependent vectors.
  loom_type_t static_type = {};
};

// Canonical rank-generic source view. Dynamic extents and its encoding are
// transported before the dependent High view so its type references this
// binding's own component identities at every call and region boundary.
struct ViewPartition final : Partition {
  // Concrete source specialization retaining nominal and cv-qualified rules.
  cxx::ClassSymbol* source;
  // Source element type, including cv qualifiers for access admission.
  const cxx::Type* element_type;
  // High scalar representation of one logical element.
  loom_scalar_type_t element;
  // Observation semantics retained from the source element qualifier.
  loom_memory_access_flags_t access_flags;
  // Semantic role of the attached layout or physical-storage encoding.
  loom_encoding_role_t encoding_role;
  // Static extent, or -1 when the axis has a transported dynamic extent.
  ShapeExtents extents;
  // Component labels in transport order; an empty label names the view itself.
  std::vector<std::string> component_names;
};

// Canonical dense rank-one source tensor handle. Scalar objects project to one
// High tensor axis. Vector objects append their logical lane axis so the High
// tensor retains both the source object count and every scalar lane. A dynamic
// object extent is transported before the dependent High tensor so its type
// references this binding's own dimension identity across value boundaries.
struct TensorPartition final : Partition {
  // Binds the High tensor type to |dynamic_extents|. Static tensors require an
  // empty span; dynamic tensors require their one transported dimension.
  loom_type_t bind_type(std::span<const loom_value_id_t> dynamic_extents) const;
  // Returns the tensor handle from one complete component binding.
  loom_value_id_t tensor_value(
      std::span<const loom_value_id_t> components) const;
  bool is_dynamic() const { return extent < 0; }

  // Concrete source specialization retaining copy and lifetime semantics.
  cxx::ClassSymbol* source;
  // Source element type used for generation payloads and scalar admission.
  const cxx::Type* element_type;
  // Source scalar lane type; identical to element_type for scalar objects.
  const cxx::Type* lane_type;
  // High scalar representation of one logical lane.
  loom_scalar_type_t element;
  // Number of scalar lanes in a vector object, or zero for a scalar object.
  uint64_t vector_lane_count;
  // Static source object count, or -1 for one transported dynamic extent.
  int64_t extent;
  // Source element byte width used to translate element origins to bytes.
  uint64_t element_bytes;
};

// Projects resolved C++ types using the translation unit's explicit data model.
// Source signedness remains available even when both types share one IR
// carrier. Enums use their resolved underlying integer representation; nominal
// source types remain intact for C++ semantic queries. The source unit and
// diagnostics outlive this projection.
class Types {
 public:
  Types(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
        loom_module_t* module = nullptr)
      : unit_(unit), diagnostics_(diagnostics), module_(module) {}

  // Projects a leaf value's representation independently of top-level cv
  // qualifiers. Object pointers carry a buffer and byte offset independently
  // of pointee layout; storage_size() admits actual object projections.
  // Aggregate values use partition(). Access semantics belong to
  // memory_access_flags().
  // Unsupported representations diagnose at owner and throw SourceRejected.
  loom_type_t get(const cxx::Type* input, cxx::AST* owner);
  // Admits an addressable scalar, vector, plain record or fixed array of those
  // elements and returns its source-owned byte footprint. Memory admission is
  // independent of the SSA partition: projection does not copy an aggregate.
  int64_t storage_size(const cxx::Type* input, cxx::AST* owner);
  // Admits a source value and returns its stable, identity-free partition.
  // Leaf carriers are static; admitted records are owned by this Types object.
  // Volatile objects cannot use SSA-only transport without addressable storage.
  const Partition& partition(const cxx::Type* input, cxx::AST* owner);
  // Observation semantics of an access through this cv-qualified object type.
  // Pointers pass their pointee type; the pointer binding's own qualifiers do
  // not affect the accessed object.
  loom_memory_access_flags_t memory_access_flags(const cxx::Type* input);
  // Returns whether projected component types reference destination SSA
  // identities. Such values must use append_bound() after reserving every
  // destination component ID in transport order.
  bool requires_binding(const cxx::Type* input, cxx::AST* owner);
  // Returns a record's admitted source schema, or null for a non-record type.
  const RecordPartition* record(const cxx::Type* input, cxx::AST* owner);
  // Returns a fixed array's value schema, or null for a non-array type.
  const ArrayPartition* array(const cxx::Type* input, cxx::AST* owner);
  // Direct lookup of a member slice retained by its owning record's admission.
  const MemberPartition& member(cxx::FieldSymbol* field, cxx::AST* owner);
  // Admits a resolved constructor when the source partition represents a
  // trivial copy or move, or when an admitted record supplies the default
  // member-initialization plan. A null constructor requires no lifecycle
  // action.
  void admit_construction(cxx::FunctionSymbol* constructor,
                          const cxx::Type* type, cxx::AST* owner);
  // Admits mutation of the source object before projection removes qualifiers.
  // A const pointer binding is immutable; a pointer to const has an immutable
  // pointee but the binding itself may still change.
  void require_mutable(const cxx::Type* input, cxx::AST* owner);
  // Appends today's admitted static High signature for an ordinary function
  // or structured result. A dependent source value requires destination SSA
  // identities before type binding and cannot use this static projection.
  // Kernel parameters instead use get(), retaining the launch binding ABI.
  void append(const cxx::Type* input, cxx::AST* owner,
              std::vector<loom_type_t>& output);
  // Appends a High signature bound to this destination's flattened component
  // IDs. Static leaves ignore their IDs; dependent views reference their own
  // dynamic extents and encoding. |identities| is either empty for a wholly
  // static projection or exactly the partition's component count. |storage|
  // must outlive the builder call that consumes the returned types.
  void append_bound(const cxx::Type* input, cxx::AST* owner,
                    std::span<const loom_value_id_t> identities,
                    BoundTypeStorage& storage,
                    std::vector<loom_type_t>& output);
  // Removes top-level cv-qualification and typedef alignment sugar while
  // retaining any reference wrapper.
  const cxx::Type* unqualified(const cxx::Type* type);
  // Returns the unqualified object type denoted by a value or reference.
  const cxx::Type* unqualified_value(const cxx::Type* type);
  // Returns the resolved native C++ vector representation, or null for a
  // scalar or class-backed semantic vector.
  const cxx::VectorType* native_vector(const cxx::Type* type);
  // Returns whether a native vector's source object contains trailing bytes
  // outside its logical lanes. Unknown or inconsistent source layouts reject
  // at owner. Value operations never observe these bytes; storage_size()
  // retains them for allocation and address arithmetic.
  bool has_vector_object_padding(const cxx::Type* type, cxx::AST* owner);
  bool is_unsigned(const cxx::Type* type);
  bool is_float(const cxx::Type* type);
  // Returns the native address domain represented by a source class, or
  // LOOM_SCALAR_TYPE_NONE for an ordinary C++ value.
  loom_scalar_type_t address_domain(const cxx::Type* type, cxx::AST* owner);
  // Whether the source class projects directly to Loom's native index scalar.
  bool is_index(const cxx::Type* type, cxx::AST* owner);
  // Whether the source class projects directly to Loom's native offset scalar.
  bool is_offset(const cxx::Type* type, cxx::AST* owner);
  // Whether the source class names this exact registered opaque dialect type.
  bool is_opaque_dialect(const cxx::Type* type, std::string_view name,
                         cxx::AST* owner);

 private:
  void require_record_storage(const cxx::ClassType* input, cxx::AST* owner);
  const Partition* special(const cxx::Type* input, cxx::AST* owner);
  const BufferPartition* buffer(const cxx::ClassType* input, cxx::AST* owner);
  const AddressPartition* address(const cxx::ClassType* input,
                                  loom_scalar_type_t element_type,
                                  cxx::AST* owner);
  const EncodingPartition* encoding(const cxx::ClassType* input,
                                    cxx::AST* owner);
  const OpaqueDialectPartition* opaque_dialect(const cxx::ClassType* input,
                                               std::string_view name,
                                               cxx::AST* owner);
  ShapeExtents parse_shape(const cxx::Type* input, std::string_view value_kind,
                           cxx::AST* owner);
  const VectorPartition* semantic_vector(const cxx::ClassType* input,
                                         cxx::AST* owner);
  const ViewPartition* view(const cxx::ClassType* input, cxx::AST* owner);
  const TensorPartition* tensor(const cxx::ClassType* input, cxx::AST* owner);
  void append_component_names(const Partition& partition,
                              std::string_view prefix,
                              std::vector<std::string>& output);

  // Resolved source traits and configured memory layout.
  cxx::TranslationUnit& unit_;
  // Source rejection boundary for unsupported types.
  Diagnostics& diagnostics_;
  // Destination owning interned dialect type names; null in scalar-only tests.
  loom_module_t* module_;
  // Stable source partitions, independent of every particular SSA binding.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<RecordPartition>>
      records_;
  // Admitted typed buffer handles keyed by concrete source specialization.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<BufferPartition>>
      buffers_;
  // Admitted native address handles keyed by their source class.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<AddressPartition>>
      addresses_;
  // Array schemas retain the element partition once, independently of length.
  std::unordered_map<const cxx::BoundedArrayType*,
                     std::unique_ptr<ArrayPartition>>
      arrays_;
  // Admitted special source objects, keyed by concrete specialization.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<EncodingPartition>>
      encodings_;
  // Admitted opaque dialect handles keyed by concrete source class.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<OpaqueDialectPartition>>
      opaque_dialects_;
  // Admitted class-backed vector specializations and their extent contracts.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<VectorPartition>>
      semantic_vectors_;
  // Admitted view specializations retaining element and extent contracts.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<ViewPartition>> views_;
  // Admitted tensor handles retaining their element and static extent.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<TensorPartition>>
      tensors_;
  // Member identity indexes the slice established by record admission.
  std::unordered_map<cxx::FieldSymbol*, MemberPartition> members_;
  // Completed memory admission; layouts remain owned by the source symbols.
  std::unordered_set<cxx::ClassSymbol*> storage_records_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_TYPES_H_
