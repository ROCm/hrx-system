// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_PARTITION_H_
#define LOOM_IMPORT_CXX_VALUE_PARTITION_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <forward_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "loom/import/cxx/value/representation.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

// A runtime object handle, identified independently of any native layout.
struct ObjectPartition final : Partition {
  // Complete source wrapper retaining trivial-copy semantics.
  cxx::ClassSymbol* source;
  // Exact High reference ownership and imported payload identity.
  loom_type_t type;
};

// One direct subobject's position in an admitted record value. Component
// slices describe High value transport, independently of C++ object offsets.
struct SubobjectPartition {
  // Source properties remain owned by the frontend's direct declaration.
  const cxx::Type* type() const;
  uint64_t byte_offset() const;

  // Direct field or nonvirtual base declaration identity, borrowed from Source.
  cxx::Symbol* source;
  // Admitted member structure, owned by Types or a static leaf partition.
  const Partition* partition;
  // First component of this member in its containing record binding.
  size_t component_offset;
};

// One independently constructed source subobject. Dependent handles remain
// complete groups: a view's dimensions, encoding, and payload travel together.
struct ValueProjection {
  // Source type supplying the destination's dependent type scheme.
  const cxx::Type* type;
  // Complete semantic partition of this subobject.
  const Partition* partition;
  // First component relative to the complete enclosing source value.
  size_t component_offset;
  // Physical displacement when the enclosing object admits native storage.
  // Semantic handles and reference snapshots use only the component offset.
  uint64_t byte_offset = 0;
  // Passing this group transfers a managed owner. A C++ reference parameter
  // must receive the current owner back even when its referent is unchanged.
  bool carries_owner = false;
};

// Canonical source record structure, built once at admission. Members retain
// their own partitions so nested dependent values bind types from the current
// destination identities instead of caching another value's extents/layout.
struct RecordPartition final : Partition {
  // Canonical definition retaining nominal identity and source object layout.
  cxx::ClassSymbol* source;
  // Direct bases followed by fields, including zero-component empty records.
  std::vector<SubobjectPartition> members;
  // Retained member paths used to name newly bound High components.
  std::vector<std::string> component_names;
  // Whether source initialization may bind members without a constructor call.
  bool is_aggregate = false;
  // Whether any component needs destination identities to bind its High type.
  bool requires_binding = false;
  // Whether a const receiver may write a mutable field, including subobjects.
  bool has_mutable_subobject = false;
};

// One frontend-selected subobject path, retained once for shared use by source
// ownership analysis and emission. The path is relative to a complete record;
// its root distinguishes repeated instances of the same inherited declaration.
struct SubobjectProjection {
  // Direct subobject selected from the complete source record.
  const SubobjectPartition* root;
  // Structure of the final selected subobject, before an optional leaf field.
  const Partition* partition;
  // First component of the selected subobject in the complete record.
  size_t component_offset;
  // Physical displacement from the complete object, excluding its own origin.
  uint64_t byte_offset;
};

// Complete implicit-object selection for one member call. The frontend may
// expose the first class adjustment in the member's base expression while the
// remaining path to the declaring class stays implicit on the member.
struct MemberObjectSelection {
  // Complete object expression beneath a derived-to-base adjustment.
  cxx::ExpressionAST* object;
  // Selected receiver subobject, absent when the member belongs directly to
  // object. A base-to-derived cast remains in object and requires storage.
  std::optional<SubobjectProjection> projection;
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
  // Whether an element contains a field that remains writable through const.
  bool has_mutable_subobject;
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
// one immediately following builder call. Scalar-element rank-one and rank-two
// types retain dimensions inline and leave this storage empty.
class BoundTypeStorage {
 public:
  BoundTypeStorage() = default;
  BoundTypeStorage(const BoundTypeStorage&) = delete;
  BoundTypeStorage& operator=(const BoundTypeStorage&) = delete;
  BoundTypeStorage(BoundTypeStorage&&) = default;
  BoundTypeStorage& operator=(BoundTypeStorage&&) = default;

  // Rebuilds only a shaped type's dimensions while preserving its carrier,
  // element, alignment, and complete encoding attachment.
  loom_type_t with_dimensions(loom_type_t type,
                              std::span<const uint64_t> dimensions);
  // Attaches one optional encoding identity, rejecting IDs outside the shaped
  // type's representable range.
  static loom_type_t with_encoding(loom_type_t type,
                                   std::optional<loom_value_id_t> encoding);

  // Retains one bounded dimension sequence and returns stable contiguous
  // storage. The sequence must have rank in [1, LOOM_TYPE_MAX_RANK].
  const loom_overflow_dim_t* retain_dimensions(
      std::span<const loom_overflow_dim_t> dimensions);

 private:
  // Each node keeps one complete shaped-type payload at a stable address.
  std::forward_list<std::array<loom_overflow_dim_t, LOOM_TYPE_MAX_RANK>>
      dimension_payloads_;
};

// Static columnar storage for a source value represented entirely by managed
// references. Each flattened source component has one rank-one cell view;
// source records and arrays retain their ordinary component ordering.
struct ReferenceStorageLayout {
  // Complete source value reconstructed by loads from the component cells.
  const Partition* value_partition;
  // Exact managed-reference element types in flattened source order.
  std::vector<loom_type_t> element_types;
  // Rank-one cell-view types in the same order as |element_types|.
  std::vector<loom_type_t> cell_types;
  // Stable out-of-line reference-view payloads owned with this layout.
  BoundTypeStorage type_storage;
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

// Canonical rank-generic tile value. Dynamic extents and an optional encoding
// are transported before the dependent tile so its High type references this
// binding's own component identities at every call and region boundary.
struct TilePartition final : Partition {
  // Retains the fully static High type once during source admission.
  void bind_static_type();

  // Binds the High tile type to |dependencies|. Dependencies contain dynamic
  // extents in source-axis order followed by an optional encoding identity.
  loom_type_t bind_type(std::span<const loom_value_id_t> dependencies,
                        BoundTypeStorage& storage) const;
  // Returns the tile SSA value from one complete component binding.
  loom_value_id_t tile_value(std::span<const loom_value_id_t> components) const;
  bool requires_binding() const { return component_count > 1; }

  // Concrete source specialization retaining nominal copy semantics.
  cxx::ClassSymbol* source;
  // Unqualified source scalar element type.
  const cxx::Type* element_type;
  // High scalar representation of one logical element.
  loom_scalar_type_t element;
  // Attached semantic encoding, or null for an unencoded tile.
  const EncodingPartition* encoding = nullptr;
  // Static extent, or -1 when the axis has a transported dynamic extent.
  ShapeExtents extents;
  // Component labels in transport order; an empty label names the tile.
  std::vector<std::string> component_names;
  // Stable overflow storage for an all-static unencoded tile type.
  BoundTypeStorage static_type_storage;
  // Prebound all-static unencoded High type; unused by dependent tiles.
  loom_type_t static_type = {};
};

// Semantic identity of one channel-family source handle.
enum class ChannelHandleKind {
  Channel,
  Read,
  Write,
};

// Canonical channel, read, or write handle parameterized by a tile payload.
// Payload dependencies travel before the final handle and bind its nested tile
// type locally at every destination edge.
struct ChannelPartition final : Partition {
  // Retains the fully static High type once during source admission.
  void bind_static_type(loom_module_t* module);

  // Binds the parameterized High handle type to |dependencies|, which follow
  // the payload tile's dynamic-extent and optional-encoding order.
  loom_type_t bind_type(loom_module_t* module,
                        std::span<const loom_value_id_t> dependencies,
                        BoundTypeStorage& storage) const;
  // Returns the final handle SSA value from one complete component binding.
  loom_value_id_t handle_value(
      std::span<const loom_value_id_t> components) const;
  bool requires_binding() const { return component_count > 1; }

  // Concrete source specialization retaining nominal copy semantics.
  cxx::ClassSymbol* source;
  // Canonical tile payload shared with direct tile values.
  const TilePartition* payload;
  // Distinguishes communication, consumption, and production handles.
  ChannelHandleKind handle_kind;
  // Whether a read carries exclusive mutable permission; false otherwise.
  bool mutable_read = false;
  // Component labels in transport order; an empty label names the handle.
  std::vector<std::string> component_names;
  // Prebound parameterized High type; unused by dependent payloads.
  loom_type_t static_type = {};
};

// Canonical source handle projected to Loom's builtin allocation pool type.
struct PoolPartition final : Partition {
  // Concrete source class retaining copy and lifetime semantics.
  cxx::ClassSymbol* source;
};

// Canonical rank-generic source vector. Dynamic extents are transported before
// the dependent High vector so its type references this binding's own
// component identities at every call and region boundary. Native C++ vectors
// remain independent static rank-one source values.
struct VectorPartition final : Partition {
  // Retains the fully static High type once during source admission.
  void bind_static_type();

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

// One scalar or reference field of a columnar source view. Nested records and
// fixed arrays use the same field order as their ordinary value projection.
struct ViewColumn {
  // Complete High element, including imported reference identity.
  loom_type_t element;
  // Observation semantics inherited from the field and enclosing row.
  loom_memory_access_flags_t access_flags;
  // Whether a row store may write this field.
  bool writable;
  // Prebound all-static plain column type; unused by dependent views.
  loom_type_t static_type = {};
};

// Canonical rank-generic source view. Record elements project to separate
// scalar/reference columns sharing one set of dynamic extents. Optional column
// encodings precede the High views; calls and joins bind all dependent
// types to the destination's own component identities.
struct ViewPartition final : Partition {
  // Retains the fully static High type once during source admission.
  void bind_static_type();

  // Binds one High column to shared dynamic extents in source-axis order
  // followed by the optional per-column encoding identities.
  loom_type_t bind_type(std::span<const loom_value_id_t> dependencies,
                        BoundTypeStorage& storage, size_t column = 0) const;
  // Returns the view SSA value from one complete component binding.
  loom_value_id_t view_value(std::span<const loom_value_id_t> components,
                             size_t column = 0) const;
  // Whether the High type references a transported extent or encoding value.
  bool requires_binding() const { return dynamic_count != 0 || encoding; }
  bool is_columnar() const { return !columns.empty(); }
  size_t column_count() const { return columns.empty() ? 1 : columns.size(); }
  size_t column_component(size_t column) const {
    return dynamic_count + (encoding ? column_count() : 0) + column;
  }
  const loom_type_t& column_element(size_t column) const {
    return columns.empty() ? element : columns[column].element;
  }
  loom_memory_access_flags_t column_access_flags(size_t column) const {
    return columns.empty() ? access_flags : columns[column].access_flags;
  }

  // Concrete source specialization retaining nominal and cv-qualified rules.
  cxx::ClassSymbol* source;
  // Source element type, including cv qualifiers for access admission.
  const cxx::Type* element_type;
  // Primitive High element; absent for record elements represented by columns.
  loom_type_t element;
  // Observation semantics retained from the source element qualifier.
  loom_memory_access_flags_t access_flags;
  // Record field projections; empty for a primitive-element view.
  std::vector<ViewColumn> columns;
  // Number of shared dynamic dimensions at the start of each binding.
  size_t dynamic_count = 0;
  // Attached layout or physical-storage encoding, or null for a plain view.
  const EncodingPartition* encoding = nullptr;
  // Static extent, or -1 when the axis has a transported dynamic extent.
  ShapeExtents extents;
  // Component labels in transport order; an empty label names the view itself.
  std::vector<std::string> component_names;
  // Stable overflow storage for an all-static plain view type.
  BoundTypeStorage static_type_storage;
  // Prebound all-static plain High type; unused by dependent views.
  loom_type_t static_type = {};
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

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_PARTITION_H_
