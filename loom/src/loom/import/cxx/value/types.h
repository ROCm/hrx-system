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
#include <optional>
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
// one immediately following builder call. Rank-one and rank-two types retain
// dimensions inline and leave this storage empty.
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

// Canonical rank-generic tile value. Dynamic extents and an optional encoding
// are transported before the dependent tile so its High type references this
// binding's own component identities at every call and region boundary.
struct TilePartition final : Partition {
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

// Canonical rank-generic source view. Dynamic extents and an optional encoding
// are transported before the dependent High view so its type references this
// binding's own component identities at every call and region boundary.
struct ViewPartition final : Partition {
  // Binds the High view type to dynamic extents in source-axis order followed
  // by an optional encoding identity.
  loom_type_t bind_type(std::span<const loom_value_id_t> dependencies,
                        BoundTypeStorage& storage) const;
  // Returns the view SSA value from one complete component binding.
  loom_value_id_t view_value(std::span<const loom_value_id_t> components) const;
  // Whether the High type references a transported extent or encoding value.
  bool requires_binding() const { return component_count > 1; }

  // Concrete source specialization retaining nominal and cv-qualified rules.
  cxx::ClassSymbol* source;
  // Source element type, including cv qualifiers for access admission.
  const cxx::Type* element_type;
  // High scalar representation of one logical element.
  loom_scalar_type_t element;
  // Observation semantics retained from the source element qualifier.
  loom_memory_access_flags_t access_flags;
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
  // Qualifiers affect access and ownership, not this representation schema.
  const Partition& partition(const cxx::Type* input, cxx::AST* owner);
  // Observation semantics of an access through this cv-qualified object type.
  // Pointers pass their pointee type; the pointer binding's own qualifiers do
  // not affect the accessed object.
  loom_memory_access_flags_t memory_access_flags(const cxx::Type* input);
  // Returns whether projected component types reference destination SSA
  // identities. Such values must use append_bound() after reserving every
  // destination component ID in transport order.
  bool requires_binding(const cxx::Type* input, cxx::AST* owner);
  // Whether this source type contains handles without a native byte-storage
  // representation. Native objects retain their ordinary storage semantics.
  bool requires_value_transport(const cxx::Type* input);
  // Whether this object or an owned subobject is volatile. References and
  // pointers do not inherit the properties of the objects they designate.
  bool has_volatile_subobject(const cxx::Type* input);
  // Cached independently constructible groups, in source declaration order.
  std::span<const ValueProjection> projections(const cxx::Type* input,
                                               cxx::AST* owner);
  // Returns a record's admitted source schema, or null for a non-record type.
  const RecordPartition* record(const cxx::Type* input, cxx::AST* owner);
  // Whether a member call may replace value components of its receiver. Const
  // calls consume the record's retained transitive mutable-subobject fact.
  bool implicit_object_may_mutate(cxx::FunctionSymbol* function,
                                  cxx::AST* owner);
  // Whether this constructor executes an ordinary admitted record body. Loom
  // source wrapper constructors retain their intrinsic projection paths.
  bool record_constructor_requires_call(const cxx::Type* type,
                                        cxx::FunctionSymbol* constructor,
                                        cxx::AST* owner);
  // Whether the selected copy/move special member has no source-level body
  // effects. Ordinary overloads and destructors are never replaced by
  // component transport.
  bool is_trivial_value_operation(cxx::FunctionSymbol* function);
  // Returns a fixed array's value schema, or null for a non-array type.
  const ArrayPartition* array(const cxx::Type* input, cxx::AST* owner);
  // Direct lookup of a member slice retained by its owning record's admission.
  const SubobjectPartition& member(cxx::Symbol* subobject, cxx::AST* owner);
  // Projects a nonempty immutable path already selected by source semantics.
  // Direct fields use member() and require no independently retained path.
  const SubobjectProjection& projection(cxx::List<cxx::Symbol*>* path,
                                        cxx::AST* owner);
  // Physical-only access does not admit an SSA record partition: volatile
  // fields may be accessed in memory while never forming a by-value record.
  uint64_t subobject_offset(cxx::List<cxx::Symbol*>* path);
  // Includes the leaf field after a possibly empty frontend-selected path.
  SubobjectProjection selection(cxx::Symbol* field,
                                cxx::List<cxx::Symbol*>* path, cxx::AST* owner);
  // Returns the inherited-base or anonymous-member projection between a
  // frontend-converted member base and the class declaring the member. Direct
  // members require no object projection and return null.
  const SubobjectProjection* member_object_projection(
      cxx::MemberExpressionAST* member, cxx::AST* owner);
  // Combines an explicit derived-to-base adjustment on the member base with
  // the implicit path from that converted class to the declaration owner.
  MemberObjectSelection member_object_selection(
      cxx::MemberExpressionAST* member, cxx::AST* owner);
  // Returns only the source-layout displacement between a member's converted
  // object class and its declaring class. Physical storage access does not
  // admit the enclosing classes as SSA record values.
  uint64_t member_object_offset(cxx::MemberExpressionAST* member);
  // Selects a member's leaf field after its implicit object projection.
  SubobjectProjection selection(cxx::MemberExpressionAST* member,
                                cxx::AST* owner);
  // Selects value transport for a reference field whose referent remains
  // stable and whose identity is unobservable. Selection may precede record
  // admission when the ordinary reference-storage representation cannot carry
  // the referent at all.
  void select_reference_snapshot(cxx::FieldSymbol* field);
  // Whether this reference field transports the referent's value instead of a
  // pointer to its source object.
  bool reference_is_snapshot(cxx::FieldSymbol* field) const;
  // Source type represented by this reference field's selected value
  // partition: either the stable referent snapshot or an ordinary pointer to
  // the source object.
  const cxx::Type* reference_transport_type(cxx::FieldSymbol* field) const;
  // Records that a concrete closure use observes this reference capture's
  // source-object identity. Returns whether the shared fact was newly added.
  bool require_reference_capture_identity(cxx::FieldSymbol* field);
  // Whether any reached closure specialization observes this capture's source
  // object identity.
  bool reference_capture_requires_identity(cxx::FieldSymbol* field) const;
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
  // IDs. Static leaves ignore their IDs; dependent shaped values and handles
  // reference their own dynamic extents and encoding. |identities| is either
  // empty for a wholly static projection or exactly the partition's component
  // count. |storage| must outlive the builder call consuming the returned
  // types.
  void append_bound(const cxx::Type* input, cxx::AST* owner,
                    std::span<const loom_value_id_t> identities,
                    BoundTypeStorage& storage,
                    std::vector<loom_type_t>& output);
  // Removes top-level source qualifiers and typedef alignment sugar while
  // retaining any reference wrapper. Restrict remains an access-path contract
  // consumed before structural type admission, not part of a High value type.
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
  struct ObjectProperties {
    // Whether any owned component lacks a native byte-storage representation.
    bool value_transport = false;
    // Whether an owned subobject has observable volatile accesses.
    bool volatile_subobject = false;
  };
  ObjectProperties object_properties(const cxx::Type* input);
  void require_record_storage(const cxx::ClassType* input, cxx::AST* owner);
  const Partition* special(const cxx::Type* input, cxx::AST* owner);
  const BufferPartition* buffer(const cxx::ClassType* input, cxx::AST* owner);
  const AddressPartition* address(const cxx::ClassType* input,
                                  loom_scalar_type_t element_type,
                                  cxx::AST* owner);
  const EncodingPartition* encoding(const cxx::ClassType* input,
                                    cxx::AST* owner);
  const PoolPartition* pool(const cxx::ClassType* input, cxx::AST* owner);
  const TilePartition* tile(const cxx::ClassType* input, cxx::AST* owner);
  const ChannelPartition* channel(const cxx::ClassType* input,
                                  std::string_view name, cxx::AST* owner);
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
  struct MemberObjectPath {
    // Nonvirtual base declarations from the converted object to the class
    // that declares the selected member.
    std::vector<cxx::Symbol*> symbols;
    // Complete physical displacement of symbols in source object layout.
    uint64_t byte_offset = 0;
    // Lazily admitted SSA partition path, absent for storage-only access.
    std::optional<SubobjectProjection> projection;
  };
  MemberObjectPath* member_object_path(cxx::MemberExpressionAST* member);

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
  // Admitted allocation pool handles keyed by their source class.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<PoolPartition>> pools_;
  // Admitted tile specializations retaining element and extent contracts.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<TilePartition>> tiles_;
  // Admitted channel-family handles retaining their payload contracts.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<ChannelPartition>>
      channels_;
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
  std::unordered_map<cxx::Symbol*, SubobjectPartition> members_;
  using ProjectionMap =
      std::unordered_map<cxx::List<cxx::Symbol*>*, SubobjectProjection>;
  // Present only when source access selects a nested class subobject path.
  // A null partition retains a physical-only path; projection() completes the
  // value facts before exposing the entry to its SSA consumers.
  std::unique_ptr<ProjectionMap> projections_;
  using MemberObjectPathMap = std::unordered_map<
      cxx::ClassSymbol*,
      std::unordered_map<cxx::ClassSymbol*, MemberObjectPath>>;
  // Sparse paths keyed first by the converted object class and then by the
  // selected member's declaring class. Physical access retains only layout;
  // SSA consumers admit the semantic projection lazily. Direct members
  // allocate nothing.
  std::unique_ptr<MemberObjectPathMap> member_object_paths_;
  // Reference fields selected to transport their referent value. A provisional
  // selection that later observes identity is rejected before translation.
  std::unordered_set<cxx::FieldSymbol*> reference_snapshots_;
  // Reference capture fields whose source object identity is observed by a
  // reached closure specialization.
  std::unordered_set<cxx::FieldSymbol*> identity_reference_captures_;
  // Completed memory admission; layouts remain owned by the source symbols.
  std::unordered_set<cxx::ClassSymbol*> storage_records_;
  // Source representation classification, independent of SSA admission.
  std::unordered_map<const cxx::Type*, ObjectProperties> object_properties_;
  // Flattened construction groups computed once by source type admission.
  std::unordered_map<const cxx::Type*, std::vector<ValueProjection>>
      construction_groups_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_TYPES_H_
