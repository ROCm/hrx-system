// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_TYPES_H_
#define LOOM_IMPORT_CXX_VALUE_TYPES_H_

#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loom/import/cxx/value/partition.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Diagnostics;

// Retained projection of an ordinary record onto explicit kernel arguments.
// Native object padding and pointer displacements are absent from this ABI:
// each pointer contributes a buffer whose source displacement starts at zero.
struct KernelRecordProjection {
  // A source pointer displacement supplied by the entry instead of a caller.
  static constexpr size_t kZeroOrigin = SIZE_MAX;

  // High kernel argument types in base/member/array-element order.
  std::vector<loom_type_t> types;
  // One native argument ordinal per source value component, or kZeroOrigin
  // for the displacement immediately following a pointer's buffer component.
  std::vector<size_t> argument_indices;
};

// Kernel entry transport of one flattened source component. Kernel launch
// arguments carry storage as untyped zero-origin buffers, so a pointer's byte
// offset is supplied by the entry instead of the launch.
enum class KernelComponent : uint8_t {
  // Transported unchanged as one kernel argument.
  Value,
  // Pointer storage root transported as one kernel buffer argument.
  PointerRoot,
  // Pointer byte offset reconstructed as zero at kernel entry.
  PointerOffset,
};

// One flattened component of a kernel parameter with the source subobject that
// owns it. A pointer's root and offset components share one leaf subobject.
struct KernelLeaf {
  // Kernel transport of this component.
  KernelComponent component;
  // Source type of the owning leaf subobject (a scalar, vector, or pointer).
  const cxx::Type* type;
  // Byte offset of the owning leaf subobject within the host parameter object.
  uint64_t byte_offset;
};

// Projects resolved C++ types using the translation unit's explicit data model.
// Source signedness remains available even when both types share one IR
// carrier. Enums use their resolved underlying integer representation; nominal
// source types remain intact for C++ semantic queries. The source unit and
// diagnostics outlive this projection.
class Types {
 public:
  Types(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
        loom_module_t* module = nullptr);
  ~Types();
  Types(const Types&) = delete;
  Types& operator=(const Types&) = delete;

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
  // Observation semantics of an access through this source object type.
  // Pointers pass their pointee type; the pointer binding's own qualifiers do
  // not affect the accessed object. C observations and C++ character, byte,
  // may_alias, aggregate, and otherwise type-universal objects retain
  // universal aliasing.
  loom_memory_access_flags_t memory_access_flags(const cxx::Type* input);
  // Returns whether projected component types reference destination SSA
  // identities. Such values must use append_bound() after reserving every
  // destination component ID in transport order.
  bool requires_binding(const cxx::Type* input, cxx::AST* owner);
  // Whether this source type contains handles without a native byte-storage
  // representation. Native objects retain their ordinary storage semantics.
  bool requires_value_transport(const cxx::Type* input);
  // Finds the independently addressable member containing a borrowed interval
  // in a value-transported aggregate. Keeping a whole numeric array together
  // preserves pointer arithmetic from an interior element.
  std::optional<ValueProjection> storage_member(const cxx::Type* input,
                                                size_t begin, size_t end,
                                                cxx::AST* owner);
  // Source-layout displacement of a typed subobject within an admitted
  // numeric storage member. Same-component nested records may have different
  // addresses, so component position alone does not identify a byte offset.
  std::optional<uint64_t> member_displacement(ValueProjection group,
                                              size_t component,
                                              const cxx::Type* type);
  // Returns columnar reference-cell storage when every flattened component is
  // a managed ref or ptr. Other semantic values have no such representation.
  const ReferenceStorageLayout* reference_storage(const cxx::Type* input,
                                                  cxx::AST* owner);
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
  // Kernel parameters instead use append_kernel_parameter().
  void append(const cxx::Type* input, cxx::AST* owner,
              std::vector<loom_type_t>& output);
  // Appends the launch binding ABI of one kernel parameter. Leaf parameters
  // keep their single High type. Records (including lambda closures) are
  // flattened in member declaration order; each pointer member contributes
  // one buffer and no byte offset. Record members must be scalars, native
  // vectors, typed buffers, index/offset values, object pointers, or nested
  // records and fixed arrays of those; references and dependent shaped values
  // diagnose.
  void append_kernel_parameter(const cxx::Type* input, cxx::AST* owner,
                               std::vector<loom_type_t>& output);
  // Returns a record's cached kernel projection, or null for a non-record.
  // Admission walks source partitions once; entry construction and predicates
  // share the retained source-to-argument mapping.
  const KernelRecordProjection* kernel_record(const cxx::Type* input,
                                              cxx::AST* owner);
  // Returns the kernel transport of every flattened component of a record
  // kernel parameter, in partition order. Diagnoses as
  // append_kernel_parameter() does.
  std::vector<KernelComponent> kernel_components(const cxx::Type* input,
                                                 cxx::AST* owner);
  // Returns every flattened component of a kernel parameter with its owning
  // leaf subobject's source type and host byte offset, in partition order. The
  // host packs record kernel arguments from these offsets.
  std::vector<KernelLeaf> kernel_leaves(const cxx::Type* input,
                                        cxx::AST* owner);
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

  // Establishes an admitted incomplete payload's exact module-local identity.
  // Called in source order before admitting any callable source types.
  void declare_object_import(cxx::ClassSymbol* source,
                             std::string_view namespace_name,
                             std::string_view payload_name,
                             std::string_view declaration_name);

  // Exact borrowed reference type for one admitted incomplete payload. This
  // bypasses any source proxy layout while preserving nominal object identity.
  loom_type_t borrowed_object_type(cxx::ClassSymbol* payload,
                                   cxx::AST* owner) const;

 private:
  std::optional<ValueProjection> component_child(ValueProjection group,
                                                 size_t component);

  struct ObjectProperties {
    // Whether any owned component lacks a native byte-storage representation.
    bool value_transport = false;
    // Whether an owned subobject has observable volatile accesses.
    bool volatile_subobject = false;
  };
  ObjectProperties object_properties(const cxx::Type* input);
  void require_record_storage(const cxx::ClassType* input, cxx::AST* owner);
  const Partition* special(const cxx::Type* input, cxx::AST* owner);
  const ObjectPartition* object(const cxx::ClassType* input,
                                loom_reference_ownership_t ownership,
                                cxx::AST* owner);
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
  // Admits each scalar/reference row field once, retaining its qualifiers.
  void append_view_columns(const cxx::Type* type, bool writable,
                           loom_memory_access_flags_t access_flags,
                           cxx::AST* owner, std::vector<ViewColumn>& columns);
  const TensorPartition* tensor(const cxx::ClassType* input, cxx::AST* owner);
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
  // Exact imported payload declarations, keyed by canonical incomplete class.
  std::unordered_map<cxx::ClassSymbol*, loom_type_t> object_imports_;
  // Concrete borrowed wrappers, retaining their payload's exact High type.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<ObjectPartition>>
      objects_;
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
  using KernelRecordMap =
      std::unordered_map<const RecordPartition*, KernelRecordProjection>;
  // Allocated only when a record crosses a kernel entry boundary. Ordinary
  // callables and scalar/pointer kernel parameters allocate no projection.
  std::unique_ptr<KernelRecordMap> kernel_records_;
  // Reference-only source values admitted to typed cell storage. Null entries
  // memoize source values containing a non-reference component.
  std::unordered_map<const cxx::Type*, std::unique_ptr<ReferenceStorageLayout>>
      reference_storage_layouts_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_TYPES_H_
