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
};

// Canonical source encoding object projected to one first-class High encoding
// value. Rank remains a source type refinement used to reject mismatched view
// construction; the High encoding role owns the runtime semantic type.
struct EncodingPartition final : Partition {
  // Concrete source specialization retaining copy and object-layout semantics.
  cxx::ClassSymbol* source;
  // Number of axes described by this source layout object.
  size_t rank;
};

// Canonical rank-two source view. Dynamic extents and the address layout are
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
  // Static extent, or -1 when the axis has a transported dynamic extent.
  std::array<int64_t, 2> extents;
  // Component labels in transport order; an empty label names the view itself.
  std::vector<std::string> component_names;
};

// Projects resolved C++ types using the translation unit's explicit data model.
// Source signedness remains available even when both types share one IR
// carrier. Enums use their resolved underlying integer representation; nominal
// source types remain intact for C++ semantic queries. The source unit and
// diagnostics outlive this projection.
class Types {
 public:
  Types(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Projects a loaded value's representation independently of top-level cv
  // qualifiers. Object storage admission belongs to partition(); access
  // semantics belong to memory_access_flags(). Unsupported representations
  // diagnose at owner and throw SourceRejected.
  loom_type_t get(const cxx::Type* input, cxx::AST* owner);
  // Admits an addressable scalar, vector or plain record and returns its
  // source-owned byte footprint. Memory admission is independent of the SSA
  // partition: field projection does not load or copy a complete record.
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
  // Direct lookup of a member slice retained by its owning record's admission.
  const MemberPartition& member(cxx::FieldSymbol* field, cxx::AST* owner);
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
  // dynamic extents and layout. |identities| is either empty for a wholly
  // static projection or exactly the partition's component count.
  void append_bound(const cxx::Type* input, cxx::AST* owner,
                    std::span<const loom_value_id_t> identities,
                    std::vector<loom_type_t>& output);
  const cxx::Type* unqualified(const cxx::Type* type);
  // Returns the resolved vector representation, or null for a scalar/object.
  const cxx::VectorType* vector(const cxx::Type* type);
  bool is_unsigned(const cxx::Type* type);
  bool is_float(const cxx::Type* type);

 private:
  void require_record_storage(const cxx::ClassType* input, cxx::AST* owner);
  const Partition* special(const cxx::Type* input, cxx::AST* owner);
  const EncodingPartition* encoding(const cxx::ClassType* input,
                                    cxx::AST* owner);
  const ViewPartition* view(const cxx::ClassType* input, cxx::AST* owner);
  void append_component_names(const Partition& partition,
                              std::string_view prefix,
                              std::vector<std::string>& output);

  // Resolved source traits and configured memory layout.
  cxx::TranslationUnit& unit_;
  // Source rejection boundary for unsupported types.
  Diagnostics& diagnostics_;
  // Stable source partitions, independent of every particular SSA binding.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<RecordPartition>>
      records_;
  // Admitted special source objects, keyed by concrete specialization.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<EncodingPartition>>
      encodings_;
  // Admitted view specializations retaining element and extent contracts.
  std::unordered_map<cxx::ClassSymbol*, std::unique_ptr<ViewPartition>> views_;
  // Member identity indexes the slice established by record admission.
  std::unordered_map<cxx::FieldSymbol*, MemberPartition> members_;
  // Completed memory admission; layouts remain owned by the source symbols.
  std::unordered_set<cxx::ClassSymbol*> storage_records_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_TYPES_H_
