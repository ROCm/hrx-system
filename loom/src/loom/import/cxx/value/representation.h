// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_REPRESENTATION_H_
#define LOOM_IMPORT_CXX_VALUE_REPRESENTATION_H_

#include <memory_resource>
#include <span>
#include <vector>

#include "loom/ir/types.h"

namespace loom::cxx_import {

// A source pointer retains an opaque supplied origin separately from its
// signed byte displacement. Neither component exposes a native address.
struct Pointer {
  // Opaque buffer origin, which may itself denote an interior native pointer.
  loom_value_id_t root;
  // Signed index displacement from that origin, including one-past positions.
  loom_value_id_t byte_offset;
};

// Source identity determines the operations available on a value independently
// of its number of High components. A two-scalar record is not a pointer.
enum class ValueKind {
  SSA,
  Pointer,
  Record,
  Array,
  Buffer,
  Address,
  Object,
  OpaqueDialect,
  Encoding,
  Pool,
  Tile,
  Channel,
  Vector,
  View,
  Tensor
};

// Admitted source structure shared by all bindings of that structure. A
// partition contains no SSA identities or dependent types: each Value supplies
// its own component bindings. Source-specific partitions may retain nominal
// identity and member slices while source admission remains alive.
struct Partition {
  // Semantic category, independent of the flattened component count.
  ValueKind kind;
  // Number of High values transported at an ordinary call or region edge.
  size_t component_count;
};

inline constexpr Partition kSSAPartition{ValueKind::SSA, 1};
inline constexpr Partition kPointerPartition{ValueKind::Pointer, 2};

class ValueArena;

// One immutable binding of an admitted source partition to High values.
// At most two components are inline; larger bindings borrow ValueArena storage.
// Copies and projections preserve those identities without copying large
// component arrays. The partition and arena outlive every referencing binding.
// No source-value storage is retained by the output module.
class Value {
 public:
  Value() = default;
  Value(loom_value_id_t value) { storage_.inline_values[0] = value; }
  Value(Pointer pointer) : partition_(&kPointerPartition) {
    storage_.inline_values[0] = pointer.root;
    storage_.inline_values[1] = pointer.byte_offset;
  }

  const Partition& partition() const { return *partition_; }
  bool is_pointer() const { return partition_->kind == ValueKind::Pointer; }
  bool is_record() const { return partition_->kind == ValueKind::Record; }
  bool is_array() const { return partition_->kind == ValueKind::Array; }
  bool is_buffer() const { return partition_->kind == ValueKind::Buffer; }
  bool is_address() const { return partition_->kind == ValueKind::Address; }
  bool is_object() const { return partition_->kind == ValueKind::Object; }
  bool is_opaque_dialect() const {
    return partition_->kind == ValueKind::OpaqueDialect;
  }
  bool is_encoding() const { return partition_->kind == ValueKind::Encoding; }
  bool is_pool() const { return partition_->kind == ValueKind::Pool; }
  bool is_tile() const { return partition_->kind == ValueKind::Tile; }
  bool is_channel() const { return partition_->kind == ValueKind::Channel; }
  bool is_vector() const { return partition_->kind == ValueKind::Vector; }
  bool is_view() const { return partition_->kind == ValueKind::View; }
  bool is_tensor() const { return partition_->kind == ValueKind::Tensor; }
  loom_value_id_t ssa() const {
    IREE_ASSERT(partition_->kind == ValueKind::SSA ||
                partition_->kind == ValueKind::Buffer ||
                partition_->kind == ValueKind::Address ||
                partition_->kind == ValueKind::Object ||
                partition_->kind == ValueKind::OpaqueDialect ||
                partition_->kind == ValueKind::Encoding ||
                partition_->kind == ValueKind::Pool);
    return storage_.inline_values[0];
  }
  Pointer pointer() const {
    IREE_ASSERT(is_pointer());
    return {storage_.inline_values[0], storage_.inline_values[1]};
  }
  std::span<const loom_value_id_t> components() const {
    return {partition_->component_count <= 2 ? storage_.inline_values
                                             : storage_.indirect_values,
            partition_->component_count};
  }
  // Projects a retained member slice. Large members share immutable parent
  // storage; small members copy their IDs into the returned value.
  Value project(const Partition& member, size_t component_offset) const {
    return Value(
        member, components().subspan(component_offset, member.component_count));
  }
  void append_to(std::vector<loom_value_id_t>& values) const {
    auto source = components();
    values.insert(values.end(), source.begin(), source.end());
  }

 private:
  friend class ValueArena;

  Value(const Partition& partition,
        std::span<const loom_value_id_t> components);

  // Source structure, with lifetime independent of this particular binding.
  const Partition* partition_ = &kSSAPartition;
  // The partition's count selects inline IDs or immutable arena storage.
  union Storage {
    // Scalar, pointer or small-record components, copied with the binding.
    loom_value_id_t inline_values[2];
    // Larger component array owned by the enclosing ValueArena.
    const loom_value_id_t* indirect_values;
  } storage_ = {{LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID}};
};

// Owns immutable component arrays for one function's source translation.
// Capturing transient builder or stack storage copies it once. Replacement
// creates a new binding; it never writes into a snapshot or another source
// copy. Inline bindings do not allocate. Allocation failure propagates as
// bad_alloc to the import API's existing infrastructure-failure boundary.
class ValueArena {
 public:
  Value capture(const Partition& partition,
                std::span<const loom_value_id_t> components);
  Value replace(Value original, size_t component_offset, Value replacement);
  // Releases all large bindings after the function's binding maps are cleared.
  void reset() { storage_.release(); }

 private:
  // Lazily allocated blocks; ordinary scalar/vector/pointer functions use none.
  std::pmr::monotonic_buffer_resource storage_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_REPRESENTATION_H_
