// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_VECTOR_MEMORY_H_
#define LOOM_IMPORT_CXX_BINDING_VECTOR_MEMORY_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

// One concrete contiguous vector memory specialization. Source declarations
// retain their native footprint or semantic vector projection together with
// the typed view contract, so calls only bind dependent result dimensions and
// the already admitted full-rank origin to the corresponding High op.
class VectorMemoryIntrinsic {
 public:
  // Returns whether |name| belongs to this binding family.
  static bool supports(std::string_view name);

  // Resolves a concrete source specialization. Unknown operation names return
  // no value; malformed recognized declarations diagnose at |owner|.
  static std::optional<VectorMemoryIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  // Emits the admitted operation. A store has no result.
  std::optional<Value> call(std::span<const Value> arguments, ValueArena& arena,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  // Compares the complete concrete contract for one canonical declaration.
  bool equivalent(const VectorMemoryIntrinsic& other) const;

 private:
  enum class Operation {
    Load,
    Store,
    LoadMask,
    StoreMask,
    LoadExpand,
    StoreCompress,
  };

  static std::optional<Operation> parse_operation(std::string_view name);

  explicit VectorMemoryIntrinsic(Operation operation) : operation_(operation) {}

  bool is_load() const;
  bool has_mask() const;
  loom_value_id_t build_mask(loom_value_id_t source, loom_builder_t* builder,
                             loom_location_id_t location) const;

  // Selected contiguous vector memory operation.
  Operation operation_;
  // Admitted source or destination view.
  const ViewPartition* view_ = nullptr;
  // Exact register-vector footprint loaded or stored.
  loom_type_t vector_type_ = {};
  // Semantic vector result or operand; null for a native C++ vector.
  const VectorPartition* semantic_vector_ = nullptr;
  // Result-dimension argument; absent for stores and static two-argument loads.
  std::optional<size_t> dimensions_index_;
  // Source integer mask carrier; unset for ordinary load/store.
  loom_type_t mask_type_ = {};
  // Internal i1 mask shape; unset for ordinary load/store.
  loom_type_t predicate_type_ = {};
  // Unsigned source interpretations in the full-rank origin record.
  uint16_t origin_unsigned_mask_ = 0;
  // Explicit physical-root proof selected by the source specialization.
  loom_memory_access_flags_t memory_flags_ = 0;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_VECTOR_MEMORY_H_
