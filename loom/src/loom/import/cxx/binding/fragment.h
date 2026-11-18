// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_FRAGMENT_H_
#define LOOM_IMPORT_CXX_BINDING_FRAGMENT_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {

// One concrete vector.fragment*, or vector.mma, source specialization. The
// binding retains logical role/shape and named-operand structure while all
// physical fragment carriers remain ordinary source vectors.
class FragmentIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Resolves a concrete source specialization. Leading constant template
  // arguments select the fragment role, memory observation, or MMA flags;
  // source records describe logical shapes, origins, and named parameters.
  static std::optional<FragmentIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      loom_module_t* module, cxx::AST* owner);

  // Emits the admitted operation from already evaluated source values. Store
  // has no result; every other operation returns one physical vector value.
  std::optional<Value> call(std::span<const Value> arguments,
                            loom_builder_t* builder,
                            loom_location_id_t location) const;

  bool equivalent(const FragmentIntrinsic& other) const;

 private:
  enum class Operation { Attach, Repack, Load, Store, Mma };

  struct Shape {
    // Two for [rows, columns], or three for [blocks, rows, columns].
    uint8_t rank = 0;
    // Unsigned source interpretations in flattened component order.
    uint16_t unsigned_mask = 0;

    bool operator==(const Shape&) const = default;
  };

  static std::optional<Operation> parse_operation(std::string_view name);
  static Shape require_shape(cxx::TranslationUnit& unit,
                             Diagnostics& diagnostics, Types& types,
                             const cxx::Type* type, cxx::AST* owner);

  explicit FragmentIntrinsic(Operation operation) : operation_(operation) {}

  // Selected vector operation.
  Operation operation_;
  // Fragment role for attach/repack/load/store.
  loom_vector_role_t role_ = LOOM_VECTOR_ROLE_LHS;
  // Optional vector.mma semantic flags.
  uint8_t mma_flags_ = 0;
  // Logical shape record carried by fragment operations.
  Shape shape_;
  // Unsigned source interpretations for runtime origin components.
  uint16_t origin_unsigned_mask_ = 0;
  // Logical shape record argument.
  uint8_t shape_argument_ = 0;
  // Named parameter/auxiliary record argument, or UINT8_MAX when absent.
  uint8_t named_operand_argument_ = UINT8_MAX;
  // Admitted source/destination view for load/store.
  const ViewPartition* view_ = nullptr;
  // Inferred and explicitly selected semantics for one memory observation.
  loom_memory_access_flags_t memory_flags_ = 0;
  // Exact physical vector result for non-store operations.
  loom_type_t result_type_ = {};
  // Module-interned record keys in source component order.
  std::vector<loom_string_id_t> named_operand_names_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_FRAGMENT_H_
