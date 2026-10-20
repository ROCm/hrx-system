// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_SHAPED_H_
#define LOOM_IMPORT_CXX_BINDING_SHAPED_H_

#include <cxx/attributes.h>
#include <cxx/types.h>

#include <array>
#include <optional>
#include <span>
#include <variant>

#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {

// An admitted register-vector intrinsic. Source signatures preserve the
// operation's heterogeneous shapes and element interpretations. Calls consume
// the retained result type and semantic kind without reexamining source types.
class ShapedIntrinsic {
 public:
  struct Splat {
    bool operator==(const Splat&) const = default;
  };
  struct Slice {
    // Unsigned integer interpretations of the flattened offset components.
    uint16_t offset_unsigned_mask = 0;
    // Whether the source call supplies dependent result dimensions.
    bool has_dimensions = false;
    bool operator==(const Slice&) const = default;
  };
  struct TableLookup {
    bool operator==(const TableLookup&) const = default;
  };
  struct Absolute {
    enum class Kind { Float, Integer };

    // Element category selected by the operation spelling.
    Kind kind;
    // Explicit source permissions; integer absolute has none.
    uint8_t flags;
    bool operator==(const Absolute&) const = default;
  };
  struct FloatExtremum {
    enum class Kind { Minimum, Maximum, MinNumber, MaxNumber };

    // NaN/comparison semantics selected by the operation spelling.
    Kind kind;
    // Explicit source permissions.
    uint8_t flags;
    bool operator==(const FloatExtremum&) const = default;
  };
  struct Dot4i {
    // Explicit interpretation of each input's byte lanes. An absent kind is
    // inferred from a concrete declaration's source element signedness and
    // resolved before this operation can be called.
    std::optional<loom_vector_dot4i_kind_t> kind;
    bool operator==(const Dot4i&) const = default;
  };
  struct Dot2f {
    bool operator==(const Dot2f&) const = default;
  };
  struct Dotf {
    // Source permissions in addition to invocation permissions.
    uint8_t flags;
    bool operator==(const Dotf&) const = default;
  };
  struct Reduction {
    // Combining operation applied in logical lane order.
    loom_combining_kind_t kind;
    // Source permissions; zero for integer combining operations.
    uint8_t flags;
    bool operator==(const Reduction&) const = default;
  };
  using Operation = std::variant<Splat, Slice, TableLookup, Absolute,
                                 FloatExtremum, Dot4i, Dot2f, Dotf, Reduction>;

  // Admits a string-only loom::op attribute independently of source types.
  // Unknown names return nullopt; recognized names with invalid declarations
  // diagnose at owner and throw SourceRejected. Templates retain this operation
  // until a concrete specialization supplies the signature.
  static std::optional<Operation> admit(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        const cxx::Attribute& attribute,
                                        cxx::AST* owner);

  // Validates a concrete signature for an admitted operation. Source objects
  // are borrowed during admission; the binding owns no frontend storage.
  static ShapedIntrinsic resolve(Operation operation,
                                 cxx::TranslationUnit& unit,
                                 Diagnostics& diagnostics, Types& types,
                                 const cxx::FunctionType* signature,
                                 cxx::AST* owner);

  // Emits the admitted operation using already-converted source arguments.
  Value call(std::span<const Value> arguments, ValueArena& arena,
             uint8_t math_flags, loom_builder_t* builder,
             loom_location_id_t location) const;

  // Compares operation semantics for redeclarations of one canonical symbol.
  bool equivalent(const ShapedIntrinsic& other) const;

 private:
  ShapedIntrinsic(Operation operation, loom_type_t result_type,
                  const VectorPartition* result_vector = nullptr,
                  std::array<const VectorPartition*, 3> semantic_operands = {})
      : operation_(operation),
        result_type_(result_type),
        result_vector_(result_vector),
        semantic_operands_(semantic_operands) {}

  // Operation family admitted from the source declaration.
  Operation operation_;
  // Exact source scalar or vector result type.
  loom_type_t result_type_;
  // Semantic vector result representation, or null for a native/scalar result.
  const VectorPartition* result_vector_;
  // Semantic vector operands by source position; null denotes a native vector.
  std::array<const VectorPartition*, 3> semantic_operands_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_SHAPED_H_
