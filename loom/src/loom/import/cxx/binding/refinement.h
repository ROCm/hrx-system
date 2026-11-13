// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_REFINEMENT_H_
#define LOOM_IMPORT_CXX_BINDING_REFINEMENT_H_

#include <cxx/attributes.h>

#include <array>
#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// Source shape retained from an admitted carrier. Nominal element/access
// semantics stay exact while explicit refinement or template selection may
// assert different extents and complete encodings.
struct ShapedSignature {
  // Source value partition, owned by Types or static scalar storage.
  const Partition* partition;
  // Native C++ vector type; other carriers bind through their partition.
  loom_type_t native_type = {};
  // Native carrier kind; nominal aggregates remain outside shape matching.
  loom_type_kind_t kind;
  // Source element identity, including view access qualifiers.
  const cxx::Type* element;
  // Canonical rank-generic extents, or null for a tensor/native vector.
  const ShapeExtents* extents = nullptr;
  // Rank-one or rank-two extents for tensor and native vector carriers.
  std::array<int64_t, 2> inline_extents = {};
  // Logical native rank, including a tensor's vector-element lane axis.
  size_t rank = 0;
  // Complete encoding role, absent for native dense representation.
  const EncodingPartition* encoding = nullptr;

  // Binds the payload type to explicit dimensions and optional encoding.
  loom_type_t bind_type(std::span<const loom_value_id_t> dependencies,
                        BoundTypeStorage& storage) const;

  int64_t extent(size_t axis) const {
    return extents ? (*extents)[axis] : inline_extents[axis];
  }
};

// Returns the admitted shape of a supported source carrier, or nullopt for an
// ordinary scalar, aggregate, or handle without shaped-value semantics.
std::optional<ShapedSignature> shaped_signature(Types& types,
                                                const cxx::Type* source,
                                                cxx::AST* owner);

// Emits an explicit identity assertion, eliding only an exactly identical type.
loom_value_id_t refine_value(loom_value_id_t value, loom_type_t type,
                             loom_builder_t* builder,
                             loom_location_id_t location);

// Concrete explicit refinement. Admission owns source signature validation;
// calls consume the retained partition and supplied result dependencies.
class RefinementIntrinsic {
 public:
  static bool supports(std::string_view name);
  static std::optional<RefinementIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  Value call(std::span<const Value> arguments, ValueArena& arena,
             loom_builder_t* builder, loom_location_id_t location) const;

  bool equivalent(const RefinementIntrinsic& other) const {
    return result_.partition == other.result_.partition &&
           loom_type_equal(result_.native_type, other.result_.native_type) &&
           dynamic_count_ == other.dynamic_count_ &&
           explicit_encoding_ == other.explicit_encoding_;
  }

 private:
  RefinementIntrinsic(ShapedSignature result, size_t dynamic_count,
                      bool explicit_encoding)
      : result_(result),
        dynamic_count_(dynamic_count),
        explicit_encoding_(explicit_encoding) {}

  // Exact result carrier and its source-owned extent/encoding declarations.
  ShapedSignature result_;
  // Number of explicitly supplied index-domain result extents.
  size_t dynamic_count_;
  // Whether the caller replaces the encoding instead of retaining the source's.
  bool explicit_encoding_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_REFINEMENT_H_
