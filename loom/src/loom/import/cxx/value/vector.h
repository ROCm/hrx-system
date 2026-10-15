// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_VECTOR_H_
#define LOOM_IMPORT_CXX_VALUE_VECTOR_H_

#include <span>

#include "loom/import/cxx/value/scalar.h"

namespace loom::cxx_import {

// Projects explicit source vector operations, without scalar integer promotion
// or automatic vectorization. Operands are evaluated by the caller. Source
// signedness selects lane operations; comparison results retain C++'s
// full-width integer masks rather than exposing Loom's internal i1 predicates.
class Vectors {
 public:
  Vectors(cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
          Scalars& scalars, Locations& locations, loom_builder_t& builder)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        scalars_(scalars),
        locations_(locations),
        builder_(builder) {}

  // Applies a resolved vector splat or equal-width bit reinterpretation.
  // Numeric lane conversion is a different source operation and is not inferred
  // here.
  loom_value_id_t convert(loom_value_id_t value, const cxx::Type* input_type,
                          const cxx::Type* output_type, cxx::AST* owner,
                          uint8_t fastmath = 0);
  // Applies an explicit numeric conversion to each lane. Frontend semantics
  // have established vector operands with the same number of elements.
  loom_value_id_t convert_elements(loom_value_id_t value,
                                   const cxx::Type* input_type,
                                   const cxx::Type* output_type,
                                   cxx::AST* owner);
  // Operands already have the source operation's common vector type.
  loom_value_id_t binary(cxx::TokenKind token, loom_value_id_t left,
                         loom_value_id_t right, const cxx::Type* input_type,
                         const cxx::Type* output_type, cxx::AST* owner,
                         uint8_t fastmath = 0);
  // Preserves vector lane width for sign, complement and logical negation.
  loom_value_id_t unary(cxx::TokenKind token, loom_value_id_t value,
                        const cxx::Type* input_type,
                        const cxx::Type* output_type, cxx::AST* owner,
                        uint8_t fastmath = 0);
  // Constructs an initialized vector from source-order scalar lanes. Omitted
  // trailing lanes are zero, as in a C/C++ aggregate initializer.
  loom_value_id_t construct(std::span<const loom_value_id_t> elements,
                            const cxx::Type* source_type, cxx::AST* owner);
  // Materializes the typed lanes of a frontend vector constant.
  loom_value_id_t constant(const cxx::ConstValue& value,
                           const cxx::Type* source_type, cxx::AST* owner);
  // Reads one lane. Source-defined indices are nonnegative and below the lane
  // count; the index carrier does not change that source bounds contract.
  loom_value_id_t extract(loom_value_id_t value, loom_value_id_t index,
                          const cxx::Type* source_type,
                          const cxx::Type* index_type, cxx::AST* owner);
  // Replaces one lane and returns the updated vector. Source-defined indices
  // are nonnegative and below the lane count; the index carrier does not
  // change that source bounds contract.
  loom_value_id_t insert(loom_value_id_t destination, loom_value_id_t lane,
                         loom_value_id_t index, const cxx::Type* source_type,
                         const cxx::Type* index_type, cxx::AST* owner);

 private:
  // Projects a source integral lane selector to Loom's index domain.
  loom_value_id_t cast_index(loom_value_id_t index, const cxx::Type* index_type,
                             cxx::AST* owner);

  // Resolved frontend types and object layout.
  cxx::TranslationUnit& unit_;
  // Source admission boundary.
  Diagnostics& diagnostics_;
  // Validated source vector representation.
  Types& types_;
  // Scalar conversion of a splatted lane.
  Scalars& scalars_;
  // Source provenance retained in the output module.
  Locations& locations_;
  // Borrowed insertion point owned by the AST driver.
  loom_builder_t& builder_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_VECTOR_H_
