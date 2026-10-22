// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/bitcast.h"

#include "loom/import/cxx/source/error.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {

int64_t bit_count(loom_type_t type) {
  auto count = loom_type_kind(type) == LOOM_TYPE_VECTOR
                   ? loom_type_dim_static_size_at(type, 0)
                   : 1;
  return count * loom_scalar_type_bitwidth(loom_type_element_type(type));
}

}  // namespace

BitCast::BitCast(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                 Types& types, const cxx::Type* input_type,
                 const cxx::Type* output_type, cxx::AST* owner)
    : input_(types.get(input_type, owner)),
      output_(types.get(output_type, owner)) {
  for (auto type : {input_, output_}) {
    if (loom_type_kind(type) != LOOM_TYPE_SCALAR &&
        loom_type_kind(type) != LOOM_TYPE_VECTOR) {
      diagnostics.reject(unit, owner,
                         "bit_cast requires scalar or vector values");
    }
  }
  // Source object sizes alone are insufficient: padded vectors carry fewer
  // value bits than their footprint, and bool occupies a byte in C++ but is
  // represented by one predicate bit in High. Equal value widths preserve all
  // observable bits while leaving destination padding unspecified.
  if (bit_count(input_) != bit_count(output_)) {
    diagnostics.reject(unit, owner,
                       "bit_cast requires equal value bit widths; bool has a "
                       "one-bit value representation");
  }
}

loom_value_id_t BitCast::build(loom_builder_t& builder, loom_value_id_t value,
                               loom_location_id_t location) const {
  if (loom_type_equal(input_, output_)) {
    return value;
  }
  loom_op_t* op;
  if (loom_type_kind(input_) == LOOM_TYPE_SCALAR &&
      loom_type_kind(output_) == LOOM_TYPE_SCALAR) {
    check(loom_scalar_bitcast_build(&builder, value, input_, output_, location,
                                    &op));
    return loom_op_results(op)[0];
  }
  auto input = input_;
  auto output = output_;
  if (loom_type_kind(input) == LOOM_TYPE_SCALAR) {
    input = loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(input),
                                1, 0);
    check(loom_vector_from_elements_build(&builder, &value, 1, input, location,
                                          &op));
    value = loom_op_results(op)[0];
  }
  if (loom_type_kind(output) == LOOM_TYPE_SCALAR) {
    output = loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                                 loom_type_element_type(output), 1, 0);
  }
  if (!loom_type_equal(input, output)) {
    check(loom_vector_bitcast_build(&builder, value, input, output, location,
                                    &op));
    value = loom_op_results(op)[0];
  }
  if (loom_type_kind(output_) == LOOM_TYPE_SCALAR) {
    int64_t index = 0;
    check(loom_vector_extract_build(&builder, value, nullptr, 0, &index, 1,
                                    output_, location, &op));
    value = loom_op_results(op)[0];
  }
  return value;
}

}  // namespace loom::cxx_import
