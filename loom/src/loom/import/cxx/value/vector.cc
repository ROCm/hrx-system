// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/vector.h"

#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {

loom_value_id_t Vectors::convert(loom_value_id_t value,
                                 const cxx::Type* input_type,
                                 const cxx::Type* output_type, cxx::AST* owner,
                                 uint8_t fastmath) {
  auto input = types_.get(input_type, owner);
  auto output = types_.get(output_type, owner);
  auto* vector = types_.native_vector(output_type);
  if (!vector) {
    diagnostics_.reject(unit_, owner,
                        "vector conversion requires a vector destination");
  }
  if (loom_type_equal(input, output)) {
    return value;
  }
  loom_op_t* op;
  if (loom_type_kind(input) == LOOM_TYPE_SCALAR) {
    value = scalars_.convert(value, input_type, vector->elementType(), owner,
                             fastmath);
    check(loom_vector_splat_build(&builder_, value, output,
                                  locations_.get(owner), &op));
  } else if (loom_type_kind(input) == LOOM_TYPE_VECTOR &&
             unit_.control()->memoryLayout()->sizeOf(input_type) ==
                 unit_.control()->memoryLayout()->sizeOf(output_type)) {
    check(loom_vector_bitcast_build(&builder_, value, input, output,
                                    locations_.get(owner), &op));
  } else {
    diagnostics_.reject(unit_, owner,
                        "vector reinterpretation requires equal object sizes");
  }
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::convert_elements(loom_value_id_t value,
                                          const cxx::Type* input_type,
                                          const cxx::Type* output_type,
                                          cxx::AST* owner) {
  auto input = types_.get(input_type, owner);
  auto output = types_.get(output_type, owner);
  if (loom_type_equal(input, output)) {
    return value;
  }
  auto* input_vector = types_.native_vector(input_type);
  auto* input_element = input_vector->elementType();
  auto* output_element = types_.native_vector(output_type)->elementType();
  bool floating_input = types_.is_float(input_element);
  bool floating_output = types_.is_float(output_element);
  auto* layout = unit_.control()->memoryLayout();
  auto input_size = layout->sizeOf(input_element);
  auto output_size = layout->sizeOf(output_element);
  auto location = locations_.get(owner);
  loom_op_t* op;
  if (floating_input && floating_output && input_size == output_size) {
    // F32 represents every value of the supported equal-width narrow formats
    // exactly. Only the final conversion rounds to the destination format.
    auto widened = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                       input_vector->elementCount(), 0);
    check(loom_vector_extf_build(&builder_, 0, value, input, widened, location,
                                 &op));
    check(loom_vector_fptrunc_build(&builder_, 0, loom_op_results(op)[0],
                                    widened, output, location, &op));
    return loom_op_results(op)[0];
  }
  bool narrows = input_size > output_size;
  bool unsigned_input = types_.is_unsigned(input_element);
  if (floating_input && floating_output) {
    check((narrows ? loom_vector_fptrunc_build : loom_vector_extf_build)(
        &builder_, 0, value, input, output, location, &op));
    return loom_op_results(op)[0];
  }
  auto build =
      floating_output
          ? (unsigned_input ? loom_vector_uitofp_build
                            : loom_vector_sitofp_build)
          : (floating_input
                 ? (types_.is_unsigned(output_element)
                        ? loom_vector_fptoui_build
                        : loom_vector_fptosi_build)
                 : (narrows ? loom_vector_trunci_build
                            : (unsigned_input ? loom_vector_extui_build
                                              : loom_vector_extsi_build)));
  check(build(&builder_, value, input, output, location, &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::binary(cxx::TokenKind token, loom_value_id_t left,
                                loom_value_id_t right,
                                const cxx::Type* input_type,
                                const cxx::Type* output_type, cxx::AST* owner,
                                uint8_t fastmath) {
  auto input = types_.get(input_type, owner);
  auto output = types_.get(output_type, owner);
  auto* vector = types_.native_vector(input_type);
  bool floating = types_.is_float(vector->elementType());
  bool unsigned_input = types_.is_unsigned(vector->elementType());
  auto source = locations_.get(owner);
  loom_op_t* op;
  int predicate = -1;
  switch (token) {
    case cxx::TokenKind::T_EQUAL_EQUAL:
      predicate = 0;
      break;
    case cxx::TokenKind::T_EXCLAIM_EQUAL:
      predicate = 1;
      break;
    case cxx::TokenKind::T_LESS:
      predicate = 2;
      break;
    case cxx::TokenKind::T_LESS_EQUAL:
      predicate = 3;
      break;
    case cxx::TokenKind::T_GREATER:
      predicate = 4;
      break;
    case cxx::TokenKind::T_GREATER_EQUAL:
      predicate = 5;
      break;
    default:
      break;
  }
  if (predicate >= 0) {
    auto mask_type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1,
                                         vector->elementCount(), 0);
    if (floating) {
      const loom_vector_cmpf_predicate_t predicates[] = {
          LOOM_VECTOR_CMPF_PREDICATE_OEQ, LOOM_VECTOR_CMPF_PREDICATE_UNE,
          LOOM_VECTOR_CMPF_PREDICATE_OLT, LOOM_VECTOR_CMPF_PREDICATE_OLE,
          LOOM_VECTOR_CMPF_PREDICATE_OGT, LOOM_VECTOR_CMPF_PREDICATE_OGE};
      check(loom_vector_cmpf_build(&builder_, fastmath, predicates[predicate],
                                   left, right, input, mask_type, source, &op));
    } else {
      auto selected =
          predicate >= 2 && unsigned_input ? predicate + 4 : predicate;
      check(loom_vector_cmpi_build(&builder_, selected, left, right, input,
                                   mask_type, source, &op));
    }
    // GNU/Clang vectors expose all bits set in each true integer lane. Zero
    // extension would silently change both bitwise selection and stored masks.
    check(loom_vector_extsi_build(&builder_, loom_op_results(op)[0], mask_type,
                                  output, source, &op));
    return loom_op_results(op)[0];
  }
  if (!floating) {
    auto build = token == cxx::TokenKind::T_SLASH
                     ? (unsigned_input ? loom_vector_divui_build
                                       : loom_vector_divsi_build)
                 : token == cxx::TokenKind::T_PERCENT
                     ? (unsigned_input ? loom_vector_remui_build
                                       : loom_vector_remsi_build)
                 : token == cxx::TokenKind::T_AMP   ? loom_vector_andi_build
                 : token == cxx::TokenKind::T_BAR   ? loom_vector_ori_build
                 : token == cxx::TokenKind::T_CARET ? loom_vector_xori_build
                 : token == cxx::TokenKind::T_GREATER_GREATER
                     ? (unsigned_input ? loom_vector_shrui_build
                                       : loom_vector_shrsi_build)
                     : nullptr;
    if (build) {
      check(build(&builder_, left, right, output, source, &op));
      return loom_op_results(op)[0];
    }
  }
  auto build =
      token == cxx::TokenKind::T_PLUS
          ? (floating ? loom_vector_addf_build : loom_vector_addi_build)
      : token == cxx::TokenKind::T_MINUS
          ? (floating ? loom_vector_subf_build : loom_vector_subi_build)
      : token == cxx::TokenKind::T_STAR
          ? (floating ? loom_vector_mulf_build : loom_vector_muli_build)
      : token == cxx::TokenKind::T_SLASH && floating ? loom_vector_divf_build
      : token == cxx::TokenKind::T_LESS_LESS && !floating
          ? loom_vector_shli_build
          : nullptr;
  if (!build) {
    diagnostics_.reject(unit_, owner, "unsupported vector binary operator");
  }
  const uint8_t flags = floating ? fastmath : 0;
  check(build(&builder_, flags, left, right, output, source, &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::unary(cxx::TokenKind token, loom_value_id_t value,
                               const cxx::Type* input_type,
                               const cxx::Type* output_type, cxx::AST* owner,
                               uint8_t fastmath) {
  auto input = types_.get(input_type, owner);
  auto output = types_.get(output_type, owner);
  bool floating =
      types_.is_float(types_.native_vector(input_type)->elementType());
  auto source = locations_.get(owner);
  loom_op_t* op;
  switch (token) {
    case cxx::TokenKind::T_PLUS:
      return value;
    case cxx::TokenKind::T_MINUS:
      if (floating) {
        check(loom_vector_negf_build(&builder_, fastmath, value, output, source,
                                     &op));
      } else {
        check(loom_vector_negi_build(&builder_, value, output, source, &op));
      }
      break;
    case cxx::TokenKind::T_TILDE:
      check(loom_vector_constant_build(&builder_, loom_attr_i64(-1), input,
                                       source, &op));
      check(loom_vector_xori_build(&builder_, value, loom_op_results(op)[0],
                                   output, source, &op));
      break;
    case cxx::TokenKind::T_EXCLAIM:
      check(loom_vector_constant_build(
          &builder_, floating ? loom_attr_f64(0.0) : loom_attr_i64(0), input,
          source, &op));
      return binary(cxx::TokenKind::T_EQUAL_EQUAL, value,
                    loom_op_results(op)[0], input_type, output_type, owner,
                    fastmath);
    default:
      diagnostics_.reject(unit_, owner, "unsupported vector unary operator");
  }
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::construct(std::span<const loom_value_id_t> elements,
                                   const cxx::Type* source_type,
                                   cxx::AST* owner) {
  auto output = types_.get(source_type, owner);
  auto* vector = types_.native_vector(source_type);
  if (elements.size() > vector->elementCount()) {
    diagnostics_.reject(unit_, owner, "too many vector initializer lanes");
  }
  std::vector<loom_value_id_t> lanes(elements.begin(), elements.end());
  if (lanes.size() < vector->elementCount()) {
    auto zero = scalars_.constant(int64_t{0}, vector->elementType(), owner);
    lanes.resize(vector->elementCount(), zero);
  }
  loom_op_t* op;
  check(loom_vector_from_elements_build(&builder_, lanes.data(), lanes.size(),
                                        output, locations_.get(owner), &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::constant(const cxx::ConstValue& value,
                                  const cxx::Type* source_type,
                                  cxx::AST* owner) {
  const auto& list = *std::get<std::shared_ptr<cxx::InitializerList>>(value);
  std::vector<loom_value_id_t> elements;
  elements.reserve(list.elements.size());
  for (const auto& [element, element_type] : list.elements) {
    elements.push_back(scalars_.constant(element, element_type, owner));
  }
  return construct(elements, source_type, owner);
}

loom_value_id_t Vectors::extract(loom_value_id_t value, loom_value_id_t index,
                                 const cxx::Type* source_type,
                                 const cxx::Type* index_type, cxx::AST* owner) {
  index = cast_index(index, index_type, owner);
  auto* vector = types_.native_vector(source_type);
  types_.get(source_type, owner);
  auto output = types_.get(vector->elementType(), owner);
  loom_op_t* op;
  int64_t selector = INT64_MIN;
  check(loom_vector_extract_build(&builder_, value, &index, 1, &selector, 1,
                                  output, locations_.get(owner), &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::insert(loom_value_id_t destination,
                                loom_value_id_t lane, loom_value_id_t index,
                                const cxx::Type* source_type,
                                const cxx::Type* index_type, cxx::AST* owner) {
  index = cast_index(index, index_type, owner);
  auto output = types_.get(source_type, owner);
  loom_op_t* op;
  int64_t selector = INT64_MIN;
  check(loom_vector_insert_build(&builder_, lane, destination, &index, 1,
                                 &selector, 1, output, locations_.get(owner),
                                 &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Vectors::cast_index(loom_value_id_t index,
                                    const cxx::Type* index_type,
                                    cxx::AST* owner) {
  if (!unit_.typeTraits().is_integral(index_type)) {
    diagnostics_.reject(unit_, owner,
                        "vector subscripts require integral indices");
  }
  loom_op_t* op;
  check(loom_index_cast_build(&builder_, index, types_.get(index_type, owner),
                              loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                              locations_.get(owner), &op));
  return loom_op_results(op)[0];
}

}  // namespace loom::cxx_import
