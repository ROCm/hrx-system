// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/scalar.h"

#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include <bit>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

loom_attribute_t integer_attribute(uint64_t bits, unsigned bytes) {
  // IR integer attributes use signed storage at their scalar type's width.
  int64_t stored =
      bytes == 1   ? std::bit_cast<int8_t>(static_cast<uint8_t>(bits))
      : bytes == 2 ? std::bit_cast<int16_t>(static_cast<uint16_t>(bits))
      : bytes == 4 ? std::bit_cast<int32_t>(static_cast<uint32_t>(bits))
                   : std::bit_cast<int64_t>(bits);
  return loom_attr_i64(stored);
}

const cxx::ConstValue& address_constant_payload(cxx::TranslationUnit& unit,
                                                Diagnostics& diagnostics,
                                                const cxx::ConstValue& value,
                                                loom_scalar_type_t domain,
                                                cxx::AST* owner) {
  auto* object = std::get_if<std::shared_ptr<cxx::ConstObject>>(&value);
  if (!object || !*object || (*object)->members().size() != 1) {
    diagnostics.reject(unit, owner,
                       "address constants require one carrier storage member");
  }
  const auto& member = (*object)->members().front();
  auto* field =
      member.symbol && member.symbol->kind() == cxx::SymbolKind::kField
          ? static_cast<const cxx::FieldSymbol*>(member.symbol)
          : nullptr;
  const cxx::Type* expected =
      domain == LOOM_SCALAR_TYPE_INDEX
          ? static_cast<const cxx::Type*>(unit.control()->getLongLongIntType())
          : static_cast<const cxx::Type*>(
                unit.control()->getUnsignedLongLongIntType());
  if (!field || field->isStatic() ||
      !unit.typeTraits().is_same(unit.typeTraits().remove_cv(field->type()),
                                 expected)) {
    diagnostics.reject(
        unit, owner, "address constant storage does not match its signedness");
  }
  return member.value;
}

}  // namespace

loom_value_id_t Scalars::convert(loom_value_id_t value,
                                 const cxx::Type* input_type,
                                 const cxx::Type* output_type, cxx::AST* owner,
                                 uint8_t fastmath) {
  auto input = types_.get(input_type, owner);
  auto output = types_.get(output_type, owner);
  if (loom_type_kind(input) != LOOM_TYPE_SCALAR ||
      loom_type_kind(output) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, owner,
                        "scalar conversion requires numeric source types");
  }
  if (loom_type_equal(input, output)) {
    return value;
  }
  loom_op_t* op;
  if (loom_type_element_type(output) == LOOM_SCALAR_TYPE_I1) {
    loom_op_t* zero;
    check(loom_scalar_constant_build(
        &builder_,
        types_.is_float(input_type) ? loom_attr_f64(0.0) : loom_attr_i64(0),
        input, locations_.get(owner), &zero));
    if (types_.is_float(input_type)) {
      check(loom_scalar_cmpf_build(
          &builder_, fastmath, LOOM_SCALAR_CMPF_PREDICATE_UNE, value,
          loom_op_results(zero)[0], locations_.get(owner), &op));
    } else {
      check(loom_scalar_cmpi_build(&builder_, LOOM_SCALAR_CMPI_PREDICATE_NE,
                                   value, loom_op_results(zero)[0],
                                   locations_.get(owner), &op));
    }
    return loom_op_results(op)[0];
  }
  bool unsigned_input = types_.is_unsigned(input_type) ||
                        loom_type_element_type(input) == LOOM_SCALAR_TYPE_I1;
  auto* layout = unit_.control()->memoryLayout();
  if (types_.is_float(input_type) && types_.is_float(output_type) &&
      layout->sizeOf(input_type) == layout->sizeOf(output_type)) {
    // Distinct narrow formats can have equal storage widths. F32 represents
    // each exactly, so only the final cast rounds.
    auto widened = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    check(loom_scalar_extf_build(&builder_, 0, value, input, widened,
                                 locations_.get(owner), &op));
    value = loom_op_results(op)[0];
    check(loom_scalar_fptrunc_build(&builder_, 0, value, widened, output,
                                    locations_.get(owner), &op));
    return loom_op_results(op)[0];
  }
  bool narrows = layout->sizeOf(input_type) > layout->sizeOf(output_type);
  if (types_.is_float(input_type) && types_.is_float(output_type)) {
    check((narrows ? loom_scalar_fptrunc_build : loom_scalar_extf_build)(
        &builder_, 0, value, input, output, locations_.get(owner), &op));
    return loom_op_results(op)[0];
  }
  auto build =
      types_.is_float(output_type)
          ? (unsigned_input ? loom_scalar_uitofp_build
                            : loom_scalar_sitofp_build)
          : (types_.is_float(input_type)
                 ? (types_.is_unsigned(output_type) ? loom_scalar_fptoui_build
                                                    : loom_scalar_fptosi_build)
                 : (narrows ? loom_scalar_trunci_build
                            : (unsigned_input ? loom_scalar_extui_build
                                              : loom_scalar_extsi_build)));
  check(build(&builder_, value, input, output, locations_.get(owner), &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Scalars::binary(cxx::TokenKind token, loom_value_id_t left,
                                loom_value_id_t right,
                                const cxx::Type* input_type,
                                const cxx::Type* output_type, cxx::AST* ast,
                                uint8_t fastmath) {
  auto source = locations_.get(ast);
  bool floating = types_.is_float(input_type);
  bool unsigned_input = types_.is_unsigned(input_type);
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
    if (floating) {
      const loom_scalar_cmpf_predicate_t predicates[] = {
          LOOM_SCALAR_CMPF_PREDICATE_OEQ, LOOM_SCALAR_CMPF_PREDICATE_UNE,
          LOOM_SCALAR_CMPF_PREDICATE_OLT, LOOM_SCALAR_CMPF_PREDICATE_OLE,
          LOOM_SCALAR_CMPF_PREDICATE_OGT, LOOM_SCALAR_CMPF_PREDICATE_OGE};
      check(loom_scalar_cmpf_build(&builder_, fastmath, predicates[predicate],
                                   left, right, source, &op));
    } else {
      auto selected = static_cast<loom_scalar_cmpi_predicate_t>(
          predicate >= 2 && unsigned_input ? predicate + 4 : predicate);
      check(loom_scalar_cmpi_build(&builder_, selected, left, right, source,
                                   &op));
    }
    return loom_op_results(op)[0];
  }
  if (!floating) {
    auto build = token == cxx::TokenKind::T_SLASH
                     ? (unsigned_input ? loom_scalar_divui_build
                                       : loom_scalar_divsi_build)
                 : token == cxx::TokenKind::T_PERCENT
                     ? (unsigned_input ? loom_scalar_remui_build
                                       : loom_scalar_remsi_build)
                 : token == cxx::TokenKind::T_AMP   ? loom_scalar_andi_build
                 : token == cxx::TokenKind::T_BAR   ? loom_scalar_ori_build
                 : token == cxx::TokenKind::T_CARET ? loom_scalar_xori_build
                 : token == cxx::TokenKind::T_GREATER_GREATER
                     ? (unsigned_input ? loom_scalar_shrui_build
                                       : loom_scalar_shrsi_build)
                     : nullptr;
    if (build) {
      check(build(&builder_, left, right, types_.get(output_type, ast), source,
                  &op));
      return loom_op_results(op)[0];
    }
  }
  auto build =
      token == cxx::TokenKind::T_PLUS
          ? (floating ? loom_scalar_addf_build : loom_scalar_addi_build)
      : token == cxx::TokenKind::T_MINUS
          ? (floating ? loom_scalar_subf_build : loom_scalar_subi_build)
      : token == cxx::TokenKind::T_STAR
          ? (floating ? loom_scalar_mulf_build : loom_scalar_muli_build)
      : token == cxx::TokenKind::T_SLASH && floating ? loom_scalar_divf_build
      : token == cxx::TokenKind::T_LESS_LESS && !floating
          ? loom_scalar_shli_build
          : nullptr;
  if (!build) {
    diagnostics_.reject(unit_, ast, "unsupported binary operator");
  }
  const uint8_t flags = floating ? fastmath : 0;
  check(build(&builder_, flags, left, right, types_.get(output_type, ast),
              source, &op));
  return loom_op_results(op)[0];
}

loom_value_id_t Scalars::integer(int64_t value, loom_scalar_type_t scalar,
                                 loom_location_id_t source) {
  return materialize(loom_attr_i64(value), loom_type_scalar(scalar), source);
}

loom_value_id_t Scalars::materialize(loom_attribute_t value, loom_type_t type,
                                     loom_location_id_t source) {
  loom_op_t* op;
  const auto scalar = loom_type_element_type(type);
  auto* build =
      scalar == LOOM_SCALAR_TYPE_INDEX || scalar == LOOM_SCALAR_TYPE_OFFSET
          ? loom_index_constant_build
          : loom_scalar_constant_build;
  check(build(&builder_, value, type, source, &op));
  return loom_op_results(op)[0];
}

loom_attribute_t Scalars::constant_attribute(const cxx::ConstValue& value,
                                             const cxx::Type* source_type,
                                             cxx::AST* ast) {
  cxx::ASTInterpreter interpreter(&unit_);
  auto target = types_.get(source_type, ast);
  if (loom_type_kind(target) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, ast, "constants require a numeric scalar type");
  }
  const auto element_type = loom_type_element_type(target);
  const bool is_address = element_type == LOOM_SCALAR_TYPE_INDEX ||
                          element_type == LOOM_SCALAR_TYPE_OFFSET;
  const cxx::ConstValue& payload =
      is_address ? address_constant_payload(unit_, diagnostics_, value,
                                            element_type, ast)
                 : value;
  loom_attribute_t attribute;
  if (types_.is_float(source_type)) {
    if (auto* floating = std::get_if<cxx::ConstFloat>(&payload);
        floating && floating->isNaN()) {
      diagnostics_.reject(
          unit_, ast,
          "numeric metadata cannot preserve NaN representation bits; "
          "use an integer encoding and bit-cast it in the function body");
    }
    auto number = interpreter.toDouble(payload);
    if (!number) {
      diagnostics_.reject(unit_, ast, "invalid floating literal");
    }
    attribute = loom_attr_f64(*number);
  } else {
    auto number = interpreter.toInt(payload);
    if (!number) {
      diagnostics_.reject(unit_, ast, "invalid integer literal");
    }
    // Literals are represented in the signed storage width of their IR
    // scalar type; signedness remains a source fact at each operation.
    const cxx::Type* storage_type = source_type;
    if (is_address) {
      storage_type = element_type == LOOM_SCALAR_TYPE_INDEX
                         ? static_cast<const cxx::Type*>(
                               unit_.control()->getLongLongIntType())
                         : static_cast<const cxx::Type*>(
                               unit_.control()->getUnsignedLongLongIntType());
    }
    auto bytes = *unit_.control()->memoryLayout()->sizeOf(storage_type);
    attribute = integer_attribute(std::bit_cast<uint64_t>(*number), bytes);
  }
  return attribute;
}

loom_value_id_t Scalars::constant(const cxx::ConstValue& value,
                                  const cxx::Type* source_type, cxx::AST* ast) {
  loom_op_t* op;
  if (auto* floating = std::get_if<cxx::ConstFloat>(&value);
      floating && floating->isNaN()) {
    auto bits = floating->bits();
    auto bytes = floating->bitWidth() / 8;
    auto storage = loom_type_scalar(bytes == 1   ? LOOM_SCALAR_TYPE_I8
                                    : bytes == 2 ? LOOM_SCALAR_TYPE_I16
                                    : bytes == 4 ? LOOM_SCALAR_TYPE_I32
                                                 : LOOM_SCALAR_TYPE_I64);
    check(loom_scalar_constant_build(&builder_, integer_attribute(bits, bytes),
                                     storage, locations_.get(ast), &op));
    check(loom_scalar_bitcast_build(&builder_, loom_op_results(op)[0], storage,
                                    types_.get(source_type, ast),
                                    locations_.get(ast), &op));
    return loom_op_results(op)[0];
  }
  return materialize(constant_attribute(value, source_type, ast),
                     types_.get(source_type, ast), locations_.get(ast));
}

}  // namespace loom::cxx_import
