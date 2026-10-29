// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/vector_arithmetic.h"

#include <cxx/names.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <string_view>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/scalar_bindings.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {

using Family = VectorArithmeticIntrinsic::Family;
using Kind = VectorArithmeticIntrinsic::Kind;
using Operation = VectorArithmeticIntrinsic::Operation;

std::optional<Family> parse_operation(std::string_view name) {
  if (name == "vector.add") {
    return Family::Add;
  }
  if (name == "vector.sub") {
    return Family::Sub;
  }
  if (name == "vector.mul") {
    return Family::Mul;
  }
  if (name == "vector.div") {
    return Family::Div;
  }
  if (name == "vector.rem") {
    return Family::Rem;
  }
  if (name == "vector.neg") {
    return Family::Neg;
  }
  if (name == "vector.and") {
    return Family::And;
  }
  if (name == "vector.or") {
    return Family::Or;
  }
  if (name == "vector.xor") {
    return Family::Xor;
  }
  if (name == "vector.complement") {
    return Family::Complement;
  }
  if (name == "vector.shl") {
    return Family::Shl;
  }
  if (name == "vector.shr") {
    return Family::Shr;
  }
  return std::nullopt;
}

bool is_unary(Family family) {
  return family == Family::Neg || family == Family::Complement;
}

bool is_unary(Kind kind) {
  return kind == Kind::NegF || kind == Kind::NegI || kind == Kind::ComplementI;
}

bool requires_integer(Family family) {
  switch (family) {
    case Family::And:
    case Family::Or:
    case Family::Xor:
    case Family::Complement:
    case Family::Shl:
    case Family::Shr:
      return true;
    default:
      return false;
  }
}

Kind resolve_float(Family family) {
  switch (family) {
    case Family::Add:
      return Kind::AddF;
    case Family::Sub:
      return Kind::SubF;
    case Family::Mul:
      return Kind::MulF;
    case Family::Div:
      return Kind::DivF;
    case Family::Rem:
      return Kind::RemF;
    case Family::Neg:
      return Kind::NegF;
    default:
      IREE_ASSERT_UNREACHABLE("integer-only vector arithmetic family");
      IREE_BUILTIN_UNREACHABLE();
  }
}

Kind resolve_integer(Family family, bool is_unsigned) {
  switch (family) {
    case Family::Add:
      return Kind::AddI;
    case Family::Sub:
      return Kind::SubI;
    case Family::Mul:
      return Kind::MulI;
    case Family::Div:
      return is_unsigned ? Kind::DivUI : Kind::DivSI;
    case Family::Rem:
      return is_unsigned ? Kind::RemUI : Kind::RemSI;
    case Family::Neg:
      return Kind::NegI;
    case Family::And:
      return Kind::AndI;
    case Family::Or:
      return Kind::OrI;
    case Family::Xor:
      return Kind::XorI;
    case Family::Complement:
      return Kind::ComplementI;
    case Family::Shl:
      return Kind::ShlI;
    case Family::Shr:
      return is_unsigned ? Kind::ShrUI : Kind::ShrSI;
    default:
      IREE_ASSERT_UNREACHABLE("unknown vector arithmetic family");
      IREE_BUILTIN_UNREACHABLE();
  }
}

}  // namespace

std::optional<VectorArithmeticIntrinsic::Operation>
VectorArithmeticIntrinsic::admit(cxx::TranslationUnit& unit,
                                 Diagnostics& diagnostics,
                                 const cxx::Attribute& attribute,
                                 cxx::AST* owner) {
  auto kind = parse_operation(attribute.arguments[0]->name());
  if (!kind) {
    return std::nullopt;
  }
  Operation operation{*kind};
  if (requires_integer(*kind) && attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "integer vector arithmetic has no semantic arguments");
  }
  for (size_t index = 1; index < attribute.arguments.size(); ++index) {
    uint8_t flag;
    if (!loom_cxx_fastmath_flag_parse(view(attribute.arguments[index]->name()),
                                      &flag)) {
      diagnostics.reject(unit, owner,
                         "intrinsic has an unsupported fast-math flag");
    }
    operation.flags |= flag;
  }
  return operation;
}

VectorArithmeticIntrinsic VectorArithmeticIntrinsic::resolve(
    Operation operation, cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    Types& types, const cxx::FunctionType* signature, cxx::AST* owner) {
  if (signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "vector arithmetic requires a fixed source signature");
  }
  auto parameters = signature->parameterTypes();
  const size_t expected_count = is_unary(operation.family) ? 1 : 2;
  if (parameters.size() != expected_count) {
    diagnostics.reject(unit, owner,
                       "vector arithmetic has the wrong operand count");
  }

  auto require_vector = [&](const cxx::Type* type) -> const VectorPartition& {
    const auto& partition = types.partition(type, owner);
    if (partition.kind != ValueKind::Vector) {
      diagnostics.reject(unit, owner,
                         "vector arithmetic requires semantic vector values");
    }
    return static_cast<const VectorPartition&>(partition);
  };
  const auto& result = require_vector(signature->returnType());
  for (const auto* parameter : parameters) {
    if (&require_vector(parameter) != &result) {
      diagnostics.reject(
          unit, owner,
          "vector arithmetic operands and result must have one vector type");
    }
  }

  const auto* element = types.unqualified(result.element_type);
  if (types.is_float(element)) {
    if (requires_integer(operation.family)) {
      diagnostics.reject(unit, owner,
                         "integer vector arithmetic requires non-boolean "
                         "integer elements");
    }
    return VectorArithmeticIntrinsic(resolve_float(operation.family),
                                     operation.flags, &result);
  }
  if (!unit.typeTraits().is_integral(element) ||
      (element->kind() == cxx::TypeKind::kBool &&
       operation.family != Family::And && operation.family != Family::Or &&
       operation.family != Family::Xor &&
       operation.family != Family::Complement)) {
    diagnostics.reject(unit, owner,
                       "vector arithmetic requires floating-point or "
                       "integer elements; boolean vectors support bitwise "
                       "operations only");
  }
  if (operation.flags) {
    diagnostics.reject(unit, owner,
                       "integer vector arithmetic has no semantic arguments");
  }
  return VectorArithmeticIntrinsic(
      resolve_integer(operation.family, types.is_unsigned(element)), 0,
      &result);
}

Value VectorArithmeticIntrinsic::call(std::span<const Value> arguments,
                                      ValueArena& arena, uint8_t math_flags,
                                      loom_builder_t* builder,
                                      loom_location_id_t location) const {
  auto left_components = arguments[0].components();
  auto left_dimensions = left_components.first(vector_->component_count - 1);
  BoundTypeStorage result_type_storage;
  const auto result_type =
      vector_->bind_type(left_dimensions, result_type_storage);
  const auto left = vector_->vector_value(left_components);

  loom_value_id_t right = LOOM_VALUE_ID_INVALID;
  if (!is_unary(kind_)) {
    auto right_components = arguments[1].components();
    auto right_dimensions =
        right_components.first(vector_->component_count - 1);
    BoundTypeStorage right_type_storage;
    const auto right_type =
        vector_->bind_type(right_dimensions, right_type_storage);
    right = vector_->vector_value(right_components);
    if (!loom_type_equal(right_type, result_type)) {
      loom_op_t* refine;
      check(loom_vector_refine_build(builder, right, result_type, location,
                                     &refine));
      right = loom_op_results(refine)[0];
    }
  }

  loom_op_t* op;
  const uint8_t flags = flags_ | math_flags;
  switch (kind_) {
    case Kind::AddF:
      check(loom_vector_addf_build(builder, flags, left, right, result_type,
                                   location, &op));
      break;
    case Kind::SubF:
      check(loom_vector_subf_build(builder, flags, left, right, result_type,
                                   location, &op));
      break;
    case Kind::MulF:
      check(loom_vector_mulf_build(builder, flags, left, right, result_type,
                                   location, &op));
      break;
    case Kind::DivF:
      check(loom_vector_divf_build(builder, flags, left, right, result_type,
                                   location, &op));
      break;
    case Kind::RemF:
      check(loom_vector_remf_build(builder, flags, left, right, result_type,
                                   location, &op));
      break;
    case Kind::NegF:
      check(loom_vector_negf_build(builder, flags, left, result_type, location,
                                   &op));
      break;
    case Kind::AddI:
      check(loom_vector_addi_build(builder, 0, left, right, result_type,
                                   location, &op));
      break;
    case Kind::SubI:
      check(loom_vector_subi_build(builder, 0, left, right, result_type,
                                   location, &op));
      break;
    case Kind::MulI:
      check(loom_vector_muli_build(builder, 0, left, right, result_type,
                                   location, &op));
      break;
    case Kind::DivSI:
      check(loom_vector_divsi_build(builder, left, right, result_type, location,
                                    &op));
      break;
    case Kind::DivUI:
      check(loom_vector_divui_build(builder, left, right, result_type, location,
                                    &op));
      break;
    case Kind::RemSI:
      check(loom_vector_remsi_build(builder, left, right, result_type, location,
                                    &op));
      break;
    case Kind::RemUI:
      check(loom_vector_remui_build(builder, left, right, result_type, location,
                                    &op));
      break;
    case Kind::NegI:
      check(loom_vector_negi_build(builder, left, result_type, location, &op));
      break;
    case Kind::AndI:
      check(loom_vector_andi_build(builder, left, right, result_type, location,
                                   &op));
      break;
    case Kind::OrI:
      check(loom_vector_ori_build(builder, left, right, result_type, location,
                                  &op));
      break;
    case Kind::XorI:
      check(loom_vector_xori_build(builder, left, right, result_type, location,
                                   &op));
      break;
    case Kind::ComplementI:
      check(loom_vector_splati_build(builder, -1, result_type, location, &op));
      check(loom_vector_xori_build(builder, left, loom_op_results(op)[0],
                                   result_type, location, &op));
      break;
    case Kind::ShlI:
      check(loom_vector_shli_build(builder, 0, left, right, result_type,
                                   location, &op));
      break;
    case Kind::ShrSI:
      check(loom_vector_shrsi_build(builder, left, right, result_type, location,
                                    &op));
      break;
    case Kind::ShrUI:
      check(loom_vector_shrui_build(builder, left, right, result_type, location,
                                    &op));
      break;
  }

  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> components;
  std::copy(left_dimensions.begin(), left_dimensions.end(), components.begin());
  components[left_dimensions.size()] = loom_op_results(op)[0];
  return arena.capture(*vector_, std::span<const loom_value_id_t>(components)
                                     .first(left_dimensions.size() + 1));
}

}  // namespace loom::cxx_import
