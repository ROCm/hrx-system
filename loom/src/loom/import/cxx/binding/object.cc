// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/object.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <array>
#include <utility>

#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/ref/ops.h"

namespace loom::cxx_import {

std::optional<ObjectIntrinsic::Operation> ObjectIntrinsic::find(
    std::string_view name) {
  static constexpr std::pair<std::string_view, Operation> operations[] = {
      {"ref.null", Operation::Null},
      {"ref.duplicate", Operation::Duplicate},
      {"ref.borrow", Operation::Borrow},
      {"ref.release", Operation::Release},
      {"ref.compare.null", Operation::CompareNull},
      {"ref.compare.eq", Operation::CompareEqual},
  };
  for (const auto& [spelling, operation] : operations) {
    if (name == spelling) {
      return operation;
    }
  }
  return std::nullopt;
}

bool ObjectIntrinsic::supports(std::string_view name) {
  return find(name).has_value();
}

std::optional<ObjectIntrinsic> ObjectIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto operation = find(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  const size_t operand_count = *operation == Operation::Null           ? 0
                               : *operation == Operation::CompareEqual ? 2
                                                                       : 1;
  if (attribute.arguments.size() != 1 || !signature ||
      signature->isVariadic() ||
      signature->parameterTypes().size() != operand_count) {
    diagnostics.reject(unit, owner,
                       "object operation has an invalid signature");
  }
  std::array<loom_type_t, 2> operands;
  size_t index = 0;
  for (auto* parameter : signature->parameterTypes()) {
    const auto& partition = types.partition(parameter, owner);
    if (partition.kind != ValueKind::Object) {
      diagnostics.reject(unit, owner,
                         "object operation requires handle operands");
    }
    operands[index++] = static_cast<const ObjectPartition&>(partition).type;
  }
  const bool returns_void = unit.typeTraits().is_void(signature->returnType());
  if (returns_void != (*operation == Operation::Release)) {
    diagnostics.reject(unit, owner, "object operation has an invalid result");
  }
  if (returns_void) {
    return ObjectIntrinsic(*operation, nullptr, {});
  }
  const auto& result = types.partition(signature->returnType(), owner);
  const auto result_type = types.get(signature->returnType(), owner);
  const bool comparison = *operation == Operation::CompareNull ||
                          *operation == Operation::CompareEqual;
  if (comparison) {
    if (!loom_type_equal(result_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1))) {
      diagnostics.reject(unit, owner,
                         "object comparison requires a bool result");
    }
  } else if (result.kind != ValueKind::Object) {
    diagnostics.reject(unit, owner,
                       "object operation requires a handle result");
  }
  if (*operation == Operation::Duplicate &&
      !loom_type_equal(operands[0], result_type)) {
    diagnostics.reject(
        unit, owner,
        "object duplication must preserve handle type and ownership");
  }
  if (*operation == Operation::Borrow ||
      *operation == Operation::CompareEqual) {
    const auto expected =
        loom_type_reference(*operation == Operation::Borrow
                                ? LOOM_REFERENCE_OWNERSHIP_BORROWED
                                : loom_type_reference_ownership(operands[1]),
                            loom_type_reference_namespace_id(operands[0]),
                            loom_type_reference_name_id(operands[0]),
                            loom_type_reference_declaration_id(operands[0]));
    if (!loom_type_equal(
            *operation == Operation::Borrow ? result_type : operands[1],
            expected) ||
        (*operation == Operation::Borrow &&
         loom_type_reference_ownership(operands[0]) !=
             LOOM_REFERENCE_OWNERSHIP_OWNED)) {
      diagnostics.reject(
          unit, owner,
          "object operation must preserve payload identity and ownership");
    }
  }
  return ObjectIntrinsic(*operation, &result, result_type);
}

std::optional<Value> ObjectIntrinsic::call(std::span<const Value> arguments,
                                           ValueArena& arena,
                                           loom_builder_t* builder,
                                           loom_location_id_t location) const {
  loom_op_t* op;
  switch (operation_) {
    case Operation::Null:
      check(loom_ref_null_build(builder, result_type_, location, &op));
      break;
    case Operation::Duplicate:
      check(loom_ref_duplicate_build(builder, arguments[0].components().data(),
                                     1, &result_type_, 1, location, &op));
      break;
    case Operation::Borrow:
      check(loom_ref_borrow_build(builder, arguments[0].components()[0],
                                  result_type_, location, &op));
      break;
    case Operation::Release:
      check(loom_ref_release_build(builder, arguments[0].components()[0],
                                   location, &op));
      return std::nullopt;
    case Operation::CompareNull:
      check(loom_ref_compare_null_build(builder, arguments[0].components()[0],
                                        location, &op));
      break;
    case Operation::CompareEqual:
      check(loom_ref_compare_eq_build(builder, arguments[0].components()[0],
                                      arguments[1].components()[0], location,
                                      &op));
      break;
  }
  return arena.capture(*result_, std::span(loom_op_results(op), 1));
}

}  // namespace loom::cxx_import
