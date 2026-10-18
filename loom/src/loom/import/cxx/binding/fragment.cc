// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/fragment.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <limits>
#include <string>

#include "loom/import/cxx/binding/index_values.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/auxiliary.h"

namespace loom::cxx_import {
namespace {

uint8_t constant_selector(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          cxx::FunctionSymbol* function, size_t index,
                          std::string_view role, cxx::AST* owner) {
  auto arguments = function->templateArguments();
  if (arguments.size() <= index) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " requires a leading constant template argument");
  }
  auto value = cxx::template_argument_value(arguments[index]);
  auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!number || number->isNegative() || number->toUWide() > UINT8_MAX) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " requires a constant enum template argument");
  }
  return static_cast<uint8_t>(number->toUIntMax());
}

const ViewPartition& require_view(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::View) {
    diagnostics.reject(unit, owner,
                       "fragment memory operation requires a view value");
  }
  return static_cast<const ViewPartition&>(partition);
}

std::vector<loom_string_id_t> require_named_operands(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::Type* type, bool permits_schema, loom_module_t* module,
    cxx::AST* owner) {
  auto* record = types.record(type, owner);
  if (!record || record->component_count != record->members.size()) {
    diagnostics.reject(unit, owner,
                       "fragment named operands must be a flat record");
  }
  const size_t maximum =
      LOOM_ENCODING_AUXILIARY_KEY_COUNT_ + static_cast<size_t>(permits_schema);
  if (record->members.size() > maximum) {
    diagnostics.reject(unit, owner, "too many fragment named operands");
  }
  std::vector<loom_string_id_t> names;
  names.reserve(record->members.size());
  bool has_schema = false;
  bool has_auxiliary = false;
  for (const auto& member : record->members) {
    auto name = cxx::to_string(member.field->name());
    if (name == "schema") {
      const auto& partition = types.partition(member.field->type(), owner);
      if (!permits_schema || partition.kind != ValueKind::Encoding ||
          static_cast<const EncodingPartition&>(partition).role !=
              LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
        diagnostics.reject(unit, owner,
                           "fragment schema must be a schema encoding value");
      }
      has_schema = true;
    } else {
      loom_encoding_auxiliary_key_t key;
      if (!loom_encoding_auxiliary_key_lookup(view(name), &key)) {
        diagnostics.reject(unit, owner,
                           "unknown fragment named operand '" + name + "'");
      }
      if (!types.native_vector(member.field->type())) {
        diagnostics.reject(unit, owner,
                           "fragment auxiliary operands must be vectors");
      }
      has_auxiliary = true;
    }
    loom_string_id_t name_id;
    check(loom_module_intern_string(module, view(name), &name_id));
    names.push_back(name_id);
  }
  if (permits_schema && has_auxiliary && !has_schema) {
    diagnostics.reject(unit, owner, "fragment parameter 'schema' is required");
  }
  return names;
}

loom_value_id_t view_component(Value value) {
  return value.components().back();
}

}  // namespace

std::optional<FragmentIntrinsic::Operation> FragmentIntrinsic::parse_operation(
    std::string_view name) {
  if (name == "vector.fragment") {
    return Operation::Attach;
  }
  if (name == "vector.fragment.repack") {
    return Operation::Repack;
  }
  if (name == "vector.fragment.load") {
    return Operation::Load;
  }
  if (name == "vector.fragment.store") {
    return Operation::Store;
  }
  if (name == "vector.mma") {
    return Operation::Mma;
  }
  return std::nullopt;
}

FragmentIntrinsic::Shape FragmentIntrinsic::require_shape(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::Type* type, cxx::AST* owner) {
  auto* record = types.record(type, owner);
  if (!record || (record->members.size() != 2 && record->members.size() != 3) ||
      record->component_count != record->members.size()) {
    diagnostics.reject(unit, owner,
                       "fragment shape requires flat {rows, columns} or "
                       "{blocks, rows, columns} integer fields");
  }
  const std::array<std::string_view, 3> unblocked = {"rows", "columns", {}};
  const std::array<std::string_view, 3> blocked = {"blocks", "rows", "columns"};
  const auto& expected = record->members.size() == 2 ? unblocked : blocked;
  uint16_t unsigned_mask = 0;
  for (size_t index = 0; index < record->members.size(); ++index) {
    const auto& member = record->members[index];
    auto name = cxx::to_string(member.field->name());
    if (name != expected[index]) {
      diagnostics.reject(unit, owner,
                         "fragment shape requires flat {rows, columns} or "
                         "{blocks, rows, columns} integer fields");
    }
    if (require_index_value(unit, diagnostics, types, member.field->type(),
                            owner, "fragment shape")) {
      unsigned_mask |= uint16_t{1} << index;
    }
  }
  return {
      .rank = static_cast<uint8_t>(record->members.size()),
      .unsigned_mask = unsigned_mask,
  };
}

bool FragmentIntrinsic::supports(std::string_view name) {
  return parse_operation(name).has_value();
}

std::optional<FragmentIntrinsic> FragmentIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    loom_module_t* module, cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto operation = parse_operation(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "fragment operations accept one operation name");
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "fragment operations require fixed source signatures");
  }
  auto parameters = signature->parameterTypes();
  FragmentIntrinsic result(*operation);

  if (*operation == Operation::Mma) {
    if (parameters.size() != 3) {
      diagnostics.reject(unit, owner,
                         "vector.mma requires lhs, rhs, and init vectors");
    }
    result.mma_flags_ =
        constant_selector(unit, diagnostics, function, 0, "vector.mma", owner);
    if ((result.mma_flags_ & ~LOOM_VECTOR_MMAFLAGS_SATURATE) != 0) {
      diagnostics.reject(unit, owner, "vector.mma has unsupported flags");
    }
    auto* result_vector = types.native_vector(signature->returnType());
    if (!result_vector || !types.native_vector(parameters[0]) ||
        !types.native_vector(parameters[1]) ||
        types.native_vector(parameters[2]) != result_vector) {
      diagnostics.reject(
          unit, owner,
          "vector.mma requires vector operands and a result matching init");
    }
    result.result_type_ = types.get(signature->returnType(), owner);
    return result;
  }

  uint8_t role = constant_selector(unit, diagnostics, function, 0,
                                   "fragment operation", owner);
  if (role >= LOOM_VECTOR_ROLE_COUNT_) {
    diagnostics.reject(unit, owner, "fragment operation has an unknown role");
  }
  result.role_ = static_cast<loom_vector_role_t>(role);

  size_t shape_index = 0;
  switch (*operation) {
    case Operation::Attach: {
      if (parameters.size() != 2 && parameters.size() != 3) {
        diagnostics.reject(
            unit, owner,
            "vector.fragment requires data, shape, and optional parameters");
      }
      auto* data = types.native_vector(parameters[0]);
      auto* output = types.native_vector(signature->returnType());
      if (!data || data != output) {
        diagnostics.reject(unit, owner,
                           "vector.fragment must preserve its vector type");
      }
      shape_index = 1;
      if (parameters.size() == 3) {
        result.named_operand_names_ = require_named_operands(
            unit, diagnostics, types, parameters[2], true, module, owner);
      }
      break;
    }
    case Operation::Repack: {
      if (parameters.size() != 2 || !types.native_vector(parameters[0]) ||
          !types.native_vector(signature->returnType())) {
        diagnostics.reject(
            unit, owner,
            "vector.fragment.repack requires source and result vectors");
      }
      shape_index = 1;
      break;
    }
    case Operation::Load: {
      if (parameters.size() != 3 && parameters.size() != 4) {
        diagnostics.reject(
            unit, owner,
            "vector.fragment.load requires view, origin, shape, and optional "
            "auxiliaries");
      }
      result.view_ =
          &require_view(unit, diagnostics, types, parameters[0], owner);
      if (result.view_->access_flags != 0) {
        diagnostics.reject(unit, owner,
                           "vector.fragment.load does not support volatile "
                           "views");
      }
      result.origin_unsigned_mask_ = require_index_record(
          unit, diagnostics, types, parameters[1], result.view_->extents.size(),
          owner, "fragment origin");
      if (!types.native_vector(signature->returnType())) {
        diagnostics.reject(unit, owner,
                           "vector.fragment.load must return a vector");
      }
      shape_index = 2;
      if (parameters.size() == 4) {
        result.named_operand_names_ = require_named_operands(
            unit, diagnostics, types, parameters[3], false, module, owner);
      }
      break;
    }
    case Operation::Store: {
      if (parameters.size() != 4 ||
          !unit.typeTraits().is_void(signature->returnType()) ||
          !types.native_vector(parameters[0])) {
        diagnostics.reject(
            unit, owner,
            "vector.fragment.store requires value, view, origin, and shape");
      }
      result.view_ =
          &require_view(unit, diagnostics, types, parameters[1], owner);
      if (unit.typeTraits().is_const(result.view_->element_type)) {
        diagnostics.reject(unit, owner,
                           "vector.fragment.store requires a mutable view");
      }
      if (result.view_->access_flags != 0) {
        diagnostics.reject(unit, owner,
                           "vector.fragment.store does not support volatile "
                           "views");
      }
      result.origin_unsigned_mask_ = require_index_record(
          unit, diagnostics, types, parameters[2], result.view_->extents.size(),
          owner, "fragment origin");
      shape_index = 3;
      break;
    }
    case Operation::Mma:
      IREE_ASSERT_UNREACHABLE("vector.mma returned above");
      IREE_BUILTIN_UNREACHABLE();
  }
  result.shape_ =
      require_shape(unit, diagnostics, types, parameters[shape_index], owner);
  if (*operation != Operation::Store) {
    result.result_type_ = types.get(signature->returnType(), owner);
  }
  return result;
}

std::optional<Value> FragmentIntrinsic::call(
    std::span<const Value> arguments, loom_builder_t* builder,
    loom_location_id_t location) const {
  auto cast_components = [&](Value record, uint16_t unsigned_mask,
                             std::span<loom_value_id_t> output) {
    auto components = record.components();
    for (size_t index = 0; index < output.size(); ++index) {
      output[index] = cast_index(Value(components[index]),
                                 is_unsigned_component(unsigned_mask, index),
                                 builder, location);
    }
  };
  loom_op_t* op;
  if (operation_ == Operation::Mma) {
    check(loom_vector_mma_build(builder, mma_flags_, arguments[0].ssa(),
                                arguments[1].ssa(), arguments[2].ssa(),
                                result_type_, location, &op));
    return Value(loom_op_results(op)[0]);
  }

  std::array<loom_value_id_t, 3> shape_values;
  cast_components(arguments[operation_ == Operation::Attach ||
                                    operation_ == Operation::Repack
                                ? 1
                            : operation_ == Operation::Load ? 2
                                                            : 3],
                  shape_.unsigned_mask,
                  std::span(shape_values).first(shape_.rank));
  const bool blocked = shape_.rank == 3;
  const loom_value_id_t blocks =
      blocked ? shape_values[0] : LOOM_VALUE_ID_INVALID;
  const loom_value_id_t rows = shape_values[shape_.rank - 2];
  const loom_value_id_t columns = shape_values[shape_.rank - 1];

  std::array<loom_named_value_t, LOOM_ENCODING_AUXILIARY_KEY_COUNT_ + 1>
      named_operands;
  auto bind_named = [&](Value record) {
    auto components = record.components();
    for (size_t index = 0; index < named_operand_names_.size(); ++index) {
      named_operands[index] = {named_operand_names_[index], 0,
                               components[index]};
    }
  };

  switch (operation_) {
    case Operation::Attach: {
      if (!named_operand_names_.empty()) {
        bind_named(arguments[2]);
      }
      check(loom_vector_fragment_build(
          builder, blocked ? LOOM_VECTOR_FRAGMENT_BUILD_FLAG_HAS_BLOCKS : 0,
          role_, arguments[0].ssa(), blocks, rows, columns,
          named_operands.data(), named_operand_names_.size(), nullptr, 0,
          result_type_, location, &op));
      return Value(loom_op_results(op)[0]);
    }
    case Operation::Repack:
      check(loom_vector_fragment_repack_build(
          builder,
          blocked ? LOOM_VECTOR_FRAGMENT_REPACK_BUILD_FLAG_HAS_BLOCKS : 0,
          role_, arguments[0].ssa(), blocks, rows, columns, result_type_,
          location, &op));
      return Value(loom_op_results(op)[0]);
    case Operation::Load: {
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
      cast_components(arguments[1], origin_unsigned_mask_,
                      std::span(indices).first(view_->extents.size()));
      std::array<int64_t, LOOM_TYPE_MAX_RANK> static_indices;
      static_indices.fill(std::numeric_limits<int64_t>::min());
      if (!named_operand_names_.empty()) {
        bind_named(arguments[3]);
      }
      auto build_flags =
          blocked ? LOOM_VECTOR_FRAGMENT_LOAD_BUILD_FLAG_HAS_BLOCKS : 0;
      check(loom_vector_fragment_load_build(
          builder, build_flags, role_, view_component(arguments[0]),
          indices.data(), view_->extents.size(), static_indices.data(),
          view_->extents.size(), blocks, rows, columns, named_operands.data(),
          named_operand_names_.size(), 0, 0, result_type_, location, &op));
      return Value(loom_op_results(op)[0]);
    }
    case Operation::Store: {
      std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK> indices;
      cast_components(arguments[2], origin_unsigned_mask_,
                      std::span(indices).first(view_->extents.size()));
      std::array<int64_t, LOOM_TYPE_MAX_RANK> static_indices;
      static_indices.fill(std::numeric_limits<int64_t>::min());
      auto build_flags =
          blocked ? LOOM_VECTOR_FRAGMENT_STORE_BUILD_FLAG_HAS_BLOCKS : 0;
      check(loom_vector_fragment_store_build(
          builder, build_flags, role_, arguments[0].ssa(),
          view_component(arguments[1]), indices.data(), view_->extents.size(),
          static_indices.data(), view_->extents.size(), blocks, rows, columns,
          0, 0, location, &op));
      return std::nullopt;
    }
    case Operation::Mma:
      IREE_ASSERT_UNREACHABLE("vector.mma returned above");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("unknown fragment operation");
  IREE_BUILTIN_UNREACHABLE();
}

bool FragmentIntrinsic::equivalent(const FragmentIntrinsic& other) const {
  return operation_ == other.operation_ && role_ == other.role_ &&
         mma_flags_ == other.mma_flags_ && shape_ == other.shape_ &&
         origin_unsigned_mask_ == other.origin_unsigned_mask_ &&
         view_ == other.view_ &&
         loom_type_equal(result_type_, other.result_type_) &&
         named_operand_names_ == other.named_operand_names_;
}

}  // namespace loom::cxx_import
