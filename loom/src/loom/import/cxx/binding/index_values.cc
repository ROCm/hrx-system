// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/index_values.h"

#include <cxx/control.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

static_assert(LOOM_TYPE_MAX_RANK <= std::numeric_limits<uint16_t>::digits);

using AddressOperation = AddressIntrinsic::Operation;

std::optional<AddressOperation> parse_operation(std::string_view name) {
  static constexpr std::pair<std::string_view, AddressOperation> operations[] =
      {
          {"index.add", AddressOperation::Add},
          {"index.andi", AddressOperation::And},
          {"index.cast", AddressOperation::Cast},
          {"index.cmp", AddressOperation::Compare},
          {"index.ctlzi", AddressOperation::CountLeadingZeros},
          {"index.ctpopi", AddressOperation::CountPopulation},
          {"index.cttzi", AddressOperation::CountTrailingZeros},
          {"index.div", AddressOperation::Div},
          {"index.divrem", AddressOperation::DivRem},
          {"index.madd", AddressOperation::Madd},
          {"index.max", AddressOperation::Max},
          {"index.min", AddressOperation::Min},
          {"index.mul", AddressOperation::Mul},
          {"index.ori", AddressOperation::Or},
          {"index.rem", AddressOperation::Rem},
          {"index.rotli", AddressOperation::RotateLeft},
          {"index.rotri", AddressOperation::RotateRight},
          {"index.scale", AddressOperation::Scale},
          {"index.shli", AddressOperation::ShiftLeft},
          {"index.shrsi", AddressOperation::ShiftRightSigned},
          {"index.shrui", AddressOperation::ShiftRightUnsigned},
          {"index.sub", AddressOperation::Sub},
          {"index.xori", AddressOperation::Xor},
      };
  for (auto [candidate, operation] : operations) {
    if (name == candidate) {
      return operation;
    }
  }
  return std::nullopt;
}

std::optional<loom_index_cmp_predicate_t> parse_predicate(
    std::string_view name) {
  static constexpr std::pair<std::string_view, loom_index_cmp_predicate_t>
      predicates[] = {
          {"eq", LOOM_INDEX_CMP_PREDICATE_EQ},
          {"ne", LOOM_INDEX_CMP_PREDICATE_NE},
          {"slt", LOOM_INDEX_CMP_PREDICATE_SLT},
          {"sle", LOOM_INDEX_CMP_PREDICATE_SLE},
          {"sgt", LOOM_INDEX_CMP_PREDICATE_SGT},
          {"sge", LOOM_INDEX_CMP_PREDICATE_SGE},
          {"ult", LOOM_INDEX_CMP_PREDICATE_ULT},
          {"ule", LOOM_INDEX_CMP_PREDICATE_ULE},
          {"ugt", LOOM_INDEX_CMP_PREDICATE_UGT},
          {"uge", LOOM_INDEX_CMP_PREDICATE_UGE},
      };
  for (auto [candidate, predicate] : predicates) {
    if (name == candidate) {
      return predicate;
    }
  }
  return std::nullopt;
}

void require_index_components(cxx::TranslationUnit& unit,
                              Diagnostics& diagnostics, Types& types,
                              const cxx::Type* type, const Partition& partition,
                              cxx::AST* owner, std::string_view role,
                              uint16_t& unsigned_component_mask,
                              size_t& component_index) {
  switch (partition.kind) {
    case ValueKind::SSA: {
      if (require_index_value(unit, diagnostics, types, type, owner, role)) {
        unsigned_component_mask = static_cast<uint16_t>(
            unsigned_component_mask | (uint16_t{1} << component_index));
      }
      ++component_index;
      return;
    }
    case ValueKind::Address:
      if (!types.is_index(type, owner)) {
        diagnostics.reject(
            unit, owner, std::string(role) + " requires logical index fields");
      }
      ++component_index;
      return;
    case ValueKind::Record: {
      const auto& record = static_cast<const RecordPartition&>(partition);
      for (const auto& member : record.members) {
        require_index_components(unit, diagnostics, types, member.type(),
                                 *member.partition, owner, role,
                                 unsigned_component_mask, component_index);
      }
      return;
    }
    case ValueKind::Array: {
      const auto& array = static_cast<const ArrayPartition&>(partition);
      for (size_t index = 0; index < array.source->size(); ++index) {
        require_index_components(unit, diagnostics, types,
                                 array.source->elementType(), *array.element,
                                 owner, role, unsigned_component_mask,
                                 component_index);
      }
      return;
    }
    default:
      diagnostics.reject(unit, owner,
                         std::string(role) + " requires integer fields");
  }
}

}  // namespace

bool require_integral(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                      Types& types, const cxx::Type* type, cxx::AST* owner,
                      std::string_view role) {
  auto projected = types.get(type, owner);
  if (!unit.typeTraits().is_integral(type) ||
      loom_type_kind(projected) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(projected) == LOOM_SCALAR_TYPE_I1) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " requires an integer value");
  }
  return types.is_unsigned(type);
}

bool require_index_value(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         Types& types, const cxx::Type* type, cxx::AST* owner,
                         std::string_view role) {
  return types.is_index(type, owner)
             ? false
             : require_integral(unit, diagnostics, types, type, owner, role);
}

uint16_t require_index_record(cxx::TranslationUnit& unit,
                              Diagnostics& diagnostics, Types& types,
                              const cxx::Type* type, size_t count,
                              cxx::AST* owner, std::string_view role) {
  auto* record = types.record(type, owner);
  if (!record || record->component_count != count) {
    diagnostics.reject(unit, owner, std::string(role) + " has the wrong arity");
  }
  uint16_t unsigned_component_mask = 0;
  size_t component_index = 0;
  require_index_components(unit, diagnostics, types, type, *record, owner, role,
                           unsigned_component_mask, component_index);
  return unsigned_component_mask;
}

void require_logical_index_record(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type, size_t count,
                                  cxx::AST* owner, std::string_view role) {
  auto* record = types.record(type, owner);
  if (!record || record->component_count != count) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " have the wrong arity");
  }
  std::vector<loom_type_t> component_types;
  types.append(type, owner, component_types);
  const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (!std::all_of(component_types.begin(), component_types.end(),
                   [&](loom_type_t component_type) {
                     return loom_type_equal(component_type, index_type);
                   })) {
    diagnostics.reject(unit, owner,
                       std::string(role) + " require logical indices");
  }
}

bool is_unsigned_component(uint32_t mask, size_t index) {
  return (mask & (uint32_t{1} << index)) != 0;
}

loom_value_id_t cast_index(Value value, bool unsigned_source,
                           loom_builder_t* builder,
                           loom_location_id_t location) {
  auto input = value.ssa();
  auto input_type = loom_module_value_type(builder->module, input);
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(input_type, index_type)) {
    return input;
  }
  loom_op_t* op;
  auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  if (unsigned_source && !loom_type_equal(input_type, i64_type)) {
    check(loom_scalar_extui_build(builder, input, input_type, i64_type,
                                  location, &op));
    input = loom_op_results(op)[0];
    input_type = i64_type;
  }
  check(loom_index_cast_build(builder, input, input_type, index_type, location,
                              &op));
  return loom_op_results(op)[0];
}

bool AddressIntrinsic::supports(std::string_view name) {
  return parse_operation(name).has_value();
}

std::optional<AddressIntrinsic> AddressIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto operation = parse_operation(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "address operations require fixed source signatures");
  }
  auto parameters = signature->parameterTypes();
  auto require_count = [&](size_t expected) {
    if (parameters.size() != expected) {
      diagnostics.reject(unit, owner,
                         std::string(attribute.arguments[0]->name()) +
                             " requires " + std::to_string(expected) +
                             (expected == 1 ? " operand" : " operands"));
    }
  };
  auto domain = [&](const cxx::Type* type) {
    return types.address_domain(type, owner);
  };
  auto require_domain = [&](const cxx::Type* type,
                            loom_scalar_type_t expected) {
    if (domain(type) != expected) {
      diagnostics.reject(unit, owner,
                         std::string(attribute.arguments[0]->name()) +
                             (expected == LOOM_SCALAR_TYPE_INDEX
                                  ? " requires logical index values"
                                  : " requires physical offset values"));
    }
  };

  auto predicate = LOOM_INDEX_CMP_PREDICATE_EQ;
  if (*operation == Operation::Compare) {
    auto selected = attribute.arguments.size() == 2
                        ? parse_predicate(attribute.arguments[1]->name())
                        : std::nullopt;
    if (!selected) {
      diagnostics.reject(unit, owner,
                         "index.cmp requires one comparison predicate");
    }
    predicate = *selected;
  } else if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       std::string(attribute.arguments[0]->name()) +
                           " does not accept selectors");
  }

  const auto& result = types.partition(signature->returnType(), owner);
  auto result_type = *operation == Operation::Cast
                         ? types.get(signature->returnType(), owner)
                         : loom_type_none();
  auto result_domain = domain(signature->returnType());
  bool unsigned_source = false;
  loom_scalar_type_t source_domain = LOOM_SCALAR_TYPE_NONE;
  if (*operation == Operation::Cast) {
    require_count(1);
    source_domain = domain(parameters[0]);
    bool result_is_address = result_domain == LOOM_SCALAR_TYPE_INDEX ||
                             result_domain == LOOM_SCALAR_TYPE_OFFSET;
    bool source_is_address = source_domain == LOOM_SCALAR_TYPE_INDEX ||
                             source_domain == LOOM_SCALAR_TYPE_OFFSET;
    if (!result_is_address) {
      require_integral(unit, diagnostics, types, signature->returnType(), owner,
                       "index.cast result");
      if (!source_is_address) {
        diagnostics.reject(
            unit, owner,
            "index.cast requires an address source or result domain");
      }
    } else if (!source_is_address) {
      unsigned_source = require_integral(unit, diagnostics, types,
                                         parameters[0], owner, "index.cast");
    }
    if (source_is_address) {
      unsigned_source = source_domain == LOOM_SCALAR_TYPE_OFFSET;
    }
  } else if (*operation == Operation::Compare) {
    require_count(2);
    auto left_domain = domain(parameters[0]);
    auto right_domain = domain(parameters[1]);
    if ((left_domain != LOOM_SCALAR_TYPE_INDEX &&
         left_domain != LOOM_SCALAR_TYPE_OFFSET) ||
        left_domain != right_domain ||
        types.unqualified(signature->returnType()) !=
            unit.control()->getBoolType()) {
      diagnostics.reject(
          unit, owner,
          "index.cmp requires matching address operands and a bool result");
    }
  } else if (*operation == Operation::Scale) {
    require_count(2);
    require_domain(parameters[0], LOOM_SCALAR_TYPE_INDEX);
    require_domain(parameters[1], LOOM_SCALAR_TYPE_OFFSET);
    require_domain(signature->returnType(), LOOM_SCALAR_TYPE_OFFSET);
  } else if (*operation == Operation::Add || *operation == Operation::Sub) {
    require_count(2);
    auto left_domain = domain(parameters[0]);
    if ((left_domain != LOOM_SCALAR_TYPE_INDEX &&
         left_domain != LOOM_SCALAR_TYPE_OFFSET) ||
        domain(parameters[1]) != left_domain || result_domain != left_domain) {
      diagnostics.reject(
          unit, owner,
          "index.add/sub require one matching address-domain type");
    }
  } else if (*operation == Operation::DivRem) {
    require_count(2);
    require_domain(parameters[0], LOOM_SCALAR_TYPE_INDEX);
    require_domain(parameters[1], LOOM_SCALAR_TYPE_INDEX);
    auto* record = result.kind == ValueKind::Record
                       ? static_cast<const RecordPartition*>(&result)
                       : nullptr;
    if (!record || record->members.size() != 2 ||
        record->component_count != 2 ||
        !types.is_index(record->members[0].type(), owner) ||
        !types.is_index(record->members[1].type(), owner)) {
      diagnostics.reject(unit, owner,
                         "index.divrem requires a two-index result record");
    }
  } else {
    size_t expected =
        *operation == Operation::Madd
            ? 3
            : (*operation == Operation::CountLeadingZeros ||
                       *operation == Operation::CountTrailingZeros ||
                       *operation == Operation::CountPopulation
                   ? 1
                   : 2);
    require_count(expected);
    for (const auto* parameter : parameters) {
      require_domain(parameter, LOOM_SCALAR_TYPE_INDEX);
    }
    require_domain(signature->returnType(), LOOM_SCALAR_TYPE_INDEX);
  }
  return AddressIntrinsic(*operation, predicate, unsigned_source, source_domain,
                          result_type, &result);
}

Value AddressIntrinsic::call(std::span<const Value> arguments,
                             ValueArena& arena, loom_builder_t* builder,
                             loom_location_id_t location) const {
  loom_op_t* op;
  if (operation_ == Operation::Cast) {
    auto input = arguments[0].ssa();
    auto input_type = loom_module_value_type(builder->module, input);
    if (loom_type_equal(input_type, result_type_)) {
      return arena.capture(*result_partition_, {&input, 1});
    }
    auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    if (loom_type_element_type(result_type_) == LOOM_SCALAR_TYPE_INDEX &&
        source_domain_ == LOOM_SCALAR_TYPE_NONE && unsigned_source_ &&
        !loom_type_equal(input_type, i64_type)) {
      check(loom_scalar_extui_build(builder, input, input_type, i64_type,
                                    location, &op));
      input = loom_op_results(op)[0];
      input_type = i64_type;
    }
    check(loom_index_cast_build(builder, input, input_type, result_type_,
                                location, &op));
    auto result = loom_op_results(op)[0];
    return result_partition_->kind == ValueKind::Address
               ? arena.capture(*result_partition_, {&result, 1})
               : Value(result);
  }

  std::array<loom_value_id_t, 3> operands = {};
  for (size_t index = 0; index < arguments.size(); ++index) {
    operands[index] = arguments[index].ssa();
  }
  switch (operation_) {
    case Operation::Cast:
      IREE_ASSERT_UNREACHABLE("cast handled above");
      IREE_BUILTIN_UNREACHABLE();
    case Operation::Add:
      check(loom_index_add_build(
          builder, operands[0], operands[1],
          loom_module_value_type(builder->module, operands[0]), location, &op));
      break;
    case Operation::Sub:
      check(loom_index_sub_build(
          builder, operands[0], operands[1],
          loom_module_value_type(builder->module, operands[0]), location, &op));
      break;
    case Operation::Mul:
      check(loom_index_mul_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::Scale:
      check(loom_index_scale_build(builder, operands[0], operands[1],
                                   loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                   location, &op));
      break;
    case Operation::Div:
      check(loom_index_div_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::Rem:
      check(loom_index_rem_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::DivRem:
      check(loom_index_divrem_build(builder, operands[0], operands[1], location,
                                    &op));
      break;
    case Operation::Min:
      check(loom_index_min_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::Max:
      check(loom_index_max_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::Madd:
      check(loom_index_madd_build(builder, operands[0], operands[1],
                                  operands[2], location, &op));
      break;
    case Operation::And:
      check(loom_index_andi_build(builder, operands[0], operands[1], location,
                                  &op));
      break;
    case Operation::Or:
      check(loom_index_ori_build(builder, operands[0], operands[1], location,
                                 &op));
      break;
    case Operation::Xor:
      check(loom_index_xori_build(builder, operands[0], operands[1], location,
                                  &op));
      break;
    case Operation::ShiftLeft:
      check(loom_index_shli_build(builder, operands[0], operands[1], location,
                                  &op));
      break;
    case Operation::ShiftRightSigned:
      check(loom_index_shrsi_build(builder, operands[0], operands[1], location,
                                   &op));
      break;
    case Operation::ShiftRightUnsigned:
      check(loom_index_shrui_build(builder, operands[0], operands[1], location,
                                   &op));
      break;
    case Operation::RotateLeft:
      check(loom_index_rotli_build(builder, operands[0], operands[1], location,
                                   &op));
      break;
    case Operation::RotateRight:
      check(loom_index_rotri_build(builder, operands[0], operands[1], location,
                                   &op));
      break;
    case Operation::CountLeadingZeros:
      check(loom_index_ctlzi_build(builder, operands[0], location, &op));
      break;
    case Operation::CountTrailingZeros:
      check(loom_index_cttzi_build(builder, operands[0], location, &op));
      break;
    case Operation::CountPopulation:
      check(loom_index_ctpopi_build(builder, operands[0], location, &op));
      break;
    case Operation::Compare:
      check(loom_index_cmp_build(builder, predicate_, operands[0], operands[1],
                                 location, &op));
      break;
  }
  return arena.capture(
      *result_partition_,
      {loom_op_results(op), result_partition_->component_count});
}

}  // namespace loom::cxx_import
