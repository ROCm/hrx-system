// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/async.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <array>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

uint64_t constant_selector(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                           cxx::FunctionSymbol* function, size_t index,
                           uint64_t maximum, std::string_view operation,
                           cxx::AST* owner) {
  auto arguments = function->templateArguments();
  if (arguments.size() <= index) {
    diagnostics.reject(unit, owner,
                       std::string(operation) +
                           " requires leading constant template arguments");
  }
  auto value = cxx::template_argument_value(arguments[index]);
  auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!number || number->isNegative() || number->toUWide() > maximum) {
    diagnostics.reject(
        unit, owner,
        std::string(operation) +
            " selectors require nonnegative constant enum values");
  }
  return number->toUWide();
}

const ViewPartition& require_view(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type,
                                  std::string_view operand, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  if (partition.kind != ValueKind::View) {
    diagnostics.reject(
        unit, owner,
        "kernel async " + std::string(operand) + " requires a view value");
  }
  const auto& view = static_cast<const ViewPartition&>(partition);
  if (view.is_columnar()) {
    diagnostics.reject(unit, owner,
                       "kernel async transfers require a primitive column");
  }
  if (view.access_flags != 0) {
    diagnostics.reject(unit, owner,
                       "kernel async transfers do not support volatile views");
  }
  return view;
}

const OpaqueDialectPartition& require_opaque(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::Type* type, std::string_view name, std::string_view role,
    cxx::AST* owner) {
  if (!types.is_opaque_dialect(type, name, owner)) {
    diagnostics.reject(
        unit, owner,
        std::string(role) + " must be loom::type::" +
            (name == "kernel.async.token" ? "async_token" : "async_group"));
  }
  return static_cast<const OpaqueDialectPartition&>(
      types.partition(type, owner));
}

void require_bool(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                  Types& types, const cxx::Type* type, cxx::AST* owner) {
  if (types.unqualified(type)->kind() != cxx::TypeKind::kBool) {
    diagnostics.reject(unit, owner,
                       "kernel async predicate requires a bool value");
  }
}

void require_cluster_mask(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          Types& types, const cxx::Type* type,
                          cxx::AST* owner) {
  if (!loom_type_equal(types.get(type, owner),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I32)) ||
      !types.is_unsigned(type)) {
    diagnostics.reject(
        unit, owner,
        "kernel async cluster participant mask requires unsigned i32");
  }
}

void require_cache_policy(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          loom_cache_scope_t scope,
                          loom_cache_temporal_t temporal,
                          loom_cache_policy_access_t access, cxx::AST* owner) {
  switch (loom_cache_policy_validate(scope, temporal, access)) {
    case LOOM_CACHE_POLICY_ERROR_NONE:
      return;
    case LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE:
      diagnostics.reject(unit, owner, "unknown kernel async cache scope");
    case LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL:
      diagnostics.reject(unit, owner,
                         "unknown kernel async temporal cache policy");
    case LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL:
      diagnostics.reject(
          unit, owner,
          "kernel async load requires a load-compatible temporal policy");
    case LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL:
      diagnostics.reject(
          unit, owner,
          "kernel async store requires a store-compatible temporal policy");
    case LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL:
      diagnostics.reject(unit, owner,
                         "kernel async transfer has an invalid cache policy");
    case LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE:
      diagnostics.reject(
          unit, owner, "last_use requires a cache scope narrower than system");
    case LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE:
      diagnostics.reject(unit, owner, "bypass requires system cache scope");
  }
}

loom_value_id_t view_component(Value value) {
  return value.components().back();
}

}  // namespace

std::optional<AsyncIntrinsic::Operation> AsyncIntrinsic::find(
    std::string_view name) {
  static constexpr std::pair<std::string_view, Operation> operations[] = {
      {"kernel.async.copy", Operation::Copy},
      {"kernel.async.copy.mask", Operation::CopyMask},
      {"kernel.async.gather", Operation::Gather},
      {"kernel.async.gather.mask", Operation::GatherMask},
      {"kernel.async.cluster.gather", Operation::ClusterGather},
      {"kernel.async.cluster.gather.mask", Operation::ClusterGatherMask},
      {"kernel.async.group", Operation::Group},
      {"kernel.async.wait", Operation::Wait},
  };
  for (auto [candidate, operation] : operations) {
    if (candidate == name) {
      return operation;
    }
  }
  return std::nullopt;
}

bool AsyncIntrinsic::supports(std::string_view name) {
  return find(name).has_value();
}

std::optional<AsyncIntrinsic> AsyncIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (attribute.arguments.empty()) {
    return std::nullopt;
  }
  auto operation = find(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "kernel async binding accepts one operation name");
  }

  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "kernel async operations require fixed signatures");
  }
  auto parameters = signature->parameterTypes();
  auto cache_scope = LOOM_CACHE_SCOPE_WORKGROUP;
  auto cache_temporal = LOOM_CACHE_TEMPORAL_REGULAR;
  auto direction = LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP;
  uint16_t newer_groups = 0;
  const OpaqueDialectPartition* result = nullptr;

  if (*operation == Operation::Wait) {
    if (parameters.size() != 1 ||
        types.unqualified(signature->returnType())->kind() !=
            cxx::TypeKind::kVoid) {
      diagnostics.reject(unit, owner,
                         "kernel.async.wait requires void(async_group)");
    }
    require_opaque(unit, diagnostics, types, parameters[0],
                   "kernel.async.group", "kernel.async.wait operand", owner);
    newer_groups = static_cast<uint16_t>(
        constant_selector(unit, diagnostics, function, 0, UINT16_MAX,
                          "kernel.async.wait", owner));
    return AsyncIntrinsic(*operation, cache_scope, cache_temporal, direction,
                          newer_groups, result);
  }

  if (*operation == Operation::Group) {
    result = &require_opaque(unit, diagnostics, types, signature->returnType(),
                             "kernel.async.group", "kernel.async.group result",
                             owner);
    for (const auto* parameter : parameters) {
      require_opaque(unit, diagnostics, types, parameter, "kernel.async.token",
                     "kernel.async.group operand", owner);
    }
    return AsyncIntrinsic(*operation, cache_scope, cache_temporal, direction,
                          newer_groups, result);
  }

  size_t operand_count = 2;
  if (*operation == Operation::CopyMask ||
      *operation == Operation::GatherMask ||
      *operation == Operation::ClusterGather) {
    operand_count = 3;
  } else if (*operation == Operation::ClusterGatherMask) {
    operand_count = 4;
  }
  if (parameters.size() != operand_count) {
    diagnostics.reject(unit, owner,
                       "kernel async declaration has the wrong operand count");
  }
  require_view(unit, diagnostics, types, parameters[0], "source", owner);
  const auto& destination = require_view(unit, diagnostics, types,
                                         parameters[1], "destination", owner);
  if (unit.typeTraits().is_const(destination.element_type)) {
    diagnostics.reject(unit, owner,
                       "kernel async destination requires a mutable view");
  }
  result = &require_opaque(unit, diagnostics, types, signature->returnType(),
                           "kernel.async.token", "kernel async result", owner);

  if (*operation == Operation::Copy || *operation == Operation::CopyMask) {
    auto selected_direction = constant_selector(
        unit, diagnostics, function, 0, UINT8_MAX, "kernel.async.copy", owner);
    if (selected_direction >= LOOM_KERNEL_DIRECTION_COUNT_) {
      diagnostics.reject(unit, owner, "unknown kernel async copy direction");
    }
    direction = static_cast<loom_kernel_direction_t>(selected_direction);
  }
  size_t cache_index =
      *operation == Operation::Copy || *operation == Operation::CopyMask ? 1
                                                                         : 0;
  auto selected_scope =
      constant_selector(unit, diagnostics, function, cache_index, UINT8_MAX,
                        "kernel async transfer", owner);
  auto selected_temporal =
      constant_selector(unit, diagnostics, function, cache_index + 1, UINT8_MAX,
                        "kernel async transfer", owner);
  cache_scope = static_cast<loom_cache_scope_t>(selected_scope);
  cache_temporal = static_cast<loom_cache_temporal_t>(selected_temporal);
  auto access = direction == LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL &&
                        (*operation == Operation::Copy ||
                         *operation == Operation::CopyMask)
                    ? LOOM_CACHE_POLICY_ACCESS_STORE
                    : LOOM_CACHE_POLICY_ACCESS_LOAD;
  require_cache_policy(unit, diagnostics, cache_scope, cache_temporal, access,
                       owner);

  if (*operation == Operation::CopyMask ||
      *operation == Operation::GatherMask) {
    require_bool(unit, diagnostics, types, parameters[2], owner);
  } else if (*operation == Operation::ClusterGather) {
    require_cluster_mask(unit, diagnostics, types, parameters[2], owner);
  } else if (*operation == Operation::ClusterGatherMask) {
    require_cluster_mask(unit, diagnostics, types, parameters[2], owner);
    require_bool(unit, diagnostics, types, parameters[3], owner);
  }

  return AsyncIntrinsic(*operation, cache_scope, cache_temporal, direction,
                        newer_groups, result);
}

std::optional<Value> AsyncIntrinsic::call(std::span<const Value> arguments,
                                          ValueArena& arena,
                                          loom_builder_t* builder,
                                          loom_location_id_t location) const {
  loom_op_t* op;
  switch (operation_) {
    case Operation::Copy:
      check(loom_kernel_async_copy_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          cache_scope_, cache_temporal_, direction_, result_->type, location,
          &op));
      break;
    case Operation::CopyMask:
      check(loom_kernel_async_copy_mask_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          arguments[2].ssa(), cache_scope_, cache_temporal_, direction_,
          result_->type, location, &op));
      break;
    case Operation::Gather:
      check(loom_kernel_async_gather_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          cache_scope_, cache_temporal_, result_->type, location, &op));
      break;
    case Operation::GatherMask:
      check(loom_kernel_async_gather_mask_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          arguments[2].ssa(), cache_scope_, cache_temporal_, result_->type,
          location, &op));
      break;
    case Operation::ClusterGather:
      check(loom_kernel_async_cluster_gather_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          arguments[2].ssa(), cache_scope_, cache_temporal_, result_->type,
          location, &op));
      break;
    case Operation::ClusterGatherMask:
      check(loom_kernel_async_cluster_gather_mask_build(
          builder, view_component(arguments[0]), view_component(arguments[1]),
          arguments[2].ssa(), arguments[3].ssa(), cache_scope_, cache_temporal_,
          result_->type, location, &op));
      break;
    case Operation::Group: {
      std::array<loom_value_id_t, 8> inline_tokens;
      std::vector<loom_value_id_t> overflow_tokens;
      auto* tokens = inline_tokens.data();
      if (arguments.size() > inline_tokens.size()) {
        overflow_tokens.reserve(arguments.size());
        for (auto argument : arguments) {
          overflow_tokens.push_back(argument.ssa());
        }
        tokens = overflow_tokens.data();
      } else {
        for (size_t index = 0; index < arguments.size(); ++index) {
          inline_tokens[index] = arguments[index].ssa();
        }
      }
      check(loom_kernel_async_group_build(builder, tokens, arguments.size(),
                                          result_->type, location, &op));
      break;
    }
    case Operation::Wait:
      check(loom_kernel_async_wait_build(builder, arguments[0].ssa(),
                                         newer_groups_, location, &op));
      return std::nullopt;
  }
  auto result = loom_op_results(op)[0];
  return arena.capture(*result_, std::span(&result, 1));
}

bool AsyncIntrinsic::equivalent(const AsyncIntrinsic& other) const {
  if (operation_ != other.operation_ || cache_scope_ != other.cache_scope_ ||
      cache_temporal_ != other.cache_temporal_ ||
      direction_ != other.direction_ || newer_groups_ != other.newer_groups_ ||
      (result_ == nullptr) != (other.result_ == nullptr)) {
    return false;
  }
  return !result_ || loom_type_equal(result_->type, other.result_->type);
}

}  // namespace loom::cxx_import
