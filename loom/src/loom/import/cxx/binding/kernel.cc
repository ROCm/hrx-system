// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/kernel.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <array>
#include <cstdint>
#include <utility>
#include <variant>

#include "loom/import/cxx/binding/combining.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"

namespace loom::cxx_import {

std::optional<KernelCollectiveIntrinsic::Operation>
KernelCollectiveIntrinsic::find(std::string_view name) {
  static constexpr std::pair<std::string_view, Operation> operations[] = {
      {"kernel.subgroup.id", Operation::Id},
      {"kernel.subgroup.count", Operation::Count},
      {"kernel.subgroup.size", Operation::Size},
      {"kernel.subgroup.lane.id", Operation::LaneId},
      {"kernel.subgroup.vote.any", Operation::Any},
      {"kernel.subgroup.vote.all", Operation::All},
      {"kernel.subgroup.vote.ballot", Operation::Ballot},
      {"kernel.subgroup.active.mask", Operation::ActiveMask},
      {"kernel.subgroup.broadcast", Operation::Broadcast},
      {"kernel.subgroup.broadcast.first", Operation::BroadcastFirst},
      {"kernel.subgroup.reduce", Operation::SubgroupReduce},
      {"kernel.workgroup.reduce", Operation::WorkgroupReduce},
      {"kernel.subgroup.scan", Operation::SubgroupScan},
      {"kernel.workgroup.scan", Operation::WorkgroupScan},
  };
  for (auto [candidate, operation] : operations) {
    if (candidate == name) {
      return operation;
    }
  }
  return std::nullopt;
}

bool KernelCollectiveIntrinsic::supports(std::string_view name) {
  return find(name).has_value();
}

std::optional<KernelCollectiveIntrinsic> KernelCollectiveIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto operation = find(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  auto* result = types.unqualified(signature->returnType());
  const bool combines = *operation == Operation::SubgroupReduce ||
                        *operation == Operation::WorkgroupReduce ||
                        *operation == Operation::SubgroupScan ||
                        *operation == Operation::WorkgroupScan;
  const bool scans = *operation == Operation::SubgroupScan ||
                     *operation == Operation::WorkgroupScan;
  const bool clustered = *operation == Operation::SubgroupReduce ||
                         *operation == Operation::SubgroupScan;
  const size_t attribute_count = combines ? 2u : 1u;
  if (attribute.arguments.size() != attribute_count ||
      signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       combines
                           ? "kernel collective requires one combining kind "
                             "and a non-variadic signature"
                           : "subgroup binding requires one operation name "
                             "and a non-variadic signature");
  }

  size_t operand_count = 0;
  switch (*operation) {
    case Operation::Any:
    case Operation::All:
    case Operation::Ballot:
    case Operation::BroadcastFirst:
    case Operation::SubgroupReduce:
    case Operation::WorkgroupReduce:
    case Operation::SubgroupScan:
    case Operation::WorkgroupScan:
      operand_count = 1;
      break;
    case Operation::Broadcast:
      operand_count = 2;
      break;
    default:
      break;
  }
  if (parameters.size() != operand_count) {
    diagnostics.reject(
        unit, owner,
        "kernel collective declaration has the wrong operand count");
  }

  auto result_type = types.get(result, owner);
  auto i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  auto i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  auto i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  std::optional<Combining> combining;
  switch (*operation) {
    case Operation::Id:
    case Operation::Count:
    case Operation::Size:
    case Operation::LaneId:
      if (!loom_type_equal(result_type, i32) || !types.is_unsigned(result)) {
        diagnostics.reject(unit, owner,
                           "subgroup topology requires an unsigned i32 result");
      }
      break;
    case Operation::Any:
    case Operation::All:
      if (!loom_type_equal(result_type, i1)) {
        diagnostics.reject(unit, owner, "subgroup vote requires a bool result");
      }
      [[fallthrough]];
    case Operation::Ballot:
      if (types.unqualified(parameters[0])->kind() != cxx::TypeKind::kBool) {
        diagnostics.reject(unit, owner,
                           "subgroup vote requires a bool predicate");
      }
      if (*operation != Operation::Ballot) {
        break;
      }
      [[fallthrough]];
    case Operation::ActiveMask:
      if (!loom_type_equal(result_type, i32) &&
          !loom_type_equal(result_type, i64)) {
        diagnostics.reject(unit, owner,
                           "subgroup mask requires an i32 or i64 result");
      }
      break;
    case Operation::Broadcast:
      if (!loom_type_equal(types.get(parameters[1], owner), i32)) {
        diagnostics.reject(unit, owner,
                           "subgroup broadcast requires an i32 lane");
      }
      [[fallthrough]];
    case Operation::BroadcastFirst:
      if (types.unqualified(parameters[0]) != result ||
          (!loom_type_is_scalar(result_type) &&
           !(loom_type_is_vector(result_type) &&
             loom_type_rank(result_type) == 1))) {
        diagnostics.reject(unit, owner,
                           "subgroup broadcast requires matching scalar or "
                           "rank-one vector value and result types");
      }
      break;
    case Operation::SubgroupReduce:
    case Operation::WorkgroupReduce:
    case Operation::SubgroupScan:
    case Operation::WorkgroupScan: {
      auto parsed_kind = parse_combining_kind(attribute.arguments[1]->name());
      if (!parsed_kind) {
        diagnostics.reject(unit, owner,
                           "unsupported kernel collective combining kind");
      }
      Combining semantics = {
          .combining_kind = *parsed_kind,
          .cluster_size = 0,
          .cluster_stride = 0,
          .math_flags = 0,
          .scan = std::nullopt,
      };
      if (types.unqualified(parameters[0]) != result ||
          (!loom_type_is_scalar(result_type) &&
           !(loom_type_is_vector(result_type) &&
             loom_type_rank(result_type) == 1))) {
        diagnostics.reject(unit, owner,
                           "kernel collective requires matching scalar or "
                           "rank-one vector value and result types");
      }
      const auto* element = result;
      if (const auto* vector = types.native_vector(result)) {
        element = types.unqualified(vector->elementType());
      }
      const bool floating = types.is_float(element);
      const bool integer = unit.typeTraits().is_integral(element) &&
                           element->kind() != cxx::TypeKind::kBool;
      if ((floating &&
           !loom_combining_kind_accepts_float(semantics.combining_kind)) ||
          (integer &&
           !loom_combining_kind_accepts_integer(semantics.combining_kind)) ||
          (!floating && !integer)) {
        diagnostics.reject(
            unit, owner,
            "kernel collective combining kind does not match the value type");
      }
      if (((semantics.combining_kind == LOOM_COMBINING_KIND_MINSI ||
            semantics.combining_kind == LOOM_COMBINING_KIND_MAXSI) &&
           types.is_unsigned(element)) ||
          ((semantics.combining_kind == LOOM_COMBINING_KIND_MINUI ||
            semantics.combining_kind == LOOM_COMBINING_KIND_MAXUI) &&
           !types.is_unsigned(element))) {
        diagnostics.reject(
            unit, owner,
            "kernel collective signedness does not match the combining kind");
      }
      auto template_arguments = function->templateArguments();
      size_t cluster_begin = 0;
      if (scans) {
        if (template_arguments.size() < 2) {
          diagnostics.reject(unit, owner,
                             "kernel scan templates require leading mode and "
                             "direction arguments");
        }
        std::array<uint8_t, 2> selectors;
        const bool subgroup = *operation == Operation::SubgroupScan;
        const auto* descriptor =
            loom_kernel_dialect_vtables(nullptr)[loom_op_dialect_index(
                subgroup ? LOOM_OP_KERNEL_SUBGROUP_SCAN
                         : LOOM_OP_KERNEL_WORKGROUP_SCAN)];
        const std::array fields = {
            subgroup ? loom_kernel_subgroup_scan_mode_diagnostic_ref()
                     : loom_kernel_workgroup_scan_mode_diagnostic_ref(),
            subgroup ? loom_kernel_subgroup_scan_direction_diagnostic_ref()
                     : loom_kernel_workgroup_scan_direction_diagnostic_ref(),
        };
        for (size_t index = 0; index < selectors.size(); ++index) {
          auto value = cxx::template_argument_value(template_arguments[index]);
          auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
          if (!number || number->isNegative() ||
              number->toUWide() > UINT8_MAX ||
              !loom_attr_descriptor_has_enum_case(
                  &descriptor->attr_descriptors[fields[index].index],
                  number->toUIntMax())) {
            diagnostics.reject(
                unit, owner,
                "kernel scan selector is not a permitted enum value");
          }
          selectors[index] = static_cast<uint8_t>(number->toUIntMax());
        }
        semantics.scan = Scan{selectors[0], selectors[1]};
        cluster_begin = 2;
      }
      if (clustered && !template_arguments.empty()) {
        if (template_arguments.size() < cluster_begin + 2) {
          diagnostics.reject(unit, owner,
                             "kernel collective templates require leading "
                             "cluster size and stride arguments");
        }
        uint32_t* destinations[] = {&semantics.cluster_size,
                                    &semantics.cluster_stride};
        for (size_t index = 0; index < 2; ++index) {
          auto value = cxx::template_argument_value(
              template_arguments[cluster_begin + index]);
          auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
          if (!number || number->isNegative() ||
              number->toUWide() > UINT32_MAX) {
            diagnostics.reject(unit, owner,
                               "kernel collective cluster arguments must be "
                               "unsigned 32-bit constants");
          }
          *destinations[index] = static_cast<uint32_t>(number->toUIntMax());
        }
      }
      if (semantics.cluster_stride != 0 && semantics.cluster_size == 0) {
        diagnostics.reject(
            unit, owner,
            "kernel collective cluster stride requires a cluster size");
      }
      const size_t math_index = cluster_begin + (clustered ? 2u : 0u);
      if (template_arguments.size() > math_index) {
        auto value =
            cxx::template_argument_value(template_arguments[math_index]);
        if (value) {
          auto* number = std::get_if<cxx::ConstInt>(&*value);
          if (!number || number->isNegative() || number->toUWide() > 1) {
            diagnostics.reject(
                unit, owner,
                "kernel collective arithmetic must be native or preserve");
          }
          semantics.math_flags = static_cast<uint8_t>(number->toUIntMax());
          if (semantics.math_flags != 0 && !floating) {
            diagnostics.reject(unit, owner,
                               "preservation requirements apply only to "
                               "floating-point collectives");
          }
        }
      }
      combining = semantics;
      break;
    }
  }
  if (combining) {
    return KernelCollectiveIntrinsic(*operation, result_type, *combining);
  }
  return KernelCollectiveIntrinsic(*operation, result_type);
}

loom_value_id_t KernelCollectiveIntrinsic::call(
    std::span<const Value> arguments, loom_builder_t* builder,
    loom_location_id_t location) const {
  loom_op_t* op;
  auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto index_result = [&] {
    auto value = loom_op_results(op)[0];
    check(loom_index_cast_build(builder, value, index_type, result_type_,
                                location, &op));
    return loom_op_results(op)[0];
  };
  switch (operation_) {
    case Operation::Id:
      check(loom_kernel_subgroup_id_build(builder, index_type, location, &op));
      return index_result();
    case Operation::Count:
      check(
          loom_kernel_subgroup_count_build(builder, index_type, location, &op));
      return index_result();
    case Operation::Size:
      check(
          loom_kernel_subgroup_size_build(builder, index_type, location, &op));
      return index_result();
    case Operation::LaneId:
      check(loom_kernel_subgroup_lane_id_build(builder, index_type, location,
                                               &op));
      return index_result();
    case Operation::Any:
      check(loom_kernel_subgroup_vote_any_build(builder, arguments[0].ssa(),
                                                location, &op));
      break;
    case Operation::All:
      check(loom_kernel_subgroup_vote_all_build(builder, arguments[0].ssa(),
                                                location, &op));
      break;
    case Operation::Ballot:
      check(loom_kernel_subgroup_vote_ballot_build(
          builder, arguments[0].ssa(), result_type_, location, &op));
      break;
    case Operation::ActiveMask:
      check(loom_kernel_subgroup_active_mask_build(builder, result_type_,
                                                   location, &op));
      break;
    case Operation::Broadcast:
      check(loom_kernel_subgroup_broadcast_build(builder, arguments[0].ssa(),
                                                 arguments[1].ssa(),
                                                 result_type_, location, &op));
      break;
    case Operation::BroadcastFirst:
      check(loom_kernel_subgroup_broadcast_first_build(
          builder, arguments[0].ssa(), result_type_, location, &op));
      break;
    case Operation::SubgroupReduce: {
      const auto& reduction = *combining_;
      loom_kernel_subgroup_reduce_build_flags_t flags = 0;
      if (reduction.cluster_size != 0) {
        flags |= LOOM_KERNEL_SUBGROUP_REDUCE_BUILD_FLAG_HAS_CLUSTER_SIZE;
      }
      if (reduction.cluster_stride != 0) {
        flags |= LOOM_KERNEL_SUBGROUP_REDUCE_BUILD_FLAG_HAS_CLUSTER_STRIDE;
      }
      check(loom_kernel_subgroup_reduce_build(
          builder, flags, reduction.combining_kind, reduction.math_flags,
          arguments[0].ssa(), reduction.cluster_size, reduction.cluster_stride,
          result_type_, location, &op));
      break;
    }
    case Operation::WorkgroupReduce: {
      const loom_value_id_t value = arguments[0].ssa();
      check(loom_kernel_workgroup_reduce_build(
          builder, combining_->combining_kind, combining_->math_flags, &value,
          1, &result_type_, 1, location, &op));
      break;
    }
    case Operation::SubgroupScan: {
      const auto& combining = *combining_;
      loom_kernel_subgroup_scan_build_flags_t flags = 0;
      if (combining.cluster_size != 0) {
        flags |= LOOM_KERNEL_SUBGROUP_SCAN_BUILD_FLAG_HAS_CLUSTER_SIZE;
      }
      if (combining.cluster_stride != 0) {
        flags |= LOOM_KERNEL_SUBGROUP_SCAN_BUILD_FLAG_HAS_CLUSTER_STRIDE;
      }
      check(loom_kernel_subgroup_scan_build(
          builder, flags, combining.combining_kind, combining.math_flags,
          arguments[0].ssa(), combining.cluster_size, combining.cluster_stride,
          static_cast<loom_kernel_subgroup_scan_mode_t>(combining.scan->mode),
          static_cast<loom_kernel_subgroup_scan_direction_t>(
              combining.scan->direction),
          result_type_, location, &op));
      break;
    }
    case Operation::WorkgroupScan:
      check(loom_kernel_workgroup_scan_build(
          builder, combining_->combining_kind, combining_->math_flags,
          arguments[0].ssa(),
          static_cast<loom_kernel_workgroup_scan_mode_t>(
              combining_->scan->mode),
          static_cast<loom_kernel_workgroup_scan_direction_t>(
              combining_->scan->direction),
          result_type_, location, &op));
      break;
  }
  return loom_op_results(op)[0];
}

bool KernelCollectiveIntrinsic::equivalent(
    const KernelCollectiveIntrinsic& other) const {
  if (operation_ != other.operation_ ||
      !loom_type_equal(result_type_, other.result_type_)) {
    return false;
  }
  if (!combining_) {
    return true;
  }
  return combining_->combining_kind == other.combining_->combining_kind &&
         combining_->math_flags == other.combining_->math_flags &&
         combining_->cluster_size == other.combining_->cluster_size &&
         combining_->cluster_stride == other.combining_->cluster_stride &&
         combining_->scan == other.combining_->scan;
}

bool BarrierIntrinsic::supports(std::string_view name) {
  return name == "kernel.barrier";
}

std::optional<BarrierIntrinsic> BarrierIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (!supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic() || !signature->parameterTypes().empty() ||
      !unit.typeTraits().is_void(signature->returnType())) {
    diagnostics.reject(unit, owner, "kernel.barrier requires void()");
  }
  auto arguments = function->templateArguments();
  if (attribute.arguments.size() != 1 || arguments.size() < 3) {
    diagnostics.reject(unit, owner,
                       "kernel.barrier requires leading memory-space, scope, "
                       "and ordering template arguments");
  }
  std::array<uint8_t, 3> selectors;
  const std::array fields = {
      loom_kernel_barrier_memory_space_diagnostic_ref(),
      loom_kernel_barrier_scope_diagnostic_ref(),
      loom_kernel_barrier_ordering_diagnostic_ref(),
  };
  const auto* operation = loom_kernel_dialect_vtables(
      nullptr)[loom_op_dialect_index(LOOM_OP_KERNEL_BARRIER)];
  for (size_t index = 0; index < selectors.size(); ++index) {
    auto value = cxx::template_argument_value(arguments[index]);
    auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
    if (!number || number->isNegative() || number->toUWide() > UINT8_MAX ||
        !loom_attr_descriptor_has_enum_case(
            &operation->attr_descriptors[fields[index].index],
            number->toUIntMax())) {
      diagnostics.reject(
          unit, owner, "kernel.barrier selector is not a permitted enum value");
    }
    selectors[index] = static_cast<uint8_t>(number->toUIntMax());
  }
  return BarrierIntrinsic(
      static_cast<loom_value_fact_memory_space_t>(selectors[0]),
      static_cast<loom_atomic_scope_t>(selectors[1]),
      static_cast<loom_atomic_ordering_t>(selectors[2]));
}

void BarrierIntrinsic::call(loom_builder_t* builder,
                            loom_location_id_t location) const {
  loom_op_t* op;
  check(loom_kernel_barrier_build(builder, memory_space_, scope_, ordering_,
                                  location, &op));
}

bool BarrierIntrinsic::equivalent(const BarrierIntrinsic& other) const {
  return memory_space_ == other.memory_space_ && scope_ == other.scope_ &&
         ordering_ == other.ordering_;
}

}  // namespace loom::cxx_import
