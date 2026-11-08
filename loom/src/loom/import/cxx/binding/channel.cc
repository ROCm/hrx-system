// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/channel.h"

#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/assert.h"
#include "loom/import/cxx/source/error.h"
#include "loom/ops/channel/ops.h"

namespace loom::cxx_import {
namespace {

const ChannelPartition* require_handle(cxx::TranslationUnit& unit,
                                       Diagnostics& diagnostics, Types& types,
                                       const cxx::Type* type,
                                       ChannelHandleKind kind,
                                       std::string_view operation,
                                       std::string_view role, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  auto* handle = partition.kind == ValueKind::Channel
                     ? &static_cast<const ChannelPartition&>(partition)
                     : nullptr;
  if (!handle || handle->handle_kind != kind) {
    diagnostics.reject(unit, owner,
                       std::string(operation) + " requires " +
                           std::string(role) + " channel capability");
  }
  return handle;
}

const ViewPartition* require_view(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics, Types& types,
                                  const cxx::Type* type,
                                  std::string_view operation,
                                  std::string_view role, cxx::AST* owner) {
  const auto& partition = types.partition(type, owner);
  auto* view = partition.kind == ValueKind::View
                   ? &static_cast<const ViewPartition&>(partition)
                   : nullptr;
  if (!view) {
    diagnostics.reject(unit, owner,
                       std::string(operation) + " requires " +
                           std::string(role) + " Loom view");
  }
  return view;
}

const RecordPartition* require_record(cxx::TranslationUnit& unit,
                                      Diagnostics& diagnostics, Types& types,
                                      const cxx::Type* type, size_t members,
                                      std::string_view operation,
                                      cxx::AST* owner) {
  const auto* record = types.record(type, owner);
  if (!record || record->members.size() != members) {
    diagnostics.reject(unit, owner,
                       std::string(operation) + " result requires a " +
                           std::to_string(members) + "-member aggregate");
  }
  return record;
}

bool payload_shape_matches(const TilePartition& payload,
                           const ViewPartition& view, size_t prefix_rank) {
  if (payload.element != view.element ||
      view.extents.size() != payload.extents.size() + prefix_rank) {
    return false;
  }
  for (size_t axis = 0; axis < payload.extents.size(); ++axis) {
    if (payload.extents[axis] != view.extents[axis + prefix_rank]) {
      return false;
    }
  }
  return true;
}

bool payload_view_matches(const TilePartition& payload,
                          const ViewPartition& view) {
  if (!payload_shape_matches(payload, view, 0)) {
    return false;
  }
  // A result view can omit the payload encoding. When it retains one, the
  // existing payload identity is the only value available to bind it.
  return !view.encoding || view.encoding == payload.encoding;
}

bool is_const_view(cxx::TranslationUnit& unit, const ViewPartition& view) {
  return unit.typeTraits().is_const(view.element_type);
}

std::span<const loom_value_id_t> handle_dependencies(
    Value value, std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1>& storage) {
  auto components = value.components();
  auto dependencies = components.first(components.size() - 1);
  std::copy(dependencies.begin(), dependencies.end(), storage.begin());
  return {storage.data(), dependencies.size()};
}

std::span<const loom_value_id_t> view_dependencies(
    const TilePartition& payload, const ViewPartition& view,
    std::span<const loom_value_id_t> payload_dependencies,
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1>& storage) {
  size_t count = 0;
  for (auto extent : payload.extents) {
    if (extent < 0) {
      storage[count] = payload_dependencies[count];
      ++count;
    }
  }
  if (view.encoding) {
    storage[count] = payload_dependencies[count];
    ++count;
  }
  return {storage.data(), count};
}

Value capture_dependent(const Partition& partition,
                        std::span<const loom_value_id_t> dependencies,
                        loom_value_id_t result, ValueArena& arena) {
  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 2> components;
  std::copy(dependencies.begin(), dependencies.end(), components.begin());
  components[dependencies.size()] = result;
  return arena.capture(partition, std::span<const loom_value_id_t>(components)
                                      .first(dependencies.size() + 1));
}

void append_dependent(std::vector<loom_value_id_t>& components,
                      std::span<const loom_value_id_t> dependencies,
                      loom_value_id_t result) {
  components.insert(components.end(), dependencies.begin(), dependencies.end());
  components.push_back(result);
}

bool count_fanout_reads(const Partition& partition,
                        const ChannelPartition* expected, size_t& count) {
  if (partition.kind == ValueKind::Channel) {
    if (&partition != expected) {
      return false;
    }
    ++count;
    return true;
  }
  if (partition.kind == ValueKind::Record) {
    const auto& record = static_cast<const RecordPartition&>(partition);
    return std::all_of(
        record.members.begin(), record.members.end(), [&](const auto& member) {
          return count_fanout_reads(*member.partition, expected, count);
        });
  }
  return false;
}

}  // namespace

std::optional<ChannelIntrinsic::Operation> ChannelIntrinsic::parse_operation(
    std::string_view name) {
  static constexpr std::pair<std::string_view, Operation> operations[] = {
      {"channel.bind", Operation::Bind},
      {"channel.reserve", Operation::Reserve},
      {"channel.accept", Operation::Accept},
      {"channel.wait", Operation::Wait},
      {"channel.acquire", Operation::Acquire},
      {"channel.publish", Operation::Publish},
      {"channel.release", Operation::Release},
      {"channel.fanout", Operation::Fanout},
      {"channel.copy", Operation::Copy},
      {"channel.select", Operation::Select},
  };
  for (auto [candidate, operation] : operations) {
    if (candidate == name) {
      return operation;
    }
  }
  return std::nullopt;
}

bool ChannelIntrinsic::supports(std::string_view name) {
  return parse_operation(name).has_value();
}

bool ChannelIntrinsic::equivalent(const ChannelIntrinsic& other) const {
  return operation_ == other.operation_ && discipline_ == other.discipline_ &&
         storage_view_ == other.storage_view_ &&
         source_handle_ == other.source_handle_ &&
         result_handle_ == other.result_handle_ &&
         result_view_ == other.result_view_ &&
         result_record_ == other.result_record_ &&
         fanout_count_ == other.fanout_count_;
}

std::optional<ChannelIntrinsic> ChannelIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  auto operation = parse_operation(attribute.arguments[0]->name());
  if (!operation) {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "channel operations accept no string selectors");
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto parameters = signature->parameterTypes();
  auto operation_name = attribute.arguments[0]->name();
  if (signature->isVariadic()) {
    diagnostics.reject(
        unit, owner,
        std::string(operation_name) + " requires a non-variadic declaration");
  }
  ChannelIntrinsic result(*operation);
  auto require_parameters = [&](size_t count) {
    if (parameters.size() != count) {
      diagnostics.reject(unit, owner,
                         std::string(operation_name) + " requires " +
                             std::to_string(count) + " operand" +
                             (count == 1 ? "" : "s"));
    }
  };
  auto require_void = [&] {
    if (!unit.typeTraits().is_void(signature->returnType())) {
      diagnostics.reject(
          unit, owner, std::string(operation_name) + " requires a void result");
    }
  };

  switch (*operation) {
    case Operation::Bind: {
      if (parameters.size() < 2 || parameters.size() > 3) {
        diagnostics.reject(
            unit, owner,
            "channel.bind requires storage, capacity, and an optional "
            "payload encoding");
      }
      result.storage_view_ =
          require_view(unit, diagnostics, types, parameters[0], operation_name,
                       "slot-storage", owner);
      result.result_handle_ = require_handle(
          unit, diagnostics, types, signature->returnType(),
          ChannelHandleKind::Channel, operation_name, "result", owner);
      if (!payload_shape_matches(*result.result_handle_->payload,
                                 *result.storage_view_, 1) ||
          is_const_view(unit, *result.storage_view_) ||
          types.address_domain(parameters[1], owner) !=
              LOOM_SCALAR_TYPE_INDEX) {
        diagnostics.reject(
            unit, owner,
            "channel.bind requires writable slot storage with one leading "
            "dimension, a matching tile result, and an index capacity");
      }
      const auto* payload_encoding = result.result_handle_->payload->encoding;
      if (payload_encoding) {
        if (parameters.size() != 3 ||
            &types.partition(parameters[2], owner) != payload_encoding) {
          diagnostics.reject(
              unit, owner,
              "encoded channel.bind requires its exact payload encoding as "
              "the third operand");
        }
      } else if (parameters.size() != 2) {
        diagnostics.reject(
            unit, owner,
            "unencoded channel.bind does not accept a payload encoding");
      }
      auto arguments = function->templateArguments();
      auto discipline = arguments.empty()
                            ? std::nullopt
                            : cxx::template_argument_value(arguments[0]);
      auto* number =
          discipline ? std::get_if<cxx::ConstInt>(&*discipline) : nullptr;
      const auto* descriptor = loom_channel_dialect_vtables(
          nullptr)[loom_op_dialect_index(LOOM_OP_CHANNEL_BIND)];
      auto field = loom_channel_bind_discipline_diagnostic_ref();
      if (!number || number->isNegative() || number->toUWide() > UINT8_MAX ||
          !loom_attr_descriptor_has_enum_case(
              &descriptor->attr_descriptors[field.index],
              number->toUIntMax())) {
        diagnostics.reject(unit, owner,
                           "channel.bind requires a supported discipline "
                           "template argument");
      }
      result.discipline_ = static_cast<uint8_t>(number->toUIntMax());
      break;
    }
    case Operation::Reserve: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Channel,
          operation_name, "input", owner);
      result.result_record_ =
          require_record(unit, diagnostics, types, signature->returnType(), 2,
                         operation_name, owner);
      result.result_handle_ = require_handle(
          unit, diagnostics, types, result.result_record_->members[0].type(),
          ChannelHandleKind::Write, operation_name, "first result", owner);
      result.result_view_ = require_view(
          unit, diagnostics, types, result.result_record_->members[1].type(),
          operation_name, "second-result", owner);
      if (result.result_handle_->payload != result.source_handle_->payload ||
          !payload_view_matches(*result.source_handle_->payload,
                                *result.result_view_) ||
          is_const_view(unit, *result.result_view_)) {
        diagnostics.reject(unit, owner,
                           "channel.reserve result requires a matching write "
                           "and writable payload view");
      }
      break;
    }
    case Operation::Accept: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Channel,
          operation_name, "input", owner);
      result.result_handle_ = require_handle(
          unit, diagnostics, types, signature->returnType(),
          ChannelHandleKind::Read, operation_name, "result", owner);
      if (result.result_handle_->payload != result.source_handle_->payload) {
        diagnostics.reject(unit, owner,
                           "channel.accept result must read the input payload");
      }
      break;
    }
    case Operation::Wait: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Read,
          operation_name, "input", owner);
      result.result_view_ =
          require_view(unit, diagnostics, types, signature->returnType(),
                       operation_name, "result", owner);
      if (!payload_view_matches(*result.source_handle_->payload,
                                *result.result_view_) ||
          is_const_view(unit, *result.result_view_) ==
              result.source_handle_->mutable_read) {
        diagnostics.reject(
            unit, owner,
            "channel.wait result must be a const view for immutable reads or "
            "a writable view for mutable reads");
      }
      break;
    }
    case Operation::Acquire: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Channel,
          operation_name, "input", owner);
      result.result_record_ =
          require_record(unit, diagnostics, types, signature->returnType(), 2,
                         operation_name, owner);
      result.result_handle_ = require_handle(
          unit, diagnostics, types, result.result_record_->members[0].type(),
          ChannelHandleKind::Read, operation_name, "first result", owner);
      result.result_view_ = require_view(
          unit, diagnostics, types, result.result_record_->members[1].type(),
          operation_name, "second-result", owner);
      if (result.result_handle_->payload != result.source_handle_->payload ||
          !payload_view_matches(*result.source_handle_->payload,
                                *result.result_view_) ||
          is_const_view(unit, *result.result_view_) ==
              result.result_handle_->mutable_read) {
        diagnostics.reject(
            unit, owner,
            "channel.acquire result requires a matching read and a view with "
            "its access permission");
      }
      break;
    }
    case Operation::Publish:
      require_parameters(1);
      require_void();
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Write,
          operation_name, "input", owner);
      break;
    case Operation::Release:
      require_parameters(1);
      require_void();
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Read,
          operation_name, "input", owner);
      break;
    case Operation::Fanout: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Read,
          operation_name, "input", owner);
      result.result_record_ = types.record(signature->returnType(), owner);
      result.result_handle_ = result.source_handle_;
      if (result.source_handle_->mutable_read || !result.result_record_ ||
          !count_fanout_reads(*result.result_record_, result.result_handle_,
                              result.fanout_count_) ||
          result.fanout_count_ < 2) {
        diagnostics.reject(
            unit, owner,
            "channel.fanout requires an immutable read and a result aggregate "
            "containing at least two matching reads");
      }
      break;
    }
    case Operation::Copy: {
      require_parameters(2);
      require_void();
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Read,
          operation_name, "source", owner);
      auto* destination = require_handle(
          unit, diagnostics, types, parameters[1], ChannelHandleKind::Write,
          operation_name, "destination", owner);
      if (destination->payload != result.source_handle_->payload) {
        diagnostics.reject(unit, owner,
                           "channel.copy requires matching source and "
                           "destination payloads");
      }
      break;
    }
    case Operation::Select: {
      require_parameters(1);
      result.source_handle_ = require_handle(
          unit, diagnostics, types, parameters[0], ChannelHandleKind::Channel,
          operation_name, "input", owner);
      result.result_record_ =
          require_record(unit, diagnostics, types, signature->returnType(), 2,
                         operation_name, owner);
      result.result_handle_ = require_handle(
          unit, diagnostics, types, result.result_record_->members[1].type(),
          ChannelHandleKind::Read, operation_name, "second result", owner);
      if (types.address_domain(result.result_record_->members[0].type(),
                               owner) != LOOM_SCALAR_TYPE_INDEX ||
          result.result_handle_->mutable_read ||
          result.result_handle_->payload != result.source_handle_->payload) {
        diagnostics.reject(
            unit, owner,
            "channel.select result requires an index source rank and a "
            "matching immutable read");
      }
      break;
    }
  }
  return result;
}

std::optional<Value> ChannelIntrinsic::call(std::span<const Value> arguments,
                                            ValueArena& arena,
                                            loom_builder_t* builder,
                                            loom_location_id_t location) const {
  loom_op_t* op;
  BoundTypeStorage type_storage;
  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> dependency_storage;
  std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 1> view_dependency_storage;

  if (operation_ == Operation::Bind) {
    auto storage_components = arguments[0].components();
    size_t storage_component = 0;
    size_t result_dependency = 0;
    for (size_t axis = 0; axis < storage_view_->extents.size(); ++axis) {
      if (storage_view_->extents[axis] < 0) {
        if (axis != 0) {
          dependency_storage[result_dependency++] =
              storage_components[storage_component];
        }
        ++storage_component;
      }
    }
    if (storage_view_->encoding) {
      ++storage_component;
    }
    if (result_handle_->payload->encoding) {
      dependency_storage[result_dependency++] = arguments[2].ssa();
    }
    auto dependencies = std::span<const loom_value_id_t>(dependency_storage)
                            .first(result_dependency);
    auto result_type =
        result_handle_->bind_type(builder->module, dependencies, type_storage);
    check(loom_channel_bind_build(
        builder, static_cast<loom_channel_bind_discipline_t>(discipline_),
        storage_view_->view_value(storage_components), arguments[1].ssa(),
        result_type, location, &op));
    return capture_dependent(*result_handle_, dependencies,
                             loom_op_results(op)[0], arena);
  }

  auto dependencies = handle_dependencies(arguments[0], dependency_storage);
  auto handle = source_handle_->handle_value(arguments[0].components());
  auto result_handle_type = [&] {
    return result_handle_->bind_type(builder->module, dependencies,
                                     type_storage);
  };
  auto result_view_dependencies = [&] {
    return view_dependencies(*source_handle_->payload, *result_view_,
                             dependencies, view_dependency_storage);
  };

  switch (operation_) {
    case Operation::Bind:
      break;
    case Operation::Reserve: {
      auto view_dependencies = result_view_dependencies();
      check(loom_channel_reserve_build(
          builder, handle, result_handle_type(),
          result_view_->bind_type(view_dependencies, type_storage), location,
          &op));
      std::vector<loom_value_id_t> components;
      components.reserve(result_record_->component_count);
      append_dependent(components, dependencies, loom_op_results(op)[0]);
      append_dependent(components, view_dependencies, loom_op_results(op)[1]);
      return arena.capture(*result_record_, components);
    }
    case Operation::Accept: {
      auto flags = result_handle_->mutable_read
                       ? LOOM_CHANNEL_ACCEPT_BUILD_FLAG_HAS_MODE
                       : 0;
      check(loom_channel_accept_build(
          builder, flags,
          result_handle_->mutable_read ? LOOM_CHANNEL_MODE_MUTABLE : 0, handle,
          result_handle_type(), location, &op));
      return capture_dependent(*result_handle_, dependencies,
                               loom_op_results(op)[0], arena);
    }
    case Operation::Wait: {
      auto view_dependencies = result_view_dependencies();
      check(loom_channel_wait_build(
          builder, handle,
          result_view_->bind_type(view_dependencies, type_storage), location,
          &op));
      return capture_dependent(*result_view_, view_dependencies,
                               loom_op_results(op)[0], arena);
    }
    case Operation::Acquire: {
      auto view_dependencies = result_view_dependencies();
      auto flags = result_handle_->mutable_read
                       ? LOOM_CHANNEL_ACQUIRE_BUILD_FLAG_HAS_MODE
                       : 0;
      check(loom_channel_acquire_build(
          builder, flags,
          result_handle_->mutable_read ? LOOM_CHANNEL_MODE_MUTABLE : 0, handle,
          result_handle_type(),
          result_view_->bind_type(view_dependencies, type_storage), location,
          &op));
      std::vector<loom_value_id_t> components;
      components.reserve(result_record_->component_count);
      append_dependent(components, dependencies, loom_op_results(op)[0]);
      append_dependent(components, view_dependencies, loom_op_results(op)[1]);
      return arena.capture(*result_record_, components);
    }
    case Operation::Publish:
      check(loom_channel_publish_build(builder, handle, location, &op));
      return std::nullopt;
    case Operation::Release:
      check(loom_channel_release_build(builder, handle, location, &op));
      return std::nullopt;
    case Operation::Fanout: {
      const size_t count = fanout_count_;
      auto type = result_handle_type();
      std::vector<loom_type_t> result_types(count, type);
      check(loom_channel_fanout_build(builder, handle, result_types.data(),
                                      result_types.size(), nullptr, 0, location,
                                      &op));
      std::vector<loom_value_id_t> components;
      components.reserve(result_record_->component_count);
      for (size_t index = 0; index < count; ++index) {
        append_dependent(components, dependencies, loom_op_results(op)[index]);
      }
      return arena.capture(*result_record_, components);
    }
    case Operation::Copy:
      check(loom_channel_copy_build(
          builder, handle,
          static_cast<const ChannelPartition&>(arguments[1].partition())
              .handle_value(arguments[1].components()),
          location, &op));
      return std::nullopt;
    case Operation::Select: {
      check(loom_channel_select_build(builder, handle,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      result_handle_type(), location, &op));
      std::vector<loom_value_id_t> components;
      components.reserve(result_record_->component_count);
      components.push_back(loom_op_results(op)[0]);
      append_dependent(components, dependencies, loom_op_results(op)[1]);
      return arena.capture(*result_record_, components);
    }
  }
  IREE_CHECK_UNREACHABLE("invalid admitted channel operation");
  return std::nullopt;
}

}  // namespace loom::cxx_import
