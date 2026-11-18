// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/component_names.h"

#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/partition.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {
namespace {

// An empty prefix during type admission denotes a field path relative to its
// parent. An anonymous SSA root still separates its name from a named suffix.
std::string join(std::string_view prefix, std::string_view suffix,
                 bool relative) {
  if (suffix.empty()) {
    return std::string(prefix);
  }
  if (relative && prefix.empty()) {
    return std::string(suffix);
  }
  std::string result;
  result.reserve(prefix.size() + 1 + suffix.size());
  result.append(prefix).append("_").append(suffix);
  return result;
}

template <typename Emit>
void visit_names(const Partition& partition, std::string_view prefix,
                 bool relative, Emit& emit) {
  std::span<const std::string> names;
  switch (partition.kind) {
    case ValueKind::Array: {
      const auto& array = static_cast<const ArrayPartition&>(partition);
      for (size_t index = 0; index < array.source->size(); ++index) {
        visit_names(*array.element,
                    join(prefix, std::to_string(index), relative), relative,
                    emit);
      }
      return;
    }
    case ValueKind::Record:
      names = static_cast<const RecordPartition&>(partition).component_names;
      break;
    case ValueKind::Vector:
      names = static_cast<const VectorPartition&>(partition).component_names;
      break;
    case ValueKind::Tile:
      names = static_cast<const TilePartition&>(partition).component_names;
      break;
    case ValueKind::Channel:
      names = static_cast<const ChannelPartition&>(partition).component_names;
      break;
    case ValueKind::View:
      names = static_cast<const ViewPartition&>(partition).component_names;
      break;
    case ValueKind::Pointer:
      emit(prefix, {});
      emit(prefix, "byte_offset");
      return;
    case ValueKind::Tensor:
      if (static_cast<const TensorPartition&>(partition).is_dynamic()) {
        emit(prefix, "dim0");
      }
      [[fallthrough]];
    case ValueKind::SSA:
    case ValueKind::Buffer:
    case ValueKind::Address:
    case ValueKind::Object:
    case ValueKind::OpaqueDialect:
    case ValueKind::Encoding:
    case ValueKind::Pool:
      emit(prefix, {});
      return;
  }
  for (const auto& name : names) {
    emit(prefix, name);
  }
}

void set_name(loom_module_t* module, loom_value_id_t value,
              std::string_view name) {
  loom_string_id_t id;
  check(loom_module_intern_string(
      module, iree_make_string_view(name.data(), name.size()), &id));
  check(loom_module_set_value_name(module, value, id));
}

}  // namespace

void append_component_names(const Partition& partition, std::string_view prefix,
                            std::vector<std::string>& output) {
  auto append = [&](std::string_view path, std::string_view suffix) {
    output.push_back(join(path, suffix, /*relative=*/true));
  };
  visit_names(partition, prefix, /*relative=*/true, append);
}

loom_value_id_t name_value(loom_module_t* module, loom_value_id_t value,
                           std::string_view hint) {
  if (!hint.empty() &&
      loom_module_value(module, value)->name_id == LOOM_STRING_ID_INVALID) {
    set_name(module, value, hint);
  }
  return value;
}

Value name_value(loom_module_t* module, Value value, std::string_view hint) {
  auto components = value.components();
  auto assign = [&](std::string_view path, std::string_view suffix) {
    const auto id = components.front();
    components = components.subspan(1);
    if ((!path.empty() || !suffix.empty()) &&
        loom_module_value(module, id)->name_id == LOOM_STRING_ID_INVALID) {
      if (suffix.empty()) {
        set_name(module, id, path);
      } else {
        set_name(module, id, join(path, suffix, /*relative=*/false));
      }
    }
  };
  visit_names(value.partition(), hint, /*relative=*/false, assign);
  return value;
}

}  // namespace loom::cxx_import
