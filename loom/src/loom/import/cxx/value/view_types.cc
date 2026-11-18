// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/value/component_names.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {
namespace {

bool is_view_element(loom_type_t element) {
  return loom_type_is_reference(element) ||
         (loom_type_is_scalar(element) &&
          loom_type_element_type(element) != LOOM_SCALAR_TYPE_I1);
}

}  // namespace

void Types::append_view_columns(const cxx::Type* type, bool writable,
                                loom_memory_access_flags_t access_flags,
                                cxx::AST* owner,
                                std::vector<ViewColumn>& columns) {
  writable &= !unit_.typeTraits().is_const(type);
  access_flags |= memory_access_flags(type) & LOOM_MEMORY_ACCESS_FLAG_VOLATILE;
  const auto& admitted = partition(type, owner);
  if (admitted.kind == ValueKind::Record) {
    const auto& row = static_cast<const RecordPartition&>(admitted);
    for (const auto& member : row.members) {
      append_view_columns(member.type(), writable, access_flags, owner,
                          columns);
    }
    return;
  }
  if (admitted.kind == ValueKind::Array) {
    const auto& array = static_cast<const ArrayPartition&>(admitted);
    for (size_t i = 0; i < array.source->size(); ++i) {
      append_view_columns(array.source->elementType(), writable, access_flags,
                          owner, columns);
    }
    return;
  }
  auto element = get(type, owner);
  if (!is_view_element(element)) {
    diagnostics_.reject(unit_, owner,
                        "view elements require a supported non-boolean scalar "
                        "or object handle");
  }
  columns.push_back({element, access_flags, writable});
}

const ViewPartition* Types::view(const cxx::ClassType* input, cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = views_.find(source); found != views_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (traits.is_volatile(input) || !source || !source->isComplete() ||
      source->isUnion() || !source->baseClasses().empty() ||
      !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "view values require a complete trivial class without "
                        "bases");
  }
  auto arguments = source->templateArguments();
  if (arguments.size() != 3) {
    diagnostics_.reject(unit_, owner,
                        "view values require shape, element, and optional "
                        "encoding type arguments");
  }
  auto extents =
      parse_shape(cxx::template_argument_type(arguments[0]), "view", owner);
  auto* element_type = cxx::template_argument_type(arguments[1]);
  if (!element_type) {
    diagnostics_.reject(unit_, owner,
                        "view element requires a source type argument");
  }
  const auto& element_partition = partition(element_type, owner);
  const bool columnar = element_partition.kind == ValueKind::Record ||
                        element_partition.kind == ValueKind::Array;
  loom_type_t element = {};
  std::vector<ViewColumn> columns;
  if (columnar) {
    append_view_columns(element_type, true, 0, owner, columns);
    if (columns.empty()) {
      diagnostics_.reject(unit_, owner, "view rows require at least one field");
    }
  } else {
    element = get(element_type, owner);
    if (!is_view_element(element)) {
      diagnostics_.reject(unit_, owner,
                          "view elements require a supported non-boolean "
                          "scalar or object handle");
    }
  }
  const EncodingPartition* view_encoding = nullptr;
  auto* encoding_type = cxx::template_argument_type(arguments[2]);
  if (!encoding_type) {
    diagnostics_.reject(unit_, owner,
                        "view encoding requires a source type argument");
  }
  if (unqualified(encoding_type)->kind() != cxx::TypeKind::kVoid) {
    const auto& admitted_encoding = partition(encoding_type, owner);
    if (admitted_encoding.kind != ValueKind::Encoding) {
      diagnostics_.reject(unit_, owner,
                          "view encoding must be void or a Loom encoding type");
    }
    view_encoding = &static_cast<const EncodingPartition&>(admitted_encoding);
    if ((view_encoding->role != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT &&
         view_encoding->role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) ||
        view_encoding->rank != extents.size()) {
      diagnostics_.reject(unit_, owner,
                          "view encoding must be a layout or physical storage "
                          "encoding with the same rank");
    }
  }
  size_t dynamic_count = 0;
  for (auto extent : extents) {
    if (extent < 0) {
      ++dynamic_count;
    }
  }
  auto result = std::make_unique<ViewPartition>();
  result->kind = ValueKind::View;
  result->component_count =
      dynamic_count + (columnar ? columns.size() : 1) * (view_encoding ? 2 : 1);
  result->dynamic_count = dynamic_count;
  result->source = source;
  result->element_type = element_type;
  result->element = element;
  result->access_flags =
      memory_access_flags(element_type) & LOOM_MEMORY_ACCESS_FLAG_VOLATILE;
  result->encoding = view_encoding;
  result->extents = std::move(extents);
  if (columnar) {
    result->columns = std::move(columns);
  }
  for (size_t axis = 0; axis < result->extents.size(); ++axis) {
    if (result->extents[axis] < 0) {
      result->component_names.push_back("dim" + std::to_string(axis));
    }
  }
  std::vector<std::string> column_names;
  if (columnar) {
    append_component_names(element_partition, {}, column_names);
  } else {
    column_names.emplace_back();
  }
  for (const auto& name : column_names) {
    if (view_encoding) {
      result->component_names.push_back(
          (name.empty() ? "" : name + "_") +
          (view_encoding->role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT
               ? "layout"
               : "storage"));
    }
  }
  result->component_names.insert(result->component_names.end(),
                                 column_names.begin(), column_names.end());
  if (!result->requires_binding()) {
    result->bind_static_type();
  }
  auto* admitted = result.get();
  views_.emplace(source, std::move(result));
  return admitted;
}

}  // namespace loom::cxx_import
