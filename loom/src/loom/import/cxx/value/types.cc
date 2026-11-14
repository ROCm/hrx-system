// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/types.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/attributes.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <limits>
#include <optional>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/type_registry.h"

namespace loom::cxx_import {
namespace {

bool has_mutable_subobject(const Partition& partition) {
  return partition.kind == ValueKind::Record
             ? static_cast<const RecordPartition&>(partition)
                   .has_mutable_subobject
         : partition.kind == ValueKind::Array
             ? static_cast<const ArrayPartition&>(partition)
                   .has_mutable_subobject
             : false;
}

bool same_subobject_path(cxx::List<cxx::Symbol*>* left,
                         cxx::List<cxx::Symbol*>* right) {
  while (left && right && left->value == right->value) {
    left = left->next;
    right = right->next;
  }
  return !left && !right;
}

const cxx::Attribute* type_binding(cxx::ClassSymbol* source) {
  auto find = [](cxx::ClassSymbol* symbol) -> const cxx::Attribute* {
    if (!symbol || !symbol->attributes()) {
      return nullptr;
    }
    for (const auto& attribute : *symbol->attributes()) {
      if (attribute.attributeNamespace && attribute.name &&
          attribute.attributeNamespace->name() == "loom" &&
          attribute.name->name() == "type") {
        return &attribute;
      }
    }
    return nullptr;
  };
  if (auto* binding = find(source)) {
    return binding;
  }
  auto* primary = cxx::class_template_of(source);
  return primary == source ? nullptr : find(primary);
}

const cxx::ParameterPackSymbol* template_argument_pack(
    cxx::ClassSymbol* source) {
  auto arguments = source->templateArguments();
  if (arguments.size() != 1) {
    return nullptr;
  }
  auto* symbol = std::get_if<cxx::Symbol*>(&arguments.front());
  return symbol ? cxx::symbol_cast<cxx::ParameterPackSymbol>(*symbol) : nullptr;
}

loom_type_t make_shaped_type(loom_type_kind_t kind, loom_scalar_type_t element,
                             std::span<const uint64_t> dimensions,
                             BoundTypeStorage& storage) {
  const loom_type_t base = {
      .header = loom_type_make_header(kind, element, 0, 0),
  };
  return storage.with_dimensions(base, dimensions);
}

loom_type_t make_encoded_shaped_type(loom_type_kind_t kind,
                                     loom_scalar_type_t element,
                                     std::span<const uint64_t> dimensions,
                                     std::optional<loom_value_id_t> encoding,
                                     BoundTypeStorage& storage) {
  auto result = make_shaped_type(kind, element, dimensions, storage);
  if (encoding) {
    if (*encoding > UINT16_MAX) {
      check(iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "encoding value ID %u exceeds the shaped type limit of %u", *encoding,
          UINT16_MAX));
    }
    result.encoding_id = static_cast<uint16_t>(*encoding);
    result.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  }
  return result;
}

loom_type_t make_channel_handle_type(loom_module_t* module,
                                     loom_type_t payload_type,
                                     ChannelHandleKind handle_kind,
                                     bool mutable_read) {
  loom_type_id_t payload_id = LOOM_TYPE_ID_INVALID;
  check(loom_module_intern_type_id(module, payload_type, &payload_id));
  loom_type_t result;
  switch (handle_kind) {
    case ChannelHandleKind::Channel:
      check(loom_channel_type_make(module, payload_id, &result));
      break;
    case ChannelHandleKind::Read:
      check(loom_read_type_make(
          module, mutable_read ? LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE : 0,
          payload_id,
          mutable_read ? LOOM_READ_TYPE_MODE_MUTABLE
                       : static_cast<loom_read_type_mode_t>(0),
          &result));
      break;
    case ChannelHandleKind::Write:
      check(loom_write_type_make(module, payload_id, &result));
      break;
  }
  return result;
}

}  // namespace

loom_type_t BoundTypeStorage::with_dimensions(
    loom_type_t type, std::span<const uint64_t> dimensions) {
  bool all_static = true;
  for (auto dimension : dimensions) {
    all_static &= !loom_dim_is_dynamic(dimension);
  }
  uint8_t flags = all_static ? LOOM_TYPE_FLAG_ALL_STATIC : 0;
  if (dimensions.size() <= 2) {
    flags |= LOOM_TYPE_FLAG_INLINE_DIMS;
  }
  type.header =
      loom_type_make_header(loom_type_kind(type), loom_type_element_type(type),
                            static_cast<uint8_t>(dimensions.size()), flags) |
      (type.header & 0xFF000000u);
  type.dims[0] = type.dims[1] = 0;
  if (dimensions.size() <= 2) {
    std::copy(dimensions.begin(), dimensions.end(), type.dims);
  } else {
    type.dims[0] = reinterpret_cast<uintptr_t>(retain_dimensions(dimensions));
  }
  return type;
}

const loom_overflow_dim_t* BoundTypeStorage::retain_dimensions(
    std::span<const loom_overflow_dim_t> dimensions) {
  IREE_ASSERT(dimensions.size() >= 3 &&
              dimensions.size() <= LOOM_TYPE_MAX_RANK);
  auto& payload = dimension_payloads_.emplace_front();
  std::copy(dimensions.begin(), dimensions.end(), payload.begin());
  return payload.data();
}

void ShapeExtents::assign(std::span<const int64_t> extents) {
  IREE_ASSERT(!extents.empty() && extents.size() <= LOOM_TYPE_MAX_RANK);
  rank_ = extents.size();
  if (rank_ <= inline_extents_.size()) {
    std::copy(extents.begin(), extents.end(), inline_extents_.begin());
    overflow_extents_.reset();
    return;
  }
  overflow_extents_ = std::make_unique<int64_t[]>(rank_);
  std::copy(extents.begin(), extents.end(), overflow_extents_.get());
}

loom_type_t TilePartition::bind_type(
    std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(dependencies.size() == component_count - 1);
  if (!requires_binding()) {
    return static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  size_t dependency = 0;
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    dimensions[axis] = extents[axis] < 0
                           ? loom_dim_pack_dynamic(dependencies[dependency++])
                           : loom_dim_pack_static(extents[axis]);
  }
  std::optional<loom_value_id_t> encoding_value;
  if (encoding) {
    encoding_value = dependencies[dependency++];
  }
  IREE_ASSERT_EQ(dependency, dependencies.size());
  return make_encoded_shaped_type(
      LOOM_TYPE_TILE, element,
      std::span<const uint64_t>(dimensions).first(extents.size()),
      encoding_value, storage);
}

loom_value_id_t TilePartition::tile_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

loom_type_t ChannelPartition::bind_type(
    loom_module_t* module, std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(module);
  IREE_ASSERT(dependencies.size() == component_count - 1);
  if (!requires_binding()) {
    return static_type;
  }
  const auto payload_type = payload->bind_type(dependencies, storage);
  return make_channel_handle_type(module, payload_type, handle_kind,
                                  mutable_read);
}

loom_value_id_t ChannelPartition::handle_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

loom_type_t VectorPartition::bind_type(
    std::span<const loom_value_id_t> dynamic_extents,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(dynamic_extents.size() == component_count - 1);
  if (!is_dynamic()) {
    return static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  size_t dynamic_index = 0;
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    dimensions[axis] =
        extents[axis] < 0
            ? loom_dim_pack_dynamic(dynamic_extents[dynamic_index++])
            : loom_dim_pack_static(extents[axis]);
  }
  return make_shaped_type(
      LOOM_TYPE_VECTOR, element,
      std::span<const uint64_t>(dimensions).first(extents.size()), storage);
}

loom_value_id_t VectorPartition::vector_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

loom_type_t ViewPartition::bind_type(
    std::span<const loom_value_id_t> dependencies,
    BoundTypeStorage& storage) const {
  IREE_ASSERT(dependencies.size() == component_count - 1);
  if (!requires_binding()) {
    return static_type;
  }
  std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
  size_t dependency = 0;
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    dimensions[axis] = extents[axis] < 0
                           ? loom_dim_pack_dynamic(dependencies[dependency++])
                           : loom_dim_pack_static(extents[axis]);
  }
  std::optional<loom_value_id_t> encoding_value;
  if (encoding) {
    encoding_value = dependencies[dependency++];
  }
  IREE_ASSERT_EQ(dependency, dependencies.size());
  return make_encoded_shaped_type(
      LOOM_TYPE_VIEW, element,
      std::span<const uint64_t>(dimensions).first(extents.size()),
      encoding_value, storage);
}

loom_value_id_t ViewPartition::view_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

loom_type_t TensorPartition::bind_type(
    std::span<const loom_value_id_t> dynamic_extents) const {
  IREE_ASSERT(dynamic_extents.size() == (is_dynamic() ? 1u : 0u));
  auto dimension = is_dynamic() ? loom_dim_pack_dynamic(dynamic_extents[0])
                                : loom_dim_pack_static(extent);
  if (vector_lane_count) {
    return loom_type_shaped_2d(LOOM_TYPE_TENSOR, element, dimension,
                               loom_dim_pack_static(vector_lane_count), 0);
  }
  return loom_type_shaped_1d(LOOM_TYPE_TENSOR, element, dimension, 0);
}

loom_value_id_t TensorPartition::tensor_value(
    std::span<const loom_value_id_t> components) const {
  IREE_ASSERT(components.size() == component_count);
  return components.back();
}

const cxx::Type* Types::unqualified(const cxx::Type* type) {
  auto traits = unit_.typeTraits();
  return cxx::unaligned_type(traits.remove_restrict(traits.remove_cv(type)));
}

const cxx::Type* Types::unqualified_value(const cxx::Type* type) {
  auto traits = unit_.typeTraits();
  type = traits.remove_reference(type);
  return cxx::unaligned_type(traits.remove_restrict(traits.remove_cv(type)));
}

const cxx::VectorType* Types::native_vector(const cxx::Type* type) {
  return cxx::type_cast<cxx::VectorType>(unqualified(type));
}

bool Types::has_vector_object_padding(const cxx::Type* type, cxx::AST* owner) {
  auto* vector = native_vector(type);
  if (!vector) {
    return false;
  }
  auto* layout = unit_.control()->memoryLayout();
  auto object_bytes = layout->sizeOf(type);
  auto lane_bytes = layout->sizeOf(vector->elementType());
  if (!vector->elementCount() || !object_bytes || !lane_bytes || !*lane_bytes ||
      vector->elementCount() > UINT64_MAX / *lane_bytes) {
    diagnostics_.reject(unit_, owner, "vector has unknown object layout");
  }
  auto value_bytes = vector->elementCount() * *lane_bytes;
  if (*object_bytes < value_bytes) {
    diagnostics_.reject(unit_, owner,
                        "vector object cannot contain its logical lanes");
  }
  return *object_bytes > value_bytes;
}

loom_memory_access_flags_t Types::memory_access_flags(const cxx::Type* input) {
  return unit_.typeTraits().is_volatile(input)
             ? LOOM_MEMORY_ACCESS_FLAG_VOLATILE
             : 0;
}

const Partition& Types::partition(const cxx::Type* input, cxx::AST* owner) {
  if (auto* admitted = array(input, owner)) {
    return *admitted;
  }
  if (auto* admitted = special(input, owner)) {
    return *admitted;
  }
  if (auto* admitted = record(input, owner)) {
    return *admitted;
  }
  return loom_type_kind(get(input, owner)) == LOOM_TYPE_BUFFER
             ? kPointerPartition
             : kSSAPartition;
}

bool Types::requires_binding(const cxx::Type* input, cxx::AST* owner) {
  const auto& admitted = partition(input, owner);
  if ((admitted.kind == ValueKind::Vector &&
       static_cast<const VectorPartition&>(admitted).is_dynamic()) ||
      (admitted.kind == ValueKind::Tile &&
       static_cast<const TilePartition&>(admitted).requires_binding()) ||
      (admitted.kind == ValueKind::Channel &&
       static_cast<const ChannelPartition&>(admitted).requires_binding()) ||
      (admitted.kind == ValueKind::View &&
       static_cast<const ViewPartition&>(admitted).requires_binding()) ||
      (admitted.kind == ValueKind::Tensor &&
       static_cast<const TensorPartition&>(admitted).is_dynamic())) {
    return true;
  }
  return admitted.kind == ValueKind::Record
             ? static_cast<const RecordPartition&>(admitted).requires_binding
         : admitted.kind == ValueKind::Array
             ? static_cast<const ArrayPartition&>(admitted).requires_binding
             : false;
}

bool Types::requires_value_transport(const cxx::Type* input) {
  return object_properties(input).value_transport;
}

bool Types::has_volatile_subobject(const cxx::Type* input) {
  return unit_.typeTraits().is_volatile(input) ||
         object_properties(input).volatile_subobject;
}

Types::ObjectProperties Types::object_properties(const cxx::Type* input) {
  input = unqualified(input);
  auto traits = unit_.typeTraits();
  if (traits.is_pointer(input) || traits.is_reference(input) ||
      traits.is_null_pointer(input)) {
    return {.value_transport = true};
  }
  if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(input)) {
    auto result = object_properties(array->elementType());
    result.volatile_subobject |= traits.is_volatile(array->elementType());
    return result;
  }
  auto* record_type = cxx::type_cast<cxx::ClassType>(input);
  if (!record_type) {
    return {};
  }
  if (auto found = object_properties_.find(input);
      found != object_properties_.end()) {
    return found->second;
  }
  auto* source = record_type->definition();
  traits.require_complete_class(source);
  source = record_type->definition();
  ObjectProperties result = {
      .value_transport = type_binding(source) != nullptr,
  };
  auto append = [&](const cxx::Type* type) {
    const auto properties = object_properties(type);
    result.value_transport |= properties.value_transport;
    result.volatile_subobject |=
        traits.is_volatile(type) || properties.volatile_subobject;
  };
  for (auto* base : source->baseClasses()) {
    append(source_object_type(base));
  }
  for (auto* symbol : source->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
    if (field && !field->isStatic()) {
      append(field->type());
    }
  }
  object_properties_.emplace(input, result);
  return result;
}

std::span<const ValueProjection> Types::projections(const cxx::Type* input,
                                                    cxx::AST* owner) {
  input = unqualified(input);
  if (auto found = construction_groups_.find(input);
      found != construction_groups_.end()) {
    return found->second;
  }
  const auto& admitted = partition(input, owner);
  std::vector<ValueProjection> result;
  auto append = [&](const cxx::Type* type, size_t offset) {
    for (auto child : projections(type, owner)) {
      child.component_offset += offset;
      result.push_back(child);
    }
  };
  if (admitted.kind == ValueKind::Record && admitted.component_count) {
    for (const auto& member :
         static_cast<const RecordPartition&>(admitted).members) {
      auto traits = unit_.typeTraits();
      if (traits.is_reference(member.type())) {
        auto* field = static_cast<cxx::FieldSymbol*>(member.source);
        if (reference_is_snapshot(field)) {
          append(reference_transport_type(field), member.component_offset);
        } else {
          // Ordinary reference members retain their binding as a pointer
          // group. The referent is not a subobject of the record.
          result.push_back({reference_transport_type(field), member.partition,
                            member.component_offset});
        }
      } else {
        append(member.type(), member.component_offset);
      }
    }
  } else if (admitted.kind == ValueKind::Array) {
    const auto& array = static_cast<const ArrayPartition&>(admitted);
    for (size_t offset = 0; offset < admitted.component_count;
         offset += array.element->component_count) {
      append(array.source->elementType(), offset);
    }
  } else {
    result.push_back({input, &admitted, 0});
  }
  return construction_groups_.emplace(input, std::move(result)).first->second;
}

const ArrayPartition* Types::array(const cxx::Type* input, cxx::AST* owner) {
  auto* source = cxx::type_cast<cxx::BoundedArrayType>(unqualified(input));
  if (!source) {
    return nullptr;
  }
  if (auto found = arrays_.find(source); found != arrays_.end()) {
    return found->second.get();
  }
  const auto& element = partition(source->elementType(), owner);
  if (element.component_count &&
      source->size() > SIZE_MAX / element.component_count) {
    diagnostics_.reject(unit_, owner, "array value component count overflows");
  }
  auto result = std::make_unique<ArrayPartition>();
  result->kind = ValueKind::Array;
  result->component_count = source->size() * element.component_count;
  result->source = source;
  result->element = &element;
  result->requires_binding = requires_binding(source->elementType(), owner);
  result->has_mutable_subobject = has_mutable_subobject(element);
  auto* admitted = result.get();
  arrays_.emplace(source, std::move(result));
  return admitted;
}

const Partition* Types::special(const cxx::Type* input, cxx::AST* owner) {
  if (!input) {
    return nullptr;
  }
  auto* type = cxx::type_cast<cxx::ClassType>(unqualified(input));
  if (!type) {
    return nullptr;
  }
  auto* source = type->definition();
  auto* binding = type_binding(source);
  if (!binding) {
    return nullptr;
  }
  if (unit_.typeTraits().is_volatile(input)) {
    diagnostics_.reject(unit_, owner,
                        "volatile Loom source types require an addressable "
                        "storage representation");
  }
  if (binding->arguments.size() != 1) {
    diagnostics_.reject(unit_, owner,
                        "loom source types require one string type binding");
  }
  auto name = binding->arguments[0]->name();
  if (name == "buffer") {
    return buffer(type, owner);
  }
  if (name == "index") {
    return address(type, LOOM_SCALAR_TYPE_INDEX, owner);
  }
  if (name == "offset") {
    return address(type, LOOM_SCALAR_TYPE_OFFSET, owner);
  }
  if (name == "encoding") {
    return encoding(type, owner);
  }
  if (name == "pool") {
    return pool(type, owner);
  }
  if (name == "tile") {
    return tile(type, owner);
  }
  if (name == "channel" || name == "read" || name == "write") {
    return channel(type, name, owner);
  }
  if (name == "vector") {
    return semantic_vector(type, owner);
  }
  if (name == "view") {
    return view(type, owner);
  }
  if (name == "tensor") {
    return tensor(type, owner);
  }
  return opaque_dialect(type, name, owner);
}

const BufferPartition* Types::buffer(const cxx::ClassType* input,
                                     cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = buffers_.find(source); found != buffers_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "buffer values require a complete trivial class "
                        "without bases");
  }
  auto arguments = source->templateArguments();
  auto* element = arguments.size() == 1
                      ? cxx::template_argument_type(arguments[0])
                      : nullptr;
  if (!element) {
    diagnostics_.reject(
        unit_, owner,
        "buffer values require one supported non-boolean scalar or vector "
        "element type");
  }
  auto element_type = get(element, owner);
  if ((loom_type_kind(element_type) != LOOM_TYPE_SCALAR &&
       loom_type_kind(element_type) != LOOM_TYPE_VECTOR) ||
      loom_type_element_type(element_type) == LOOM_SCALAR_TYPE_I1) {
    diagnostics_.reject(
        unit_, owner,
        "buffer values require one supported non-boolean scalar or vector "
        "element type");
  }
  auto result = std::make_unique<BufferPartition>();
  result->kind = ValueKind::Buffer;
  result->component_count = 1;
  result->source = source;
  result->element_type = element;
  auto* admitted = result.get();
  buffers_.emplace(source, std::move(result));
  return admitted;
}

const AddressPartition* Types::address(const cxx::ClassType* input,
                                       loom_scalar_type_t element_type,
                                       cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = addresses_.find(source); found != addresses_.end()) {
    if (found->second->element_type != element_type) {
      diagnostics_.reject(unit_, owner,
                          "source class has conflicting address domains");
    }
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !source->templateArguments().empty() ||
      !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(
        unit_, owner,
        "address values require a complete non-template trivial "
        "class without bases");
  }
  auto result = std::make_unique<AddressPartition>();
  result->kind = ValueKind::Address;
  result->component_count = 1;
  result->source = source;
  result->element_type = element_type;
  auto* admitted = result.get();
  addresses_.emplace(source, std::move(result));
  return admitted;
}

const OpaqueDialectPartition* Types::opaque_dialect(const cxx::ClassType* input,
                                                    std::string_view name,
                                                    cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = opaque_dialects_.find(source);
      found != opaque_dialects_.end()) {
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
    diagnostics_.reject(
        unit_, owner,
        "dialect values require a complete trivial class without bases");
  }
  if (!module_) {
    diagnostics_.reject(unit_, owner,
                        "dialect source types require a destination module");
  }
  const auto name_view = iree_make_string_view(name.data(), name.size());
  const auto* descriptor =
      loom_type_registry_lookup(module_->context, name_view);
  if (!descriptor || descriptor->ir_kind != LOOM_TYPE_DIALECT ||
      descriptor->param_count != 0 || descriptor->format_element_count != 0) {
    diagnostics_.reject(
        unit_, owner,
        "Loom source type binding must name a registered opaque dialect type");
  }
  loom_string_id_t name_id;
  check(loom_module_intern_string(module_, name_view, &name_id));
  auto result = std::make_unique<OpaqueDialectPartition>();
  result->kind = ValueKind::OpaqueDialect;
  result->component_count = 1;
  result->source = source;
  result->type = loom_type_dialect_opaque(name_id);
  auto* admitted = result.get();
  opaque_dialects_.emplace(source, std::move(result));
  return admitted;
}

const EncodingPartition* Types::encoding(const cxx::ClassType* input,
                                         cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = encodings_.find(source); found != encodings_.end()) {
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
    diagnostics_.reject(
        unit_, owner,
        "encoding values require a complete trivial class without bases");
  }
  auto arguments = source->templateArguments();
  if (arguments.size() != 2) {
    diagnostics_.reject(unit_, owner,
                        "encoding values require role and rank arguments");
  }
  cxx::ASTInterpreter interpreter(&unit_);
  auto role = cxx::template_argument_value(arguments[0]);
  auto rank = cxx::template_argument_value(arguments[1]);
  auto role_value = role ? interpreter.toInt(*role) : std::nullopt;
  auto rank_value = rank ? interpreter.toInt(*rank) : std::nullopt;
  bool ranked = (role_value == 0 || role_value == 2) && rank_value &&
                *rank_value >= 1 && *rank_value <= LOOM_TYPE_MAX_RANK;
  bool unranked = (role_value == 1 || role_value == 3) && rank_value == 0;
  if (!ranked && !unranked) {
    diagnostics_.reject(
        unit_, owner,
        "encoding values require layout/storage rank in [1, 15] or "
        "schema/transform rank zero");
  }
  auto result = std::make_unique<EncodingPartition>();
  result->kind = ValueKind::Encoding;
  result->component_count = 1;
  result->source = source;
  switch (*role_value) {
    case 0:
      result->role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
      break;
    case 1:
      result->role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA;
      break;
    case 2:
      result->role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE;
      break;
    case 3:
      result->role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM;
      break;
  }
  result->rank = static_cast<size_t>(*rank_value);
  auto* admitted = result.get();
  encodings_.emplace(source, std::move(result));
  return admitted;
}

const PoolPartition* Types::pool(const cxx::ClassType* input, cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = pools_.find(source); found != pools_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !source->templateArguments().empty() ||
      !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "pool values require a complete non-template trivial "
                        "class without bases");
  }
  auto result = std::make_unique<PoolPartition>();
  result->kind = ValueKind::Pool;
  result->component_count = 1;
  result->source = source;
  auto* admitted = result.get();
  pools_.emplace(source, std::move(result));
  return admitted;
}

ShapeExtents Types::parse_shape(const cxx::Type* input,
                                std::string_view value_kind, cxx::AST* owner) {
  auto* shape_type =
      input ? cxx::type_cast<cxx::ClassType>(unqualified(input)) : nullptr;
  auto* shape_source = shape_type ? shape_type->definition() : nullptr;
  auto* shape_binding = shape_source ? type_binding(shape_source) : nullptr;
  if (!shape_binding || shape_binding->arguments.size() != 1 ||
      shape_binding->arguments[0]->name() != "shape") {
    diagnostics_.reject(unit_, owner,
                        std::string(value_kind) +
                            " shape must be a loom::type::shape "
                            "specialization");
  }
  auto* shape_pack = template_argument_pack(shape_source);
  auto shape_arguments =
      shape_pack ? std::span<cxx::Symbol* const>(shape_pack->elements())
                 : std::span<cxx::Symbol* const>{};
  if (shape_arguments.empty() || shape_arguments.size() > LOOM_TYPE_MAX_RANK) {
    diagnostics_.reject(unit_, owner,
                        std::string(value_kind) +
                            " rank must be in Loom's supported range [1, 15]");
  }

  cxx::ASTInterpreter interpreter(&unit_);
  auto pointer_bytes = unit_.control()->memoryLayout()->sizeOfPointer();
  uint64_t dynamic = pointer_bytes == 4 ? UINT32_MAX : UINT64_MAX;
  std::array<int64_t, LOOM_TYPE_MAX_RANK> parsed_extents;
  for (size_t axis = 0; axis < shape_arguments.size(); ++axis) {
    auto value = cxx::template_argument_value(shape_arguments[axis]);
    auto number = value ? interpreter.toInt(*value) : std::nullopt;
    if (!number) {
      diagnostics_.reject(
          unit_, owner,
          std::string(value_kind) + " extents require constant size arguments");
    }
    uint64_t bits = static_cast<uint64_t>(*number);
    if (pointer_bytes == 4) {
      bits &= UINT32_MAX;
    }
    if (bits == dynamic) {
      parsed_extents[axis] = -1;
    } else if (*number < 0 || bits > LOOM_DIM_MAX_STATIC_SIZE) {
      diagnostics_.reject(unit_, owner,
                          std::string(value_kind) +
                              " extent is outside Loom's static dimension "
                              "range");
    } else {
      parsed_extents[axis] = static_cast<int64_t>(bits);
    }
  }
  ShapeExtents result;
  result.assign(
      std::span<const int64_t>(parsed_extents).first(shape_arguments.size()));
  return result;
}

const TilePartition* Types::tile(const cxx::ClassType* input, cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = tiles_.find(source); found != tiles_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "tile values require a complete trivial class without "
                        "bases");
  }
  auto arguments = source->templateArguments();
  if (arguments.size() != 3) {
    diagnostics_.reject(
        unit_, owner,
        "tile values require shape, element, and optional encoding type "
        "arguments");
  }
  auto extents =
      parse_shape(cxx::template_argument_type(arguments[0]), "tile", owner);
  auto* element_type = cxx::template_argument_type(arguments[1]);
  if (!element_type || traits.is_const(element_type) ||
      traits.is_volatile(element_type)) {
    diagnostics_.reject(unit_, owner,
                        "tile elements require an unqualified source type");
  }
  auto element = get(element_type, owner);
  if (loom_type_kind(element) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, owner,
                        "tile elements require a supported scalar type");
  }
  const EncodingPartition* tile_encoding = nullptr;
  auto* encoding_type = cxx::template_argument_type(arguments[2]);
  if (!encoding_type) {
    diagnostics_.reject(unit_, owner,
                        "tile encoding requires a source type argument");
  }
  if (unqualified(encoding_type)->kind() != cxx::TypeKind::kVoid) {
    const auto& admitted_encoding = partition(encoding_type, owner);
    if (admitted_encoding.kind != ValueKind::Encoding) {
      diagnostics_.reject(unit_, owner,
                          "tile encoding must be void or a Loom encoding type");
    }
    tile_encoding = &static_cast<const EncodingPartition&>(admitted_encoding);
    if ((tile_encoding->role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT ||
         tile_encoding->role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) &&
        tile_encoding->rank != extents.size()) {
      diagnostics_.reject(unit_, owner,
                          "tile address encoding must have the same rank");
    }
  }
  size_t dynamic_count = 0;
  for (auto extent : extents) {
    dynamic_count += extent < 0;
  }
  auto result = std::make_unique<TilePartition>();
  result->kind = ValueKind::Tile;
  result->component_count = dynamic_count + (tile_encoding ? 2 : 1);
  result->source = source;
  result->element_type = element_type;
  result->element = loom_type_element_type(element);
  result->encoding = tile_encoding;
  result->extents = std::move(extents);
  for (size_t axis = 0; axis < result->extents.size(); ++axis) {
    if (result->extents[axis] < 0) {
      result->component_names.push_back("dim" + std::to_string(axis));
    }
  }
  if (tile_encoding) {
    result->component_names.emplace_back("encoding");
  }
  result->component_names.emplace_back();
  if (!result->requires_binding()) {
    std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
    for (size_t axis = 0; axis < result->extents.size(); ++axis) {
      dimensions[axis] = loom_dim_pack_static(result->extents[axis]);
    }
    result->static_type = make_shaped_type(
        LOOM_TYPE_TILE, result->element,
        std::span<const uint64_t>(dimensions).first(result->extents.size()),
        result->static_type_storage);
  }
  auto* admitted = result.get();
  tiles_.emplace(source, std::move(result));
  return admitted;
}

const ChannelPartition* Types::channel(const cxx::ClassType* input,
                                       std::string_view name, cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = channels_.find(source); found != channels_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(
        unit_, owner,
        "channel handles require a complete trivial class without bases");
  }
  auto arguments = source->templateArguments();
  const size_t expected_count = name == "read" ? 2 : 1;
  if (arguments.size() != expected_count) {
    diagnostics_.reject(
        unit_, owner,
        name == "read"
            ? "read values require payload and access mode arguments"
            : std::string(name) + " values require one payload argument");
  }
  auto* payload_type = cxx::template_argument_type(arguments[0]);
  const auto* payload_partition =
      payload_type ? &partition(payload_type, owner) : nullptr;
  if (!payload_partition || payload_partition->kind != ValueKind::Tile) {
    diagnostics_.reject(
        unit_, owner, std::string(name) + " payload must be a Loom tile type");
  }
  const auto* payload = static_cast<const TilePartition*>(payload_partition);
  bool mutable_read = false;
  if (name == "read") {
    cxx::ASTInterpreter interpreter(&unit_);
    auto mode = cxx::template_argument_value(arguments[1]);
    auto mode_value = mode ? interpreter.toInt(*mode) : std::nullopt;
    if (!mode_value || (*mode_value != 0 && *mode_value != 1)) {
      diagnostics_.reject(unit_, owner,
                          "read access mode must be immutable or mutable");
    }
    mutable_read = *mode_value == 1;
  }
  if (!module_) {
    diagnostics_.reject(unit_, owner,
                        "channel source types require a destination module");
  }
  auto result = std::make_unique<ChannelPartition>();
  result->kind = ValueKind::Channel;
  result->component_count = payload->component_count;
  result->source = source;
  result->payload = payload;
  result->handle_kind = name == "channel" ? ChannelHandleKind::Channel
                        : name == "read"  ? ChannelHandleKind::Read
                                          : ChannelHandleKind::Write;
  result->mutable_read = mutable_read;
  result->component_names = payload->component_names;
  if (!result->requires_binding()) {
    result->static_type = make_channel_handle_type(
        module_, payload->static_type, result->handle_kind, mutable_read);
  }
  auto* admitted = result.get();
  channels_.emplace(source, std::move(result));
  return admitted;
}

const VectorPartition* Types::semantic_vector(const cxx::ClassType* input,
                                              cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = semantic_vectors_.find(source);
      found != semantic_vectors_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "vector values require a complete trivial class "
                        "without bases");
  }
  auto arguments = source->templateArguments();
  if (arguments.size() != 2) {
    diagnostics_.reject(unit_, owner,
                        "vector values require shape and element arguments");
  }
  auto extents =
      parse_shape(cxx::template_argument_type(arguments[0]), "vector", owner);
  auto* element_type = cxx::template_argument_type(arguments[1]);
  if (!element_type || traits.is_const(element_type) ||
      traits.is_volatile(element_type)) {
    diagnostics_.reject(unit_, owner,
                        "vector elements require an unqualified source type");
  }
  auto element = get(element_type, owner);
  if (loom_type_kind(element) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, owner,
                        "vector elements require a supported scalar type");
  }
  size_t dynamic_count = 0;
  for (auto extent : extents) {
    if (extent < 0) {
      ++dynamic_count;
    }
  }
  auto result = std::make_unique<VectorPartition>();
  result->kind = ValueKind::Vector;
  result->component_count = dynamic_count + 1;
  result->source = source;
  result->element_type = element_type;
  result->element = loom_type_element_type(element);
  result->extents = std::move(extents);
  for (size_t axis = 0; axis < result->extents.size(); ++axis) {
    if (result->extents[axis] < 0) {
      result->component_names.push_back("dim" + std::to_string(axis));
    }
  }
  result->component_names.emplace_back();
  if (!result->is_dynamic()) {
    std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
    for (size_t axis = 0; axis < result->extents.size(); ++axis) {
      dimensions[axis] = loom_dim_pack_static(result->extents[axis]);
    }
    result->static_type = make_shaped_type(
        LOOM_TYPE_VECTOR, result->element,
        std::span<const uint64_t>(dimensions).first(result->extents.size()),
        result->static_type_storage);
  }
  auto* admitted = result.get();
  semantic_vectors_.emplace(source, std::move(result));
  return admitted;
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
  auto element = get(element_type, owner);
  if (loom_type_kind(element) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(element) == LOOM_SCALAR_TYPE_I1) {
    diagnostics_.reject(
        unit_, owner,
        "view elements require a supported non-boolean scalar type");
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
  result->component_count = dynamic_count + (view_encoding ? 2 : 1);
  result->source = source;
  result->element_type = element_type;
  result->element = loom_type_element_type(element);
  result->access_flags = memory_access_flags(element_type);
  result->encoding = view_encoding;
  result->extents = std::move(extents);
  for (size_t axis = 0; axis < result->extents.size(); ++axis) {
    if (result->extents[axis] < 0) {
      result->component_names.push_back("dim" + std::to_string(axis));
    }
  }
  if (view_encoding) {
    result->component_names.push_back(
        view_encoding->role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT ? "layout"
                                                                 : "storage");
  }
  result->component_names.emplace_back();
  if (!result->requires_binding()) {
    std::array<uint64_t, LOOM_TYPE_MAX_RANK> dimensions;
    for (size_t axis = 0; axis < result->extents.size(); ++axis) {
      dimensions[axis] = loom_dim_pack_static(result->extents[axis]);
    }
    result->static_type = make_encoded_shaped_type(
        LOOM_TYPE_VIEW, result->element,
        std::span<const uint64_t>(dimensions).first(result->extents.size()),
        std::nullopt, result->static_type_storage);
  }
  auto* admitted = result.get();
  views_.emplace(source, std::move(result));
  return admitted;
}

const TensorPartition* Types::tensor(const cxx::ClassType* input,
                                     cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = tensors_.find(source); found != tensors_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "tensor values require a complete trivial class "
                        "without bases");
  }
  auto arguments = source->templateArguments();
  if (arguments.size() != 2) {
    diagnostics_.reject(unit_, owner,
                        "tensor values require shape and element arguments");
  }
  auto extents =
      parse_shape(cxx::template_argument_type(arguments[0]), "tensor", owner);
  if (extents.size() != 1) {
    diagnostics_.reject(unit_, owner,
                        "check tensors currently require rank one");
  }
  auto* element = cxx::template_argument_type(arguments[1]);
  auto projected_element = get(element, owner);
  auto element_kind = loom_type_kind(projected_element);
  auto* source_vector = native_vector(element);
  if (source_vector && has_vector_object_padding(element, owner)) {
    diagnostics_.reject(unit_, owner,
                        "tensor elements require vectors without object "
                        "padding");
  }
  auto* lane_type = source_vector ? source_vector->elementType() : element;
  auto bytes = unit_.control()->memoryLayout()->sizeOf(element);
  if ((element_kind != LOOM_TYPE_SCALAR && element_kind != LOOM_TYPE_VECTOR) ||
      loom_type_element_type(projected_element) == LOOM_SCALAR_TYPE_I1 ||
      traits.is_const(element) || traits.is_volatile(element) || !bytes ||
      !*bytes) {
    diagnostics_.reject(unit_, owner,
                        "tensor elements require unqualified non-boolean "
                        "scalar or vector types");
  }
  if (extents[0] >= 0 &&
      static_cast<uint64_t>(extents[0]) > INT64_MAX / *bytes) {
    diagnostics_.reject(unit_, owner,
                        "tensor extent has an unrepresentable byte length");
  }
  auto result = std::make_unique<TensorPartition>();
  result->kind = ValueKind::Tensor;
  result->component_count = extents[0] < 0 ? 2 : 1;
  result->source = source;
  result->element_type = element;
  result->lane_type = lane_type;
  result->element = loom_type_element_type(projected_element);
  result->vector_lane_count = source_vector ? source_vector->elementCount() : 0;
  result->extent = extents[0];
  result->element_bytes = *bytes;
  auto* admitted = result.get();
  tensors_.emplace(source, std::move(result));
  return admitted;
}

const cxx::Type* SubobjectPartition::type() const {
  return source_object_type(source);
}

uint64_t SubobjectPartition::byte_offset() const {
  return source_subobject_offset(source);
}

const RecordPartition* Types::record(const cxx::Type* input, cxx::AST* owner) {
  if (!input) {
    return nullptr;
  }
  auto* type = cxx::type_cast<cxx::ClassType>(unqualified(input));
  if (!type) {
    return nullptr;
  }
  if (special(input, owner)) {
    return nullptr;
  }
  auto traits = unit_.typeTraits();
  auto* source = type->definition();
  if (auto found = records_.find(source); found != records_.end()) {
    return found->second.get();
  }
  if (source) {
    traits.require_complete_class(source);
    source = type->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      source->isPolymorphic()) {
    diagnostics_.reject(
        unit_, owner,
        "record values require a complete non-union type without "
        "virtual members");
  }
  auto result = std::make_unique<RecordPartition>();
  result->kind = ValueKind::Record;
  result->component_count = 0;
  result->source = source;
  result->is_aggregate = traits.is_aggregate(input);
  auto append = [&](cxx::Symbol* symbol) {
    auto* type = source_object_type(symbol);
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
    if (field && traits.is_restrict(type)) {
      diagnostics_.reject(
          unit_, owner,
          "restrict-qualified fields require an object-member access scope");
    }
    const bool snapshot = field && traits.is_lvalue_reference(type) &&
                          reference_is_snapshot(field);
    const Partition* member_partition = nullptr;
    if (traits.is_lvalue_reference(type)) {
      member_partition = snapshot
                             ? &partition(traits.remove_reference(type), owner)
                             : &kPointerPartition;
    } else {
      member_partition = &partition(type, owner);
    }
    if (symbol->kind() == cxx::SymbolKind::kBaseClass &&
        member_partition->kind != ValueKind::Record) {
      diagnostics_.reject(unit_, owner,
                          "Loom semantic handles cannot be base subobjects");
    }
    SubobjectPartition member{symbol, member_partition,
                              result->component_count};
    result->members.push_back(member);
    members_.emplace(symbol, member);
    append_component_names(*member_partition, cxx::to_string(symbol->name()),
                           result->component_names);
    result->component_count += member_partition->component_count;
    result->has_mutable_subobject |= (field && field->isMutable()) ||
                                     has_mutable_subobject(*member_partition);
    if (!traits.is_reference(type)) {
      result->requires_binding |= requires_binding(type, owner);
    }
  };
  for (auto* base : source->baseClasses()) {
    if (base->isVirtual()) {
      diagnostics_.reject(
          unit_, owner, "virtual base objects require a dynamic object layout");
    }
    append(base);
  }
  for (auto* symbol : source->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
    if (!field || field->isStatic()) {
      continue;
    }
    if (field->isBitField()) {
      diagnostics_.reject(unit_, owner,
                          "record value fields cannot be bitfields");
    }
    if (traits.is_rvalue_reference(field->type())) {
      diagnostics_.reject(
          unit_, owner,
          "rvalue-reference fields require consuming object identity");
    }
    append(field);
  }
  auto* admitted = result.get();
  records_.emplace(source, std::move(result));
  return admitted;
}

bool Types::implicit_object_may_mutate(cxx::FunctionSymbol* function,
                                       cxx::AST* owner) {
  if (!function || !function->isImplicitObjectMemberFunction()) {
    return false;
  }
  auto* signature = static_cast<const cxx::FunctionType*>(function->type());
  if (!cxx::has_const(signature->cvQualifiers())) {
    return true;
  }
  const auto* admitted = record(function->parent()->type(), owner);
  return admitted && admitted->has_mutable_subobject;
}

bool Types::record_constructor_requires_call(const cxx::Type* type,
                                             cxx::FunctionSymbol* constructor,
                                             cxx::AST* owner) {
  if (!constructor || !constructor->isConstructor()) {
    return false;
  }
  type = unqualified(type);
  while (auto* array = cxx::type_cast<cxx::BoundedArrayType>(type)) {
    type = unqualified(array->elementType());
  }
  auto* admitted = record(type, owner);
  return admitted && (unit_.typeTraits().is_user_provided(constructor) ||
                      (!same_function(constructor,
                                      admitted->source->defaultConstructor()) &&
                       !is_trivial_value_operation(constructor)));
}

bool Types::is_trivial_value_operation(cxx::FunctionSymbol* function) {
  return unit_.typeTraits().is_trivial_special_member(function) &&
         !function->isDestructor();
}

void Types::append_component_names(const Partition& partition,
                                   std::string_view prefix,
                                   std::vector<std::string>& output) {
  auto append = [&](std::string_view suffix) {
    if (suffix.empty()) {
      output.emplace_back(prefix);
    } else {
      output.emplace_back(std::string(prefix) + "_" + std::string(suffix));
    }
  };
  switch (partition.kind) {
    case ValueKind::SSA:
    case ValueKind::Buffer:
    case ValueKind::Address:
    case ValueKind::OpaqueDialect:
    case ValueKind::Encoding:
    case ValueKind::Pool:
      append({});
      return;
    case ValueKind::Tile: {
      const auto& tile = static_cast<const TilePartition&>(partition);
      for (const auto& name : tile.component_names) {
        append(name);
      }
      return;
    }
    case ValueKind::Channel: {
      const auto& channel = static_cast<const ChannelPartition&>(partition);
      for (const auto& name : channel.component_names) {
        append(name);
      }
      return;
    }
    case ValueKind::Vector: {
      const auto& vector = static_cast<const VectorPartition&>(partition);
      for (const auto& name : vector.component_names) {
        append(name);
      }
      return;
    }
    case ValueKind::Tensor: {
      const auto& tensor = static_cast<const TensorPartition&>(partition);
      if (tensor.is_dynamic()) {
        append("dim0");
      }
      append({});
      return;
    }
    case ValueKind::Pointer:
      append({});
      append("byte_offset");
      return;
    case ValueKind::Record: {
      const auto& record = static_cast<const RecordPartition&>(partition);
      for (const auto& name : record.component_names) {
        append(name);
      }
      return;
    }
    case ValueKind::Array: {
      const auto& array = static_cast<const ArrayPartition&>(partition);
      for (size_t index = 0; index < array.source->size(); ++index) {
        append_component_names(
            *array.element, std::string(prefix) + "_" + std::to_string(index),
            output);
      }
      return;
    }
    case ValueKind::View: {
      const auto& view = static_cast<const ViewPartition&>(partition);
      for (const auto& name : view.component_names) {
        append(name);
      }
      return;
    }
  }
}

const SubobjectPartition& Types::member(cxx::Symbol* subobject,
                                        cxx::AST* owner) {
  if (auto found = members_.find(subobject); found != members_.end()) {
    return found->second;
  }
  record(subobject->parent()->type(), owner);
  return members_.at(subobject);
}

const SubobjectProjection& Types::projection(cxx::List<cxx::Symbol*>* path,
                                             cxx::AST* owner) {
  if (!projections_) {
    projections_ = std::make_unique<ProjectionMap>();
  }
  auto [entry, inserted] = projections_->try_emplace(path);
  if (entry->second.partition) {
    return entry->second;
  }
  const auto& root = member(path->value, owner);
  SubobjectProjection result{
      &root, root.partition, root.component_offset,
      inserted ? root.byte_offset() : entry->second.byte_offset};
  for (auto* step : cxx::ListView{path->next}) {
    const auto& selected = member(step, owner);
    result.partition = selected.partition;
    result.component_offset += selected.component_offset;
    if (inserted) {
      result.byte_offset += selected.byte_offset();
    }
  }
  return entry->second = result;
}

uint64_t Types::subobject_offset(cxx::List<cxx::Symbol*>* path) {
  if (!projections_) {
    projections_ = std::make_unique<ProjectionMap>();
  }
  auto [entry, inserted] = projections_->try_emplace(path);
  if (inserted) {
    for (auto* step : cxx::ListView{path}) {
      entry->second.byte_offset += source_subobject_offset(step);
    }
  }
  return entry->second.byte_offset;
}

SubobjectProjection Types::selection(cxx::Symbol* field,
                                     cxx::List<cxx::Symbol*>* path,
                                     cxx::AST* owner) {
  const auto& leaf = member(field, owner);
  if (!path) {
    return {&leaf, leaf.partition, leaf.component_offset, leaf.byte_offset()};
  }
  auto result = projection(path, owner);
  result.partition = leaf.partition;
  result.component_offset += leaf.component_offset;
  result.byte_offset += leaf.byte_offset();
  return result;
}

Types::MemberObjectPath* Types::member_object_path(
    cxx::MemberExpressionAST* member_expression) {
  auto classes = member_object_classes(unit_, member_expression);
  IREE_ASSERT(classes.has_value());
  if (classes->object == classes->declaring) {
    return nullptr;
  }
  if (member_object_paths_) {
    auto object = member_object_paths_->find(classes->object);
    if (object != member_object_paths_->end()) {
      auto path = object->second.find(classes->declaring);
      if (path != object->second.end()) {
        return &path->second;
      }
    }
  }

  auto path = classes->object->subobjectPath(classes->declaring);
  IREE_ASSERT(path.has_value() && !path->empty());
  MemberObjectPath result{.symbols = std::move(*path)};
  for (auto* step : result.symbols) {
    result.byte_offset += source_subobject_offset(step);
  }
  if (!member_object_paths_) {
    member_object_paths_ = std::make_unique<MemberObjectPathMap>();
  }
  auto& paths = (*member_object_paths_)[classes->object];
  return &paths.emplace(classes->declaring, std::move(result)).first->second;
}

const SubobjectProjection* Types::member_object_projection(
    cxx::MemberExpressionAST* member_expression, cxx::AST* owner) {
  auto* path = member_object_path(member_expression);
  if (!path) {
    return nullptr;
  }
  if (path->projection) {
    return &*path->projection;
  }
  const auto& root = member(path->symbols.front(), owner);
  SubobjectProjection result{
      .root = &root,
      .partition = root.partition,
      .component_offset = root.component_offset,
      .byte_offset = path->byte_offset,
  };
  for (auto* step : std::span<cxx::Symbol* const>{path->symbols}.subspan(1)) {
    const auto& selected = this->member(step, owner);
    result.partition = selected.partition;
    result.component_offset += selected.component_offset;
  }
  path->projection = result;
  return &*path->projection;
}

MemberObjectSelection Types::member_object_selection(
    cxx::MemberExpressionAST* member_expression, cxx::AST* owner) {
  MemberObjectSelection result{.object = member_expression->baseExpression};
  auto* adjusted = unwrap_object_identity(unit_, result.object);
  auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(adjusted);
  if (cast &&
      cast->castKind == cxx::ImplicitCastKind::kDerivedToBaseConversion) {
    result.object = cast->expression;
    result.projection = projection(cast->subobjectPath, owner);
  }
  adjusted = unwrap_object_identity(unit_, result.object);
  cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(adjusted);
  if (cast &&
      cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion) {
    auto* inner = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(
        unwrap_object_identity(unit_, cast->expression));
    if (inner &&
        inner->castKind == cxx::ImplicitCastKind::kDerivedToBaseConversion &&
        cast->type == inner->expression->type &&
        same_subobject_path(cast->subobjectPath, inner->subobjectPath)) {
      // Template instantiation may preserve a selected member's original
      // derived object by wrapping the frontend's base adjustment in its exact
      // inverse. The pair is an identity; the member's declaring-class path
      // below remains the one semantic projection.
      result.object = inner->expression;
    }
  }
  auto* implicit = member_object_projection(member_expression, owner);
  if (!implicit) {
    return result;
  }
  if (!result.projection) {
    result.projection = *implicit;
    return result;
  }
  result.projection->partition = implicit->partition;
  result.projection->component_offset += implicit->component_offset;
  result.projection->byte_offset += implicit->byte_offset;
  return result;
}

uint64_t Types::member_object_offset(
    cxx::MemberExpressionAST* member_expression) {
  auto* path = member_object_path(member_expression);
  return path ? path->byte_offset : 0;
}

SubobjectProjection Types::selection(cxx::MemberExpressionAST* expression,
                                     cxx::AST* owner) {
  const auto& leaf = member(expression->symbol, owner);
  auto* object = member_object_projection(expression, owner);
  if (!object) {
    return {&leaf, leaf.partition, leaf.component_offset, leaf.byte_offset()};
  }
  auto result = *object;
  result.partition = leaf.partition;
  result.component_offset += leaf.component_offset;
  result.byte_offset += leaf.byte_offset();
  return result;
}

void Types::select_reference_snapshot(cxx::FieldSymbol* field) {
  auto traits = unit_.typeTraits();
  IREE_ASSERT(traits.is_lvalue_reference(field->type()));
  auto* referent = traits.remove_reference(field->type());
  IREE_ASSERT(!traits.is_volatile(referent));
  if (reference_snapshots_.contains(field)) {
    return;
  }
  IREE_ASSERT(traits.is_pointer(referent) || !members_.contains(field));
  reference_snapshots_.insert(field);
}

bool Types::reference_is_snapshot(cxx::FieldSymbol* field) const {
  return reference_snapshots_.contains(field);
}

const cxx::Type* Types::reference_transport_type(
    cxx::FieldSymbol* field) const {
  auto traits = unit_.typeTraits();
  IREE_ASSERT(traits.is_lvalue_reference(field->type()));
  auto* referent = traits.remove_reference(field->type());
  return reference_is_snapshot(field) ? referent : traits.add_pointer(referent);
}

bool Types::require_reference_capture_identity(cxx::FieldSymbol* field) {
  auto* closure = cxx::symbol_cast<cxx::ClassSymbol>(field->parent());
  IREE_ASSERT(closure && closure->isClosureType());
  IREE_ASSERT(unit_.typeTraits().is_lvalue_reference(field->type()));
  return identity_reference_captures_.insert(field).second;
}

bool Types::reference_capture_requires_identity(cxx::FieldSymbol* field) const {
  return identity_reference_captures_.contains(field);
}

loom_type_t Types::get(const cxx::Type* input, cxx::AST* ast) {
  if (!input) {
    diagnostics_.reject(unit_, ast, "expression has no resolved C++ type");
  }
  switch (unqualified(input)->kind()) {
    case cxx::TypeKind::kBool:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    case cxx::TypeKind::kInt:
    case cxx::TypeKind::kUnsignedInt:
    case cxx::TypeKind::kChar32:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    case cxx::TypeKind::kChar:
    case cxx::TypeKind::kSignedChar:
    case cxx::TypeKind::kUnsignedChar:
    case cxx::TypeKind::kChar8:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I8);
    case cxx::TypeKind::kShortInt:
    case cxx::TypeKind::kUnsignedShortInt:
    case cxx::TypeKind::kChar16:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I16);
    case cxx::TypeKind::kWideChar:
      return loom_type_scalar(unit_.control()->memoryLayout()->sizeOf(input) ==
                                      2
                                  ? LOOM_SCALAR_TYPE_I16
                                  : LOOM_SCALAR_TYPE_I32);
    case cxx::TypeKind::kLongInt:
    case cxx::TypeKind::kUnsignedLongInt:
      return loom_type_scalar(unit_.control()->memoryLayout()->sizeOfLong() == 4
                                  ? LOOM_SCALAR_TYPE_I32
                                  : LOOM_SCALAR_TYPE_I64);
    case cxx::TypeKind::kLongLongInt:
    case cxx::TypeKind::kUnsignedLongLongInt:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    case cxx::TypeKind::kFloat:
      return loom_type_scalar(LOOM_SCALAR_TYPE_F32);
    case cxx::TypeKind::kDouble:
      return loom_type_scalar(LOOM_SCALAR_TYPE_F64);
    case cxx::TypeKind::kFloat16:
      return loom_type_scalar(LOOM_SCALAR_TYPE_F16);
    case cxx::TypeKind::kBFloat16:
      return loom_type_scalar(LOOM_SCALAR_TYPE_BF16);
    case cxx::TypeKind::kFloat8E4M3FN:
      return loom_type_scalar(LOOM_SCALAR_TYPE_F8E4M3);
    case cxx::TypeKind::kFloat8E5M2:
      return loom_type_scalar(LOOM_SCALAR_TYPE_F8E5M2);
    case cxx::TypeKind::kEnum:
    case cxx::TypeKind::kScopedEnum: {
      auto* underlying = unit_.typeTraits().underlying_type(input);
      if (underlying == unqualified(input)) {
        diagnostics_.reject(unit_, ast,
                            "enum has no resolved underlying representation");
      }
      return get(underlying, ast);
    }
    case cxx::TypeKind::kVector: {
      auto* source = native_vector(input);
      auto element = get(source->elementType(), ast);
      if (loom_type_kind(element) != LOOM_TYPE_SCALAR ||
          loom_type_element_type(element) == LOOM_SCALAR_TYPE_I1 ||
          !source->elementCount() ||
          source->elementCount() > LOOM_DIM_MAX_STATIC_SIZE) {
        diagnostics_.reject(unit_, ast,
                            "vectors require non-boolean scalar lanes");
      }
      // Logical SSA shape is independent of the source object's tail padding.
      // Storage admission and pointer arithmetic retain the complete footprint.
      has_vector_object_padding(input, ast);
      return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                                 loom_type_element_type(element),
                                 source->elementCount(), 0);
    }
    case cxx::TypeKind::kBoundedArray: {
      auto* array = cxx::type_cast<cxx::BoundedArrayType>(unqualified(input));
      auto element = get(array->elementType(), ast);
      if (loom_type_kind(element) != LOOM_TYPE_SCALAR) {
        diagnostics_.reject(unit_, ast,
                            "arrays require a supported scalar element");
      }
      return loom_type_buffer();
    }
    case cxx::TypeKind::kNullptr:
      return loom_type_buffer();
    case cxx::TypeKind::kPointer: {
      auto* pointer = cxx::type_cast<cxx::PointerType>(unqualified(input));
      if (unit_.typeTraits().is_function(pointer->elementType())) {
        diagnostics_.reject(
            unit_, ast,
            "function pointers have no object pointer representation");
      }
      return loom_type_buffer();
    }
    case cxx::TypeKind::kClass: {
      auto* admitted = special(input, ast);
      if (admitted && admitted->kind == ValueKind::Buffer) {
        return loom_type_buffer();
      }
      if (admitted && admitted->kind == ValueKind::Address) {
        return loom_type_scalar(
            static_cast<const AddressPartition*>(admitted)->element_type);
      }
      if (admitted && admitted->kind == ValueKind::Pool) {
        return loom_type_pool();
      }
      if (admitted && admitted->kind == ValueKind::Tile) {
        const auto& tile = static_cast<const TilePartition&>(*admitted);
        if (tile.requires_binding()) {
          diagnostics_.reject(
              unit_, ast,
              "dependent tile types require bound component identities");
        }
        return tile.static_type;
      }
      if (admitted && admitted->kind == ValueKind::Channel) {
        const auto& channel = static_cast<const ChannelPartition&>(*admitted);
        if (channel.requires_binding()) {
          diagnostics_.reject(
              unit_, ast,
              "dependent channel types require bound component identities");
        }
        return channel.static_type;
      }
      if (admitted && admitted->kind == ValueKind::Vector) {
        const auto& vector = static_cast<const VectorPartition&>(*admitted);
        if (vector.is_dynamic()) {
          diagnostics_.reject(
              unit_, ast,
              "dynamic vector types require bound extent identities");
        }
        return vector.static_type;
      }
      if (admitted && admitted->kind == ValueKind::Tensor) {
        const auto& tensor = static_cast<const TensorPartition&>(*admitted);
        if (tensor.is_dynamic()) {
          diagnostics_.reject(
              unit_, ast,
              "dynamic tensor types require a bound extent identity");
        }
        return tensor.bind_type({});
      }
      if (admitted && admitted->kind == ValueKind::OpaqueDialect) {
        return static_cast<const OpaqueDialectPartition*>(admitted)->type;
      }
      diagnostics_.reject(unit_, ast,
                          "unsupported C++ type: " + cxx::to_string(input));
    }
    default:
      diagnostics_.reject(unit_, ast,
                          "unsupported C++ type: " + cxx::to_string(input));
  }
}

void Types::require_record_storage(const cxx::ClassType* input,
                                   cxx::AST* owner) {
  auto* source = input->definition();
  if (storage_records_.contains(source)) {
    return;
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (type_binding(source)) {
    diagnostics_.reject(unit_, owner,
                        "Loom source types require SSA value transport");
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      source->isPolymorphic()) {
    diagnostics_.reject(
        unit_, owner,
        "record storage requires a complete non-union type without "
        "virtual members");
  }
  for (auto* base : source->baseClasses()) {
    if (base->isVirtual()) {
      diagnostics_.reject(
          unit_, owner, "virtual base objects require a dynamic object layout");
    }
    storage_size(source_object_type(base), owner);
  }
  for (auto* symbol : source->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
    if (!field || field->isStatic()) {
      continue;
    }
    if (!field->name() || field->isBitField() || field->isNoUniqueAddress()) {
      diagnostics_.reject(
          unit_, owner,
          "record storage requires named fields without bitfields or "
          "no_unique_address");
    }
    storage_size(field->type(), owner);
  }
  storage_records_.insert(source);
}

int64_t Types::storage_size(const cxx::Type* input, cxx::AST* owner) {
  auto traits = unit_.typeTraits();
  if (traits.is_pointer(input) || traits.is_reference(input)) {
    diagnostics_.reject(
        unit_, owner,
        "stored pointers and references require an object storage "
        "representation");
  }
  if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(unqualified(input))) {
    storage_size(array->elementType(), owner);
  } else if (auto* record =
                 cxx::type_cast<cxx::ClassType>(unqualified(input))) {
    require_record_storage(record, owner);
  } else {
    auto type = get(input, owner);
    if (loom_type_kind(type) != LOOM_TYPE_SCALAR &&
        loom_type_kind(type) != LOOM_TYPE_VECTOR) {
      diagnostics_.reject(
          unit_, owner,
          "memory objects require scalar, vector or plain record storage");
    }
  }
  auto bytes = unit_.control()->memoryLayout()->sizeOf(input);
  if (!bytes || *bytes > INT64_MAX) {
    diagnostics_.reject(unit_, owner, "unknown or unrepresentable object size");
  }
  return static_cast<int64_t>(*bytes);
}

bool Types::is_unsigned(const cxx::Type* type) {
  auto traits = unit_.typeTraits();
  return traits.is_unsigned(traits.underlying_type(type));
}

bool Types::is_opaque_dialect(const cxx::Type* type, std::string_view name,
                              cxx::AST* owner) {
  auto projected = get(type, owner);
  if (!loom_type_is_dialect(projected) ||
      loom_type_dialect_param_count(projected) != 0 || !module_) {
    return false;
  }
  auto name_id = loom_type_dialect_name_id(projected);
  return name_id != LOOM_STRING_ID_INVALID &&
         name_id < module_->strings.count &&
         iree_string_view_equal(
             loom_string_table_get(&module_->strings, name_id),
             iree_make_string_view(name.data(), name.size()));
}

// The frontend has selected the special member and checked accessibility,
// deletion, cv and overload resolution. Ordinary records execute selected
// nontrivial bodies; intrinsic source wrappers admit only trivial transport.
void Types::admit_construction(cxx::FunctionSymbol* constructor,
                               const cxx::Type* type, cxx::AST* owner) {
  if (!constructor || unit_.typeTraits().is_array(type)) {
    return;
  }
  const auto& admitted = partition(unqualified(type), owner);
  if (admitted.kind == ValueKind::Record ||
      is_trivial_value_operation(constructor)) {
    return;
  }
  diagnostics_.reject(
      unit_, owner,
      "Loom source type construction requires an operation result or "
      "a trivial copy");
}

void Types::require_mutable(const cxx::Type* input, cxx::AST* owner) {
  if (unit_.typeTraits().is_const(input)) {
    diagnostics_.reject(unit_, owner,
                        "mutation requires a non-const destination");
  }
}

void Types::append(const cxx::Type* input, cxx::AST* owner,
                   std::vector<loom_type_t>& output) {
  if (requires_binding(input, owner)) {
    diagnostics_.reject(
        unit_, owner,
        "dependent source values require reserved destination identities");
  }
  // Only dependent shaped types use temporary overflow payloads. The
  // rejection above therefore guarantees that no returned type borrows this
  // local storage; static shaped partitions retain their own payload.
  BoundTypeStorage storage;
  append_bound(input, owner, {}, storage, output);
}

void Types::append_bound(const cxx::Type* input, cxx::AST* owner,
                         std::span<const loom_value_id_t> identities,
                         BoundTypeStorage& storage,
                         std::vector<loom_type_t>& output) {
  const auto& admitted = partition(input, owner);
  if (!identities.empty() && identities.size() != admitted.component_count) {
    diagnostics_.reject(unit_, owner,
                        "source value identity count does not match its "
                        "admitted component partition");
  }
  switch (admitted.kind) {
    case ValueKind::SSA:
      output.push_back(get(input, owner));
      return;
    case ValueKind::Buffer:
      output.push_back(loom_type_buffer());
      return;
    case ValueKind::Address:
      output.push_back(loom_type_scalar(
          static_cast<const AddressPartition&>(admitted).element_type));
      return;
    case ValueKind::OpaqueDialect:
      output.push_back(
          static_cast<const OpaqueDialectPartition&>(admitted).type);
      return;
    case ValueKind::Pointer:
      output.push_back(loom_type_buffer());
      output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET));
      return;
    case ValueKind::Encoding:
      output.push_back(loom_type_encoding_with_role(
          static_cast<const EncodingPartition&>(admitted).role));
      return;
    case ValueKind::Pool:
      output.push_back(loom_type_pool());
      return;
    case ValueKind::Tile: {
      const auto& tile = static_cast<const TilePartition&>(admitted);
      if (tile.requires_binding() &&
          identities.size() != tile.component_count) {
        diagnostics_.reject(
            unit_, owner,
            "tile values require reserved destination component identities");
      }
      const size_t dependency_count = tile.component_count - 1;
      size_t dependency = 0;
      for (auto extent : tile.extents) {
        if (extent < 0) {
          output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
          ++dependency;
        }
      }
      if (tile.encoding) {
        output.push_back(loom_type_encoding_with_role(tile.encoding->role));
        ++dependency;
      }
      IREE_ASSERT_EQ(dependency, dependency_count);
      output.push_back(tile.bind_type(identities.empty()
                                          ? std::span<const loom_value_id_t>{}
                                          : identities.first(dependency_count),
                                      storage));
      return;
    }
    case ValueKind::Channel: {
      const auto& channel = static_cast<const ChannelPartition&>(admitted);
      if (channel.requires_binding() &&
          identities.size() != channel.component_count) {
        diagnostics_.reject(
            unit_, owner,
            "channel values require reserved destination component "
            "identities");
      }
      const auto& payload = *channel.payload;
      const size_t dependency_count = channel.component_count - 1;
      size_t dependency = 0;
      for (auto extent : payload.extents) {
        if (extent < 0) {
          output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
          ++dependency;
        }
      }
      if (payload.encoding) {
        output.push_back(loom_type_encoding_with_role(payload.encoding->role));
        ++dependency;
      }
      IREE_ASSERT_EQ(dependency, dependency_count);
      output.push_back(channel.bind_type(
          module_,
          identities.empty() ? std::span<const loom_value_id_t>{}
                             : identities.first(dependency_count),
          storage));
      return;
    }
    case ValueKind::Vector: {
      const auto& vector = static_cast<const VectorPartition&>(admitted);
      if (vector.is_dynamic()) {
        if (identities.size() != vector.component_count) {
          diagnostics_.reject(
              unit_, owner,
              "vector values require reserved destination component "
              "identities");
        }
        for (size_t index = 0; index + 1 < identities.size(); ++index) {
          output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
        }
        output.push_back(
            vector.bind_type(identities.first(identities.size() - 1), storage));
      } else {
        output.push_back(vector.bind_type({}, storage));
      }
      return;
    }
    case ValueKind::Tensor: {
      const auto& tensor = static_cast<const TensorPartition&>(admitted);
      if (tensor.is_dynamic()) {
        IREE_ASSERT(identities.size() == tensor.component_count);
        output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
        output.push_back(tensor.bind_type(identities.first(1)));
      } else {
        output.push_back(tensor.bind_type({}));
      }
      return;
    }
    case ValueKind::Record: {
      const auto& record = static_cast<const RecordPartition&>(admitted);
      for (const auto& member : record.members) {
        auto member_identities =
            identities.empty()
                ? std::span<const loom_value_id_t>{}
                : identities.subspan(member.component_offset,
                                     member.partition->component_count);
        auto* member_type = member.type();
        if (unit_.typeTraits().is_lvalue_reference(member_type)) {
          member_type = reference_transport_type(
              static_cast<cxx::FieldSymbol*>(member.source));
        }
        append_bound(member_type, owner, member_identities, storage, output);
      }
      return;
    }
    case ValueKind::Array: {
      const auto& array = static_cast<const ArrayPartition&>(admitted);
      for (size_t index = 0; index < array.source->size(); ++index) {
        auto element_identities =
            identities.empty()
                ? std::span<const loom_value_id_t>{}
                : identities.subspan(index * array.element->component_count,
                                     array.element->component_count);
        append_bound(array.source->elementType(), owner, element_identities,
                     storage, output);
      }
      return;
    }
    case ValueKind::View: {
      const auto& view = static_cast<const ViewPartition&>(admitted);
      if (view.requires_binding() &&
          identities.size() != admitted.component_count) {
        diagnostics_.reject(
            unit_, owner,
            "view values require reserved destination component identities");
      }
      size_t component = 0;
      for (size_t axis = 0; axis < view.extents.size(); ++axis) {
        if (view.extents[axis] < 0) {
          output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
          ++component;
        }
      }
      if (view.encoding) {
        output.push_back(loom_type_encoding_with_role(view.encoding->role));
        ++component;
      }
      output.push_back(view.bind_type(identities.empty()
                                          ? std::span<const loom_value_id_t>{}
                                          : identities.first(component),
                                      storage));
      return;
    }
  }
}

bool Types::is_float(const cxx::Type* input) {
  auto kind = unqualified(input)->kind();
  return kind == cxx::TypeKind::kFloat || kind == cxx::TypeKind::kFloat16 ||
         kind == cxx::TypeKind::kBFloat16 || kind == cxx::TypeKind::kDouble ||
         kind == cxx::TypeKind::kFloat8E4M3FN ||
         kind == cxx::TypeKind::kFloat8E5M2;
}

bool Types::is_index(const cxx::Type* type, cxx::AST* owner) {
  return address_domain(type, owner) == LOOM_SCALAR_TYPE_INDEX;
}

bool Types::is_offset(const cxx::Type* type, cxx::AST* owner) {
  return address_domain(type, owner) == LOOM_SCALAR_TYPE_OFFSET;
}

loom_scalar_type_t Types::address_domain(const cxx::Type* type,
                                         cxx::AST* owner) {
  const auto& admitted = partition(type, owner);
  return admitted.kind == ValueKind::Address
             ? static_cast<const AddressPartition&>(admitted).element_type
             : LOOM_SCALAR_TYPE_NONE;
}

}  // namespace loom::cxx_import
