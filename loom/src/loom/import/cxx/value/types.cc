// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/types.h"

#include <cxx/ast_interpreter.h>
#include <cxx/attributes.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <limits>
#include <optional>

namespace loom::cxx_import {
namespace {

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

}  // namespace

const cxx::Type* Types::unqualified(const cxx::Type* type) {
  return unit_.typeTraits().remove_cv(type);
}

const cxx::VectorType* Types::vector(const cxx::Type* type) {
  return cxx::type_cast<cxx::VectorType>(unqualified(type));
}

loom_memory_access_flags_t Types::memory_access_flags(const cxx::Type* input) {
  return unit_.typeTraits().is_volatile(input)
             ? LOOM_MEMORY_ACCESS_FLAG_VOLATILE
             : 0;
}

const Partition& Types::partition(const cxx::Type* input, cxx::AST* owner) {
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
  if (admitted.kind == ValueKind::View) {
    return true;
  }
  if (admitted.kind != ValueKind::Record) {
    return false;
  }
  const auto& record = static_cast<const RecordPartition&>(admitted);
  for (const auto& member : record.members) {
    if (requires_binding(member.field->type(), owner)) {
      return true;
    }
  }
  return false;
}

const Partition* Types::special(const cxx::Type* input, cxx::AST* owner) {
  if (!input) {
    return nullptr;
  }
  if (unit_.typeTraits().is_volatile(input)) {
    diagnostics_.reject(unit_, owner,
                        "volatile objects require addressable storage");
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
  if (binding->arguments.size() != 1) {
    diagnostics_.reject(unit_, owner,
                        "loom source types require one string type binding");
  }
  auto name = binding->arguments[0]->name();
  if (name == "encoding") {
    return encoding(type, owner);
  }
  if (name == "view") {
    return view(type, owner);
  }
  if (name == "tensor") {
    return tensor(type, owner);
  }
  diagnostics_.reject(unit_, owner, "unknown Loom source type binding");
}

const EncodingPartition* Types::encoding(const cxx::ClassType* input,
                                         cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = encodings_.find(source); found != encodings_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.requireCompleteClass(source);
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
  auto arguments = cxx::class_template_arguments(source);
  if (arguments.size() != 2) {
    diagnostics_.reject(unit_, owner,
                        "encoding values require role and rank arguments");
  }
  cxx::ASTInterpreter interpreter(&unit_);
  auto role = cxx::template_argument_value(arguments[0]);
  auto rank = cxx::template_argument_value(arguments[1]);
  auto role_value = role ? interpreter.toInt(*role) : std::nullopt;
  auto rank_value = rank ? interpreter.toInt(*rank) : std::nullopt;
  if (!role_value || *role_value != 0 || !rank_value || *rank_value < 1 ||
      *rank_value > LOOM_TYPE_MAX_RANK) {
    diagnostics_.reject(
        unit_, owner,
        "encoding values require the layout role and rank in [1, 15]");
  }
  auto result = std::make_unique<EncodingPartition>();
  result->kind = ValueKind::Encoding;
  result->component_count = 1;
  result->source = source;
  result->rank = static_cast<size_t>(*rank_value);
  auto* admitted = result.get();
  encodings_.emplace(source, std::move(result));
  return admitted;
}

const ViewPartition* Types::view(const cxx::ClassType* input, cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = views_.find(source); found != views_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.requireCompleteClass(source);
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
  auto arguments = cxx::class_template_arguments(source);
  if (arguments.size() != 3) {
    diagnostics_.reject(
        unit_, owner,
        "rank-two view values require an element and two extent arguments");
  }
  auto* element_type = cxx::template_argument_type(arguments[0]);
  auto element = get(element_type, owner);
  if (loom_type_kind(element) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(element) == LOOM_SCALAR_TYPE_I1) {
    diagnostics_.reject(
        unit_, owner,
        "view elements require a supported non-boolean scalar type");
  }
  cxx::ASTInterpreter interpreter(&unit_);
  auto pointer_bytes = unit_.control()->memoryLayout()->sizeOfPointer();
  uint64_t dynamic = pointer_bytes == 4 ? UINT32_MAX : UINT64_MAX;
  std::array<int64_t, 2> extents;
  size_t dynamic_count = 0;
  for (size_t axis = 0; axis < extents.size(); ++axis) {
    auto value = cxx::template_argument_value(arguments[axis + 1]);
    auto number = value ? interpreter.toInt(*value) : std::nullopt;
    if (!number) {
      diagnostics_.reject(unit_, owner,
                          "view extents require constant size arguments");
    }
    uint64_t bits = static_cast<uint64_t>(*number);
    if (pointer_bytes == 4) {
      bits &= UINT32_MAX;
    }
    if (bits == dynamic) {
      extents[axis] = -1;
      ++dynamic_count;
    } else if (*number < 0 || bits > LOOM_DIM_MAX_STATIC_SIZE) {
      diagnostics_.reject(unit_, owner,
                          "view extent is outside Loom's static dimension "
                          "range");
    } else {
      extents[axis] = static_cast<int64_t>(bits);
    }
  }
  auto result = std::make_unique<ViewPartition>();
  result->kind = ValueKind::View;
  result->component_count = dynamic_count + 2;
  result->source = source;
  result->element_type = element_type;
  result->element = loom_type_element_type(element);
  result->access_flags = memory_access_flags(element_type);
  result->extents = extents;
  if (extents[0] < 0) {
    result->component_names.push_back("rows");
  }
  if (extents[1] < 0) {
    result->component_names.push_back("columns");
  }
  result->component_names.push_back("layout");
  result->component_names.emplace_back();
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
    traits.requireCompleteClass(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "tensor values require a complete trivial class "
                        "without bases");
  }
  auto arguments = cxx::class_template_arguments(source);
  if (arguments.size() != 2) {
    diagnostics_.reject(unit_, owner,
                        "rank-one tensor values require an element and extent "
                        "argument");
  }
  auto* element = cxx::template_argument_type(arguments[0]);
  auto element_type = get(element, owner);
  auto bytes = unit_.control()->memoryLayout()->sizeOf(element);
  if (loom_type_kind(element_type) != LOOM_TYPE_SCALAR ||
      loom_type_element_type(element_type) == LOOM_SCALAR_TYPE_I1 ||
      traits.is_const(element) || traits.is_volatile(element) || !bytes ||
      !*bytes) {
    diagnostics_.reject(unit_, owner,
                        "tensor elements require unqualified non-boolean "
                        "scalar types");
  }
  auto extent = cxx::template_argument_value(arguments[1]);
  cxx::ASTInterpreter interpreter(&unit_);
  auto count = extent ? interpreter.toInt(*extent) : std::nullopt;
  if (!count || *count < 0 ||
      static_cast<uint64_t>(*count) > INT64_MAX / *bytes) {
    diagnostics_.reject(unit_, owner,
                        "tensor extent requires a nonnegative constant with "
                        "a representable byte length");
  }
  auto result = std::make_unique<TensorPartition>();
  result->kind = ValueKind::Tensor;
  result->component_count = 1;
  result->source = source;
  result->element_type = element;
  result->type = loom_type_shaped_1d(LOOM_TYPE_TENSOR,
                                     loom_type_element_type(element_type),
                                     loom_dim_pack_static(*count), 0);
  result->element_bytes = *bytes;
  auto* admitted = result.get();
  tensors_.emplace(source, std::move(result));
  return admitted;
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
  if (traits.is_volatile(input)) {
    diagnostics_.reject(unit_, owner,
                        "volatile objects require addressable storage");
  }
  auto* source = type->definition();
  if (auto found = records_.find(source); found != records_.end()) {
    return found->second.get();
  }
  if (source) {
    traits.requireCompleteClass(source);
    source = type->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_aggregate(input) ||
      !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(
        unit_, owner,
        "record values require a complete aggregate without unions, bases "
        "or nontrivial lifecycle operations");
  }
  auto result = std::make_unique<RecordPartition>();
  result->kind = ValueKind::Record;
  result->component_count = 0;
  result->source = source;
  for (auto* symbol : source->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
    if (!field || field->isStatic()) {
      continue;
    }
    if (field->isBitField() || traits.is_reference(field->type()) ||
        traits.is_array(field->type())) {
      diagnostics_.reject(
          unit_, owner,
          "record value fields cannot be bitfields, references or arrays");
    }
    auto& member_partition = partition(field->type(), owner);
    MemberPartition member{field, &member_partition, result->component_count};
    result->members.push_back(member);
    members_.emplace(field, member);
    auto name = cxx::to_string(field->name());
    append_component_names(member_partition, name, result->component_names);
    result->component_count += member_partition.component_count;
  }
  auto* admitted = result.get();
  records_.emplace(source, std::move(result));
  return admitted;
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
    case ValueKind::Encoding:
    case ValueKind::Tensor:
      append({});
      return;
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
    case ValueKind::View: {
      const auto& view = static_cast<const ViewPartition&>(partition);
      for (const auto& name : view.component_names) {
        append(name);
      }
      return;
    }
  }
}

const MemberPartition& Types::member(cxx::FieldSymbol* field, cxx::AST* owner) {
  if (auto found = members_.find(field); found != members_.end()) {
    return found->second;
  }
  record(field->parent()->type(), owner);
  return members_.at(field);
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
      return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    case cxx::TypeKind::kChar:
    case cxx::TypeKind::kSignedChar:
    case cxx::TypeKind::kUnsignedChar:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I8);
    case cxx::TypeKind::kShortInt:
    case cxx::TypeKind::kUnsignedShortInt:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I16);
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
      auto* source = vector(input);
      auto element = get(source->elementType(), ast);
      auto* layout = unit_.control()->memoryLayout();
      auto bytes = layout->sizeOf(source);
      auto element_bytes = layout->sizeOf(source->elementType());
      if (loom_type_kind(element) != LOOM_TYPE_SCALAR ||
          loom_type_element_type(element) == LOOM_SCALAR_TYPE_I1 ||
          !source->elementCount() ||
          source->elementCount() > LOOM_DIM_MAX_STATIC_SIZE || !bytes ||
          !element_bytes || *bytes != source->elementCount() * *element_bytes) {
        diagnostics_.reject(
            unit_, ast,
            "vectors require non-boolean scalar lanes without object padding");
      }
      return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                                 loom_type_element_type(element),
                                 source->elementCount(), 0);
    }
    case cxx::TypeKind::kBoundedArray: {
      auto* array = cxx::type_cast<cxx::BoundedArrayType>(unqualified(input));
      auto element = get(array->elementType(), ast);
      if (loom_type_kind(element) != LOOM_TYPE_SCALAR ||
          loom_type_element_type(element) == LOOM_SCALAR_TYPE_I1) {
        diagnostics_.reject(
            unit_, ast,
            "arrays require a supported non-boolean scalar element");
      }
      return loom_type_buffer();
    }
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
      if (admitted && admitted->kind == ValueKind::Tensor) {
        return static_cast<const TensorPartition*>(admitted)->type;
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
    traits.requireCompleteClass(source);
    source = input->definition();
  }
  if (type_binding(source)) {
    diagnostics_.reject(unit_, owner,
                        "Loom source types require SSA value transport");
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_aggregate(input) ||
      !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(
        unit_, owner,
        "record storage requires a complete aggregate without unions, bases "
        "or nontrivial lifecycle operations");
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
    if ((loom_type_kind(type) != LOOM_TYPE_SCALAR &&
         loom_type_kind(type) != LOOM_TYPE_VECTOR) ||
        loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1) {
      diagnostics_.reject(
          unit_, owner,
          "memory objects require non-boolean scalar, vector or plain record "
          "storage");
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

// The frontend has selected the special member and checked accessibility,
// deletion, cv and overload resolution. Source admission establishes trivial
// lifecycle semantics before an implicit copy becomes an SSA value copy.
void Types::admit_copy(cxx::FunctionSymbol* constructor, const cxx::Type* type,
                       cxx::AST* owner) {
  if (!constructor) {
    return;
  }
  const auto& admitted = partition(type, owner);
  cxx::ClassSymbol* source = nullptr;
  bool is_record = admitted.kind == ValueKind::Record;
  if (admitted.kind == ValueKind::Record) {
    source = static_cast<const RecordPartition&>(admitted).source;
  } else if (admitted.kind == ValueKind::Encoding) {
    source = static_cast<const EncodingPartition&>(admitted).source;
  } else if (admitted.kind == ValueKind::View) {
    source = static_cast<const ViewPartition&>(admitted).source;
  } else if (admitted.kind == ValueKind::Tensor) {
    source = static_cast<const TensorPartition&>(admitted).source;
  }
  if (source && constructor == source->defaultConstructor()) {
    diagnostics_.reject(
        unit_, owner,
        is_record ? "default record construction requires source object "
                    "initialization semantics"
                  : "default encoding, view or tensor construction requires "
                    "source object initialization semantics");
  }
  if (!source || (constructor != source->copyConstructor() &&
                  constructor != source->moveConstructor())) {
    diagnostics_.reject(
        unit_, owner,
        is_record
            ? "record construction requires aggregate initialization "
              "or a trivial copy"
            : "encoding, view and tensor construction requires an operation "
              "result or a trivial copy");
  }
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
  append_bound(input, owner, {}, output);
}

void Types::append_bound(const cxx::Type* input, cxx::AST* owner,
                         std::span<const loom_value_id_t> identities,
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
    case ValueKind::Pointer:
      output.push_back(loom_type_buffer());
      output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET));
      return;
    case ValueKind::Encoding:
      output.push_back(
          loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT));
      return;
    case ValueKind::Tensor:
      output.push_back(static_cast<const TensorPartition&>(admitted).type);
      return;
    case ValueKind::Record: {
      const auto& record = static_cast<const RecordPartition&>(admitted);
      for (const auto& member : record.members) {
        auto member_identities =
            identities.empty()
                ? std::span<const loom_value_id_t>{}
                : identities.subspan(member.component_offset,
                                     member.partition->component_count);
        append_bound(member.field->type(), owner, member_identities, output);
      }
      return;
    }
    case ValueKind::View: {
      if (identities.size() != admitted.component_count) {
        diagnostics_.reject(
            unit_, owner,
            "view values require reserved destination component identities");
      }
      const auto& view = static_cast<const ViewPartition&>(admitted);
      std::array<uint64_t, 2> dimensions;
      size_t component = 0;
      for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        if (view.extents[axis] < 0) {
          dimensions[axis] = loom_dim_pack_dynamic(identities[component++]);
          output.push_back(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
        } else {
          dimensions[axis] = loom_dim_pack_static(view.extents[axis]);
        }
      }
      auto layout = identities[component++];
      if (layout > UINT16_MAX) {
        diagnostics_.reject(
            unit_, owner,
            "view layout identity exceeds Loom's current type capacity");
      }
      output.push_back(
          loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT));
      auto type = loom_type_shaped_2d(LOOM_TYPE_VIEW, view.element,
                                      dimensions[0], dimensions[1], 0);
      type.encoding_id = static_cast<uint16_t>(layout);
      type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
      output.push_back(type);
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

}  // namespace loom::cxx_import
