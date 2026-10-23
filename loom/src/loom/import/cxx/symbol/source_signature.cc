// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/source_signature.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"

namespace loom::cxx_import {
namespace {

class CSourceSignatureBuilder {
 public:
  CSourceSignatureBuilder(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          loom_module_t* module, cxx::AST* source, Types& types)
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        source_(source),
        types_(types) {}

  loom_attribute_t Build(cxx::FunctionSymbol* function) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    if (!signature || signature->isVariadic()) {
      diagnostics_.reject(unit_, source_,
                          "C ABI functions require a nonvariadic signature");
    }
    for (auto* parameter : function->parameters()) {
      argument_roots_.push_back(AddType(parameter->type()));
    }
    if (!unit_.typeTraits().is_void(signature->returnType())) {
      result_roots_.push_back(AddType(signature->returnType()));
    }
    loom_attribute_t result;
    check(loom_func_abi_c_signature_attr_make(
        module_,
        loom_make_i64_array(argument_roots_.data(), argument_roots_.size()),
        loom_make_i64_array(result_roots_.data(), result_roots_.size()),
        loom_make_parameterized_attr_array(type_nodes_.data(),
                                           type_nodes_.size()),
        &result));
    return result;
  }

 private:
  struct ObjectLayout {
    int64_t length;
    int64_t alignment;
  };

  ObjectLayout Layout(const cxx::Type* type) {
    auto* layout = unit_.control()->memoryLayout();
    auto length = layout->sizeOf(type);
    auto alignment = layout->alignmentOf(type);
    if (!length || !alignment || *length > INT64_MAX ||
        *alignment > INT64_MAX) {
      diagnostics_.reject(unit_, source_,
                          "C ABI type has an unknown or unrepresentable "
                          "object layout");
    }
    return {static_cast<int64_t>(*length), static_cast<int64_t>(*alignment)};
  }

  int64_t AddType(const cxx::Type* input) {
    auto traits = unit_.typeTraits();
    if (traits.is_reference(input)) {
      diagnostics_.reject(unit_, source_,
                          "C ABI signatures cannot contain C++ references");
    }
    const cxx::Type* type = types_.unqualified(input);
    if (auto found = type_indices_.find(type); found != type_indices_.end()) {
      return found->second;
    }

    // Composite nodes are appended after their children. Cache only completed
    // nodes: admitted records cannot be recursively value-containing, and
    // pointers intentionally do not traverse their pointee type.
    loom_attribute_t node;
    const ObjectLayout layout = Layout(type);
    const size_t projected_component_count =
        types_.partition(type, source_).component_count;
    if (projected_component_count > INT64_MAX) {
      diagnostics_.reject(unit_, source_,
                          "C ABI type projection is unrepresentably large");
    }
    const int64_t component_count =
        static_cast<int64_t>(projected_component_count);
    const auto absent_string = LOOM_STRING_ID_INVALID;
    const auto no_fields = loom_parameterized_attr_array_empty();
    switch (type->kind()) {
      case cxx::TypeKind::kBool:
        check(loom_func_abi_c_type_attr_make(
            module_, 0, LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN, layout.length,
            layout.alignment, component_count, 0,
            (loom_func_abi_c_type_format_t)0, 0, 0, 0, 0, absent_string,
            no_fields, &node));
        break;
      case cxx::TypeKind::kChar:
      case cxx::TypeKind::kSignedChar:
      case cxx::TypeKind::kUnsignedChar:
      case cxx::TypeKind::kShortInt:
      case cxx::TypeKind::kUnsignedShortInt:
      case cxx::TypeKind::kInt:
      case cxx::TypeKind::kUnsignedInt:
      case cxx::TypeKind::kLongInt:
      case cxx::TypeKind::kUnsignedLongInt:
      case cxx::TypeKind::kLongLongInt:
      case cxx::TypeKind::kUnsignedLongLongInt:
      case cxx::TypeKind::kEnum:
      case cxx::TypeKind::kScopedEnum:
        check(loom_func_abi_c_type_attr_make(
            module_, LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_BITWIDTH,
            types_.is_unsigned(type)
                ? LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER
                : LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER,
            layout.length, layout.alignment, component_count, layout.length * 8,
            (loom_func_abi_c_type_format_t)0, 0, 0, 0, 0, absent_string,
            no_fields, &node));
        break;
      case cxx::TypeKind::kFloat:
      case cxx::TypeKind::kDouble:
      case cxx::TypeKind::kFloat16:
      case cxx::TypeKind::kBFloat16:
      case cxx::TypeKind::kFloat8E4M3FN:
      case cxx::TypeKind::kFloat8E5M2:
        check(loom_func_abi_c_type_attr_make(
            module_, LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FORMAT,
            LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING, layout.length, layout.alignment,
            component_count, 0, FloatFormat(type), 0, 0, 0, 0, absent_string,
            no_fields, &node));
        break;
      case cxx::TypeKind::kPointer: {
        auto* pointer = cxx::type_cast<cxx::PointerType>(type);
        if (traits.is_function(pointer->elementType())) {
          diagnostics_.reject(unit_, source_,
                              "C ABI signatures cannot contain function "
                              "pointers");
        }
        check(loom_func_abi_c_type_attr_make(
            module_, 0, LOOM_FUNC_ABI_C_TYPE_KIND_POINTER, layout.length,
            layout.alignment, component_count, 0,
            (loom_func_abi_c_type_format_t)0, 0, 0, 0, 0, absent_string,
            no_fields, &node));
        break;
      }
      case cxx::TypeKind::kVector: {
        auto* vector = types_.native_vector(type);
        const int64_t element = AddType(vector->elementType());
        check(loom_func_abi_c_type_attr_make(
            module_,
            LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT,
            LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR, layout.length, layout.alignment,
            component_count, 0, (loom_func_abi_c_type_format_t)0, 0, element,
            static_cast<int64_t>(vector->elementCount()), 0, absent_string,
            no_fields, &node));
        break;
      }
      case cxx::TypeKind::kBoundedArray: {
        auto* array = cxx::type_cast<cxx::BoundedArrayType>(type);
        const int64_t element = AddType(array->elementType());
        const ObjectLayout element_layout = Layout(array->elementType());
        check(loom_func_abi_c_type_attr_make(
            module_,
            LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT |
                LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_STRIDE,
            LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY, layout.length, layout.alignment,
            component_count, 0, (loom_func_abi_c_type_format_t)0, 0, element,
            static_cast<int64_t>(array->size()), element_layout.length,
            absent_string, no_fields, &node));
        break;
      }
      case cxx::TypeKind::kClass:
        node = AddRecord(type, layout, component_count);
        break;
      default:
        diagnostics_.reject(unit_, source_,
                            "unsupported C ABI type: " + cxx::to_string(type));
    }
    if (type_nodes_.size() > INT64_MAX) {
      diagnostics_.reject(unit_, source_, "C ABI type table is too large");
    }
    const int64_t index = static_cast<int64_t>(type_nodes_.size());
    type_nodes_.push_back(node);
    type_indices_.emplace(type, index);
    return index;
  }

  loom_attribute_t AddRecord(const cxx::Type* type, ObjectLayout object_layout,
                             int64_t component_count) {
    const RecordPartition* record = types_.record(type, source_);
    if (!record || !record->source || !record->source->name()) {
      diagnostics_.reject(unit_, source_,
                          "C ABI records require a named plain aggregate");
    }
    std::vector<loom_attribute_t> fields;
    fields.reserve(record->members.size());
    for (const auto& member : record->members) {
      auto* field = member.field;
      if (!field->name() || field->isNoUniqueAddress() ||
          unit_.typeTraits().is_reference(field->type())) {
        diagnostics_.reject(
            unit_, source_,
            "C ABI record fields require distinct named object storage");
      }
      const int64_t type_index = AddType(field->type());
      const auto byte_offset = field->offsetInClass();
      if (!byte_offset || *byte_offset > INT64_MAX) {
        diagnostics_.reject(unit_, source_,
                            "C ABI record field has an unknown or "
                            "unrepresentable offset");
      }
      loom_string_id_t field_name;
      const std::string field_spelling = cxx::to_string(field->name());
      check(loom_module_intern_string(module_, view(field_spelling),
                                      &field_name));
      loom_attribute_t field_attr;
      check(loom_func_abi_c_field_attr_make(module_, field_name, type_index,
                                            static_cast<int64_t>(*byte_offset),
                                            &field_attr));
      fields.push_back(field_attr);
    }
    loom_string_id_t record_name;
    const std::string record_spelling = cxx::to_string(record->source->name());
    check(loom_module_intern_string(module_, view(record_spelling),
                                    &record_name));
    loom_attribute_t node;
    check(loom_func_abi_c_type_attr_make(
        module_,
        LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_NAME |
            LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FIELDS,
        LOOM_FUNC_ABI_C_TYPE_KIND_RECORD, object_layout.length,
        object_layout.alignment, component_count, 0,
        (loom_func_abi_c_type_format_t)0, 0, 0, 0, 0, record_name,
        loom_make_parameterized_attr_array(fields.data(), fields.size()),
        &node));
    return node;
  }

  loom_func_abi_c_type_format_t FloatFormat(const cxx::Type* type) {
    switch (type->kind()) {
      case cxx::TypeKind::kFloat8E4M3FN:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN;
      case cxx::TypeKind::kFloat8E5M2:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E5M2;
      case cxx::TypeKind::kFloat16:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_F16;
      case cxx::TypeKind::kBFloat16:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_BF16;
      case cxx::TypeKind::kFloat:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_F32;
      case cxx::TypeKind::kDouble:
        return LOOM_FUNC_ABI_C_TYPE_FORMAT_F64;
      default:
        IREE_CHECK_UNREACHABLE();
    }
  }

  cxx::TranslationUnit& unit_;
  Diagnostics& diagnostics_;
  loom_module_t* module_;
  cxx::AST* source_;
  Types& types_;
  std::unordered_map<const cxx::Type*, int64_t> type_indices_;
  std::vector<loom_attribute_t> type_nodes_;
  std::vector<int64_t> argument_roots_;
  std::vector<int64_t> result_roots_;
};

}  // namespace

loom_attribute_t build_c_source_signature(cxx::TranslationUnit& unit,
                                          Diagnostics& diagnostics,
                                          loom_module_t* module,
                                          cxx::FunctionSymbol* function,
                                          cxx::AST* source, Types& types) {
  if (!function->hasCLinkage()) {
    return loom_attr_absent();
  }
  return CSourceSignatureBuilder(unit, diagnostics, module, source, types)
      .Build(function);
}

}  // namespace loom::cxx_import
