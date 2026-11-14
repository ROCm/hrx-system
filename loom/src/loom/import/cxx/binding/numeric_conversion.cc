// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/numeric_conversion.h"

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "iree/base/api.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/vector.h"

namespace loom::cxx_import {
namespace {

bool is_numeric(cxx::TranslationUnit& unit, Types& types,
                const cxx::Type* type) {
  type = types.unqualified(type);
  return types.is_float(type) || (unit.typeTraits().is_integral(type) &&
                                  type->kind() != cxx::TypeKind::kBool);
}

bool same_shape(const VectorPartition& lhs, const VectorPartition& rhs) {
  if (lhs.extents.size() != rhs.extents.size()) {
    return false;
  }
  for (size_t axis = 0; axis < lhs.extents.size(); ++axis) {
    if (lhs.extents[axis] != rhs.extents[axis]) {
      return false;
    }
  }
  return true;
}

uint8_t subnormal_flags(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                        cxx::FunctionSymbol* function, cxx::AST* owner) {
  auto arguments =
      cxx::expand_template_arguments(function->templateArguments());
  if (arguments.size() < 2) {
    diagnostics.reject(unit, owner,
                       "numeric conversion lacks a subnormal policy");
  }
  auto value = cxx::template_argument_value(arguments[1]);
  auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!number || number->isNegative() || number->toUWide() > 3) {
    diagnostics.reject(unit, owner,
                       "numeric conversion has an invalid subnormal policy");
  }
  return static_cast<uint8_t>(number->toUIntMax());
}

bool permits_subnormal_replacement(cxx::FunctionSymbol* function) {
  auto arguments =
      cxx::expand_template_arguments(function->templateArguments());
  if (arguments.size() < 2) {
    return false;
  }
  auto value = cxx::template_argument_value(arguments[1]);
  auto* number = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  return number && !number->isNegative() && number->toUWide() != 0;
}

}  // namespace

std::optional<cxx::ConstValue> NumericConversionConstantEvaluator::evaluate(
    cxx::ASTInterpreter& interpreter, cxx::FunctionSymbol* function,
    std::span<const cxx::ConstValue> arguments) {
  const auto* operation = annotation(function, "op");
  if (!operation || operation->arguments.size() != 1 ||
      operation->arguments[0]->name() != "numeric.convert" ||
      arguments.size() != 1 || permits_subnormal_replacement(function)) {
    return std::nullopt;
  }

  auto* unit = interpreter.translationUnit();
  auto traits = unit->typeTraits();
  const auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (!signature || signature->parameterTypes().size() != 1) {
    return std::nullopt;
  }
  const auto* input_type =
      traits.remove_cv(signature->parameterTypes().front());
  const auto* output_type = traits.remove_cv(signature->returnType());
  const auto* input_vector = cxx::type_cast<cxx::VectorType>(input_type);
  const auto* output_vector = cxx::type_cast<cxx::VectorType>(output_type);
  if (!input_vector && !output_vector) {
    return interpreter.convertArithmetic(arguments.front(), input_type,
                                         output_type);
  }
  if (!input_vector || !output_vector ||
      input_vector->elementCount() != output_vector->elementCount()) {
    return std::nullopt;
  }

  const auto* input =
      std::get_if<std::shared_ptr<cxx::InitializerList>>(&arguments.front());
  if (!input || !*input || (*input)->size() != input_vector->elementCount()) {
    return std::nullopt;
  }
  auto output = std::make_shared<cxx::InitializerList>();
  for (size_t index = 0; index < (*input)->size(); ++index) {
    const auto& [element, element_type] = (*input)->element(index);
    auto converted = interpreter.convertArithmetic(
        element, element_type, output_vector->elementType());
    if (!converted) {
      return std::nullopt;
    }
    output->append(std::move(*converted), output_vector->elementType());
  }
  return cxx::ConstValue{std::move(output)};
}

std::optional<NumericConversionIntrinsic::Operation>
NumericConversionIntrinsic::admit(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics,
                                  const cxx::Attribute& attribute,
                                  cxx::AST* owner) {
  if (attribute.arguments.empty() ||
      attribute.arguments[0]->name() != "numeric.convert") {
    return std::nullopt;
  }
  if (attribute.arguments.size() != 1) {
    diagnostics.reject(unit, owner,
                       "numeric conversion has no semantic arguments");
  }
  return Operation{};
}

NumericConversionIntrinsic NumericConversionIntrinsic::resolve(
    Operation operation, cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    Types& types, cxx::FunctionSymbol* function, cxx::AST* owner) {
  (void)operation;
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  if (signature->isVariadic() || signature->parameterTypes().size() != 1) {
    diagnostics.reject(unit, owner,
                       "numeric conversion requires exactly one operand");
  }
  const auto* input = types.unqualified(signature->parameterTypes()[0]);
  const auto* output = types.unqualified(signature->returnType());
  const uint8_t subnormal = subnormal_flags(unit, diagnostics, function, owner);
  const auto& input_partition = types.partition(input, owner);
  const auto& output_partition = types.partition(output, owner);

  if (input_partition.kind == ValueKind::Vector ||
      output_partition.kind == ValueKind::Vector) {
    if (input_partition.kind != ValueKind::Vector ||
        output_partition.kind != ValueKind::Vector) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires two scalars or two vectors");
    }
    const auto& input_vector =
        static_cast<const VectorPartition&>(input_partition);
    const auto& output_vector =
        static_cast<const VectorPartition&>(output_partition);
    if (!same_shape(input_vector, output_vector)) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires vectors with the same logical shape");
    }
    if (!is_numeric(unit, types, input_vector.element_type) ||
        !is_numeric(unit, types, output_vector.element_type)) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires integer or floating-point elements");
    }
    if (subnormal != 0 &&
        (!types.is_float(input_vector.element_type) ||
         !types.is_float(output_vector.element_type) || input == output)) {
      diagnostics.reject(
          unit, owner,
          "subnormal permissions require a floating-point format conversion");
    }
    return NumericConversionIntrinsic(Representation::SemanticVector, input,
                                      output, &input_vector, &output_vector,
                                      subnormal);
  }

  const auto* input_vector = types.native_vector(input);
  const auto* output_vector = types.native_vector(output);
  if (input_vector || output_vector) {
    if (!input_vector || !output_vector ||
        input_vector->elementCount() != output_vector->elementCount()) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires vectors with the same lane count");
    }
    if (!is_numeric(unit, types, input_vector->elementType()) ||
        !is_numeric(unit, types, output_vector->elementType())) {
      diagnostics.reject(
          unit, owner,
          "numeric conversion requires integer or floating-point elements");
    }
    if (subnormal != 0 &&
        (!types.is_float(input_vector->elementType()) ||
         !types.is_float(output_vector->elementType()) || input == output)) {
      diagnostics.reject(
          unit, owner,
          "subnormal permissions require a floating-point format conversion");
    }
    return NumericConversionIntrinsic(Representation::NativeVector, input,
                                      output, nullptr, nullptr, subnormal);
  }

  if (input_partition.kind != ValueKind::SSA ||
      output_partition.kind != ValueKind::SSA ||
      !is_numeric(unit, types, input) || !is_numeric(unit, types, output)) {
    diagnostics.reject(
        unit, owner,
        "numeric conversion requires integer or floating-point scalars");
  }
  if (subnormal != 0 &&
      (!types.is_float(input) || !types.is_float(output) || input == output)) {
    diagnostics.reject(
        unit, owner,
        "subnormal permissions require a floating-point format conversion");
  }
  return NumericConversionIntrinsic(Representation::Scalar, input, output,
                                    nullptr, nullptr, subnormal);
}

Value NumericConversionIntrinsic::call(std::span<const Value> arguments,
                                       ValueArena& arena, Scalars& scalars,
                                       Vectors& vectors,
                                       cxx::AST* owner) const {
  IREE_ASSERT(arguments.size() == 1);
  switch (representation_) {
    case Representation::Scalar:
      return Value(scalars.convert(arguments[0].ssa(), input_type_,
                                   output_type_, owner, 0, subnormal_flags_));
    case Representation::NativeVector:
      return Value(vectors.convert_elements(arguments[0].ssa(), input_type_,
                                            output_type_, owner,
                                            subnormal_flags_));
    case Representation::SemanticVector:
      return vectors.convert_elements(arguments[0], *input_vector_,
                                      *output_vector_, arena, owner,
                                      subnormal_flags_);
  }
  IREE_ASSERT_UNREACHABLE("unknown numeric conversion representation");
  IREE_BUILTIN_UNREACHABLE();
}

}  // namespace loom::cxx_import
