// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/testing/source_generator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace loom::cxx_import::testing {
namespace {

enum class TypeCategory {
  kInteger,
  kFloatingPoint,
};

struct TypeDescriptor {
  // C++ source spelling for the type.
  const char* spelling;
  // Operations that may be emitted for values of the type.
  TypeCategory category;
};

// Complete scalar family admitted by the importer, excluding bool (which is
// generated as a predicate rather than an arithmetic value) and unsupported
// long double.
constexpr std::array<TypeDescriptor, 21> kScalarTypes = {{
    {"signed char", TypeCategory::kInteger},
    {"char", TypeCategory::kInteger},
    {"short", TypeCategory::kInteger},
    {"int", TypeCategory::kInteger},
    {"long", TypeCategory::kInteger},
    {"long long", TypeCategory::kInteger},
    {"wchar_t", TypeCategory::kInteger},
    {"unsigned char", TypeCategory::kInteger},
    {"char8_t", TypeCategory::kInteger},
    {"char16_t", TypeCategory::kInteger},
    {"char32_t", TypeCategory::kInteger},
    {"unsigned short", TypeCategory::kInteger},
    {"unsigned", TypeCategory::kInteger},
    {"unsigned long", TypeCategory::kInteger},
    {"unsigned long long", TypeCategory::kInteger},
    {"__float8_e4m3fn", TypeCategory::kFloatingPoint},
    {"__float8_e5m2", TypeCategory::kFloatingPoint},
    {"_Float16", TypeCategory::kFloatingPoint},
    {"__bf16", TypeCategory::kFloatingPoint},
    {"float", TypeCategory::kFloatingPoint},
    {"double", TypeCategory::kFloatingPoint},
}};

constexpr std::array<const char*, 10> kIntegerOperators = {
    "+", "-", "*", "/", "%", "^", "&", "|", "<<", ">>",
};
constexpr std::array<const char*, 4> kFloatingPointOperators = {
    "+",
    "-",
    "*",
    "/",
};
constexpr std::array<const char*, 6> kComparisonOperators = {
    "<", ">", "<=", ">=", "==", "!=",
};

class ByteStream {
 public:
  explicit ByteStream(std::span<const uint8_t> bytes) : bytes_(bytes) {
    for (uint8_t byte : bytes) {
      fallback_state_ ^= byte;
      fallback_state_ *= 1099511628211ull;
    }
  }

  size_t Choose(size_t count) { return Next() % count; }

 private:
  uint8_t Next() {
    if (position_ < bytes_.size()) {
      return bytes_[position_++];
    }
    fallback_state_ ^= fallback_state_ >> 12;
    fallback_state_ ^= fallback_state_ << 25;
    fallback_state_ ^= fallback_state_ >> 27;
    fallback_state_ *= 2685821657736338717ull;
    return static_cast<uint8_t>(fallback_state_ >> 56);
  }

  // Caller-owned mutation bytes consumed in source order.
  std::span<const uint8_t> bytes_;
  // Position of the next unconsumed mutation byte.
  size_t position_ = 0;
  // Deterministic choices used after all mutation bytes are consumed.
  uint64_t fallback_state_ = 1469598103934665603ull;
};

const char* ChooseArithmeticOperator(ByteStream& stream,
                                     TypeCategory category) {
  if (category == TypeCategory::kFloatingPoint) {
    return kFloatingPointOperators[stream.Choose(
        kFloatingPointOperators.size())];
  }
  return kIntegerOperators[stream.Choose(kIntegerOperators.size())];
}

std::string Cast(const TypeDescriptor& type, size_t value) {
  return "static_cast<" + std::string(type.spelling) + ">(" +
         std::to_string(value) + ")";
}

std::string GenerateExpression(ByteStream& stream, const TypeDescriptor& type,
                               size_t depth, const char* left_name,
                               const char* right_name) {
  if (depth == 0) {
    switch (stream.Choose(3)) {
      case 0:
        return left_name;
      case 1:
        return right_name;
      default:
        return Cast(type, 1 + stream.Choose(15));
    }
  }
  const std::string left =
      GenerateExpression(stream, type, depth - 1, left_name, right_name);
  const std::string right =
      GenerateExpression(stream, type, depth - 1, left_name, right_name);
  return "(" + left + " " + ChooseArithmeticOperator(stream, type.category) +
         " " + right + ")";
}

void AppendPreamble(std::string& source) {
  source +=
      "#include <loomcxx/numeric.h>\n"
      "#include <loomcxx/scalar.h>\n"
      "#include <loomcxx/view.h>\n\n"
      "template <class T>\n"
      "static T generated_arithmetic(T left, T right) {\n"
      "  return left + right;\n"
      "}\n\n"
      "template <class T>\n"
      "static T generated_select(bool predicate, T true_value, T false_value) "
      "{\n"
      "  return predicate ? true_value : false_value;\n"
      "}\n\n"
      "static int generated_bias(int value) { return value + 1; }\n"
      "static float generated_bias(float value) { return value + 1.0f; }\n\n"
      "[[loom::symbol(\"fuzz.overload.i32\")]] int generated_overload_probe("
      "int value) {\n"
      "  return generated_bias(value);\n"
      "}\n\n"
      "[[loom::symbol(\"fuzz.overload.f32\")]] float "
      "generated_overload_probe(float value) {\n"
      "  return generated_bias(value);\n"
      "}\n\n"
      "[[loom::symbol(\"fuzz.intrinsic.f32\")]] float generated_intrinsic("
      "float value) {\n"
      "  return loom::scalar::fmaf(value, value, 1.0f);\n"
      "}\n\n";
}

void AppendScalarFunction(ByteStream& stream, size_t function_ordinal,
                          std::string& source) {
  const TypeDescriptor& type = kScalarTypes[stream.Choose(kScalarTypes.size())];
  const TypeDescriptor& converted_type =
      kScalarTypes[stream.Choose(kScalarTypes.size())];
  const char* loop_operator = ChooseArithmeticOperator(stream, type.category);
  const char* true_operator = ChooseArithmeticOperator(stream, type.category);
  const char* false_operator = ChooseArithmeticOperator(stream, type.category);
  const char* comparison =
      kComparisonOperators[stream.Choose(kComparisonOperators.size())];
  const size_t constant = 1 + stream.Choose(15);
  const size_t loop_count = 1 + stream.Choose(8);
  const size_t expression_depth = 1 + stream.Choose(3);
  const std::string ordinal = std::to_string(function_ordinal);

  source += "[[loom::symbol(\"fuzz.scalar." + ordinal + "\")]] ";
  source += type.spelling;
  source += " generated_scalar_" + ordinal + "(";
  source += type.spelling;
  source += " left, ";
  source += type.spelling;
  source += " right) {\n  ";
  source += type.spelling;
  source += " value = generated_arithmetic(left, right);\n  value = value ";
  source += ChooseArithmeticOperator(stream, type.category);
  source += " " + Cast(type, constant) + ";\n  value = ";
  source +=
      GenerateExpression(stream, type, expression_depth, "value", "right");
  source += ";\n  bool predicate = value ";
  source += comparison;
  source += " right;\n  value = generated_select(predicate, value, right);\n  ";
  source += converted_type.spelling;
  source += " converted = loom::convert<";
  source += converted_type.spelling;
  source += ">(value);\n  value = loom::convert<";
  source += type.spelling;
  source += ">(converted);\n  for (unsigned index = 0; index < ";
  source += std::to_string(loop_count);
  source += "u; ++index) {\n    value = value ";
  source += loop_operator;
  source += " static_cast<";
  source += type.spelling;
  source += ">(index + 1u);\n  }\n  if (predicate) {\n    value = value ";
  source += true_operator;
  source += " " + Cast(type, 1 + stream.Choose(15));
  source += ";\n  } else {\n    value = value ";
  source += false_operator;
  source += " " + Cast(type, 1 + stream.Choose(15));
  source += ";\n  }\n  return value;\n}\n\n";
}

void AppendKernel(ByteStream& stream, std::string& source) {
  const TypeDescriptor& type = kScalarTypes[stream.Choose(kScalarTypes.size())];
  const char* arithmetic_operator =
      ChooseArithmeticOperator(stream, type.category);
  const size_t element_count = 1 + stream.Choose(8);
  const size_t constant = 1 + stream.Choose(15);
  const bool use_buffer_view = stream.Choose(2) != 0;

  source +=
      "[[loom::kernel, loom::workgroup_count(1, 1, 1),\n"
      "  loom::workgroup_size(1, 1, 1)]] void generated_kernel(";
  if (use_buffer_view) {
    source += "loom::type::buffer<";
    source += type.spelling;
    source += "> output, loom::type::buffer<";
    source += type.spelling;
    source += "> input) {\n  auto source = loom::buffer::view<";
    source += std::to_string(element_count);
    source += ">(loom::buffer::cast<const ";
    source += type.spelling;
    source += ">(input), {}, loom::encoding::layout::dense<1>());\n";
    source += "  auto destination = loom::buffer::view<";
    source += std::to_string(element_count);
    source += ">(";
    source += "output, {}, loom::encoding::layout::dense<1>());\n";
  } else {
    source += type.spelling;
    source += "* output, const ";
    source += type.spelling;
    source += "* input) {\n";
  }
  source += "  for (unsigned index = 0; index < ";
  source += std::to_string(element_count);
  source += "u; ++index) {\n    ";
  source += type.spelling;
  source += use_buffer_view ? " value = loom::view::load(source, index);\n"
                            : " value = input[index];\n";
  source += "    value = value ";
  source += arithmetic_operator;
  source += " " + Cast(type, constant) + ";\n    ";
  source += use_buffer_view ? "loom::view::store(value, destination, index);\n"
                            : "output[index] = value;\n";
  source += "  }\n}\n";
}

}  // namespace

std::string GenerateSource(std::span<const uint8_t> bytes) {
  bytes = bytes.first(std::min(bytes.size(), kMaximumMutationByteCount));
  ByteStream stream(bytes);
  std::string source;
  source.reserve(kMaximumGeneratedSourceByteCount);
  AppendPreamble(source);
  const size_t function_count = 1 + stream.Choose(4);
  for (size_t i = 0; i < function_count; ++i) {
    AppendScalarFunction(stream, i, source);
  }
  AppendKernel(stream, source);
  return source;
}

}  // namespace loom::cxx_import::testing
