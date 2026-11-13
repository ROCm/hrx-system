// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/math.h"

#include <cxx/builtin_signature.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <string_view>

namespace loom::cxx_import {
namespace {

struct MathOperation {
  // Standard spelling before a float or long-double suffix.
  std::string_view name;
  // Existing scalar projection, or null for a distinct unsupported contract.
  const char* scalar;
};

// Scalar projections own both numeric signatures and operation emission. Frexp
// adapts its exponent result to the C pointer-output calling convention.
// Other contracts remain explicit rejections, not approximations to a nearby
// operation (in particular rounding controlled by the floating environment).
constexpr MathOperation kOperations[] = {
    {"abs", "scalar.absi"},
    {"acos", "scalar.acosf"},
    {"acosh", "scalar.acoshf"},
    {"asin", "scalar.asinf"},
    {"asinh", "scalar.asinhf"},
    {"atan", "scalar.atanf"},
    {"atan2", "scalar.atan2f"},
    {"atanh", "scalar.atanhf"},
    {"cbrt", "scalar.cbrtf"},
    {"ceil", "scalar.ceilf"},
    {"copysign", "scalar.copysignf"},
    {"cos", "scalar.cosf"},
    {"cosh", "scalar.coshf"},
    {"erf", "scalar.erff"},
    {"erfc", "scalar.erfcf"},
    {"exp", "scalar.expf"},
    {"exp2", "scalar.exp2f"},
    {"expm1", "scalar.expm1f"},
    {"fabs", "scalar.absf"},
    {"fdim", nullptr},
    {"floor", "scalar.floorf"},
    {"fma", "scalar.fmaf"},
    {"fmax", "scalar.maxnumf"},
    {"fmin", "scalar.minnumf"},
    {"fmod", "scalar.remf"},
    {"fpclassify", nullptr},
    {"frexp", "scalar.frexpf"},
    {"hypot", nullptr},
    {"ilogb", nullptr},
    {"isfinite", nullptr},
    {"isgreater", nullptr},
    {"isgreaterequal", nullptr},
    {"isinf", nullptr},
    {"isless", nullptr},
    {"islessequal", nullptr},
    {"islessgreater", nullptr},
    {"isnan", nullptr},
    {"isnormal", nullptr},
    {"isunordered", nullptr},
    {"labs", "scalar.absi"},
    {"ldexp", "scalar.ldexpf"},
    {"lgamma", nullptr},
    {"llabs", "scalar.absi"},
    {"llrint", nullptr},
    {"llround", nullptr},
    {"log", "scalar.logf"},
    {"log10", "scalar.log10f"},
    {"log1p", "scalar.log1pf"},
    {"log2", "scalar.log2f"},
    {"logb", nullptr},
    {"lrint", nullptr},
    {"lround", nullptr},
    {"modf", nullptr},
    {"nearbyint", nullptr},
    {"nextafter", nullptr},
    {"nexttoward", nullptr},
    {"pow", "scalar.powf"},
    {"remainder", nullptr},
    {"remquo", nullptr},
    {"rint", nullptr},
    {"round", "scalar.roundf"},
    {"roundeven", "scalar.roundevenf"},
    {"scalbln", nullptr},
    {"scalbn", "scalar.ldexpf"},
    {"signbit", nullptr},
    {"sin", "scalar.sinf"},
    {"sincos", nullptr},
    {"sinh", "scalar.sinhf"},
    {"sqrt", "scalar.sqrtf"},
    {"tan", "scalar.tanf"},
    {"tanh", "scalar.tanhf"},
    {"tgamma", nullptr},
    {"trunc", "scalar.truncf"},
};
static_assert([] {
  for (size_t index = 1; index < std::size(kOperations); ++index) {
    if (kOperations[index - 1].name >= kOperations[index].name) {
      return false;
    }
  }
  return true;
}());

const MathOperation* find_operation(std::string_view name) {
  auto* position = std::lower_bound(
      std::begin(kOperations), std::end(kOperations), name,
      [](const MathOperation& operation, std::string_view name) {
        return operation.name < name;
      });
  return position != std::end(kOperations) && position->name == name ? position
                                                                     : nullptr;
}

// Reuse the frontend's signature catalog instead of duplicating C prototypes.
// The spelling buffer is bounded by this table plus the builtin prefix and an
// optional precision suffix; no temporary strings or symbols are allocated.
bool matches_library_signature(cxx::TranslationUnit& unit,
                               const cxx::FunctionType* signature,
                               std::string_view name) {
  constexpr std::string_view prefix = "__builtin_";
  constexpr size_t maximum_name_length = [] {
    size_t length = 0;
    for (const auto& operation : kOperations) {
      length = std::max(length, operation.name.size());
    }
    return length + 1;
  }();
  std::array<char, prefix.size() + maximum_name_length> spelling;
  std::copy(prefix.begin(), prefix.end(), spelling.begin());
  std::copy(name.begin(), name.end(), spelling.begin() + prefix.size());
  const auto kind = cxx::Token::builtinFunctionKind(
      {spelling.data(), prefix.size() + name.size()});
  const auto catalog = cxx::builtinSignatureOf(kind);
  if (catalog.count != 1) {
    return false;
  }
  const auto expected = cxx::decodeBuiltinSignature(unit.control(), catalog, 0);
  const auto same_type = [&](const cxx::Type* left, const cxx::Type* right) {
    return unit.typeTraits().remove_cv(left) ==
           unit.typeTraits().remove_cv(right);
  };
  return expected.type &&
         signature->isVariadic() == expected.type->isVariadic() &&
         same_type(signature->returnType(), expected.type->returnType()) &&
         std::ranges::equal(signature->parameterTypes(),
                            expected.type->parameterTypes(), same_type);
}

}  // namespace

const loom_cxx_scalar_binding_t* math_binding(cxx::TranslationUnit& unit,
                                              Diagnostics& diagnostics,
                                              cxx::FunctionSymbol* function,
                                              cxx::AST* owner) {
  const bool builtin =
      function->builtinKind() != cxx::BuiltinFunctionKind::T_NONE;
  if (!builtin && !function->hasCLinkage()) {
    return nullptr;
  }
  const auto& spelling =
      cxx::name_cast<cxx::Identifier>(function->name())->name();
  std::string_view name(spelling);
  if (builtin) {
    name.remove_prefix(std::string_view("__builtin_").size());
  }
  const auto* operation = find_operation(name);
  bool long_double = false;
  if (!operation && (name.ends_with('f') || name.ends_with('l'))) {
    long_double = name.ends_with('l');
    operation = find_operation(name.substr(0, name.size() - 1));
  }
  if (!operation) {
    return nullptr;
  }
  if (!builtin &&
      !matches_library_signature(
          unit, cxx::type_cast<cxx::FunctionType>(function->type()), name)) {
    return nullptr;
  }
  const auto category = builtin ? "math builtin '" : "standard math function '";
  if (long_double) {
    diagnostics.reject(unit, owner,
                       category + std::string(spelling) +
                           "' requires long-double lowering; implicit "
                           "narrowing is not permitted");
  }
  if (!operation->scalar) {
    diagnostics.reject(
        unit, owner,
        std::string("unsupported ") + category + std::string(spelling) + "'");
  }
  return loom_cxx_scalar_binding_find(
      iree_make_cstring_view(operation->scalar));
}

}  // namespace loom::cxx_import
