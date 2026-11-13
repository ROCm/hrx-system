// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_MATH_H_
#define LOOM_IMPORT_CXX_BINDING_MATH_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>

#include "loom/import/cxx/binding/scalar_bindings.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

// Resolves frontend-owned standard math builtins to process-lifetime scalar
// projections. Unsupported math diagnoses at the call; unrelated builtins and
// ordinary functions return null. Source signatures and conversions belong to
// the frontend, while the intrinsic owner retains the concrete typed binding.
const loom_cxx_scalar_binding_t* builtin_math_binding(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, cxx::AST* owner);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_MATH_H_
