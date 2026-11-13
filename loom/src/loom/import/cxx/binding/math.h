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

// Resolves standard math builtins and exact C library prototypes to existing
// scalar projections. The caller owns declaration admission (definitions and
// explicit bindings); the frontend owns signatures and argument conversions.
// Unsupported math diagnoses at the call; unrelated functions return null.
const loom_cxx_scalar_binding_t* math_binding(cxx::TranslationUnit& unit,
                                              Diagnostics& diagnostics,
                                              cxx::FunctionSymbol* function,
                                              cxx::AST* owner);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_MATH_H_
