// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_SOURCE_SIGNATURE_H_
#define LOOM_IMPORT_CXX_SYMBOL_SOURCE_SIGNATURE_H_

#include <cxx/symbols_fwd.h>

#include "loom/ir/attribute.h"

typedef struct loom_module_t loom_module_t;

namespace cxx {
class AST;
class TranslationUnit;
}  // namespace cxx

namespace loom::cxx_import {

class Diagnostics;
class Types;

// Builds the retained C source signature for an ordinary function with C
// language linkage. Functions with C++ linkage return an absent attribute.
// Unsupported C boundary types diagnose at |source| and throw SourceRejected.
loom_attribute_t build_c_source_signature(cxx::TranslationUnit& unit,
                                          Diagnostics& diagnostics,
                                          loom_module_t* module,
                                          cxx::FunctionSymbol* function,
                                          cxx::AST* source, Types& types);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_SOURCE_SIGNATURE_H_
