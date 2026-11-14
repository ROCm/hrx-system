// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_MEMORY_ACCESS_H_
#define LOOM_IMPORT_CXX_BINDING_MEMORY_ACCESS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstddef>

#include "loom/ops/op_defs.h"

namespace loom::cxx_import {

class Diagnostics;

// Admits an optional constant memory-access selector against the operation's
// supported flags. An absent selector or a type parameter carries no flags.
// Called once when binding a source function; calls retain the admitted bits.
loom_memory_access_flags_t read_memory_access_flags(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, size_t argument_index,
    loom_memory_access_flags_t allowed_flags, cxx::AST* owner);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_MEMORY_ACCESS_H_
