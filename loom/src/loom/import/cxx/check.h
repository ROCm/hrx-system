// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CHECK_H_
#define LOOM_IMPORT_CXX_CHECK_H_

#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/value/scalar.h"

namespace loom::cxx_import {

// Projects an admitted check case into existing check sources, direct function
// invocations, and terminal expectations. The caller has entered the case body.
// All collaborators and source AST remain live until translation returns.
void translate_check_body(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                          Functions& functions, Intrinsics& intrinsics,
                          Types& types, Scalars& scalars, Locations& locations,
                          loom_builder_t& builder, const FunctionBody& body);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CHECK_H_
