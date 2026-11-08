// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_OBJECTS_H_
#define LOOM_IMPORT_CXX_SOURCE_OBJECTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>

namespace loom::cxx_import {

// Source type of a variable, field, or direct base subobject. Base declarations
// retain their class through symbol(), independently of Symbol::type().
const cxx::Type* source_object_type(cxx::Symbol* symbol);

// Physical position of one admitted direct field or nonvirtual base. Complete
// class layout is owned by the frontend; no inheritance lookup is performed.
uint64_t source_subobject_offset(cxx::Symbol* symbol);

// Whether this source conversion changes the selected class subobject. Its
// retained path always runs from derived to base; the kind selects direction.
bool is_class_adjustment(const cxx::ImplicitCastExpressionAST* expression);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_OBJECTS_H_
