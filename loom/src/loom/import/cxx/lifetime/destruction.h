// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_LIFETIME_DESTRUCTION_H_
#define LOOM_IMPORT_CXX_LIFETIME_DESTRUCTION_H_

#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>
#include <optional>

namespace cxx {
class TranslationUnit;
}

namespace loom::cxx_import {

// Complete-object destruction selected by the verified C++ frontend. A
// bounded array uses the element class destructor once per element in reverse
// index order; a class object has one element. Absence means that destruction
// is trivial and emits no runtime work.
struct Destruction {
  // Complete class element type after removing array extents and cv qualifiers.
  const cxx::Type* element_type;
  // Selected non-deleted class destructor.
  cxx::FunctionSymbol* function;
  // Number of complete class elements in the object.
  size_t element_count;
};

// Returns the frontend-selected runtime destruction for |type|. The source has
// already crossed parsing and type checking, so a nontrivial object is a
// complete class or bounded array with one usable selected destructor.
std::optional<Destruction> destruction_for(cxx::TranslationUnit& unit,
                                           const cxx::Type* type);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_LIFETIME_DESTRUCTION_H_
