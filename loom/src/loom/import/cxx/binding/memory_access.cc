// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/memory_access.h"

#include <cxx/const_value.h>
#include <cxx/symbols.h>

#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {

loom_memory_access_flags_t read_memory_access_flags(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, size_t argument_index,
    loom_memory_access_flags_t allowed_flags, cxx::AST* owner) {
  auto arguments = function->templateArguments();
  if (argument_index >= arguments.size()) {
    return 0;
  }
  auto value = cxx::template_argument_value(arguments[argument_index]);
  if (!value) {
    return 0;
  }
  auto* number = std::get_if<cxx::ConstInt>(&*value);
  if (!number || number->isNegative() ||
      (number->toUWide() | allowed_flags) != allowed_flags) {
    diagnostics.reject(unit, owner, "unsupported memory access flags");
  }
  return static_cast<loom_memory_access_flags_t>(number->toUIntMax());
}

}  // namespace loom::cxx_import
