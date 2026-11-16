// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/memory_access.h"

#include <cxx/const_value.h>
#include <cxx/symbols.h>

#include <cstdint>

#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {
namespace {

// Must match loom::memory_access::may_alias. This selector controls C++
// inference and is removed before operation construction.
constexpr uint64_t kMayAlias = uint64_t{1} << 31;

}  // namespace

loom_memory_access_flags_t read_memory_access_flags(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::FunctionSymbol* function, size_t argument_index,
    loom_memory_access_flags_t allowed_flags,
    loom_memory_access_flags_t inferred_flags, cxx::AST* owner) {
  inferred_flags &= allowed_flags;
  auto arguments = function->templateArguments();
  if (argument_index >= arguments.size()) {
    return inferred_flags;
  }
  auto value = cxx::template_argument_value(arguments[argument_index]);
  if (!value) {
    return inferred_flags;
  }
  auto* number = std::get_if<cxx::ConstInt>(&*value);
  uint64_t supported_flags = allowed_flags;
  if (iree_any_bit_set(allowed_flags, LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT)) {
    supported_flags |= kMayAlias;
  }
  if (!number || number->isNegative() ||
      (number->toUWide() | supported_flags) != supported_flags) {
    diagnostics.reject(unit, owner, "unsupported memory access flags");
  }
  const uint64_t selected = number->toUWide();
  if (iree_all_bits_set(selected,
                        kMayAlias | LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT)) {
    diagnostics.reject(unit, owner, "conflicting memory aliasing flags");
  }
  if (iree_any_bit_set(selected, kMayAlias)) {
    inferred_flags &= ~LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT;
  }
  return inferred_flags |
         static_cast<loom_memory_access_flags_t>(selected & allowed_flags);
}

}  // namespace loom::cxx_import
