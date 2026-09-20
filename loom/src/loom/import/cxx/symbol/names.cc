// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/names.h"

namespace loom::cxx_import {

bool is_symbol_name(std::string_view name) {
  if (name.empty()) {
    return false;
  }
  for (char character : name) {
    if (!((character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9') || character == '_' ||
          character == '$' || character == '.' || character == '-')) {
      return false;
    }
  }
  return true;
}

void SymbolNames::reserve(std::string_view name, cxx::AST* source) {
  if (!names_.try_emplace(std::string(name), 0).second) {
    diagnostics_.reject(
        unit_, source,
        "Loom symbol name is already reserved: " + std::string(name));
  }
}

std::string SymbolNames::unique(std::string spelling) {
  auto [base, inserted] = names_.try_emplace(spelling, 0);
  if (inserted) {
    return spelling;
  }
  auto& ordinal = base->second;
  while (true) {
    auto candidate = spelling + "_" + std::to_string(++ordinal);
    if (names_.try_emplace(candidate, 0).second) {
      return candidate;
    }
  }
}

}  // namespace loom::cxx_import
