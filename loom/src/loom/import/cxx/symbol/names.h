// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SYMBOL_NAMES_H_
#define LOOM_IMPORT_CXX_SYMBOL_NAMES_H_

#include <string>
#include <string_view>
#include <unordered_map>

#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

// Whether a nonempty spelling can be printed and parsed as one Loom symbol.
bool is_symbol_name(std::string_view name);

// Owns the shared output namespace for functions and configuration symbols.
// Declaration admission reserves explicit names before any reachable private
// helper receives a generated name. Source objects outlive this owner; output
// producers copy chosen spellings into their module.
class SymbolNames {
 public:
  SymbolNames(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  // Claims an exact name once for its admitted producer. Collisions diagnose
  // at the source boundary, including generated launch configuration names.
  void reserve(std::string_view name, cxx::AST* source);

  // Assigns a private name, disambiguating against all earlier reservations.
  std::string unique(std::string spelling);

 private:
  // Invocation-owned source used to locate conflicting declarations.
  cxx::TranslationUnit& unit_;
  // Public source admission diagnostics.
  Diagnostics& diagnostics_;
  // Exact names and private suffixes share one collision domain. Each entry
  // retains the last private suffix attempted for that occupied spelling.
  std::unordered_map<std::string, size_t> names_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SYMBOL_NAMES_H_
