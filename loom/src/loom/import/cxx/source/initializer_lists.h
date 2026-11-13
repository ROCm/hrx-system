// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_INITIALIZER_LISTS_H_
#define LOOM_IMPORT_CXX_SOURCE_INITIALIZER_LISTS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <unordered_map>

namespace loom::cxx_import {

class Diagnostics;

// One checked list initializes an array, independently of its library facade.
// The original clauses retain the frontend's copy-initialization conversions.
struct InitializerList {
  // Arena-owned const-array construction; its extent can be zero.
  cxx::BracedInitListAST* backing;
  // Library field carrying the beginning of the backing array.
  cxx::FieldSymbol* begin;
  // Library field carrying either its element count or past-the-end pointer.
  cxx::FieldSymbol* limit;
};

// Retains source adaptations during the owning evaluated-expression walk.
// Lookups consume the retained result without changing the original AST.
class InitializerLists {
 public:
  InitializerLists(cxx::TranslationUnit& unit, Diagnostics& diagnostics)
      : unit_(unit), diagnostics_(diagnostics) {}

  const InitializerList& retain(cxx::BracedInitListAST* source);
  const InitializerList* get(cxx::BracedInitListAST* source) const;

 private:
  // Owns source types, field identities, and the adaptation's arena storage.
  cxx::TranslationUnit& unit_;
  // Unsupported library layouts are diagnosed at the source initializer.
  Diagnostics& diagnostics_;
  // Sparse plans exist only for checked std::initializer_list constructions.
  std::unordered_map<cxx::BracedInitListAST*, InitializerList> lists_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_INITIALIZER_LISTS_H_
