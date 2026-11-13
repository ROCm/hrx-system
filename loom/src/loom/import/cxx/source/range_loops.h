// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_
#define LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>

#include <unordered_map>
#include <unordered_set>

namespace loom::cxx_import {

// Ordinary declaration and loop forms for checked range iteration. The owning
// function walk creates each expansion once; translation consumes the retained
// scope. Wrappers belong to the source arena and borrow checked expressions,
// symbols, and the written body without modifying the frontend's source tree.
class RangeLoops {
 public:
  explicit RangeLoops(cxx::TranslationUnit& unit) : unit_(unit) {}

  cxx::CompoundStatementAST* expand(cxx::ForRangeStatementAST* source);
  cxx::CompoundStatementAST* scope(cxx::ForRangeStatementAST* source) const;

  // The hidden range initializer owns the loop's extended temporary lifetimes.
  bool initializes_range(cxx::InitDeclaratorAST* declaration) const;

 private:
  // Source arena owning both the original syntax and these semantic wrappers.
  cxx::TranslationUnit& unit_;
  // Completed expansions indexed by the original source loop identity.
  std::unordered_map<cxx::ForRangeStatementAST*, cxx::CompoundStatementAST*>
      scopes_;
  // Synthetic range declarations, distinct from iterator and element locals.
  std::unordered_set<cxx::InitDeclaratorAST*> initializers_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_
