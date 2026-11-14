// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_
#define LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_

#include <cxx/ast_fwd.h>

#include <unordered_set>

namespace loom::cxx_import {

// Indexes the frontend's checked ordinary declaration and loop forms for range
// iteration. The lowered scope remains owned by the source unit; this class
// only retains which synthesized declaration owns the range temporary's
// extended lifetimes.
class RangeLoops {
 public:
  RangeLoops() = default;

  cxx::CompoundStatementAST* expand(cxx::ForRangeStatementAST* source);
  cxx::CompoundStatementAST* scope(cxx::ForRangeStatementAST* source) const;

  // The hidden range initializer owns the loop's extended temporary lifetimes.
  bool initializes_range(cxx::InitDeclaratorAST* declaration) const;

 private:
  // Frontend-lowered range declarations, distinct from iterator and element
  // locals.
  std::unordered_set<cxx::InitDeclaratorAST*> initializers_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_RANGE_LOOPS_H_
