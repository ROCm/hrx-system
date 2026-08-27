// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_IREE_RECURSION_CHECK_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_IREE_RECURSION_CHECK_H_

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::iree {

// Diagnoses call-graph cycles whose native stack depth has no mechanically
// proven fixed bound.
class UnboundedRecursionCheck final : public ClangTidyCheck {
 public:
  UnboundedRecursionCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

}  // namespace clang::tidy::iree

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_IREE_RECURSION_CHECK_H_
