// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_IREE_ISA_DISPATCH_CHAIN_CHECK_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_IREE_ISA_DISPATCH_CHAIN_CHECK_H_

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::iree {

// Bounds concrete type predicates across a maximal if/else-if chain. Small
// local distinctions remain direct; larger families require a named semantic
// owner instead of continuing to grow a predicate ladder.
class IsaDispatchChainCheck final : public ClangTidyCheck {
 public:
  IsaDispatchChainCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
  void storeOptions(ClangTidyOptions::OptionMap& Options) override;

 private:
  unsigned maximum_isa_predicates_;
};

}  // namespace clang::tidy::iree

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_IREE_ISA_DISPATCH_CHAIN_CHECK_H_
