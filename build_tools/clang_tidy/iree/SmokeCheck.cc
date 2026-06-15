// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/SmokeCheck.h"

#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::tidy::iree {

SmokeCheck::SmokeCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void SmokeCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      functionDecl(isDefinition(), hasName("iree_clang_tidy_smoke_bad"))
          .bind("function"),
      this);
}

void SmokeCheck::check(const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function) {
    return;
  }
  diag(Function->getLocation(), "IREE clang-tidy plugin smoke diagnostic");
}

}  // namespace clang::tidy::iree
