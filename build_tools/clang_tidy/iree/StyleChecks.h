// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_IREE_STYLE_CHECKS_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_IREE_STYLE_CHECKS_H_

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::iree {

class DirectGotoCheck final : public ClangTidyCheck {
 public:
  DirectGotoCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

class DesignatedInitializerCheck final : public ClangTidyCheck {
 public:
  DesignatedInitializerCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

class GuardedReleaseCheck final : public ClangTidyCheck {
 public:
  GuardedReleaseCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

class TestStatusMacroCheck final : public ClangTidyCheck {
 public:
  TestStatusMacroCheck(StringRef Name, ClangTidyContext* Context);

  void registerPPCallbacks(const SourceManager& SM, Preprocessor* PP,
                           Preprocessor* ModuleExpanderPP) override;
};

class TestStatusPredicateCheck final : public ClangTidyCheck {
 public:
  TestStatusPredicateCheck(StringRef Name, ClangTidyContext* Context);

  void registerPPCallbacks(const SourceManager& SM, Preprocessor* PP,
                           Preprocessor* ModuleExpanderPP) override;
};

}  // namespace clang::tidy::iree

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_IREE_STYLE_CHECKS_H_
