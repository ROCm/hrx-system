// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_IREE_DESIGNATED_INITIALIZER_CHECK_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_IREE_DESIGNATED_INITIALIZER_CHECK_H_

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::iree {

// Enforces real C++20 designators for aggregate member labels and folds
// aggregate setup statements when initialization and assignment are proven
// equivalent.
class DesignatedInitializerCheck final : public ClangTidyCheck {
 public:
  DesignatedInitializerCheck(StringRef Name, ClangTidyContext* Context);

  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
  void storeOptions(ClangTidyOptions::OptionMap& Options) override;

 private:
  // Whether to diagnose and convert comment field labels.
  const bool enable_comment_label_conversion_;

  // Whether to diagnose and fold empty-initializer setup blocks.
  const bool enable_setup_block_folding_;
};

}  // namespace clang::tidy::iree

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_IREE_DESIGNATED_INITIALIZER_CHECK_H_
