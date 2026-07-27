// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_IREE_TRACE_CHECKS_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_IREE_TRACE_CHECKS_H_

#include <string>

#include "clang-tidy/ClangTidyCheck.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::tidy::iree {

enum class TraceMacroKind {
  kNone,
  kBegin,
  kEnd,
  kReturn,
  kReturnIfError,
  kReturnAndEndIfError,
  kReturnAndEnd,
  kAdopt,
  kTransfer,
};

struct TraceMacroExpansion {
  TraceMacroKind Kind = TraceMacroKind::kNone;
  std::string Name;
  std::string ZoneId;
  SourceLocation BeginLocation;
  SourceLocation EndLocation;
};

class TraceZoneCheck final : public ClangTidyCheck {
 public:
  TraceZoneCheck(StringRef Name, ClangTidyContext* Context);

  void registerPPCallbacks(const SourceManager& SM, Preprocessor* PP,
                           Preprocessor* ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder* Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

  void addTraceMacroExpansion(TraceMacroKind Kind, StringRef Name,
                              std::string ZoneId, SourceLocation BeginLocation,
                              SourceLocation EndLocation);
  ArrayRef<TraceMacroExpansion> traceMacroExpansions() const {
    return TraceMacroExpansions_;
  }

 private:
  llvm::SmallVector<TraceMacroExpansion, 256> TraceMacroExpansions_;
};

}  // namespace clang::tidy::iree

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_IREE_TRACE_CHECKS_H_
