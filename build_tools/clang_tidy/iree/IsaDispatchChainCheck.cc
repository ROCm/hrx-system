// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/IsaDispatchChainCheck.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

constexpr unsigned kDefaultMaximumIsaPredicates = 3;

unsigned GetUnsignedOption(ClangTidyCheck::OptionsView& Options,
                           StringRef CheckName, StringRef OptionName,
                           unsigned Default, ClangTidyContext* Context) {
  std::optional<StringRef> Value = Options.get(OptionName);
  if (!Value) {
    return Default;
  }
  unsigned long long Parsed = 0;
  if (!Value->getAsInteger(10, Parsed) &&
      Parsed <= std::numeric_limits<unsigned>::max()) {
    return static_cast<unsigned>(Parsed);
  }

  std::string FullName = (CheckName + "." + OptionName).str();
  Context->configurationDiag(
      "invalid configuration value '%0' for option '%1'; expected a "
      "non-negative integer")
      << *Value << FullName;
  return Default;
}

bool IsIsaPredicate(const CallExpr* Call) {
  const FunctionDecl* Callee = Call->getDirectCallee();
  if (!Callee) {
    return false;
  }
  if (Callee->getName().ends_with("_isa")) {
    return true;
  }
  const FunctionTemplateDecl* PrimaryTemplate = Callee->getPrimaryTemplate();
  return PrimaryTemplate && PrimaryTemplate->getName() == "isa";
}

class IsaPredicateCollector final
    : public RecursiveASTVisitor<IsaPredicateCollector> {
 public:
  bool VisitCallExpr(CallExpr* Call) {
    if (IsIsaPredicate(Call)) {
      calls.push_back(Call);
    }
    return true;
  }

  // A lambda body is a distinct function, even when the lambda expression is
  // nested inside a condition. Its predicates do not participate in the
  // surrounding branch's dispatch decision.
  bool TraverseLambdaExpr(LambdaExpr*) { return true; }

  std::vector<const CallExpr*> calls;
};

bool IsElseIf(const IfStmt* If, ASTContext& Context) {
  for (const DynTypedNode& Parent : Context.getParents(*If)) {
    const auto* ParentIf = Parent.get<IfStmt>();
    if (ParentIf && ParentIf->getElse() == If) {
      return true;
    }
  }
  return false;
}

SourceLocation PredicateLocation(const CallExpr* Call,
                                 const SourceManager& SourceManager) {
  SourceLocation Location = Call->getCallee()->getExprLoc();
  return SourceManager.getExpansionLoc(Location);
}

void CollectConditionPredicates(const Expr* Condition,
                                const SourceManager& SourceManager,
                                std::vector<const CallExpr*>& Calls) {
  IsaPredicateCollector Collector;
  Collector.TraverseStmt(const_cast<Expr*>(Condition));
  std::stable_sort(Collector.calls.begin(), Collector.calls.end(),
                   [&SourceManager](const CallExpr* Lhs, const CallExpr* Rhs) {
                     return SourceManager.isBeforeInTranslationUnit(
                         PredicateLocation(Lhs, SourceManager),
                         PredicateLocation(Rhs, SourceManager));
                   });
  Calls.insert(Calls.end(), Collector.calls.begin(), Collector.calls.end());
}

}  // namespace

IsaDispatchChainCheck::IsaDispatchChainCheck(StringRef Name,
                                             ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      maximum_isa_predicates_(
          GetUnsignedOption(Options, Name, "MaximumIsaPredicates",
                            kDefaultMaximumIsaPredicates, Context)) {}

void IsaDispatchChainCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      ifStmt(unless(isExpansionInSystemHeader())).bind("if_chain"), this);
}

void IsaDispatchChainCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Root = Result.Nodes.getNodeAs<IfStmt>("if_chain");
  if (!Root || IsElseIf(Root, *Result.Context)) {
    return;
  }

  const SourceManager& SourceManager = *Result.SourceManager;
  std::vector<const CallExpr*> Predicates;
  for (const IfStmt* Branch = Root; Branch;) {
    CollectConditionPredicates(Branch->getCond(), SourceManager, Predicates);
    Branch = dyn_cast_or_null<IfStmt>(Branch->getElse());
  }
  if (Predicates.size() <= maximum_isa_predicates_) {
    return;
  }

  SourceLocation RootLocation = SourceManager.getExpansionLoc(Root->getIfLoc());
  diag(RootLocation,
       "concrete isa dispatch contains %0 predicates; the configured maximum "
       "is %1; use a semantic interface or trait, a producer-retained kind, "
       "or an owner-local descriptor")
      << Predicates.size() << maximum_isa_predicates_;
  diag(PredicateLocation(Predicates[maximum_isa_predicates_], SourceManager),
       "the first predicate over the limit is here", DiagnosticIDs::Note);
}

void IsaDispatchChainCheck::storeOptions(ClangTidyOptions::OptionMap& Options) {
  this->Options.store(Options, "MaximumIsaPredicates",
                      std::to_string(maximum_isa_predicates_));
}

}  // namespace clang::tidy::iree
