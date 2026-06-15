// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/StyleChecks.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

bool IsExternalMacroBody(SourceLocation Location,
                         const SourceManager& SourceManager) {
  if (!Location.isMacroID()) {
    return false;
  }
  SourceLocation SpellingLocation = SourceManager.getSpellingLoc(Location);
  if (SourceManager.isInSystemHeader(SpellingLocation)) {
    return true;
  }
  llvm::StringRef Filename = SourceManager.getFilename(SpellingLocation);
  return Filename.contains("/external/") || Filename.starts_with("external/");
}

bool IsStatusTestMacroName(StringRef Name) {
  return Name == "IREE_EXPECT_OK" || Name == "IREE_ASSERT_OK" ||
         Name == "IREE_EXPECT_NOT_OK" || Name == "IREE_ASSERT_NOT_OK" ||
         Name == "IREE_EXPECT_STATUS_IS" || Name == "IREE_ASSERT_STATUS_IS" ||
         Name == "IREE_ASSERT_OK_AND_ASSIGN";
}

bool IsGTestBooleanMacroName(StringRef Name) {
  return Name == "EXPECT_TRUE" || Name == "ASSERT_TRUE" ||
         Name == "EXPECT_FALSE" || Name == "ASSERT_FALSE";
}

bool IsTestFilename(StringRef Filename) {
  if (Filename.empty()) {
    return false;
  }
  return Filename.contains("/test/") || Filename.contains("/testing/") ||
         Filename.contains("/cts/") || Filename.contains("_test.") ||
         Filename.contains("_test_") || Filename.ends_with("test_base.h");
}

bool FirstMacroArgumentContainsIdentifier(const MacroArgs* Args,
                                          StringRef Identifier) {
  if (!Args || Args->getNumMacroArguments() == 0) {
    return false;
  }
  const Token* Argument = Args->getUnexpArgument(0);
  if (!Argument) {
    return false;
  }
  unsigned ArgumentLength = MacroArgs::getArgLength(Argument);
  for (unsigned I = 0; I < ArgumentLength; ++I) {
    const IdentifierInfo* ArgumentIdentifier = Argument[I].getIdentifierInfo();
    if (ArgumentIdentifier && ArgumentIdentifier->getName() == Identifier) {
      return true;
    }
  }
  return false;
}

std::optional<std::string> SourceText(const Expr* Expr,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  CharSourceRange Range =
      CharSourceRange::getTokenRange(Expr->getSourceRange());
  bool Invalid = false;
  StringRef Text =
      Lexer::getSourceText(Range, SourceManager, LangOptions, &Invalid);
  if (Invalid || Text.empty()) {
    return std::nullopt;
  }
  std::string Result = Text.str();
  Result.erase(std::remove_if(Result.begin(), Result.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               Result.end());
  return Result;
}

std::optional<std::string> SourceText(CharSourceRange Range,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  bool Invalid = false;
  StringRef Text =
      Lexer::getSourceText(Range, SourceManager, LangOptions, &Invalid);
  if (Invalid || Text.empty()) {
    return std::nullopt;
  }
  return Text.str();
}

std::optional<std::string> CompactSourceText(CharSourceRange Range,
                                             const SourceManager& SourceManager,
                                             const LangOptions& LangOptions) {
  std::optional<std::string> Text =
      SourceText(Range, SourceManager, LangOptions);
  if (!Text) {
    return std::nullopt;
  }
  Text->erase(std::remove_if(Text->begin(), Text->end(),
                             [](unsigned char c) { return std::isspace(c); }),
              Text->end());
  return Text;
}

std::optional<CharSourceRange> StatementCharRange(
    const Stmt* Statement, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  SourceLocation Begin = Statement->getBeginLoc();
  SourceLocation End = Statement->getEndLoc();
  if (Begin.isInvalid() || End.isInvalid() || Begin.isMacroID()) {
    return std::nullopt;
  }
  End = SourceManager.getExpansionLoc(End);
  if (SourceManager.getFileID(Begin) != SourceManager.getFileID(End)) {
    return std::nullopt;
  }
  End = Lexer::getLocForEndOfToken(End, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  std::pair<FileID, unsigned> Decomposed = SourceManager.getDecomposedLoc(End);
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
  if (Invalid) {
    return std::nullopt;
  }
  if (Decomposed.second < Buffer.size() && Buffer[Decomposed.second] == ';') {
    End = End.getLocWithOffset(1);
  }
  return CharSourceRange::getCharRange(Begin, End);
}

std::optional<std::string> StatementSourceText(
    const Stmt* Statement, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  std::optional<CharSourceRange> Range =
      StatementCharRange(Statement, SourceManager, LangOptions);
  if (!Range) {
    return std::nullopt;
  }
  return SourceText(*Range, SourceManager, LangOptions);
}

const Expr* IgnoreExpressionNoise(const Expr* Expr) {
  if (!Expr) {
    return nullptr;
  }
  return Expr->IgnoreParenImpCasts();
}

bool IsNullPointerConstant(const Expr* Expr, ASTContext& Context) {
  Expr = IgnoreExpressionNoise(Expr);
  return Expr && Expr->isNullPointerConstant(Context,
                                             Expr::NPC_ValueDependentIsNotNull);
}

const Expr* NonNullGuardedExpression(const Expr* Condition,
                                     ASTContext& Context) {
  Condition = IgnoreExpressionNoise(Condition);
  if (!Condition) {
    return nullptr;
  }
  if (Condition->getType()->isAnyPointerType()) {
    return Condition;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(Condition);
  if (!Binary || Binary->getOpcode() != BO_NE) {
    return nullptr;
  }
  const Expr* Lhs = IgnoreExpressionNoise(Binary->getLHS());
  const Expr* Rhs = IgnoreExpressionNoise(Binary->getRHS());
  if (IsNullPointerConstant(Lhs, Context) && Rhs &&
      Rhs->getType()->isAnyPointerType()) {
    return Rhs;
  }
  if (IsNullPointerConstant(Rhs, Context) && Lhs &&
      Lhs->getType()->isAnyPointerType()) {
    return Lhs;
  }
  return nullptr;
}

const CallExpr* CallStatement(const Stmt* Statement) {
  const auto* Expr = dyn_cast_or_null<clang::Expr>(Statement);
  if (!Expr) {
    return nullptr;
  }
  return dyn_cast<CallExpr>(IgnoreExpressionNoise(Expr));
}

bool IsPointerLikeSingleArgumentRelease(const CallExpr* Call,
                                        const SourceManager& SourceManager,
                                        const LangOptions& LangOptions) {
  if (!Call || Call->getNumArgs() != 1) {
    return false;
  }
  std::optional<std::string> CalleeText =
      SourceText(Call->getCallee(), SourceManager, LangOptions);
  if (!CalleeText || !StringRef(*CalleeText).ends_with("_release")) {
    return false;
  }
  const FunctionDecl* Callee = Call->getDirectCallee();
  if (!Callee || Callee->getNumParams() != 1 ||
      !Callee->getReturnType()->isVoidType() ||
      !Callee->getName().ends_with("_release")) {
    return false;
  }
  return Callee->getParamDecl(0)->getType()->isAnyPointerType();
}

class TestStatusMacroRecorder final : public PPCallbacks {
 public:
  TestStatusMacroRecorder(TestStatusMacroCheck& Check, Preprocessor& PP)
      : Check_(Check), PP_(PP) {}

  void MacroExpands(const Token& MacroNameTok, const MacroDefinition&,
                    SourceRange Range, const MacroArgs*) override {
    const IdentifierInfo* Identifier = MacroNameTok.getIdentifierInfo();
    if (!Identifier) {
      return;
    }
    StringRef Name = Identifier->getName();
    if (!IsStatusTestMacroName(Name)) {
      return;
    }
    const SourceManager& SourceManager = PP_.getSourceManager();
    if (SourceManager.isInSystemHeader(MacroNameTok.getLocation())) {
      return;
    }
    if (SourceManager.isMacroBodyExpansion(MacroNameTok.getLocation())) {
      return;
    }
    SourceLocation Location = SourceManager.getExpansionLoc(Range.getBegin());
    if (Location.isInvalid()) {
      return;
    }
    StringRef Filename = SourceManager.getFilename(Location);
    if (IsTestFilename(Filename)) {
      return;
    }
    Check_.diag(Location, "%0 is a test-only status assertion macro") << Name;
  }

 private:
  TestStatusMacroCheck& Check_;
  Preprocessor& PP_;
};

class TestStatusPredicateRecorder final : public PPCallbacks {
 public:
  TestStatusPredicateRecorder(TestStatusPredicateCheck& Check, Preprocessor& PP)
      : Check_(Check), PP_(PP) {}

  void MacroExpands(const Token& MacroNameTok, const MacroDefinition&,
                    SourceRange Range, const MacroArgs* Args) override {
    const IdentifierInfo* Identifier = MacroNameTok.getIdentifierInfo();
    if (!Identifier) {
      return;
    }
    StringRef Name = Identifier->getName();
    if (!IsGTestBooleanMacroName(Name) ||
        !FirstMacroArgumentContainsIdentifier(Args, "iree_status_is_ok")) {
      return;
    }
    const SourceManager& SourceManager = PP_.getSourceManager();
    if (SourceManager.isInSystemHeader(MacroNameTok.getLocation())) {
      return;
    }
    if (SourceManager.isMacroBodyExpansion(MacroNameTok.getLocation())) {
      return;
    }
    SourceLocation Location = SourceManager.getExpansionLoc(Range.getBegin());
    if (Location.isInvalid()) {
      return;
    }
    Check_.diag(Location,
                "use IREE status test macros instead of %0 with "
                "iree_status_is_ok")
        << Name;
  }

 private:
  TestStatusPredicateCheck& Check_;
  Preprocessor& PP_;
};

bool IsNullAssignmentToGuard(const Stmt* Statement, const Expr* GuardExpression,
                             ASTContext& Context,
                             const SourceManager& SourceManager,
                             const LangOptions& LangOptions) {
  const auto* Expr = dyn_cast_or_null<clang::Expr>(Statement);
  if (!Expr) {
    return false;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(IgnoreExpressionNoise(Expr));
  if (!Binary || !Binary->isAssignmentOp() ||
      !IsNullPointerConstant(Binary->getRHS(), Context)) {
    return false;
  }
  std::optional<std::string> GuardText =
      SourceText(GuardExpression, SourceManager, LangOptions);
  std::optional<std::string> LhsText =
      SourceText(Binary->getLHS(), SourceManager, LangOptions);
  return GuardText && LhsText && *GuardText == *LhsText;
}

const CallExpr* ReleaseCallInSimpleBranch(const Stmt* Branch,
                                          const Expr* GuardExpression,
                                          ASTContext& Context,
                                          const SourceManager& SourceManager,
                                          const LangOptions& LangOptions) {
  if (const auto* Call = CallStatement(Branch)) {
    return IsPointerLikeSingleArgumentRelease(Call, SourceManager, LangOptions)
               ? Call
               : nullptr;
  }
  const auto* Compound = dyn_cast_or_null<CompoundStmt>(Branch);
  if (!Compound) {
    return nullptr;
  }
  const CallExpr* ReleaseCall = nullptr;
  unsigned StatementCount = 0;
  for (const Stmt* Child : Compound->body()) {
    if (isa<NullStmt>(Child)) {
      continue;
    }
    ++StatementCount;
    if (const auto* Call = CallStatement(Child);
        IsPointerLikeSingleArgumentRelease(Call, SourceManager, LangOptions)) {
      if (ReleaseCall) {
        return nullptr;
      }
      ReleaseCall = Call;
      continue;
    }
    if (IsNullAssignmentToGuard(Child, GuardExpression, Context, SourceManager,
                                LangOptions)) {
      continue;
    }
    return nullptr;
  }
  return StatementCount <= 2 ? ReleaseCall : nullptr;
}

std::optional<std::string> GuardedReleaseReplacementText(
    const IfStmt* If, const Stmt* Branch, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  if (CallStatement(Branch)) {
    std::optional<std::string> Replacement =
        StatementSourceText(Branch, SourceManager, LangOptions);
    if (!Replacement || Replacement->find('\n') != std::string::npos) {
      return std::nullopt;
    }
    return Replacement;
  }
  const auto* Compound = dyn_cast<CompoundStmt>(Branch);
  if (!Compound) {
    return std::nullopt;
  }
  SourceLocation IfLocation = If->getIfLoc();
  if (IfLocation.isInvalid() || IfLocation.isMacroID()) {
    return std::nullopt;
  }
  unsigned Column = SourceManager.getSpellingColumnNumber(IfLocation);
  std::string Indent(Column > 0 ? Column - 1 : 0, ' ');
  std::string Replacement;
  for (const Stmt* Child : Compound->body()) {
    if (isa<NullStmt>(Child)) {
      continue;
    }
    std::optional<std::string> ChildText =
        StatementSourceText(Child, SourceManager, LangOptions);
    if (!ChildText || ChildText->find('\n') != std::string::npos) {
      return std::nullopt;
    }
    if (!Replacement.empty()) {
      Replacement.append("\n");
      Replacement.append(Indent);
    }
    Replacement.append(*ChildText);
  }
  return Replacement.empty() ? std::nullopt
                             : std::optional<std::string>(Replacement);
}

std::optional<CharSourceRange> DesignatedInitializerFixRange(
    const DesignatedInitExpr* Init, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  if (!Init || Init->size() == 0 || Init->isDirectInit()) {
    return std::nullopt;
  }
  SourceLocation Begin = Init->getDesignator(0)->getBeginLoc();
  SourceLocation Equal = Init->getEqualOrColonLoc();
  if (Begin.isInvalid() || Equal.isInvalid()) {
    return std::nullopt;
  }
  if (Begin.isMacroID() || Equal.isMacroID() ||
      SourceManager.getFileID(Begin) != SourceManager.getFileID(Equal)) {
    return std::nullopt;
  }
  SourceLocation End =
      Lexer::getLocForEndOfToken(Equal, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  std::pair<FileID, unsigned> Decomposed = SourceManager.getDecomposedLoc(End);
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
  if (Invalid) {
    return std::nullopt;
  }
  while (
      Decomposed.second < Buffer.size() &&
      (Buffer[Decomposed.second] == ' ' || Buffer[Decomposed.second] == '\t')) {
    End = End.getLocWithOffset(1);
    ++Decomposed.second;
  }
  return CharSourceRange::getCharRange(Begin, End);
}

const InitListExpr* SourceInitializerList(const InitListExpr* Parent) {
  if (!Parent) {
    return nullptr;
  }
  return Parent->getSyntacticForm() ? Parent->getSyntacticForm() : Parent;
}

const FieldDecl* SingleFieldDesignator(const DesignatedInitExpr* Init) {
  if (!Init || Init->size() != 1) {
    return nullptr;
  }
  const DesignatedInitExpr::Designator* Designator = Init->getDesignator(0);
  if (!Designator || !Designator->isFieldDesignator()) {
    return nullptr;
  }
  return Designator->getFieldDecl();
}

std::optional<std::vector<const FieldDecl*>> PositionalFields(
    const RecordDecl* Record) {
  if (!Record) {
    return std::nullopt;
  }
  std::vector<const FieldDecl*> Fields;
  Fields.reserve(std::distance(Record->field_begin(), Record->field_end()));
  for (const FieldDecl* Field : Record->fields()) {
    if (Field->isUnnamedBitField()) {
      continue;
    }
    if (Field->getName().empty()) {
      return std::nullopt;
    }
    Fields.push_back(Field);
  }
  return Fields;
}

std::optional<unsigned> PositionalFieldIndex(const FieldDecl* Field) {
  if (!Field) {
    return std::nullopt;
  }
  unsigned FieldIndex = 0;
  for (const FieldDecl* Candidate : Field->getParent()->fields()) {
    if (Candidate->isUnnamedBitField()) {
      continue;
    }
    if (Candidate->getCanonicalDecl() == Field->getCanonicalDecl()) {
      return FieldIndex;
    }
    ++FieldIndex;
  }
  return std::nullopt;
}

std::optional<std::string> DesignatedInitializerReplacementText(
    const DesignatedInitExpr* Init, const InitListExpr* Parent,
    StringRef DesignatorText) {
  const InitListExpr* SourceParent = SourceInitializerList(Parent);
  if (!SourceParent || SourceParent->getNumInits() == 0) {
    return std::nullopt;
  }
  const RecordDecl* Record = nullptr;
  std::optional<std::vector<const FieldDecl*>> Fields;
  unsigned NextFieldIndex = 0;
  std::optional<std::string> Replacement;
  for (const Expr* Child : SourceParent->inits()) {
    const auto* DesignatedChild = dyn_cast_or_null<DesignatedInitExpr>(Child);
    const FieldDecl* Field = SingleFieldDesignator(DesignatedChild);
    if (!Field) {
      return std::nullopt;
    }
    if (!Record) {
      Record = Field->getParent();
      Fields = PositionalFields(Record);
      if (!Fields) {
        return std::nullopt;
      }
    } else if (Record->getCanonicalDecl() !=
               Field->getParent()->getCanonicalDecl()) {
      return std::nullopt;
    }
    std::optional<unsigned> FieldIndex = PositionalFieldIndex(Field);
    if (!FieldIndex || *FieldIndex < NextFieldIndex ||
        *FieldIndex >= Fields->size()) {
      return std::nullopt;
    }
    if (DesignatedChild == Init) {
      std::string Text;
      for (unsigned I = NextFieldIndex; I < *FieldIndex; ++I) {
        Text.append("/*.");
        Text.append((*Fields)[I]->getNameAsString());
        Text.append("=*/{}, ");
      }
      Text.append("/*");
      Text.append(DesignatorText);
      Text.append("*/");
      Replacement = std::move(Text);
    }
    NextFieldIndex = *FieldIndex + 1;
  }
  return Replacement;
}

}  // namespace

DirectGotoCheck::DirectGotoCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void DirectGotoCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(gotoStmt(unless(isExpansionInSystemHeader())).bind("goto"),
                     this);
}

void DirectGotoCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Goto = Result.Nodes.getNodeAs<GotoStmt>("goto");
  if (!Goto) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Goto->getGotoLoc(), SourceManager)) {
    return;
  }
  SourceLocation Location = SourceManager.getExpansionLoc(Goto->getGotoLoc());
  diag(Location,
       "direct goto is not allowed; split lifetime management from mutation "
       "logic or use structured control flow");
}

DesignatedInitializerCheck::DesignatedInitializerCheck(
    StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void DesignatedInitializerCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      initListExpr(
          forEach(designatedInitExpr(unless(isExpansionInSystemHeader()))
                      .bind("designated_init")))
          .bind("syntactic_init"),
      this);
}

void DesignatedInitializerCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  if (!Result.Context->getLangOpts().CPlusPlus) {
    return;
  }
  const auto* Init =
      Result.Nodes.getNodeAs<DesignatedInitExpr>("designated_init");
  const auto* ParentInit =
      Result.Nodes.getNodeAs<InitListExpr>("syntactic_init");
  if (!Init || !ParentInit || Init->size() == 0) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Init->getBeginLoc(), SourceManager)) {
    return;
  }
  SourceLocation DesignatorLocation = Init->getDesignator(0)->getBeginLoc();
  SourceLocation Location =
      SourceManager.isMacroBodyExpansion(DesignatorLocation)
          ? SourceManager.getSpellingLoc(DesignatorLocation)
          : SourceManager.getExpansionLoc(DesignatorLocation);
  DiagnosticBuilder Diagnostic =
      diag(Location,
           "C++ code must use comment field labels instead of designated "
           "initializers for MSVC portability");

  std::optional<CharSourceRange> ReplacementRange =
      DesignatedInitializerFixRange(Init, SourceManager,
                                    Result.Context->getLangOpts());
  if (!ReplacementRange) {
    return;
  }
  std::optional<std::string> DesignatorText = CompactSourceText(
      *ReplacementRange, SourceManager, Result.Context->getLangOpts());
  if (!DesignatorText || !StringRef(*DesignatorText).ends_with("=")) {
    return;
  }
  std::optional<std::string> ReplacementText =
      DesignatedInitializerReplacementText(Init, ParentInit, *DesignatorText);
  if (!ReplacementText) {
    return;
  }
  Diagnostic << FixItHint::CreateReplacement(*ReplacementRange,
                                             *ReplacementText);
}

GuardedReleaseCheck::GuardedReleaseCheck(StringRef Name,
                                         ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void GuardedReleaseCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      ifStmt(unless(hasElse(stmt())), hasCondition(expr().bind("condition")),
             hasThen(stmt().bind("then")))
          .bind("if"),
      this);
}

void GuardedReleaseCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto* Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto* Then = Result.Nodes.getNodeAs<Stmt>("then");
  if (!If || !Condition || !Then) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(If->getIfLoc(), SourceManager)) {
    return;
  }
  ASTContext& Context = *Result.Context;
  const Expr* GuardExpression = NonNullGuardedExpression(Condition, Context);
  if (!GuardExpression) {
    return;
  }
  const CallExpr* ReleaseCall =
      ReleaseCallInSimpleBranch(Then, GuardExpression, Context, SourceManager,
                                Result.Context->getLangOpts());
  if (!ReleaseCall) {
    return;
  }
  std::optional<std::string> GuardText =
      SourceText(GuardExpression, SourceManager, Result.Context->getLangOpts());
  std::optional<std::string> ReleaseArgText = SourceText(
      ReleaseCall->getArg(0), SourceManager, Result.Context->getLangOpts());
  if (!GuardText || !ReleaseArgText || *GuardText != *ReleaseArgText) {
    return;
  }
  DiagnosticBuilder Diagnostic =
      diag(SourceManager.getExpansionLoc(If->getIfLoc()),
           "release functions are null-safe; call %0 unconditionally");
  Diagnostic << ReleaseCall->getDirectCallee()->getName();
  std::optional<CharSourceRange> ReplacementRange =
      StatementCharRange(If, SourceManager, Result.Context->getLangOpts());
  std::optional<std::string> ReplacementText = GuardedReleaseReplacementText(
      If, Then, SourceManager, Result.Context->getLangOpts());
  if (ReplacementRange && ReplacementText) {
    Diagnostic << FixItHint::CreateReplacement(*ReplacementRange,
                                               *ReplacementText);
  }
}

TestStatusMacroCheck::TestStatusMacroCheck(StringRef Name,
                                           ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void TestStatusMacroCheck::registerPPCallbacks(const SourceManager&,
                                               Preprocessor* PP,
                                               Preprocessor*) {
  if (!PP) {
    return;
  }
  PP->addPPCallbacks(std::make_unique<TestStatusMacroRecorder>(*this, *PP));
}

TestStatusPredicateCheck::TestStatusPredicateCheck(StringRef Name,
                                                   ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void TestStatusPredicateCheck::registerPPCallbacks(const SourceManager&,
                                                   Preprocessor* PP,
                                                   Preprocessor*) {
  if (!PP) {
    return;
  }
  PP->addPPCallbacks(std::make_unique<TestStatusPredicateRecorder>(*this, *PP));
}

}  // namespace clang::tidy::iree
