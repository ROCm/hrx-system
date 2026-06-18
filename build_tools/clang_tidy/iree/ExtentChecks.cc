// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/ExtentChecks.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

struct ExtentValueContract {
  const char* TypeName;
  const char* EmptyFunction;
  const char* IsEmptyFunction;
  const char* PointerField;
  const char* ExtentField;
};

constexpr ExtentValueContract kExtentValueContracts[] = {
    {"iree_byte_span_t", "iree_byte_span_empty", "iree_byte_span_is_empty",
     "data", "data_length"},
    {"iree_const_byte_span_t", "iree_const_byte_span_empty",
     "iree_const_byte_span_is_empty", "data", "data_length"},
    {"iree_string_view_t", "iree_string_view_empty",
     "iree_string_view_is_empty", "data", "size"},
    {"iree_mutable_string_view_t", "iree_mutable_string_view_empty",
     "iree_mutable_string_view_is_empty", "data", "size"},
    {"iree_string_pair_list_t", "iree_string_pair_list_empty", nullptr, nullptr,
     "count"},
    {"iree_string_view_list_t", "iree_string_view_list_empty", nullptr, nullptr,
     "count"},
    {"iree_async_span_t", "iree_async_span_empty", "iree_async_span_is_empty",
     nullptr, "length"},
    {"iree_async_span_list_t", "iree_async_span_list_empty",
     "iree_async_span_list_is_empty", nullptr, "count"},
    {"iree_async_continuation_list_t", "iree_async_continuation_list_empty",
     "iree_async_continuation_list_is_empty", nullptr, "count"},
    {"iree_async_operation_list_t", "iree_async_operation_list_empty",
     "iree_async_operation_list_is_empty", nullptr, "count"},
    {"iree_hal_buffer_ref_list_t", "iree_hal_buffer_ref_list_empty", nullptr,
     nullptr, "count"},
    {"iree_hal_buffer_binding_table_t", "iree_hal_buffer_binding_table_empty",
     "iree_hal_buffer_binding_table_is_empty", nullptr, "count"},
    {"iree_hal_semaphore_list_t", "iree_hal_semaphore_list_empty",
     "iree_hal_semaphore_list_is_empty", nullptr, "count"},
};

const Expr* IgnoreExpressionNoise(const Expr* Expr) {
  if (!Expr) {
    return nullptr;
  }
  return Expr->IgnoreParenImpCasts();
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

std::optional<std::string> SourceText(const Expr* Expr,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  if (!Expr) {
    return std::nullopt;
  }
  SourceLocation Begin = Expr->getBeginLoc();
  SourceLocation End = Expr->getEndLoc();
  if (Begin.isInvalid() || End.isInvalid() || Begin.isMacroID() ||
      End.isMacroID()) {
    return std::nullopt;
  }
  End = Lexer::getLocForEndOfToken(End, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  return SourceText(CharSourceRange::getCharRange(Begin, End), SourceManager,
                    LangOptions);
}

std::optional<std::string> CompactSourceText(const Expr* Expr,
                                             const SourceManager& SourceManager,
                                             const LangOptions& LangOptions) {
  std::optional<std::string> Text =
      SourceText(Expr, SourceManager, LangOptions);
  if (!Text) {
    return std::nullopt;
  }
  Text->erase(std::remove_if(Text->begin(), Text->end(),
                             [](unsigned char c) { return std::isspace(c); }),
              Text->end());
  return Text;
}

std::optional<CharSourceRange> ExprCharRange(const Expr* Expr,
                                             const SourceManager& SourceManager,
                                             const LangOptions& LangOptions) {
  if (!Expr) {
    return std::nullopt;
  }
  SourceLocation Begin = Expr->getBeginLoc();
  SourceLocation End = Expr->getEndLoc();
  if (Begin.isInvalid() || End.isInvalid() || Begin.isMacroID() ||
      End.isMacroID()) {
    return std::nullopt;
  }
  End = Lexer::getLocForEndOfToken(End, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  if (SourceManager.getFileID(Begin) != SourceManager.getFileID(End)) {
    return std::nullopt;
  }
  return CharSourceRange::getCharRange(Begin, End);
}

std::optional<StringRef> TypedefName(QualType Qual) {
  Qual = Qual.getUnqualifiedType();
  const Type* TypePtr = Qual.getTypePtrOrNull();
  if (!TypePtr) {
    return std::nullopt;
  }
  const auto* Typedef = TypePtr->getAs<TypedefType>();
  return Typedef ? std::optional<StringRef>(Typedef->getDecl()->getName())
                 : std::nullopt;
}

const ExtentValueContract* ContractForValueType(QualType Qual) {
  std::optional<StringRef> Name = TypedefName(Qual);
  if (!Name) {
    return nullptr;
  }
  for (const ExtentValueContract& Contract : kExtentValueContracts) {
    if (*Name == Contract.TypeName) {
      return &Contract;
    }
  }
  return nullptr;
}

const InitListExpr* SourceInitializerList(const Expr* Init) {
  const auto* List =
      dyn_cast_or_null<InitListExpr>(IgnoreExpressionNoise(Init));
  if (!List) {
    return nullptr;
  }
  return List->getSyntacticForm() ? List->getSyntacticForm() : List;
}

bool IsZeroOrNullConstant(const Expr* Expr, ASTContext& Context) {
  Expr = IgnoreExpressionNoise(Expr);
  if (!Expr) {
    return false;
  }
  if (Expr->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull)) {
    return true;
  }
  const auto* Integer = dyn_cast<IntegerLiteral>(Expr);
  return Integer && Integer->getValue().isZero();
}

bool IsAggregateZeroInitializer(const InitListExpr* Init, ASTContext& Context) {
  if (!Init) {
    return false;
  }
  if (Init->getNumInits() == 0) {
    return true;
  }
  for (const Expr* Child : Init->inits()) {
    if (!IsZeroOrNullConstant(Child, Context)) {
      return false;
    }
  }
  return true;
}

bool IsInsideOwnEmptyFunction(const VarDecl* Var,
                              const ExtentValueContract& Contract) {
  const auto* Function = dyn_cast_or_null<FunctionDecl>(Var->getDeclContext());
  return Function && Function->getName() == Contract.EmptyFunction;
}

bool IsZeroIntegerConstant(const Expr* Expr) {
  Expr = IgnoreExpressionNoise(Expr);
  const auto* Integer = dyn_cast_or_null<IntegerLiteral>(Expr);
  return Integer && Integer->getValue().isZero();
}

struct ExtentMemberUse {
  const ExtentValueContract* Contract = nullptr;
  const Expr* Base = nullptr;
  std::string BaseText;
  std::string CompactBaseText;
};

std::optional<ExtentMemberUse> ExtentMemberUseForField(
    const Expr* Input, StringRef FieldName, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  Input = IgnoreExpressionNoise(Input);
  const auto* Member = dyn_cast_or_null<MemberExpr>(Input);
  if (!Member || Member->isArrow()) {
    return std::nullopt;
  }
  const auto* Field = dyn_cast_or_null<FieldDecl>(Member->getMemberDecl());
  if (!Field || Field->getName() != FieldName) {
    return std::nullopt;
  }
  const Expr* Base = Member->getBase();
  const ExtentValueContract* Contract = ContractForValueType(Base->getType());
  if (!Contract || !Contract->IsEmptyFunction) {
    return std::nullopt;
  }
  std::optional<std::string> BaseText =
      SourceText(Base, SourceManager, LangOptions);
  std::optional<std::string> CompactBaseText =
      CompactSourceText(Base, SourceManager, LangOptions);
  if (!BaseText || !CompactBaseText) {
    return std::nullopt;
  }
  return ExtentMemberUse{Contract, Base, *BaseText, *CompactBaseText};
}

std::optional<ExtentMemberUse> PointerNullPredicate(
    const Expr* Input, ASTContext& Context, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  Input = IgnoreExpressionNoise(Input);
  if (!Input) {
    return std::nullopt;
  }
  if (const auto* Unary = dyn_cast<UnaryOperator>(Input);
      Unary && Unary->getOpcode() == UO_LNot) {
    for (const ExtentValueContract& Contract : kExtentValueContracts) {
      if (!Contract.PointerField) {
        continue;
      }
      std::optional<ExtentMemberUse> Use =
          ExtentMemberUseForField(Unary->getSubExpr(), Contract.PointerField,
                                  SourceManager, LangOptions);
      if (Use && Use->Contract == &Contract) {
        return Use;
      }
    }
    return std::nullopt;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(Input);
  if (!Binary || Binary->getOpcode() != BO_EQ) {
    return std::nullopt;
  }
  const Expr* Lhs = IgnoreExpressionNoise(Binary->getLHS());
  const Expr* Rhs = IgnoreExpressionNoise(Binary->getRHS());
  const Expr* Candidate = nullptr;
  if (Lhs &&
      Lhs->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull)) {
    Candidate = Rhs;
  } else if (Rhs && Rhs->isNullPointerConstant(
                        Context, Expr::NPC_ValueDependentIsNotNull)) {
    Candidate = Lhs;
  }
  if (!Candidate) {
    return std::nullopt;
  }
  for (const ExtentValueContract& Contract : kExtentValueContracts) {
    if (!Contract.PointerField) {
      continue;
    }
    std::optional<ExtentMemberUse> Use = ExtentMemberUseForField(
        Candidate, Contract.PointerField, SourceManager, LangOptions);
    if (Use && Use->Contract == &Contract) {
      return Use;
    }
  }
  return std::nullopt;
}

std::optional<ExtentMemberUse> ExtentZeroPredicate(
    const Expr* Input, ASTContext& Context, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  Input = IgnoreExpressionNoise(Input);
  if (!Input) {
    return std::nullopt;
  }
  if (const auto* Unary = dyn_cast<UnaryOperator>(Input);
      Unary && Unary->getOpcode() == UO_LNot) {
    for (const ExtentValueContract& Contract : kExtentValueContracts) {
      std::optional<ExtentMemberUse> Use =
          ExtentMemberUseForField(Unary->getSubExpr(), Contract.ExtentField,
                                  SourceManager, LangOptions);
      if (Use && Use->Contract == &Contract) {
        return Use;
      }
    }
    return std::nullopt;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(Input);
  if (!Binary || Binary->getOpcode() != BO_EQ) {
    return std::nullopt;
  }
  const Expr* Lhs = IgnoreExpressionNoise(Binary->getLHS());
  const Expr* Rhs = IgnoreExpressionNoise(Binary->getRHS());
  const Expr* Candidate = nullptr;
  if (IsZeroIntegerConstant(Lhs)) {
    Candidate = Rhs;
  } else if (IsZeroIntegerConstant(Rhs)) {
    Candidate = Lhs;
  }
  if (!Candidate) {
    return std::nullopt;
  }
  for (const ExtentValueContract& Contract : kExtentValueContracts) {
    std::optional<ExtentMemberUse> Use = ExtentMemberUseForField(
        Candidate, Contract.ExtentField, SourceManager, LangOptions);
    if (Use && Use->Contract == &Contract) {
      return Use;
    }
  }
  return std::nullopt;
}

bool SameExtentBase(const ExtentMemberUse& Lhs, const ExtentMemberUse& Rhs) {
  return Lhs.Contract == Rhs.Contract &&
         Lhs.CompactBaseText == Rhs.CompactBaseText;
}

bool IsEmptyHelperFunction(const FunctionDecl* Function) {
  return Function && Function->getName().ends_with("_is_empty");
}

}  // namespace

ExtentEmptyInitializerCheck::ExtentEmptyInitializerCheck(
    StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void ExtentEmptyInitializerCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(varDecl(unless(isExpansionInSystemHeader()),
                             hasInitializer(initListExpr().bind("init")))
                         .bind("var"),
                     this);
}

void ExtentEmptyInitializerCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Var = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto* Init = Result.Nodes.getNodeAs<InitListExpr>("init");
  if (!Var || !Init || !Var->hasLocalStorage()) {
    return;
  }
  const ExtentValueContract* Contract = ContractForValueType(Var->getType());
  if (!Contract || IsInsideOwnEmptyFunction(Var, *Contract) ||
      !IsAggregateZeroInitializer(SourceInitializerList(Init),
                                  *Result.Context)) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (Init->getBeginLoc().isMacroID()) {
    return;
  }
  DiagnosticBuilder Diagnostic =
      diag(SourceManager.getExpansionLoc(Init->getBeginLoc()),
           "use %0() instead of aggregate zero initialization for empty %1 "
           "values")
      << Contract->EmptyFunction << Contract->TypeName;
  std::optional<CharSourceRange> ReplacementRange =
      ExprCharRange(Init, SourceManager, Result.Context->getLangOpts());
  if (ReplacementRange) {
    Diagnostic << FixItHint::CreateReplacement(
        *ReplacementRange, (Twine(Contract->EmptyFunction) + "()").str());
  }
}

ExtentEmptyPredicateCheck::ExtentEmptyPredicateCheck(StringRef Name,
                                                     ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void ExtentEmptyPredicateCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      functionDecl(
          forEachDescendant(binaryOperator(unless(isExpansionInSystemHeader()),
                                           anyOf(hasOperatorName("||"),
                                                 hasOperatorName("&&")))
                                .bind("binary")))
          .bind("function"),
      this);
}

void ExtentEmptyPredicateCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Binary = Result.Nodes.getNodeAs<BinaryOperator>("binary");
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Binary) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (Binary->getOperatorLoc().isMacroID()) {
    return;
  }
  ASTContext& Context = *Result.Context;
  const LangOptions& LangOptions = Result.Context->getLangOpts();
  std::optional<ExtentMemberUse> LhsPointer = PointerNullPredicate(
      Binary->getLHS(), Context, SourceManager, LangOptions);
  std::optional<ExtentMemberUse> RhsPointer = PointerNullPredicate(
      Binary->getRHS(), Context, SourceManager, LangOptions);
  std::optional<ExtentMemberUse> LhsExtent = ExtentZeroPredicate(
      Binary->getLHS(), Context, SourceManager, LangOptions);
  std::optional<ExtentMemberUse> RhsExtent = ExtentZeroPredicate(
      Binary->getRHS(), Context, SourceManager, LangOptions);

  std::optional<ExtentMemberUse> PointerUse;
  if (LhsPointer && RhsExtent && SameExtentBase(*LhsPointer, *RhsExtent)) {
    PointerUse = LhsPointer;
  } else if (RhsPointer && LhsExtent &&
             SameExtentBase(*RhsPointer, *LhsExtent)) {
    PointerUse = RhsPointer;
  } else {
    return;
  }
  if (Binary->getOpcode() == BO_LOr && !IsEmptyHelperFunction(Function)) {
    return;
  }

  DiagnosticBuilder Diagnostic =
      diag(SourceManager.getExpansionLoc(Binary->getOperatorLoc()),
           "test empty %0 values with %1(); pointer-null is malformedness, "
           "not emptiness")
      << PointerUse->Contract->TypeName
      << PointerUse->Contract->IsEmptyFunction;
  std::optional<CharSourceRange> ReplacementRange =
      ExprCharRange(Binary, SourceManager, LangOptions);
  if (ReplacementRange) {
    Diagnostic << FixItHint::CreateReplacement(
        *ReplacementRange, (Twine(PointerUse->Contract->IsEmptyFunction) + "(" +
                            PointerUse->BaseText + ")")
                               .str());
  }
}

}  // namespace clang::tidy::iree
