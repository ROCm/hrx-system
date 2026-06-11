// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/RefCountChecks.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
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

std::optional<std::string> FunctionBodySourceText(
    const FunctionDecl* Function, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  const Stmt* Body = Function->getBody();
  if (!Body) {
    return std::nullopt;
  }
  SourceRange BodyRange = Body->getSourceRange();
  if (BodyRange.getBegin().isInvalid() || BodyRange.getEnd().isInvalid()) {
    return std::nullopt;
  }
  return SourceText(CharSourceRange::getTokenRange(BodyRange), SourceManager,
                    LangOptions);
}

bool IsIdentifierCharacter(char Character) {
  return std::isalnum(static_cast<unsigned char>(Character)) ||
         Character == '_';
}

bool SourceContainsIdentifier(StringRef Source, StringRef Identifier) {
  enum class State {
    kCode,
    kLineComment,
    kBlockComment,
    kString,
    kCharacter,
  };
  State ParserState = State::kCode;
  for (size_t I = 0; I < Source.size(); ++I) {
    char Character = Source[I];
    char Next = I + 1 < Source.size() ? Source[I + 1] : '\0';
    switch (ParserState) {
      case State::kCode:
        if (Character == '/' && Next == '/') {
          ParserState = State::kLineComment;
          ++I;
        } else if (Character == '/' && Next == '*') {
          ParserState = State::kBlockComment;
          ++I;
        } else if (Character == '"') {
          ParserState = State::kString;
        } else if (Character == '\'') {
          ParserState = State::kCharacter;
        } else if (Source.substr(I).starts_with(Identifier) &&
                   (I == 0 || !IsIdentifierCharacter(Source[I - 1])) &&
                   (I + Identifier.size() == Source.size() ||
                    !IsIdentifierCharacter(Source[I + Identifier.size()]))) {
          return true;
        }
        break;
      case State::kLineComment:
        if (Character == '\n' || Character == '\r') {
          ParserState = State::kCode;
        }
        break;
      case State::kBlockComment:
        if (Character == '*' && Next == '/') {
          ParserState = State::kCode;
          ++I;
        }
        break;
      case State::kString:
      case State::kCharacter:
        if (Character == '\\' && Next != '\0') {
          ++I;
        } else if ((ParserState == State::kString && Character == '"') ||
                   (ParserState == State::kCharacter && Character == '\'')) {
          ParserState = State::kCode;
        }
        break;
    }
  }
  return false;
}

std::string NormalizedCodeSource(StringRef Source) {
  enum class State {
    kCode,
    kLineComment,
    kBlockComment,
    kString,
    kCharacter,
  };
  State ParserState = State::kCode;
  std::string Result;
  for (size_t I = 0; I < Source.size(); ++I) {
    char Character = Source[I];
    char Next = I + 1 < Source.size() ? Source[I + 1] : '\0';
    switch (ParserState) {
      case State::kCode:
        if (Character == '/' && Next == '/') {
          ParserState = State::kLineComment;
          ++I;
        } else if (Character == '/' && Next == '*') {
          ParserState = State::kBlockComment;
          ++I;
        } else if (Character == '"') {
          ParserState = State::kString;
        } else if (Character == '\'') {
          ParserState = State::kCharacter;
        } else if (!std::isspace(static_cast<unsigned char>(Character))) {
          Result.push_back(Character);
        }
        break;
      case State::kLineComment:
        if (Character == '\n' || Character == '\r') {
          ParserState = State::kCode;
        }
        break;
      case State::kBlockComment:
        if (Character == '*' && Next == '/') {
          ParserState = State::kCode;
          ++I;
        }
        break;
      case State::kString:
      case State::kCharacter:
        if (Character == '\\' && Next != '\0') {
          ++I;
        } else if ((ParserState == State::kString && Character == '"') ||
                   (ParserState == State::kCharacter && Character == '\'')) {
          ParserState = State::kCode;
        }
        break;
    }
  }
  return Result;
}

bool IsRefCountLifecycleFunctionName(StringRef FunctionName) {
  return FunctionName.ends_with("_retain") ||
         FunctionName.ends_with("_release");
}

bool IsReleaseFunctionName(StringRef FunctionName) {
  return FunctionName.ends_with("_release");
}

bool IsRetainFunctionName(StringRef FunctionName) {
  return FunctionName.ends_with("_retain");
}

bool IsNamedTypedef(QualType QualType, StringRef Name) {
  QualType = QualType.getUnqualifiedType();
  const Type* TypePtr = QualType.getTypePtrOrNull();
  if (!TypePtr) {
    return false;
  }
  if (const auto* Typedef = TypePtr->getAs<TypedefType>()) {
    return Typedef->getDecl()->getName() == Name;
  }
  if (const auto* Attributed = dyn_cast<AttributedType>(TypePtr)) {
    return IsNamedTypedef(Attributed->getModifiedType(), Name);
  }
  return false;
}

bool IsRefCountField(const FieldDecl* Field) {
  return Field && IsNamedTypedef(Field->getType(), "iree_atomic_ref_count_t");
}

const RecordDecl* EnclosingRecord(const FieldDecl* Field) {
  return dyn_cast_or_null<RecordDecl>(Field->getDeclContext());
}

const FieldDecl* FirstField(const RecordDecl* Record) {
  if (!Record) {
    return nullptr;
  }
  for (const FieldDecl* Field : Record->fields()) {
    return Field;
  }
  return nullptr;
}

bool IsAllowedRefCountFieldName(const FieldDecl* Field,
                                const SourceManager& SourceManager) {
  if (Field->getName() == "ref_count") {
    return true;
  }

  SourceLocation Location = SourceManager.getExpansionLoc(Field->getLocation());
  StringRef Filename = SourceManager.getFilename(Location);
  return Field->getName() == "counter" &&
         Filename.ends_with("runtime/src/iree/vm/ref.h");
}

bool IsRefCountAnchorField(const FieldDecl* Field,
                           const SourceManager& SourceManager) {
  return IsRefCountField(Field) &&
         Field == FirstField(EnclosingRecord(Field)) &&
         IsAllowedRefCountFieldName(Field, SourceManager);
}

bool IsRefCountAnchoredRecord(const RecordDecl* Record,
                              const SourceManager& SourceManager) {
  return IsRefCountAnchorField(FirstField(Record), SourceManager);
}

enum class RefCountFieldDiagnostic {
  kNone,
  kMisusedCounter,
  kRefCountNotFirst,
};

RefCountFieldDiagnostic DiagnoseRefCountField(
    const FieldDecl* Field, const SourceManager& SourceManager) {
  if (!IsRefCountField(Field)) {
    return RefCountFieldDiagnostic::kNone;
  }
  const RecordDecl* Record = EnclosingRecord(Field);
  if (Field == FirstField(Record)) {
    return IsAllowedRefCountFieldName(Field, SourceManager)
               ? RefCountFieldDiagnostic::kNone
               : RefCountFieldDiagnostic::kMisusedCounter;
  }
  if (Field->getName() == "ref_count") {
    return RefCountFieldDiagnostic::kRefCountNotFirst;
  }
  return IsRefCountAnchoredRecord(Record, SourceManager)
             ? RefCountFieldDiagnostic::kMisusedCounter
             : RefCountFieldDiagnostic::kNone;
}

bool BodyContainsRefCountMutation(StringRef BodyText) {
  return SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_inc") ||
         SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_dec");
}

bool HasIgnoredRefCountDecrement(StringRef NormalizedBody) {
  static constexpr StringRef RefCountDec = "iree_atomic_ref_count_dec";
  size_t SearchPosition = 0;
  while (true) {
    size_t DecPosition = NormalizedBody.find(RefCountDec, SearchPosition);
    if (DecPosition == StringRef::npos) {
      return false;
    }
    bool IsStatementStart = DecPosition == 0 ||
                            NormalizedBody[DecPosition - 1] == '{' ||
                            NormalizedBody[DecPosition - 1] == ';';
    static constexpr StringRef VoidCast = "(void)";
    if (!IsStatementStart && DecPosition >= VoidCast.size()) {
      size_t CastPosition = DecPosition - VoidCast.size();
      IsStatementStart =
          NormalizedBody.substr(CastPosition, VoidCast.size()) == VoidCast &&
          (CastPosition == 0 || NormalizedBody[CastPosition - 1] == '{' ||
           NormalizedBody[CastPosition - 1] == ';');
    }
    if (IsStatementStart) {
      return true;
    }
    SearchPosition = DecPosition + RefCountDec.size();
  }
}

bool RefCountMutationReferencesParameter(StringRef NormalizedBody,
                                         StringRef MutationName,
                                         StringRef ParameterName) {
  size_t SearchPosition = 0;
  while (true) {
    size_t MutationPosition = NormalizedBody.find(MutationName, SearchPosition);
    if (MutationPosition == StringRef::npos) {
      return false;
    }
    size_t StatementEnd = NormalizedBody.find(';', MutationPosition);
    if (StatementEnd == StringRef::npos) {
      StatementEnd = NormalizedBody.size();
    }
    if (SourceContainsIdentifier(
            NormalizedBody.substr(MutationPosition,
                                  StatementEnd - MutationPosition),
            ParameterName)) {
      return true;
    }
    SearchPosition = MutationPosition + MutationName.size();
  }
}

bool HasEarlyNullReturn(StringRef NormalizedBody, StringRef ParameterName) {
  std::string Parameter = ParameterName.str();
  for (std::string Pattern :
       {"if(!" + Parameter + ")return;", "if(!" + Parameter + "){return;}",
        "if(" + Parameter + "==NULL)return;",
        "if(" + Parameter + "==NULL){return;}",
        "if(NULL==" + Parameter + ")return;",
        "if(NULL==" + Parameter + "){return;}",
        "if(IREE_UNLIKELY(!" + Parameter + "))return;",
        "if(IREE_UNLIKELY(!" + Parameter + ")){return;}"}) {
    if (NormalizedBody.contains(StringRef(Pattern))) {
      return true;
    }
  }
  return false;
}

bool HasGuardedRefCountDecrement(StringRef NormalizedBody,
                                 StringRef ParameterName) {
  static constexpr StringRef RefCountDec = "iree_atomic_ref_count_dec";
  size_t SearchPosition = 0;
  while (true) {
    size_t DecPosition = NormalizedBody.find(RefCountDec, SearchPosition);
    if (DecPosition == StringRef::npos) {
      return false;
    }
    StringRef Prefix = NormalizedBody.substr(0, DecPosition);
    size_t IfPosition = Prefix.rfind("if(");
    size_t StatementPosition = NormalizedBody.rfind(';', DecPosition);
    size_t BlockPosition = NormalizedBody.rfind('{', DecPosition);
    size_t Boundary =
        std::max(StatementPosition == StringRef::npos ? 0 : StatementPosition,
                 BlockPosition == StringRef::npos ? 0 : BlockPosition);
    if (IfPosition != StringRef::npos && IfPosition >= Boundary) {
      StringRef ConditionPrefix =
          NormalizedBody.substr(IfPosition + 3, DecPosition - IfPosition - 3);
      if (ConditionPrefix.contains("&&") &&
          SourceContainsIdentifier(ConditionPrefix, ParameterName)) {
        return true;
      }
    }
    SearchPosition = DecPosition + RefCountDec.size();
  }
}

bool HasArgumentAssert(StringRef NormalizedBody, StringRef ParameterName) {
  std::string Parameter = ParameterName.str();
  for (std::string Pattern : {"IREE_ASSERT_ARGUMENT(" + Parameter + ");",
                              "IREE_ASSERT(" + Parameter + ");"}) {
    if (NormalizedBody.contains(StringRef(Pattern))) {
      return true;
    }
  }
  return false;
}

bool IsInsideOpenBlock(StringRef NormalizedBody, size_t BlockPosition,
                       size_t TargetPosition) {
  int Depth = 0;
  for (size_t I = BlockPosition; I < TargetPosition; ++I) {
    if (NormalizedBody[I] == '{') {
      ++Depth;
    } else if (NormalizedBody[I] == '}') {
      --Depth;
    }
  }
  return Depth > 0;
}

bool HasGuardedRefCountIncrement(StringRef NormalizedBody,
                                 StringRef ParameterName) {
  static constexpr StringRef RefCountInc = "iree_atomic_ref_count_inc";
  size_t SearchPosition = 0;
  while (true) {
    size_t IncPosition = NormalizedBody.find(RefCountInc, SearchPosition);
    if (IncPosition == StringRef::npos) {
      return false;
    }
    StringRef Prefix = NormalizedBody.substr(0, IncPosition);
    size_t IfPosition = Prefix.rfind("if(");
    if (IfPosition != StringRef::npos) {
      size_t BlockPosition = NormalizedBody.find('{', IfPosition);
      bool IncrementIsInsideThenBlock =
          BlockPosition != StringRef::npos && BlockPosition < IncPosition &&
          IsInsideOpenBlock(NormalizedBody, BlockPosition, IncPosition);
      if (IncrementIsInsideThenBlock) {
        StringRef Condition = NormalizedBody.substr(
            IfPosition + 3, BlockPosition - IfPosition - 3);
        if (SourceContainsIdentifier(Condition, ParameterName)) {
          return true;
        }
      }
    }
    SearchPosition = IncPosition + RefCountInc.size();
  }
}

bool IsRefCountReleaseNullSafe(const FunctionDecl* Function,
                               StringRef NormalizedBody) {
  if (Function->getNumParams() != 1) {
    return true;
  }
  const ParmVarDecl* Parameter = Function->getParamDecl(0);
  if (!Parameter->getType()->isAnyPointerType()) {
    return true;
  }
  StringRef ParameterName = Parameter->getName();
  if (ParameterName.empty() ||
      !RefCountMutationReferencesParameter(
          NormalizedBody, "iree_atomic_ref_count_dec", ParameterName)) {
    return true;
  }
  if (HasArgumentAssert(NormalizedBody, ParameterName)) {
    return false;
  }
  return HasEarlyNullReturn(NormalizedBody, ParameterName) ||
         HasGuardedRefCountDecrement(NormalizedBody, ParameterName);
}

bool IsRefCountRetainNullSafe(const FunctionDecl* Function,
                              StringRef NormalizedBody) {
  if (Function->getNumParams() != 1) {
    return true;
  }
  const ParmVarDecl* Parameter = Function->getParamDecl(0);
  if (!Parameter->getType()->isAnyPointerType()) {
    return true;
  }
  StringRef ParameterName = Parameter->getName();
  if (ParameterName.empty() ||
      !RefCountMutationReferencesParameter(
          NormalizedBody, "iree_atomic_ref_count_inc", ParameterName)) {
    return true;
  }
  if (HasArgumentAssert(NormalizedBody, ParameterName)) {
    return false;
  }
  return HasEarlyNullReturn(NormalizedBody, ParameterName) ||
         HasGuardedRefCountIncrement(NormalizedBody, ParameterName);
}

const Expr* IgnoreExprNoise(const Expr* Expression) {
  return Expression ? Expression->IgnoreParenCasts() : nullptr;
}

const VarDecl* ReferencedVariable(const Expr* Expression) {
  Expression = IgnoreExprNoise(Expression);
  const auto* Reference = dyn_cast_or_null<DeclRefExpr>(Expression);
  if (!Reference) {
    return nullptr;
  }
  return dyn_cast<VarDecl>(Reference->getDecl());
}

const VarDecl* ReferencedVariableThroughIndirection(const Expr* Expression) {
  Expression = IgnoreExprNoise(Expression);
  if (const VarDecl* Variable = ReferencedVariable(Expression)) {
    return Variable;
  }
  const auto* Unary = dyn_cast_or_null<UnaryOperator>(Expression);
  if (Unary && Unary->getOpcode() == UO_Deref) {
    return ReferencedVariable(Unary->getSubExpr());
  }
  return nullptr;
}

bool IsNullPointerConstant(const Expr* Expression, ASTContext& Context) {
  Expression = IgnoreExprNoise(Expression);
  return Expression && Expression->isNullPointerConstant(
                           Context, Expr::NPC_ValueDependentIsNotNull);
}

const VarDecl* PositivePointerGuardVariable(const Expr* Condition,
                                            ASTContext& Context) {
  Condition = IgnoreExprNoise(Condition);
  if (const VarDecl* Variable = ReferencedVariable(Condition)) {
    return Variable->getType()->isAnyPointerType() ? Variable : nullptr;
  }

  const auto* Binary = dyn_cast_or_null<BinaryOperator>(Condition);
  if (!Binary || Binary->getOpcode() != BO_NE) {
    return nullptr;
  }
  const VarDecl* LeftVariable = ReferencedVariable(Binary->getLHS());
  const VarDecl* RightVariable = ReferencedVariable(Binary->getRHS());
  if (LeftVariable && !RightVariable &&
      IsNullPointerConstant(Binary->getRHS(), Context)) {
    return LeftVariable->getType()->isAnyPointerType() ? LeftVariable : nullptr;
  }
  if (RightVariable && !LeftVariable &&
      IsNullPointerConstant(Binary->getLHS(), Context)) {
    return RightVariable->getType()->isAnyPointerType() ? RightVariable
                                                        : nullptr;
  }
  return nullptr;
}

const Stmt* SingleStatementBody(const Stmt* Body) {
  if (!Body) {
    return nullptr;
  }
  const auto* Compound = dyn_cast<CompoundStmt>(Body);
  if (!Compound) {
    return Body;
  }
  const Stmt* OnlyStatement = nullptr;
  for (const Stmt* Statement : Compound->body()) {
    if (!Statement) {
      continue;
    }
    if (OnlyStatement) {
      return nullptr;
    }
    OnlyStatement = Statement;
  }
  return OnlyStatement;
}

struct DirectRefCountOperation {
  const VarDecl* Variable = nullptr;
  const FunctionDecl* Function = nullptr;
  SourceLocation Location;
};

std::optional<DirectRefCountOperation> DirectRefCountOperationStatement(
    const Stmt* Statement,
    bool (*FunctionNamePredicate)(llvm::StringRef FunctionName)) {
  if (!Statement) {
    return std::nullopt;
  }
  const auto* Expression =
      dyn_cast<Expr>(Statement->IgnoreContainers(/*IgnoreCaptured=*/true));
  if (!Expression) {
    return std::nullopt;
  }
  Expression = IgnoreExprNoise(Expression);
  const auto* Call = dyn_cast_or_null<CallExpr>(Expression);
  if (!Call || Call->getNumArgs() != 1) {
    return std::nullopt;
  }
  const FunctionDecl* Callee = Call->getDirectCallee();
  if (!Callee || !Callee->getReturnType()->isVoidType() ||
      !FunctionNamePredicate(Callee->getName())) {
    return std::nullopt;
  }
  const VarDecl* Variable = ReferencedVariable(Call->getArg(0));
  if (!Variable || !Variable->getType()->isAnyPointerType()) {
    return std::nullopt;
  }
  return DirectRefCountOperation{
      .Variable = Variable,
      .Function = Callee,
      .Location = Call->getBeginLoc(),
  };
}

std::optional<DirectRefCountOperation> DirectReleaseStatement(
    const Stmt* Statement) {
  return DirectRefCountOperationStatement(Statement, IsReleaseFunctionName);
}

std::optional<DirectRefCountOperation> DirectRetainStatement(
    const Stmt* Statement) {
  return DirectRefCountOperationStatement(Statement, IsRetainFunctionName);
}

std::optional<DirectRefCountOperation> GuardedReleaseStatement(
    const IfStmt* Statement, ASTContext& Context) {
  if (!Statement || Statement->getElse()) {
    return std::nullopt;
  }
  const VarDecl* GuardedVariable =
      PositivePointerGuardVariable(Statement->getCond(), Context);
  if (!GuardedVariable) {
    return std::nullopt;
  }
  std::optional<DirectRefCountOperation> Release =
      DirectReleaseStatement(SingleStatementBody(Statement->getThen()));
  if (!Release || Release->Variable != GuardedVariable) {
    return std::nullopt;
  }
  return Release;
}

std::optional<DirectRefCountOperation> IfElseMergedReleaseStatement(
    const IfStmt* Statement) {
  if (!Statement || !Statement->getElse()) {
    return std::nullopt;
  }
  std::optional<DirectRefCountOperation> ThenRelease =
      DirectReleaseStatement(SingleStatementBody(Statement->getThen()));
  std::optional<DirectRefCountOperation> ElseRelease =
      DirectReleaseStatement(SingleStatementBody(Statement->getElse()));
  if (!ThenRelease || !ElseRelease ||
      ThenRelease->Variable != ElseRelease->Variable ||
      ThenRelease->Function != ElseRelease->Function) {
    return std::nullopt;
  }
  return DirectRefCountOperation{
      .Variable = ThenRelease->Variable,
      .Function = ThenRelease->Function,
      .Location = Statement->getIfLoc(),
  };
}

const VarDecl* DirectVariableAssignment(const Stmt* Statement) {
  if (!Statement) {
    return nullptr;
  }
  const auto* Expression =
      dyn_cast<Expr>(Statement->IgnoreContainers(/*IgnoreCaptured=*/true));
  if (!Expression) {
    return nullptr;
  }
  Expression = IgnoreExprNoise(Expression);
  const auto* Binary = dyn_cast_or_null<BinaryOperator>(Expression);
  if (!Binary || !Binary->isAssignmentOp()) {
    return nullptr;
  }
  return ReferencedVariable(Binary->getLHS());
}

struct ReleaseInfo {
  const FunctionDecl* Function = nullptr;
  SourceLocation Location;
};

using ReleasedVariables = llvm::DenseMap<const VarDecl*, ReleaseInfo>;

class ReleasedUseVisitor final
    : public ConstStmtVisitor<ReleasedUseVisitor, void> {
 public:
  ReleasedUseVisitor(ClangTidyCheck& Check, const SourceManager& SourceManager,
                     const ReleasedVariables& Released)
      : Check(Check), SourceManager(SourceManager), Released(Released) {}

  void VisitStmt(const Stmt* Statement) { VisitChildren(Statement); }

  void VisitMemberExpr(const MemberExpr* Expression) {
    if (const VarDecl* Variable =
            ReferencedVariableThroughIndirection(Expression->getBase())) {
      DiagnoseDereference(Variable, Expression->getOperatorLoc());
    }
    VisitChildren(Expression);
  }

  void VisitUnaryOperator(const UnaryOperator* Expression) {
    if (Expression->getOpcode() == UO_Deref) {
      if (const VarDecl* Variable =
              ReferencedVariable(Expression->getSubExpr())) {
        DiagnoseDereference(Variable, Expression->getOperatorLoc());
      }
    }
    VisitChildren(Expression);
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr* Expression) {
    if (const VarDecl* Variable = ReferencedVariable(Expression->getBase())) {
      DiagnoseDereference(Variable, Expression->getBeginLoc());
    }
    VisitChildren(Expression);
  }

  void VisitBinaryOperator(const BinaryOperator* Expression) {
    if (Expression->isAssignmentOp()) {
      if (const VarDecl* Variable = ReferencedVariable(Expression->getRHS())) {
        DiagnoseUse(Variable, Expression->getRHS()->getBeginLoc());
      }
    }
    VisitChildren(Expression);
  }

  void VisitCallExpr(const CallExpr* Expression) {
    const FunctionDecl* Callee = Expression->getDirectCallee();
    if (Callee && (IsReleaseFunctionName(Callee->getName()) ||
                   IsRetainFunctionName(Callee->getName()))) {
      VisitChildren(Expression);
      return;
    }
    for (const Expr* Argument : Expression->arguments()) {
      if (const VarDecl* Variable = ReferencedVariable(Argument)) {
        DiagnoseUse(Variable, Argument->getBeginLoc());
      }
    }
    VisitChildren(Expression);
  }

  void VisitDeclStmt(const DeclStmt* Statement) {
    for (const Decl* Declaration : Statement->decls()) {
      const auto* Variable = dyn_cast<VarDecl>(Declaration);
      if (!Variable || !Variable->hasInit()) {
        continue;
      }
      if (const VarDecl* Referenced = ReferencedVariable(Variable->getInit())) {
        DiagnoseUse(Referenced, Variable->getInit()->getBeginLoc());
      }
    }
    VisitChildren(Statement);
  }

  void VisitReturnStmt(const ReturnStmt* Statement) {
    if (const Expr* ReturnValue = Statement->getRetValue()) {
      if (const VarDecl* Variable = ReferencedVariable(ReturnValue)) {
        DiagnoseUse(Variable, ReturnValue->getBeginLoc());
      }
    }
    VisitChildren(Statement);
  }

 private:
  void VisitChildren(const Stmt* Statement) {
    for (const Stmt* Child : Statement->children()) {
      if (Child) {
        Visit(Child);
      }
    }
  }

  void DiagnoseDereference(const VarDecl* Variable, SourceLocation Location) {
    auto It = Released.find(Variable);
    if (It == Released.end() || Diagnosed.contains(Variable)) {
      return;
    }
    if (IsExternalMacroBody(Location, SourceManager)) {
      return;
    }
    Diagnosed.insert(Variable);
    Check.diag(SourceManager.getExpansionLoc(Location),
               "%0 is dereferenced after %1 releases it")
        << Variable->getName() << It->second.Function->getName();
  }

  void DiagnoseUse(const VarDecl* Variable, SourceLocation Location) {
    auto It = Released.find(Variable);
    if (It == Released.end() || Diagnosed.contains(Variable)) {
      return;
    }
    if (IsExternalMacroBody(Location, SourceManager)) {
      return;
    }
    Diagnosed.insert(Variable);
    Check.diag(SourceManager.getExpansionLoc(Location),
               "%0 is used after %1 releases it")
        << Variable->getName() << It->second.Function->getName();
  }

  ClangTidyCheck& Check;
  const SourceManager& SourceManager;
  const ReleasedVariables& Released;
  llvm::SmallPtrSet<const VarDecl*, 4> Diagnosed;
};

class ReleaseFlowAnalyzer final
    : public ConstStmtVisitor<ReleaseFlowAnalyzer, void> {
 public:
  ReleaseFlowAnalyzer(ClangTidyCheck& Check, const SourceManager& SourceManager)
      : Check(Check), SourceManager(SourceManager) {}

  void VisitStmt(const Stmt* Statement) { VisitChildren(Statement); }

  void VisitCompoundStmt(const CompoundStmt* Compound) {
    ReleasedVariables Released;
    llvm::DenseMap<const VarDecl*, unsigned> ReferenceCounts;
    for (const Stmt* Child : Compound->body()) {
      if (!Child) {
        continue;
      }
      ReleasedUseVisitor UseVisitor(Check, SourceManager, Released);
      UseVisitor.Visit(Child);
      if (std::optional<DirectRefCountOperation> Release =
              DirectReleaseStatement(Child)) {
        ConsumeRelease(*Release, Released, ReferenceCounts);
      } else if (const auto* If = dyn_cast<IfStmt>(Child);
                 std::optional<DirectRefCountOperation> Release =
                     IfElseMergedReleaseStatement(If)) {
        ConsumeRelease(*Release, Released, ReferenceCounts);
      } else if (std::optional<DirectRefCountOperation> Retain =
                     DirectRetainStatement(Child)) {
        auto It = Released.find(Retain->Variable);
        if (It != Released.end() &&
            !IsExternalMacroBody(Retain->Location, SourceManager)) {
          Check.diag(SourceManager.getExpansionLoc(Retain->Location),
                     "%0 is retained by %1 after %2 already released it")
              << Retain->Variable->getName() << Retain->Function->getName()
              << It->second.Function->getName();
        } else {
          unsigned& ReferenceCount = ReferenceCounts[Retain->Variable];
          if (ReferenceCount == 0) {
            ReferenceCount = 1;
          }
          ++ReferenceCount;
        }
      } else if (const VarDecl* Assigned = DirectVariableAssignment(Child)) {
        Released.erase(Assigned);
        ReferenceCounts.erase(Assigned);
      }
      Visit(Child);
    }
  }

 private:
  void ConsumeRelease(
      const DirectRefCountOperation& Release, ReleasedVariables& Released,
      llvm::DenseMap<const VarDecl*, unsigned>& ReferenceCounts) {
    auto It = Released.find(Release.Variable);
    if (It != Released.end() &&
        !IsExternalMacroBody(Release.Location, SourceManager)) {
      Check.diag(SourceManager.getExpansionLoc(Release.Location),
                 "%0 is released by %1 after %2 already released it")
          << Release.Variable->getName() << Release.Function->getName()
          << It->second.Function->getName();
      return;
    }
    unsigned& ReferenceCount = ReferenceCounts[Release.Variable];
    if (ReferenceCount == 0) {
      ReferenceCount = 1;
    }
    --ReferenceCount;
    if (ReferenceCount == 0) {
      Released[Release.Variable] = ReleaseInfo{.Function = Release.Function,
                                               .Location = Release.Location};
    }
  }

  void VisitChildren(const Stmt* Statement) {
    for (const Stmt* Child : Statement->children()) {
      if (Child) {
        Visit(Child);
      }
    }
  }

  ClangTidyCheck& Check;
  const SourceManager& SourceManager;
};

}  // namespace

RefCountLifecycleCheck::RefCountLifecycleCheck(StringRef Name,
                                               ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void RefCountLifecycleCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(isExpansionInSystemHeader()))
          .bind("function"),
      this);
  Finder->addMatcher(
      fieldDecl(unless(isExpansionInSystemHeader())).bind("refcount-field"),
      this);
  Finder->addMatcher(
      ifStmt(unless(isExpansionInSystemHeader())).bind("guarded-release-if"),
      this);
}

void RefCountLifecycleCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  if (const auto* Field = Result.Nodes.getNodeAs<FieldDecl>("refcount-field")) {
    const SourceManager& SourceManager = *Result.SourceManager;
    if (IsExternalMacroBody(Field->getLocation(), SourceManager)) {
      return;
    }
    switch (DiagnoseRefCountField(Field, SourceManager)) {
      case RefCountFieldDiagnostic::kNone:
        return;
      case RefCountFieldDiagnostic::kMisusedCounter:
        diag(SourceManager.getExpansionLoc(Field->getLocation()),
             "iree_atomic_ref_count_t field %0 must model object lifetime; "
             "use an explicit atomic integer type for counters")
            << Field->getName();
        return;
      case RefCountFieldDiagnostic::kRefCountNotFirst:
        diag(SourceManager.getExpansionLoc(Field->getLocation()),
             "iree_atomic_ref_count_t field ref_count must be the first field "
             "in a refcounted object");
        return;
    }
  }

  if (const auto* If = Result.Nodes.getNodeAs<IfStmt>("guarded-release-if")) {
    const SourceManager& SourceManager = *Result.SourceManager;
    if (IsExternalMacroBody(If->getIfLoc(), SourceManager)) {
      return;
    }
    std::optional<DirectRefCountOperation> Release =
        GuardedReleaseStatement(If, *Result.Context);
    if (!Release || IsExternalMacroBody(Release->Location, SourceManager)) {
      return;
    }
    diag(SourceManager.getExpansionLoc(If->getIfLoc()),
         "%0 is null-guarded before %1, but release functions must be "
         "null-safe")
        << Release->Variable->getName() << Release->Function->getName();
    return;
  }

  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Function->getLocation(), SourceManager)) {
    return;
  }
  const Stmt* Body = Function->getBody();
  if (!Body) {
    return;
  }
  ReleaseFlowAnalyzer FlowAnalyzer(*this, SourceManager);
  FlowAnalyzer.Visit(Body);

  std::optional<std::string> BodyText = FunctionBodySourceText(
      Function, SourceManager, Result.Context->getLangOpts());
  if (!BodyText || !BodyContainsRefCountMutation(*BodyText)) {
    return;
  }
  std::string NormalizedBody = NormalizedCodeSource(*BodyText);
  if (HasIgnoredRefCountDecrement(NormalizedBody)) {
    diag(SourceManager.getExpansionLoc(Function->getLocation()),
         "iree_atomic_ref_count_dec return value must be checked");
    return;
  }
  StringRef FunctionName = Function->getName();
  if (!IsRefCountLifecycleFunctionName(FunctionName)) {
    return;
  }
  if (Function->getReturnType()->isVoidType() &&
      FunctionName.ends_with("_release") &&
      !IsRefCountReleaseNullSafe(Function, NormalizedBody)) {
    diag(SourceManager.getExpansionLoc(Function->getLocation()),
         "refcounted release function %0 must be null-safe")
        << FunctionName;
    return;
  }
  if (Function->getReturnType()->isVoidType() &&
      FunctionName.ends_with("_retain") &&
      !IsRefCountRetainNullSafe(Function, NormalizedBody)) {
    diag(SourceManager.getExpansionLoc(Function->getLocation()),
         "refcounted retain function %0 must be null-safe")
        << FunctionName;
    return;
  }
  if (Function->getReturnType()->isVoidType()) {
    return;
  }
  diag(SourceManager.getExpansionLoc(Function->getLocation()),
       "refcounted retain/release function %0 must return void")
      << FunctionName;
}

}  // namespace clang::tidy::iree
