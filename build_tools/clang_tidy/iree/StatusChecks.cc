// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/StatusChecks.h"

#include <optional>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::tidy::iree {
namespace {

bool IsNamedTypedef(QualType Type, StringRef Name) {
  if (Type.isNull()) {
    return false;
  }
  Type = Type.getNonReferenceType();
  if (Type.isNull() || Type->isDependentType()) {
    return false;
  }
  Type = Type.getLocalUnqualifiedType();
  const auto* Typedef = Type->getAs<TypedefType>();
  return Typedef && Typedef->getDecl()->getName() == Name;
}

bool IsStatusType(QualType Type) {
  return IsNamedTypedef(Type, "iree_status_t");
}

bool IsBorrowedStatusType(QualType Type) {
  return IsStatusType(Type) && Type.isConstQualified();
}

QualType ReturnTypeFromCalleeExpr(const Expr* Callee) {
  if (!Callee) {
    return QualType();
  }
  QualType CalleeType = Callee->IgnoreParenImpCasts()->getType();
  if (CalleeType.isNull() || CalleeType->isDependentType()) {
    return QualType();
  }
  CalleeType = CalleeType.getLocalUnqualifiedType();
  if (const auto* Pointer = CalleeType->getAs<PointerType>()) {
    CalleeType = Pointer->getPointeeType().getLocalUnqualifiedType();
  }
  if (const auto* Proto = CalleeType->getAs<FunctionProtoType>()) {
    return Proto->getReturnType();
  }
  if (const auto* NoProto = CalleeType->getAs<FunctionNoProtoType>()) {
    return NoProto->getReturnType();
  }
  return QualType();
}

QualType ParameterTypeFromCalleeExpr(const Expr* Callee, unsigned Index) {
  if (!Callee) {
    return QualType();
  }
  QualType CalleeType = Callee->IgnoreParenImpCasts()->getType();
  if (CalleeType.isNull() || CalleeType->isDependentType()) {
    return QualType();
  }
  CalleeType = CalleeType.getLocalUnqualifiedType();
  if (const auto* Pointer = CalleeType->getAs<PointerType>()) {
    CalleeType = Pointer->getPointeeType().getLocalUnqualifiedType();
  }
  const auto* Proto = CalleeType->getAs<FunctionProtoType>();
  if (!Proto || Index >= Proto->getNumParams()) {
    return QualType();
  }
  return Proto->getParamType(Index);
}

QualType ReturnTypeFromCall(const CallExpr* Call) {
  if (!Call) {
    return QualType();
  }
  if (const FunctionDecl* Callee = Call->getDirectCallee()) {
    return Callee->getReturnType();
  }
  return ReturnTypeFromCalleeExpr(Call->getCallee());
}

QualType ParameterTypeFromCall(const CallExpr* Call, unsigned Index) {
  if (!Call) {
    return QualType();
  }
  if (const FunctionDecl* Callee = Call->getDirectCallee();
      Callee && Index < Callee->getNumParams()) {
    return Callee->getParamDecl(Index)->getType();
  }
  return ParameterTypeFromCalleeExpr(Call->getCallee(), Index);
}

StringRef CalleeName(const CallExpr* Call) {
  if (const FunctionDecl* Callee = Call->getDirectCallee()) {
    if (const auto* Identifier = Callee->getIdentifier()) {
      return Identifier->getName();
    }
  }
  return StringRef();
}

bool IsAllowedExplicitStatusConsumer(const CallExpr* Call) {
  return CalleeName(Call) == "iree_status_ignore";
}

bool IsMacroStatusTemporary(const VarDecl* Var) {
  if (!Var) {
    return false;
  }
  return Var->getName().starts_with("__status_") &&
         Var->getLocation().isMacroID();
}

const Expr* IgnoreExprNoise(const Expr* Expr) {
  while (Expr) {
    Expr = Expr->IgnoreParenImpCasts();
    if (const auto* Cleanups = dyn_cast<ExprWithCleanups>(Expr)) {
      Expr = Cleanups->getSubExpr();
      continue;
    }
    if (const auto* Materialized = dyn_cast<MaterializeTemporaryExpr>(Expr)) {
      Expr = Materialized->getSubExpr();
      continue;
    }
    break;
  }
  return Expr;
}

const VarDecl* ReferencedStatusLocal(const Expr* Expr) {
  Expr = IgnoreExprNoise(Expr);
  const auto* Ref = dyn_cast_or_null<DeclRefExpr>(Expr);
  if (!Ref) {
    return nullptr;
  }
  const auto* Var = dyn_cast<VarDecl>(Ref->getDecl());
  if (!Var || !Var->hasLocalStorage() || IsMacroStatusTemporary(Var) ||
      !IsStatusType(Var->getType())) {
    return nullptr;
  }
  return Var->getCanonicalDecl();
}

const VarDecl* ReferencedStatusLocalThroughCasts(const Expr* Expr) {
  Expr = IgnoreExprNoise(Expr);
  while (const auto* Cast = dyn_cast_or_null<CastExpr>(Expr)) {
    Expr = IgnoreExprNoise(Cast->getSubExpr());
  }
  return ReferencedStatusLocal(Expr);
}

const VarDecl* ForwardedStatusLocal(const Expr* Expr) {
  if (const VarDecl* Var = ReferencedStatusLocal(Expr)) {
    return Var;
  }
  Expr = IgnoreExprNoise(Expr);
  if (const auto* Call = dyn_cast_or_null<CallExpr>(Expr);
      Call && CalleeName(Call) == "move" && Call->getNumArgs() == 1) {
    return ForwardedStatusLocal(Call->getArg(0));
  }
  return nullptr;
}

const VarDecl* ForwardedStatusLocalThroughCasts(const Expr* Expr) {
  Expr = IgnoreExprNoise(Expr);
  while (const auto* Cast = dyn_cast_or_null<CastExpr>(Expr)) {
    Expr = IgnoreExprNoise(Cast->getSubExpr());
  }
  return ForwardedStatusLocal(Expr);
}

bool IsStatusObserver(StringRef Name) {
  return Name == "iree_status_code" || Name == "iree_status_is_ok" ||
         Name.starts_with("iree_status_is_") || Name == "iree_status_format" ||
         Name == "iree_status_format_to" || Name == "iree_status_to_string" ||
         Name == "iree_status_fprint" || Name == "iree_status_format_message" ||
         Name == "iree_status_format_message_to" ||
         Name == "iree_status_source_location" ||
         Name == "iree_status_message" ||
         Name == "iree_status_enumerate_payloads";
}

bool IsStatusReportingObserver(StringRef Name) {
  return Name == "iree_status_format" || Name == "iree_status_format_to" ||
         Name == "iree_status_to_string" || Name == "iree_status_fprint" ||
         Name == "iree_status_format_message" ||
         Name == "iree_status_format_message_to";
}

bool IsKnownStatusNoOwnerProducer(StringRef Name) {
  return Name == "iree_ok_status" || Name == "iree_status_from_code" ||
         Name == "iree_status_ignore" ||
         Name == "iree_async_socket_query_failure";
}

bool IsKnownStatusConsumer(StringRef Name) {
  return Name == "iree_status_free" || Name == "iree_status_ignore" ||
         Name == "iree_status_consume_code" || Name == "iree_status_abort" ||
         Name == "ConsumeForTest" || Name == "StatusToStringAndFree";
}

std::optional<unsigned> KnownStatusSinkArgument(StringRef Name) {
  if (Name == "iree_async_proactor_io_uring_push_software_completion") {
    return 2;
  }
  return std::nullopt;
}

bool IsKnownStatusTransferProducer(StringRef Name) {
  return Name == "iree_status_annotate" || Name == "iree_status_annotate_f" ||
         Name == "iree_status_freeze" || Name == "iree_status_join";
}

bool IsKnownStatusClone(StringRef Name) { return Name == "iree_status_clone"; }

bool IsNoReturnStatusConsumer(StringRef Name) {
  return Name == "iree_status_abort";
}

std::optional<std::pair<const VarDecl*, bool>> CompareExchangeStatusCondition(
    const Expr* Condition) {
  Condition = IgnoreExprNoise(Condition);
  if (!Condition) {
    return std::nullopt;
  }
  if (const auto* Unary = dyn_cast<UnaryOperator>(Condition);
      Unary && Unary->getOpcode() == UO_LNot) {
    if (auto Inner = CompareExchangeStatusCondition(Unary->getSubExpr())) {
      return std::make_pair(Inner->first, !Inner->second);
    }
  }
  if (const auto* Atomic = dyn_cast<AtomicExpr>(Condition);
      Atomic && Atomic->isCmpXChg()) {
    if (const VarDecl* Var =
            ForwardedStatusLocalThroughCasts(Atomic->getVal2())) {
      return std::make_pair(Var, true);
    }
  }
  if (const auto* Call = dyn_cast<CallExpr>(Condition);
      Call && Call->getNumArgs() >= 3 &&
      CalleeName(Call).contains("compare_exchange")) {
    if (const VarDecl* Var =
            ForwardedStatusLocalThroughCasts(Call->getArg(2))) {
      return std::make_pair(Var, true);
    }
  }
  return std::nullopt;
}

bool IsNoReturnCall(const CallExpr* Call) {
  if (const FunctionDecl* Callee = Call->getDirectCallee()) {
    if (Callee->isNoReturn()) {
      return true;
    }
  }
  StringRef Name = CalleeName(Call);
  return Name == "abort" || Name == "_Exit" || Name == "exit" ||
         Name == "quick_exit" || Name == "__builtin_abort" ||
         Name == "__builtin_trap" || Name == "iree_abort";
}

bool IsIreeOkStatusCall(const Expr* Expr) {
  Expr = IgnoreExprNoise(Expr);
  const auto* Call = dyn_cast_or_null<CallExpr>(Expr);
  return Call && CalleeName(Call) == "iree_ok_status";
}

struct StatusValue {
  bool IsStatus = false;
  bool MayOwn = false;
  bool Unknown = false;
  SourceLocation Location;

  static StatusValue NonStatus() { return StatusValue(); }

  static StatusValue NoOwner(SourceLocation Location = SourceLocation()) {
    StatusValue Value;
    Value.IsStatus = true;
    Value.Location = Location;
    return Value;
  }

  static StatusValue Owned(SourceLocation Location) {
    StatusValue Value;
    Value.IsStatus = true;
    Value.MayOwn = true;
    Value.Location = Location;
    return Value;
  }

  static StatusValue UnknownStatus(SourceLocation Location = SourceLocation()) {
    StatusValue Value;
    Value.IsStatus = true;
    Value.Unknown = true;
    Value.Location = Location;
    return Value;
  }
};

struct VariableState {
  bool MayOwn = false;
  bool MayBeConsumed = false;
  bool Unknown = false;
  bool Reported = false;
  bool OkInitializerLive = false;
  SourceLocation OwnershipLocation;
  SourceLocation ConsumeLocation;
  SourceLocation ReportLocation;
  SourceLocation OkInitLocation;
  SourceRange OkInitRange;

  static VariableState NoOwner() { return VariableState(); }

  static VariableState Owned(SourceLocation Location) {
    VariableState State;
    State.MayOwn = true;
    State.OwnershipLocation = Location;
    return State;
  }

  static VariableState UnknownState() {
    VariableState State;
    State.Unknown = true;
    return State;
  }

  static VariableState OkInitialized(SourceLocation Location,
                                     SourceRange Range) {
    VariableState State;
    State.OkInitializerLive = true;
    State.OkInitLocation = Location;
    State.OkInitRange = Range;
    return State;
  }
};

struct AnalysisState {
  llvm::DenseMap<const VarDecl*, VariableState> Variables;
  bool Terminal = false;
  bool SuppressOkOverwriteDiagnostics = false;
};

struct ImmediateOkOverwriteFix {
  const Expr* ReplacementExpr = nullptr;
  const Stmt* RemovalStatement = nullptr;
};

VariableState MergeVariableState(const VariableState& Lhs,
                                 const VariableState& Rhs) {
  if (Lhs.Unknown || Rhs.Unknown) {
    return VariableState::UnknownState();
  }
  VariableState Result;
  Result.MayOwn = Lhs.MayOwn || Rhs.MayOwn;
  Result.MayBeConsumed = Lhs.MayBeConsumed || Rhs.MayBeConsumed;
  Result.Reported = Lhs.Reported || Rhs.Reported;
  Result.OkInitializerLive = Lhs.OkInitializerLive && Rhs.OkInitializerLive;
  Result.OwnershipLocation = Lhs.OwnershipLocation.isValid()
                                 ? Lhs.OwnershipLocation
                                 : Rhs.OwnershipLocation;
  Result.ConsumeLocation =
      Lhs.ConsumeLocation.isValid() ? Lhs.ConsumeLocation : Rhs.ConsumeLocation;
  Result.ReportLocation =
      Lhs.ReportLocation.isValid() ? Lhs.ReportLocation : Rhs.ReportLocation;
  Result.OkInitLocation =
      Lhs.OkInitLocation.isValid() ? Lhs.OkInitLocation : Rhs.OkInitLocation;
  Result.OkInitRange =
      Lhs.OkInitRange.isValid() ? Lhs.OkInitRange : Rhs.OkInitRange;
  return Result;
}

AnalysisState MergeStates(const AnalysisState& Lhs, const AnalysisState& Rhs) {
  if (Lhs.Terminal) {
    return Rhs;
  }
  if (Rhs.Terminal) {
    return Lhs;
  }
  AnalysisState Result = Lhs;
  Result.SuppressOkOverwriteDiagnostics =
      Lhs.SuppressOkOverwriteDiagnostics && Rhs.SuppressOkOverwriteDiagnostics;
  for (const auto& [Var, RhsState] : Rhs.Variables) {
    auto It = Result.Variables.find(Var);
    if (It == Result.Variables.end()) {
      Result.Variables[Var] = RhsState;
    } else {
      It->second = MergeVariableState(It->second, RhsState);
    }
  }
  return Result;
}

class StatusRefCollector {
 public:
  void collect(const Stmt* Statement) {
    if (!Statement) {
      return;
    }
    if (const auto* ExprStatement = dyn_cast<Expr>(Statement)) {
      if (const VarDecl* Var = ReferencedStatusLocal(ExprStatement)) {
        Variables_.insert(Var);
      }
    }
    for (const Stmt* Child : Statement->children()) {
      collect(Child);
    }
  }

  const llvm::SmallPtrSetImpl<const VarDecl*>& variables() const {
    return Variables_;
  }

 private:
  llvm::SmallPtrSet<const VarDecl*, 8> Variables_;
};

class GotoFinder {
 public:
  bool containsGoto(const Stmt* Statement) {
    if (!Statement) {
      return false;
    }
    if (isa<GotoStmt>(Statement) || isa<IndirectGotoStmt>(Statement)) {
      return true;
    }
    for (const Stmt* Child : Statement->children()) {
      if (containsGoto(Child)) {
        return true;
      }
    }
    return false;
  }
};

struct StatusParameterUseState {
  bool Observed = false;
  bool OwnsOrEscapes = false;
  bool ExplicitBorrowed = false;
  SourceLocation ObserveLocation;
  SourceLocation OwnsOrEscapesLocation;
};

const VarDecl* ReferencedStatusParameter(const Expr* Expr) {
  Expr = IgnoreExprNoise(Expr);
  const auto* Ref = dyn_cast_or_null<DeclRefExpr>(Expr);
  if (!Ref) {
    return nullptr;
  }
  const auto* Param = dyn_cast<ParmVarDecl>(Ref->getDecl());
  if (!Param || !IsStatusType(Param->getType())) {
    return nullptr;
  }
  return Param->getCanonicalDecl();
}

bool IsBorrowedStatusCheckExemptFunction(StringRef Name) {
  return IsStatusObserver(Name) || IsKnownStatusConsumer(Name) ||
         IsKnownStatusTransferProducer(Name) || IsKnownStatusClone(Name) ||
         IsKnownStatusNoOwnerProducer(Name) ||
         Name == "iree_async_proactor_io_uring_push_software_completion" ||
         Name == "iree_hal_status_as_semaphore_failure" ||
         Name == "iree_status_from_loomc" || Name == "loomc_status_from_iree" ||
         Name == "iree_status_from_loomc_status" ||
         Name == "loomc_status_from_iree_status" ||
         Name == "iree_to_hrx_status";
}

bool IsStatusCppObserverMethod(const FunctionDecl* Function) {
  const auto* Method = dyn_cast<CXXMethodDecl>(Function);
  if (!Method || !Method->getIdentifier()) {
    return false;
  }
  const CXXRecordDecl* Parent = Method->getParent();
  return Parent && Parent->getName() == "Status" &&
         Method->getName() == "ToString";
}

class BorrowedStatusParameterAnalyzer {
 public:
  explicit BorrowedStatusParameterAnalyzer(BorrowedStatusParameterCheck& Check)
      : Check_(Check) {}

  void analyzeFunction(const FunctionDecl* Function) {
    if (!Function->hasBody()) {
      return;
    }
    if (const auto* Identifier = Function->getIdentifier();
        Identifier &&
        IsBorrowedStatusCheckExemptFunction(Identifier->getName())) {
      return;
    }
    if (IsStatusCppObserverMethod(Function)) {
      return;
    }
    for (const ParmVarDecl* Param : Function->parameters()) {
      if (!Param->getType()->isReferenceType() &&
          IsStatusType(Param->getType())) {
        StatusParameterUseState ParamState;
        ParamState.ExplicitBorrowed = IsBorrowedStatusType(Param->getType());
        State_[Param->getCanonicalDecl()] = ParamState;
      }
    }
    if (State_.empty()) {
      return;
    }
    FunctionReturnsStatus_ = IsStatusType(Function->getReturnType());
    analyzeStatement(Function->getBody());
    for (const auto& [Param, State] : State_) {
      if (State.ExplicitBorrowed) {
        if (State.OwnsOrEscapes) {
          Check_.diag(State.OwnsOrEscapesLocation.isValid()
                          ? State.OwnsOrEscapesLocation
                          : Param->getLocation(),
                      "const iree_status_t parameter %0 is borrowed and must "
                      "not be consumed, returned, stored, or transferred")
              << Param;
        }
        continue;
      }
      if (!State.Observed || State.OwnsOrEscapes) {
        continue;
      }
      Check_.diag(State.ObserveLocation.isValid() ? State.ObserveLocation
                                                  : Param->getLocation(),
                  "iree_status_t parameter %0 is only observed; use "
                  "iree_status_code_t or another non-owning representation")
          << Param;
    }
  }

 private:
  void markObserved(const VarDecl* Param, SourceLocation Location) {
    auto It = State_.find(Param);
    if (It == State_.end() || It->second.OwnsOrEscapes) {
      return;
    }
    It->second.Observed = true;
    if (!It->second.ObserveLocation.isValid()) {
      It->second.ObserveLocation = Location;
    }
  }

  void markOwnsOrEscapes(const VarDecl* Param, SourceLocation Location) {
    auto It = State_.find(Param);
    if (It == State_.end()) {
      return;
    }
    It->second.OwnsOrEscapes = true;
    if (!It->second.OwnsOrEscapesLocation.isValid()) {
      It->second.OwnsOrEscapesLocation = Location;
    }
  }

  bool referencesTrackedStatusParameter(const Expr* Expr) {
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return false;
    }
    if (const VarDecl* Param = ReferencedStatusParameter(Expr);
        Param && State_.contains(Param)) {
      return true;
    }
    if (const auto* Lambda = dyn_cast<LambdaExpr>(Expr)) {
      for (const LambdaCapture& Capture : Lambda->captures()) {
        if (const auto* Param =
                dyn_cast_or_null<ParmVarDecl>(Capture.getCapturedVar())) {
          if (State_.contains(Param->getCanonicalDecl())) {
            return true;
          }
        }
      }
      return false;
    }
    for (const Stmt* Child : Expr->children()) {
      if (const auto* ChildExpr = dyn_cast_or_null<clang::Expr>(Child);
          ChildExpr && referencesTrackedStatusParameter(ChildExpr)) {
        return true;
      }
    }
    return false;
  }

  void markReferencedParametersOwnOrEscaped(const Expr* Expression) {
    Expression = IgnoreExprNoise(Expression);
    if (!Expression) {
      return;
    }
    if (const auto* Call = dyn_cast<CallExpr>(Expression);
        Call && IsKnownStatusClone(CalleeName(Call))) {
      for (const Expr* Arg : Call->arguments()) {
        analyzeExpression(Arg, /*transferred=*/false);
      }
      return;
    }
    if (const VarDecl* Param = ReferencedStatusParameter(Expression);
        Param && State_.contains(Param)) {
      markOwnsOrEscapes(Param, Expression->getExprLoc());
      return;
    }
    for (const Stmt* Child : Expression->children()) {
      if (const auto* ChildExpr = dyn_cast_or_null<clang::Expr>(Child)) {
        markReferencedParametersOwnOrEscaped(ChildExpr);
      }
    }
  }

  bool isStatusDestination(const Expr* Expr) const {
    Expr = IgnoreExprNoise(Expr);
    return Expr && IsStatusType(Expr->getType());
  }

  void analyzeStatement(const Stmt* Statement) {
    if (!Statement) {
      return;
    }
    if (const auto* Lambda = dyn_cast<LambdaExpr>(Statement)) {
      for (const LambdaCapture& Capture : Lambda->captures()) {
        if (const auto* Param =
                dyn_cast_or_null<ParmVarDecl>(Capture.getCapturedVar())) {
          markOwnsOrEscapes(Param->getCanonicalDecl(), Capture.getLocation());
        }
      }
      return;
    }
    if (const auto* Decl = dyn_cast<DeclStmt>(Statement)) {
      analyzeDeclStatement(Decl);
      return;
    }
    if (const auto* Return = dyn_cast<ReturnStmt>(Statement)) {
      if (const Expr* Value = Return->getRetValue()) {
        analyzeExpression(Value, /*transferred=*/FunctionReturnsStatus_);
      }
      return;
    }
    if (const auto* ExprStatement = dyn_cast<Expr>(Statement)) {
      analyzeExpression(ExprStatement, /*transferred=*/false);
      return;
    }
    for (const Stmt* Child : Statement->children()) {
      analyzeStatement(Child);
    }
  }

  void analyzeDeclStatement(const DeclStmt* DeclStatement) {
    for (const Decl* D : DeclStatement->decls()) {
      const auto* Var = dyn_cast<VarDecl>(D);
      if (!Var || !Var->hasInit()) {
        continue;
      }
      analyzeExpression(Var->getInit(),
                        /*transferred=*/IsStatusType(Var->getType()));
    }
  }

  void analyzeExpression(const Expr* Expr, bool transferred) {
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return;
    }
    if (const VarDecl* Param = ReferencedStatusParameter(Expr);
        Param && State_.contains(Param)) {
      if (transferred) {
        markOwnsOrEscapes(Param, Expr->getExprLoc());
      } else {
        markObserved(Param, Expr->getExprLoc());
      }
      return;
    }
    if (const auto* Cast = dyn_cast<CastExpr>(Expr)) {
      analyzeExpression(Cast->getSubExpr(), transferred);
      return;
    }
    if (const auto* Call = dyn_cast<CallExpr>(Expr)) {
      analyzeCall(Call, transferred);
      return;
    }
    if (const auto* Construct = dyn_cast<CXXConstructExpr>(Expr)) {
      const CXXConstructorDecl* Constructor = Construct->getConstructor();
      const bool IsStatusWrapper =
          Constructor && Constructor->getParent() &&
          (Constructor->getParent()->getName() == "Status" ||
           Constructor->getParent()->getName() == "StatusOr");
      for (unsigned I = 0; I < Construct->getNumArgs(); ++I) {
        analyzeExpression(Construct->getArg(I), IsStatusWrapper);
      }
      return;
    }
    if (const auto* Binary = dyn_cast<BinaryOperator>(Expr)) {
      if (Binary->isAssignmentOp()) {
        analyzeExpression(Binary->getLHS(), /*transferred=*/false);
        analyzeExpression(
            Binary->getRHS(),
            /*transferred=*/isStatusDestination(Binary->getLHS()));
        return;
      }
      analyzeExpression(Binary->getLHS(), /*transferred=*/false);
      analyzeExpression(Binary->getRHS(), /*transferred=*/false);
      return;
    }
    if (const auto* Unary = dyn_cast<UnaryOperator>(Expr)) {
      if (Unary->getOpcode() == UO_AddrOf) {
        markReferencedParametersOwnOrEscaped(Unary->getSubExpr());
      } else {
        analyzeExpression(Unary->getSubExpr(), transferred);
      }
      return;
    }
    if (const auto* Conditional = dyn_cast<AbstractConditionalOperator>(Expr)) {
      analyzeExpression(Conditional->getCond(), /*transferred=*/false);
      analyzeExpression(Conditional->getTrueExpr(), transferred);
      analyzeExpression(Conditional->getFalseExpr(), transferred);
      return;
    }
    for (const Stmt* Child : Expr->children()) {
      if (const auto* ChildExpr = dyn_cast_or_null<clang::Expr>(Child)) {
        analyzeExpression(ChildExpr, transferred);
      } else {
        analyzeStatement(Child);
      }
    }
  }

  void analyzeCall(const CallExpr* Call, bool transferred) {
    StringRef Name = CalleeName(Call);
    if (IsKnownStatusConsumer(Name) && Call->getNumArgs() >= 1) {
      analyzeExpression(Call->getArg(0), /*transferred=*/true);
      for (unsigned I = 1; I < Call->getNumArgs(); ++I) {
        analyzeExpression(Call->getArg(I), /*transferred=*/false);
      }
      return;
    }
    if (IsKnownStatusTransferProducer(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        bool ConsumesArg = Name == "iree_status_join" ? I <= 1 : I == 0;
        analyzeExpression(Call->getArg(I), /*transferred=*/ConsumesArg);
      }
      return;
    }
    if (IsKnownStatusClone(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        const Expr* Arg = Call->getArg(I);
        if (const VarDecl* Param = ReferencedStatusParameter(Arg);
            Param && State_.contains(Param) && State_[Param].ExplicitBorrowed) {
          analyzeExpression(Arg, /*transferred=*/false);
        } else {
          analyzeExpression(Arg, /*transferred=*/true);
        }
      }
      return;
    }
    if (std::optional<unsigned> SinkArg = KnownStatusSinkArgument(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        analyzeExpression(Call->getArg(I), /*transferred=*/I == *SinkArg);
      }
      return;
    }
    if (IsStatusObserver(Name)) {
      for (const Expr* Arg : Call->arguments()) {
        analyzeExpression(Arg, /*transferred=*/false);
      }
      return;
    }

    for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
      const Expr* Arg = Call->getArg(I);
      QualType ParamType = ParameterTypeFromCall(Call, I);
      const bool ParamIsBorrowedStatus =
          !ParamType.isNull() && IsBorrowedStatusType(ParamType);
      const bool ArgIsStatus = IsStatusType(Arg->getType()) ||
                               (!ParamType.isNull() && IsStatusType(ParamType));
      if (ParamIsBorrowedStatus) {
        analyzeExpression(Arg, /*transferred=*/false);
      } else if (transferred || ArgIsStatus) {
        markReferencedParametersOwnOrEscaped(Arg);
      } else if (referencesTrackedStatusParameter(Arg)) {
        analyzeExpression(Arg, /*transferred=*/false);
      } else {
        analyzeExpression(Arg, /*transferred=*/false);
      }
    }
  }

  BorrowedStatusParameterCheck& Check_;
  llvm::DenseMap<const VarDecl*, StatusParameterUseState> State_;
  bool FunctionReturnsStatus_ = false;
};

struct StatusFullExpressionUse {
  const VarDecl* Var = nullptr;
  bool Transfers = false;
  SourceLocation Location;
};

class StatusTransferOrderAnalyzer {
 public:
  explicit StatusTransferOrderAnalyzer(StatusTransferOrderCheck& Check)
      : Check_(Check) {}

  void analyzeFunction(const FunctionDecl* Function) {
    if (!Function->hasBody()) {
      return;
    }
    analyzeStatement(Function->getBody());
  }

 private:
  void checkFullExpression(const Expr* Expr) {
    SmallVector<StatusFullExpressionUse, 8> Uses;
    collectExpressionUses(Expr, /*transferred=*/false, Uses);

    llvm::DenseMap<const VarDecl*, unsigned> Counts;
    llvm::DenseMap<const VarDecl*, SourceLocation> TransferLocations;
    for (const StatusFullExpressionUse& Use : Uses) {
      if (!Use.Var) {
        continue;
      }
      const VarDecl* Var = Use.Var->getCanonicalDecl();
      Counts[Var] += 1;
      if (Use.Transfers && !TransferLocations[Var].isValid()) {
        TransferLocations[Var] = Use.Location;
      }
    }

    for (const auto& [Var, Count] : Counts) {
      SourceLocation TransferLocation = TransferLocations[Var];
      if (Count < 2 || !TransferLocation.isValid()) {
        continue;
      }
      Check_.diag(TransferLocation,
                  "iree_status_t %0 is used more than once in one full "
                  "expression while ownership is transferred; sequence the "
                  "operations through temporaries or clone before fanout")
          << Var;
    }
  }

  void recordUse(const VarDecl* Var, bool transferred, SourceLocation Location,
                 SmallVectorImpl<StatusFullExpressionUse>& Uses) {
    if (!Var) {
      return;
    }
    Uses.push_back(StatusFullExpressionUse{
        Var->getCanonicalDecl(),
        transferred,
        Location,
    });
  }

  bool isStatusDestination(const Expr* Expr) const {
    Expr = IgnoreExprNoise(Expr);
    return Expr && IsStatusType(Expr->getType());
  }

  void checkSequencedExpression(const Expr* Expr) { checkFullExpression(Expr); }

  void collectExpressionUses(const Expr* Expr, bool transferred,
                             SmallVectorImpl<StatusFullExpressionUse>& Uses) {
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return;
    }

    if (const VarDecl* Var = ForwardedStatusLocal(Expr)) {
      recordUse(Var, transferred, Expr->getExprLoc(), Uses);
      return;
    }
    if (const VarDecl* Var = ReferencedStatusLocal(Expr)) {
      recordUse(Var, transferred, Expr->getExprLoc(), Uses);
      return;
    }
    if (const auto* Cast = dyn_cast<CastExpr>(Expr)) {
      collectExpressionUses(Cast->getSubExpr(), transferred, Uses);
      return;
    }
    if (const auto* Call = dyn_cast<CallExpr>(Expr)) {
      collectCallUses(Call, transferred, Uses);
      return;
    }
    if (const auto* Construct = dyn_cast<CXXConstructExpr>(Expr)) {
      const CXXConstructorDecl* Constructor = Construct->getConstructor();
      const bool IsStatusWrapper =
          Constructor && Constructor->getParent() &&
          (Constructor->getParent()->getName() == "Status" ||
           Constructor->getParent()->getName() == "StatusOr");
      for (unsigned I = 0; I < Construct->getNumArgs(); ++I) {
        collectExpressionUses(Construct->getArg(I),
                              /*transferred=*/IsStatusWrapper, Uses);
      }
      return;
    }
    if (const auto* Binary = dyn_cast<BinaryOperator>(Expr)) {
      if (Binary->getOpcode() == BO_Comma || Binary->getOpcode() == BO_LAnd ||
          Binary->getOpcode() == BO_LOr) {
        checkSequencedExpression(Binary->getLHS());
        checkSequencedExpression(Binary->getRHS());
        return;
      }
      if (Binary->isAssignmentOp()) {
        collectExpressionUses(
            Binary->getRHS(),
            /*transferred=*/isStatusDestination(Binary->getLHS()), Uses);
        return;
      }
      collectExpressionUses(Binary->getLHS(), /*transferred=*/false, Uses);
      collectExpressionUses(Binary->getRHS(), /*transferred=*/false, Uses);
      return;
    }
    if (const auto* Unary = dyn_cast<UnaryOperator>(Expr)) {
      collectExpressionUses(Unary->getSubExpr(), transferred, Uses);
      return;
    }
    if (const auto* Conditional = dyn_cast<AbstractConditionalOperator>(Expr)) {
      checkSequencedExpression(Conditional->getCond());
      checkSequencedExpression(Conditional->getTrueExpr());
      checkSequencedExpression(Conditional->getFalseExpr());
      return;
    }

    for (const Stmt* Child : Expr->children()) {
      if (const auto* ChildExpr = dyn_cast_or_null<clang::Expr>(Child)) {
        collectExpressionUses(ChildExpr, transferred, Uses);
      }
    }
  }

  void collectCallUses(const CallExpr* Call, bool transferred,
                       SmallVectorImpl<StatusFullExpressionUse>& Uses) {
    StringRef Name = CalleeName(Call);
    if (IsKnownStatusConsumer(Name) && Call->getNumArgs() >= 1) {
      collectExpressionUses(Call->getArg(0), /*transferred=*/true, Uses);
      for (unsigned I = 1; I < Call->getNumArgs(); ++I) {
        collectExpressionUses(Call->getArg(I), /*transferred=*/false, Uses);
      }
      return;
    }
    if (IsKnownStatusTransferProducer(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        bool ConsumesArg = Name == "iree_status_join" ? I <= 1 : I == 0;
        collectExpressionUses(Call->getArg(I), /*transferred=*/ConsumesArg,
                              Uses);
      }
      return;
    }
    if (IsKnownStatusClone(Name)) {
      for (const Expr* Arg : Call->arguments()) {
        collectExpressionUses(Arg, /*transferred=*/false, Uses);
      }
      return;
    }
    if (std::optional<unsigned> SinkArg = KnownStatusSinkArgument(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        collectExpressionUses(Call->getArg(I), /*transferred=*/I == *SinkArg,
                              Uses);
      }
      return;
    }
    if (IsStatusObserver(Name)) {
      for (const Expr* Arg : Call->arguments()) {
        collectExpressionUses(Arg, /*transferred=*/false, Uses);
      }
      return;
    }

    for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
      const Expr* Arg = Call->getArg(I);
      QualType ParamType = ParameterTypeFromCall(Call, I);
      const bool ArgIsStatus = IsStatusType(Arg->getType()) ||
                               (!ParamType.isNull() && IsStatusType(ParamType));
      collectExpressionUses(Arg, /*transferred=*/transferred || ArgIsStatus,
                            Uses);
    }
  }

  void analyzeStatement(const Stmt* Statement) {
    if (!Statement) {
      return;
    }
    if (const auto* Compound = dyn_cast<CompoundStmt>(Statement)) {
      for (const Stmt* Child : Compound->body()) {
        analyzeStatement(Child);
      }
      return;
    }
    if (const auto* DeclarationStatement = dyn_cast<DeclStmt>(Statement)) {
      for (const Decl* D : DeclarationStatement->decls()) {
        const auto* Var = dyn_cast<VarDecl>(D);
        if (Var && Var->hasInit()) {
          checkFullExpression(Var->getInit());
        }
      }
      return;
    }
    if (const auto* Return = dyn_cast<ReturnStmt>(Statement)) {
      checkFullExpression(Return->getRetValue());
      return;
    }
    if (const auto* If = dyn_cast<IfStmt>(Statement)) {
      analyzeStatement(If->getInit());
      if (const DeclStmt* ConditionVariable =
              If->getConditionVariableDeclStmt()) {
        analyzeStatement(ConditionVariable);
      }
      checkFullExpression(If->getCond());
      analyzeStatement(If->getThen());
      analyzeStatement(If->getElse());
      return;
    }
    if (const auto* For = dyn_cast<ForStmt>(Statement)) {
      analyzeStatement(For->getInit());
      checkFullExpression(For->getCond());
      checkFullExpression(For->getInc());
      analyzeStatement(For->getBody());
      return;
    }
    if (const auto* While = dyn_cast<WhileStmt>(Statement)) {
      checkFullExpression(While->getCond());
      analyzeStatement(While->getBody());
      return;
    }
    if (const auto* Do = dyn_cast<DoStmt>(Statement)) {
      analyzeStatement(Do->getBody());
      checkFullExpression(Do->getCond());
      return;
    }
    if (const auto* Switch = dyn_cast<SwitchStmt>(Statement)) {
      checkFullExpression(Switch->getCond());
      analyzeStatement(Switch->getBody());
      return;
    }
    if (const auto* ExprStatement = dyn_cast<Expr>(Statement)) {
      checkFullExpression(ExprStatement);
      return;
    }
    for (const Stmt* Child : Statement->children()) {
      analyzeStatement(Child);
    }
  }

  StatusTransferOrderCheck& Check_;
};

class StatusLifetimeAnalyzer {
 public:
  StatusLifetimeAnalyzer(StatusLifetimeCheck& Check, ASTContext& Context)
      : Check_(Check), Context_(Context) {}

  void analyzeFunction(const FunctionDecl* Function) {
    const Stmt* Body = Function->getBody();
    if (!Body) {
      return;
    }
    GotoFinder Finder;
    if (Finder.containsGoto(Body)) {
      return;
    }
    AnalysisState State;
    analyzeStatement(Body, State, /*is_function_body=*/true);
  }

 private:
  void diagnoseUseAfterConsume(const VarDecl* Var, SourceLocation Location,
                               const VariableState& State) {
    if (State.ConsumeLocation.isValid()) {
      Check_.diag(Location,
                  "iree_status_t %0 is used after its ownership was consumed "
                  "or transferred here")
          << Var << State.ConsumeLocation;
    } else {
      Check_.diag(Location,
                  "iree_status_t %0 is used after its ownership was consumed "
                  "or transferred")
          << Var;
    }
  }

  void diagnoseDoubleConsume(const VarDecl* Var, SourceLocation Location,
                             const VariableState& State) {
    if (State.ConsumeLocation.isValid()) {
      Check_.diag(Location,
                  "iree_status_t %0 is consumed more than once; previous "
                  "consume or transfer was here")
          << Var << State.ConsumeLocation;
    } else {
      Check_.diag(Location, "iree_status_t %0 is consumed more than once")
          << Var;
    }
  }

  void diagnoseReportedIgnore(const VarDecl* Var, SourceLocation Location,
                              const VariableState& State) {
    if (State.ReportLocation.isValid()) {
      Check_.diag(Location,
                  "iree_status_t %0 was reported here before being passed to "
                  "iree_status_ignore; use iree_status_free for statuses that "
                  "have already been handled")
          << Var << State.ReportLocation;
    } else {
      Check_.diag(Location,
                  "iree_status_t %0 was passed to iree_status_ignore after "
                  "being reported; use iree_status_free for statuses that have "
                  "already been handled")
          << Var;
    }
  }

  bool isFileSourceRange(SourceRange Range) const {
    return Range.isValid() && Range.getBegin().isFileID() &&
           Range.getEnd().isFileID();
  }

  std::optional<std::string> sourceText(SourceRange Range) const {
    if (!isFileSourceRange(Range)) {
      return std::nullopt;
    }
    StringRef Text = Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                                          Context_.getSourceManager(),
                                          Context_.getLangOpts());
    if (Text.empty()) {
      return std::nullopt;
    }
    return Text.str();
  }

  std::optional<CharSourceRange> statementRemovalRange(
      const Stmt* Statement) const {
    if (!Statement || !isFileSourceRange(Statement->getSourceRange())) {
      return std::nullopt;
    }
    std::optional<SourceLocation> Begin = statementRemovalBegin(Statement);
    if (!Begin) {
      return std::nullopt;
    }
    std::optional<unsigned> EndOffset = statementRemovalEndOffset(Statement);
    if (!EndOffset) {
      return std::nullopt;
    }
    auto BeginOffset =
        Context_.getSourceManager().getDecomposedLoc(*Begin).second;
    if (*EndOffset <= BeginOffset) {
      return std::nullopt;
    }
    SourceLocation End = Begin->getLocWithOffset(*EndOffset - BeginOffset - 1);
    return CharSourceRange::getCharRange(*Begin, End);
  }

  std::optional<SourceLocation> statementRemovalBegin(
      const Stmt* Statement) const {
    const SourceManager& SourceManager = Context_.getSourceManager();
    SourceLocation Begin = Statement->getBeginLoc();
    if (!Begin.isFileID()) {
      return std::nullopt;
    }
    auto Decomposed = SourceManager.getDecomposedLoc(Begin);
    bool Invalid = false;
    StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
    if (Invalid || Decomposed.second > Buffer.size()) {
      return std::nullopt;
    }

    unsigned LineStart = Decomposed.second;
    while (LineStart > 0 && Buffer[LineStart - 1] != '\n' &&
           Buffer[LineStart - 1] != '\r') {
      --LineStart;
    }
    for (unsigned I = LineStart; I < Decomposed.second; ++I) {
      if (Buffer[I] != ' ' && Buffer[I] != '\t') {
        return std::nullopt;
      }
    }
    return SourceManager.getLocForStartOfFile(Decomposed.first)
        .getLocWithOffset(LineStart);
  }

  std::optional<unsigned> statementRemovalEndOffset(
      const Stmt* Statement) const {
    const SourceManager& SourceManager = Context_.getSourceManager();
    SourceLocation Begin = Statement->getBeginLoc();
    if (!Begin.isFileID()) {
      return std::nullopt;
    }
    auto Decomposed = SourceManager.getDecomposedLoc(Begin);
    bool Invalid = false;
    StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
    if (Invalid || Decomposed.second > Buffer.size()) {
      return std::nullopt;
    }

    for (unsigned I = Decomposed.second; I < Buffer.size(); ++I) {
      if (Buffer[I] == '\r' || Buffer[I] == '\n') {
        return std::nullopt;
      }
      if (Buffer[I] != ';') {
        continue;
      }
      for (unsigned End = I + 1; End < Buffer.size(); ++End) {
        if (Buffer[End] == ' ' || Buffer[End] == '\t') {
          continue;
        }
        if (Buffer[End] == '\r') {
          if (End + 1 < Buffer.size() && Buffer[End + 1] == '\n') {
            return End + 2;
          }
          return End + 1;
        }
        if (Buffer[End] == '\n') {
          return End + 1;
        }
        return std::nullopt;
      }
      return Buffer.size();
    }
    return std::nullopt;
  }

  void appendImmediateOkOverwriteFix(DiagnosticBuilder& Diagnostic,
                                     const BinaryOperator* Assignment,
                                     const VariableState& State) const {
    auto It = ImmediateOkOverwriteFixes_.find(Assignment);
    if (It == ImmediateOkOverwriteFixes_.end()) {
      return;
    }
    std::optional<std::string> ReplacementText =
        sourceText(It->second.ReplacementExpr->getSourceRange());
    std::optional<CharSourceRange> RemovalRange =
        statementRemovalRange(It->second.RemovalStatement);
    if (!ReplacementText || !RemovalRange ||
        !isFileSourceRange(State.OkInitRange)) {
      return;
    }
    Diagnostic << FixItHint::CreateRemoval(*RemovalRange)
               << FixItHint::CreateReplacement(
                      CharSourceRange::getTokenRange(State.OkInitRange),
                      *ReplacementText);
  }

  void diagnoseImmediateOkOverwrite(const VarDecl* Var,
                                    const BinaryOperator* Assignment,
                                    const VariableState& State) {
    if (State.OkInitLocation.isValid()) {
      DiagnosticBuilder Diagnostic = Check_.diag(
          Assignment->getOperatorLoc(),
          "iree_status_t %0 initialized to iree_ok_status here is overwritten "
          "before the initializer is used; initialize from the producer "
          "directly or use a return-if-error helper");
      Diagnostic << Var << State.OkInitLocation;
      appendImmediateOkOverwriteFix(Diagnostic, Assignment, State);
    } else {
      DiagnosticBuilder Diagnostic = Check_.diag(
          Assignment->getOperatorLoc(),
          "iree_status_t %0 initialized to iree_ok_status is overwritten "
          "before the initializer is used; initialize from the producer "
          "directly or use a return-if-error helper");
      Diagnostic << Var;
      appendImmediateOkOverwriteFix(Diagnostic, Assignment, State);
    }
  }

  void diagnoseLeak(const VarDecl* Var, SourceLocation Location,
                    const VariableState& State) {
    if (!State.MayOwn || State.Unknown) {
      return;
    }
    SourceLocation DiagnosticLocation =
        Location.isValid() ? Location : State.OwnershipLocation;
    if (State.OwnershipLocation.isValid() &&
        State.OwnershipLocation != DiagnosticLocation) {
      Check_.diag(DiagnosticLocation,
                  "iree_status_t %0 may leave scope without being returned, "
                  "stored, or consumed; ownership was acquired here")
          << Var << State.OwnershipLocation;
    } else {
      Check_.diag(DiagnosticLocation,
                  "iree_status_t %0 may leave scope without being returned, "
                  "stored, or consumed")
          << Var;
    }
  }

  VariableState* findState(const VarDecl* Var, AnalysisState& State) {
    if (!Var) {
      return nullptr;
    }
    auto It = State.Variables.find(Var->getCanonicalDecl());
    return It == State.Variables.end() ? nullptr : &It->second;
  }

  const VariableState* findState(const VarDecl* Var,
                                 const AnalysisState& State) {
    if (!Var) {
      return nullptr;
    }
    auto It = State.Variables.find(Var->getCanonicalDecl());
    return It == State.Variables.end() ? nullptr : &It->second;
  }

  void useVariable(const VarDecl* Var, SourceLocation Location,
                   AnalysisState& State) {
    VariableState* VarState = findState(Var, State);
    if (!VarState || VarState->Unknown) {
      return;
    }
    VarState->OkInitializerLive = false;
    if (VarState->MayBeConsumed) {
      diagnoseUseAfterConsume(Var, Location, *VarState);
    }
  }

  void reportVariable(const VarDecl* Var, SourceLocation Location,
                      AnalysisState& State) {
    VariableState* VarState = findState(Var, State);
    if (!VarState || VarState->Unknown) {
      return;
    }
    VarState->Reported = true;
    if (!VarState->ReportLocation.isValid()) {
      VarState->ReportLocation = Location;
    }
  }

  void consumeVariable(const VarDecl* Var, SourceLocation Location,
                       AnalysisState& State) {
    VariableState* VarState = findState(Var, State);
    if (!VarState || VarState->Unknown) {
      return;
    }
    VarState->OkInitializerLive = false;
    if (VarState->MayBeConsumed) {
      diagnoseDoubleConsume(Var, Location, *VarState);
    }
    if (VarState->MayOwn) {
      VarState->MayOwn = false;
      VarState->MayBeConsumed = true;
      VarState->ConsumeLocation = Location;
    }
  }

  void ignoreVariable(const VarDecl* Var, SourceLocation Location,
                      AnalysisState& State) {
    VariableState* VarState = findState(Var, State);
    if (VarState && !VarState->Unknown && VarState->Reported) {
      diagnoseReportedIgnore(Var, Location, *VarState);
    }
    consumeVariable(Var, Location, State);
  }

  void storeVariable(const VarDecl* Var, SourceLocation Location,
                     AnalysisState& State) {
    consumeVariable(Var, Location, State);
  }

  StatusValue statusValueForVariable(const VarDecl* Var,
                                     SourceLocation Location,
                                     const AnalysisState& State) {
    if (const VariableState* VarState = findState(Var, State)) {
      if (VarState->Unknown) {
        return StatusValue::UnknownStatus(Location);
      }
      return VarState->MayOwn ? StatusValue::Owned(Location)
                              : StatusValue::NoOwner(Location);
    }
    return StatusValue::UnknownStatus(Location);
  }

  void setVariableFromStatusValue(const VarDecl* Var, const StatusValue& Value,
                                  SourceLocation Location,
                                  AnalysisState& State) {
    if (!Var) {
      return;
    }
    if (!Value.IsStatus) {
      State.Variables[Var] = VariableState::UnknownState();
      return;
    }
    if (Value.Unknown) {
      State.Variables[Var] = VariableState::UnknownState();
      return;
    }
    if (Value.MayOwn) {
      State.Variables[Var] = VariableState::Owned(
          Value.Location.isValid() ? Value.Location : Location);
      return;
    }
    State.Variables[Var] = VariableState::NoOwner();
  }

  void diagnoseOutstandingStatus(AnalysisState& State,
                                 SourceLocation Location) {
    for (const auto& [Var, VarState] : State.Variables) {
      diagnoseLeak(Var, Location, VarState);
    }
  }

  void eraseVariables(ArrayRef<const VarDecl*> Vars, AnalysisState& State) {
    for (const VarDecl* Var : Vars) {
      State.Variables.erase(Var);
    }
  }

  void markLoopTouchedVariablesUnknown(const Stmt* Statement,
                                       AnalysisState& State) {
    StatusRefCollector Collector;
    Collector.collect(Statement);
    for (const VarDecl* Var : Collector.variables()) {
      if (findState(Var, State)) {
        State.Variables[Var] = VariableState::UnknownState();
      }
    }
  }

  void analyzeStatement(const Stmt* Statement, AnalysisState& State,
                        bool is_function_body = false) {
    if (!Statement || State.Terminal) {
      return;
    }
    if (const auto* Compound = dyn_cast<CompoundStmt>(Statement)) {
      analyzeCompound(Compound, State, is_function_body);
      return;
    }
    if (const auto* Decl = dyn_cast<DeclStmt>(Statement)) {
      analyzeDeclStatement(Decl, State);
      return;
    }
    if (const auto* Return = dyn_cast<ReturnStmt>(Statement)) {
      analyzeReturn(Return, State);
      return;
    }
    if (const auto* If = dyn_cast<IfStmt>(Statement)) {
      analyzeIf(If, State);
      return;
    }
    if (isa<BreakStmt>(Statement) || isa<ContinueStmt>(Statement)) {
      State.Terminal = true;
      return;
    }
    if (const auto* For = dyn_cast<ForStmt>(Statement)) {
      analyzeFor(For, State);
      return;
    }
    if (const auto* While = dyn_cast<WhileStmt>(Statement)) {
      analyzeWhile(While, State);
      return;
    }
    if (isa<DoStmt>(Statement) || isa<SwitchStmt>(Statement)) {
      AnalysisState InnerState = State;
      InnerState.SuppressOkOverwriteDiagnostics = true;
      for (const Stmt* Child : Statement->children()) {
        analyzeStatement(Child, InnerState);
      }
      markLoopTouchedVariablesUnknown(Statement, State);
      return;
    }
    if (const auto* ExprStatement = dyn_cast<Expr>(Statement)) {
      StatusValue Value = analyzeExpression(ExprStatement, State);
      (void)Value;
      return;
    }
    for (const Stmt* Child : Statement->children()) {
      analyzeStatement(Child, State);
    }
  }

  void recordImmediateOkOverwriteFix(const Stmt* DeclarationStatement,
                                     const Stmt* AssignmentStatement) {
    const auto* Declaration = dyn_cast_or_null<DeclStmt>(DeclarationStatement);
    if (!Declaration || !Declaration->isSingleDecl()) {
      return;
    }
    const auto* Var = dyn_cast_or_null<VarDecl>(Declaration->getSingleDecl());
    if (!Var || !Var->hasLocalStorage() || IsMacroStatusTemporary(Var) ||
        !Var->hasInit() || !IsStatusType(Var->getType()) ||
        !IsIreeOkStatusCall(Var->getInit())) {
      return;
    }

    const auto* AssignmentExpr = dyn_cast_or_null<Expr>(AssignmentStatement);
    const auto* Assignment =
        dyn_cast_or_null<BinaryOperator>(IgnoreExprNoise(AssignmentExpr));
    if (!Assignment || Assignment->getOpcode() != BO_Assign) {
      return;
    }
    const VarDecl* LhsVar = ReferencedStatusLocal(Assignment->getLHS());
    if (LhsVar != Var->getCanonicalDecl()) {
      return;
    }

    ImmediateOkOverwriteFixes_[Assignment] = ImmediateOkOverwriteFix{
        Assignment->getRHS(),
        AssignmentStatement,
    };
  }

  void analyzeCompound(const CompoundStmt* Compound, AnalysisState& State,
                       bool is_function_body) {
    SmallVector<const VarDecl*, 8> BlockVariables;
    const Stmt* PreviousChild = nullptr;
    for (const Stmt* Child : Compound->body()) {
      if (State.Terminal) {
        break;
      }
      if (PreviousChild) {
        recordImmediateOkOverwriteFix(PreviousChild, Child);
      }
      if (const auto* DeclarationStatement = dyn_cast<DeclStmt>(Child)) {
        for (const Decl* D : DeclarationStatement->decls()) {
          if (const auto* Var = dyn_cast<VarDecl>(D);
              Var && Var->hasLocalStorage() && !IsMacroStatusTemporary(Var) &&
              IsStatusType(Var->getType())) {
            BlockVariables.push_back(Var->getCanonicalDecl());
          }
        }
      }
      analyzeStatement(Child, State);
      PreviousChild = Child;
    }
    if (!State.Terminal) {
      if (is_function_body) {
        diagnoseOutstandingStatus(State, Compound->getEndLoc());
      } else {
        for (const VarDecl* Var : BlockVariables) {
          if (const VariableState* VarState = findState(Var, State)) {
            diagnoseLeak(Var, Compound->getEndLoc(), *VarState);
          }
        }
      }
    }
    if (!is_function_body) {
      eraseVariables(BlockVariables, State);
    }
  }

  void analyzeDeclStatement(const DeclStmt* DeclStatement,
                            AnalysisState& State) {
    for (const Decl* D : DeclStatement->decls()) {
      const auto* Var = dyn_cast<VarDecl>(D);
      if (!Var || !Var->hasLocalStorage()) {
        continue;
      }
      if (!IsStatusType(Var->getType())) {
        if (Var->hasInit()) {
          analyzeExpression(Var->getInit(), State);
        }
        continue;
      }
      if (IsMacroStatusTemporary(Var)) {
        if (Var->hasInit()) {
          if (const VarDecl* InitVar = ForwardedStatusLocal(Var->getInit())) {
            useVariable(InitVar, Var->getInit()->getExprLoc(), State);
            markKnownOk(InitVar, State);
          } else {
            analyzeExpression(Var->getInit(), State);
          }
        }
        continue;
      }
      const VarDecl* CanonicalVar = Var->getCanonicalDecl();
      if (!Var->hasInit()) {
        State.Variables[CanonicalVar] = VariableState::UnknownState();
        continue;
      }
      if (IsIreeOkStatusCall(Var->getInit())) {
        State.Variables[CanonicalVar] = VariableState::OkInitialized(
            Var->getLocation(), Var->getInit()->getSourceRange());
        continue;
      }
      StatusValue InitValue =
          analyzeTransferredExpression(Var->getInit(), State);
      setVariableFromStatusValue(CanonicalVar, InitValue, Var->getLocation(),
                                 State);
    }
  }

  void analyzeReturn(const ReturnStmt* Return, AnalysisState& State) {
    if (const Expr* Value = Return->getRetValue()) {
      StatusValue ReturnValue = analyzeTransferredExpression(Value, State);
      (void)ReturnValue;
    }
    diagnoseOutstandingStatus(State, Return->getBeginLoc());
    State.Terminal = true;
  }

  std::optional<std::pair<const VarDecl*, bool>> statusOkCondition(
      const Expr* Condition) {
    Condition = IgnoreExprNoise(Condition);
    if (!Condition) {
      return std::nullopt;
    }
    if (const auto* Unary = dyn_cast<UnaryOperator>(Condition);
        Unary && Unary->getOpcode() == UO_LNot) {
      if (auto Inner = statusOkCondition(Unary->getSubExpr())) {
        return std::make_pair(Inner->first, !Inner->second);
      }
      if (const VarDecl* Var = ReferencedStatusLocal(Unary->getSubExpr())) {
        return std::make_pair(Var, true);
      }
    }
    if (const auto* Call = dyn_cast<CallExpr>(Condition);
        Call && CalleeName(Call) == "iree_status_is_ok" &&
        Call->getNumArgs() == 1) {
      if (const VarDecl* Var = ReferencedStatusLocal(Call->getArg(0))) {
        return std::make_pair(Var, true);
      }
    }
    if (const auto* Call = dyn_cast<CallExpr>(Condition);
        Call && CalleeName(Call) == "__builtin_expect" &&
        Call->getNumArgs() >= 1) {
      return statusOkCondition(Call->getArg(0));
    }
    if (const auto* Binary = dyn_cast<BinaryOperator>(Condition)) {
      if (Binary->getOpcode() == BO_LAnd) {
        if (auto LhsCondition = statusOkCondition(Binary->getLHS());
            LhsCondition && LhsCondition->second) {
          return LhsCondition;
        }
        if (auto RhsCondition = statusOkCondition(Binary->getRHS());
            RhsCondition && RhsCondition->second) {
          return RhsCondition;
        }
      }
      if (Binary->getOpcode() == BO_LOr) {
        if (auto LhsCondition = statusOkCondition(Binary->getLHS());
            LhsCondition && !LhsCondition->second) {
          return LhsCondition;
        }
        if (auto RhsCondition = statusOkCondition(Binary->getRHS());
            RhsCondition && !RhsCondition->second) {
          return RhsCondition;
        }
      }
      if (Binary->getOpcode() == BO_EQ || Binary->getOpcode() == BO_NE) {
        if (const VarDecl* Var =
                ReferencedStatusLocalThroughCasts(Binary->getLHS())) {
          if (isZeroConstant(Binary->getRHS())) {
            return std::make_pair(Var, Binary->getOpcode() == BO_EQ);
          }
        }
        if (const VarDecl* Var =
                ReferencedStatusLocalThroughCasts(Binary->getRHS())) {
          if (isZeroConstant(Binary->getLHS())) {
            return std::make_pair(Var, Binary->getOpcode() == BO_EQ);
          }
        }
      }
    }
    if (const VarDecl* Var = ReferencedStatusLocal(Condition)) {
      return std::make_pair(Var, false);
    }
    return std::nullopt;
  }

  bool isZeroConstant(const Expr* Expr) {
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return false;
    }
    if (const auto* Literal = dyn_cast<IntegerLiteral>(Expr)) {
      return Literal->getValue().isZero();
    }
    if (const auto* Ref = dyn_cast<DeclRefExpr>(Expr)) {
      if (const auto* Constant = dyn_cast<EnumConstantDecl>(Ref->getDecl())) {
        return Constant->getInitVal().isZero();
      }
    }
    Expr::EvalResult Result;
    if (!Expr->EvaluateAsInt(Result, Context_)) {
      return false;
    }
    return Result.Val.getInt().isZero();
  }

  void markKnownOk(const VarDecl* Var, AnalysisState& State) {
    VariableState* VarState = findState(Var, State);
    if (!VarState || VarState->Unknown) {
      return;
    }
    VarState->MayOwn = false;
  }

  void analyzeIf(const IfStmt* If, AnalysisState& State) {
    if (const Stmt* Init = If->getInit()) {
      analyzeStatement(Init, State);
    }
    if (const DeclStmt* ConditionVariable =
            If->getConditionVariableDeclStmt()) {
      analyzeDeclStatement(ConditionVariable, State);
    }
    analyzeExpression(If->getCond(), State);

    AnalysisState ThenState = State;
    AnalysisState ElseState = State;
    if (auto Condition = statusOkCondition(If->getCond())) {
      const VarDecl* Var = Condition->first;
      bool TrueMeansOk = Condition->second;
      if (TrueMeansOk) {
        markKnownOk(Var, ThenState);
      } else {
        markKnownOk(Var, ElseState);
      }
    }
    if (auto Condition = CompareExchangeStatusCondition(If->getCond())) {
      const VarDecl* Var = Condition->first;
      bool TrueTransfers = Condition->second;
      if (TrueTransfers) {
        consumeVariable(Var, If->getCond()->getExprLoc(), ThenState);
      } else {
        consumeVariable(Var, If->getCond()->getExprLoc(), ElseState);
      }
    }

    const bool SuppressOkOverwriteDiagnostics =
        State.SuppressOkOverwriteDiagnostics;
    ThenState.SuppressOkOverwriteDiagnostics = true;
    analyzeStatement(If->getThen(), ThenState);
    ThenState.SuppressOkOverwriteDiagnostics = SuppressOkOverwriteDiagnostics;
    if (const Stmt* Else = If->getElse()) {
      ElseState.SuppressOkOverwriteDiagnostics = true;
      analyzeStatement(Else, ElseState);
      ElseState.SuppressOkOverwriteDiagnostics = SuppressOkOverwriteDiagnostics;
    }
    State = MergeStates(ThenState, ElseState);
    State.SuppressOkOverwriteDiagnostics = SuppressOkOverwriteDiagnostics;
  }

  void markConditionTrueFacts(const Expr* Condition, AnalysisState& State) {
    if (auto ConditionFact = statusOkCondition(Condition);
        ConditionFact && ConditionFact->second) {
      markKnownOk(ConditionFact->first, State);
    }
  }

  void analyzeFor(const ForStmt* For, AnalysisState& State) {
    AnalysisState InnerState = State;
    analyzeStatement(For->getInit(), InnerState);
    analyzeExpression(For->getCond(), InnerState);

    AnalysisState BodyState = InnerState;
    BodyState.SuppressOkOverwriteDiagnostics = true;
    markConditionTrueFacts(For->getCond(), BodyState);
    analyzeStatement(For->getBody(), BodyState);
    analyzeExpression(For->getInc(), BodyState);

    markLoopTouchedVariablesUnknown(For, State);
  }

  void analyzeWhile(const WhileStmt* While, AnalysisState& State) {
    AnalysisState InnerState = State;
    analyzeExpression(While->getCond(), InnerState);

    AnalysisState BodyState = InnerState;
    BodyState.SuppressOkOverwriteDiagnostics = true;
    markConditionTrueFacts(While->getCond(), BodyState);
    analyzeStatement(While->getBody(), BodyState);

    markLoopTouchedVariablesUnknown(While, State);
  }

  StatusValue analyzeExpression(const Expr* Expr, AnalysisState& State) {
    if (!Expr) {
      return StatusValue::NonStatus();
    }
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return StatusValue::NonStatus();
    }

    if (const VarDecl* Var = ReferencedStatusLocal(Expr)) {
      useVariable(Var, Expr->getExprLoc(), State);
      if (const VariableState* VarState = findState(Var, State)) {
        if (VarState->Unknown) {
          return StatusValue::UnknownStatus(Expr->getExprLoc());
        }
        return VarState->MayOwn ? StatusValue::Owned(Expr->getExprLoc())
                                : StatusValue::NoOwner(Expr->getExprLoc());
      }
      return StatusValue::UnknownStatus(Expr->getExprLoc());
    }

    if (const auto* Cast = dyn_cast<CastExpr>(Expr);
        Cast && Cast->getType()->isVoidType()) {
      analyzeExpression(Cast->getSubExpr(), State);
      return StatusValue::NonStatus();
    }

    if (const auto* Call = dyn_cast<CallExpr>(Expr)) {
      return analyzeCall(Call, State);
    }

    if (const auto* Construct = dyn_cast<CXXConstructExpr>(Expr)) {
      return analyzeCXXConstruct(Construct, State);
    }

    if (const auto* Binary = dyn_cast<BinaryOperator>(Expr)) {
      if (Binary->isAssignmentOp()) {
        return analyzeAssignment(Binary, State);
      }
      analyzeExpression(Binary->getLHS(), State);
      analyzeExpression(Binary->getRHS(), State);
      return IsStatusType(Binary->getType())
                 ? StatusValue::UnknownStatus(Binary->getExprLoc())
                 : StatusValue::NonStatus();
    }

    if (const auto* Unary = dyn_cast<UnaryOperator>(Expr)) {
      analyzeExpression(Unary->getSubExpr(), State);
      return IsStatusType(Unary->getType())
                 ? StatusValue::UnknownStatus(Unary->getExprLoc())
                 : StatusValue::NonStatus();
    }

    if (const auto* Conditional = dyn_cast<AbstractConditionalOperator>(Expr)) {
      analyzeExpression(Conditional->getCond(), State);
      AnalysisState TrueState = State;
      AnalysisState FalseState = State;
      StatusValue TrueValue =
          analyzeExpression(Conditional->getTrueExpr(), TrueState);
      StatusValue FalseValue =
          analyzeExpression(Conditional->getFalseExpr(), FalseState);
      State = MergeStates(TrueState, FalseState);
      if (!TrueValue.IsStatus && !FalseValue.IsStatus) {
        return StatusValue::NonStatus();
      }
      if (TrueValue.Unknown || FalseValue.Unknown) {
        return StatusValue::UnknownStatus(Expr->getExprLoc());
      }
      return (TrueValue.MayOwn || FalseValue.MayOwn)
                 ? StatusValue::Owned(Expr->getExprLoc())
                 : StatusValue::NoOwner(Expr->getExprLoc());
    }

    for (const Stmt* Child : Expr->children()) {
      if (const auto* ChildExpr =
              Child ? dyn_cast<clang::Expr>(Child) : nullptr) {
        analyzeExpression(ChildExpr, State);
      } else {
        analyzeStatement(Child, State);
      }
    }
    return IsStatusType(Expr->getType())
               ? StatusValue::UnknownStatus(Expr->getExprLoc())
               : StatusValue::NonStatus();
  }

  StatusValue analyzeTransferredExpression(const Expr* Expr,
                                           AnalysisState& State) {
    if (!Expr) {
      return StatusValue::NonStatus();
    }
    Expr = IgnoreExprNoise(Expr);
    if (!Expr) {
      return StatusValue::NonStatus();
    }

    if (const VarDecl* Var = ForwardedStatusLocal(Expr)) {
      SourceLocation Location = Expr->getExprLoc();
      useVariable(Var, Location, State);
      StatusValue Value = statusValueForVariable(Var, Location, State);
      consumeVariable(Var, Location, State);
      return Value;
    }

    if (const auto* Conditional = dyn_cast<AbstractConditionalOperator>(Expr)) {
      analyzeExpression(Conditional->getCond(), State);

      AnalysisState TrueState = State;
      AnalysisState FalseState = State;
      if (auto Condition = statusOkCondition(Conditional->getCond())) {
        const VarDecl* Var = Condition->first;
        bool TrueMeansOk = Condition->second;
        if (TrueMeansOk) {
          markKnownOk(Var, TrueState);
        } else {
          markKnownOk(Var, FalseState);
        }
      }

      StatusValue TrueValue =
          analyzeTransferredExpression(Conditional->getTrueExpr(), TrueState);
      StatusValue FalseValue =
          analyzeTransferredExpression(Conditional->getFalseExpr(), FalseState);
      State = MergeStates(TrueState, FalseState);
      if (!TrueValue.IsStatus && !FalseValue.IsStatus) {
        return StatusValue::NonStatus();
      }
      if (TrueValue.Unknown || FalseValue.Unknown) {
        return StatusValue::UnknownStatus(Expr->getExprLoc());
      }
      return (TrueValue.MayOwn || FalseValue.MayOwn)
                 ? StatusValue::Owned(Expr->getExprLoc())
                 : StatusValue::NoOwner(Expr->getExprLoc());
    }

    return analyzeExpression(Expr, State);
  }

  StatusValue analyzeAssignment(const BinaryOperator* Binary,
                                AnalysisState& State) {
    const VarDecl* LhsVar = ReferencedStatusLocal(Binary->getLHS());
    VariableState LhsBefore = LhsVar && findState(LhsVar, State)
                                  ? *findState(LhsVar, State)
                                  : VariableState::NoOwner();
    StatusValue RhsValue =
        analyzeTransferredExpression(Binary->getRHS(), State);
    if (LhsVar) {
      const VariableState* LhsAfterRhs = findState(LhsVar, State);
      bool RhsConsumedLhs =
          LhsAfterRhs && (!LhsBefore.MayOwn || !LhsAfterRhs->MayOwn ||
                          LhsAfterRhs->MayBeConsumed || LhsAfterRhs->Unknown);
      if (!State.SuppressOkOverwriteDiagnostics &&
          LhsBefore.OkInitializerLive && LhsAfterRhs &&
          LhsAfterRhs->OkInitializerLive && RhsValue.IsStatus &&
          !IsIreeOkStatusCall(Binary->getRHS())) {
        diagnoseImmediateOkOverwrite(LhsVar, Binary, LhsBefore);
      }
      if (LhsBefore.MayOwn && !LhsBefore.Unknown && !RhsConsumedLhs) {
        diagnoseLeak(LhsVar, Binary->getOperatorLoc(), LhsBefore);
      }
      setVariableFromStatusValue(LhsVar, RhsValue, Binary->getOperatorLoc(),
                                 State);
    }
    return IsStatusType(Binary->getType())
               ? StatusValue::UnknownStatus(Binary->getExprLoc())
               : StatusValue::NonStatus();
  }

  StatusValue analyzeCall(const CallExpr* Call, AnalysisState& State) {
    StringRef Name = CalleeName(Call);

    if (Name == "iree_atomic_compare_exchange_strong" &&
        Call->getNumArgs() >= 3) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        if (I == 2) {
          if (const VarDecl* Var =
                  ForwardedStatusLocalThroughCasts(Call->getArg(I))) {
            useVariable(Var, Call->getArg(I)->getExprLoc(), State);
            State.Variables[Var] = VariableState::UnknownState();
            continue;
          }
        }
        analyzeExpression(Call->getArg(I), State);
      }
      return StatusValue::NonStatus();
    }

    if (std::optional<unsigned> SinkArg = KnownStatusSinkArgument(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        if (I == *SinkArg) {
          analyzeTransferredExpression(Call->getArg(I), State);
        } else {
          analyzeExpression(Call->getArg(I), State);
        }
      }
      return StatusValue::NonStatus();
    }

    if (IsKnownStatusConsumer(Name) && Call->getNumArgs() >= 1) {
      if (Name == "iree_status_ignore") {
        const Expr* Arg = Call->getArg(0);
        if (const VarDecl* Var = ForwardedStatusLocal(Arg)) {
          SourceLocation Location = Arg->getExprLoc();
          useVariable(Var, Location, State);
          ignoreVariable(Var, Location, State);
        } else {
          analyzeTransferredExpression(Arg, State);
        }
      } else {
        analyzeTransferredExpression(Call->getArg(0), State);
      }
      for (unsigned I = 1; I < Call->getNumArgs(); ++I) {
        analyzeExpression(Call->getArg(I), State);
      }
      if (IsNoReturnStatusConsumer(Name)) {
        diagnoseOutstandingStatus(State, Call->getExprLoc());
        State.Terminal = true;
      }
      if (IsNamedTypedef(ReturnTypeFromCall(Call), "iree_status_t")) {
        return StatusValue::NoOwner(Call->getExprLoc());
      }
      return StatusValue::NonStatus();
    }

    if (IsKnownStatusTransferProducer(Name)) {
      bool ResultMayOwn = false;
      bool ResultUnknown = false;
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        bool ConsumesArg = false;
        if (Name == "iree_status_join") {
          ConsumesArg = I == 0 || I == 1;
        } else {
          ConsumesArg = I == 0;
        }
        StatusValue ArgValue =
            ConsumesArg ? analyzeTransferredExpression(Call->getArg(I), State)
                        : analyzeExpression(Call->getArg(I), State);
        if (ConsumesArg) {
          ResultMayOwn = ResultMayOwn || ArgValue.MayOwn;
          ResultUnknown = ResultUnknown || ArgValue.Unknown;
        }
      }
      if (ResultUnknown) {
        return StatusValue::UnknownStatus(Call->getExprLoc());
      }
      if (Name == "iree_status_annotate" || Name == "iree_status_annotate_f") {
        ResultMayOwn = true;
      }
      return ResultMayOwn ? StatusValue::Owned(Call->getExprLoc())
                          : StatusValue::NoOwner(Call->getExprLoc());
    }

    if (IsKnownStatusClone(Name)) {
      for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
        analyzeExpression(Call->getArg(I), State);
      }
      return StatusValue::Owned(Call->getExprLoc());
    }

    bool HasUnknownStatusArgument = false;
    for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
      StatusValue ArgValue = analyzeExpression(Call->getArg(I), State);
      if (IsStatusReportingObserver(Name)) {
        if (const VarDecl* Var =
                ReferencedStatusLocalThroughCasts(Call->getArg(I))) {
          reportVariable(Var, Call->getArg(I)->getExprLoc(), State);
        }
      }
      QualType ParamType = ParameterTypeFromCall(Call, I);
      const bool ParamIsStatus =
          ParamType.isNull() ? ArgValue.IsStatus : IsStatusType(ParamType);
      const bool ParamIsBorrowedStatus =
          !ParamType.isNull() && IsBorrowedStatusType(ParamType);
      if (ParamIsStatus && !ParamIsBorrowedStatus) {
        if (!IsStatusObserver(Name)) {
          HasUnknownStatusArgument = true;
          if (const VarDecl* Var = ForwardedStatusLocal(Call->getArg(I))) {
            State.Variables[Var] = VariableState::UnknownState();
          }
        }
      }
    }

    if (IsNoReturnCall(Call)) {
      diagnoseOutstandingStatus(State, Call->getExprLoc());
      State.Terminal = true;
      return StatusValue::NonStatus();
    }

    if (!IsNamedTypedef(ReturnTypeFromCall(Call), "iree_status_t")) {
      return StatusValue::NonStatus();
    }
    if (IsKnownStatusNoOwnerProducer(Name)) {
      return StatusValue::NoOwner(Call->getExprLoc());
    }
    if (HasUnknownStatusArgument) {
      return StatusValue::UnknownStatus(Call->getExprLoc());
    }
    return StatusValue::Owned(Call->getExprLoc());
  }

  StatusValue analyzeCXXConstruct(const CXXConstructExpr* Construct,
                                  AnalysisState& State) {
    const CXXConstructorDecl* Constructor = Construct->getConstructor();
    const bool IsStatusConstructor =
        Constructor && Constructor->getParent() &&
        (Constructor->getParent()->getName() == "Status" ||
         Constructor->getParent()->getName() == "StatusOr");
    for (unsigned I = 0; I < Construct->getNumArgs(); ++I) {
      StatusValue ArgValue =
          IsStatusConstructor
              ? analyzeTransferredExpression(Construct->getArg(I), State)
              : analyzeExpression(Construct->getArg(I), State);
      (void)ArgValue;
    }
    return StatusValue::NonStatus();
  }

  StatusLifetimeCheck& Check_;
  ASTContext& Context_;
  llvm::DenseMap<const BinaryOperator*, ImmediateOkOverwriteFix>
      ImmediateOkOverwriteFixes_;
};

}  // namespace

DiscardedStatusCheck::DiscardedStatusCheck(StringRef Name,
                                           ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void DiscardedStatusCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      expr(ignoringParenCasts(
               callExpr(callee(functionDecl())).bind("status_call")),
           hasParent(compoundStmt()))
          .bind("discarded_expr"),
      this);
}

void DiscardedStatusCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Call = Result.Nodes.getNodeAs<CallExpr>("status_call");
  if (!Call) {
    return;
  }
  if (!IsNamedTypedef(ReturnTypeFromCall(Call), "iree_status_t")) {
    return;
  }
  if (IsAllowedExplicitStatusConsumer(Call)) {
    return;
  }

  if (StringRef Name = CalleeName(Call); !Name.empty()) {
    diag(Call->getExprLoc(),
         "iree_status_t result from %0 must be returned, stored for later "
         "consumption, or explicitly consumed")
        << Name;
    return;
  }
  diag(Call->getExprLoc(),
       "iree_status_t result must be returned, stored for later consumption, "
       "or explicitly consumed");
}

StatusLifetimeCheck::StatusLifetimeCheck(StringRef Name,
                                         ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void StatusLifetimeCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(functionDecl(isDefinition()).bind("function"), this);
}

void StatusLifetimeCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function || !Function->hasBody()) {
    return;
  }
  StatusLifetimeAnalyzer Analyzer(*this, *Result.Context);
  Analyzer.analyzeFunction(Function);
}

StatusTransferOrderCheck::StatusTransferOrderCheck(StringRef Name,
                                                   ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void StatusTransferOrderCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(functionDecl(isDefinition()).bind("function"), this);
}

void StatusTransferOrderCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function || !Function->hasBody()) {
    return;
  }
  StatusTransferOrderAnalyzer Analyzer(*this);
  Analyzer.analyzeFunction(Function);
}

BorrowedStatusParameterCheck::BorrowedStatusParameterCheck(
    StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void BorrowedStatusParameterCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(functionDecl(isDefinition()).bind("function"), this);
}

void BorrowedStatusParameterCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function || !Function->hasBody()) {
    return;
  }
  BorrowedStatusParameterAnalyzer Analyzer(*this);
  Analyzer.analyzeFunction(Function);
}

}  // namespace clang::tidy::iree
