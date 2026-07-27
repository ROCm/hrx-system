// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/TraceChecks.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

struct TraceMacro {
  TraceMacroKind Kind = TraceMacroKind::kNone;
  StringRef Name;
  std::string ZoneId;
};

struct TraceZone {
  std::string Id;
  llvm::SmallVector<std::string, 4> Aliases;
  SourceLocation BeginLocation;
};

struct TraceState {
  llvm::SmallVector<TraceZone, 8> Zones;
  llvm::SmallVector<std::string, 8> KnownZoneIds;
  bool Terminal = false;
  bool Unknown = false;
  bool SuppressEndDiagnostics = false;
};

enum class EndZoneResult {
  kValid,
  kInvalid,
  kDynamic,
};

std::string Trim(std::string Text) {
  StringRef Ref(Text);
  Ref = Ref.trim();
  return Ref.str();
}

TraceMacroKind TraceMacroKindForName(StringRef Name) {
  if (Name == "IREE_TRACE_ZONE_BEGIN" ||
      Name == "IREE_TRACE_ZONE_BEGIN_NAMED" ||
      Name == "IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC" ||
      Name == "IREE_TRACE_ZONE_BEGIN_EXTERNAL" ||
      Name == "IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN" ||
      Name == "HRX_TRACE_ZONE_BEGIN") {
    return TraceMacroKind::kBegin;
  }
  if (Name == "IREE_TRACE_ZONE_END" || Name == "HRX_TRACE_ZONE_END") {
    return TraceMacroKind::kEnd;
  }
  if (Name == "HIP_RETURN_ERROR") {
    return TraceMacroKind::kReturn;
  }
  if (Name == "IREE_RETURN_IF_ERROR" || Name == "HRX_RETURN_IF_IREE_ERROR") {
    return TraceMacroKind::kReturnIfError;
  }
  if (Name == "IREE_RETURN_AND_END_ZONE_IF_ERROR" ||
      Name == "HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR" ||
      Name == "HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR" ||
      Name == "HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_2" ||
      Name == "HIP_RETURN_STATUS_AND_END_ZONE_IF_ERROR_3") {
    return TraceMacroKind::kReturnAndEndIfError;
  }
  if (Name == "IREE_RETURN_AND_END_ZONE" || Name == "HRX_RETURN_AND_END_ZONE" ||
      Name == "HRX_RETURN_VOID_AND_END_ZONE") {
    return TraceMacroKind::kReturnAndEnd;
  }
  if (Name == "IREE_TRACE_ZONE_ADOPT") {
    return TraceMacroKind::kAdopt;
  }
  if (Name == "IREE_TRACE_ZONE_TRANSFER") {
    return TraceMacroKind::kTransfer;
  }
  return TraceMacroKind::kNone;
}

std::string FirstMacroArgument(const MacroArgs* Args, const Preprocessor& PP) {
  if (!Args || Args->getNumMacroArguments() == 0) {
    return "";
  }
  const Token* Arg = Args->getUnexpArgument(0);
  if (!Arg || MacroArgs::getArgLength(Arg) == 0) {
    return "";
  }
  bool Invalid = false;
  std::string Spelling = PP.getSpelling(Arg[0], &Invalid);
  return Invalid ? "" : Trim(std::move(Spelling));
}

class TraceMacroRecorder final : public PPCallbacks {
 public:
  TraceMacroRecorder(TraceZoneCheck& Check, Preprocessor& PP)
      : Check_(Check), PP_(PP) {}

  void MacroExpands(const Token& MacroNameTok, const MacroDefinition&,
                    SourceRange Range, const MacroArgs* Args) override {
    const IdentifierInfo* Identifier = MacroNameTok.getIdentifierInfo();
    if (!Identifier) {
      return;
    }
    StringRef Name = Identifier->getName();
    TraceMacroKind Kind = TraceMacroKindForName(Name);
    if (Kind == TraceMacroKind::kNone) {
      return;
    }
    const SourceManager& SourceManager = PP_.getSourceManager();
    if (SourceManager.isMacroBodyExpansion(MacroNameTok.getLocation())) {
      return;
    }
    SourceLocation BeginLocation =
        SourceManager.getExpansionLoc(Range.getBegin());
    if (SourceManager.getExpansionLoc(MacroNameTok.getLocation()) !=
        BeginLocation) {
      return;
    }
    SourceLocation EndLocation = SourceManager.getExpansionLoc(Range.getEnd());
    if (EndLocation.isInvalid()) {
      EndLocation = BeginLocation;
    }
    Check_.addTraceMacroExpansion(Kind, Name, FirstMacroArgument(Args, PP_),
                                  BeginLocation, EndLocation);
  }

 private:
  TraceZoneCheck& Check_;
  Preprocessor& PP_;
};

class TraceZoneAnalyzer {
 public:
  TraceZoneAnalyzer(TraceZoneCheck& Check, ASTContext& Context)
      : Check_(Check), Context_(Context) {
    ConsumedMacroExpansions_.resize(Check_.traceMacroExpansions().size(), 0);
  }

  void analyzeFunction(const FunctionDecl* Function) {
    const Stmt* Body = Function->getBody();
    if (!Body) {
      return;
    }
    TraceState State;
    if (const auto* Compound = dyn_cast<CompoundStmt>(Body)) {
      analyzeFunctionCompound(Compound, State);
    } else {
      analyzeStatement(Body, State);
    }
    if (!State.Terminal && !State.Zones.empty()) {
      diagnoseActiveFunctionEnd(Body->getEndLoc(), State);
    }
  }

 private:
  bool isTraceMacroContainer(const Stmt* Statement) const {
    return isa<CompoundStmt>(Statement) || isa<IfStmt>(Statement) ||
           isa<ForStmt>(Statement) || isa<WhileStmt>(Statement) ||
           isa<SwitchStmt>(Statement) || isa<LabelStmt>(Statement);
  }

  SourceLocation expansionLocation(SourceLocation Location) const {
    const SourceManager& SourceManager = Context_.getSourceManager();
    return Location.isMacroID() ? SourceManager.getTopMacroCallerLoc(Location)
                                : SourceManager.getExpansionLoc(Location);
  }

  bool rangesOverlap(SourceLocation LhsBegin, SourceLocation LhsEnd,
                     SourceLocation RhsBegin, SourceLocation RhsEnd) const {
    if (LhsBegin.isInvalid() || LhsEnd.isInvalid() || RhsBegin.isInvalid() ||
        RhsEnd.isInvalid()) {
      return false;
    }
    const SourceManager& SourceManager = Context_.getSourceManager();
    if (SourceManager.isBeforeInTranslationUnit(LhsEnd, LhsBegin)) {
      std::swap(LhsBegin, LhsEnd);
    }
    if (SourceManager.isBeforeInTranslationUnit(RhsEnd, RhsBegin)) {
      std::swap(RhsBegin, RhsEnd);
    }
    return !SourceManager.isBeforeInTranslationUnit(LhsEnd, RhsBegin) &&
           !SourceManager.isBeforeInTranslationUnit(RhsEnd, LhsBegin);
  }

  bool sameExpansionLine(SourceLocation Lhs, SourceLocation Rhs) const {
    if (Lhs.isInvalid() || Rhs.isInvalid()) {
      return false;
    }
    const SourceManager& SourceManager = Context_.getSourceManager();
    Lhs = Lhs.isMacroID() ? SourceManager.getTopMacroCallerLoc(Lhs)
                          : SourceManager.getExpansionLoc(Lhs);
    Rhs = Rhs.isMacroID() ? SourceManager.getTopMacroCallerLoc(Rhs)
                          : SourceManager.getExpansionLoc(Rhs);
    if (SourceManager.getFileID(Lhs) != SourceManager.getFileID(Rhs)) {
      return false;
    }
    return SourceManager.getExpansionLineNumber(Lhs) ==
           SourceManager.getExpansionLineNumber(Rhs);
  }

  int traceMacroPriority(TraceMacroKind Kind) const {
    switch (Kind) {
      case TraceMacroKind::kNone:
        return 0;
      case TraceMacroKind::kReturnAndEnd:
      case TraceMacroKind::kReturnAndEndIfError:
        return 4;
      case TraceMacroKind::kBegin:
      case TraceMacroKind::kEnd:
        return 3;
      case TraceMacroKind::kReturnIfError:
        return 2;
      case TraceMacroKind::kReturn:
        return 1;
      case TraceMacroKind::kAdopt:
      case TraceMacroKind::kTransfer:
        return 3;
    }
    return 0;
  }

  const TraceMacroExpansion* findTraceMacroExpansionForStatement(
      const Stmt* Statement) {
    if (!Statement || isTraceMacroContainer(Statement)) {
      return nullptr;
    }
    SourceRange StatementRange = Statement->getSourceRange();
    SourceLocation StatementBegin =
        expansionLocation(StatementRange.getBegin());
    SourceLocation StatementEnd = expansionLocation(StatementRange.getEnd());
    if (StatementEnd.isInvalid()) {
      StatementEnd = StatementBegin;
    }
    ArrayRef<TraceMacroExpansion> Expansions = Check_.traceMacroExpansions();
    size_t BestIndex = Expansions.size();
    int BestPriority = 0;
    for (size_t I = 0; I < Expansions.size(); ++I) {
      if (ConsumedMacroExpansions_[I]) {
        continue;
      }
      const TraceMacroExpansion& Expansion = Expansions[I];
      if (!rangesOverlap(StatementBegin, StatementEnd, Expansion.BeginLocation,
                         Expansion.EndLocation) &&
          !sameExpansionLine(StatementBegin, Expansion.BeginLocation)) {
        continue;
      }
      int Priority = traceMacroPriority(Expansion.Kind);
      if (Priority > BestPriority) {
        BestIndex = I;
        BestPriority = Priority;
      }
    }
    if (BestIndex != Expansions.size()) {
      ConsumedMacroExpansions_[BestIndex] = 1;
      return &Expansions[BestIndex];
    }
    return nullptr;
  }

  TraceMacro traceMacroForStatement(const Stmt* Statement) {
    const TraceMacroExpansion* Expansion =
        findTraceMacroExpansionForStatement(Statement);
    if (Expansion) {
      return TraceMacro{Expansion->Kind, Expansion->Name, Expansion->ZoneId};
    }
    return TraceMacro{};
  }

  const TraceZone* activeZone(const TraceState& State) const {
    return State.Zones.empty() ? nullptr : &State.Zones.back();
  }

  bool zoneHasAlias(const TraceZone& Zone, StringRef ZoneId) const {
    for (StringRef Alias : Zone.Aliases) {
      if (Alias == ZoneId) {
        return true;
      }
    }
    return false;
  }

  bool zonesEquivalent(const TraceZone& Lhs, const TraceZone& Rhs) const {
    for (StringRef LhsAlias : Lhs.Aliases) {
      if (zoneHasAlias(Rhs, LhsAlias)) {
        return true;
      }
    }
    return false;
  }

  bool isKnownZoneId(const TraceState& State, StringRef ZoneId) const {
    for (StringRef KnownZoneId : State.KnownZoneIds) {
      if (KnownZoneId == ZoneId) {
        return true;
      }
    }
    return false;
  }

  void addKnownZoneId(TraceState& State, StringRef ZoneId) {
    if (ZoneId.empty() || ZoneId == "<unknown>" ||
        isKnownZoneId(State, ZoneId)) {
      return;
    }
    State.KnownZoneIds.push_back(ZoneId.str());
  }

  TraceZone* findZoneByAlias(TraceState& State, StringRef ZoneId) {
    for (TraceZone& Zone : llvm::reverse(State.Zones)) {
      if (zoneHasAlias(Zone, ZoneId)) {
        return &Zone;
      }
    }
    return nullptr;
  }

  void addZoneAlias(TraceState& State, StringRef SourceId, StringRef TargetId) {
    if (State.Unknown || SourceId.empty() || TargetId.empty() ||
        TargetId == "<unknown>") {
      return;
    }
    TraceZone* Zone = findZoneByAlias(State, SourceId);
    if (!Zone || zoneHasAlias(*Zone, TargetId)) {
      return;
    }
    Zone->Aliases.push_back(TargetId.str());
    addKnownZoneId(State, TargetId);
  }

  void removeZoneAlias(TraceState& State, StringRef ZoneId) {
    if (State.Unknown || ZoneId.empty()) {
      return;
    }
    for (TraceZone& Zone : State.Zones) {
      llvm::erase_if(Zone.Aliases,
                     [&](const std::string& Alias) { return Alias == ZoneId; });
    }
  }

  void diagnoseActiveReturn(SourceLocation Location, const TraceState& State,
                            StringRef ReturnKind) {
    const TraceZone* Zone = activeZone(State);
    if (!Zone || State.Unknown) {
      return;
    }
    if (Zone->BeginLocation.isValid()) {
      Check_.diag(Location,
                  "%0 exits with active trace zone %1; end the zone or use a "
                  "trace-zone return helper")
          << ReturnKind << Zone->Id << Zone->BeginLocation;
    } else {
      Check_.diag(Location,
                  "%0 exits with active trace zone %1; end the zone or use a "
                  "trace-zone return helper")
          << ReturnKind << Zone->Id;
    }
  }

  void diagnoseActiveFunctionEnd(SourceLocation Location,
                                 const TraceState& State) {
    const TraceZone* Zone = activeZone(State);
    if (!Zone || State.Unknown) {
      return;
    }
    if (Zone->BeginLocation.isValid()) {
      Check_.diag(Location,
                  "function exits with active trace zone %0; end the zone "
                  "before the function exits")
          << Zone->Id << Zone->BeginLocation;
    } else {
      Check_.diag(Location,
                  "function exits with active trace zone %0; end the zone "
                  "before the function exits")
          << Zone->Id;
    }
  }

  void diagnoseUnmatchedEnd(SourceLocation Location, const TraceMacro& Macro) {
    Check_.diag(Location, "%0 ends trace zone %1 but no trace zone is active")
        << Macro.Name << Macro.ZoneId;
  }

  void diagnoseMismatchedEnd(SourceLocation Location, const TraceState& State,
                             const TraceMacro& Macro) {
    const TraceZone* Zone = activeZone(State);
    if (!Zone || State.Unknown) {
      return;
    }
    if (Zone->BeginLocation.isValid()) {
      Check_.diag(Location,
                  "%0 ends trace zone %1 but the active trace zone is %2")
          << Macro.Name << Macro.ZoneId << Zone->Id << Zone->BeginLocation;
    } else {
      Check_.diag(Location,
                  "%0 ends trace zone %1 but the active trace zone is %2")
          << Macro.Name << Macro.ZoneId << Zone->Id;
    }
  }

  void beginZone(const TraceMacro& Macro, SourceLocation Location,
                 TraceState& State) {
    std::string ZoneId = Macro.ZoneId.empty() ? "<unknown>" : Macro.ZoneId;
    addKnownZoneId(State, ZoneId);
    TraceZone Zone;
    Zone.Id = std::move(ZoneId);
    Zone.Aliases.push_back(Zone.Id);
    Zone.BeginLocation = Location;
    State.Zones.push_back(std::move(Zone));
  }

  EndZoneResult validateEndZone(const TraceMacro& Macro,
                                SourceLocation Location,
                                const TraceState& State) {
    if (State.Unknown) {
      return EndZoneResult::kDynamic;
    }
    if (State.Zones.empty()) {
      if (Macro.ZoneId.empty() || !isKnownZoneId(State, Macro.ZoneId)) {
        return EndZoneResult::kDynamic;
      }
      if (!State.SuppressEndDiagnostics) {
        diagnoseUnmatchedEnd(Location, Macro);
      }
      return EndZoneResult::kInvalid;
    }
    if (Macro.ZoneId.empty()) {
      return EndZoneResult::kValid;
    }
    const TraceZone& Zone = State.Zones.back();
    if (Zone.Id == "<unknown>" || zoneHasAlias(Zone, Macro.ZoneId)) {
      return EndZoneResult::kValid;
    }
    if (!isKnownZoneId(State, Macro.ZoneId)) {
      return EndZoneResult::kDynamic;
    }
    if (!State.SuppressEndDiagnostics) {
      diagnoseMismatchedEnd(Location, State, Macro);
    }
    return EndZoneResult::kInvalid;
  }

  void markUnknown(TraceState& State) {
    State.Unknown = true;
    State.Zones.clear();
  }

  void endZone(const TraceMacro& Macro, SourceLocation Location,
               TraceState& State) {
    if (State.Unknown) {
      return;
    }
    EndZoneResult Result = validateEndZone(Macro, Location, State);
    if (Result == EndZoneResult::kInvalid ||
        Result == EndZoneResult::kDynamic) {
      markUnknown(State);
      return;
    }
    State.Zones.pop_back();
  }

  void adoptZone(const TraceMacro& Macro, SourceLocation Location,
                 TraceState& State) {
    beginZone(Macro, Location, State);
  }

  void transferZone(const TraceMacro& Macro, SourceLocation Location,
                    TraceState& State) {
    endZone(Macro, Location, State);
  }

  void validateEndZoneWithoutPop(const TraceMacro& Macro,
                                 SourceLocation Location, TraceState& State) {
    if (State.Unknown) {
      return;
    }
    EndZoneResult Result = validateEndZone(Macro, Location, State);
    if (Result == EndZoneResult::kInvalid) {
      markUnknown(State);
    }
  }

  bool sameZones(const TraceState& Lhs, const TraceState& Rhs) const {
    if (Lhs.Zones.size() != Rhs.Zones.size()) {
      return false;
    }
    for (size_t I = 0; I < Lhs.Zones.size(); ++I) {
      if (Lhs.Zones[I].Id != Rhs.Zones[I].Id) {
        if (zonesEquivalent(Lhs.Zones[I], Rhs.Zones[I])) {
          continue;
        }
        return false;
      }
    }
    return true;
  }

  void addZoneAliases(TraceZone& Target, const TraceZone& Source) {
    for (StringRef Alias : Source.Aliases) {
      if (!zoneHasAlias(Target, Alias)) {
        Target.Aliases.push_back(Alias.str());
      }
    }
  }

  TraceState mergeEquivalentStates(const TraceState& ThenState,
                                   const TraceState& ElseState) {
    TraceState MergedState = ThenState;
    for (size_t I = 0; I < MergedState.Zones.size(); ++I) {
      addZoneAliases(MergedState.Zones[I], ElseState.Zones[I]);
    }
    for (StringRef KnownZoneId : ElseState.KnownZoneIds) {
      addKnownZoneId(MergedState, KnownZoneId);
    }
    return MergedState;
  }

  const TraceZone* firstDifferentZone(const TraceState& Lhs,
                                      const TraceState& Rhs) const {
    size_t CommonSize = std::min(Lhs.Zones.size(), Rhs.Zones.size());
    for (size_t I = 0; I < CommonSize; ++I) {
      if (Lhs.Zones[I].Id != Rhs.Zones[I].Id) {
        return &Rhs.Zones[I];
      }
    }
    if (Rhs.Zones.size() > Lhs.Zones.size()) {
      return &Rhs.Zones[CommonSize];
    }
    if (Lhs.Zones.size() > Rhs.Zones.size()) {
      return &Lhs.Zones[CommonSize];
    }
    return nullptr;
  }

  void diagnoseBranchStackMismatch(SourceLocation Location,
                                   const TraceState& LhsState,
                                   const TraceState& RhsState,
                                   StringRef BranchName) {
    if (LhsState.Terminal || RhsState.Terminal || LhsState.Unknown ||
        RhsState.Unknown || sameZones(LhsState, RhsState)) {
      return;
    }
    const TraceZone* Zone = firstDifferentZone(LhsState, RhsState);
    if (!Zone) {
      return;
    }
    if (RhsState.Zones.size() > LhsState.Zones.size()) {
      if (Zone->BeginLocation.isValid()) {
        Check_.diag(Location,
                    "%0 branch falls through with trace zone %1 still active; "
                    "end branch-local trace zones before branch exit")
            << BranchName << Zone->Id << Zone->BeginLocation;
      } else {
        Check_.diag(Location,
                    "%0 branch falls through with trace zone %1 still active; "
                    "end branch-local trace zones before branch exit")
            << BranchName << Zone->Id;
      }
      return;
    }
    if (Zone->BeginLocation.isValid()) {
      Check_.diag(Location,
                  "%0 branch changes trace-zone stack around trace zone %1 "
                  "before branch exit; fallthrough branches must preserve "
                  "active trace zones")
          << BranchName << Zone->Id << Zone->BeginLocation;
    } else {
      Check_.diag(Location,
                  "%0 branch changes trace-zone stack around trace zone %1 "
                  "before branch exit; fallthrough branches must preserve "
                  "active trace zones")
          << BranchName << Zone->Id;
    }
  }

  TraceState mergeStates(const TraceState& EntryState,
                         const TraceState& ThenState,
                         const TraceState& ElseState) {
    if (ThenState.Terminal && ElseState.Terminal) {
      TraceState Terminal = EntryState;
      Terminal.Terminal = true;
      return Terminal;
    }
    if (ThenState.Terminal) {
      return ElseState;
    }
    if (ElseState.Terminal) {
      return ThenState;
    }
    if (ThenState.Unknown || ElseState.Unknown) {
      TraceState Unknown = EntryState;
      Unknown.Unknown = true;
      Unknown.Zones.clear();
      return Unknown;
    }
    if (!sameZones(ThenState, ElseState)) {
      TraceState Unknown = EntryState;
      Unknown.Unknown = true;
      Unknown.Zones.clear();
      return Unknown;
    }
    return mergeEquivalentStates(ThenState, ElseState);
  }

  bool handleTraceMacro(const Stmt* Statement, TraceState& State) {
    TraceMacro Macro = traceMacroForStatement(Statement);
    switch (Macro.Kind) {
      case TraceMacroKind::kNone:
        return false;
      case TraceMacroKind::kBegin:
        beginZone(Macro, Statement->getBeginLoc(), State);
        return true;
      case TraceMacroKind::kEnd:
        endZone(Macro, Statement->getBeginLoc(), State);
        return true;
      case TraceMacroKind::kReturn:
        if (!State.Zones.empty()) {
          diagnoseActiveReturn(Statement->getBeginLoc(), State, Macro.Name);
        }
        State.Terminal = true;
        return true;
      case TraceMacroKind::kReturnIfError:
        if (!State.Zones.empty()) {
          diagnoseActiveReturn(Statement->getBeginLoc(), State, Macro.Name);
        }
        // The macro may continue on success, so active zones stay open.
        return true;
      case TraceMacroKind::kReturnAndEndIfError:
        // The macro is a conditional failure path: it ends the zone only on
        // failure and falls through with the zone still active on success.
        validateEndZoneWithoutPop(Macro, Statement->getBeginLoc(), State);
        return true;
      case TraceMacroKind::kReturnAndEnd:
        endZone(Macro, Statement->getBeginLoc(), State);
        State.Terminal = true;
        return true;
      case TraceMacroKind::kAdopt:
        adoptZone(Macro, Statement->getBeginLoc(), State);
        return true;
      case TraceMacroKind::kTransfer:
        transferZone(Macro, Statement->getBeginLoc(), State);
        return true;
    }
    return false;
  }

  std::string zoneIdFromExpr(const Expr* Expression) const {
    if (!Expression) {
      return "";
    }
    Expression = Expression->IgnoreParenImpCasts();
    if (const auto* DeclRef = dyn_cast<DeclRefExpr>(Expression)) {
      return DeclRef->getDecl()->getNameAsString();
    }
    return "";
  }

  bool isZeroLiteral(const Expr* Expression) const {
    if (!Expression) {
      return false;
    }
    Expression = Expression->IgnoreParenImpCasts();
    if (const auto* Integer = dyn_cast<IntegerLiteral>(Expression)) {
      return Integer->getValue() == 0;
    }
    return false;
  }

  bool handleZoneAssignment(const Stmt* Statement, TraceState& State) {
    const auto* Assignment = dyn_cast<BinaryOperator>(Statement);
    if (!Assignment || !Assignment->isAssignmentOp()) {
      return false;
    }
    std::string TargetId = zoneIdFromExpr(Assignment->getLHS());
    if (TargetId.empty()) {
      return false;
    }
    if (isZeroLiteral(Assignment->getRHS())) {
      removeZoneAlias(State, TargetId);
      return true;
    }
    std::string SourceId = zoneIdFromExpr(Assignment->getRHS());
    if (SourceId.empty()) {
      return false;
    }
    addZoneAlias(State, SourceId, TargetId);
    return true;
  }

  void analyzeStatement(const Stmt* Statement, TraceState& State) {
    if (!Statement || State.Terminal) {
      return;
    }
    if (handleTraceMacro(Statement, State)) {
      return;
    }
    if (handleZoneAssignment(Statement, State)) {
      return;
    }
    if (const auto* Compound = dyn_cast<CompoundStmt>(Statement)) {
      analyzeCompound(Compound, State);
      return;
    }
    if (isa<ReturnStmt>(Statement)) {
      if (!Statement->getBeginLoc().isMacroID() && !State.Zones.empty()) {
        diagnoseActiveReturn(Statement->getBeginLoc(), State, "return");
      }
      State.Terminal = true;
      return;
    }
    if (const auto* If = dyn_cast<IfStmt>(Statement)) {
      analyzeIf(If, State);
      return;
    }
    if (const auto* Label = dyn_cast<LabelStmt>(Statement)) {
      analyzeStatement(Label->getSubStmt(), State);
      return;
    }
    if (isa<GotoStmt>(Statement) || isa<IndirectGotoStmt>(Statement)) {
      return;
    }
    if (isa<BreakStmt>(Statement) || isa<ContinueStmt>(Statement)) {
      return;
    }
    if (const auto* For = dyn_cast<ForStmt>(Statement)) {
      analyzeLoopLike(For->getBody(), State,
                      isInfiniteLoopCondition(For->getCond()));
      return;
    }
    if (const auto* While = dyn_cast<WhileStmt>(Statement)) {
      analyzeLoopLike(While->getBody(), State,
                      isInfiniteLoopCondition(While->getCond()));
      return;
    }
    if (const auto* Do = dyn_cast<DoStmt>(Statement)) {
      analyzeLoopLike(Do->getBody(), State,
                      isInfiniteLoopCondition(Do->getCond()));
      return;
    }
    if (const auto* Switch = dyn_cast<SwitchStmt>(Statement)) {
      analyzeLoopLike(Switch->getBody(), State,
                      /*LoopCannotFallThrough=*/false);
      return;
    }
    for (const Stmt* Child : Statement->children()) {
      analyzeStatement(Child, State);
    }
  }

  void analyzeCompound(const CompoundStmt* Compound, TraceState& State) {
    for (const Stmt* Child : Compound->body()) {
      if (State.Terminal) {
        break;
      }
      analyzeStatement(Child, State);
    }
  }

  void analyzeFunctionCompound(const CompoundStmt* Compound,
                               TraceState& State) {
    for (const Stmt* Child : Compound->body()) {
      if (State.Terminal) {
        break;
      }
      analyzeStatement(Child, State);
    }
  }

  void analyzeIf(const IfStmt* If, TraceState& State) {
    TraceState EntryState = State;
    TraceState ThenState = State;
    analyzeStatement(If->getThen(), ThenState);

    TraceState ElseState = State;
    if (const Stmt* Else = If->getElse()) {
      analyzeStatement(Else, ElseState);
      diagnoseBranchStackMismatch(If->getEndLoc(), ThenState, ElseState, "if");
    } else {
      diagnoseBranchStackMismatch(If->getThen()->getEndLoc(), EntryState,
                                  ThenState, "then");
    }

    State = mergeStates(EntryState, ThenState, ElseState);
  }

  bool isInfiniteLoopCondition(const Expr* Condition) const {
    if (!Condition) {
      return true;
    }
    Expr::EvalResult Result;
    if (!Condition->EvaluateAsInt(Result, Context_)) {
      return false;
    }
    return Result.Val.getInt().getBoolValue();
  }

  bool containsEscapingLoopControl(const Stmt* Statement) const {
    if (!Statement) {
      return false;
    }
    if (isa<BreakStmt>(Statement) || isa<GotoStmt>(Statement) ||
        isa<IndirectGotoStmt>(Statement)) {
      return true;
    }
    for (const Stmt* Child : Statement->children()) {
      if (containsEscapingLoopControl(Child)) {
        return true;
      }
    }
    return false;
  }

  void analyzeLoopLike(const Stmt* Body, TraceState& State,
                       bool LoopCannotFallThrough) {
    TraceState BodyState = State;
    BodyState.SuppressEndDiagnostics = true;
    analyzeStatement(Body, BodyState);
    // A loop or switch body may execute zero times or leave through break, so
    // the outer state remains the entry state. Diagnostics inside the body
    // still report active-zone status returns.
    if (LoopCannotFallThrough && !containsEscapingLoopControl(Body)) {
      State.Terminal = true;
    }
  }

  TraceZoneCheck& Check_;
  ASTContext& Context_;
  llvm::SmallVector<char, 256> ConsumedMacroExpansions_;
};

}  // namespace

TraceZoneCheck::TraceZoneCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void TraceZoneCheck::registerPPCallbacks(const SourceManager&, Preprocessor* PP,
                                         Preprocessor*) {
  if (!PP) {
    return;
  }
  PP->addPPCallbacks(std::make_unique<TraceMacroRecorder>(*this, *PP));
}

void TraceZoneCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(functionDecl(isDefinition()).bind("function"), this);
}

void TraceZoneCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function || !Function->hasBody()) {
    return;
  }
  TraceZoneAnalyzer Analyzer(*this, *Result.Context);
  Analyzer.analyzeFunction(Function);
}

void TraceZoneCheck::addTraceMacroExpansion(TraceMacroKind Kind, StringRef Name,
                                            std::string ZoneId,
                                            SourceLocation BeginLocation,
                                            SourceLocation EndLocation) {
  TraceMacroExpansions_.push_back(TraceMacroExpansion{
      Kind, Name.str(), std::move(ZoneId), BeginLocation, EndLocation});
}

}  // namespace clang::tidy::iree
