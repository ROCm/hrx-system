// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/RecursionCheck.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::tidy::iree {
namespace {

const FunctionDecl* CanonicalFunction(const FunctionDecl* Function) {
  return Function ? Function->getCanonicalDecl() : nullptr;
}

class FunctionCollector final : public RecursiveASTVisitor<FunctionCollector> {
 public:
  bool VisitFunctionDecl(FunctionDecl* Function) {
    if (!Function || !Function->isThisDeclarationADefinition() ||
        Function->isImplicit()) {
      return true;
    }
    const FunctionDecl* Canonical = CanonicalFunction(Function);
    if (SeenFunctions.insert(Canonical).second) {
      Functions.push_back(Function);
    }
    return true;
  }

  llvm::SmallVector<const FunctionDecl*, 64> Functions;

 private:
  llvm::SmallPtrSet<const FunctionDecl*, 32> SeenFunctions;
};

struct CollectedCall {
  const FunctionDecl* Caller = nullptr;
  const CallExpr* Call = nullptr;
  const FunctionDecl* DirectCallee = nullptr;
};

struct AssignedValue {
  const Expr* Value = nullptr;
  const FunctionDecl* ContextFunction = nullptr;
};

const ValueDecl* ReferencedStorage(const Expr* Expression) {
  while (Expression) {
    Expression = Expression->IgnoreParenImpCasts();
    if (const auto* Reference = dyn_cast<DeclRefExpr>(Expression)) {
      return dyn_cast<ValueDecl>(Reference->getDecl());
    }
    if (const auto* Member = dyn_cast<MemberExpr>(Expression)) {
      Expression = Member->getBase();
      continue;
    }
    if (const auto* Subscript = dyn_cast<ArraySubscriptExpr>(Expression)) {
      Expression = Subscript->getBase();
      continue;
    }
    if (const auto* Unary = dyn_cast<UnaryOperator>(Expression)) {
      Expression = Unary->getSubExpr();
      continue;
    }
    break;
  }
  return nullptr;
}

class FunctionBodyCollector final
    : public RecursiveASTVisitor<FunctionBodyCollector> {
 public:
  FunctionBodyCollector(
      const FunctionDecl* Function, llvm::SmallVectorImpl<CollectedCall>& Calls,
      llvm::DenseMap<const ValueDecl*, llvm::SmallVector<AssignedValue, 2>>&
          AssignedValues)
      : Function(Function), Calls(Calls), AssignedValues(AssignedValues) {}

  bool VisitCallExpr(CallExpr* Call) {
    Calls.push_back(CollectedCall{
        Function,
        Call,
        CanonicalFunction(Call ? Call->getDirectCallee() : nullptr),
    });
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator* Binary) {
    if (!Binary || !Binary->isAssignmentOp()) {
      return true;
    }
    const ValueDecl* Storage = ReferencedStorage(Binary->getLHS());
    if (Storage) {
      AssignedValues[Storage].push_back(
          AssignedValue{Binary->getRHS(), Function});
    }
    return true;
  }

  // A lambda's call operator is a separate function. Calls in its body must
  // not be attributed to the enclosing function.
  bool TraverseLambdaExpr(LambdaExpr*) { return true; }

 private:
  const FunctionDecl* Function;
  llvm::SmallVectorImpl<CollectedCall>& Calls;
  llvm::DenseMap<const ValueDecl*, llvm::SmallVector<AssignedValue, 2>>&
      AssignedValues;
};

struct TargetWorkItem {
  const Expr* Expression = nullptr;
  const FunctionDecl* ContextFunction = nullptr;
  const FunctionDecl* GraphCaller = nullptr;
  const CallExpr* GraphCall = nullptr;
};

struct ResolvedTarget {
  const FunctionDecl* Target = nullptr;
  const FunctionDecl* GraphCaller = nullptr;
  const FunctionDecl* IndirectCaller = nullptr;
  const CallExpr* GraphCall = nullptr;
};

class IndirectTargetResolver {
 public:
  IndirectTargetResolver(
      ASTContext& Context, llvm::ArrayRef<CollectedCall> Calls,
      const llvm::DenseMap<const FunctionDecl*, llvm::SmallVector<unsigned, 4>>&
          DirectCallsites,
      const llvm::DenseMap<const ValueDecl*,
                           llvm::SmallVector<AssignedValue, 2>>& AssignedValues)
      : Context(Context),
        Calls(Calls),
        DirectCallsites(DirectCallsites),
        AssignedValues(AssignedValues) {}

  llvm::SmallVector<ResolvedTarget, 8> Resolve(
      const CollectedCall& IndirectCall) const {
    llvm::SmallVector<ResolvedTarget, 8> Targets;
    llvm::SmallVector<std::pair<const FunctionDecl*, const FunctionDecl*>, 8>
        SeenTargets;
    llvm::SmallVector<std::pair<const Expr*, const FunctionDecl*>, 32>
        SeenExpressions;
    llvm::SmallVector<std::pair<const ValueDecl*, const FunctionDecl*>, 32>
        SeenStorage;
    llvm::SmallVector<TargetWorkItem, 32> Worklist;
    Worklist.push_back(TargetWorkItem{IndirectCall.Call->getCallee(),
                                      IndirectCall.Caller, IndirectCall.Caller,
                                      IndirectCall.Call});

    auto Push = [&Worklist](const Expr* Expression,
                            const TargetWorkItem& From) {
      if (Expression) {
        Worklist.push_back(TargetWorkItem{Expression, From.ContextFunction,
                                          From.GraphCaller, From.GraphCall});
      }
    };

    while (!Worklist.empty()) {
      TargetWorkItem Item = Worklist.pop_back_val();
      const Expr* Expression = Item.Expression;
      std::pair<const Expr*, const FunctionDecl*> ExpressionKey{
          Expression, Item.GraphCaller};
      if (!Expression || llvm::is_contained(SeenExpressions, ExpressionKey)) {
        continue;
      }
      SeenExpressions.push_back(ExpressionKey);
      Expression = Expression->IgnoreParenImpCasts();

      if (const auto* Reference = dyn_cast<DeclRefExpr>(Expression)) {
        if (const auto* Function =
                dyn_cast<FunctionDecl>(Reference->getDecl())) {
          const FunctionDecl* Canonical = CanonicalFunction(Function);
          std::pair<const FunctionDecl*, const FunctionDecl*> TargetKey{
              Item.GraphCaller, Canonical};
          if (FunctionTypeMatches(IndirectCall.Call, Canonical) &&
              !llvm::is_contained(SeenTargets, TargetKey)) {
            SeenTargets.push_back(TargetKey);
            Targets.push_back(ResolvedTarget{Canonical, Item.GraphCaller,
                                             IndirectCall.Caller,
                                             Item.GraphCall});
          }
          continue;
        }
        const auto* Storage = dyn_cast<ValueDecl>(Reference->getDecl());
        std::pair<const ValueDecl*, const FunctionDecl*> StorageKey{
            Storage, Item.GraphCaller};
        if (!Storage || llvm::is_contained(SeenStorage, StorageKey)) {
          continue;
        }
        SeenStorage.push_back(StorageKey);
        if (const auto* Variable = dyn_cast<VarDecl>(Storage)) {
          Push(Variable->getInit(), Item);
        }
        if (const auto* Parameter = dyn_cast<ParmVarDecl>(Storage)) {
          ResolveParameter(*Parameter, Worklist);
        }
        auto Assigned = AssignedValues.find(Storage);
        if (Assigned != AssignedValues.end()) {
          for (const AssignedValue& Value : Assigned->second) {
            TargetWorkItem AssignmentItem = Item;
            AssignmentItem.ContextFunction = Value.ContextFunction;
            Push(Value.Value, AssignmentItem);
          }
        }
        continue;
      }

      if (const auto* Member = dyn_cast<MemberExpr>(Expression)) {
        Push(Member->getBase(), Item);
        continue;
      }
      if (const auto* Subscript = dyn_cast<ArraySubscriptExpr>(Expression)) {
        Push(Subscript->getBase(), Item);
        continue;
      }
      if (const auto* Initializers = dyn_cast<InitListExpr>(Expression)) {
        for (const Expr* Initializer : Initializers->inits()) {
          Push(Initializer, Item);
        }
        continue;
      }
      if (const auto* Designated = dyn_cast<DesignatedInitExpr>(Expression)) {
        Push(Designated->getInit(), Item);
        continue;
      }
      if (const auto* Compound = dyn_cast<CompoundLiteralExpr>(Expression)) {
        Push(Compound->getInitializer(), Item);
        continue;
      }
      if (const auto* Unary = dyn_cast<UnaryOperator>(Expression)) {
        Push(Unary->getSubExpr(), Item);
        continue;
      }
      if (const auto* Cast = dyn_cast<CastExpr>(Expression)) {
        Push(Cast->getSubExpr(), Item);
        continue;
      }
      if (const auto* Conditional =
              dyn_cast<AbstractConditionalOperator>(Expression)) {
        Push(Conditional->getTrueExpr(), Item);
        Push(Conditional->getFalseExpr(), Item);
        continue;
      }
      if (const auto* Binary = dyn_cast<BinaryOperator>(Expression)) {
        Push(Binary->getRHS(), Item);
        continue;
      }
      if (const auto* Generic = dyn_cast<GenericSelectionExpr>(Expression)) {
        if (!Generic->isResultDependent()) {
          Push(Generic->getResultExpr(), Item);
        }
        continue;
      }
      if (const auto* Opaque = dyn_cast<OpaqueValueExpr>(Expression)) {
        Push(Opaque->getSourceExpr(), Item);
        continue;
      }
      if (const auto* DefaultArgument =
              dyn_cast<CXXDefaultArgExpr>(Expression)) {
        Push(DefaultArgument->getExpr(), Item);
        continue;
      }
      if (const auto* BoundTemporary =
              dyn_cast<CXXBindTemporaryExpr>(Expression)) {
        Push(BoundTemporary->getSubExpr(), Item);
        continue;
      }
      if (const auto* Materialized =
              dyn_cast<MaterializeTemporaryExpr>(Expression)) {
        Push(Materialized->getSubExpr(), Item);
        continue;
      }
    }

    return Targets;
  }

 private:
  bool FunctionTypeMatches(const CallExpr* Call,
                           const FunctionDecl* Target) const {
    if (!Call || !Target) {
      return false;
    }
    QualType CalleeType = Call->getCallee()->getType();
    if (const auto* Pointer = CalleeType->getAs<PointerType>()) {
      CalleeType = Pointer->getPointeeType();
    }
    if (!CalleeType->isFunctionType()) {
      return false;
    }
    if (Context.getLangOpts().CPlusPlus) {
      return Context.hasSameType(CalleeType, Target->getType());
    }
    return Context.typesAreCompatible(CalleeType, Target->getType());
  }

  void ResolveParameter(const ParmVarDecl& Parameter,
                        llvm::SmallVectorImpl<TargetWorkItem>& Worklist) const {
    const auto* Owner = dyn_cast<FunctionDecl>(Parameter.getDeclContext());
    if (!Owner) {
      return;
    }
    auto Found = DirectCallsites.find(CanonicalFunction(Owner));
    if (Found == DirectCallsites.end()) {
      return;
    }
    const unsigned ParameterIndex = Parameter.getFunctionScopeIndex();
    for (unsigned CallIndex : Found->second) {
      const CollectedCall& Callsite = Calls[CallIndex];
      if (ParameterIndex < Callsite.Call->getNumArgs()) {
        Worklist.push_back(TargetWorkItem{Callsite.Call->getArg(ParameterIndex),
                                          Callsite.Caller, Callsite.Caller,
                                          Callsite.Call});
      }
    }
  }

  ASTContext& Context;
  llvm::ArrayRef<CollectedCall> Calls;
  const llvm::DenseMap<const FunctionDecl*, llvm::SmallVector<unsigned, 4>>&
      DirectCallsites;
  const llvm::DenseMap<const ValueDecl*, llvm::SmallVector<AssignedValue, 2>>&
      AssignedValues;
};

struct CallEdge {
  unsigned Target = 0;
  const CallExpr* Call = nullptr;
  const FunctionDecl* IndirectCaller = nullptr;
  bool IsIndirect = false;
};

struct ResolvedCallEdge {
  const FunctionDecl* Source = nullptr;
  const FunctionDecl* Target = nullptr;
  const CallExpr* Call = nullptr;
  const FunctionDecl* IndirectCaller = nullptr;
  bool IsIndirect = false;
};

struct FunctionNode {
  const FunctionDecl* Definition = nullptr;
  llvm::SmallVector<CallEdge, 8> Edges;
  llvm::SmallVector<unsigned, 8> ReverseEdges;
};

struct DepthFirstFrame {
  unsigned Node = 0;
  unsigned NextEdge = 0;
};

std::vector<std::vector<unsigned>> ComputeStronglyConnectedComponents(
    llvm::ArrayRef<FunctionNode> Nodes) {
  std::vector<bool> Visited(Nodes.size(), false);
  std::vector<unsigned> FinishOrder;
  FinishOrder.reserve(Nodes.size());
  llvm::SmallVector<DepthFirstFrame, 32> Frames;

  for (unsigned Root = 0; Root < Nodes.size(); ++Root) {
    if (Visited[Root]) {
      continue;
    }
    Visited[Root] = true;
    Frames.push_back(DepthFirstFrame{Root, 0});
    while (!Frames.empty()) {
      DepthFirstFrame& Frame = Frames.back();
      const FunctionNode& Node = Nodes[Frame.Node];
      if (Frame.NextEdge < Node.Edges.size()) {
        unsigned Target = Node.Edges[Frame.NextEdge++].Target;
        if (!Visited[Target]) {
          Visited[Target] = true;
          Frames.push_back(DepthFirstFrame{Target, 0});
        }
        continue;
      }
      FinishOrder.push_back(Frame.Node);
      Frames.pop_back();
    }
  }

  std::vector<bool> Assigned(Nodes.size(), false);
  std::vector<std::vector<unsigned>> Components;
  llvm::SmallVector<unsigned, 32> Worklist;
  for (auto It = FinishOrder.rbegin(); It != FinishOrder.rend(); ++It) {
    unsigned Root = *It;
    if (Assigned[Root]) {
      continue;
    }
    std::vector<unsigned> Component;
    Assigned[Root] = true;
    Worklist.push_back(Root);
    while (!Worklist.empty()) {
      unsigned NodeIndex = Worklist.pop_back_val();
      Component.push_back(NodeIndex);
      for (unsigned Predecessor : Nodes[NodeIndex].ReverseEdges) {
        if (!Assigned[Predecessor]) {
          Assigned[Predecessor] = true;
          Worklist.push_back(Predecessor);
        }
      }
    }
    Components.push_back(std::move(Component));
  }
  return Components;
}

bool IsRecursiveComponent(llvm::ArrayRef<unsigned> Component,
                          llvm::ArrayRef<FunctionNode> Nodes) {
  if (Component.size() > 1) {
    return true;
  }
  if (Component.empty()) {
    return false;
  }
  unsigned NodeIndex = Component.front();
  return llvm::any_of(
      Nodes[NodeIndex].Edges,
      [NodeIndex](const CallEdge& Edge) { return Edge.Target == NodeIndex; });
}

struct CycleEdge {
  unsigned Source = 0;
  unsigned EdgeIndex = 0;
};

std::vector<CycleEdge> FindRepresentativeCycle(
    unsigned Start, llvm::ArrayRef<unsigned> Component,
    llvm::ArrayRef<FunctionNode> Nodes) {
  std::vector<bool> InComponent(Nodes.size(), false);
  for (unsigned Node : Component) {
    InComponent[Node] = true;
  }

  const FunctionNode& StartNode = Nodes[Start];
  for (unsigned FirstEdgeIndex = 0; FirstEdgeIndex < StartNode.Edges.size();
       ++FirstEdgeIndex) {
    const CallEdge& FirstEdge = StartNode.Edges[FirstEdgeIndex];
    if (!InComponent[FirstEdge.Target]) {
      continue;
    }
    CycleEdge First{Start, FirstEdgeIndex};
    if (FirstEdge.Target == Start) {
      return {First};
    }

    const unsigned Unset = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> PredecessorNode(Nodes.size(), Unset);
    std::vector<unsigned> PredecessorEdge(Nodes.size(), Unset);
    llvm::SmallVector<unsigned, 32> Queue;
    size_t QueuePosition = 0;
    PredecessorNode[FirstEdge.Target] = FirstEdge.Target;
    Queue.push_back(FirstEdge.Target);
    while (QueuePosition < Queue.size() && PredecessorNode[Start] == Unset) {
      unsigned Source = Queue[QueuePosition++];
      for (unsigned EdgeIndex = 0; EdgeIndex < Nodes[Source].Edges.size();
           ++EdgeIndex) {
        unsigned Target = Nodes[Source].Edges[EdgeIndex].Target;
        if (!InComponent[Target] || PredecessorNode[Target] != Unset) {
          continue;
        }
        PredecessorNode[Target] = Source;
        PredecessorEdge[Target] = EdgeIndex;
        Queue.push_back(Target);
      }
    }
    if (PredecessorNode[Start] == Unset) {
      continue;
    }

    std::vector<CycleEdge> ReversePath;
    unsigned Current = Start;
    while (Current != FirstEdge.Target) {
      unsigned Source = PredecessorNode[Current];
      ReversePath.push_back(CycleEdge{Source, PredecessorEdge[Current]});
      Current = Source;
    }
    std::reverse(ReversePath.begin(), ReversePath.end());
    std::vector<CycleEdge> Cycle;
    Cycle.reserve(1 + ReversePath.size());
    Cycle.push_back(First);
    Cycle.insert(Cycle.end(), ReversePath.begin(), ReversePath.end());
    return Cycle;
  }
  return {};
}

std::string FunctionName(const FunctionDecl* Function) {
  if (!Function) {
    return "<unknown>";
  }
  std::string Name = Function->getQualifiedNameAsString();
  return Name.empty() ? "<anonymous>" : Name;
}

struct SourcePoint {
  std::string File;
  unsigned Line = 0;
  unsigned Column = 0;
};

std::string NormalizeSourcePath(StringRef Path) {
  constexpr llvm::StringLiteral kBazelExecrootMarker = "/execroot/_main/";
  size_t Marker = Path.find(kBazelExecrootMarker);
  if (Marker != StringRef::npos) {
    return Path.drop_front(Marker + kBazelExecrootMarker.size()).str();
  }
  return Path.str();
}

SourcePoint GetSourcePoint(SourceLocation Location,
                           const SourceManager& SourceManager) {
  if (Location.isInvalid()) {
    return {};
  }
  PresumedLoc Presumed =
      SourceManager.getPresumedLoc(SourceManager.getExpansionLoc(Location));
  if (Presumed.isInvalid()) {
    return {};
  }
  return SourcePoint{NormalizeSourcePath(Presumed.getFilename()),
                     Presumed.getLine(), Presumed.getColumn()};
}

class FunctionIdentityMap {
 public:
  FunctionIdentityMap(ASTContext& Context, const SourceManager& SourceManager,
                      StringRef TranslationUnit)
      : SourceManager(SourceManager),
        TranslationUnit(TranslationUnit),
        Mangler(Context.createMangleContext()) {}

  const std::string& Get(const FunctionDecl* Function) {
    Function = CanonicalFunction(Function);
    auto Found = Ids.find(Function);
    if (Found != Ids.end()) {
      return Found->second;
    }

    std::string Symbol;
    llvm::raw_string_ostream SymbolStream(Symbol);
    if (Mangler->shouldMangleDeclName(Function)) {
      Mangler->mangleName(GlobalDecl(Function), SymbolStream);
    } else {
      SymbolStream << FunctionName(Function);
    }
    SymbolStream.flush();

    std::string Id;
    llvm::raw_string_ostream IdStream(Id);
    if (Function->getFormalLinkage() == Linkage::External) {
      IdStream << "external:" << Symbol;
    } else {
      const FunctionDecl* Definition = Function->getDefinition();
      SourcePoint Point = GetSourcePoint(
          Definition ? Definition->getLocation() : Function->getLocation(),
          SourceManager);
      IdStream << "internal:" << TranslationUnit << ':' << Point.File << ':'
               << Point.Line << ':' << Point.Column << ':' << Symbol;
    }
    IdStream.flush();
    return Ids.try_emplace(Function, std::move(Id)).first->second;
  }

 private:
  const SourceManager& SourceManager;
  std::string TranslationUnit;
  std::unique_ptr<MangleContext> Mangler;
  llvm::DenseMap<const FunctionDecl*, std::string> Ids;
};

std::string MainSourcePath(const SourceManager& SourceManager) {
  SourceLocation Start =
      SourceManager.getLocForStartOfFile(SourceManager.getMainFileID());
  return NormalizeSourcePath(SourceManager.getFilename(Start));
}

bool IsDefinedInMainFile(const FunctionDecl* Function,
                         const SourceManager& SourceManager) {
  const FunctionDecl* Definition =
      Function ? Function->getDefinition() : nullptr;
  return Definition &&
         SourceManager.isWrittenInMainFile(
             SourceManager.getSpellingLoc(Definition->getLocation()));
}

std::error_code WriteRecursionSummary(
    StringRef OutputPath, llvm::ArrayRef<const FunctionDecl*> Functions,
    llvm::ArrayRef<ResolvedCallEdge> Edges, ASTContext& Context,
    const SourceManager& SourceManager) {
  std::error_code Error;
  llvm::raw_fd_ostream Output(OutputPath, Error);
  if (Error) {
    return Error;
  }

  std::string TranslationUnit = MainSourcePath(SourceManager);
  FunctionIdentityMap Ids(Context, SourceManager, TranslationUnit);
  llvm::json::OStream Json(Output, /*IndentSize=*/2);
  Json.object([&] {
    Json.attribute("version", 1);
    Json.attribute("translation_unit", TranslationUnit);
    Json.attributeArray("functions", [&] {
      for (const FunctionDecl* Function : Functions) {
        if (!IsDefinedInMainFile(Function, SourceManager)) {
          continue;
        }
        const FunctionDecl* Definition = Function->getDefinition();
        SourcePoint Point =
            GetSourcePoint(Definition->getLocation(), SourceManager);
        Json.object([&] {
          Json.attribute("id", Ids.Get(Function));
          Json.attribute("name", FunctionName(Function));
          Json.attribute("file", Point.File);
          Json.attribute("line", Point.Line);
          Json.attribute("column", Point.Column);
        });
      }
    });
    Json.attributeArray("edges", [&] {
      for (const ResolvedCallEdge& Edge : Edges) {
        if (!IsDefinedInMainFile(Edge.Source, SourceManager)) {
          continue;
        }
        SourcePoint Point =
            GetSourcePoint(Edge.Call->getExprLoc(), SourceManager);
        Json.object([&] {
          Json.attribute("caller", Ids.Get(Edge.Source));
          Json.attribute("callee", Ids.Get(Edge.Target));
          Json.attribute("caller_name", FunctionName(Edge.Source));
          Json.attribute("callee_name", FunctionName(Edge.Target));
          Json.attribute("file", Point.File);
          Json.attribute("line", Point.Line);
          Json.attribute("column", Point.Column);
          Json.attribute("indirect", Edge.IsIndirect);
          if (Edge.IsIndirect && Edge.IndirectCaller) {
            Json.attribute("dispatcher", FunctionName(Edge.IndirectCaller));
          }
        });
      }
    });
  });
  Output << '\n';
  Output.close();
  Error = Output.error();
  Output.clear_error();
  return Error;
}

bool IsBeforeInSource(const FunctionDecl* Lhs, const FunctionDecl* Rhs,
                      const SourceManager& SourceManager) {
  SourceLocation LhsLocation = SourceManager.getSpellingLoc(Lhs->getLocation());
  SourceLocation RhsLocation = SourceManager.getSpellingLoc(Rhs->getLocation());
  StringRef LhsFile = SourceManager.getFilename(LhsLocation);
  StringRef RhsFile = SourceManager.getFilename(RhsLocation);
  if (LhsFile != RhsFile) {
    return LhsFile < RhsFile;
  }
  if (LhsLocation != RhsLocation) {
    return SourceManager.isBeforeInTranslationUnit(LhsLocation, RhsLocation);
  }
  return FunctionName(Lhs) < FunctionName(Rhs);
}

}  // namespace

UnboundedRecursionCheck::UnboundedRecursionCheck(StringRef Name,
                                                 ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void UnboundedRecursionCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  Finder->addMatcher(
      ast_matchers::translationUnitDecl().bind("translation_unit"), this);
}

void UnboundedRecursionCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* TranslationUnit =
      Result.Nodes.getNodeAs<TranslationUnitDecl>("translation_unit");
  if (!TranslationUnit || !Result.Context || !Result.SourceManager) {
    return;
  }

  FunctionCollector Functions;
  Functions.TraverseDecl(const_cast<TranslationUnitDecl*>(TranslationUnit));
  std::sort(Functions.Functions.begin(), Functions.Functions.end(),
            [&](const FunctionDecl* Lhs, const FunctionDecl* Rhs) {
              return IsBeforeInSource(Lhs, Rhs, *Result.SourceManager);
            });

  llvm::SmallVector<CollectedCall, 128> Calls;
  llvm::DenseMap<const ValueDecl*, llvm::SmallVector<AssignedValue, 2>>
      AssignedValues;
  for (const FunctionDecl* Function : Functions.Functions) {
    FunctionBodyCollector Collector(Function, Calls, AssignedValues);
    Collector.TraverseStmt(const_cast<Stmt*>(Function->getBody()));
  }

  llvm::DenseMap<const FunctionDecl*, llvm::SmallVector<unsigned, 4>>
      DirectCallsites;
  for (unsigned Index = 0; Index < Calls.size(); ++Index) {
    if (Calls[Index].DirectCallee) {
      DirectCallsites[Calls[Index].DirectCallee].push_back(Index);
    }
  }

  std::vector<FunctionNode> Nodes;
  Nodes.reserve(Functions.Functions.size());
  llvm::DenseMap<const FunctionDecl*, unsigned> NodeIndices;
  for (const FunctionDecl* Definition : Functions.Functions) {
    const FunctionDecl* Canonical = CanonicalFunction(Definition);
    NodeIndices[Canonical] = Nodes.size();
    Nodes.push_back(FunctionNode{Definition, {}, {}});
  }

  IndirectTargetResolver Resolver(*Result.Context, Calls, DirectCallsites,
                                  AssignedValues);
  llvm::SmallVector<ResolvedCallEdge, 128> ResolvedEdges;
  for (const CollectedCall& Call : Calls) {
    if (Call.DirectCallee) {
      ResolvedEdges.push_back(ResolvedCallEdge{Call.Caller, Call.DirectCallee,
                                               Call.Call, nullptr, false});
      continue;
    }
    for (const ResolvedTarget& Resolved : Resolver.Resolve(Call)) {
      ResolvedEdges.push_back(
          ResolvedCallEdge{Resolved.GraphCaller, Resolved.Target,
                           Resolved.GraphCall, Resolved.IndirectCaller, true});
    }
  }

  for (const ResolvedCallEdge& Edge : ResolvedEdges) {
    auto Source = NodeIndices.find(CanonicalFunction(Edge.Source));
    auto Target = NodeIndices.find(CanonicalFunction(Edge.Target));
    if (Source == NodeIndices.end() || Target == NodeIndices.end()) {
      continue;
    }
    Nodes[Source->second].Edges.push_back(CallEdge{
        Target->second, Edge.Call, Edge.IndirectCaller, Edge.IsIndirect});
    Nodes[Target->second].ReverseEdges.push_back(Source->second);
  }

  if (std::optional<std::string> SummaryPath =
          llvm::sys::Process::GetEnv("IREE_CLANG_TIDY_RECURSION_SUMMARY")) {
    if (std::error_code Error = WriteRecursionSummary(
            *SummaryPath, Functions.Functions, ResolvedEdges, *Result.Context,
            *Result.SourceManager)) {
      SourceLocation Start = Result.SourceManager->getLocForStartOfFile(
          Result.SourceManager->getMainFileID());
      diag(Start, "failed to write recursion call-graph summary %0: %1")
          << *SummaryPath << Error.message();
    }
  }

  if (llvm::sys::Process::GetEnv("IREE_CLANG_TIDY_RECURSION_DIAGNOSTICS") ==
      std::optional<std::string>("0")) {
    return;
  }

  for (std::vector<unsigned>& Component :
       ComputeStronglyConnectedComponents(Nodes)) {
    if (!IsRecursiveComponent(Component, Nodes)) {
      continue;
    }
    std::sort(Component.begin(), Component.end());
    std::optional<unsigned> Representative;
    for (unsigned NodeIndex : Component) {
      SourceLocation Location = Nodes[NodeIndex].Definition->getLocation();
      if (Location.isValid() &&
          !Result.SourceManager->isInSystemHeader(Location)) {
        Representative = NodeIndex;
        break;
      }
    }
    if (!Representative) {
      continue;
    }

    const FunctionNode& Node = Nodes[*Representative];
    diag(Node.Definition->getLocation(),
         "potentially unbounded native recursion in a call cycle containing "
         "%0 function%select{|s}1; use an explicit worklist or establish a "
         "mechanically checked fixed bound")
        << Component.size() << (Component.size() != 1);

    for (const CycleEdge& CycleEdge :
         FindRepresentativeCycle(*Representative, Component, Nodes)) {
      const FunctionNode& Source = Nodes[CycleEdge.Source];
      const CallEdge& Edge = Source.Edges[CycleEdge.EdgeIndex];
      if (Edge.IsIndirect && CanonicalFunction(Edge.IndirectCaller) !=
                                 CanonicalFunction(Source.Definition)) {
        diag(Edge.Call->getExprLoc(),
             "call from %0 through callback dispatcher %1 to %2 participates "
             "in this cycle",
             DiagnosticIDs::Note)
            << FunctionName(Source.Definition)
            << FunctionName(Edge.IndirectCaller)
            << FunctionName(Nodes[Edge.Target].Definition);
      } else {
        diag(Edge.Call->getExprLoc(),
             "%select{direct|indirect}0 call from %1 to %2 participates in "
             "this cycle",
             DiagnosticIDs::Note)
            << Edge.IsIndirect << FunctionName(Source.Definition)
            << FunctionName(Nodes[Edge.Target].Definition);
      }
    }
  }
}

}  // namespace clang::tidy::iree
