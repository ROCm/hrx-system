// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/LifecycleChecks.h"

#include <optional>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
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

const RecordDecl* RecordDefinition(const RecordDecl* Record) {
  if (!Record) {
    return nullptr;
  }
  return Record->getDefinition() ? Record->getDefinition() : Record;
}

const RecordDecl* RecordFromType(QualType Type) {
  Type = Type.getCanonicalType().getUnqualifiedType();
  const auto* Record = Type->getAs<RecordType>();
  return RecordDefinition(Record ? Record->getDecl() : nullptr);
}

const RecordDecl* CallerOwnedOutParameterRecord(const ParmVarDecl* Parameter) {
  if (!Parameter || !Parameter->getName().starts_with("out_")) {
    return nullptr;
  }
  QualType Type = Parameter->getType().getCanonicalType().getUnqualifiedType();
  if (!Type->isPointerType()) {
    return nullptr;
  }
  QualType PointeeType =
      Type->getPointeeType().getCanonicalType().getUnqualifiedType();
  if (PointeeType->isPointerType()) {
    return nullptr;
  }
  return RecordFromType(PointeeType);
}

const RecordDecl* PublishedOutParameterRecord(const ParmVarDecl* Parameter) {
  if (!Parameter || !Parameter->getName().starts_with("out_")) {
    return nullptr;
  }
  QualType Type = Parameter->getType().getCanonicalType().getUnqualifiedType();
  if (!Type->isPointerType()) {
    return nullptr;
  }
  QualType PointeeType =
      Type->getPointeeType().getCanonicalType().getUnqualifiedType();
  if (!PointeeType->isPointerType()) {
    return nullptr;
  }
  return RecordFromType(PointeeType->getPointeeType());
}

bool IsStorageParameterName(StringRef Name) {
  return Name == "storage" || Name.ends_with("_storage");
}

StringRef SimpleFunctionName(const FunctionDecl* Function) {
  const IdentifierInfo* Identifier =
      Function ? Function->getIdentifier() : nullptr;
  return Identifier ? Identifier->getName() : StringRef();
}

bool HasStorageParameter(const FunctionDecl* Function) {
  for (const ParmVarDecl* Parameter : Function->parameters()) {
    if (IsStorageParameterName(Parameter->getName())) {
      return true;
    }
  }
  return false;
}

bool FirstParameterHasRecordType(const FunctionDecl* Function,
                                 const RecordDecl* Record) {
  if (!Function || Function->getNumParams() < 1) {
    return false;
  }
  QualType Type = Function->getParamDecl(0)
                      ->getType()
                      .getCanonicalType()
                      .getUnqualifiedType();
  if (!Type->isPointerType()) {
    return false;
  }
  const RecordDecl* ParameterRecord = RecordFromType(Type->getPointeeType());
  return ParameterRecord && ParameterRecord == RecordDefinition(Record);
}

const FunctionDecl* MatchingDeinitializeFunction(const FunctionDecl* Function,
                                                 const RecordDecl* Record) {
  StringRef FunctionName = SimpleFunctionName(Function);
  if (!Function || !Record || !FunctionName.ends_with("_allocate")) {
    return nullptr;
  }
  std::string DeinitializeName =
      (FunctionName.drop_back(StringRef("_allocate").size()) + "_deinitialize")
          .str();
  for (const Decl* Declaration : Function->getDeclContext()->decls()) {
    const auto* Candidate = dyn_cast<FunctionDecl>(Declaration);
    if (!Candidate || SimpleFunctionName(Candidate) != DeinitializeName ||
        !Candidate->getReturnType()->isVoidType()) {
      continue;
    }
    if (FirstParameterHasRecordType(Candidate, Record)) {
      return Candidate;
    }
  }
  return nullptr;
}

struct CallerOwnedAllocateOutParameter {
  const ParmVarDecl* Parameter = nullptr;
  const FunctionDecl* DeinitializeFunction = nullptr;
};

std::optional<CallerOwnedAllocateOutParameter>
CallerOwnedAllocateOutParameterFor(const FunctionDecl* Function) {
  if (!Function || !SimpleFunctionName(Function).ends_with("_allocate")) {
    return std::nullopt;
  }
  for (const ParmVarDecl* Parameter : Function->parameters()) {
    const RecordDecl* Record = CallerOwnedOutParameterRecord(Parameter);
    const FunctionDecl* DeinitializeFunction =
        MatchingDeinitializeFunction(Function, Record);
    if (DeinitializeFunction) {
      return CallerOwnedAllocateOutParameter{
          .Parameter = Parameter,
          .DeinitializeFunction = DeinitializeFunction,
      };
    }
  }
  return std::nullopt;
}

const ParmVarDecl* InitializePublishedOutParameterWithoutStorage(
    const FunctionDecl* Function) {
  if (!Function || !SimpleFunctionName(Function).ends_with("_initialize") ||
      HasStorageParameter(Function)) {
    return nullptr;
  }
  for (const ParmVarDecl* Parameter : Function->parameters()) {
    if (PublishedOutParameterRecord(Parameter)) {
      return Parameter;
    }
  }
  return nullptr;
}

}  // namespace

LifecycleNamingCheck::LifecycleNamingCheck(StringRef Name,
                                           ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void LifecycleNamingCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(isExpansionInSystemHeader()))
          .bind("function"),
      this);
}

void LifecycleNamingCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Function->getLocation(), SourceManager)) {
    return;
  }
  std::optional<CallerOwnedAllocateOutParameter> OutParameter =
      CallerOwnedAllocateOutParameterFor(Function);
  if (OutParameter) {
    SourceLocation Location =
        SourceManager.getExpansionLoc(OutParameter->Parameter->getLocation());
    diag(Location,
         "caller-owned output %0 from %1 uses allocate naming but is cleaned "
         "up by %2; use initialize/deinitialize naming")
        << OutParameter->Parameter->getName() << SimpleFunctionName(Function)
        << SimpleFunctionName(OutParameter->DeinitializeFunction);
    return;
  }

  const ParmVarDecl* PublishedOutParameter =
      InitializePublishedOutParameterWithoutStorage(Function);
  if (!PublishedOutParameter) {
    return;
  }
  SourceLocation Location =
      SourceManager.getExpansionLoc(PublishedOutParameter->getLocation());
  diag(Location,
       "pointer-to-pointer output %0 from %1 uses initialize naming without "
       "an explicit storage parameter")
      << PublishedOutParameter->getName() << SimpleFunctionName(Function);
}

}  // namespace clang::tidy::iree
