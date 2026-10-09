// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/DesignatedInitializerCheck.h"

#include <cctype>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

struct FieldLabel {
  SourceLocation location;
  CharSourceRange range;
  std::string name;
};

struct MemberAssignment {
  const FieldDecl* field;
  std::string value_text;
  CharSourceRange removal_range;
};

struct LabeledInitializer {
  enum class Issue {
    kNone,
    kUnrepresentable,
    kMismatch,
  };

  FieldLabel label;
  const FieldDecl* field;
  Issue issue;
};

bool IsMainFileLocation(SourceLocation Location,
                        const SourceManager& SourceManager) {
  if (Location.isInvalid() || Location.isMacroID()) {
    return false;
  }
  return SourceManager.isWrittenInMainFile(Location);
}

bool IsIdentifier(StringRef Text) {
  if (Text.empty() ||
      !(Text.front() == '_' ||
        std::isalpha(static_cast<unsigned char>(Text.front())))) {
    return false;
  }
  for (char Character : Text.drop_front()) {
    if (Character != '_' &&
        !std::isalnum(static_cast<unsigned char>(Character))) {
      return false;
    }
  }
  return true;
}

std::optional<FieldLabel> FindFieldLabel(const Expr* Initializer,
                                         const SourceManager& SourceManager) {
  SourceLocation Begin =
      SourceManager.getExpansionLoc(Initializer->getBeginLoc());
  if (!IsMainFileLocation(Begin, SourceManager)) {
    return std::nullopt;
  }
  std::pair<FileID, unsigned> Decomposed =
      SourceManager.getDecomposedLoc(Begin);
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
  if (Invalid || Decomposed.second > Buffer.size()) {
    return std::nullopt;
  }

  size_t CommentEnd = Decomposed.second;
  while (CommentEnd > 0 &&
         std::isspace(static_cast<unsigned char>(Buffer[CommentEnd - 1]))) {
    --CommentEnd;
  }
  if (CommentEnd < 2 || Buffer.substr(CommentEnd - 2, 2) != "*/") {
    return std::nullopt;
  }
  size_t CommentBegin = Buffer.take_front(CommentEnd).rfind("/*");
  if (CommentBegin == StringRef::npos) {
    return std::nullopt;
  }
  StringRef Comment =
      Buffer.slice(static_cast<unsigned>(CommentBegin), CommentEnd);
  if (!Comment.starts_with("/*.") || !Comment.ends_with("=*/")) {
    return std::nullopt;
  }
  StringRef Name = Comment.drop_front(3).drop_back(3);
  if (!IsIdentifier(Name)) {
    return std::nullopt;
  }

  SourceLocation FileBegin =
      SourceManager.getLocForStartOfFile(Decomposed.first);
  SourceLocation LabelBegin =
      FileBegin.getLocWithOffset(static_cast<int>(CommentBegin));
  SourceLocation LabelEnd =
      FileBegin.getLocWithOffset(static_cast<int>(CommentEnd));
  return FieldLabel{
      LabelBegin,
      CharSourceRange::getCharRange(LabelBegin, LabelEnd),
      Name.str(),
  };
}

const RecordDecl* DefinedRecord(QualType Type) {
  const auto* TypeRecord = Type->getAs<RecordType>();
  if (!TypeRecord) {
    return nullptr;
  }
  return TypeRecord->getDecl()->getDefinition();
}

const RecordDecl* AggregateRecord(QualType Type) {
  const RecordDecl* Record = DefinedRecord(Type);
  if (!Record) {
    return nullptr;
  }
  if (const auto* CxxRecord = dyn_cast<CXXRecordDecl>(Record)) {
    if (!CxxRecord->isAggregate() || CxxRecord->getNumBases() != 0) {
      return nullptr;
    }
  }
  return Record;
}

std::vector<const FieldDecl*> InitializableFields(const RecordDecl* Record) {
  std::vector<const FieldDecl*> Fields;
  for (const FieldDecl* Field : Record->fields()) {
    if (!Field->isUnnamedBitField()) {
      Fields.push_back(Field);
    }
  }
  return Fields;
}

const FieldDecl* InitializedField(const InitListExpr* Initializer,
                                  unsigned InitializerIndex) {
  const InitListExpr* SemanticInitializer = Initializer->getSemanticForm();
  if (!SemanticInitializer) {
    SemanticInitializer = Initializer;
  }
  const RecordDecl* Record = AggregateRecord(SemanticInitializer->getType());
  if (!Record) {
    return nullptr;
  }
  std::vector<const FieldDecl*> Fields = InitializableFields(Record);
  if (Record->isUnion()) {
    if (InitializerIndex != 0) {
      return nullptr;
    }
    if (SemanticInitializer->getInitializedFieldInUnion()) {
      return SemanticInitializer->getInitializedFieldInUnion();
    }
    return Fields.empty() ? nullptr : Fields.front();
  }
  return InitializerIndex < Fields.size() ? Fields[InitializerIndex] : nullptr;
}

bool CanUseDirectDesignator(const FieldDecl* Field, const Expr* Initializer,
                            ASTContext& Context) {
  if (!Field || Field->getName().empty() || Field->isAnonymousStructOrUnion()) {
    return false;
  }
  const Expr* SourceExpression = Initializer->IgnoreParenImpCasts();
  if (isa<InitListExpr>(SourceExpression)) {
    return true;
  }
  QualType FieldType = Field->getType();
  if (Context.hasSameUnqualifiedType(FieldType, SourceExpression->getType())) {
    return true;
  }
  if (FieldType->isArrayType()) {
    return isa<StringLiteral>(SourceExpression);
  }
  const auto* FieldRecord = FieldType->getAsCXXRecordDecl();
  return !FieldRecord || !FieldRecord->isAggregate();
}

const InitListExpr* SourceInitializerList(const Expr* Initializer) {
  const auto* List = dyn_cast_or_null<InitListExpr>(
      Initializer ? Initializer->IgnoreParenImpCasts() : nullptr);
  if (!List) {
    return nullptr;
  }
  if (const InitListExpr* Syntactic = List->getSyntacticForm()) {
    return Syntactic;
  }
  return List->isSyntacticForm() ? List : nullptr;
}

const Expr* SpelledExpression(const Stmt* Statement) {
  const auto* Expression = dyn_cast_or_null<Expr>(Statement);
  return Expression ? Expression->IgnoreUnlessSpelledInSource() : nullptr;
}

const VarDecl* ReferencedVariable(const Expr* Expression) {
  const auto* Reference = dyn_cast_or_null<DeclRefExpr>(
      Expression ? Expression->IgnoreParenImpCasts() : nullptr);
  return Reference ? dyn_cast<VarDecl>(Reference->getDecl()) : nullptr;
}

struct ParsedMemberAssignment {
  const FieldDecl* field;
  const Expr* value;
};

std::optional<ParsedMemberAssignment> ParseMemberAssignment(
    const Stmt* Statement, const VarDecl* Variable) {
  const Expr* Expression = SpelledExpression(Statement);
  const Expr* Left = nullptr;
  const Expr* Right = nullptr;
  if (const auto* Assignment = dyn_cast_or_null<BinaryOperator>(Expression)) {
    if (Assignment->getOpcode() != BO_Assign) {
      return std::nullopt;
    }
    Left = Assignment->getLHS();
    Right = Assignment->getRHS();
  } else if (const auto* Assignment =
                 dyn_cast_or_null<CXXOperatorCallExpr>(Expression)) {
    if (Assignment->getOperator() != OO_Equal ||
        Assignment->getNumArgs() != 2) {
      return std::nullopt;
    }
    Left = Assignment->getArg(0);
    Right = Assignment->getArg(1);
  } else {
    return std::nullopt;
  }
  const auto* Member = dyn_cast<MemberExpr>(Left->IgnoreParenImpCasts());
  if (!Member || Member->isArrow() ||
      ReferencedVariable(Member->getBase()) != Variable) {
    return std::nullopt;
  }
  const auto* Field = dyn_cast<FieldDecl>(Member->getMemberDecl());
  if (!Field) {
    return std::nullopt;
  }
  return ParsedMemberAssignment{Field, Right};
}

bool ReferencesVariable(const Stmt* Statement, const VarDecl* Variable) {
  if (!Statement) {
    return false;
  }
  if (const auto* Reference = dyn_cast<DeclRefExpr>(Statement)) {
    if (Reference->getDecl()->getCanonicalDecl() ==
        Variable->getCanonicalDecl()) {
      return true;
    }
  }
  for (const Stmt* Child : Statement->children()) {
    if (ReferencesVariable(Child, Variable)) {
      return true;
    }
  }
  return false;
}

bool ContainsMemberAssignment(const Stmt* Statement, const VarDecl* Variable) {
  if (!Statement) {
    return false;
  }
  if (ParseMemberAssignment(Statement, Variable)) {
    return true;
  }
  for (const Stmt* Child : Statement->children()) {
    if (ContainsMemberAssignment(Child, Variable)) {
      return true;
    }
  }
  return false;
}

std::optional<unsigned> FieldIndex(const RecordDecl* Record,
                                   const FieldDecl* Field) {
  unsigned Index = 0;
  for (const FieldDecl* Candidate : Record->fields()) {
    if (Candidate->isUnnamedBitField()) {
      continue;
    }
    if (Candidate->getCanonicalDecl() == Field->getCanonicalDecl()) {
      return Index;
    }
    ++Index;
  }
  return std::nullopt;
}

bool IntegerConstantFits(const Expr* Expression, QualType TargetType,
                         ASTContext& Context) {
  if (!TargetType->isIntegerType() || TargetType->isBooleanType()) {
    return false;
  }
  std::optional<llvm::APSInt> Value =
      Expression->getIntegerConstantExpr(Context);
  if (!Value) {
    return false;
  }
  unsigned Width = Context.getIntWidth(TargetType);
  if (Value->isNegative()) {
    return TargetType->isSignedIntegerOrEnumerationType() &&
           Value->isSignedIntN(Width);
  }
  unsigned ValueBits = Value->getActiveBits();
  unsigned AvailableBits =
      TargetType->isSignedIntegerOrEnumerationType() ? Width - 1 : Width;
  return ValueBits <= AvailableBits;
}

bool CanMoveAssignmentValue(const Expr* Value, const FieldDecl* Field,
                            const VarDecl* Variable, ASTContext& Context,
                            const SourceManager& SourceManager) {
  if (!Value || ReferencesVariable(Value, Variable) ||
      !IsMainFileLocation(Value->getBeginLoc(), SourceManager) ||
      !IsMainFileLocation(Value->getEndLoc(), SourceManager)) {
    return false;
  }
  QualType FieldType = Field->getType();
  if (FieldType.isVolatileQualified() || FieldType->isAtomicType()) {
    return false;
  }
  const Expr* SourceValue = Value->IgnoreParenImpCasts();
  if (Context.hasSameUnqualifiedType(FieldType, SourceValue->getType())) {
    return true;
  }
  if (IntegerConstantFits(SourceValue, FieldType, Context)) {
    return true;
  }
  if (FieldType->isPointerType() &&
      Context.hasSameUnqualifiedType(FieldType, Value->getType())) {
    return SourceValue->getType()->isPointerType() ||
           SourceValue->getType()->isArrayType() ||
           SourceValue->getType()->isFunctionType() ||
           SourceValue->isNullPointerConstant(
               Context, Expr::NPC_ValueDependentIsNotNull);
  }
  return false;
}

std::optional<std::string> SourceText(const Expr* Expression,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  if (!Expression || Expression->getBeginLoc().isMacroID() ||
      Expression->getEndLoc().isMacroID()) {
    return std::nullopt;
  }
  bool Invalid = false;
  StringRef Text = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Expression->getSourceRange()),
      SourceManager, LangOptions, &Invalid);
  if (Invalid || Text.empty()) {
    return std::nullopt;
  }
  return Text.str();
}

bool ContainsOnlyWhitespaceOrSemicolon(SourceLocation Begin, SourceLocation End,
                                       const SourceManager& SourceManager) {
  if (!IsMainFileLocation(Begin, SourceManager) ||
      !IsMainFileLocation(End, SourceManager) || Begin.isMacroID() ||
      End.isMacroID()) {
    return false;
  }
  std::pair<FileID, unsigned> BeginOffset =
      SourceManager.getDecomposedLoc(Begin);
  std::pair<FileID, unsigned> EndOffset = SourceManager.getDecomposedLoc(End);
  if (BeginOffset.first != EndOffset.first ||
      BeginOffset.second > EndOffset.second) {
    return false;
  }
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(BeginOffset.first, &Invalid);
  if (Invalid || EndOffset.second > Buffer.size()) {
    return false;
  }
  for (char Character : Buffer.slice(BeginOffset.second, EndOffset.second)) {
    if (Character != ';' &&
        !std::isspace(static_cast<unsigned char>(Character))) {
      return false;
    }
  }
  return true;
}

std::optional<CharSourceRange> WholeLineStatementRange(
    const Stmt* Statement, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  SourceLocation Begin = Statement->getBeginLoc();
  SourceLocation End = Statement->getEndLoc();
  if (!IsMainFileLocation(Begin, SourceManager) ||
      !IsMainFileLocation(End, SourceManager) || Begin.isMacroID() ||
      End.isMacroID()) {
    return std::nullopt;
  }
  End = Lexer::getLocForEndOfToken(End, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  std::pair<FileID, unsigned> BeginOffset =
      SourceManager.getDecomposedLoc(Begin);
  std::pair<FileID, unsigned> EndOffset = SourceManager.getDecomposedLoc(End);
  if (BeginOffset.first != EndOffset.first) {
    return std::nullopt;
  }
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(BeginOffset.first, &Invalid);
  if (Invalid) {
    return std::nullopt;
  }
  if (EndOffset.second < Buffer.size() && Buffer[EndOffset.second] == ';') {
    ++EndOffset.second;
  }
  size_t PreviousNewline = Buffer.rfind('\n', BeginOffset.second);
  size_t LineBegin =
      PreviousNewline == StringRef::npos ? 0 : PreviousNewline + 1;
  for (size_t I = LineBegin; I < BeginOffset.second; ++I) {
    if (Buffer[I] != ' ' && Buffer[I] != '\t') {
      return std::nullopt;
    }
  }
  size_t LineEnd = Buffer.find('\n', EndOffset.second);
  size_t ContentEnd = LineEnd == StringRef::npos ? Buffer.size() : LineEnd;
  for (size_t I = EndOffset.second; I < ContentEnd; ++I) {
    if (Buffer[I] != ' ' && Buffer[I] != '\t' && Buffer[I] != '\r') {
      return std::nullopt;
    }
  }
  size_t RemovalBegin =
      PreviousNewline == StringRef::npos ? LineBegin : PreviousNewline;
  size_t RemovalEnd = LineEnd == StringRef::npos ? Buffer.size() : LineEnd;
  SourceLocation FileBegin =
      SourceManager.getLocForStartOfFile(BeginOffset.first);
  return CharSourceRange::getCharRange(
      FileBegin.getLocWithOffset(static_cast<int>(RemovalBegin)),
      FileBegin.getLocWithOffset(static_cast<int>(RemovalEnd)));
}

std::optional<StringRef> SetupAggregateIssue(const VarDecl* Variable,
                                             const RecordDecl* Record,
                                             ASTContext& Context) {
  if (!Variable->hasLocalStorage() || Variable->isStaticLocal()) {
    return "the object does not have automatic storage";
  }
  if (Variable->getType().isVolatileQualified()) {
    return "the object is volatile";
  }
  if (Record->isUnion()) {
    return "union member activation differs between initialization and "
           "assignment";
  }
  for (const FieldDecl* Field : Record->fields()) {
    if (Field->isAnonymousStructOrUnion()) {
      return "the aggregate contains an anonymous struct or union";
    }
    if (Field->hasInClassInitializer()) {
      return "the aggregate has a default member initializer";
    }
  }
  if (!Variable->getType().isTrivialType(Context) ||
      !Variable->getType().isTriviallyCopyableType(Context)) {
    return "the aggregate has nontrivial initialization or assignment";
  }
  return std::nullopt;
}

void DiagnoseSetupIssue(DesignatedInitializerCheck& Check,
                        const InitListExpr* Initializer, StringRef Issue) {
  Check.diag(Initializer->getLBraceLoc(),
             "aggregate setup cannot be folded: %0")
      << Issue;
}

void CheckCommentLabels(DesignatedInitializerCheck& Check,
                        const InitListExpr* Initializer, ASTContext& Context,
                        const SourceManager& SourceManager) {
  // Paired syntactic forms are not children of the semantic AST. If a matcher
  // nevertheless reaches one, let the semantic form own the diagnostics.
  if (Initializer->getSemanticForm()) {
    return;
  }
  const InitListExpr* SourceInitializer = Initializer->getSyntacticForm();
  if (!SourceInitializer) {
    SourceInitializer = Initializer;
  }
  if (!SourceInitializer->isExplicit() ||
      !IsMainFileLocation(SourceInitializer->getLBraceLoc(), SourceManager)) {
    return;
  }
  std::vector<LabeledInitializer> LabeledInitializers;
  bool CanFixInitializer = true;
  for (unsigned I = 0; I < SourceInitializer->getNumInits(); ++I) {
    const Expr* Value = SourceInitializer->getInit(I);
    if (!Value) {
      CanFixInitializer = false;
      continue;
    }
    if (isa<DesignatedInitExpr>(Value)) {
      CanFixInitializer = false;
      continue;
    }
    std::optional<FieldLabel> Label = FindFieldLabel(Value, SourceManager);
    if (!Label) {
      CanFixInitializer = false;
      continue;
    }
    const FieldDecl* Field = InitializedField(SourceInitializer, I);
    if (!Field || !CanUseDirectDesignator(Field, Value, Context)) {
      CanFixInitializer = false;
      LabeledInitializers.push_back(LabeledInitializer{
          std::move(*Label),
          Field,
          LabeledInitializer::Issue::kUnrepresentable,
      });
      continue;
    }
    if (Label->name != Field->getName()) {
      CanFixInitializer = false;
      LabeledInitializers.push_back(LabeledInitializer{
          std::move(*Label),
          Field,
          LabeledInitializer::Issue::kMismatch,
      });
      continue;
    }
    LabeledInitializers.push_back(LabeledInitializer{
        std::move(*Label),
        Field,
        LabeledInitializer::Issue::kNone,
    });
  }
  for (const LabeledInitializer& Labeled : LabeledInitializers) {
    if (Labeled.issue == LabeledInitializer::Issue::kUnrepresentable) {
      Check.diag(Labeled.label.location,
                 "comment field label cannot be represented as a C++20 "
                 "designator for this aggregate");
    } else if (Labeled.issue == LabeledInitializer::Issue::kMismatch) {
      Check.diag(Labeled.label.location,
                 "comment label names '%0', but this positional initializer "
                 "selects '%1'")
          << Labeled.label.name << Labeled.field->getName();
    } else if (!CanFixInitializer) {
      Check.diag(Labeled.label.location,
                 "comment field label cannot be converted until every "
                 "initializer element is representable as a designator");
    } else {
      Check.diag(Labeled.label.location,
                 "replace comment field label with a C++20 designated "
                 "initializer")
          << FixItHint::CreateReplacement(Labeled.label.range,
                                          "." + Labeled.label.name + " = ");
    }
  }
}

void CheckSetupBlocks(DesignatedInitializerCheck& Check,
                      const CompoundStmt* Compound, ASTContext& Context,
                      const SourceManager& SourceManager) {
  ArrayRef<Stmt*> Statements = Compound->body();
  for (size_t I = 0; I < Statements.size(); ++I) {
    const auto* Declaration = dyn_cast<DeclStmt>(Statements[I]);
    if (!Declaration || !Declaration->isSingleDecl()) {
      continue;
    }
    const auto* Variable = dyn_cast<VarDecl>(Declaration->getSingleDecl());
    if (!Variable ||
        !IsMainFileLocation(Variable->getLocation(), SourceManager)) {
      continue;
    }
    const InitListExpr* Initializer =
        SourceInitializerList(Variable->getInit());
    if (!Initializer || Initializer->getNumInits() != 0 ||
        I + 1 >= Statements.size() ||
        !ParseMemberAssignment(Statements[I + 1], Variable)) {
      continue;
    }
    const RecordDecl* Record = DefinedRecord(Variable->getType());
    if (!Record) {
      continue;
    }
    if (const auto* CxxRecord = dyn_cast<CXXRecordDecl>(Record)) {
      if (!CxxRecord->isAggregate()) {
        continue;
      }
      if (CxxRecord->getNumBases() != 0) {
        DiagnoseSetupIssue(Check, Initializer,
                           "base subobjects cannot be named by C++20 "
                           "designators");
        continue;
      }
    }
    if (Initializer->getLBraceLoc().isMacroID() ||
        Initializer->getRBraceLoc().isMacroID()) {
      DiagnoseSetupIssue(Check, Initializer,
                         "the empty initializer is produced by a macro");
      continue;
    }
    SourceLocation InitializerInteriorBegin = Lexer::getLocForEndOfToken(
        Initializer->getLBraceLoc(), 0, SourceManager, Context.getLangOpts());
    SourceLocation InitializerEnd = Lexer::getLocForEndOfToken(
        Initializer->getRBraceLoc(), 0, SourceManager, Context.getLangOpts());
    if (InitializerInteriorBegin.isInvalid() || InitializerEnd.isInvalid() ||
        !ContainsOnlyWhitespaceOrSemicolon(InitializerInteriorBegin,
                                           Initializer->getRBraceLoc(),
                                           SourceManager)) {
      DiagnoseSetupIssue(
          Check, Initializer,
          "the source spelling cannot be rewritten without moving "
          "comments or macros");
      continue;
    }
    if (std::optional<StringRef> Issue =
            SetupAggregateIssue(Variable, Record, Context)) {
      DiagnoseSetupIssue(Check, Initializer, *Issue);
      continue;
    }

    std::vector<MemberAssignment> Assignments;
    std::optional<StringRef> Rejection;
    unsigned PreviousFieldIndex = 0;
    bool HasPreviousField = false;
    SourceLocation PreviousStatementEnd = InitializerEnd;
    size_t NextStatement = I + 1;
    for (; NextStatement < Statements.size(); ++NextStatement) {
      std::optional<ParsedMemberAssignment> Parsed =
          ParseMemberAssignment(Statements[NextStatement], Variable);
      if (!Parsed) {
        break;
      }
      std::optional<unsigned> Index = FieldIndex(Record, Parsed->field);
      std::optional<std::string> ValueText =
          SourceText(Parsed->value, SourceManager, Context.getLangOpts());
      std::optional<CharSourceRange> RemovalRange = WholeLineStatementRange(
          Statements[NextStatement], SourceManager, Context.getLangOpts());
      bool PreservesInterstatementText = ContainsOnlyWhitespaceOrSemicolon(
          PreviousStatementEnd, Statements[NextStatement]->getBeginLoc(),
          SourceManager);
      if (!Index || Parsed->field->getParent()->getCanonicalDecl() !=
                        Record->getCanonicalDecl()) {
        Rejection = "an assignment does not name a direct aggregate member";
      } else if (HasPreviousField && *Index <= PreviousFieldIndex) {
        Rejection = "member assignments are not in declaration order";
      } else if (!CanMoveAssignmentValue(Parsed->value, Parsed->field, Variable,
                                         Context, SourceManager)) {
        Rejection = "an assignment value may change initialization semantics";
      } else if (!ValueText || !RemovalRange || !PreservesInterstatementText) {
        Rejection =
            "the source spelling cannot be rewritten without moving "
            "comments or macros";
      }
      if (Rejection) {
        Assignments.clear();
        break;
      }
      Assignments.push_back(MemberAssignment{
          Parsed->field,
          std::move(*ValueText),
          *RemovalRange,
      });
      PreviousFieldIndex = *Index;
      HasPreviousField = true;
      PreviousStatementEnd =
          Lexer::getLocForEndOfToken(Statements[NextStatement]->getEndLoc(), 0,
                                     SourceManager, Context.getLangOpts());
      if (PreviousStatementEnd.isInvalid()) {
        Rejection =
            "the source spelling cannot be rewritten without moving "
            "comments or macros";
        Assignments.clear();
        break;
      }
    }
    if (Rejection) {
      DiagnoseSetupIssue(Check, Initializer, *Rejection);
      continue;
    }
    if (Assignments.empty()) {
      continue;
    }
    bool HasLaterAssignment = false;
    for (size_t J = NextStatement; J < Statements.size(); ++J) {
      if (ContainsMemberAssignment(Statements[J], Variable)) {
        HasLaterAssignment = true;
        break;
      }
    }
    if (HasLaterAssignment) {
      DiagnoseSetupIssue(
          Check, Initializer,
          "member assignments are separated by observation or control flow");
      continue;
    }

    std::string Replacement = "{";
    for (size_t J = 0; J < Assignments.size(); ++J) {
      if (J != 0) {
        Replacement.append(", ");
      }
      Replacement.append(".");
      Replacement.append(Assignments[J].field->getNameAsString());
      Replacement.append(" = ");
      Replacement.append(Assignments[J].value_text);
    }
    Replacement.append("}");

    DiagnosticBuilder Diagnostic =
        Check.diag(Initializer->getLBraceLoc(),
                   "fold aggregate setup into C++20 designated initialization");
    Diagnostic << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(Initializer->getSourceRange()),
        Replacement);
    Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
        Assignments.front().removal_range.getBegin(),
        Assignments.back().removal_range.getEnd()));
  }
}

}  // namespace

DesignatedInitializerCheck::DesignatedInitializerCheck(
    StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      enable_comment_label_conversion_(
          Options.get("EnableCommentLabelConversion", true)),
      enable_setup_block_folding_(
          Options.get("EnableSetupBlockFolding", true)) {}

void DesignatedInitializerCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  if (!getLangOpts().CPlusPlus20) {
    return;
  }
  using namespace ast_matchers;
  if (enable_comment_label_conversion_) {
    Finder->addMatcher(initListExpr(isExpansionInMainFile()).bind("init_list"),
                       this);
  }
  if (enable_setup_block_folding_) {
    Finder->addMatcher(compoundStmt(isExpansionInMainFile(),
                                    has(declStmt(hasSingleDecl(varDecl(
                                        hasInitializer(initListExpr()))))))
                           .bind("compound"),
                       this);
  }
}

void DesignatedInitializerCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  if (const auto* Initializer =
          Result.Nodes.getNodeAs<InitListExpr>("init_list")) {
    CheckCommentLabels(*this, Initializer, *Result.Context,
                       *Result.SourceManager);
  }
  if (const auto* Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound")) {
    CheckSetupBlocks(*this, Compound, *Result.Context, *Result.SourceManager);
  }
}

void DesignatedInitializerCheck::storeOptions(
    ClangTidyOptions::OptionMap& Options) {
  this->Options.store(Options, "EnableCommentLabelConversion",
                      enable_comment_label_conversion_);
  this->Options.store(Options, "EnableSetupBlockFolding",
                      enable_setup_block_folding_);
}

}  // namespace clang::tidy::iree
