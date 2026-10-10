// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/DesignatedInitializerCheck.h"

#include <algorithm>
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
#include "llvm/Support/YAMLParser.h"

namespace clang::tidy::iree {
namespace {

// Clang-tidy's boolean option accessors are out-of-line template
// specializations that some distributions do not export to loadable plugins.
// Use the exported string and YAML APIs while preserving upstream semantics.
bool GetBooleanOption(ClangTidyCheck::OptionsView& Options, StringRef CheckName,
                      StringRef OptionName, bool Default,
                      ClangTidyContext* Context) {
  std::optional<StringRef> Value = Options.get(OptionName);
  if (!Value) {
    return Default;
  }
  if (std::optional<bool> Parsed = llvm::yaml::parseBool(*Value)) {
    return *Parsed;
  }
  long long Number = 0;
  if (!Value->getAsInteger(10, Number)) {
    return Number != 0;
  }

  std::string FullName = (CheckName + "." + OptionName).str();
  Context->configurationDiag(
      "invalid configuration value '%0' for option '%1'; expected a bool")
      << *Value << FullName;
  return Default;
}

void StoreBooleanOption(ClangTidyCheck::OptionsView& Options,
                        ClangTidyOptions::OptionMap& OptionMap,
                        StringRef OptionName, bool Value) {
  Options.store(OptionMap, OptionName,
                Value ? StringRef("true") : StringRef("false"));
}

struct FieldLabel {
  SourceLocation location;
  CharSourceRange range;
  std::string name;
};

struct MemberAssignment {
  const FieldDecl* field;
  std::string value_text;
  CharSourceRange removal_range;
  unsigned field_index;
  bool has_side_effects;
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

// An invalid insertion location means the initializer already has the desired
// spelling or fits on one line. A missing result means its source could not be
// analyzed safely and prevents a fix for that initializer.
std::optional<SourceLocation> TrailingCommaInsertion(
    const InitListExpr* Initializer, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  if (Initializer->getNumInits() == 0) {
    return SourceLocation();
  }
  CharSourceRange InitializerRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Initializer->getSourceRange()),
      SourceManager, LangOptions);
  CharSourceRange LastValueRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(
          Initializer->getInit(Initializer->getNumInits() - 1)
              ->getSourceRange()),
      SourceManager, LangOptions);
  if (InitializerRange.isInvalid() || LastValueRange.isInvalid()) {
    return std::nullopt;
  }
  bool Invalid = false;
  StringRef InitializerText = Lexer::getSourceText(
      InitializerRange, SourceManager, LangOptions, &Invalid);
  if (Invalid) {
    return std::nullopt;
  }
  if (!InitializerText.contains('\n')) {
    return SourceLocation();
  }
  SourceLocation LastToken = LastValueRange.getEnd().getLocWithOffset(-1);
  std::optional<Token> Next = Lexer::findNextToken(
      LastToken, SourceManager, LangOptions, /*IncludeComments=*/false);
  if (!Next) {
    return std::nullopt;
  }
  if (Next->is(tok::comma)) {
    return SourceLocation();
  }
  if (!Next->is(tok::r_brace)) {
    return std::nullopt;
  }
  return LastValueRange.getEnd();
}

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
                                         const SourceManager& SourceManager,
                                         const LangOptions& LangOptions) {
  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Initializer->getSourceRange()),
      SourceManager, LangOptions);
  if (FileRange.isInvalid()) {
    return std::nullopt;
  }
  SourceLocation Begin = FileRange.getBegin();
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

bool IsMainFileInitializerBrace(SourceLocation Location,
                                const SourceManager& SourceManager) {
  if (Location.isInvalid()) {
    return false;
  }
  if (Location.isMacroID()) {
    if (!SourceManager.isMacroArgExpansion(Location)) {
      return false;
    }
    Location = SourceManager.getSpellingLoc(Location);
  }
  return IsMainFileLocation(Location, SourceManager);
}

void CheckDesignatedTrailingComma(DesignatedInitializerCheck& Check,
                                  const InitListExpr* Initializer,
                                  ASTContext& Context,
                                  const SourceManager& SourceManager) {
  // Paired syntactic forms are not children of the semantic AST. If a matcher
  // nevertheless reaches one, let the semantic form own the diagnostic.
  if (Initializer->getSemanticForm()) {
    return;
  }
  const InitListExpr* SourceInitializer = Initializer->getSyntacticForm();
  if (!SourceInitializer) {
    SourceInitializer = Initializer;
  }
  if (!SourceInitializer->isExplicit() ||
      !IsMainFileInitializerBrace(SourceInitializer->getLBraceLoc(),
                                  SourceManager)) {
    return;
  }
  bool HasDesignator = false;
  for (const Expr* Value : SourceInitializer->inits()) {
    HasDesignator |= isa<DesignatedInitExpr>(Value);
  }
  if (!HasDesignator) {
    return;
  }
  std::optional<SourceLocation> CommaInsertion = TrailingCommaInsertion(
      SourceInitializer, SourceManager, Context.getLangOpts());
  if (!CommaInsertion || !CommaInsertion->isValid()) {
    return;
  }
  Check.diag(*CommaInsertion,
             "add a trailing comma to multiline designated initializer")
      << FixItHint::CreateInsertion(*CommaInsertion, ",");
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

const FieldDecl* DirectlyDesignatableField(const FieldDecl* Field,
                                           const Expr* Initializer) {
  if (!Field || !Field->isAnonymousStructOrUnion()) {
    return Field;
  }
  const RecordDecl* AnonymousRecord = DefinedRecord(Field->getType());
  if (!AnonymousRecord || !AnonymousRecord->isUnion()) {
    return Field;
  }
  if (const InitListExpr* NestedInitializer =
          SourceInitializerList(Initializer)) {
    return InitializedField(NestedInitializer, 0);
  }
  std::vector<const FieldDecl*> Fields = InitializableFields(AnonymousRecord);
  return Fields.empty() ? nullptr : Fields.front();
}

bool CanUseDirectDesignator(const FieldDecl* Field, const Expr* Initializer,
                            ASTContext& Context) {
  if (!Field || Field->getName().empty() || Field->isAnonymousStructOrUnion()) {
    return false;
  }
  QualType FieldType = Field->getType();
  const Expr* SourceExpression = Initializer->IgnoreParenImpCasts();
  if (const auto* List = dyn_cast<InitListExpr>(SourceExpression)) {
    // An empty braced scalar initializer value-initializes the selected
    // subobject before and after designation. It is also accepted under
    // -Wbraced-scalar-init, unlike populated scalar initializer lists.
    if (List->getNumInits() == 0) {
      return true;
    }
    // Braces around the anonymous aggregate containing a promoted scalar
    // member become braces around the scalar itself after direct designation.
    // Besides triggering -Wbraced-scalar-init, removing those braces would
    // change list-initialization and narrowing semantics. Leave the source for
    // an evidence-backed repair instead.
    return FieldType->isArrayType() || DefinedRecord(FieldType);
  }
  if (Context.hasSameUnqualifiedType(FieldType, SourceExpression->getType())) {
    return true;
  }
  if (FieldType->isArrayType()) {
    return isa<StringLiteral>(SourceExpression);
  }
  const auto* FieldRecord = FieldType->getAsCXXRecordDecl();
  return !FieldRecord || !FieldRecord->isAggregate();
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

const VarDecl* PromotedMemberBaseVariable(const MemberExpr* Member) {
  const Expr* Base = Member->getBase()->IgnoreParenImpCasts();
  while (const auto* BaseMember = dyn_cast<MemberExpr>(Base)) {
    const auto* BaseField = dyn_cast<FieldDecl>(BaseMember->getMemberDecl());
    if (!BaseField || !BaseField->isAnonymousStructOrUnion()) {
      return nullptr;
    }
    Base = BaseMember->getBase()->IgnoreParenImpCasts();
  }
  return ReferencedVariable(Base);
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
      PromotedMemberBaseVariable(Member) != Variable) {
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
  if (const auto* TypeTrait = dyn_cast<UnaryExprOrTypeTraitExpr>(Statement)) {
    // Moving an unevaluated reference into the object's initializer does not
    // make it observe the object under construction. Keep variably modified
    // operands conservative because evaluating their bound is observable.
    if (!TypeTrait->getTypeOfArgument()->isVariablyModifiedType()) {
      return false;
    }
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

std::optional<unsigned> DesignatorFieldIndex(const RecordDecl* Record,
                                             const FieldDecl* Field) {
  if (Field->getParent()->getCanonicalDecl() == Record->getCanonicalDecl()) {
    return FieldIndex(Record, Field);
  }
  unsigned Index = 0;
  for (const FieldDecl* Candidate : Record->fields()) {
    if (Candidate->isUnnamedBitField()) {
      continue;
    }
    const RecordDecl* AnonymousRecord = DefinedRecord(Candidate->getType());
    if (Candidate->isAnonymousStructOrUnion() && AnonymousRecord &&
        AnonymousRecord->isUnion() &&
        AnonymousRecord->getCanonicalDecl() ==
            Field->getParent()->getCanonicalDecl()) {
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
                            const VarDecl* Variable, ASTContext& Context) {
  if (!Value || ReferencesVariable(Value, Variable)) {
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
  if (FieldType->isPointerType()) {
    // The assignment's semantic analysis already proved that its right-hand
    // side has an implicit conversion sequence to the pointer field. Pointer
    // copy-initialization accepts the same sequence; unlike arithmetic list
    // initialization, it does not introduce a narrowing restriction.
    return true;
  }
  return false;
}

std::optional<std::string> SourceText(const Expr* Expression,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  if (!Expression) {
    return std::nullopt;
  }
  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Expression->getSourceRange()),
      SourceManager, LangOptions);
  if (FileRange.isInvalid() ||
      !IsMainFileLocation(FileRange.getBegin(), SourceManager) ||
      !IsMainFileLocation(FileRange.getEnd(), SourceManager)) {
    return std::nullopt;
  }
  bool Invalid = false;
  StringRef Text =
      Lexer::getSourceText(FileRange, SourceManager, LangOptions, &Invalid);
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
  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Statement->getSourceRange()),
      SourceManager, LangOptions);
  if (FileRange.isInvalid()) {
    return std::nullopt;
  }
  SourceLocation Begin = FileRange.getBegin();
  SourceLocation End = FileRange.getEnd();
  if (!IsMainFileLocation(Begin, SourceManager) ||
      !IsMainFileLocation(End, SourceManager)) {
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

bool CanFoldSetupAggregate(const VarDecl* Variable, const RecordDecl* Record,
                           ASTContext& Context) {
  if (!Variable->hasLocalStorage() || Variable->isStaticLocal()) {
    return false;
  }
  if (Variable->getType().isVolatileQualified()) {
    return false;
  }
  if (Record->isUnion()) {
    return false;
  }
  for (const FieldDecl* Field : Record->fields()) {
    if (Field->isAnonymousStructOrUnion()) {
      const RecordDecl* AnonymousRecord = DefinedRecord(Field->getType());
      if (!AnonymousRecord || !AnonymousRecord->isUnion()) {
        return false;
      }
    }
    if (Field->hasInClassInitializer()) {
      return false;
    }
  }
  if (!Variable->getType().isTrivialType(Context) ||
      !Variable->getType().isTriviallyCopyableType(Context)) {
    return false;
  }
  return true;
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
      !IsMainFileInitializerBrace(SourceInitializer->getLBraceLoc(),
                                  SourceManager)) {
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
    std::optional<FieldLabel> Label =
        FindFieldLabel(Value, SourceManager, Context.getLangOpts());
    if (!Label) {
      CanFixInitializer = false;
      continue;
    }
    const FieldDecl* Field = DirectlyDesignatableField(
        InitializedField(SourceInitializer, I), Value);
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
  std::optional<SourceLocation> CommaInsertion = TrailingCommaInsertion(
      SourceInitializer, SourceManager, Context.getLangOpts());
  CanFixInitializer &= CommaInsertion.has_value();
  for (size_t I = 0; I < LabeledInitializers.size(); ++I) {
    const LabeledInitializer& Labeled = LabeledInitializers[I];
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
      DiagnosticBuilder Diagnostic = Check.diag(
          Labeled.label.location,
          "replace comment field label with a C++20 designated initializer");
      Diagnostic << FixItHint::CreateReplacement(
          Labeled.label.range, "." + Labeled.label.name + " = ");
      if (I + 1 == LabeledInitializers.size() && CommaInsertion->isValid()) {
        Diagnostic << FixItHint::CreateInsertion(*CommaInsertion, ",");
      }
    }
  }
}

void CheckDesignatedCompoundLiteral(DesignatedInitializerCheck& Check,
                                    const CompoundLiteralExpr* Literal,
                                    ASTContext& Context,
                                    const SourceManager& SourceManager) {
  const InitListExpr* Initializer =
      SourceInitializerList(Literal->getInitializer());
  if (!Initializer) {
    return;
  }

  bool UsesDesignatedInitialization = false;
  for (const Expr* Value : Initializer->inits()) {
    if (isa<DesignatedInitExpr>(Value) ||
        FindFieldLabel(Value, SourceManager, Context.getLangOpts())) {
      UsesDesignatedInitialization = true;
      break;
    }
  }
  if (!UsesDesignatedInitialization) {
    return;
  }

  SourceLocation LParen = Literal->getLParenLoc();
  SourceLocation TypeEnd =
      Literal->getTypeSourceInfo()->getTypeLoc().getEndLoc();
  std::optional<Token> RParen = Lexer::findNextToken(
      TypeEnd, SourceManager, Context.getLangOpts(), /*IncludeComments=*/false);
  if (!IsMainFileLocation(LParen, SourceManager) || LParen.isMacroID() ||
      !RParen || !RParen->is(tok::r_paren) ||
      !IsMainFileLocation(RParen->getLocation(), SourceManager) ||
      RParen->getLocation().isMacroID()) {
    return;
  }

  Check.diag(LParen,
             "replace C-style compound literal with standard C++ list "
             "initialization")
      << FixItHint::CreateRemoval(
             CharSourceRange::getTokenRange(LParen, LParen))
      << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
             RParen->getLocation(), RParen->getLocation()));
}

template <typename CallExpression>
void CheckArgumentLabels(DesignatedInitializerCheck& Check,
                         const CallExpression* Call, ASTContext& Context,
                         const SourceManager& SourceManager) {
  for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
    const Expr* Argument = Call->getArg(I);
    if (!Argument) {
      continue;
    }
    std::optional<FieldLabel> Label =
        FindFieldLabel(Argument, SourceManager, Context.getLangOpts());
    if (!Label) {
      continue;
    }
    Check.diag(Label->location,
               "replace aggregate-style comment label on call argument with "
               "a parameter label")
        << FixItHint::CreateReplacement(Label->range,
                                        "/*" + Label->name + "=*/");
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
        continue;
      }
    }
    if (Initializer->getLBraceLoc().isMacroID() ||
        Initializer->getRBraceLoc().isMacroID()) {
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
      continue;
    }
    if (!CanFoldSetupAggregate(Variable, Record, Context)) {
      continue;
    }

    std::vector<MemberAssignment> Assignments;
    SourceLocation PreviousStatementEnd = InitializerEnd;
    size_t NextStatement = I + 1;
    for (; NextStatement < Statements.size(); ++NextStatement) {
      std::optional<ParsedMemberAssignment> Parsed =
          ParseMemberAssignment(Statements[NextStatement], Variable);
      if (!Parsed) {
        break;
      }
      std::optional<unsigned> Index =
          DesignatorFieldIndex(Record, Parsed->field);
      std::optional<std::string> ValueText =
          SourceText(Parsed->value, SourceManager, Context.getLangOpts());
      std::optional<CharSourceRange> RemovalRange = WholeLineStatementRange(
          Statements[NextStatement], SourceManager, Context.getLangOpts());
      bool PreservesInterstatementText = ContainsOnlyWhitespaceOrSemicolon(
          PreviousStatementEnd, Statements[NextStatement]->getBeginLoc(),
          SourceManager);
      if (!Index ||
          !CanMoveAssignmentValue(Parsed->value, Parsed->field, Variable,
                                  Context) ||
          !ValueText || !RemovalRange || !PreservesInterstatementText) {
        break;
      }
      Assignments.push_back(MemberAssignment{
          Parsed->field,
          std::move(*ValueText),
          *RemovalRange,
          *Index,
          Parsed->value->HasSideEffects(Context),
      });
      PreviousStatementEnd = RemovalRange->getEnd();
    }
    if (Assignments.empty()) {
      continue;
    }

    SourceLocation RemovalBegin = Assignments.front().removal_range.getBegin();
    size_t FoldedAssignmentCount = Assignments.size();
    bool IsDeclarationOrder = true;
    bool HasDuplicateField = false;
    for (size_t J = 1; J < Assignments.size(); ++J) {
      if (Assignments[J].field_index <= Assignments[J - 1].field_index) {
        IsDeclarationOrder = false;
      }
      for (size_t K = 0; K < J; ++K) {
        HasDuplicateField |=
            Assignments[J].field_index == Assignments[K].field_index;
      }
    }
    if (!IsDeclarationOrder) {
      bool HasSideEffects = false;
      for (const MemberAssignment& Assignment : Assignments) {
        HasSideEffects |= Assignment.has_side_effects;
      }
      if (!HasSideEffects && !HasDuplicateField) {
        std::stable_sort(
            Assignments.begin(), Assignments.end(),
            [](const MemberAssignment& Lhs, const MemberAssignment& Rhs) {
              return Lhs.field_index < Rhs.field_index;
            });
      } else {
        FoldedAssignmentCount = 1;
        while (FoldedAssignmentCount < Assignments.size() &&
               Assignments[FoldedAssignmentCount - 1].field_index <
                   Assignments[FoldedAssignmentCount].field_index) {
          ++FoldedAssignmentCount;
        }
        Assignments.resize(FoldedAssignmentCount);
      }
    }

    std::optional<CharSourceRange> LastRemovalRange =
        WholeLineStatementRange(Statements[I + FoldedAssignmentCount],
                                SourceManager, Context.getLangOpts());
    if (!LastRemovalRange) {
      continue;
    }
    SourceLocation RemovalEnd = LastRemovalRange->getEnd();

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
    Replacement.append(",}");

    DiagnosticBuilder Diagnostic =
        Check.diag(Initializer->getLBraceLoc(),
                   "fold aggregate setup into C++20 designated initialization");
    Diagnostic << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(Initializer->getSourceRange()),
        Replacement);
    Diagnostic << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(RemovalBegin, RemovalEnd));
  }
}

}  // namespace

DesignatedInitializerCheck::DesignatedInitializerCheck(
    StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      enable_comment_label_conversion_(GetBooleanOption(
          Options, Name, "EnableCommentLabelConversion", true, Context)),
      enable_setup_block_folding_(GetBooleanOption(
          Options, Name, "EnableSetupBlockFolding", true, Context)) {}

void DesignatedInitializerCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  if (!getLangOpts().CPlusPlus20) {
    return;
  }
  using namespace ast_matchers;
  Finder->addMatcher(
      compoundLiteralExpr(isExpansionInMainFile()).bind("compound_literal"),
      this);
  Finder->addMatcher(initListExpr(isExpansionInMainFile()).bind("init_list"),
                     this);
  if (enable_comment_label_conversion_) {
    Finder->addMatcher(callExpr(isExpansionInMainFile()).bind("call"), this);
    Finder->addMatcher(
        cxxConstructExpr(isExpansionInMainFile()).bind("constructor"), this);
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
  if (const auto* Literal =
          Result.Nodes.getNodeAs<CompoundLiteralExpr>("compound_literal")) {
    CheckDesignatedCompoundLiteral(*this, Literal, *Result.Context,
                                   *Result.SourceManager);
  }
  if (const auto* Initializer =
          Result.Nodes.getNodeAs<InitListExpr>("init_list")) {
    CheckDesignatedTrailingComma(*this, Initializer, *Result.Context,
                                 *Result.SourceManager);
    if (enable_comment_label_conversion_) {
      CheckCommentLabels(*this, Initializer, *Result.Context,
                         *Result.SourceManager);
    }
  }
  if (const auto* Call = Result.Nodes.getNodeAs<CallExpr>("call")) {
    CheckArgumentLabels(*this, Call, *Result.Context, *Result.SourceManager);
  }
  if (const auto* Constructor =
          Result.Nodes.getNodeAs<CXXConstructExpr>("constructor")) {
    // Implicit copy construction of an aggregate member has no source-spelled
    // argument delimiters. Its argument may still be preceded by the enclosing
    // aggregate's field label, which the InitListExpr matcher owns.
    if (Constructor->getParenOrBraceRange().isValid()) {
      CheckArgumentLabels(*this, Constructor, *Result.Context,
                          *Result.SourceManager);
    }
  }
  if (const auto* Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound")) {
    CheckSetupBlocks(*this, Compound, *Result.Context, *Result.SourceManager);
  }
}

void DesignatedInitializerCheck::storeOptions(
    ClangTidyOptions::OptionMap& Options) {
  StoreBooleanOption(this->Options, Options, "EnableCommentLabelConversion",
                     enable_comment_label_conversion_);
  StoreBooleanOption(this->Options, Options, "EnableSetupBlockFolding",
                     enable_setup_block_folding_);
}

}  // namespace clang::tidy::iree
