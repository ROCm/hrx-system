// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_
#define LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_

#include <cxx/ast_visitor.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// Retained ownership of a source assignment destination. A field write replaces
// a slice of its owning automatic binding; it is not a storage-backed store.
struct Destination {
  // Owning local or parameter identity, borrowed from the source function.
  cxx::Symbol* binding;
  // First component of a nested member in the complete owning binding.
  size_t component_offset;
  // Member structure, or null when the destination is the entire binding.
  const Partition* member;
};

// Retained proof that a source loop has a stable, nonwrapping unsigned
// interval.
struct CountedLoop {
  struct Bound {
    // Resolved identifier supplying the loop-invariant bound.
    cxx::Symbol* binding;
    // Evaluated source expression, including its integer promotions.
    cxx::ExpressionAST* expression;
  };
  // Source binding replaced by the structured loop's induction argument.
  cxx::Symbol* induction;
  // Proven constant or stable runtime bound evaluated once by translation.
  std::variant<unsigned, Bound> upper;
  // Positive constant step in the source's unsigned-int width.
  unsigned step;
};

// Syntactic paths through a statement that take a particular exit. Other
// outcomes remain possible for None and Some; All takes that exit on every
// path.
enum class ExitFlow { None, Some, All };

// One analysis owns control facts for an immutable source function body.
// Ordered writes include nested constructs and preserve source symbol identity.
// Constexpr branches visit their initializers, decision and source-selected
// arm. Exit outcomes aggregate each statement's already visited children, and
// counted-loop classification evaluates each header's constants once after its
// writes are complete. Queries only consume retained facts; they do not
// traverse source or output IR. The source unit and body outlive this object
// and every returned reference.
class ControlFlow final : private cxx::ASTVisitor {
 public:
  ControlFlow(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
              Types& types, cxx::StatementAST* body);
  ControlFlow(const ControlFlow&) = delete;
  ControlFlow& operator=(const ControlFlow&) = delete;

  // Unique bindings mutated under a structured statement or conditional value
  // expression, in encounter order. Includes mutations in conditions.
  std::span<cxx::Symbol* const> written(cxx::AST* owner) const;
  // Whether an evaluated address expression requires this binding's object
  // identity. The complete function is classified before translation begins.
  bool addressed(cxx::Symbol* binding) const;
  // Retained automatic-object destination, or no value for a memory access or
  // unsupported lvalue. Whole identifiers need no indexed projection record.
  std::optional<Destination> destination(cxx::ExpressionAST* expression) const;
  // Whether this member projects existing memory instead of an SSA record.
  // Object identity is propagated from visited children once during analysis.
  bool storage_backed(cxx::MemberExpressionAST* expression) const;
  // Null retains ordinary while semantics; a result permits scf.for lowering.
  const CountedLoop* counted(cxx::ForStatementAST* loop) const;
  // Declaration for a nonnull decisionVariable retained by a statement in this
  // function. The returned syntax is borrowed from the source arena.
  cxx::ConditionExpressionAST* condition_declaration(
      cxx::VariableSymbol* variable) const;
  // Retained return/fallthrough summary, including nested statements.
  ExitFlow returns(cxx::StatementAST* statement) const;
  // Continues escaping this statement to its enclosing loop. A nested loop
  // consumes its own continues rather than propagating them to its parent.
  ExitFlow continues(cxx::StatementAST* statement) const;

 private:
  enum Path : unsigned { Fallthrough = 1, Return = 2, Continue = 4 };

  bool preVisit(cxx::AST* ast) override;
  void postVisit(cxx::AST* ast) override;
  void visit(cxx::IfStatementAST* ast) override;
  void visit(cxx::ConditionExpressionAST* ast) override;
  void visit(cxx::AssignmentExpressionAST* ast) override;
  void visit(cxx::CompoundAssignmentExpressionAST* ast) override;
  void visit(cxx::PostIncrExpressionAST* ast) override;
  void visit(cxx::UnaryExpressionAST* ast) override;
  static bool structured(cxx::AST* ast);
  void record(cxx::ExpressionAST* expression);
  std::optional<Destination> classify_destination(
      cxx::ExpressionAST* expression);
  bool classify_storage(cxx::ExpressionAST* expression) const;
  std::optional<CountedLoop> classify(cxx::ForStatementAST* loop);
  unsigned paths(cxx::StatementAST* statement) const;
  unsigned classify_paths(cxx::StatementAST* statement) const;
  ExitFlow exits(cxx::StatementAST* statement, Path path) const;

  // Resolved source types and literal interpretation for loop admission.
  cxx::TranslationUnit& unit_;
  // Statement annotation admission during the existing source body walk.
  Diagnostics& diagnostics_;
  // Admitted source member partitions outlive all retained destination slices.
  Types& types_;
  // Active structured ancestors during construction only.
  std::vector<cxx::AST*> owners_;
  // Stable encounter order determines region argument/result order.
  std::unordered_map<cxx::AST*, std::vector<cxx::Symbol*>> writes_;
  // Automatic bindings whose addresses occur in evaluated source expressions.
  std::unordered_set<cxx::Symbol*> addressed_;
  // Nested lvalue ownership and transitive component offsets computed once.
  std::unordered_map<cxx::ExpressionAST*, Destination> destinations_;
  // Memory record objects and member projections, including nested fields.
  std::unordered_set<cxx::ExpressionAST*> storage_expressions_;
  // Proven intervals retained after each source loop's children are visited.
  std::unordered_map<cxx::ForStatementAST*, CountedLoop> counted_;
  // Declaration syntax indexed by the statement's retained decision variable.
  std::unordered_map<cxx::VariableSymbol*, cxx::ConditionExpressionAST*>
      conditions_;
  // Sparse path sets retain statements with function or iteration exits.
  std::unordered_map<cxx::StatementAST*, unsigned> paths_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_
