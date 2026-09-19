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
#include <vector>

namespace loom::cxx_import {

// Retained proof that a source loop has a stable, nonwrapping unsigned
// interval.
struct CountedLoop {
  // Source binding replaced by the structured loop's induction argument.
  cxx::Symbol* induction;
  // Stable upper bound, evaluated once by the composition driver.
  cxx::ExpressionAST* upper;
  // Positive constant step in the source's unsigned-int width.
  unsigned step;
};

// Source paths through a statement that reach a function return. Fallthrough
// remains possible for None and Some; All terminates every syntactic path.
enum class ReturnFlow { None, Some, All };

// One traversal owns control facts for an immutable source function body.
// Ordered writes include nested constructs and preserve source symbol identity.
// Return outcomes aggregate each statement's already visited children, and
// counted-loop classification happens after its writes are complete. Queries
// only consume retained facts; they do not traverse source or output IR.
// The source unit and body outlive this object and every returned reference.
class ControlFlow final : private cxx::ASTVisitor {
 public:
  ControlFlow(cxx::TranslationUnit& unit, cxx::StatementAST* body);
  ControlFlow(const ControlFlow&) = delete;
  ControlFlow& operator=(const ControlFlow&) = delete;

  // Unique bindings mutated under a structured statement or conditional value
  // expression, in encounter order. Includes mutations in conditions.
  std::span<cxx::Symbol* const> written(cxx::AST* owner) const;
  // Null retains ordinary while semantics; a result permits scf.for lowering.
  const CountedLoop* counted(cxx::ForStatementAST* loop) const;
  // Retained return/fallthrough summary, including nested statements.
  ReturnFlow returns(cxx::StatementAST* statement) const;

 private:
  bool preVisit(cxx::AST* ast) override;
  void postVisit(cxx::AST* ast) override;
  void visit(cxx::AssignmentExpressionAST* ast) override;
  void visit(cxx::CompoundAssignmentExpressionAST* ast) override;
  void visit(cxx::PostIncrExpressionAST* ast) override;
  void visit(cxx::UnaryExpressionAST* ast) override;
  static bool structured(cxx::AST* ast);
  void record(cxx::ExpressionAST* expression);
  std::optional<CountedLoop> classify(cxx::ForStatementAST* loop);
  ReturnFlow classify_returns(cxx::StatementAST* statement) const;

  // Resolved source types and literal interpretation for loop admission.
  cxx::TranslationUnit& unit_;
  // Active structured ancestors during construction only.
  std::vector<cxx::AST*> owners_;
  // Stable encounter order determines region argument/result order.
  std::unordered_map<cxx::AST*, std::vector<cxx::Symbol*>> writes_;
  // Proven intervals retained after each source loop's children are visited.
  std::unordered_map<cxx::ForStatementAST*, CountedLoop> counted_;
  // Sparse summaries retain only statements with function-return paths.
  std::unordered_map<cxx::StatementAST*, ReturnFlow> returns_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_
