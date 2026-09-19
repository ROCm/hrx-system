// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>

namespace loom::cxx_import {

ControlFlow::ControlFlow(cxx::TranslationUnit& unit, cxx::StatementAST* body)
    : unit_(unit) {
  accept(body);
}

std::span<cxx::Symbol* const> ControlFlow::written(cxx::AST* owner) const {
  auto found = writes_.find(owner);
  return found == writes_.end() ? std::span<cxx::Symbol* const>{}
                                : found->second;
}

const CountedLoop* ControlFlow::counted(cxx::ForStatementAST* loop) const {
  auto found = counted_.find(loop);
  return found == counted_.end() ? nullptr : &found->second;
}

ReturnFlow ControlFlow::returns(cxx::StatementAST* statement) const {
  auto found = returns_.find(statement);
  return found == returns_.end() ? ReturnFlow::None : found->second;
}

bool ControlFlow::preVisit(cxx::AST* ast) {
  if (structured(ast)) {
    owners_.push_back(ast);
  }
  return true;
}

void ControlFlow::postVisit(cxx::AST* ast) {
  if (auto* statement = cxx::ast_cast<cxx::StatementAST>(ast)) {
    auto flow = classify_returns(statement);
    if (flow != ReturnFlow::None) {
      returns_.emplace(statement, flow);
    }
  }
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(ast)) {
    if (auto counted = classify(loop)) {
      counted_.emplace(loop, *counted);
    }
  }
  if (structured(ast)) {
    owners_.pop_back();
  }
}

ReturnFlow ControlFlow::classify_returns(cxx::StatementAST* statement) const {
  if (cxx::ast_cast<cxx::ReturnStatementAST>(statement)) {
    return ReturnFlow::All;
  }
  if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(statement)) {
    auto flow = ReturnFlow::None;
    for (auto* child : cxx::ListView{compound->statementList}) {
      auto child_flow = returns(child);
      if (child_flow == ReturnFlow::All) {
        return ReturnFlow::All;
      }
      if (child_flow == ReturnFlow::Some) {
        flow = ReturnFlow::Some;
      }
    }
    return flow;
  }
  if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(statement)) {
    auto then_flow = returns(branch->statement);
    auto else_flow = returns(branch->elseStatement);
    return then_flow == else_flow ? then_flow : ReturnFlow::Some;
  }
  cxx::StatementAST* body = nullptr;
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(statement)) {
    return returns(loop->statement);
  }
  return returns(body) == ReturnFlow::None ? ReturnFlow::None
                                           : ReturnFlow::Some;
}

void ControlFlow::visit(cxx::AssignmentExpressionAST* ast) {
  record(ast->leftExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::CompoundAssignmentExpressionAST* ast) {
  record(ast->targetExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::PostIncrExpressionAST* ast) {
  record(ast->baseExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::UnaryExpressionAST* ast) {
  if (ast->op == cxx::TokenKind::T_PLUS_PLUS ||
      ast->op == cxx::TokenKind::T_MINUS_MINUS) {
    record(ast->expression);
  }
  cxx::ASTVisitor::visit(ast);
}

bool ControlFlow::structured(cxx::AST* ast) {
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast)) {
    return !binary->symbol && (binary->op == cxx::TokenKind::T_AMP_AMP ||
                               binary->op == cxx::TokenKind::T_BAR_BAR);
  }
  return cxx::ast_cast<cxx::IfStatementAST>(ast) ||
         cxx::ast_cast<cxx::ForStatementAST>(ast) ||
         cxx::ast_cast<cxx::WhileStatementAST>(ast) ||
         cxx::ast_cast<cxx::DoStatementAST>(ast) ||
         cxx::ast_cast<cxx::CompoundStatementAST>(ast) ||
         cxx::ast_cast<cxx::ConditionalExpressionAST>(ast);
}

void ControlFlow::record(cxx::ExpressionAST* expression) {
  while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    expression = nested->expression;
  }
  auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression);
  if (!id) {
    return;
  }
  for (auto* owner : owners_) {
    auto& writes = writes_[owner];
    if (std::ranges::find(writes, id->symbol) == writes.end()) {
      writes.push_back(id->symbol);
    }
  }
}

// The unit-step unsigned interval cannot wrap before its strict upper bound.
// A stable scalar/literal upper bound in that same unsigned width may be
// evaluated once. Wider comparisons can observe induction wraparound and
// retain their general while semantics, as do mutable bounds.
std::optional<CountedLoop> ControlFlow::classify(cxx::ForStatementAST* loop) {
  unsigned step_value = 1;
  auto* declaration =
      cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer);
  auto* initial =
      declaration
          ? cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration)
          : nullptr;
  auto* condition = cxx::ast_cast<cxx::BinaryExpressionAST>(loop->condition);
  auto* step = cxx::ast_cast<cxx::UnaryExpressionAST>(loop->expression);
  auto* compound =
      cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(loop->expression);
  if (!initial || !initial->initDeclaratorList ||
      initial->initDeclaratorList->next || !condition || condition->symbol ||
      condition->op != cxx::TokenKind::T_LESS ||
      !cxx::ast_cast<cxx::CompoundStatementAST>(loop->statement)) {
    return std::nullopt;
  }
  auto* induction = initial->initDeclaratorList->value->symbol;
  auto* left = cxx::ast_cast<cxx::IdExpressionAST>(
      cxx::Initializer::stripImplicitCasts(condition->leftExpression));
  cxx::IdExpressionAST* increment = nullptr;
  if (step && !step->symbol && step->op == cxx::TokenKind::T_PLUS_PLUS) {
    increment = cxx::ast_cast<cxx::IdExpressionAST>(step->expression);
  } else if (compound && !compound->symbol &&
             compound->op == cxx::TokenKind::T_PLUS_EQUAL &&
             cxx::ast_cast<cxx::IntLiteralExpressionAST>(
                 compound->rightExpression) &&
             cxx::ast_cast<cxx::IntLiteralExpressionAST>(
                 condition->rightExpression)) {
    cxx::ASTInterpreter interpreter(&unit_);
    auto amount =
        interpreter.toInt(*interpreter.evaluate(compound->rightExpression));
    auto upper =
        interpreter.toInt(*interpreter.evaluate(condition->rightExpression));
    if (amount && upper && *amount > 0 && *amount <= UINT32_MAX &&
        *upper >= 0 && *upper <= UINT32_MAX - *amount + 1) {
      step_value = static_cast<unsigned>(*amount);
      increment =
          cxx::ast_cast<cxx::IdExpressionAST>(compound->targetExpression);
    }
  }
  auto* upper =
      cxx::Initializer::stripImplicitCasts(condition->rightExpression);
  auto* bound = cxx::ast_cast<cxx::IdExpressionAST>(upper);
  if (!left || !increment || left->symbol != induction ||
      increment->symbol != induction ||
      unit_.typeTraits().remove_cv(induction->type())->kind() !=
          cxx::TypeKind::kUnsignedInt ||
      unit_.typeTraits().remove_cv(condition->leftExpression->type)->kind() !=
          cxx::TypeKind::kUnsignedInt ||
      unit_.typeTraits().remove_cv(condition->rightExpression->type)->kind() !=
          cxx::TypeKind::kUnsignedInt ||
      (!bound && !cxx::ast_cast<cxx::IntLiteralExpressionAST>(upper))) {
    return std::nullopt;
  }
  const auto& body_writes = written(loop->statement);
  const auto& loop_writes = written(loop);
  if (std::ranges::find(body_writes, induction) != body_writes.end() ||
      (bound &&
       std::ranges::find(loop_writes, bound->symbol) != loop_writes.end())) {
    return std::nullopt;
  }
  return CountedLoop{induction, condition->rightExpression, step_value};
}

}  // namespace loom::cxx_import
