// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"

namespace loom::cxx_import {
namespace {

cxx::ExpressionAST* unwrapped(cxx::ExpressionAST* expression) {
  expression = cxx::Initializer::stripImplicitCasts(expression);
  while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    expression = cxx::Initializer::stripImplicitCasts(nested->expression);
  }
  return expression;
}

}  // namespace

ControlFlow::ControlFlow(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         Types& types, cxx::StatementAST* body)
    : unit_(unit), diagnostics_(diagnostics), types_(types) {
  accept(body);
}

std::optional<Destination> ControlFlow::destination(
    cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    return Destination{id->symbol, 0, nullptr};
  }
  auto found = destinations_.find(expression);
  return found == destinations_.end() ? std::nullopt
                                      : std::optional(found->second);
}

std::span<cxx::Symbol* const> ControlFlow::written(cxx::AST* owner) const {
  auto found = writes_.find(owner);
  return found == writes_.end() ? std::span<cxx::Symbol* const>{}
                                : found->second;
}

bool ControlFlow::addressed(cxx::Symbol* binding) const {
  return addressed_.contains(binding);
}

bool ControlFlow::storage_backed(cxx::MemberExpressionAST* expression) const {
  return storage_expressions_.contains(expression);
}

const CountedLoop* ControlFlow::counted(cxx::ForStatementAST* loop) const {
  auto found = counted_.find(loop);
  if (found == counted_.end() || addressed(found->second.induction)) {
    return nullptr;
  }
  // An aliased bound can change through a helper without a syntactic write in
  // the loop. Consult complete address demand, including later source uses.
  auto* bound = std::get_if<CountedLoop::Bound>(&found->second.upper);
  return bound && addressed(bound->binding) ? nullptr : &found->second;
}

cxx::ConditionExpressionAST* ControlFlow::condition_declaration(
    cxx::VariableSymbol* variable) const {
  return conditions_.at(variable);
}

unsigned ControlFlow::paths(cxx::StatementAST* statement) const {
  auto found = paths_.find(statement);
  return found == paths_.end() ? unsigned(Fallthrough) : found->second;
}

ExitFlow ControlFlow::exits(cxx::StatementAST* statement, Path path) const {
  auto outcomes = paths(statement);
  return !(outcomes & path) ? ExitFlow::None
         : outcomes == path ? ExitFlow::All
                            : ExitFlow::Some;
}

ExitFlow ControlFlow::returns(cxx::StatementAST* statement) const {
  return exits(statement, Return);
}

ExitFlow ControlFlow::continues(cxx::StatementAST* statement) const {
  return exits(statement, Continue);
}

bool ControlFlow::preVisit(cxx::AST* ast) {
  switch (ast->kind()) {
    case cxx::ASTKind::ConstExpression:
    case cxx::ASTKind::SizeofExpression:
    case cxx::ASTKind::SizeofTypeExpression:
    case cxx::ASTKind::AlignofTypeExpression:
      return false;
    default:
      break;
  }
  if (structured(ast)) {
    owners_.push_back(ast);
  }
  return true;
}

void ControlFlow::postVisit(cxx::AST* ast) {
  if (auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast)) {
    if (classify_storage(expression)) {
      storage_expressions_.insert(expression);
    }
  }
  if (auto* statement = cxx::ast_cast<cxx::StatementAST>(ast)) {
    reject_misplaced_binding_statement(unit_, diagnostics_, statement);
    auto outcomes = classify_paths(statement);
    if (outcomes != Fallthrough) {
      paths_.emplace(statement, outcomes);
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

unsigned ControlFlow::classify_paths(cxx::StatementAST* statement) const {
  if (cxx::ast_cast<cxx::ReturnStatementAST>(statement)) {
    return Return;
  }
  if (cxx::ast_cast<cxx::ContinueStatementAST>(statement)) {
    return Continue;
  }
  if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(statement)) {
    unsigned outcomes = Fallthrough;
    for (auto* child : cxx::ListView{compound->statementList}) {
      if (!(outcomes & Fallthrough)) {
        break;
      }
      outcomes = (outcomes & ~Fallthrough) | paths(child);
    }
    return outcomes;
  }
  if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(statement)) {
    if (branch->constexprValue.has_value()) {
      return paths(*branch->constexprValue ? branch->statement
                                           : branch->elseStatement);
    }
    return paths(branch->statement) | paths(branch->elseStatement);
  }
  cxx::StatementAST* body = nullptr;
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(statement)) {
    auto body_paths = paths(loop->statement);
    return (body_paths & Return) |
           (body_paths & (Fallthrough | Continue) ? unsigned(Fallthrough) : 0);
  }
  return Fallthrough | (paths(body) & Return);
}

void ControlFlow::visit(cxx::IfStatementAST* ast) {
  if (ast->constexprValue.has_value()) {
    accept(ast->initializer);
    accept(ast->condition);
    accept(*ast->constexprValue ? ast->statement : ast->elseStatement);
  } else {
    cxx::ASTVisitor::visit(ast);
  }
}

void ControlFlow::visit(cxx::ConditionExpressionAST* ast) {
  conditions_.emplace(ast->symbol, ast);
  cxx::ASTVisitor::visit(ast);
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
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP) {
    if (auto target = classify_destination(ast->expression)) {
      addressed_.insert(target->binding);
    }
  }
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
  auto target = classify_destination(expression);
  if (!target) {
    return;
  }
  if (!cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    destinations_.emplace(expression, *target);
  }
  for (auto* owner : owners_) {
    auto& writes = writes_[owner];
    if (std::ranges::find(writes, target->binding) == writes.end()) {
      writes.push_back(target->binding);
    }
  }
}

std::optional<Destination> ControlFlow::classify_destination(
    cxx::ExpressionAST* expression) {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    return Destination{id->symbol, 0, nullptr};
  }
  if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    return classify_destination(nested->expression);
  }
  auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
  if (!member || member->accessOp != cxx::TokenKind::T_DOT) {
    return std::nullopt;
  }
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
  if (!field || field->isStatic()) {
    return std::nullopt;
  }
  // Establish the automatic owner before requesting any SSA partition: a dot
  // member can also project a record reached through a pointer or subscript.
  auto target = classify_destination(member->baseExpression);
  if (target) {
    const auto& slice = types_.member(field, member);
    target->component_offset += slice.component_offset;
    target->member = slice.partition;
  }
  return target;
}

bool ControlFlow::classify_storage(cxx::ExpressionAST* expression) const {
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    return field && !field->isStatic() &&
           (member->accessOp == cxx::TokenKind::T_MINUS_GREATER ||
            storage_expressions_.contains(member->baseExpression));
  }
  if (!expression->type || !unit_.typeTraits().is_class(expression->type)) {
    return false;
  }
  if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    return storage_expressions_.contains(nested->expression);
  }
  if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
    return storage_expressions_.contains(cast->expression);
  }
  if (auto* subscript =
          cxx::ast_cast<cxx::SubscriptExpressionAST>(expression)) {
    return !subscript->symbol;
  }
  if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression)) {
    return !unary->symbol && unary->op == cxx::TokenKind::T_STAR;
  }
  return false;
}

// The unit-step unsigned interval cannot wrap before its strict upper bound.
// Larger constant steps require room for the final increment even when the
// bound is not a multiple of the step. Pure constants and stable scalar bounds
// in that same unsigned width may be evaluated once. Wider comparisons can
// observe induction wraparound and retain their general while semantics, as do
// mutable or volatile bounds.
std::optional<CountedLoop> ControlFlow::classify(cxx::ForStatementAST* loop) {
  auto* declaration =
      cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer);
  auto* initial =
      declaration
          ? cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration)
          : nullptr;
  auto* condition =
      cxx::ast_cast<cxx::BinaryExpressionAST>(unwrapped(loop->condition));
  if (!initial || !initial->initDeclaratorList ||
      initial->initDeclaratorList->next || !condition || condition->symbol ||
      condition->op != cxx::TokenKind::T_LESS ||
      !cxx::ast_cast<cxx::CompoundStatementAST>(loop->statement)) {
    return std::nullopt;
  }
  auto* induction = initial->initDeclaratorList->value->symbol;
  auto* left =
      cxx::ast_cast<cxx::IdExpressionAST>(unwrapped(condition->leftExpression));
  auto traits = unit_.typeTraits();
  if (!left || left->symbol != induction ||
      traits.is_volatile(induction->type()) ||
      traits.remove_cv(induction->type())->kind() !=
          cxx::TypeKind::kUnsignedInt ||
      traits.remove_cv(condition->leftExpression->type)->kind() !=
          cxx::TypeKind::kUnsignedInt ||
      traits.remove_cv(condition->rightExpression->type)->kind() !=
          cxx::TypeKind::kUnsignedInt) {
    return std::nullopt;
  }

  unsigned step_value = 1;
  auto* increment_expression = unwrapped(loop->expression);
  auto* step = cxx::ast_cast<cxx::UnaryExpressionAST>(increment_expression);
  auto* post = cxx::ast_cast<cxx::PostIncrExpressionAST>(increment_expression);
  auto* compound =
      cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(increment_expression);
  cxx::IdExpressionAST* increment = nullptr;
  if (step && !step->symbol && step->op == cxx::TokenKind::T_PLUS_PLUS) {
    increment =
        cxx::ast_cast<cxx::IdExpressionAST>(unwrapped(step->expression));
  } else if (post && !post->symbol && post->op == cxx::TokenKind::T_PLUS_PLUS) {
    increment =
        cxx::ast_cast<cxx::IdExpressionAST>(unwrapped(post->baseExpression));
  } else if (compound && !compound->symbol &&
             compound->op == cxx::TokenKind::T_PLUS_EQUAL) {
    auto amount = integer_constant(unit_, compound->rightExpression);
    if (amount && *amount > 0 && *amount <= UINT32_MAX) {
      step_value = static_cast<unsigned>(*amount);
      increment = cxx::ast_cast<cxx::IdExpressionAST>(
          unwrapped(compound->targetExpression));
    }
  }
  const auto& body_writes = written(loop->statement);
  if (!increment || increment->symbol != induction ||
      std::ranges::find(body_writes, induction) != body_writes.end()) {
    return std::nullopt;
  }

  if (auto upper = integer_constant(unit_, condition->rightExpression)) {
    if (*upper >= 0 && *upper <= UINT32_MAX - step_value + 1) {
      return CountedLoop{induction, static_cast<unsigned>(*upper), step_value};
    }
    return std::nullopt;
  }
  auto* bound = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrapped(condition->rightExpression));
  const auto& loop_writes = written(loop);
  if (step_value != 1 || !bound || !traits.is_integral_or_enum(bound->type) ||
      traits.is_volatile(bound->type) ||
      std::ranges::find(loop_writes, bound->symbol) != loop_writes.end()) {
    return std::nullopt;
  }
  return CountedLoop{
      induction, CountedLoop::Bound{bound->symbol, condition->rightExpression},
      step_value};
}

}  // namespace loom::cxx_import
