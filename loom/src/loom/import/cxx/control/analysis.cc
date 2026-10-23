// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/function_body.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/expressions.h"

namespace loom::cxx_import {
namespace {

bool operation_is(cxx::Symbol* symbol, std::string_view operation) {
  auto* binding = annotation(symbol, "op");
  return binding && binding->arguments.size() == 1 &&
         binding->arguments[0]->name() == operation;
}

bool operation_is(cxx::Symbol* symbol, std::string_view operation,
                  std::string_view parameter) {
  auto* binding = annotation(symbol, "op");
  return binding && binding->arguments.size() == 2 &&
         binding->arguments[0]->name() == operation &&
         binding->arguments[1]->name() == parameter;
}

// CXX materializes trivial by-value class arguments as an implicit copy around
// a one-element paren initializer. Peel only those same-type copies so loop
// admission still reasons about the resolved source binding identity.
cxx::IdExpressionAST* address_identity(cxx::TranslationUnit& unit,
                                       const AddressPartition& address,
                                       cxx::ExpressionAST* expression) {
  while (expression) {
    if (auto* cast =
            cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
      if (cast->conversionFunction &&
          cast->conversionFunction != address.source->copyConstructor() &&
          cast->conversionFunction != address.source->moveConstructor()) {
        return nullptr;
      }
      expression = cast->expression;
      continue;
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (auto* copy = cxx::ast_cast<cxx::ParenInitializerAST>(expression)) {
      if (!copy->expressionList || copy->expressionList->next) {
        return nullptr;
      }
      expression = copy->expressionList->value;
      continue;
    }
    break;
  }
  auto* identity = cxx::ast_cast<cxx::IdExpressionAST>(expression);
  if (!identity) {
    return nullptr;
  }
  auto traits = unit.typeTraits();
  return traits.is_same(traits.remove_cv(identity->type),
                        address.source->type())
             ? identity
             : nullptr;
}

bool is_address_unit(cxx::TranslationUnit& unit,
                     const AddressPartition& address,
                     cxx::ExpressionAST* expression) {
  expression = unwrap_expression(expression);
  cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
  if (auto* construction =
          cxx::ast_cast<cxx::BracedTypeConstructionAST>(expression)) {
    arguments = construction->bracedInitList
                    ? construction->bracedInitList->expressionList
                    : nullptr;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::TypeConstructionAST>(expression)) {
    arguments = construction->expressionList;
  }
  auto value = arguments && !arguments->next
                   ? integer_constant(unit, arguments->value)
                   : std::nullopt;
  auto traits = unit.typeTraits();
  return traits.is_same(traits.remove_cv(expression->type),
                        address.source->type()) &&
         value && *value == 1;
}

struct IntegerIncrement {
  cxx::IdExpressionAST* induction;
  unsigned step;
};

std::optional<IntegerIncrement> classify_integer_increment(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression,
    std::uintmax_t maximum_step) {
  expression = unwrap_expression(expression);
  if (auto* prefix = cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
      prefix && !prefix->symbol && prefix->op == cxx::TokenKind::T_PLUS_PLUS) {
    return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                                unwrap_expression(prefix->expression)),
                            1};
  }
  if (auto* postfix = cxx::ast_cast<cxx::PostIncrExpressionAST>(expression);
      postfix && !postfix->symbol &&
      postfix->op == cxx::TokenKind::T_PLUS_PLUS) {
    return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                                unwrap_expression(postfix->baseExpression)),
                            1};
  }
  auto* compound =
      cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(expression);
  if (!compound || compound->symbol ||
      compound->op != cxx::TokenKind::T_PLUS_EQUAL) {
    return std::nullopt;
  }
  auto amount = integer_constant(unit, compound->rightExpression);
  if (!amount || *amount <= 0 ||
      static_cast<std::uintmax_t>(*amount) > maximum_step) {
    return std::nullopt;
  }
  return IntegerIncrement{cxx::ast_cast<cxx::IdExpressionAST>(
                              unwrap_expression(compound->targetExpression)),
                          static_cast<unsigned>(*amount)};
}

}  // namespace

ControlFlow::ControlFlow(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         Types& types, cxx::StatementAST* body,
                         cxx::FunctionSymbol* function)
    : unit_(unit),
      diagnostics_(diagnostics),
      types_(types),
      function_(function) {
  accept(body);
}

std::optional<Destination> ControlFlow::destination(
    cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      return destination(unwrap_expression(found->second));
    }
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

bool ControlFlow::reference_requires_storage(cxx::Symbol* parameter) const {
  return storage_references_.contains(parameter);
}

bool ControlFlow::reference_written(cxx::Symbol* parameter) const {
  return written_references_.contains(parameter);
}

bool ControlFlow::argument_requires_storage(
    const CallArgument& argument) const {
  if (argument.inherent_storage ||
      (argument.current_receiver && receiver_requires_storage_)) {
    return true;
  }
  auto traits = unit_.typeTraits();
  for (auto* binding : argument.identity_bindings) {
    if (traits.is_lvalue_reference(binding->type())) {
      if (cxx::symbol_cast<cxx::FieldSymbol>(binding) ||
          storage_references_.contains(binding)) {
        return true;
      }
      continue;
    }
    if (addressed(binding) || traits.is_array(binding->type()) ||
        traits.is_volatile(binding->type())) {
      return true;
    }
  }
  return false;
}

bool ControlFlow::require_reference_storage(cxx::Symbol* parameter) {
  return storage_references_.insert(parameter).second;
}

bool ControlFlow::storage_backed(cxx::MemberExpressionAST* expression) const {
  auto* owner = object_owner(expression);
  return storage_expressions_.contains(expression) ||
         (owner &&
          (addressed(owner) || unit_.typeTraits().is_volatile(owner->type())));
}

cxx::Symbol* ControlFlow::object_owner(cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    return id->symbol;
  }
  auto found = object_owners_.find(expression);
  return found == object_owners_.end() ? nullptr : found->second;
}

const CountedLoop* ControlFlow::counted(cxx::ForStatementAST* loop) const {
  auto found = counted_.find(loop);
  if (found == counted_.end() || addressed(found->second.induction)) {
    return nullptr;
  }
  // An aliased bound can change through a helper without a syntactic write in
  // the loop. Consult complete address demand, including later source uses.
  const CountedLoop::Bound* bound = nullptr;
  if (auto* range =
          std::get_if<CountedLoop::UnsignedRange>(&found->second.range)) {
    bound = std::get_if<CountedLoop::Bound>(&range->upper);
  } else if (auto* range =
                 std::get_if<CountedLoop::SignedRange>(&found->second.range)) {
    bound = std::get_if<CountedLoop::Bound>(&range->upper);
  } else {
    bound = &std::get<CountedLoop::AddressRange>(found->second.range).upper;
  }
  return bound && addressed(bound->binding) ? nullptr : &found->second;
}

const SwitchLayout& ControlFlow::switch_layout(
    cxx::SwitchStatementAST* statement) const {
  return switches_.at(statement).layout;
}

unsigned ControlFlow::paths(cxx::StatementAST* statement) const {
  auto found = paths_.find(statement);
  return found == paths_.end() ? unsigned(Fallthrough) : found->second;
}

unsigned ControlFlow::sequence_paths(
    std::span<cxx::StatementAST* const> statements) const {
  unsigned outcomes = Fallthrough;
  for (auto* statement : statements) {
    if (!(outcomes & Fallthrough)) {
      break;
    }
    outcomes = (outcomes & ~Fallthrough) | paths(statement);
  }
  return outcomes;
}

ExitFlow ControlFlow::exits(cxx::StatementAST* statement,
                            unsigned selected_paths) const {
  auto outcomes = paths(statement);
  return !(outcomes & selected_paths)    ? ExitFlow::None
         : !(outcomes & ~selected_paths) ? ExitFlow::All
                                         : ExitFlow::Some;
}

ExitFlow ControlFlow::returns(cxx::StatementAST* statement) const {
  return exits(statement, Return);
}

ExitFlow ControlFlow::continues(cxx::StatementAST* statement) const {
  return exits(statement, Continue);
}

ExitFlow ControlFlow::breaks(cxx::StatementAST* statement) const {
  return exits(statement, Break);
}

ExitFlow ControlFlow::iteration_exits(cxx::StatementAST* statement) const {
  return exits(statement, Continue | Break);
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
  // Counted-loop admission queries body writes separately from header writes.
  // Keep that owner even when the body is a single unstructured statement.
  auto* loop = owners_.empty()
                   ? nullptr
                   : cxx::ast_cast<cxx::ForStatementAST>(owners_.back());
  if (structured(ast) || (loop && loop->statement == ast)) {
    owners_.push_back(ast);
  }
  return true;
}

void ControlFlow::postVisit(cxx::AST* ast) {
  if (auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast)) {
    cxx::Symbol* owner = nullptr;
    bool reference_member = false;
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
      if (member->accessOp == cxx::TokenKind::T_DOT) {
        owner = object_owner(member->baseExpression);
        auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
        reference_member =
            field && unit_.typeTraits().is_reference(field->type());
      }
    } else if (auto* nested =
                   cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      owner = object_owner(nested->expression);
    } else if (auto* cast =
                   cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        owner = object_owner(cast->expression);
      }
    }
    if (owner) {
      object_owners_.emplace(expression, owner);
      // The resolved frontend can retain an array operand directly on builtin
      // subscripts instead of inserting an explicit decay cast. Evaluating an
      // array lvalue borrows its storage in either spelling.
      if (!reference_member && expression->type &&
          unit_.typeTraits().is_array(expression->type)) {
        addressed_.insert(owner);
      }
    }
    if (classify_storage(expression)) {
      storage_expressions_.insert(expression);
    }
  }
  if (auto* statement = cxx::ast_cast<cxx::StatementAST>(ast)) {
    reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                       BindingAttributeScope::Statement);
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
  if (!owners_.empty() && owners_.back() == ast) {
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
  if (cxx::ast_cast<cxx::BreakStatementAST>(statement)) {
    return Break;
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
  if (auto* selection = cxx::ast_cast<cxx::SwitchStatementAST>(statement)) {
    const auto& facts = switches_.at(selection);
    unsigned outcomes = facts.layout.has_default ? 0 : unsigned(Fallthrough);
    unsigned suffix = Fallthrough;
    for (size_t index = facts.clause_paths.size(); index > 0; --index) {
      auto clause = facts.clause_paths[index - 1];
      auto entry =
          (clause & ~Fallthrough) | (clause & Fallthrough ? suffix : 0);
      outcomes |= entry;
      suffix = entry;
    }
    // A break selects the ordinary switch continuation. Continue and return
    // still escape to their enclosing loop and function, respectively.
    return (outcomes & ~Break) | (outcomes & Break ? unsigned(Fallthrough) : 0);
  }
  cxx::StatementAST* body = nullptr;
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(statement)) {
    body = loop->statement;
  } else if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(statement)) {
    auto body_paths = paths(loop->statement);
    return (body_paths & Return) |
           (body_paths & (Fallthrough | Continue | Break)
                ? unsigned(Fallthrough)
                : 0);
  }
  return Fallthrough | (paths(body) & Return);
}

void ControlFlow::visit(cxx::CaseStatementAST* ast) {
  if (!switch_stack_.empty() && active_switch_label_ != ast) {
    diagnostics_.reject(
        unit_, ast,
        "case labels must be direct children of a compound switch body");
  }
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::DefaultStatementAST* ast) {
  if (!switch_stack_.empty() && active_switch_label_ != ast) {
    diagnostics_.reject(
        unit_, ast,
        "default labels must be direct children of a compound switch body");
  }
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::CompoundStatementAST* ast) {
  if (switch_stack_.empty() || switch_stack_.back()->statement != ast) {
    cxx::ASTVisitor::visit(ast);
    return;
  }

  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }

  SwitchFacts facts;
  bool clause_owner_active = false;
  auto finish_clause = [&] {
    if (!clause_owner_active) {
      return;
    }
    auto& clause = facts.layout.clauses.back();
    auto* owner = clause.labels.front().source;
    auto found = writes_.find(owner);
    if (found != writes_.end()) {
      clause.writes = std::move(found->second);
      writes_.erase(found);
    }
    auto outcomes = sequence_paths(clause.statements);
    clause.falls_through = outcomes & Fallthrough;
    facts.clause_paths.push_back(outcomes);
    owners_.pop_back();
    clause_owner_active = false;
  };

  for (auto* child : cxx::ListView{ast->statementList}) {
    auto* case_label = cxx::ast_cast<cxx::CaseStatementAST>(child);
    auto* default_label = cxx::ast_cast<cxx::DefaultStatementAST>(child);
    if (case_label || default_label) {
      bool starts_clause = facts.layout.clauses.empty() ||
                           !facts.layout.clauses.back().statements.empty();
      if (starts_clause) {
        finish_clause();
        facts.layout.clauses.push_back({});
      }
      active_switch_label_ = child;
      accept(child);
      active_switch_label_ = nullptr;
      auto& clause = facts.layout.clauses.back();
      std::optional<std::int64_t> key;
      if (case_label) {
        key = case_label->caseValue;
      }
      clause.labels.push_back({child, key});
      if (default_label) {
        facts.layout.has_default = true;
      }
      if (starts_clause) {
        owners_.push_back(child);
        clause_owner_active = true;
      }
      continue;
    }
    if (facts.layout.clauses.empty()) {
      diagnostics_.reject(
          unit_, child,
          "switch body statements require a preceding case or default label");
    }
    facts.layout.clauses.back().statements.push_back(child);
    accept(child);
  }
  finish_clause();

  for (size_t index = 0; index < facts.layout.clauses.size(); ++index) {
    const auto& clause = facts.layout.clauses[index];
    if (clause.labels.size() != 1 ||
        (index + 1 < facts.layout.clauses.size() && clause.falls_through)) {
      facts.layout.direct = false;
      break;
    }
  }
  switches_.emplace(switch_stack_.back(), std::move(facts));
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

void ControlFlow::visit(cxx::SwitchStatementAST* ast) {
  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }
  accept(ast->initializer);
  accept(ast->condition);
  if (!cxx::ast_cast<cxx::CompoundStatementAST>(ast->statement)) {
    diagnostics_.reject(unit_, ast->statement,
                        "runtime switch requires a compound statement body");
  }
  switch_stack_.push_back(ast);
  accept(ast->statement);
  switch_stack_.pop_back();
}

void ControlFlow::visit(cxx::AssignmentExpressionAST* ast) {
  record(ast->leftExpression);
  record_callable_write(ast->leftExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::CompoundAssignmentExpressionAST* ast) {
  record(ast->targetExpression);
  record_callable_write(ast->targetExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::PostIncrExpressionAST* ast) {
  record(ast->baseExpression);
  record_callable_write(ast->baseExpression);
  cxx::ASTVisitor::visit(ast);
}

void ControlFlow::visit(cxx::UnaryExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP) {
    require_identity(ast->expression);
  }
  if (ast->op == cxx::TokenKind::T_PLUS_PLUS ||
      ast->op == cxx::TokenKind::T_MINUS_MINUS) {
    record(ast->expression);
    record_callable_write(ast->expression);
  }
}

void ControlFlow::visit(cxx::ReturnStatementAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* signature = function_
                        ? cxx::type_cast<cxx::FunctionType>(function_->type())
                        : nullptr;
  if (signature &&
      unit_.typeTraits().is_lvalue_reference(signature->returnType()) &&
      ast->expression) {
    require_identity(ast->expression);
  }
}

void ControlFlow::visit(cxx::CallExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* function = direct_callee(ast);
  if (!function) {
    return;
  }
  retain_call(ast, function);
  record_implicit_object_call(ast, function);
  auto parameters = function->parameters();
  auto* argument = ast->expressionList;
  auto traits = unit_.typeTraits();
  for (size_t ordinal = 0; ordinal < parameters.size() && argument;
       ++ordinal, argument = argument->next) {
    auto* parameter_type = parameters[ordinal]->type();
    if (!traits.is_lvalue_reference(parameter_type) ||
        argument->value->valueCategory != cxx::ValueCategory::kLValue) {
      continue;
    }
    if (traits.is_const(traits.remove_reference(parameter_type))) {
      retain_destination(argument->value);
    } else {
      record(argument->value);
    }
  }
}

void ControlFlow::visit(cxx::SubscriptExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(ast->symbol)) {
    retain_call(ast, function);
    return;
  }
  auto traits = unit_.typeTraits();
  auto* base_type = traits.remove_reference(ast->baseExpression->type);
  auto* index_type = traits.remove_reference(ast->indexExpression->type);
  if (traits.is_array(base_type)) {
    require_identity(ast->baseExpression);
  } else if (traits.is_array(index_type)) {
    require_identity(ast->indexExpression);
  }
}

void ControlFlow::analyze_default_constructor(const cxx::Type* type,
                                              cxx::FunctionSymbol* constructor,
                                              cxx::AST* owner) {
  type = types_.unqualified(type);
  while (auto* array = cxx::type_cast<cxx::BoundedArrayType>(type)) {
    type = types_.unqualified(array->elementType());
  }
  auto* record = types_.record(type, owner);
  if (!record || !constructor ||
      constructor != record->source->defaultConstructor() ||
      !analyzed_constructors_.insert(constructor).second) {
    return;
  }
  cxx::ASTRewriter::requireFunctionDefinition(&unit_, constructor);
  auto* definition = constructor->resolvedDefinition()->declaration();
  auto* body = definition ? definition->functionBody : nullptr;
  if (!body) {
    return;
  }

  // A synthesized constructor is an expression-local initialization plan, not
  // a lexical child of every source construct that invokes it. Its own
  // structured expressions still retain their mutation summaries.
  auto enclosing_owners = std::move(owners_);
  owners_.clear();
  for (auto* initializer : cxx::ListView{cxx::ctor_initializers(body)}) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(initializer->symbol);
    if (field && unit_.typeTraits().is_lvalue_reference(field->type())) {
      auto* source = cxx::memInitializerClause(unit_.arena(), initializer);
      cxx::Initializer reference(source);
      auto* referent = reference.singleExpression();
      if (!referent) {
        diagnostics_.reject(unit_, source,
                            "reference members require one initializer "
                            "expression");
      }
      require_identity(referent);
    }
    accept(initializer);
  }
  owners_ = std::move(enclosing_owners);
}

void ControlFlow::retain_call(cxx::CallExpressionAST* call,
                              cxx::FunctionSymbol* function) {
  DirectCall retained = {function, call, {}, std::nullopt, false, false};
  for (auto* argument : cxx::ListView{call->expressionList}) {
    retained.arguments.push_back(retain_call_argument(argument));
  }
  if (function->isImplicitObjectMemberFunction()) {
    auto* base = unwrap_expression(call->baseExpression);
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
      retained.current_receiver = is_current_object(member->baseExpression);
      retained.storage_receiver =
          member->accessOp == cxx::TokenKind::T_MINUS_GREATER;
      if (!retained.current_receiver) {
        retained.receiver = retain_call_argument(member->baseExpression);
      }
    } else {
      retained.current_receiver = true;
    }
  }
  calls_.push_back(std::move(retained));
}

void ControlFlow::retain_call(cxx::SubscriptExpressionAST* call,
                              cxx::FunctionSymbol* function) {
  DirectCall retained = {function, call, {}, std::nullopt, false, false};
  retained.arguments.push_back(retain_call_argument(call->indexExpression));
  if (function->isImplicitObjectMemberFunction()) {
    retained.current_receiver = is_current_object(call->baseExpression);
    if (!retained.current_receiver) {
      retained.receiver = retain_call_argument(call->baseExpression);
    }
  }
  calls_.push_back(std::move(retained));
}

CallArgument ControlFlow::retain_call_argument(cxx::ExpressionAST* expression) {
  CallArgument argument = {expression, {}, nullptr, false, false};
  collect_identity(expression, argument);
  auto target = classify_destination(unwrap_expression(expression));
  argument.write_binding = target ? target->binding : nullptr;
  return argument;
}

void ControlFlow::collect_identity(cxx::ExpressionAST* expression,
                                   CallArgument& argument) {
  if (!expression) {
    return;
  }
  expression = unwrap_expression(expression);
  auto traits = unit_.typeTraits();
  auto append_binding = [&](cxx::Symbol* binding) {
    if (std::ranges::find(argument.identity_bindings, binding) ==
        argument.identity_bindings.end()) {
      argument.identity_bindings.push_back(binding);
    }
  };
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      collect_identity(found->second, argument);
      return;
    }
    if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol)) {
      if (traits.is_lvalue_reference(field->type())) {
        argument.inherent_storage = true;
      } else {
        argument.current_receiver = true;
      }
      return;
    }
    append_binding(id->symbol);
    return;
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    if (field && traits.is_lvalue_reference(field->type())) {
      argument.inherent_storage = true;
      return;
    }
    if (member->accessOp == cxx::TokenKind::T_MINUS_GREATER ||
        storage_expressions_.contains(member)) {
      argument.inherent_storage = true;
      return;
    }
    if (is_current_object(member->baseExpression)) {
      argument.current_receiver = true;
      return;
    }
    collect_identity(member->baseExpression, argument);
    return;
  }
  if (auto* subscript =
          cxx::ast_cast<cxx::SubscriptExpressionAST>(expression)) {
    if (subscript->symbol) {
      argument.inherent_storage =
          expression->valueCategory == cxx::ValueCategory::kLValue;
      return;
    }
    auto* base_type = subscript->baseExpression->type;
    auto* index_type = subscript->indexExpression->type;
    if (types_.native_vector(base_type)) {
      collect_identity(subscript->baseExpression, argument);
      return;
    }
    if (types_.native_vector(index_type)) {
      collect_identity(subscript->indexExpression, argument);
      return;
    }
    argument.inherent_storage = true;
    if (traits.is_array(base_type)) {
      collect_identity(subscript->baseExpression, argument);
    } else if (traits.is_array(index_type)) {
      collect_identity(subscript->indexExpression, argument);
    }
    return;
  }
  if (auto* conditional =
          cxx::ast_cast<cxx::ConditionalExpressionAST>(expression)) {
    collect_identity(conditional->iftrueExpression, argument);
    collect_identity(conditional->iffalseExpression, argument);
    return;
  }
  if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression)) {
    if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
      argument.inherent_storage = true;
      return;
    }
  }
  if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(expression)) {
    auto* callee = direct_callee(call);
    auto* signature =
        callee ? cxx::type_cast<cxx::FunctionType>(callee->type()) : nullptr;
    if (signature && traits.is_lvalue_reference(signature->returnType())) {
      argument.inherent_storage = true;
      return;
    }
  }
  if (auto* owner = object_owner(expression)) {
    append_binding(owner);
  }
}

bool ControlFlow::require_identity(cxx::ExpressionAST* expression) {
  return require_identity(retain_call_argument(expression));
}

bool ControlFlow::require_identity(const CallArgument& argument) {
  bool changed = false;
  if (argument.current_receiver) {
    changed = !receiver_requires_storage_;
    receiver_requires_storage_ = true;
  }
  auto traits = unit_.typeTraits();
  for (auto* binding : argument.identity_bindings) {
    if (traits.is_lvalue_reference(binding->type())) {
      if (function_) {
        auto parameters = function_->parameters();
        if (std::ranges::find(parameters, binding) != parameters.end()) {
          changed = require_reference_storage(binding) || changed;
        }
      }
      continue;
    }
    if (cxx::symbol_cast<cxx::FieldSymbol>(binding)) {
      changed = !receiver_requires_storage_ || changed;
      receiver_requires_storage_ = true;
      continue;
    }
    changed = addressed_.insert(binding).second || changed;
  }
  return changed;
}

bool ControlFlow::record_callable_write(cxx::ExpressionAST* expression) {
  if (!expression) {
    return false;
  }
  expression = unwrap_expression(expression);
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      return record_callable_write(found->second);
    }
  }
  auto target = classify_destination(expression);
  return target && record_callable_write(target->binding);
}

bool ControlFlow::record_callable_write(cxx::Symbol* binding) {
  if (!binding) {
    return false;
  }
  auto traits = unit_.typeTraits();
  if (traits.is_lvalue_reference(binding->type()) && function_) {
    auto parameters = function_->parameters();
    if (std::ranges::find(parameters, binding) != parameters.end()) {
      return written_references_.insert(binding).second;
    }
  }
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(binding);
  if (!field || traits.is_lvalue_reference(field->type())) {
    return false;
  }
  return add_receiver_write(field);
}

bool ControlFlow::add_receiver_write(cxx::FieldSymbol* field) {
  if (std::ranges::find(receiver_writes_, field) != receiver_writes_.end()) {
    return false;
  }
  receiver_writes_.push_back(field);
  return true;
}

void ControlFlow::retain_reference_initializers(cxx::VariableSymbol* variable,
                                                cxx::ExpressionAST* initializer,
                                                cxx::AST* owner) {
  if (!initializer || unit_.typeTraits().is_reference(variable->type()) ||
      !unit_.typeTraits().is_class(variable->type())) {
    return;
  }
  auto* record = types_.record(types_.unqualified(variable->type()), owner);
  if (!record) {
    return;
  }
  cxx::Initializer source(initializer);
  auto arguments = source.arguments();
  if (arguments.size() == 1 && types_.unqualified(arguments[0]->type) ==
                                   types_.unqualified(variable->type())) {
    return;
  }
  const size_t count = std::min(arguments.size(), record->members.size());
  for (size_t index = 0; index < count; ++index) {
    if (unit_.typeTraits().is_lvalue_reference(
            record->members[index].field->type())) {
      require_identity(arguments[index]);
    }
  }
}

CallPropagation ControlFlow::propagate_call(const DirectCall& call,
                                            ControlFlow& callee) {
  bool caller_changed = false;
  bool callee_changed = false;
  auto parameters = callee.function_->parameters();
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    if (!unit_.typeTraits().is_lvalue_reference(parameter->type()) ||
        ordinal >= call.arguments.size()) {
      continue;
    }
    const auto& argument = call.arguments[ordinal];
    if (argument_requires_storage(argument)) {
      callee_changed =
          callee.require_reference_storage(parameter) || callee_changed;
    }
    if (callee.reference_requires_storage(parameter)) {
      caller_changed = require_identity(argument) || caller_changed;
    }
    if (callee.reference_written(parameter)) {
      caller_changed =
          record_callable_write(argument.write_binding) || caller_changed;
    }
  }
  if (!call.function->isImplicitObjectMemberFunction()) {
    return {caller_changed, callee_changed};
  }
  if (callee.receiver_requires_storage()) {
    if (call.current_receiver) {
      caller_changed = !receiver_requires_storage_ || caller_changed;
      receiver_requires_storage_ = true;
    } else if (!call.storage_receiver && call.receiver) {
      caller_changed = require_identity(*call.receiver) || caller_changed;
    }
  }
  if (callee.receiver_writes().empty()) {
    return {caller_changed, callee_changed};
  }
  if (call.current_receiver) {
    for (auto* field : callee.receiver_writes()) {
      caller_changed = add_receiver_write(field) || caller_changed;
    }
  } else if (call.receiver) {
    caller_changed =
        record_callable_write(call.receiver->write_binding) || caller_changed;
  }
  return {caller_changed, callee_changed};
}

void ControlFlow::finalize_callable_effects() {
  auto* record = function_
                     ? cxx::symbol_cast<cxx::ClassSymbol>(function_->parent())
                     : nullptr;
  if (!record || receiver_writes_.size() < 2) {
    return;
  }
  std::vector<cxx::FieldSymbol*> ordered;
  ordered.reserve(receiver_writes_.size());
  for (auto* member : record->members()) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
    if (field &&
        std::ranges::find(receiver_writes_, field) != receiver_writes_.end()) {
      ordered.push_back(field);
    }
  }
  for (auto* field : receiver_writes_) {
    if (std::ranges::find(ordered, field) == ordered.end()) {
      ordered.push_back(field);
    }
  }
  receiver_writes_ = std::move(ordered);
}

void ControlFlow::visit(cxx::InitDeclaratorAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol);
  if (variable) {
    if (unit_.typeTraits().is_lvalue_reference(variable->type()) &&
        ast->initializer) {
      cxx::Initializer source(ast->initializer);
      auto* referent = source.singleExpression();
      if (!referent) {
        diagnostics_.reject(unit_, ast->initializer,
                            "reference locals require one initializer "
                            "expression");
      }
      reference_aliases_[variable] = referent;
    }
    retain_reference_initializers(variable, ast->initializer, ast);
    analyze_default_constructor(variable->type(), variable->constructor(), ast);
  }
}

void ControlFlow::visit(cxx::ConditionExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (ast->symbol) {
    analyze_default_constructor(ast->symbol->type(), ast->symbol->constructor(),
                                ast);
  }
}

void ControlFlow::visit(cxx::BracedInitListAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (!ast->implicitElement) {
    return;
  }
  accept(ast->implicitElement->initializer());
  analyze_default_constructor(ast->implicitElement->type(),
                              ast->implicitElement->constructor(), ast);
}

void ControlFlow::visit(cxx::TypeConstructionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  analyze_default_constructor(ast->type, ast->constructorSymbol, ast);
}

void ControlFlow::visit(cxx::BracedTypeConstructionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  analyze_default_constructor(ast->type, ast->constructorSymbol, ast);
}

bool ControlFlow::structured(cxx::AST* ast) {
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast)) {
    return !binary->symbol && (binary->op == cxx::TokenKind::T_AMP_AMP ||
                               binary->op == cxx::TokenKind::T_BAR_BAR);
  }
  return cxx::ast_cast<cxx::IfStatementAST>(ast) ||
         cxx::ast_cast<cxx::SwitchStatementAST>(ast) ||
         cxx::ast_cast<cxx::ForStatementAST>(ast) ||
         cxx::ast_cast<cxx::WhileStatementAST>(ast) ||
         cxx::ast_cast<cxx::DoStatementAST>(ast) ||
         cxx::ast_cast<cxx::CompoundStatementAST>(ast) ||
         cxx::ast_cast<cxx::ConditionalExpressionAST>(ast);
}

std::optional<Destination> ControlFlow::retain_destination(
    cxx::ExpressionAST* expression) {
  auto target = classify_destination(unwrap_expression(expression));
  if (!target) {
    return std::nullopt;
  }
  if (!cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    destinations_.emplace(expression, *target);
  }
  return target;
}

void ControlFlow::record(cxx::ExpressionAST* expression) {
  auto target = retain_destination(expression);
  if (!target) {
    return;
  }
  record(target->binding);
}

void ControlFlow::record(cxx::Symbol* binding) {
  for (auto* owner : owners_) {
    auto& writes = writes_[owner];
    if (std::ranges::find(writes, binding) == writes.end()) {
      writes.push_back(binding);
    }
  }
}

void ControlFlow::record_implicit_object_call(cxx::CallExpressionAST* call,
                                              cxx::FunctionSymbol* callee) {
  if (!callee->isImplicitObjectMemberFunction()) {
    return;
  }
  auto* base = unwrap_expression(call->baseExpression);
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
    auto* receiver = member->baseExpression;
    if (!is_current_object(receiver)) {
      if (member->accessOp == cxx::TokenKind::T_DOT &&
          receiver->valueCategory == cxx::ValueCategory::kLValue) {
        if (implicit_object_may_mutate(callee)) {
          record(receiver);
        } else {
          retain_destination(receiver);
        }
      }
      return;
    }
  }
  if (!implicit_object_may_mutate(callee)) {
    return;
  }
  auto* object_type = implicit_object_type(unit_, function_);
  auto* receiver = object_type
                       ? types_.record(types_.unqualified(object_type), call)
                       : nullptr;
  if (!receiver) {
    diagnostics_.reject(unit_, call,
                        "implicit member call requires a current record "
                        "object");
  }
  for (const auto& member : receiver->members) {
    record(member.field);
  }
}

std::optional<Destination> ControlFlow::classify_destination(
    cxx::ExpressionAST* expression) {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      return classify_destination(unwrap_expression(found->second));
    }
    return Destination{id->symbol, 0, nullptr};
  }
  if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
    return classify_destination(nested->expression);
  }
  if (auto* subscript =
          cxx::ast_cast<cxx::SubscriptExpressionAST>(expression)) {
    if (subscript->symbol) {
      return std::nullopt;
    }
    if (types_.native_vector(subscript->baseExpression->type)) {
      return classify_destination(subscript->baseExpression);
    }
    if (types_.native_vector(subscript->indexExpression->type)) {
      return classify_destination(subscript->indexExpression);
    }
    return std::nullopt;
  }
  auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
  if (!member) {
    return std::nullopt;
  }
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
  if (!field || field->isStatic()) {
    return std::nullopt;
  }
  if (is_current_object(member->baseExpression)) {
    return Destination{field, 0, nullptr};
  }
  if (member->accessOp != cxx::TokenKind::T_DOT) {
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

std::optional<CountedLoop> ControlFlow::classify(cxx::ForStatementAST* loop) {
  auto* declaration =
      cxx::ast_cast<cxx::DeclarationStatementAST>(loop->initializer);
  auto* initial =
      declaration
          ? cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration)
          : nullptr;
  auto* condition = cxx::ast_cast<cxx::BinaryExpressionAST>(
      unwrap_expression(loop->condition));
  if (!initial || !initial->initDeclaratorList ||
      initial->initDeclaratorList->next || !condition ||
      condition->op != cxx::TokenKind::T_LESS) {
    return std::nullopt;
  }
  auto* induction = initial->initDeclaratorList->value->symbol;
  auto traits = unit_.typeTraits();
  if (!induction || traits.is_volatile(induction->type())) {
    return std::nullopt;
  }
  auto kind = cxx::unaligned_type(traits.remove_cv(induction->type()))->kind();
  if (kind == cxx::TypeKind::kUnsignedInt) {
    return classify_unsigned(loop, induction, condition);
  }
  if (kind == cxx::TypeKind::kInt) {
    return classify_signed(loop, induction, condition);
  }
  return traits.is_class(induction->type())
             ? classify_address(loop, induction, condition)
             : std::nullopt;
}

// Larger unsigned-int steps require room for the final source increment even
// when the bound is not a multiple of the step. Wider comparisons, mutable or
// volatile bounds, and address escapes retain general while semantics.
std::optional<CountedLoop> ControlFlow::classify_unsigned(
    cxx::ForStatementAST* loop, cxx::Symbol* induction,
    cxx::BinaryExpressionAST* condition) {
  auto traits = unit_.typeTraits();
  auto* left = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->leftExpression));
  if (condition->symbol || !left || left->symbol != induction ||
      cxx::unaligned_type(traits.remove_cv(condition->leftExpression->type))
              ->kind() != cxx::TypeKind::kUnsignedInt ||
      cxx::unaligned_type(traits.remove_cv(condition->rightExpression->type))
              ->kind() != cxx::TypeKind::kUnsignedInt) {
    return std::nullopt;
  }

  auto increment = classify_integer_increment(
      unit_, loop->expression, std::numeric_limits<std::uint32_t>::max());
  const auto& body_writes = written(loop->statement);
  if (!increment || !increment->induction ||
      increment->induction->symbol != induction ||
      std::ranges::find(body_writes, induction) != body_writes.end()) {
    return std::nullopt;
  }
  auto step_value = increment->step;

  if (auto upper = integer_constant(unit_, condition->rightExpression)) {
    if (*upper >= 0 &&
        static_cast<std::uintmax_t>(*upper) <=
            std::numeric_limits<std::uint32_t>::max() - step_value + 1) {
      return CountedLoop{
          induction, CountedLoop::UnsignedRange{static_cast<unsigned>(*upper),
                                                step_value}};
    }
    return std::nullopt;
  }
  auto* bound = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->rightExpression));
  const auto& loop_writes = written(loop);
  if (step_value != 1 || !bound || !traits.is_integral_or_enum(bound->type) ||
      traits.is_volatile(bound->type) ||
      std::ranges::find(loop_writes, bound->symbol) != loop_writes.end()) {
    return std::nullopt;
  }
  return CountedLoop{
      induction,
      CountedLoop::UnsignedRange{
          CountedLoop::Bound{bound->symbol, condition->rightExpression},
          step_value}};
}

// A strict signed-int interval cannot overflow with a unit step: an executed
// induction is at most upper-1, including when upper is INT32_MAX. Larger
// steps require a constant upper bound that leaves room for the final source
// increment. Wider signed types retain general-loop semantics because a
// target-selected index carrier is permitted to be only 32 bits.
std::optional<CountedLoop> ControlFlow::classify_signed(
    cxx::ForStatementAST* loop, cxx::Symbol* induction,
    cxx::BinaryExpressionAST* condition) {
  auto traits = unit_.typeTraits();
  auto* left = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->leftExpression));
  if (condition->symbol || !left || left->symbol != induction ||
      cxx::unaligned_type(traits.remove_cv(condition->leftExpression->type))
              ->kind() != cxx::TypeKind::kInt ||
      cxx::unaligned_type(traits.remove_cv(condition->rightExpression->type))
              ->kind() != cxx::TypeKind::kInt) {
    return std::nullopt;
  }

  if (auto* compound = cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(
          unwrap_expression(loop->expression));
      compound &&
      cxx::unaligned_type(traits.remove_cv(compound->rightExpression->type))
              ->kind() != cxx::TypeKind::kInt) {
    return std::nullopt;
  }

  auto increment = classify_integer_increment(
      unit_, loop->expression,
      static_cast<std::uintmax_t>(std::numeric_limits<std::int32_t>::max()));
  const auto& body_writes = written(loop->statement);
  if (!increment || !increment->induction ||
      increment->induction->symbol != induction ||
      std::ranges::find(body_writes, induction) != body_writes.end()) {
    return std::nullopt;
  }
  auto step_value = increment->step;

  if (auto upper = integer_constant(unit_, condition->rightExpression)) {
    auto maximum_upper = std::numeric_limits<std::int32_t>::max() -
                         static_cast<std::intmax_t>(step_value) + 1;
    if (*upper >= std::numeric_limits<std::int32_t>::min() &&
        *upper <= maximum_upper) {
      return CountedLoop{
          induction, CountedLoop::SignedRange{static_cast<std::int32_t>(*upper),
                                              step_value}};
    }
    return std::nullopt;
  }
  auto* bound = cxx::ast_cast<cxx::IdExpressionAST>(
      unwrap_expression(condition->rightExpression));
  const auto& loop_writes = written(loop);
  if (step_value != 1 || !bound || !traits.is_integral_or_enum(bound->type) ||
      traits.is_volatile(bound->type) ||
      std::ranges::find(loop_writes, bound->symbol) != loop_writes.end()) {
    return std::nullopt;
  }
  return CountedLoop{
      induction,
      CountedLoop::SignedRange{
          CountedLoop::Bound{bound->symbol, condition->rightExpression},
          step_value}};
}

// A strict native address interval with a unit step cannot wrap before its
// upper bound, independently of the target-selected carrier width. Non-unit
// steps retain while semantics because an overshoot may wrap.
std::optional<CountedLoop> ControlFlow::classify_address(
    cxx::ForStatementAST* loop, cxx::Symbol* induction,
    cxx::BinaryExpressionAST* condition) {
  const auto& partition = types_.partition(induction->type(), loop);
  if (partition.kind != ValueKind::Address) {
    return std::nullopt;
  }
  const auto& address = static_cast<const AddressPartition&>(partition);
  auto domain = address.element_type;
  if (domain != LOOM_SCALAR_TYPE_INDEX && domain != LOOM_SCALAR_TYPE_OFFSET) {
    return std::nullopt;
  }
  auto predicate = domain == LOOM_SCALAR_TYPE_INDEX ? "slt" : "ult";
  auto* left = address_identity(unit_, address, condition->leftExpression);
  auto* bound = address_identity(unit_, address, condition->rightExpression);
  if (!left || left->symbol != induction || !bound ||
      !operation_is(condition->symbol, "index.cmp", predicate)) {
    return std::nullopt;
  }
  auto* assignment = cxx::ast_cast<cxx::AssignmentExpressionAST>(
      unwrap_expression(loop->expression));
  auto* increment = assignment
                        ? cxx::ast_cast<cxx::IdExpressionAST>(
                              unwrap_expression(assignment->leftExpression))
                        : nullptr;
  auto* addition = assignment
                       ? cxx::ast_cast<cxx::BinaryExpressionAST>(
                             unwrap_expression(assignment->rightExpression))
                       : nullptr;
  if (!assignment || assignment->op != cxx::TokenKind::T_EQUAL || !increment ||
      increment->symbol != induction || !addition ||
      !operation_is(addition->symbol, "index.add")) {
    return std::nullopt;
  }
  auto* addition_left =
      address_identity(unit_, address, addition->leftExpression);
  auto* addition_right =
      address_identity(unit_, address, addition->rightExpression);
  cxx::ExpressionAST* amount = nullptr;
  if (addition_left && addition_left->symbol == induction) {
    amount = addition->rightExpression;
  } else if (addition_right && addition_right->symbol == induction) {
    amount = addition->leftExpression;
  } else {
    return std::nullopt;
  }
  if (!is_address_unit(unit_, address, amount)) {
    return std::nullopt;
  }

  const auto& body_writes = written(loop->statement);
  const auto& loop_writes = written(loop);
  if (std::ranges::find(body_writes, induction) != body_writes.end() ||
      unit_.typeTraits().is_volatile(bound->type) ||
      std::ranges::find(loop_writes, bound->symbol) != loop_writes.end()) {
    return std::nullopt;
  }
  return CountedLoop{
      induction,
      CountedLoop::AddressRange{
          CountedLoop::Bound{bound->symbol, condition->rightExpression},
          domain == LOOM_SCALAR_TYPE_INDEX
              ? CountedLoop::AddressDomain::Index
              : CountedLoop::AddressDomain::Offset}};
}

}  // namespace loom::cxx_import
