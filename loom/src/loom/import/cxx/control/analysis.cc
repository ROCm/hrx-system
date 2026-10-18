// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/function_body.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
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
  } else {
    bound = &std::get<CountedLoop::AddressRange>(found->second.range).upper;
  }
  return bound && addressed(bound->binding) ? nullptr : &found->second;
}

unsigned ControlFlow::paths(cxx::StatementAST* statement) const {
  auto found = paths_.find(statement);
  return found == paths_.end() ? unsigned(Fallthrough) : found->second;
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
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
      if (member->accessOp == cxx::TokenKind::T_DOT) {
        owner = object_owner(member->baseExpression);
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
      if (expression->type && unit_.typeTraits().is_array(expression->type)) {
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

void ControlFlow::visit(cxx::IfStatementAST* ast) {
  if (ast->constexprValue.has_value()) {
    accept(ast->initializer);
    accept(ast->condition);
    accept(*ast->constexprValue ? ast->statement : ast->elseStatement);
  } else {
    cxx::ASTVisitor::visit(ast);
  }
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
  cxx::ASTVisitor::visit(ast);
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP) {
    if (auto* owner = object_owner(ast->expression)) {
      addressed_.insert(owner);
    }
  }
  if (ast->op == cxx::TokenKind::T_PLUS_PLUS ||
      ast->op == cxx::TokenKind::T_MINUS_MINUS) {
    record(ast->expression);
  }
}

void ControlFlow::visit(cxx::CallExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* function = direct_callee(ast);
  if (!function) {
    return;
  }
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
    accept(initializer);
  }
  owners_ = std::move(enclosing_owners);
}

void ControlFlow::visit(cxx::InitDeclaratorAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol);
  if (variable) {
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
  if (cxx::unaligned_type(traits.remove_cv(induction->type()))->kind() ==
      cxx::TypeKind::kUnsignedInt) {
    return classify_unsigned(loop, induction, condition);
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

  unsigned step_value = 1;
  auto* increment_expression = unwrap_expression(loop->expression);
  auto* step = cxx::ast_cast<cxx::UnaryExpressionAST>(increment_expression);
  auto* post = cxx::ast_cast<cxx::PostIncrExpressionAST>(increment_expression);
  auto* compound =
      cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(increment_expression);
  cxx::IdExpressionAST* increment = nullptr;
  if (step && !step->symbol && step->op == cxx::TokenKind::T_PLUS_PLUS) {
    increment = cxx::ast_cast<cxx::IdExpressionAST>(
        unwrap_expression(step->expression));
  } else if (post && !post->symbol && post->op == cxx::TokenKind::T_PLUS_PLUS) {
    increment = cxx::ast_cast<cxx::IdExpressionAST>(
        unwrap_expression(post->baseExpression));
  } else if (compound && !compound->symbol &&
             compound->op == cxx::TokenKind::T_PLUS_EQUAL) {
    auto amount = integer_constant(unit_, compound->rightExpression);
    if (amount && *amount > 0 && *amount <= UINT32_MAX) {
      step_value = static_cast<unsigned>(*amount);
      increment = cxx::ast_cast<cxx::IdExpressionAST>(
          unwrap_expression(compound->targetExpression));
    }
  }
  const auto& body_writes = written(loop->statement);
  if (!increment || increment->symbol != induction ||
      std::ranges::find(body_writes, induction) != body_writes.end()) {
    return std::nullopt;
  }

  if (auto upper = integer_constant(unit_, condition->rightExpression)) {
    if (*upper >= 0 && *upper <= UINT32_MAX - step_value + 1) {
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
