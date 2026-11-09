// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/analysis.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/function_body.h>
#include <cxx/initialization.h>
#include <cxx/lambda_captures.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/objects.h"

namespace loom::cxx_import {

struct ControlFlow::TemporaryDestructionFacts {
  // Unique anonymous result-object destructions in source encounter order.
  std::vector<std::pair<Destruction, cxx::ExpressionAST*>> destructions;
  // Canonical destructors already represented by an ordinary cleanup edge.
  std::unordered_set<cxx::FunctionSymbol*> retained;
  // Canonical destructors admitted once into |destructions|.
  std::unordered_set<cxx::FunctionSymbol*> candidates;
};

struct ControlFlow::ReferenceIdentityFacts {
  // Sparse member-slice vectors referenced directly by retained call edges.
  std::vector<std::unique_ptr<std::vector<ObjectInterval>>> argument_slices;
  // Exact automatic-object intervals supplied by a reference-returning call.
  std::unordered_map<cxx::AST*, std::vector<ObjectInterval>> results;
  // Calls whose selected callable ABI produces durable storage.
  std::unordered_set<cxx::AST*> storage_results;
};

struct ControlFlow::ReferenceWriteFacts {
  struct Projection {
    // Component range within the complete returned reference.
    ReferenceProjection range;
    // Whether the write must also reach the enclosing callable signature.
    WriteEffect effect;
  };
  struct Result {
    // Structured source constructs whose carried binding must be updated.
    std::vector<cxx::AST*> owners;
    // Component ranges changed through the complete returned reference.
    std::vector<Projection> projections;
  };

  // Lexical scopes for calls whose receiver is a reference-returning call.
  std::unordered_map<cxx::AST*, std::vector<cxx::AST*>> receiver_owners;
  // Writes mapped backward through exact parameter or receiver aliases.
  std::unordered_map<cxx::AST*, Result> results;
};

namespace {

ReferenceProjection compose_projection(ReferenceProjection parent,
                                       ReferenceProjection child) {
  return {parent.component_offset + child.component_offset,
          child.component_count != SIZE_MAX ? child.component_count
          : parent.component_count == SIZE_MAX
              ? SIZE_MAX
              : parent.component_count - child.component_offset};
}

std::optional<ObjectInterval> project_interval(ObjectInterval interval,
                                               ReferenceProjection projection) {
  const size_t interval_end =
      interval.end == ObjectInterval::kWholeObjectEnd
          ? SIZE_MAX
          : interval.reference_offset + interval.end - interval.begin;
  const size_t projection_end =
      projection.component_count == SIZE_MAX
          ? SIZE_MAX
          : projection.component_offset + projection.component_count;
  const size_t begin =
      std::max(interval.reference_offset, projection.component_offset);
  const size_t end = std::min(interval_end, projection_end);
  if (begin >= end) {
    return std::nullopt;
  }
  return ObjectInterval{
      interval.binding, interval.begin + begin - interval.reference_offset,
      end == SIZE_MAX ? ObjectInterval::kWholeObjectEnd
                      : interval.begin + end - interval.reference_offset,
      begin - projection.component_offset};
}

// Reference capture fields name pointer objects; the concrete call operator's
// lvalue-to-rvalue conversions distinguish value reads from identity uses.
cxx::FieldSymbol* pointer_reference_capture(cxx::TranslationUnit& unit,
                                            cxx::Symbol* symbol) {
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol);
  if (!field) {
    return nullptr;
  }
  auto* closure = cxx::symbol_cast<cxx::ClassSymbol>(field->parent());
  if (!closure || !closure->isClosureType()) {
    return nullptr;
  }
  auto traits = unit.typeTraits();
  auto* referent = traits.remove_reference(field->type());
  return traits.is_lvalue_reference(field->type()) &&
                 traits.is_pointer(referent) && !traits.is_volatile(referent)
             ? field
             : nullptr;
}

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
          !same_function(cast->conversionFunction,
                         address.source->copyConstructor()) &&
          !same_function(cast->conversionFunction,
                         address.source->moveConstructor())) {
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

std::optional<std::intmax_t> address_constant(cxx::TranslationUnit& unit,
                                              const AddressPartition& address,
                                              cxx::ExpressionAST* expression) {
  expression = unwrap_expression(expression);
  auto traits = unit.typeTraits();
  if (!traits.is_same(traits.remove_cv(expression->type),
                      address.source->type())) {
    return std::nullopt;
  }
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
  return arguments && !arguments->next
             ? integer_constant(unit, arguments->value)
             : std::nullopt;
}

bool is_address_unit(cxx::TranslationUnit& unit,
                     const AddressPartition& address,
                     cxx::ExpressionAST* expression) {
  auto value = address_constant(unit, address, expression);
  return value && *value == 1;
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

std::optional<bool> constant_truth(cxx::TranslationUnit& unit,
                                   cxx::ExpressionAST* expression) {
  if (!expression) {
    return true;
  }
  auto constant = scalar_constant(unit, expression);
  if (!constant) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit);
  return interpreter.toBool(*constant);
}

}  // namespace

ControlFlow::ControlFlow(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         Types& types, cxx::StatementAST* body,
                         cxx::FunctionSymbol* function)
    : unit_(unit),
      diagnostics_(diagnostics),
      types_(types),
      function_(function) {
  if (function_) {
    auto* closure = cxx::symbol_cast<cxx::ClassSymbol>(function_->parent());
    if (closure && closure->isClosureType()) {
      auto* name = cxx::name_cast<cxx::OperatorId>(function_->name());
      observes_pointer_captures_ =
          name && name->op() == cxx::TokenKind::T_LPAREN;
    }
    for (auto* parameter : function_->parameters()) {
      retain_automatic_destruction(parameter, function_->declaration());
    }
  }
  if (function_ && function_->isConstructor()) {
    auto* definition = function_->declaration();
    auto* function_body = definition ? definition->functionBody : nullptr;
    for (auto* initializer :
         cxx::ListView{cxx::ctor_initializers(function_body)}) {
      accept(initializer);
    }
  }
  accept(body);
  if (function_ && function_->isDestructor()) {
    retain_subobject_destructions();
  }
  retain_temporary_destructions();
}

const Destruction* ControlFlow::automatic_destruction(
    cxx::Symbol* binding) const {
  auto found = automatic_destructions_.find(binding);
  return found == automatic_destructions_.end() ? nullptr : &found->second;
}

ControlFlow::~ControlFlow() = default;

std::optional<Destination> ControlFlow::destination(
    cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression);
      id && !id->subobjectPath) {
    if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
        field && is_captured_object_field(field)) {
      return std::nullopt;
    }
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      return found->second.destination;
    }
    return Destination{id->symbol, 0, nullptr};
  }
  if (auto* decision = cxx::ast_cast<cxx::ConditionExpressionAST>(expression)) {
    return Destination{decision->symbol, 0, nullptr};
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
  auto found = binding_storage_.find(binding);
  return found != binding_storage_.end() && found->second.addressed;
}

bool ControlFlow::storage_backed(cxx::Symbol* binding) const {
  auto found = binding_storage_.find(binding);
  return found != binding_storage_.end() &&
         (found->second.addressed || found->second.automatic_storage);
}

bool ControlFlow::reference_requires_storage(cxx::Symbol* binding) const {
  if (auto found = reference_aliases_.find(binding);
      found != reference_aliases_.end()) {
    return argument_requires_storage(found->second.identity);
  }
  return storage_references_.contains(binding);
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
  if (reference_identity_facts_ &&
      std::ranges::any_of(
          argument.reference_calls, [&](const ReferenceCall& reference) {
            return reference_identity_facts_->storage_results.contains(
                reference.call);
          })) {
    return true;
  }
  auto traits = unit_.typeTraits();
  auto requires_storage = [&](cxx::Symbol* binding) {
    if (traits.is_reference(source_object_type(binding))) {
      return cxx::symbol_cast<cxx::FieldSymbol>(binding) ||
             storage_references_.contains(binding);
    }
    return is_direct_subobject(binding) ? receiver_requires_storage_
                                        : storage_backed(binding);
  };
  if (std::ranges::any_of(argument.identity_bindings, requires_storage)) {
    return true;
  }
  if (argument.identity_intervals &&
      std::ranges::any_of(*argument.identity_intervals,
                          [&](const ObjectInterval& interval) {
                            return requires_storage(interval.binding);
                          })) {
    return true;
  }
  return false;
}

bool ControlFlow::require_reference_storage(cxx::Symbol* parameter) {
  return storage_references_.insert(parameter).second;
}

bool ControlFlow::storage_backed(cxx::MemberExpressionAST* expression) const {
  auto* owner = object_owner(expression);
  return storage_expressions_.contains(expression) ||
         (owner && storage_backed(owner));
}

cxx::Symbol* ControlFlow::object_owner(cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    return id->subobjectPath ? id->subobjectPath->value : id->symbol;
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
  return bound && bound->binding && addressed(bound->binding) ? nullptr
                                                              : &found->second;
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

bool ControlFlow::requires_callable_exit(cxx::StatementAST* statement) const {
  return callable_exits_.contains(statement);
}

size_t ControlFlow::callable_exit_count(
    cxx::CompoundStatementAST* statement) const {
  auto found = callable_exit_counts_.find(statement);
  return found == callable_exit_counts_.end() ? 0 : found->second;
}

bool ControlFlow::falls_through(cxx::StatementAST* statement) const {
  return paths(statement) & Fallthrough;
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
    construction_.enter();
  }
  return true;
}

void ControlFlow::postVisit(cxx::AST* ast) {
  if (auto* expression = cxx::ast_cast<cxx::ExpressionAST>(ast)) {
    cxx::Symbol* owner = nullptr;
    bool reference_member = false;
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
      if (observes_pointer_captures_ && expression != pointer_value_read_) {
        retain_pointer_capture_identity(member->symbol);
      }
      if (member->accessOp == cxx::TokenKind::T_DOT) {
        owner = object_owner(member->baseExpression);
        auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
        reference_member =
            field && unit_.typeTraits().is_reference(field->type());
      }
    } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
      if (observes_pointer_captures_ && expression != pointer_value_read_) {
        retain_pointer_capture_identity(id->symbol);
      }
    } else if (auto* nested =
                   cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      owner = object_owner(nested->expression);
    } else if (auto* cast =
                   cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression)) {
      if (cast->valueCategory != cxx::ValueCategory::kPrValue) {
        owner = object_owner(cast->expression);
      }
    } else if (auto* operand = explicit_reference_operand(unit_, expression)) {
      owner = object_owner(operand);
    }
    if (owner) {
      object_owners_.emplace(expression, owner);
      // The resolved frontend can retain an array operand directly on builtin
      // subscripts instead of inserting an explicit decay cast. Evaluating an
      // array lvalue borrows its storage in either spelling.
      if (!reference_member && expression->type &&
          unit_.typeTraits().is_array(expression->type)) {
        require_address(owner);
      }
    }
    if (classify_storage(expression)) {
      storage_expressions_.insert(expression);
    }
    if (cxx::ast_cast<cxx::MemberExpressionAST>(expression) &&
        construction_.tracked(owner)) {
      if (auto destination = classify_destination(expression)) {
        destinations_.insert_or_assign(expression, *destination);
      }
    }
    retain_temporary_destruction(expression);
  }
  if (auto* statement = cxx::ast_cast<cxx::StatementAST>(ast)) {
    reject_misplaced_binding_statement(unit_, diagnostics_, statement,
                                       BindingAttributeScope::Statement);
    auto outcomes = classify_paths(statement);
    // Schedule and strand bodies complete at their own region boundary.
    // Reject escaping source exits before return/loop lowering can flatten
    // the attributed compound into its enclosing control flow.
    if (outcomes & (Return | Continue | Break)) {
      visit_loom_attributes(
          unit_, statement_attributes(statement),
          [&](std::string_view name, cxx::AttributeAST*) {
            std::string_view region;
            if (name == "serial" || name == "concurrent") {
              region = "schedule regions";
            } else if (name == "strand") {
              region = "pipeline strands";
            } else {
              return;
            }
            diagnostics_.reject(
                unit_, statement,
                std::string(region) + ((outcomes & Return)
                                           ? " cannot contain function returns"
                                           : " cannot exit an enclosing loop"));
          });
    }
    if (outcomes != Fallthrough) {
      paths_.emplace(statement, outcomes);
    }
    if (classify_callable_exit(statement)) {
      callable_exits_.insert(statement);
    }
  }
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(ast)) {
    if (auto counted = classify(loop)) {
      counted_.emplace(loop, *counted);
    }
  }
  if (!owners_.empty() && owners_.back() == ast) {
    construction_.leave(ast, written(ast));
    owners_.pop_back();
  }
  switch (ast->kind()) {
    case cxx::ASTKind::ReturnStatement:
      construction_.return_from_function();
      break;
    case cxx::ASTKind::BreakStatement:
      construction_.break_from_region();
      break;
    case cxx::ASTKind::ContinueStatement:
      construction_.continue_iteration();
      break;
    default:
      break;
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
  cxx::ExpressionAST* condition = nullptr;
  if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(statement)) {
    body = loop->statement;
    condition = loop->condition;
  } else if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(statement)) {
    body = loop->statement;
    condition = loop->condition;
  } else if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(statement)) {
    auto body_paths = paths(loop->statement);
    unsigned outcomes = body_paths & (Return | Diverge);
    if (body_paths & Break) {
      outcomes |= Fallthrough;
    }
    if (body_paths & (Fallthrough | Continue)) {
      auto truth = constant_truth(unit_, loop->expression);
      outcomes |= truth && *truth ? unsigned(Diverge) : unsigned(Fallthrough);
    }
    return outcomes;
  }
  if (!body) {
    return Fallthrough;
  }
  auto body_paths = paths(body);
  unsigned outcomes = body_paths & (Return | Diverge);
  auto truth = constant_truth(unit_, condition);
  if (!truth || !*truth) {
    return outcomes | Fallthrough;
  }
  if (body_paths & Break) {
    outcomes |= Fallthrough;
  }
  if (body_paths & (Fallthrough | Continue)) {
    outcomes |= Diverge;
  }
  return outcomes;
}

bool ControlFlow::classify_callable_exit(cxx::StatementAST* statement) {
  if (cxx::ast_cast<cxx::ForStatementAST>(statement) ||
      cxx::ast_cast<cxx::WhileStatementAST>(statement) ||
      cxx::ast_cast<cxx::DoStatementAST>(statement) ||
      cxx::ast_cast<cxx::SwitchStatementAST>(statement)) {
    return returns(statement) != ExitFlow::None;
  }
  if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(statement)) {
    size_t count = std::ranges::count_if(cxx::ListView{compound->statementList},
                                         [&](cxx::StatementAST* child) {
                                           return requires_callable_exit(child);
                                         });
    if (count) {
      callable_exit_counts_.emplace(compound, count);
    }
    return count != 0;
  }
  if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(statement)) {
    if (branch->constexprValue.has_value()) {
      return requires_callable_exit(
          *branch->constexprValue ? branch->statement : branch->elseStatement);
    }
    return requires_callable_exit(branch->statement) ||
           requires_callable_exit(branch->elseStatement);
  }
  return false;
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
  auto incoming = construction_.state();
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
    construction_.leave(owner, clause.writes);
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
        construction_.restore(
            Construction::join(construction_.state(), incoming));
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
        construction_.enter(/*declaration_scope=*/false);
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
  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }
  accept(ast->initializer);
  accept(ast->condition);
  if (ast->constexprValue.has_value()) {
    accept(*ast->constexprValue ? ast->statement : ast->elseStatement);
  } else {
    auto incoming = construction_.state();
    accept(ast->statement);
    auto then_state = construction_.state();
    construction_.restore(std::move(incoming));
    accept(ast->elseStatement);
    construction_.restore(
        Construction::join(std::move(then_state), construction_.state()));
  }
}

void ControlFlow::visit(cxx::WhileStatementAST* ast) {
  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }
  accept(ast->condition);
  auto condition_exit = construction_.state();
  accept(ast->statement);
  construction_.finish_loop(std::move(condition_exit));
}

void ControlFlow::visit(cxx::DoStatementAST* ast) {
  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }
  accept(ast->statement);
  construction_.reach_latch();
  accept(ast->expression);
  construction_.finish_loop(construction_.state());
}

void ControlFlow::visit(cxx::ForStatementAST* ast) {
  for (auto* attribute : cxx::ListView{ast->attributeList}) {
    accept(attribute);
  }
  accept(ast->initializer);
  accept(ast->condition);
  auto condition_exit = construction_.state();
  // Retain the frontend's lexical write order while keeping latch-only
  // assignments out of both the first body entry and the zero-trip exit.
  accept(ast->expression);
  construction_.restore(condition_exit);
  accept(ast->statement);
  construction_.finish_loop(std::move(condition_exit));
}

void ControlFlow::visit(cxx::ConditionalExpressionAST* ast) {
  accept(ast->condition);
  auto incoming = construction_.state();
  accept(ast->iftrueExpression);
  auto then_state = construction_.state();
  construction_.restore(std::move(incoming));
  accept(ast->iffalseExpression);
  construction_.restore(
      Construction::join(std::move(then_state), construction_.state()));
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
  auto incoming = construction_.state();
  accept(ast->statement);
  if (switches_.at(ast).layout.has_default) {
    incoming = {false, {}};
  }
  construction_.finish_switch(std::move(incoming));
  switch_stack_.pop_back();
}

void ControlFlow::visit(cxx::AssignmentExpressionAST* ast) {
  const bool ordinary_call =
      ast->symbol && !types_.is_trivial_value_operation(ast->symbol);
  cxx::ASTVisitor::visit(ast);
  if (!ordinary_call) {
    if (!record(ast->leftExpression, WriteEffect::Callable)) {
      require_identity(ast->leftExpression);
    }
  }
  if (ordinary_call) {
    retain_call(*direct_operator_call(ast), ast);
  } else if (auto target = destination(ast->leftExpression)) {
    construction_.assign(
        target->binding, target->component_offset,
        target->member ? target->member->component_count : SIZE_MAX);
  }
}

void ControlFlow::visit(cxx::BinaryExpressionAST* ast) {
  if (!ast->symbol && (ast->op == cxx::TokenKind::T_AMP_AMP ||
                       ast->op == cxx::TokenKind::T_BAR_BAR)) {
    accept(ast->leftExpression);
    auto skipped = construction_.state();
    accept(ast->rightExpression);
    construction_.restore(
        Construction::join(std::move(skipped), construction_.state()));
  } else {
    cxx::ASTVisitor::visit(ast);
  }
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
  }
}

void ControlFlow::visit(cxx::CompoundAssignmentExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (!ast->symbol) {
    if (!record(ast->targetExpression, WriteEffect::Callable)) {
      require_identity(ast->targetExpression);
    }
  }
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
  }
}

void ControlFlow::visit(cxx::PostIncrExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (!ast->symbol) {
    if (!record(ast->baseExpression, WriteEffect::Callable)) {
      require_identity(ast->baseExpression);
    }
  }
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
  }
}

void ControlFlow::visit(cxx::UnaryExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
    return;
  }
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP) {
    require_identity(ast->expression);
  }
  if (ast->op == cxx::TokenKind::T_PLUS_PLUS ||
      ast->op == cxx::TokenKind::T_MINUS_MINUS) {
    if (!record(ast->expression, WriteEffect::Callable)) {
      require_identity(ast->expression);
    }
  }
}

void ControlFlow::visit(cxx::ReturnStatementAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* signature = function_
                        ? cxx::type_cast<cxx::FunctionType>(function_->type())
                        : nullptr;
  if (!signature || !unit_.typeTraits().is_reference(signature->returnType()) ||
      !ast->expression) {
    return;
  }

  auto parameters = function_->parameters();
  std::optional<size_t> parameter_ordinal;
  auto returned =
      classify_destination(unwrap_object_identity(unit_, ast->expression));
  if (returned && !returned->member && returned->component_offset == 0) {
    auto found = std::ranges::find(parameters, returned->binding);
    if (found != parameters.end() &&
        unit_.typeTraits().is_reference((*found)->type())) {
      parameter_ordinal = static_cast<size_t>(found - parameters.begin());
    }
  }

  ReferenceReturn incoming = ReferenceReturn::Storage;
  if (is_current_object(unit_, ast->expression)) {
    incoming = ReferenceReturn::CurrentReceiver;
  } else if (parameter_ordinal) {
    incoming = ReferenceReturn::Parameter;
  }
  auto require_storage = [&](ReferenceReturn origin,
                             std::optional<size_t> ordinal,
                             cxx::ExpressionAST* expression) {
    if (origin == ReferenceReturn::CurrentReceiver) {
      receiver_requires_storage_ = true;
    } else if (origin == ReferenceReturn::Parameter) {
      require_reference_storage(parameters[*ordinal]);
    } else if (origin == ReferenceReturn::Storage && expression) {
      require_identity(expression);
    }
  };

  if (reference_return_ == ReferenceReturn::None) {
    reference_return_ = incoming;
    if (parameter_ordinal) {
      reference_return_parameter_ = *parameter_ordinal;
    }
    if (incoming == ReferenceReturn::Storage) {
      require_storage(incoming, std::nullopt, ast->expression);
    }
    return;
  }
  const bool same_origin = reference_return_ == incoming &&
                           (incoming != ReferenceReturn::Parameter ||
                            reference_return_parameter_ == *parameter_ordinal);
  if (same_origin) {
    if (incoming == ReferenceReturn::Storage) {
      require_storage(incoming, std::nullopt, ast->expression);
    }
    return;
  }
  auto previous_ordinal = reference_return_ == ReferenceReturn::Parameter
                              ? std::optional(reference_return_parameter_)
                              : std::nullopt;
  require_storage(reference_return_, previous_ordinal, nullptr);
  require_storage(incoming, parameter_ordinal, ast->expression);
  reference_return_ = ReferenceReturn::Storage;
}

void ControlFlow::visit(cxx::CallExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  auto* function = direct_callee(ast);
  if (!function) {
    return;
  }
  retain_call(ast, function);
  record_implicit_object_call(ast, function);
}

void ControlFlow::visit(cxx::LambdaExpressionAST* ast) {
  // Constructing a closure evaluates its captures, not its deferred callable
  // body. The synthesized operator() definition owns analysis of that body
  // when ordinary reachable-call discovery admits it.
  for (auto* capture : cxx::ListView{ast->captureList}) {
    if (cxx::is_pack_capture(capture)) {
      diagnostics_.reject(unit_, capture,
                          "lambda pack captures must be expanded by the C++ "
                          "frontend");
    }
    auto* field = cxx::capture_field(capture);
    auto* initializer = cxx::capture_initializer(capture);
    if (!field || !initializer) {
      diagnostics_.reject(unit_, capture,
                          "lambda capture has no normalized field and "
                          "initializer");
    }
    if (!retain_array_copy(field->type(), initializer, field->constructor(),
                           capture) &&
        types_.record_constructor_requires_call(
            field->type(), field->constructor(), capture)) {
      auto* arguments = initializer;
      if (auto* conversion = constructor_conversion(initializer);
          conversion &&
          same_function(conversion->conversionFunction, field->constructor())) {
        bound_constructor_conversions_.insert(conversion);
        arguments = conversion->expression;
      }
      retain_constructor_call(field->constructor(), arguments,
                              CallArgument{.inherent_storage = true},
                              /*storage_receiver=*/true, capture);
    }
    if (unit_.typeTraits().is_lvalue_reference(field->type())) {
      cxx::Initializer reference(initializer);
      auto* referent = reference.singleExpression();
      if (!referent) {
        diagnostics_.reject(unit_, capture,
                            "lambda reference capture requires one "
                            "initializer expression");
      }
      auto argument = retain_call_argument(referent);
      const bool direct_automatic =
          argument.identity_bindings.size() == 1 &&
          !argument.identity_intervals && !argument.current_receiver &&
          !argument.inherent_storage && argument.reference_calls.empty() &&
          argument.write_binding == argument.identity_bindings.front() &&
          !cxx::symbol_cast<cxx::FieldSymbol>(argument.write_binding);
      if (cxx::ast_cast<cxx::RefLambdaCaptureAST>(capture) &&
          pointer_reference_capture(unit_, field) && direct_automatic) {
        pointer_reference_facts().captures.push_back(
            {.field = field, .referent = std::move(argument)});
      } else {
        require_identity(argument);
      }
    } else if (cxx::ast_cast<cxx::ThisLambdaCaptureAST>(capture)) {
      receiver_requires_storage_ = true;
    }
    accept(capture);
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
  if (!record || !constructor || cxx::isUserProvided(constructor) ||
      !same_function(constructor, record->source->defaultConstructor())) {
    return;
  }
  auto* canonical = constructor->canonical();
  if (!analyzed_constructors_.insert(canonical).second) {
    return;
  }
  cxx::ASTRewriter::requireFunctionDefinition(&unit_, constructor);
  auto* definition = constructor->resolvedDefinition()->declaration();
  auto* body = definition ? definition->functionBody : nullptr;
  if (!body) {
    diagnostics_.reject(unit_, owner,
                        "default record constructor has no source "
                        "initialization plan");
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
    auto* empty = cxx::ast_cast<cxx::ParenMemInitializerAST>(initializer);
    if (empty && !empty->lparenLoc && !empty->expressionList) {
      // The frontend includes implicit default construction of class members
      // in this list. An empty semantic handle begins its source lifetime but
      // supplies no value; a nested record may still have real initializers.
      auto* member_type = source_object_type(initializer->symbol);
      analyze_default_constructor(member_type, initializer->constructor,
                                  initializer);
      if (default_constructor_initializes(initializer->constructor) ||
          types_.record_constructor_requires_call(
              member_type, initializer->constructor, initializer)) {
        initializing_constructors_.insert(canonical);
      }
    } else {
      initializing_constructors_.insert(canonical);
    }
  }
  owners_ = std::move(enclosing_owners);
}

bool ControlFlow::default_constructor_initializes(
    cxx::FunctionSymbol* constructor) const {
  return constructor &&
         initializing_constructors_.contains(constructor->canonical());
}

const ArrayCopyInitialization* ControlFlow::array_copy(
    cxx::AST* initializer) const {
  auto found = array_copies_.find(initializer);
  return found == array_copies_.end() ? nullptr : &found->second;
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
      retained.current_receiver =
          !member->subobjectPath &&
          is_current_object(unit_, member->baseExpression);
      retained.storage_receiver =
          member->accessOp == cxx::TokenKind::T_MINUS_GREATER;
      if (!retained.current_receiver) {
        retained.receiver = retain_member_receiver(
            member->baseExpression, member->subobjectPath, member);
      }
    } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base);
               id && id->subobjectPath) {
      retained.receiver =
          retain_member_receiver(nullptr, id->subobjectPath, id);
    } else {
      retained.current_receiver = true;
    }
  }
  retain_reference_receiver_write(retained);
  retain_reference_arguments(retained);
  calls_.push_back(std::move(retained));
}

void ControlFlow::retain_call(cxx::SubscriptExpressionAST* call,
                              cxx::FunctionSymbol* function) {
  DirectCall retained = {function, call, {}, std::nullopt, false, false};
  retained.arguments.push_back(retain_call_argument(call->indexExpression));
  if (function->isImplicitObjectMemberFunction()) {
    retained.current_receiver = is_current_object(unit_, call->baseExpression);
    if (!retained.current_receiver) {
      retained.receiver = retain_call_argument(call->baseExpression);
    }
  }
  retain_reference_receiver_write(retained);
  retain_reference_arguments(retained);
  calls_.push_back(std::move(retained));
}

void ControlFlow::retain_call(const DirectOperatorCall& call,
                              cxx::AST* source) {
  DirectCall retained = {call.function, source, {}, std::nullopt, false, false};
  retained.arguments.reserve(call.argument_count + call.has_postfix_dummy);
  for (size_t index = 0; index < call.argument_count; ++index) {
    retained.arguments.push_back(retain_call_argument(call.arguments[index]));
  }
  if (call.has_postfix_dummy) {
    retained.arguments.push_back({});
  }
  if (call.receiver) {
    retained.current_receiver = is_current_object(unit_, call.receiver);
    if (!retained.current_receiver) {
      retained.receiver = retain_call_argument(call.receiver);
      retained.storage_receiver = retained.receiver->inherent_storage;
    }
  }
  retain_reference_receiver_write(retained);
  retain_reference_arguments(retained);
  calls_.push_back(std::move(retained));

  if (!call.receiver || !call.function->isImplicitObjectMemberFunction()) {
    return;
  }
  if (!is_current_object(unit_, call.receiver)) {
    if (call.receiver->valueCategory != cxx::ValueCategory::kPrValue) {
      if (types_.implicit_object_may_mutate(call.function, source)) {
        record(call.receiver);
      } else {
        retain_destination(call.receiver);
      }
    }
    return;
  }
  if (!types_.implicit_object_may_mutate(call.function, source)) {
    return;
  }
  auto* object_type = implicit_object_type(unit_, function_);
  auto* receiver = object_type
                       ? types_.record(types_.unqualified(object_type), source)
                       : nullptr;
  if (!receiver) {
    diagnostics_.reject(unit_, source,
                        "implicit operator call requires a current record "
                        "object");
  }
  for (const auto& member : receiver->members) {
    record(member.source);
  }
}

void ControlFlow::retain_reference_arguments(const DirectCall& call) {
  auto parameters = call.function->parameters();
  auto traits = unit_.typeTraits();
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* argument = call.arguments[ordinal].source;
    auto* parameter_type = parameters[ordinal]->type();
    if (!traits.is_reference(parameter_type) || !argument ||
        argument->valueCategory == cxx::ValueCategory::kPrValue) {
      continue;
    }
    if (traits.is_const(traits.remove_reference(parameter_type))) {
      retain_destination(argument);
    } else {
      record(argument);
    }
  }
}

CallArgument ControlFlow::retain_call_argument(cxx::ExpressionAST* expression) {
  CallArgument argument = {.source = expression};
  collect_identity(expression, argument);
  auto target = classify_destination(unwrap_object_identity(unit_, expression));
  argument.write_binding = target ? target->binding : nullptr;
  return argument;
}

CallArgument ControlFlow::retain_member_receiver(cxx::ExpressionAST* expression,
                                                 cxx::List<cxx::Symbol*>* path,
                                                 cxx::AST* owner) {
  if (!path) {
    return retain_call_argument(expression);
  }
  const auto& selected = types_.projection(path, owner);
  if (!expression || is_current_object(unit_, expression)) {
    CallArgument argument{
        .source = expression,
        .write_binding = selected.root->source,
    };
    append_identity(
        {selected.root->source,
         selected.component_offset - selected.root->component_offset,
         selected.partition},
        argument);
    return argument;
  }
  auto origin = retain_call_argument(expression);
  CallArgument argument = {
      .source = expression,
      .write_binding = origin.write_binding,
  };
  append_identity(
      origin, argument,
      {selected.component_offset, selected.partition->component_count});
  return argument;
}

void ControlFlow::append_identity(Destination destination,
                                  CallArgument& argument) {
  append_identity(
      ObjectInterval{destination.binding, destination.component_offset,
                     destination.member
                         ? destination.component_offset +
                               destination.member->component_count
                         : ObjectInterval::kWholeObjectEnd},
      argument);
}

void ControlFlow::append_identity(ObjectInterval interval,
                                  CallArgument& argument) {
  if (interval.begin == 0 && interval.end == ObjectInterval::kWholeObjectEnd &&
      interval.reference_offset == 0) {
    if (std::ranges::find(argument.identity_bindings, interval.binding) ==
        argument.identity_bindings.end()) {
      argument.identity_bindings.push_back(interval.binding);
    }
    return;
  }
  if (!argument.identity_intervals) {
    if (!reference_identity_facts_) {
      reference_identity_facts_ = std::make_unique<ReferenceIdentityFacts>();
    }
    auto intervals = std::make_unique<std::vector<ObjectInterval>>();
    argument.identity_intervals = intervals.get();
    reference_identity_facts_->argument_slices.push_back(std::move(intervals));
  }
  append_interval(*argument.identity_intervals, interval);
}

void ControlFlow::append_receiver_identity(CallArgument& argument,
                                           ReferenceProjection projection) {
  argument.current_receiver |= projection == ReferenceProjection{};
  const auto* record =
      types_.record(types_.unqualified(implicit_object_type(unit_, function_)),
                    function_->declaration());
  for (const auto& member : record->members) {
    auto interval =
        project_interval({member.source, 0, member.partition->component_count,
                          member.component_offset},
                         projection);
    if (interval) {
      append_identity(*interval, argument);
    }
  }
}

void ControlFlow::append_identity(const CallArgument& source,
                                  CallArgument& argument,
                                  ReferenceProjection projection) {
  for (auto* binding : source.identity_bindings) {
    if (auto interval = project_interval(
            {binding, 0, ObjectInterval::kWholeObjectEnd}, projection)) {
      append_identity(*interval, argument);
    }
  }
  if (source.identity_intervals) {
    for (const auto& origin : *source.identity_intervals) {
      if (auto interval = project_interval(origin, projection)) {
        append_identity(*interval, argument);
      }
    }
  }
  argument.current_receiver |=
      source.current_receiver && projection == ReferenceProjection{};
  argument.inherent_storage |= source.inherent_storage;
  for (const auto& source_call : source.reference_calls) {
    ReferenceCall reference{
        source_call.call,
        compose_projection(source_call.projection, projection)};
    if (std::ranges::find(argument.reference_calls, reference) ==
        argument.reference_calls.end()) {
      argument.reference_calls.push_back(reference);
    }
  }
}

void ControlFlow::collect_identity(cxx::ExpressionAST* expression,
                                   CallArgument& argument,
                                   ReferenceProjection projection) {
  if (!expression) {
    return;
  }
  if (is_current_object(unit_, expression)) {
    append_receiver_identity(argument, projection);
    return;
  }
  expression = unwrap_object_identity(unit_, expression);
  auto append_destination = [&](Destination destination) {
    auto interval = project_interval(
        {destination.binding, destination.component_offset,
         destination.member ? destination.component_offset +
                                  destination.member->component_count
                            : ObjectInterval::kWholeObjectEnd},
        projection);
    if (interval) {
      append_identity(*interval, argument);
    }
  };
  auto traits = unit_.typeTraits();
  if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
      cast && is_class_adjustment(cast)) {
    if (cast->castKind == cxx::ImplicitCastKind::kDerivedToBaseConversion) {
      if (auto selected = classify_destination(cast)) {
        append_destination(*selected);
        return;
      }
      const auto& selected = types_.projection(cast->subobjectPath, cast);
      collect_identity(cast->expression, argument,
                       compose_projection({selected.component_offset,
                                           selected.partition->component_count},
                                          projection));
      return;
    }
    collect_identity(cast->expression, argument, projection);
    argument.inherent_storage |=
        cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion;
    return;
  }
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      append_identity(found->second.identity, argument, projection);
      return;
    }
    if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol)) {
      auto* root = id->subobjectPath ? id->subobjectPath->value : field;
      if (is_captured_object_field(root)) {
        argument.inherent_storage = true;
        return;
      }
      if (traits.is_lvalue_reference(field->type())) {
        argument.inherent_storage = true;
      } else {
        append_destination(*classify_destination(id));
      }
      return;
    }
    append_destination({id->symbol, 0, nullptr});
    return;
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    auto* root = member->subobjectPath ? member->subobjectPath->value : field;
    if (field && is_current_object(unit_, member->baseExpression) &&
        is_captured_object_field(root)) {
      argument.inherent_storage = true;
      return;
    }
    if (field && traits.is_lvalue_reference(field->type())) {
      argument.inherent_storage = true;
      return;
    }
    if (member->accessOp == cxx::TokenKind::T_MINUS_GREATER ||
        storage_expressions_.contains(member)) {
      argument.inherent_storage = true;
      return;
    }
    if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
      collect_identity(member->baseExpression, argument, projection);
      return;
    }
    if (is_current_object(unit_, member->baseExpression)) {
      if (field) {
        append_destination(*classify_destination(member));
      }
      return;
    }
    if (auto destination = classify_destination(expression)) {
      append_destination(*destination);
    } else {
      const auto selected =
          types_.selection(member->symbol, member->subobjectPath, member);
      collect_identity(member->baseExpression, argument,
                       compose_projection({selected.component_offset,
                                           selected.partition->component_count},
                                          projection));
    }
    return;
  }
  if (auto* subscript =
          cxx::ast_cast<cxx::SubscriptExpressionAST>(expression)) {
    if (subscript->symbol) {
      argument.reference_calls.push_back({expression, projection});
      return;
    }
    auto* base_type = subscript->baseExpression->type;
    auto* index_type = subscript->indexExpression->type;
    if (types_.native_vector(base_type)) {
      if (auto destination = retain_destination(expression)) {
        append_destination(*destination);
      } else {
        collect_identity(subscript->baseExpression, argument, projection);
      }
      return;
    }
    if (types_.native_vector(index_type)) {
      if (auto destination = retain_destination(expression)) {
        append_destination(*destination);
      } else {
        collect_identity(subscript->indexExpression, argument, projection);
      }
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
    collect_identity(conditional->iftrueExpression, argument, projection);
    collect_identity(conditional->iffalseExpression, argument, projection);
    return;
  }
  if (auto call = direct_operator_call(expression)) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(call->function->type());
    if (signature && traits.is_reference(signature->returnType())) {
      argument.reference_calls.push_back({expression, projection});
      return;
    }
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
    if (signature && traits.is_reference(signature->returnType())) {
      argument.reference_calls.push_back({expression, projection});
      return;
    }
  }
  if (auto* owner = object_owner(expression)) {
    append_destination({owner, 0, nullptr});
  }
}

bool ControlFlow::require_identity(cxx::ExpressionAST* expression) {
  return require_identity(retain_call_argument(expression));
}

bool ControlFlow::require_address(cxx::Symbol* binding) {
  auto& facts = binding_storage_[binding];
  const bool changed = !facts.addressed;
  facts.addressed = true;
  return changed;
}

void ControlFlow::retain_automatic_storage(cxx::VariableSymbol* variable,
                                           cxx::ExpressionAST* initializer,
                                           bool record_constructor_call) {
  auto traits = unit_.typeTraits();
  if (!traits.is_reference(variable->type()) &&
      ((!initializer && !types_.requires_value_transport(variable->type())) ||
       record_constructor_call || traits.is_array(variable->type()) ||
       traits.is_volatile(variable->type()))) {
    binding_storage_[variable].automatic_storage = true;
  }
}

bool ControlFlow::require_identity(const CallArgument& argument) {
  bool changed = false;
  for (const auto& reference : argument.reference_calls) {
    changed = identity_call_results_.insert(reference.call).second || changed;
  }
  if (argument.current_receiver) {
    changed = !receiver_requires_storage_ || changed;
    receiver_requires_storage_ = true;
  }
  auto traits = unit_.typeTraits();
  auto require_binding = [&](cxx::Symbol* binding) {
    if (traits.is_reference(source_object_type(binding))) {
      if (function_) {
        auto parameters = function_->parameters();
        if (std::ranges::find(parameters, binding) != parameters.end()) {
          changed = require_reference_storage(binding) || changed;
        }
      }
      return;
    }
    if (is_direct_subobject(binding)) {
      changed = !receiver_requires_storage_ || changed;
      receiver_requires_storage_ = true;
      return;
    }
    changed = require_address(binding) || changed;
  };
  for (auto* binding : argument.identity_bindings) {
    require_binding(binding);
  }
  if (argument.identity_intervals) {
    for (const auto& interval : *argument.identity_intervals) {
      require_binding(interval.binding);
    }
  }
  return changed;
}

bool ControlFlow::record_callable_write(cxx::Symbol* binding) {
  if (!binding) {
    return false;
  }
  auto traits = unit_.typeTraits();
  if (traits.is_reference(source_object_type(binding)) && function_) {
    auto parameters = function_->parameters();
    if (std::ranges::find(parameters, binding) != parameters.end()) {
      return written_references_.insert(binding).second;
    }
  }
  if (!is_direct_subobject(binding) ||
      traits.is_lvalue_reference(source_object_type(binding))) {
    return false;
  }
  return add_receiver_write(binding);
}

bool ControlFlow::record_scoped_write(cxx::Symbol* binding,
                                      std::span<cxx::AST* const> owners) {
  if (!binding) {
    return false;
  }
  bool changed =
      pointer_reference_facts_ &&
      pointer_reference_facts_->written_bindings.insert(binding).second;
  for (auto* owner : owners) {
    auto& writes = writes_[owner];
    if (std::ranges::find(writes, binding) == writes.end()) {
      writes.push_back(binding);
      changed = true;
    }
  }
  return changed;
}

ControlFlow::PointerReferenceFacts& ControlFlow::pointer_reference_facts() {
  if (!pointer_reference_facts_) {
    pointer_reference_facts_ = std::make_unique<PointerReferenceFacts>();
    for (const auto& entry : writes_) {
      pointer_reference_facts_->written_bindings.insert(entry.second.begin(),
                                                        entry.second.end());
    }
  }
  return *pointer_reference_facts_;
}

void ControlFlow::retain_pointer_capture_identity(cxx::Symbol* symbol) {
  auto* field = pointer_reference_capture(unit_, symbol);
  if (field && function_->parent() == field->parent()) {
    pointer_reference_facts().identity_captures.insert(field);
  }
}

void ControlFlow::collect_pointer_capture_identities(
    std::unordered_set<cxx::FieldSymbol*>& identities) const {
  if (pointer_reference_facts_) {
    const auto& retained = pointer_reference_facts_->identity_captures;
    identities.insert(retained.begin(), retained.end());
  }
}

bool ControlFlow::resolve_pointer_reference_captures(
    const std::unordered_set<cxx::FieldSymbol*>& identities) {
  if (!pointer_reference_facts_) {
    return false;
  }
  bool changed = false;
  auto binding_is_written = [&](cxx::Symbol* binding) {
    return pointer_reference_facts_->written_bindings.contains(binding);
  };
  for (auto& capture : pointer_reference_facts_->captures) {
    if (capture.requires_storage) {
      continue;
    }
    const auto& referent = capture.referent;
    const bool written =
        std::ranges::any_of(referent.identity_bindings, binding_is_written) ||
        (referent.identity_intervals &&
         std::ranges::any_of(*referent.identity_intervals,
                             [&](const ObjectInterval& interval) {
                               return binding_is_written(interval.binding);
                             }));
    if (!identities.contains(capture.field) && !written &&
        !argument_requires_storage(referent)) {
      continue;
    }
    capture.requires_storage = true;
    changed = require_identity(referent) || changed;
  }
  return changed;
}

void ControlFlow::publish_pointer_reference_snapshots(Types& types) const {
  if (!pointer_reference_facts_) {
    return;
  }
  for (const auto& capture : pointer_reference_facts_->captures) {
    if (!capture.requires_storage) {
      types.select_pointer_reference_snapshot(capture.field,
                                              capture.referent.source);
    }
  }
}

bool ControlFlow::record_argument_write(
    const CallArgument& argument, std::span<cxx::AST* const> owners,
    std::span<cxx::Symbol* const> receiver_fields) {
  if (receiver_fields.empty()) {
    return record_projected_write(argument, owners, {}, WriteEffect::Callable);
  }
  bool changed = false;
  for (auto* field : receiver_fields) {
    const auto& member = types_.member(field, argument.source);
    changed = record_projected_write(
                  argument, owners,
                  {member.component_offset, member.partition->component_count},
                  WriteEffect::Callable) ||
              changed;
  }
  return changed;
}

bool ControlFlow::record_projected_write(const CallArgument& argument,
                                         std::span<cxx::AST* const> owners,
                                         ReferenceProjection projection,
                                         WriteEffect effect) {
  // Storage origins keep addresses live; writing through them does not replace
  // the pointer or array binding that supplied the address.
  if (argument.inherent_storage) {
    return false;
  }
  bool changed = false;
  visit_intervals(argument, projection, [&](ObjectInterval interval) {
    changed = record_scoped_write(interval.binding, owners) || changed;
    if (effect == WriteEffect::Callable) {
      changed = record_callable_write(interval.binding) || changed;
    }
  });
  for (const auto& reference : argument.reference_calls) {
    changed = record_reference_result_write(
                  {reference.call,
                   compose_projection(reference.projection, projection)},
                  owners, effect) ||
              changed;
  }
  return changed;
}

bool ControlFlow::record_reference_result_write(
    ReferenceCall reference, std::span<cxx::AST* const> owners,
    WriteEffect effect) {
  if (!reference_write_facts_) {
    reference_write_facts_ = std::make_unique<ReferenceWriteFacts>();
  }
  auto& write = reference_write_facts_->results[reference.call];
  bool changed = false;
  for (auto* owner : owners) {
    if (std::ranges::find(write.owners, owner) == write.owners.end()) {
      write.owners.push_back(owner);
      changed = true;
    }
  }
  auto found = std::ranges::find(write.projections, reference.projection,
                                 &ReferenceWriteFacts::Projection::range);
  if (found == write.projections.end()) {
    write.projections.push_back({reference.projection, effect});
    changed = true;
  } else if (effect == WriteEffect::Callable && found->effect != effect) {
    found->effect = effect;
    changed = true;
  }
  return changed;
}

void ControlFlow::retain_reference_receiver_write(const DirectCall& call) {
  if (!call.receiver || call.receiver->reference_calls.empty() ||
      !types_.implicit_object_may_mutate(call.function, call.source)) {
    return;
  }
  if (!reference_write_facts_) {
    reference_write_facts_ = std::make_unique<ReferenceWriteFacts>();
  }
  reference_write_facts_->receiver_owners.emplace(call.source, owners_);
}

bool ControlFlow::add_receiver_write(cxx::Symbol* field) {
  if (std::ranges::find(receiver_writes_, field) != receiver_writes_.end()) {
    return false;
  }
  receiver_writes_.push_back(field);
  return true;
}

void ControlFlow::retain_constructor_call(cxx::FunctionSymbol* constructor,
                                          cxx::ExpressionAST* initializer,
                                          CallArgument receiver,
                                          bool storage_receiver,
                                          cxx::AST* owner) {
  DirectCall retained = {
      .function = constructor,
      .source = owner,
      .arguments = {},
      .receiver = std::nullopt,
      .current_receiver = receiver.current_receiver,
      .storage_receiver = storage_receiver,
  };
  for (auto* argument : constructor_arguments(initializer)) {
    retained.arguments.push_back(retain_call_argument(argument));
  }
  retain_reference_arguments(retained);
  if (!retained.current_receiver) {
    retained.receiver = std::move(receiver);
  }
  calls_.push_back(std::move(retained));
}

bool ControlFlow::retain_constructor(cxx::VariableSymbol* variable,
                                     cxx::ExpressionAST* initializer,
                                     cxx::AST* owner) {
  if (!types_.record_constructor_requires_call(
          variable->type(), variable->constructor(), owner)) {
    return false;
  }
  if (auto* conversion = constructor_conversion(initializer);
      conversion &&
      same_function(conversion->conversionFunction, variable->constructor())) {
    initializer = conversion->expression;
  }
  CallArgument receiver = {
      .source = nullptr,
      .write_binding = variable,
  };
  append_identity({variable, 0, nullptr}, receiver);
  retain_constructor_call(variable->constructor(), initializer,
                          std::move(receiver),
                          /*storage_receiver=*/false, owner);
  return true;
}

void ControlFlow::retain_destructor_call(const Destruction& destruction,
                                         std::optional<CallArgument> receiver,
                                         bool current_receiver,
                                         cxx::AST* owner) {
  if (temporary_destruction_facts_) {
    temporary_destruction_facts_->retained.insert(
        destruction.function->canonical());
  }
  calls_.push_back({
      .function = destruction.function,
      .source = owner,
      .arguments = {},
      .receiver = std::move(receiver),
      .current_receiver = current_receiver,
      .storage_receiver = false,
  });
}

void ControlFlow::retain_temporary_destruction(cxx::ExpressionAST* expression) {
  if (!expression ||
      expression->valueCategory != cxx::ValueCategory::kPrValue ||
      !unit_.typeTraits().is_class(expression->type)) {
    return;
  }
  auto selected = destruction_for(unit_, expression->type);
  if (!selected) {
    return;
  }
  if (!temporary_destruction_facts_) {
    temporary_destruction_facts_ =
        std::make_unique<TemporaryDestructionFacts>();
    for (const auto& call : calls_) {
      if (call.function && call.function->isDestructor()) {
        temporary_destruction_facts_->retained.insert(
            call.function->canonical());
      }
    }
  }
  auto* canonical = selected->function->canonical();
  if (temporary_destruction_facts_->candidates.insert(canonical).second) {
    temporary_destruction_facts_->destructions.emplace_back(*selected,
                                                            expression);
  }
}

void ControlFlow::retain_temporary_destructions() {
  if (!temporary_destruction_facts_) {
    return;
  }
  for (const auto& [destruction, expression] :
       temporary_destruction_facts_->destructions) {
    if (!temporary_destruction_facts_->retained.contains(
            destruction.function->canonical())) {
      retain_destructor_call(destruction, std::nullopt,
                             /*current_receiver=*/false, expression);
    }
  }
}

void ControlFlow::retain_automatic_destruction(cxx::Symbol* binding,
                                               cxx::AST* owner) {
  if (!binding) {
    return;
  }
  if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(binding);
      variable && (variable->isStatic() || variable->isExtern() ||
                   variable->isThreadLocal())) {
    return;
  }
  auto selected = destruction_for(unit_, binding->type());
  if (!selected) {
    return;
  }
  automatic_destructions_.emplace(binding, *selected);
  CallArgument receiver = {
      .source = nullptr,
      .write_binding = binding,
  };
  append_identity({binding, 0, nullptr}, receiver);
  retain_destructor_call(*selected, std::move(receiver),
                         /*current_receiver=*/false, owner);
}

void ControlFlow::retain_subobject_destructions() {
  auto* record = cxx::symbol_cast<cxx::ClassSymbol>(function_->parent());
  auto* owner = function_->declaration();
  auto* partition =
      record ? types_.record(types_.unqualified(record->type()), owner)
             : nullptr;
  if (!partition) {
    diagnostics_.reject(unit_, owner,
                        "destructor requires an admitted record object");
  }
  for (const auto& member : partition->members) {
    auto selected = destruction_for(unit_, member.type());
    if (!selected) {
      continue;
    }
    subobject_destructions_.push_back({member.source, *selected});
    retain_destructor_call(*selected, std::nullopt,
                           /*current_receiver=*/true, owner);
  }
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
    if (unit_.typeTraits().is_lvalue_reference(record->members[index].type())) {
      require_identity(arguments[index]);
    }
  }
}

bool ControlFlow::append_interval(std::vector<ObjectInterval>& intervals,
                                  ObjectInterval interval) {
  if ((interval.begin == interval.end &&
       interval.end != ObjectInterval::kWholeObjectEnd) ||
      std::ranges::find_if(intervals, [&](const ObjectInterval& retained) {
        return retained.binding == interval.binding &&
               retained.begin == interval.begin &&
               retained.end == interval.end &&
               retained.reference_offset == interval.reference_offset;
      }) != intervals.end()) {
    return false;
  }
  intervals.push_back(interval);
  return true;
}

template <typename Visitor>
void ControlFlow::visit_intervals(const CallArgument& argument,
                                  ReferenceProjection projection,
                                  Visitor&& visitor) const {
  auto visit = [&](ObjectInterval origin, ReferenceProjection selected) {
    if (auto interval = project_interval(origin, selected)) {
      visitor(*interval);
    }
  };
  for (auto* binding : argument.identity_bindings) {
    visit({binding, 0, ObjectInterval::kWholeObjectEnd}, projection);
  }
  if (argument.identity_intervals) {
    for (const auto& interval : *argument.identity_intervals) {
      visit(interval, projection);
    }
  }
  if (!reference_identity_facts_) {
    return;
  }
  for (const auto& reference : argument.reference_calls) {
    auto found = reference_identity_facts_->results.find(reference.call);
    if (found == reference_identity_facts_->results.end()) {
      continue;
    }
    const auto selected = compose_projection(reference.projection, projection);
    for (const auto& interval : found->second) {
      visit(interval, selected);
    }
  }
}

void ControlFlow::append_intervals(
    const CallArgument& argument,
    std::vector<ObjectInterval>& intervals) const {
  visit_intervals(argument, {}, [&](ObjectInterval interval) {
    append_interval(intervals, interval);
  });
}

bool ControlFlow::identities_overlap(const CallArgument& left,
                                     const CallArgument& right) const {
  bool overlaps = false;
  visit_intervals(left, {}, [&](ObjectInterval left_interval) {
    visit_intervals(right, {}, [&](ObjectInterval right_interval) {
      overlaps =
          overlaps || (left_interval.binding == right_interval.binding &&
                       (left_interval.end == ObjectInterval::kWholeObjectEnd ||
                        right_interval.end == ObjectInterval::kWholeObjectEnd ||
                        (left_interval.begin < right_interval.end &&
                         right_interval.begin < left_interval.end)));
    });
  });
  if (overlaps) {
    return true;
  }

  auto has_unresolved_identity = [&](const CallArgument& argument) {
    return std::ranges::any_of(
        argument.reference_calls, [&](const ReferenceCall& reference) {
          return !reference_identity_facts_ ||
                 !reference_identity_facts_->results.contains(reference.call);
        });
  };
  const bool left_has_identity =
      left.inherent_storage || left.current_receiver ||
      !left.identity_bindings.empty() || left.identity_intervals ||
      !left.reference_calls.empty();
  const bool right_has_identity =
      right.inherent_storage || right.current_receiver ||
      !right.identity_bindings.empty() || right.identity_intervals ||
      !right.reference_calls.empty();
  return (has_unresolved_identity(left) && right_has_identity) ||
         (has_unresolved_identity(right) && left_has_identity);
}

bool ControlFlow::references_current_receiver_field(
    const CallArgument& argument) const {
  bool references_field = false;
  visit_intervals(argument, {}, [&](ObjectInterval interval) {
    references_field |= is_direct_subobject(interval.binding);
  });
  return references_field;
}

bool ControlFlow::retain_reference_result_identity(const DirectCall& call,
                                                   const ControlFlow& callee) {
  // Retain an exact automatic origin whenever every return aliases one
  // parameter or its receiver. Outer call edges can compare nested aliases
  // without traversing either function body again. The selected storage ABI is
  // retained separately so another reference call can transport it directly.
  bool result_storage = callee.reference_return_ == ReferenceReturn::Storage;
  std::vector<ObjectInterval> result_intervals;
  if (auto ordinal = callee.reference_return_parameter();
      ordinal && *ordinal < call.arguments.size()) {
    auto parameters = callee.function_->parameters();
    result_storage = callee.reference_requires_storage(parameters[*ordinal]) ||
                     argument_requires_storage(call.arguments[*ordinal]);
    append_intervals(call.arguments[*ordinal], result_intervals);
  } else if (callee.returns_current_receiver_reference()) {
    result_storage =
        (call.current_receiver && receiver_requires_storage_) ||
        callee.receiver_requires_storage_ || call.storage_receiver ||
        (call.receiver && argument_requires_storage(*call.receiver));
    if (call.current_receiver) {
      auto* object_type = implicit_object_type(unit_, function_);
      auto* record =
          object_type
              ? types_.record(types_.unqualified(object_type), call.source)
              : nullptr;
      if (record) {
        for (const auto& member : record->members) {
          append_interval(result_intervals,
                          {member.source, 0, member.partition->component_count,
                           member.component_offset});
        }
      }
    } else if (call.receiver) {
      append_intervals(*call.receiver, result_intervals);
    }
  }
  bool changed = false;
  if (result_storage) {
    if (!reference_identity_facts_) {
      reference_identity_facts_ = std::make_unique<ReferenceIdentityFacts>();
    }
    changed =
        reference_identity_facts_->storage_results.insert(call.source).second;
  }
  if (result_intervals.empty()) {
    return changed;
  }

  if (!reference_identity_facts_) {
    reference_identity_facts_ = std::make_unique<ReferenceIdentityFacts>();
  }
  auto& retained = reference_identity_facts_->results[call.source];
  for (const auto& interval : result_intervals) {
    changed = append_interval(retained, interval) || changed;
  }
  return changed;
}

CallPropagation ControlFlow::propagate_overlapping_references(
    const DirectCall& call, ControlFlow& callee) {
  bool caller_changed = false;
  bool callee_changed = false;
  auto parameters = callee.function_->parameters();
  auto select_storage = [&](size_t ordinal) {
    auto* parameter = parameters[ordinal];
    callee_changed =
        callee.require_reference_storage(parameter) || callee_changed;
    caller_changed =
        require_identity(call.arguments[ordinal]) || caller_changed;
  };

  // A value-reference call represents each mutable parameter as an independent
  // result writeback. Overlapping source intervals require one shared storage
  // identity so reads and writes observe the source language's aliasing order.
  const size_t parameter_count =
      std::min(parameters.size(), call.arguments.size());
  for (size_t first = 0; first < parameter_count; ++first) {
    if (!unit_.typeTraits().is_reference(parameters[first]->type())) {
      continue;
    }
    for (size_t second = first + 1; second < parameter_count; ++second) {
      if (!unit_.typeTraits().is_reference(parameters[second]->type()) ||
          (!callee.reference_written(parameters[first]) &&
           !callee.reference_written(parameters[second])) ||
          !identities_overlap(call.arguments[first], call.arguments[second])) {
        continue;
      }
      select_storage(first);
      select_storage(second);
    }
  }

  if (!call.function->isImplicitObjectMemberFunction()) {
    return {caller_changed, callee_changed};
  }
  const bool receiver_mutable = !callee.receiver_writes().empty();
  for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
    auto* parameter = parameters[ordinal];
    const auto& argument = call.arguments[ordinal];
    if (!unit_.typeTraits().is_reference(parameter->type()) ||
        (!receiver_mutable && !callee.reference_written(parameter))) {
      continue;
    }
    bool receiver_overlap = false;
    if (call.current_receiver) {
      receiver_overlap = argument.current_receiver ||
                         references_current_receiver_field(argument);
    } else if (call.receiver) {
      receiver_overlap = identities_overlap(*call.receiver, argument);
    }
    if (!receiver_overlap) {
      continue;
    }

    select_storage(ordinal);
    const bool receiver_already_storage =
        call.storage_receiver ||
        (call.current_receiver
             ? receiver_requires_storage_
             : call.receiver && argument_requires_storage(*call.receiver));
    if (!receiver_already_storage && !callee.receiver_requires_storage_) {
      callee.receiver_requires_storage_ = true;
      callee_changed = true;
    }
  }
  return {caller_changed, callee_changed};
}

CallPropagation ControlFlow::propagate_call(const DirectCall& call,
                                            ControlFlow& callee) {
  bool caller_changed = false;
  bool callee_changed = false;
  auto parameters = callee.function_->parameters();
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    if (!unit_.typeTraits().is_reference(parameter->type()) ||
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
          record_projected_write(argument, {}, {}, WriteEffect::Callable) ||
          caller_changed;
    }
  }
  caller_changed =
      retain_reference_result_identity(call, callee) || caller_changed;
  auto overlap_changes = propagate_overlapping_references(call, callee);
  caller_changed = overlap_changes.caller_changed || caller_changed;
  callee_changed = overlap_changes.callee_changed || callee_changed;

  if (identity_call_results_.contains(call.source)) {
    if (auto ordinal = callee.reference_return_parameter()) {
      callee_changed = callee.require_reference_storage(parameters[*ordinal]) ||
                       callee_changed;
      caller_changed =
          require_identity(call.arguments[*ordinal]) || caller_changed;
    }
  }
  const ReferenceWriteFacts::Result* result_write = nullptr;
  if (reference_write_facts_) {
    auto found = reference_write_facts_->results.find(call.source);
    if (found != reference_write_facts_->results.end()) {
      result_write = &found->second;
    }
  }
  if (result_write) {
    const auto& write = *result_write;
    for (const auto& projection : write.projections) {
      if (auto ordinal = callee.reference_return_parameter()) {
        caller_changed =
            record_projected_write(call.arguments[*ordinal], write.owners,
                                   projection.range, projection.effect) ||
            caller_changed;
      } else if (call.function->isImplicitObjectMemberFunction() &&
                 callee.returns_current_receiver_reference()) {
        if (call.current_receiver) {
          const auto* record = types_.record(
              types_.unqualified(implicit_object_type(unit_, function_)),
              call.source);
          for (const auto& member : record->members) {
            if (project_interval(
                    {member.source, 0, member.partition->component_count,
                     member.component_offset},
                    projection.range)) {
              caller_changed =
                  record_scoped_write(member.source, write.owners) ||
                  caller_changed;
              if (projection.effect == WriteEffect::Callable) {
                caller_changed =
                    add_receiver_write(member.source) || caller_changed;
              }
            }
          }
        } else if (!call.storage_receiver && call.receiver) {
          caller_changed =
              record_projected_write(*call.receiver, write.owners,
                                     projection.range, projection.effect) ||
              caller_changed;
        }
      }
    }
  }

  if (!call.function->isImplicitObjectMemberFunction()) {
    return {caller_changed, callee_changed};
  }
  if (identity_call_results_.contains(call.source) &&
      callee.returns_current_receiver_reference()) {
    caller_changed =
        storage_call_results_.insert(call.source).second || caller_changed;
    if (call.current_receiver) {
      caller_changed = !receiver_requires_storage_ || caller_changed;
      receiver_requires_storage_ = true;
    } else if (!call.storage_receiver && call.receiver) {
      caller_changed = require_identity(*call.receiver) || caller_changed;
    }
  }
  if (callee.receiver_requires_storage()) {
    if (call.current_receiver) {
      caller_changed = !receiver_requires_storage_ || caller_changed;
      receiver_requires_storage_ = true;
    } else if (!call.storage_receiver && call.receiver) {
      caller_changed = require_identity(*call.receiver) || caller_changed;
    }
  }
  if (call.function->isDestructor()) {
    return {caller_changed, callee_changed};
  }
  if (callee.receiver_writes().empty()) {
    return {caller_changed, callee_changed};
  }
  if (call.current_receiver) {
    for (auto* field : callee.receiver_writes()) {
      caller_changed = add_receiver_write(field) || caller_changed;
    }
  } else if (call.receiver) {
    std::span<cxx::AST* const> owners;
    if (reference_write_facts_) {
      auto found = reference_write_facts_->receiver_owners.find(call.source);
      if (found != reference_write_facts_->receiver_owners.end()) {
        owners = found->second;
      }
    }
    caller_changed = record_argument_write(*call.receiver, owners,
                                           callee.receiver_writes()) ||
                     caller_changed;
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
  std::vector<cxx::Symbol*> ordered;
  ordered.reserve(receiver_writes_.size());
  for (const auto& member :
       types_.record(record->type(), function_->declaration())->members) {
    auto* field = member.source;
    if (std::ranges::find(receiver_writes_, field) != receiver_writes_.end()) {
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
  auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(ast->symbol);
  if (variable && variable->isStatic()) {
    static_locals_.push_back(ast);
  }
  if (variable) {
    if (auto* conversion = constructor_conversion(ast->initializer);
        conversion &&
        same_function(conversion->conversionFunction,
                      variable->constructor()) &&
        types_.record_constructor_requires_call(
            variable->type(), conversion->conversionFunction, ast)) {
      bound_constructor_conversions_.insert(conversion);
    }
  }
  cxx::ASTVisitor::visit(ast);
  if (variable) {
    if (unit_.typeTraits().is_reference(variable->type()) && ast->initializer) {
      cxx::Initializer source(ast->initializer);
      auto* referent = source.singleExpression();
      if (!referent) {
        diagnostics_.reject(unit_, ast->initializer,
                            "reference locals require one initializer "
                            "expression");
      }
      auto destination = retain_destination(referent);
      auto identity = retain_call_argument(referent);
      if (argument_requires_storage(identity)) {
        require_identity(identity);
      }
      reference_aliases_.emplace(
          variable, ReferenceAlias{destination, std::move(identity)});
    }
    retain_reference_initializers(variable, ast->initializer, ast);
    const bool record_constructor_call =
        retain_constructor(variable, ast->initializer, ast);
    retain_automatic_storage(variable, ast->initializer,
                             record_constructor_call);
    analyze_default_constructor(variable->type(), variable->constructor(), ast);
    if (!ast->initializer && !storage_backed(variable)) {
      construction_.declare(
          variable, types_.partition(variable->type(), ast).component_count);
      if (default_constructor_initializes(variable->constructor())) {
        construction_.assign(variable, 0, SIZE_MAX);
      }
    }
    retain_automatic_destruction(variable, ast);
  }
}

void ControlFlow::visit(cxx::StructuredBindingDeclarationAST* ast) {
  // Raggi retains both the source initializer and the synthesized hidden
  // variable that owns it. The latter is the semantic declaration: visiting
  // both would count every initializer effect twice. Analyze the lowered
  // declaration sequence in its C++ initialization order instead.
  accept(ast->hiddenVariable);
  for (auto* binding : cxx::ListView{ast->bindingDeclaratorList}) {
    accept(binding);
  }
}

void ControlFlow::visit(cxx::ImplicitCastExpressionAST* ast) {
  if (observes_pointer_captures_ && !ast->conversionFunction &&
      ast->castKind == cxx::ImplicitCastKind::kLValueToRValueConversion) {
    auto* saved_read = pointer_value_read_;
    pointer_value_read_ = unwrap_expression(ast->expression);
    cxx::ASTVisitor::visit(ast);
    pointer_value_read_ = saved_read;
  } else {
    cxx::ASTVisitor::visit(ast);
  }
  if (is_class_adjustment(ast)) {
    if (unit_.typeTraits().is_pointer(ast->type)) {
      diagnostics_.reject(
          unit_, ast,
          "class-pointer conversions require null-preserving buffer identity; "
          "use a reference conversion for a live object");
    }
    if (ast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion) {
      require_identity(ast->expression);
    } else {
      retain_destination(ast);
    }
    return;
  }
  auto* conversion_object =
      implicit_object_type(unit_, ast->conversionFunction);
  if (ast->conversionFunction && !ast->conversionFunction->isConstructor() &&
      types_.record(conversion_object, ast)) {
    retain_call(
        DirectOperatorCall{
            .function = ast->conversionFunction,
            .receiver = ast->expression,
            .arguments = {},
            .argument_count = 0,
            .evaluation_order = OperatorEvaluationOrder::Forward,
            .has_postfix_dummy = false,
            .is_virtual_dispatch = ast->isVirtualDispatch,
        },
        ast);
    return;
  }
  if (bound_constructor_conversions_.contains(ast) ||
      !types_.record_constructor_requires_call(ast->type,
                                               ast->conversionFunction, ast)) {
    return;
  }
  retain_constructor_call(ast->conversionFunction, ast->expression,
                          CallArgument{.inherent_storage = true},
                          /*storage_receiver=*/true, ast);
}

bool ControlFlow::retain_array_copy(const cxx::Type* type,
                                    cxx::ExpressionAST* initializer,
                                    cxx::FunctionSymbol* constructor,
                                    cxx::AST* owner) {
  if (!constructor || types_.is_trivial_value_operation(constructor)) {
    return false;
  }
  if (auto copy = array_copy_initialization(unit_, function_, type, initializer,
                                            constructor, owner)) {
    diagnostics_.finish();
    require_identity(copy->source);
    for (auto* argument : copy->default_arguments) {
      accept(argument);
    }
    DirectCall call{
        .function = copy->constructor,
        .source = owner,
        .arguments = {retain_call_argument(copy->source)},
        .receiver = CallArgument{.inherent_storage = true},
        .current_receiver = false,
        .storage_receiver = true,
    };
    // Each leaf is a subobject of this already-addressable source array.
    call.arguments.front().inherent_storage = true;
    for (auto* argument : copy->default_arguments) {
      call.arguments.push_back(retain_call_argument(argument));
    }
    retain_reference_arguments(call);
    calls_.push_back(std::move(call));
    array_copies_.emplace(owner, std::move(*copy));
    return true;
  }
  return false;
}

void ControlFlow::retain_member_constructor(
    cxx::MemInitializerAST* initializer) {
  if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(initializer->symbol);
      field && unit_.typeTraits().is_array(field->type()) &&
      retain_array_copy(field->type(),
                        cxx::memInitializerClause(unit_.arena(), initializer),
                        initializer->constructor, initializer)) {
    return;
  }
  auto* delegate = function_->delegatingConstructor();
  if (delegate && same_function(delegate, initializer->constructor)) {
    auto* object_type = implicit_object_type(unit_, function_);
    if (types_.record_constructor_requires_call(object_type, delegate,
                                                initializer)) {
      auto* source = cxx::memInitializerClause(unit_.arena(), initializer);
      // Delegation has no source receiver expression: the selected call reuses
      // the constructor's current object and its existing storage identity.
      retain_constructor_call(
          delegate, source,
          CallArgument{.current_receiver = true, .inherent_storage = true},
          /*storage_receiver=*/true, initializer);
    } else {
      analyze_default_constructor(object_type, delegate, initializer);
    }
    return;
  }
  auto* field = initializer->symbol;
  if (!is_direct_subobject(field) ||
      !types_.record_constructor_requires_call(
          source_object_type(field), initializer->constructor, initializer)) {
    return;
  }
  auto* source = cxx::memInitializerClause(unit_.arena(), initializer);
  retain_constructor_call(initializer->constructor, source,
                          CallArgument{
                              .write_binding = initializer->symbol,
                              .inherent_storage = true,
                          },
                          /*storage_receiver=*/true, initializer);
}

void ControlFlow::visit(cxx::ParenMemInitializerAST* ast) {
  cxx::ASTVisitor::visit(ast);
  retain_member_constructor(ast);
}

void ControlFlow::visit(cxx::BracedMemInitializerAST* ast) {
  cxx::ASTVisitor::visit(ast);
  retain_member_constructor(ast);
}

void ControlFlow::visit(cxx::ConditionExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (ast->symbol) {
    retain_reference_initializers(ast->symbol, ast->initializer, ast);
    const bool record_constructor_call =
        retain_constructor(ast->symbol, ast->initializer, ast);
    retain_automatic_storage(ast->symbol, ast->initializer,
                             record_constructor_call);
    analyze_default_constructor(ast->symbol->type(), ast->symbol->constructor(),
                                ast);
    retain_automatic_destruction(ast->symbol, ast);
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
  if (types_.record_constructor_requires_call(ast->type, ast->constructorSymbol,
                                              ast)) {
    retain_constructor_call(ast->constructorSymbol, ast,
                            CallArgument{.inherent_storage = true},
                            /*storage_receiver=*/true, ast);
  }
  analyze_default_constructor(ast->type, ast->constructorSymbol, ast);
}

void ControlFlow::visit(cxx::BracedTypeConstructionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (types_.record_constructor_requires_call(ast->type, ast->constructorSymbol,
                                              ast)) {
    retain_constructor_call(ast->constructorSymbol, ast,
                            CallArgument{.inherent_storage = true},
                            /*storage_receiver=*/true, ast);
  }
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
  auto target = classify_destination(unwrap_object_identity(unit_, expression));
  if (!target) {
    return std::nullopt;
  }
  auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression);
  if (!id || id->subobjectPath) {
    destinations_.emplace(expression, *target);
  }
  return target;
}

bool ControlFlow::record(cxx::ExpressionAST* expression, WriteEffect effect) {
  auto target = retain_destination(expression);
  if (!target) {
    auto argument = retain_call_argument(expression);
    record_projected_write(argument, owners_, {}, effect);
    return !argument.inherent_storage;
  }
  record(target->binding);
  if (effect == WriteEffect::Callable) {
    record_callable_write(target->binding);
  }
  return true;
}

void ControlFlow::record(cxx::Symbol* binding) {
  record_scoped_write(binding, owners_);
}

void ControlFlow::record_implicit_object_call(cxx::CallExpressionAST* call,
                                              cxx::FunctionSymbol* callee) {
  if (!callee->isImplicitObjectMemberFunction()) {
    return;
  }
  auto* base = unwrap_expression(call->baseExpression);
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base);
      id && id->subobjectPath) {
    if (types_.implicit_object_may_mutate(callee, call)) {
      record(types_.projection(id->subobjectPath, id).root->source);
    }
    return;
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
    auto* receiver = member->baseExpression;
    if (member->subobjectPath && is_current_object(unit_, receiver)) {
      if (types_.implicit_object_may_mutate(callee, call)) {
        record(types_.projection(member->subobjectPath, member).root->source);
      }
      return;
    }
    if (!is_current_object(unit_, receiver)) {
      if (member->accessOp == cxx::TokenKind::T_DOT &&
          receiver->valueCategory != cxx::ValueCategory::kPrValue) {
        if (types_.implicit_object_may_mutate(callee, call)) {
          record(receiver);
        } else {
          retain_destination(receiver);
        }
      }
      return;
    }
  }
  if (!types_.implicit_object_may_mutate(callee, call)) {
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
    record(member.source);
  }
}

std::optional<Destination> ControlFlow::classify_destination(
    cxx::ExpressionAST* expression) {
  if (auto found = destinations_.find(expression);
      found != destinations_.end()) {
    return found->second;
  }
  auto* identity = unwrap_object_identity(unit_, expression);
  if (identity != expression) {
    return classify_destination(identity);
  }
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    if (auto found = reference_aliases_.find(id->symbol);
        found != reference_aliases_.end()) {
      return found->second.destination;
    }
    auto* root = id->subobjectPath ? id->subobjectPath->value : id->symbol;
    if (is_direct_subobject(root) && is_captured_object_field(root)) {
      return std::nullopt;
    }
    if (id->subobjectPath) {
      auto selected = types_.selection(id->symbol, id->subobjectPath, id);
      return Destination{
          root, selected.component_offset - selected.root->component_offset,
          selected.partition};
    }
    return Destination{id->symbol, 0, nullptr};
  }
  if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
      cast && is_class_adjustment(cast)) {
    if (cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion) {
      return std::nullopt;
    }
    if (is_current_object(unit_, cast->expression)) {
      const auto& selected = types_.projection(cast->subobjectPath, cast);
      return Destination{
          selected.root->source,
          selected.component_offset - selected.root->component_offset,
          selected.partition};
    }
    auto target = classify_destination(cast->expression);
    if (target) {
      const auto& selected = types_.projection(cast->subobjectPath, cast);
      target->component_offset += selected.component_offset;
      target->member = selected.partition;
    }
    return target;
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
  if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
    return member->accessOp == cxx::TokenKind::T_DOT
               ? classify_destination(member->baseExpression)
               : std::nullopt;
  }
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
  if (!field || field->isStatic()) {
    return std::nullopt;
  }
  if (is_current_object(unit_, member->baseExpression)) {
    auto selected = types_.selection(field, member->subobjectPath, member);
    if (is_captured_object_field(selected.root->source)) {
      return std::nullopt;
    }
    return Destination{
        selected.root->source,
        selected.component_offset - selected.root->component_offset,
        member->subobjectPath ? selected.partition : nullptr};
  }
  if (member->accessOp != cxx::TokenKind::T_DOT) {
    return std::nullopt;
  }
  // Establish the automatic owner before requesting any SSA partition: a dot
  // member can also project a record reached through a pointer or subscript.
  auto target = classify_destination(member->baseExpression);
  if (target) {
    auto slice = types_.selection(field, member->subobjectPath, member);
    target->component_offset += slice.component_offset;
    target->member = slice.partition;
  }
  return target;
}

bool ControlFlow::classify_storage(cxx::ExpressionAST* expression) const {
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    const bool object_member =
        member->vectorComponentKind != cxx::VectorComponentKind::kNone ||
        (field && !field->isStatic());
    return object_member &&
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
  if (auto* operand = explicit_reference_operand(unit_, expression)) {
    return storage_expressions_.contains(operand);
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

bool ControlFlow::is_captured_object_field(cxx::Symbol* field) const {
  auto* closure = function_
                      ? cxx::symbol_cast<cxx::ClassSymbol>(function_->parent())
                      : nullptr;
  return captured_object_member(unit_, closure, field).has_value();
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
  const bool constant_bound =
      address_constant(unit_, address, condition->rightExpression).has_value();
  if (!left || left->symbol != induction || (!bound && !constant_bound) ||
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
      (bound &&
       (unit_.typeTraits().is_volatile(bound->type) ||
        std::ranges::find(loop_writes, bound->symbol) != loop_writes.end()))) {
    return std::nullopt;
  }
  return CountedLoop{induction,
                     CountedLoop::AddressRange{
                         CountedLoop::Bound{bound ? bound->symbol : nullptr,
                                            condition->rightExpression},
                         domain == LOOM_SCALAR_TYPE_INDEX
                             ? CountedLoop::AddressDomain::Index
                             : CountedLoop::AddressDomain::Offset}};
}

}  // namespace loom::cxx_import
