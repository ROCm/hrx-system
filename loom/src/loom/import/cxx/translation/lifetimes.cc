// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation/lifetimes.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/import/cxx/value/objects.h"
#include "loom/ops/scf/ops.h"

namespace loom::cxx_import {

namespace {

DestructionAction local_destruction(const Destruction& destruction,
                                    cxx::Symbol* binding, cxx::AST* source) {
  return {
      .destruction = destruction,
      .binding = binding,
      .object_type = source_object_type(binding),
      .source = source,
  };
}

template <typename Emit>
void emit_cleanup_if(loom_builder_t& builder, loom_value_id_t condition,
                     loom_location_id_t source, Emit emit) {
  loom_op_t* op;
  check(loom_scf_if_build(&builder, 0, condition, nullptr, 0, nullptr, 0,
                          source, &op));
  auto saved =
      loom_builder_enter_region(&builder, op, loom_scf_if_then_region(op));
  emit();
  loom_op_t* yield;
  check(loom_scf_yield_build(&builder, nullptr, 0, source, &yield));
  loom_builder_restore(&builder, saved);
}

}  // namespace

ObjectLifetimes::ObjectLifetimes(cxx::TranslationUnit& unit, Scalars& scalars,
                                 Storage& storage, Objects& objects,
                                 Locations& locations, ValueArena& arena,
                                 loom_builder_t& builder, Emitter& emitter)
    : unit_(unit),
      scalars_(scalars),
      storage_(storage),
      objects_(objects),
      locations_(locations),
      value_arena_(arena),
      builder_(builder),
      emitter_(emitter) {}

ObjectLifetimes::~ObjectLifetimes() = default;

void ObjectLifetimes::reset(const ControlFlow& control) {
  control_ = &control;
  active_destructions_.clear();
  temporary_destructions_.clear();
  temporary_results_.clear();
  full_expression_mark_.reset();
  extends_temporaries_ = false;
  conditional_cleanup_anchors_.clear();
}

void ObjectLifetimes::leave_conditional(loom_op_t* operation) {
  IREE_ASSERT(!conditional_cleanup_anchors_.empty());
  IREE_ASSERT(!operation ||
              conditional_cleanup_anchors_.back().before_op == operation);
  conditional_cleanup_anchors_.pop_back();
}

void ObjectLifetimes::emit_destruction(const DestructionAction& active) {
  if (!active.activity) {
    emitter_.destroy_object(active);
    return;
  }
  auto* bool_type = unit_.control()->getBoolType();
  auto condition = storage_.load(*active.activity, bool_type, active.source);
  emit_cleanup_if(builder_, condition, locations_.get(active.source),
                  [&] { emitter_.destroy_object(active); });
}

void ObjectLifetimes::emit_destructions(size_t mark) {
  IREE_ASSERT_LE(mark, active_destructions_.size());
  for (size_t index = active_destructions_.size(); index > mark; --index) {
    emit_destruction(active_destructions_[index - 1]);
  }
}

void ObjectLifetimes::emit_automatic_destruction(cxx::Symbol* binding,
                                                 cxx::AST* source) {
  if (const auto* destruction = control_->automatic_destruction(binding)) {
    emitter_.destroy_object(local_destruction(*destruction, binding, source));
  }
}

void ObjectLifetimes::emit_automatic_destruction_if(loom_value_id_t condition,
                                                    cxx::Symbol* binding,
                                                    cxx::AST* source) {
  if (const auto* destruction = control_->automatic_destruction(binding)) {
    emit_cleanup_if(builder_, condition, locations_.get(source), [&] {
      emitter_.destroy_object(local_destruction(*destruction, binding, source));
    });
  }
}

void ObjectLifetimes::close_scope(size_t mark) {
  emit_destructions(mark);
  active_destructions_.resize(mark);
}

void ObjectLifetimes::close_scope_if(loom_value_id_t condition, size_t mark,
                                     cxx::AST* source) {
  if (mark == active_destructions_.size()) {
    return;
  }
  emit_cleanup_if(builder_, condition, locations_.get(source),
                  [&] { emit_destructions(mark); });
  active_destructions_.resize(mark);
}

void ObjectLifetimes::activate_local(cxx::Symbol* binding, cxx::AST* source) {
  if (const auto* destruction = control_->automatic_destruction(binding)) {
    activate(local_destruction(*destruction, binding, source));
  }
}

void ObjectLifetimes::activate_subobjects(cxx::AST* source) {
  for (const auto& subobject : control_->subobject_destructions()) {
    active_destructions_.push_back({
        .destruction = subobject.destruction,
        .field = subobject.field,
        .object_type = source_object_type(subobject.field),
        .source = source,
    });
  }
}

void ObjectLifetimes::allocate_temporary_storage(DestructionAction& action) {
  auto saved = loom_builder_save(&builder_);
  if (!conditional_cleanup_anchors_.empty()) {
    loom_builder_restore(&builder_, conditional_cleanup_anchors_.front());
  }
  auto object_allocation =
      storage_.allocate(action.object_type,
                        LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, action.source);
  action.storage = storage_.project(object_allocation.pointer,
                                    action.object_type, action.source);
  if (!conditional_cleanup_anchors_.empty()) {
    auto* bool_type = unit_.control()->getBoolType();
    auto activity_allocation = storage_.allocate(
        bool_type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, action.source);
    action.activity = StorageAccess{activity_allocation.view, std::nullopt};
    auto inactive = scalars_.integer(false, LOOM_SCALAR_TYPE_I1,
                                     locations_.get(action.source));
    storage_.store(*action.activity, inactive, bool_type, action.source);
    loom_builder_restore(&builder_, saved);
  }
}

void ObjectLifetimes::activate_temporary_storage(
    const DestructionAction& action) {
  if (action.activity) {
    auto active = scalars_.integer(true, LOOM_SCALAR_TYPE_I1,
                                   locations_.get(action.source));
    storage_.store(*action.activity, active, unit_.control()->getBoolType(),
                   action.source);
  }
}

void ObjectLifetimes::materialize_temporary_value(DestructionAction& action) {
  allocate_temporary_storage(action);
  objects_.store(*action.storage, *action.value, action.object_type,
                 action.source);
  activate_temporary_storage(action);
  action.value.reset();
}

void ObjectLifetimes::materialize_conditional_temporaries(size_t mark) {
  IREE_ASSERT(!conditional_cleanup_anchors_.empty());
  IREE_ASSERT_LE(mark, temporary_destructions_.size());
  for (size_t index = mark; index < temporary_destructions_.size(); ++index) {
    auto& action = temporary_destructions_[index];
    if (action && action->value) {
      materialize_temporary_value(*action);
    }
  }
}

std::optional<TemporaryReference> ObjectLifetimes::temporary_result(
    cxx::ExpressionAST* expression) const {
  auto* origin = control_->temporary_origins().materialization(expression);
  auto found = temporary_results_.find(origin);
  return found == temporary_results_.end()
             ? std::nullopt
             : std::optional(TemporaryReference{found->second});
}

void ObjectLifetimes::forget_temporary(size_t index) {
  const auto& action = *temporary_destructions_[index];
  auto found = temporary_results_.find(action.origin);
  if (found != temporary_results_.end() && found->second == index) {
    temporary_results_.erase(found);
  }
  temporary_destructions_[index].reset();
}

void ObjectLifetimes::extend_materialized_temporary(size_t index) {
  auto& action = *temporary_destructions_[index];
  if (!extends_temporaries_ &&
      control_->temporary_origins().extended(action.origin)) {
    active_destructions_.push_back(action);
    forget_temporary(index);
  }
}

void ObjectLifetimes::retain_temporary(cxx::ExpressionAST* origin,
                                       cxx::ExpressionAST* expression,
                                       Value value, size_t evaluation_mark) {
  const auto& destruction = control_->temporary_origins().destruction(origin);
  // Transparent syntax can complete an object already created by a child.
  // A later evaluation of the same source construction is a distinct object.
  auto found = temporary_results_.find(origin);
  if (found != temporary_results_.end() && found->second >= evaluation_mark) {
    auto& action = *temporary_destructions_[found->second];
    if (!action.storage) {
      action.value = value;
    }
    return;
  }
  IREE_ASSERT(full_expression_mark_);
  temporary_results_.insert_or_assign(origin, temporary_destructions_.size());
  temporary_destructions_.push_back(DestructionAction{
      .destruction = destruction,
      .object_type = expression->type,
      .value = value,
      .origin = origin,
      .source = expression,
  });
}

FullExpression ObjectLifetimes::begin_full_expression(bool extend_temporaries) {
  FullExpression expression = {
      .mark = temporary_destructions_.size(),
      .parent_mark = full_expression_mark_,
      .parent_extends_temporaries = extends_temporaries_,
  };
  full_expression_mark_ = expression.mark;
  extends_temporaries_ = extend_temporaries;
  return expression;
}

void ObjectLifetimes::finish_full_expression(FullExpression expression) {
  IREE_ASSERT(full_expression_mark_);
  IREE_ASSERT_EQ(*full_expression_mark_, expression.mark);
  IREE_ASSERT_LE(expression.mark, temporary_destructions_.size());
  if (extends_temporaries_) {
    // The range initializer's surviving caller-owned temporaries become
    // lexical objects in their original construction order. Adopted function
    // arguments and nested full expressions have already retired their slots.
    for (size_t index = expression.mark; index < temporary_destructions_.size();
         ++index) {
      if (temporary_destructions_[index]) {
        active_destructions_.push_back(*temporary_destructions_[index]);
        forget_temporary(index);
      }
    }
  }
  for (size_t index = temporary_destructions_.size(); index > expression.mark;
       --index) {
    if (temporary_destructions_[index - 1]) {
      emit_destruction(*temporary_destructions_[index - 1]);
      forget_temporary(index - 1);
    }
  }
  temporary_destructions_.resize(expression.mark);
  full_expression_mark_ = expression.parent_mark;
  extends_temporaries_ = expression.parent_extends_temporaries;
}

// Result ownership transfers by construction identity, independently of
// the immutable payload and the number of transparent syntax wrappers.
void ObjectLifetimes::adopt_temporary_result(cxx::ExpressionAST* expression) {
  auto* origin = control_->temporary_origins().result(expression);
  auto found = temporary_results_.find(origin);
  if (found != temporary_results_.end()) {
    forget_temporary(found->second);
  }
}

// A reference can extend a complete temporary through one of its
// subobjects. Adopt the complete materialization rather than requiring the
// initializer expression itself to be the owning prvalue.
void ObjectLifetimes::bind_temporary_materialization(
    cxx::ExpressionAST* expression, cxx::Symbol* binding, cxx::AST* source) {
  auto* origin = control_->temporary_origins().materialization(expression);
  auto found = temporary_results_.find(origin);
  if (found != temporary_results_.end()) {
    if (extends_temporaries_) {
      // Preserve this object's construction position among the initializer's
      // other temporaries, but read its mutable named value at scope exit.
      auto& action = *temporary_destructions_[found->second];
      action.binding = binding;
      action.value.reset();
      action.source = source;
      return;
    }
    forget_temporary(found->second);
  }
  activate_local(binding, source);
}

void ObjectLifetimes::update_temporary(
    std::optional<TemporaryReference> temporary, Value updated,
    const cxx::Type* type, cxx::AST* owner) {
  if (temporary) {
    auto& action = *temporary_destructions_[temporary->index];
    if (action.storage) {
      objects_.store(
          storage_.subobject(*action.storage, temporary->byte_offset, owner),
          updated, type, owner);
    } else {
      action.value = value_arena_.replace(*action.value,
                                          temporary->component_offset, updated);
    }
  }
}

StorageProjection ObjectLifetimes::materialize_temporary(
    cxx::ExpressionAST* expression, Value value,
    std::optional<TemporaryReference> temporary) {
  if (!temporary) {
    temporary = temporary_result(expression);
  }
  if (temporary) {
    auto& action = *temporary_destructions_[temporary->index];
    if (!action.storage) {
      materialize_temporary_value(action);
    }
    auto storage = *action.storage;
    if (temporary->byte_offset) {
      storage = storage_.subobject(storage, temporary->byte_offset, expression);
    }
    extend_materialized_temporary(temporary->index);
    return storage;
  }
  auto allocation = storage_.allocate(
      expression->type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, expression);
  auto storage =
      storage_.project(allocation.pointer, expression->type, expression);
  objects_.store(storage, value, expression->type, expression);
  return storage;
}

PendingConstruction ObjectLifetimes::begin_construction(const cxx::Type* type,
                                                        cxx::AST* owner) {
  auto* expression = cxx::ast_cast<cxx::ExpressionAST>(owner);
  auto* origin =
      expression ? control_->temporary_origins().result(expression) : nullptr;
  PendingConstruction object;
  if (origin) {
    object.cleanup.emplace(DestructionAction{
        .destruction = control_->temporary_origins().destruction(origin),
        .object_type = type,
        .origin = origin,
        .source = owner,
    });
    allocate_temporary_storage(*object.cleanup);
    object.storage = *object.cleanup->storage;
  } else {
    auto allocation =
        storage_.allocate(type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, owner);
    object.storage = storage_.project(allocation.pointer, type, owner);
  }
  return object;
}

void ObjectLifetimes::complete_construction(PendingConstruction& object) {
  if (object.cleanup) {
    activate_temporary_storage(*object.cleanup);
    auto index = temporary_destructions_.size();
    temporary_results_.insert_or_assign(object.cleanup->origin, index);
    temporary_destructions_.push_back(std::move(*object.cleanup));
    extend_materialized_temporary(index);
  }
}

}  // namespace loom::cxx_import
