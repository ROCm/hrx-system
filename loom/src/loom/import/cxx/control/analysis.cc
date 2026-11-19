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
#include <string>
#include <string_view>
#include <utility>

#include "loom/import/cxx/binding/imports.h"
#include "loom/import/cxx/control/call_aliases.h"
#include "loom/import/cxx/control/object_identity.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/initializer_lists.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/import/cxx/source/range_loops.h"

namespace loom::cxx_import {

struct ControlFlow::ReferenceIdentityFacts {
  struct IntervalHash {
    size_t operator()(const ObjectInterval& interval) const {
      auto mix = [](size_t hash, size_t value) {
        return hash ^
               (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
      };
      size_t hash = reinterpret_cast<uintptr_t>(interval.binding);
      hash = mix(hash, interval.begin);
      hash = mix(hash, interval.end);
      return mix(hash, interval.reference_offset);
    }
  };

  struct Result {
    // Exact automatic-object intervals in stable source encounter order.
    std::vector<ObjectInterval> intervals;
    // Wide fixed-point canonicalization index, absent for small results.
    std::unique_ptr<std::unordered_set<ObjectInterval, IntervalHash>> index;
    // Complete fixed array reached by a pointer-returning accessor.
    const cxx::Type* array_type = nullptr;
  };

  // Sparse member-slice vectors referenced directly by retained call edges.
  std::vector<std::unique_ptr<std::vector<ObjectInterval>>> argument_slices;
  // Exact automatic-object results and their optional wide analysis indices.
  std::unordered_map<cxx::AST*, Result> results;
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

struct ControlFlow::ProjectedReferenceReturnFacts {
  struct OriginHash {
    size_t operator()(const ReferenceResultOrigin& origin) const {
      auto mix = [](size_t hash, size_t value) {
        return hash ^
               (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
      };
      size_t hash = origin.owner_ordinal;
      hash = mix(hash, reinterpret_cast<uintptr_t>(origin.projection.type));
      hash =
          mix(hash, reinterpret_cast<uintptr_t>(origin.projection.partition));
      return mix(hash, origin.projection.component_offset);
    }
  };

  struct Selection {
    // Canonical origin indices in source selection order.
    std::vector<size_t> origins;
    // Callee origins already translated through this call. SIZE_MAX denotes a
    // direct source selection that does not participate in propagation.
    size_t propagated_origin_count = SIZE_MAX;
  };

  struct ArrayResult {
    // Returned element expression indexing its retained array selection.
    cxx::ExpressionAST* expression;
    // Array subobject within the call's complete returned reference.
    ReferenceProjection projection;
    // Callee origins already expanded into this element selection.
    size_t propagated_origin_count = 0;
  };

  // Possible caller-owned subobjects in source selection order.
  std::vector<ReferenceResultOrigin> origins;
  // Canonicalization index created only after the bounded linear prefix.
  std::unordered_map<ReferenceResultOrigin, size_t, OriginHash> origin_indices;
  // Source leaves indexing their possible origins and propagation progress.
  // A forwarded call can gain origins over several fixed-point iterations
  // without moving other leaves.
  std::unordered_map<cxx::AST*, Selection> selections;
  // Element returns waiting on an accessor's origins, indexed by that call.
  std::unordered_map<cxx::AST*, std::vector<ArrayResult>> array_results;
  // Whole-parameter leaves promoted into a mixed projected result.
  std::unordered_map<cxx::Symbol*, size_t> complete_parameter_indices;
  // Whole current-object leaf promoted into a mixed projected result.
  size_t complete_receiver_index = SIZE_MAX;
  // Reference parameters indexed once when a projected origin needs them.
  std::unordered_map<cxx::Symbol*, size_t> parameter_ordinals;
  // Direct receiver fields indexed once with their complete component offset.
  std::unordered_map<cxx::Symbol*, size_t> receiver_offsets;
  // Whether the wide parameter index has been populated.
  bool parameter_ordinals_indexed = false;
  // Whether the wide receiver-field index has been populated.
  bool receiver_offsets_indexed = false;
  // Parameter ordinals that own at least one projected result candidate.
  std::vector<bool> parameter_has_origin;
  // Parameters with a candidate smaller than their storage ABI carrier.
  std::unordered_set<cxx::Symbol*> partial_parameter_origins;
  // All discovered origins are complete caller arguments. Survives disposal
  // of the per-parameter origin index after the call graph converges.
  bool only_whole_parameters = true;
  // Whether a result candidate is smaller than the storage receiver.
  bool has_partial_receiver_origin = false;
  // Whether every origin must use durable pointer transport.
  bool requires_storage = false;
};

namespace {

// Small reference sets dominate ordinary source calls. Keep them inline while
// bounding the linear prefix before switching large sets to hashed lookup.
constexpr size_t kLinearCanonicalizationThreshold = 8;

bool pointer_returning_call(cxx::ExpressionAST* expression) {
  cxx::FunctionSymbol* function = nullptr;
  if (auto call = direct_operator_call(expression)) {
    function = call->function;
  } else if (auto* subscript =
                 cxx::ast_cast<cxx::SubscriptExpressionAST>(expression);
             subscript && subscript->symbol) {
    function = cxx::symbol_cast<cxx::FunctionSymbol>(subscript->symbol);
  } else if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(expression)) {
    function = direct_callee(call);
  }
  return function &&
         cxx::type_cast<cxx::PointerType>(
             cxx::type_cast<cxx::FunctionType>(function->type())->returnType());
}

// Most projected origins translate through one complete argument owner. Keep
// that result inline and spill only when the argument itself has several
// possible identities.
struct TranslatedReferenceOrigins {
  void append(ReferenceResultOrigin origin) {
    if (!single && multiple.empty()) {
      single = std::move(origin);
      return;
    }
    if (multiple.empty()) {
      multiple.reserve(4);
      multiple.push_back(std::move(*single));
      single.reset();
    }
    multiple.push_back(std::move(origin));
  }

  void invalidate() {
    valid = false;
    single.reset();
    multiple.clear();
  }

  bool empty() const { return !single && multiple.empty(); }

  std::span<ReferenceResultOrigin> values() {
    return single ? std::span<ReferenceResultOrigin>{&*single, 1}
                  : std::span<ReferenceResultOrigin>{multiple};
  }

  // Whether every visited interval maps to one complete caller-owned origin.
  bool valid = true;
  // Allocation-free ordinary result, absent after spilling or invalidation.
  std::optional<ReferenceResultOrigin> single;
  // Multiple concrete origins retained only for an actually selected argument.
  std::vector<ReferenceResultOrigin> multiple;
};

ReferenceProjection compose_projection(ReferenceProjection parent,
                                       ReferenceProjection child) {
  return {parent.component_offset + child.component_offset,
          child.component_count != SIZE_MAX ? child.component_count
          : parent.component_count == SIZE_MAX
              ? SIZE_MAX
              : parent.component_count - child.component_offset};
}

bool has_nonlocal_storage(cxx::VariableSymbol* variable) {
  return variable->isStatic() || variable->isExtern() ||
         variable->isThreadLocal() ||
         cxx::symbol_cast<cxx::NamespaceSymbol>(
             variable->canonical()->parent());
}

ObjectInterval object_interval(const Destination& destination) {
  return {destination.binding, destination.component_offset,
          destination.member ? destination.component_offset +
                                   destination.member->component_count
                             : ObjectInterval::kWholeObjectEnd};
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

// A pointer reference can use the existing two-component value partition. A
// nonpointer reference is a candidate only after Types has selected value
// transport before admitting the containing closure record.
cxx::FieldSymbol* reference_capture(cxx::TranslationUnit& unit, Types& types,
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
                 !traits.is_volatile(referent) &&
                 (traits.is_pointer(referent) ||
                  types.reference_is_snapshot(field))
             ? field
             : nullptr;
}

// Returns the overloaded subscript at the root of an immediate view element
// projection. Value conversion and dot field selection consume the synthetic
// lvalue without giving its source reference an independent identity.
cxx::SubscriptExpressionAST* view_element_origin(
    cxx::TranslationUnit& unit, cxx::ExpressionAST* expression) {
  expression = unwrap_object_identity(unit, expression);
  if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
      cast && !cast->conversionFunction &&
      cast->castKind == cxx::ImplicitCastKind::kLValueToRValueConversion) {
    return view_element_origin(unit, cast->expression);
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
      member && member->accessOp == cxx::TokenKind::T_DOT) {
    return view_element_origin(unit, member->baseExpression);
  }
  auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(expression);
  return subscript && has_op_binding(subscript->symbol, "view.element")
             ? subscript
             : nullptr;
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
                         Types& types, const Imports& imports,
                         cxx::StatementAST* body, cxx::FunctionSymbol* function)
    : unit_(unit),
      diagnostics_(diagnostics),
      types_(types),
      imports_(imports),
      function_(function),
      temporary_origins_(unit, types) {
  if (function_) {
    if (auto* object_type = implicit_object_type(unit_, function_)) {
      receiver_requires_storage_ = types_.has_volatile_subobject(object_type);
      receiver_partition_ = types_.record(types_.unqualified(object_type),
                                          function_->declaration());
      for (const auto& member : receiver_partition_->members) {
        construction_.declare_parameter(member.source,
                                        member.partition->component_count,
                                        !function_->isDestructor());
      }
    }
    auto* closure = cxx::symbol_cast<cxx::ClassSymbol>(function_->parent());
    if (closure && closure->isClosureType()) {
      auto* name = cxx::name_cast<cxx::OperatorId>(function_->name());
      observes_reference_captures_ =
          name && name->op() == cxx::TokenKind::T_LPAREN;
    }
    for (auto* parameter : function_->parameters()) {
      retain_owned_binding(parameter, nullptr);
      const auto traits = unit_.typeTraits();
      const auto* type = traits.remove_reference(parameter->type());
      if (!traits.is_reference(parameter->type()) && traits.is_volatile(type)) {
        binding_storage_[parameter].automatic_storage = true;
      }
      if (const auto* object_type = candidate_pointer_object_type(parameter)) {
        construction_.declare_parameter(
            parameter, types_
                           .partition(types_.unqualified(object_type),
                                      function_->declaration())
                           .component_count);
        CallArgument identity{.source = nullptr, .write_binding = parameter};
        append_identity({parameter, 0, nullptr}, identity);
        pointer_aliases_.emplace(
            parameter, PointerAlias{nullptr, object_type, std::move(identity)});
      } else if (traits.is_reference(parameter->type())) {
        if (types_.has_volatile_subobject(type)) {
          storage_references_.insert(parameter);
        }
        if (!traits.is_array(type) && !traits.is_volatile(type)) {
          construction_.declare_parameter(
              parameter,
              types_
                  .partition(types_.unqualified(type), function_->declaration())
                  .component_count);
        }
      }
      retain_automatic_destruction(parameter, function_->declaration());
    }
    if (function_->isDestructor()) {
      retain_subobject_destructions();
    }
  }
  if (function_ && function_->isConstructor()) {
    constructor_initializers_ = cxx::ctor_initializers(function_);
    for (auto* initializer : cxx::ListView{constructor_initializers_}) {
      accept(initializer);
    }
  }
  accept(body);
  for (auto& entry : pointer_aliases_) {
    auto& alias = entry.second;
    if (!alias.transparent) {
      if (alias.parameter_object_type) {
        require_borrowed_pointer_storage(entry.first);
      } else {
        require_identity(alias.identity);
      }
    }
  }
  if (reference_return_ == ReferenceReturn::Parameter) {
    IREE_ASSERT(reference_return_parameter_ != SIZE_MAX);
  }
  if (reference_return_ != ReferenceReturn::Projected) {
    projected_reference_return_facts_.reset();
  }
  temporary_origins_.finish();
  if (object_identity_) {
    object_identity_->finish();
  }
  construction_.finish_function();
}

const Destruction* ControlFlow::automatic_destruction(
    cxx::Symbol* binding) const {
  auto found = automatic_destructions_.find(binding);
  return found == automatic_destructions_.end() ? nullptr : &found->second;
}

ControlFlow::~ControlFlow() = default;

std::optional<size_t> ControlFlow::reference_return_parameter() const {
  return reference_return_ == ReferenceReturn::Parameter
             ? std::optional(reference_return_parameter_)
             : std::nullopt;
}

bool ControlFlow::selects_whole_parameters() const {
  const auto* facts = projected_reference_return_facts_.get();
  return facts && facts->only_whole_parameters && !facts->requires_storage;
}

std::span<const ReferenceResultOrigin>
ControlFlow::projected_reference_return_origins(bool value_receiver) const {
  if (reference_return_ != ReferenceReturn::Projected ||
      !projected_reference_return_facts_) {
    return {};
  }
  const auto& facts = *projected_reference_return_facts_;
  if (facts.requires_storage ||
      (!value_receiver && facts.has_partial_receiver_origin)) {
    return {};
  }
  return facts.origins;
}

std::span<const size_t> ControlFlow::reference_return_parameters() const {
  return reference_return_ == ReferenceReturn::Parameter
             ? std::span<const size_t>{&reference_return_parameter_, 1}
             : std::span<const size_t>{};
}

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
  return (found != binding_storage_.end() && found->second.addressed) ||
         (object_identity_ && object_identity_->requires_storage(binding));
}

bool ControlFlow::storage_backed(cxx::Symbol* binding) const {
  auto found = binding_storage_.find(binding);
  return (found != binding_storage_.end() &&
          (found->second.addressed || found->second.automatic_storage)) ||
         (object_identity_ && object_identity_->requires_storage(binding));
}

bool ControlFlow::object_requires_storage(cxx::AST* source) const {
  return object_identity_ && object_identity_->requires_storage(source);
}

const TemporaryMember* ControlFlow::temporary_member(
    cxx::ExpressionAST* source) const {
  return object_identity_ ? object_identity_->temporary_member(source)
                          : nullptr;
}

std::span<const ValueProjection> ControlFlow::storage_members(
    cxx::Symbol* binding) const {
  if (storage_members_) {
    auto found = storage_members_->find(binding);
    if (found != storage_members_->end()) {
      return found->second;
    }
  }
  return {};
}

ObjectInterval ControlFlow::storage_interval(ObjectInterval interval,
                                             cxx::AST* source) const {
  if (receiver_partition_ && is_direct_subobject(interval.binding) &&
      interval.binding->parent() == receiver_partition_->source &&
      !function_->isConstructor() && !function_->isDestructor() &&
      types_.requires_value_transport(receiver_partition_->source->type())) {
    const auto& field = types_.member(interval.binding, source);
    if (interval.end == ObjectInterval::kWholeObjectEnd) {
      interval.end = field.partition->component_count;
    }
    interval.binding = receiver_partition_->source;
    interval.begin += field.component_offset;
    interval.end += field.component_offset;
  }
  return interval;
}

bool ControlFlow::require_storage_member(ObjectInterval interval,
                                         cxx::AST* source, bool& changed) {
  interval = storage_interval(interval, source);
  if (interval.end == ObjectInterval::kWholeObjectEnd) {
    return false;
  }
  auto* type = source_object_type(interval.binding);
  // An indirect pointer parameter's identity describes its pointee.
  if (auto found = pointer_aliases_.find(interval.binding);
      found != pointer_aliases_.end() && found->second.parameter_object_type) {
    type = found->second.parameter_object_type;
  }
  if (unit_.typeTraits().is_reference(type)) {
    if (auto* origin = reference_materialization(interval.binding)) {
      type = origin->type;
    }
  }
  auto group =
      types_.storage_member(type, interval.begin, interval.end, source);
  if (!group) {
    return false;
  }
  if (!storage_members_) {
    storage_members_ = std::make_unique<StorageMembers>();
  }
  auto& members = (*storage_members_)[interval.binding];
  auto insert = std::ranges::lower_bound(members, group->component_offset, {},
                                         &ValueProjection::component_offset);
  if (insert == members.end() ||
      insert->component_offset != group->component_offset) {
    members.insert(insert, *group);
    changed = true;
  }
  return true;
}

bool ControlFlow::result_requires_storage() const {
  return object_identity_ && object_identity_->requires_storage(function_);
}

void ControlFlow::retain_owned_binding(cxx::Symbol* binding,
                                       cxx::ExpressionAST* initializer) {
  auto* type = source_object_type(binding);
  auto traits = unit_.typeTraits();
  if (!traits.is_class(type) && !(object_identity_ && traits.is_array(type) &&
                                  types_.requires_value_transport(type))) {
    return;
  }
  if (!object_identity_) {
    object_identity_ = std::make_unique<ObjectIdentity>(unit_, types_);
  }
  object_identity_->bind(binding, initializer, addressed(binding));
}

bool ControlFlow::reference_requires_storage(cxx::Symbol* binding) const {
  if (auto found = reference_aliases_.find(binding);
      found != reference_aliases_.end()) {
    return argument_requires_storage(found->second.identity);
  }
  return storage_references_.contains(binding);
}

cxx::ExpressionAST* ControlFlow::reference_materialization(
    cxx::Symbol* binding) const {
  auto found = reference_aliases_.find(binding);
  return found == reference_aliases_.end() ? nullptr
                                           : found->second.materialization;
}

bool ControlFlow::indirect_parameter_written(cxx::Symbol* parameter) const {
  return written_indirect_parameters_.contains(parameter);
}

const cxx::Type* ControlFlow::borrowed_pointer_object_type(
    cxx::Symbol* parameter) const {
  auto found = pointer_aliases_.find(parameter);
  return found == pointer_aliases_.end() ? nullptr
                                         : found->second.parameter_object_type;
}

bool ControlFlow::borrowed_pointer_requires_storage(
    cxx::Symbol* parameter) const {
  auto found = pointer_aliases_.find(parameter);
  return found != pointer_aliases_.end() &&
         found->second.parameter_object_type &&
         (!found->second.transparent ||
          storage_borrowed_pointers_.contains(parameter));
}

const cxx::Type* ControlFlow::candidate_pointer_object_type(
    cxx::Symbol* parameter) {
  auto traits = unit_.typeTraits();
  auto* pointer =
      cxx::type_cast<cxx::PointerType>(types_.unqualified(parameter->type()));
  if (!pointer) {
    return nullptr;
  }
  auto* object_type = pointer->elementType();
  auto* unqualified = types_.unqualified(object_type);
  if (traits.is_volatile(object_type) || traits.is_array(unqualified) ||
      !traits.is_class(unqualified) || !traits.is_complete(unqualified) ||
      !types_.requires_value_transport(unqualified)) {
    return nullptr;
  }
  return object_type;
}

std::span<const size_t> ControlFlow::projected_reference_return_indices(
    cxx::ExpressionAST* expression) const {
  if (!projected_reference_return_facts_) {
    return {};
  }
  expression = unwrap_object_identity(unit_, expression);
  auto found = projected_reference_return_facts_->selections.find(expression);
  if (found != projected_reference_return_facts_->selections.end()) {
    return found->second.origins;
  }
  if (projected_reference_return_facts_->complete_receiver_index != SIZE_MAX &&
      is_current_object(unit_, expression)) {
    return {&projected_reference_return_facts_->complete_receiver_index, 1};
  }
  auto returned = destination(expression);
  if (!returned || returned->member || returned->component_offset != 0) {
    return {};
  }
  auto parameter =
      projected_reference_return_facts_->complete_parameter_indices.find(
          returned->binding);
  return parameter == projected_reference_return_facts_
                          ->complete_parameter_indices.end()
             ? std::span<const size_t>{}
             : std::span<const size_t>{&parameter->second, 1};
}

std::pair<size_t, bool> ControlFlow::retain_projected_reference_origin(
    ProjectedReferenceReturnFacts& facts, ReferenceResultOrigin origin) {
  size_t retained_index = SIZE_MAX;
  bool inserted = false;
  if (facts.origin_indices.empty()) {
    auto found = std::ranges::find(facts.origins, origin);
    if (found != facts.origins.end()) {
      return {static_cast<size_t>(found - facts.origins.begin()), false};
    }
    if (facts.origins.size() < kLinearCanonicalizationThreshold) {
      retained_index = facts.origins.size();
      inserted = true;
    } else {
      facts.origin_indices.reserve(
          std::max(facts.origins.size() + 1, facts.origins.capacity()));
      for (size_t index = 0; index < facts.origins.size(); ++index) {
        facts.origin_indices.emplace(facts.origins[index], index);
      }
    }
  }
  if (retained_index == SIZE_MAX) {
    auto retained =
        facts.origin_indices.try_emplace(origin, facts.origins.size());
    retained_index = retained.first->second;
    inserted = retained.second;
  }
  if (inserted) {
    facts.origins.push_back(std::move(origin));
    const auto& retained = facts.origins.back();
    if (facts.origins.size() > 1 &&
        retained.projection.type != facts.origins.front().projection.type &&
        unit_.typeTraits().is_pointer(
            cxx::type_cast<cxx::FunctionType>(function_->type())
                ->returnType())) {
      promote_projected_reference_storage(facts);
    }
    if (retained.owner_ordinal == ReferenceResultOrigin::kReceiver) {
      facts.only_whole_parameters = false;
      const bool partial = retained.projection.component_offset != 0 ||
                           retained.projection.partition != receiver_partition_;
      facts.has_partial_receiver_origin |= partial;
      if (facts.requires_storage) {
        receiver_requires_storage_ = true;
      }
    } else {
      auto parameters = function_->parameters();
      auto* parameter = parameters[retained.owner_ordinal];
      if (facts.parameter_has_origin.empty()) {
        facts.parameter_has_origin.resize(parameters.size());
      }
      facts.parameter_has_origin[retained.owner_ordinal] = true;
      auto* type = types_.unqualified_value(parameter->type());
      const bool partial =
          retained.projection.component_offset != 0 ||
          retained.projection.partition !=
              &types_.partition(type, function_->declaration());
      if (partial) {
        facts.only_whole_parameters = false;
        facts.partial_parameter_origins.insert(parameter);
      }
      if (facts.requires_storage) {
        storage_references_.insert(parameter);
      } else if (partial && reference_uses_storage_abi(parameter)) {
        promote_projected_reference_storage(facts);
      }
    }
  }
  return {retained_index, inserted};
}

bool ControlFlow::promote_projected_reference_storage(
    ProjectedReferenceReturnFacts& facts) {
  if (facts.requires_storage) {
    return false;
  }
  facts.requires_storage = true;
  auto parameters = function_->parameters();
  for (const auto& origin : facts.origins) {
    if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
      receiver_requires_storage_ = true;
    } else {
      storage_references_.insert(parameters[origin.owner_ordinal]);
    }
  }
  for (const auto& [expression, selection] : facts.selections) {
    if (selection.propagated_origin_count != SIZE_MAX) {
      identity_call_results_.insert(expression);
    }
  }
  for (const auto& [call, results] : facts.array_results) {
    identity_call_results_.insert(call);
  }
  return true;
}

bool ControlFlow::reference_uses_storage_abi(cxx::Symbol* parameter) const {
  auto traits = unit_.typeTraits();
  auto* object_type = traits.remove_reference(parameter->type());
  return reference_requires_storage(parameter) ||
         traits.is_volatile(object_type) || traits.is_array(object_type);
}

std::optional<std::pair<size_t, size_t>> ControlFlow::projected_reference_owner(
    ProjectedReferenceReturnFacts& facts, cxx::Symbol* binding) {
  auto parameters = function_->parameters();
  if (parameters.size() <= kLinearCanonicalizationThreshold) {
    auto parameter = std::ranges::find(parameters, binding);
    if (parameter != parameters.end() &&
        unit_.typeTraits().is_reference((*parameter)->type())) {
      return std::pair{static_cast<size_t>(parameter - parameters.begin()),
                       size_t{0}};
    }
  } else {
    if (!facts.parameter_ordinals_indexed) {
      facts.parameter_ordinals.reserve(parameters.size());
      for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
        if (unit_.typeTraits().is_reference(parameters[ordinal]->type())) {
          facts.parameter_ordinals.emplace(parameters[ordinal], ordinal);
        }
      }
      facts.parameter_ordinals_indexed = true;
    }
    if (auto parameter = facts.parameter_ordinals.find(binding);
        parameter != facts.parameter_ordinals.end()) {
      return std::pair{parameter->second, size_t{0}};
    }
  }

  if (!receiver_partition_) {
    return std::nullopt;
  }
  const auto& members = receiver_partition_->members;
  if (members.size() <= kLinearCanonicalizationThreshold) {
    auto receiver =
        std::ranges::find(members, binding, &SubobjectPartition::source);
    if (receiver == members.end()) {
      return std::nullopt;
    }
    return std::pair{ReferenceResultOrigin::kReceiver,
                     receiver->component_offset};
  }
  if (!facts.receiver_offsets_indexed) {
    facts.receiver_offsets.reserve(members.size());
    for (const auto& member : members) {
      facts.receiver_offsets.emplace(member.source, member.component_offset);
    }
    facts.receiver_offsets_indexed = true;
  }
  if (auto receiver = facts.receiver_offsets.find(binding);
      receiver != facts.receiver_offsets.end()) {
    return std::pair{ReferenceResultOrigin::kReceiver, receiver->second};
  }
  return std::nullopt;
}

void ControlFlow::merge_projected_reference_returns(
    ProjectedReferenceReturnFacts& target,
    ProjectedReferenceReturnFacts source) {
  std::vector<size_t> remapped_indices;
  remapped_indices.reserve(source.origins.size());
  for (auto& origin : source.origins) {
    remapped_indices.push_back(
        retain_projected_reference_origin(target, std::move(origin)).first);
  }
  for (auto& [expression, selection] : source.selections) {
    for (size_t& index : selection.origins) {
      index = remapped_indices[index];
    }
    const bool propagated = selection.propagated_origin_count != SIZE_MAX;
    target.selections.emplace(expression, std::move(selection));
    if (target.requires_storage && propagated) {
      identity_call_results_.insert(expression);
    }
  }
  for (auto& [call, results] : source.array_results) {
    auto& target_results = target.array_results[call];
    target_results.insert(target_results.end(), results.begin(), results.end());
    if (target.requires_storage) {
      identity_call_results_.insert(call);
    }
  }
  for (const auto& [parameter, index] : source.complete_parameter_indices) {
    target.complete_parameter_indices.insert_or_assign(parameter,
                                                       remapped_indices[index]);
  }
  if (source.complete_receiver_index != SIZE_MAX) {
    target.complete_receiver_index =
        remapped_indices[source.complete_receiver_index];
  }
  if (source.requires_storage) {
    promote_projected_reference_storage(target);
  }
}

void ControlFlow::retain_complete_parameter_reference_origins(
    ProjectedReferenceReturnFacts& facts, std::span<const size_t> ordinals) {
  auto parameters = function_->parameters();
  for (size_t ordinal : ordinals) {
    auto* type = types_.unqualified_value(parameters[ordinal]->type());
    const size_t index =
        retain_projected_reference_origin(
            facts,
            {ordinal,
             {type, &types_.partition(type, function_->declaration()), 0}})
            .first;
    facts.complete_parameter_indices.insert_or_assign(parameters[ordinal],
                                                      index);
  }
}

void ControlFlow::retain_complete_receiver_reference_origin(
    ProjectedReferenceReturnFacts& facts) {
  auto* object_type = implicit_object_type(unit_, function_);
  auto* type = object_type ? types_.unqualified(object_type) : nullptr;
  const auto* record =
      type ? types_.record(type, function_->declaration()) : nullptr;
  IREE_ASSERT(record);
  facts.complete_receiver_index =
      retain_projected_reference_origin(
          facts, {ReferenceResultOrigin::kReceiver, {type, record, 0}})
          .first;
}

bool ControlFlow::argument_requires_storage(const CallArgument& argument,
                                            StorageDemand demand) const {
  if (object_requires_storage(argument.source) || argument.inherent_storage ||
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
    if (auto found = pointer_aliases_.find(binding);
        found != pointer_aliases_.end() &&
        found->second.parameter_object_type) {
      return storage_borrowed_pointers_.contains(binding);
    }
    auto* source_type = source_object_type(binding);
    if (traits.is_reference(source_type)) {
      auto* object_type = traits.remove_reference(source_type);
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(binding);
      return (field && !types_.reference_is_snapshot(field)) ||
             storage_references_.contains(binding) ||
             traits.is_volatile(object_type) || traits.is_array(object_type);
    }
    if (is_direct_subobject(binding)) {
      return receiver_requires_storage_;
    }
    if (demand == StorageDemand::Identity && !addressed(binding)) {
      auto* type = types_.unqualified_value(source_object_type(binding));
      if (traits.is_scalar(type) || types_.native_vector(type)) {
        return false;
      }
    }
    return storage_backed(binding);
  };
  if (std::ranges::any_of(argument.identity_bindings, requires_storage)) {
    return true;
  }
  if (argument.identity_intervals &&
      std::ranges::any_of(
          *argument.identity_intervals, [&](const ObjectInterval& interval) {
            for (const auto& home : storage_members(interval.binding)) {
              if (interval.begin >= home.component_offset &&
                  interval.end <=
                      home.component_offset + home.partition->component_count) {
                return true;
              }
            }
            return requires_storage(interval.binding);
          })) {
    return true;
  }
  return false;
}

bool ControlFlow::require_reference_storage(cxx::Symbol* parameter) {
  bool changed = storage_references_.insert(parameter).second;
  if (changed && projected_reference_return_facts_ &&
      projected_reference_return_facts_->partial_parameter_origins.contains(
          parameter)) {
    changed = promote_projected_reference_storage(
                  *projected_reference_return_facts_) ||
              changed;
  }
  return changed;
}

bool ControlFlow::require_borrowed_pointer_storage(cxx::Symbol* parameter) {
  return storage_borrowed_pointers_.insert(parameter).second;
}

bool ControlFlow::storage_backed(cxx::MemberExpressionAST* expression) const {
  if (expression->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
    auto* base = cxx::ast_cast<cxx::IdExpressionAST>(
        unwrap_expression(expression->baseExpression));
    if (base) {
      auto found = pointer_aliases_.find(base->symbol);
      if (found != pointer_aliases_.end() && found->second.transparent &&
          !argument_requires_storage(found->second.identity)) {
        return false;
      }
    }
  }
  auto* owner = object_owner(expression);
  return storage_expressions_.contains(expression) ||
         object_requires_storage(expression) ||
         (owner && storage_backed(owner));
}

cxx::ExpressionAST* ControlFlow::pointer_alias(cxx::Symbol* binding) const {
  auto found = pointer_aliases_.find(binding);
  return found != pointer_aliases_.end() && found->second.transparent &&
                 found->second.referent &&
                 (!found->second.array_origin ||
                  borrowed_array_type(found->second.array_origin)) &&
                 !argument_requires_storage(found->second.identity)
             ? found->second.referent
             : nullptr;
}

cxx::ExpressionAST* ControlFlow::array_borrow_origin(
    cxx::ExpressionAST* expression) const {
  auto* source = unwrap_expression(expression);
  if (unit_.typeTraits().is_array(source->type)) {
    return source;
  }
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(source)) {
    auto found = pointer_aliases_.find(id->symbol);
    return found != pointer_aliases_.end() && found->second.transparent
               ? found->second.array_origin
               : nullptr;
  }
  return pointer_returning_call(source) ? source : nullptr;
}

const cxx::Type* ControlFlow::borrowed_array_type(
    cxx::ExpressionAST* expression) const {
  auto* source = array_borrow_origin(expression);
  if (!source) {
    return nullptr;
  }
  if (unit_.typeTraits().is_array(source->type)) {
    return source->type;
  }
  if (!reference_identity_facts_ ||
      reference_identity_facts_->storage_results.contains(source)) {
    return nullptr;
  }
  auto found = reference_identity_facts_->results.find(source);
  return found == reference_identity_facts_->results.end()
             ? nullptr
             : found->second.array_type;
}

CallArgument ControlFlow::retain_array_identity(
    cxx::ExpressionAST* expression) {
  if (auto* id =
          cxx::ast_cast<cxx::IdExpressionAST>(unwrap_expression(expression))) {
    auto found = pointer_aliases_.find(id->symbol);
    if (found != pointer_aliases_.end() && found->second.transparent &&
        found->second.array_origin) {
      return found->second.identity;
    }
  }
  return retain_call_argument(expression);
}

void ControlFlow::require_pointer_result_storage() {
  auto* signature = cxx::type_cast<cxx::FunctionType>(function_->type());
  if (unit_.typeTraits().is_pointer(signature->returnType()) &&
      projected_reference_return_facts_) {
    promote_projected_reference_storage(*projected_reference_return_facts_);
  }
}

cxx::Symbol* ControlFlow::object_owner(cxx::ExpressionAST* expression) const {
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    return id->subobjectPath ? id->subobjectPath->value : id->symbol;
  }
  auto found = object_owners_.find(expression);
  return found == object_owners_.end() ? nullptr : found->second;
}

void ControlFlow::retain_counted(cxx::ForStatementAST* loop) {
  auto interval = CountedLoop::recognize(unit_, types_, loop);
  if (!interval) {
    return;
  }
  const Destination* referent = nullptr;
  if (auto* bound = interval->upper_identifier()) {
    if (auto alias = reference_aliases_.find(bound->symbol);
        alias != reference_aliases_.end()) {
      if (!alias->second.destination) {
        return;
      }
      referent = &*alias->second.destination;
    } else if (bound->subobjectPath) {
      if (!retain_destination(bound)) {
        return;
      }
      referent = &destinations_.at(bound);
    }
  }
  has_projected_loop_bounds_ |= referent && referent->member;
  counted_.emplace(loop, LoopCandidate{*interval, referent});
}

const CountedLoop* ControlFlow::counted(cxx::ForStatementAST* loop) const {
  auto found = counted_.find(loop);
  if (found == counted_.end()) {
    return nullptr;
  }
  const auto& [interval, referent] = found->second;
  auto body_writes = written(loop->statement);
  if (addressed(interval.induction) ||
      std::ranges::find(body_writes, interval.induction) != body_writes.end()) {
    return nullptr;
  }
  auto* identifier = interval.upper_identifier();
  if (!identifier) {
    return &interval;
  }
  auto* binding = identifier->symbol;
  auto* owner = referent ? referent->binding : binding;
  // A storage-backed reference may alias an entirely different parameter.
  // Only the completed callable analysis knows which arguments share storage.
  if (addressed(owner) ||
      ((referent || unit_.typeTraits().is_reference(binding->type())) &&
       reference_requires_storage(binding))) {
    return nullptr;
  }
  if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(owner);
      field &&
      (field->isStatic()
           ? !unit_.typeTraits().is_const(field->type())
           : receiver_requires_storage_ || is_captured_object_field(field) ||
                 (unit_.typeTraits().is_reference(field->type()) &&
                  !types_.reference_is_snapshot(field)))) {
    return nullptr;
  }
  if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(owner);
      variable && !unit_.typeTraits().is_const(variable->type()) &&
      has_nonlocal_storage(variable)) {
    return nullptr;
  }
  // Member writes were checked by component as they were recorded. For a
  // whole object the ordinary region write summary is already exact.
  if (!referent || !referent->member) {
    auto writes = written(loop);
    if (std::ranges::find(writes, owner) != writes.end()) {
      return nullptr;
    }
  }
  return &interval;
}

const SwitchLayout& ControlFlow::switch_layout(
    cxx::SwitchStatementAST* statement) const {
  return switches_.at(statement);
}

StatementExits ControlFlow::exits(cxx::StatementAST* statement) const {
  auto found = exits_.find(statement);
  return found == exits_.end() ? StatementExits{} : found->second;
}

void ControlFlow::retain_exits(cxx::StatementAST* statement,
                               StatementExits paths) {
  if (paths != StatementExits{}) {
    exits_.emplace(statement, paths);
  }
}

cxx::CompoundStatementAST* ControlFlow::range_scope(
    cxx::ForRangeStatementAST* loop) const {
  return range_loops_->scope(loop);
}

bool ControlFlow::initializes_range(cxx::InitDeclaratorAST* declaration) const {
  return range_loops_ && range_loops_->initializes_range(declaration);
}

const InitializerList* ControlFlow::initializer_list(
    cxx::BracedInitListAST* source) const {
  return initializer_lists_ ? initializer_lists_->get(source) : nullptr;
}

bool ControlFlow::preVisit(cxx::AST* ast) {
  switch (ast->kind()) {
    case cxx::ASTKind::ConstExpression: {
      auto* constant = cxx::ast_cast<cxx::ConstExpressionAST>(ast);
      return constant->constValue &&
             constant_contains_address(*constant->constValue);
    }
    case cxx::ASTKind::SizeofExpression:
    case cxx::ASTKind::SizeofTypeExpression:
    case cxx::ASTKind::AlignofTypeExpression:
    case cxx::ASTKind::NoexceptExpression:
    case cxx::ASTKind::DecltypeSpecifier:
    case cxx::ASTKind::RequiresExpression:
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
    if (!object_identity_ && expression->type &&
        expression->valueCategory == cxx::ValueCategory::kPrValue &&
        unit_.typeTraits().is_class(expression->type)) {
      object_identity_ = std::make_unique<ObjectIdentity>(unit_, types_);
    }
    if (object_identity_) {
      object_identity_->retain(expression);
    }
    auto* list = cxx::ast_cast<cxx::BracedInitListAST>(expression);
    auto* initialized = list ? initializer_list(list) : nullptr;
    if (initialized) {
      temporary_origins_.retain_initializer_list(
          list, initialized->backing, retain_reference_materializations_);
    } else {
      temporary_origins_.retain(expression, retain_reference_materializations_);
    }
    cxx::Symbol* owner = nullptr;
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression)) {
      record_member_binding_read(expression);
      if (member->symbol == initializing_binding_ &&
          is_current_object(unit_, member->baseExpression) &&
          (unit_.typeTraits().is_array(member->type) ||
           unit_.typeTraits().is_class(member->type))) {
        require_identity(member);
      }
      if (observes_reference_captures_ && expression != capture_value_read_) {
        retain_reference_capture_identity(member->symbol);
      }
      if (member->accessOp == cxx::TokenKind::T_DOT) {
        owner = object_owner(member->baseExpression);
      }
    } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
      record_member_binding_read(expression);
      if (id->symbol == initializing_binding_ &&
          (unit_.typeTraits().is_array(id->type) ||
           unit_.typeTraits().is_class(id->type))) {
        require_identity(id);
      }
      if (!pointer_aliases_.empty()) {
        if (auto found = pointer_aliases_.find(id->symbol);
            found != pointer_aliases_.end() &&
            pointer_alias_use_ != id->symbol &&
            borrowed_pointer_expression_ != id) {
          found->second.transparent = false;
        }
      }
      if (observes_reference_captures_ && expression != capture_value_read_) {
        retain_reference_capture_identity(id->symbol);
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
    } else if (auto* binary =
                   cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
               binary && !binary->symbol &&
               binary->op == cxx::TokenKind::T_COMMA) {
      owner = object_owner(binary->rightExpression);
    } else if (auto* selection = array_selection(expression)) {
      owner = object_owner(selection->array);
    }
    if (owner) {
      object_owners_.emplace(expression, owner);
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
    // A switch records its combined exits at the end of its last clause.
    StatementExits paths;
    if (statement->kind() == cxx::ASTKind::SwitchStatement) {
      paths = exits(statement);
    } else {
      paths = classify_exits(statement);
      retain_exits(statement, paths);
    }
    // Schedule and strand bodies complete at their own region boundary.
    // Reject escaping source exits before return/loop lowering can flatten
    // the attributed compound into its enclosing control flow.
    if (paths.escapes_region()) {
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
                std::string(region) + ((paths.returns() != ExitFlow::None)
                                           ? " cannot contain function returns"
                                           : " cannot exit an enclosing loop"));
          });
    }
  }
  if (!owners_.empty() && owners_.back() == ast) {
    construction_.leave(ast);
    owners_.pop_back();
  }
  switch (ast->kind()) {
    case cxx::ASTKind::ReturnStatement:
      construction_.return_from_function();
      construction_.end_path();
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

StatementExits ControlFlow::classify_exits(cxx::StatementAST* statement) const {
  using Kind = cxx::ASTKind;
  using LoopTest = StatementExits::LoopTest;
  switch (statement->kind()) {
    case Kind::ExpressionStatement:
      // Construction already joined evaluated conditional and short-circuit
      // paths.
      return construction_.state().reachable
                 ? StatementExits{}
                 : StatementExits::terminal(StatementExits::Diverge);
    case Kind::ReturnStatement:
      return StatementExits::terminal(StatementExits::Return);
    case Kind::ContinueStatement:
      return StatementExits::terminal(StatementExits::Continue);
    case Kind::BreakStatement:
      return StatementExits::terminal(StatementExits::Break);
    case Kind::CompoundStatement: {
      StatementExits result;
      if (!exits_.empty()) {
        auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(statement);
        for (auto* child : cxx::ListView{compound->statementList}) {
          result.append(exits(child));
        }
      }
      return result;
    }
    case Kind::IfStatement: {
      auto* branch = cxx::ast_cast<cxx::IfStatementAST>(statement);
      if (branch->constexprValue.has_value()) {
        return exits(*branch->constexprValue ? branch->statement
                                             : branch->elseStatement);
      }
      return StatementExits::branch(exits(branch->statement),
                                    exits(branch->elseStatement));
    }
    case Kind::ForRangeStatement:
      return exits(
          range_scope(cxx::ast_cast<cxx::ForRangeStatementAST>(statement)));
    case Kind::WhileStatement: {
      auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(statement);
      return exits(loop->statement)
          .leave_loop(LoopTest::BeforeBody,
                      constant_truth(unit_, loop->condition).value_or(false));
    }
    case Kind::ForStatement: {
      auto* loop = cxx::ast_cast<cxx::ForStatementAST>(statement);
      return exits(loop->statement)
          .leave_loop(LoopTest::BeforeBody,
                      constant_truth(unit_, loop->condition).value_or(false));
    }
    case Kind::DoStatement: {
      auto* loop = cxx::ast_cast<cxx::DoStatementAST>(statement);
      auto body = exits(loop->statement);
      return body.leave_loop(
          LoopTest::AfterBody,
          body.reaches_latch() &&
              constant_truth(unit_, loop->expression).value_or(false));
    }
    default:
      return {};
  }
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

  SwitchLayout layout;
  SwitchExits switch_exits;
  StatementExits clause_exits;
  auto incoming = construction_.state();
  bool clause_owner_active = false;
  auto finish_clause = [&](bool has_successor) {
    if (!clause_owner_active) {
      return;
    }
    auto& clause = layout.clauses.back();
    auto* owner = clause.labels.front().source;
    auto found = writes_.find(owner);
    if (found != writes_.end()) {
      clause.writes = std::move(found->second);
      writes_.erase(found);
    }
    clause.falls_through = clause_exits.falls_through();
    if (clause.labels.size() != 1 || (has_successor && clause.falls_through)) {
      layout.direct = false;
    }
    switch_exits.append_clause(clause_exits);
    clause_exits = {};
    construction_.leave(owner);
    owners_.pop_back();
    clause_owner_active = false;
  };

  for (auto* child : cxx::ListView{ast->statementList}) {
    auto* case_label = cxx::ast_cast<cxx::CaseStatementAST>(child);
    auto* default_label = cxx::ast_cast<cxx::DefaultStatementAST>(child);
    if (case_label || default_label) {
      bool starts_clause =
          layout.clauses.empty() || !layout.clauses.back().statements.empty();
      if (starts_clause) {
        finish_clause(/*has_successor=*/true);
        construction_.restore(
            construction_.join(construction_.state(), incoming));
        layout.clauses.push_back({});
      }
      active_switch_label_ = child;
      accept(child);
      active_switch_label_ = nullptr;
      auto& clause = layout.clauses.back();
      std::optional<std::int64_t> key;
      if (case_label) {
        key = case_label->caseValue;
      }
      clause.labels.push_back({child, key});
      if (default_label) {
        layout.has_default = true;
      }
      if (starts_clause) {
        owners_.push_back(child);
        construction_.enter(/*declaration_scope=*/false);
        clause_owner_active = true;
      }
      continue;
    }
    if (layout.clauses.empty()) {
      diagnostics_.reject(
          unit_, child,
          "switch body statements require a preceding case or default label");
    }
    layout.clauses.back().statements.push_back(child);
    accept(child);
    clause_exits.append(exits(child));
  }
  finish_clause(/*has_successor=*/false);
  auto* selection = switch_stack_.back();
  retain_exits(selection, switch_exits.finish(layout.has_default));
  switches_.emplace(selection, std::move(layout));
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
        construction_.join(std::move(then_state), construction_.state()));
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
  retain_counted(ast);
  auto condition_exit = construction_.state();
  accept(ast->statement);
  construction_.reach_latch();
  accept(ast->expression);
  construction_.finish_loop(std::move(condition_exit));
}

void ControlFlow::visit(cxx::ForRangeStatementAST* ast) {
  if (!range_loops_) {
    range_loops_ = std::make_unique<RangeLoops>();
  }
  accept(range_loops_->expand(ast));
}

void ControlFlow::visit(cxx::ConditionalExpressionAST* ast) {
  accept(ast->condition);
  auto incoming = construction_.state();
  accept(ast->iftrueExpression);
  auto then_state = construction_.state();
  construction_.restore(std::move(incoming));
  accept(ast->iffalseExpression);
  construction_.restore(
      construction_.join(std::move(then_state), construction_.state()));
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
  if (switches_.at(ast).has_default) {
    incoming = {false, {}};
  }
  construction_.finish_switch(std::move(incoming));
  switch_stack_.pop_back();
}

void ControlFlow::visit(cxx::AssignmentExpressionAST* ast) {
  const bool ordinary_call =
      ast->symbol && !types_.is_trivial_value_operation(ast->symbol);
  // Assignment evaluates its right operand before the destination, including
  // overloaded operator syntax. Field initialization depends on that order.
  accept(ast->rightExpression);
  if (!ordinary_call) {
    record_parameter_read(ast->rightExpression);
  }
  accept(ast->leftExpression);
  auto* subscript =
      cxx::ast_cast<cxx::SubscriptExpressionAST>(ast->leftExpression);
  const bool view_element =
      view_element_origin(unit_, ast->leftExpression) != nullptr;
  const auto* selection =
      array_selection(unwrap_object_identity(unit_, ast->leftExpression));
  if (!ordinary_call &&
      (((selection && !selection->constant_index) ||
        (subscript && !selection && !view_element)) ||
       (cxx::ast_cast<cxx::MemberExpressionAST>(ast->leftExpression) &&
        cxx::ast_cast<cxx::MemberExpressionAST>(ast->leftExpression)
                ->vectorComponentKind != cxx::VectorComponentKind::kNone))) {
    record_parameter_read(ast->leftExpression);
  }
  if (!ordinary_call) {
    if (!record(ast->leftExpression, WriteEffect::Callable)) {
      require_identity(ast->leftExpression);
    }
  }
  if (ordinary_call) {
    retain_call(*direct_operator_call(ast), ast);
  } else {
    record_parameter_assignment(ast->leftExpression);
  }
}

void ControlFlow::visit(cxx::BinaryExpressionAST* ast) {
  if (!ast->symbol && (ast->op == cxx::TokenKind::T_AMP_AMP ||
                       ast->op == cxx::TokenKind::T_BAR_BAR)) {
    accept(ast->leftExpression);
    auto skipped = construction_.state();
    accept(ast->rightExpression);
    construction_.restore(
        construction_.join(std::move(skipped), construction_.state()));
  } else {
    cxx::ASTVisitor::visit(ast);
  }
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
  }
}

void ControlFlow::visit(cxx::CompoundAssignmentExpressionAST* ast) {
  accept(ast->rightExpression);
  accept(ast->targetExpression);
  accept(ast->leftExpression);
  accept(ast->adjustExpression);
  if (!ast->symbol) {
    record_parameter_read(ast->targetExpression);
    record_parameter_assignment(ast->targetExpression);
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
    record_parameter_read(ast->baseExpression);
    record_parameter_assignment(ast->baseExpression);
    if (!record(ast->baseExpression, WriteEffect::Callable)) {
      require_identity(ast->baseExpression);
    }
  }
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
  }
}

void ControlFlow::visit(cxx::UnaryExpressionAST* ast) {
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP &&
      ast != borrowed_pointer_expression_ && ast != borrowed_call_address_) {
    auto* dereference = cxx::ast_cast<cxx::UnaryExpressionAST>(
        unwrap_expression(ast->expression));
    if (dereference && !dereference->symbol &&
        dereference->op == cxx::TokenKind::T_STAR) {
      // &* observes the pointer, not its pointee. In particular, an opaque
      // pointee needs no array partition or source-storage representation.
      accept(dereference->expression);
      return;
    }
  }
  if (!ast->symbol && ast->op == cxx::TokenKind::T_STAR) {
    auto* source = unwrap_expression(ast->expression);
    ArraySelection selection{ast->expression, nullptr, nullptr, true, false, 0};
    if (auto* arithmetic = cxx::ast_cast<cxx::BinaryExpressionAST>(source);
        arithmetic && !arithmetic->symbol &&
        unit_.typeTraits().is_pointer(arithmetic->type) &&
        (arithmetic->op == cxx::TokenKind::T_PLUS ||
         arithmetic->op == cxx::TokenKind::T_MINUS)) {
      selection.array_first =
          unit_.typeTraits().is_pointer(arithmetic->leftExpression->type);
      selection.array = selection.array_first ? arithmetic->leftExpression
                                              : arithmetic->rightExpression;
      selection.index = selection.array_first ? arithmetic->rightExpression
                                              : arithmetic->leftExpression;
      selection.subtract = arithmetic->op == cxx::TokenKind::T_MINUS;
    }
    if (array_borrow_origin(selection.array)) {
      selection.type = borrowed_array_type(selection.array);
      analyze_array_selection(ast, selection);
      return;
    }
  }
  auto* alias_binding = [&]() -> cxx::Symbol* {
    if (ast->symbol || ast->op != cxx::TokenKind::T_STAR) {
      return nullptr;
    }
    auto* source =
        cxx::ast_cast<cxx::IdExpressionAST>(unwrap_expression(ast->expression));
    return source && !pointer_aliases_.empty() &&
                   pointer_aliases_.contains(source->symbol)
               ? source->symbol
               : nullptr;
  }();
  auto* saved_alias_use = pointer_alias_use_;
  if (alias_binding) {
    pointer_alias_use_ = alias_binding;
  }
  cxx::ASTVisitor::visit(ast);
  pointer_alias_use_ = saved_alias_use;
  if (auto call = direct_operator_call(ast)) {
    retain_call(*call, ast);
    return;
  }
  if (!ast->symbol && ast->op == cxx::TokenKind::T_AMP) {
    if (ast != borrowed_pointer_expression_ && ast != borrowed_call_address_) {
      require_identity(ast->expression);
    }
  }
  if (ast->op == cxx::TokenKind::T_PLUS_PLUS ||
      ast->op == cxx::TokenKind::T_MINUS_MINUS) {
    record_parameter_read(ast->expression);
    record_parameter_assignment(ast->expression);
    if (!record(ast->expression, WriteEffect::Callable)) {
      require_identity(ast->expression);
    }
  }
}

void ControlFlow::visit(cxx::MemberExpressionAST* ast) {
  cxx::Symbol* alias_binding = nullptr;
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(ast->symbol);
  if (!pointer_aliases_.empty() &&
      ast->accessOp == cxx::TokenKind::T_MINUS_GREATER && field &&
      !unit_.typeTraits().is_array(field->type())) {
    if (auto* source = cxx::ast_cast<cxx::IdExpressionAST>(
            unwrap_expression(ast->baseExpression));
        source && pointer_aliases_.contains(source->symbol)) {
      alias_binding = source->symbol;
    }
  }
  auto* saved_alias_use = pointer_alias_use_;
  if (alias_binding) {
    pointer_alias_use_ = alias_binding;
  }
  cxx::ASTVisitor::visit(ast);
  pointer_alias_use_ = saved_alias_use;
}

bool ControlFlow::collect_reference_return_parameters(
    cxx::ExpressionAST* expression, std::vector<size_t>& ordinals,
    std::unordered_map<cxx::Symbol*, size_t>* parameter_indices) {
  expression = unwrap_object_identity(unit_, expression);
  if (auto* conditional =
          cxx::ast_cast<cxx::ConditionalExpressionAST>(expression);
      conditional &&
      expression->valueCategory != cxx::ValueCategory::kPrValue) {
    std::unordered_map<cxx::Symbol*, size_t> local_indices;
    if (!parameter_indices) {
      auto parameters = function_->parameters();
      if (parameters.size() > kLinearCanonicalizationThreshold) {
        local_indices.reserve(parameters.size());
        for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
          local_indices.emplace(parameters[ordinal], ordinal);
        }
        parameter_indices = &local_indices;
      }
    }
    const size_t ordinal_mark = ordinals.size();
    if (collect_reference_return_parameters(conditional->iftrueExpression,
                                            ordinals, parameter_indices) &&
        collect_reference_return_parameters(conditional->iffalseExpression,
                                            ordinals, parameter_indices)) {
      return true;
    }
    ordinals.resize(ordinal_mark);
    return false;
  }

  auto returned = classify_destination(expression);
  if (!returned || returned->member || returned->component_offset != 0) {
    return false;
  }
  auto parameters = function_->parameters();
  size_t ordinal = SIZE_MAX;
  if (parameter_indices) {
    auto found = parameter_indices->find(returned->binding);
    if (found != parameter_indices->end()) {
      ordinal = found->second;
    }
  } else if (auto found = std::ranges::find(parameters, returned->binding);
             found != parameters.end()) {
    ordinal = static_cast<size_t>(found - parameters.begin());
  }
  if (ordinal == SIZE_MAX ||
      !unit_.typeTraits().is_reference(returned->binding->type())) {
    return false;
  }
  ordinals.push_back(ordinal);
  return true;
}

bool ControlFlow::collect_projected_reference_return(
    cxx::ExpressionAST* expression, ProjectedReferenceReturnFacts& facts) {
  expression = unwrap_object_identity(unit_, expression);
  if (auto* conditional =
          cxx::ast_cast<cxx::ConditionalExpressionAST>(expression);
      conditional &&
      expression->valueCategory != cxx::ValueCategory::kPrValue) {
    return collect_projected_reference_return(conditional->iftrueExpression,
                                              facts) &&
           collect_projected_reference_return(conditional->iffalseExpression,
                                              facts);
  }

  auto origin =
      [&](Destination destination, const cxx::Type* type,
          const Partition* partition,
          size_t component_offset) -> std::optional<ReferenceResultOrigin> {
    auto owner = projected_reference_owner(facts, destination.binding);
    if (!owner) {
      return std::nullopt;
    }
    return ReferenceResultOrigin{
        owner->first, {type, partition, owner->second + component_offset}};
  };
  auto append = [&](cxx::ExpressionAST* source, ReferenceResultOrigin value) {
    const size_t index =
        retain_projected_reference_origin(facts, std::move(value)).first;
    facts.selections[source].origins.push_back(index);
  };

  if (is_current_object(unit_, expression)) {
    auto* object_type = implicit_object_type(unit_, function_);
    const auto* type = object_type ? types_.unqualified(object_type) : nullptr;
    const auto* record = type ? types_.record(type, expression) : nullptr;
    if (!record) {
      return false;
    }
    append(expression, {ReferenceResultOrigin::kReceiver, {type, record, 0}});
    return true;
  }

  if (auto returned = classify_destination(expression)) {
    const auto* type = types_.unqualified_value(expression->type);
    const auto* partition = returned->member
                                ? returned->member
                                : &types_.partition(type, expression);
    auto value = origin(*returned, type, partition, returned->component_offset);
    if (value) {
      append(expression, *value);
      return true;
    }
  }

  if (view_element_origin(unit_, expression)) {
    return false;
  }

  auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(expression);
  auto traits = unit_.typeTraits();
  const auto* selection = array_selection(expression);
  if (!selection) {
    bool reference_call = false;
    if (subscript && subscript->symbol) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
      reference_call =
          signature && (traits.is_reference(signature->returnType()) ||
                        traits.is_pointer(signature->returnType()));
    } else if (auto call = direct_operator_call(expression)) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(call->function->type());
      reference_call =
          signature && (traits.is_reference(signature->returnType()) ||
                        traits.is_pointer(signature->returnType()));
    } else if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(expression)) {
      auto* function = direct_callee(call);
      auto* signature =
          function ? cxx::type_cast<cxx::FunctionType>(function->type())
                   : nullptr;
      reference_call =
          signature && (traits.is_reference(signature->returnType()) ||
                        traits.is_pointer(signature->returnType()));
    }
    if (reference_call) {
      auto selection = facts.selections.try_emplace(expression).first;
      selection->second.propagated_origin_count = 0;
      if (facts.requires_storage) {
        identity_call_results_.insert(expression);
      }
    }
    return reference_call;
  }
  auto* source = array_borrow_origin(selection->array);
  auto target = source ? classify_destination(source) : std::nullopt;
  auto identity = retain_array_identity(selection->array);
  if (argument_requires_storage(identity)) {
    return false;
  }
  if (!target) {
    if (identity.reference_calls.size() != 1 ||
        !identity.identity_bindings.empty() || identity.identity_intervals) {
      return false;
    }
    const auto& reference = identity.reference_calls.front();
    facts.array_results[reference.call].push_back(
        {expression, reference.projection});
    facts.selections.try_emplace(expression);
    if (facts.requires_storage) {
      identity_call_results_.insert(reference.call);
    }
    return true;
  }
  if (!selection->type) {
    return false;
  }
  const auto* array = types_.array(selection->type, expression);
  if (!array || !array->source->size()) {
    return false;
  }

  const size_t base_offset = target->component_offset;
  const size_t element_count = array->element->component_count;
  const auto* element_type = array->source->elementType();
  if (selection->constant_index) {
    auto value =
        origin(*target, element_type, array->element,
               base_offset + *selection->constant_index * element_count);
    if (value) {
      append(expression, *value);
    }
    return value.has_value();
  }
  facts.origins.reserve(facts.origins.size() + array->source->size());
  facts.selections[expression].origins.reserve(array->source->size());
  for (size_t index = 0; index < array->source->size(); ++index) {
    auto value = origin(*target, element_type, array->element,
                        base_offset + index * element_count);
    if (!value) {
      return false;
    }
    append(expression, *value);
  }
  return true;
}

void ControlFlow::visit(cxx::ReturnStatementAST* ast) {
  auto* signature = function_
                        ? cxx::type_cast<cxx::FunctionType>(function_->type())
                        : nullptr;
  const bool pointer_result =
      signature && unit_.typeTraits().is_pointer(signature->returnType());
  auto* returned_source = ast->expression;
  auto* saved_borrowed_expression = borrowed_pointer_expression_;
  if (pointer_result && returned_source) {
    returned_source = unwrap_expression(returned_source);
    if (unit_.typeTraits().is_array(returned_source->type) ||
        pointer_returning_call(returned_source)) {
      borrowed_pointer_expression_ = returned_source;
    }
  }
  cxx::ASTVisitor::visit(ast);
  borrowed_pointer_expression_ = saved_borrowed_expression;
  if (signature && ast->expression &&
      unit_.typeTraits().is_class(signature->returnType())) {
    if (!object_identity_) {
      object_identity_ = std::make_unique<ObjectIdentity>(unit_, types_);
    }
    object_identity_->returned(function_, ast->expression);
  }
  if (signature && !unit_.typeTraits().is_reference(signature->returnType()) &&
      ast->expression) {
    record_parameter_read(ast->expression);
  }
  if (!signature ||
      (!unit_.typeTraits().is_reference(signature->returnType()) &&
       !pointer_result) ||
      !ast->expression) {
    return;
  }

  if (reference_return_ == ReferenceReturn::Storage) {
    if (!pointer_result || unit_.typeTraits().is_array(returned_source->type) ||
        pointer_returning_call(returned_source)) {
      require_identity(returned_source);
    }
    return;
  }

  auto parameters = function_->parameters();
  const bool returns_current_receiver =
      !pointer_result && is_current_object(unit_, ast->expression);
  std::vector<size_t> parameter_ordinals;
  const bool returns_parameters =
      !pointer_result && !returns_current_receiver &&
      collect_reference_return_parameters(ast->expression, parameter_ordinals);
  ProjectedReferenceReturnFacts projected_origins;
  const bool returns_projection =
      !returns_current_receiver && !returns_parameters &&
      (!pointer_result || unit_.typeTraits().is_array(returned_source->type) ||
       pointer_returning_call(returned_source)) &&
      collect_projected_reference_return(returned_source, projected_origins);
  ReferenceReturn incoming = ReferenceReturn::Storage;
  size_t incoming_parameter = SIZE_MAX;
  if (returns_current_receiver) {
    incoming = ReferenceReturn::CurrentReceiver;
  } else if (returns_projection) {
    incoming = ReferenceReturn::Projected;
  } else if (returns_parameters) {
    incoming_parameter = parameter_ordinals.front();
    const bool common_parameter = std::ranges::all_of(
        parameter_ordinals,
        [&](size_t ordinal) { return ordinal == incoming_parameter; });
    if (common_parameter) {
      incoming = ReferenceReturn::Parameter;
    } else {
      incoming = ReferenceReturn::Projected;
      retain_complete_parameter_reference_origins(projected_origins,
                                                  parameter_ordinals);
    }
  }
  auto require_storage = [&](ReferenceReturn origin,
                             std::span<const size_t> ordinals,
                             const ProjectedReferenceReturnFacts* projected,
                             cxx::ExpressionAST* expression) {
    if (origin == ReferenceReturn::CurrentReceiver) {
      receiver_requires_storage_ = true;
    } else if (origin == ReferenceReturn::Projected) {
      if (!projected) {
        if (expression) {
          require_identity(expression);
        }
        return;
      }
      for (const auto& origin : projected->origins) {
        if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
          receiver_requires_storage_ = true;
        } else {
          require_reference_storage(parameters[origin.owner_ordinal]);
        }
      }
      for (const auto& [expression, selection] : projected->selections) {
        if (selection.propagated_origin_count != SIZE_MAX) {
          identity_call_results_.insert(expression);
        }
      }
      for (const auto& [call, results] : projected->array_results) {
        identity_call_results_.insert(call);
      }
    } else if (origin == ReferenceReturn::Parameter) {
      for (size_t ordinal : ordinals) {
        require_reference_storage(parameters[ordinal]);
      }
    } else if (origin == ReferenceReturn::Storage && expression) {
      auto* source =
          pointer_result ? unwrap_expression(expression) : expression;
      if (!pointer_result || unit_.typeTraits().is_array(source->type) ||
          pointer_returning_call(source)) {
        require_identity(source);
      }
    }
  };

  if (reference_return_ == ReferenceReturn::None) {
    reference_return_ = incoming;
    if (incoming == ReferenceReturn::Parameter) {
      reference_return_parameter_ = incoming_parameter;
    } else if (incoming == ReferenceReturn::Storage) {
      require_storage(incoming, {}, nullptr, ast->expression);
      projected_reference_return_facts_.reset();
    } else if (incoming == ReferenceReturn::Projected) {
      projected_reference_return_facts_ =
          std::make_unique<ProjectedReferenceReturnFacts>(
              std::move(projected_origins));
    }
    return;
  }
  if (reference_return_ == incoming) {
    if (incoming == ReferenceReturn::Parameter) {
      if (reference_return_parameter_ == incoming_parameter) {
        return;
      }
    } else if (incoming == ReferenceReturn::Projected) {
      merge_projected_reference_returns(*projected_reference_return_facts_,
                                        std::move(projected_origins));
      return;
    } else {
      return;
    }
  }

  auto value_return = [](ReferenceReturn kind) {
    return kind == ReferenceReturn::CurrentReceiver ||
           kind == ReferenceReturn::Parameter ||
           kind == ReferenceReturn::Projected;
  };
  if (value_return(reference_return_) && value_return(incoming)) {
    auto facts = reference_return_ == ReferenceReturn::Projected
                     ? std::move(projected_reference_return_facts_)
                     : std::make_unique<ProjectedReferenceReturnFacts>();
    if (reference_return_ == ReferenceReturn::CurrentReceiver) {
      retain_complete_receiver_reference_origin(*facts);
    } else if (reference_return_ == ReferenceReturn::Parameter) {
      retain_complete_parameter_reference_origins(
          *facts, reference_return_parameters());
    }
    if (incoming == ReferenceReturn::CurrentReceiver) {
      retain_complete_receiver_reference_origin(*facts);
    } else if (incoming == ReferenceReturn::Parameter) {
      retain_complete_parameter_reference_origins(*facts, parameter_ordinals);
    } else {
      merge_projected_reference_returns(*facts, std::move(projected_origins));
    }
    reference_return_ = ReferenceReturn::Projected;
    reference_return_parameter_ = SIZE_MAX;
    projected_reference_return_facts_ = std::move(facts);
    return;
  }

  require_storage(reference_return_, reference_return_parameters(),
                  projected_reference_return_facts_.get(), nullptr);
  require_storage(incoming, parameter_ordinals, &projected_origins,
                  ast->expression);
  reference_return_ = ReferenceReturn::Storage;
  projected_reference_return_facts_.reset();
}

void ControlFlow::visit(cxx::CallExpressionAST* ast) {
  if (ast->isVirtualDispatch) {
    diagnostics_.reject(unit_, ast, "virtual member dispatch is not admitted");
  }
  auto* function = direct_callee(ast);
  auto object_receiver =
      function ? imports_.object_method_receiver(function, ast) : std::nullopt;
  if (object_receiver) {
    accept(object_receiver->handle);
    for (auto* argument : cxx::ListView{ast->expressionList}) {
      accept(argument);
    }
    retain_object_method_call(ast, function, object_receiver->handle);
    return;
  }
  if (!function) {
    cxx::ASTVisitor::visit(ast);
  } else {
    auto* definition = function->resolvedDefinition();
    const bool has_definition =
        function->isDefined() || function->hasPendingBody() ||
        (definition &&
         (definition->isDefined() || definition->hasPendingBody()));
    // A resolved function designator has no runtime value to evaluate. Visit
    // only an explicit member receiver; template argument expressions belong
    // to frontend substitution and must not become runtime reads or storage.
    auto* designator = unwrap_expression(ast->baseExpression);
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(designator)) {
      auto* saved_read = capture_value_read_;
      if (has_definition) {
        // The retained call edge decides whether the concrete member function
        // needs receiver identity after all reached bodies are analyzed.
        capture_value_read_ = unwrap_expression(member->baseExpression);
      }
      accept(member->baseExpression);
      capture_value_read_ = saved_read;
    }
    auto parameters = function->parameters();
    size_t ordinal = 0;
    for (auto* argument : cxx::ListView{ast->expressionList}) {
      const cxx::Type* object_type =
          has_definition && ordinal < parameters.size()
              ? candidate_pointer_object_type(parameters[ordinal])
              : nullptr;
      auto* source = object_type ? unwrap_expression(argument) : nullptr;
      auto* address = cxx::ast_cast<cxx::UnaryExpressionAST>(source);
      if (!address || address->symbol || address->op != cxx::TokenKind::T_AMP ||
          types_.unqualified(address->expression->type) !=
              types_.unqualified(object_type)) {
        address = nullptr;
      }
      auto* forwarded = cxx::ast_cast<cxx::IdExpressionAST>(source);
      if (!forwarded || !pointer_aliases_.contains(forwarded->symbol)) {
        forwarded = nullptr;
      }
      auto* saved_address = borrowed_call_address_;
      auto* saved_alias_use = pointer_alias_use_;
      borrowed_call_address_ = address;
      if (forwarded) {
        pointer_alias_use_ = forwarded->symbol;
      }
      accept(argument);
      borrowed_call_address_ = saved_address;
      pointer_alias_use_ = saved_alias_use;
      ++ordinal;
    }
  }
  if (cxx::resolveBuiltinFunctionKind(
          cxx::ast_cast<cxx::IdExpressionAST>(ast->baseExpression)) ==
      cxx::BuiltinFunctionKind::T___BUILTIN_UNREACHABLE) {
    construction_.end_path();
    return;
  }
  if (!function) {
    record_memory_write();
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
      retain_constructor_call(field->constructor(), arguments, CallArgument{},
                              /*storage_receiver=*/false, capture);
    }
    if (unit_.typeTraits().is_lvalue_reference(field->type())) {
      auto* referent = reference_initializer(unit_, initializer);
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
      auto traits = unit_.typeTraits();
      auto* referent_type = traits.remove_reference(field->type());
      const bool pointer = traits.is_pointer(referent_type);
      const bool value_transport_required =
          !pointer && !traits.is_array(referent_type) &&
          !traits.is_volatile(referent_type) &&
          types_.requires_value_transport(referent_type);
      if (cxx::ast_cast<cxx::RefLambdaCaptureAST>(capture) &&
          direct_automatic &&
          ((pointer && !traits.is_volatile(referent_type)) ||
           value_transport_required)) {
        if (value_transport_required) {
          // The nested closure record must know its member partition before
          // its call operator is analyzed. Unsafe candidates fail after the
          // call graph reaches a fixed point; storage cannot represent them.
          types_.select_reference_snapshot(field);
        }
        reference_capture_facts().captures.push_back({
            .field = field,
            .referent = std::move(argument),
            .source = capture,
            .value_transport_required = value_transport_required,
        });
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
  if (has_op_binding(ast->symbol, "view.element")) {
    accept(ast->baseExpression);
    for (auto* index : cxx::ListView{ast->expressionList}) {
      accept(index);
    }
    return;
  }
  if (!ast->symbol) {
    auto traits = unit_.typeTraits();
    auto* index = subscript_index(ast);
    for (auto* operand : {ast->baseExpression, index}) {
      const auto* type = traits.remove_reference(operand->type);
      if (!traits.is_array(type) &&
          !(traits.is_pointer(type) && array_borrow_origin(operand))) {
        continue;
      }
      const bool array_first = operand == ast->baseExpression;
      analyze_array_selection(
          ast, {operand, array_first ? index : ast->baseExpression,
                traits.is_array(type) ? type : borrowed_array_type(operand),
                array_first, false, std::nullopt});
      return;
    }
  }
  cxx::ASTVisitor::visit(ast);
  if (auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(ast->symbol)) {
    retain_call(ast, function);
  }
}

void ControlFlow::analyze_array_selection(cxx::ExpressionAST* expression,
                                          ArraySelection selection) {
  auto analyze_array = [&] {
    auto* saved_alias_use = pointer_alias_use_;
    auto* saved_borrowed_expression = borrowed_pointer_expression_;
    borrowed_pointer_expression_ = unwrap_expression(selection.array);
    if (auto* id =
            cxx::ast_cast<cxx::IdExpressionAST>(borrowed_pointer_expression_)) {
      pointer_alias_use_ = id->symbol;
    }
    accept(selection.array);
    pointer_alias_use_ = saved_alias_use;
    borrowed_pointer_expression_ = saved_borrowed_expression;
  };
  if (selection.array_first) {
    analyze_array();
    accept(selection.index);
  } else {
    accept(selection.index);
    analyze_array();
  }
  // Visit conversions before selecting their array origin so lifetime facts
  // retain materializations. Translation borrows the array, not its decay.
  if (unit_.typeTraits().is_pointer(selection.array->type)) {
    selection.array = unwrap_expression(selection.array);
  }
  if (selection.index) {
    selection.constant_index.reset();
    const auto* type = cxx::type_cast<cxx::BoundedArrayType>(
        selection.type ? types_.unqualified(selection.type) : nullptr);
    if (type || !selection.type) {
      cxx::ASTInterpreter interpreter(&unit_);
      auto constant = interpreter.evaluate(selection.index);
      auto* integer =
          constant ? std::get_if<cxx::ConstInt>(&*constant) : nullptr;
      if (integer &&
          (integer->isZero() || integer->isNegative() == selection.subtract) &&
          integer->magnitude() <= static_cast<cxx::ConstInt::UWide>(SIZE_MAX) &&
          (!type || integer->magnitude() <
                        static_cast<cxx::ConstInt::UWide>(type->size()))) {
        selection.constant_index = static_cast<size_t>(integer->magnitude());
      }
    }
  }
  array_selections_.emplace(expression, selection);
  retain_destination(selection.array);
  retain_destination(expression);
}

const ArraySelection* ControlFlow::array_selection(
    cxx::ExpressionAST* expression) const {
  if (expression->kind() != cxx::ASTKind::SubscriptExpression &&
      expression->kind() != cxx::ASTKind::UnaryExpression) {
    return nullptr;
  }
  auto found = array_selections_.find(expression);
  if (found != array_selections_.end() &&
      unit_.typeTraits().is_pointer(found->second.array->type)) {
    auto* id = cxx::ast_cast<cxx::IdExpressionAST>(
        unwrap_expression(found->second.array));
    if (id) {
      auto alias = pointer_aliases_.find(id->symbol);
      if (alias == pointer_aliases_.end() || !alias->second.transparent ||
          argument_requires_storage(alias->second.identity)) {
        return nullptr;
      }
    }
  }
  return found == array_selections_.end() ? nullptr : &found->second;
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
      unit_.typeTraits().is_user_provided(constructor) ||
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
  for (auto* initializer : cxx::ListView{cxx::ctor_initializers(constructor)}) {
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
  auto parameters = function->parameters();
  auto* definition = function->resolvedDefinition();
  const bool has_definition =
      function->isDefined() || function->hasPendingBody() ||
      (definition && (definition->isDefined() || definition->hasPendingBody()));
  size_t ordinal = 0;
  for (auto* argument : cxx::ListView{call->expressionList}) {
    retained.arguments.push_back(retain_call_argument(
        argument, has_definition && ordinal < parameters.size()
                      ? candidate_pointer_object_type(parameters[ordinal])
                      : nullptr));
    ++ordinal;
  }
  if (function->isImplicitObjectMemberFunction()) {
    auto* base = unwrap_expression(call->baseExpression);
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
      auto selected = types_.member_object_selection(member, member);
      auto* projection = selected.projection ? &*selected.projection : nullptr;
      retained.current_receiver =
          !projection && is_current_object(unit_, selected.object);
      retained.storage_receiver =
          member->accessOp == cxx::TokenKind::T_MINUS_GREATER;
      if (!retained.current_receiver) {
        retained.receiver = retain_member_receiver(selected.object, projection);
      }
    } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base);
               id && id->subobjectPath) {
      retained.receiver = retain_member_receiver(
          nullptr, &types_.projection(id->subobjectPath, id));
    } else {
      retained.current_receiver = true;
    }
  }
  retain_reference_receiver_write(retained);
  retain_reference_arguments(retained);
  calls_.push_back(std::move(retained));
}

void ControlFlow::retain_object_method_call(cxx::CallExpressionAST* call,
                                            cxx::FunctionSymbol* function,
                                            cxx::ExpressionAST* receiver) {
  DirectCall retained = {
      .function = function,
      .source = call,
      .arguments = {},
      .receiver = retain_call_argument(receiver),
      .current_receiver = false,
      .storage_receiver = false,
  };
  for (auto* argument : cxx::ListView{call->expressionList}) {
    retained.arguments.push_back(retain_call_argument(argument));
  }
  retain_reference_arguments(retained);
  calls_.push_back(std::move(retained));
}

void ControlFlow::retain_call(cxx::SubscriptExpressionAST* call,
                              cxx::FunctionSymbol* function) {
  DirectCall retained = {function, call, {}, std::nullopt, false, false};
  auto parameters = function->parameters();
  size_t ordinal = 0;
  for (auto* argument : cxx::ListView{call->expressionList}) {
    retained.arguments.push_back(retain_call_argument(
        argument, ordinal < parameters.size()
                      ? candidate_pointer_object_type(parameters[ordinal])
                      : nullptr));
    ++ordinal;
  }
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
  auto parameters = call.function->parameters();
  for (size_t index = 0; index < call.argument_count; ++index) {
    retained.arguments.push_back(
        retain_call_argument(call.arguments[index],
                             candidate_pointer_object_type(parameters[index])));
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
  const bool transports_owner =
      may_transport_owner(implicit_object_type(unit_, call.function), source) &&
      !call.function->isDestructor();
  if (!is_current_object(unit_, call.receiver)) {
    if (call.receiver->valueCategory != cxx::ValueCategory::kPrValue) {
      if (types_.implicit_object_may_mutate(call.function, source) ||
          transports_owner) {
        record(call.receiver);
      } else {
        retain_destination(call.receiver);
      }
    }
    return;
  }
  const bool may_mutate =
      types_.implicit_object_may_mutate(call.function, source);
  if (!may_mutate && !transports_owner) {
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
    if (may_mutate || may_transport_owner(member.type(), source)) {
      record(member.source);
    }
  }
}

void ControlFlow::retain_reference_arguments(DirectCall& call) {
  call.pointer_result_observed =
      call.source != borrowed_pointer_expression_ &&
      unit_.typeTraits().is_pointer(
          cxx::type_cast<cxx::FunctionType>(call.function->type())
              ->returnType());
  auto parameters = call.function->parameters();
  auto traits = unit_.typeTraits();
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* argument = call.arguments[ordinal].source;
    auto* parameter_type = parameters[ordinal]->type();
    const bool borrowed_pointer =
        candidate_pointer_object_type(parameters[ordinal]) != nullptr;
    if (!traits.is_reference(parameter_type) && !borrowed_pointer && argument) {
      record_parameter_read(argument);
    }
    if (!traits.is_reference(parameter_type) || !argument ||
        argument->valueCategory == cxx::ValueCategory::kPrValue) {
      continue;
    }
    if (traits.is_const(traits.remove_reference(parameter_type)) &&
        !may_transport_owner(parameter_type, argument)) {
      retain_destination(argument);
    } else {
      record(argument);
    }
  }
  construction_.call(calls_.size());
}

void ControlFlow::record_parameter_read(cxx::ExpressionAST* expression) {
  if (!construction_.has_parameters()) {
    return;
  }
  // Materializing a by-value parameter can retain a parenthesized source
  // initializer beneath its frontend conversion (including semantic handles).
  cxx::Initializer initializer(expression);
  auto* value = initializer.singleExpression();
  if (value && view_element_origin(unit_, value)) {
    return;
  }
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(
          value ? unwrap_expression(value) : nullptr);
      id && pointer_alias_use_ == id->symbol) {
    // Arrow and dereference syntax evaluates the pointer expression, but a
    // transparent semantic alias has no pointer value to read. The enclosing
    // object/member conversion records the exact referent components.
    return;
  }
  if (value && !pointer_aliases_.empty() && record_pointer_alias_read(value)) {
    return;
  }
  if (auto* list = cxx::ast_cast<cxx::BracedInitListAST>(expression)) {
    record_copy_read(list->type, cxx::Initializer(list).expressionList());
    return;
  }
  if (value && receiver_partition_ && is_current_object(unit_, value)) {
    for (const auto& member : receiver_partition_->members) {
      construction_.read(member.source, 0, member.partition->component_count);
    }
    return;
  }
  if (auto selected = value ? retain_destination(value) : std::nullopt) {
    construction_.read(
        selected->binding, selected->component_offset,
        selected->member ? selected->member->component_count : SIZE_MAX);
  } else if (value) {
    auto identity = retain_call_argument(value);
    visit_intervals(identity, {}, [&](ObjectInterval interval) {
      construction_.read(interval.binding, interval.begin,
                         interval.end == ObjectInterval::kWholeObjectEnd
                             ? SIZE_MAX
                             : interval.end - interval.begin);
    });
  }
}

bool ControlFlow::record_pointer_alias_read(cxx::ExpressionAST* expression) {
  expression = unwrap_object_identity(unit_, expression);
  cxx::ExpressionAST* pointer = nullptr;
  ReferenceProjection projection;
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
      member && member->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    if (!field) {
      return false;
    }
    pointer = member->baseExpression;
    const auto selected = types_.selection(member, member);
    projection = {selected.component_offset,
                  selected.partition->component_count};
  } else if (auto* dereference =
                 cxx::ast_cast<cxx::UnaryExpressionAST>(expression);
             dereference && !dereference->symbol &&
             dereference->op == cxx::TokenKind::T_STAR) {
    pointer = dereference->expression;
  } else {
    return false;
  }
  auto* source =
      cxx::ast_cast<cxx::IdExpressionAST>(unwrap_expression(pointer));
  auto found =
      source ? pointer_aliases_.find(source->symbol) : pointer_aliases_.end();
  if (found == pointer_aliases_.end() || !found->second.transparent) {
    return false;
  }
  if (found->second.array_origin && projection.component_count == SIZE_MAX) {
    projection.component_count =
        types_.partition(expression->type, expression).component_count;
  }
  visit_intervals(
      found->second.identity, projection, [&](ObjectInterval interval) {
        construction_.read(interval.binding, interval.begin,
                           interval.end == ObjectInterval::kWholeObjectEnd
                               ? SIZE_MAX
                               : interval.end - interval.begin);
      });
  return true;
}

void ControlFlow::record_copy_read(const cxx::Type* type,
                                   cxx::List<cxx::ExpressionAST*>* arguments) {
  // A single same-type initializer copies the object, including when the
  // frontend preserves a glvalue instead of inserting a value conversion.
  if (construction_.has_parameters() && type && arguments && !arguments->next &&
      types_.unqualified(type) == types_.unqualified(arguments->value->type)) {
    record_parameter_read(arguments->value);
  }
}

void ControlFlow::record_parameter_assignment(cxx::ExpressionAST* expression) {
  if (!construction_.has_bindings()) {
    return;
  }
  if (view_element_origin(unit_, expression)) {
    return;
  }
  auto* identity = unwrap_object_identity(unit_, expression);
  cxx::FieldSymbol* field = nullptr;
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(identity)) {
    field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
  } else if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(identity)) {
    field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
  }
  if (field && unit_.typeTraits().is_reference(field->type())) {
    return;
  }
  if (receiver_partition_ && is_current_object(unit_, expression)) {
    for (const auto& member : receiver_partition_->members) {
      construction_.assign(member.source, 0, member.partition->component_count);
    }
    return;
  }
  if (auto selected =
          retain_destination(unwrap_object_identity(unit_, expression))) {
    construction_.assign(
        selected->binding, selected->component_offset,
        selected->member ? selected->member->component_count : SIZE_MAX);
  } else {
    auto target = retain_call_argument(expression);
    if (!target.inherent_storage &&
        (!target.identity_bindings.empty() || target.identity_intervals ||
         !target.reference_calls.empty())) {
      construction_.assign_indirect(field_assignments_.size());
      field_assignments_.push_back(std::move(target));
    }
  }
}

void ControlFlow::record_member_binding_read(cxx::ExpressionAST* expression) {
  cxx::Symbol* field;
  cxx::Symbol* root = nullptr;
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
    field = id->symbol;
    if (observes_reference_captures_) {
      root = id->subobjectPath ? id->subobjectPath->value : field;
    }
  } else {
    auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
    if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
      return;
    }
    field = member->symbol;
    if (observes_reference_captures_) {
      auto* projection = types_.member_object_projection(member, member);
      root = projection ? projection->root->source : field;
    }
  }
  if (observes_reference_captures_) {
    if (auto captured =
            captured_object_member(unit_, receiver_partition_->source, root)) {
      construction_.read(captured->pointer_field, 0, SIZE_MAX);
      return;
    }
  }
  if (cxx::symbol_cast<cxx::FieldSymbol>(field) &&
      unit_.typeTraits().is_reference(field->type())) {
    // A source access reads the stored binding even when it only writes the
    // referent. That write never defines or replaces the binding itself.
    if (auto target = retain_destination(expression)) {
      construction_.read(
          target->binding, target->component_offset,
          target->member ? target->member->component_count : SIZE_MAX);
    }
  }
}

CallArgument ControlFlow::retain_call_argument(
    cxx::ExpressionAST* expression, const cxx::Type* borrowed_object_type) {
  CallArgument argument = {.source = expression};
  if (unit_.typeTraits().is_pointer(expression->type) &&
      is_current_object(unit_, unwrap_expression(expression))) {
    // An explicit this pointer crosses the call as an address. Implicit
    // receivers and value references still select their own native transport.
    receiver_requires_storage_ = true;
  }
  if (borrowed_object_type) {
    auto* source = unwrap_expression(expression);
    auto* address = cxx::ast_cast<cxx::UnaryExpressionAST>(source);
    if (address && !address->symbol && address->op == cxx::TokenKind::T_AMP &&
        types_.unqualified(address->expression->type) ==
            types_.unqualified(borrowed_object_type)) {
      collect_identity(address->expression, argument);
      auto target = classify_destination(
          unwrap_object_identity(unit_, address->expression));
      argument.write_binding = target ? target->binding : nullptr;
      return argument;
    }
    auto* forwarded = cxx::ast_cast<cxx::IdExpressionAST>(source);
    if (forwarded) {
      auto found = pointer_aliases_.find(forwarded->symbol);
      if (found != pointer_aliases_.end() && found->second.transparent) {
        append_identity(found->second.identity, argument);
        argument.write_binding = found->second.identity.write_binding;
        return argument;
      }
    }
    argument.inherent_storage = true;
    return argument;
  }
  collect_identity(expression, argument);
  auto target = classify_destination(unwrap_object_identity(unit_, expression));
  argument.write_binding = target ? target->binding : nullptr;
  return argument;
}

CallArgument ControlFlow::retain_member_receiver(
    cxx::ExpressionAST* expression, const SubobjectProjection* projection) {
  if (!projection) {
    return retain_call_argument(expression);
  }
  if (!expression || is_current_object(unit_, expression)) {
    CallArgument argument{
        .source = expression,
        .write_binding = projection->root->source,
    };
    append_identity(
        {projection->root->source,
         projection->component_offset - projection->root->component_offset,
         projection->partition},
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
      {projection->component_offset, projection->partition->component_count});
  return argument;
}

void ControlFlow::append_identity(Destination destination,
                                  CallArgument& argument) {
  append_identity(object_interval(destination), argument);
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
  if (view_element_origin(unit_, expression)) {
    argument.inherent_storage = true;
    return;
  }
  auto append_destination = [&](Destination destination) {
    auto interval = project_interval(object_interval(destination), projection);
    if (interval) {
      append_identity(*interval, argument);
    }
  };
  auto traits = unit_.typeTraits();
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
      binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
    collect_identity(binary->rightExpression, argument, projection);
    return;
  }
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
    if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol);
        variable && has_nonlocal_storage(variable)) {
      argument.inherent_storage = true;
      return;
    }
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
    if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
      collect_identity(member->baseExpression, argument, projection);
      return;
    }
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    if (field && member->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
      auto* base = cxx::ast_cast<cxx::IdExpressionAST>(
          unwrap_expression(member->baseExpression));
      if (base) {
        auto found = pointer_aliases_.find(base->symbol);
        if (found != pointer_aliases_.end() && found->second.transparent) {
          const auto selected = types_.selection(member, member);
          append_identity(
              found->second.identity, argument,
              compose_projection({selected.component_offset,
                                  selected.partition->component_count},
                                 projection));
          return;
        }
      }
      argument.inherent_storage = true;
      return;
    }
    auto* object = types_.member_object_projection(member, member);
    auto* root = object ? object->root->source : field;
    if (field && is_current_object(unit_, member->baseExpression) &&
        is_captured_object_field(root)) {
      argument.inherent_storage = true;
      return;
    }
    if (field && traits.is_lvalue_reference(field->type())) {
      argument.inherent_storage = true;
      return;
    }
    if (storage_expressions_.contains(member)) {
      argument.inherent_storage = true;
      return;
    }
    if (is_current_object(unit_, member->baseExpression)) {
      if (field) {
        append_destination(*classify_destination(member));
      }
      return;
    }
    if (auto destination = classify_destination(expression)) {
      if (types_.requires_value_transport(types_.unqualified_value(
              source_object_type(destination->binding)))) {
        destinations_.emplace(expression, *destination);
      }
      append_destination(*destination);
    } else {
      const auto selected = types_.selection(member, member);
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
    auto* index = subscript_index(subscript);
    auto* base_type = subscript->baseExpression->type;
    auto* index_type = index->type;
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
        collect_identity(index, argument, projection);
      }
      return;
    }
    if (!array_selection(subscript)) {
      argument.inherent_storage = true;
      return;
    }
  }
  if (const auto* selection = array_selection(expression)) {
    auto array_identity = retain_array_identity(selection->array);
    if (argument_requires_storage(array_identity)) {
      append_identity(array_identity, argument);
      argument.inherent_storage = true;
      return;
    }
    if (!selection->type) {
      const size_t count =
          types_.partition(expression->type, expression).component_count;
      append_identity(
          array_identity, argument,
          selection->constant_index
              ? compose_projection({*selection->constant_index * count, count},
                                   projection)
              : projection);
      return;
    }
    const auto* array = types_.array(selection->type, expression);
    const size_t first = selection->constant_index.value_or(0);
    const size_t end =
        selection->constant_index ? first + 1 : array->source->size();
    for (size_t index = first; index < end; ++index) {
      append_identity(
          array_identity, argument,
          compose_projection({index * array->element->component_count,
                              array->element->component_count},
                             projection));
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
    if (signature && (traits.is_reference(signature->returnType()) ||
                      traits.is_pointer(signature->returnType()))) {
      argument.reference_calls.push_back({expression, projection});
      return;
    }
  }
  if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(expression)) {
    if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
      auto* source = cxx::ast_cast<cxx::IdExpressionAST>(
          unwrap_expression(unary->expression));
      if (source) {
        auto found = pointer_aliases_.find(source->symbol);
        if (found != pointer_aliases_.end() && found->second.transparent) {
          if (found->second.array_origin) {
            projection = compose_projection(
                {0, types_.partition(expression->type, expression)
                        .component_count},
                projection);
          }
          append_identity(found->second.identity, argument, projection);
          return;
        }
      }
      argument.inherent_storage = true;
      return;
    }
  }
  if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(expression)) {
    auto* callee = direct_callee(call);
    auto* signature =
        callee ? cxx::type_cast<cxx::FunctionType>(callee->type()) : nullptr;
    if (signature && (traits.is_reference(signature->returnType()) ||
                      traits.is_pointer(signature->returnType()))) {
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
  bool changed = !facts.addressed;
  facts.addressed = true;
  if (object_identity_) {
    changed = object_identity_->require_storage(binding) || changed;
  }
  return changed;
}

void ControlFlow::retain_automatic_storage(cxx::VariableSymbol* variable,
                                           cxx::ExpressionAST* initializer) {
  if (variable->isStructuredBindingAlias()) {
    return;
  }
  auto traits = unit_.typeTraits();
  if (!traits.is_reference(variable->type()) &&
      ((!initializer && !types_.requires_value_transport(variable->type())) ||
       (traits.is_array(variable->type()) &&
        !types_.requires_value_transport(variable->type())) ||
       traits.is_volatile(variable->type()))) {
    binding_storage_[variable].automatic_storage = true;
  }
}

bool ControlFlow::require_identity(const CallArgument& argument) {
  bool changed = false;
  bool complete =
      !argument.identity_intervals && argument.identity_bindings.empty();
  for (const auto& reference : argument.reference_calls) {
    changed = identity_call_results_.insert(reference.call).second || changed;
  }
  if (argument.current_receiver) {
    changed = !receiver_requires_storage_ || changed;
    receiver_requires_storage_ = true;
  }
  auto traits = unit_.typeTraits();
  auto require_binding = [&](cxx::Symbol* binding) {
    if (auto found = pointer_aliases_.find(binding);
        found != pointer_aliases_.end() &&
        found->second.parameter_object_type) {
      changed = require_borrowed_pointer_storage(binding) || changed;
      return;
    }
    if (traits.is_reference(source_object_type(binding))) {
      if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(binding)) {
        changed = retain_reference_capture_identity(field) || changed;
        return;
      }
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
    if (!is_direct_subobject(binding) ||
        !require_storage_member({binding, 0, ObjectInterval::kWholeObjectEnd},
                                argument.source, changed)) {
      complete = true;
      require_binding(binding);
    }
  }
  if (argument.identity_intervals) {
    for (const auto& interval : *argument.identity_intervals) {
      if (!require_storage_member(interval, argument.source, changed)) {
        complete = true;
        require_binding(interval.binding);
      }
    }
  }
  if (complete && object_identity_) {
    changed = object_identity_->require_storage(argument.source) || changed;
  }
  return changed;
}

bool ControlFlow::record_callable_write(cxx::Symbol* binding) {
  if (!binding) {
    return false;
  }
  if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(binding);
      variable && has_nonlocal_storage(variable)) {
    return record_memory_write();
  }
  if (auto found = pointer_aliases_.find(binding);
      found != pointer_aliases_.end() && found->second.parameter_object_type) {
    return written_indirect_parameters_.insert(binding).second;
  }
  auto traits = unit_.typeTraits();
  if (traits.is_reference(source_object_type(binding)) && function_) {
    auto parameters = function_->parameters();
    if (std::ranges::find(parameters, binding) != parameters.end()) {
      return written_indirect_parameters_.insert(binding).second;
    }
  }
  if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(binding);
      field && traits.is_reference(field->type())) {
    return record_memory_write();
  }
  return is_direct_subobject(binding) && add_receiver_write(binding);
}

void ControlFlow::invalidate_projected_bound(cxx::AST* owner,
                                             ObjectInterval write) {
  auto* loop = cxx::ast_cast<cxx::ForStatementAST>(owner);
  if (!loop) {
    return;
  }
  auto found = counted_.find(loop);
  if (found == counted_.end()) {
    return;
  }
  auto* bound = found->second.referent;
  if (bound && bound->member && bound->binding == write.binding) {
    auto selected = object_interval(*bound);
    if (write.begin < selected.end && selected.begin < write.end) {
      counted_.erase(found);
    }
  }
}

bool ControlFlow::record_scoped_write(ObjectInterval interval,
                                      std::span<cxx::AST* const> owners) {
  auto* binding = interval.binding;
  if (!binding) {
    return false;
  }
  bool changed = observes_reference_captures_ &&
                 retain_reference_capture_identity(binding);
  if (reference_capture_facts_) {
    changed =
        reference_capture_facts_->written_bindings.insert(binding).second ||
        changed;
  }
  // A write through a reference member observes its identity and updates its
  // referent. The member's address stays fixed across structured edges.
  if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(binding);
      field && unit_.typeTraits().is_reference(field->type())) {
    return changed;
  }
  for (auto* owner : owners) {
    if (has_projected_loop_bounds_) {
      invalidate_projected_bound(owner, interval);
    }
    auto& writes = writes_[owner];
    if (std::ranges::find(writes, binding) == writes.end()) {
      writes.push_back(binding);
      changed = true;
    }
  }
  return changed;
}

ControlFlow::ReferenceCaptureFacts& ControlFlow::reference_capture_facts() {
  if (!reference_capture_facts_) {
    reference_capture_facts_ = std::make_unique<ReferenceCaptureFacts>();
    for (const auto& entry : writes_) {
      reference_capture_facts_->written_bindings.insert(entry.second.begin(),
                                                        entry.second.end());
    }
  }
  return *reference_capture_facts_;
}

bool ControlFlow::retain_reference_capture_identity(cxx::Symbol* symbol) {
  if (!observes_reference_captures_) {
    return false;
  }
  auto* field = reference_capture(unit_, types_, symbol);
  if (field && function_->parent() == field->parent()) {
    return types_.require_reference_capture_identity(field);
  }
  return false;
}

bool ControlFlow::resolve_member_constructions() {
  if (!object_identity_) {
    return false;
  }
  bool changed = false;
  for (const auto& member : object_identity_->member_constructions()) {
    if (object_identity_->requires_storage(member.initializer)) {
      const auto& part =
          types_.partition(member.initializer->type, member.initializer);
      require_storage_member(
          {member.binding, member.offset, member.offset + part.component_count},
          member.initializer, changed);
    }
  }
  return changed;
}

const Destination* ControlFlow::aggregate_destination(
    cxx::ExpressionAST* initializer) const {
  return object_identity_ ? object_identity_->aggregate_destination(initializer)
                          : nullptr;
}

std::optional<Destination> ControlFlow::member_construction(
    cxx::ExpressionAST* source) const {
  const auto* member = object_identity_
                           ? object_identity_->member_construction(source)
                           : nullptr;
  if (!member) {
    return {};
  }
  return Destination{member->binding, member->offset,
                     &types_.partition(source->type, source)};
}

bool ControlFlow::resolve_reference_captures() {
  if (!reference_capture_facts_) {
    return false;
  }
  bool changed = false;
  auto binding_is_written = [&](cxx::Symbol* binding) {
    return reference_capture_facts_->written_bindings.contains(binding);
  };
  for (auto& capture : reference_capture_facts_->captures) {
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
    if (!types_.reference_capture_requires_identity(capture.field) &&
        !written && !argument_requires_storage(referent)) {
      continue;
    }
    if (capture.value_transport_required) {
      diagnostics_.reject(
          unit_, capture.source,
          "lambda reference capture requires object identity, but its "
          "referent can only be transported as a value");
    }
    capture.requires_storage = true;
    changed = require_identity(referent) || changed;
  }
  return changed;
}

void ControlFlow::publish_reference_snapshots(Types& types) const {
  if (!reference_capture_facts_) {
    return;
  }
  for (const auto& capture : reference_capture_facts_->captures) {
    if (!capture.requires_storage) {
      types.select_reference_snapshot(capture.field);
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
    return effect == WriteEffect::Callable && record_memory_write();
  }
  bool changed = false;
  visit_intervals(argument, projection, [&](ObjectInterval interval) {
    changed = record_scoped_write(interval, owners) || changed;
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
  if (!receiver.source && !receiver.write_binding &&
      !receiver.current_receiver && receiver.identity_bindings.empty()) {
    receiver.source = cxx::ast_cast<cxx::ExpressionAST>(owner);
  }
  DirectCall retained = {
      .function = constructor,
      .source = owner,
      .arguments = {},
      .receiver = std::nullopt,
      .current_receiver = receiver.current_receiver,
      .storage_receiver = storage_receiver,
  };
  auto parameters = constructor->parameters();
  auto arguments = constructor_arguments(initializer);
  size_t ordinal = 0;
  for (auto* argument : arguments) {
    retained.arguments.push_back(retain_call_argument(
        argument, ordinal < parameters.size()
                      ? candidate_pointer_object_type(parameters[ordinal])
                      : nullptr));
    ++ordinal;
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
  auto* origin = temporary_origins_.result(expression);
  if (origin != expression) {
    return;
  }
  const auto& selected = temporary_origins_.destruction(origin);
  // Each constructed result retains its own receiver edge. A destructor can
  // observe that object's identity even when its constructor did not.
  retain_destructor_call(selected, CallArgument{.source = expression},
                         /*current_receiver=*/false, expression);
}

void ControlFlow::retain_automatic_destruction(cxx::Symbol* binding,
                                               cxx::AST* owner,
                                               const cxx::Type* object_type) {
  if (!binding) {
    return;
  }
  if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(binding)) {
    if (variable->isStructuredBindingAlias() || variable->isStatic() ||
        variable->isExtern() || variable->isThreadLocal()) {
      return;
    }
  }
  auto selected =
      destruction_for(unit_, object_type ? object_type : binding->type());
  if (!selected) {
    return;
  }
  automatic_destructions_.emplace(binding, *selected);
  const auto* element = types_.record(selected->element_type, owner);
  for (size_t index = 0; index < selected->element_count; ++index) {
    CallArgument receiver = {.source = nullptr, .write_binding = binding};
    append_identity({binding, index * element->component_count, element},
                    receiver);
    construction_.defer_call(calls_.size());
    retain_destructor_call(*selected, std::move(receiver),
                           /*current_receiver=*/false, owner);
  }
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
    const auto* element = types_.record(selected->element_type, owner);
    for (size_t index = 0; index < selected->element_count; ++index) {
      CallArgument receiver = {
          .source = nullptr,
          .write_binding = member.source,
      };
      append_identity(
          {member.source, index * element->component_count, element}, receiver);
      construction_.defer_call(calls_.size());
      retain_destructor_call(*selected, std::move(receiver),
                             /*current_receiver=*/false, owner);
    }
  }
}

void ControlFlow::configure_receiver(Construction::ParameterMode mode) {
  if (!receiver_partition_) {
    return;
  }
  for (const auto& member : receiver_partition_->members) {
    construction_.configure_parameter(
        member.source, mode,
        std::ranges::find(receiver_writes_, member.source) !=
            receiver_writes_.end());
  }
}

bool ControlFlow::receiver_can_construct_value() const {
  if (!function_ || !function_->isConstructor() || !receiver_partition_) {
    return false;
  }
  for (const auto& member : receiver_partition_->members) {
    const auto* parameter = construction_.parameter(member.source);
    if (!parameter || !parameter->inputs.empty() ||
        !parameter->outputs.contains({0, parameter->component_count})) {
      return false;
    }
  }
  return true;
}

void ControlFlow::select_constructor_receiver_storage() {
  if (function_ && function_->isConstructor() &&
      !receiver_can_construct_value()) {
    receiver_requires_storage_ = true;
  }
}

void ControlFlow::retain_aggregate_reference_identities(
    const cxx::Type* type, cxx::List<cxx::ExpressionAST*>* elements,
    cxx::AST* owner) {
  auto traits = unit_.typeTraits();
  if (!elements || !traits.is_class(type) || !traits.is_aggregate(type)) {
    return;
  }
  auto* record = types_.record(types_.unqualified(type), owner);
  if (!record) {
    return;
  }
  // Copy construction transfers existing bindings instead of creating a new
  // reference to the source record itself.
  if (!elements->next &&
      types_.unqualified(elements->value->type) == types_.unqualified(type)) {
    return;
  }
  for (const auto& member : record->members) {
    if (!elements) {
      break;
    }
    if (traits.is_reference(member.type())) {
      require_identity(elements->value);
    }
    elements = elements->next;
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
    for (const auto& interval : found->second.intervals) {
      visit(interval, selected);
    }
  }
}

bool ControlFlow::references_current_receiver_field(
    const CallArgument& argument) const {
  bool references_field = false;
  visit_intervals(argument, {}, [&](ObjectInterval interval) {
    references_field |= is_direct_subobject(interval.binding);
  });
  return references_field;
}

bool ControlFlow::propagate_projected_reference_return(
    const DirectCall& call, const ControlFlow& callee) {
  if (reference_return_ != ReferenceReturn::Projected ||
      !projected_reference_return_facts_) {
    return false;
  }
  auto& facts = *projected_reference_return_facts_;
  bool changed = false;
  if (auto selection = facts.selections.find(call.source);
      selection != facts.selections.end()) {
    changed = propagate_projected_reference_selection(
        call, callee, call.source, {},
        selection->second.propagated_origin_count);
  }
  if (!facts.array_results.empty()) {
    auto found = facts.array_results.find(call.source);
    if (found != facts.array_results.end()) {
      for (auto& result : found->second) {
        changed = propagate_projected_reference_selection(
                      call, callee, result.expression, result.projection,
                      result.propagated_origin_count) ||
                  changed;
      }
    }
  }
  return changed;
}

bool ControlFlow::propagate_projected_reference_selection(
    const DirectCall& call, const ControlFlow& callee, cxx::AST* expression,
    ReferenceProjection projection, size_t& propagated_count) {
  auto& facts = *projected_reference_return_facts_;
  auto& selection = facts.selections.find(expression)->second;
  const auto* array_element =
      array_selection(cxx::ast_cast<cxx::ExpressionAST>(expression));
  IREE_ASSERT(propagated_count != SIZE_MAX);
  bool changed = false;
  auto select_storage = [&] {
    changed = promote_projected_reference_storage(facts) || changed;
  };
  if (callee.reference_return_ == ReferenceReturn::Storage ||
      (callee.projected_reference_return_facts_ &&
       callee.projected_reference_return_facts_->requires_storage)) {
    select_storage();
    return changed;
  }

  auto translate_argument =
      [&](const CallArgument& argument,
          const ReferenceResultOrigin& source) -> TranslatedReferenceOrigins {
    TranslatedReferenceOrigins translated;
    visit_intervals(argument,
                    {source.projection.component_offset,
                     source.projection.partition->component_count},
                    [&](ObjectInterval interval) {
                      if (!translated.valid) {
                        return;
                      }
                      if (interval.reference_offset != 0 ||
                          (interval.end != ObjectInterval::kWholeObjectEnd &&
                           interval.end - interval.begin !=
                               source.projection.partition->component_count)) {
                        translated.valid = false;
                        return;
                      }
                      auto owner =
                          projected_reference_owner(facts, interval.binding);
                      if (owner) {
                        translated.append({owner->first,
                                           {source.projection.type,
                                            source.projection.partition,
                                            owner->second + interval.begin}});
                        return;
                      }
                      translated.valid = false;
                    });
    if (!translated.valid) {
      translated.invalidate();
    }
    return translated;
  };
  auto append_origin = [&](const ReferenceResultOrigin& source) {
    TranslatedReferenceOrigins translated;
    if (source.owner_ordinal == ReferenceResultOrigin::kReceiver) {
      if (call.current_receiver) {
        translated.append(source);
      } else if (call.receiver) {
        translated = translate_argument(*call.receiver, source);
      }
    } else if (source.owner_ordinal < call.arguments.size()) {
      translated =
          translate_argument(call.arguments[source.owner_ordinal], source);
    }
    if (translated.empty()) {
      select_storage();
      return;
    }
    for (auto& origin : translated.values()) {
      const size_t index =
          retain_projected_reference_origin(facts, std::move(origin)).first;
      selection.origins.push_back(index);
      // The selection itself is fixed-point state even when another leaf
      // already retained the same canonical origin.
      changed = true;
    }
  };
  size_t source_count = 0;
  auto append = [&](const ReferenceResultOrigin& source) {
    if (source_count++ < propagated_count) {
      return;
    }
    if (!array_element) {
      append_origin(source);
      return;
    }
    const auto* array = types_.array(
        array_element->type ? array_element->type : source.projection.type,
        expression);
    const bool constant =
        array_element->constant_index &&
        *array_element->constant_index < array->source->size();
    const size_t first = constant ? *array_element->constant_index : 0;
    const size_t end = constant ? first + 1 : array->source->size();
    for (size_t index = first; index < end; ++index) {
      append_origin(
          {source.owner_ordinal,
           {array->source->elementType(), array->element,
            source.projection.component_offset + projection.component_offset +
                index * array->element->component_count}});
    }
  };

  if (callee.reference_return_ == ReferenceReturn::Projected) {
    selection.origins.reserve(
        callee.projected_reference_return_facts_->origins.size());
    for (const auto& origin :
         callee.projected_reference_return_facts_->origins) {
      append(origin);
    }
  } else if (callee.reference_return_ == ReferenceReturn::Parameter) {
    auto parameters = callee.function_->parameters();
    selection.origins.reserve(callee.reference_return_parameters().size());
    for (size_t ordinal : callee.reference_return_parameters()) {
      auto* type = callee.types_.unqualified_value(parameters[ordinal]->type());
      append({ordinal, {type, &callee.types_.partition(type, call.source), 0}});
    }
  } else if (callee.returns_current_receiver_reference()) {
    auto* object_type = implicit_object_type(callee.unit_, callee.function_);
    auto* record =
        object_type ? callee.types_.record(
                          callee.types_.unqualified(object_type), call.source)
                    : nullptr;
    if (record) {
      selection.origins.reserve(1);
      append({ReferenceResultOrigin::kReceiver,
              {callee.types_.unqualified(object_type), record, 0}});
    }
  }
  propagated_count = source_count;
  return changed;
}

bool ControlFlow::retain_reference_result_identity(const DirectCall& call,
                                                   const ControlFlow& callee) {
  if (callee.reference_return_ == ReferenceReturn::None ||
      (callee.reference_return_ == ReferenceReturn::Storage &&
       !callee.projected_reference_return_facts_ &&
       unit_.typeTraits().is_pointer(
           cxx::type_cast<cxx::FunctionType>(callee.function_->type())
               ->returnType()))) {
    return false;
  }
  // Retain an exact automatic origin whenever every return aliases one
  // parameter or its receiver. Outer call edges can compare nested aliases
  // without traversing either function body again. The selected storage ABI is
  // retained separately so another reference call can transport it directly.
  bool result_storage =
      callee.reference_return_ == ReferenceReturn::Storage ||
      (callee.projected_reference_return_facts_ &&
       callee.projected_reference_return_facts_->requires_storage);
  bool changed = false;
  ReferenceIdentityFacts::Result* retained_result = nullptr;
  const size_t expected_interval_count =
      callee.projected_reference_return_facts_
          ? callee.projected_reference_return_facts_->origins.size()
          : size_t{1};
  auto append_result_interval = [&](ObjectInterval interval) {
    if (interval.begin == interval.end &&
        interval.end != ObjectInterval::kWholeObjectEnd) {
      return;
    }
    if (!reference_identity_facts_) {
      reference_identity_facts_ = std::make_unique<ReferenceIdentityFacts>();
    }
    if (!retained_result) {
      retained_result = &reference_identity_facts_->results[call.source];
    }
    auto& retained_intervals = retained_result->intervals;
    if (!retained_result->index &&
        retained_intervals.size() < kLinearCanonicalizationThreshold) {
      if (std::ranges::find(retained_intervals, interval) ==
          retained_intervals.end()) {
        retained_intervals.push_back(interval);
        changed = true;
      }
      return;
    }
    if (!retained_result->index) {
      retained_result->index = std::make_unique<std::unordered_set<
          ObjectInterval, ReferenceIdentityFacts::IntervalHash>>();
      retained_result->index->reserve(
          std::max(retained_intervals.size() + 1, expected_interval_count));
      retained_result->index->insert(retained_intervals.begin(),
                                     retained_intervals.end());
    }
    if (retained_result->index->insert(interval).second) {
      retained_intervals.push_back(interval);
      changed = true;
    }
  };
  if (callee.reference_return_ == ReferenceReturn::Parameter) {
    auto parameters = callee.function_->parameters();
    for (size_t ordinal : callee.reference_return_parameters()) {
      if (ordinal >= call.arguments.size()) {
        continue;
      }
      result_storage |=
          callee.reference_requires_storage(parameters[ordinal]) ||
          argument_requires_storage(call.arguments[ordinal]);
      visit_intervals(call.arguments[ordinal], {}, append_result_interval);
    }
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
          append_result_interval({member.source, 0,
                                  member.partition->component_count,
                                  member.component_offset});
        }
      }
    } else if (call.receiver) {
      visit_intervals(*call.receiver, {}, append_result_interval);
    }
  } else if (callee.reference_return_ == ReferenceReturn::Projected) {
    auto callee_parameters = callee.function_->parameters();
    for (const auto& origin :
         callee.projected_reference_return_facts_->origins) {
      ReferenceProjection projection{
          origin.projection.component_offset,
          origin.projection.partition->component_count};
      if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver &&
          call.current_receiver) {
        result_storage |=
            callee.receiver_requires_storage_ || receiver_requires_storage_;
        auto* object_type = implicit_object_type(unit_, function_);
        auto* record =
            object_type
                ? types_.record(types_.unqualified(object_type), call.source)
                : nullptr;
        if (!record) {
          continue;
        }
        for (const auto& member : record->members) {
          auto projected = project_interval(
              {member.source, 0, member.partition->component_count,
               member.component_offset},
              projection);
          if (projected) {
            append_result_interval(*projected);
          }
        }
        continue;
      }
      const CallArgument* source = nullptr;
      if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
        result_storage |=
            callee.receiver_requires_storage_ || call.storage_receiver;
        source = call.receiver ? &*call.receiver : nullptr;
      } else if (origin.owner_ordinal < call.arguments.size()) {
        result_storage |= callee.reference_requires_storage(
            callee_parameters[origin.owner_ordinal]);
        source = &call.arguments[origin.owner_ordinal];
      }
      if (source) {
        result_storage |= argument_requires_storage(*source);
        visit_intervals(*source, projection, append_result_interval);
      }
    }
  }
  if (!result_storage && callee.projected_reference_return_facts_ &&
      retained_result &&
      unit_.typeTraits().is_pointer(
          cxx::type_cast<cxx::FunctionType>(callee.function_->type())
              ->returnType())) {
    const auto& origins = callee.projected_reference_return_facts_->origins;
    const auto* array_type =
        origins.empty() ? nullptr : origins.front().projection.type;
    if (array_type && retained_result->array_type != array_type) {
      retained_result->array_type = array_type;
      changed = true;
    }
  }
  if (result_storage) {
    if (!reference_identity_facts_) {
      reference_identity_facts_ = std::make_unique<ReferenceIdentityFacts>();
    }
    changed =
        reference_identity_facts_->storage_results.insert(call.source).second ||
        changed;
  }
  return changed;
}

CallPropagation ControlFlow::propagate_overlapping_indirect_parameters(
    const DirectCall& call, ControlFlow& callee,
    iree_arena_allocator_t* scratch_arena) {
  bool caller_changed = false;
  bool callee_changed = false;
  auto parameters = callee.function_->parameters();
  const bool member_call = call.function->isImplicitObjectMemberFunction();
  const bool receiver_mutable =
      member_call && !callee.receiver_writes().empty();
  const bool indexed_receiver =
      member_call && !call.current_receiver && call.receiver.has_value();
  auto borrowed_pointer = [&](cxx::Symbol* parameter) {
    auto found = callee.pointer_aliases_.find(parameter);
    return found != callee.pointer_aliases_.end() &&
           found->second.parameter_object_type;
  };
  auto indirect_parameter = [&](cxx::Symbol* parameter) {
    return unit_.typeTraits().is_reference(parameter->type()) ||
           borrowed_pointer(parameter);
  };
  auto transfers_owner = [&](cxx::Symbol* parameter) {
    if (!unit_.typeTraits().is_reference(parameter->type())) {
      return false;
    }
    // Parameter input facts are still converging while call aliases select
    // identity. Querying them here can suppress selection before later call
    // effects arrive, so any owning group is identity-sensitive during this
    // fixed point. Only overlapping arguments materialize shared cells.
    auto* object_type = types_.unqualified(
        unit_.typeTraits().remove_reference(parameter->type()));
    const auto groups = types_.projections(object_type, call.source);
    return std::ranges::any_of(
        groups, [](const auto& group) { return group.carries_owner; });
  };
  auto requires_shared_identity = [&](cxx::Symbol* parameter) {
    return callee.indirect_parameter_written(parameter) ||
           transfers_owner(parameter);
  };
  auto select_storage = [&](size_t ordinal) {
    auto* parameter = parameters[ordinal];
    callee_changed =
        (unit_.typeTraits().is_reference(parameter->type())
             ? callee.require_reference_storage(parameter)
             : callee.require_borrowed_pointer_storage(parameter)) ||
        callee_changed;
    caller_changed =
        require_identity(call.arguments[ordinal]) || caller_changed;
  };

  // A value-indirect call represents each mutable parameter as an independent
  // result writeback and transports each managed owner through one argument.
  // Overlapping source intervals require one shared storage identity so reads,
  // writes, and owner transfer observe the source language's aliasing order.
  const size_t parameter_count =
      std::min(parameters.size(), call.arguments.size());
  size_t indirect_count = 0;
  size_t written_count = 0;
  for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
    if (indirect_parameter(parameters[ordinal])) {
      ++indirect_count;
      written_count += requires_shared_identity(parameters[ordinal]);
    }
  }
  const size_t alias_count = indirect_count + indexed_receiver;
  written_count += indexed_receiver && receiver_mutable;
  auto has_unresolved_identity = [&](const CallArgument& argument) {
    return std::ranges::any_of(
        argument.reference_calls, [&](const ReferenceCall& reference) {
          return !reference_identity_facts_ ||
                 !reference_identity_facts_->results.contains(reference.call);
        });
  };
  bool indexed_receiver_overlap = false;
  if (alias_count > 1 && written_count) {
    iree_arena_reset(scratch_arena);
    CallAliasParameter* alias_parameters = nullptr;
    check(iree_arena_allocate_array_aligned(
        scratch_arena, alias_count, sizeof(*alias_parameters),
        alignof(CallAliasParameter),
        reinterpret_cast<void**>(&alias_parameters)));
    size_t alias_index = 0;
    size_t interval_count = 0;
    auto retain_alias = [&](const CallArgument& argument, bool written) {
      alias_parameters[alias_index++] = {
          .written = written,
          .has_identity =
              argument.inherent_storage || argument.current_receiver ||
              !argument.identity_bindings.empty() ||
              argument.identity_intervals || !argument.reference_calls.empty(),
          .unresolved_identity = has_unresolved_identity(argument),
      };
      visit_intervals(argument, {}, [&](ObjectInterval) { ++interval_count; });
    };
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      if (indirect_parameter(parameters[ordinal])) {
        retain_alias(call.arguments[ordinal],
                     requires_shared_identity(parameters[ordinal]));
      }
    }
    if (indexed_receiver) {
      retain_alias(*call.receiver, receiver_mutable);
    }
    CallAliasInterval* alias_intervals = nullptr;
    if (interval_count) {
      check(iree_arena_allocate_array_aligned(
          scratch_arena, interval_count, sizeof(*alias_intervals),
          alignof(CallAliasInterval),
          reinterpret_cast<void**>(&alias_intervals)));
    }
    size_t interval_index = 0;
    alias_index = 0;
    auto retain_intervals = [&](const CallArgument& argument) {
      const size_t owner = alias_index++;
      visit_intervals(argument, {}, [&](ObjectInterval interval) {
        alias_intervals[interval_index++] = {interval.binding, interval.begin,
                                             interval.end, owner};
      });
    };
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      if (indirect_parameter(parameters[ordinal])) {
        retain_intervals(call.arguments[ordinal]);
      }
    }
    if (indexed_receiver) {
      retain_intervals(*call.receiver);
    }
    std::span<const uint8_t> storage_aliases;
    check(CallAliases({alias_parameters, alias_count},
                      {alias_intervals, interval_count})
              .select_storage(scratch_arena, &storage_aliases));
    alias_index = 0;
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      if (indirect_parameter(parameters[ordinal])) {
        if (storage_aliases[alias_index]) {
          select_storage(ordinal);
        }
        ++alias_index;
      }
    }
    if (indexed_receiver) {
      indexed_receiver_overlap = storage_aliases[alias_index];
    }
  }

  if (!member_call) {
    return {caller_changed, callee_changed};
  }
  if (indexed_receiver_overlap) {
    const bool receiver_already_storage =
        call.storage_receiver || argument_requires_storage(*call.receiver);
    if (!receiver_already_storage && !callee.receiver_requires_storage_) {
      callee.receiver_requires_storage_ = true;
      callee_changed = true;
    }
  }
  if (!call.current_receiver) {
    return {caller_changed, callee_changed};
  }
  for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
    auto* parameter = parameters[ordinal];
    const auto& argument = call.arguments[ordinal];
    if (!indirect_parameter(parameter) ||
        (!receiver_mutable && !callee.indirect_parameter_written(parameter))) {
      continue;
    }
    const bool receiver_overlap = argument.current_receiver ||
                                  references_current_receiver_field(argument);
    if (!receiver_overlap) {
      continue;
    }

    select_storage(ordinal);
    const bool receiver_already_storage =
        call.storage_receiver || receiver_requires_storage_;
    if (!receiver_already_storage && !callee.receiver_requires_storage_) {
      callee.receiver_requires_storage_ = true;
      callee_changed = true;
    }
  }
  return {caller_changed, callee_changed};
}

void ControlFlow::transfer_storage_members(
    ObjectInterval actual, ControlFlow& callee, cxx::Symbol* formal,
    cxx::AST* source, bool& caller_changed, bool& callee_changed) {
  actual = storage_interval(actual, source);
  // A private reference sees the caller's selected members, including
  // writes through a second pointer supplied in the very same call.
  // Copy descriptors before insertion since recursive edges can name the
  // same binding on both sides.
  const size_t count = storage_members(actual.binding).size();
  for (size_t i = 0; i < count; ++i) {
    const auto home = storage_members(actual.binding)[i];
    const size_t begin = home.component_offset;
    const size_t end = begin + home.partition->component_count;
    if (begin >= actual.begin && end <= actual.end) {
      const size_t offset = actual.reference_offset + begin - actual.begin;
      callee.require_storage_member(
          {formal, offset, offset + home.partition->component_count}, source,
          callee_changed);
    }
  }
  const size_t demanded = callee.storage_members(formal).size();
  for (size_t i = 0; i < demanded; ++i) {
    const auto home = callee.storage_members(formal)[i];
    auto mapped = project_interval(
        actual, {home.component_offset, home.partition->component_count});
    if (mapped) {
      require_storage_member(*mapped, source, caller_changed);
    }
  }
}

CallPropagation ControlFlow::propagate_call(
    const DirectCall& call, ControlFlow& callee,
    iree_arena_allocator_t* scratch_arena) {
  bool caller_changed = propagate_projected_reference_return(call, callee);
  bool callee_changed = false;
  if ((storage_members_ || callee.storage_members_) &&
      callee.receiver_partition_ && !call.function->isConstructor() &&
      !call.function->isDestructor()) {
    auto* formal = callee.receiver_partition_->source;
    if (call.receiver) {
      visit_intervals(*call.receiver, {}, [&](ObjectInterval actual) {
        transfer_storage_members(actual, callee, formal, call.source,
                                 caller_changed, callee_changed);
      });
    } else if (call.current_receiver && receiver_partition_) {
      transfer_storage_members(
          {receiver_partition_->source, 0, ObjectInterval::kWholeObjectEnd},
          callee, formal, call.source, caller_changed, callee_changed);
    }
  }
  const bool stored_receiver_write =
      !callee.receiver_writes().empty() &&
      (call.storage_receiver || callee.receiver_requires_storage_ ||
       (call.current_receiver && receiver_requires_storage_) ||
       (call.receiver && argument_requires_storage(*call.receiver)));
  if (callee.writes_memory_ || stored_receiver_write) {
    caller_changed = record_memory_write() || caller_changed;
  }
  // Value transport snapshots a reference at the call boundary. A write can
  // change it through another alias. Untracked return origins also need their
  // address. A whole-parameter selector returns an ordinal to the caller,
  // which already holds the original lvalues.
  const bool observes_reference_storage =
      callee.writes_memory_ || !callee.written_indirect_parameters_.empty() ||
      stored_receiver_write ||
      (callee.reference_return_ != ReferenceReturn::None &&
       !callee.selects_whole_parameters());
  if (object_identity_ && callee.object_identity_ &&
      (object_requires_storage(call.source) ||
       callee.result_requires_storage())) {
    caller_changed =
        object_identity_->require_storage(call.source) || caller_changed;
    callee_changed =
        callee.object_identity_->require_storage(callee.function_) ||
        callee_changed;
  }
  auto parameters = callee.function_->parameters();
  const bool storage_receiver_result =
      call.storage_receiver && callee.projected_reference_return_facts_ &&
      callee.projected_reference_return_facts_->has_partial_receiver_origin;
  for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
    auto* parameter = parameters[ordinal];
    if (ordinal >= call.arguments.size()) {
      continue;
    }
    const bool reference = unit_.typeTraits().is_reference(parameter->type());
    if (!reference && unit_.typeTraits().is_class(parameter->type()) &&
        callee.object_identity_ &&
        (callee.addressed(parameter) ||
         object_requires_storage(call.arguments[ordinal].source))) {
      callee_changed = callee.require_address(parameter) || callee_changed;
      if (object_identity_) {
        caller_changed =
            object_identity_->require_storage(call.arguments[ordinal].source) ||
            caller_changed;
      }
    }
    auto borrowed = callee.pointer_aliases_.find(parameter);
    const bool borrowed_pointer = borrowed != callee.pointer_aliases_.end() &&
                                  borrowed->second.parameter_object_type;
    if (!reference && !borrowed_pointer) {
      continue;
    }
    const auto& argument = call.arguments[ordinal];
    if (storage_members_ || callee.storage_members_) {
      visit_intervals(argument, {}, [&](ObjectInterval actual) {
        transfer_storage_members(actual, callee, parameter, call.source,
                                 caller_changed, callee_changed);
      });
    }
    if (reference && storage_receiver_result &&
        ordinal < callee.projected_reference_return_facts_->parameter_has_origin
                      .size() &&
        callee.projected_reference_return_facts_
            ->parameter_has_origin[ordinal]) {
      callee_changed =
          callee.require_reference_storage(parameter) || callee_changed;
    }
    if ((!reference || observes_reference_storage) &&
        argument_requires_storage(argument, StorageDemand::Identity)) {
      callee_changed =
          (reference ? callee.require_reference_storage(parameter)
                     : callee.require_borrowed_pointer_storage(parameter)) ||
          callee_changed;
    }
    const bool requires_storage =
        reference ? callee.reference_uses_storage_abi(parameter)
                  : callee.storage_borrowed_pointers_.contains(parameter);
    if (requires_storage) {
      caller_changed = require_identity(argument) || caller_changed;
    }
    if (callee.indirect_parameter_written(parameter)) {
      caller_changed =
          record_projected_write(argument, {}, {}, WriteEffect::Callable) ||
          caller_changed;
    }
  }
  caller_changed =
      retain_reference_result_identity(call, callee) || caller_changed;
  auto overlap_changes =
      propagate_overlapping_indirect_parameters(call, callee, scratch_arena);
  caller_changed = overlap_changes.caller_changed || caller_changed;
  callee_changed = overlap_changes.callee_changed || callee_changed;

  if (call.pointer_result_observed ||
      identity_call_results_.contains(call.source)) {
    if (auto ordinal = callee.reference_return_parameter()) {
      callee_changed = callee.require_reference_storage(parameters[*ordinal]) ||
                       callee_changed;
      caller_changed =
          require_identity(call.arguments[*ordinal]) || caller_changed;
    } else if (callee.projected_reference_return_facts_) {
      auto& facts = *callee.projected_reference_return_facts_;
      callee_changed =
          callee.promote_projected_reference_storage(facts) || callee_changed;
      for (const auto& origin : facts.origins) {
        if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
          if (call.current_receiver) {
            caller_changed = !receiver_requires_storage_ || caller_changed;
            receiver_requires_storage_ = true;
          } else if (!call.storage_receiver && call.receiver) {
            caller_changed = require_identity(*call.receiver) || caller_changed;
          }
        } else if (origin.owner_ordinal < call.arguments.size()) {
          callee_changed = callee.require_reference_storage(
                               parameters[origin.owner_ordinal]) ||
                           callee_changed;
          caller_changed =
              require_identity(call.arguments[origin.owner_ordinal]) ||
              caller_changed;
        }
      }
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
      } else if (callee.reference_return_ == ReferenceReturn::Parameter) {
        for (size_t ordinal : callee.reference_return_parameters()) {
          caller_changed =
              record_projected_write(call.arguments[ordinal], write.owners,
                                     projection.range, projection.effect) ||
              caller_changed;
        }
      } else if (callee.reference_return_ == ReferenceReturn::Projected ||
                 (call.function->isImplicitObjectMemberFunction() &&
                  callee.returns_current_receiver_reference())) {
        auto record_receiver_projection = [&](ReferenceProjection range) {
          if (call.current_receiver) {
            const auto* record = types_.record(
                types_.unqualified(implicit_object_type(unit_, function_)),
                call.source);
            for (const auto& member : record->members) {
              if (auto selected = project_interval(
                      {member.source, 0, member.partition->component_count,
                       member.component_offset},
                      range)) {
                caller_changed = record_scoped_write(*selected, write.owners) ||
                                 caller_changed;
                if (projection.effect == WriteEffect::Callable) {
                  caller_changed =
                      add_receiver_write(member.source) || caller_changed;
                }
              }
            }
          } else if (!call.storage_receiver && call.receiver) {
            caller_changed =
                record_projected_write(*call.receiver, write.owners, range,
                                       projection.effect) ||
                caller_changed;
          }
        };
        if (callee.reference_return_ == ReferenceReturn::Projected) {
          for (const auto& origin :
               callee.projected_reference_return_facts_->origins) {
            auto range = compose_projection(
                {origin.projection.component_offset,
                 origin.projection.partition->component_count},
                projection.range);
            if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
              record_receiver_projection(range);
            } else if (origin.owner_ordinal < call.arguments.size()) {
              caller_changed = record_projected_write(
                                   call.arguments[origin.owner_ordinal],
                                   write.owners, range, projection.effect) ||
                               caller_changed;
            }
          }
        } else {
          record_receiver_projection(projection.range);
        }
      }
    }
  }

  if (!call.function->isImplicitObjectMemberFunction()) {
    return {caller_changed, callee_changed};
  }
  if ((call.pointer_result_observed ||
       identity_call_results_.contains(call.source)) &&
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

void ControlFlow::bind_field_calls(
    const std::unordered_map<cxx::FunctionSymbol*,
                             std::unique_ptr<ControlFlow>>& functions) {
  for (auto& assignment : construction_.assignments()) {
    const auto& target = field_assignments_[assignment.source_index];
    auto count = types_
                     .partition(types_.unqualified_value(target.source->type),
                                target.source)
                     .component_count;
    visit_intervals(target, {}, [&](ObjectInterval interval) {
      assignment.destinations.push_back(
          {interval.binding,
           {interval.begin, interval.end == ObjectInterval::kWholeObjectEnd
                                ? interval.begin + count
                                : interval.end},
           interval.reference_offset,
           true});
    });
    Construction::classify_definite_destinations(assignment.destinations);
    std::erase_if(assignment.destinations, [&](const auto& destination) {
      return !construction_.declared(destination.binding);
    });
  }
  for (auto& invocation : construction_.calls()) {
    const auto& call = calls_[invocation.source_index];
    auto found = functions.find(call.function->canonical());
    auto* callee = found == functions.end() ? nullptr : found->second.get();
    auto bind = [&](const CallArgument& argument, size_t total,
                    const Construction::Parameter* parameter, bool writable,
                    ReferenceProjection projection = {}) {
      const size_t first = invocation.arguments.size();
      visit_intervals(argument, projection, [&](ObjectInterval interval) {
        const size_t count = interval.end == ObjectInterval::kWholeObjectEnd
                                 ? total - interval.reference_offset
                                 : interval.end - interval.begin;
        invocation.arguments.push_back(
            {interval.binding,
             interval.begin,
             {interval.reference_offset, interval.reference_offset + count},
             parameter,
             true,
             writable});
      });
      // A selected reference may name several objects. A returned field is
      // definite only for intervals with one source identity on every path.
      Construction::classify_definite_arguments(
          std::span(invocation.arguments).subspan(first));
      // A choice between a tracked object and an already initialized value is
      // still conditional. Remove nonparticipating identities only after they
      // have constrained definite assignment of the tracked alternatives.
      auto start = invocation.arguments.begin() + first;
      invocation.arguments.erase(
          std::remove_if(start, invocation.arguments.end(),
                         [&](const Construction::Argument& actual) {
                           return !construction_.declared(actual.binding);
                         }),
          invocation.arguments.end());
    };
    auto parameters = call.function->parameters();
    for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
      auto* parameter = parameters[ordinal];
      const bool reference = unit_.typeTraits().is_reference(parameter->type());
      const cxx::Type* borrowed_object_type = nullptr;
      if (callee) {
        auto found = callee->pointer_aliases_.find(
            callee->function_->parameters()[ordinal]);
        if (found != callee->pointer_aliases_.end()) {
          borrowed_object_type = found->second.parameter_object_type;
        }
      }
      if (!reference && !borrowed_object_type) {
        continue;
      }
      auto* object_type = reference
                              ? types_.unqualified_value(parameter->type())
                              : types_.unqualified(borrowed_object_type);
      bind(
          call.arguments[ordinal],
          types_.partition(object_type, call.source).component_count,
          callee ? callee->construction_.parameter(
                       callee->function_->parameters()[ordinal])
                 : nullptr,
          !unit_.typeTraits().is_const(
              reference ? unit_.typeTraits().remove_reference(parameter->type())
                        : borrowed_object_type));
    }
    if (callee && callee->receiver_partition_) {
      for (const auto& member : callee->receiver_partition_->members) {
        auto* parameter = callee->construction_.parameter(member.source);
        if (call.current_receiver) {
          invocation.arguments.push_back(
              {member.source,
               0,
               {0, member.partition->component_count},
               parameter,
               true,
               parameter->returned});
        } else if (call.receiver) {
          bind(*call.receiver, member.partition->component_count, parameter,
               parameter->returned,
               {member.component_offset, member.partition->component_count});
        }
      }
    } else if (call.receiver) {
      const size_t count =
          types_
              .partition(types_.unqualified(
                             implicit_object_type(unit_, call.function)),
                         call.source)
              .component_count;
      bind(*call.receiver, count, nullptr,
           types_.implicit_object_may_mutate(call.function, call.source));
    }
  }
}

bool ControlFlow::normalize_field_groups(bool inputs) {
  bool changed = false;
  auto normalize = [&](cxx::Symbol* binding, const cxx::Type* type) {
    auto* effects = construction_.parameter(binding);
    if (!effects || effects->mode != Construction::ParameterMode::Projected) {
      return;
    }
    auto& components = inputs ? effects->inputs : effects->outputs;
    if (components.empty()) {
      return;
    }
    for (const auto& group :
         types_.projections(type, function_->declaration())) {
      const ComponentRange range{
          group.component_offset,
          group.component_offset + group.partition->component_count};
      if (components.intersects(range)) {
        changed = components.add(range) || changed;
      }
    }
  };
  for (auto* parameter : function_->parameters()) {
    auto found = pointer_aliases_.find(parameter);
    auto* object_type =
        found != pointer_aliases_.end() && found->second.parameter_object_type
            ? types_.unqualified(found->second.parameter_object_type)
            : types_.unqualified_value(parameter->type());
    normalize(parameter, object_type);
  }
  if (receiver_partition_) {
    auto traits = unit_.typeTraits();
    for (const auto& member : receiver_partition_->members) {
      normalize(member.source,
                traits.is_reference(member.type())
                    ? traits.add_pointer(traits.remove_reference(member.type()))
                    : types_.unqualified(member.type()));
    }
  }
  return changed;
}

bool ControlFlow::propagate_field_outputs() {
  const bool changed = construction_.propagate_outputs();
  return normalize_field_groups(false) || changed;
}

bool ControlFlow::propagate_field_inputs() {
  const bool changed = construction_.propagate_inputs();
  return normalize_field_groups(true) || changed;
}

void ControlFlow::finalize_callable_effects() {
  for (auto entry = array_selections_.begin();
       entry != array_selections_.end();) {
    auto& selection = entry->second;
    if (selection.type) {
      ++entry;
      continue;
    }
    selection.type = borrowed_array_type(selection.array);
    if (!selection.type) {
      entry = array_selections_.erase(entry);
      continue;
    }
    const auto* array = cxx::type_cast<cxx::BoundedArrayType>(
        types_.unqualified(selection.type));
    if (selection.constant_index &&
        *selection.constant_index >= array->size()) {
      selection.constant_index.reset();
    }
    ++entry;
  }
  if (reference_identity_facts_) {
    for (auto& entry : reference_identity_facts_->results) {
      entry.second.index.reset();
    }
  }
  if (projected_reference_return_facts_) {
    auto& facts = *projected_reference_return_facts_;
    facts.array_results = {};
    facts.origin_indices = {};
    facts.parameter_ordinals = {};
    facts.receiver_offsets = {};
    facts.parameter_ordinals_indexed = false;
    facts.receiver_offsets_indexed = false;
    facts.parameter_has_origin = {};
    facts.partial_parameter_origins = {};
  }
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
  auto* initializer = variable && variable->initializer()
                          ? variable->initializer()
                          : ast->initializer;
  if (variable) {
    retain_owned_binding(variable, nullptr);
  }
  auto* saved_initializing_binding = initializing_binding_;
  initializing_binding_ = variable;
  auto* saved_borrowed_expression = borrowed_pointer_expression_;
  const bool saved_retain_reference_materializations =
      retain_reference_materializations_;
  if (variable && variable->isStatic()) {
    static_locals_.push_back(ast);
  }
  if (variable) {
    retain_reference_materializations_ |=
        initializer && unit_.typeTraits().is_reference(variable->type());
    if (auto* conversion = constructor_conversion(initializer);
        conversion &&
        same_function(conversion->conversionFunction,
                      variable->constructor()) &&
        types_.record_constructor_requires_call(
            variable->type(), conversion->conversionFunction, ast)) {
      bound_constructor_conversions_.insert(conversion);
    }
    auto traits = unit_.typeTraits();
    if (initializer &&
        traits.is_pointer(types_.unqualified_value(variable->type())) &&
        !traits.is_volatile(types_.unqualified_value(variable->type()))) {
      auto* initialized = cxx::Initializer(initializer).singleExpression();
      auto* source = initialized ? unwrap_expression(initialized) : nullptr;
      auto* address = cxx::ast_cast<cxx::UnaryExpressionAST>(source);
      cxx::ExpressionAST* referent = nullptr;
      cxx::ExpressionAST* array_origin = nullptr;
      CallArgument identity{};
      if (address && !address->symbol && address->op == cxx::TokenKind::T_AMP &&
          !traits.is_array(address->expression->type)) {
        referent = address->expression;
        identity = retain_call_argument(referent);
      } else if (source && traits.is_array(source->type)) {
        referent = array_origin = source;
        identity = retain_call_argument(referent);
      } else if (source && pointer_returning_call(source)) {
        referent = array_origin = source;
        identity = retain_call_argument(referent);
      } else if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(source)) {
        auto found = pointer_aliases_.find(id->symbol);
        if (found != pointer_aliases_.end() && found->second.transparent &&
            found->second.array_origin) {
          referent = source;
          array_origin = found->second.array_origin;
          identity = found->second.identity;
        }
      }
      if (referent &&
          (!traits.is_reference(variable->type()) ||
           initialized->valueCategory == cxx::ValueCategory::kPrValue) &&
          !identity.inherent_storage &&
          (identity.current_receiver || !identity.identity_bindings.empty() ||
           identity.identity_intervals || !identity.reference_calls.empty())) {
        borrowed_pointer_expression_ = source;
        pointer_aliases_.emplace(
            variable, PointerAlias{referent, nullptr, std::move(identity), true,
                                   array_origin});
      }
    }
  }
  accept(ast->declarator);
  accept(ast->requiresClause);
  accept(initializer);
  initializing_binding_ = saved_initializing_binding;
  borrowed_pointer_expression_ = saved_borrowed_expression;
  retain_reference_materializations_ = saved_retain_reference_materializations;
  if (variable) {
    auto traits = unit_.typeTraits();
    retain_owned_binding(variable, initializer);
    auto* folded_constant = folded_variable_constant(variable);
    const bool aliases_subobject = variable->isStructuredBindingAlias();
    const bool reference_binding =
        aliases_subobject || traits.is_reference(variable->type());
    const cxx::Type* automatic_object_type = nullptr;
    if (!folded_constant && !reference_binding && initializer) {
      record_parameter_read(initializer);
    }
    if (reference_binding && initializer) {
      auto* referent = reference_initializer(unit_, initializer);
      if (!referent) {
        diagnostics_.reject(unit_, initializer,
                            "reference bindings require one initializer "
                            "expression");
      }
      auto* materialization = temporary_origins_.materialization(referent);
      if (materialization &&
          !types_.requires_value_transport(materialization->type)) {
        materialization = nullptr;
      }
      automatic_object_type = materialization ? materialization->type : nullptr;
      auto destination = materialization
                             ? classify_materialized_destination(
                                   referent, materialization, variable)
                             : retain_destination(referent);
      auto identity = retain_call_argument(referent);
      if (materialization) {
        if (object_identity_) {
          object_identity_->own_members(variable, materialization);
        }
        identity = {.source = referent, .write_binding = variable};
        append_identity(destination.value_or(Destination{variable, 0, nullptr}),
                        identity);
      }
      if (argument_requires_storage(identity)) {
        require_identity(identity);
      }
      reference_aliases_.emplace(
          variable,
          ReferenceAlias{destination, std::move(identity), materialization});
      temporary_origins_.extend(referent);
    } else if (!folded_constant) {
      temporary_origins_.extend_members(initializer);
    }
    if (!folded_constant) {
      retain_constructor(variable, initializer, ast);
    }
    retain_automatic_storage(variable, initializer);
    if (!folded_constant) {
      analyze_default_constructor(variable->type(), variable->constructor(),
                                  ast);
    }
    if (!initializer && !storage_backed(variable)) {
      construction_.declare(
          variable, types_.partition(variable->type(), ast).component_count);
      if (default_constructor_initializes(variable->constructor())) {
        construction_.assign(variable, 0, SIZE_MAX);
      }
    }
    retain_automatic_destruction(variable, ast, automatic_object_type);
  }
}

void ControlFlow::visit(cxx::StructuredBindingDeclarationAST* ast) {
  // Raggi retains both the source initializer and the synthesized hidden
  // variable that owns it. The latter is the semantic declaration: visiting
  // both would count every initializer effect twice. Analyze the lowered
  // declaration sequence in its C++ initialization order instead.
  accept(ast->binding.hiddenVariable);
  for (auto* binding : cxx::ListView{ast->binding.bindingDeclaratorList}) {
    accept(binding);
  }
}

void ControlFlow::visit(cxx::ImplicitCastExpressionAST* ast) {
  if (observes_reference_captures_ && !ast->conversionFunction &&
      ast->castKind == cxx::ImplicitCastKind::kLValueToRValueConversion) {
    auto* saved_read = capture_value_read_;
    capture_value_read_ = unwrap_expression(ast->expression);
    cxx::ASTVisitor::visit(ast);
    capture_value_read_ = saved_read;
  } else {
    cxx::ASTVisitor::visit(ast);
  }
  if (!ast->conversionFunction &&
      ast->castKind == cxx::ImplicitCastKind::kLValueToRValueConversion) {
    record_parameter_read(ast->expression);
  }
  if (!ast->conversionFunction &&
      ast->castKind == cxx::ImplicitCastKind::kArrayToPointerConversion &&
      ast->expression != borrowed_pointer_expression_) {
    require_identity(ast->expression);
  }
  if (is_class_adjustment(ast)) {
    if (unit_.typeTraits().is_pointer(ast->type)) {
      // Pointer values already retain their allocation root and byte origin.
      // Translation adjusts that origin while preserving null; no pointee
      // value destination or object identity is required here.
      return;
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
    retain_call(*direct_operator_call(ast), ast);
    return;
  }
  if (bound_constructor_conversions_.contains(ast) ||
      !types_.record_constructor_requires_call(ast->type,
                                               ast->conversionFunction, ast)) {
    return;
  }
  retain_constructor_call(ast->conversionFunction, ast->expression,
                          CallArgument{}, /*storage_receiver=*/false, ast);
}

void ControlFlow::visit(cxx::BuiltinBitCastExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  record_parameter_read(ast->expression);
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
      field && unit_.typeTraits().is_reference(field->type())) {
    auto* source = cxx::memInitializerClause(unit_.arena(), initializer);
    auto* referent = reference_initializer(unit_, source);
    if (!referent) {
      diagnostics_.reject(unit_, source,
                          "reference members require one initializer "
                          "expression");
    }
    require_identity(referent);
    return;
  }
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
      retain_constructor_call(delegate, source,
                              CallArgument{.current_receiver = true},
                              /*storage_receiver=*/false, initializer);
    } else {
      analyze_default_constructor(object_type, delegate, initializer);
    }
    return;
  }
  auto* field = initializer->symbol;
  if (!is_direct_subobject(field)) {
    return;
  }
  if (!types_.record_constructor_requires_call(
          source_object_type(field), initializer->constructor, initializer)) {
    if (auto** arguments = cxx::memInitializerListSlot(initializer)) {
      record_copy_read(source_object_type(field), *arguments);
    }
    return;
  }
  auto* source = cxx::memInitializerClause(unit_.arena(), initializer);
  retain_constructor_call(initializer->constructor, source,
                          CallArgument{
                              .write_binding = initializer->symbol,
                          },
                          /*storage_receiver=*/false, initializer);
}

void ControlFlow::visit(cxx::ParenMemInitializerAST* ast) {
  auto* saved_initializing_binding = initializing_binding_;
  initializing_binding_ = ast->symbol;
  accept(cxx::memInitializerClause(unit_.arena(), ast));
  initializing_binding_ = saved_initializing_binding;
  retain_member_constructor(ast);
  if (construction_.parameter(ast->symbol)) {
    construction_.assign(ast->symbol, 0, SIZE_MAX);
  }
}

void ControlFlow::visit(cxx::BracedMemInitializerAST* ast) {
  auto* saved_initializing_binding = initializing_binding_;
  initializing_binding_ = ast->symbol;
  accept(cxx::memInitializerClause(unit_.arena(), ast));
  initializing_binding_ = saved_initializing_binding;
  retain_member_constructor(ast);
  if (construction_.parameter(ast->symbol)) {
    construction_.assign(ast->symbol, 0, SIZE_MAX);
  }
}

void ControlFlow::visit(cxx::ConditionExpressionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (ast->symbol) {
    temporary_origins_.extend_members(ast->initializer);
    retain_constructor(ast->symbol, ast->initializer, ast);
    retain_automatic_storage(ast->symbol, ast->initializer);
    analyze_default_constructor(ast->symbol->type(), ast->symbol->constructor(),
                                ast);
    retain_automatic_destruction(ast->symbol, ast);
  }
}

void ControlFlow::visit(cxx::BracedInitListAST* ast) {
  if (unit_.typeTraits().initializer_list_element_type(ast->type)) {
    if (!initializer_lists_) {
      initializer_lists_ =
          std::make_unique<InitializerLists>(unit_, diagnostics_);
    }
    auto* backing = initializer_lists_->retain(ast).backing;
    accept(backing);
    return;
  }
  for (auto* element : cxx::ListView{cxx::Initializer(ast).expressionList()}) {
    accept(element);
  }
  retain_aggregate_reference_identities(
      ast->type, cxx::Initializer(ast).expressionList(), ast);
  if (!ast->implicitElement) {
    return;
  }
  accept(ast->implicitElement->initializer());
  analyze_default_constructor(ast->implicitElement->type(),
                              ast->implicitElement->constructor(), ast);
}

void ControlFlow::visit(cxx::ParenInitializerAST* ast) {
  for (auto* element : cxx::ListView{cxx::Initializer(ast).expressionList()}) {
    accept(element);
  }
  retain_aggregate_reference_identities(
      ast->type, cxx::Initializer(ast).expressionList(), ast);
}

void ControlFlow::visit(cxx::TypeConstructionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  retain_aggregate_reference_identities(ast->type, ast->expressionList, ast);
  if (types_.record_constructor_requires_call(ast->type, ast->constructorSymbol,
                                              ast)) {
    retain_constructor_call(ast->constructorSymbol, ast, CallArgument{},
                            /*storage_receiver=*/false, ast);
  } else {
    record_copy_read(ast->type, ast->expressionList);
  }
  analyze_default_constructor(ast->type, ast->constructorSymbol, ast);
}

void ControlFlow::visit(cxx::BracedTypeConstructionAST* ast) {
  cxx::ASTVisitor::visit(ast);
  if (types_.record_constructor_requires_call(ast->type, ast->constructorSymbol,
                                              ast)) {
    retain_constructor_call(ast->constructorSymbol, ast, CallArgument{},
                            /*storage_receiver=*/false, ast);
  } else if (ast->bracedInitList) {
    record_copy_read(ast->type,
                     cxx::Initializer(ast->bracedInitList).expressionList());
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
  if (view_element_origin(unit_, expression)) {
    if (effect == WriteEffect::Callable) {
      record_memory_write();
    }
    return true;
  }
  auto target = retain_destination(expression);
  if (!target) {
    auto argument = retain_call_argument(expression);
    record_projected_write(argument, owners_, {}, effect);
    return !argument.inherent_storage;
  }
  record_scoped_write(object_interval(*target), owners_);
  if (effect == WriteEffect::Callable) {
    record_callable_write(target->binding);
  }
  return true;
}

void ControlFlow::record(cxx::Symbol* binding) {
  record_scoped_write({binding, 0, ObjectInterval::kWholeObjectEnd}, owners_);
}

bool ControlFlow::may_transport_owner(const cxx::Type* type, cxx::AST* owner) {
  if (!type) {
    return false;
  }
  auto groups = types_.projections(types_.unqualified_value(type), owner);
  return std::ranges::any_of(
      groups, [](const auto& group) { return group.carries_owner; });
}

void ControlFlow::record_implicit_object_call(cxx::CallExpressionAST* call,
                                              cxx::FunctionSymbol* callee) {
  if (!callee->isImplicitObjectMemberFunction()) {
    return;
  }
  const bool transports_owner =
      may_transport_owner(implicit_object_type(unit_, callee), call) &&
      !callee->isDestructor();
  const bool may_mutate = types_.implicit_object_may_mutate(callee, call);
  auto* base = unwrap_expression(call->baseExpression);
  if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base);
      id && id->subobjectPath) {
    if (may_mutate || transports_owner) {
      record(types_.projection(id->subobjectPath, id).root->source);
    }
    return;
  }
  if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base)) {
    auto selected = types_.member_object_selection(member, member);
    auto* receiver = selected.object;
    auto* projection = selected.projection ? &*selected.projection : nullptr;
    if (projection && is_current_object(unit_, receiver)) {
      if (may_mutate || transports_owner) {
        record(projection->root->source);
      }
      return;
    }
    if (!is_current_object(unit_, receiver)) {
      if (member->accessOp == cxx::TokenKind::T_DOT &&
          receiver->valueCategory != cxx::ValueCategory::kPrValue) {
        if (may_mutate || transports_owner) {
          record(receiver);
        } else {
          retain_destination(receiver);
        }
      }
      return;
    }
  }
  if (!may_mutate && !transports_owner) {
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
    if (may_mutate || may_transport_owner(member.type(), call)) {
      record(member.source);
    }
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
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
      binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
    return retain_destination(binary->rightExpression);
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
    auto* index = subscript_index(subscript);
    if (types_.native_vector(index->type)) {
      return classify_destination(index);
    }
  }
  if (auto* selection = array_selection(expression)) {
    if (selection->constant_index) {
      auto* source = array_borrow_origin(selection->array);
      auto target = source ? classify_destination(source) : std::nullopt;
      if (target && selection->type) {
        auto* binding_type = source_object_type(target->binding);
        if (storage_backed(target->binding) ||
            (!is_direct_subobject(target->binding) &&
             unit_.typeTraits().is_array(
                 unit_.typeTraits().remove_reference(binding_type)) &&
             !types_.requires_value_transport(binding_type))) {
          return std::nullopt;
        }
        const auto* array = types_.array(selection->type, expression);
        target->component_offset +=
            *selection->constant_index * array->element->component_count;
        target->member = array->element;
      }
      return target;
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
    auto* object = types_.member_object_projection(member, member);
    auto selected = types_.selection(member, member);
    if (is_captured_object_field(selected.root->source)) {
      return std::nullopt;
    }
    return Destination{
        selected.root->source,
        selected.component_offset - selected.root->component_offset,
        object ? selected.partition : nullptr};
  }
  if (member->accessOp != cxx::TokenKind::T_DOT) {
    return std::nullopt;
  }
  // Establish the automatic owner before requesting any SSA partition: a dot
  // member can also project a record reached through a pointer or subscript.
  auto target = classify_destination(member->baseExpression);
  if (target && target->member && target->member->kind == ValueKind::Pointer) {
    // A reference member's binding is part of the enclosing value, but the
    // referenced object's fields are in separate storage. Flattening through
    // that boundary would mistake its buffer identity for the first field.
    return std::nullopt;
  }
  if (target) {
    auto slice = types_.selection(member, member);
    target->component_offset += slice.component_offset;
    target->member = slice.partition;
  }
  return target;
}

std::optional<Destination> ControlFlow::classify_materialized_destination(
    cxx::ExpressionAST* expression, cxx::ExpressionAST* materialization,
    cxx::Symbol* owner) {
  if (!expression) {
    return std::nullopt;
  }
  if (expression->valueCategory == cxx::ValueCategory::kPrValue &&
      temporary_origins_.materialization(expression) == materialization) {
    return Destination{owner, 0, nullptr};
  }
  auto* unwrapped = unwrap_object_identity(unit_, expression);
  if (unwrapped != expression) {
    return classify_materialized_destination(unwrapped, materialization, owner);
  }
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
      binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
    return classify_materialized_destination(binary->rightExpression,
                                             materialization, owner);
  }
  if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(expression);
      cast && is_class_adjustment(cast)) {
    if (cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion) {
      return std::nullopt;
    }
    auto target = classify_materialized_destination(cast->expression,
                                                    materialization, owner);
    if (target) {
      const auto& selected = types_.projection(cast->subobjectPath, cast);
      target->component_offset += selected.component_offset;
      target->member = selected.partition;
    }
    return target;
  }
  auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
  if (!member || member->accessOp != cxx::TokenKind::T_DOT ||
      member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
    return std::nullopt;
  }
  auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
  if (!field || field->isStatic() ||
      unit_.typeTraits().is_reference(field->type())) {
    return std::nullopt;
  }
  auto target = classify_materialized_destination(member->baseExpression,
                                                  materialization, owner);
  if (target) {
    auto selected = types_.selection(member, member);
    target->component_offset += selected.component_offset;
    target->member = selected.partition;
  }
  return target;
}

bool ControlFlow::classify_storage(cxx::ExpressionAST* expression) const {
  if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(expression);
      binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
    return storage_expressions_.contains(binary->rightExpression);
  }
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
  if (auto* selection = array_selection(expression)) {
    return storage_expressions_.contains(selection->array);
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

}  // namespace loom::cxx_import
