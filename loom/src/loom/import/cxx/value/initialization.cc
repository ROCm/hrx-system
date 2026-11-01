// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/initialization.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/function_body.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstring>
#include <utility>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

// Restores the enclosing source current-object binding on every normal or
// diagnostic exit from one recursive initialization.
class ReceiverScope {
 public:
  ReceiverScope(Initialization::Evaluation& evaluation,
                const RecordPartition& record)
      : evaluation_(evaluation) {
    evaluation_.enter_value_receiver(record);
  }
  ReceiverScope(Initialization::Evaluation& evaluation,
                const RecordPartition& record, StorageProjection destination,
                const cxx::Type* object_type)
      : evaluation_(evaluation) {
    evaluation_.enter_storage_receiver(record, destination, object_type);
  }
  ReceiverScope(const ReceiverScope&) = delete;
  ReceiverScope& operator=(const ReceiverScope&) = delete;
  ~ReceiverScope() { evaluation_.leave_receiver(); }

 private:
  Initialization::Evaluation& evaluation_;
};

// Importer-owned construction cost. Expression evaluation is common to both
// candidates and is deliberately excluded; these counts describe only the
// operations and SSA edges that assemble the final vector.
struct VectorConstructionCost {
  size_t operation_count = 0;
  size_t operand_count = 0;

  bool strictly_smaller_than(const VectorConstructionCost& other) const {
    return operation_count < other.operation_count &&
           operand_count < other.operand_count;
  }
};

struct VectorInitializerLane {
  cxx::ExpressionAST* expression = nullptr;
  std::optional<cxx::ConstValue> constant;
  bool literal_encodable = false;
};

bool vector_literal_can_encode(const cxx::ConstValue& value) {
  const auto* floating = std::get_if<cxx::ConstFloat>(&value);
  return !floating || !floating->isNaN();
}

// Walks the frontend-owned declaration-order plan once. The importer supplies
// only the destination policy; constructor synthesis, omitted members, and
// default-member clauses remain source semantic facts.
template <typename Callback>
void for_each_default_initializer(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics,
                                  cxx::FunctionSymbol* constructor,
                                  cxx::AST* owner, Callback&& callback) {
  cxx::ASTRewriter::requireFunctionDefinition(&unit, constructor);
  auto* definition = constructor->resolvedDefinition()->declaration();
  auto* body = definition ? definition->functionBody : nullptr;
  if (!body) {
    diagnostics.reject(unit, owner,
                       "default record constructor has no source "
                       "initialization plan");
  }
  for (auto* initializer : cxx::ListView{cxx::ctor_initializers(body)}) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(initializer->symbol);
    if (!field) {
      diagnostics.reject(unit, initializer,
                         "record value constructors cannot initialize bases "
                         "or delegate");
    }
    auto* source = cxx::memInitializerClause(unit.arena(), initializer);
    callback(field, source, initializer);
  }
}

}  // namespace

loom_string_id_t Initialization::zero_name(loom_scalar_type_t scalar_type) {
  IREE_ASSERT(loom_scalar_type_is_valid(scalar_type));
  auto& name_id = zero_name_ids_[scalar_type];
  if (name_id != LOOM_STRING_ID_INVALID) {
    return name_id;
  }
  static constexpr char kPrefix[] = "zero$";
  char storage[32];
  const char* scalar_name = loom_scalar_type_name(scalar_type);
  const iree_host_size_t prefix_length = sizeof(kPrefix) - 1;
  const iree_host_size_t scalar_length = std::strlen(scalar_name);
  IREE_ASSERT(prefix_length + scalar_length <= sizeof(storage));
  std::memcpy(storage, kPrefix, prefix_length);
  std::memcpy(storage + prefix_length, scalar_name, scalar_length);
  check(loom_builder_intern_string(
      &builder_, iree_make_string_view(storage, prefix_length + scalar_length),
      &name_id));
  return name_id;
}

Value Initialization::value(const cxx::Type* type,
                            cxx::ExpressionAST* initializer,
                            cxx::FunctionSymbol* constructor, cxx::AST* owner,
                            std::optional<Value> retained) {
  if (auto* conversion = constructor_conversion(initializer);
      conversion && types_.is_user_record_constructor(
                        type, conversion->conversionFunction, owner)) {
    constructor = conversion->conversionFunction;
    initializer = conversion->expression;
  }
  type = types_.unqualified(type);
  cxx::List<cxx::ExpressionAST*>* elements = nullptr;
  cxx::VariableSymbol* implicit_element = nullptr;
  bool value_initialized = false;
  if (auto* construction =
          cxx::ast_cast<cxx::TypeConstructionAST>(initializer)) {
    elements = construction->expressionList;
    constructor = construction->constructorSymbol;
    value_initialized = construction->lparenLoc && !elements;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::BracedTypeConstructionAST>(initializer)) {
    auto* list = construction->bracedInitList;
    elements = list ? list->expressionList : nullptr;
    implicit_element = list ? list->implicitElement : nullptr;
    constructor = construction->constructorSymbol;
    value_initialized = list && !elements;
  } else if (auto* list = cxx::ast_cast<cxx::BracedInitListAST>(initializer)) {
    elements = list->expressionList;
    implicit_element = list->implicitElement;
  } else if (auto* list =
                 cxx::ast_cast<cxx::ParenInitializerAST>(initializer)) {
    elements = list->expressionList;
    value_initialized = list->lparenLoc && !elements;
  } else if (initializer) {
    return evaluation_.convert(initializer, type, owner);
  }

  const auto* array = types_.array(type, owner);
  if (constructor && !array) {
    types_.admit_construction(constructor, type, owner);
  }
  if (!array && types_.is_user_record_constructor(type, constructor, owner)) {
    std::vector<cxx::ExpressionAST*> arguments;
    for (auto* element : cxx::ListView{elements}) {
      arguments.push_back(element);
    }
    auto allocation =
        storage_.allocate(type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, owner);
    auto destination = storage_.project(allocation.pointer, type, owner);
    evaluation_.call_constructor(destination, *types_.record(type, owner), type,
                                 constructor, arguments, owner);
    return objects_.load(destination, type, owner);
  }
  if (auto* record = types_.record(type, owner);
      record && constructor &&
      same_function(constructor, record->source->defaultConstructor())) {
    return default_record(
        *record, constructor,
        value_initialized &&
            unit_.typeTraits().requires_zero_initialization(type, constructor),
        owner, retained);
  }
  return from_elements(type, elements, implicit_element, constructor, owner,
                       retained);
}

Value Initialization::zero(const cxx::Type* type, cxx::AST* owner) {
  type = types_.unqualified(type);
  if (types_.native_vector(type)) {
    return vectors_.zero(type, owner);
  }
  if (auto* array = types_.array(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(array->component_count);
    for (size_t index = 0; index < array->source->size(); ++index) {
      zero(array->source->elementType(), owner).append_to(components);
    }
    return arena_.capture(*array, components);
  }
  if (auto* record = types_.record(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(record->component_count);
    for (const auto& member : record->members) {
      if (unit_.typeTraits().is_reference(member.field->type())) {
        diagnostics_.reject(unit_, owner,
                            "reference members require an initializer");
      }
      zero(member.field->type(), owner).append_to(components);
    }
    return arena_.capture(*record, components);
  }
  const auto& partition = types_.partition(type, owner);
  auto output = types_.get(type, owner);
  if (partition.kind != ValueKind::SSA ||
      loom_type_kind(output) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, owner,
                        "zero initialization requires an admitted scalar, "
                        "array or record value");
  }
  loom_op_t* op;
  check(loom_scalar_constant_build(
      &builder_, types_.is_float(type) ? loom_attr_f64(0.0) : loom_attr_i64(0),
      output, locations_.get(owner), &op));
  loom_value_id_t result = loom_op_results(op)[0];
  check(loom_module_set_value_name(builder_.module, result,
                                   zero_name(loom_type_element_type(output))));
  return result;
}

std::optional<loom_value_id_t> Initialization::constant_vector(
    const cxx::Type* type, cxx::List<cxx::ExpressionAST*>* elements,
    cxx::AST* owner) {
  auto* vector = types_.native_vector(type);
  auto values = std::make_shared<cxx::InitializerList>();
  cxx::ASTInterpreter interpreter(&unit_);
  for (auto* element : cxx::ListView{elements}) {
    auto value = scalar_constant(unit_, element);
    if (!value) {
      return std::nullopt;
    }
    auto converted = interpreter.convertArithmetic(*value, element->type,
                                                   vector->elementType());
    if (!converted) {
      return std::nullopt;
    }
    values->elements.emplace_back(std::move(*converted), vector->elementType());
  }
  return vectors_.constant(cxx::ConstValue{std::move(values)}, type, owner);
}

loom_value_id_t Initialization::initialize_vector(
    const cxx::Type* type, cxx::List<cxx::ExpressionAST*>* elements,
    cxx::AST* owner) {
  if (auto constant = constant_vector(type, elements, owner)) {
    return *constant;
  }

  auto* vector = types_.native_vector(type);
  const size_t lane_count = vector->elementCount();
  std::vector<VectorInitializerLane> lanes;
  lanes.reserve(lane_count);
  cxx::ASTInterpreter interpreter(&unit_);
  for (auto* element : cxx::ListView{elements}) {
    VectorInitializerLane lane{.expression = element};
    auto value = scalar_constant(unit_, element);
    if (value) {
      lane.constant = interpreter.convertArithmetic(*value, element->type,
                                                    vector->elementType());
      lane.literal_encodable =
          lane.constant && vector_literal_can_encode(*lane.constant);
    }
    lanes.push_back(std::move(lane));
  }

  // The ordinary constructor remains the diagnostic owner for excess lanes.
  // Pure constants can be materialized from their converted frontend value;
  // this is the same exact conversion used by the all-literal path and makes
  // the construction cost below concrete rather than syntax-dependent.
  auto emit_from_elements = [&]() {
    std::vector<loom_value_id_t> components;
    components.reserve(lanes.size());
    for (const auto& lane : lanes) {
      components.push_back(
          lane.constant
              ? scalars_.constant(*lane.constant, vector->elementType(),
                                  lane.expression)
              : evaluation_
                    .convert(lane.expression, vector->elementType(), owner)
                    .ssa());
    }
    return vectors_.construct(components, type, owner);
  };

  if (lanes.size() > lane_count) {
    return emit_from_elements();
  }

  VectorConstructionCost from_elements_cost{
      .operation_count = 1,
      .operand_count = lane_count,
  };
  VectorConstructionCost literal_patch_cost{
      .operation_count = 1,
      .operand_count = 0,
  };
  for (const auto& lane : lanes) {
    if (lane.literal_encodable) {
      ++from_elements_cost.operation_count;
      continue;
    }

    // Each patch adds one static vector.insert with two SSA operands. A pure
    // NaN additionally uses the same payload constant and bitcast in both
    // candidates; only its insertion is unique to the compact form.
    ++literal_patch_cost.operation_count;
    literal_patch_cost.operand_count += 2;
    if (lane.constant) {
      from_elements_cost.operation_count += 2;
      ++from_elements_cost.operand_count;
      literal_patch_cost.operation_count += 2;
      ++literal_patch_cost.operand_count;
    }
  }
  if (lanes.size() < lane_count) {
    // vector.from_elements shares one scalar zero across omitted lanes.
    ++from_elements_cost.operation_count;
  }
  if (!literal_patch_cost.strictly_smaller_than(from_elements_cost)) {
    return emit_from_elements();
  }

  auto base = std::make_shared<cxx::InitializerList>();
  base->elements.reserve(lane_count);
  for (size_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    const bool literal_lane =
        lane_index < lanes.size() && lanes[lane_index].literal_encodable;
    base->elements.emplace_back(literal_lane ? *lanes[lane_index].constant
                                             : cxx::ConstValue{int64_t{0}},
                                vector->elementType());
  }
  const auto& first_value = std::get<0>(base->elements.front());
  const auto first_attribute =
      scalars_.constant_attribute(first_value, vector->elementType(), owner);
  bool uniform = true;
  for (size_t lane_index = 1; lane_index < lane_count; ++lane_index) {
    const auto& value = std::get<0>(base->elements[lane_index]);
    const auto attribute =
        scalars_.constant_attribute(value, vector->elementType(), owner);
    uniform &= loom_attribute_equal(&first_attribute, &attribute);
  }
  // Uniform literals become typed splats. Splat-plus-patch target selection is
  // not proven, so only a nonuniform literal authorizes compaction.
  if (uniform) {
    return emit_from_elements();
  }

  loom_value_id_t result =
      vectors_.constant(cxx::ConstValue{std::move(base)}, type, owner);
  for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    const auto& lane = lanes[lane_index];
    if (lane.literal_encodable) {
      continue;
    }
    const loom_value_id_t value =
        lane.constant
            ? scalars_.constant(*lane.constant, vector->elementType(),
                                lane.expression)
            : evaluation_.convert(lane.expression, vector->elementType(), owner)
                  .ssa();
    result = vectors_.insert_static(
        result, value, static_cast<int64_t>(lane_index), type, owner);
  }
  return result;
}

Value Initialization::finish_value_receiver(const RecordPartition& record,
                                            cxx::AST* owner) {
  std::vector<loom_value_id_t> components;
  components.reserve(record.component_count);
  for (const auto& member : record.members) {
    evaluation_.receiver_member(member.field, owner).append_to(components);
  }
  return arena_.capture(record, components);
}

Value Initialization::default_record(const RecordPartition& record,
                                     cxx::FunctionSymbol* constructor,
                                     bool zero_initialize, cxx::AST* owner,
                                     std::optional<Value> retained) {
  ReceiverScope receiver(evaluation_, record);
  if (retained && !zero_initialize) {
    for (const auto& member : record.members) {
      evaluation_.publish_receiver_member(
          member.field,
          retained->project(*member.partition, member.component_offset));
    }
  } else if (zero_initialize) {
    for (const auto& member : record.members) {
      if (unit_.typeTraits().is_lvalue_reference(member.field->type())) {
        continue;
      }
      evaluation_.publish_receiver_member(member.field,
                                          zero(member.field->type(), owner));
    }
  }

  for_each_default_initializer(
      unit_, diagnostics_, constructor, owner,
      [&](cxx::FieldSymbol* field, cxx::ExpressionAST* source,
          cxx::MemInitializerAST* initializer) {
        const bool is_reference =
            unit_.typeTraits().is_lvalue_reference(field->type());
        std::optional<Value> prior;
        if (!is_reference && (retained || zero_initialize)) {
          prior = evaluation_.receiver_member(field, initializer);
        }
        cxx::ExpressionAST* referent = nullptr;
        if (is_reference) {
          cxx::Initializer reference(source);
          referent = reference.singleExpression();
          if (!referent) {
            diagnostics_.reject(unit_, source,
                                "reference members require one initializer "
                                "expression");
          }
        }
        evaluation_.publish_receiver_member(
            field, referent
                       ? evaluation_.reference(referent, initializer)
                       : value(field->type(), source, initializer->constructor,
                               initializer, prior));
      });
  return finish_value_receiver(record, owner);
}

Value Initialization::from_elements(const cxx::Type* type,
                                    cxx::List<cxx::ExpressionAST*>* elements,
                                    cxx::VariableSymbol* implicit_element,
                                    cxx::FunctionSymbol* constructor,
                                    cxx::AST* owner,
                                    std::optional<Value> retained) {
  type = types_.unqualified(type);
  if (auto* array = types_.array(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(array->component_count);
    for (size_t index = 0; index < array->source->size(); ++index) {
      std::optional<Value> prior;
      if (retained) {
        prior = retained->project(*array->element,
                                  index * array->element->component_count);
      }
      Value initialized;
      if (elements) {
        initialized = evaluation_.convert(elements->value,
                                          array->source->elementType(), owner);
        elements = elements->next;
      } else if (implicit_element) {
        initialized =
            value(array->source->elementType(), implicit_element->initializer(),
                  implicit_element->constructor(), owner, prior);
      } else if (constructor) {
        initialized = value(array->source->elementType(), nullptr, constructor,
                            owner, prior);
      } else if (prior) {
        initialized = *prior;
      } else {
        initialized = zero(array->source->elementType(), owner);
      }
      initialized.append_to(components);
    }
    return arena_.capture(*array, components);
  }
  if (auto* record = types_.record(type, owner)) {
    // Single same-type arguments are copy-list/direct initialization, not
    // the first aggregate field. The normalized AST retains that distinction.
    if (elements && !elements->next &&
        types_.unqualified(elements->value->type) == types_.unqualified(type)) {
      return evaluation_.expression(elements->value);
    }
    ReceiverScope receiver(evaluation_, *record);
    for (const auto& member : record->members) {
      Value initialized;
      if (unit_.typeTraits().is_lvalue_reference(member.field->type())) {
        if (!elements) {
          diagnostics_.reject(unit_, owner,
                              "reference members require an initializer");
        }
        initialized = evaluation_.reference(elements->value, owner);
      } else {
        initialized = elements
                          ? evaluation_.convert(elements->value,
                                                member.field->type(), owner)
                          : zero(member.field->type(), owner);
      }
      evaluation_.publish_receiver_member(member.field, initialized);
      if (elements) {
        elements = elements->next;
      }
    }
    return finish_value_receiver(*record, owner);
  }
  const auto& partition = types_.partition(type, owner);
  if (partition.kind == ValueKind::Address) {
    if (elements && !elements->next) {
      return evaluation_.convert(elements->value, type, owner);
    }
    diagnostics_.reject(unit_, owner,
                        "address values require one integer or address value");
  }
  if (partition.kind == ValueKind::Buffer ||
      partition.kind == ValueKind::OpaqueDialect ||
      partition.kind == ValueKind::Encoding ||
      partition.kind == ValueKind::Pool || partition.kind == ValueKind::Tile ||
      partition.kind == ValueKind::Channel ||
      partition.kind == ValueKind::Vector ||
      partition.kind == ValueKind::View ||
      partition.kind == ValueKind::Tensor) {
    if (elements && !elements->next &&
        types_.unqualified(elements->value->type) == types_.unqualified(type)) {
      return evaluation_.expression(elements->value);
    }
    diagnostics_.reject(
        unit_, owner,
        "Loom source values require an operation result or a copy");
  }
  if (types_.native_vector(type)) {
    return initialize_vector(type, elements, owner);
  }
  auto output = types_.get(type, owner);
  if (elements && !elements->next) {
    return evaluation_.convert(elements->value, type, owner);
  }
  if (elements || loom_type_kind(output) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(
        unit_, owner,
        "value initialization requires a scalar, vector or admitted record");
  }
  return zero(type, owner);
}

void Initialization::default_storage_record(StorageProjection destination,
                                            const RecordPartition& record,
                                            const cxx::Type* object_type,
                                            cxx::FunctionSymbol* constructor,
                                            bool zero_initialize, bool retained,
                                            cxx::AST* owner) {
  types_.admit_construction(constructor, object_type, owner);
  ReceiverScope receiver(evaluation_, record, destination, object_type);
  if (zero_initialize) {
    for (const auto& member : record.members) {
      auto member_type = objects_.member_type(object_type, member.field);
      objects_.store(storage_.member(destination, member.field, owner),
                     zero(member_type, owner), member_type, owner);
      evaluation_.complete_storage_receiver_member(member.field);
    }
  } else if (retained) {
    for (const auto& member : record.members) {
      evaluation_.complete_storage_receiver_member(member.field);
    }
  }

  for_each_default_initializer(
      unit_, diagnostics_, constructor, owner,
      [&](cxx::FieldSymbol* field, cxx::ExpressionAST* source,
          cxx::MemInitializerAST* initializer) {
        auto field_type = objects_.member_type(object_type, field);
        object(storage_.member(destination, field, initializer), field_type,
               source, initializer->constructor, initializer,
               zero_initialize || retained);
        evaluation_.complete_storage_receiver_member(field);
      });
}

// A list initializes subobjects in place: a later default clause may observe
// an already initialized field. Explicit nested clauses still resolve their
// lexical enclosing receiver by field identity. Copy initialization captures
// the complete source value before replacing destination storage.
void Initialization::object(StorageProjection destination,
                            const cxx::Type* type,
                            cxx::ExpressionAST* initializer,
                            cxx::FunctionSymbol* constructor, cxx::AST* owner,
                            bool retained) {
  if (auto* conversion = constructor_conversion(initializer);
      conversion && types_.is_user_record_constructor(
                        type, conversion->conversionFunction, owner)) {
    constructor = conversion->conversionFunction;
    initializer = conversion->expression;
  }
  if (unit_.typeTraits().is_array(type) &&
      unit_.typeTraits().is_char_type(
          unit_.typeTraits().get_element_type(type)) &&
      unit_.typeTraits().is_volatile(type)) {
    if (auto* string = string_initializer(initializer)) {
      auto* array =
          cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(type));
      auto* element_type = unit_.typeTraits().get_element_type(type);
      for (size_t index = 0; index < array->size(); ++index) {
        const int64_t code_unit =
            index < string->charCount() ? string->charAt(index) : 0;
        auto value =
            scalars_.constant(cxx::ConstValue{code_unit}, element_type, owner);
        auto element = storage_.element(destination, array, index, owner);
        storage_.store(storage_.dereference(element, element_type, owner),
                       value, element_type, owner);
      }
      return;
    }
  }
  if (auto contents = literal(type, initializer, owner)) {
    auto source = storage_.root(
        storage_.literal(iree_make_const_byte_span(contents->bytes.data(),
                                                   contents->bytes.size()),
                         contents->is_utf8,
                         LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT,
                         destination.alignment, owner),
        owner);
    auto length = scalars_.integer(
        contents->bytes.size(), LOOM_SCALAR_TYPE_OFFSET, locations_.get(owner));
    loom_op_t* copy;
    check(loom_buffer_copy_build(
        &builder_, source.root, source.byte_offset, destination.pointer.root,
        destination.pointer.byte_offset, length, locations_.get(owner), &copy));
    return;
  }
  cxx::Initializer source(initializer);
  bool list = !initializer || source.form() == cxx::InitializerForm::kList ||
              source.form() == cxx::InitializerForm::kParen;
  bool value_initialized = false;
  std::vector<cxx::ExpressionAST*> elements;
  cxx::VariableSymbol* implicit_element = nullptr;
  if (list) {
    elements = source.arguments();
    auto* braced = source.bracedInitList();
    implicit_element = braced ? braced->implicitElement : nullptr;
    if (auto* paren = cxx::ast_cast<cxx::ParenInitializerAST>(initializer)) {
      value_initialized = paren->lparenLoc && !paren->expressionList;
    }
  } else if (unit_.typeTraits().is_class(type)) {
    // A same-type prvalue constructs this destination directly. Evaluating it
    // as a detached value first would hide already initialized subobjects from
    // later clauses that name the destination.
    auto* clause = source.singleExpression();
    if (clause && clause->valueCategory == cxx::ValueCategory::kPrValue &&
        types_.unqualified(clause->type) == types_.unqualified(type)) {
      if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(clause)) {
        object(destination, type, nested->expression, constructor, owner,
               retained);
        return;
      }
      if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(clause)) {
        evaluation_.conditional_object(destination, type, select, retained);
        return;
      }
      if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(clause);
          binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
        evaluation_.effect(binary->leftExpression);
        object(destination, type, binary->rightExpression, constructor, owner,
               retained);
        return;
      }
      cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
      if (auto* construction =
              cxx::ast_cast<cxx::BracedTypeConstructionAST>(clause)) {
        constructor = construction->constructorSymbol;
        arguments = construction->bracedInitList->expressionList;
        implicit_element = construction->bracedInitList->implicitElement;
        list = true;
      } else if (auto* construction =
                     cxx::ast_cast<cxx::TypeConstructionAST>(clause)) {
        constructor = construction->constructorSymbol;
        arguments = construction->expressionList;
        value_initialized =
            construction->lparenLoc && !construction->expressionList;
        list = true;
      }
      for (auto* argument : cxx::ListView{arguments}) {
        elements.push_back(argument);
      }
    }
  }

  auto* object_type = type;
  auto* shape = types_.unqualified(type);
  if (!types_.array(shape, owner) &&
      types_.is_user_record_constructor(shape, constructor, owner)) {
    evaluation_.call_constructor(destination, *types_.record(shape, owner),
                                 object_type, constructor, elements, owner);
    return;
  }
  if (auto* record = types_.record(shape, owner);
      record && constructor &&
      same_function(constructor, record->source->defaultConstructor())) {
    default_storage_record(
        destination, *record, object_type, constructor,
        value_initialized &&
            unit_.typeTraits().requires_zero_initialization(shape, constructor),
        retained, owner);
    return;
  }
  if (list) {
    if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(shape)) {
      for (size_t index = 0; index < array->size(); ++index) {
        const bool explicit_clause = index < elements.size();
        auto* clause = index < elements.size() ? elements[index] : nullptr;
        auto* element_constructor = clause ? nullptr : constructor;
        if (!clause && implicit_element) {
          clause = implicit_element->initializer();
          element_constructor = implicit_element->constructor();
        }
        object(storage_.element(destination, array, index, owner),
               unit_.typeTraits().get_element_type(object_type), clause,
               element_constructor, owner, retained && !explicit_clause);
      }
      return;
    }
    if (auto* record = types_.record(shape, owner)) {
      if (elements.size() == 1 &&
          types_.unqualified(elements[0]->type) == shape) {
        object(destination, object_type, elements[0], nullptr, owner);
        return;
      }
      ReceiverScope receiver(evaluation_, *record, destination, object_type);
      for (size_t index = 0; index < record->members.size(); ++index) {
        const auto& member = record->members[index];
        object(storage_.member(destination, member.field, owner),
               objects_.member_type(object_type, member.field),
               index < elements.size() ? elements[index] : nullptr, nullptr,
               owner, false);
        evaluation_.complete_storage_receiver_member(member.field);
      }
      return;
    }
  }
  auto initialized = value(shape, initializer, constructor, owner);
  objects_.store(destination, initialized, object_type, owner);
}

void Initialization::constructor_receiver(StorageProjection destination,
                                          const RecordPartition& record,
                                          const cxx::Type* object_type,
                                          cxx::FunctionSymbol* constructor,
                                          cxx::AST* owner) {
  default_storage_record(destination, record, object_type, constructor,
                         /*zero_initialize=*/false, /*retained=*/false, owner);
}

std::optional<ObjectRepresentation> Initialization::literal(
    const cxx::Type* type, cxx::ExpressionAST* initializer, cxx::AST* owner) {
  auto traits = unit_.typeTraits();
  if (!traits.is_array(type) ||
      !traits.is_char_type(traits.get_element_type(type))) {
    return std::nullopt;
  }
  auto* string = string_initializer(initializer);
  if (!string) {
    return std::nullopt;
  }
  cxx::ASTInterpreter interpreter(&unit_);
  auto value = interpreter.evaluateInitializer(type, initializer);
  auto representation =
      value ? object_representation(unit_, type, *value, string) : std::nullopt;
  if (!representation) {
    diagnostics_.reject(
        unit_, owner,
        "string initializer has no complete source object representation");
  }
  return representation;
}

void Initialization::storage(StorageAllocation allocation,
                             const cxx::Type* type,
                             cxx::ExpressionAST* initializer,
                             cxx::FunctionSymbol* constructor,
                             cxx::AST* owner) {
  if (auto* array =
          cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(type))) {
    if (unit_.typeTraits().is_char_type(array->elementType()) &&
        string_initializer(initializer)) {
      object(storage_.project(allocation.pointer, type, owner), type,
             initializer, constructor, owner);
      return;
    }
    cxx::Initializer source(initializer);
    if (!constructor && source.form() != cxx::InitializerForm::kList &&
        source.form() != cxx::InitializerForm::kParen) {
      diagnostics_.reject(
          unit_, owner, "automatic arrays require element-wise initialization");
    }
    auto elements = source.arguments();
    auto* braced = source.bracedInitList();
    auto* implicit_element = braced ? braced->implicitElement : nullptr;
    auto* element_type = unit_.typeTraits().get_element_type(type);
    for (size_t index = 0; index < array->size(); ++index) {
      auto* clause = index < elements.size() ? elements[index] : nullptr;
      auto* element_constructor = clause ? nullptr : constructor;
      if (!clause && implicit_element) {
        clause = implicit_element->initializer();
        element_constructor = implicit_element->constructor();
      }
      if (unit_.typeTraits().is_class(element_type) ||
          unit_.typeTraits().is_array(element_type) ||
          types_.native_vector(element_type)) {
        object(
            storage_.element(storage_.project(allocation.pointer, type, owner),
                             array, index, owner),
            element_type, clause, element_constructor, owner);
      } else {
        auto element = value(element_type, clause, element_constructor, owner);
        auto position = scalars_.integer(index, LOOM_SCALAR_TYPE_INDEX,
                                         locations_.get(owner));
        storage_.store({allocation.view, position}, element.ssa(), element_type,
                       owner);
      }
    }
  } else if (unit_.typeTraits().is_class(type)) {
    object(storage_.project(allocation.pointer, type, owner), type, initializer,
           constructor, owner);
  } else {
    storage_.store({allocation.view, std::nullopt},
                   evaluation_.expression(initializer).ssa(), type, owner);
  }
}

}  // namespace loom::cxx_import
