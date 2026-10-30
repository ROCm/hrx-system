// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/translation.h"

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/assumptions.h"
#include "loom/import/cxx/binding/command_programs.h"
#include "loom/import/cxx/binding/command_schedule.h"
#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/constexpr.h"
#include "loom/import/cxx/binding/exponent.h"
#include "loom/import/cxx/binding/fastmath.h"
#include "loom/import/cxx/binding/function_contracts.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/loop_schedule.h"
#include "loom/import/cxx/binding/target_definitions.h"
#include "loom/import/cxx/binding/template_definitions.h"
#include "loom/import/cxx/check.h"
#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/symbol/immutable_objects.h"
#include "loom/import/cxx/value/bitcast.h"
#include "loom/import/cxx/value/initialization.h"
#include "loom/import/cxx/value/objects.h"
#include "loom/import/cxx/value/representation.h"
#include "loom/import/cxx/value/scalar.h"
#include "loom/import/cxx/value/signature.h"
#include "loom/import/cxx/value/storage.h"
#include "loom/import/cxx/value/types.h"
#include "loom/import/cxx/value/vector.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {
class Translator final : private Initialization::Evaluation {
 public:
  Translator(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
             loom_module_t* module, const loom_cxx_import_options_t& options)
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        locations_(unit, diagnostics, module, options.source_observer),
        types_(unit, diagnostics, module),
        scalars_(unit, diagnostics, types_, locations_, builder_),
        names_(unit, diagnostics),
        immutable_objects_(unit, diagnostics, locations_, names_, module),
        fastmath_(unit, diagnostics),
        function_contracts_(unit, diagnostics),
        configs_(unit, diagnostics, types_, scalars_, locations_, names_),
        target_definitions_(unit, diagnostics, locations_, names_, module),
        template_definitions_(unit, diagnostics),
        vectors_(unit, diagnostics, types_, scalars_, locations_, builder_),
        storage_(unit, diagnostics, types_, scalars_, locations_, builder_),
        objects_(unit, types_, storage_, value_arena_),
        initialization_(unit, diagnostics, types_, scalars_, vectors_, storage_,
                        objects_, value_arena_, locations_, builder_, *this),
        launches_(unit, diagnostics),
        command_programs_(unit, diagnostics),
        pipeline_contracts_(unit, diagnostics),
        intrinsics_(unit, diagnostics, types_, scalars_, vectors_, locations_,
                    names_, launches_, command_programs_, pipeline_contracts_,
                    function_contracts_, module),
        functions_(unit, diagnostics, module, immutable_objects_, intrinsics_,
                   launches_, command_programs_, pipeline_contracts_, configs_,
                   fastmath_, function_contracts_, target_definitions_,
                   template_definitions_, names_),
        options_(options),
        default_fastmath_flags_(options.fp_model == LOOM_CXX_FP_MODEL_FAST
                                    ? LOOM_SCALAR_FASTMATHFLAGS_FAST
                                    : 0) {
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void translate() {
    functions_.select({options_.roots, options_.root_count});
    functions_.analyze(types_);
    for (size_t index = 0; index < functions_.pending().size(); ++index) {
      function(functions_.pending()[index]);
    }
    functions_.build_declarations(types_, locations_, &builder_);
    functions_.build_benchmarks(locations_, &builder_);
    immutable_objects_.build(&builder_);
    assembly_fragments_.verify(module_, options_.diagnostic_sink);
  }

 private:
  class FastMathScope {
   public:
    FastMathScope(uint8_t& active, uint8_t value)
        : active_(active), saved_(active) {
      active_ = value;
    }
    ~FastMathScope() { active_ = saved_; }

   private:
    // Translator state installed for this lexical source subtree.
    uint8_t& active_;
    // Enclosing source subtree state restored on every exit path.
    uint8_t saved_;
  };

  uint8_t resolve_fastmath(std::optional<uint8_t> contract) const {
    return contract.value_or(lexical_fastmath_flags_);
  }

  [[noreturn]] void fail(cxx::AST* ast, const std::string& message) {
    diagnostics_.reject(unit_, ast, message);
  }

  Value reference(cxx::ExpressionAST* expression, cxx::AST* owner) override {
    return reference_value(expression, owner);
  }

  void require_kernel_context(cxx::AST* owner) {
    // Required inlining defers the IR ancestor requirement until expansion.
    if (current_function_.kind == FunctionKind::Kernel ||
        current_function_.kind == FunctionKind::TemplateDefinition ||
        (current_function_.kind == FunctionKind::Ordinary &&
         loom_func_def_inline_policy(current_function_.operation) ==
             LOOM_INLINE_POLICY_INLINE)) {
      return;
    }
    fail(owner, "kernel intrinsic requires a kernel or force-inline helper");
  }

  void require_launch_configuration_context(cxx::AST* owner) {
    // Required inlining defers the IR ancestor requirement until expansion.
    if (current_function_.kind == FunctionKind::LaunchConfiguration ||
        current_function_.kind == FunctionKind::ClusteredLaunchConfiguration ||
        current_function_.kind == FunctionKind::TemplateDefinition ||
        (current_function_.kind == FunctionKind::Ordinary &&
         loom_func_def_inline_policy(current_function_.operation) ==
             LOOM_INLINE_POLICY_INLINE)) {
      return;
    }
    fail(owner,
         "target launch query requires a kernel configuration or force-inline "
         "helper");
  }

  struct ReceiverOwner {
    // Active receiver stack index.
    size_t index;
    // Whether source initialization has completed this subobject.
    bool complete;
  };

  struct ReceiverFieldBinding {
    // Member identity owned by this source current-object scope.
    cxx::FieldSymbol* field;
    // Enclosing receiver for the same declaration, when recursively active.
    std::optional<ReceiverOwner> previous_owner;
    // Enclosing value binding hidden while this receiver is current.
    std::optional<Value> previous_value;
  };

  struct ActiveReceiver {
    // Canonical flattened structure shared by value and storage receivers.
    const RecordPartition* record;
    // Addressable destination, absent for a record assembled as SSA values.
    std::optional<StorageProjection> storage;
    // Cv-qualified complete object type governing storage observations.
    const cxx::Type* object_type;
    // State restored when this source current-object scope ends.
    std::vector<ReceiverFieldBinding> fields;
  };

  void enter_receiver(const RecordPartition& record,
                      std::optional<StorageProjection> storage,
                      const cxx::Type* object_type) {
    ActiveReceiver receiver{&record, storage, object_type, {}};
    receiver.fields.reserve(record.members.size());
    const size_t receiver_index = receivers_.size();
    for (const auto& member : record.members) {
      auto owner = receiver_owners_.find(member.field);
      auto value = values_.find(member.field);
      receiver.fields.push_back(
          {member.field,
           owner == receiver_owners_.end()
               ? std::nullopt
               : std::optional<ReceiverOwner>{owner->second},
           value == values_.end() ? std::nullopt
                                  : std::optional<Value>{value->second}});
      receiver_owners_[member.field] = {receiver_index, false};
      values_.erase(member.field);
    }
    receivers_.push_back(std::move(receiver));
  }

  void enter_value_receiver(const RecordPartition& record) override {
    enter_receiver(record, std::nullopt, record.source->type());
  }

  void enter_storage_receiver(const RecordPartition& record,
                              StorageProjection destination,
                              const cxx::Type* object_type) override {
    enter_receiver(record, destination, object_type);
  }

  void bind_value_receiver(const RecordPartition& record, Value value) {
    enter_receiver(record, std::nullopt, record.source->type());
    for (const auto& member : record.members) {
      publish_receiver_member(
          member.field,
          value.project(*member.partition, member.component_offset));
    }
  }

  void bind_storage_receiver(const RecordPartition& record,
                             StorageProjection destination,
                             const cxx::Type* object_type) {
    enter_receiver(record, destination, object_type);
    for (const auto& member : record.members) {
      complete_storage_receiver_member(member.field);
    }
  }

  Value current_receiver_value(cxx::AST* owner) {
    IREE_ASSERT(!receivers_.empty());
    const auto& active = receivers_.back();
    IREE_ASSERT(!active.storage);
    std::vector<loom_value_id_t> components;
    components.reserve(active.record->component_count);
    for (const auto& member : active.record->members) {
      receiver_member(member.field, owner).append_to(components);
    }
    return value_arena_.capture(*active.record, components);
  }

  void replace_current_receiver(Value value) {
    IREE_ASSERT(!receivers_.empty());
    const auto& active = receivers_.back();
    IREE_ASSERT(!active.storage && &value.partition() == active.record);
    for (const auto& member : active.record->members) {
      publish_receiver_member(
          member.field,
          value.project(*member.partition, member.component_offset));
    }
  }

  ActiveReceiver* receiver(cxx::FieldSymbol* field) {
    auto found = receiver_owners_.find(field);
    return found == receiver_owners_.end() ? nullptr
                                           : &receivers_[found->second.index];
  }

  cxx::FieldSymbol* current_receiver_field(cxx::ExpressionAST* expression) {
    expression = unwrap_expression(expression);
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
      return field && receiver(field) ? field : nullptr;
    }
    auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(expression);
    if (!member || !is_current_object(member->baseExpression)) {
      return nullptr;
    }
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    return field && receiver(field) ? field : nullptr;
  }

  std::optional<StorageProjection> receiver_storage(cxx::FieldSymbol* field,
                                                    cxx::AST* owner) {
    auto* active = receiver(field);
    if (!active || !active->storage) {
      return std::nullopt;
    }
    return storage_.member(*active->storage, field, owner);
  }

  void publish_receiver_member(cxx::FieldSymbol* field, Value value) override {
    auto* active = receiver(field);
    IREE_ASSERT(active && !active->storage);
    values_[field] = value;
    receiver_owners_.at(field).complete = true;
  }

  void complete_storage_receiver_member(cxx::FieldSymbol* field) override {
    auto* active = receiver(field);
    IREE_ASSERT(active && active->storage);
    receiver_owners_.at(field).complete = true;
  }

  Value receiver_member(cxx::FieldSymbol* field, cxx::AST* owner) override {
    auto* active = receiver(field);
    if (!active) {
      fail(owner, "member expression requires a current source object");
    }
    if (!receiver_owners_.at(field).complete) {
      fail(owner, "member is read before its initialization is complete");
    }
    if (active->storage) {
      auto projection = storage_.member(*active->storage, field, owner);
      auto* member_type = objects_.member_type(active->object_type, field);
      if (unit_.typeTraits().is_array(member_type)) {
        return projection.pointer;
      }
      return unit_.typeTraits().is_class(member_type)
                 ? objects_.load(projection, member_type, owner)
                 : Value(storage_.load(
                       storage_.dereference(projection, member_type, owner),
                       member_type, owner));
    }
    auto found = values_.find(field);
    IREE_ASSERT(found != values_.end());
    return found->second;
  }

  void leave_receiver() override {
    IREE_ASSERT(!receivers_.empty());
    const size_t receiver_index = receivers_.size() - 1;
    auto receiver = std::move(receivers_.back());
    receivers_.pop_back();
    for (const auto& binding : receiver.fields) {
      auto found = receiver_owners_.find(binding.field);
      IREE_ASSERT(found != receiver_owners_.end() &&
                  found->second.index == receiver_index);
      if (binding.previous_owner) {
        found->second = *binding.previous_owner;
      } else {
        receiver_owners_.erase(found);
      }
      if (binding.previous_value) {
        values_[binding.field] = *binding.previous_value;
      } else {
        values_.erase(binding.field);
      }
    }
  }

  Value convert(cxx::ExpressionAST* input_ast, const cxx::Type* output_type,
                cxx::AST* owner) override {
    auto value = expression(input_ast);
    if (value.is_address()) {
      auto input_domain =
          static_cast<const AddressPartition&>(value.partition()).element_type;
      auto output = types_.get(output_type, owner);
      auto output_kind = loom_type_kind(output);
      auto output_domain = output_kind == LOOM_TYPE_SCALAR
                               ? loom_type_element_type(output)
                               : LOOM_SCALAR_TYPE_NONE;
      auto input = loom_type_scalar(input_domain);
      if (loom_type_equal(output, input)) {
        return value;
      }
      if (output_kind != LOOM_TYPE_SCALAR ||
          (!unit_.typeTraits().is_integral(output_type) &&
           output_domain != LOOM_SCALAR_TYPE_INDEX &&
           output_domain != LOOM_SCALAR_TYPE_OFFSET)) {
        fail(owner,
             "address conversion requires an address domain or C++ integer");
      }
      loom_op_t* cast;
      check(loom_index_cast_build(&builder_, value.ssa(), input, output,
                                  locations_.get(owner), &cast));
      auto result = loom_op_results(cast)[0];
      if (output_domain == LOOM_SCALAR_TYPE_INDEX ||
          output_domain == LOOM_SCALAR_TYPE_OFFSET) {
        return value_arena_.capture(types_.partition(output_type, owner),
                                    {&result, 1});
      }
      return Value(result);
    }
    if (value.is_record() || value.is_array() || value.is_buffer() ||
        value.is_opaque_dialect() || value.is_encoding() || value.is_vector() ||
        value.is_view() || value.is_tensor()) {
      if (&types_.partition(output_type, owner) != &value.partition()) {
        fail(owner, "conversion must preserve the source value type");
      }
      return value;
    }
    if (value.is_pointer()) {
      if (loom_type_kind(types_.get(output_type, owner)) != LOOM_TYPE_BUFFER) {
        fail(owner, "conversion must preserve pointer representation");
      }
      return value;
    }
    const auto& output_partition = types_.partition(output_type, owner);
    if (output_partition.kind == ValueKind::Address) {
      auto input = value.ssa();
      auto input_type = value_type(input);
      const auto& address =
          static_cast<const AddressPartition&>(output_partition);
      auto output = loom_type_scalar(address.element_type);
      if (loom_type_equal(input_type, output)) {
        return value_arena_.capture(output_partition, {&input, 1});
      }
      auto traits = unit_.typeTraits();
      auto input_domain = types_.address_domain(input_ast->type, owner);
      if (input_domain == LOOM_SCALAR_TYPE_NONE &&
          (!traits.is_integral(input_ast->type) ||
           types_.unqualified(input_ast->type)->kind() ==
               cxx::TypeKind::kBool)) {
        fail(owner,
             "address construction requires a non-Boolean integer or address");
      }
      loom_op_t* op;
      auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
      if (address.element_type == LOOM_SCALAR_TYPE_INDEX &&
          input_domain == LOOM_SCALAR_TYPE_NONE &&
          types_.is_unsigned(input_ast->type) &&
          !loom_type_equal(input_type, i64_type)) {
        check(loom_scalar_extui_build(&builder_, input, input_type, i64_type,
                                      locations_.get(owner), &op));
        input = loom_op_results(op)[0];
        input_type = i64_type;
      }
      check(loom_index_cast_build(&builder_, input, input_type, output,
                                  locations_.get(owner), &op));
      auto result = loom_op_results(op)[0];
      return value_arena_.capture(output_partition, {&result, 1});
    }
    return numeric_convert(value.ssa(), input_ast->type, output_type, owner);
  }

  loom_value_id_t numeric_convert(loom_value_id_t value,
                                  const cxx::Type* input_type,
                                  const cxx::Type* output_type,
                                  cxx::AST* owner) {
    return types_.native_vector(input_type) || types_.native_vector(output_type)
               ? vectors_.convert(value, input_type, output_type, owner,
                                  lexical_fastmath_flags_)
               : scalars_.convert(value, input_type, output_type, owner,
                                  lexical_fastmath_flags_);
  }

  loom_value_id_t binary(cxx::TokenKind token, loom_value_id_t left,
                         loom_value_id_t right, const cxx::Type* input_type,
                         const cxx::Type* output_type, cxx::AST* owner) {
    return types_.native_vector(input_type)
               ? vectors_.binary(token, left, right, input_type, output_type,
                                 owner, lexical_fastmath_flags_)
               : scalars_.binary(token, left, right, input_type, output_type,
                                 owner, lexical_fastmath_flags_);
  }

  loom_type_t value_type(loom_value_id_t value) {
    return loom_module_value_type(module_, value);
  }

  loom_string_id_t string(const std::string& value) {
    loom_string_id_t result;
    check(loom_builder_intern_string(
        &builder_, iree_make_cstring_view(value.c_str()), &result));
    return result;
  }

  loom_value_id_t name(loom_value_id_t value, const std::string& hint) {
    // A C++ alias of an existing SSA value keeps the original value's name.
    if (!hint.empty() &&
        loom_module_value(module_, value)->name_id == LOOM_STRING_ID_INVALID) {
      check(loom_module_set_value_name(module_, value, string(hint)));
    }
    return value;
  }

  Value name(Value value, const std::string& hint) {
    if (value.is_record()) {
      const auto& partition =
          static_cast<const RecordPartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        name(components[index], hint + "_" + partition.component_names[index]);
      }
    } else if (value.is_vector()) {
      const auto& partition =
          static_cast<const VectorPartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        auto suffix = partition.component_names[index];
        name(components[index], suffix.empty() ? hint : hint + "_" + suffix);
      }
    } else if (value.is_view()) {
      const auto& partition =
          static_cast<const ViewPartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        auto suffix = partition.component_names[index];
        name(components[index], suffix.empty() ? hint : hint + "_" + suffix);
      }
    } else if (value.is_tensor()) {
      const auto& partition =
          static_cast<const TensorPartition&>(value.partition());
      auto components = value.components();
      if (partition.is_dynamic()) {
        name(components[0], hint + "_dim0");
      }
      name(partition.tensor_value(components), hint);
    } else if (value.is_pointer()) {
      auto pointer = value.pointer();
      name(pointer.root, hint);
      name(pointer.byte_offset, hint + "_byte_offset");
    } else if (value.is_buffer() || value.is_encoding()) {
      name(value.components()[0], hint);
    } else {
      name(value.ssa(), hint);
    }
    return value;
  }

  Value region_value(loom_region_t* region, size_t& index,
                     const Partition& partition) {
    auto* block = loom_region_entry_block(region);
    auto components =
        std::span<const loom_value_id_t>(block->arg_ids, block->arg_count)
            .subspan(index, partition.component_count);
    index += partition.component_count;
    return value_arena_.capture(partition, components);
  }

  Value command_region_value(loom_region_t* region,
                             const StagedSignatureProjection& program,
                             const StagedParameterProjection& parameter,
                             const Partition& partition) {
    auto* block = loom_region_entry_block(region);
    auto arguments =
        std::span<const loom_value_id_t>(block->arg_ids, block->arg_count);
    if (!parameter.binding_count) {
      return value_arena_.capture(
          partition, arguments.subspan(parameter.specialization_offset,
                                       parameter.component_count));
    }
    const size_t binding_index =
        program.specialization_count + parameter.binding_offset;
    if (!parameter.specialization_count) {
      return value_arena_.capture(
          partition, arguments.subspan(binding_index, parameter.binding_count));
    }
    std::array<loom_value_id_t, LOOM_TYPE_MAX_RANK + 2> components;
    IREE_ASSERT_LE(parameter.component_count, components.size());
    std::copy_n(arguments.begin() + parameter.specialization_offset,
                parameter.specialization_count, components.begin());
    std::copy_n(arguments.begin() + binding_index, parameter.binding_count,
                components.begin() + parameter.specialization_count);
    return value_arena_.capture(
        partition, std::span<const loom_value_id_t>(components.data(),
                                                    parameter.component_count));
  }

  loom_value_id_t result(loom_op_t* op, const std::string& hint = "") {
    auto value = loom_op_results(op)[0];
    return hint.empty() ? value : name(value, hint);
  }

  // Source tails are borrowed stack frames. A returning branch consumes its
  // remaining source in at most one arm, so no continuation is duplicated.
  struct ReturnSequence {
    // Remaining statements in the current lexical block.
    cxx::List<cxx::StatementAST*>* remaining;
    // Enclosing block's tail, or null at the function boundary.
    const ReturnSequence* continuation;
    // Floating-point permissions at the lexical block boundary.
    uint8_t fastmath;
    // Remaining direct children containing a loop/switch return, including
    // enclosing lexical tails. Translation decrements this without rescanning.
    size_t callable_exit_count;
  };

  struct Returned {
    // Native return operands; empty for a void source function.
    std::vector<loom_value_id_t> values;
    // Return or branch whose source location owns the resulting exit.
    cxx::AST* source;
    // Whether the callable's ordinary return is unreachable on this path.
    bool unreachable = false;
  };

  const ReferenceParameter* reference_parameter(size_t ordinal) const {
    for (const auto& reference : current_function_.reference_parameters) {
      if (reference.ordinal == ordinal) {
        return &reference;
      }
      if (reference.ordinal > ordinal) {
        break;
      }
    }
    return nullptr;
  }

  std::vector<const cxx::Type*> function_result_sources() {
    std::vector<const cxx::Type*> sources;
    const bool returns_receiver_alias =
        current_function_.implicit_object && current_function_.control &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value &&
        current_function_.control->returns_current_receiver_reference();
    if (!unit_.typeTraits().is_void(current_function_.return_type) &&
        !returns_receiver_alias) {
      sources.push_back(
          unit_.typeTraits().is_lvalue_reference(current_function_.return_type)
              ? unit_.typeTraits().add_pointer(
                    unit_.typeTraits().remove_reference(
                        current_function_.return_type))
              : current_function_.return_type);
    }
    if (current_function_.implicit_object &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value) {
      for (auto* field : current_function_.implicit_object->written_fields) {
        sources.push_back(types_.unqualified_value(field->type()));
      }
    }
    auto parameters = current_function_.source->symbol->parameters();
    for (const auto& reference : current_function_.reference_parameters) {
      if (reference.abi == ReferenceABI::Value && reference.mutable_access) {
        sources.push_back(
            types_.unqualified_value(parameters[reference.ordinal]->type()));
      }
    }
    return sources;
  }

  Returned complete_return(Returned returned) {
    if (current_function_.implicit_object &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value) {
      for (auto* field : current_function_.implicit_object->written_fields) {
        receiver_member(field, returned.source).append_to(returned.values);
      }
    }
    auto parameters = current_function_.source->symbol->parameters();
    for (const auto& reference : current_function_.reference_parameters) {
      if (reference.abi == ReferenceABI::Value && reference.mutable_access) {
        values_.at(parameters[reference.ordinal]).append_to(returned.values);
      }
    }
    return returned;
  }

  bool supports_function_exit() const {
    return current_function_.kind == FunctionKind::Ordinary ||
           current_function_.kind == FunctionKind::Kernel;
  }

  Returned return_values(cxx::ReturnStatementAST* statement) {
    bool returns_void =
        unit_.typeTraits().is_void(current_function_.return_type);
    if (!statement->expression) {
      if (!returns_void) {
        fail(statement, "non-void return requires a value");
      }
      return complete_return({{}, statement});
    }
    if (unit_.typeTraits().is_void(statement->expression->type)) {
      if (!returns_void) {
        fail(statement, "non-void return requires a value");
      }
      effect(statement->expression);
      return complete_return({{}, statement});
    }
    if (returns_void) {
      fail(statement, "void return cannot carry a value");
    }
    Returned returned = {{}, statement};
    if (unit_.typeTraits().is_lvalue_reference(current_function_.return_type)) {
      const bool returns_receiver_alias =
          current_function_.implicit_object && current_function_.control &&
          current_function_.implicit_object->abi == ImplicitObjectABI::Value &&
          current_function_.control->returns_current_receiver_reference();
      if (returns_receiver_alias) {
        IREE_ASSERT(is_current_object(statement->expression));
      } else {
        reference_value(statement->expression, statement)
            .append_to(returned.values);
      }
    } else {
      convert(statement->expression, current_function_.return_type, statement)
          .append_to(returned.values);
    }
    return complete_return(std::move(returned));
  }

  void function_exit(cxx::ReturnStatementAST* statement) {
    if (!supports_function_exit()) {
      fail(statement,
           "returns inside structured control require a runtime function or "
           "kernel body");
    }
    auto outer_values = values_;
    auto returned = return_values(statement);
    auto source = locations_.get(statement);
    auto condition = scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
    loom_op_t* op;
    check(loom_func_exit_build(&builder_, condition, source, &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_func_exit_body(op));
    loom_op_t* terminator;
    if (current_function_.kind == FunctionKind::Kernel) {
      check(loom_kernel_return_build(&builder_, source, &terminator));
    } else {
      check(loom_func_return_build(&builder_, returned.values.data(),
                                   returned.values.size(), source,
                                   &terminator));
    }
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
  }

  void function(FunctionInstance instance) {
    auto defined = functions_.define(instance, types_, locations_, &builder_);
    if (defined.configuration) {
      function_body(*defined.configuration);
    }
    function_body(defined.body);
  }

  void function_body(const FunctionBody& body_contract) {
    FastMathScope math_scope(lexical_fastmath_flags_,
                             fastmath_.function(body_contract.source->symbol)
                                 .value_or(default_fastmath_flags_));
    current_function_ = body_contract;
    const auto& defined = current_function_;
    auto* body = defined.body;
    auto* op = defined.operation;
    auto* region = defined.region;
    if (defined.kind == FunctionKind::CheckCase ||
        defined.kind == FunctionKind::CheckScenario) {
      auto saved = loom_builder_enter_region(&builder_, op, region);
      if (defined.kind == FunctionKind::CheckCase) {
        translate_check_case_body(unit_, diagnostics_, functions_, intrinsics_,
                                  types_, scalars_, locations_, builder_,
                                  defined);
      } else {
        translate_check_scenario_body(unit_, diagnostics_, functions_,
                                      intrinsics_, types_, scalars_, locations_,
                                      builder_, defined);
      }
      loom_builder_restore(&builder_, saved);
      return;
    }
    auto parameters = defined.source->symbol->parameters();
    owned_control_.reset();
    if (defined.control) {
      control_ = defined.control;
    } else {
      owned_control_.emplace(unit_, diagnostics_, types_, body,
                             defined.source->symbol);
      control_ = &*owned_control_;
    }
    auto saved = loom_builder_enter_region(&builder_, op, region);
    values_.clear();
    locals_.clear();
    immutable_storage_.clear();
    immutable_preamble_end_ = nullptr;
    references_.clear();
    receivers_.clear();
    receiver_owners_.clear();
    value_arena_.reset();
    size_t argument_index = 0;
    if (defined.implicit_object) {
      const auto& implicit = *defined.implicit_object;
      auto* record = types_.record(types_.unqualified(implicit.object_type),
                                   defined.source);
      if (!record) {
        fail(defined.source,
             "implicit object receiver requires an admitted record type");
      }
      if (implicit.abi == ImplicitObjectABI::Value) {
        auto value =
            name(region_value(region, argument_index, *record), "this");
        bind_value_receiver(*record, value);
      } else {
        auto pointer = name(
            region_value(region, argument_index, kPointerPartition), "this");
        bind_storage_receiver(
            *record,
            storage_.project(pointer.pointer(), implicit.object_type,
                             defined.source),
            implicit.object_type);
      }
    }
    size_t parameter_index = 0;
    for (auto* parameter : parameters) {
      const auto* reference = reference_parameter(parameter_index);
      auto* value_type =
          !reference ? types_.unqualified(parameter->type())
          : reference->abi == ReferenceABI::Storage
              ? unit_.typeTraits().add_pointer(
                    unit_.typeTraits().remove_reference(parameter->type()))
              : types_.unqualified_value(parameter->type());
      const auto& partition = types_.partition(value_type, defined.source);
      bool kernel_pointer = defined.kind == FunctionKind::Kernel &&
                            partition.kind == ValueKind::Pointer;
      // Only pointers change representation at kernel entry: the ABI supplies
      // one buffer, and source pointer transport adds its zero byte origin.
      auto value =
          defined.kind == FunctionKind::CommandProgram
              ? command_region_value(
                    region, *defined.command_program,
                    defined.command_program->parameters[parameter_index],
                    partition)
              : region_value(region, argument_index,
                             kernel_pointer ? kSSAPartition : partition);
      if (kernel_pointer) {
        auto buffer = value.ssa();
        if (!defined.parameter_contracts.empty()) {
          buffer = apply_parameter_contract(
              defined.parameter_contracts[parameter_index], buffer, locations_,
              &builder_);
        }
        value = storage_.root(buffer, defined.source);
      }
      value = name(value, cxx::to_string(parameter->name()));
      if (reference && reference->abi == ReferenceABI::Storage) {
        auto projection = storage_.project(
            value.pointer(),
            unit_.typeTraits().remove_reference(parameter->type()),
            defined.source);
        references_[parameter] = {
            ObjectLvalue{unit_.typeTraits().remove_reference(parameter->type()),
                         projection}};
        ++parameter_index;
        continue;
      }
      if (control_->addressed(parameter) ||
          unit_.typeTraits().is_volatile(parameter->type())) {
        auto access = allocate_local(parameter, 0, defined.source);
        if (partition.kind == ValueKind::SSA) {
          storage_.store(access, value.ssa(), parameter->type(),
                         defined.source);
        } else {
          objects_.store(local_object(parameter, defined.source), value,
                         parameter->type(), defined.source);
        }
      } else {
        values_[parameter] = value;
      }
      ++parameter_index;
    }
    auto returned =
        return_sequence({body->statementList, nullptr, lexical_fastmath_flags_,
                         control_->callable_exit_count(body)});
    if (defined.implicit_object) {
      leave_receiver();
    }
    loom_op_t* terminator;
    if (returned.unreachable) {
      check(loom_func_unreachable_build(
          &builder_, locations_.get(returned.source), &terminator));
    } else if (defined.kind == FunctionKind::CommandProgram) {
      check(loom_command_return_build(
          &builder_, locations_.get(returned.source), &terminator));
    } else if (defined.kind == FunctionKind::Kernel) {
      check(loom_kernel_return_build(&builder_, locations_.get(returned.source),
                                     &terminator));
    } else if (defined.kind == FunctionKind::TemplateDefinition) {
      check(loom_template_return_build(
          &builder_, returned.values.data(), returned.values.size(),
          locations_.get(returned.source), &terminator));
    } else if (defined.kind == FunctionKind::LaunchConfiguration ||
               defined.kind == FunctionKind::ClusteredLaunchConfiguration) {
      auto source = locations_.get(returned.source);
      std::array<loom_value_id_t, 9> dimensions = {};
      auto i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
      auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
      auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      for (size_t i = 0; i < returned.values.size(); ++i) {
        loom_op_t* extended;
        check(loom_scalar_extui_build(&builder_, returned.values[i], i32_type,
                                      i64_type, source, &extended));
        check(loom_index_cast_build(&builder_, loom_op_results(extended)[0],
                                    i64_type, index_type, source, &terminator));
        dimensions[i] = result(terminator);
      }
      bool clustered =
          defined.kind == FunctionKind::ClusteredLaunchConfiguration;
      loom_kernel_launch_config_build_flags_t flags =
          clustered
              ? LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_X |
                    LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Y |
                    LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Z
              : 0;
      check(loom_kernel_launch_config_build(
          &builder_, flags, dimensions[0], dimensions[1], dimensions[2],
          dimensions[3], dimensions[4], dimensions[5],
          clustered ? dimensions[6] : 0, clustered ? dimensions[7] : 0,
          clustered ? dimensions[8] : 0, source, &terminator));
    } else {
      check(loom_func_return_build(
          &builder_, returned.values.data(), returned.values.size(),
          locations_.get(returned.source), &terminator));
    }
    loom_builder_restore(&builder_, saved);
  }

  Returned return_sequence(ReturnSequence sequence) {
    FastMathScope math_scope(lexical_fastmath_flags_, sequence.fastmath);
    for (;;) {
      if (!sequence.remaining) {
        if (sequence.continuation) {
          return return_sequence(*sequence.continuation);
        }
        if (!unit_.typeTraits().is_void(current_function_.return_type)) {
          fail(current_function_.source,
               "non-void function can reach its end without returning a value");
        }
        return complete_return({{}, current_function_.source});
      }
      auto* child = sequence.remaining->value;
      sequence.remaining = sequence.remaining->next;
      if (control_->requires_callable_exit(child)) {
        IREE_ASSERT_GT(sequence.callable_exit_count, 0);
        --sequence.callable_exit_count;
      }
      if (control_->returns(child) != ExitFlow::None) {
        return returning_statement(child, sequence);
      }
      statement(child);
      if (supports_function_exit() && !control_->falls_through(child)) {
        return {{}, child, true};
      }
    }
  }

  StorageAccess allocate_local(
      cxx::Symbol* symbol, int64_t alignment, cxx::AST* owner,
      const ObjectRepresentation* initializer = nullptr) {
    auto allocation =
        storage_.allocate(symbol->type(), LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
                          alignment, owner, initializer);
    auto spelling = cxx::to_string(symbol->name());
    name(Value(allocation.pointer), spelling + "_storage");
    name(allocation.view, spelling + "_view");
    locals_[symbol] = allocation;
    return {allocation.view, std::nullopt};
  }

  StorageProjection string_literal(cxx::StringLiteralExpressionAST* literal) {
    auto contents = initialization_.literal(literal->type, literal, literal);
    auto allocation = storage_.constant(literal->type, *contents, literal);
    return storage_.project(allocation.pointer, literal->type, literal);
  }

  StorageProjection immutable_object(cxx::VariableSymbol* variable,
                                     cxx::AST* owner) {
    auto* canonical = variable->canonical();
    auto object = immutable_objects_.reference(canonical, owner);
    auto found = immutable_storage_.find(canonical);
    if (found == immutable_storage_.end()) {
      auto saved = loom_builder_enter_region(
          &builder_, current_function_.operation, current_function_.region);
      auto* entry = loom_region_entry_block(current_function_.region);
      if (immutable_preamble_end_) {
        loom_builder_set_after(&builder_, immutable_preamble_end_);
      } else if (entry->first_op) {
        loom_builder_set_before(&builder_, entry->first_op);
      }
      auto type = loom_type_buffer();
      loom_op_t* op;
      check(loom_global_load_build(&builder_, object.symbol, &type, 1,
                                   locations_.get(owner), &op));
      auto allocation = storage_.admit_fixed(
          loom_global_load_result(op).values[0], object.type,
          LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT, owner);
      std::string spelling(object.value_name.data, object.value_name.size);
      name(Value(allocation.pointer), spelling + "_storage");
      name(allocation.view, spelling + "_view");
      immutable_preamble_end_ =
          loom_value_def_op(loom_module_value(module_, allocation.view));
      found = immutable_storage_.emplace(canonical, allocation).first;
      loom_builder_restore(&builder_, saved);
    }
    return storage_.project(found->second.pointer, object.type, owner);
  }

  Value binding(cxx::Symbol* symbol, cxx::AST* owner) {
    auto found = locals_.find(symbol);
    if (found == locals_.end()) {
      return values_.at(symbol);
    }
    if (unit_.typeTraits().is_array(symbol->type())) {
      return found->second.pointer;
    }
    auto value =
        unit_.typeTraits().is_class(symbol->type())
            ? objects_.load(local_object(symbol, owner), symbol->type(), owner)
            : Value(storage_.load({found->second.view, std::nullopt},
                                  symbol->type(), owner));
    return name(value, cxx::to_string(symbol->name()));
  }

  StorageProjection local_object(cxx::Symbol* symbol, cxx::AST* owner) {
    return storage_.project(locals_.at(symbol).pointer, symbol->type(), owner);
  }

  Value apply_constexpr_contract(Value value, ConstexprContract contract) {
    if (!contract) {
      return value;
    }
    auto required = cxx_import::apply_constexpr_contract(contract, value.ssa(),
                                                         locations_, &builder_);
    std::array<loom_value_id_t, 1> components = {required};
    return value_arena_.capture(value.partition(), components);
  }

  void initialize_variable(cxx::VariableSymbol* variable,
                           cxx::ExpressionAST* initializer,
                           ConstexprContract constexpr_contract,
                           cxx::AST* owner) {
    if (unit_.typeTraits().is_lvalue_reference(variable->type())) {
      if (!initializer) {
        fail(owner, "reference locals require an initializer");
      }
      cxx::Initializer source(initializer);
      auto* referent_expression = source.singleExpression();
      if (!referent_expression) {
        fail(initializer,
             "reference locals require one initializer expression");
      }
      auto* referent = unit_.typeTraits().remove_reference(variable->type());
      references_.emplace(variable,
                          reference_lvalue(referent_expression, owner,
                                           unit_.typeTraits().is_const(referent)
                                               ? LvalueAccess::Read
                                               : LvalueAccess::Write));
      return;
    }
    if (unit_.typeTraits().is_array(variable->type()) ||
        control_->addressed(variable) ||
        unit_.typeTraits().is_volatile(variable->type())) {
      auto contents =
          unit_.typeTraits().is_volatile(variable->type())
              ? std::nullopt
              : initialization_.literal(variable->type(), initializer, owner);
      allocate_local(variable, variable->explicitAlignment(), owner,
                     contents ? &*contents : nullptr);
      if (contents) {
        return;
      }
      if (initializer) {
        if (types_.address_domain(types_.unqualified(variable->type()),
                                  owner) == LOOM_SCALAR_TYPE_NONE) {
          types_.admit_construction(variable->constructor(), variable->type(),
                                    owner);
        }
        initialization_.storage(locals_.at(variable), variable->type(),
                                initializer, variable->constructor(), owner);
      }
      return;
    }
    if (!initializer) {
      const auto& partition = types_.partition(variable->type(), owner);
      if (partition.kind != ValueKind::SSA) {
        fail(owner,
             "uninitialized locals require scalar or native vector object "
             "storage");
      }
      allocate_local(variable, variable->explicitAlignment(), owner);
      return;
    }
    types_.partition(variable->type(), owner);
    if (types_.address_domain(types_.unqualified(variable->type()), owner) ==
        LOOM_SCALAR_TYPE_NONE) {
      types_.admit_construction(variable->constructor(), variable->type(),
                                owner);
    }
    auto value =
        apply_constexpr_contract(expression(initializer), constexpr_contract);
    values_[variable] = name(value, cxx::to_string(variable->name()));
  }

  ConstexprContract initialize_condition(cxx::ExpressionAST* expression) {
    auto* declaration = condition_declaration(expression);
    if (!declaration) {
      return {};
    }
    auto* variable = declaration->symbol;
    reject_misplaced_binding_attributes(
        unit_, diagnostics_, declaration->attributeList,
        BindingAttributeScope::LocalDeclaration);
    reject_misplaced_binding_declarator(unit_, diagnostics_,
                                        declaration->declarator);
    auto constexpr_contract = admit_constexpr_contract(
        unit_, diagnostics_, types_, declaration->attributeList, variable,
        control_->addressed(variable), declaration);
    if (variable->isStatic() || variable->isExtern() ||
        variable->isThreadLocal()) {
      fail(declaration, "condition storage duration must be automatic");
    }
    initialize_variable(variable, declaration->initializer, constexpr_contract,
                        declaration);
    return constexpr_contract;
  }

  void reinitialize_condition(cxx::ConditionExpressionAST* declaration,
                              ConstexprContract constexpr_contract,
                              cxx::AST* owner) {
    auto* decision = declaration->symbol;
    auto value = apply_constexpr_contract(expression(declaration->initializer),
                                          constexpr_contract);
    if (auto found = locals_.find(decision); found != locals_.end()) {
      storage_.store({found->second.view, std::nullopt}, value.ssa(),
                     decision->type(), owner);
    } else {
      values_[decision] = name(value, cxx::to_string(decision->name()));
    }
  }

  // Source-selected branches are transparent to every statement/exit path.
  // Their initializers still execute once; a discarded arm contributes no IR.
  cxx::StatementAST* selected_statement(cxx::StatementAST* ast) {
    while (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      if (!branch->constexprValue.has_value()) {
        break;
      }
      if (branch->initializer) {
        statement(branch->initializer);
      }
      initialize_condition(branch->condition);
      ast = *branch->constexprValue ? branch->statement : branch->elseStatement;
      if (ast) {
        if (auto contract = fastmath_.statement(ast)) {
          lexical_fastmath_flags_ = *contract;
        }
      }
    }
    return ast;
  }

  loom_value_id_t branch_condition(cxx::IfStatementAST* branch) {
    if (branch->initializer) {
      statement(branch->initializer);
    }
    initialize_condition(branch->condition);
    return expression(branch->condition).ssa();
  }

  Returned returning_statement(cxx::StatementAST* ast,
                               const ReturnSequence& continuation) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    if (auto* ret = cxx::ast_cast<cxx::ReturnStatementAST>(ast)) {
      return return_values(ret);
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return return_sequence({
          compound->statementList,
          &continuation,
          lexical_fastmath_flags_,
          control_->callable_exit_count(compound) +
              continuation.callable_exit_count,
      });
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && control_->returns(branch) != ExitFlow::None) {
      if ((control_->returns(branch->statement) != ExitFlow::All &&
           control_->returns(branch->elseStatement) != ExitFlow::All) ||
          control_->requires_callable_exit(branch) ||
          continuation.callable_exit_count != 0) {
        if (!supports_function_exit()) {
          fail(ast,
               "returns inside structured control require a runtime function "
               "or kernel body");
        }
        statement(ast);
        if (!control_->falls_through(ast)) {
          return {{}, ast, true};
        }
        return return_sequence(continuation);
      }
      auto condition = branch_condition(branch);
      auto outer_values = values_;
      auto result_sources = function_result_sources();
      auto result_signature =
          bind_signature(types_, result_sources, ast, &builder_);
      loom_op_t* op;
      auto source = locations_.get(ast);
      check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                              condition, result_signature.types.data(),
                              result_signature.types.size(), nullptr, 0, source,
                              &op));
      auto saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
      auto returned = returning_statement(branch->statement, continuation);
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, returned.values.data(),
                                 returned.values.size(),
                                 locations_.get(returned.source), &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;
      saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
      returned = returning_statement(branch->elseStatement, continuation);
      check(loom_scf_yield_build(&builder_, returned.values.data(),
                                 returned.values.size(),
                                 locations_.get(returned.source), &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;
      returned.values.assign(
          loom_op_results(op),
          loom_op_results(op) + result_signature.types.size());
      returned.source = ast;
      return returned;
    }
    if (control_->returns(ast) != ExitFlow::None) {
      if (!supports_function_exit()) {
        fail(ast,
             "returns inside structured control require a runtime function or "
             "kernel body");
      }
      statement(ast);
      if (!control_->falls_through(ast)) {
        return {{}, ast, true};
      }
      return return_sequence(continuation);
    }
    if (ast) {
      statement(ast);
    }
    return return_sequence(continuation);
  }

  StorageProjection pointer_projection(Value value, const cxx::Type* type,
                                       cxx::AST* owner) {
    auto* pointer = cxx::type_cast<cxx::PointerType>(types_.unqualified(type));
    return storage_.project(value.pointer(),
                            pointer ? pointer->elementType() : type, owner);
  }

  StorageProjection pointer_expression(cxx::ExpressionAST* ast) {
    auto* source = strip_implicit_casts(ast);
    auto* source_type = unit_.typeTraits().remove_reference(source->type);
    if (unit_.typeTraits().is_array(source_type)) {
      // Direct array indexing retains its enclosing lvalue's alignment. An
      // actual pointer value, including an explicit cast or call result, starts
      // with its pointee contract and carries no hidden alignment component.
      auto projection = object_address(source);
      if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(source)) {
        name(projection.pointer, cxx::to_string(member->symbol->name()));
      }
      return projection;
    }
    return pointer_projection(expression(ast), ast->type, ast);
  }

  struct SubscriptOperands {
    // Evaluated array or pointer origin with its source alignment.
    StorageProjection base;
    // Evaluated integral displacement, before address-width conversion.
    loom_value_id_t index;
    // Source array or pointer type owning the element stride.
    const cxx::Type* base_type;
    // Source displacement type owning its signedness and width.
    const cxx::Type* index_type;
  };

  SubscriptOperands subscript_operands(cxx::SubscriptExpressionAST* ast) {
    if (ast->symbol) {
      fail(ast, "overloaded indexing is not admitted");
    }
    auto* base_type =
        unit_.typeTraits().remove_reference(ast->baseExpression->type);
    auto* index_type =
        unit_.typeTraits().remove_reference(ast->indexExpression->type);
    if (types_.native_vector(base_type) || types_.native_vector(index_type)) {
      fail(ast, "vector lane subscripts require a value projection");
    }
    if (unit_.typeTraits().is_pointer(base_type) ||
        unit_.typeTraits().is_array(base_type)) {
      auto base = pointer_expression(ast->baseExpression);
      auto index = expression(ast->indexExpression);
      return {base, index.ssa(), base_type, index_type};
    }
    // The commuted spelling i[p] still evaluates its written base first.
    auto index = expression(ast->baseExpression);
    auto base = pointer_expression(ast->indexExpression);
    return {base, index.ssa(), index_type, base_type};
  }

  StorageAccess address(cxx::ExpressionAST* ast) {
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return address(nested->expression);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      auto found = locals_.find(id->symbol);
      if (found != locals_.end()) {
        return {found->second.view, std::nullopt};
      }
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      auto operands = subscript_operands(subscript);
      return storage_.subscript(operands.base, operands.index,
                                operands.base_type, operands.index_type, ast);
    }
    return storage_.dereference(object_address(ast), ast->type, ast);
  }

  std::optional<StorageProjection> reference_field_address(
      cxx::ExpressionAST* ast) {
    ast = strip_implicit_casts(ast);
    cxx::FieldSymbol* field = nullptr;
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
      if (!field || !receiver(field)) {
        return std::nullopt;
      }
    } else if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (!field) {
        return std::nullopt;
      }
      if (!unit_.typeTraits().is_lvalue_reference(field->type())) {
        return std::nullopt;
      }
      if (auto* active = receiver(field);
          active && is_current_object(member->baseExpression)) {
        auto pointer = receiver_member(field, ast);
        return storage_.project(
            pointer.pointer(),
            unit_.typeTraits().remove_reference(field->type()), ast);
      }
      if (control_->storage_backed(member)) {
        fail(ast,
             "reference-bearing records cannot be loaded from physical "
             "object storage");
      }
      auto base = expression(member->baseExpression);
      const auto& slice = types_.member(field, ast);
      auto pointer = base.project(*slice.partition, slice.component_offset);
      return storage_.project(
          pointer.pointer(), unit_.typeTraits().remove_reference(field->type()),
          ast);
    }
    if (!field || !unit_.typeTraits().is_lvalue_reference(field->type())) {
      return std::nullopt;
    }
    auto pointer = receiver_member(field, ast);
    return storage_.project(pointer.pointer(),
                            unit_.typeTraits().remove_reference(field->type()),
                            ast);
  }

  StorageProjection object_address(cxx::ExpressionAST* ast) {
    ast = strip_implicit_casts(ast);
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return object_address(nested->expression);
    }
    if (auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(ast)) {
      return string_literal(literal);
    }
    if (auto reference = reference_field_address(ast)) {
      return *reference;
    }
    if (auto* field = current_receiver_field(ast)) {
      if (auto projection = receiver_storage(field, ast)) {
        return *projection;
      }
      fail(ast,
           "addressing a member during initialization requires addressable "
           "record storage");
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = references_.find(id->symbol);
          found != references_.end()) {
        const auto* object =
            std::get_if<ObjectLvalue>(&found->second.projection);
        auto* projection =
            object ? std::get_if<StorageProjection>(&object->location)
                   : nullptr;
        if (!projection) {
          fail(ast,
               "reference identity escaped an automatic value without "
               "addressable storage");
        }
        return *projection;
      }
      if (auto found = locals_.find(id->symbol); found != locals_.end()) {
        return storage_.project(found->second.pointer, id->type, id);
      }
      if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol);
          variable &&
          cxx::symbol_cast<cxx::NamespaceSymbol>(
              variable->canonical()->parent()) &&
          !configs_.contains(variable)) {
        return immutable_object(variable, id);
      }
      if (unit_.typeTraits().is_array(id->type)) {
        return storage_.project(expression(id).pointer(), id->type, id);
      }
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (!field || field->isStatic()) {
        fail(ast, "record storage access requires a non-static data member");
      }
      auto base = member->accessOp == cxx::TokenKind::T_MINUS_GREATER
                      ? pointer_expression(member->baseExpression)
                      : object_address(member->baseExpression);
      return storage_.member(base, field, ast);
    }
    if (auto call = direct_operator_call(ast)) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(call->function->type());
      if (!signature ||
          !unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
        fail(ast, "addressed operator must return an lvalue reference");
      }
      auto value = ordinary_operator_call(ast, *call);
      if (!value || !value->is_pointer()) {
        fail(ast, "reference-returning operator lost its object identity");
      }
      return storage_.project(
          value->pointer(),
          unit_.typeTraits().remove_reference(signature->returnType()), ast);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      if (subscript->symbol) {
        auto* signature =
            cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
        if (!signature ||
            !unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
          fail(ast, "addressed overloaded indexing must return a reference");
        }
        auto value = overloaded_subscript_call(subscript);
        if (!value || !value->is_pointer()) {
          fail(ast, "reference-returning subscript lost its object identity");
        }
        return storage_.project(
            value->pointer(),
            unit_.typeTraits().remove_reference(signature->returnType()), ast);
      }
      auto operands = subscript_operands(subscript);
      return storage_.advance(operands.base, operands.index, operands.base_type,
                              operands.index_type, cxx::TokenKind::T_PLUS, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        return pointer_expression(unary->expression);
      }
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      auto* function = direct_callee(call);
      auto* signature =
          function ? cxx::type_cast<cxx::FunctionType>(function->type())
                   : nullptr;
      if (signature &&
          unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
        auto value = ordinary_call(call, function, locations_.get(call));
        if (!value || !value->is_pointer()) {
          fail(ast, "reference-returning call lost its object identity");
        }
        return storage_.project(
            value->pointer(),
            unit_.typeTraits().remove_reference(signature->returnType()), ast);
      }
    }
    if (ast->valueCategory == cxx::ValueCategory::kPrValue &&
        !unit_.typeTraits().is_array(ast->type)) {
      auto value = expression(ast);
      auto allocation = storage_.allocate(
          ast->type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, ast);
      auto object = storage_.project(allocation.pointer, ast->type, ast);
      if (unit_.typeTraits().is_class(ast->type)) {
        objects_.store(object, value, ast->type, ast);
      } else {
        storage_.store(storage_.dereference(object, ast->type, ast),
                       value.ssa(), ast->type, ast);
      }
      return object;
    }
    fail(ast, "address-of requires an existing storage-backed element");
  }

  Value load(cxx::ExpressionAST* ast) {
    // An array lvalue denotes borrowed storage. Decay and subsequent element
    // projections preserve that origin without loading or copying the array.
    if (unit_.typeTraits().is_array(ast->type)) {
      return object_address(ast).pointer;
    }
    if (unit_.typeTraits().is_class(ast->type)) {
      return objects_.load(object_address(ast), ast->type, ast);
    }
    return storage_.load(address(ast), ast->type, ast);
  }

  enum class IncrementResult { Previous, Updated };

  // One complete object selected by an automatic binding or evaluated memory
  // address. Reads and writes share this projection even when another binding
  // changes.
  struct ObjectLvalue {
    // Source object type retains qualifiers and storage access rules.
    const cxx::Type* type;
    // An automatic SSA partition or an evaluated storage projection.
    std::variant<Destination, StorageAccess, StorageProjection> location;
  };

  // One lane projected from its complete vector lvalue. The source base and
  // index have both been evaluated exactly once.
  struct VectorLaneLvalue {
    // Complete vector destination receiving the functional lane update.
    ObjectLvalue destination;
    // Vector snapshot used by one immediate compound operation. A durable C++
    // reference has no snapshot and observes the current destination value on
    // every later use while preserving its once-evaluated lane selector.
    std::optional<Value> value;
    // Lane selector before projection to Loom's index domain.
    loom_value_id_t index;
    // Source selector type retaining its integer width and signedness.
    const cxx::Type* index_type;
  };

  // Resolving a source lvalue executes every address and index expression once.
  // An immediate lane operation may retain the complete vector snapshot;
  // durable references instead reload their destination on each later use.
  struct Lvalue {
    // Complete object or one evaluated lane within a complete vector object.
    std::variant<ObjectLvalue, VectorLaneLvalue> projection;
  };

  enum class LvalueAccess { Read, Write };

  ObjectLvalue object_destination(
      cxx::ExpressionAST* expression, cxx::AST* owner,
      LvalueAccess access = LvalueAccess::Write,
      std::optional<Destination> retained_destination = std::nullopt,
      bool preserve_identity = false) {
    if (access == LvalueAccess::Write) {
      types_.require_mutable(expression->type, expression);
    }
    auto destination = retained_destination ? retained_destination
                                            : control_->destination(expression);
    if (destination) {
      if (auto* field =
              cxx::symbol_cast<cxx::FieldSymbol>(destination->binding);
          field && receiver(field)) {
        auto* active = receiver(field);
        if (!receiver_owners_.at(field).complete) {
          fail(owner,
               "member is mutated before its initialization is complete");
        }
        if (active->storage) {
          auto traits = unit_.typeTraits();
          auto* target_type = traits.is_volatile(active->object_type)
                                  ? traits.add_volatile(expression->type)
                                  : expression->type;
          if (preserve_identity || traits.is_class(target_type) ||
              traits.is_array(target_type)) {
            return {target_type, object_address(expression)};
          }
          return {target_type, address(expression)};
        }
      }
      if (!locals_.contains(destination->binding)) {
        if (references_.contains(destination->binding)) {
          auto traits = unit_.typeTraits();
          if (preserve_identity || traits.is_class(expression->type) ||
              traits.is_array(expression->type)) {
            return {expression->type, object_address(expression)};
          }
          return {expression->type, address(expression)};
        }
        if (!values_.contains(destination->binding)) {
          fail(owner, "mutation requires an owned automatic source binding");
        }
        return {expression->type, *destination};
      }
    }
    if (preserve_identity || unit_.typeTraits().is_class(expression->type) ||
        unit_.typeTraits().is_array(expression->type)) {
      return {expression->type, object_address(expression)};
    }
    return {expression->type, address(expression)};
  }

  Value read(const ObjectLvalue& target, cxx::AST* owner) {
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      auto value = values_.at(destination->binding);
      return destination->member ? value.project(*destination->member,
                                                 destination->component_offset)
                                 : value;
    }
    if (auto* object = std::get_if<StorageProjection>(&target.location)) {
      return objects_.load(*object, target.type, owner);
    }
    return storage_.load(std::get<StorageAccess>(target.location), target.type,
                         owner);
  }

  void write(const ObjectLvalue& target, Value value, cxx::AST* owner) {
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      auto& binding = values_.at(destination->binding);
      binding = name(destination->member
                         ? value_arena_.replace(
                               binding, destination->component_offset, value)
                         : value,
                     cxx::to_string(destination->binding->name()));
    } else if (auto* object =
                   std::get_if<StorageProjection>(&target.location)) {
      objects_.store(*object, value, target.type, owner);
    } else {
      storage_.store(std::get<StorageAccess>(target.location), value.ssa(),
                     target.type, owner);
    }
  }

  Lvalue vector_lane_destination(
      cxx::SubscriptExpressionAST* subscript, cxx::AST* owner,
      std::optional<Destination> retained_destination,
      LvalueAccess access = LvalueAccess::Write, bool retain_value = true) {
    if (subscript->symbol) {
      fail(subscript, "overloaded indexing is not admitted");
    }
    auto* left_type = subscript->baseExpression->type;
    auto* right_type = subscript->indexExpression->type;
    bool left_vector = types_.native_vector(left_type);
    bool right_vector = types_.native_vector(right_type);
    if (!left_vector && !right_vector) {
      fail(subscript, "vector lane destination requires a vector operand");
    }

    cxx::ExpressionAST* vector_expression =
        left_vector ? subscript->baseExpression : subscript->indexExpression;
    cxx::ExpressionAST* index_expression =
        left_vector ? subscript->indexExpression : subscript->baseExpression;
    if (left_vector) {
      auto vector_destination = object_destination(
          vector_expression, owner, access, retained_destination);
      auto index_value = expression(index_expression);
      std::optional<Value> vector_value;
      if (retain_value) {
        vector_value = read(vector_destination, vector_expression);
      }
      return {VectorLaneLvalue{vector_destination, std::move(vector_value),
                               index_value.ssa(), index_expression->type}};
    }

    // The commuted spelling i[v] still evaluates its written left operand
    // before resolving and loading the vector on the right.
    auto index_value = expression(index_expression);
    auto vector_destination = object_destination(vector_expression, owner,
                                                 access, retained_destination);
    std::optional<Value> vector_value;
    if (retain_value) {
      vector_value = read(vector_destination, vector_expression);
    }
    return {VectorLaneLvalue{vector_destination, std::move(vector_value),
                             index_value.ssa(), index_expression->type}};
  }

  Lvalue destination(cxx::ExpressionAST* expression, cxx::AST* owner,
                     LvalueAccess access = LvalueAccess::Write,
                     bool preserve_identity = false) {
    auto* projected = strip_implicit_casts(expression);
    while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(projected)) {
      projected = strip_implicit_casts(nested->expression);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(projected)) {
      if (auto found = references_.find(id->symbol);
          found != references_.end()) {
        if (access == LvalueAccess::Write) {
          types_.require_mutable(projected->type, owner);
        }
        return found->second;
      }
    }
    if (auto reference = reference_field_address(projected)) {
      if (access == LvalueAccess::Write) {
        types_.require_mutable(projected->type, owner);
      }
      return {ObjectLvalue{projected->type, *reference}};
    }
    if ((cxx::ast_cast<cxx::CallExpressionAST>(projected) ||
         (cxx::ast_cast<cxx::SubscriptExpressionAST>(projected) &&
          cxx::ast_cast<cxx::SubscriptExpressionAST>(projected)->symbol)) &&
        projected->valueCategory == cxx::ValueCategory::kLValue) {
      if (access == LvalueAccess::Write) {
        types_.require_mutable(projected->type, owner);
      }
      return {ObjectLvalue{projected->type, object_address(projected)}};
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(projected);
        subscript && (types_.native_vector(subscript->baseExpression->type) ||
                      types_.native_vector(subscript->indexExpression->type))) {
      return vector_lane_destination(subscript, owner,
                                     control_->destination(expression), access);
    }
    return {object_destination(expression, owner, access, std::nullopt,
                               preserve_identity)};
  }

  Lvalue reference_lvalue(cxx::ExpressionAST* expression, cxx::AST* owner,
                          LvalueAccess access) {
    auto* projected = unwrap_expression(expression);
    if (auto* conditional =
            cxx::ast_cast<cxx::ConditionalExpressionAST>(projected)) {
      auto condition = this->expression(conditional->condition).ssa();
      auto select = conditional_reference(conditional, condition);
      return {ObjectLvalue{
          projected->type,
          storage_.project(select.pointer(), projected->type, owner)}};
    }
    auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(projected);
    const bool vector_lane =
        subscript && (types_.native_vector(subscript->baseExpression->type) ||
                      types_.native_vector(subscript->indexExpression->type));
    auto target =
        vector_lane ? vector_lane_destination(subscript, owner,
                                              control_->destination(projected),
                                              access, false)
                    : destination(projected, owner, access, true);
    return target;
  }

  Value reference_value(cxx::ExpressionAST* expression, cxx::AST* owner) {
    auto target = reference_lvalue(expression, owner, LvalueAccess::Read);
    auto* object = std::get_if<ObjectLvalue>(&target.projection);
    auto* projection =
        object ? std::get_if<StorageProjection>(&object->location) : nullptr;
    if (!projection) {
      fail(owner,
           "reference identity escaped an automatic value without "
           "addressable storage");
    }
    return Value(projection->pointer);
  }

  Value conditional_reference(cxx::ConditionalExpressionAST* ast,
                              loom_value_id_t condition) {
    auto written = live_mutations(ast);
    auto* pointer_type = unit_.typeTraits().add_pointer(ast->type);
    std::vector<const cxx::Type*> sources = {pointer_type};
    sources.reserve(1 + written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto initial = current(written);
    auto source = locations_.get(ast);
    auto outputs = bind_signature(types_, sources, ast, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
    auto build_arm = [&](loom_region_t* region,
                         cxx::ExpressionAST* expression) {
      auto saved = loom_builder_enter_region(&builder_, op, region);
      std::vector<loom_value_id_t> yielded;
      Value(object_address(expression).pointer).append_to(yielded);
      auto mutations = current(written);
      yielded.insert(yielded.end(), mutations.begin(), mutations.end());
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 source, &yield));
      loom_builder_restore(&builder_, saved);
    };
    build_arm(loom_scf_if_then_region(op), ast->iftrueExpression);
    bind_values(written, initial.data());
    build_arm(loom_scf_if_else_region(op), ast->iffalseExpression);
    bind_values(written,
                loom_op_results(op) + kPointerPartition.component_count);
    return value_arena_.capture(
        kPointerPartition,
        {loom_op_results(op), kPointerPartition.component_count});
  }

  Value read(const Lvalue& target, cxx::AST* owner) {
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      return read(*object, owner);
    }
    const auto& lane = std::get<VectorLaneLvalue>(target.projection);
    auto vector = lane.value ? *lane.value : read(lane.destination, owner);
    return Value(vectors_.extract(vector.ssa(), lane.index,
                                  lane.destination.type, lane.index_type,
                                  owner));
  }

  void write(const Lvalue& target, Value value, cxx::AST* owner) {
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      write(*object, value, owner);
      return;
    }
    const auto& lane = std::get<VectorLaneLvalue>(target.projection);
    auto vector = lane.value ? *lane.value : read(lane.destination, owner);
    auto updated =
        vectors_.insert(vector.ssa(), value.ssa(), lane.index,
                        lane.destination.type, lane.index_type, owner);
    write(lane.destination, Value(updated), owner);
  }

  struct ReferenceOrigin {
    // Automatic object that owns this flattened component interval.
    cxx::Symbol* binding;
    // Inclusive first flattened component within the owning object.
    size_t begin;
    // Exclusive end of the flattened component interval.
    size_t end;
    // Whether the callee may replace this interval.
    bool mutable_access;
  };

  struct ReferenceWriteback {
    // Once-evaluated lvalue receiving the mutable result.
    Lvalue target;
    // Admitted value representation transported through the call.
    const Partition* partition;
    // Unqualified source value type used to bind the tied result.
    const cxx::Type* value_type;
    // First flattened call operand for this source parameter.
    size_t argument_offset;
  };

  // One source argument evaluated exactly once, before its components are
  // arranged in callable ABI order. A mutable value reference retains its
  // destination until the tied call results are available.
  struct EvaluatedCallArgument {
    // Source value or storage pointer transported through the call.
    Value value;
    // Mutable automatic lvalue receiving a value-reference writeback.
    std::optional<Lvalue> writeback_target;
    // Admitted value structure of a mutable value reference.
    const Partition* writeback_partition = nullptr;
    // Unqualified value type of a mutable value reference.
    const cxx::Type* writeback_type = nullptr;
    // Automatic component interval participating in alias admission.
    std::optional<ReferenceOrigin> origin;
  };

  ReferenceOrigin reference_origin(const Lvalue& target, const Value& value,
                                   bool mutable_access, cxx::AST* owner) {
    const ObjectLvalue* object = std::get_if<ObjectLvalue>(&target.projection);
    size_t component_count = value.partition().component_count;
    if (!object) {
      const auto& lane = std::get<VectorLaneLvalue>(target.projection);
      object = &lane.destination;
      component_count =
          types_.partition(lane.destination.type, owner).component_count;
    }
    auto* destination = std::get_if<Destination>(&object->location);
    if (!destination) {
      fail(owner,
           "runtime value-reference arguments require an automatic SSA "
           "object");
    }
    return {
        destination->binding,
        destination->component_offset,
        destination->component_offset + component_count,
        mutable_access,
    };
  }

  void require_nonoverlapping_references(std::vector<ReferenceOrigin> origins,
                                         cxx::AST* owner) {
    if (std::ranges::none_of(origins, [](const ReferenceOrigin& origin) {
          return origin.mutable_access;
        })) {
      return;
    }
    std::ranges::sort(
        origins, [](const ReferenceOrigin& left, const ReferenceOrigin& right) {
          if (left.binding != right.binding) {
            return std::less<cxx::Symbol*>{}(left.binding, right.binding);
          }
          return left.begin < right.begin;
        });
    cxx::Symbol* binding = nullptr;
    size_t active_end = 0;
    bool active_mutable = false;
    for (const auto& origin : origins) {
      if (origin.begin == origin.end) {
        continue;
      }
      if (origin.binding != binding || origin.begin >= active_end) {
        binding = origin.binding;
        active_end = origin.end;
        active_mutable = origin.mutable_access;
        continue;
      }
      if (origin.mutable_access || active_mutable) {
        fail(owner,
             "mutable value-reference arguments must select nonoverlapping "
             "automatic objects or member slices");
      }
      active_end = std::max(active_end, origin.end);
    }
  }

  EvaluatedCallArgument evaluate_call_argument(
      cxx::ExpressionAST* argument, cxx::ParameterSymbol* parameter,
      const ReferenceParameter* reference, cxx::AST* owner) {
    if (!reference) {
      return {.value = expression(argument)};
    }
    auto* expected_type = types_.unqualified_value(parameter->type());
    const auto& expected_partition =
        reference->abi == ReferenceABI::Storage
            ? kPointerPartition
            : types_.partition(expected_type, owner);
    if (reference->abi == ReferenceABI::Storage) {
      return {.value = reference_value(argument, argument)};
    }
    std::optional<Lvalue> target;
    Value value;
    std::optional<ReferenceOrigin> origin;
    if (argument->valueCategory == cxx::ValueCategory::kLValue) {
      target = destination(
          argument, owner,
          reference->mutable_access ? LvalueAccess::Write : LvalueAccess::Read);
      value = read(*target, argument);
      origin =
          reference_origin(*target, value, reference->mutable_access, argument);
    } else {
      if (reference->mutable_access) {
        fail(argument, "mutable value references require an lvalue argument");
      }
      value = expression(argument);
    }
    if (&value.partition() != &expected_partition) {
      fail(argument,
           "runtime value-reference binding must preserve the admitted "
           "source value type");
    }
    return {
        .value = value,
        .writeback_target =
            reference->mutable_access ? std::move(target) : std::nullopt,
        .writeback_partition =
            reference->mutable_access ? &expected_partition : nullptr,
        .writeback_type = reference->mutable_access ? expected_type : nullptr,
        .origin = origin,
    };
  }

  void append_call_argument(EvaluatedCallArgument argument,
                            size_t receiver_argument_count,
                            std::vector<loom_value_id_t>& arguments,
                            std::vector<ReferenceWriteback>& writebacks,
                            std::vector<ReferenceOrigin>& origins) {
    const size_t argument_offset = receiver_argument_count + arguments.size();
    argument.value.append_to(arguments);
    if (argument.origin) {
      origins.push_back(*argument.origin);
    }
    if (argument.writeback_target) {
      writebacks.push_back({std::move(*argument.writeback_target),
                            argument.writeback_partition,
                            argument.writeback_type, argument_offset});
    }
  }

  struct ReceiverArgument {
    // Native callable specialization selected by receiver ownership.
    ImplicitObjectABI abi;
    // Prvalue record or resolved storage pointer, when evaluated eagerly.
    std::optional<Value> value;
    // Value structure used by a tied mutable receiver result.
    const RecordPartition* record;
    // Once-evaluated external lvalue, absent for temporaries/current object.
    std::optional<Lvalue> target;
    // Whether mutable results replace the active value receiver.
    bool current;
    // Exact top-level fields returned by a value receiver call.
    std::span<cxx::FieldSymbol* const> written_fields;
    // Automatic receiver slices participating in reference overlap checks.
    std::vector<ReferenceOrigin> origins;
  };

  ReceiverArgument current_receiver_argument(
      const RecordPartition& record, bool requires_storage,
      std::span<cxx::FieldSymbol* const> written_fields, cxx::AST* owner) {
    if (receivers_.empty() || receivers_.back().record != &record) {
      fail(owner,
           "implicit member call requires a matching current source object");
    }
    const auto& active = receivers_.back();
    if (active.storage) {
      return {ImplicitObjectABI::Storage,
              Value(active.storage->pointer),
              &record,
              std::nullopt,
              true,
              written_fields,
              {}};
    }
    if (requires_storage) {
      fail(owner, "member function requires a storage-backed receiver");
    }
    return {ImplicitObjectABI::Value,
            std::nullopt,
            &record,
            std::nullopt,
            true,
            written_fields,
            {}};
  }

  ReceiverArgument receiver_argument(cxx::FunctionSymbol* function,
                                     cxx::ExpressionAST* receiver,
                                     cxx::TokenKind access_op,
                                     cxx::AST* owner) {
    auto* object_type = implicit_object_type(unit_, function);
    auto* record = object_type
                       ? types_.record(types_.unqualified(object_type), owner)
                       : nullptr;
    if (!record) {
      fail(owner, "non-static member call requires an admitted record object");
    }
    auto written_fields = functions_.receiver_writes(function);
    const bool mutable_access = !written_fields.empty();
    const bool requires_storage =
        unit_.typeTraits().is_volatile(object_type) ||
        functions_.receiver_requires_storage(function) ||
        control_->call_result_requires_storage(owner);
    auto admit = [&](ReceiverArgument argument) {
      if (requires_storage && argument.abi == ImplicitObjectABI::Value) {
        fail(owner, "member call requires a storage-backed receiver");
      }
      return argument;
    };
    if (!receiver || is_current_object(receiver)) {
      return admit(current_receiver_argument(*record, requires_storage,
                                             written_fields, owner));
    }
    if (access_op == cxx::TokenKind::T_MINUS_GREATER) {
      auto projection = pointer_expression(receiver);
      return admit({ImplicitObjectABI::Storage,
                    Value(projection.pointer),
                    record,
                    std::nullopt,
                    false,
                    written_fields,
                    {}});
    }
    if (access_op != cxx::TokenKind::T_DOT) {
      fail(owner, "member call requires a dot or arrow receiver");
    }
    if (receiver->valueCategory == cxx::ValueCategory::kPrValue) {
      if (requires_storage) {
        auto projection = object_address(receiver);
        return {ImplicitObjectABI::Storage,
                Value(projection.pointer),
                record,
                std::nullopt,
                false,
                written_fields,
                {}};
      }
      auto value = expression(receiver);
      if (&value.partition() != record) {
        fail(receiver,
             "member call receiver must preserve its admitted record type");
      }
      return admit({ImplicitObjectABI::Value,
                    value,
                    record,
                    std::nullopt,
                    false,
                    written_fields,
                    {}});
    }

    auto target =
        destination(receiver, owner,
                    mutable_access ? LvalueAccess::Write : LvalueAccess::Read);
    auto* object = std::get_if<ObjectLvalue>(&target.projection);
    if (!object) {
      fail(receiver, "member call receiver requires a complete object");
    }
    if (auto* storage = std::get_if<StorageProjection>(&object->location)) {
      return admit({ImplicitObjectABI::Storage,
                    Value(storage->pointer),
                    record,
                    std::nullopt,
                    false,
                    written_fields,
                    {}});
    }
    return admit({ImplicitObjectABI::Value,
                  std::nullopt,
                  record,
                  std::move(target),
                  false,
                  written_fields,
                  {}});
  }

  ReceiverArgument receiver_argument(cxx::CallExpressionAST* call,
                                     cxx::FunctionSymbol* function) {
    auto* base = unwrap_expression(call->baseExpression);
    auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base);
    return receiver_argument(
        function, member ? member->baseExpression : nullptr,
        member ? member->accessOp : cxx::TokenKind::T_DOT, call);
  }

  ReceiverArgument receiver_argument(cxx::SubscriptExpressionAST* call,
                                     cxx::FunctionSymbol* function) {
    return receiver_argument(function, call->baseExpression,
                             cxx::TokenKind::T_DOT, call);
  }

  Value materialize_receiver(ReceiverArgument& receiver, cxx::AST* owner) {
    if (receiver.value) {
      return *receiver.value;
    }
    Value value = receiver.current ? current_receiver_value(owner)
                                   : read(*receiver.target, owner);
    if (&value.partition() != receiver.record) {
      fail(owner,
           "member call receiver must preserve its admitted record type");
    }
    if (receiver.current) {
      receiver.origins.reserve(receiver.written_fields.size());
      for (auto* field : receiver.written_fields) {
        const auto& member = types_.member(field, owner);
        receiver.origins.push_back(
            {field, 0, member.partition->component_count, true});
      }
    } else {
      receiver.origins.push_back(reference_origin(
          *receiver.target, value, !receiver.written_fields.empty(), owner));
    }
    receiver.value = value;
    return value;
  }

  void write_receiver(const ReceiverArgument& receiver, Value value,
                      cxx::AST* owner) {
    if (receiver.current) {
      replace_current_receiver(value);
    } else if (receiver.target) {
      write(*receiver.target, value, owner);
    }
  }

  std::optional<Value> ordinary_call(cxx::CallExpressionAST* call,
                                     cxx::FunctionSymbol* function,
                                     loom_location_id_t source) {
    if (call->isVirtualDispatch) {
      fail(call, "virtual member dispatch is not admitted");
    }
    if (function->hasExplicitObjectParameter()) {
      fail(call, "explicit object member functions are not admitted");
    }
    std::optional<ReceiverArgument> receiver;
    if (function->isImplicitObjectMemberFunction()) {
      receiver = receiver_argument(call, function);
    } else if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(
                   unwrap_expression(call->baseExpression))) {
      // C++ evaluates the object expression even when member lookup selects a
      // static function. Preserve that sequencing before explicit arguments.
      effect(member->baseExpression);
    }
    return emit_ordinary_call(call, function, {call->expressionList, nullptr},
                              std::move(receiver), source);
  }

  std::optional<Value> ordinary_call(cxx::SubscriptExpressionAST* call,
                                     cxx::FunctionSymbol* function,
                                     loom_location_id_t source) {
    if (call->isVirtualDispatch) {
      fail(call, "virtual member dispatch is not admitted");
    }
    if (function->hasExplicitObjectParameter()) {
      fail(call, "explicit object member functions are not admitted");
    }
    std::optional<ReceiverArgument> receiver;
    if (function->isImplicitObjectMemberFunction()) {
      receiver = receiver_argument(call, function);
    } else {
      // A static overloaded subscript still evaluates its written object
      // expression before the explicit index argument.
      effect(call->baseExpression);
    }
    return emit_ordinary_call(call, function, {nullptr, call->indexExpression},
                              std::move(receiver), source);
  }

  std::optional<Value> overloaded_subscript_call(
      cxx::SubscriptExpressionAST* call) {
    auto* function = call->symbol;
    IREE_ASSERT(function);
    if (intrinsics_.check_binding(function, call) ||
        functions_.is_check_case(function) ||
        annotated(function, "check_benchmark")) {
      fail(call, "check declarations cannot be used as overloaded subscripts");
    }
    if (intrinsics_.lookup(function, call)) {
      fail(call, "intrinsics cannot be used as overloaded subscripts");
    }
    if (annotated(function, "kernel")) {
      fail(call, "kernels require an explicit launch operation");
    }
    return ordinary_call(call, function, locations_.get(call));
  }

  cxx::CallExpressionAST* direct_wrapped_call(cxx::ExpressionAST* source) {
    while (source) {
      if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
        source = nested->expression;
        continue;
      }
      if (auto* initializer =
              cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
        source = initializer->expression;
        continue;
      }
      if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source);
          cast && !cast->conversionFunction) {
        source = cast->expression;
        continue;
      }
      return cxx::ast_cast<cxx::CallExpressionAST>(source);
    }
    return nullptr;
  }

  const cxx::Type* storage_element_type(const cxx::Type* source_type,
                                        const Partition& partition) {
    if (auto* pointer =
            cxx::type_cast<cxx::PointerType>(types_.unqualified(source_type))) {
      return pointer->elementType();
    }
    return partition.kind == ValueKind::Buffer
               ? static_cast<const BufferPartition&>(partition).element_type
               : nullptr;
  }

  bool storage_elements_compatible(const cxx::Type* expected,
                                   const cxx::Type* actual) {
    if (!expected || !actual ||
        types_.unqualified(expected) != types_.unqualified(actual)) {
      return false;
    }
    auto traits = unit_.typeTraits();
    return !(traits.is_const(actual) && !traits.is_const(expected)) &&
           !(traits.is_volatile(actual) && !traits.is_volatile(expected));
  }

  void append_kernel_workloads(cxx::ExpressionAST* source,
                               const KernelLaunchIntrinsic& launch,
                               std::vector<loom_value_id_t>& workloads) {
    auto* call = direct_wrapped_call(source);
    auto* function = call ? direct_callee(call) : nullptr;
    if (!function || !annotated(function, "workload")) {
      fail(source,
           "kernel workloads require a direct loom::kernel::workload(...) "
           "call");
    }
    auto parameters = launch.configuration->parameters();
    auto* argument = call->expressionList;
    for (auto* parameter : parameters) {
      if (!argument) {
        fail(source,
             "loom::kernel::workload argument count must match the kernel "
             "configuration");
      }
      auto* parameter_type = types_.unqualified(parameter->type());
      const auto& expected = types_.partition(parameter_type, argument->value);
      auto value = expression(argument->value);
      const auto& actual = value.partition();
      auto* expected_element = storage_element_type(parameter_type, expected);
      auto* actual_element =
          storage_element_type(argument->value->type, actual);
      if (expected.kind == ValueKind::Pointer && value.is_buffer()) {
        if (!storage_elements_compatible(expected_element, actual_element)) {
          fail(argument->value,
               "kernel workload buffer element type must match the "
               "configuration pointer element type");
        }
        workloads.push_back(value.ssa());
        auto byte_offset = scalars_.integer(0, LOOM_SCALAR_TYPE_OFFSET,
                                            locations_.get(argument->value));
        workloads.push_back(name(
            byte_offset,
            cxx::to_string(parameter->name()) + std::string("_byte_offset")));
      } else {
        if (types_.unqualified(argument->value->type) != parameter_type ||
            actual.component_count != expected.component_count) {
          fail(argument->value,
               "loom::kernel::workload arguments must match the kernel "
               "configuration parameters");
        }
        value.append_to(workloads);
      }
      argument = argument->next;
    }
    if (argument) {
      fail(source,
           "loom::kernel::workload argument count must match the kernel "
           "configuration");
    }
  }

  void kernel_launch(cxx::CallExpressionAST* call,
                     const KernelLaunchIntrinsic& launch) {
    if (current_function_.kind != FunctionKind::CommandProgram) {
      fail(call, "kernel.launch requires a command_program body");
    }
    std::vector<loom_value_id_t> workloads;
    auto* argument = call->expressionList;
    if (launch.pipeline) {
      std::vector<loom_value_id_t> arguments;
      append_staged_arguments(argument, *launch.pipeline,
                              launch.pipeline_buffer_materializations,
                              workloads, arguments);
      loom_op_t* op;
      check(loom_kernel_launch_build(
          &builder_, functions_.declare(launch.kernel, call), workloads.data(),
          workloads.size(), arguments.data(), arguments.size(),
          locations_.get(call), &op));
      return;
    }
    if (launch.configuration && !launch.configuration->parameters().empty()) {
      append_kernel_workloads(argument->value, launch, workloads);
      argument = argument->next;
    }
    auto* kernel_type =
        cxx::type_cast<cxx::FunctionType>(launch.kernel->type());
    auto kernel_parameters = kernel_type->parameterTypes();
    std::vector<loom_value_id_t> arguments;
    size_t ordinal = 0;
    for (; argument; argument = argument->next, ++ordinal) {
      auto value = expression(argument->value);
      auto expected = types_.get(kernel_parameters[ordinal], argument->value);
      if (loom_type_kind(expected) == LOOM_TYPE_BUFFER) {
        if (!value.is_buffer()) {
          fail(argument->value,
               "command kernel pointer arguments require typed buffers");
        }
        const auto& actual =
            static_cast<const BufferPartition&>(value.partition());
        auto* expected_element = storage_element_type(
            kernel_parameters[ordinal],
            types_.partition(kernel_parameters[ordinal], argument->value));
        if (!storage_elements_compatible(expected_element,
                                         actual.element_type)) {
          fail(argument->value,
               "command kernel buffer element type must match the kernel "
               "pointer element type");
        }
        arguments.push_back(value.ssa());
      } else {
        arguments.push_back(value.ssa());
      }
    }
    loom_op_t* op;
    check(loom_kernel_launch_build(
        &builder_, functions_.declare(launch.kernel, call), workloads.data(),
        workloads.size(), arguments.data(), arguments.size(),
        locations_.get(call), &op));
  }

  void append_staged_arguments(cxx::List<cxx::ExpressionAST*>* source,
                               const StagedSignatureProjection& projection,
                               std::span<const uint8_t> buffer_materializations,
                               std::vector<loom_value_id_t>& specializations,
                               std::vector<loom_value_id_t>& bindings) {
    specializations.resize(projection.specialization_count);
    bindings.resize(projection.binding_count);
    size_t ordinal = 0;
    for (auto* argument : cxx::ListView{source}) {
      auto value = expression(argument);
      const auto& parameter = projection.parameters[ordinal];
      auto components = value.components();
      loom_value_id_t storage_operand = LOOM_VALUE_ID_INVALID;
      if (!buffer_materializations.empty() &&
          buffer_materializations[ordinal]) {
        if (value.is_pointer()) {
          storage_operand =
              storage_.zero_origin_root(value.pointer(), argument);
        } else {
          storage_operand = components.back();
        }
        components = {&storage_operand, 1};
      }
      ++ordinal;
      IREE_ASSERT_EQ(components.size(), parameter.component_count);
      std::copy_n(components.begin(), parameter.specialization_count,
                  specializations.begin() + parameter.specialization_offset);
      std::copy_n(components.begin() + parameter.specialization_count,
                  parameter.binding_count,
                  bindings.begin() + parameter.binding_offset);
    }
    IREE_ASSERT_EQ(ordinal, projection.parameters.size());
  }

  void command_launch(cxx::CallExpressionAST* call,
                      const CommandLaunchIntrinsic& launch) {
    if (current_function_.kind != FunctionKind::CommandProgram) {
      fail(call, "command.launch requires a command_program body");
    }
    std::vector<loom_value_id_t> specializations;
    std::vector<loom_value_id_t> bindings;
    append_staged_arguments(call->expressionList, *launch.projection,
                            /*buffer_materializations=*/{}, specializations,
                            bindings);
    loom_op_t* op;
    check(loom_command_launch_build(
        &builder_, functions_.declare(launch.program, call),
        specializations.data(), specializations.size(), bindings.data(),
        bindings.size(), locations_.get(call), &op));
  }

  std::string_view required_string_literal(cxx::ExpressionAST* source,
                                           const char* message) {
    while (source) {
      if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(source)) {
        source = nested->expression;
        continue;
      }
      if (auto* initializer =
              cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(source)) {
        source = initializer->expression;
        continue;
      }
      if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source);
          cast && !cast->conversionFunction) {
        source = cast->expression;
        continue;
      }
      auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(source);
      if (!literal) {
        fail(source, message);
      }
      return literal->literal->stringValue();
    }
    fail(current_function_.source, message);
  }

  Value command_parameter(cxx::CallExpressionAST* call,
                          const CommandParameterIntrinsic& binding) {
    if (current_function_.kind != FunctionKind::CommandProgram) {
      fail(call, "command parameter operations require a command_program body");
    }
    auto* argument = call->expressionList;
    if (binding.operation == CommandParameterIntrinsic::Operation::Base) {
      auto source = expression(argument->value);
      loom_op_t* op;
      check(loom_command_parameter_base_build(
          &builder_, source.components().back(), types_.get(call->type, call),
          locations_.get(call), &op));
      auto result = loom_op_results(op)[0];
      return value_arena_.capture(*binding.result, {&result, 1});
    }

    auto source = expression(argument->value);
    argument = argument->next;
    std::vector<loom_value_id_t> components;
    components.reserve(binding.binding_component_count + 1);
    for (size_t index = 0; index < binding.binding_component_count; ++index) {
      auto value = expression(argument->value);
      IREE_ASSERT_EQ(value.components().size(), 1);
      components.push_back(value.components()[0]);
      argument = argument->next;
    }
    auto pattern = required_string_literal(
        argument->value,
        "command.parameter requires a string literal key pattern");
    argument = argument->next;
    std::vector<loom_value_id_t> substitutions;
    for (; argument; argument = argument->next) {
      substitutions.push_back(expression(argument->value).ssa());
    }

    std::vector<loom_value_id_t> type_identities = components;
    type_identities.push_back(LOOM_VALUE_ID_INVALID);
    BoundTypeStorage type_storage;
    std::vector<loom_type_t> component_types;
    types_.append_bound(call->type, call, type_identities, type_storage,
                        component_types);
    IREE_ASSERT_EQ(component_types.size(), type_identities.size());
    loom_string_id_t pattern_id;
    check(loom_module_intern_string(module_, view(pattern), &pattern_id));
    loom_op_t* op;
    check(loom_command_parameter_build(
        &builder_, source.ssa(), pattern_id, substitutions.data(),
        substitutions.size(), component_types.back(), locations_.get(call),
        &op));
    components.push_back(loom_op_results(op)[0]);
    return value_arena_.capture(*binding.result, components);
  }

  void admit_ordinary_operator(const DirectOperatorCall& call,
                               cxx::ExpressionAST* expression) {
    auto* function = call.function;
    IREE_ASSERT(function);
    if (call.is_virtual_dispatch) {
      fail(expression, "virtual member dispatch is not admitted");
    }
    if (function->hasExplicitObjectParameter()) {
      fail(expression, "explicit object member functions are not admitted");
    }
    if (intrinsics_.check_binding(function, expression) ||
        functions_.is_check_case(function) ||
        annotated(function, "check_benchmark")) {
      fail(expression,
           "check declarations cannot be used as ordinary operators");
    }
    if (intrinsics_.lookup(function, expression)) {
      fail(expression,
           "intrinsic operators must use their declared Loom operation");
    }
    if (annotated(function, "kernel")) {
      fail(expression, "kernels require an explicit launch operation");
    }
  }

  struct SourceArguments {
    // Remaining arguments owned by an ordinary call's frontend list.
    cxx::List<cxx::ExpressionAST*>* remaining;
    // Single argument supplied by overloaded subscript syntax.
    cxx::ExpressionAST* single;

    cxx::ExpressionAST* next() {
      if (remaining) {
        auto* argument = remaining->value;
        remaining = remaining->next;
        return argument;
      }
      return std::exchange(single, nullptr);
    }
  };

  std::optional<Value> emit_simple_call(cxx::ExpressionAST* call,
                                        cxx::FunctionSymbol* function,
                                        loom_symbol_ref_t callee,
                                        std::vector<loom_value_id_t> arguments,
                                        loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
    loom_op_t* op;
    if (unit_.typeTraits().is_void(signature->returnType())) {
      check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee,
                                 arguments.data(), arguments.size(), nullptr, 0,
                                 nullptr, 0, source, &op));
      return std::nullopt;
    }
    const bool returns_reference =
        unit_.typeTraits().is_lvalue_reference(signature->returnType());
    std::array<const cxx::Type*, 1> result_sources = {
        returns_reference
            ? unit_.typeTraits().add_pointer(
                  unit_.typeTraits().remove_reference(signature->returnType()))
            : signature->returnType()};
    auto results = bind_signature(types_, result_sources, call, &builder_);
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee, arguments.data(),
                               arguments.size(), results.types.data(),
                               results.types.size(), nullptr, 0, source, &op));
    return value_arena_.capture(
        returns_reference ? kPointerPartition
                          : types_.partition(signature->returnType(), call),
        {loom_op_results(op), results.types.size()});
  }

  std::optional<Value> emit_evaluated_call(
      cxx::ExpressionAST* call, cxx::FunctionSymbol* function,
      loom_symbol_ref_t callee, std::optional<ReceiverArgument> receiver,
      std::vector<loom_value_id_t> arguments,
      std::vector<ReferenceWriteback> writebacks,
      std::vector<ReferenceOrigin> origins, loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
    if (receiver) {
      auto receiver_value = materialize_receiver(*receiver, call);
      auto receiver_components = receiver_value.components();
      arguments.insert(arguments.begin(), receiver_components.begin(),
                       receiver_components.end());
      origins.insert(origins.end(), receiver->origins.begin(),
                     receiver->origins.end());
    }
    require_nonoverlapping_references(std::move(origins), call);

    const bool returns_void =
        unit_.typeTraits().is_void(signature->returnType());
    const bool returns_reference =
        unit_.typeTraits().is_lvalue_reference(signature->returnType());
    const bool returns_receiver_alias =
        returns_reference && receiver &&
        receiver->abi == ImplicitObjectABI::Value &&
        functions_.returns_current_receiver_reference(function);
    std::vector<const cxx::Type*> result_sources;
    const bool receiver_writeback = receiver &&
                                    receiver->abi == ImplicitObjectABI::Value &&
                                    !receiver->written_fields.empty();
    result_sources.reserve((!returns_void && !returns_receiver_alias) +
                           (receiver ? receiver->written_fields.size() : 0) +
                           writebacks.size());
    size_t ordinary_result_count = 0;
    if (!returns_void && !returns_receiver_alias) {
      auto* result_type = returns_reference
                              ? unit_.typeTraits().add_pointer(
                                    unit_.typeTraits().remove_reference(
                                        signature->returnType()))
                              : signature->returnType();
      result_sources.push_back(result_type);
      ordinary_result_count =
          types_.partition(result_type, call).component_count;
    }
    if (receiver_writeback) {
      for (auto* field : receiver->written_fields) {
        result_sources.push_back(types_.unqualified_value(field->type()));
      }
    }
    for (const auto& writeback : writebacks) {
      result_sources.push_back(writeback.value_type);
    }
    BoundSignature results;
    if (!result_sources.empty()) {
      results = bind_signature(types_, result_sources, call, &builder_);
    }
    std::vector<loom_tied_result_t> tied_results;
    size_t result_offset = ordinary_result_count;
    if (receiver_writeback) {
      for (auto* field : receiver->written_fields) {
        const auto& member = types_.member(field, call);
        const size_t component_count = member.partition->component_count;
        if (member.component_offset + component_count > UINT16_MAX ||
            result_offset + component_count > UINT16_MAX) {
          fail(call,
               "implicit object call exceeds native operand or result limits");
        }
        for (size_t component = 0; component < component_count; ++component) {
          const size_t operand = member.component_offset + component;
          tied_results.push_back({
              .result_index = static_cast<uint16_t>(result_offset + component),
              .operand_index = static_cast<uint16_t>(operand),
              .has_type_change = !loom_type_equal(
                  loom_module_value_type(module_, arguments[operand]),
                  results.types[result_offset + component]),
          });
        }
        result_offset += component_count;
      }
    }
    for (const auto& writeback : writebacks) {
      const size_t component_count = writeback.partition->component_count;
      if (writeback.argument_offset + component_count > UINT16_MAX ||
          result_offset + component_count > UINT16_MAX) {
        fail(call,
             "value-reference call exceeds native operand or result limits");
      }
      for (size_t component = 0; component < component_count; ++component) {
        tied_results.push_back({
            .result_index = static_cast<uint16_t>(result_offset + component),
            .operand_index =
                static_cast<uint16_t>(writeback.argument_offset + component),
            .has_type_change = !loom_type_equal(
                loom_module_value_type(
                    module_, arguments[writeback.argument_offset + component]),
                results.types[result_offset + component]),
        });
      }
      result_offset += component_count;
    }

    loom_op_t* op;
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee, arguments.data(),
                               arguments.size(), results.types.data(),
                               results.types.size(), tied_results.data(),
                               tied_results.size(), source, &op));

    result_offset = ordinary_result_count;
    std::optional<Value> receiver_result;
    if (receiver_writeback) {
      auto updated = *receiver->value;
      for (auto* field : receiver->written_fields) {
        const auto& member = types_.member(field, call);
        const size_t component_count = member.partition->component_count;
        auto field_value = value_arena_.capture(
            *member.partition,
            {loom_op_results(op) + result_offset, component_count});
        updated =
            value_arena_.replace(updated, member.component_offset, field_value);
        result_offset += component_count;
      }
      write_receiver(*receiver, updated, call);
      receiver_result = updated;
    }
    for (const auto& writeback : writebacks) {
      const size_t component_count = writeback.partition->component_count;
      auto updated = value_arena_.capture(
          *writeback.partition,
          {loom_op_results(op) + result_offset, component_count});
      write(writeback.target, updated, call);
      result_offset += component_count;
    }
    if (returns_void) {
      return std::nullopt;
    }
    if (returns_receiver_alias) {
      return receiver_result ? *receiver_result : *receiver->value;
    }
    return value_arena_.capture(
        returns_reference ? kPointerPartition
                          : types_.partition(signature->returnType(), call),
        {loom_op_results(op), ordinary_result_count});
  }

  std::optional<Value> emit_ordinary_call(
      cxx::ExpressionAST* call, cxx::FunctionSymbol* function,
      SourceArguments source_arguments,
      std::optional<ReceiverArgument> receiver, loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
    auto parameters = function->parameters();
    auto receiver_abi = receiver ? receiver->abi : ImplicitObjectABI::None;
    auto callee = functions_.declare(function, call, receiver_abi);
    auto references = functions_.reference_parameters(function);
    if (references.empty() && !receiver) {
      std::vector<loom_value_id_t> arguments;
      while (auto* argument = source_arguments.next()) {
        expression(argument).append_to(arguments);
      }
      return emit_simple_call(call, function, callee, std::move(arguments),
                              source);
    }
    std::vector<loom_value_id_t> arguments;
    std::vector<ReferenceWriteback> writebacks;
    std::vector<ReferenceOrigin> origins;
    writebacks.reserve(references.size());
    origins.reserve(references.size() +
                    (receiver && receiver->abi == ImplicitObjectABI::Value
                         ? receiver->record->members.size()
                         : 0));
    const size_t receiver_argument_count =
        !receiver ? 0
        : receiver->abi == ImplicitObjectABI::Value
            ? receiver->record->component_count
            : kPointerPartition.component_count;

    size_t reference_index = 0;
    for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
      auto* argument = source_arguments.next();
      IREE_ASSERT(argument);
      const ReferenceParameter* reference = nullptr;
      if (reference_index < references.size() &&
          references[reference_index].ordinal == ordinal) {
        reference = &references[reference_index++];
      }
      append_call_argument(evaluate_call_argument(argument, parameters[ordinal],
                                                  reference, call),
                           receiver_argument_count, arguments, writebacks,
                           origins);
    }
    while (auto* argument = source_arguments.next()) {
      expression(argument).append_to(arguments);
    }
    return emit_evaluated_call(call, function, callee, std::move(receiver),
                               std::move(arguments), std::move(writebacks),
                               std::move(origins), source);
  }

  std::optional<Value> ordinary_operator_call(cxx::ExpressionAST* expression,
                                              const DirectOperatorCall& call) {
    admit_ordinary_operator(call, expression);
    auto* function = call.function;
    auto parameters = function->parameters();
    const size_t parameter_count =
        call.argument_count + (call.has_postfix_dummy ? 1u : 0u);
    IREE_ASSERT(parameters.size() == parameter_count);

    auto references = functions_.reference_parameters(function);
    std::array<const ReferenceParameter*, 3> reference_by_ordinal = {};
    for (const auto& reference : references) {
      IREE_ASSERT(reference.ordinal < parameter_count);
      reference_by_ordinal[reference.ordinal] = &reference;
    }

    std::array<std::optional<EvaluatedCallArgument>, 3> evaluated_arguments;
    std::optional<ReceiverArgument> receiver;
    auto evaluate_receiver = [&] {
      IREE_ASSERT(call.receiver && !receiver);
      receiver = receiver_argument(function, call.receiver,
                                   cxx::TokenKind::T_DOT, expression);
    };
    auto evaluate_argument = [&](size_t ordinal) {
      IREE_ASSERT(ordinal < call.argument_count &&
                  !evaluated_arguments[ordinal]);
      evaluated_arguments[ordinal] =
          evaluate_call_argument(call.arguments[ordinal], parameters[ordinal],
                                 reference_by_ordinal[ordinal], expression);
    };

    switch (call.evaluation_order) {
      case OperatorEvaluationOrder::Forward:
        if (call.receiver) {
          evaluate_receiver();
        }
        for (size_t ordinal = 0; ordinal < call.argument_count; ++ordinal) {
          evaluate_argument(ordinal);
        }
        break;
      case OperatorEvaluationOrder::ArgumentsThenReceiver:
        for (size_t ordinal = 0; ordinal < call.argument_count; ++ordinal) {
          evaluate_argument(ordinal);
        }
        evaluate_receiver();
        break;
      case OperatorEvaluationOrder::ReverseArguments:
        IREE_ASSERT(!call.receiver);
        for (size_t ordinal = call.argument_count; ordinal > 0; --ordinal) {
          evaluate_argument(ordinal - 1);
        }
        break;
    }

    if (call.has_postfix_dummy) {
      const size_t ordinal = call.argument_count;
      IREE_ASSERT(!reference_by_ordinal[ordinal]);
      auto type = types_.get(parameters[ordinal]->type(), expression);
      IREE_ASSERT(loom_type_kind(type) == LOOM_TYPE_SCALAR);
      evaluated_arguments[ordinal] = EvaluatedCallArgument{
          .value = Value(scalars_.integer(0, loom_type_element_type(type),
                                          locations_.get(expression))),
      };
    }

    auto receiver_abi = receiver ? receiver->abi : ImplicitObjectABI::None;
    auto callee = functions_.declare(function, expression, receiver_abi);
    if (references.empty() && !receiver) {
      std::vector<loom_value_id_t> arguments;
      for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
        IREE_ASSERT(evaluated_arguments[ordinal]);
        evaluated_arguments[ordinal]->value.append_to(arguments);
      }
      return emit_simple_call(expression, function, callee,
                              std::move(arguments), locations_.get(expression));
    }

    const size_t receiver_argument_count =
        !receiver ? 0
        : receiver->abi == ImplicitObjectABI::Value
            ? receiver->record->component_count
            : kPointerPartition.component_count;
    std::vector<loom_value_id_t> arguments;
    std::vector<ReferenceWriteback> writebacks;
    std::vector<ReferenceOrigin> origins;
    writebacks.reserve(references.size());
    origins.reserve(references.size() +
                    (receiver && receiver->abi == ImplicitObjectABI::Value
                         ? receiver->record->members.size()
                         : 0));
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      IREE_ASSERT(evaluated_arguments[ordinal]);
      append_call_argument(std::move(*evaluated_arguments[ordinal]),
                           receiver_argument_count, arguments, writebacks,
                           origins);
    }
    return emit_evaluated_call(
        expression, function, callee, std::move(receiver), std::move(arguments),
        std::move(writebacks), std::move(origins), locations_.get(expression));
  }

  std::optional<Value> ordinary_operator_call(cxx::ExpressionAST* expression) {
    auto call = direct_operator_call(expression);
    IREE_ASSERT(call);
    return ordinary_operator_call(expression, *call);
  }

  Value ordinary_operator_value(cxx::ExpressionAST* expression,
                                const DirectOperatorCall& call) {
    auto value = ordinary_operator_call(expression, call);
    if (!value) {
      fail(expression, "void operator cannot be used as a value");
    }
    auto* signature = cxx::type_cast<cxx::FunctionType>(call.function->type());
    IREE_ASSERT(signature);
    if (!unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
      return *value;
    }
    if (!value->is_pointer()) {
      return *value;
    }
    auto projection = storage_.project(
        value->pointer(),
        unit_.typeTraits().remove_reference(signature->returnType()),
        expression);
    if (unit_.typeTraits().is_class(expression->type)) {
      return objects_.load(projection, expression->type, expression);
    }
    return storage_.load(
        storage_.dereference(projection, expression->type, expression),
        expression->type, expression);
  }

  Value increment(cxx::ExpressionAST* destination, cxx::TokenKind token,
                  IncrementResult selected, cxx::AST* owner) {
    auto target = this->destination(destination, owner);
    auto previous = read(target, destination);
    auto source = locations_.get(owner);
    auto operation = token == cxx::TokenKind::T_MINUS_MINUS
                         ? cxx::TokenKind::T_MINUS
                         : cxx::TokenKind::T_PLUS;
    Value updated;
    if (auto* pointer = cxx::type_cast<cxx::PointerType>(
            types_.unqualified(destination->type))) {
      auto one = scalars_.integer(1, LOOM_SCALAR_TYPE_I32, source);
      updated =
          storage_
              .advance(pointer_projection(previous, pointer, owner), one,
                       pointer, unit_.control()->getIntType(), operation, owner)
              .pointer;
    } else if (types_.is_float(destination->type)) {
      auto one = scalars_.convert(
          scalars_.integer(1, LOOM_SCALAR_TYPE_I32, source),
          unit_.control()->getIntType(), destination->type, owner);
      updated = scalars_.binary(operation, previous.ssa(), one,
                                destination->type, destination->type, owner);
    } else {
      if (!unit_.typeTraits().is_integral(destination->type) ||
          types_.unqualified(destination->type)->kind() ==
              cxx::TypeKind::kBool) {
        fail(owner, "increment requires an arithmetic or pointer binding");
      }
      // Builtin ++/-- performs promoted arithmetic before converting back to
      // the lvalue type, just like compound assignment with an integer one.
      auto* promoted =
          unit_.typeTraits().promoted_integer_type(destination->type);
      auto value =
          scalars_.convert(previous.ssa(), destination->type, promoted, owner);
      auto one = scalars_.integer(1, loom_type_element_type(value_type(value)),
                                  source);
      value = scalars_.binary(operation, value, one, promoted, promoted, owner);
      updated = scalars_.convert(value, promoted, destination->type, owner);
    }
    write(target, updated, owner);
    return selected == IncrementResult::Previous ? previous : updated;
  }

  Value compound_assignment(cxx::CompoundAssignmentExpressionAST* assignment) {
    auto* ast = assignment;
    auto* destination = assignment->targetExpression;
    if (assignment->symbol) {
      auto call = direct_operator_call(ast);
      IREE_ASSERT(call);
      return ordinary_operator_value(ast, *call);
    }
    auto value = expression(assignment->rightExpression);
    // The right operand precedes both the address evaluation and value read.
    auto target = this->destination(destination, ast);
    auto old = read(target, destination);
    if (old.is_pointer()) {
      auto updated =
          storage_
              .advance(pointer_projection(old, destination->type, ast),
                       value.ssa(), destination->type,
                       assignment->rightExpression->type,
                       cxx::get_underlying_binary_op(assignment->op), ast)
              .pointer;
      write(target, Value(updated), ast);
      return Value(updated);
    }
    auto* promoted = assignment->leftExpression->type;
    old = numeric_convert(old.ssa(), destination->type, promoted, ast);
    value = numeric_convert(value.ssa(), assignment->rightExpression->type,
                            promoted, ast);
    auto updated = binary(cxx::get_underlying_binary_op(assignment->op),
                          old.ssa(), value.ssa(), promoted, promoted, ast);
    updated = numeric_convert(updated, promoted, destination->type, ast);
    write(target, updated, ast);
    return updated;
  }

  Value assignment(cxx::AssignmentExpressionAST* assignment) {
    auto* ast = assignment;
    const auto& partition = types_.partition(
        types_.unqualified(assignment->leftExpression->type), ast);
    cxx::ClassSymbol* source = nullptr;
    if (partition.kind == ValueKind::Record) {
      source = static_cast<const RecordPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Buffer) {
      source = static_cast<const BufferPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Address) {
      source = static_cast<const AddressPartition&>(partition).source;
    } else if (partition.kind == ValueKind::OpaqueDialect) {
      source = static_cast<const OpaqueDialectPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Encoding) {
      source = static_cast<const EncodingPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Vector) {
      source = static_cast<const VectorPartition&>(partition).source;
    } else if (partition.kind == ValueKind::View) {
      source = static_cast<const ViewPartition&>(partition).source;
    } else if (partition.kind == ValueKind::Tensor) {
      source = static_cast<const TensorPartition&>(partition).source;
    }
    if ((assignment->symbol &&
         (!source ||
          (assignment->symbol != source->copyAssignmentOperator() &&
           assignment->symbol != source->moveAssignmentOperator()))) ||
        assignment->op != cxx::TokenKind::T_EQUAL) {
      fail(ast, "only builtin plain assignment is admitted here");
    }
    auto value = expression(assignment->rightExpression);
    write(destination(assignment->leftExpression, ast), value, ast);
    return value;
  }

  // The condition has already executed. Each arm starts from that same state
  // and yields its optional value followed by bindings it may have changed.
  // Object initialization uses void arms that publish directly to storage.
  template <typename Then, typename Else>
  auto conditional(cxx::ExpressionAST* ast, loom_value_id_t condition,
                   Then then_value, Else else_value) {
    constexpr bool has_value = !std::is_void_v<std::invoke_result_t<Then>>;
    auto written = live_mutations(ast);
    auto value_count =
        has_value ? types_.partition(ast->type, ast).component_count : 0;
    std::vector<const cxx::Type*> sources;
    sources.reserve((has_value ? 1 : 0) + written.size());
    if constexpr (has_value) {
      sources.push_back(ast->type);
    }
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto initial = current(written);
    auto source = locations_.get(ast);
    auto outputs = bind_signature(types_, sources, ast, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    std::vector<loom_value_id_t> yielded;
    if constexpr (has_value) {
      then_value().append_to(yielded);
    } else {
      then_value();
    }
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    bind_values(written, initial.data());
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    yielded.clear();
    if constexpr (has_value) {
      else_value().append_to(yielded);
    } else {
      else_value();
    }
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    bind_values(written, loom_op_results(op) + value_count);
    if constexpr (has_value) {
      return value_arena_.capture(types_.partition(ast->type, ast),
                                  {loom_op_results(op), value_count});
    }
  }

  static cxx::ConditionalExpressionAST* parenthesized_conditional(
      cxx::ExpressionAST* expression) {
    // Parentheses do not transform a value. Implicit casts do, so leave those
    // for expression() instead of using the broader unwrap_expression().
    while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(expression)) {
      expression = nested->expression;
    }
    return cxx::ast_cast<cxx::ConditionalExpressionAST>(expression);
  }

  struct ConditionalValueFrame {
    // Source conditional whose unfinished arms this frame owns.
    cxx::ConditionalExpressionAST* expression;
    // Source bindings that either arm may replace.
    std::vector<cxx::Symbol*> written;
    // Binding components restored before entering the else arm.
    std::vector<loom_value_id_t> initial;
    // Structured conditional receiving the two source arm values.
    loom_op_t* operation;
    // Insertion point enclosing |operation|.
    loom_builder_ip_t enclosing;
    // Leading operation results occupied by the conditional value.
    iree_host_size_t value_count;
    // Whether the active operation region is the then arm.
    bool building_then;
  };

  ConditionalValueFrame begin_conditional_value(
      cxx::ConditionalExpressionAST* expression) {
    if (types_.native_vector(expression->condition->type)) {
      fail(expression, "vector conditions require lane-wise selection");
    }
    auto condition = this->expression(expression->condition).ssa();
    auto written = live_mutations(expression);
    auto value_count =
        types_.partition(expression->type, expression).component_count;
    std::vector<const cxx::Type*> sources = {expression->type};
    sources.reserve(1 + written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto initial = current(written);
    auto outputs = bind_signature(types_, sources, expression, &builder_);
    loom_op_t* operation;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0,
                            locations_.get(expression), &operation));
    auto enclosing = loom_builder_enter_region(
        &builder_, operation, loom_scf_if_then_region(operation));
    return {expression, std::move(written), std::move(initial),
            operation,  enclosing,          value_count,
            true};
  }

  void yield_conditional_value(const ConditionalValueFrame& frame,
                               Value value) {
    std::vector<loom_value_id_t> yielded;
    value.append_to(yielded);
    auto mutations = current(frame.written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               locations_.get(frame.expression), &yield));
  }

  Value finish_conditional_value(const ConditionalValueFrame& frame) {
    loom_builder_restore(&builder_, frame.enclosing);
    bind_values(frame.written,
                loom_op_results(frame.operation) + frame.value_count);
    return value_arena_.capture(
        types_.partition(frame.expression->type, frame.expression),
        {loom_op_results(frame.operation), frame.value_count});
  }

  Value conditional_value(cxx::ConditionalExpressionAST* expression) {
    if (!parenthesized_conditional(expression->iftrueExpression) &&
        !parenthesized_conditional(expression->iffalseExpression)) {
      if (types_.native_vector(expression->condition->type)) {
        fail(expression, "vector conditions require lane-wise selection");
      }
      auto condition = this->expression(expression->condition).ssa();
      return conditional(
          expression, condition,
          [&] { return this->expression(expression->iftrueExpression); },
          [&] { return this->expression(expression->iffalseExpression); });
    }

    // Generated decision tables commonly form long trees of parenthesized
    // conditional expressions. Keep the source evaluation and nested scf.if
    // shape while storing unfinished arms on the heap instead of retaining
    // one C++ call frame per source level.
    std::vector<ConditionalValueFrame> frames;
    frames.push_back(begin_conditional_value(expression));
    std::optional<Value> value;
    for (;;) {
      auto& frame = frames.back();
      if (!value) {
        auto* arm = frame.building_then ? frame.expression->iftrueExpression
                                        : frame.expression->iffalseExpression;
        if (auto* nested = parenthesized_conditional(arm)) {
          frames.push_back(begin_conditional_value(nested));
        } else {
          value = this->expression(arm);
        }
        continue;
      }

      yield_conditional_value(frame, *value);
      value.reset();
      if (frame.building_then) {
        loom_builder_restore(&builder_, frame.enclosing);
        bind_values(frame.written, frame.initial.data());
        loom_builder_enter_region(&builder_, frame.operation,
                                  loom_scf_if_else_region(frame.operation));
        frame.building_then = false;
        continue;
      }

      value = finish_conditional_value(frame);
      frames.pop_back();
      if (frames.empty()) {
        return *value;
      }
    }
  }

  void conditional_object(StorageProjection destination, const cxx::Type* type,
                          cxx::ConditionalExpressionAST* select,
                          bool retained) override {
    if (types_.native_vector(select->condition->type)) {
      fail(select, "vector conditions require lane-wise selection");
    }
    auto condition = expression(select->condition).ssa();
    conditional(
        select, condition,
        [&] {
          initialization_.object(destination, type, select->iftrueExpression,
                                 nullptr, select, retained);
        },
        [&] {
          initialization_.object(destination, type, select->iffalseExpression,
                                 nullptr, select, retained);
        });
  }

  std::optional<IntrinsicCallResult> builtin(cxx::CallExpressionAST* call) {
    if (auto atomic =
            AtomicBuiltin::resolve(unit_, diagnostics_, types_, call)) {
      std::array<Value, 4> arguments;
      auto* argument = call->expressionList;
      // Admission consumed the constant orderings. Evaluate every remaining
      // operand once, including weak even though High CAS is always strong.
      for (size_t index = 0; index < atomic->argument_count(); ++index) {
        arguments[index] = expression(argument->value);
        argument = argument->next;
      }
      return IntrinsicCallResult{atomic->call(
          std::span(arguments).first(atomic->argument_count()), storage_,
          scalars_, call, &builder_, locations_.get(call))};
    }
    if (auto fence =
            FenceIntrinsic::resolve_builtin(unit_, diagnostics_, call)) {
      fence->call(&builder_, locations_.get(call));
      return IntrinsicCallResult{std::nullopt};
    }
    if (auto exponent = ExponentBuiltin::resolve(types_, call)) {
      std::array<Value, 2> arguments;
      auto* argument = call->expressionList;
      for (auto& value : arguments) {
        value = expression(argument->value);
        argument = argument->next;
      }
      return IntrinsicCallResult{exponent->call(
          arguments, storage_, call, &builder_, locations_.get(call))};
    }
    return std::nullopt;
  }

  Value constant(const cxx::ConstValue& value, cxx::ExpressionAST* ast) {
    return types_.native_vector(ast->type)
               ? vectors_.constant(value, ast->type, ast)
               : scalars_.constant(value, ast->type, ast);
  }

  std::optional<Value> fold_constant_reference_call(
      cxx::ExpressionAST* expression, cxx::CallExpressionAST* call) {
    auto* callee =
        call ? cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression)
             : nullptr;
    auto* function = callee
                         ? cxx::symbol_cast<cxx::FunctionSymbol>(callee->symbol)
                         : nullptr;
    auto* signature = function
                          ? cxx::type_cast<cxx::FunctionType>(function->type())
                          : nullptr;
    if (!signature) {
      return std::nullopt;
    }
    // A constant invocation has no runtime reference ABI. Fold it before
    // reachable-function discovery so constexpr-only helpers remain source
    // semantics rather than unsupported Loom call signatures.
    auto traits = unit_.typeTraits();
    bool has_reference = traits.is_reference(signature->returnType());
    for (auto* parameter : function->parameters()) {
      if (traits.is_reference(parameter->type())) {
        has_reference = true;
        break;
      }
    }
    if (!has_reference || (!traits.is_arithmetic(expression->type) &&
                           !traits.is_enum(expression->type))) {
      return std::nullopt;
    }
    cxx::ASTInterpreter interpreter(&unit_);
    auto value = interpreter.evaluate(expression);
    if (!value ||
        (types_.is_float(expression->type) ? !interpreter.toDouble(*value)
                                           : !interpreter.toInt(*value))) {
      return std::nullopt;
    }
    return constant(*value, expression);
  }

  Value expression(cxx::ExpressionAST* ast) override {
    if (!ast) {
      throw std::runtime_error("missing expression");
    }
    if (auto* assignment = cxx::ast_cast<cxx::AssignmentExpressionAST>(ast)) {
      return this->assignment(assignment);
    }
    if (auto* assignment =
            cxx::ast_cast<cxx::CompoundAssignmentExpressionAST>(ast)) {
      return compound_assignment(assignment);
    }
    auto source = locations_.get(ast);
    if (auto* expression = cxx::ast_cast<cxx::ConstExpressionAST>(ast)) {
      return constant(*expression->constValue, ast);
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return expression(nested->expression);
    }
    if (auto* equal = cxx::ast_cast<cxx::EqualInitializerAST>(ast)) {
      return expression(equal->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(ast)) {
      return expression(initializer->expression);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast)) {
      if (cast->castKind == cxx::ImplicitCastKind::kLValueToRValueConversion) {
        if (auto* call =
                cxx::ast_cast<cxx::CallExpressionAST>(cast->expression)) {
          if (auto constant = fold_constant_reference_call(ast, call)) {
            return *constant;
          }
        }
      }
      if (cast->conversionFunction &&
          types_.address_domain(cast->expression->type, ast) ==
              LOOM_SCALAR_TYPE_NONE) {
        types_.admit_construction(cast->conversionFunction, ast->type, ast);
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast =
            cxx::ast_cast<cxx::BuiltinConvertVectorExpressionAST>(ast)) {
      auto value = expression(cast->expression).ssa();
      return vectors_.convert_elements(value, cast->expression->type,
                                       cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::BuiltinBitCastExpressionAST>(ast)) {
      BitCast conversion(unit_, diagnostics_, types_, cast->expression->type,
                         cast->type, ast);
      auto value = expression(cast->expression).ssa();
      return conversion.build(builder_, value, source);
    }
    if (auto* cast = cxx::ast_cast<cxx::CastExpressionAST>(ast)) {
      auto input = loom_type_kind(types_.get(cast->expression->type, ast));
      auto output = loom_type_kind(types_.get(cast->type, ast));
      if (input != output) {
        fail(ast,
             "explicit casts require two scalars, two vectors or two pointers");
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(ast)) {
      auto input = loom_type_kind(types_.get(cast->expression->type, ast));
      auto output = loom_type_kind(types_.get(cast->type, ast));
      if (input != output ||
          (cast->castOp != cxx::TokenKind::T_STATIC_CAST &&
           !(cast->castOp == cxx::TokenKind::T_REINTERPRET_CAST &&
             (input == LOOM_TYPE_BUFFER || input == LOOM_TYPE_VECTOR)))) {
        fail(ast,
             "casts require numeric static_cast or pointer/vector "
             "reinterpret_cast");
      }
      return convert(cast->expression, cast->type, ast);
    }
    if (auto* cast = cxx::ast_cast<cxx::TypeConstructionAST>(ast)) {
      if (types_.address_domain(ast->type, ast) != LOOM_SCALAR_TYPE_NONE) {
        if (!cast->expressionList || cast->expressionList->next) {
          fail(ast, "address construction requires one integer argument");
        }
        return convert(cast->expressionList->value, ast->type, ast);
      }
      if (types_.record(ast->type, ast)) {
        types_.admit_construction(cast->constructorSymbol, ast->type, ast);
        return initialization_.value(ast->type, ast, cast->constructorSymbol,
                                     ast);
      }
      auto output = types_.get(ast->type, ast);
      if (cast->constructorSymbol ||
          loom_type_kind(output) != LOOM_TYPE_SCALAR ||
          (cast->expressionList && cast->expressionList->next)) {
        fail(ast, "functional casts require a scalar and at most one argument");
      }
      if (cast->expressionList) {
        return convert(cast->expressionList->value, ast->type, ast);
      }
      loom_op_t* op;
      check(loom_scalar_constant_build(
          &builder_,
          types_.is_float(ast->type) ? loom_attr_f64(0.0) : loom_attr_i64(0),
          output, source, &op));
      return result(op);
    }
    if (auto* construction =
            cxx::ast_cast<cxx::BracedTypeConstructionAST>(ast)) {
      if (types_.address_domain(ast->type, ast) != LOOM_SCALAR_TYPE_NONE) {
        auto* arguments = construction->bracedInitList->expressionList;
        if (!arguments || arguments->next) {
          fail(ast, "address construction requires one integer argument");
        }
        return convert(arguments->value, ast->type, ast);
      }
      types_.admit_construction(construction->constructorSymbol, ast->type,
                                ast);
      return initialization_.value(ast->type, ast,
                                   construction->constructorSymbol, ast);
    }
    if (cxx::ast_cast<cxx::BracedInitListAST>(ast)) {
      return initialization_.value(ast->type, ast, nullptr, ast);
    }
    if (cxx::ast_cast<cxx::ParenInitializerAST>(ast)) {
      return initialization_.value(ast->type, ast, nullptr, ast);
    }
    if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(ast)) {
      return conditional_value(select);
    }

    if (auto* decision = cxx::ast_cast<cxx::ConditionExpressionAST>(ast)) {
      return binding(decision->symbol, ast);
    }
    if (cxx::ast_cast<cxx::ThisExpressionAST>(ast)) {
      if (receivers_.empty() || !receivers_.back().storage) {
        fail(ast, "this pointer requires a storage-backed receiver");
      }
      return Value(receivers_.back().storage->pointer);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = references_.find(id->symbol);
          found != references_.end()) {
        return name(read(found->second, ast),
                    cxx::to_string(id->symbol->name()));
      }
      if (auto found = values_.find(id->symbol); found != values_.end()) {
        return found->second;
      }
      if (locals_.contains(id->symbol)) {
        return binding(id->symbol, ast);
      }
      if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
          field && receiver(field)) {
        if (unit_.typeTraits().is_lvalue_reference(field->type())) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        return name(receiver_member(field, ast), cxx::to_string(field->name()));
      }
      if (auto* enumerator =
              cxx::symbol_cast<cxx::EnumeratorSymbol>(id->symbol)) {
        if (!enumerator->value()) {
          fail(ast, "enumerator has no resolved constant value");
        }
        return name(scalars_.constant(*enumerator->value(), ast->type, ast),
                    cxx::to_string(enumerator->name()));
      }
      if (auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(id->symbol)) {
        if (auto value = configs_.read(variable, &builder_, source)) {
          std::array<loom_value_id_t, 1> components = {*value};
          return name(value_arena_.capture(
                          types_.partition(variable->type(), ast), components),
                      cxx::to_string(variable->name()));
        }
        if (unit_.typeTraits().is_array(variable->type())) {
          return immutable_object(variable, id).pointer;
        }
        if (variable->constValue() &&
            !unit_.typeTraits().is_volatile(variable->type()) &&
            (variable->isConstexpr() ||
             unit_.typeTraits().is_const(variable->type()))) {
          cxx::ASTInterpreter interpreter(&unit_);
          if (auto value = interpreter.evaluate(ast)) {
            auto result = constant(*value, ast);
            // A reference parameter names its source object, not the value
            // produced by reading that object in this expression.
            return unit_.typeTraits().is_reference(variable->type())
                       ? result
                       : name(result, cxx::to_string(variable->name()));
          }
        }
      }
      if (cxx::symbol_cast<cxx::FieldSymbol>(id->symbol)) {
        fail(ast,
             "member-dependent initialization requires source object "
             "initialization semantics");
      }
      fail(ast, "unbound source value: " +
                    cxx::to_string(id->symbol ? id->symbol->name() : nullptr));
    }
    if (auto* literal = cxx::ast_cast<cxx::BoolLiteralExpressionAST>(ast)) {
      return scalars_.integer(literal->isTrue, LOOM_SCALAR_TYPE_I1, source);
    }
    if (auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(ast)) {
      return string_literal(literal).pointer;
    }
    if (cxx::ast_cast<cxx::IntLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::FloatLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::CharLiteralExpressionAST>(ast) ||
        cxx::ast_cast<cxx::SizeofExpressionAST>(ast) ||
        cxx::ast_cast<cxx::SizeofTypeExpressionAST>(ast) ||
        cxx::ast_cast<cxx::AlignofTypeExpressionAST>(ast) ||
        cxx::ast_cast<cxx::BuiltinOffsetofExpressionAST>(ast)) {
      cxx::ASTInterpreter interpreter(&unit_);
      auto value = interpreter.evaluate(ast);
      if (!value) {
        fail(ast, "literal or layout query has no constant value");
      }
      return scalars_.constant(*value, ast->type, ast);
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      if (auto* field = current_receiver_field(member)) {
        if (unit_.typeTraits().is_lvalue_reference(field->type())) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        return name(receiver_member(field, ast), cxx::to_string(field->name()));
      }
      auto* base = cxx::ast_cast<cxx::IdExpressionAST>(member->baseExpression);
      if (control_->storage_backed(member) ||
          unit_.typeTraits().is_array(member->type)) {
        return name(load(ast), cxx::to_string(member->symbol->name()));
      }
      // Concise source leaves such as workitem::id retain their complete
      // topology family in the durable IR value name.
      const char* topology_name = nullptr;
      decltype(&loom_kernel_workitem_id_build) topology_build = nullptr;
      if (base) {
        if (annotated(base->symbol, "workitem_id")) {
          topology_name = "workitem_id";
          topology_build = loom_kernel_workitem_id_build;
        } else if (annotated(base->symbol, "workgroup_id")) {
          topology_name = "workgroup_id";
          topology_build = loom_kernel_workgroup_id_build;
        } else if (annotated(base->symbol, "workgroup_size")) {
          topology_name = "workgroup_size";
          topology_build = loom_kernel_workgroup_size_build;
        } else if (annotated(base->symbol, "workgroup_count")) {
          topology_name = "workgroup_count";
          topology_build = loom_kernel_workgroup_count_build;
        }
      }
      if (!topology_build) {
        auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
        if (!field || field->isStatic()) {
          fail(ast, "record value access requires a non-static data member");
        }
        if (unit_.typeTraits().is_lvalue_reference(field->type())) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        auto value = expression(member->baseExpression);
        const auto& slice = types_.member(field, ast);
        return value.project(*slice.partition, slice.component_offset);
      }
      require_kernel_context(ast);
      auto axis = cxx::to_string(member->symbol->name());
      loom_kernel_dimension_t dimension;
      if (axis == "x") {
        dimension = LOOM_KERNEL_DIMENSION_X;
      } else if (axis == "y") {
        dimension = LOOM_KERNEL_DIMENSION_Y;
      } else if (axis == "z") {
        dimension = LOOM_KERNEL_DIMENSION_Z;
      } else {
        fail(ast, "unknown topology axis");
      }
      loom_op_t* op;
      auto coordinate = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      check(topology_build(&builder_, dimension, coordinate, source, &op));
      auto raw_value = loom_op_results(op)[0];
      const auto& partition = types_.partition(ast->type, ast);
      if (partition.kind == ValueKind::Address) {
        const auto& address = static_cast<const AddressPartition&>(partition);
        if (address.element_type != LOOM_SCALAR_TYPE_INDEX) {
          fail(ast, "kernel topology requires the Loom index domain");
        }
        std::array<loom_value_id_t, 1> components = {raw_value};
        return name(value_arena_.capture(partition, components),
                    std::string(topology_name) + "_" + axis);
      }
      auto value = name(raw_value, std::string(topology_name) + "_" + axis);
      check(loom_index_cast_build(&builder_, value, coordinate,
                                  types_.get(ast->type, ast), source, &op));
      return result(op);
    }
    if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast)) {
      if (binary->symbol) {
        auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(binary->symbol);
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (!binding) {
          auto call = direct_operator_call(ast);
          IREE_ASSERT(call);
          return ordinary_operator_value(ast, *call);
        }
        std::array<Value, 2> arguments = {
            expression(binary->leftExpression),
            expression(binary->rightExpression),
        };
        auto called =
            intrinsics_.call(*binding, arguments, value_arena_, storage_, ast,
                             lexical_fastmath_flags_, &builder_, source);
        if (!called.value) {
          fail(ast, "overloaded binary operation must produce a value");
        }
        return *called.value;
      }
      if (binary->op == cxx::TokenKind::T_COMMA) {
        effect(binary->leftExpression);
        return expression(binary->rightExpression);
      }
      auto left = expression(binary->leftExpression);
      if (binary->op == cxx::TokenKind::T_AMP_AMP ||
          binary->op == cxx::TokenKind::T_BAR_BAR) {
        if (types_.native_vector(ast->type)) {
          fail(ast, "vector logical operators require lane-wise evaluation");
        }
        // The semantic AST supplies both contextual boolean conversions.
        // The skipped arm forwards the left value, preserving its identity
        // without evaluating right-side memory accesses or binding updates.
        auto evaluate_right = [&] {
          return expression(binary->rightExpression);
        };
        auto skipped = [&] { return left; };
        return binary->op == cxx::TokenKind::T_AMP_AMP
                   ? conditional(ast, left.ssa(), evaluate_right, skipped)
                   : conditional(ast, left.ssa(), skipped, evaluate_right);
      }
      auto right = expression(binary->rightExpression);
      if (left.is_pointer() || right.is_pointer()) {
        if (left.is_pointer() && !right.is_pointer()) {
          return storage_
              .advance(
                  pointer_projection(left, binary->leftExpression->type, ast),
                  right.ssa(), binary->leftExpression->type,
                  binary->rightExpression->type, binary->op, ast)
              .pointer;
        }
        if (right.is_pointer() && !left.is_pointer() &&
            binary->op == cxx::TokenKind::T_PLUS) {
          return storage_
              .advance(
                  pointer_projection(right, binary->rightExpression->type, ast),
                  left.ssa(), binary->rightExpression->type,
                  binary->leftExpression->type, binary->op, ast)
              .pointer;
        }
        fail(ast,
             "pointer arithmetic requires a pointer and an integral "
             "displacement");
      }
      // C++ promotes shift operands independently. Loom's shift operands have
      // one width. Every defined source shift count fits the promoted left
      // width; normalize that count without changing any defined execution.
      if (binary->op == cxx::TokenKind::T_LESS_LESS ||
          binary->op == cxx::TokenKind::T_GREATER_GREATER) {
        right = numeric_convert(right.ssa(), binary->rightExpression->type,
                                binary->leftExpression->type, ast);
      }
      return this->binary(binary->op, left.ssa(), right.ssa(),
                          binary->leftExpression->type, ast->type, ast);
    }
    if (auto* postfix = cxx::ast_cast<cxx::PostIncrExpressionAST>(ast)) {
      if (postfix->symbol) {
        auto call = direct_operator_call(ast);
        IREE_ASSERT(call);
        return ordinary_operator_value(ast, *call);
      }
      return increment(postfix->baseExpression, postfix->op,
                       IncrementResult::Previous, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (unary->symbol) {
        auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(unary->symbol);
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (!binding) {
          auto call = direct_operator_call(ast);
          IREE_ASSERT(call);
          return ordinary_operator_value(ast, *call);
        }
        std::array<Value, 1> arguments = {expression(unary->expression)};
        auto called =
            intrinsics_.call(*binding, arguments, value_arena_, storage_, ast,
                             lexical_fastmath_flags_, &builder_, source);
        if (!called.value) {
          fail(ast, "overloaded unary operation must produce a value");
        }
        return *called.value;
      }
      if (unary->op == cxx::TokenKind::T_PLUS_PLUS ||
          unary->op == cxx::TokenKind::T_MINUS_MINUS) {
        return increment(unary->expression, unary->op, IncrementResult::Updated,
                         ast);
      }
      if (unary->op == cxx::TokenKind::T_AMP) {
        auto* operand = strip_implicit_casts(unary->expression);
        while (auto* nested =
                   cxx::ast_cast<cxx::NestedExpressionAST>(operand)) {
          operand = strip_implicit_casts(nested->expression);
        }
        if (auto* dereference = cxx::ast_cast<cxx::UnaryExpressionAST>(operand);
            dereference && !dereference->symbol &&
            dereference->op == cxx::TokenKind::T_STAR) {
          // Taking the address of an indirect object preserves the pointer;
          // it neither reads the object nor needs its storage layout.
          return expression(dereference->expression);
        }
        return object_address(unary->expression).pointer;
      }
      if (unary->op == cxx::TokenKind::T_STAR) {
        return load(ast);
      }
      if (unary->op != cxx::TokenKind::T_PLUS &&
          unary->op != cxx::TokenKind::T_MINUS &&
          unary->op != cxx::TokenKind::T_TILDE &&
          unary->op != cxx::TokenKind::T_EXCLAIM) {
        fail(ast, "unsupported unary value expression");
      }
      auto operand = expression(unary->expression);
      if (types_.native_vector(unary->expression->type)) {
        return vectors_.unary(unary->op, operand.ssa(), unary->expression->type,
                              ast->type, ast, lexical_fastmath_flags_);
      }
      if (unary->op == cxx::TokenKind::T_PLUS) {
        return operand;
      }
      loom_op_t* op;
      auto output = types_.get(ast->type, ast);
      auto value = operand.ssa();
      if (unary->op == cxx::TokenKind::T_MINUS) {
        if (types_.is_float(ast->type)) {
          check(loom_scalar_negf_build(&builder_, lexical_fastmath_flags_,
                                       value, output, source, &op));
        } else {
          auto zero =
              scalars_.integer(0, loom_type_element_type(output), source);
          check(loom_scalar_subi_build(&builder_, 0, zero, value, output,
                                       source, &op));
        }
        return result(op);
      }
      if (unary->op == cxx::TokenKind::T_TILDE ||
          unary->op == cxx::TokenKind::T_EXCLAIM) {
        value = scalars_.convert(value, unary->expression->type, ast->type, ast,
                                 lexical_fastmath_flags_);
        auto mask =
            scalars_.integer(unary->op == cxx::TokenKind::T_EXCLAIM ? 1 : -1,
                             loom_type_element_type(output), source);
        check(loom_scalar_xori_build(&builder_, value, mask, output, source,
                                     &op));
        return result(op);
      }
      fail(ast, "unsupported unary value expression");
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      if (subscript->symbol) {
        auto called = overloaded_subscript_call(subscript);
        if (!called) {
          fail(ast, "void overloaded subscript cannot be used as a value");
        }
        auto* signature =
            cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
        if (signature &&
            unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
          auto projection = storage_.project(
              called->pointer(),
              unit_.typeTraits().remove_reference(signature->returnType()),
              ast);
          if (unit_.typeTraits().is_class(ast->type)) {
            return objects_.load(projection, ast->type, ast);
          }
          return storage_.load(storage_.dereference(projection, ast->type, ast),
                               ast->type, ast);
        }
        return *called;
      }
      auto* base_type = subscript->baseExpression->type;
      auto* index_type = subscript->indexExpression->type;
      if (types_.native_vector(base_type) || types_.native_vector(index_type)) {
        auto base = expression(subscript->baseExpression);
        auto index = expression(subscript->indexExpression);
        if (!types_.native_vector(base_type)) {
          std::swap(base, index);
          std::swap(base_type, index_type);
        }
        return vectors_.extract(base.ssa(), index.ssa(), base_type, index_type,
                                ast);
      }
      return load(ast);
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      if (auto constant = fold_constant_reference_call(ast, call)) {
        return *constant;
      }
      if (auto called = builtin(call)) {
        return *called->value;
      }
      auto* function = direct_callee(call);
      if (!function) {
        fail(ast, "call must resolve to a function symbol");
      }
      if (intrinsics_.check_binding(function, ast) ||
          functions_.is_check_case(function) ||
          annotated(function, "check_benchmark")) {
        fail(ast,
             "check declarations cannot be called from ordinary functions");
      }
      auto* binding = intrinsics_.lookup(function, ast);
      if (binding &&
          (std::holds_alternative<KernelCollectiveIntrinsic>(*binding) ||
           std::holds_alternative<SubgroupShuffleIntrinsic>(*binding) ||
           std::holds_alternative<AsyncIntrinsic>(*binding))) {
        require_kernel_context(ast);
      }
      if (binding && std::holds_alternative<TargetIntrinsic>(*binding)) {
        require_launch_configuration_context(ast);
      }
      if (!binding) {
        if (annotated(function, "kernel")) {
          fail(ast, "kernels require an explicit launch operation");
        }
        if (functions_.is_command_program(function)) {
          fail(ast,
               "command programs require an explicit command launch "
               "operation");
        }
        if (pipeline_contracts_.contains(function)) {
          fail(ast,
               "pipeline contracts require an explicit kernel launch "
               "operation");
        }
        auto called = ordinary_call(call, function, source);
        if (!called) {
          fail(ast, "void function cannot be used as a value");
        }
        auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
        if (signature &&
            unit_.typeTraits().is_lvalue_reference(signature->returnType())) {
          auto projection = storage_.project(
              called->pointer(),
              unit_.typeTraits().remove_reference(signature->returnType()),
              ast);
          if (unit_.typeTraits().is_class(ast->type)) {
            return objects_.load(projection, ast->type, ast);
          }
          return storage_.load(storage_.dereference(projection, ast->type, ast),
                               ast->type, ast);
        }
        return *called;
      }
      if (auto* parameter = std::get_if<CommandParameterIntrinsic>(binding)) {
        return command_parameter(call, *parameter);
      }
      if (std::holds_alternative<KernelLaunchIntrinsic>(*binding) ||
          std::holds_alternative<CommandLaunchIntrinsic>(*binding)) {
        fail(ast, "void launch operation cannot be used as a value");
      }
      if (auto* buffer = std::get_if<BufferIntrinsic>(binding);
          buffer && buffer->has_literal()) {
        return buffer->call_literal(unit_, diagnostics_,
                                    call->expressionList->value, storage_, ast);
      }
      auto* assembly = std::get_if<AssemblyIntrinsic>(binding);
      loom_symbol_ref_t fragment = {};
      auto* expressions = call->expressionList;
      if (assembly) {
        fragment =
            assembly->fragment(unit_, diagnostics_, expressions->value, names_,
                               assembly_fragments_, module_, options_);
        expressions = expressions->next;
      }
      std::vector<Value> source_arguments;
      for (auto* argument : cxx::ListView{expressions}) {
        source_arguments.push_back(expression(argument));
      }
      if (assembly) {
        return *assembly->call(fragment, source_arguments, value_arena_,
                               &builder_, source);
      }
      auto called =
          intrinsics_.call(*binding, source_arguments, value_arena_, storage_,
                           ast, lexical_fastmath_flags_, &builder_, source);
      if (!called.value) {
        fail(ast, "void intrinsic cannot be used as a value");
      }
      return *called.value;
    }
    fail(ast,
         "unsupported expression: " + std::string(cxx::to_string(ast->kind())));
  }

  struct IterationFlow {
    // Whether execution reaches the remaining source statements in the body.
    loom_value_id_t reaches_tail;
    // Whether execution reaches the step or trailing condition. Invalid when
    // the body contains no break and only tail reachability is required.
    loom_value_id_t reaches_latch;
  };

  // Borrowed remaining statements from either the frontend's linked compound
  // body or a retained contiguous switch clause. Copies advance independently
  // without rebuilding or rescanning either source representation.
  struct StatementSequence {
    explicit StatementSequence(cxx::List<cxx::StatementAST*>* statements)
        : linked(statements) {}
    explicit StatementSequence(std::span<cxx::StatementAST* const> statements)
        : contiguous(statements) {}

    bool empty() const { return !linked && contiguous.empty(); }
    cxx::StatementAST* take() {
      if (linked) {
        auto* statement = linked->value;
        linked = linked->next;
        return statement;
      }
      auto* statement = contiguous.front();
      contiguous = contiguous.subspan(1);
      return statement;
    }

    // Remaining frontend list, when traversing an ordinary compound body.
    cxx::List<cxx::StatementAST*>* linked = nullptr;
    // Remaining retained clause segment, when traversing a switch body.
    std::span<cxx::StatementAST* const> contiguous;
  };

  ExitFlow iteration_exit_flow(cxx::StatementAST* statement,
                               bool tracks_break) {
    return tracks_break ? control_->iteration_exits(statement)
                        : control_->continues(statement);
  }

  IterationFlow constant_iteration_flow(bool reaches_tail, bool reaches_latch,
                                        bool tracks_break, cxx::AST* owner) {
    auto source = locations_.get(owner);
    auto tail = scalars_.integer(reaches_tail, LOOM_SCALAR_TYPE_I1, source);
    if (!tracks_break) {
      return {tail, LOOM_VALUE_ID_INVALID};
    }
    auto latch =
        reaches_tail == reaches_latch
            ? tail
            : scalars_.integer(reaches_latch, LOOM_SCALAR_TYPE_I1, source);
    return {tail, latch};
  }

  // Each arm yields body-tail reachability and, for break-bearing bodies,
  // latch reachability followed by the live bindings it changed. Arm-local
  // declarations stay inside the arm.
  template <typename Then, typename Else>
  IterationFlow iteration_branch(
      cxx::AST* owner, loom_value_id_t condition, bool tracks_break,
      std::span<cxx::Symbol* const> mutation_candidates, Then then_statement,
      Else else_statement) {
    auto written = live_mutations(mutation_candidates);
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
    if (tracks_break) {
      sources.push_back(unit_.control()->getBoolType());
    }
    const size_t control_count = sources.size();
    sources.reserve(control_count + written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto source = locations_.get(owner);
    auto result_signature = bind_signature(types_, sources, owner, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, result_signature.types.data(),
                            result_signature.types.size(), nullptr, 0, source,
                            &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    auto then_flow = then_statement();
    std::vector<loom_value_id_t> yielded = {then_flow.reaches_tail};
    if (tracks_break) {
      yielded.push_back(then_flow.reaches_latch);
    }
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    auto else_flow = else_statement();
    yielded = {else_flow.reaches_tail};
    if (tracks_break) {
      yielded.push_back(else_flow.reaches_latch);
    }
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
    bind_values(written, loom_op_results(op) + control_count);
    return {loom_op_results(op)[0],
            tracks_break ? loom_op_results(op)[1] : LOOM_VALUE_ID_INVALID};
  }

  IterationFlow iteration_sequence(cxx::AST* owner, StatementSequence remaining,
                                   uint8_t fastmath, bool tracks_break) {
    return iteration_sequence(owner, remaining, fastmath, tracks_break,
                              control_->written(owner));
  }

  IterationFlow iteration_sequence(cxx::AST* owner, StatementSequence remaining,
                                   uint8_t fastmath, bool tracks_break,
                                   std::span<cxx::Symbol* const> mutations) {
    FastMathScope math_scope(lexical_fastmath_flags_, fastmath);
    while (!remaining.empty()) {
      auto* child = remaining.take();
      auto flow = iteration_exit_flow(child, tracks_break);
      if (flow == ExitFlow::None) {
        statement(child);
        if (!control_->falls_through(child)) {
          return constant_iteration_flow(false, false, tracks_break, child);
        }
        continue;
      }
      return iteration_tail(owner, child, remaining, fastmath, tracks_break,
                            mutations);
    }
    return constant_iteration_flow(true, true, tracks_break, owner);
  }

  // A tail enters at most one arm directly. When both arms can reach it, join
  // their fallthrough values first and emit the shared source just once.
  IterationFlow iteration_tail(cxx::AST* owner, cxx::StatementAST* ast,
                               StatementSequence remaining,
                               uint8_t continuation_fastmath, bool tracks_break,
                               std::span<cxx::Symbol* const> mutations) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    auto flow = iteration_exit_flow(ast, tracks_break);
    if (flow == ExitFlow::None) {
      if (ast) {
        statement(ast);
      }
      if (!control_->falls_through(ast)) {
        return constant_iteration_flow(false, false, tracks_break, ast);
      }
      return iteration_sequence(owner, remaining, continuation_fastmath,
                                tracks_break, mutations);
    }
    if (flow == ExitFlow::All || remaining.empty()) {
      return iteration_statement(ast, tracks_break);
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && (iteration_exit_flow(branch->statement, tracks_break) ==
                       ExitFlow::All ||
                   iteration_exit_flow(branch->elseStatement, tracks_break) ==
                       ExitFlow::All)) {
      auto condition = branch_condition(branch);
      return iteration_branch(
          owner, condition, tracks_break, mutations,
          [&] {
            return iteration_tail(owner, branch->statement, remaining,
                                  continuation_fastmath, tracks_break,
                                  mutations);
          },
          [&] {
            return iteration_tail(owner, branch->elseStatement, remaining,
                                  continuation_fastmath, tracks_break,
                                  mutations);
          });
    }
    auto fallthrough = iteration_statement(ast, tracks_break);
    return iteration_branch(
        owner, fallthrough.reaches_tail, tracks_break, mutations,
        [&] {
          return iteration_sequence(owner, remaining, continuation_fastmath,
                                    tracks_break, mutations);
        },
        [&] {
          auto stopped =
              constant_iteration_flow(false, false, tracks_break, ast);
          stopped.reaches_latch = fallthrough.reaches_latch;
          return stopped;
        });
  }

  loom_value_id_t predicate_or(loom_value_id_t left, loom_value_id_t right,
                               cxx::AST* owner) {
    loom_op_t* op;
    check(loom_scalar_ori_build(&builder_, left, right,
                                loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                locations_.get(owner), &op));
    return result(op);
  }

  loom_value_id_t predicate_not(loom_value_id_t value, cxx::AST* owner) {
    auto source = locations_.get(owner);
    auto one = scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
    loom_op_t* op;
    check(loom_scalar_xori_build(&builder_, value, one,
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I1), source,
                                 &op));
    return result(op);
  }

  loom_value_id_t switch_case_match(loom_value_id_t selector, std::int64_t key,
                                    cxx::AST* owner) {
    auto source = locations_.get(owner);
    loom_op_t* constant;
    check(loom_index_constant_build(&builder_, loom_attr_i64(key),
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    source, &constant));
    loom_op_t* comparison;
    check(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_EQ, selector,
                               loom_index_constant_result(constant), source,
                               &comparison));
    return loom_index_cmp_result(comparison);
  }

  loom_value_id_t switch_selector(cxx::SwitchStatementAST* selection) {
    if (selection->initializer) {
      statement(selection->initializer);
    }
    initialize_condition(selection->condition);
    auto selector = expression(selection->condition).ssa();
    auto input_type = value_type(selector);
    auto i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    auto source = locations_.get(selection->condition);
    loom_op_t* op;
    // index.cast interprets fixed-width payloads as signed. Preserve the C++
    // value of a narrower unsigned selector before entering that domain.
    if (types_.is_unsigned(selection->condition->type) &&
        !loom_type_equal(input_type, i64_type)) {
      check(loom_scalar_extui_build(&builder_, selector, input_type, i64_type,
                                    source, &op));
      selector = result(op);
      input_type = i64_type;
    }
    check(loom_index_cast_build(&builder_, selector, input_type,
                                loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                source, &op));
    return name(result(op), "switch_selector");
  }

  IterationFlow switch_clause_flow(const SwitchClause& clause) {
    return iteration_sequence(
        clause.labels.front().source, StatementSequence(clause.statements),
        lexical_fastmath_flags_, /*tracks_break=*/true, clause.writes);
  }

  // A clause's internal break reaches the switch continuation, while continue
  // skips that continuation for the enclosing iteration. Normal completion
  // also reaches the switch continuation.
  loom_value_id_t switch_reaches_tail(IterationFlow flow, cxx::AST* owner) {
    return predicate_or(flow.reaches_tail,
                        predicate_not(flow.reaches_latch, owner), owner);
  }

  IterationFlow direct_switch(cxx::SwitchStatementAST* selection,
                              loom_value_id_t selector,
                              const SwitchLayout& layout,
                              bool tracks_outer_break) {
    const bool tracks_continue =
        control_->continues(selection) != ExitFlow::None;
    auto written = live_mutations(selection);
    std::vector<const cxx::Type*> sources;
    if (tracks_continue) {
      sources.push_back(unit_.control()->getBoolType());
    }
    const size_t control_count = sources.size();
    sources.reserve(control_count + written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto signature = bind_signature(types_, sources, selection, &builder_);

    struct Case {
      std::int64_t key;
      const SwitchClause* clause;
    };
    std::vector<Case> cases;
    const SwitchClause* fallback = nullptr;
    for (const auto& clause : layout.clauses) {
      const auto& label = clause.labels.front();
      if (label.key) {
        cases.push_back({*label.key, &clause});
      } else {
        fallback = &clause;
      }
    }
    std::ranges::sort(cases, {}, &Case::key);
    std::vector<std::int64_t> keys;
    keys.reserve(cases.size());
    for (const auto& source_case : cases) {
      keys.push_back(source_case.key);
    }

    auto source = locations_.get(selection);
    loom_op_t* op;
    check(loom_scf_switch_build(&builder_, selector, signature.types.data(),
                                signature.types.size(), nullptr, 0, keys.data(),
                                keys.size(), source, &op));
    auto outer_values = values_;
    auto yield_clause = [&](const SwitchClause* clause, loom_region_t* region) {
      auto* clause_source = clause ? clause->labels.front().source : selection;
      auto saved = loom_builder_enter_region(&builder_, op, region);
      values_ = outer_values;
      auto flow = clause ? switch_clause_flow(*clause)
                         : constant_iteration_flow(
                               true, true, /*tracks_break=*/true, selection);
      std::vector<loom_value_id_t> yielded;
      if (tracks_continue) {
        yielded.push_back(switch_reaches_tail(flow, clause_source));
      }
      auto mutations = current(written);
      yielded.insert(yielded.end(), mutations.begin(), mutations.end());
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(clause_source), &yield));
      loom_builder_restore(&builder_, saved);
    };

    auto case_regions = loom_scf_switch_case_regions(op);
    for (size_t index = 0; index < cases.size(); ++index) {
      yield_clause(cases[index].clause, case_regions.regions[index]);
    }
    yield_clause(fallback, loom_scf_switch_default_region(op));
    values_ = std::move(outer_values);
    bind_values(written, loom_op_results(op) + control_count);

    auto reaches_tail =
        tracks_continue ? loom_op_results(op)[0]
                        : scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
    auto reaches_latch =
        tracks_outer_break ? scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source)
                           : LOOM_VALUE_ID_INVALID;
    return {reaches_tail, reaches_latch};
  }

  IterationFlow linear_switch(cxx::SwitchStatementAST* selection,
                              loom_value_id_t selector,
                              const SwitchLayout& layout,
                              bool tracks_outer_break) {
    auto source = locations_.get(selection);
    auto false_value = scalars_.integer(false, LOOM_SCALAR_TYPE_I1, source);
    auto true_value = scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
    std::vector<loom_value_id_t> activations(layout.clauses.size(),
                                             false_value);
    auto any_match = false_value;
    std::optional<size_t> fallback;
    for (size_t index = 0; index < layout.clauses.size(); ++index) {
      for (const auto& label : layout.clauses[index].labels) {
        if (!label.key) {
          fallback = index;
          continue;
        }
        auto match = switch_case_match(selector, *label.key, label.source);
        activations[index] =
            predicate_or(activations[index], match, label.source);
        any_match = predicate_or(any_match, match, label.source);
      }
    }
    if (fallback) {
      activations[*fallback] =
          predicate_or(activations[*fallback],
                       predicate_not(any_match, selection), selection);
    }

    const bool tracks_continue =
        control_->continues(selection) != ExitFlow::None;
    auto active = false_value;
    auto reaches_tail = true_value;
    for (size_t index = 0; index < layout.clauses.size(); ++index) {
      const auto& clause = layout.clauses[index];
      auto entered = predicate_or(active, activations[index],
                                  clause.labels.front().source);
      auto written = live_mutations(clause.writes);
      std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
      if (tracks_continue) {
        sources.push_back(unit_.control()->getBoolType());
      }
      const size_t control_count = sources.size();
      sources.reserve(control_count + written.size());
      for (auto* symbol : written) {
        sources.push_back(types_.unqualified_value(symbol->type()));
      }
      auto signature = bind_signature(types_, sources, selection, &builder_);
      loom_op_t* op;
      check(loom_scf_if_build(
          &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, entered,
          signature.types.data(), signature.types.size(), nullptr, 0,
          locations_.get(clause.labels.front().source), &op));
      auto outer_values = values_;
      auto saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
      auto flow = switch_clause_flow(clause);
      std::vector<loom_value_id_t> yielded = {flow.reaches_tail};
      if (tracks_continue) {
        yielded.push_back(
            switch_reaches_tail(flow, clause.labels.front().source));
      }
      auto mutations = current(written);
      yielded.insert(yielded.end(), mutations.begin(), mutations.end());
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(clause.labels.front().source),
                                 &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;

      saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
      yielded = {active};
      if (tracks_continue) {
        yielded.push_back(reaches_tail);
      }
      mutations = current(written);
      yielded.insert(yielded.end(), mutations.begin(), mutations.end());
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(clause.labels.front().source),
                                 &yield));
      loom_builder_restore(&builder_, saved);
      values_ = std::move(outer_values);
      active = loom_op_results(op)[0];
      if (tracks_continue) {
        reaches_tail = loom_op_results(op)[1];
      }
      bind_values(written, loom_op_results(op) + control_count);
    }

    auto reaches_latch =
        tracks_outer_break ? true_value : LOOM_VALUE_ID_INVALID;
    return {reaches_tail, reaches_latch};
  }

  IterationFlow switch_statement(cxx::SwitchStatementAST* selection,
                                 bool tracks_outer_break) {
    auto selector = switch_selector(selection);
    const auto& layout = control_->switch_layout(selection);
    auto flow =
        layout.direct
            ? direct_switch(selection, selector, layout, tracks_outer_break)
            : linear_switch(selection, selector, layout, tracks_outer_break);
    if (auto* decision = condition_declaration(selection->condition)) {
      values_.erase(decision->symbol);
      locals_.erase(decision->symbol);
    }
    return flow;
  }

  IterationFlow iteration_statement(cxx::StatementAST* ast, bool tracks_break) {
    auto* source = ast;
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    if (!ast) {
      return constant_iteration_flow(true, true, tracks_break, source);
    }
    if (iteration_exit_flow(ast, tracks_break) == ExitFlow::None) {
      statement(ast);
      auto falls_through = control_->falls_through(ast);
      return constant_iteration_flow(falls_through, falls_through, tracks_break,
                                     ast);
    }
    if (cxx::ast_cast<cxx::ContinueStatementAST>(ast)) {
      return constant_iteration_flow(false, true, tracks_break, ast);
    }
    if (cxx::ast_cast<cxx::BreakStatementAST>(ast)) {
      return constant_iteration_flow(false, false, tracks_break, ast);
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return iteration_sequence(compound,
                                StatementSequence(compound->statementList),
                                lexical_fastmath_flags_, tracks_break);
    }
    if (auto* selection = cxx::ast_cast<cxx::SwitchStatementAST>(ast)) {
      return switch_statement(selection, tracks_break);
    }
    // The retained path analysis propagates iteration exits through blocks
    // and admitted control statements only; nested loops consume their own
    // exits. A retained construct must still fail through the ordinary source
    // diagnostic until translation admits it.
    auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
    if (!branch) {
      fail(ast, "unsupported statement: " +
                    std::string(cxx::to_string(ast->kind())));
    }
    auto condition = branch_condition(branch);
    return iteration_branch(
        ast, condition, tracks_break, control_->written(ast),
        [&] { return iteration_statement(branch->statement, tracks_break); },
        [&] {
          return branch->elseStatement
                     ? iteration_statement(branch->elseStatement, tracks_break)
                     : constant_iteration_flow(true, true, tracks_break, ast);
        });
  }

  // Iteration transfers end only this body. The caller decides whether the
  // flow reaches the for step or do condition. Break-free bodies preserve
  // their existing structured IR and carry no latch value.
  IterationFlow loop_body(cxx::StatementAST* body, bool tracks_break = false) {
    if (iteration_exit_flow(body, tracks_break) == ExitFlow::None) {
      statement(body);
      return tracks_break
                 ? constant_iteration_flow(true, true, tracks_break, body)
                 : IterationFlow{LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
    }
    return iteration_statement(body, tracks_break);
  }

  void command_schedule(const CommandSchedule& schedule) {
    if (current_function_.kind != FunctionKind::CommandProgram) {
      fail(schedule.body,
           "serial and concurrent regions require a command_program body");
    }
    if (!live_mutations(schedule.body).empty()) {
      fail(schedule.body,
           "command schedule regions cannot update enclosing automatic "
           "values because they have no results");
    }
    if (control_->returns(schedule.body) != ExitFlow::None) {
      fail(schedule.body,
           "command schedule regions cannot contain function returns");
    }
    if (control_->iteration_exits(schedule.body) != ExitFlow::None) {
      fail(schedule.body,
           "command schedule regions cannot exit an enclosing loop");
    }

    loom_op_t* op;
    if (schedule.kind == CommandSchedule::Kind::Serial) {
      check(loom_command_serial_build(&builder_, locations_.get(schedule.body),
                                      &op));
    } else {
      check(loom_command_concurrent_build(&builder_,
                                          locations_.get(schedule.body), &op));
    }
    auto* region = schedule.kind == CommandSchedule::Kind::Serial
                       ? loom_command_serial_body(op)
                       : loom_command_concurrent_body(op);
    auto saved = loom_builder_enter_region(&builder_, op, region);
    auto outer_values = values_;
    for (auto* child : cxx::ListView{schedule.body->statementList}) {
      statement(child);
      if (!control_->falls_through(child)) {
        break;
      }
    }
    loom_op_t* terminator;
    check(loom_command_yield_build(&builder_, locations_.get(schedule.body),
                                   &terminator));
    values_ = std::move(outer_values);
    loom_builder_restore(&builder_, saved);
  }

  void statement(cxx::StatementAST* ast) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    if (!ast) {
      return;
    }
    if (auto schedule = CommandSchedule::resolve(unit_, diagnostics_, ast)) {
      command_schedule(*schedule);
      return;
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      for (auto* child : cxx::ListView{compound->statementList}) {
        statement(child);
        if (!control_->falls_through(child)) {
          break;
        }
      }
      return;
    }
    if (auto* returned = cxx::ast_cast<cxx::ReturnStatementAST>(ast)) {
      function_exit(returned);
      return;
    }
    if (auto* declaration = cxx::ast_cast<cxx::DeclarationStatementAST>(ast)) {
      if (cxx::ast_cast<cxx::EmptyDeclarationAST>(declaration->declaration)) {
        return;
      }
      // The frontend checks assertions, including dependent conditions after
      // instantiation. Successful assertions have no runtime semantics.
      if (cxx::ast_cast<cxx::StaticAssertDeclarationAST>(
              declaration->declaration)) {
        return;
      }
      if (auto* alias = cxx::ast_cast<cxx::AliasDeclarationAST>(
              declaration->declaration)) {
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            alias->attributeList);
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            alias->typeId->attributeList);
        reject_misplaced_binding_declarator(unit_, diagnostics_,
                                            alias->typeId->declarator);
        return;
      }
      if (auto* directive =
              cxx::ast_cast<cxx::UsingDirectiveAST>(declaration->declaration)) {
        reject_misplaced_binding_attributes(unit_, diagnostics_,
                                            directive->attributeList);
        return;
      }
      if (cxx::ast_cast<cxx::UsingDeclarationAST>(declaration->declaration)) {
        return;
      }
      auto* simple =
          cxx::ast_cast<cxx::SimpleDeclarationAST>(declaration->declaration);
      if (!simple) {
        fail(ast, "unsupported local declaration");
      }
      reject_misplaced_binding_attributes(
          unit_, diagnostics_, simple->attributeList,
          BindingAttributeScope::LocalDeclaration);
      if (!simple->initDeclaratorList) {
        admit_constexpr_contract(unit_, diagnostics_, types_,
                                 simple->attributeList, nullptr, false, simple);
      }
      for (auto* variable : cxx::ListView{simple->initDeclaratorList}) {
        reject_misplaced_binding_declarator(unit_, diagnostics_,
                                            variable->declarator);
        auto* source_variable =
            cxx::symbol_cast<cxx::VariableSymbol>(variable->symbol);
        auto constexpr_contract = admit_constexpr_contract(
            unit_, diagnostics_, types_, simple->attributeList, source_variable,
            source_variable && control_->addressed(source_variable), variable);
        if (cxx::symbol_cast<cxx::TypeAliasSymbol>(variable->symbol)) {
          continue;
        }
        if (!source_variable || source_variable->isStatic() ||
            source_variable->isExtern() || source_variable->isThreadLocal()) {
          fail(ast, "local storage duration must be automatic or __shared__");
        }
        if (variable->symbol && annotated(variable->symbol, "workgroup")) {
          if (current_function_.kind != FunctionKind::Kernel &&
              current_function_.kind != FunctionKind::TemplateDefinition) {
            fail(variable,
                 "workgroup storage requires a kernel or template definition "
                 "body");
          }
          if (variable->initializer) {
            fail(variable, "workgroup storage cannot have an initializer");
          }
          auto* source_type = types_.unqualified(variable->symbol->type());
          auto* array = cxx::type_cast<cxx::BoundedArrayType>(source_type);
          auto allocation = storage_.allocate(
              source_type, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
              source_variable->explicitAlignment(), variable);
          auto spelling = cxx::to_string(variable->symbol->name());
          if (array) {
            values_[variable->symbol] =
                name(Value(allocation.pointer), spelling);
          } else {
            name(Value(allocation.pointer), spelling + "_storage");
            locals_[variable->symbol] = allocation;
          }
          name(allocation.view, spelling + "_view");
          continue;
        }
        initialize_variable(source_variable, variable->initializer,
                            constexpr_contract, variable);
      }
      return;
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      auto condition = branch_condition(branch);
      auto saved_values = values_;
      auto written = live_mutations(ast);
      std::vector<const cxx::Type*> sources;
      sources.reserve(written.size());
      for (auto* symbol : written) {
        sources.push_back(types_.unqualified_value(symbol->type()));
      }
      auto results = bind_signature(types_, sources, ast, &builder_);
      loom_op_t* op;
      auto flags = branch->elseStatement || !written.empty()
                       ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION
                       : 0;
      check(loom_scf_if_build(&builder_, flags, condition, results.types.data(),
                              results.types.size(), nullptr, 0,
                              locations_.get(ast), &op));
      auto saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
      statement(branch->statement);
      loom_op_t* yield;
      auto yielded = current(written);
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(ast), &yield));
      values_ = saved_values;
      loom_builder_restore(&builder_, saved);
      if (flags) {
        saved = loom_builder_enter_region(&builder_, op,
                                          loom_scf_if_else_region(op));
        if (branch->elseStatement) {
          statement(branch->elseStatement);
        }
        yielded = current(written);
        check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                   locations_.get(ast), &yield));
        values_ = saved_values;
        loom_builder_restore(&builder_, saved);
      }
      bind_values(written, loom_op_results(op));
      return;
    }
    if (auto* selection = cxx::ast_cast<cxx::SwitchStatementAST>(ast)) {
      switch_statement(selection, /*tracks_outer_break=*/false);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(ast)) {
      LoopSchedule schedule(unit_, diagnostics_, types_, loop->attributeList);
      if (loop->initializer) {
        statement(loop->initializer);
      }
      if (control_->breaks(loop->statement) != ExitFlow::None) {
        if (!schedule.empty()) {
          fail(ast, "loop scheduling cannot preserve break exits");
        }
        conditional_loop(ast, loop->condition, loop->statement,
                         loop->expression, LoopTest::BeforeBody);
        return;
      }
      if (auto* counted = control_->counted(loop)) {
        counted_loop(loop, *counted, schedule);
        return;
      }
      if (!schedule.empty()) {
        fail(ast, "loop scheduling requires a nonwrapping counted for loop");
      }
      conditional_loop(ast, loop->condition, loop->statement, loop->expression,
                       LoopTest::BeforeBody);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::WhileStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, types_, loop->attributeList)
               .empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->condition, loop->statement, nullptr,
                       LoopTest::BeforeBody);
      return;
    }
    if (auto* loop = cxx::ast_cast<cxx::DoStatementAST>(ast)) {
      if (!LoopSchedule(unit_, diagnostics_, types_, loop->attributeList)
               .empty()) {
        fail(ast, "loop scheduling requires a counted for loop");
      }
      conditional_loop(ast, loop->expression, loop->statement, nullptr,
                       LoopTest::AfterBody);
      return;
    }
    if (auto* expression_statement =
            cxx::ast_cast<cxx::ExpressionStatementAST>(ast)) {
      if (expression_statement->expression) {
        effect(expression_statement->expression);
      }
      return;
    }
    fail(ast,
         "unsupported statement: " + std::string(cxx::to_string(ast->kind())));
  }

  enum class LoopTest { BeforeBody, AfterBody };

  // Evaluates a source condition only when its loop remains active. The
  // inactive arm yields false and forwards source state without evaluating
  // condition side effects.
  template <typename Evaluate>
  loom_value_id_t guarded_condition(cxx::AST* owner, loom_value_id_t active,
                                    Evaluate evaluate) {
    auto written = live_mutations(owner);
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
    sources.reserve(1 + written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto source = locations_.get(owner);
    auto result_signature = bind_signature(types_, sources, owner, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            active, result_signature.types.data(),
                            result_signature.types.size(), nullptr, 0, source,
                            &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    std::vector<loom_value_id_t> yielded = {evaluate()};
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    yielded = {
        scalars_.integer(false, LOOM_SCALAR_TYPE_I1, locations_.get(owner))};
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
    bind_values(written, loom_op_results(op) + 1);
    return loom_op_results(op)[0];
  }

  // Executes a loop latch only when the body reaches it. The inactive arm
  // forwards source state, skipping a for step and condition reinitialization.
  template <typename Execute>
  void guarded_latch(cxx::AST* owner, const std::vector<cxx::Symbol*>& written,
                     loom_value_id_t active, Execute execute) {
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources;
    sources.reserve(written.size());
    for (auto* symbol : written) {
      sources.push_back(types_.unqualified_value(symbol->type()));
    }
    auto source = locations_.get(owner);
    auto result_signature = bind_signature(types_, sources, owner, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            active, result_signature.types.data(),
                            result_signature.types.size(), nullptr, 0, source,
                            &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    execute();
    auto yielded = current(written);
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    yielded = current(written);
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = std::move(outer_values);
    bind_values(written, loom_op_results(op));
  }

  // The before region's forwarded values are also the loop's final results.
  // A post-test loop therefore runs its body there and uses an identity after
  // region; the final body mutations survive the false condition.
  void conditional_loop(cxx::StatementAST* ast,
                        cxx::ExpressionAST* condition_expression,
                        cxx::StatementAST* body, cxx::ExpressionAST* step,
                        LoopTest test) {
    auto* declaration = condition_declaration(condition_expression);
    auto* decision = declaration ? declaration->symbol : nullptr;
    auto constexpr_contract = initialize_condition(declaration);
    auto written = live_mutations(ast);
    if (decision && values_.contains(decision)) {
      std::erase(written, decision);
      written.push_back(decision);
    }
    auto initial = current(written);
    auto outer_values = values_;
    auto source = locations_.get(ast);
    auto dependent_state_sources = dependent_source_types(written, ast);
    auto bind_loop_signatures =
        [&](std::span<const loom_type_t> header_suffix_types,
            bool require_result_types) {
          if (!dependent_state_sources.empty()) {
            return bind_condition_loop_signatures(
                types_, dependent_state_sources, ast, &builder_,
                header_suffix_types, require_result_types);
          }
          if (require_result_types) {
            auto state_sources = source_types(written);
            return bind_condition_loop_signatures(
                types_, state_sources, ast, &builder_, header_suffix_types,
                require_result_types);
          }
          return BoundConditionLoopSignatures{};
        };
    auto evaluate_condition = [&] {
      return condition_expression
                 ? expression(condition_expression).ssa()
                 : scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
    };

    if (control_->breaks(body) != ExitFlow::None) {
      loom_op_t* op;
      if (test == LoopTest::BeforeBody) {
        auto active =
            scalars_.integer(true, LOOM_SCALAR_TYPE_I1, locations_.get(ast));
        auto header_initial = initial;
        header_initial.push_back(active);
        const std::array<loom_type_t, 1> header_suffix_types = {
            loom_type_scalar(LOOM_SCALAR_TYPE_I1)};
        auto signatures = bind_loop_signatures(header_suffix_types,
                                               /*require_result_types=*/true);
        check(loom_scf_while_build(
            &builder_, header_initial.data(), header_initial.size(),
            signatures.header_types.empty() ? nullptr
                                            : signatures.header_types.data(),
            signatures.result_types.data(), initial.size(), nullptr, 0, source,
            &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(written, before);
        auto header_active =
            loom_region_entry_arg_id(before, header_initial.size() - 1u);
        auto condition =
            guarded_condition(ast, header_active, evaluate_condition);
        auto forwarded = current(written);
        loom_op_t* terminator;
        check(loom_scf_condition_build(&builder_, condition, forwarded.data(),
                                       forwarded.size(), source, &terminator));
        loom_builder_restore(&builder_, saved);
        auto* after = loom_scf_while_after(op);
        saved = loom_builder_enter_region(&builder_, op, after);
        values_ = outer_values;
        bind(written, after);
        auto flow = loop_body(body, /*tracks_break=*/true);
        if (step || decision) {
          guarded_latch(ast, written, flow.reaches_latch, [&] {
            if (step) {
              effect(step);
            }
            if (decision) {
              reinitialize_condition(declaration, constexpr_contract, ast);
            }
          });
        }
        auto yielded = current(written);
        yielded.push_back(flow.reaches_latch);
        check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                   source, &terminator));
        loom_builder_restore(&builder_, saved);
      } else {
        auto signatures =
            bind_loop_signatures({}, /*require_result_types=*/false);
        check(loom_scf_while_build(
            &builder_, initial.data(), initial.size(),
            signatures.header_types.empty() ? nullptr
                                            : signatures.header_types.data(),
            signatures.result_types.empty() ? nullptr
                                            : signatures.result_types.data(),
            initial.size(), nullptr, 0, source, &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(written, before);
        auto flow = loop_body(body, /*tracks_break=*/true);
        auto condition =
            guarded_condition(ast, flow.reaches_latch, evaluate_condition);
        auto forwarded = current(written);
        loom_op_t* terminator;
        check(loom_scf_condition_build(&builder_, condition, forwarded.data(),
                                       forwarded.size(), source, &terminator));
        loom_builder_restore(&builder_, saved);
        auto* after = loom_scf_while_after(op);
        saved = loom_builder_enter_region(&builder_, op, after);
        values_ = outer_values;
        bind(written, after);
        if (decision) {
          reinitialize_condition(declaration, constexpr_contract, ast);
        }
        auto yielded = current(written);
        check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                   source, &terminator));
        loom_builder_restore(&builder_, saved);
      }
      values_ = outer_values;
      bind_values(written, loom_op_results(op));
      if (decision) {
        values_.erase(decision);
        locals_.erase(decision);
      }
      return;
    }

    loom_op_t* op;
    auto signatures = bind_loop_signatures({}, /*require_result_types=*/false);
    check(loom_scf_while_build(
        &builder_, initial.data(), initial.size(),
        signatures.header_types.empty() ? nullptr
                                        : signatures.header_types.data(),
        signatures.result_types.empty() ? nullptr
                                        : signatures.result_types.data(),
        initial.size(), nullptr, 0, source, &op));
    auto* before = loom_scf_while_before(op);
    auto saved = loom_builder_enter_region(&builder_, op, before);
    bind(written, before);
    if (test == LoopTest::AfterBody) {
      loop_body(body);
    }
    auto condition = evaluate_condition();
    auto forwarded = current(written);
    loom_op_t* terminator;
    check(loom_scf_condition_build(&builder_, condition, forwarded.data(),
                                   forwarded.size(), source, &terminator));
    loom_builder_restore(&builder_, saved);
    auto* after = loom_scf_while_after(op);
    saved = loom_builder_enter_region(&builder_, op, after);
    values_ = outer_values;
    bind(written, after);
    if (test == LoopTest::BeforeBody) {
      loop_body(body);
    }
    if (step) {
      effect(step);
    }
    // A decision object is recreated after the body and for-loop increment.
    // The preheader supplied its first value, so every check sees a real value.
    if (decision) {
      reinitialize_condition(declaration, constexpr_contract, ast);
    }
    auto yielded = current(written);
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &terminator));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    bind_values(written, loom_op_results(op));
    if (decision) {
      values_.erase(decision);
      locals_.erase(decision);
    }
  }

  loom_value_id_t unsigned_offset(loom_value_id_t value,
                                  loom_location_id_t source) {
    loom_op_t* cast;
    check(loom_index_cast_build(&builder_, value, value_type(value),
                                loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                source, &cast));
    return result(cast);
  }

  loom_value_id_t signed_index(loom_value_id_t value,
                               loom_location_id_t source) {
    loom_op_t* cast;
    check(loom_index_cast_build(&builder_, value, value_type(value),
                                loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                source, &cast));
    return result(cast);
  }

  void counted_loop(cxx::ForStatementAST* loop, const CountedLoop& counted,
                    const LoopSchedule& schedule) {
    auto source = locations_.get(loop);
    auto* induction = counted.induction;
    loom_value_id_t lower;
    loom_value_id_t upper;
    loom_value_id_t step;
    bool native_address = false;
    if (auto* range = std::get_if<CountedLoop::UnsignedRange>(&counted.range)) {
      lower = unsigned_offset(values_.at(induction).ssa(), source);
      auto* constant_upper = std::get_if<unsigned>(&range->upper);
      auto bound =
          constant_upper
              ? scalars_.integer(static_cast<int32_t>(*constant_upper),
                                 LOOM_SCALAR_TYPE_I32, source)
              : expression(
                    std::get<CountedLoop::Bound>(range->upper).expression)
                    .ssa();
      upper = unsigned_offset(bound, source);
      step = scalars_.integer(range->step, LOOM_SCALAR_TYPE_OFFSET, source);
    } else if (auto* range =
                   std::get_if<CountedLoop::SignedRange>(&counted.range)) {
      lower = signed_index(values_.at(induction).ssa(), source);
      auto* constant_upper = std::get_if<std::int32_t>(&range->upper);
      auto bound =
          constant_upper
              ? scalars_.integer(*constant_upper, LOOM_SCALAR_TYPE_I32, source)
              : expression(
                    std::get<CountedLoop::Bound>(range->upper).expression)
                    .ssa();
      upper = signed_index(bound, source);
      step = scalars_.integer(range->step, LOOM_SCALAR_TYPE_INDEX, source);
    } else {
      const auto& address_range =
          std::get<CountedLoop::AddressRange>(counted.range);
      auto domain = address_range.domain == CountedLoop::AddressDomain::Index
                        ? LOOM_SCALAR_TYPE_INDEX
                        : LOOM_SCALAR_TYPE_OFFSET;
      lower = values_.at(induction).ssa();
      upper = expression(address_range.upper.expression).ssa();
      step = scalars_.integer(1, domain, source);
      native_address = true;
    }
    auto depth = schedule.pipeline_depth()
                     ? expression(schedule.pipeline_depth()).ssa()
                     : 0;
    auto factor = schedule.unroll_factor()
                      ? expression(schedule.unroll_factor()).ssa()
                      : 0;
    auto written = live_mutations(loop);
    std::erase(written, induction);
    auto initial = current(written);
    BoundSignature result_signature;
    auto state_sources = dependent_source_types(written, loop);
    if (!state_sources.empty()) {
      result_signature = bind_signature(types_, state_sources, loop, &builder_);
    }
    auto outer_values = values_;
    auto* op = schedule.build(&builder_, lower, upper, step, initial,
                              result_signature.types, depth, factor, source);
    auto* body = loom_scf_for_body(op);
    auto saved = loom_builder_enter_region(&builder_, op, body);
    auto iteration = name(loom_region_entry_arg_id(body, 0),
                          cxx::to_string(induction->name()));
    if (native_address) {
      std::array<loom_value_id_t, 1> components = {iteration};
      values_[induction] = value_arena_.capture(
          types_.partition(induction->type(), loop), components);
    } else {
      loom_op_t* cast;
      check(loom_index_cast_build(&builder_, iteration, value_type(iteration),
                                  types_.get(induction->type(), loop), source,
                                  &cast));
      values_[induction] = result(cast);
    }
    bind(written, body, 1);
    loop_body(loop->statement);
    auto yielded = current(written);
    loom_op_t* terminator;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &terminator));
    loom_builder_restore(&builder_, saved);
    values_ = outer_values;
    values_.erase(induction);
    bind_values(written, loom_op_results(op));
  }

  std::vector<cxx::Symbol*> live_mutations(cxx::AST* owner) {
    return live_mutations(control_->written(owner));
  }

  std::vector<cxx::Symbol*> live_mutations(
      std::span<cxx::Symbol* const> candidates) {
    std::vector<cxx::Symbol*> result;
    for (auto* symbol : candidates) {
      if (values_.contains(symbol)) {
        result.push_back(symbol);
      }
    }
    return result;
  }

  std::vector<const cxx::Type*> source_types(
      std::span<cxx::Symbol* const> symbols) {
    std::vector<const cxx::Type*> result;
    result.reserve(symbols.size());
    for (auto* symbol : symbols) {
      result.push_back(types_.unqualified_value(symbol->type()));
    }
    return result;
  }

  // Returns the complete state type list only when at least one source type
  // depends on destination identities. Ordinary scalar loops take the empty,
  // allocation-free path and let the generated builders infer their schemes.
  std::vector<const cxx::Type*> dependent_source_types(
      std::span<cxx::Symbol* const> symbols, cxx::AST* owner) {
    for (auto* symbol : symbols) {
      if (types_.requires_binding(types_.unqualified_value(symbol->type()),
                                  owner)) {
        return source_types(symbols);
      }
    }
    return {};
  }

  std::vector<loom_value_id_t> current(
      const std::vector<cxx::Symbol*>& symbols) {
    std::vector<loom_value_id_t> result;
    for (auto* symbol : symbols) {
      values_.at(symbol).append_to(result);
    }
    return result;
  }
  void bind(const std::vector<cxx::Symbol*>& symbols, loom_region_t* region,
            size_t index = 0) {
    for (auto* symbol : symbols) {
      const auto& partition = values_.at(symbol).partition();
      values_[symbol] = name(region_value(region, index, partition),
                             cxx::to_string(symbol->name()));
    }
  }
  void bind_values(const std::vector<cxx::Symbol*>& symbols,
                   const loom_value_id_t* components) {
    for (auto* symbol : symbols) {
      const auto& partition = values_.at(symbol).partition();
      auto count = partition.component_count;
      values_[symbol] =
          name(value_arena_.capture(partition, {components, count}),
               cxx::to_string(symbol->name()));
      if (count) {
        components += count;
      }
    }
  }

  void effect(cxx::ExpressionAST* ast) override {
    if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(ast);
        binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
      effect(binary->leftExpression);
      effect(binary->rightExpression);
      return;
    }
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      effect(nested->expression);
      return;
    }
    if (auto call = direct_operator_call(ast);
        call && types_.unqualified(ast->type)->kind() == cxx::TypeKind::kVoid) {
      if (ordinary_operator_call(ast, *call)) {
        fail(ast, "value-producing operator reached a void expression");
      }
      return;
    }
    if (types_.unqualified(ast->type)->kind() == cxx::TypeKind::kVoid) {
      if (auto* cast = cxx::ast_cast<cxx::CastExpressionAST>(ast)) {
        effect(cast->expression);
        return;
      }
      if (auto* cast = cxx::ast_cast<cxx::CppCastExpressionAST>(ast)) {
        effect(cast->expression);
        return;
      }
      if (auto* cast = cxx::ast_cast<cxx::TypeConstructionAST>(ast)) {
        for (auto* operand : cxx::ListView{cast->expressionList}) {
          effect(operand);
        }
        return;
      }
    }
    if (auto* call = cxx::ast_cast<cxx::CallExpressionAST>(ast)) {
      if (builtin(call)) {
        return;
      }
      auto* id = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      if (id && annotated(id->symbol, "assume")) {
        auto predicates = assumption_predicates(unit_, diagnostics_, call);
        for (const auto& predicate : predicates) {
          for (size_t i = 0; i < predicate.value_count; ++i) {
            const auto& source_value = predicate.values[i];
            if (source_value.origin == PredicateValueOrigin::Expression) {
              for (const auto& dependency : source_value.dependencies) {
                if (locals_.contains(dependency.binding)) {
                  fail(dependency.source,
                       "assume expressions cannot read an addressable binding "
                       "whose storage may change through an alias");
                }
                if (!values_.contains(dependency.binding)) {
                  fail(dependency.source,
                       "assume expressions can read only owned automatic "
                       "scalar bindings");
                }
              }
              continue;
            }
            if (source_value.origin != PredicateValueOrigin::Binding ||
                !source_value.members.empty()) {
              fail(source_value.source,
                   "assume can refine only direct automatic scalar bindings");
            }
            auto* symbol = source_value.binding;
            if (locals_.contains(symbol)) {
              fail(source_value.source,
                   "assume cannot retain facts for an addressable binding "
                   "whose storage may change through an alias");
            }
            if (!values_.contains(symbol)) {
              fail(source_value.source,
                   "assume can retain facts only for an owned automatic "
                   "scalar binding");
            }
          }
        }
        for (const auto& admitted : predicates) {
          std::array<loom_value_id_t, 3> values;
          std::array<loom_type_t, 3> value_types;
          for (size_t i = 0; i < admitted.value_count; ++i) {
            values[i] = expression(admitted.values[i].converted).ssa();
            value_types[i] =
                types_.get(admitted.values[i].converted->type, ast);
          }
          loom_predicate_t predicate = admitted.predicate;
          for (uint8_t i = 0; i < predicate.arg_count; ++i) {
            if (predicate.arg_tags[i] == LOOM_PRED_ARG_VALUE) {
              predicate.args[i] = values[predicate.args[i]];
            }
          }
          bool index_domain = admitted.value_count > 0;
          for (size_t i = 0; i < admitted.value_count; ++i) {
            auto element = loom_type_element_type(value_types[i]);
            index_domain &=
                loom_type_kind(value_types[i]) == LOOM_TYPE_SCALAR &&
                (element == LOOM_SCALAR_TYPE_INDEX ||
                 element == LOOM_SCALAR_TYPE_OFFSET);
          }
          loom_op_t* op;
          if (index_domain) {
            check(loom_index_assume_build(
                &builder_, values.data(), admitted.value_count, &predicate, 1,
                value_types.data(), admitted.value_count, locations_.get(ast),
                &op));
          } else {
            check(loom_scalar_assume_build(
                &builder_, values.data(), admitted.value_count, &predicate, 1,
                value_types.data(), admitted.value_count, locations_.get(ast),
                &op));
          }
          for (size_t i = 0; i < admitted.value_count; ++i) {
            const auto& source_value = admitted.values[i];
            if (source_value.origin == PredicateValueOrigin::Expression) {
              continue;
            }
            auto result = loom_op_results(op)[i];
            auto refined =
                index_domain
                    ? value_arena_.capture(
                          types_.partition(source_value.source->type, ast),
                          {&result, 1})
                    : Value(numeric_convert(result,
                                            source_value.converted->type,
                                            source_value.source->type, ast));
            values_[source_value.binding] =
                name(refined, cxx::to_string(source_value.binding->name()));
          }
        }
        return;
      }
      if (!id || !annotated(id->symbol, "barrier")) {
        if (types_.unqualified(ast->type)->kind() != cxx::TypeKind::kVoid) {
          expression(ast);
          return;
        }
        auto* function = direct_callee(call);
        if (function && (intrinsics_.check_binding(function, ast) ||
                         functions_.is_check_case(function) ||
                         annotated(function, "check_benchmark"))) {
          fail(ast,
               "check declarations cannot be called from ordinary functions");
        }
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (binding) {
          if (auto* launch = std::get_if<KernelLaunchIntrinsic>(binding)) {
            kernel_launch(call, *launch);
            return;
          }
          if (auto* launch = std::get_if<CommandLaunchIntrinsic>(binding)) {
            command_launch(call, *launch);
            return;
          }
          if (std::holds_alternative<AsyncIntrinsic>(*binding)) {
            require_kernel_context(ast);
          }
          auto* assembly = std::get_if<AssemblyIntrinsic>(binding);
          loom_symbol_ref_t fragment = {};
          auto* expressions = call->expressionList;
          if (assembly) {
            fragment = assembly->fragment(
                unit_, diagnostics_, expressions->value, names_,
                assembly_fragments_, module_, options_);
            expressions = expressions->next;
          }
          std::vector<Value> arguments;
          for (auto* argument : cxx::ListView{expressions}) {
            arguments.push_back(expression(argument));
          }
          if (assembly) {
            assembly->call(fragment, arguments, value_arena_, &builder_,
                           locations_.get(ast));
            return;
          }
          auto called = intrinsics_.call(*binding, arguments, value_arena_,
                                         storage_, ast, lexical_fastmath_flags_,
                                         &builder_, locations_.get(ast));
          if (called.value) {
            fail(ast, "value-producing intrinsic reached a void call");
          }
          return;
        }
        if (!function) {
          fail(ast, "void call must resolve to a function symbol");
        }
        if (annotated(function, "kernel")) {
          fail(ast, "kernels require an explicit launch operation");
        }
        if (functions_.is_command_program(function)) {
          fail(ast,
               "command programs require an explicit command launch "
               "operation");
        }
        if (pipeline_contracts_.contains(function)) {
          fail(ast,
               "pipeline contracts require an explicit kernel launch "
               "operation");
        }
        auto called = ordinary_call(call, function, locations_.get(ast));
        if (called) {
          fail(ast, "value-producing function reached a void call");
        }
        return;
      }
      if (call->expressionList) {
        fail(ast, "barrier does not take arguments");
      }
      // HIP __syncthreads covers both global and shared memory. Loom names one
      // memory space per barrier, so preserve both fences explicitly.
      loom_op_t* op;
      check(loom_kernel_barrier_build(
          &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
          LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
          locations_.get(ast), &op));
      check(loom_kernel_barrier_build(
          &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
          LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL,
          locations_.get(ast), &op));
      return;
    }
    expression(ast);
  }

  // Source AST/symbol lifetime ends after construction and verification.
  cxx::TranslationUnit& unit_;
  // Source diagnostics share frontend byte ranges and the caller sink.
  Diagnostics& diagnostics_;
  // Output arena owner.
  loom_module_t* module_;
  // Current insertion point in the structured output.
  loom_builder_t builder_ = {};
  // Retained source ranges copied into the output module.
  Locations locations_;
  // Source type and scalar representation contracts.
  Types types_;
  // Numeric builders consume evaluated operands without AST callbacks.
  Scalars scalars_;
  // Shared output namespace for callables and configuration symbols.
  SymbolNames names_;
  // Reached source-owned immutable objects retain translation-unit identity.
  ImmutableObjects immutable_objects_;
  // Explicit lexical floating-point permissions indexed by source scope.
  FastMathContracts fastmath_;
  // Callable predicates shared by definitions and template declarations.
  FunctionContracts function_contracts_;
  // Namespace-scope scalar configs retain key identity across source aliases.
  Configs configs_;
  // Source target definitions bind function-like operations without
  // target-specific importer code.
  TargetDefinitions target_definitions_;
  // Bodyful template providers retain late linker selection contracts.
  TemplateDefinitions template_definitions_;
  // Explicit vector builders retain lane widths and full-width source masks.
  Vectors vectors_;
  // Memory representations retain declared array extents and access shape.
  Storage storage_;
  // Immutable large source bindings, released after each function's maps.
  ValueArena value_arena_;
  // Source-layout object snapshots share the callable value partitions.
  Objects objects_;
  // Aggregate clauses preserve in-place initialization and copy ordering.
  Initialization initialization_;
  // Admitted launch contracts, including bounds from function redeclarations.
  LaunchContracts launches_;
  // Command annotations and the retained source/native signature permutation.
  CommandPrograms command_programs_;
  // External kernel-scoped pipeline contracts and staged signatures.
  PipelineContracts pipeline_contracts_;
  // Retained generated operation bindings for reached source declarations.
  Intrinsics intrinsics_;
  // Literal admission records one batch for source-boundary verification.
  AssemblyFragments assembly_fragments_;
  // Root selection, native definition contracts and reachable identities.
  Functions functions_;
  // Borrowed source configuration for this invocation.
  const loom_cxx_import_options_t& options_;
  // Translation-unit permissions inherited by unannotated function bodies.
  uint8_t default_fastmath_flags_;
  // Function and nested statement permissions active at the insertion point.
  uint8_t lexical_fastmath_flags_ = 0;
  // Source and native definition contracts for the body being translated.
  FunctionBody current_function_ = {};
  // Bound symbols, never identifier spellings, key source-to-SSA mappings.
  std::unordered_map<cxx::Symbol*, Value> values_;
  // Storage-backed automatic objects retain one allocation across direct and
  // aliased accesses. They are not mutable SSA bindings at region edges.
  std::unordered_map<cxx::Symbol*, StorageAllocation> locals_;
  // Each reached immutable source object has one entry-dominating root per
  // function. The translation-unit owner separately retains its payload.
  std::unordered_map<cxx::VariableSymbol*, StorageAllocation>
      immutable_storage_;
  // Last immutable-root setup op in the current function entry preamble.
  loom_op_t* immutable_preamble_end_ = nullptr;
  // Runtime C++ reference bindings retain one resolved lvalue and never
  // rebind. Storage references therefore preserve their root and byte offset
  // without re-evaluating a source address expression.
  std::unordered_map<cxx::Symbol*, Lvalue> references_;
  // Nested source current objects retain declaration identity independently of
  // spelling. The direct index avoids rediscovering an owning receiver through
  // the active stack for every member expression.
  std::vector<ActiveReceiver> receivers_;
  std::unordered_map<cxx::FieldSymbol*, ReceiverOwner> receiver_owners_;
  // Launch configurations not owned by the ordinary reachable-call analysis
  // retain a function-local analysis here.
  std::optional<ControlFlow> owned_control_;
  // Immutable control facts for the function currently being translated.
  const ControlFlow* control_ = nullptr;
};
}  // namespace

void translate(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
               loom_module_t* module,
               const loom_cxx_import_options_t& options) {
  Translator(unit, diagnostics, module, options).translate();
}

}  // namespace loom::cxx_import
