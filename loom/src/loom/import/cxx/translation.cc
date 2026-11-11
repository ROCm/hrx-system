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
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/assumptions.h"
#include "loom/import/cxx/binding/command_programs.h"
#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/constexpr.h"
#include "loom/import/cxx/binding/exponent.h"
#include "loom/import/cxx/binding/fastmath.h"
#include "loom/import/cxx/binding/function_contracts.h"
#include "loom/import/cxx/binding/index_expression.h"
#include "loom/import/cxx/binding/intrinsics.h"
#include "loom/import/cxx/binding/launch.h"
#include "loom/import/cxx/binding/loop_schedule.h"
#include "loom/import/cxx/binding/pipeline_strand.h"
#include "loom/import/cxx/binding/schedule_region.h"
#include "loom/import/cxx/binding/target_definitions.h"
#include "loom/import/cxx/binding/template_definitions.h"
#include "loom/import/cxx/check.h"
#include "loom/import/cxx/control/analysis.h"
#include "loom/import/cxx/control/regions.h"
#include "loom/import/cxx/receiver/scopes.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/import/cxx/source/locations.h"
#include "loom/import/cxx/source/objects.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/symbol/functions.h"
#include "loom/import/cxx/symbol/immutable_objects.h"
#include "loom/import/cxx/value/bindings.h"
#include "loom/import/cxx/value/bitcast.h"
#include "loom/import/cxx/value/initialization.h"
#include "loom/import/cxx/value/lvalues.h"
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
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom::cxx_import {
namespace {
class Translator final : private Initialization::Evaluation {
  struct Mutation {
    // Automatic binding owning this complete source subobject.
    cxx::Symbol* binding;
    // Retained type and component slice carried across the region edge.
    ValueProjection projection;
  };

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
        (current_function_.kind == FunctionKind::Pipeline &&
         pipeline_strand_depth_ != 0) ||
        (current_function_.kind == FunctionKind::Ordinary &&
         loom_func_def_inline_policy(current_function_.operation) ==
             LOOM_INLINE_POLICY_INLINE)) {
      return;
    }
    if (current_function_.kind == FunctionKind::Pipeline) {
      fail(owner, "kernel intrinsic in a pipeline requires a strand body");
    }
    fail(owner, "kernel intrinsic requires a kernel or inline helper");
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
         "target launch query requires a kernel configuration or inline "
         "helper");
  }

  void enter_value_receiver(const RecordPartition& record) override {
    receivers_.enter(record, std::nullopt, record.source->type());
  }

  void enter_storage_receiver(const RecordPartition& record,
                              StorageProjection destination,
                              const cxx::Type* object_type) override {
    receivers_.enter(record, destination, object_type);
  }

  bool suspend_current_receiver() override { return receivers_.suspend(); }

  void resume_current_receiver() override { receivers_.resume(); }

  void bind_value_receiver(const RecordPartition& record, Value value) {
    receivers_.enter(record, std::nullopt, record.source->type());
    for (const auto& member : record.members) {
      publish_receiver_member(
          member.source,
          value.project(*member.partition, member.component_offset));
    }
  }

  void bind_storage_receiver(const RecordPartition& record,
                             StorageProjection destination,
                             const cxx::Type* object_type) {
    receivers_.enter(record, destination, object_type);
    for (const auto& member : record.members) {
      complete_storage_receiver_member(member.source);
    }
  }

  Value current_receiver_value(cxx::AST* owner) {
    IREE_ASSERT(receivers_.current());
    const auto& active = *receivers_.current();
    IREE_ASSERT(!active.storage);
    std::vector<loom_value_id_t> components;
    components.reserve(active.record->component_count);
    for (const auto& member : active.record->members) {
      receiver_member(member.source, owner).append_to(components);
    }
    return value_arena_.capture(*active.record, components);
  }

  void replace_current_receiver(Value value) {
    IREE_ASSERT(receivers_.current());
    const auto& active = *receivers_.current();
    IREE_ASSERT(!active.storage && &value.partition() == active.record);
    for (const auto& member : active.record->members) {
      publish_receiver_member(
          member.source,
          value.project(*member.partition, member.component_offset));
    }
  }

  std::optional<SubobjectProjection> current_receiver_selection(
      cxx::ExpressionAST* expression) {
    expression = unwrap_object_identity(unit_, expression);
    cxx::Symbol* field = nullptr;
    cxx::List<cxx::Symbol*>* path = nullptr;
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(expression)) {
      field = id->symbol;
      path = id->subobjectPath;
    } else if (auto* member =
                   cxx::ast_cast<cxx::MemberExpressionAST>(expression);
               member && is_current_object(unit_, member->baseExpression)) {
      field = member->symbol;
      path = member->subobjectPath;
    }
    auto* root = path ? path->value : field;
    if (!is_direct_subobject(field) || !receivers_.find(root)) {
      return std::nullopt;
    }
    return types_.selection(field, path, expression);
  }

  Value selected_receiver_value(const SubobjectProjection& selected,
                                const cxx::Type* type, cxx::AST* owner) {
    auto* root = selected.root->source;
    if (auto* active = receivers_.find(root); active && active->storage) {
      if (!receivers_.is_complete(root)) {
        fail(owner, "member is read before its initialization is complete");
      }
      auto projection =
          storage_.subobject(*active->storage, selected.byte_offset, owner);
      auto traits = unit_.typeTraits();
      if (traits.is_volatile(active->object_type)) {
        type = traits.add_volatile(type);
      }
      return traits.is_array(type) ? Value(projection.pointer)
                                   : objects_.load(projection, type, owner);
    }
    if (!receivers_.is_complete(root)) {
      fail(owner, "member is read before its initialization is complete");
    }
    auto value = values_.project(
        root, *selected.partition,
        selected.component_offset - selected.root->component_offset,
        value_arena_);
    if (!value) {
      fail(owner, "source subobject is read before initialization");
    }
    return *value;
  }

  std::optional<StorageProjection> captured_object_storage(
      cxx::Symbol* field, cxx::List<cxx::Symbol*>* path, cxx::AST* owner) {
    if (!receivers_.current()) {
      return std::nullopt;
    }
    auto* closure = receivers_.current()->record->source;
    auto captured =
        captured_object_member(unit_, closure, path ? path->value : field);
    if (!captured) {
      return std::nullopt;
    }
    auto pointer = receiver_member(captured->pointer_field, owner);
    return storage_.subobject(
        storage_.project(pointer.pointer(), captured->object_type, owner),
        source_subobject_offset(field) +
            (path ? types_.subobject_offset(path) : 0),
        owner);
  }

  std::optional<StorageProjection> receiver_storage(cxx::Symbol* field,
                                                    cxx::AST* owner) {
    auto* active = receivers_.find(field);
    if (!active || !active->storage) {
      return std::nullopt;
    }
    return storage_.member(*active->storage, field, owner);
  }

  void publish_receiver_member(cxx::Symbol* field, Value value) override {
    receivers_.publish(field, value);
  }

  void complete_storage_receiver_member(cxx::Symbol* field) override {
    receivers_.mark_complete(field);
  }

  Value receiver_member(cxx::Symbol* field, cxx::AST* owner) override {
    auto* active = receivers_.find(field);
    if (!active) {
      fail(owner, "member expression requires a current source object");
    }
    if (!receivers_.is_complete(field)) {
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
    if (found == values_.end()) {
      fail(owner, "source subobject is read before initialization");
    }
    return found->second;
  }

  void leave_receiver() override { receivers_.leave(); }

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
        value.is_opaque_dialect() || value.is_encoding() || value.is_pool() ||
        value.is_tile() || value.is_channel() || value.is_vector() ||
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
    } else if (value.is_array()) {
      const auto& partition =
          static_cast<const ArrayPartition&>(value.partition());
      for (size_t index = 0; index < partition.source->size(); ++index) {
        name(value.project(*partition.element,
                           index * partition.element->component_count),
             hint + "_" + std::to_string(index));
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
    } else if (value.is_tile()) {
      const auto& partition =
          static_cast<const TilePartition&>(value.partition());
      auto components = value.components();
      for (size_t index = 0; index < components.size(); ++index) {
        auto suffix = partition.component_names[index];
        name(components[index], suffix.empty() ? hint : hint + "_" + suffix);
      }
    } else if (value.is_channel()) {
      const auto& partition =
          static_cast<const ChannelPartition&>(value.partition());
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

  Value staged_region_value(loom_region_t* region,
                            const StagedSignatureProjection& signature,
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
        signature.specialization_count + parameter.binding_offset;
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
    // Active destruction count at this lexical block's entry.
    size_t destruction_mark;
  };

  // Evaluated source reference into one nontrivial temporary object. Geometry
  // remains explicit when a returned reference denotes a base or subobject.
  struct TemporaryReference {
    // Stable construction-instance index in the current cleanup queue.
    size_t index;
    // First SSA component within the complete temporary's current value.
    size_t component_offset = 0;
    // Source-layout displacement if the complete object becomes addressable.
    uint64_t byte_offset = 0;
  };

  // One completed source object awaiting destruction. Automatic bindings and
  // current receiver fields resolve their latest value at the cleanup edge;
  // temporary objects retain the value or storage created by their expression.
  struct DestructionAction {
    // Frontend-selected complete-object destruction.
    Destruction destruction;
    // Automatic local or parameter binding, mutually exclusive with |field|,
    // |value| and |storage|.
    cxx::Symbol* binding = nullptr;
    // Current receiver subobject, mutually exclusive with |binding|, |value|
    // and |storage|.
    cxx::Symbol* field = nullptr;
    // Complete source object type, including any bounded array extents.
    const cxx::Type* object_type = nullptr;
    // Direct SSA object retained by an unmaterialized temporary.
    std::optional<Value> value;
    // Direct object storage retained by an addressed temporary.
    std::optional<StorageProjection> storage;
    // Conditional temporary activation flag, stored outside its source arm.
    std::optional<StorageAccess> activity;
    // Canonical construction site; each evaluation gets its own cleanup entry.
    cxx::ExpressionAST* origin = nullptr;
    // Source declaration or expression owning the implicit call location.
    cxx::AST* source = nullptr;
  };

  void call_destructor(const Destruction& destruction,
                       std::span<const loom_value_id_t> arguments,
                       ImplicitObjectABI receiver_abi, cxx::AST* source) {
    auto callee =
        functions_.declare(destruction.function, source, receiver_abi);
    loom_op_t* op;
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee, arguments.data(),
                               arguments.size(), nullptr, 0, nullptr, 0,
                               locations_.get(source), &op));
  }

  template <typename Read>
  void destroy_components(const Destruction& destruction, cxx::AST* source,
                          Read&& read) {
    const auto* record = types_.record(destruction.element_type, source);
    const auto* projection =
        functions_.receiver_projection(destruction.function);
    std::vector<loom_value_id_t> arguments;
    arguments.reserve(projection ? projection->input_component_count
                                 : record->component_count);
    for (size_t index = destruction.element_count; index > 0; --index) {
      const size_t offset = (index - 1) * record->component_count;
      arguments.clear();
      if (projection) {
        for (const auto& group : projection->inputs) {
          read(*group.partition, offset + group.component_offset)
              .append_to(arguments);
        }
      } else {
        read(*record, offset).append_to(arguments);
      }
      call_destructor(destruction, arguments, ImplicitObjectABI::Value, source);
    }
  }

  void destroy_value(Value value, const Destruction& destruction,
                     cxx::AST* source) {
    destroy_components(destruction, source,
                       [&](const Partition& partition, size_t offset) {
                         return value.project(partition, offset);
                       });
  }

  void destroy_binding(cxx::Symbol* binding, const Destruction& destruction,
                       cxx::AST* source) {
    destroy_components(
        destruction, source, [&](const Partition& partition, size_t offset) {
          auto value =
              values_.project(binding, partition, offset, value_arena_);
          if (!value) {
            fail(source, "destructor reads an uninitialized source subobject");
          }
          return *value;
        });
  }

  void destroy_storage(StorageProjection object, const cxx::Type* type,
                       const Destruction& destruction, cxx::AST* source) {
    type = types_.unqualified(type);
    if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(type)) {
      for (size_t index = array->size(); index > 0; --index) {
        destroy_storage(storage_.element(object, array, index - 1, source),
                        array->elementType(), destruction, source);
      }
      return;
    }
    call_destructor(destruction, Value(object.pointer).components(),
                    ImplicitObjectABI::Storage, source);
  }

  void emit_destruction_object(const DestructionAction& active) {
    if (active.binding) {
      auto local = locals_.find(active.binding);
      if (local != locals_.end()) {
        destroy_storage(local_object(active.binding, active.source),
                        source_object_type(active.binding), active.destruction,
                        active.source);
      } else {
        destroy_binding(active.binding, active.destruction, active.source);
      }
      return;
    }
    if (active.field) {
      if (auto storage = receiver_storage(active.field, active.source)) {
        destroy_storage(*storage, source_object_type(active.field),
                        active.destruction, active.source);
      } else {
        destroy_binding(active.field, active.destruction, active.source);
      }
      return;
    }
    if (active.storage) {
      destroy_storage(*active.storage, active.object_type, active.destruction,
                      active.source);
      return;
    }
    IREE_ASSERT(active.value);
    destroy_value(*active.value, active.destruction, active.source);
  }

  void emit_destruction(const DestructionAction& active) {
    if (!active.activity) {
      emit_destruction_object(active);
      return;
    }
    auto* bool_type = unit_.control()->getBoolType();
    auto condition = storage_.load(*active.activity, bool_type, active.source);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, 0, condition, nullptr, 0, nullptr, 0,
                            locations_.get(active.source), &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    emit_destruction_object(active);
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, nullptr, 0,
                               locations_.get(active.source), &yield));
    loom_builder_restore(&builder_, saved);
  }

  void emit_destructions(size_t mark) {
    IREE_ASSERT_LE(mark, active_destructions_.size());
    for (size_t index = active_destructions_.size(); index > mark; --index) {
      emit_destruction(active_destructions_[index - 1]);
    }
  }

  void emit_automatic_destruction(cxx::Symbol* binding, cxx::AST* source) {
    if (const auto* destruction = control_->automatic_destruction(binding)) {
      emit_destruction({
          .destruction = *destruction,
          .binding = binding,
          .object_type = source_object_type(binding),
          .source = source,
      });
    }
  }

  void emit_automatic_destruction_if(loom_value_id_t condition,
                                     cxx::Symbol* binding, cxx::AST* source) {
    if (!control_->automatic_destruction(binding)) {
      return;
    }
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, 0, condition, nullptr, 0, nullptr, 0,
                            locations_.get(source), &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    emit_automatic_destruction(binding, source);
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, nullptr, 0, locations_.get(source),
                               &yield));
    loom_builder_restore(&builder_, saved);
  }

  void close_destruction_scope(size_t mark) {
    emit_destructions(mark);
    active_destructions_.resize(mark);
  }

  void close_destruction_scope_if(loom_value_id_t condition, size_t mark,
                                  cxx::AST* source) {
    if (mark == active_destructions_.size()) {
      return;
    }
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, 0, condition, nullptr, 0, nullptr, 0,
                            locations_.get(source), &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    emit_destructions(mark);
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, nullptr, 0, locations_.get(source),
                               &yield));
    loom_builder_restore(&builder_, saved);
    active_destructions_.resize(mark);
  }

  void activate_automatic_destruction(cxx::Symbol* binding, cxx::AST* source) {
    if (const auto* destruction = control_->automatic_destruction(binding)) {
      active_destructions_.push_back({
          .destruction = *destruction,
          .binding = binding,
          .object_type = source_object_type(binding),
          .source = source,
      });
    }
  }

  void activate_subobject_destructions(cxx::AST* source) {
    for (const auto& subobject : control_->subobject_destructions()) {
      active_destructions_.push_back({
          .destruction = subobject.destruction,
          .field = subobject.field,
          .object_type = source_object_type(subobject.field),
          .source = source,
      });
    }
  }

  void materialize_conditional_temporary(DestructionAction& action) {
    IREE_ASSERT(!conditional_cleanup_anchors_.empty());
    IREE_ASSERT(action.value);
    IREE_ASSERT(!action.storage);
    IREE_ASSERT(!action.activity);
    auto saved = loom_builder_save(&builder_);
    loom_builder_restore(&builder_, conditional_cleanup_anchors_.front());
    auto object_allocation = storage_.allocate(
        action.object_type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0,
        action.source);
    auto object = storage_.project(object_allocation.pointer,
                                   action.object_type, action.source);
    auto* bool_type = unit_.control()->getBoolType();
    auto activity_allocation = storage_.allocate(
        bool_type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, action.source);
    StorageAccess activity = {activity_allocation.view, std::nullopt};
    auto inactive = scalars_.integer(false, LOOM_SCALAR_TYPE_I1,
                                     locations_.get(action.source));
    storage_.store(activity, inactive, bool_type, action.source);
    loom_builder_restore(&builder_, saved);

    objects_.store(object, *action.value, action.object_type, action.source);
    auto active = scalars_.integer(true, LOOM_SCALAR_TYPE_I1,
                                   locations_.get(action.source));
    storage_.store(activity, active, bool_type, action.source);
    action.value.reset();
    action.storage = object;
    action.activity = activity;
  }

  void materialize_conditional_temporaries(size_t mark) {
    IREE_ASSERT(!conditional_cleanup_anchors_.empty());
    IREE_ASSERT_LE(mark, temporary_destructions_.size());
    for (size_t index = mark; index < temporary_destructions_.size(); ++index) {
      auto& action = temporary_destructions_[index];
      if (action && action->value) {
        materialize_conditional_temporary(*action);
      }
    }
  }

  std::optional<TemporaryReference> temporary_result(
      cxx::ExpressionAST* expression) const {
    auto* origin = control_->temporary_origins().materialization(expression);
    auto found = temporary_results_.find(origin);
    return found == temporary_results_.end()
               ? std::nullopt
               : std::optional(TemporaryReference{found->second});
  }

  void forget_temporary(size_t index) {
    const auto& action = *temporary_destructions_[index];
    auto found = temporary_results_.find(action.origin);
    if (found != temporary_results_.end() && found->second == index) {
      temporary_results_.erase(found);
    }
    temporary_destructions_[index].reset();
  }

  Value retain_temporary(cxx::ExpressionAST* expression, Value value,
                         size_t evaluation_mark) {
    auto* origin = control_->temporary_origins().result(expression);
    if (!origin) {
      return value;
    }
    const auto& destruction = control_->temporary_origins().destruction(origin);
    // Transparent syntax can complete an object already created by a child.
    // A later evaluation of the same source construction is a distinct object.
    auto found = temporary_results_.find(origin);
    if (found != temporary_results_.end() && found->second >= evaluation_mark) {
      temporary_destructions_[found->second]->value = value;
      return value;
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
    return value;
  }

  FullExpression begin_full_expression() override {
    FullExpression expression = {
        .mark = temporary_destructions_.size(),
        .parent_mark = full_expression_mark_,
    };
    full_expression_mark_ = expression.mark;
    return expression;
  }

  void finish_full_expression(FullExpression expression) override {
    IREE_ASSERT(full_expression_mark_);
    IREE_ASSERT_EQ(*full_expression_mark_, expression.mark);
    IREE_ASSERT_LE(expression.mark, temporary_destructions_.size());
    for (size_t index = temporary_destructions_.size(); index > expression.mark;
         --index) {
      if (temporary_destructions_[index - 1]) {
        emit_destruction(*temporary_destructions_[index - 1]);
        forget_temporary(index - 1);
      }
    }
    temporary_destructions_.resize(expression.mark);
    full_expression_mark_ = expression.parent_mark;
  }

  // Result ownership transfers by construction identity, independently of
  // the immutable payload and the number of transparent syntax wrappers.
  void adopt_temporary_result(cxx::ExpressionAST* expression) {
    auto* origin = control_->temporary_origins().result(expression);
    auto found = temporary_results_.find(origin);
    if (found != temporary_results_.end()) {
      forget_temporary(found->second);
    }
  }

  void update_temporary(std::optional<TemporaryReference> temporary,
                        Value updated, const cxx::Type* type, cxx::AST* owner) {
    if (temporary) {
      auto& action = *temporary_destructions_[temporary->index];
      if (action.storage) {
        objects_.store(
            storage_.subobject(*action.storage, temporary->byte_offset, owner),
            updated, type, owner);
      } else {
        action.value = value_arena_.replace(
            *action.value, temporary->component_offset, updated);
      }
    }
  }

  StorageProjection materialize_temporary(
      cxx::ExpressionAST* expression, Value value,
      std::optional<TemporaryReference> temporary = std::nullopt) {
    if (!temporary) {
      temporary = temporary_result(expression);
    }
    if (temporary) {
      auto& action = *temporary_destructions_[temporary->index];
      if (!action.storage) {
        if (conditional_cleanup_anchors_.empty()) {
          auto allocation = storage_.allocate(
              action.object_type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0,
              expression);
          action.storage = storage_.project(allocation.pointer,
                                            action.object_type, expression);
          objects_.store(*action.storage, *action.value, action.object_type,
                         expression);
          action.value.reset();
        } else {
          materialize_conditional_temporary(action);
        }
      }
      auto storage = *action.storage;
      if (temporary->byte_offset) {
        storage =
            storage_.subobject(storage, temporary->byte_offset, expression);
      }
      if (control_->temporary_origins().extended(action.origin)) {
        active_destructions_.push_back(action);
        forget_temporary(temporary->index);
      }
      return storage;
    }
    auto allocation = storage_.allocate(
        expression->type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, expression);
    auto storage =
        storage_.project(allocation.pointer, expression->type, expression);
    objects_.store(storage, value, expression->type, expression);
    return storage;
  }

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

  bool returns_value_reference_parameter(cxx::FunctionSymbol* function) const {
    auto returned = functions_.reference_return_parameter(function);
    if (!returned) {
      return false;
    }
    for (const auto& reference : functions_.reference_parameters(function)) {
      if (reference.ordinal == *returned) {
        return reference.abi == ReferenceABI::Value;
      }
    }
    return false;
  }

  bool returns_value_reference(cxx::FunctionSymbol* function,
                               const Value& result) const {
    return returns_value_reference_parameter(function) ||
           (!result.is_pointer() &&
            (functions_.returns_current_receiver_reference(function) ||
             !functions_
                  .projected_reference_return_origins(function,
                                                      ImplicitObjectABI::Value)
                  .empty()));
  }

  std::vector<const cxx::Type*> function_result_sources() {
    std::vector<const cxx::Type*> sources;
    const bool returns_reference =
        unit_.typeTraits().is_reference(current_function_.return_type);
    const bool returns_parameter_alias =
        returns_reference && current_function_.control &&
        current_function_.control->reference_return_parameter().has_value();
    const auto projected_origins =
        functions_.projected_reference_return_origins(
            current_function_.source->symbol,
            current_function_.implicit_object
                ? current_function_.implicit_object->abi
                : ImplicitObjectABI::None);
    const bool returns_projected_reference =
        returns_reference && !projected_origins.empty();
    const bool returns_receiver_alias =
        current_function_.implicit_object && current_function_.control &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value &&
        current_function_.control->returns_current_receiver_reference();
    const bool returns_exact_projection =
        returns_projected_reference && projected_origins.size() == 1;
    if (!unit_.typeTraits().is_void(current_function_.return_type) &&
        !returns_receiver_alias && !returns_parameter_alias &&
        !returns_exact_projection) {
      const cxx::Type* result_type = current_function_.return_type;
      if (returns_projected_reference) {
        result_type = unit_.control()->getUnsignedIntType();
      } else if (returns_reference) {
        result_type = unit_.typeTraits().add_pointer(
            unit_.typeTraits().remove_reference(result_type));
      }
      sources.push_back(result_type);
    }
    if (current_function_.implicit_object &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value) {
      const auto& implicit = *current_function_.implicit_object;
      if (implicit.projection) {
        for (const auto& group : implicit.projection->outputs) {
          sources.push_back(group.type);
        }
      } else {
        for (auto* field : implicit.written_fields) {
          sources.push_back(
              types_.unqualified_value(source_object_type(field)));
        }
      }
    }
    auto parameters = current_function_.source->symbol->parameters();
    for (const auto& reference : current_function_.reference_parameters) {
      if (reference.abi == ReferenceABI::Value && reference.mutable_access) {
        if (reference.projection) {
          for (const auto& group : reference.projection->outputs) {
            sources.push_back(group.type);
          }
        } else {
          sources.push_back(
              types_.unqualified_value(parameters[reference.ordinal]->type()));
        }
      }
    }
    return sources;
  }

  Returned complete_return(Returned returned) {
    if (current_function_.implicit_object &&
        current_function_.implicit_object->abi == ImplicitObjectABI::Value) {
      const auto& implicit = *current_function_.implicit_object;
      if (implicit.projection) {
        for (const auto& group : implicit.projection->outputs) {
          read(project_call_target(current_receiver_target(), group,
                                   returned.source),
               returned.source)
              .append_to(returned.values);
        }
      } else {
        for (auto* field : implicit.written_fields) {
          receiver_member(field, returned.source).append_to(returned.values);
        }
      }
    }
    auto parameters = current_function_.source->symbol->parameters();
    for (const auto& reference : current_function_.reference_parameters) {
      if (reference.abi == ReferenceABI::Value && reference.mutable_access) {
        if (reference.projection) {
          for (const auto& group : reference.projection->outputs) {
            auto value =
                values_.project(parameters[reference.ordinal], *group.partition,
                                group.component_offset, value_arena_);
            if (!value) {
              fail(returned.source,
                   "output field is not initialized on this return path");
            }
            value->append_to(returned.values);
          }
        } else {
          values_.at(parameters[reference.ordinal]).append_to(returned.values);
        }
      }
    }
    return returned;
  }

  bool supports_function_exit() const {
    return current_function_.kind == FunctionKind::Ordinary ||
           current_function_.kind == FunctionKind::Kernel;
  }

  Returned return_values(cxx::ReturnStatementAST* statement) {
    auto mark = begin_full_expression();
    auto returned = return_values_impl(statement);
    finish_full_expression(mark);
    return returned;
  }

  void append_reference_return_indices(cxx::ExpressionAST* expression,
                                       std::vector<size_t>& indices) {
    expression = unwrap_object_identity(unit_, expression);
    if (auto* conditional =
            cxx::ast_cast<cxx::ConditionalExpressionAST>(expression);
        conditional &&
        expression->valueCategory != cxx::ValueCategory::kPrValue) {
      append_reference_return_indices(conditional->iftrueExpression, indices);
      append_reference_return_indices(conditional->iffalseExpression, indices);
      return;
    }
    auto projected_indices =
        current_function_.control->projected_reference_return_indices(
            expression);
    IREE_ASSERT(!projected_indices.empty());
    indices.insert(indices.end(), projected_indices.begin(),
                   projected_indices.end());
  }

  Value encode_reference_return_index(const Lvalue& target,
                                      std::span<const size_t> indices,
                                      size_t& leaf_index, cxx::AST* owner) {
    const auto* selected = target.selection();
    if (!selected) {
      IREE_ASSERT_LT(leaf_index, indices.size());
      return Value(scalars_.integer(indices[leaf_index++], LOOM_SCALAR_TYPE_I32,
                                    locations_.get(owner)));
    }
    return dispatch_ordinal(
        selected->selector, selected->alternatives.size(),
        unit_.control()->getUnsignedIntType(), owner, [&](size_t ordinal) {
          return encode_reference_return_index(selected->alternatives[ordinal],
                                               indices, leaf_index, owner);
        });
  }

  loom_value_id_t selected_reference_return(cxx::ExpressionAST* expression,
                                            cxx::AST* owner) {
    std::vector<size_t> indices;
    append_reference_return_indices(expression, indices);
    IREE_ASSERT(!indices.empty());

    auto target = reference_lvalue(expression, owner, LvalueAccess::Read,
                                   /*force_ordinal=*/true);
    if (indices.size() == 1) {
      return scalars_.integer(indices.front(), LOOM_SCALAR_TYPE_I32,
                              locations_.get(owner));
    }
    const auto* selected = target.selection();
    IREE_ASSERT(selected);
    if (selected->alternatives.size() == indices.size() &&
        std::ranges::none_of(selected->alternatives,
                             [](const Lvalue& alternative) {
                               return alternative.selection();
                             })) {
      bool identity_mapping = true;
      for (size_t ordinal = 0; ordinal < indices.size(); ++ordinal) {
        identity_mapping &= indices[ordinal] == ordinal;
      }
      if (identity_mapping) {
        return selected->selector;
      }
    }

    size_t leaf_index = 0;
    auto encoded =
        encode_reference_return_index(target, indices, leaf_index, owner);
    IREE_ASSERT_EQ(leaf_index, indices.size());
    return encoded.ssa();
  }

  Returned return_values_impl(cxx::ReturnStatementAST* statement) {
    bool returns_void =
        unit_.typeTraits().is_void(current_function_.return_type);
    if (!statement->expression) {
      if (!returns_void) {
        fail(statement, "non-void return requires a value");
      }
      return {{}, statement};
    }
    if (unit_.typeTraits().is_void(statement->expression->type)) {
      if (!returns_void) {
        fail(statement, "non-void return requires a value");
      }
      effect(statement->expression);
      return {{}, statement};
    }
    if (returns_void) {
      fail(statement, "void return cannot carry a value");
    }
    Returned returned = {{}, statement};
    if (unit_.typeTraits().is_reference(current_function_.return_type)) {
      const bool returns_parameter_alias =
          current_function_.control &&
          current_function_.control->reference_return_parameter().has_value();
      const auto projected_origins =
          functions_.projected_reference_return_origins(
              current_function_.source->symbol,
              current_function_.implicit_object
                  ? current_function_.implicit_object->abi
                  : ImplicitObjectABI::None);
      const bool returns_projected_reference = !projected_origins.empty();
      const bool returns_receiver_alias =
          current_function_.implicit_object && current_function_.control &&
          current_function_.implicit_object->abi == ImplicitObjectABI::Value &&
          current_function_.control->returns_current_receiver_reference();
      if (returns_receiver_alias) {
        IREE_ASSERT(is_current_object(unit_, statement->expression));
      } else if (returns_projected_reference && projected_origins.size() > 1) {
        returned.values.push_back(
            selected_reference_return(statement->expression, statement));
      } else if (returns_parameter_alias || returns_projected_reference) {
        // The alias needs no runtime payload, but selecting it can still
        // evaluate a condition or call an effectful forwarding helper.
        effect(statement->expression);
      } else {
        reference_value(statement->expression, statement)
            .append_to(returned.values);
      }
    } else {
      auto value = convert(statement->expression, current_function_.return_type,
                           statement);
      adopt_temporary_result(statement->expression);
      value.append_to(returned.values);
    }
    return returned;
  }

  void function_exit(cxx::ReturnStatementAST* statement) {
    if (!supports_function_exit()) {
      fail(statement,
           "returns inside structured control require a runtime function or "
           "kernel body");
    }
    auto outer_values = values_;
    auto returned = return_values(statement);
    emit_destructions(0);
    returned = complete_return(std::move(returned));
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
    pipeline_strand_depth_ = 0;
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
    immutable_objects_.retain_locals(defined.source->symbol,
                                     control_->static_locals());
    auto saved = loom_builder_enter_region(&builder_, op, region);
    values_.clear();
    locals_.clear();
    active_destructions_.clear();
    temporary_destructions_.clear();
    temporary_results_.clear();
    full_expression_mark_.reset();
    conditional_cleanup_anchors_.clear();
    break_destruction_marks_.clear();
    continue_destruction_marks_.clear();
    immutable_storage_.clear();
    immutable_preamble_end_ = nullptr;
    references_.clear();
    receivers_.clear();
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
      if (implicit.projection) {
        receivers_.enter(*record, std::nullopt, record->source->type());
        for (const auto& member : record->members) {
          values_.declare(member.source, *member.partition);
          receivers_.mark_complete(member.source);
        }
        for (const auto& group : implicit.projection->inputs) {
          auto value = region_value(region, argument_index, *group.partition);
          auto components = value.components();
          for (size_t index = 0; index < components.size(); ++index) {
            name(components[index],
                 "this_" +
                     record->component_names[group.component_offset + index]);
          }
          write(project_call_target(current_receiver_target(), group,
                                    defined.source),
                value, defined.source);
        }
      } else if (implicit.abi == ImplicitObjectABI::Value) {
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
      if (reference && reference->projection) {
        values_.declare(parameter, partition);
        for (const auto& group : reference->projection->inputs) {
          auto value = region_value(region, argument_index, *group.partition);
          auto parameter_name = cxx::to_string(parameter->name());
          if (partition.kind == ValueKind::Record) {
            const auto& record = static_cast<const RecordPartition&>(partition);
            auto components = value.components();
            for (size_t index = 0; index < components.size(); ++index) {
              name(components[index],
                   parameter_name + "_" +
                       record.component_names[group.component_offset + index]);
            }
          } else {
            name(value, parameter_name);
          }
          values_.replace(parameter, group.component_offset, value,
                          value_arena_);
        }
        activate_automatic_destruction(parameter, defined.source);
        ++parameter_index;
        continue;
      }
      bool kernel_pointer = defined.kind == FunctionKind::Kernel &&
                            partition.kind == ValueKind::Pointer;
      // Only pointers change representation at kernel entry: the ABI supplies
      // one buffer, and source pointer transport adds its zero byte origin.
      auto value =
          defined.staged_signature
              ? staged_region_value(
                    region, *defined.staged_signature,
                    defined.staged_signature->parameters[parameter_index],
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
        activate_automatic_destruction(parameter, defined.source);
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
        values_.set(parameter, value);
      }
      activate_automatic_destruction(parameter, defined.source);
      ++parameter_index;
    }
    if (defined.source->symbol->isConstructor()) {
      IREE_ASSERT(defined.implicit_object &&
                  defined.implicit_object->abi == ImplicitObjectABI::Storage &&
                  receivers_.current() && receivers_.current()->storage);
      initialization_.constructor_receiver(
          *receivers_.current()->storage, *receivers_.current()->record,
          defined.implicit_object->object_type, defined.source->symbol,
          defined.source);
    }
    if (defined.source->symbol->isDestructor()) {
      activate_subobject_destructions(defined.source);
    }
    auto returned = return_sequence({body ? body->statementList : nullptr,
                                     nullptr, lexical_fastmath_flags_,
                                     control_->callable_exit_count(body), 0});
    active_destructions_.clear();
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
    } else if (defined.kind == FunctionKind::Pipeline) {
      check(loom_pipeline_finish_build(
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
        close_destruction_scope(sequence.destruction_mark);
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
    auto allocation = storage_.allocate(source_object_type(symbol),
                                        LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
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
      if (!values_.initialized(symbol)) {
        fail(owner, "source value is read before initialization");
      }
      return values_.at(symbol);
    }
    if (unit_.typeTraits().is_array(source_object_type(symbol))) {
      return found->second.pointer;
    }
    auto value = unit_.typeTraits().is_class(source_object_type(symbol))
                     ? objects_.load(local_object(symbol, owner),
                                     source_object_type(symbol), owner)
                     : Value(storage_.load({found->second.view, std::nullopt},
                                           source_object_type(symbol), owner));
    return name(value, cxx::to_string(symbol->name()));
  }

  StorageProjection local_object(cxx::Symbol* symbol, cxx::AST* owner) {
    return storage_.project(locals_.at(symbol).pointer,
                            source_object_type(symbol), owner);
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
    if (unit_.typeTraits().is_reference(variable->type())) {
      if (!initializer) {
        fail(owner, "reference locals require an initializer");
      }
      auto* referent_expression =
          cxx::Initializer(initializer).referenceExpression(unit_.typeTraits());
      if (!referent_expression) {
        fail(initializer,
             "reference locals require one initializer expression");
      }
      auto* referent = unit_.typeTraits().remove_reference(variable->type());
      auto target =
          control_->reference_requires_storage(variable)
              ? Lvalue{ObjectLvalue{
                    referent,
                    storage_.project(
                        reference_value(referent_expression, owner).pointer(),
                        referent, owner)}}
              : reference_lvalue(referent_expression, owner,
                                 unit_.typeTraits().is_const(referent)
                                     ? LvalueAccess::Read
                                     : LvalueAccess::Write);
      references_.emplace(variable, std::move(target));
      return;
    }
    const bool record_constructor_call =
        types_.record_constructor_requires_call(variable->type(),
                                                variable->constructor(), owner);
    const bool requires_initialization =
        initializer || record_constructor_call ||
        control_->default_constructor_initializes(variable->constructor());
    if (control_->storage_backed(variable)) {
      auto contents =
          unit_.typeTraits().is_volatile(variable->type())
              ? std::nullopt
              : initialization_.literal(variable->type(), initializer, owner);
      allocate_local(variable, variable->explicitAlignment(), owner,
                     contents ? &*contents : nullptr);
      if (contents) {
        activate_automatic_destruction(variable, owner);
        return;
      }
      if (requires_initialization) {
        if (types_.address_domain(types_.unqualified(variable->type()),
                                  owner) == LOOM_SCALAR_TYPE_NONE) {
          types_.admit_construction(variable->constructor(), variable->type(),
                                    owner);
        }
        initialization_.storage(locals_.at(variable), variable->type(),
                                initializer, variable->constructor(), owner);
      }
      activate_automatic_destruction(variable, owner);
      return;
    }
    if (!requires_initialization) {
      values_.declare(variable, types_.partition(variable->type(), owner));
      activate_automatic_destruction(variable, owner);
      return;
    }
    types_.partition(variable->type(), owner);
    if (types_.address_domain(types_.unqualified(variable->type()), owner) ==
        LOOM_SCALAR_TYPE_NONE) {
      types_.admit_construction(variable->constructor(), variable->type(),
                                owner);
    }
    auto value = apply_constexpr_contract(
        initializer ? expression(initializer)
                    : initialization_.value(variable->type(), nullptr,
                                            variable->constructor(), owner),
        constexpr_contract);
    adopt_temporary_result(initializer);
    values_.set(variable, name(value, cxx::to_string(variable->name())));
    activate_automatic_destruction(variable, owner);
  }

  void initialize_local_declarator(
      cxx::InitDeclaratorAST* declaration,
      cxx::List<cxx::AttributeSpecifierAST*>* attributes, cxx::AST* owner) {
    reject_misplaced_binding_declarator(unit_, diagnostics_,
                                        declaration->declarator);
    auto* variable = cxx::symbol_cast<cxx::VariableSymbol>(declaration->symbol);
    auto constexpr_contract = admit_constexpr_contract(
        unit_, diagnostics_, types_, attributes, variable,
        variable && control_->addressed(variable), declaration);
    if (cxx::symbol_cast<cxx::TypeAliasSymbol>(declaration->symbol)) {
      return;
    }
    if (variable && variable->isStatic() && !variable->isThreadLocal()) {
      if (annotated(variable, "workgroup")) {
        fail(owner, "static immutable storage cannot have workgroup scope");
      }
      return;
    }
    if (!variable || variable->isExtern() || variable->isThreadLocal()) {
      fail(owner, "local storage duration must be automatic or __shared__");
    }
    if (annotated(declaration->symbol, "workgroup")) {
      if (current_function_.kind != FunctionKind::Kernel &&
          current_function_.kind != FunctionKind::TemplateDefinition) {
        fail(declaration,
             "workgroup storage requires a kernel or template definition "
             "body");
      }
      if (declaration->initializer) {
        fail(declaration, "workgroup storage cannot have an initializer");
      }
      if (control_->automatic_destruction(variable)) {
        fail(declaration, "workgroup storage requires trivial destruction");
      }
      auto* source_type = types_.unqualified(declaration->symbol->type());
      auto* array = cxx::type_cast<cxx::BoundedArrayType>(source_type);
      auto allocation =
          storage_.allocate(source_type, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
                            variable->explicitAlignment(), declaration);
      auto spelling = cxx::to_string(declaration->symbol->name());
      if (array) {
        values_.set(declaration->symbol,
                    name(Value(allocation.pointer), spelling));
      } else {
        name(Value(allocation.pointer), spelling + "_storage");
        locals_[declaration->symbol] = allocation;
      }
      name(allocation.view, spelling + "_view");
      return;
    }
    auto mark = begin_full_expression();
    initialize_variable(variable, declaration->initializer, constexpr_contract,
                        declaration);
    finish_full_expression(mark);
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
    adopt_temporary_result(declaration->initializer);
    if (auto found = locals_.find(decision); found != locals_.end()) {
      storage_.store({found->second.view, std::nullopt}, value.ssa(),
                     decision->type(), owner);
    } else {
      values_.set(decision, name(value, cxx::to_string(decision->name())));
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
      auto mark = begin_full_expression();
      initialize_condition(branch->condition);
      finish_full_expression(mark);
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
    auto mark = begin_full_expression();
    initialize_condition(branch->condition);
    if (condition_declaration(branch->condition)) {
      finish_full_expression(mark);
      mark = begin_full_expression();
    }
    auto condition = expression(branch->condition).ssa();
    finish_full_expression(mark);
    return condition;
  }

  Returned returning_statement(cxx::StatementAST* ast,
                               const ReturnSequence& continuation) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    const size_t selection_mark = active_destructions_.size();
    auto* selected = selected_statement(ast);
    if (selected != ast) {
      const ReturnSequence selected_continuation = {
          nullptr,
          &continuation,
          lexical_fastmath_flags_,
          continuation.callable_exit_count,
          selection_mark,
      };
      return selected ? returning_statement(selected, selected_continuation)
                      : return_sequence(selected_continuation);
    }
    ast = selected;
    if (auto* ret = cxx::ast_cast<cxx::ReturnStatementAST>(ast)) {
      auto returned = return_values(ret);
      emit_destructions(0);
      return complete_return(std::move(returned));
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return return_sequence({
          compound->statementList,
          &continuation,
          lexical_fastmath_flags_,
          control_->callable_exit_count(compound) +
              continuation.callable_exit_count,
          active_destructions_.size(),
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
      const size_t destruction_mark = active_destructions_.size();
      auto condition = branch_condition(branch);
      auto outer_values = values_;
      auto outer_destructions = active_destructions_;
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
      active_destructions_ = outer_destructions;
      saved =
          loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
      returned = returning_statement(branch->elseStatement, continuation);
      check(loom_scf_yield_build(&builder_, returned.values.data(),
                                 returned.values.size(),
                                 locations_.get(returned.source), &yield));
      loom_builder_restore(&builder_, saved);
      values_ = outer_values;
      active_destructions_ = std::move(outer_destructions);
      returned.values.assign(
          loom_op_results(op),
          loom_op_results(op) + result_signature.types.size());
      returned.source = ast;
      active_destructions_.resize(destruction_mark);
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
    ast = unwrap_object_identity(unit_, ast);
    cxx::FieldSymbol* field = nullptr;
    auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast);
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
    } else if (member) {
      field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
    }
    if (!field || !unit_.typeTraits().is_lvalue_reference(field->type())) {
      return std::nullopt;
    }
    auto* referent = unit_.typeTraits().remove_reference(field->type());
    if (auto selected = current_receiver_selection(ast)) {
      auto value = selected_receiver_value(*selected, field->type(), ast);
      return storage_.project(value.pointer(), referent, ast);
    }
    if (!member) {
      return std::nullopt;
    }
    if (control_->storage_backed(member)) {
      fail(ast,
           "reference-bearing records cannot be loaded from physical object "
           "storage");
    }
    auto base = expression(member->baseExpression);
    auto selected = types_.selection(field, member->subobjectPath, ast);
    auto value = base.project(*selected.partition, selected.component_offset);
    return storage_.project(value.pointer(), referent, ast);
  }

  StorageProjection object_address(cxx::ExpressionAST* ast) {
    ast = unwrap_object_identity(unit_, ast);
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast);
        cast && is_class_adjustment(cast)) {
      auto object = object_address(cast->expression);
      auto byte_offset = types_.subobject_offset(cast->subobjectPath);
      return cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion
                 ? storage_.enclosing(object, byte_offset, cast)
                 : storage_.subobject(object, byte_offset, cast);
    }
    if (auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(ast)) {
      return string_literal(literal);
    }
    if (auto reference = reference_field_address(ast)) {
      return *reference;
    }
    if (auto selected = current_receiver_selection(ast)) {
      auto* active = receivers_.find(selected->root->source);
      if (active->storage) {
        return storage_.subobject(*active->storage, selected->byte_offset, ast);
      }
      fail(ast,
           "addressing a member during initialization requires addressable "
           "record storage");
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
      if (field) {
        if (auto captured =
                captured_object_storage(field, id->subobjectPath, ast)) {
          return *captured;
        }
      }
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
          (variable->isStatic() || cxx::symbol_cast<cxx::NamespaceSymbol>(
                                       variable->canonical()->parent())) &&
          !configs_.contains(variable)) {
        return immutable_object(variable, id);
      }
      if (unit_.typeTraits().is_array(id->type)) {
        return storage_.project(expression(id).pointer(), id->type, id);
      }
    }
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
      if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
        fail(member, "extended-vector components are not addressable");
      }
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member->symbol);
      if (!field || field->isStatic()) {
        fail(ast, "record storage access requires a non-static data member");
      }
      auto base = member->accessOp == cxx::TokenKind::T_MINUS_GREATER
                      ? pointer_expression(member->baseExpression)
                      : object_address(member->baseExpression);
      auto byte_offset = source_subobject_offset(field);
      if (member->subobjectPath) {
        byte_offset += types_.subobject_offset(member->subobjectPath);
      }
      return storage_.subobject(base, byte_offset, ast);
    }
    // Value-producing operators and conversions materialize their result just
    // like ordinary calls. Only glvalues below follow an existing identity.
    if (ast->valueCategory == cxx::ValueCategory::kPrValue) {
      auto value = expression(ast);
      if (unit_.typeTraits().is_class(ast->type) ||
          unit_.typeTraits().is_array(ast->type)) {
        return materialize_temporary(ast, value);
      }
      auto allocation = storage_.allocate(
          ast->type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, ast);
      auto object = storage_.project(allocation.pointer, ast->type, ast);
      storage_.store(storage_.dereference(object, ast->type, ast), value.ssa(),
                     ast->type, ast);
      return object;
    }
    if (auto call = direct_operator_call(ast)) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(call->function->type());
      if (!signature ||
          !unit_.typeTraits().is_reference(signature->returnType())) {
        fail(ast, "addressed operator must return a reference");
      }
      auto called = ordinary_operator_call(ast, *call);
      if (!called.value || !called.value->is_pointer()) {
        fail(ast, "reference-returning operator lost its object identity");
      }
      return storage_.project(
          called.value->pointer(),
          unit_.typeTraits().remove_reference(signature->returnType()), ast);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      if (subscript->symbol) {
        auto* signature =
            cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
        if (!signature ||
            !unit_.typeTraits().is_reference(signature->returnType())) {
          fail(ast, "addressed overloaded indexing must return a reference");
        }
        auto called = overloaded_subscript_call(subscript);
        if (!called.value || !called.value->is_pointer()) {
          fail(ast, "reference-returning subscript lost its object identity");
        }
        return storage_.project(
            called.value->pointer(),
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
          unit_.typeTraits().is_reference(signature->returnType())) {
        auto called = ordinary_call(call, function, locations_.get(call));
        if (!called.value || !called.value->is_pointer()) {
          fail(ast, "reference-returning call lost its object identity");
        }
        return storage_.project(
            called.value->pointer(),
            unit_.typeTraits().remove_reference(signature->returnType()), ast);
      }
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

  // These arms only read evaluated lvalues or compute their replacements.
  // Source expression effects and temporary construction have already finished.
  Value selected_value(loom_value_id_t condition, const cxx::Type* result_type,
                       cxx::AST* owner,
                       const std::function<Value(size_t)>& operation) {
    std::array<const cxx::Type*, 1> sources = {result_type};
    auto outputs = bind_signature(types_, sources, owner, &builder_);
    auto source = locations_.get(owner);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
    for (size_t arm = 0; arm < 2; ++arm) {
      auto* region =
          arm == 0 ? loom_scf_if_then_region(op) : loom_scf_if_else_region(op);
      auto saved = loom_builder_enter_region(&builder_, op, region);
      auto value = operation(arm);
      auto components = value.components();
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, components.data(),
                                 components.size(), source, &yield));
      loom_builder_restore(&builder_, saved);
    }
    return value_arena_.capture(types_.partition(result_type, owner),
                                {loom_op_results(op), outputs.types.size()});
  }

  loom_value_id_t ordinal_condition(loom_value_id_t selector,
                                    cxx::TokenKind predicate, size_t ordinal,
                                    cxx::AST* owner) {
    auto* ordinal_type = unit_.control()->getUnsignedIntType();
    auto expected =
        scalars_.integer(ordinal, LOOM_SCALAR_TYPE_I32, locations_.get(owner));
    return scalars_.binary(predicate, selector, expected, ordinal_type,
                           unit_.control()->getBoolType(), owner);
  }

  loom_value_id_t selection_condition(const SelectedLvalue& selected,
                                      size_t ordinal, cxx::AST* owner) {
    return ordinal_condition(selected.selector, cxx::TokenKind::T_EQUAL_EQUAL,
                             ordinal, owner);
  }

  Value dispatch_ordinal_range(loom_value_id_t selector,
                               const cxx::Type* result_type, cxx::AST* owner,
                               const std::function<Value(size_t)>& operation,
                               size_t begin, size_t end) {
    IREE_ASSERT_LT(begin, end);
    if (begin + 1 == end) {
      return operation(begin);
    }
    const bool compact_tail = end - begin <= 3;
    const size_t split = compact_tail ? begin : begin + (end - begin) / 2;
    auto condition = ordinal_condition(
        selector,
        compact_tail ? cxx::TokenKind::T_EQUAL_EQUAL : cxx::TokenKind::T_LESS,
        split, owner);
    return selected_value(condition, result_type, owner, [&](size_t arm) {
      if (compact_tail) {
        return arm == 0 ? operation(begin)
                        : dispatch_ordinal_range(selector, result_type, owner,
                                                 operation, begin + 1, end);
      }
      return arm == 0 ? dispatch_ordinal_range(selector, result_type, owner,
                                               operation, begin, split)
                      : dispatch_ordinal_range(selector, result_type, owner,
                                               operation, split, end);
    });
  }

  Value dispatch_ordinal(loom_value_id_t selector, size_t count,
                         const cxx::Type* result_type, cxx::AST* owner,
                         const std::function<Value(size_t)>& operation) {
    // Region construction visits every ordinal exactly once in ascending
    // order. Reference-result encoding consumes its retained leaf map in that
    // same source order without building a parallel flattened target list.
    return dispatch_ordinal_range(selector, result_type, owner, operation, 0,
                                  count);
  }

  Value dispatch_selection(const SelectedLvalue& selected, cxx::AST* owner,
                           const std::function<Value(size_t)>& operation) {
    return dispatch_ordinal(selected.selector, selected.alternatives.size(),
                            types_.unqualified_value(selected.type), owner,
                            operation);
  }

  // Result of one ordinary source call. Exact reference aliases retain the
  // already evaluated destination in addition to its current value so a
  // subsequent member call can update the same automatic object.
  struct EvaluatedCallResult {
    // Ordinary result value, absent for a void source function.
    std::optional<Value> value;
    // Exact source destination named by a value-reference result.
    std::optional<Lvalue> reference_target;
    // Unnamed source object aliased by this reference result, if nontrivial.
    std::optional<TemporaryReference> temporary;
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
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(
            unwrap_object_identity(unit_, expression))) {
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
      if (field) {
        if (auto captured =
                captured_object_storage(field, id->subobjectPath, owner)) {
          return {expression->type, *captured};
        }
      }
    }
    auto destination = retained_destination ? retained_destination
                                            : control_->destination(expression);
    if (destination) {
      if (auto* field = destination->binding; receivers_.find(field)) {
        auto* active = receivers_.find(field);
        if (!receivers_.is_complete(field)) {
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
        if (values_.contains(destination->binding)) {
          return {expression->type, *destination};
        }
        if (access == LvalueAccess::Write) {
          fail(owner, "mutation requires an owned automatic source binding");
        }
      }
    }
    if (preserve_identity || unit_.typeTraits().is_class(expression->type) ||
        unit_.typeTraits().is_array(expression->type)) {
      return {expression->type, object_address(expression)};
    }
    return {expression->type, address(expression)};
  }

  Value read(const ObjectLvalue& target, cxx::AST* owner) {
    if (std::holds_alternative<CurrentReceiver>(target.location)) {
      return current_receiver_value(owner);
    }
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      const auto& partition = destination->member
                                  ? *destination->member
                                  : values_.partition(destination->binding);
      auto value = values_.project(destination->binding, partition,
                                   destination->component_offset, value_arena_);
      if (!value) {
        fail(owner, "source subobject is read before initialization");
      }
      return *value;
    }
    if (auto* object = std::get_if<StorageProjection>(&target.location)) {
      return objects_.load(*object, target.type, owner);
    }
    return storage_.load(std::get<StorageAccess>(target.location), target.type,
                         owner);
  }

  void write(const ObjectLvalue& target, Value value, cxx::AST* owner) {
    if (std::holds_alternative<CurrentReceiver>(target.location)) {
      replace_current_receiver(value);
      return;
    }
    if (auto* destination = std::get_if<Destination>(&target.location)) {
      if (destination->member) {
        values_.replace(destination->binding, destination->component_offset,
                        value, value_arena_);
        if (values_.initialized(destination->binding)) {
          values_.set(destination->binding,
                      name(values_.at(destination->binding),
                           cxx::to_string(destination->binding->name())));
        }
      } else {
        values_.set(destination->binding,
                    name(value, cxx::to_string(destination->binding->name())));
      }
    } else if (auto* object =
                   std::get_if<StorageProjection>(&target.location)) {
      objects_.store(*object, value, target.type, owner);
    } else {
      storage_.store(std::get<StorageAccess>(target.location), value.ssa(),
                     target.type, owner);
    }
  }

  struct ArraySubscriptExpressions {
    // Source array operand, independent of the commuted i[array] spelling.
    cxx::ExpressionAST* array;
    // Integral source selector evaluated in its written position.
    cxx::ExpressionAST* index;
    // Whether the array is evaluated before the selector.
    bool array_first;
  };

  std::optional<ArraySubscriptExpressions> array_subscript_expressions(
      cxx::SubscriptExpressionAST* subscript) {
    if (subscript->symbol) {
      return std::nullopt;
    }
    auto traits = unit_.typeTraits();
    auto* base_type = traits.remove_reference(subscript->baseExpression->type);
    auto* index_type =
        traits.remove_reference(subscript->indexExpression->type);
    if (traits.is_array(base_type)) {
      return ArraySubscriptExpressions{subscript->baseExpression,
                                       subscript->indexExpression, true};
    }
    if (traits.is_array(index_type)) {
      return ArraySubscriptExpressions{subscript->indexExpression,
                                       subscript->baseExpression, false};
    }
    return std::nullopt;
  }

  bool is_semantic_array(cxx::ExpressionAST* expression) {
    auto* type = unit_.typeTraits().remove_reference(expression->type);
    return unit_.typeTraits().is_array(type) &&
           types_.requires_value_transport(type);
  }

  bool is_semantic_array_element(cxx::ExpressionAST* expression) {
    expression = unwrap_object_identity(unit_, expression);
    auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(expression);
    auto operands =
        subscript ? array_subscript_expressions(subscript) : std::nullopt;
    return operands && is_semantic_array(operands->array);
  }

  Lvalue project_array_element(const Lvalue& target,
                               const ArrayPartition& array, size_t index,
                               const cxx::Type* element_type, cxx::AST* owner) {
    const size_t component_offset = index * array.element->component_count;
    if (const auto* selected = target.selection()) {
      std::vector<Lvalue> alternatives;
      alternatives.reserve(selected->alternatives.size());
      for (const auto& alternative : selected->alternatives) {
        alternatives.push_back(project_array_element(alternative, array, index,
                                                     element_type, owner));
      }
      return {std::make_shared<const SelectedLvalue>(
          selected->selector, element_type, std::move(alternatives),
          selected->lane)};
    }
    const auto* object = std::get_if<ObjectLvalue>(&target.projection);
    if (!object) {
      fail(owner, "array element requires a complete source object");
    }
    if (const auto* destination = std::get_if<Destination>(&object->location)) {
      return {ObjectLvalue{
          element_type,
          Destination{destination->binding,
                      destination->component_offset + component_offset,
                      array.element}}};
    }
    if (const auto* storage =
            std::get_if<StorageProjection>(&object->location)) {
      return {
          ObjectLvalue{element_type,
                       storage_.element(*storage, array.source, index, owner)}};
    }
    fail(owner, "semantic array element lost its retained source owner");
  }

  std::optional<Lvalue> semantic_array_element(
      cxx::SubscriptExpressionAST* subscript, cxx::AST* owner,
      LvalueAccess access) {
    auto operands = array_subscript_expressions(subscript);
    if (!operands || !is_semantic_array(operands->array)) {
      return std::nullopt;
    }

    std::optional<Lvalue> array_target;
    std::optional<Value> index_value;
    if (operands->array_first) {
      array_target = reference_lvalue(operands->array, owner, access);
      index_value = expression(operands->index);
    } else {
      index_value = expression(operands->index);
      array_target = reference_lvalue(operands->array, owner, access);
    }

    auto* array = types_.array(operands->array->type, owner);
    if (!array || !array->source->size()) {
      fail(owner, "semantic subscript requires a nonempty bounded array");
    }
    auto* element_type = array->source->elementType();
    cxx::ASTInterpreter interpreter(&unit_);
    auto constant = interpreter.evaluate(operands->index);
    auto* integer = constant ? std::get_if<cxx::ConstInt>(&*constant) : nullptr;
    if (integer && !integer->isNegative() &&
        integer->toUWide() <
            static_cast<cxx::ConstInt::UWide>(array->source->size())) {
      return project_array_element(*array_target, *array,
                                   static_cast<size_t>(integer->toUIntMax()),
                                   element_type, owner);
    }
    if (array->source->size() == 1) {
      return project_array_element(*array_target, *array, 0, element_type,
                                   owner);
    }

    auto* ordinal_type = unit_.control()->getUnsignedIntType();
    auto selector = scalars_.convert(index_value->ssa(), operands->index->type,
                                     ordinal_type, owner);
    std::vector<Lvalue> alternatives;
    std::optional<loom_value_id_t> lane;
    if (const auto* outer = array_target->selection()) {
      alternatives.reserve(outer->alternatives.size() * array->source->size());
      for (const auto& alternative : outer->alternatives) {
        for (size_t index = 0; index < array->source->size(); ++index) {
          alternatives.push_back(project_array_element(
              alternative, *array, index, element_type, owner));
        }
      }
      auto extent = scalars_.integer(
          array->source->size(), LOOM_SCALAR_TYPE_I32, locations_.get(owner));
      auto base = scalars_.binary(cxx::TokenKind::T_STAR, outer->selector,
                                  extent, ordinal_type, ordinal_type, owner);
      selector = scalars_.binary(cxx::TokenKind::T_PLUS, base, selector,
                                 ordinal_type, ordinal_type, owner);
      lane = outer->lane;
    } else {
      alternatives.reserve(array->source->size());
      for (size_t index = 0; index < array->source->size(); ++index) {
        alternatives.push_back(project_array_element(
            *array_target, *array, index, element_type, owner));
      }
    }
    return Lvalue{std::make_shared<const SelectedLvalue>(
        selector, element_type, std::move(alternatives), lane)};
  }

  Lvalue project_vector_lane(const Lvalue& target, loom_value_id_t index,
                             const cxx::Type* index_type,
                             const cxx::Type* element_type, cxx::AST* owner,
                             bool retain_value) {
    if (const auto* selected = target.selection()) {
      auto* lane_type = unit_.control()->getUnsignedLongLongIntType();
      auto lane = scalars_.convert(index, index_type, lane_type, owner);
      std::vector<Lvalue> alternatives;
      alternatives.reserve(selected->alternatives.size());
      for (const auto& alternative : selected->alternatives) {
        alternatives.push_back(project_vector_lane(
            alternative, lane, lane_type, element_type, owner, retain_value));
      }
      return {std::make_shared<const SelectedLvalue>(
          selected->selector, element_type, std::move(alternatives), lane)};
    }
    const auto& object = std::get<ObjectLvalue>(target.projection);
    if (const auto* storage =
            std::get_if<StorageProjection>(&object.location)) {
      auto* pointer_type = unit_.typeTraits().add_pointer(element_type);
      return {ObjectLvalue{
          element_type,
          storage_.advance(*storage, index, pointer_type, index_type,
                           cxx::TokenKind::T_PLUS, owner)}};
    }
    std::optional<Value> snapshot;
    if (retain_value) {
      snapshot = read(object, owner);
    }
    return {VectorLaneLvalue{object, snapshot, index, index_type}};
  }

  Lvalue vector_lane_destination(cxx::SubscriptExpressionAST* subscript,
                                 cxx::AST* owner,
                                 LvalueAccess access = LvalueAccess::Write,
                                 bool retain_value = true) {
    const bool left_vector =
        types_.native_vector(subscript->baseExpression->type);
    auto* vector =
        left_vector ? subscript->baseExpression : subscript->indexExpression;
    auto* index =
        left_vector ? subscript->indexExpression : subscript->baseExpression;
    if (left_vector) {
      auto target = reference_lvalue(vector, owner, access);
      auto coordinate = expression(index);
      return project_vector_lane(target, coordinate.ssa(), index->type,
                                 subscript->type, owner, retain_value);
    }
    // Commuted i[v] syntax still evaluates its written left operand first.
    auto coordinate = expression(index);
    auto target = reference_lvalue(vector, owner, access);
    return project_vector_lane(target, coordinate.ssa(), index->type,
                               subscript->type, owner, retain_value);
  }

  const cxx::Identifier* vector_component_identifier(
      cxx::MemberExpressionAST* selection) const {
    auto* name = cxx::ast_cast<cxx::NameIdAST>(selection->unqualifiedId);
    IREE_ASSERT(selection->vectorComponentKind !=
                cxx::VectorComponentKind::kNone);
    IREE_ASSERT(name && name->identifier);
    return name->identifier;
  }

  const cxx::Type* vector_component_source_type(
      cxx::MemberExpressionAST* selection) const {
    if (selection->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
      return unit_.typeTraits().get_element_type(
          selection->baseExpression->type);
    }
    return selection->baseExpression->type;
  }

  size_t vector_component_index(cxx::MemberExpressionAST* selection,
                                size_t result_position) {
    auto* source =
        types_.native_vector(vector_component_source_type(selection));
    IREE_ASSERT(source);
    return cxx::vector_component_index(selection->vectorComponentKind,
                                       vector_component_identifier(selection),
                                       source->elementCount(), result_position);
  }

  Value read_vector_component(Value source,
                              cxx::MemberExpressionAST* selection) {
    auto* result = types_.native_vector(selection->type);
    if (!result) {
      return Value(vectors_.extract_static(
          source.ssa(),
          static_cast<int64_t>(vector_component_index(selection, 0)),
          vector_component_source_type(selection), selection));
    }
    std::vector<loom_value_id_t> lanes;
    lanes.reserve(result->elementCount());
    for (size_t position = 0; position < result->elementCount(); ++position) {
      lanes.push_back(vectors_.extract_static(
          source.ssa(),
          static_cast<int64_t>(vector_component_index(selection, position)),
          vector_component_source_type(selection), selection));
    }
    return Value(vectors_.construct(lanes, selection->type, selection));
  }

  Value update_vector_component(Value destination, Value replacement,
                                cxx::MemberExpressionAST* selection) {
    auto* source =
        types_.native_vector(vector_component_source_type(selection));
    auto* result = types_.native_vector(selection->type);
    const size_t result_count = result ? result->elementCount() : 1;
    auto updated = destination.ssa();
    for (size_t position = 0; position < result_count; ++position) {
      const size_t index = vector_component_index(selection, position);
      if (index == source->elementCount()) {
        continue;
      }
      auto lane = result
                      ? vectors_.extract_static(replacement.ssa(),
                                                static_cast<int64_t>(position),
                                                selection->type, selection)
                      : replacement.ssa();
      updated = vectors_.insert_static(
          updated, lane, static_cast<int64_t>(index),
          vector_component_source_type(selection), selection);
    }
    return Value(updated);
  }

  Lvalue vector_component_destination(
      cxx::MemberExpressionAST* selection, cxx::AST* owner,
      std::optional<Destination> retained_destination,
      LvalueAccess access = LvalueAccess::Write, bool retain_value = true) {
    ObjectLvalue vector_destination;
    if (selection->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
      auto* source_type = vector_component_source_type(selection);
      if (access == LvalueAccess::Write) {
        types_.require_mutable(source_type, selection);
      }
      vector_destination = {source_type,
                            pointer_expression(selection->baseExpression)};
    } else {
      vector_destination = object_destination(selection->baseExpression, owner,
                                              access, retained_destination);
    }
    std::optional<Value> vector_value;
    if (retain_value) {
      vector_value = read(vector_destination, selection);
    }
    return {VectorComponentLvalue{vector_destination, std::move(vector_value),
                                  selection}};
  }

  Value vector_component_value(cxx::MemberExpressionAST* selection) {
    if (selection->accessOp == cxx::TokenKind::T_MINUS_GREATER) {
      ObjectLvalue source = {vector_component_source_type(selection),
                             pointer_expression(selection->baseExpression)};
      return read_vector_component(read(source, selection), selection);
    }
    return read_vector_component(expression(selection->baseExpression),
                                 selection);
  }

  Lvalue destination(cxx::ExpressionAST* expression, cxx::AST* owner,
                     LvalueAccess access = LvalueAccess::Write,
                     bool preserve_identity = false) {
    auto* projected = unwrap_object_identity(unit_, expression);
    if (unit_.typeTraits().is_class(projected->type) &&
        is_current_object(unit_, projected)) {
      if (access == LvalueAccess::Write) {
        types_.require_mutable(projected->type, owner);
      }
      const auto& receiver = *receivers_.current();
      if (receiver.storage) {
        return {ObjectLvalue{projected->type, *receiver.storage}};
      }
      return {ObjectLvalue{projected->type,
                           CurrentReceiver{receivers_.current()->record}}};
    }
    if (cxx::ast_cast<cxx::ConditionalExpressionAST>(projected)) {
      return reference_lvalue(projected, owner, access);
    }
    if (auto* subscript =
            cxx::ast_cast<cxx::SubscriptExpressionAST>(projected)) {
      if (auto target = semantic_array_element(subscript, owner, access)) {
        return std::move(*target);
      }
    }
    if (auto called = reference_call(projected)) {
      if (called->reference_target) {
        return *called->reference_target;
      }
      if (called->value && called->value->is_pointer()) {
        return {ObjectLvalue{projected->type,
                             storage_.project(called->value->pointer(),
                                              projected->type, owner)}};
      }
      if (called->value) {
        // An exact alias of an unnamed receiver still names that temporary.
        // Materialization moves its existing cleanup action onto storage.
        return {ObjectLvalue{projected->type,
                             materialize_temporary(projected, *called->value,
                                                   called->temporary)}};
      }
      fail(owner, "reference result has no retained source destination");
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
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(projected);
        member && member->accessOp == cxx::TokenKind::T_DOT &&
        member->vectorComponentKind == cxx::VectorComponentKind::kNone &&
        ((!control_->destination(member) &&
          !control_->storage_backed(member)) ||
         is_semantic_array_element(member->baseExpression))) {
      auto base = reference_lvalue(member->baseExpression, owner, access);
      return base.project(
          types_.selection(member->symbol, member->subobjectPath, member),
          member->type, storage_, owner);
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
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(projected);
        member &&
        member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
      auto retained = control_->destination(expression);
      if (!retained && projected != expression) {
        retained = control_->destination(projected);
      }
      return vector_component_destination(member, owner, retained, access);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(projected);
        subscript && (types_.native_vector(subscript->baseExpression->type) ||
                      types_.native_vector(subscript->indexExpression->type))) {
      return vector_lane_destination(subscript, owner, access);
    }
    auto retained = control_->destination(expression);
    if (!retained && projected != expression) {
      retained = control_->destination(projected);
    }
    return {object_destination(expression, owner, access, retained,
                               preserve_identity)};
  }

  Lvalue reference_conditional(cxx::ConditionalExpressionAST* expression,
                               cxx::AST* owner, LvalueAccess access,
                               bool force_ordinal = false) {
    auto condition = this->expression(expression->condition).ssa();
    auto written = live_mutations(expression);
    auto initial = values_;
    ConditionalRegions regions(builder_);
    conditional_cleanup_anchors_.push_back(loom_builder_save(&builder_));
    std::array<std::optional<Lvalue>, 2> targets;
    std::array<std::vector<loom_value_id_t>, 2> mutations;
    for (size_t arm = 0; arm < targets.size(); ++arm) {
      auto saved = regions.enter(arm);
      const size_t temporary_mark = temporary_destructions_.size();
      targets[arm] = reference_lvalue(arm == 0 ? expression->iftrueExpression
                                               : expression->iffalseExpression,
                                      owner, access, force_ordinal);
      materialize_conditional_temporaries(temporary_mark);
      mutations[arm] = current(written);
      loom_builder_restore(&builder_, saved);
      values_ = initial;
    }
    conditional_cleanup_anchors_.pop_back();

    std::array<const StorageProjection*, 2> storage = {};
    bool has_lane = false;
    for (size_t arm = 0; arm < targets.size(); ++arm) {
      const auto& target = *targets[arm];
      if (const auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
        storage[arm] = std::get_if<StorageProjection>(&object->location);
      }
      has_lane |= std::holds_alternative<VectorLaneLvalue>(target.projection);
      if (const auto* nested = target.selection()) {
        has_lane |= nested->lane.has_value();
      }
    }
    const bool addressable = !force_ordinal && storage[0] && storage[1];
    if (!force_ordinal && !addressable && (storage[0] || storage[1])) {
      fail(owner, "mixed reference selection requires addressable origins");
    }

    std::vector<Lvalue> alternatives;
    auto* ordinal_type = unit_.control()->getUnsignedIntType();
    auto* lane_type = unit_.control()->getUnsignedLongLongIntType();
    auto source = locations_.get(owner);
    for (size_t arm = 0; arm < targets.size(); ++arm) {
      auto saved = regions.enter(arm);
      auto& target = *targets[arm];
      std::vector<loom_value_id_t> yielded;
      if (addressable) {
        Value(storage[arm]->pointer).append_to(yielded);
      } else {
        auto ordinal =
            scalars_.integer(alternatives.size(), LOOM_SCALAR_TYPE_I32, source);
        std::optional<loom_value_id_t> lane;
        if (const auto* nested = target.selection()) {
          alternatives.insert(alternatives.end(), nested->alternatives.begin(),
                              nested->alternatives.end());
          ordinal = scalars_.binary(cxx::TokenKind::T_PLUS, nested->selector,
                                    ordinal, ordinal_type, ordinal_type, owner);
          lane = nested->lane;
        } else {
          if (const auto* indexed =
                  std::get_if<VectorLaneLvalue>(&target.projection)) {
            lane = scalars_.convert(indexed->index, indexed->index_type,
                                    lane_type, owner);
          }
          alternatives.push_back(std::move(target));
        }
        yielded.push_back(ordinal);
        if (has_lane) {
          yielded.push_back(
              lane ? *lane : scalars_.integer(0, LOOM_SCALAR_TYPE_I64, source));
        }
      }
      yielded.insert(yielded.end(), mutations[arm].begin(),
                     mutations[arm].end());
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 source, &yield));
      loom_builder_restore(&builder_, saved);
    }

    std::vector<const cxx::Type*> sources;
    if (addressable) {
      sources.push_back(unit_.typeTraits().add_pointer(expression->type));
    } else {
      sources.push_back(ordinal_type);
      if (has_lane) {
        sources.push_back(lane_type);
      }
    }
    const size_t result_count =
        addressable ? kPointerPartition.component_count : 1 + size_t(has_lane);
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
    }
    auto outputs = bind_signature(types_, sources, owner, &builder_);
    auto* op = regions.build(condition, outputs.types, source);
    auto* results = loom_op_results(op);
    bind_values(written, results + result_count);
    if (addressable) {
      return {ObjectLvalue{
          expression->type,
          StorageProjection{
              {results[0], results[1]},
              std::min(storage[0]->alignment, storage[1]->alignment),
              storage[0]->pointer_width_constrained &&
                  storage[1]->pointer_width_constrained}}};
    }
    std::optional<loom_value_id_t> coordinate;
    if (has_lane) {
      coordinate = results[1];
      // A destination is read or written only in its selected arm, using the
      // coordinate evaluated by that same arm.
      for (auto& alternative : alternatives) {
        auto* lane = std::get_if<VectorLaneLvalue>(&alternative.projection);
        if (!lane) {
          continue;
        }
        lane->index = *coordinate;
        lane->index_type = lane_type;
      }
    }
    return {std::make_shared<const SelectedLvalue>(
        results[0], expression->type, std::move(alternatives), coordinate)};
  }

  Lvalue reference_lvalue(cxx::ExpressionAST* expression, cxx::AST* owner,
                          LvalueAccess access, bool force_ordinal = false) {
    auto* projected = unwrap_object_identity(unit_, expression);
    if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(projected);
        member &&
        member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
      fail(member, "extended-vector components are not addressable");
    }
    if (auto* conditional =
            cxx::ast_cast<cxx::ConditionalExpressionAST>(projected)) {
      if (projected->valueCategory == cxx::ValueCategory::kPrValue) {
        return {ObjectLvalue{projected->type, object_address(projected)}};
      }
      return reference_conditional(conditional, owner, access, force_ordinal);
    }
    auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(projected);
    const bool vector_lane =
        subscript && (types_.native_vector(subscript->baseExpression->type) ||
                      types_.native_vector(subscript->indexExpression->type));
    if (vector_lane) {
      return vector_lane_destination(subscript, owner, access,
                                     /*retain_value=*/false);
    }

    auto target =
        destination(projected, owner, access, /*preserve_identity=*/true);
    return target;
  }

  Value reference_value(cxx::ExpressionAST* expression, cxx::AST* owner) {
    auto* projected = unwrap_object_identity(unit_, expression);
    if (auto* conditional =
            cxx::ast_cast<cxx::ConditionalExpressionAST>(projected);
        conditional &&
        projected->valueCategory != cxx::ValueCategory::kPrValue) {
      auto condition = this->expression(conditional->condition).ssa();
      return conditional_reference(conditional, condition);
    }
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
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
    }
    auto initial = values_;
    auto source = locations_.get(ast);
    auto outputs = bind_signature(types_, sources, ast, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
    conditional_cleanup_anchors_.push_back(
        {op->parent_block, op->parent_op, op});
    auto build_arm = [&](loom_region_t* region,
                         cxx::ExpressionAST* expression) {
      auto saved = loom_builder_enter_region(&builder_, op, region);
      const size_t temporary_mark = temporary_destructions_.size();
      std::vector<loom_value_id_t> yielded;
      reference_value(expression, ast).append_to(yielded);
      materialize_conditional_temporaries(temporary_mark);
      auto mutations = current(written);
      yielded.insert(yielded.end(), mutations.begin(), mutations.end());
      loom_op_t* yield;
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 source, &yield));
      loom_builder_restore(&builder_, saved);
    };
    build_arm(loom_scf_if_then_region(op), ast->iftrueExpression);
    values_ = initial;
    build_arm(loom_scf_if_else_region(op), ast->iffalseExpression);
    conditional_cleanup_anchors_.pop_back();
    values_ = std::move(initial);
    bind_values(written,
                loom_op_results(op) + kPointerPartition.component_count);
    return value_arena_.capture(
        kPointerPartition,
        {loom_op_results(op), kPointerPartition.component_count});
  }

  Value read(const Lvalue& target, cxx::AST* owner) {
    if (const auto* selected = target.selection()) {
      return dispatch_selection(*selected, owner, [&](size_t arm) {
        return read(selected->alternatives[arm], owner);
      });
    }
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      return read(*object, owner);
    }
    if (auto* lane = std::get_if<VectorLaneLvalue>(&target.projection)) {
      auto vector = lane->value ? *lane->value : read(lane->destination, owner);
      return Value(vectors_.extract(vector.ssa(), lane->index,
                                    lane->destination.type, lane->index_type,
                                    owner));
    }
    const auto& component = std::get<VectorComponentLvalue>(target.projection);
    auto vector =
        component.value ? *component.value : read(component.destination, owner);
    return read_vector_component(vector, component.selection);
  }

  Value read_selected_object(const ObjectLvalue& object,
                             ValueBindings::Updates& updates, cxx::AST* owner) {
    auto project = [&](cxx::Symbol* binding, const Partition& partition,
                       size_t offset) {
      auto value = updates.project(binding, partition, offset);
      if (!value) {
        fail(owner, "source subobject is read before initialization");
      }
      return *value;
    };
    if (const auto* destination = std::get_if<Destination>(&object.location)) {
      const auto& partition = destination->member
                                  ? *destination->member
                                  : values_.partition(destination->binding);
      return project(destination->binding, partition,
                     destination->component_offset);
    }
    const auto& receiver = std::get<CurrentReceiver>(object.location);
    std::vector<loom_value_id_t> components;
    components.reserve(receiver.record->component_count);
    for (const auto& member : receiver.record->members) {
      project(member.source, *member.partition, 0).append_to(components);
    }
    return value_arena_.capture(*receiver.record, components);
  }

  loom_value_id_t combine_selection_condition(
      std::optional<loom_value_id_t> outer, loom_value_id_t inner,
      cxx::AST* owner) {
    if (!outer) {
      return inner;
    }
    auto* bool_type = unit_.control()->getBoolType();
    return scalars_.binary(cxx::TokenKind::T_AMP, *outer, inner, bool_type,
                           bool_type, owner);
  }

  void accumulate_selection_write(
      const SelectedLvalue& selected, Value value, cxx::AST* owner,
      std::optional<loom_value_id_t> outer_condition,
      ValueBindings::Updates& updates) {
    for (size_t ordinal = 0; ordinal < selected.alternatives.size();
         ++ordinal) {
      const auto& alternative = selected.alternatives[ordinal];
      if (const auto* nested = alternative.selection()) {
        auto condition = combine_selection_condition(
            outer_condition, selection_condition(selected, ordinal, owner),
            owner);
        accumulate_selection_write(*nested, value, owner, condition, updates);
        continue;
      }
      const auto& object = alternative.object();
      if (std::holds_alternative<StorageAccess>(object.location) ||
          std::holds_alternative<StorageProjection>(object.location)) {
        auto condition = combine_selection_condition(
            outer_condition, selection_condition(selected, ordinal, owner),
            owner);
        auto source = locations_.get(owner);
        loom_op_t* op;
        check(loom_scf_if_build(&builder_, 0, condition, nullptr, 0, nullptr, 0,
                                source, &op));
        auto saved = loom_builder_enter_region(&builder_, op,
                                               loom_scf_if_then_region(op));
        write(alternative, value, owner);
        loom_op_t* yield;
        check(loom_scf_yield_build(&builder_, nullptr, 0, source, &yield));
        loom_builder_restore(&builder_, saved);
        continue;
      }
      // Binary choices share one predicate with opposite arm polarity. This
      // retains the complement known by the selection owner without requiring
      // a later pass to recover it from the ordinal's range.
      const bool complementary =
          !outer_condition && selected.alternatives.size() == 2 && ordinal == 1;
      auto condition = combine_selection_condition(
          outer_condition,
          selection_condition(selected, complementary ? 0 : ordinal, owner),
          owner);
      auto updated =
          selected_value(condition, object.type, owner, [&](size_t arm) {
            const bool replace = (arm == 0) != complementary;
            if (replace &&
                std::holds_alternative<ObjectLvalue>(alternative.projection)) {
              return value;
            }
            auto current = read_selected_object(object, updates, owner);
            if (!replace) {
              return current;
            }
            if (const auto* lane =
                    std::get_if<VectorLaneLvalue>(&alternative.projection)) {
              auto original = lane->value ? *lane->value : current;
              return Value(vectors_.insert(original.ssa(), value.ssa(),
                                           lane->index, object.type,
                                           lane->index_type, owner));
            }
            const auto& component =
                std::get<VectorComponentLvalue>(alternative.projection);
            return update_vector_component(
                component.value ? *component.value : current, value,
                component.selection);
          });
      if (const auto* destination =
              std::get_if<Destination>(&object.location)) {
        updates.replace(destination->binding, destination->component_offset,
                        updated);
      } else {
        const auto& receiver = std::get<CurrentReceiver>(object.location);
        for (const auto& member : receiver.record->members) {
          updates.replace(
              member.source, 0,
              updated.project(*member.partition, member.component_offset));
        }
      }
    }
  }

  void write_selection(const SelectedLvalue& selected, Value value,
                       cxx::AST* owner) {
    ValueBindings::Updates updates(values_, value_arena_);
    accumulate_selection_write(selected, value, owner, std::nullopt, updates);
    updates.apply();
    for (auto* binding : selected.bindings) {
      if (auto found = values_.find(binding); found != values_.end()) {
        name(found->second, cxx::to_string(binding->name()));
      }
    }
  }

  void write(const Lvalue& target, Value value, cxx::AST* owner) {
    if (const auto* selected = target.selection()) {
      write_selection(*selected, value, owner);
      return;
    }
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      write(*object, value, owner);
      return;
    }
    if (auto* lane = std::get_if<VectorLaneLvalue>(&target.projection)) {
      auto vector = lane->value ? *lane->value : read(lane->destination, owner);
      auto updated =
          vectors_.insert(vector.ssa(), value.ssa(), lane->index,
                          lane->destination.type, lane->index_type, owner);
      write(lane->destination, Value(updated), owner);
      return;
    }
    const auto& component = std::get<VectorComponentLvalue>(target.projection);
    auto vector =
        component.value ? *component.value : read(component.destination, owner);
    write(component.destination,
          update_vector_component(vector, value, component.selection), owner);
  }

  struct ReferenceWriteback {
    // Once-evaluated lvalue receiving the mutable result, absent for a prvalue
    // temporary identified by its evaluated cleanup token.
    std::optional<Lvalue> target;
    // Admitted value representation transported through the call.
    const Partition* partition;
    // Unqualified source value type used to bind the returned value.
    const cxx::Type* value_type;
    // First flattened call operand for this source parameter.
    size_t argument_offset;
    // Exact evaluated temporary receiving this writeback, if nontrivial.
    std::optional<TemporaryReference> temporary;
    // Sparse output groups, or null for whole-value writeback.
    const ParameterProjection* projection = nullptr;
    // Complete temporary payload retained when only some groups are returned.
    std::optional<Value> source_value;
  };

  // One reference result proven to alias an already evaluated source argument.
  // The value changes only when a mutable value-reference writeback replaces
  // the corresponding flattened operand interval.
  struct ReferenceResultAlias {
    // Evaluated argument value reused as the source call result.
    Value value;
    // Once-evaluated automatic destination named by the result.
    std::optional<Lvalue> target;
    // First flattened call operand occupied by the aliased parameter.
    size_t argument_offset;
    // Evaluated unnamed object retained when the reference escapes this call.
    std::optional<TemporaryReference> temporary;
  };

  // One caller-owned origin admitted by a finite selected-reference result.
  // Named and storage-backed arguments retain their lvalue; an unnamed const
  // value remains an SSA payload because no valid source use can mutate it.
  struct SelectedReferenceCandidate {
    // Concrete caller destination, when the argument has one.
    std::optional<Lvalue> target;
    // Unnamed read-only value, present only when there is no destination.
    std::optional<Value> value;
  };

  // One source argument evaluated exactly once, before its components are
  // arranged in callable ABI order. A mutable value reference retains its
  // destination until the call results are available.
  struct EvaluatedCallArgument {
    // Completed value or storage pointer. An SSA lvalue reference is read from
    // |target| only after all argument evaluations have finished.
    std::optional<Value> value;
    // Automatic lvalue named by a value-reference argument. Read-only
    // parameters retain it because an exact reference result can be mutated by
    // a later call.
    std::optional<Lvalue> target;
    // Admitted value structure of a value reference, including read-only ones.
    const Partition* reference_partition = nullptr;
    // Unqualified value type of a mutable value reference.
    const cxx::Type* writeback_type = nullptr;
    // Exact temporary construction evaluated for this argument.
    std::optional<TemporaryReference> temporary;
    // Retained source input/output selection for this parameter.
    const ParameterProjection* projection = nullptr;
  };

  EvaluatedCallArgument evaluate_call_argument(
      cxx::ExpressionAST* argument, cxx::ParameterSymbol* parameter,
      const ReferenceParameter* reference, cxx::AST* owner) {
    if (!reference) {
      auto value = expression(argument);
      adopt_temporary_result(argument);
      return {.value = value};
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
    std::optional<Value> value;
    std::optional<TemporaryReference> temporary;
    if (argument->valueCategory != cxx::ValueCategory::kPrValue) {
      if (auto called = reference_call(argument)) {
        target = std::move(called->reference_target);
        temporary = called->temporary;
        if (!target) {
          IREE_ASSERT(called->value && !called->value->is_pointer());
          value = *called->value;
        }
      } else {
        target = destination(argument, owner,
                             reference->mutable_access ? LvalueAccess::Write
                                                       : LvalueAccess::Read);
      }
    } else {
      value = expression(argument);
      temporary = temporary_result(argument);
    }
    return {
        .value = value,
        .target = std::move(target),
        .reference_partition = &expected_partition,
        .writeback_type = reference->mutable_access ? expected_type : nullptr,
        .temporary = temporary,
        .projection = reference->projection.get(),
    };
  }

  SelectedReferenceCandidate selected_call_argument_candidate(
      EvaluatedCallArgument& argument, cxx::ParameterSymbol* parameter,
      const ReferenceParameter& reference, cxx::AST* owner) {
    if (argument.target) {
      return {.target = *argument.target};
    }
    auto* object_type = unit_.typeTraits().remove_reference(parameter->type());
    IREE_ASSERT(argument.value);
    if (reference.abi == ReferenceABI::Storage) {
      IREE_ASSERT(argument.value->is_pointer());
      auto projection =
          storage_.project(argument.value->pointer(), object_type, owner);
      if (unit_.typeTraits().is_class(object_type) ||
          unit_.typeTraits().is_array(object_type)) {
        return {.target = Lvalue{ObjectLvalue{object_type, projection}}};
      }
      return {
          .target = Lvalue{ObjectLvalue{
              object_type,
              storage_.dereference(projection, object_type, owner),
          }},
      };
    }
    return {.value = *argument.value};
  }

  SelectedReferenceCandidate project_reference_candidate(
      SelectedReferenceCandidate candidate, const ValueProjection& projection,
      cxx::AST* owner) {
    if (candidate.target) {
      candidate.target =
          project_call_target(*candidate.target, projection, owner);
    } else {
      IREE_ASSERT(candidate.value);
      candidate.value = candidate.value->project(*projection.partition,
                                                 projection.component_offset);
    }
    return candidate;
  }

  Value selected_call_result_value(
      loom_value_id_t selector,
      std::span<const SelectedReferenceCandidate> candidates,
      const cxx::Type* result_type, cxx::AST* owner) {
    auto evaluate = [&](size_t ordinal) {
      const auto& candidate = candidates[ordinal];
      return candidate.target ? read(*candidate.target, owner)
                              : *candidate.value;
    };
    return dispatch_ordinal(selector, candidates.size(), result_type, owner,
                            evaluate);
  }

  Lvalue project_call_target(const Lvalue& target, const ValueProjection& group,
                             cxx::AST* owner) {
    if (const auto* selected = target.selection()) {
      std::vector<Lvalue> alternatives;
      alternatives.reserve(selected->alternatives.size());
      for (const auto& alternative : selected->alternatives) {
        alternatives.push_back(project_call_target(alternative, group, owner));
      }
      return {std::make_shared<const SelectedLvalue>(
          selected->selector, group.type, std::move(alternatives),
          selected->lane)};
    }
    const auto* object = std::get_if<ObjectLvalue>(&target.projection);
    if (!object) {
      return target;
    }
    if (auto* destination = std::get_if<Destination>(&object->location)) {
      return {ObjectLvalue{
          group.type,
          Destination{destination->binding,
                      destination->component_offset + group.component_offset,
                      group.partition}}};
    }
    if (auto* receiver = std::get_if<CurrentReceiver>(&object->location)) {
      const auto& members = receiver->record->members;
      auto member = std::upper_bound(
          members.begin(), members.end(), group.component_offset,
          [](size_t offset, const SubobjectPartition& value) {
            return offset < value.component_offset;
          });
      --member;
      return {ObjectLvalue{group.type, Destination{member->source,
                                                   group.component_offset -
                                                       member->component_offset,
                                                   group.partition}}};
    }
    if (group.component_offset == 0 &&
        group.partition ==
            &types_.partition(types_.unqualified_value(object->type), owner)) {
      return target;
    }
    fail(owner,
         "projected reference call requires a source value or complete scalar "
         "storage");
  }

  Lvalue current_receiver_target() const {
    const auto* record = receivers_.current()->record;
    return {ObjectLvalue{record->source->type(), CurrentReceiver{record}}};
  }

  void materialize_call_argument(EvaluatedCallArgument& argument,
                                 cxx::AST* owner) {
    if (argument.projection) {
      return;
    }
    if (!argument.value) {
      argument.value = read(*argument.target, owner);
    }
    if (argument.reference_partition &&
        &argument.value->partition() != argument.reference_partition) {
      fail(owner,
           "runtime value-reference binding must preserve the admitted "
           "source value type");
    }
  }

  void append_call_argument(EvaluatedCallArgument argument,
                            size_t receiver_argument_count,
                            std::vector<loom_value_id_t>& arguments,
                            std::vector<ReferenceWriteback>& writebacks,
                            cxx::AST* owner) {
    const size_t argument_offset = receiver_argument_count + arguments.size();
    if (argument.projection) {
      for (const auto& group : argument.projection->inputs) {
        auto input =
            argument.value
                ? argument.value->project(*group.partition,
                                          group.component_offset)
                : read(project_call_target(*argument.target, group, owner),
                       owner);
        input.append_to(arguments);
      }
    } else {
      argument.value->append_to(arguments);
    }
    if (argument.writeback_type) {
      writebacks.push_back({
          .target = std::move(argument.target),
          .partition = argument.reference_partition,
          .value_type = argument.writeback_type,
          .argument_offset = argument_offset,
          .temporary = argument.temporary,
          .projection = argument.projection,
          .source_value = argument.value,
      });
    }
  }

  struct ReceiverArgument {
    // Native callable specialization selected by receiver ownership.
    ImplicitObjectABI abi;
    // Prvalue record or resolved storage pointer, when evaluated eagerly.
    std::optional<Value> value;
    // Value structure used by a mutable receiver result.
    const RecordPartition* record;
    // Once-evaluated external lvalue, absent for temporaries/current object.
    std::optional<Lvalue> target;
    // Whether mutable results replace the active value receiver.
    bool current;
    // Exact top-level fields returned by a value receiver call.
    std::span<cxx::Symbol* const> written_fields;
    // Exact evaluated temporary whose current payload cleanup must observe.
    std::optional<TemporaryReference> temporary;
  };

  ReceiverArgument current_receiver_argument(
      const RecordPartition& record, bool requires_storage,
      std::span<cxx::Symbol* const> written_fields, cxx::AST* owner) {
    if (!receivers_.current() || receivers_.current()->record != &record) {
      fail(owner,
           "implicit member call requires a matching current source object");
    }
    const auto& active = *receivers_.current();
    if (active.storage) {
      return {ImplicitObjectABI::Storage,
              Value(active.storage->pointer),
              &record,
              std::nullopt,
              true,
              written_fields};
    }
    if (requires_storage) {
      fail(owner, "member function requires a storage-backed receiver");
    }
    return {ImplicitObjectABI::Value,
            std::nullopt,
            &record,
            std::nullopt,
            true,
            written_fields};
  }

  std::optional<EvaluatedCallResult> reference_call(
      cxx::ExpressionAST* expression) {
    expression = unwrap_object_identity(unit_, expression);
    if (auto call = direct_operator_call(expression)) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(call->function->type());
      if (signature &&
          unit_.typeTraits().is_reference(signature->returnType())) {
        return ordinary_operator_call(expression, *call);
      }
      return std::nullopt;
    }
    if (auto* subscript =
            cxx::ast_cast<cxx::SubscriptExpressionAST>(expression);
        subscript && subscript->symbol) {
      auto* signature =
          cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
      if (signature &&
          unit_.typeTraits().is_reference(signature->returnType())) {
        return overloaded_subscript_call(subscript);
      }
      return std::nullopt;
    }
    auto* call = cxx::ast_cast<cxx::CallExpressionAST>(expression);
    auto* function = call ? direct_callee(call) : nullptr;
    auto* signature = function
                          ? cxx::type_cast<cxx::FunctionType>(function->type())
                          : nullptr;
    if (!signature ||
        !unit_.typeTraits().is_reference(signature->returnType())) {
      return std::nullopt;
    }
    if (intrinsics_.check_binding(function, expression) ||
        functions_.is_check_case(function) ||
        annotated(function, "check_benchmark") ||
        intrinsics_.lookup(function, expression)) {
      fail(expression,
           "intrinsic and check calls cannot supply an implicit object");
    }
    if (annotated(function, "kernel")) {
      fail(expression, "kernels require an explicit launch operation");
    }
    if (functions_.is_command_program(function)) {
      fail(expression,
           "command programs require an explicit command launch operation");
    }
    if (pipeline_contracts_.contains(function)) {
      fail(expression,
           "pipelines require explicit composition or kernel launch");
    }
    return ordinary_call(call, function, locations_.get(call));
  }

  ReceiverArgument unprojected_receiver_argument(
      cxx::FunctionSymbol* function, cxx::ExpressionAST* receiver,
      cxx::TokenKind access_op, cxx::AST* owner,
      const RecordPartition* enclosing_record = nullptr) {
    auto* object_type = implicit_object_type(unit_, function);
    auto* record = object_type
                       ? types_.record(types_.unqualified(object_type), owner)
                       : nullptr;
    if (enclosing_record) {
      record = enclosing_record;
    }
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
    if (!receiver || is_current_object(unit_, receiver)) {
      return admit(current_receiver_argument(*record, requires_storage,
                                             written_fields, owner));
    }
    if (access_op == cxx::TokenKind::T_MINUS_GREATER) {
      auto projection = pointer_expression(receiver);
      return admit({ImplicitObjectABI::Storage, Value(projection.pointer),
                    record, std::nullopt, false, written_fields});
    }
    if (access_op != cxx::TokenKind::T_DOT) {
      fail(owner, "member call requires a dot or arrow receiver");
    }

    if (auto called = reference_call(receiver)) {
      if (called->reference_target) {
        return admit({ImplicitObjectABI::Value, std::nullopt, record,
                      std::move(called->reference_target), false,
                      written_fields, called->temporary});
      }
      if (!called->value) {
        fail(receiver, "void function cannot supply an implicit object");
      }
      if (called->value->is_pointer()) {
        return admit({ImplicitObjectABI::Storage, *called->value, record,
                      std::nullopt, false, written_fields});
      }
      if (&called->value->partition() != record) {
        fail(receiver,
             "member call receiver must preserve its admitted record type");
      }
      return admit({ImplicitObjectABI::Value,
                    called->reference_target ? std::nullopt : called->value,
                    record, std::move(called->reference_target), false,
                    written_fields, called->temporary});
    }

    bool direct_prvalue =
        receiver->valueCategory == cxx::ValueCategory::kPrValue;
    if (!direct_prvalue &&
        receiver->valueCategory == cxx::ValueCategory::kXValue) {
      auto* materialization =
          cxx::ast_cast<cxx::ImplicitCastExpressionAST>(receiver);
      direct_prvalue =
          materialization &&
          materialization->castKind ==
              cxx::ImplicitCastKind::kTemporaryMaterializationConversion &&
          materialization->expression->valueCategory ==
              cxx::ValueCategory::kPrValue;
    }
    if (direct_prvalue) {
      auto value = expression(receiver);
      if (&value.partition() != record) {
        fail(receiver,
             "member call receiver must preserve its admitted record type");
      }
      if (requires_storage) {
        auto projection = materialize_temporary(receiver, value);
        return {ImplicitObjectABI::Storage,
                Value(projection.pointer),
                record,
                std::nullopt,
                false,
                written_fields};
      }
      return admit({ImplicitObjectABI::Value, value, record, std::nullopt,
                    false, written_fields, temporary_result(receiver)});
    }

    auto target =
        destination(receiver, owner,
                    mutable_access ? LvalueAccess::Write : LvalueAccess::Read);
    if (target.selection()) {
      return admit({ImplicitObjectABI::Value, std::nullopt, record,
                    std::move(target), false, written_fields});
    }
    auto* object = std::get_if<ObjectLvalue>(&target.projection);
    if (!object) {
      fail(receiver, "member call receiver requires a complete object");
    }
    if (auto* storage = std::get_if<StorageProjection>(&object->location)) {
      return admit({ImplicitObjectABI::Storage, Value(storage->pointer), record,
                    std::nullopt, false, written_fields});
    }
    return admit({ImplicitObjectABI::Value, std::nullopt, record,
                  std::move(target), false, written_fields});
  }

  ReceiverArgument receiver_argument(cxx::FunctionSymbol* function,
                                     cxx::ExpressionAST* receiver,
                                     cxx::TokenKind access_op, cxx::AST* owner,
                                     cxx::List<cxx::Symbol*>* path = nullptr) {
    if (!path) {
      return unprojected_receiver_argument(function, receiver, access_op,
                                           owner);
    }
    const auto& selected = types_.projection(path, owner);
    auto* enclosing = types_.record(path->value->parent()->type(), owner);
    auto result = unprojected_receiver_argument(function, receiver, access_op,
                                                owner, enclosing);
    auto* type = implicit_object_type(unit_, function);
    result.record = static_cast<const RecordPartition*>(selected.partition);
    if (result.abi == ImplicitObjectABI::Storage) {
      result.value =
          storage_
              .subobject(storage_.project(result.value->pointer(),
                                          enclosing->source->type(), owner),
                         selected.byte_offset, owner)
              .pointer;
    } else if (result.current) {
      result.target = Lvalue{ObjectLvalue{
          type, Destination{
                    selected.root->source,
                    selected.component_offset - selected.root->component_offset,
                    selected.partition}}};
    } else if (result.target) {
      result.target = result.target->project(selected, type, storage_, owner);
    } else {
      if (result.temporary) {
        result.temporary->component_offset += selected.component_offset;
        result.temporary->byte_offset += selected.byte_offset;
      }
      result.value =
          result.value->project(*selected.partition, selected.component_offset);
    }
    result.current = false;
    return result;
  }

  ReceiverArgument receiver_argument(cxx::CallExpressionAST* call,
                                     cxx::FunctionSymbol* function) {
    auto* base = unwrap_expression(call->baseExpression);
    auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(base);
    auto* id = cxx::ast_cast<cxx::IdExpressionAST>(base);
    return receiver_argument(
        function, member ? member->baseExpression : nullptr,
        member ? member->accessOp : cxx::TokenKind::T_DOT, call,
        member ? member->subobjectPath
        : id   ? id->subobjectPath
               : nullptr);
  }

  ReceiverArgument receiver_argument(cxx::SubscriptExpressionAST* call,
                                     cxx::FunctionSymbol* function) {
    return receiver_argument(function, call->baseExpression,
                             cxx::TokenKind::T_DOT, call);
  }

  size_t receiver_input_count(
      cxx::FunctionSymbol* function,
      const std::optional<ReceiverArgument>& receiver) const {
    if (!receiver) {
      return 0;
    }
    if (receiver->abi == ImplicitObjectABI::Storage) {
      return kPointerPartition.component_count;
    }
    const auto* projection = functions_.receiver_projection(function);
    return projection ? projection->input_component_count
                      : receiver->record->component_count;
  }

  Value materialize_receiver(ReceiverArgument& receiver, cxx::AST* owner) {
    if (receiver.abi == ImplicitObjectABI::Storage) {
      IREE_ASSERT(receiver.value && receiver.value->is_pointer());
      return *receiver.value;
    }
    if (!receiver.value) {
      receiver.value = receiver.current ? current_receiver_value(owner)
                                        : read(*receiver.target, owner);
    }
    Value value = *receiver.value;
    if (&value.partition() != receiver.record) {
      fail(owner,
           "member call receiver must preserve its admitted record type");
    }
    return value;
  }

  SelectedReferenceCandidate receiver_reference_candidate(
      ReceiverArgument& receiver, const ValueProjection& origin,
      const std::optional<Value>& updated, cxx::AST* owner) {
    if (receiver.abi == ImplicitObjectABI::Storage) {
      IREE_ASSERT(receiver.value && receiver.value->is_pointer());
      IREE_ASSERT_EQ(origin.component_offset, 0u);
      IREE_ASSERT(origin.type ==
                  types_.unqualified(receiver.record->source->type()));
      auto projection =
          storage_.project(receiver.value->pointer(), origin.type, owner);
      return {.target = Lvalue{ObjectLvalue{origin.type, projection}}};
    }
    if (receiver.current || receiver.target) {
      auto target =
          receiver.current ? current_receiver_target() : *receiver.target;
      return {.target = project_call_target(target, origin, owner)};
    }
    const Value* value = updated          ? &*updated
                         : receiver.value ? &*receiver.value
                                          : nullptr;
    IREE_ASSERT(value);
    return {
        .value = value->project(*origin.partition, origin.component_offset),
    };
  }

  void write_receiver(const ReceiverArgument& receiver, Value value,
                      cxx::AST* owner) {
    if (receiver.current) {
      replace_current_receiver(value);
    } else if (receiver.target) {
      write(*receiver.target, value, owner);
    } else {
      IREE_ASSERT(receiver.value);
      update_temporary(receiver.temporary, value,
                       receiver.record->source->type(), owner);
    }
  }

  EvaluatedCallResult ordinary_call(cxx::CallExpressionAST* call,
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

  EvaluatedCallResult ordinary_call(cxx::SubscriptExpressionAST* call,
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

  EvaluatedCallResult overloaded_subscript_call(
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

  std::vector<loom_value_id_t> dynamic_indices(const IndexList& list) {
    std::vector<loom_value_id_t> values;
    values.reserve(list.dynamic_expressions().size());
    for (auto* source : list.dynamic_expressions()) {
      auto value = expression(source).ssa();
      values.push_back(IndexExpression::build(unit_, &builder_, source, value,
                                              locations_.get(source)));
    }
    return values;
  }

  Value pipeline_memory(cxx::CallExpressionAST* call,
                        const PipelineMemoryIntrinsic& binding) {
    if (current_function_.kind != FunctionKind::Pipeline) {
      fail(call, "pipeline.memory requires a pipeline definition body");
    }
    if (pipeline_strand_depth_) {
      fail(call, "pipeline.memory belongs outside pipeline strand bodies");
    }
    auto coordinates = IndexList::admit(unit_, diagnostics_, types_,
                                        call->expressionList->value,
                                        "pipeline.memory coordinates");
    auto dynamic = dynamic_indices(coordinates);
    loom_op_t* op;
    check(loom_pipeline_memory_build(
        &builder_, binding.memory_space, dynamic.data(), dynamic.size(),
        coordinates.static_values().data(), coordinates.static_values().size(),
        types_.get(call->type, call), locations_.get(call), &op));
    auto result = loom_op_results(op)[0];
    return value_arena_.capture(*binding.result, {&result, 1});
  }

  void pipeline_compose(cxx::CallExpressionAST* call,
                        const PipelineComposeIntrinsic& binding) {
    if (current_function_.kind != FunctionKind::Pipeline) {
      fail(call, "pipeline.compose requires a pipeline definition body");
    }
    std::vector<loom_value_id_t> specializations;
    std::vector<loom_value_id_t> bindings;
    append_staged_arguments(call->expressionList, *binding.projection,
                            /*buffer_materializations=*/{}, specializations,
                            bindings);
    loom_op_t* op;
    check(loom_pipeline_compose_build(
        &builder_, functions_.declare(binding.pipeline, call),
        specializations.data(), specializations.size(), bindings.data(),
        bindings.size(), locations_.get(call), &op));
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
    cxx::List<cxx::ExpressionAST*>* remaining = nullptr;
    // Single argument supplied by overloaded subscript syntax.
    cxx::ExpressionAST* single = nullptr;
    // Arguments recovered from a declaration initializer.
    std::span<cxx::ExpressionAST* const> range;
    // Next unevaluated argument in the initializer range.
    size_t range_index = 0;
    // Projected source element for an array copy's first reference parameter.
    // Callable analysis has already selected its storage-reference ABI.
    std::optional<Value> leading_reference;

    cxx::ExpressionAST* next() {
      if (remaining) {
        auto* argument = remaining->value;
        remaining = remaining->next;
        return argument;
      }
      if (single) {
        return std::exchange(single, nullptr);
      }
      return range_index < range.size() ? range[range_index++] : nullptr;
    }
  };

  void call_constructor(StorageProjection destination,
                        const RecordPartition& record,
                        const cxx::Type* object_type,
                        cxx::FunctionSymbol* constructor,
                        std::span<cxx::ExpressionAST* const> arguments,
                        cxx::AST* owner) override {
    emit_constructor_call(destination, record, constructor,
                          SourceArguments{.range = arguments}, owner);
  }

  const ArrayCopyInitialization* array_copy(cxx::AST* owner) override {
    return control_->array_copy(owner);
  }

  void call_copy_constructor(
      StorageProjection destination, const RecordPartition& record,
      cxx::FunctionSymbol* constructor, StorageProjection source,
      std::span<cxx::ExpressionAST* const> default_arguments,
      cxx::AST* owner) override {
    emit_constructor_call(destination, record, constructor,
                          SourceArguments{
                              .range = default_arguments,
                              .leading_reference = Value(source.pointer),
                          },
                          owner);
  }

  void emit_constructor_call(StorageProjection destination,
                             const RecordPartition& record,
                             cxx::FunctionSymbol* constructor,
                             SourceArguments arguments, cxx::AST* owner) {
    ReceiverArgument receiver = {
        ImplicitObjectABI::Storage,
        Value(destination.pointer),
        &record,
        std::nullopt,
        false,
        functions_.receiver_writes(constructor),
    };
    auto result =
        emit_ordinary_call(owner, constructor, std::move(arguments),
                           std::move(receiver), locations_.get(owner));
    IREE_ASSERT(!result.value);
  }

  std::optional<Value> emit_simple_call(cxx::AST* call,
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
        unit_.typeTraits().is_reference(signature->returnType());
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

  EvaluatedCallResult emit_evaluated_call(
      cxx::AST* call, cxx::FunctionSymbol* function, loom_symbol_ref_t callee,
      std::optional<ReceiverArgument> receiver,
      std::vector<loom_value_id_t> arguments,
      std::vector<ReferenceWriteback> writebacks,
      std::optional<ReferenceResultAlias> reference_result_alias,
      std::span<const ReferenceResultOrigin> projected_origins,
      std::vector<SelectedReferenceCandidate> selected_reference_candidates,
      loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
    const auto* receiver_projection =
        receiver && receiver->abi == ImplicitObjectABI::Value
            ? functions_.receiver_projection(function)
            : nullptr;
    if (receiver) {
      const size_t explicit_count = arguments.size();
      if (receiver_projection) {
        for (const auto& group : receiver_projection->inputs) {
          auto value =
              receiver->value
                  ? receiver->value->project(*group.partition,
                                             group.component_offset)
                  : read(project_call_target(receiver->current
                                                 ? current_receiver_target()
                                                 : *receiver->target,
                                             group, call),
                         call);
          value.append_to(arguments);
        }
      } else {
        materialize_receiver(*receiver, call).append_to(arguments);
      }
      std::rotate(arguments.begin(), arguments.begin() + explicit_count,
                  arguments.end());
    }

    const bool returns_void =
        unit_.typeTraits().is_void(signature->returnType());
    const bool returns_reference =
        unit_.typeTraits().is_reference(signature->returnType());
    const bool returns_parameter_alias = reference_result_alias.has_value();
    const bool returns_projected_reference =
        returns_reference && !projected_origins.empty();
    const bool returns_exact_projected_reference =
        returns_projected_reference && projected_origins.size() == 1;
    const bool returns_selected_projected_reference =
        returns_projected_reference && projected_origins.size() > 1;
    const bool returns_receiver_alias =
        returns_reference && receiver &&
        receiver->abi == ImplicitObjectABI::Value &&
        functions_.returns_current_receiver_reference(function);
    std::vector<const cxx::Type*> result_sources;
    const bool receiver_writeback = receiver &&
                                    receiver->abi == ImplicitObjectABI::Value &&
                                    !receiver->written_fields.empty();
    result_sources.reserve(
        (!returns_void && !returns_receiver_alias && !returns_parameter_alias &&
         !returns_exact_projected_reference) +
        (receiver ? receiver->written_fields.size() : 0) + writebacks.size());
    size_t ordinary_result_count = 0;
    if (!returns_void && !returns_receiver_alias && !returns_parameter_alias &&
        !returns_exact_projected_reference) {
      const cxx::Type* result_type = signature->returnType();
      if (returns_selected_projected_reference) {
        result_type = unit_.control()->getUnsignedIntType();
      } else if (returns_reference) {
        result_type = unit_.typeTraits().add_pointer(
            unit_.typeTraits().remove_reference(result_type));
      }
      result_sources.push_back(result_type);
      ordinary_result_count =
          types_.partition(result_type, call).component_count;
    }
    if (receiver_projection) {
      for (const auto& group : receiver_projection->outputs) {
        result_sources.push_back(group.type);
      }
    } else if (receiver_writeback) {
      for (auto* field : receiver->written_fields) {
        result_sources.push_back(
            types_.unqualified_value(source_object_type(field)));
      }
    }
    for (const auto& writeback : writebacks) {
      if (writeback.projection) {
        for (const auto& group : writeback.projection->outputs) {
          result_sources.push_back(group.type);
        }
      } else {
        result_sources.push_back(writeback.value_type);
      }
    }
    BoundSignature results;
    if (!result_sources.empty()) {
      results = bind_signature(types_, result_sources, call, &builder_);
    }
    loom_op_t* op;
    check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee, arguments.data(),
                               arguments.size(), results.types.data(),
                               results.types.size(), nullptr, 0, source, &op));

    size_t result_offset = ordinary_result_count;
    std::optional<Value> receiver_result;
    if (receiver_projection) {
      auto updated = receiver->value;
      for (const auto& group : receiver_projection->outputs) {
        const size_t count = group.partition->component_count;
        auto value = value_arena_.capture(
            *group.partition, {loom_op_results(op) + result_offset, count});
        if (receiver->current || receiver->target) {
          write(
              project_call_target(receiver->current ? current_receiver_target()
                                                    : *receiver->target,
                                  group, call),
              value, call);
        } else {
          updated =
              value_arena_.replace(*updated, group.component_offset, value);
        }
        result_offset += count;
      }
      if (updated && !receiver_projection->outputs.empty()) {
        write_receiver(*receiver, *updated, call);
        receiver_result = *updated;
      }
    } else if (receiver_writeback) {
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
      if (writeback.projection) {
        auto temporary_value = writeback.source_value;
        for (const auto& group : writeback.projection->outputs) {
          const size_t count = group.partition->component_count;
          auto updated = value_arena_.capture(
              *group.partition, {loom_op_results(op) + result_offset, count});
          if (writeback.target) {
            write(project_call_target(*writeback.target, group, call), updated,
                  call);
          } else {
            temporary_value = value_arena_.replace(
                *temporary_value, group.component_offset, updated);
          }
          result_offset += count;
        }
        if (!writeback.target) {
          update_temporary(writeback.temporary, *temporary_value,
                           writeback.value_type, call);
        }
        continue;
      }
      const size_t component_count = writeback.partition->component_count;
      auto updated = value_arena_.capture(
          *writeback.partition,
          {loom_op_results(op) + result_offset, component_count});
      if (writeback.target) {
        write(*writeback.target, updated, call);
      } else {
        update_temporary(writeback.temporary, updated, writeback.value_type,
                         call);
      }
      if (reference_result_alias && reference_result_alias->argument_offset ==
                                        writeback.argument_offset) {
        reference_result_alias->value = updated;
      }
      result_offset += component_count;
    }
    if (returns_void) {
      return {};
    }
    if (returns_receiver_alias) {
      return {
          .value = receiver_result ? *receiver_result : *receiver->value,
          .reference_target =
              receiver->current
                  ? Lvalue{ObjectLvalue{
                        unit_.typeTraits().remove_reference(
                            signature->returnType()),
                        CurrentReceiver{receivers_.current()->record}}}
                  : receiver->target,
          .temporary = receiver->temporary,
      };
    }
    if (returns_projected_reference) {
      if (selected_reference_candidates.empty()) {
        selected_reference_candidates.resize(projected_origins.size());
      } else {
        IREE_ASSERT_EQ(selected_reference_candidates.size(),
                       projected_origins.size());
      }
      for (size_t ordinal = 0; ordinal < projected_origins.size(); ++ordinal) {
        const auto& origin = projected_origins[ordinal];
        if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
          IREE_ASSERT(receiver);
          selected_reference_candidates[ordinal] = receiver_reference_candidate(
              *receiver, origin.projection, receiver_result, call);
        } else {
          IREE_ASSERT(selected_reference_candidates[ordinal].target ||
                      selected_reference_candidates[ordinal].value);
        }
      }
      if (returns_exact_projected_reference) {
        auto candidate = std::move(selected_reference_candidates.front());
        return {
            .value = std::move(candidate.value),
            .reference_target = std::move(candidate.target),
            .temporary = receiver ? receiver->temporary : std::nullopt,
        };
      }
    }
    if (reference_result_alias) {
      return {
          .value = reference_result_alias->value,
          .reference_target = std::move(reference_result_alias->target),
          .temporary = reference_result_alias->temporary,
      };
    }
    if (returns_selected_projected_reference) {
      IREE_ASSERT_EQ(ordinary_result_count, 1u);
      auto selector = loom_op_results(op)[0];
      auto* result_type =
          unit_.typeTraits().remove_reference(signature->returnType());
      const bool has_complete_targets =
          std::ranges::all_of(selected_reference_candidates,
                              [](const SelectedReferenceCandidate& candidate) {
                                return candidate.target.has_value();
                              });
      if (!has_complete_targets) {
        return {
            .value = selected_call_result_value(
                selector, selected_reference_candidates, result_type, call),
            .temporary = receiver ? receiver->temporary : std::nullopt,
        };
      }
      std::vector<Lvalue> targets;
      targets.reserve(selected_reference_candidates.size());
      for (auto& candidate : selected_reference_candidates) {
        targets.push_back(std::move(*candidate.target));
      }
      return {
          .reference_target = Lvalue{std::make_shared<const SelectedLvalue>(
              selector, result_type, std::move(targets))},
          .temporary = receiver ? receiver->temporary : std::nullopt,
      };
    }
    return {
        .value = value_arena_.capture(
            returns_reference ? kPointerPartition
                              : types_.partition(signature->returnType(), call),
            {loom_op_results(op), ordinary_result_count}),
    };
  }

  EvaluatedCallResult emit_ordinary_call(
      cxx::AST* call, cxx::FunctionSymbol* function,
      SourceArguments source_arguments,
      std::optional<ReceiverArgument> receiver, loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
    auto parameters = function->parameters();
    auto receiver_abi = receiver ? receiver->abi : ImplicitObjectABI::None;
    auto callee = functions_.declare(function, call, receiver_abi);
    auto references = functions_.reference_parameters(function);
    if (references.empty() &&
        !(receiver && receiver->target && receiver->target->selection())) {
      std::vector<loom_value_id_t> arguments;
      for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
        auto* argument = source_arguments.next();
        IREE_ASSERT(argument);
        auto value = expression(argument);
        adopt_temporary_result(argument);
        value.append_to(arguments);
      }
      while (auto* variadic_argument = source_arguments.next()) {
        expression(variadic_argument).append_to(arguments);
      }
      if (receiver) {
        auto projected_origins = functions_.projected_reference_return_origins(
            function, receiver->abi);
        return emit_evaluated_call(call, function, callee, std::move(receiver),
                                   std::move(arguments), {}, std::nullopt,
                                   projected_origins, {}, source);
      }
      return {
          .value = emit_simple_call(call, function, callee,
                                    std::move(arguments), source),
      };
    }
    std::vector<EvaluatedCallArgument> evaluated_arguments;
    evaluated_arguments.reserve(parameters.size());
    size_t reference_index = 0;
    for (size_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
      const ReferenceParameter* reference = nullptr;
      if (reference_index < references.size() &&
          references[reference_index].ordinal == ordinal) {
        reference = &references[reference_index++];
      }
      EvaluatedCallArgument evaluated;
      if (ordinal == 0 && source_arguments.leading_reference) {
        IREE_ASSERT(reference && reference->abi == ReferenceABI::Storage);
        evaluated.value = *source_arguments.leading_reference;
      } else {
        auto* argument = source_arguments.next();
        IREE_ASSERT(argument);
        evaluated = evaluate_call_argument(argument, parameters[ordinal],
                                           reference, call);
      }
      evaluated_arguments.push_back(std::move(evaluated));
    }
    std::vector<loom_value_id_t> variadic_arguments;
    while (auto* argument = source_arguments.next()) {
      expression(argument).append_to(variadic_arguments);
    }
    return emit_projected_call(call, function, callee, std::move(receiver),
                               std::move(evaluated_arguments),
                               std::move(variadic_arguments), source);
  }

  EvaluatedCallResult emit_projected_call(
      cxx::AST* call, cxx::FunctionSymbol* function, loom_symbol_ref_t callee,
      std::optional<ReceiverArgument> receiver,
      std::vector<EvaluatedCallArgument> evaluated_arguments,
      std::vector<loom_value_id_t> variadic_arguments,
      loom_location_id_t source) {
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    auto return_parameter =
        unit_.typeTraits().is_reference(signature->returnType())
            ? functions_.reference_return_parameter(function)
            : std::nullopt;
    const size_t receiver_argument_count =
        receiver_input_count(function, receiver);
    std::vector<loom_value_id_t> arguments;
    std::vector<ReferenceWriteback> writebacks;
    std::optional<ReferenceResultAlias> reference_result_alias;
    for (auto& evaluated : evaluated_arguments) {
      materialize_call_argument(evaluated, call);
    }
    std::vector<SelectedReferenceCandidate> selected_reference_candidates;
    auto projected_origins = functions_.projected_reference_return_origins(
        function, receiver ? receiver->abi : ImplicitObjectABI::None);
    if (!projected_origins.empty()) {
      auto parameters = function->parameters();
      auto references = functions_.reference_parameters(function);
      std::vector<const ReferenceParameter*> references_by_ordinal(
          parameters.size());
      for (const auto& reference : references) {
        references_by_ordinal[reference.ordinal] = &reference;
      }
      selected_reference_candidates.reserve(projected_origins.size());
      for (const auto& origin : projected_origins) {
        if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
          IREE_ASSERT(receiver);
          selected_reference_candidates.push_back(receiver_reference_candidate(
              *receiver, origin.projection, std::nullopt, call));
          continue;
        }
        const auto* reference = references_by_ordinal[origin.owner_ordinal];
        IREE_ASSERT(reference);
        auto candidate = selected_call_argument_candidate(
            evaluated_arguments[origin.owner_ordinal],
            parameters[origin.owner_ordinal], *reference, call);
        selected_reference_candidates.push_back(project_reference_candidate(
            std::move(candidate), origin.projection, call));
      }
    }
    for (size_t ordinal = 0; ordinal < evaluated_arguments.size(); ++ordinal) {
      auto& evaluated = evaluated_arguments[ordinal];
      if (return_parameter == ordinal) {
        reference_result_alias = {*evaluated.value, evaluated.target,
                                  receiver_argument_count + arguments.size(),
                                  evaluated.temporary};
      }
      append_call_argument(std::move(evaluated), receiver_argument_count,
                           arguments, writebacks, call);
    }
    arguments.insert(arguments.end(), variadic_arguments.begin(),
                     variadic_arguments.end());
    return emit_evaluated_call(
        call, function, callee, std::move(receiver), std::move(arguments),
        std::move(writebacks), std::move(reference_result_alias),
        projected_origins, std::move(selected_reference_candidates), source);
  }

  EvaluatedCallResult ordinary_operator_call(cxx::ExpressionAST* expression,
                                             const DirectOperatorCall& call) {
    admit_ordinary_operator(call, expression);
    auto* function = call.function;
    auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
    IREE_ASSERT(signature);
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
        evaluated_arguments[ordinal]->value->append_to(arguments);
      }
      return {
          .value = emit_simple_call(expression, function, callee,
                                    std::move(arguments),
                                    locations_.get(expression)),
      };
    }

    const size_t receiver_argument_count =
        receiver_input_count(function, receiver);
    std::vector<loom_value_id_t> arguments;
    std::vector<ReferenceWriteback> writebacks;
    std::optional<ReferenceResultAlias> reference_result_alias;
    auto return_parameter =
        unit_.typeTraits().is_reference(signature->returnType())
            ? functions_.reference_return_parameter(function)
            : std::nullopt;
    writebacks.reserve(references.size());
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      IREE_ASSERT(evaluated_arguments[ordinal]);
      materialize_call_argument(*evaluated_arguments[ordinal], expression);
    }
    std::vector<SelectedReferenceCandidate> selected_reference_candidates;
    auto projected_origins =
        functions_.projected_reference_return_origins(function, receiver_abi);
    selected_reference_candidates.reserve(projected_origins.size());
    for (const auto& origin : projected_origins) {
      if (origin.owner_ordinal == ReferenceResultOrigin::kReceiver) {
        IREE_ASSERT(receiver);
        selected_reference_candidates.push_back(receiver_reference_candidate(
            *receiver, origin.projection, std::nullopt, expression));
        continue;
      }
      IREE_ASSERT(evaluated_arguments[origin.owner_ordinal]);
      const auto* reference = reference_by_ordinal[origin.owner_ordinal];
      IREE_ASSERT(reference);
      auto candidate = selected_call_argument_candidate(
          *evaluated_arguments[origin.owner_ordinal],
          parameters[origin.owner_ordinal], *reference, expression);
      selected_reference_candidates.push_back(project_reference_candidate(
          std::move(candidate), origin.projection, expression));
    }
    for (size_t ordinal = 0; ordinal < parameter_count; ++ordinal) {
      IREE_ASSERT(evaluated_arguments[ordinal]);
      auto evaluated = std::move(*evaluated_arguments[ordinal]);
      if (return_parameter == ordinal) {
        reference_result_alias = {*evaluated.value, evaluated.target,
                                  receiver_argument_count + arguments.size()};
      }
      append_call_argument(std::move(evaluated), receiver_argument_count,
                           arguments, writebacks, expression);
    }
    return emit_evaluated_call(
        expression, function, callee, std::move(receiver), std::move(arguments),
        std::move(writebacks), std::move(reference_result_alias),
        projected_origins, std::move(selected_reference_candidates),
        locations_.get(expression));
  }

  EvaluatedCallResult ordinary_operator_call(cxx::ExpressionAST* expression) {
    auto call = direct_operator_call(expression);
    IREE_ASSERT(call);
    return ordinary_operator_call(expression, *call);
  }

  Value ordinary_operator_value(cxx::ExpressionAST* expression,
                                const DirectOperatorCall& call) {
    auto called = ordinary_operator_call(expression, call);
    auto* signature = cxx::type_cast<cxx::FunctionType>(call.function->type());
    IREE_ASSERT(signature);
    if (called.reference_target) {
      return read(*called.reference_target, expression);
    }
    if (!called.value) {
      fail(expression, "void operator cannot be used as a value");
    }
    if (!unit_.typeTraits().is_reference(signature->returnType())) {
      return *called.value;
    }
    if (returns_value_reference(call.function, *called.value)) {
      return *called.value;
    }
    if (!called.value->is_pointer()) {
      fail(expression, "reference-returning operator lost its object identity");
    }
    auto projection = storage_.project(
        called.value->pointer(),
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
    if (assignment->symbol &&
        !types_.is_trivial_value_operation(assignment->symbol)) {
      auto call = direct_operator_call(ast);
      return ordinary_operator_value(ast, *call);
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
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
    }
    auto initial = values_;
    auto source = locations_.get(ast);
    auto outputs = bind_signature(types_, sources, ast, &builder_);
    loom_op_t* op;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0, source, &op));
    conditional_cleanup_anchors_.push_back(
        {op->parent_block, op->parent_op, op});
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_then_region(op));
    size_t temporary_mark = temporary_destructions_.size();
    std::vector<loom_value_id_t> yielded;
    if constexpr (has_value) {
      auto value = then_value();
      value.append_to(yielded);
    } else {
      then_value();
    }
    materialize_conditional_temporaries(temporary_mark);
    auto mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    loom_op_t* yield;
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    values_ = initial;
    saved =
        loom_builder_enter_region(&builder_, op, loom_scf_if_else_region(op));
    temporary_mark = temporary_destructions_.size();
    yielded.clear();
    if constexpr (has_value) {
      auto value = else_value();
      value.append_to(yielded);
    } else {
      else_value();
    }
    materialize_conditional_temporaries(temporary_mark);
    mutations = current(written);
    yielded.insert(yielded.end(), mutations.begin(), mutations.end());
    check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                               source, &yield));
    loom_builder_restore(&builder_, saved);
    conditional_cleanup_anchors_.pop_back();
    values_ = std::move(initial);
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
    std::vector<Mutation> written;
    // Binding availability restored before entering either continuation.
    ValueBindings initial;
    // Structured conditional receiving the two source arm values.
    loom_op_t* operation;
    // Insertion point enclosing |operation|.
    loom_builder_ip_t enclosing;
    // Leading operation results occupied by the conditional value.
    iree_host_size_t value_count;
    // Temporary prefix preceding the arm currently under construction.
    size_t temporary_mark;
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
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
    }
    auto initial = values_;
    auto outputs = bind_signature(types_, sources, expression, &builder_);
    loom_op_t* operation;
    check(loom_scf_if_build(&builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                            condition, outputs.types.data(),
                            outputs.types.size(), nullptr, 0,
                            locations_.get(expression), &operation));
    conditional_cleanup_anchors_.push_back(
        {operation->parent_block, operation->parent_op, operation});
    auto enclosing = loom_builder_enter_region(
        &builder_, operation, loom_scf_if_then_region(operation));
    return {expression,
            std::move(written),
            std::move(initial),
            operation,
            enclosing,
            value_count,
            temporary_destructions_.size(),
            true};
  }

  void yield_conditional_value(const ConditionalValueFrame& frame,
                               Value value) {
    auto* arm = frame.building_then ? frame.expression->iftrueExpression
                                    : frame.expression->iffalseExpression;
    adopt_temporary_result(arm);
    materialize_conditional_temporaries(frame.temporary_mark);
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
    IREE_ASSERT(!conditional_cleanup_anchors_.empty());
    IREE_ASSERT_EQ(conditional_cleanup_anchors_.back().before_op,
                   frame.operation);
    conditional_cleanup_anchors_.pop_back();
    values_ = frame.initial;
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
          [&] {
            auto value = this->expression(expression->iftrueExpression);
            adopt_temporary_result(expression->iftrueExpression);
            return value;
          },
          [&] {
            auto value = this->expression(expression->iffalseExpression);
            adopt_temporary_result(expression->iffalseExpression);
            return value;
          });
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
        values_ = frame.initial;
        loom_builder_enter_region(&builder_, frame.operation,
                                  loom_scf_if_else_region(frame.operation));
        frame.building_then = false;
        frame.temporary_mark = temporary_destructions_.size();
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

  std::optional<IntrinsicCallResult> multiprecision_arithmetic_builtin(
      cxx::CallExpressionAST* call) {
    auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
    using Builtin = cxx::BuiltinFunctionKind;
    bool addition;
    switch (cxx::resolveBuiltinFunctionKind(callee)) {
      case Builtin::T___BUILTIN_ADDCB:
      case Builtin::T___BUILTIN_ADDCS:
      case Builtin::T___BUILTIN_ADDC:
      case Builtin::T___BUILTIN_ADDCL:
      case Builtin::T___BUILTIN_ADDCLL:
        addition = true;
        break;
      case Builtin::T___BUILTIN_SUBCB:
      case Builtin::T___BUILTIN_SUBCS:
      case Builtin::T___BUILTIN_SUBC:
      case Builtin::T___BUILTIN_SUBCL:
      case Builtin::T___BUILTIN_SUBCLL:
        addition = false;
        break;
      default:
        return std::nullopt;
    }

    std::array<Value, 4> arguments;
    auto* argument = call->expressionList;
    for (auto& value : arguments) {
      value = expression(argument->value);
      argument = argument->next;
    }

    const loom_type_t type = types_.get(call->type, call);
    const loom_location_id_t location = locations_.get(call);
    auto build =
        addition ? loom_scalar_addcarryui_build : loom_scalar_subborrowui_build;
    loom_op_t* first_op = nullptr;
    check(build(&builder_, arguments[0].ssa(), arguments[1].ssa(), type,
                location, &first_op));
    loom_op_t* second_op = nullptr;
    // Carry/borrow input is a full-width value rather than an i1. Each
    // addition or subtraction can independently contribute to the output bit.
    check(build(&builder_, loom_op_results(first_op)[0], arguments[2].ssa(),
                type, location, &second_op));

    const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_op_t* flag_op = nullptr;
    check(loom_scalar_ori_build(&builder_, loom_op_results(first_op)[1],
                                loom_op_results(second_op)[1], i1_type,
                                location, &flag_op));
    loom_op_t* extended_flag_op = nullptr;
    check(loom_scalar_extui_build(&builder_, loom_op_results(flag_op)[0],
                                  i1_type, type, location, &extended_flag_op));

    auto* output_expression = call->expressionList->next->next->next->value;
    auto* output_type =
        cxx::type_cast<cxx::PointerType>(output_expression->type)
            ->elementType();
    auto output = storage_.dereference(
        storage_.project(arguments[3].pointer(), output_type, call),
        output_type, call);
    storage_.store(output, loom_op_results(extended_flag_op)[0], output_type,
                   call);
    return IntrinsicCallResult{Value(loom_op_results(second_op)[0])};
  }

  std::optional<IntrinsicCallResult> builtin(cxx::CallExpressionAST* call) {
    if (cxx::resolveBuiltinFunctionKind(
            cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression)) ==
        cxx::BuiltinFunctionKind::T___BUILTIN_UNREACHABLE) {
      // This path has no defined continuation or runtime effect. The retained
      // source exit summary stops the sequence and supplies func.unreachable
      // when the enclosing callable cannot reach an ordinary return.
      return IntrinsicCallResult{std::nullopt};
    }
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
    if (auto arithmetic = multiprecision_arithmetic_builtin(call)) {
      return arithmetic;
    }
    return std::nullopt;
  }

  Value constant(const cxx::ConstValue& value, cxx::ExpressionAST* ast) {
    return initialization_.constant(value, ast->type, ast);
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
    const size_t evaluation_mark = temporary_destructions_.size();
    auto value = expression_impl(ast);
    return retain_temporary(ast, value, evaluation_mark);
  }

  void adopt(cxx::ExpressionAST* expression) override {
    adopt_temporary_result(expression);
  }

  Value expression_impl(cxx::ExpressionAST* ast) {
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
    if (auto* lambda = cxx::ast_cast<cxx::LambdaExpressionAST>(ast)) {
      return initialization_.closure(lambda);
    }
    if (auto* equal = cxx::ast_cast<cxx::EqualInitializerAST>(ast)) {
      return expression(equal->expression);
    }
    if (auto* initializer =
            cxx::ast_cast<cxx::DefaultInitializerExpressionAST>(ast)) {
      return expression(initializer->expression);
    }
    if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast)) {
      if (is_class_adjustment(cast)) {
        if (cast->castKind == cxx::ImplicitCastKind::kBaseToDerivedConversion) {
          return load(cast);
        }
        const auto& selected = types_.projection(cast->subobjectPath, cast);
        if (is_current_object(unit_, cast->expression)) {
          return selected_receiver_value(selected, cast->type, cast);
        }
        return expression(cast->expression)
            .project(*selected.partition, selected.component_offset);
      }
      if (cast->conversionFunction &&
          !cast->conversionFunction->isConstructor() &&
          types_.record(implicit_object_type(unit_, cast->conversionFunction),
                        cast)) {
        auto called = ordinary_operator_call(ast, *direct_operator_call(ast));
        if (called.reference_target) {
          return read(*called.reference_target, ast);
        }
        if (!called.value) {
          fail(ast, "conversion function must return a value");
        }
        return *called.value;
      }
      if (cast->conversionFunction &&
          cast->conversionFunction->isConstructor() &&
          types_.record_constructor_requires_call(
              cast->type, cast->conversionFunction, cast)) {
        return initialization_.value(cast->type, cast->expression,
                                     cast->conversionFunction, cast);
      }
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
      if (explicit_reference_operand(unit_, cast)) {
        auto value = expression(cast->expression);
        if (&value.partition() != &types_.partition(cast->type, cast)) {
          fail(cast,
               "reference casts must preserve the admitted source object type");
        }
        return value;
      }
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
      if (receivers_.current()) {
        auto* closure = receivers_.current()->record->source;
        auto* captured_this =
            closure->isClosureType() ? closure->capturedThisField() : nullptr;
        if (captured_this) {
          return receiver_member(captured_this, ast);
        }
      }
      if (!receivers_.current() || !receivers_.current()->storage) {
        fail(ast, "this pointer requires a storage-backed receiver");
      }
      return Value(receivers_.current()->storage->pointer);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = references_.find(id->symbol);
          found != references_.end()) {
        return name(read(found->second, ast),
                    cxx::to_string(id->symbol->name()));
      }
      auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol);
      auto receiver_field =
          field ? current_receiver_selection(id) : std::nullopt;
      if (receiver_field &&
          unit_.typeTraits().is_lvalue_reference(field->type())) {
        if (!types_.pointer_reference_is_snapshot(field)) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        return name(
            selected_receiver_value(*receiver_field, field->type(), ast),
            cxx::to_string(field->name()));
      }
      if (receiver_field && id->subobjectPath) {
        return name(
            selected_receiver_value(*receiver_field, field->type(), ast),
            cxx::to_string(field->name()));
      }
      if (auto found = values_.find(id->symbol); found != values_.end()) {
        return found->second;
      }
      if (values_.contains(id->symbol)) {
        fail(ast, "source value is read before initialization");
      }
      if (locals_.contains(id->symbol)) {
        return binding(id->symbol, ast);
      }
      if (receiver_field) {
        return name(
            selected_receiver_value(*receiver_field, field->type(), ast),
            cxx::to_string(field->name()));
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
      if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(id->symbol)) {
        if (auto captured =
                captured_object_storage(field, id->subobjectPath, ast)) {
          return name(objects_.load(*captured, ast->type, ast),
                      cxx::to_string(field->name()));
        }
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
      if (member->vectorComponentKind != cxx::VectorComponentKind::kNone) {
        return vector_component_value(member);
      }
      if (auto selected = current_receiver_selection(member)) {
        auto* field = static_cast<cxx::FieldSymbol*>(member->symbol);
        if (unit_.typeTraits().is_lvalue_reference(field->type()) &&
            !types_.pointer_reference_is_snapshot(field)) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        return name(selected_receiver_value(*selected, field->type(), ast),
                    cxx::to_string(field->name()));
      }
      auto* base = cxx::ast_cast<cxx::IdExpressionAST>(member->baseExpression);
      if (is_semantic_array_element(member->baseExpression)) {
        return name(read(destination(member, ast, LvalueAccess::Read), ast),
                    cxx::to_string(member->symbol->name()));
      }
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
        if (unit_.typeTraits().is_lvalue_reference(field->type()) &&
            !types_.pointer_reference_is_snapshot(field)) {
          return name(load(ast), cxx::to_string(field->name()));
        }
        if (auto destination = control_->destination(member);
            destination && values_.contains(destination->binding) &&
            !locals_.contains(destination->binding)) {
          return read(ObjectLvalue{member->type, *destination}, member);
        }
        auto value = expression(member->baseExpression);
        auto slice = types_.selection(field, member->subobjectPath, ast);
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
        auto* signature =
            cxx::type_cast<cxx::FunctionType>(subscript->symbol->type());
        if (signature &&
            unit_.typeTraits().is_reference(signature->returnType())) {
          if (called.reference_target) {
            return read(*called.reference_target, ast);
          }
          if (!called.value) {
            fail(ast, "reference-returning subscript produced no value");
          }
          if (returns_value_reference(subscript->symbol, *called.value)) {
            return *called.value;
          }
          if (!called.value->is_pointer()) {
            fail(ast, "reference-returning subscript lost its object identity");
          }
          auto projection = storage_.project(
              called.value->pointer(),
              unit_.typeTraits().remove_reference(signature->returnType()),
              ast);
          if (unit_.typeTraits().is_class(ast->type)) {
            return objects_.load(projection, ast->type, ast);
          }
          return storage_.load(storage_.dereference(projection, ast->type, ast),
                               ast->type, ast);
        }
        if (!called.value) {
          fail(ast, "void overloaded subscript cannot be used as a value");
        }
        return *called.value;
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
      if (auto target =
              semantic_array_element(subscript, ast, LvalueAccess::Read)) {
        return read(*target, ast);
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
          fail(ast, "pipelines require explicit composition or kernel launch");
        }
        auto called = ordinary_call(call, function, source);
        auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
        if (signature &&
            unit_.typeTraits().is_reference(signature->returnType())) {
          if (called.reference_target) {
            return read(*called.reference_target, ast);
          }
          if (!called.value) {
            fail(ast, "reference-returning function produced no value");
          }
          if (returns_value_reference(function, *called.value)) {
            return *called.value;
          }
          if (!called.value->is_pointer()) {
            fail(ast, "reference-returning call lost its object identity");
          }
          auto projection = storage_.project(
              called.value->pointer(),
              unit_.typeTraits().remove_reference(signature->returnType()),
              ast);
          if (unit_.typeTraits().is_class(ast->type)) {
            return objects_.load(projection, ast->type, ast);
          }
          return storage_.load(storage_.dereference(projection, ast->type, ast),
                               ast->type, ast);
        }
        if (!called.value) {
          fail(ast, "void function cannot be used as a value");
        }
        return *called.value;
      }
      if (auto* parameter = std::get_if<CommandParameterIntrinsic>(binding)) {
        return command_parameter(call, *parameter);
      }
      if (auto* memory = std::get_if<PipelineMemoryIntrinsic>(binding)) {
        return pipeline_memory(call, *memory);
      }
      if (std::holds_alternative<KernelLaunchIntrinsic>(*binding) ||
          std::holds_alternative<CommandLaunchIntrinsic>(*binding) ||
          std::holds_alternative<PipelineComposeIntrinsic>(*binding)) {
        fail(ast, "void graph operation cannot be used as a value");
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
    auto outer_destructions = active_destructions_;
    std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
    if (tracks_break) {
      sources.push_back(unit_.control()->getBoolType());
    }
    const size_t control_count = sources.size();
    sources.reserve(control_count + written.size());
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
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
    active_destructions_ = outer_destructions;
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
    active_destructions_ = std::move(outer_destructions);
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
    return iteration_sequence(owner, remaining, fastmath, tracks_break,
                              mutations, active_destructions_.size());
  }

  IterationFlow iteration_sequence(cxx::AST* owner, StatementSequence remaining,
                                   uint8_t fastmath, bool tracks_break,
                                   std::span<cxx::Symbol* const> mutations,
                                   size_t completion_mark) {
    auto outer_destructions = active_destructions_;
    const size_t lexical_mark = active_destructions_.size();
    auto flow = iteration_sequence(owner, remaining, fastmath, tracks_break,
                                   mutations, lexical_mark, completion_mark);
    active_destructions_ = std::move(outer_destructions);
    return flow;
  }

  IterationFlow iteration_sequence(cxx::AST* owner, StatementSequence remaining,
                                   uint8_t fastmath, bool tracks_break,
                                   std::span<cxx::Symbol* const> mutations,
                                   size_t lexical_mark,
                                   size_t completion_mark) {
    FastMathScope math_scope(lexical_fastmath_flags_, fastmath);
    while (!remaining.empty()) {
      auto* child = remaining.take();
      auto flow = iteration_exit_flow(child, tracks_break);
      if (flow == ExitFlow::None) {
        statement(child);
        if (!control_->falls_through(child)) {
          active_destructions_.resize(lexical_mark);
          return constant_iteration_flow(false, false, tracks_break, child);
        }
        continue;
      }
      return iteration_tail(owner, child, remaining, fastmath, tracks_break,
                            mutations, lexical_mark, completion_mark);
    }
    close_destruction_scope(completion_mark);
    return constant_iteration_flow(true, true, tracks_break, owner);
  }

  IterationFlow iteration_scoped_tail(cxx::AST* owner, cxx::StatementAST* ast,
                                      StatementSequence remaining,
                                      uint8_t continuation_fastmath,
                                      bool tracks_break,
                                      std::span<cxx::Symbol* const> mutations,
                                      size_t scope_mark, size_t lexical_mark,
                                      size_t completion_mark) {
    auto flow = iteration_exit_flow(ast, tracks_break);
    if (flow == ExitFlow::None) {
      if (ast) {
        statement(ast);
      }
      if (ast && !control_->falls_through(ast)) {
        return constant_iteration_flow(false, false, tracks_break, ast);
      }
      close_destruction_scope(scope_mark);
      return iteration_sequence(owner, remaining, continuation_fastmath,
                                tracks_break, mutations, lexical_mark,
                                completion_mark);
    }
    if (flow == ExitFlow::All) {
      return iteration_statement(ast, tracks_break);
    }
    auto fallthrough = iteration_statement(ast, tracks_break, scope_mark);
    return iteration_branch(
        owner, fallthrough.reaches_tail, tracks_break, mutations,
        [&] {
          active_destructions_.resize(scope_mark);
          return iteration_sequence(owner, remaining, continuation_fastmath,
                                    tracks_break, mutations, lexical_mark,
                                    completion_mark);
        },
        [&] {
          auto stopped =
              constant_iteration_flow(false, false, tracks_break, ast);
          stopped.reaches_latch = fallthrough.reaches_latch;
          return stopped;
        });
  }

  // A tail enters at most one arm directly. When both arms can reach it, join
  // their fallthrough values first and emit the shared source just once.
  IterationFlow iteration_tail(cxx::AST* owner, cxx::StatementAST* ast,
                               StatementSequence remaining,
                               uint8_t continuation_fastmath, bool tracks_break,
                               std::span<cxx::Symbol* const> mutations,
                               size_t lexical_mark, size_t completion_mark) {
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
                                tracks_break, mutations, lexical_mark,
                                completion_mark);
    }
    if (flow == ExitFlow::All) {
      return iteration_statement(ast, tracks_break);
    }
    if (remaining.empty()) {
      return iteration_statement(ast, tracks_break, completion_mark);
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && (iteration_exit_flow(branch->statement, tracks_break) ==
                       ExitFlow::All ||
                   iteration_exit_flow(branch->elseStatement, tracks_break) ==
                       ExitFlow::All)) {
      const size_t condition_mark = active_destructions_.size();
      auto condition = branch_condition(branch);
      return iteration_branch(
          owner, condition, tracks_break, mutations,
          [&] {
            return iteration_scoped_tail(owner, branch->statement, remaining,
                                         continuation_fastmath, tracks_break,
                                         mutations, condition_mark,
                                         lexical_mark, completion_mark);
          },
          [&] {
            return iteration_scoped_tail(
                owner, branch->elseStatement, remaining, continuation_fastmath,
                tracks_break, mutations, condition_mark, lexical_mark,
                completion_mark);
          });
    }
    auto fallthrough = iteration_statement(ast, tracks_break);
    return iteration_branch(
        owner, fallthrough.reaches_tail, tracks_break, mutations,
        [&] {
          return iteration_sequence(owner, remaining, continuation_fastmath,
                                    tracks_break, mutations, lexical_mark,
                                    completion_mark);
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
    auto mark = begin_full_expression();
    initialize_condition(selection->condition);
    if (condition_declaration(selection->condition)) {
      finish_full_expression(mark);
      mark = begin_full_expression();
    }
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
    auto result_value = name(result(op), "switch_selector");
    finish_full_expression(mark);
    return result_value;
  }

  IterationFlow switch_clause_flow(const SwitchClause& clause,
                                   size_t completion_mark) {
    return iteration_sequence(clause.labels.front().source,
                              StatementSequence(clause.statements),
                              lexical_fastmath_flags_, /*tracks_break=*/true,
                              clause.writes, completion_mark);
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
                              bool tracks_outer_break, size_t completion_mark) {
    const bool tracks_continue =
        control_->continues(selection) != ExitFlow::None;
    auto written = live_mutations(selection);
    std::vector<const cxx::Type*> sources;
    if (tracks_continue) {
      sources.push_back(unit_.control()->getBoolType());
    }
    const size_t control_count = sources.size();
    sources.reserve(control_count + written.size());
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
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
    auto outer_destructions = active_destructions_;
    auto yield_clause = [&](const SwitchClause* clause, loom_region_t* region) {
      auto* clause_source = clause ? clause->labels.front().source : selection;
      auto saved = loom_builder_enter_region(&builder_, op, region);
      values_ = outer_values;
      active_destructions_ = outer_destructions;
      auto flow = [&] {
        if (clause) {
          return switch_clause_flow(*clause, completion_mark);
        }
        close_destruction_scope(completion_mark);
        return constant_iteration_flow(true, true, /*tracks_break=*/true,
                                       selection);
      }();
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
    active_destructions_ = std::move(outer_destructions);
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
      for (const auto& mutation : written) {
        sources.push_back(mutation.projection.type);
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
      auto flow = switch_clause_flow(clause, active_destructions_.size());
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
    const size_t destruction_mark = active_destructions_.size();
    auto selector = switch_selector(selection);
    const auto& layout = control_->switch_layout(selection);
    break_destruction_marks_.push_back(
        layout.direct ? destruction_mark : active_destructions_.size());
    auto flow =
        layout.direct
            ? direct_switch(selection, selector, layout, tracks_outer_break,
                            destruction_mark)
            : linear_switch(selection, selector, layout, tracks_outer_break);
    break_destruction_marks_.pop_back();
    if (layout.direct) {
      active_destructions_.resize(destruction_mark);
    } else if (control_->continues(selection) != ExitFlow::None) {
      close_destruction_scope_if(flow.reaches_tail, destruction_mark,
                                 selection);
    } else {
      close_destruction_scope(destruction_mark);
    }
    if (auto* decision = condition_declaration(selection->condition)) {
      values_.erase(decision->symbol);
      locals_.erase(decision->symbol);
    }
    return flow;
  }

  IterationFlow iteration_statement(
      cxx::StatementAST* ast, bool tracks_break,
      std::optional<size_t> completion_mark = std::nullopt) {
    auto* source = ast;
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    if (!ast) {
      if (completion_mark) {
        close_destruction_scope(*completion_mark);
      }
      return constant_iteration_flow(true, true, tracks_break, source);
    }
    if (iteration_exit_flow(ast, tracks_break) == ExitFlow::None) {
      statement(ast);
      auto falls_through = control_->falls_through(ast);
      if (falls_through && completion_mark) {
        close_destruction_scope(*completion_mark);
      }
      return constant_iteration_flow(falls_through, falls_through, tracks_break,
                                     ast);
    }
    if (cxx::ast_cast<cxx::ContinueStatementAST>(ast)) {
      IREE_ASSERT(!continue_destruction_marks_.empty());
      emit_destructions(continue_destruction_marks_.back());
      return constant_iteration_flow(false, true, tracks_break, ast);
    }
    if (cxx::ast_cast<cxx::BreakStatementAST>(ast)) {
      IREE_ASSERT(!break_destruction_marks_.empty());
      emit_destructions(break_destruction_marks_.back());
      return constant_iteration_flow(false, false, tracks_break, ast);
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return completion_mark
                 ? iteration_sequence(
                       compound, StatementSequence(compound->statementList),
                       lexical_fastmath_flags_, tracks_break,
                       control_->written(compound), *completion_mark)
                 : iteration_sequence(
                       compound, StatementSequence(compound->statementList),
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
    const size_t condition_mark = active_destructions_.size();
    auto condition = branch_condition(branch);
    const size_t arm_completion_mark = completion_mark.value_or(condition_mark);
    auto flow = iteration_branch(
        ast, condition, tracks_break, control_->written(ast),
        [&] {
          return iteration_statement(branch->statement, tracks_break,
                                     arm_completion_mark);
        },
        [&] {
          if (branch->elseStatement) {
            return iteration_statement(branch->elseStatement, tracks_break,
                                       arm_completion_mark);
          }
          close_destruction_scope(arm_completion_mark);
          return constant_iteration_flow(true, true, tracks_break, ast);
        });
    active_destructions_.resize(condition_mark);
    return flow;
  }

  // Iteration transfers end only this body. The caller decides whether the
  // flow reaches the for step or do condition. Break-free bodies preserve
  // their existing structured IR and carry no latch value.
  IterationFlow loop_body(cxx::StatementAST* body, bool tracks_break = false) {
    const size_t destruction_mark = active_destructions_.size();
    break_destruction_marks_.push_back(destruction_mark);
    continue_destruction_marks_.push_back(destruction_mark);
    IterationFlow flow;
    if (iteration_exit_flow(body, tracks_break) == ExitFlow::None) {
      statement(body);
      if (control_->falls_through(body)) {
        close_destruction_scope(destruction_mark);
      } else {
        active_destructions_.resize(destruction_mark);
      }
      flow = tracks_break
                 ? constant_iteration_flow(true, true, tracks_break, body)
                 : IterationFlow{LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
    } else {
      flow = iteration_statement(body, tracks_break);
    }
    active_destructions_.resize(destruction_mark);
    continue_destruction_marks_.pop_back();
    break_destruction_marks_.pop_back();
    return flow;
  }

  void schedule_region(const ScheduleRegion& schedule) {
    const bool pipeline = current_function_.kind == FunctionKind::Pipeline;
    if (!pipeline && current_function_.kind != FunctionKind::CommandProgram) {
      fail(schedule.body,
           "schedule regions require a command_program or pipeline body");
    }
    if (pipeline && pipeline_strand_depth_) {
      fail(schedule.body, "pipeline schedule regions belong outside strands");
    }
    if (!live_mutations(schedule.body).empty()) {
      fail(schedule.body,
           "schedule regions cannot update enclosing automatic "
           "values because they have no results");
    }

    loom_op_t* op;
    loom_region_t* region;
    if (pipeline && schedule.kind == ScheduleRegion::Kind::Serial) {
      check(loom_pipeline_serial_build(&builder_, locations_.get(schedule.body),
                                       &op));
      region = loom_pipeline_serial_body(op);
    } else if (pipeline) {
      check(loom_pipeline_concurrent_build(&builder_,
                                           locations_.get(schedule.body), &op));
      region = loom_pipeline_concurrent_body(op);
    } else if (schedule.kind == ScheduleRegion::Kind::Serial) {
      check(loom_command_serial_build(&builder_, locations_.get(schedule.body),
                                      &op));
      region = loom_command_serial_body(op);
    } else {
      check(loom_command_concurrent_build(&builder_,
                                          locations_.get(schedule.body), &op));
      region = loom_command_concurrent_body(op);
    }
    auto saved = loom_builder_enter_region(&builder_, op, region);
    auto outer_values = values_;
    for (auto* child : cxx::ListView{schedule.body->statementList}) {
      statement(child);
      if (!control_->falls_through(child)) {
        break;
      }
    }
    loom_op_t* terminator;
    if (pipeline) {
      check(loom_pipeline_yield_build(&builder_, locations_.get(schedule.body),
                                      &terminator));
    } else {
      check(loom_command_yield_build(&builder_, locations_.get(schedule.body),
                                     &terminator));
    }
    values_ = std::move(outer_values);
    loom_builder_restore(&builder_, saved);
  }

  void pipeline_strand(const PipelineStrand& strand) {
    if (current_function_.kind != FunctionKind::Pipeline) {
      fail(strand.source, "strand requires a pipeline definition body");
    }
    if (!live_mutations(strand.body).empty()) {
      fail(strand.body,
           "pipeline strands cannot update enclosing automatic values "
           "because they have no results");
    }

    auto origins = dynamic_indices(strand.origins);
    auto counts = dynamic_indices(strand.counts);
    auto strides = dynamic_indices(strand.strides);
    auto target = strand.target ? target_definitions_.reference(
                                      strand.target, strand.target_source)
                                : loom_symbol_ref_null();
    loom_op_t* op;
    check(loom_pipeline_strand_build(
        &builder_,
        loom_symbol_ref_is_valid(target)
            ? LOOM_PIPELINE_STRAND_BUILD_FLAG_HAS_TARGET
            : 0,
        0, target, origins.data(), origins.size(),
        strand.origins.static_values().data(),
        strand.origins.static_values().size(), counts.data(), counts.size(),
        strand.counts.static_values().data(),
        strand.counts.static_values().size(), strides.data(), strides.size(),
        strand.strides.static_values().data(),
        strand.strides.static_values().size(), locations_.get(strand.source),
        &op));
    auto saved =
        loom_builder_enter_region(&builder_, op, loom_pipeline_strand_body(op));
    auto outer_values = values_;
    ++pipeline_strand_depth_;
    for (auto* child : cxx::ListView{strand.body->statementList}) {
      statement(child);
      if (!control_->falls_through(child)) {
        break;
      }
    }
    loom_op_t* terminator;
    check(loom_pipeline_end_build(&builder_, locations_.get(strand.body),
                                  &terminator));
    --pipeline_strand_depth_;
    values_ = std::move(outer_values);
    loom_builder_restore(&builder_, saved);
  }

  void validate_statement_attributes(cxx::StatementAST* statement) {
    const bool loop = cxx::ast_cast<cxx::ForStatementAST>(statement) ||
                      cxx::ast_cast<cxx::WhileStatementAST>(statement) ||
                      cxx::ast_cast<cxx::DoStatementAST>(statement);
    const bool declaration =
        cxx::ast_cast<cxx::DeclarationStatementAST>(statement) != nullptr;
    cxx::AttributeAST* region = nullptr;
    cxx::AttributeAST* strand = nullptr;
    cxx::AttributeAST* target = nullptr;
    visit_loom_attributes(
        unit_, statement_attributes(statement),
        [&](std::string_view name, cxx::AttributeAST* attribute) {
          if (name == "fastmath" || loop) {
            return;
          }
          if (declaration && (name == "constexpr" || name == "workgroup")) {
            return;
          }
          if (name == "serial" || name == "concurrent" || name == "strand") {
            if (region) {
              diagnostics_.reject(
                  unit_, attribute,
                  "one command or pipeline region is allowed per statement");
            }
            region = attribute;
            if (name == "strand") {
              strand = attribute;
            }
            return;
          }
          if (name == "target") {
            if (target) {
              diagnostics_.reject(unit_, attribute,
                                  "one target is allowed per statement");
            }
            target = attribute;
            return;
          }
          diagnostics_.reject(unit_, attribute,
                              "unsupported Loom statement attribute");
        });
    if (target && !strand) {
      diagnostics_.reject(unit_, target,
                          "statement target requires a pipeline strand");
    }
  }

  void statement(cxx::StatementAST* ast) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    if (ast) {
      validate_statement_attributes(ast);
    }
    const size_t selection_mark = active_destructions_.size();
    auto* selected = selected_statement(ast);
    if (selected != ast) {
      if (selected) {
        statement(selected);
      }
      if (!selected || control_->falls_through(selected)) {
        close_destruction_scope(selection_mark);
      } else {
        active_destructions_.resize(selection_mark);
      }
      return;
    }
    ast = selected;
    if (!ast) {
      return;
    }
    if (auto schedule = ScheduleRegion::resolve(unit_, diagnostics_, ast)) {
      schedule_region(*schedule);
      return;
    }
    if (auto strand =
            PipelineStrand::resolve(unit_, diagnostics_, types_, ast)) {
      pipeline_strand(*strand);
      return;
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      const size_t destruction_mark = active_destructions_.size();
      bool falls_through = true;
      for (auto* child : cxx::ListView{compound->statementList}) {
        statement(child);
        if (!control_->falls_through(child)) {
          falls_through = false;
          break;
        }
      }
      if (falls_through) {
        close_destruction_scope(destruction_mark);
      } else {
        active_destructions_.resize(destruction_mark);
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
      if (auto* structured =
              cxx::ast_cast<cxx::StructuredBindingDeclarationAST>(
                  declaration->declaration)) {
        reject_misplaced_binding_attributes(
            unit_, diagnostics_, structured->attributeList,
            BindingAttributeScope::LocalDeclaration);
        if (!structured->hiddenVariable || !structured->bindingDeclaratorList) {
          fail(structured,
               "structured binding decomposition must be resolved before "
               "import");
        }
        initialize_local_declarator(structured->hiddenVariable,
                                    structured->attributeList, structured);
        for (auto* binding : cxx::ListView{structured->bindingDeclaratorList}) {
          if (!binding->initializer) {
            fail(structured,
                 "structured binding decomposition must be resolved before "
                 "import");
          }
          initialize_local_declarator(binding, nullptr, structured);
        }
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
      for (auto* declarator : cxx::ListView{simple->initDeclaratorList}) {
        initialize_local_declarator(declarator, simple->attributeList, simple);
      }
      return;
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast)) {
      const size_t destruction_mark = active_destructions_.size();
      auto condition = branch_condition(branch);
      auto saved_values = values_;
      auto saved_destructions = active_destructions_;
      auto written = live_mutations(ast);
      std::vector<const cxx::Type*> sources;
      sources.reserve(written.size());
      for (const auto& mutation : written) {
        sources.push_back(mutation.projection.type);
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
      const size_t arm_mark = active_destructions_.size();
      statement(branch->statement);
      if (control_->falls_through(branch->statement)) {
        close_destruction_scope(arm_mark);
      } else {
        active_destructions_.resize(arm_mark);
      }
      loom_op_t* yield;
      auto yielded = current(written);
      check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                 locations_.get(ast), &yield));
      values_ = saved_values;
      active_destructions_ = saved_destructions;
      loom_builder_restore(&builder_, saved);
      if (flags) {
        saved = loom_builder_enter_region(&builder_, op,
                                          loom_scf_if_else_region(op));
        if (branch->elseStatement) {
          const size_t else_mark = active_destructions_.size();
          statement(branch->elseStatement);
          if (control_->falls_through(branch->elseStatement)) {
            close_destruction_scope(else_mark);
          } else {
            active_destructions_.resize(else_mark);
          }
        }
        yielded = current(written);
        check(loom_scf_yield_build(&builder_, yielded.data(), yielded.size(),
                                   locations_.get(ast), &yield));
        values_ = saved_values;
        active_destructions_ = saved_destructions;
        loom_builder_restore(&builder_, saved);
      }
      bind_values(written, loom_op_results(op));
      close_destruction_scope(destruction_mark);
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
        auto mark = begin_full_expression();
        effect(expression_statement->expression);
        finish_full_expression(mark);
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
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
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
  void guarded_latch(cxx::AST* owner, const std::vector<Mutation>& written,
                     loom_value_id_t active, Execute execute) {
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources;
    sources.reserve(written.size());
    for (const auto& mutation : written) {
      sources.push_back(mutation.projection.type);
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
    const size_t condition_destruction_mark = active_destructions_.size();
    auto initialization_mark = begin_full_expression();
    auto constexpr_contract = initialize_condition(declaration);
    finish_full_expression(initialization_mark);
    if (decision) {
      active_destructions_.resize(condition_destruction_mark);
    }
    auto written = live_mutations(ast);
    if (decision && values_.contains(decision)) {
      std::erase_if(written, [&](const auto& value) {
        return value.binding == decision;
      });
      written.push_back(whole_mutation(decision));
    }
    auto incoming = live_mutations(control_->written(ast));
    if (decision && values_.contains(decision)) {
      std::erase_if(incoming, [&](const auto& value) {
        return value.binding == decision;
      });
      incoming.push_back(whole_mutation(decision));
    }
    size_t result_count = 0;
    for (const auto& mutation : written) {
      result_count += mutation.projection.partition->component_count;
    }
    auto initial = current(incoming);
    auto outer_values = values_;
    auto source = locations_.get(ast);
    auto dependent_state_sources = dependent_source_types(written, ast);
    auto bind_loop_signatures =
        [&](std::span<const loom_type_t> header_suffix_types,
            bool require_result_types) {
          bool distinct_header = incoming.size() != written.size();
          if (!dependent_state_sources.empty() || require_result_types ||
              distinct_header) {
            auto header_sources = source_types(incoming);
            auto result_sources = source_types(written);
            return bind_condition_loop_signatures(
                types_, header_sources, result_sources, ast, &builder_,
                header_suffix_types, require_result_types || distinct_header);
          }
          return BoundConditionLoopSignatures{};
        };
    auto evaluate_condition = [&] {
      auto mark = begin_full_expression();
      auto condition =
          condition_expression
              ? expression(condition_expression).ssa()
              : scalars_.integer(true, LOOM_SCALAR_TYPE_I1, source);
      finish_full_expression(mark);
      if (decision && control_->automatic_destruction(decision)) {
        emit_automatic_destruction_if(predicate_not(condition, ast), decision,
                                      declaration);
      }
      return condition;
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
            signatures.result_types.data(), result_count, nullptr, 0, source,
            &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(incoming, before);
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
        if (decision && step && control_->automatic_destruction(decision)) {
          emit_automatic_destruction_if(predicate_not(flow.reaches_latch, ast),
                                        decision, declaration);
        } else if (decision) {
          emit_automatic_destruction(decision, declaration);
        }
        if (step || decision) {
          guarded_latch(ast, written, flow.reaches_latch, [&] {
            if (step) {
              auto mark = begin_full_expression();
              effect(step);
              finish_full_expression(mark);
            }
            if (decision) {
              if (step) {
                emit_automatic_destruction(decision, declaration);
              }
              auto mark = begin_full_expression();
              reinitialize_condition(declaration, constexpr_contract, ast);
              finish_full_expression(mark);
            }
          });
        }
        auto yielded = current(incoming);
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
            result_count, nullptr, 0, source, &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(incoming, before);
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
          auto mark = begin_full_expression();
          reinitialize_condition(declaration, constexpr_contract, ast);
          finish_full_expression(mark);
        }
        auto yielded = current(incoming);
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
        result_count, nullptr, 0, source, &op));
    auto* before = loom_scf_while_before(op);
    auto saved = loom_builder_enter_region(&builder_, op, before);
    bind(incoming, before);
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
      auto mark = begin_full_expression();
      effect(step);
      finish_full_expression(mark);
    }
    // A decision object is recreated after the body and for-loop increment.
    // The preheader supplied its first value, so every check sees a real value.
    if (decision) {
      emit_automatic_destruction(decision, declaration);
      auto mark = begin_full_expression();
      reinitialize_condition(declaration, constexpr_contract, ast);
      finish_full_expression(mark);
    }
    auto yielded = current(incoming);
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
    std::erase_if(
        written, [&](const auto& value) { return value.binding == induction; });
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
      values_.set(induction,
                  value_arena_.capture(
                      types_.partition(induction->type(), loop), components));
    } else {
      loom_op_t* cast;
      check(loom_index_cast_build(&builder_, iteration, value_type(iteration),
                                  types_.get(induction->type(), loop), source,
                                  &cast));
      values_.set(induction, result(cast));
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

  Mutation whole_mutation(cxx::Symbol* symbol) {
    return {symbol,
            {types_.unqualified_value(source_object_type(symbol)),
             &values_.partition(symbol), 0}};
  }

  std::vector<Mutation> live_mutations(cxx::AST* owner) {
    return live_mutations(control_->written(owner), owner);
  }

  std::vector<Mutation> live_mutations(std::span<cxx::Symbol* const> candidates,
                                       cxx::AST* owner = nullptr) {
    std::vector<Mutation> result;
    for (auto* symbol : candidates) {
      if (auto found = values_.find(symbol); found != values_.end()) {
        result.push_back({symbol,
                          {types_.unqualified_value(source_object_type(symbol)),
                           &found->second.partition(), 0}});
        continue;
      }
      if (!values_.contains(symbol)) {
        continue;
      }
      auto* type = types_.unqualified_value(source_object_type(symbol));
      for (const auto& projection :
           types_.projections(type, current_function_.source)) {
        if (values_.initialized(symbol, projection.component_offset,
                                projection.partition->component_count) ||
            (owner && control_->initialized(owner, symbol, projection))) {
          result.push_back({symbol, projection});
        }
      }
    }
    return result;
  }

  std::vector<const cxx::Type*> source_types(
      std::span<const Mutation> mutations) {
    std::vector<const cxx::Type*> result;
    result.reserve(mutations.size());
    for (const auto& mutation : mutations) {
      result.push_back(mutation.projection.type);
    }
    return result;
  }

  // Ordinary scalar loops keep inferred schemes; dependent source groups bind
  // their destination identities before region or result creation.
  std::vector<const cxx::Type*> dependent_source_types(
      std::span<const Mutation> mutations, cxx::AST* owner) {
    for (const auto& mutation : mutations) {
      if (types_.requires_binding(mutation.projection.type, owner)) {
        return source_types(mutations);
      }
    }
    return {};
  }

  std::vector<loom_value_id_t> current(std::span<const Mutation> mutations) {
    std::vector<loom_value_id_t> result;
    for (const auto& mutation : mutations) {
      auto value =
          values_.project(mutation.binding, *mutation.projection.partition,
                          mutation.projection.component_offset, value_arena_);
      if (!value) {
        fail(current_function_.source,
             "source state is not initialized on every continuing path");
      }
      value->append_to(result);
    }
    return result;
  }

  void bind_mutation(const Mutation& mutation, Value value) {
    auto* binding = mutation.binding;
    value = name(value, cxx::to_string(binding->name()));
    if (&value.partition() == &values_.partition(binding)) {
      values_.set(binding, value);
    } else {
      values_.replace(binding, mutation.projection.component_offset, value,
                      value_arena_);
    }
  }

  void bind(std::span<const Mutation> mutations, loom_region_t* region,
            size_t index = 0) {
    for (const auto& mutation : mutations) {
      bind_mutation(mutation, region_value(region, index,
                                           *mutation.projection.partition));
    }
  }

  void bind_values(std::span<const Mutation> mutations,
                   const loom_value_id_t* components) {
    for (const auto& mutation : mutations) {
      const auto& partition = *mutation.projection.partition;
      auto count = partition.component_count;
      bind_mutation(mutation,
                    value_arena_.capture(partition, {components, count}));
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
    if (ast->valueCategory != cxx::ValueCategory::kPrValue &&
        !unit_.typeTraits().is_volatile(ast->type)) {
      // Discarded glvalues evaluate their selection and call effects without
      // fetching the designated value. Volatile observations retain loads.
      if (cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
        return;
      }
      if (auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(ast)) {
        effect(member->baseExpression);
        return;
      }
      if (auto* operand = explicit_reference_operand(unit_, ast)) {
        effect(operand);
        return;
      }
      if (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(ast);
          cast && !cast->conversionFunction) {
        effect(cast->expression);
        return;
      }
      if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast);
          unary && !unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        effect(unary->expression);
        return;
      }
      if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast);
          subscript && !subscript->symbol) {
        effect(subscript->baseExpression);
        effect(subscript->indexExpression);
        return;
      }
      if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(ast)) {
        conditional(
            ast, expression(select->condition).ssa(),
            [&] { effect(select->iftrueExpression); },
            [&] { effect(select->iffalseExpression); });
        return;
      }
      auto call = direct_operator_call(ast);
      if (cxx::ast_cast<cxx::CallExpressionAST>(ast) ||
          (call && !types_.is_trivial_value_operation(call->function))) {
        if (reference_call(ast)) {
          return;
        }
      }
    }
    if (auto call = direct_operator_call(ast);
        call && types_.unqualified(ast->type)->kind() == cxx::TypeKind::kVoid) {
      if (ordinary_operator_call(ast, *call).value) {
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
            values_.set(
                source_value.binding,
                name(refined, cxx::to_string(source_value.binding->name())));
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
          if (auto* compose = std::get_if<PipelineComposeIntrinsic>(binding)) {
            pipeline_compose(call, *compose);
            return;
          }
          if (std::holds_alternative<PipelineMemoryIntrinsic>(*binding)) {
            fail(ast, "value-producing pipeline.memory reached a void call");
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
          fail(ast, "pipelines require explicit composition or kernel launch");
        }
        auto called = ordinary_call(call, function, locations_.get(ast));
        if (called.value || called.reference_target) {
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
  // Lexical pipeline strand nesting admits kernel operations only in worker
  // execution regions and rejects construction-only memory queries there.
  size_t pipeline_strand_depth_ = 0;
  // Bound symbols, never identifier spellings, key source-to-SSA mappings.
  ValueBindings values_;
  // Storage-backed automatic objects retain one allocation across direct and
  // aliased accesses. They are not mutable SSA bindings at region edges.
  std::unordered_map<cxx::Symbol*, StorageAllocation> locals_;
  // Completed automatic objects and destructor subobjects active at the
  // current source insertion point, in construction order.
  std::vector<DestructionAction> active_destructions_;
  // Nontrivial prvalue results retained until the innermost full expression
  // completes. Result-object transfer removes only the selected final value;
  // side temporaries retain reverse construction order.
  std::vector<std::optional<DestructionAction>> temporary_destructions_;
  // Latest evaluated result at each retained construction site. Cleanup tokens
  // remain stable when another result is adopted or extended to lexical scope.
  std::unordered_map<cxx::ExpressionAST*, size_t> temporary_results_;
  // Temporary prefix owned by the current full expression. Lexical caller
  // tokens retain parent marks without a heap-backed translator stack.
  std::optional<size_t> full_expression_mark_;
  // Structured conditional expressions hoist branch-only temporary storage
  // before the outermost conditional operation so cleanup remains dominated.
  std::vector<loom_builder_ip_t> conditional_cleanup_anchors_;
  // Lexical cleanup boundaries for the innermost break and continue targets.
  std::vector<size_t> break_destruction_marks_;
  std::vector<size_t> continue_destruction_marks_;
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
  // Lexical current objects share bindings and retain construction completion.
  ReceiverScopes receivers_{values_};
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
