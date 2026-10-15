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
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "iree/base/api.h"
#include "loom/import/cxx/binding/assumptions.h"
#include "loom/import/cxx/binding/config.h"
#include "loom/import/cxx/binding/constexpr.h"
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
#include "loom/ops/func/ops.h"
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
        intrinsics_(unit, diagnostics, types_, locations_, names_, launches_,
                    function_contracts_, module),
        functions_(unit, diagnostics, module, intrinsics_, launches_, configs_,
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
    for (size_t index = 0; index < functions_.pending().size(); ++index) {
      function(functions_.pending()[index]);
    }
    functions_.build_declarations(types_, locations_, &builder_);
    functions_.build_benchmarks(locations_, &builder_);
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
    if (current_function_.kind != FunctionKind::LaunchConfiguration &&
        current_function_.kind != FunctionKind::ClusteredLaunchConfiguration) {
      fail(owner,
           "target launch query requires a kernel configuration function");
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
  };

  struct Returned {
    // Native return operands; empty for a void source function.
    std::vector<loom_value_id_t> values;
    // Return or branch whose source location owns the resulting exit.
    cxx::AST* source;
  };

  void function(cxx::FunctionSymbol* symbol) {
    auto defined = functions_.define(symbol, types_, locations_, &builder_);
    if (defined.configuration) {
      function_body(*defined.configuration);
      auto* region = defined.configuration->region;
      if (loom_region_has_read_effects(region) ||
          loom_region_has_write_effects(region) ||
          loom_region_has_convergent_effects(region) ||
          loom_region_has_observable_effects(region)) {
        fail(defined.configuration->source,
             "kernel configuration body must be pure");
      }
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
    control_.emplace(unit_, diagnostics_, types_, body);
    auto saved = loom_builder_enter_region(&builder_, op, region);
    values_.clear();
    locals_.clear();
    value_arena_.reset();
    size_t argument_index = 0;
    size_t parameter_index = 0;
    for (auto* parameter : parameters) {
      auto* value_type = types_.unqualified(parameter->type());
      const auto& partition = types_.partition(value_type, defined.source);
      bool kernel_pointer = defined.kind == FunctionKind::Kernel &&
                            partition.kind == ValueKind::Pointer;
      // Only pointers change representation at kernel entry: the ABI supplies
      // one buffer, and source pointer transport adds its zero byte origin.
      auto value = region_value(region, argument_index,
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
    auto returned = return_sequence(
        {body->statementList, nullptr, lexical_fastmath_flags_});
    loom_op_t* terminator;
    if (defined.kind == FunctionKind::Kernel) {
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
      auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      for (size_t i = 0; i < returned.values.size(); ++i) {
        check(loom_index_cast_build(&builder_, returned.values[i],
                                    value_type(returned.values[i]), index_type,
                                    source, &terminator));
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
        if (current_function_.return_type->kind() != cxx::TypeKind::kVoid) {
          fail(current_function_.source,
               "non-void function can reach its end without returning a value");
        }
        return {{}, current_function_.source};
      }
      auto* child = sequence.remaining->value;
      sequence.remaining = sequence.remaining->next;
      if (control_->returns(child) != ExitFlow::None) {
        return returning_statement(child, sequence);
      }
      statement(child);
    }
  }

  StorageAccess allocate_local(cxx::Symbol* symbol, int64_t alignment,
                               cxx::AST* owner) {
    auto allocation = storage_.allocate(
        symbol->type(), LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, alignment, owner);
    auto spelling = cxx::to_string(symbol->name());
    name(Value(allocation.pointer), spelling + "_storage");
    name(allocation.view, spelling + "_view");
    locals_[symbol] = allocation;
    return {allocation.view, std::nullopt};
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
    if (unit_.typeTraits().is_array(variable->type()) ||
        control_->addressed(variable) ||
        unit_.typeTraits().is_volatile(variable->type())) {
      allocate_local(variable, variable->explicitAlignment(), owner);
      if (initializer) {
        if (types_.address_domain(types_.unqualified(variable->type()),
                                  owner) == LOOM_SCALAR_TYPE_NONE) {
          types_.admit_copy(variable->constructor(), variable->type(), owner);
        }
        initialization_.storage(locals_.at(variable), variable->type(),
                                initializer, owner);
      }
      return;
    }
    if (!initializer) {
      fail(owner, "locals require initializers");
    }
    types_.partition(variable->type(), owner);
    if (types_.address_domain(types_.unqualified(variable->type()), owner) ==
        LOOM_SCALAR_TYPE_NONE) {
      types_.admit_copy(variable->constructor(), variable->type(), owner);
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
      bool returns_void =
          current_function_.return_type->kind() == cxx::TypeKind::kVoid;
      if (!ret->expression) {
        if (!returns_void) {
          fail(ast, "non-void return requires a value");
        }
        return {{}, ret};
      }
      if (ret->expression->type->kind() == cxx::TypeKind::kVoid) {
        if (!returns_void) {
          fail(ast, "non-void return requires a value");
        }
        effect(ret->expression);
        return {{}, ret};
      }
      if (returns_void) {
        fail(ast, "void return cannot carry a value");
      }
      Returned returned = {{}, ret};
      convert(ret->expression, current_function_.return_type, ret)
          .append_to(returned.values);
      return returned;
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return return_sequence(
          {compound->statementList, &continuation, lexical_fastmath_flags_});
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && control_->returns(branch) != ExitFlow::None) {
      if (control_->returns(branch->statement) != ExitFlow::All &&
          control_->returns(branch->elseStatement) != ExitFlow::All) {
        fail(ast,
             "conditional returns require at most one fallthrough arm; "
             "shared continuations need a scoped exit projection");
      }
      auto condition = branch_condition(branch);
      auto outer_values = values_;
      BoundSignature result_signature;
      if (current_function_.return_type->kind() != cxx::TypeKind::kVoid) {
        std::array<const cxx::Type*, 1> result_sources = {
            current_function_.return_type};
        result_signature =
            bind_signature(types_, result_sources, ast, &builder_);
      }
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
      fail(ast,
           "returns inside loops require a structured loop-exit projection");
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
    if (unit_.typeTraits().is_array(source->type)) {
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
    auto* base_type = ast->baseExpression->type;
    auto* index_type = ast->indexExpression->type;
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

  StorageProjection object_address(cxx::ExpressionAST* ast) {
    ast = strip_implicit_casts(ast);
    if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(ast)) {
      return object_address(nested->expression);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = locals_.find(id->symbol); found != locals_.end()) {
        return storage_.project(found->second.pointer, id->type, id);
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
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(ast)) {
      auto operands = subscript_operands(subscript);
      return storage_.advance(operands.base, operands.index, operands.base_type,
                              operands.index_type, cxx::TokenKind::T_PLUS, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (!unary->symbol && unary->op == cxx::TokenKind::T_STAR) {
        return pointer_expression(unary->expression);
      }
    }
    if (unit_.typeTraits().is_class(ast->type) &&
        ast->valueCategory == cxx::ValueCategory::kPrValue) {
      auto value = expression(ast);
      auto allocation = storage_.allocate(
          ast->type, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, 0, ast);
      auto object = storage_.project(allocation.pointer, ast->type, ast);
      objects_.store(object, value, ast->type, ast);
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

  // One lane projected from a snapshot of its complete vector lvalue. The
  // source base and index have both been evaluated exactly once.
  struct VectorLaneLvalue {
    // Complete vector destination receiving the functional lane update.
    ObjectLvalue destination;
    // Vector value loaded after evaluating the destination and lane selector.
    Value value;
    // Lane selector before projection to Loom's index domain.
    loom_value_id_t index;
    // Source selector type retaining its integer width and signedness.
    const cxx::Type* index_type;
  };

  // Resolving a source lvalue executes every address and index expression once.
  // A lane projection retains the complete vector snapshot required to replace
  // that lane without reloading memory during a compound update.
  struct Lvalue {
    // Complete object or one evaluated lane within a complete vector object.
    std::variant<ObjectLvalue, VectorLaneLvalue> projection;
  };

  ObjectLvalue object_destination(
      cxx::ExpressionAST* expression, cxx::AST* owner,
      std::optional<Destination> retained_destination = std::nullopt) {
    types_.require_mutable(expression->type, expression);
    auto destination = retained_destination ? retained_destination
                                            : control_->destination(expression);
    if (destination) {
      if (!locals_.contains(destination->binding)) {
        if (!values_.contains(destination->binding)) {
          fail(owner, "mutation requires an owned automatic source binding");
        }
        return {expression->type, *destination};
      }
    }
    if (unit_.typeTraits().is_class(expression->type)) {
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
      std::optional<Destination> retained_destination) {
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
      auto vector_destination =
          object_destination(vector_expression, owner, retained_destination);
      auto index_value = expression(index_expression);
      auto vector_value = read(vector_destination, vector_expression);
      return {VectorLaneLvalue{vector_destination, vector_value,
                               index_value.ssa(), index_expression->type}};
    }

    // The commuted spelling i[v] still evaluates its written left operand
    // before resolving and loading the vector on the right.
    auto index_value = expression(index_expression);
    auto vector_destination =
        object_destination(vector_expression, owner, retained_destination);
    auto vector_value = read(vector_destination, vector_expression);
    return {VectorLaneLvalue{vector_destination, vector_value,
                             index_value.ssa(), index_expression->type}};
  }

  Lvalue destination(cxx::ExpressionAST* expression, cxx::AST* owner) {
    auto* projected = strip_implicit_casts(expression);
    while (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(projected)) {
      projected = strip_implicit_casts(nested->expression);
    }
    if (auto* subscript = cxx::ast_cast<cxx::SubscriptExpressionAST>(projected);
        subscript && (types_.native_vector(subscript->baseExpression->type) ||
                      types_.native_vector(subscript->indexExpression->type))) {
      return vector_lane_destination(subscript, owner,
                                     control_->destination(expression));
    }
    return {object_destination(expression, owner)};
  }

  Value read(const Lvalue& target, cxx::AST* owner) {
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      return read(*object, owner);
    }
    const auto& lane = std::get<VectorLaneLvalue>(target.projection);
    return Value(vectors_.extract(lane.value.ssa(), lane.index,
                                  lane.destination.type, lane.index_type,
                                  owner));
  }

  void write(const Lvalue& target, Value value, cxx::AST* owner) {
    if (auto* object = std::get_if<ObjectLvalue>(&target.projection)) {
      write(*object, value, owner);
      return;
    }
    const auto& lane = std::get<VectorLaneLvalue>(target.projection);
    auto updated =
        vectors_.insert(lane.value.ssa(), value.ssa(), lane.index,
                        lane.destination.type, lane.index_type, owner);
    write(lane.destination, Value(updated), owner);
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
    } else {
      if (!unit_.typeTraits().is_integral(destination->type) ||
          types_.unqualified(destination->type)->kind() ==
              cxx::TypeKind::kBool) {
        fail(owner, "increment requires an integer or pointer binding");
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
      fail(ast, "overloaded compound assignment is not admitted");
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
      sources.push_back(symbol->type());
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

  void conditional_object(StorageProjection destination, const cxx::Type* type,
                          cxx::ConditionalExpressionAST* select) override {
    if (types_.native_vector(select->condition->type)) {
      fail(select, "vector conditions require lane-wise selection");
    }
    auto condition = expression(select->condition).ssa();
    conditional(
        select, condition,
        [&] {
          initialization_.object(destination, type, select->iftrueExpression,
                                 select);
        },
        [&] {
          initialization_.object(destination, type, select->iffalseExpression,
                                 select);
        });
  }

  std::optional<IntrinsicCallResult> atomic_builtin(
      cxx::CallExpressionAST* call) {
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
    return std::nullopt;
  }

  Value constant(const cxx::ConstValue& value, cxx::ExpressionAST* ast) {
    return types_.native_vector(ast->type)
               ? vectors_.constant(value, ast->type, ast)
               : scalars_.constant(value, ast->type, ast);
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
      if (cast->conversionFunction &&
          types_.address_domain(cast->expression->type, ast) ==
              LOOM_SCALAR_TYPE_NONE) {
        types_.admit_copy(cast->conversionFunction, ast->type, ast);
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
        types_.admit_copy(cast->constructorSymbol, ast->type, ast);
        return initialization_.value(ast->type, cast->expressionList, ast);
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
      types_.admit_copy(construction->constructorSymbol, ast->type, ast);
      return initialization_.value(
          ast->type, construction->bracedInitList->expressionList, ast);
    }
    if (auto* initializer = cxx::ast_cast<cxx::BracedInitListAST>(ast)) {
      return initialization_.value(ast->type, initializer->expressionList, ast);
    }
    if (auto* initializer = cxx::ast_cast<cxx::ParenInitializerAST>(ast)) {
      return initialization_.value(ast->type, initializer->expressionList, ast);
    }
    if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(ast)) {
      if (types_.native_vector(select->condition->type)) {
        fail(ast, "vector conditions require lane-wise selection");
      }
      auto condition = expression(select->condition).ssa();
      return conditional(
          ast, condition, [&] { return expression(select->iftrueExpression); },
          [&] { return expression(select->iffalseExpression); });
    }

    if (auto* decision = cxx::ast_cast<cxx::ConditionExpressionAST>(ast)) {
      return binding(decision->symbol, ast);
    }
    if (auto* id = cxx::ast_cast<cxx::IdExpressionAST>(ast)) {
      if (auto found = values_.find(id->symbol); found != values_.end()) {
        return found->second;
      }
      if (locals_.contains(id->symbol)) {
        return binding(id->symbol, ast);
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
          fail(ast, "overloaded binary operation requires a Loom operation");
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
        fail(ast, "only builtin increment/decrement is admitted");
      }
      return increment(postfix->baseExpression, postfix->op,
                       IncrementResult::Previous, ast);
    }
    if (auto* unary = cxx::ast_cast<cxx::UnaryExpressionAST>(ast)) {
      if (unary->symbol) {
        auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(unary->symbol);
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (!binding) {
          fail(ast, "overloaded unary operation requires a Loom operation");
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
      auto* base_type = subscript->baseExpression->type;
      auto* index_type = subscript->indexExpression->type;
      if (types_.native_vector(base_type) || types_.native_vector(index_type)) {
        if (subscript->symbol) {
          fail(ast, "overloaded indexing is not admitted");
        }
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
      if (auto called = atomic_builtin(call)) {
        return *called->value;
      }
      auto* callee = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      auto* function =
          callee ? cxx::symbol_cast<cxx::FunctionSymbol>(callee->symbol)
                 : nullptr;
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
           std::holds_alternative<AsyncIntrinsic>(*binding))) {
        require_kernel_context(ast);
      }
      if (binding && std::holds_alternative<TargetIntrinsic>(*binding)) {
        require_launch_configuration_context(ast);
      }
      auto* assembly =
          binding ? std::get_if<AssemblyIntrinsic>(binding) : nullptr;
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
        return *assembly->call(fragment, source_arguments, &builder_, source);
      }
      auto flatten_arguments = [&] {
        std::vector<loom_value_id_t> arguments;
        for (auto argument : source_arguments) {
          argument.append_to(arguments);
        }
        return arguments;
      };
      loom_op_t* op;
      if (annotated(function, "shuffle_xor")) {
        require_kernel_context(ast);
        auto arguments = flatten_arguments();
        if (arguments.size() != 3) {
          fail(ast, "shuffle_xor requires three scalar operands");
        }
        check(loom_kernel_subgroup_shuffle_build(
            &builder_, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR, arguments[0],
            arguments[1], arguments[2], types_.get(ast->type, ast), source,
            &op));
        return result(op);
      }
      if (binding) {
        auto called =
            intrinsics_.call(*binding, source_arguments, value_arena_, storage_,
                             ast, lexical_fastmath_flags_, &builder_, source);
        if (!called.value) {
          fail(ast, "void intrinsic cannot be used as a value");
        }
        return *called.value;
      }
      if (annotated(function, "kernel")) {
        fail(ast, "kernels require an explicit launch operation");
      }
      auto symbol = functions_.declare(function, ast);
      auto arguments = flatten_arguments();
      std::array<const cxx::Type*, 1> result_sources = {ast->type};
      auto results = bind_signature(types_, result_sources, ast, &builder_);
      check(loom_func_call_build(
          &builder_, 0, 0, 0, 0, symbol, arguments.data(), arguments.size(),
          results.types.data(), results.types.size(), nullptr, 0, source, &op));
      return value_arena_.capture(types_.partition(ast->type, ast),
                                  {loom_op_results(op), results.types.size()});
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
  IterationFlow iteration_branch(cxx::AST* owner, loom_value_id_t condition,
                                 bool tracks_break, Then then_statement,
                                 Else else_statement) {
    auto written = live_mutations(owner);
    auto outer_values = values_;
    std::vector<const cxx::Type*> sources = {unit_.control()->getBoolType()};
    if (tracks_break) {
      sources.push_back(unit_.control()->getBoolType());
    }
    const size_t control_count = sources.size();
    sources.reserve(control_count + written.size());
    for (auto* symbol : written) {
      sources.push_back(symbol->type());
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

  IterationFlow iteration_sequence(cxx::CompoundStatementAST* owner,
                                   cxx::List<cxx::StatementAST*>* remaining,
                                   uint8_t fastmath, bool tracks_break) {
    FastMathScope math_scope(lexical_fastmath_flags_, fastmath);
    while (remaining) {
      auto* child = remaining->value;
      remaining = remaining->next;
      auto flow = iteration_exit_flow(child, tracks_break);
      if (flow == ExitFlow::None) {
        statement(child);
        continue;
      }
      return iteration_tail(owner, child, remaining, fastmath, tracks_break);
    }
    return constant_iteration_flow(true, true, tracks_break, owner);
  }

  // A tail enters at most one arm directly. When both arms can reach it, join
  // their fallthrough values first and emit the shared source just once.
  IterationFlow iteration_tail(cxx::CompoundStatementAST* owner,
                               cxx::StatementAST* ast,
                               cxx::List<cxx::StatementAST*>* remaining,
                               uint8_t continuation_fastmath,
                               bool tracks_break) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    auto flow = iteration_exit_flow(ast, tracks_break);
    if (flow == ExitFlow::None) {
      if (ast) {
        statement(ast);
      }
      return iteration_sequence(owner, remaining, continuation_fastmath,
                                tracks_break);
    }
    if (flow == ExitFlow::All || !remaining) {
      return iteration_statement(ast, tracks_break);
    }
    if (auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
        branch && (iteration_exit_flow(branch->statement, tracks_break) ==
                       ExitFlow::All ||
                   iteration_exit_flow(branch->elseStatement, tracks_break) ==
                       ExitFlow::All)) {
      auto condition = branch_condition(branch);
      return iteration_branch(
          owner, condition, tracks_break,
          [&] {
            return iteration_tail(owner, branch->statement, remaining,
                                  continuation_fastmath, tracks_break);
          },
          [&] {
            return iteration_tail(owner, branch->elseStatement, remaining,
                                  continuation_fastmath, tracks_break);
          });
    }
    auto fallthrough = iteration_statement(ast, tracks_break);
    return iteration_branch(
        owner, fallthrough.reaches_tail, tracks_break,
        [&] {
          return iteration_sequence(owner, remaining, continuation_fastmath,
                                    tracks_break);
        },
        [&] {
          auto stopped =
              constant_iteration_flow(false, false, tracks_break, ast);
          stopped.reaches_latch = fallthrough.reaches_latch;
          return stopped;
        });
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
      return constant_iteration_flow(true, true, tracks_break, ast);
    }
    if (cxx::ast_cast<cxx::ContinueStatementAST>(ast)) {
      return constant_iteration_flow(false, true, tracks_break, ast);
    }
    if (cxx::ast_cast<cxx::BreakStatementAST>(ast)) {
      return constant_iteration_flow(false, false, tracks_break, ast);
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      return iteration_sequence(compound, compound->statementList,
                                lexical_fastmath_flags_, tracks_break);
    }
    // The retained path analysis propagates iteration exits through blocks
    // and conditionals only; nested loops consume their own exits.
    auto* branch = cxx::ast_cast<cxx::IfStatementAST>(ast);
    auto condition = branch_condition(branch);
    return iteration_branch(
        ast, condition, tracks_break,
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

  void statement(cxx::StatementAST* ast) {
    FastMathScope math_scope(
        lexical_fastmath_flags_,
        resolve_fastmath(ast ? fastmath_.statement(ast) : std::nullopt));
    ast = selected_statement(ast);
    if (!ast) {
      return;
    }
    if (auto* compound = cxx::ast_cast<cxx::CompoundStatementAST>(ast)) {
      for (auto* child : cxx::ListView{compound->statementList}) {
        statement(child);
      }
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
        sources.push_back(symbol->type());
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
    if (auto* loop = cxx::ast_cast<cxx::ForStatementAST>(ast)) {
      LoopSchedule schedule(unit_, diagnostics_, types_, loop->attributeList);
      if (!loop->condition) {
        fail(ast, "for loops require a condition");
      }
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
        fail(
            ast,
            "loop scheduling requires a nonwrapping unsigned counted for loop");
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
      sources.push_back(symbol->type());
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
      sources.push_back(symbol->type());
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

    if (control_->breaks(body) != ExitFlow::None) {
      loom_op_t* op;
      if (test == LoopTest::BeforeBody) {
        auto active =
            scalars_.integer(true, LOOM_SCALAR_TYPE_I1, locations_.get(ast));
        auto header_initial = initial;
        header_initial.push_back(active);
        std::vector<const cxx::Type*> result_sources;
        result_sources.reserve(written.size());
        for (auto* symbol : written) {
          result_sources.push_back(symbol->type());
        }
        auto result_signature =
            bind_signature(types_, result_sources, ast, &builder_);
        check(loom_scf_while_build(
            &builder_, header_initial.data(), header_initial.size(),
            /*iter_args_types=*/nullptr, result_signature.types.data(),
            result_signature.types.size(), nullptr, 0, source, &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(written, before);
        auto header_active =
            loom_region_entry_arg_id(before, header_initial.size() - 1u);
        auto condition = guarded_condition(ast, header_active, [&] {
          return expression(condition_expression).ssa();
        });
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
        check(loom_scf_while_build(&builder_, initial.data(), initial.size(),
                                   /*iter_args_types=*/nullptr,
                                   /*result_types=*/nullptr, initial.size(),
                                   nullptr, 0, source, &op));
        auto* before = loom_scf_while_before(op);
        auto saved = loom_builder_enter_region(&builder_, op, before);
        bind(written, before);
        auto flow = loop_body(body, /*tracks_break=*/true);
        auto condition = guarded_condition(ast, flow.reaches_latch, [&] {
          return expression(condition_expression).ssa();
        });
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
    check(loom_scf_while_build(&builder_, initial.data(), initial.size(),
                               /*iter_args_types=*/nullptr,
                               /*result_types=*/nullptr, initial.size(),
                               nullptr, 0, source, &op));
    auto* before = loom_scf_while_before(op);
    auto saved = loom_builder_enter_region(&builder_, op, before);
    bind(written, before);
    if (test == LoopTest::AfterBody) {
      loop_body(body);
    }
    auto condition = expression(condition_expression).ssa();
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

  void counted_loop(cxx::ForStatementAST* loop, const CountedLoop& counted,
                    const LoopSchedule& schedule) {
    auto source = locations_.get(loop);
    auto* induction = counted.induction;
    auto lower = unsigned_offset(values_.at(induction).ssa(), source);
    auto* constant_upper = std::get_if<unsigned>(&counted.upper);
    auto bound =
        constant_upper
            ? scalars_.integer(static_cast<int32_t>(*constant_upper),
                               LOOM_SCALAR_TYPE_I32, source)
            : expression(std::get<CountedLoop::Bound>(counted.upper).expression)
                  .ssa();
    auto upper = unsigned_offset(bound, source);
    auto step = scalars_.integer(counted.step, LOOM_SCALAR_TYPE_OFFSET, source);
    auto depth = schedule.pipeline_depth()
                     ? expression(schedule.pipeline_depth()).ssa()
                     : 0;
    auto factor = schedule.unroll_factor()
                      ? expression(schedule.unroll_factor()).ssa()
                      : 0;
    auto written = live_mutations(loop);
    std::erase(written, induction);
    auto initial = current(written);
    auto outer_values = values_;
    auto* op = schedule.build(&builder_, lower, upper, step, initial, depth,
                              factor, source);
    auto* body = loom_scf_for_body(op);
    auto saved = loom_builder_enter_region(&builder_, op, body);
    auto iteration = name(loom_region_entry_arg_id(body, 0),
                          cxx::to_string(induction->name()));
    loom_op_t* cast;
    check(loom_index_cast_build(&builder_, iteration, value_type(iteration),
                                types_.get(induction->type(), loop), source,
                                &cast));
    values_[induction] = result(cast);
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
    std::vector<cxx::Symbol*> result;
    for (auto* symbol : control_->written(owner)) {
      if (values_.contains(symbol)) {
        result.push_back(symbol);
      }
    }
    return result;
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
      if (atomic_builtin(call)) {
        return;
      }
      auto* id = cxx::ast_cast<cxx::IdExpressionAST>(call->baseExpression);
      if (id && annotated(id->symbol, "assume")) {
        auto predicates = assumption_predicates(unit_, diagnostics_, call);
        for (const auto& predicate : predicates) {
          for (size_t i = 0; i < predicate.value_count; ++i) {
            const auto& source_value = predicate.values[i];
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
        auto* function =
            id ? cxx::symbol_cast<cxx::FunctionSymbol>(id->symbol) : nullptr;
        if (function && (intrinsics_.check_binding(function, ast) ||
                         functions_.is_check_case(function) ||
                         annotated(function, "check_benchmark"))) {
          fail(ast,
               "check declarations cannot be called from ordinary functions");
        }
        auto* binding = function ? intrinsics_.lookup(function, ast) : nullptr;
        if (binding) {
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
            assembly->call(fragment, arguments, &builder_, locations_.get(ast));
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
        std::vector<loom_value_id_t> arguments;
        for (auto* argument : cxx::ListView{call->expressionList}) {
          expression(argument).append_to(arguments);
        }
        auto callee = functions_.declare(function, ast);
        loom_op_t* op;
        check(loom_func_call_build(&builder_, 0, 0, 0, 0, callee,
                                   arguments.data(), arguments.size(), nullptr,
                                   0, nullptr, 0, locations_.get(ast), &op));
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
  // Immutable control facts for the function currently being translated.
  std::optional<ControlFlow> control_;
};
}  // namespace

void translate(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
               loom_module_t* module,
               const loom_cxx_import_options_t& options) {
  Translator(unit, diagnostics, module, options).translate();
}

}  // namespace loom::cxx_import
