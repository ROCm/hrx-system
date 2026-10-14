// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/task_abi.h"

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/abi/task/parameter_layout.h"
#include "loom/target/abi/task/state_layout.h"
#include "loom/target/function_contract.h"
#include "loom/target/pass_environment.h"

static iree_status_t loom_low_task_entry_reject(loom_pass_t* pass,
                                                const loom_op_t* source,
                                                iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {loom_param_string(constraint)};
  return iree_diagnostic_emit(pass->diagnostic_emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = source,
                                  .error = LOOM_ERR_BACKEND_052,
                                  .params = params,
                                  .param_count = IREE_ARRAYSIZE(params),
                              });
}

static bool loom_low_task_entry_builder_initialize(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptors,
    loom_location_id_t location, const loom_low_task_entry_lowering_t* lowering,
    void* target_data, loom_low_task_entry_builder_t* out_builder) {
  *out_builder = (loom_low_task_entry_builder_t){
      .location = location,
      .target_data = target_data,
  };
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &out_builder->ir);
  return lowering->initialize(descriptors, out_builder);
}

typedef struct loom_low_task_entry_parameters_t {
  // Original body arguments, borrowed until their blocks are moved.
  const loom_value_id_t* arguments;
  // Admitted physical carriers, indexed by original argument ordinal.
  loom_type_t* types;
  // Number of arguments, matching the retained logical interface.
  uint16_t count;
} loom_low_task_entry_parameters_t;

static iree_status_t loom_low_task_entry_parameters_prepare(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_task_parameter_layout_t* abi,
    const loom_low_task_entry_lowering_t* lowering,
    loom_low_task_entry_parameters_t* out_parameters) {
  *out_parameters = (loom_low_task_entry_parameters_t){0};
  out_parameters->arguments =
      loom_func_like_arg_ids(function, &out_parameters->count);
  if (out_parameters->count != abi->attributes.parameter_count) {
    return loom_low_task_entry_reject(
        pass, function.op,
        IREE_SV("one argument carrier per logical parameter"));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      pass->arena, out_parameters->count, sizeof(*out_parameters->types),
      (void**)&out_parameters->types));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < out_parameters->count && !loom_pass_has_error_diagnostics(pass) &&
       iree_status_is_ok(status);
       ++i) {
    const iree_hal_executable_dispatch_parameter_v0_t* parameter =
        &abi->parameters[i];
    const uint8_t size =
        parameter->type == IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING
            ? LOOM_TASK_ABI64_POINTER_SIZE
            : parameter->size;
    const loom_type_t type =
        loom_module_value_type(module, out_parameters->arguments[i]);
    out_parameters->types[i] = type;
    const iree_string_view_t constraint =
        lowering->parameter_constraint(type, size);
    if (!iree_string_view_is_empty(constraint)) {
      const loom_diagnostic_param_t params[] = {
          loom_param_i64(i),
          loom_param_type(type),
          loom_param_string(constraint),
      };
      status = iree_diagnostic_emit(pass->diagnostic_emitter,
                                    &(loom_diagnostic_emission_t){
                                        .op = function.op,
                                        .error = LOOM_ERR_BACKEND_053,
                                        .params = params,
                                        .param_count = IREE_ARRAYSIZE(params),
                                    });
    }
  }
  return status;
}

static iree_status_t loom_low_task_entry_build_parameter_imports(
    loom_low_task_entry_builder_t* builder, loom_pass_t* pass,
    const loom_low_task_entry_lowering_t* lowering, loom_func_like_t function,
    const loom_task_parameter_layout_t* abi,
    const loom_low_task_entry_parameters_t* parameters,
    loom_value_id_t dispatch, loom_value_id_t** out_arguments) {
  const uint16_t argument_count = parameters->count;
  const loom_value_id_t* source_arguments = parameters->arguments;
  const loom_type_t* types = parameters->types;
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      pass->arena, argument_count, sizeof(*arguments), (void**)&arguments));
  loom_value_id_t constants = LOOM_VALUE_ID_INVALID;
  loom_value_id_t bindings = LOOM_VALUE_ID_INVALID;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    const iree_hal_executable_dispatch_parameter_v0_t* parameter =
        &abi->parameters[i];
    const bool binding =
        parameter->type == IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING;
    const uint8_t size =
        binding ? LOOM_TASK_ABI64_POINTER_SIZE : parameter->size;
    const loom_low_task_parameter_access_t access = {
        .carrier_type = types[i],
        .table_offset = binding ? LOOM_TASK_ABI64_BINDINGS_OFFSET
                                : LOOM_TASK_ABI64_CONSTANTS_OFFSET,
        .value_offset = binding
                            ? parameter->offset * LOOM_TASK_ABI64_POINTER_SIZE
                            : parameter->offset,
        .byte_length = size,
        .used =
            loom_module_value_has_uses(builder->ir.module, source_arguments[i]),
    };
    status = lowering->emit_parameter(builder, &access, dispatch,
                                      binding ? &bindings : &constants,
                                      &arguments[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  if (predicate_count) {
    loom_ir_remap_t remap;
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        builder->ir.module, builder->ir.module, pass->arena, NULL, &remap));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_arguments,
                                                  arguments, argument_count));
    loom_predicate_t* imported_predicates = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
        &remap, predicates, predicate_count, &imported_predicates));
    loom_op_t* assume = NULL;
    IREE_RETURN_IF_ERROR(loom_low_assume_build(
        &builder->ir, arguments, argument_count, imported_predicates,
        predicate_count, types, argument_count, builder->location, &assume));
    arguments = loom_op_results(assume);
  }
  *out_arguments = arguments;
  return iree_ok_status();
}

static iree_status_t loom_low_task_entry_materialize_body(
    loom_low_task_entry_builder_t* builder, loom_rewriter_t* rewriter,
    const loom_low_task_entry_lowering_t* lowering, loom_region_t* body,
    const loom_value_id_t* state_arguments, loom_value_id_t success) {
  iree_status_t status = iree_ok_status();
  for (uint16_t b = 1; b < body->block_count && iree_status_is_ok(status);
       ++b) {
    loom_block_t* block = loom_region_block(body, b);
    for (loom_op_t* op = block->first_op; op && iree_status_is_ok(status);) {
      loom_op_t* next = op->next_op;
      loom_builder_set_before(&builder->ir, op);
      if (loom_low_return_isa(op)) {
        loom_op_t* new_return = NULL;
        status = loom_low_return_build(&builder->ir, &success, 1, op->location,
                                       &new_return);
        if (iree_status_is_ok(status)) {
          status = loom_rewriter_erase(rewriter, op);
        }
      } else if (loom_low_live_in_isa(op)) {
        const iree_string_view_t name = loom_string_table_get(
            &builder->ir.module->strings, loom_low_live_in_source(op));
        for (unsigned i = 0;
             i < LOOM_TASK_BUILTIN_COUNT_ && iree_status_is_ok(status); ++i) {
          const loom_task_builtin_info_t* builtin = &loom_task_builtins[i];
          if (!iree_string_view_equal(name, builtin->name)) {
            continue;
          }
          loom_value_id_t widened;
          status = lowering->emit_builtin(
              builder, builtin, state_arguments[builtin->state_argument],
              &widened);
          if (iree_status_is_ok(status)) {
            status = loom_rewriter_replace_all_uses_and_erase(rewriter, op,
                                                              &widened, 1);
          }
          break;
        }
      }
      op = next;
    }
  }
  return status;
}

iree_status_t loom_low_task_materialize_kernel(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_task_entry_lowering_t* lowering, void* target_data) {
  if (!loom_low_kernel_def_isa(function.op)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_registry_t* registry =
      loom_low_pass_capability_descriptor_registry(
          loom_low_pass_capability_from_pass(pass));
  loom_symbol_fact_table_t facts;
  loom_symbol_fact_table_initialize(&facts, pass->arena);
  loom_low_resolved_target_t target;
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &facts, function.op,
      loom_target_function_version_target_facts(pass->function_version),
      registry, pass->diagnostic_emitter, &target));
  if (!target.target_facts ||
      target.target_facts->fact_type != lowering->fact_type) {
    return iree_ok_status();
  }
  const loom_op_t* source = function.op;
  if (loom_low_kernel_def_workgroup_cluster_size_x(source) > 1 ||
      loom_low_kernel_def_workgroup_cluster_size_y(source) > 1 ||
      loom_low_kernel_def_workgroup_cluster_size_z(source) > 1) {
    return loom_low_task_entry_reject(
        pass, source,
        IREE_SV("one invocation per workgroup without clustering"));
  }
  loom_low_task_entry_builder_t builder;
  if (!loom_low_task_entry_builder_initialize(module, target.descriptor_set,
                                              source->location, lowering,
                                              target_data, &builder)) {
    return loom_low_task_entry_reject(
        pass, source,
        IREE_SV("task entry instructions for the selected representation"));
  }
  loom_task_parameter_layout_t abi;
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_task_parameter_layout_parse(
      module, source, loom_low_kernel_def_abi_layout(source),
      pass->diagnostic_emitter, pass->arena, &accepted, &abi));
  if (!accepted) {
    return iree_ok_status();
  }
  loom_low_task_entry_parameters_t parameter_plan;
  IREE_RETURN_IF_ERROR(loom_low_task_entry_parameters_prepare(
      pass, module, function, &abi, lowering, &parameter_plan));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }
  loom_builder_set_before(&builder.ir, source);
  const loom_type_t argument_types[] = {
      builder.pointer_type, builder.pointer_type, builder.pointer_type};
  loom_low_func_def_build_flags_t flags =
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI |
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI_LAYOUT |
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
  if (loom_low_kernel_def_has_retain(source)) {
    flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
  }
  if (loom_low_kernel_def_has_allocation(source)) {
    flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ALLOCATION;
  }
  if (loom_low_kernel_def_has_schedule(source)) {
    flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_SCHEDULE;
  }
  loom_symbol_ref_t target_ref = loom_low_kernel_def_target(source);
  if (loom_symbol_ref_is_valid(target_ref)) {
    flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_string_id_t export_symbol = loom_low_kernel_def_export_symbol(source);
  if (loom_low_kernel_def_has_export_symbol(source)) {
    flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  loom_op_t* physical = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &builder.ir, flags, LOOM_LOW_VISIBILITY_PUBLIC,
      loom_low_kernel_def_retain(source), 0, 0, 0,
      loom_low_kernel_def_allocation(source),
      loom_low_kernel_def_schedule(source),
      loom_low_kernel_def_descriptor_set(source), target_ref,
      LOOM_TARGET_ABI_HAL_KERNEL, loom_named_attr_slice_empty(),
      loom_low_kernel_def_abi_layout(source), export_symbol,
      loom_named_attr_slice_empty(), loom_func_like_callee(function),
      argument_types, 3, &builder.word_type, 1, NULL, 0, NULL, 0,
      source->location, &physical));
  loom_region_t* body = loom_low_func_def_body(physical);
  const loom_value_id_t* state_arguments =
      loom_region_entry_block(body)->arg_ids;
  loom_builder_enter_region(&builder.ir, physical, body);
  loom_value_id_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(loom_low_task_entry_build_parameter_imports(
      &builder, pass, lowering, function, &abi, &parameter_plan,
      state_arguments[1], &parameters));
  loom_value_id_t success;
  IREE_RETURN_IF_ERROR(lowering->emit_success(&builder, &success));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  loom_block_t* moved_entry = NULL;
  iree_status_t status = loom_rewriter_move_region_blocks(
      &rewriter, loom_low_kernel_def_body(source), function.op, body, 1,
      physical, &moved_entry);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch = NULL;
    status =
        loom_low_br_build(&builder.ir, moved_entry, parameters,
                          moved_entry->arg_count, source->location, &branch);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_task_entry_materialize_body(
        &builder, &rewriter, lowering, body, state_arguments, success);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(&rewriter, function.op);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(module, physical,
                                        loom_op_vtable(module, physical));
    if (pass->function_version) {
      loom_function_version_update(pass->function_version,
                                   loom_func_like_cast(module, physical));
    }
    loom_pass_mark_changed(pass);
  }
  return status;
}

static iree_status_t loom_low_task_entry_build_query(
    loom_low_task_entry_builder_t* builder,
    const loom_low_task_entry_lowering_t* lowering,
    loom_symbol_ref_t query_symbol, loom_symbol_ref_t library_symbol,
    loom_string_id_t descriptor_key, loom_op_t** out_query) {
  const loom_type_t arguments[] = {builder->word_type, builder->pointer_type};
  loom_op_t* query = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &builder->ir,
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
          LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN |
          LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI,
      LOOM_LOW_VISIBILITY_PUBLIC, LOOM_LOW_RETAIN_RETAIN, 0, 0, 0, 0, 0,
      descriptor_key, loom_symbol_ref_null(), LOOM_TARGET_ABI_OBJECT_FUNCTION,
      loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), query_symbol,
      arguments, 2, &builder->pointer_type, 1, NULL, 0, NULL, 0,
      builder->location, &query));
  loom_region_t* body = loom_low_func_def_body(query);
  loom_builder_enter_region(&builder->ir, query, body);
  IREE_RETURN_IF_ERROR(lowering->emit_query(
      builder, body, library_symbol, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_8));
  *out_query = query;
  return iree_ok_status();
}

iree_status_t loom_low_task_materialize_query(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_task_entry_lowering_t* lowering, void* target_data) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  loom_function_version_owner_t* owner =
      loom_target_pass_capability_function_version_owner(capability);
  const loom_target_function_version_t* entry = NULL;
  for (iree_host_size_t i = 0; i < owner->list.count; ++i) {
    const loom_target_function_version_t* version =
        loom_target_function_version_const_cast(owner->list.values[i]);
    if (version &&
        version->function_target_facts->fact_type == lowering->fact_type &&
        (version->function_target_facts->storage.export_plan.abi_kind ==
             LOOM_TARGET_ABI_HAL_KERNEL ||
         iree_any_bit_set(version->base.function.vtable->flags,
                          LOOM_FUNC_LIKE_FLAG_KERNEL_ENTRY))) {
      entry = version;
      break;
    }
  }
  if (!entry) {
    return iree_ok_status();
  }
  loom_string_id_t query_name;
  loom_string_id_t library_name;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("iree_hal_executable_library_query"), &query_name));
  // Query definitions supplied in authored Low remain part of the program.
  if (loom_module_find_symbol(module, query_name) != LOOM_SYMBOL_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV(LOOM_TASK_LIBRARY_SYMBOL), &library_name));
  uint16_t query_id;
  uint16_t library_id = loom_module_find_symbol(module, library_name);
  const loom_low_descriptor_registry_t* registry =
      loom_low_pass_capability_descriptor_registry(
          loom_low_pass_capability_from_pass(pass));
  const iree_string_view_t descriptor_key =
      entry->function_target_facts->storage.config.contract_set_key;
  const loom_low_descriptor_set_t* descriptors =
      loom_low_descriptor_registry_lookup(registry, descriptor_key);
  loom_low_task_entry_builder_t builder;
  if (!loom_low_task_entry_builder_initialize(
          module, descriptors, entry->base.function.op->location, lowering,
          target_data, &builder)) {
    return loom_low_task_entry_reject(
        pass, entry->base.function.op,
        IREE_SV("task entry instructions for the selected representation"));
  }
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, query_name, &query_id));
  if (library_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, library_name, &library_id));
    loom_op_t* declaration = NULL;
    IREE_RETURN_IF_ERROR(loom_global_rodata_decl_build(
        &builder.ir, (loom_symbol_ref_t){.symbol_id = library_id},
        builder.location, &declaration));
  }
  loom_string_id_t descriptor_name;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, descriptor_key, &descriptor_name));
  loom_op_t* query = NULL;
  IREE_RETURN_IF_ERROR(loom_low_task_entry_build_query(
      &builder, lowering, (loom_symbol_ref_t){.symbol_id = query_id},
      (loom_symbol_ref_t){.symbol_id = library_id}, descriptor_name, &query));

  // The query is an ordinary callable in the entry's target context. Reapply
  // its own function contract instead of inheriting the entry's HAL overlay.
  loom_symbol_fact_table_t facts;
  loom_symbol_fact_table_initialize(&facts, pass->arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_symbol_fact_table_lookup(&facts, module, query_id, &base_facts));
  const loom_target_facts_t* query_facts = NULL;
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_facts(
      module, loom_func_symbol_facts_cast(base_facts),
      loom_target_facts_identity_name(entry->resolved_target.facts),
      entry->resolved_target.facts, pass->diagnostic_emitter, owner->arena,
      &valid, &query_facts));
  if (!valid) {
    return iree_ok_status();
  }
  loom_target_function_version_t* version = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(owner->arena, sizeof(*version), (void**)&version));
  *version = (loom_target_function_version_t){
      .base = {.type = &loom_target_function_version_type,
               .function = loom_func_like_cast(module, query),
               .flags = LOOM_FUNCTION_VERSION_FLAG_RETAIN},
      .resolved_target = entry->resolved_target,
      .target_context_ordinal = entry->target_context_ordinal,
      .function_target_facts = query_facts,
  };
  IREE_RETURN_IF_ERROR(
      loom_function_version_owner_append(owner, &version->base));
  loom_pass_mark_changed(pass);
  return iree_ok_status();
}
