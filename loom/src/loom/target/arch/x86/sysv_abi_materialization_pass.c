// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_abi_materialization_pass.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/sysv_abi.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"

#define LOOM_X86_SYSV_ABI_MATERIALIZATION_STATISTICS(V, statistics_type) \
  V(statistics_type, functions, "functions",                             \
    "Number of x86 function symbols observed.")                          \
  V(statistics_type, layouts, "layouts",                                 \
    "Number of x86 SysV ABI layouts materialized.")                      \
  V(statistics_type, stack_argument_loads, "stack_argument_loads",       \
    "Number of incoming stack argument loads materialized.")             \
  V(statistics_type, stack_argument_stores, "stack_argument_stores",     \
    "Number of outgoing stack argument stores materialized.")

LOOM_PASS_STATISTICS_DEFINE(loom_x86_sysv_abi_materialization_statistics,
                            loom_x86_sysv_abi_materialization_statistics_t,
                            LOOM_X86_SYSV_ABI_MATERIALIZATION_STATISTICS)

static const loom_pass_info_t loom_x86_materialize_sysv_abi_pass_info_storage =
    {
        .name = IREE_SVL("x86-materialize-sysv-abi"),
        .description =
            IREE_SVL("Materialize x86 SysV target-low function layouts."),
        .kind = LOOM_PASS_MODULE,
        .statistic_layout =
            &loom_x86_sysv_abi_materialization_statistics_layout,
};

const loom_pass_info_t* loom_x86_materialize_sysv_abi_pass_info(void) {
  return &loom_x86_materialize_sysv_abi_pass_info_storage;
}

static bool loom_x86_sysv_abi_function_attr_indices(
    const loom_op_t* function_op, uint8_t* out_abi_attr_index,
    uint8_t* out_layout_attr_index) {
  if (loom_low_func_def_isa(function_op)) {
    *out_abi_attr_index = loom_low_func_def_abi_ATTR_INDEX;
    *out_layout_attr_index = loom_low_func_def_abi_layout_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(function_op)) {
    *out_abi_attr_index = loom_low_func_decl_abi_ATTR_INDEX;
    *out_layout_attr_index = loom_low_func_decl_abi_layout_ATTR_INDEX;
    return true;
  }
  return false;
}

static bool loom_x86_sysv_abi_function_uses_x86_representation(
    const loom_module_t* module, loom_func_like_t function) {
  const loom_string_id_t descriptor_set_id =
      loom_func_like_repr_contract(function);
  return descriptor_set_id < module->strings.count &&
         iree_string_view_starts_with(
             module->strings.entries[descriptor_set_id], IREE_SV("x86."));
}

static bool loom_x86_sysv_abi_target_matches(
    const loom_low_resolved_target_t* target) {
  return target->descriptor_set != NULL && target->target_facts != NULL &&
         target->target_facts->fact_type == &loom_x86_target_fact_type;
}

typedef struct loom_x86_sysv_abi_function_t {
  // Target-low function definition or declaration.
  loom_op_t* op;
  // Canonical argument locations borrowing the retained ABI attribute.
  const int64_t* argument_locations;
  // Number of entries in |argument_locations|.
  uint16_t argument_count;
  // Bytes occupied by stack-classified arguments.
  uint32_t stack_argument_bytes;
} loom_x86_sysv_abi_function_t;

static iree_status_t loom_x86_sysv_abi_materialize_function(
    loom_module_t* module, loom_func_like_t function,
    const loom_low_descriptor_set_t* descriptor_set, uint8_t abi_attr_index,
    uint8_t layout_attr_index, iree_arena_allocator_t* arena, bool* out_changed,
    bool* out_layout_materialized) {
  *out_changed = false;
  *out_layout_materialized = false;
  const loom_attribute_t abi_attr =
      loom_op_const_attrs(function.op)[abi_attr_index];
  const loom_attribute_t layout_attr =
      loom_op_const_attrs(function.op)[layout_attr_index];
  const bool abi_present = !loom_attr_is_absent(abi_attr);
  const bool layout_present = !loom_attr_is_absent(layout_attr);
  if (abi_present &&
      loom_attr_as_enum(abi_attr) != LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    return iree_ok_status();
  }
  if (abi_present && layout_present) {
    return iree_ok_status();
  }

  loom_attribute_t materialized_layout_attr = loom_attr_absent();
  if (!layout_present) {
    uint16_t argument_count = 0;
    const loom_value_id_t* argument_ids =
        loom_func_like_arg_ids(function, &argument_count);
    const uint16_t result_count = function.op->result_count;
    const loom_value_id_t* result_ids = loom_op_const_results(function.op);
    loom_type_t* argument_types = NULL;
    loom_type_t* result_types = NULL;
    if (argument_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, argument_count,
                                                     sizeof(*argument_types),
                                                     (void**)&argument_types));
      for (uint16_t i = 0; i < argument_count; ++i) {
        argument_types[i] = loom_module_value_type(module, argument_ids[i]);
      }
    }
    if (result_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, result_count, sizeof(*result_types), (void**)&result_types));
      for (uint16_t i = 0; i < result_count; ++i) {
        result_types[i] = loom_module_value_type(module, result_ids[i]);
      }
    }

    loom_x86_sysv_abi_layout_t layout = {0};
    bool supported = false;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_layout_build(
        descriptor_set, argument_types, argument_count, result_types,
        result_count, arena, &layout, &supported));
    if (!supported) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "x86 SysV ABI does not yet support this target-low signature");
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_layout_make_attr(
        module, &layout, &materialized_layout_attr));
  }

  if (!layout_present) {
    IREE_RETURN_IF_ERROR(loom_op_set_attr(
        module, function.op, layout_attr_index, materialized_layout_attr));
    *out_layout_materialized = true;
  }
  if (!abi_present) {
    IREE_RETURN_IF_ERROR(
        loom_op_set_attr(module, function.op, abi_attr_index,
                         loom_attr_enum(LOOM_TARGET_ABI_OBJECT_FUNCTION)));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_abi_materialize_incoming_arguments(
    loom_module_t* module, loom_builder_t* builder,
    const loom_x86_sysv_abi_function_t* function,
    loom_x86_sysv_abi_materialization_statistics_t* statistics,
    bool* out_changed) {
  if (!loom_low_func_def_isa(function->op)) {
    return iree_ok_status();
  }
  const loom_func_like_t function_like =
      loom_func_like_cast(module, function->op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function_like, &argument_count);
  IREE_ASSERT_EQ(argument_count, function->argument_count);
  for (uint16_t ordinal = 0; ordinal < argument_count; ++ordinal) {
    if (loom_x86_sysv_abi_location_is_register(
            function->argument_locations[ordinal])) {
      continue;
    }
    const loom_value_id_t argument_id = argument_ids[ordinal];
    loom_value_t* argument = loom_module_value(module, argument_id);
    const loom_type_t argument_type = argument->type;
    while (argument->use_count != 0) {
      const loom_use_t use = loom_value_uses(argument)[argument->use_count - 1];
      loom_op_t* user_op = loom_use_user_op(use);
      loom_builder_set_before(builder, user_op);
      loom_op_t* stack_arg_op = NULL;
      const uint32_t byte_offset = loom_x86_sysv_abi_stack_location_offset(
          function->argument_locations[ordinal]);
      IREE_RETURN_IF_ERROR(loom_low_func_stack_arg_build(
          builder, ordinal, byte_offset, argument_type, user_op->location,
          &stack_arg_op));
      IREE_RETURN_IF_ERROR(
          loom_op_set_operand(module, user_op, loom_use_operand_index(use),
                              loom_low_func_stack_arg_result(stack_arg_op)));
      ++statistics->stack_argument_loads;
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_abi_copy_call_comments(
    loom_module_t* module, const loom_op_t* source_op,
    const loom_op_t* target_op) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, source_op, &comment_count);
  return loom_module_attach_op_comments(module, target_op, comments,
                                        comment_count);
}

static iree_status_t loom_x86_sysv_abi_materialize_outgoing_arguments(
    loom_module_t* module, loom_builder_t* builder,
    iree_arena_allocator_t* arena, loom_op_t* call_op,
    const loom_x86_sysv_abi_function_t* callee,
    loom_x86_sysv_abi_materialization_statistics_t* statistics,
    bool* out_changed) {
  const loom_value_slice_t old_stack_args =
      loom_low_func_call_stack_args(call_op);
  if (old_stack_args.count != 0 || callee->stack_argument_bytes == 0) {
    return iree_ok_status();
  }
  const loom_value_slice_t old_arguments = loom_low_func_call_operands(call_op);
  IREE_ASSERT_EQ(old_arguments.count, callee->argument_count);

  iree_host_size_t register_count = 0;
  for (iree_host_size_t i = 0; i < callee->argument_count; ++i) {
    register_count +=
        loom_x86_sysv_abi_location_is_register(callee->argument_locations[i]);
  }
  const iree_host_size_t stack_count = callee->argument_count - register_count;
  loom_value_id_t* partitioned_arguments = NULL;
  if (callee->argument_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, callee->argument_count, sizeof(*partitioned_arguments),
        (void**)&partitioned_arguments));
  }
  loom_value_id_t* register_arguments = partitioned_arguments;
  loom_value_id_t* stack_arguments = partitioned_arguments + register_count;

  loom_builder_set_before(builder, call_op);
  iree_host_size_t register_index = 0;
  iree_host_size_t stack_index = 0;
  for (iree_host_size_t ordinal = 0; ordinal < callee->argument_count;
       ++ordinal) {
    const loom_value_id_t argument = old_arguments.values[ordinal];
    if (loom_x86_sysv_abi_location_is_register(
            callee->argument_locations[ordinal])) {
      register_arguments[register_index++] = argument;
      continue;
    }
    loom_op_t* call_arg_op = NULL;
    const uint32_t byte_offset = loom_x86_sysv_abi_stack_location_offset(
        callee->argument_locations[ordinal]);
    IREE_RETURN_IF_ERROR(loom_low_func_call_arg_build(
        builder, loom_low_func_call_callee(call_op), (int64_t)ordinal,
        byte_offset, argument, loom_type_storage(LOOM_STORAGE_SPACE_STACK),
        call_op->location, &call_arg_op));
    stack_arguments[stack_index++] = loom_low_func_call_arg_token(call_arg_op);
    ++statistics->stack_argument_stores;
  }
  IREE_ASSERT_EQ(register_index, register_count);
  IREE_ASSERT_EQ(stack_index, stack_count);

  loom_low_func_call_build_flags_t build_flags = 0;
  uint8_t purity = 0;
  uint8_t inline_policy = 0;
  const loom_attribute_t* attrs = loom_op_const_attrs(call_op);
  if (!loom_attr_is_absent(attrs[loom_low_func_call_purity_ATTR_INDEX])) {
    build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
    purity = loom_low_func_call_purity(call_op);
  }
  if (!loom_attr_is_absent(
          attrs[loom_low_func_call_inline_policy_ATTR_INDEX])) {
    build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY;
    inline_policy = loom_low_func_call_inline_policy(call_op);
  }
  loom_type_t result_types[1] = {0};
  const loom_value_id_t* old_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < call_op->result_count; ++i) {
    IREE_ASSERT_LT(i, IREE_ARRAYSIZE(result_types));
    result_types[i] = loom_module_value_type(module, old_results[i]);
  }
  loom_op_t* replacement_call = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_call_build(
      builder, build_flags, purity, inline_policy,
      loom_low_func_call_callee(call_op), register_arguments, register_count,
      stack_arguments, stack_count, result_types, call_op->result_count,
      loom_op_tied_results(call_op), call_op->tied_result_count,
      call_op->location, &replacement_call));
  IREE_RETURN_IF_ERROR(
      loom_x86_sysv_abi_copy_call_comments(module, call_op, replacement_call));

  const loom_value_id_t* replacement_results =
      loom_op_const_results(replacement_call);
  for (uint16_t i = 0; i < call_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(module, old_results[i],
                                                     replacement_results[i]));
    IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
        module, old_results[i], replacement_results[i]));
  }
  IREE_RETURN_IF_ERROR(loom_op_erase(module, call_op));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_abi_materialize_boundaries(
    loom_pass_t* pass, loom_module_t* module,
    const loom_x86_sysv_abi_function_t* functions,
    loom_x86_sysv_abi_materialization_statistics_t* statistics) {
  loom_symbol_reference_table_t references = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, pass->arena, &references));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  bool changed = false;
  for (iree_host_size_t i = 0; i < references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &references.occurrences[i];
    if (occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
        occurrence->source_symbol_id >= references.symbol_count ||
        occurrence->target_symbol_id >= references.symbol_count ||
        functions[occurrence->source_symbol_id].op == NULL ||
        functions[occurrence->target_symbol_id].op == NULL ||
        !loom_low_func_call_isa(occurrence->user_op)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_materialize_outgoing_arguments(
        module, &builder, pass->arena, (loom_op_t*)occurrence->user_op,
        &functions[occurrence->target_symbol_id], statistics, &changed));
  }
  for (loom_symbol_id_t symbol_id = 0; symbol_id < references.symbol_count;
       ++symbol_id) {
    if (functions[symbol_id].op == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_materialize_incoming_arguments(
        module, &builder, &functions[symbol_id], statistics, &changed));
  }
  if (changed) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}

iree_status_t loom_x86_materialize_sysv_abi_run(loom_pass_t* pass,
                                                loom_module_t* module) {
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_function_version_list_t* function_versions =
      loom_target_pass_capability_function_versions(target_capability);

  loom_target_function_version_snapshot_t versions = {0};
  bool versions_built = false;
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_x86_sysv_abi_materialization_statistics_t* statistics =
      loom_x86_sysv_abi_materialization_statistics(pass);
  loom_x86_sysv_abi_function_t* functions = NULL;
  bool has_stack_arguments = false;

  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    loom_op_t* function_op = module->symbols.entries[symbol_id].defining_op;
    uint8_t abi_attr_index = LOOM_ATTR_INDEX_NONE;
    uint8_t layout_attr_index = LOOM_ATTR_INDEX_NONE;
    if (!loom_x86_sysv_abi_function_attr_indices(function_op, &abi_attr_index,
                                                 &layout_attr_index)) {
      continue;
    }
    loom_func_like_t function = loom_func_like_cast(module, function_op);
    if (!loom_x86_sysv_abi_function_uses_x86_representation(module, function)) {
      continue;
    }
    ++statistics->functions;

    if (!versions_built) {
      IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
          module, function_versions, pass->arena, &versions));
      versions_built = true;
    }
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(&versions, symbol_id);
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, &symbol_facts, function_op,
        version != NULL ? version->function_target_facts : NULL,
        descriptor_registry, pass->diagnostic_emitter, &target));
    if (!loom_x86_sysv_abi_target_matches(&target)) {
      continue;
    }

    bool changed = false;
    bool layout_materialized = false;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_materialize_function(
        module, function, target.descriptor_set, abi_attr_index,
        layout_attr_index, pass->arena, &changed, &layout_materialized));
    if (layout_materialized) {
      ++statistics->layouts;
    }
    if (changed) {
      loom_pass_mark_changed(pass);
    }
    const loom_attribute_t abi_attr =
        loom_op_const_attrs(function_op)[abi_attr_index];
    const loom_attribute_t layout_attr =
        loom_op_const_attrs(function_op)[layout_attr_index];
    if (loom_attr_is_absent(abi_attr) || loom_attr_is_absent(layout_attr) ||
        loom_attr_as_enum(abi_attr) != LOOM_TARGET_ABI_OBJECT_FUNCTION) {
      continue;
    }
    loom_x86_sysv_abi_layout_t layout = {0};
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_function_layout_parse(
        module, target.descriptor_set, function_op, pass->arena, &layout));
    if (functions == NULL) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(pass->arena, module->symbols.count,
                                    sizeof(*functions), (void**)&functions));
      memset(functions, 0, module->symbols.count * sizeof(*functions));
    }
    IREE_ASSERT_LE(layout.argument_count, UINT16_MAX);
    functions[symbol_id] = (loom_x86_sysv_abi_function_t){
        .op = function_op,
        .argument_locations = layout.argument_locations,
        .argument_count = (uint16_t)layout.argument_count,
        .stack_argument_bytes = layout.stack_argument_bytes,
    };
    has_stack_arguments |= layout.stack_argument_bytes != 0;
  }
  if (!has_stack_arguments) {
    return iree_ok_status();
  }
  return loom_x86_sysv_abi_materialize_boundaries(pass, module, functions,
                                                  statistics);
}
