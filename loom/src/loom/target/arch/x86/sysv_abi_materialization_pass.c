// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_abi_materialization_pass.h"

#include "loom/analysis/symbol_facts.h"
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
    "Number of x86 SysV ABI layouts materialized.")

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

    const loom_attribute_t abi_attr =
        loom_op_const_attrs(function_op)[abi_attr_index];
    const loom_attribute_t layout_attr =
        loom_op_const_attrs(function_op)[layout_attr_index];
    if (!loom_attr_is_absent(abi_attr) &&
        loom_attr_as_enum(abi_attr) != LOOM_TARGET_ABI_OBJECT_FUNCTION) {
      continue;
    }
    if (!loom_attr_is_absent(abi_attr) && !loom_attr_is_absent(layout_attr)) {
      continue;
    }

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
  }
  return iree_ok_status();
}
