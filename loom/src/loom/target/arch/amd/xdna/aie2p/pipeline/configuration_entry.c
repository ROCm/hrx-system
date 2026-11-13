// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/configuration_entry.h"

#include "loom/codegen/low/launch_config_program.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amd/xdna/aie2p/array/abi_layout.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/function_version.h"

static iree_status_t loom_aie2p_configuration_entry_reject(
    loom_pass_t* pass, const loom_op_t* op, iree_string_view_t requirement) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

iree_status_t loom_aie2p_configuration_entry_realize(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (loom_kernel_workload_arg_ids(module, function.op).count != 0) {
    return loom_aie2p_configuration_entry_reject(
        pass, function.op,
        IREE_SV("a configuration kernel without workload arguments"));
  }
  uint8_t export_linkage = 0;
  uint16_t predicate_count = 0;
  loom_func_like_predicates(function, &predicate_count);
  if (loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID ||
      loom_func_like_export_linkage(function, &export_linkage) ||
      predicate_count != 0) {
    return loom_aie2p_configuration_entry_reject(
        pass, function.op,
        IREE_SV("a specialized configuration kernel with default export "
                "name and linkage"));
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body->block_count != 1 ||
      !loom_kernel_return_isa(loom_region_entry_block(body)->last_op)) {
    return loom_aie2p_configuration_entry_reject(
        pass, function.op,
        IREE_SV("a single-block result-free configuration kernel"));
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_type_is_buffer(loom_module_value_type(module, arguments[i])) ||
        loom_module_value_has_uses(module, arguments[i])) {
      return loom_aie2p_configuration_entry_reject(
          pass, function.op,
          IREE_SV("buffer arguments addressed through configuration binding "
                  "ordinals"));
    }
  }
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts));
  loom_target_dispatch_workgroup_count_t count = {0};
  loom_target_workgroup_size_t size = {0};
  if (!loom_kernel_def_static_workgroup_count_from_facts(module, function.op,
                                                         facts, &count) ||
      !loom_kernel_def_static_workgroup_size_from_facts(module, function.op,
                                                        facts, &size) ||
      count.x != 1 || count.y != 1 || count.z != 1 || size.x != 1 ||
      size.y != 1 || size.z != 1) {
    return loom_aie2p_configuration_entry_reject(
        pass, function.op,
        IREE_SV("a static one-workgroup, one-workitem configuration launch"));
  }
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(function.op);
  if (loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    loom_target_workgroup_cluster_size_t cluster = {0};
    if (!loom_kernel_def_static_workgroup_cluster_size_from_facts(
            module, function.op, facts, &cluster) ||
        cluster.x != 1 || cluster.y != 1 || cluster.z != 1) {
      return loom_aie2p_configuration_entry_reject(
          pass, function.op, IREE_SV("a unit configuration workgroup cluster"));
    }
  }

  const loom_symbol_ref_t symbol = loom_func_like_callee(function);
  loom_kernel_launch_config_program_t* launch_program =
      loom_kernel_launch_config_program_from_pass(pass);
  if (launch_program) {
    const loom_kernel_launch_config_t configuration = {
        .workgroup_count = {1, 1, 1},
        .workgroup_size = {1, 1, 1},
        .workgroup_cluster_size = {1, 1, 1},
        .subgroup_size = 1,
    };
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_program_capture_static(
        launch_program,
        loom_string_table_get(
            &module->strings,
            module->symbols.entries[symbol.symbol_id].name_id),
        &configuration));
  }
  loom_string_id_t contract;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("amd.xdna.aie2p.configuration"), &contract));
  loom_attribute_t layout;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_abi_layout_make_attr(module, argument_count, &layout));
  const bool retained = iree_any_bit_set(
      module->symbols.entries[symbol.symbol_id].flags, LOOM_SYMBOL_FLAG_RETAIN);
  const uint8_t visibility = loom_func_like_visibility(function);
  const loom_symbol_ref_t target = loom_func_like_target(function);
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  loom_builder_set_before(&rewriter.builder, function.op);
  loom_op_t* entry = NULL;
  iree_status_t status = loom_low_func_def_build(
      &rewriter.builder,
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI |
          LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI_LAYOUT |
          (visibility ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY : 0) |
          (retained ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN : 0) |
          (loom_symbol_ref_is_valid(target)
               ? LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET
               : 0),
      visibility, retained ? LOOM_LOW_RETAIN_RETAIN : 0, 0, 0, 0, 0, 0, 0,
      contract, target, LOOM_TARGET_ABI_ARRAY_PROGRAM,
      loom_named_attr_slice_empty(), loom_attr_as_dict(layout),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, NULL, 0,
      NULL, 0, NULL, 0, NULL, 0, function.op->location, &entry);
  loom_block_t* source = loom_region_entry_block(body);
  loom_op_t* terminator = source->last_op;
  if (iree_status_is_ok(status)) {
    iree_host_size_t comment_count = 0;
    const iree_string_view_t* comments =
        loom_module_op_comments(module, function.op, &comment_count);
    if (comment_count != 0) {
      status = loom_module_attach_op_comments(module, entry, comments,
                                              comment_count);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_region_t* entry_body = loom_low_func_def_body(entry);
    loom_block_t* destination = loom_region_entry_block(entry_body);
    for (loom_op_t* op = source->first_op;
         op != terminator && iree_status_is_ok(status);) {
      loom_op_t* next = op->next_op;
      status =
          loom_rewriter_move_to_block_end(&rewriter, op, destination, entry);
      op = next;
    }
    if (iree_status_is_ok(status)) {
      loom_builder_enter_region(&rewriter.builder, entry, entry_body);
      loom_op_t* return_op = NULL;
      status = loom_low_return_build(&rewriter.builder, NULL, 0,
                                     terminator->location, &return_op);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(&rewriter, function.op);
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(module, entry,
                                        loom_op_vtable(module, entry));
    loom_function_version_update(pass->function_version,
                                 loom_func_like_cast(module, entry));
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
