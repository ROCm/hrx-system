// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/workgroup.h"

#include "loom/analysis/pipeline/resources.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/pass/builder.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/kernel/kernel_resources.h"

static const loom_pass_info_t kPipelineWorkgroupPassInfo = {
    .name = IREE_SVL("realize-pipeline-workgroup"),
    .description =
        IREE_SVL("Realize a cooperative strand as one workgroup kernel."),
    .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_pipeline_workgroup_pass_info(void) {
  return &kPipelineWorkgroupPassInfo;
}

static iree_status_t loom_pipeline_workgroup_reject(
    loom_pass_t* pass, const loom_op_t* operation,
    iree_string_view_t requirement) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = operation,
      .error = LOOM_ERR_LOWERING_072,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_pipeline_workgroup_apply(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_pipeline_resource_strand_t* strand) {
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  loom_builder_t* builder = &rewriter.builder;
  const loom_location_id_t location = function.op->location;
  loom_block_t* source = loom_region_entry_block(loom_func_like_body(function));
  loom_type_t* argument_types = NULL;
  iree_status_t status = iree_arena_allocate_array(
      pass->arena, source->arg_count, sizeof(*argument_types),
      (void**)&argument_types);
  for (uint16_t i = 0; iree_status_is_ok(status) && i < source->arg_count;
       ++i) {
    argument_types[i] = loom_module_value_type(module, source->arg_ids[i]);
  }
  const loom_symbol_ref_t symbol = loom_func_like_callee(function);
  const loom_symbol_ref_t target = loom_func_like_target(function);
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  loom_kernel_def_build_flags_t flags = 0;
  if (loom_symbol_ref_is_valid(target)) {
    flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET;
  }
  if (predicate_count) {
    flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_PREDICATES;
  }
  if (loom_func_like_visibility(function)) {
    flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL;
  }
  const bool retained = iree_any_bit_set(
      module->symbols.entries[symbol.symbol_id].flags, LOOM_SYMBOL_FLAG_RETAIN);
  if (retained) {
    flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_RETAIN;
  }
  loom_op_t* kernel = NULL;
  if (iree_status_is_ok(status)) {
    loom_builder_set_before(builder, function.op);
    status = loom_kernel_def_build(
        builder, flags, 0, retained ? LOOM_KERNEL_RETAIN_RETAIN : 0, target,
        module->symbols.entries[symbol.symbol_id].name_id, 0, symbol, NULL, 0,
        argument_types, source->arg_count, predicates, predicate_count,
        location, &kernel);
  }
  if (iree_status_is_ok(status)) {
    loom_builder_enter_region(builder, kernel, loom_kernel_def_config(kernel));
    loom_op_t *unit, *waves, *subgroup, *width, *launch;
    const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    status = loom_index_constant_build(builder, loom_attr_i64(1), index_type,
                                       location, &unit);
    if (iree_status_is_ok(status)) {
      status = loom_index_constant_build(
          builder, loom_attr_i64((int64_t)strand->axes[0].count), index_type,
          location, &waves);
    }
    if (iree_status_is_ok(status)) {
      status = loom_target_subgroup_size_build(builder, index_type, location,
                                               &subgroup);
    }
    if (iree_status_is_ok(status)) {
      status = loom_index_mul_build(builder, loom_index_constant_result(waves),
                                    loom_target_subgroup_size_result(subgroup),
                                    location, &width);
    }
    if (iree_status_is_ok(status)) {
      const loom_value_id_t one = loom_index_constant_result(unit);
      status = loom_kernel_launch_config_build(
          builder, 0, one, one, one, loom_index_mul_result(width), one, one,
          LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
          location, &launch);
    }
  }
  loom_block_t* destination = NULL;
  if (iree_status_is_ok(status)) {
    destination = loom_region_entry_block(loom_kernel_def_body(kernel));
    status = loom_module_assign_value_types(
        module, source->arg_ids, destination->arg_ids, source->arg_count);
  }
  for (uint16_t i = 0; iree_status_is_ok(status) && i < source->arg_count;
       ++i) {
    status = loom_module_copy_value_name(module, source->arg_ids[i],
                                         destination->arg_ids[i]);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_replace_all_uses_with(
          &rewriter, source->arg_ids[i], destination->arg_ids[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    // Consume the retained declaration directly. Its body is already an
    // ordinary call; the shared inliner owns its subsequent legalization.
    status =
        loom_rewriter_move_before(&rewriter, strand->call, strand->declaration);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(&rewriter, strand->declaration);
  }
  while (iree_status_is_ok(status) && source->first_op) {
    loom_op_t* operation = source->first_op;
    if (loom_pipeline_finish_isa(operation) ||
        loom_pipeline_strand_isa(operation)) {
      // Construction analysis omits empty strands; they have no invocation.
      status = loom_rewriter_erase(&rewriter, operation);
    } else {
      status = loom_rewriter_move_to_block_end(&rewriter, operation,
                                               destination, kernel);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_builder_enter_region(builder, kernel, loom_kernel_def_body(kernel));
    loom_op_t* terminator;
    status = loom_kernel_return_build(builder, location, &terminator);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t count = 0;
    const iree_string_view_t* comments =
        loom_module_op_comments(module, function.op, &count);
    if (count) {
      status = loom_module_attach_op_comments(module, kernel, comments, count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(&rewriter, function.op);
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(module, kernel,
                                        loom_op_vtable(module, kernel));
    if (pass->function_version) {
      loom_function_version_update(pass->function_version,
                                   loom_func_like_cast(module, kernel));
    }
    loom_target_function_version_t* version =
        loom_target_function_version_cast(pass->function_version);
    if (version) {
      // The replacement owns a HAL-kernel ABI, independently of the source
      // construction's ABI. Launch size remains derived from its config IR.
      const iree_string_view_t name = loom_string_table_get(
          &module->strings, module->symbols.entries[symbol.symbol_id].name_id);
      const loom_func_symbol_facts_t contract = {
          .func_op = kernel,
          .name = name,
          .has_abi = true,
          .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
          .export_symbol =
              (flags & LOOM_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL)
                  ? name
                  : iree_string_view_empty(),
      };
      loom_function_version_owner_t* owner =
          loom_target_pass_capability_function_version_owner(
              loom_target_pass_capability_from_pass(pass));
      bool valid = false;
      status = loom_target_function_contract_refine_internal_facts(
          module, &contract,
          loom_target_facts_identity_name(version->resolved_target.facts),
          version->resolved_target.facts, pass->diagnostic_emitter,
          owner->arena, &valid, &version->function_target_facts);
    }
    if (iree_status_is_ok(status)) {
      uint16_t normalized_count = 0;
      status = loom_kernel_resources_normalize(
          module, loom_func_like_cast(module, kernel), &normalized_count);
    }
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_pipeline_workgroup_run(loom_pass_t* pass,
                                          loom_module_t* module,
                                          loom_func_like_t function) {
  if (!loom_pipeline_def_isa(function.op)) {
    return iree_ok_status();
  }
  if (loom_pipeline_def_scope(function.op) != LOOM_PIPELINE_SCOPE_KERNEL ||
      loom_func_like_specialization_count(function) != 0) {
    return loom_pipeline_workgroup_reject(
        pass, function.op, IREE_SV("a specialized kernel-scope pipeline"));
  }
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts));
  loom_pipeline_resources_t resources;
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
      module, function, facts, NULL, 0, NULL, 0, NULL, 0,
      pass->diagnostic_emitter, pass->arena, &resources, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  if (resources.composition_count || resources.schedule_count ||
      resources.channel_count) {
    return loom_pipeline_workgroup_reject(
        pass, function.op,
        IREE_SV("realized child control and channel bindings"));
  }
  if (resources.strand_count != 1) {
    return loom_pipeline_workgroup_reject(pass, function.op,
                                          IREE_SV("one cooperative strand"));
  }
  const loom_pipeline_resource_strand_t* strand = &resources.strands[0];
  if (loom_pipeline_strand_participation(strand->declaration) !=
      LOOM_PIPELINE_STRAND_PARTICIPATION_COOPERATIVE) {
    return loom_pipeline_workgroup_reject(
        pass, strand->declaration, IREE_SV("cooperative strand participation"));
  }
  if (strand->rank != 1 || strand->axes[0].origin != 0 ||
      strand->axes[0].stride != 1 || strand->axes[0].count == 0) {
    return loom_pipeline_workgroup_reject(
        pass, strand->declaration,
        IREE_SV("a nonempty contiguous wave domain starting at zero"));
  }
  const loom_target_facts_t* entry_target =
      loom_value_fact_table_block_target_facts(
          facts, loom_region_entry_block(loom_func_like_body(function)));
  const loom_target_facts_t* strand_target =
      loom_value_fact_table_block_target_facts(
          facts, loom_region_entry_block(
                     loom_pipeline_strand_body(strand->declaration)));
  if (entry_target && strand_target &&
      !loom_target_facts_satisfy_specialization_requirement(entry_target,
                                                            strand_target)) {
    return loom_pipeline_workgroup_reject(
        pass, strand->declaration,
        IREE_SV("the enclosing kernel's execution target"));
  }
  return loom_pipeline_workgroup_apply(pass, module, function, strand);
}

static iree_status_t loom_pipeline_workgroup_build_run(loom_builder_t* builder,
                                                       void* user_data) {
  (void)user_data;
  loom_op_t* run;
  return loom_pass_ir_build_run(builder, 0,
                                IREE_SV("realize-pipeline-workgroup"),
                                loom_named_attr_slice_empty(), &run);
}

static iree_status_t loom_pipeline_workgroup_build_where(
    loom_builder_t* builder, void* user_data) {
  const loom_named_attr_t* family = user_data;
  loom_op_t* where;
  return loom_pass_ir_build_where(
      builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS, IREE_SV("target"),
      loom_make_named_attr_slice(family, 1), loom_pipeline_workgroup_build_run,
      NULL, &where);
}

iree_status_t loom_pipeline_workgroup_build_pipeline(
    loom_builder_t* builder, iree_string_view_t family) {
  loom_string_id_t key, value;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, IREE_SV("family"), &key));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, family, &value));
  loom_named_attr_t attribute = {.name_id = key,
                                 .value = loom_attr_string(value)};
  loom_op_t* for_op;
  return loom_pass_ir_build_for(builder, LOOM_PASS_ANCHOR_FUNC,
                                loom_pipeline_workgroup_build_where, &attribute,
                                &for_op);
}
