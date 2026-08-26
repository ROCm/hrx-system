// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/resolve_launches.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

typedef struct loom_kernel_launch_resolution_t {
  // Module being transformed.
  loom_module_t* module;
  // Dense configuration function projection by logical-kernel symbol ID.
  const loom_symbol_ref_t* configuration_functions;
  // Number of entries in |configuration_functions|.
  iree_host_size_t configuration_function_count;
  // Dense derived entry refs by logical-kernel symbol ID.
  loom_symbol_ref_t* entry_refs;
  // Diagnostic sink for authored contract failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Scratch storage shared by type remapping and dense tables.
  iree_arena_allocator_t* scratch_arena;
  // Preparation-wide old declaration argument to new entry argument remap.
  loom_ir_remap_t type_remap;
  // Rewriter maintaining use lists and module summaries.
  loom_rewriter_t rewriter;
} loom_kernel_launch_resolution_t;

static iree_string_view_t loom_kernel_launch_resolution_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_string_id_t name_id =
      module->symbols.entries[symbol_ref.symbol_id].name_id;
  IREE_ASSERT_LT(name_id, module->strings.count);
  return module->strings.entries[name_id];
}

static iree_status_t loom_kernel_launch_resolution_emit_missing_configuration(
    const loom_kernel_launch_resolution_t* resolution,
    const loom_op_t* declaration_op, const loom_op_t* launch_op) {
  const loom_symbol_ref_t callee = loom_kernel_decl_callee(declaration_op);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_launch_resolution_symbol_name(
          resolution->module, callee)),
  };
  const loom_diagnostic_related_op_t related_op = {
      .label = IREE_SV("launched here"),
      .module = resolution->module,
      .op = launch_op,
  };
  const loom_diagnostic_emission_t emission = {
      .module = resolution->module,
      .op = declaration_op,
      .error = LOOM_ERR_LOWERING_050,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = &related_op,
      .related_op_count = 1,
  };
  return iree_diagnostic_emit(resolution->diagnostic_emitter, &emission);
}

static iree_status_t loom_kernel_launch_resolution_build_entry(
    loom_kernel_launch_resolution_t* resolution, loom_op_t* declaration_op,
    loom_symbol_ref_t* out_entry_ref) {
  *out_entry_ref = loom_symbol_ref_null();
  IREE_ASSERT(loom_kernel_decl_isa(declaration_op));
  const loom_symbol_ref_t source_ref = loom_kernel_decl_callee(declaration_op);
  const loom_value_slice_t source_arguments =
      loom_kernel_decl_args(declaration_op);

  loom_type_t* argument_types = NULL;
  if (source_arguments.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        resolution->scratch_arena, source_arguments.count,
        sizeof(*argument_types), (void**)&argument_types));
    for (uint16_t i = 0; i < source_arguments.count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }

  loom_kernel_entry_decl_build_flags_t build_flags = 0;
  const loom_symbol_ref_t target = loom_kernel_decl_target(declaration_op);
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_KERNEL_ENTRY_DECL_BUILD_FLAG_HAS_TARGET;
  }
  loom_builder_t builder;
  loom_builder_initialize(resolution->module, &resolution->module->arena,
                          loom_module_block(resolution->module), &builder);
  loom_builder_set_before(&builder, declaration_op);
  loom_op_t* entry_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_entry_decl_build(
      &builder, build_flags, /*retain=*/0, target, source_ref, argument_types,
      source_arguments.count, declaration_op->location, &entry_op));

  const loom_value_slice_t entry_arguments =
      loom_kernel_entry_decl_args(entry_op);
  IREE_ASSERT_EQ(source_arguments.count, entry_arguments.count);
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&resolution->type_remap, source_arguments.values,
                               entry_arguments.values, source_arguments.count));
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(resolution->module, source_arguments.values[i]);
    loom_type_t entry_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&resolution->type_remap, source_type, &entry_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        resolution->module, entry_arguments.values[i], entry_type));
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        resolution->module, source_arguments.values[i],
        entry_arguments.values[i]));
  }

  // The executable entry replaces the logical declaration under the same
  // symbol identity. This keeps every dispatch and plan projection indexed by
  // one stable symbol row instead of manufacturing a duplicate textual name.
  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase(&resolution->rewriter, declaration_op));
  loom_module_link_symbol_defining_op(
      resolution->module, entry_op,
      loom_op_vtable(resolution->module, entry_op));

  *out_entry_ref = source_ref;
  return iree_ok_status();
}

static bool loom_kernel_launch_resolution_is_schedule(const loom_op_t* op) {
  return loom_kernel_launch_serial_isa(op) ||
         loom_kernel_launch_concurrent_isa(op);
}

// Returns the earliest schedule wrapper containing |launch_op|. Ordinary CFG
// parents stop the ascent so the configuration remains branch/loop local.
static loom_op_t* loom_kernel_launch_resolution_call_insertion_op(
    loom_op_t* launch_op) {
  loom_op_t* insertion_op = launch_op;
  while (insertion_op->parent_op &&
         loom_kernel_launch_resolution_is_schedule(insertion_op->parent_op)) {
    insertion_op = insertion_op->parent_op;
  }
  return insertion_op;
}

static iree_status_t loom_kernel_launch_resolution_rewrite_launch(
    loom_kernel_launch_resolution_t* resolution, loom_op_t* launch_op,
    bool* out_valid) {
  *out_valid = false;
  IREE_ASSERT(loom_kernel_launch_isa(launch_op));
  const loom_symbol_ref_t kernel_ref = loom_kernel_launch_callee(launch_op);
  IREE_ASSERT(loom_symbol_ref_is_valid(kernel_ref));
  IREE_ASSERT_EQ(kernel_ref.module_id, 0u);
  IREE_ASSERT_LT(kernel_ref.symbol_id, resolution->module->symbols.count);
  if (kernel_ref.symbol_id >= resolution->configuration_function_count ||
      !loom_symbol_ref_is_valid(
          resolution->configuration_functions[kernel_ref.symbol_id])) {
    const loom_op_t* defining_op =
        resolution->module->symbols.entries[kernel_ref.symbol_id].defining_op;
    IREE_ASSERT(defining_op != NULL);
    IREE_ASSERT(loom_kernel_decl_isa(defining_op));
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_resolution_emit_missing_configuration(
            resolution, defining_op, launch_op));
    return iree_ok_status();
  }
  const loom_symbol_ref_t configuration_ref =
      resolution->configuration_functions[kernel_ref.symbol_id];
  IREE_ASSERT_EQ(configuration_ref.module_id, 0u);
  IREE_ASSERT_LT(configuration_ref.symbol_id,
                 resolution->module->symbols.count);
  const loom_op_t* configuration_op =
      resolution->module->symbols.entries[configuration_ref.symbol_id]
          .defining_op;
  IREE_ASSERT(configuration_op != NULL);
  IREE_ASSERT(loom_func_def_isa(configuration_op));

  loom_symbol_ref_t entry_ref = resolution->entry_refs[kernel_ref.symbol_id];
  if (!loom_symbol_ref_is_valid(entry_ref)) {
    loom_op_t* declaration_op =
        resolution->module->symbols.entries[kernel_ref.symbol_id].defining_op;
    IREE_ASSERT(declaration_op != NULL);
    IREE_ASSERT(loom_kernel_decl_isa(declaration_op));
    IREE_RETURN_IF_ERROR(loom_kernel_launch_resolution_build_entry(
        resolution, declaration_op, &entry_ref));
    resolution->entry_refs[kernel_ref.symbol_id] = entry_ref;
  }

  const loom_value_slice_t workloads = loom_kernel_launch_workloads(launch_op);
  const loom_type_t count_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_builder_t* builder = &resolution->rewriter.builder;
  loom_builder_set_before(
      builder, loom_kernel_launch_resolution_call_insertion_op(launch_op));
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      builder,
      LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY |
          LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, LOOM_INLINE_POLICY_INLINE,
      configuration_ref, workloads.values, workloads.count, count_types,
      IREE_ARRAYSIZE(count_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, launch_op->location, &call_op));

  const loom_value_slice_t counts = loom_func_call_results(call_op);
  IREE_ASSERT_EQ(counts.count, IREE_ARRAYSIZE(count_types));
  const loom_value_slice_t arguments = loom_kernel_launch_arguments(launch_op);
  loom_builder_set_before(builder, launch_op);
  loom_op_t* dispatch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_dispatch_build(
      builder, entry_ref, counts.values, counts.count, arguments.values,
      arguments.count, /*workgroup_size=*/NULL,
      /*workgroup_size_count=*/0, launch_op->location, &dispatch_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(&resolution->rewriter, launch_op));
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_kernel_resolve_launches(
    loom_module_t* module, const loom_symbol_reference_table_t* references,
    const loom_symbol_ref_t* configuration_functions,
    iree_host_size_t configuration_function_count,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena, bool* out_valid) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(references);
  IREE_ASSERT(references->module == module);
  IREE_ASSERT_EQ(references->symbol_count, module->symbols.count);
  IREE_ASSERT(configuration_function_count == 0 || configuration_functions);
  IREE_ASSERT(configuration_function_count == 0 ||
              configuration_function_count == module->symbols.count);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_valid);
  *out_valid = false;

  loom_symbol_ref_t* entry_refs = NULL;
  if (references->symbol_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, references->symbol_count,
                                  sizeof(*entry_refs), (void**)&entry_refs));
    for (iree_host_size_t i = 0; i < references->symbol_count; ++i) {
      entry_refs[i] = loom_symbol_ref_null();
    }
  }

  loom_kernel_launch_resolution_t resolution = {
      .module = module,
      .configuration_functions = configuration_functions,
      .configuration_function_count = configuration_function_count,
      .entry_refs = entry_refs,
      .diagnostic_emitter = diagnostic_emitter,
      .scratch_arena = scratch_arena,
  };
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      module, module, scratch_arena, /*options=*/NULL, &resolution.type_remap));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&resolution.rewriter, module, scratch_arena));

  iree_status_t status = iree_ok_status();
  bool valid = true;
  for (iree_host_size_t i = 0;
       i < references->occurrence_count && iree_status_is_ok(status) && valid;
       ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &references->occurrences[i];
    loom_op_t* user_op = (loom_op_t*)occurrence->user_op;
    if (occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL || !user_op ||
        iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
        !loom_kernel_launch_isa(user_op)) {
      continue;
    }
    status = loom_kernel_launch_resolution_rewrite_launch(&resolution, user_op,
                                                          &valid);
  }
  loom_rewriter_deinitialize(&resolution.rewriter);
  if (iree_status_is_ok(status)) *out_valid = valid;
  return status;
}
