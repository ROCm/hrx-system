// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/launch_config_module.h"

#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/transforms/cleanup/dce.h"

typedef struct loom_kernel_launch_config_module_build_t {
  // Immutable verified source module containing the kernel definitions.
  const loom_module_t* source_module;
  // Prepared host module under construction.
  loom_module_t* module;
  // Scratch storage discarded after extraction.
  iree_arena_allocator_t* scratch_arena;
} loom_kernel_launch_config_module_build_t;

static iree_string_view_t loom_kernel_launch_config_module_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_kernel_launch_config_module_copy_value_name(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_ir_remap_t* remap, loom_value_id_t source_value,
    loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(source_module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      remap, source_name, /*allow_invalid=*/false, &target_name));
  return loom_module_set_value_name(target_module, target_value, target_name);
}

static iree_status_t loom_kernel_launch_config_module_build_function(
    loom_kernel_launch_config_module_build_t* build,
    loom_symbol_id_t source_symbol_id, loom_op_t* source_kernel_op) {
  const loom_value_slice_t source_arguments =
      loom_kernel_workload_arg_ids(build->source_module, source_kernel_op);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      build->source_module, build->module, build->scratch_arena,
      /*options=*/NULL, &remap));

  loom_type_t* argument_types = NULL;
  if (source_arguments.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, source_arguments.count, sizeof(*argument_types),
        (void**)&argument_types));
  }
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    const loom_type_t source_type = loom_module_value_type(
        build->source_module, source_arguments.values[i]);
    if (!loom_type_is_scalar(source_type)) {
      const iree_string_view_t source_name =
          loom_kernel_launch_config_module_symbol_name(build->source_module,
                                                       source_symbol_id);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "kernel '@%.*s' launch workload argument %u is not scalar",
          (int)source_name.size, source_name.data, (unsigned)i);
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, source_type, &argument_types[i]));
  }

  const loom_type_t result_types[3] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  const iree_string_view_t source_name =
      loom_kernel_launch_config_module_symbol_name(build->source_module,
                                                   source_symbol_id);
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(build->module, source_name, &target_name));
  loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(build->module, target_name, &target_symbol_id));

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &remap, source_kernel_op->location, &target_location));
  loom_builder_t builder;
  loom_builder_initialize(build->module, &build->module->arena,
                          loom_module_block(build->module), &builder);
  loom_op_t* function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY,
      LOOM_FUNC_VISIBILITY_PUBLIC, /*retain=*/0, /*cc=*/0,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id},
      argument_types, source_arguments.count, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL,
      /*predicates_count=*/0, target_location, &function_op));
  const loom_func_like_t function =
      loom_func_like_cast(build->module, function_op);

  uint16_t ignored_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(function, &ignored_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_arguments.values,
                                                target_arguments,
                                                source_arguments.count));
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_module_copy_value_name(
        build->source_module, build->module, &remap, source_arguments.values[i],
        target_arguments[i]));
  }

  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(function));
  const loom_region_t* source_config = loom_kernel_def_config(source_kernel_op);
  const loom_block_t* source_block =
      loom_region_const_entry_block(source_config);
  IREE_RETURN_IF_ERROR(loom_ir_clone_block_ops(
      &builder, source_block, &remap,
      &(loom_ir_clone_block_options_t){.omit_terminators = true}));

  const loom_op_t* source_launch_config =
      loom_kernel_def_launch_config_op(source_kernel_op);
  loom_value_id_t result_values[3] = {0};
  for (uint8_t dimension = 0; dimension < IREE_ARRAYSIZE(result_values);
       ++dimension) {
    const loom_value_id_t source_value =
        loom_kernel_launch_config_workgroup_count_operand(
            source_launch_config, (loom_kernel_dimension_t)dimension);
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &remap, source_value, &result_values[dimension]));
  }
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, result_values,
                                              IREE_ARRAYSIZE(result_values),
                                              target_location, &return_op));

  loom_dce_statistics_t statistics = {0};
  loom_pass_t dce_pass = {
      .info = loom_dce_pass_info(),
      .instance_arena = build->scratch_arena,
      .arena = build->scratch_arena,
      .statistic_storage = &statistics,
  };
  return loom_dce_run(&dce_pass, build->module, function);
}

iree_status_t loom_kernel_launch_config_module_materialize(
    const loom_module_t* source_module, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  loom_module_t* module = NULL;
  iree_status_t status =
      loom_module_allocate(source_module->context, IREE_SV("launch_config"),
                           block_pool, NULL, allocator, &module);
  iree_arena_allocator_t scratch_arena;
  bool scratch_arena_initialized = false;
  loom_kernel_launch_config_module_build_t build = {
      .source_module = source_module,
      .module = module,
  };
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &scratch_arena);
    scratch_arena_initialized = true;
    build.scratch_arena = &scratch_arena;
  }
  for (loom_symbol_id_t symbol_id = 0;
       symbol_id < source_module->symbols.count && iree_status_is_ok(status);
       ++symbol_id) {
    loom_op_t* defining_op =
        source_module->symbols.entries[symbol_id].defining_op;
    if (defining_op == NULL ||
        iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
        !loom_kernel_def_isa(defining_op)) {
      continue;
    }
    status = loom_kernel_launch_config_module_build_function(&build, symbol_id,
                                                             defining_op);
    iree_arena_reset(&scratch_arena);
    if (!iree_status_is_ok(status)) {
      const iree_string_view_t source_name =
          loom_kernel_launch_config_module_symbol_name(source_module,
                                                       symbol_id);
      status = iree_status_annotate_f(
          status, "while extracting kernel launch configuration '@%.*s'",
          (int)source_name.size, source_name.data);
    }
  }

  if (scratch_arena_initialized) {
    iree_arena_deinitialize(&scratch_arena);
  }
  if (!iree_status_is_ok(status)) {
    if (module != NULL) loom_module_free(module);
    return status;
  }
  *out_module = module;
  return iree_ok_status();
}
