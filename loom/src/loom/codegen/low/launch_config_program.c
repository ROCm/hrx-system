// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/launch_config_program.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/launch_config_abi.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/dce.h"

struct loom_kernel_launch_config_program_entry_t {
  // Next entry in capture order.
  loom_kernel_launch_config_program_entry_t* next;

  // Stable specialized-function identity, or NULL for an authored function.
  loom_function_version_t* version_handle;

  // Source symbol spelling interned in the host module for final lookup.
  loom_string_id_t source_function_name_id;

  // Host function receiving the launch computation.
  loom_func_like_t launch_function;

  // Results complete before final workgroup storage is joined.
  loom_value_id_t
      result_values[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES];
};

static const loom_pass_environment_capability_type_t
    loom_kernel_launch_config_program_capability_type = {
        .name = IREE_SVL("kernel-launch-config-program"),
};

iree_status_t loom_kernel_launch_config_program_initialize(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator,
    loom_kernel_launch_config_program_t* out_program) {
  *out_program = (loom_kernel_launch_config_program_t){0};
  out_program->capability.type =
      &loom_kernel_launch_config_program_capability_type;
  return loom_module_allocate(context, IREE_SV("launch_config"), block_pool,
                              /*hints=*/NULL, allocator, &out_program->module);
}

void loom_kernel_launch_config_program_deinitialize(
    loom_kernel_launch_config_program_t* program) {
  loom_module_free(program->module);
  *program = (loom_kernel_launch_config_program_t){0};
}

loom_kernel_launch_config_program_t*
loom_kernel_launch_config_program_from_pass(const loom_pass_t* pass) {
  if (pass == NULL || pass->environment == NULL) return NULL;
  return (loom_kernel_launch_config_program_t*)loom_pass_environment_lookup(
      pass->environment, &loom_kernel_launch_config_program_capability_type);
}

static iree_status_t loom_kernel_launch_config_copy_value_name(
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

static iree_status_t loom_kernel_launch_config_copy_workload_predicates(
    const loom_op_t* source_kernel, loom_ir_remap_t* remap,
    iree_arena_allocator_t* scratch_arena, loom_op_t* target_function_op) {
  const loom_attribute_t source_predicates =
      loom_kernel_def_workload_predicates(source_kernel);
  if (loom_attr_is_absent(source_predicates)) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      remap, source_predicates.predicate_list, source_predicates.count,
      &target_predicates));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, remap->target_module, scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, target_function_op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(target_predicates, source_predicates.count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_kernel_launch_config_build_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_constant_build(builder, loom_value_facts_exact_i64(value),
                             loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location,
                             out_value);
}

static iree_status_t loom_kernel_launch_config_intern_export_name(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    iree_string_view_t public_name, loom_string_id_t* out_string_id) {
  iree_host_size_t private_name_length = 0;
  if (!iree_host_size_checked_add(
          public_name.size, LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH,
          &private_name_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "launch config export name is too large");
  }
  char* private_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, private_name_length,
                                           (void**)&private_name_data));
  private_name_data[0] = LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX;
  memcpy(private_name_data + LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH,
         public_name.data, public_name.size);
  return loom_module_intern_string(
      module, iree_make_string_view(private_name_data, private_name_length),
      out_string_id);
}

static iree_status_t loom_kernel_launch_config_program_build_function(
    loom_kernel_launch_config_program_t* program,
    const loom_module_t* source_module, loom_func_like_t source_function,
    iree_string_view_t source_function_name,
    loom_function_version_t* version_handle,
    const loom_target_facts_t* target_facts,
    const loom_value_fact_table_t* source_facts,
    iree_arena_allocator_t* scratch_arena) {
  loom_op_t* source_kernel = source_function.op;
  loom_region_t* source_config = loom_kernel_def_config(source_kernel);
  const loom_op_t* source_launch_config =
      loom_kernel_def_launch_config_op(source_kernel);

  const loom_target_bundle_t* target_bundle =
      loom_target_facts_bundle(target_facts);
  if (target_bundle == NULL || target_bundle->snapshot == NULL ||
      target_bundle->export_plan == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel launch configuration has no complete target facts");
  }
  const iree_string_view_t export_name =
      iree_string_view_is_empty(target_bundle->export_plan->export_symbol)
          ? source_function_name
          : target_bundle->export_plan->export_symbol;

  loom_module_t* target_module = program->module;
  loom_string_id_t private_export_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_intern_export_name(
      target_module, scratch_arena, export_name, &private_export_name));
  if (loom_module_find_symbol(target_module, private_export_name) !=
      LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "duplicate launch config export '%.*s'",
                            (int)export_name.size, export_name.data);
  }
  loom_symbol_id_t function_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(
      target_module, private_export_name, &function_symbol_id));

  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      target_module, source_function_name, &source_name_id));

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source_module, target_module, scratch_arena, /*options=*/NULL, &remap));
  const loom_value_slice_t source_arguments =
      loom_kernel_workload_arg_ids(source_module, source_kernel);
  loom_type_t* argument_types = NULL;
  if (source_arguments.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, source_arguments.count, sizeof(*argument_types),
        (void**)&argument_types));
  }
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(source_module, source_arguments.values[i]);
    if (!loom_type_is_scalar(source_type)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "kernel '@%.*s' launch workload argument %u is not scalar",
          (int)source_function_name.size, source_function_name.data,
          (unsigned)i);
    }
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, source_type, &argument_types[i]));
  }

  loom_type_t result_types[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(result_types); ++i) {
    result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  }
  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &remap, source_kernel->location, &target_location));
  loom_builder_t builder;
  loom_builder_initialize(target_module, &target_module->arena,
                          loom_module_block(target_module), &builder);
  loom_op_t* function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL,
      LOOM_FUNC_VISIBILITY_PUBLIC, /*retain=*/0, /*cc=*/0,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      private_export_name, loom_named_attr_slice_empty(),
      loom_named_attr_slice_empty(),
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = function_symbol_id},
      argument_types, source_arguments.count, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL,
      /*predicates_count=*/0, target_location, &function_op));
  const loom_func_like_t launch_function =
      loom_func_like_cast(target_module, function_op);

  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(launch_function, &target_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_arguments.values,
                                                target_arguments,
                                                source_arguments.count));
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_copy_value_name(
        source_module, target_module, &remap, source_arguments.values[i],
        target_arguments[i]));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_copy_workload_predicates(
      source_kernel, &remap, scratch_arena, function_op));

  loom_builder_enter_region(&builder, function_op,
                            loom_func_like_body(launch_function));
  const loom_block_t* source_block =
      loom_region_const_entry_block(source_config);
  const loom_op_t* source_op = NULL;
  loom_block_for_each_op(source_block, source_op) {
    if (source_op == source_launch_config) continue;
    bool materialize_results = source_op->result_count != 0;
    const loom_value_id_t* source_results = loom_op_const_results(source_op);
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      materialize_results &= loom_value_facts_can_materialize_constant(
          loom_value_fact_table_lookup(source_facts, source_result),
          loom_module_value_type(source_module, source_result));
    }
    if (!materialize_results) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_ir_clone_op(&builder, source_op, &remap, &cloned_op));
      continue;
    }

    loom_location_id_t result_location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(&remap, source_op->location,
                                                   &result_location));
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      loom_type_t target_type = {0};
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          &remap, loom_module_value_type(source_module, source_result),
          &target_type));
      loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_constant_build(
          &builder, loom_value_fact_table_lookup(source_facts, source_result),
          target_type, result_location, &target_result));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(&remap, source_result, target_result));
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_copy_value_name(
          source_module, target_module, &remap, source_result, target_result));
    }
  }

  loom_kernel_launch_config_program_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&target_module->arena,
                                           sizeof(*entry), (void**)&entry));
  memset(entry, 0, sizeof(*entry));
  entry->version_handle = version_handle;
  entry->source_function_name_id = source_name_id;
  entry->launch_function = launch_function;

  const loom_value_id_t source_results[] = {
      loom_kernel_launch_config_workgroup_count_x(source_launch_config),
      loom_kernel_launch_config_workgroup_count_y(source_launch_config),
      loom_kernel_launch_config_workgroup_count_z(source_launch_config),
      loom_kernel_launch_config_workgroup_size_x(source_launch_config),
      loom_kernel_launch_config_workgroup_size_y(source_launch_config),
      loom_kernel_launch_config_workgroup_size_z(source_launch_config),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_results); ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(&remap, source_results[i],
                                                     &entry->result_values[i]));
  }

  if (loom_kernel_launch_config_has_workgroup_cluster_size(
          source_launch_config)) {
    for (uint8_t dimension = 0; dimension < 3; ++dimension) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          &remap,
          loom_kernel_launch_config_workgroup_cluster_size_operand(
              source_launch_config, (loom_kernel_dimension_t)dimension),
          &entry->result_values
               [LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X +
                dimension]));
    }
  } else {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_build_constant(
        &builder, 1, target_location, &one));
    for (uint8_t dimension = 0; dimension < 3; ++dimension) {
      entry->result_values
          [LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X +
           dimension] = one;
    }
  }

  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_build_constant(
      &builder, target_bundle->snapshot->subgroup_size, target_location,
      &entry->result_values[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE]));

  if (program->entries.tail != NULL) {
    program->entries.tail->next = entry;
  } else {
    program->entries.head = entry;
  }
  program->entries.tail = entry;
  ++program->entries.count;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_program_capture(
    loom_kernel_launch_config_program_t* program,
    const loom_module_t* source_module, loom_func_like_t source_function,
    iree_string_view_t source_function_name,
    loom_function_version_t* version_handle,
    const loom_target_facts_t* target_facts,
    const loom_value_fact_table_t* source_facts) {
  if (!loom_kernel_def_isa(source_function.op)) return iree_ok_status();

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(program->module->arena.block_pool, &scratch_arena);
  iree_status_t status = loom_kernel_launch_config_program_build_function(
      program, source_module, source_function, source_function_name,
      version_handle, target_facts, source_facts, &scratch_arena);
  iree_arena_deinitialize(&scratch_arena);
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status, "while capturing kernel launch configuration '@%.*s'",
        (int)source_function_name.size, source_function_name.data);
  }
  return status;
}

static iree_status_t loom_kernel_launch_config_resolve_low_function(
    const loom_kernel_launch_config_program_t* program,
    const loom_module_t* lowered_module,
    const loom_kernel_launch_config_program_entry_t* entry,
    loom_func_like_t* out_function) {
  *out_function = (loom_func_like_t){0};
  if (entry->version_handle != NULL) {
    *out_function = entry->version_handle->function;
    return iree_ok_status();
  }
  const iree_string_view_t source_name =
      program->module->strings.entries[entry->source_function_name_id];
  const loom_string_id_t lowered_name_id =
      loom_module_lookup_string(lowered_module, source_name);
  if (lowered_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "lowered kernel '@%.*s' was not found",
                            (int)source_name.size, source_name.data);
  }
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(lowered_module, lowered_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "lowered kernel '@%.*s' was not found",
                            (int)source_name.size, source_name.data);
  }
  *out_function = loom_func_like_cast(
      lowered_module, lowered_module->symbols.entries[symbol_id].defining_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_workgroup_storage_bytes(
    const loom_module_t* module, loom_func_like_t low_function,
    uint64_t* out_bytes) {
  *out_bytes = 0;
  if (!loom_low_kernel_def_isa(low_function.op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch config entry does not resolve to a final low kernel");
  }
  const loom_region_t* body = loom_func_like_body(low_function);
  if (body == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "final low kernel has no body for storage layout");
  }
  loom_low_storage_layout_space_sizes_t sizes = {0};
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_storage_reserve_isa(op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_storage_layout_accumulate_reservation(module, op, &sizes));
      }
    }
  }
  *out_bytes = sizes.workgroup_bytes;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_program_finalize(
    loom_kernel_launch_config_program_t* program,
    const loom_module_t* lowered_module,
    iree_arena_block_pool_t* scratch_block_pool, loom_module_t** out_module) {
  *out_module = NULL;
  if (program->entries.count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass program produced no kernel launch configurations");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(scratch_block_pool, &scratch_arena);
  iree_status_t status = iree_ok_status();
  for (loom_kernel_launch_config_program_entry_t* entry = program->entries.head;
       entry != NULL && iree_status_is_ok(status); entry = entry->next) {
    loom_func_like_t low_function = {0};
    status = loom_kernel_launch_config_resolve_low_function(
        program, lowered_module, entry, &low_function);
    uint64_t workgroup_storage_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_kernel_launch_config_workgroup_storage_bytes(
          lowered_module, low_function, &workgroup_storage_bytes);
    }
    if (!iree_status_is_ok(status)) break;
    if (workgroup_storage_bytes > INT64_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "workgroup storage size %" PRIu64
                                " exceeds launch fact domain",
                                workgroup_storage_bytes);
      break;
    }

    loom_builder_t builder;
    loom_builder_initialize(
        program->module, &program->module->arena,
        loom_region_entry_block(loom_func_like_body(entry->launch_function)),
        &builder);
    loom_value_id_t result_values[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT];
    memcpy(result_values, entry->result_values, sizeof(entry->result_values));
    status = loom_kernel_launch_config_build_constant(
        &builder, (int64_t)workgroup_storage_bytes,
        entry->launch_function.op->location,
        &result_values
            [LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES]);
    loom_op_t* return_op = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_func_return_build(
          &builder, result_values, IREE_ARRAYSIZE(result_values),
          entry->launch_function.op->location, &return_op);
    }
    if (iree_status_is_ok(status)) {
      loom_dce_statistics_t statistics = {0};
      loom_pass_t dce_pass = {
          .info = loom_dce_pass_info(),
          .instance_arena = &scratch_arena,
          .arena = &scratch_arena,
          .statistic_storage = &statistics,
      };
      status = loom_dce_run(&dce_pass, program->module, entry->launch_function);
      iree_arena_reset(&scratch_arena);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status)) {
    *out_module = program->module;
  }
  return status;
}
