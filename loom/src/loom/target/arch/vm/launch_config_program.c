// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/launch_config_program.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/facts.h"
#include "loom/target/arch/vm/ops/ops.h"
#include "loom/target/arch/vm/ops/target.h"
#include "loom/target/emit/vm/launch_config_abi.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"

typedef struct loom_vm_launch_config_target_key_t {
  // Maximum local workgroup dimensions visible to the launch program.
  loom_target_workgroup_size_t max_workgroup_size;

  // Maximum flattened local workgroup size visible to the launch program.
  uint32_t max_flat_workgroup_size;

  // Maximum workgroup-local storage visible to the launch program.
  uint64_t max_workgroup_storage_bytes;

  // Fixed subgroup size visible to the launch program.
  uint32_t subgroup_size;

  // Maximum dispatched grid dimensions visible to the launch program.
  loom_target_grid_size_t max_grid_size;

  // Maximum flattened dispatched grid size visible to the launch program.
  uint64_t max_flat_grid_size;

  // Maximum dispatched workgroup counts visible to the launch program.
  loom_target_workgroup_count_limit_t max_workgroup_count;
} loom_vm_launch_config_target_key_t;

struct loom_vm_launch_config_program_target_t {
  // Next target in materialization order.
  loom_vm_launch_config_program_target_t* next;

  // Device execution limits projected onto the VM target.
  loom_vm_launch_config_target_key_t key;

  // Module-local VM target definition.
  loom_symbol_ref_t target_ref;
};

struct loom_vm_launch_config_program_entry_t {
  // Next entry in materialization order.
  loom_vm_launch_config_program_entry_t* next;

  // Stable identity following a specialized device kernel through lowering.
  loom_function_version_t* device_version_handle;

  // Device kernel symbol spelling used when no version handle exists.
  loom_string_id_t device_function_name_id;

  // Launch function symbol spelling interned in the shared module.
  loom_string_id_t launch_function_name_id;
};

static const loom_pass_environment_capability_type_t
    loom_vm_launch_config_program_capability_type = {
        .name = IREE_SVL("vm.launch-config-program"),
};

static const loom_pass_info_t
    loom_vm_launch_config_materialize_pass_info_storage = {
        .name = IREE_SVL("materialize-kernel-launch-configs"),
        .description =
            IREE_SVL("Materialize VM launch roots beside device kernels."),
        .kind = LOOM_PASS_MODULE,
};

static const loom_pass_info_t loom_vm_launch_config_finalize_pass_info_storage =
    {
        .name = IREE_SVL("finalize-kernel-launch-configs"),
        .description = IREE_SVL(
            "Attach final device storage requirements to VM launch exports."),
        .kind = LOOM_PASS_MODULE,
};

static const loom_pass_descriptor_t kVmLaunchConfigPassDescriptors[] = {
    {
        .key = IREE_SVL("vm-finalize-kernel-launch-configs"),
        .info = loom_vm_launch_config_finalize_pass_info,
        .module_run = loom_vm_launch_config_finalize_run,
    },
    {
        .key = IREE_SVL("vm-materialize-kernel-launch-configs"),
        .info = loom_vm_launch_config_materialize_pass_info,
        .module_run = loom_vm_launch_config_materialize_run,
    },
};

const loom_pass_registry_t loom_vm_launch_config_pass_registry = {
    .descriptors = kVmLaunchConfigPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kVmLaunchConfigPassDescriptors),
};

void loom_vm_launch_config_program_initialize(
    iree_arena_allocator_t* arena,
    loom_vm_launch_config_program_t* out_program) {
  *out_program = (loom_vm_launch_config_program_t){0};
  out_program->capability.type = &loom_vm_launch_config_program_capability_type;
  out_program->arena = arena;
}

loom_vm_launch_config_program_t* loom_vm_launch_config_program_from_pass(
    const loom_pass_t* pass) {
  if (pass == NULL || pass->environment == NULL) return NULL;
  return (loom_vm_launch_config_program_t*)loom_pass_environment_lookup(
      pass->environment, &loom_vm_launch_config_program_capability_type);
}

iree_status_t loom_vm_launch_config_program_require_finalized(
    const loom_vm_launch_config_program_t* program) {
  if (program->state != LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_FINALIZED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel launch configuration product was not finalized");
  }
  return iree_ok_status();
}

const loom_pass_info_t* loom_vm_launch_config_materialize_pass_info(void) {
  return &loom_vm_launch_config_materialize_pass_info_storage;
}

const loom_pass_info_t* loom_vm_launch_config_finalize_pass_info(void) {
  return &loom_vm_launch_config_finalize_pass_info_storage;
}

static loom_vm_launch_config_target_key_t loom_vm_launch_config_target_key_make(
    const loom_target_snapshot_t* snapshot) {
  return (loom_vm_launch_config_target_key_t){
      .max_workgroup_size = snapshot->max_workgroup_size,
      .max_flat_workgroup_size = snapshot->max_flat_workgroup_size,
      .max_workgroup_storage_bytes = snapshot->max_workgroup_storage_bytes,
      .subgroup_size = snapshot->subgroup_size,
      .max_grid_size = snapshot->max_grid_size,
      .max_flat_grid_size = snapshot->max_flat_grid_size,
      .max_workgroup_count = snapshot->max_workgroup_count,
  };
}

static bool loom_vm_launch_config_target_key_equal(
    const loom_vm_launch_config_target_key_t* lhs,
    const loom_vm_launch_config_target_key_t* rhs) {
  return lhs->max_workgroup_size.x == rhs->max_workgroup_size.x &&
         lhs->max_workgroup_size.y == rhs->max_workgroup_size.y &&
         lhs->max_workgroup_size.z == rhs->max_workgroup_size.z &&
         lhs->max_flat_workgroup_size == rhs->max_flat_workgroup_size &&
         lhs->max_workgroup_storage_bytes == rhs->max_workgroup_storage_bytes &&
         lhs->subgroup_size == rhs->subgroup_size &&
         lhs->max_grid_size.x == rhs->max_grid_size.x &&
         lhs->max_grid_size.y == rhs->max_grid_size.y &&
         lhs->max_grid_size.z == rhs->max_grid_size.z &&
         lhs->max_flat_grid_size == rhs->max_flat_grid_size &&
         lhs->max_workgroup_count.x == rhs->max_workgroup_count.x &&
         lhs->max_workgroup_count.y == rhs->max_workgroup_count.y &&
         lhs->max_workgroup_count.z == rhs->max_workgroup_count.z;
}

static iree_status_t loom_vm_launch_config_reserve_target_symbol(
    loom_vm_launch_config_program_t* program, loom_module_t* module,
    loom_symbol_ref_t* out_target_ref) {
  for (iree_host_size_t ordinal = program->targets.count;; ++ordinal) {
    char name_storage[64];
    const int name_length = iree_snprintf(name_storage, sizeof(name_storage),
                                          "$launch.target.%" PRIhsz, ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_storage)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "launch target symbol name is too large");
    }
    const iree_string_view_t name =
        iree_make_string_view(name_storage, (iree_host_size_t)name_length);
    const loom_string_id_t existing_name_id =
        loom_module_lookup_string(module, name);
    if (existing_name_id != LOOM_STRING_ID_INVALID &&
        loom_module_find_symbol(module, existing_name_id) !=
            LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_target_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }
}

static iree_status_t loom_vm_launch_config_reserve_function_symbol(
    const loom_vm_launch_config_program_t* program, loom_module_t* module,
    loom_symbol_ref_t* out_function_ref) {
  for (iree_host_size_t ordinal = program->entries.count;; ++ordinal) {
    char name_storage[64];
    const int name_length = iree_snprintf(name_storage, sizeof(name_storage),
                                          "$launch.function.%" PRIhsz, ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_storage)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "launch function symbol name is too large");
    }
    const iree_string_view_t name =
        iree_make_string_view(name_storage, (iree_host_size_t)name_length);
    const loom_string_id_t existing_name_id =
        loom_module_lookup_string(module, name);
    if (existing_name_id != LOOM_STRING_ID_INVALID &&
        loom_module_find_symbol(module, existing_name_id) !=
            LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_function_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }
}

static iree_status_t loom_vm_launch_config_resolve_target(
    loom_vm_launch_config_program_t* program, loom_module_t* module,
    const loom_target_snapshot_t* device_snapshot,
    loom_symbol_ref_t* out_target_ref) {
  const loom_vm_launch_config_target_key_t key =
      loom_vm_launch_config_target_key_make(device_snapshot);
  for (loom_vm_launch_config_program_target_t* target = program->targets.head;
       target != NULL; target = target->next) {
    if (loom_vm_launch_config_target_key_equal(&target->key, &key)) {
      *out_target_ref = target->target_ref;
      return iree_ok_status();
    }
  }

  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_reserve_target_symbol(
      program, module, &target_ref));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_target_build_core_with_execution_limits(
      &builder, target_ref, device_snapshot, LOOM_LOCATION_UNKNOWN,
      &target_op));

  loom_vm_launch_config_program_target_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(program->arena, sizeof(*target), (void**)&target));
  *target = (loom_vm_launch_config_program_target_t){
      .key = key,
      .target_ref = target_ref,
  };
  if (program->targets.tail != NULL) {
    program->targets.tail->next = target;
  } else {
    program->targets.head = target;
  }
  program->targets.tail = target;
  ++program->targets.count;
  *out_target_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_vm_launch_config_copy_value_name(
    loom_module_t* module, loom_value_id_t source_value,
    loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  return loom_module_set_value_name(module, target_value, source_name);
}

static iree_status_t loom_vm_launch_config_emit_workload_type_constraint(
    const loom_module_t* module, const loom_op_t* kernel_op,
    loom_value_id_t argument, uint16_t argument_ordinal,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  char fallback_name[32];
  iree_string_view_t argument_name = loom_module_value_name(module, argument);
  if (iree_string_view_is_empty(argument_name)) {
    const int fallback_name_length =
        iree_snprintf(fallback_name, sizeof(fallback_name), "workload[%u]",
                      (unsigned)argument_ordinal);
    argument_name = iree_make_string_view(fallback_name, fallback_name_length);
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(argument_name),
      loom_param_type(loom_module_value_type(module, argument)),
      loom_param_string(IREE_SV("scalar VM launch-config ABI value")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = kernel_op,
      .error = LOOM_ERR_TYPE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_vm_launch_config_copy_workload_predicates(
    loom_module_t* module, const loom_op_t* source_kernel,
    loom_ir_remap_t* remap, iree_arena_allocator_t* scratch_arena,
    loom_op_t* launch_function_op) {
  const loom_attribute_t source_predicates =
      loom_kernel_def_workload_predicates(source_kernel);
  if (loom_attr_is_absent(source_predicates)) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      remap, source_predicates.predicate_list, source_predicates.count,
      &target_predicates));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, launch_function_op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(target_predicates, source_predicates.count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_vm_launch_config_build_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_constant_build(builder, loom_value_facts_exact_i64(value),
                             loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location,
                             out_value);
}

static iree_status_t loom_vm_launch_config_intern_export_name(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    iree_string_view_t public_name, loom_string_id_t* out_string_id) {
  iree_host_size_t private_name_length = 0;
  if (!iree_host_size_checked_add(public_name.size,
                                  LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH,
                                  &private_name_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config export name is too large");
  }
  char* private_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, private_name_length,
                                           (void**)&private_name_data));
  private_name_data[0] = LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX;
  memcpy(private_name_data + LOOM_VM_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH,
         public_name.data, public_name.size);
  return loom_module_intern_string(
      module, iree_make_string_view(private_name_data, private_name_length),
      out_string_id);
}

static iree_status_t loom_vm_launch_config_build_function(
    loom_vm_launch_config_program_t* program, loom_module_t* module,
    const loom_low_source_selection_t* selection,
    const loom_value_fact_table_t* source_facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena) {
  loom_op_t* source_kernel = selection->func.op;
  loom_region_t* source_config = loom_kernel_def_config(source_kernel);
  const loom_op_t* source_launch_config =
      loom_kernel_def_launch_config_op(source_kernel);

  const loom_target_bundle_t* target_bundle =
      loom_low_source_selection_target_bundle(selection);
  if (target_bundle == NULL || target_bundle->snapshot == NULL ||
      target_bundle->export_plan == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel launch configuration has no complete target facts");
  }
  const iree_string_view_t export_name =
      iree_string_view_is_empty(target_bundle->export_plan->export_symbol)
          ? selection->function_name
          : target_bundle->export_plan->export_symbol;

  const loom_value_slice_t source_arguments =
      loom_kernel_workload_arg_ids(module, source_kernel);
  loom_type_t* argument_types = NULL;
  if (source_arguments.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, source_arguments.count, sizeof(*argument_types),
        (void**)&argument_types));
  }
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    const loom_value_id_t source_argument = source_arguments.values[i];
    const loom_type_t source_type =
        loom_module_value_type(module, source_argument);
    if (!loom_type_is_scalar(source_type)) {
      return loom_vm_launch_config_emit_workload_type_constraint(
          module, source_kernel, source_argument, i, diagnostic_emitter);
    }
    argument_types[i] = source_type;
  }

  loom_symbol_ref_t vm_target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_resolve_target(
      program, module, target_bundle->snapshot, &vm_target_ref));

  loom_string_id_t launch_export_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_intern_export_name(
      module, scratch_arena, export_name, &launch_export_name_id));
  loom_symbol_ref_t launch_function_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_reserve_function_symbol(
      program, module, &launch_function_ref));
  const loom_string_id_t launch_function_name_id =
      module->symbols.entries[launch_function_ref.symbol_id].name_id;

  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, scratch_arena,
                                                /*options=*/NULL, &remap));

  loom_type_t result_types[LOOM_VM_LAUNCH_CONFIG_RESULT_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(result_types); ++i) {
    result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  }
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL,
      LOOM_FUNC_VISIBILITY_PUBLIC, /*retain=*/0, /*cc=*/0,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
      vm_target_ref, /*abi=*/0, loom_named_attr_slice_empty(),
      launch_export_name_id, loom_named_attr_slice_empty(),
      loom_named_attr_slice_empty(), launch_function_ref, argument_types,
      source_arguments.count, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL,
      /*predicates_count=*/0, source_kernel->location, &function_op));
  const loom_func_like_t launch_function =
      loom_func_like_cast(module, function_op);

  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(launch_function, &target_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, source_arguments.values,
                                                target_arguments,
                                                source_arguments.count));
  for (uint16_t i = 0; i < source_arguments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_launch_config_copy_value_name(
        module, source_arguments.values[i], target_arguments[i]));
  }
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_copy_workload_predicates(
      module, source_kernel, &remap, scratch_arena, function_op));

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
          loom_module_value_type(module, source_result));
    }
    if (!materialize_results) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_ir_clone_op(&builder, source_op, &remap, &cloned_op));
      continue;
    }

    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_constant_build(
          &builder, loom_value_fact_table_lookup(source_facts, source_result),
          loom_module_value_type(module, source_result), source_op->location,
          &target_result));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(&remap, source_result, target_result));
      IREE_RETURN_IF_ERROR(loom_vm_launch_config_copy_value_name(
          module, source_result, target_result));
    }
  }

  loom_value_id_t result_values[LOOM_VM_LAUNCH_CONFIG_RESULT_COUNT];
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
                                                     &result_values[i]));
  }

  if (loom_kernel_launch_config_has_workgroup_cluster_size(
          source_launch_config)) {
    for (uint8_t dimension = 0; dimension < 3; ++dimension) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          &remap,
          loom_kernel_launch_config_workgroup_cluster_size_operand(
              source_launch_config, (loom_kernel_dimension_t)dimension),
          &result_values[LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X +
                         dimension]));
    }
  } else {
    loom_value_id_t one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_launch_config_build_constant(
        &builder, 1, source_kernel->location, &one));
    for (uint8_t dimension = 0; dimension < 3; ++dimension) {
      result_values[LOOM_VM_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X +
                    dimension] = one;
    }
  }
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_build_constant(
      &builder, target_bundle->snapshot->subgroup_size, source_kernel->location,
      &result_values[LOOM_VM_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE]));

  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(
      &builder, result_values, IREE_ARRAYSIZE(result_values),
      source_kernel->location, &return_op));

  loom_string_id_t device_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, selection->function_name, &device_name_id));
  loom_vm_launch_config_program_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(program->arena, sizeof(*entry), (void**)&entry));
  *entry = (loom_vm_launch_config_program_entry_t){
      .device_version_handle = selection->version_handle,
      .device_function_name_id = device_name_id,
      .launch_function_name_id = launch_function_name_id,
  };
  if (program->entries.tail != NULL) {
    program->entries.tail->next = entry;
  } else {
    program->entries.head = entry;
  }
  program->entries.tail = entry;
  ++program->entries.count;
  return iree_ok_status();
}

iree_status_t loom_vm_launch_config_materialize_run(loom_pass_t* pass,
                                                    loom_module_t* module) {
  loom_vm_launch_config_program_t* program =
      loom_vm_launch_config_program_from_pass(pass);
  if (program == NULL) return iree_ok_status();
  if (program->state != LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_EMPTY) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel launch configurations were already materialized");
  }

  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  if (policy_registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch materialization requires the target-low policy registry");
  }
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_target_environment_t* target_environment =
      loom_target_pass_capability_target_environment(target_capability);
  if (target_environment == NULL ||
      loom_target_environment_lookup_fact_provider(
          target_environment, &loom_vm_target_fact_type) == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch materialization requires the VM target provider");
  }

  iree_arena_allocator_t selection_arena;
  iree_arena_initialize(module->arena.block_pool, &selection_arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  const loom_low_source_selection_options_t selection_options = {
      .policy_registry = policy_registry,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .lowering_kind = IREE_SV("launch configuration materialization"),
      .function_versions =
          loom_target_pass_capability_function_versions(target_capability),
  };
  loom_low_source_selection_list_t selection_list = {0};
  iree_status_t status = loom_low_select_source_funcs(
      module, &selection_options, &selection_arena, &selection_list);
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !loom_pass_has_error_diagnostics(pass);
       ++i) {
    const loom_low_source_selection_t* selection = &selection_list.values[i];
    if (!loom_kernel_def_isa(selection->func.op)) continue;
    loom_value_fact_table_t* source_facts = NULL;
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(selection->func,
                                                       selection->target_facts),
        &source_facts);
    if (iree_status_is_ok(status)) {
      status = loom_vm_launch_config_build_function(
          program, module, selection, source_facts, pass->diagnostic_emitter,
          &scratch_arena);
    }
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "while materializing kernel launch configuration '@%.*s'",
          (int)selection->function_name.size, selection->function_name.data);
    }
    iree_arena_reset(&scratch_arena);
  }
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&selection_arena);

  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
      program->entries.count == 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "pass program found no kernel launch configurations");
  }
  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass)) {
    program->state = LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_MATERIALIZED;
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    loom_pass_mark_changed(pass);
  }
  return status;
}

typedef struct loom_vm_launch_config_resolved_function_t {
  // Resolved function interface.
  loom_func_like_t function;

  // Module-local symbol defining |function|.
  loom_symbol_id_t symbol_id;
} loom_vm_launch_config_resolved_function_t;

static iree_status_t loom_vm_launch_config_resolve_function(
    const loom_module_t* module, loom_string_id_t function_name_id,
    iree_string_view_t function_kind,
    loom_vm_launch_config_resolved_function_t* out_resolved) {
  *out_resolved = (loom_vm_launch_config_resolved_function_t){0};
  const iree_string_view_t function_name =
      module->strings.entries[function_name_id];
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module, function_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "%.*s '@%.*s' was not found",
                            (int)function_kind.size, function_kind.data,
                            (int)function_name.size, function_name.data);
  }
  const loom_func_like_t function = loom_func_like_cast(
      module, module->symbols.entries[symbol_id].defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%.*s '@%.*s' is not function-like",
                            (int)function_kind.size, function_kind.data,
                            (int)function_name.size, function_name.data);
  }
  *out_resolved = (loom_vm_launch_config_resolved_function_t){
      .function = function,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

iree_status_t loom_vm_launch_config_program_build_closure(
    const loom_vm_launch_config_program_t* program, const loom_module_t* module,
    iree_arena_allocator_t* arena,
    loom_vm_launch_config_program_closure_t* out_closure) {
  *out_closure = (loom_vm_launch_config_program_closure_t){0};
  IREE_RETURN_IF_ERROR(
      loom_vm_launch_config_program_require_finalized(program));

  uint8_t* root_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*root_symbols),
                                                 (void**)&root_symbols));
  memset(root_symbols, 0, module->symbols.count * sizeof(*root_symbols));
  const loom_function_version_t** root_function_versions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*root_function_versions),
      (void**)&root_function_versions));
  memset(root_function_versions, 0,
         module->symbols.count * sizeof(*root_function_versions));

  loom_symbol_id_t* root_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, program->entries.count,
                                                 sizeof(*root_symbol_ids),
                                                 (void**)&root_symbol_ids));
  iree_host_size_t root_index = 0;
  for (const loom_vm_launch_config_program_entry_t* entry =
           program->entries.head;
       entry != NULL; entry = entry->next) {
    loom_vm_launch_config_resolved_function_t resolved = {0};
    IREE_RETURN_IF_ERROR(loom_vm_launch_config_resolve_function(
        module, entry->launch_function_name_id, IREE_SV("launch function"),
        &resolved));
    root_symbols[resolved.symbol_id] = 1;
    root_function_versions[resolved.symbol_id] = entry->device_version_handle;
    root_symbol_ids[root_index++] = resolved.symbol_id;
  }

  loom_symbol_reference_table_t* references = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*references), (void**)&references));
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, arena, references));
  const loom_symbol_liveness_options_t options = {
      .root_symbol_ids =
          {
              .values = root_symbol_ids,
              .count = root_index,
          },
  };
  IREE_RETURN_IF_ERROR(loom_symbol_liveness_compute(
      module, references, &options, arena, &out_closure->symbol_liveness));
  out_closure->root_symbols = root_symbols;
  out_closure->root_function_versions = root_function_versions;
  return iree_ok_status();
}

static iree_status_t loom_vm_launch_config_resolve_device_function(
    const loom_module_t* module,
    const loom_vm_launch_config_program_entry_t* entry,
    loom_func_like_t* out_function) {
  if (entry->device_version_handle != NULL) {
    *out_function = entry->device_version_handle->function;
    if (!loom_func_like_isa(*out_function)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "device kernel version does not reference a live function");
    }
    return iree_ok_status();
  }
  loom_vm_launch_config_resolved_function_t resolved = {0};
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_resolve_function(
      module, entry->device_function_name_id, IREE_SV("device kernel"),
      &resolved));
  *out_function = resolved.function;
  return iree_ok_status();
}

static iree_status_t loom_vm_launch_config_workgroup_storage_bytes(
    const loom_module_t* module, loom_func_like_t low_function,
    uint64_t* out_bytes) {
  // Module finalization runs after every caller-visible storage reservation is
  // materialized. Later target emission may add stack, scratch, or private
  // allocation state but cannot change this workgroup requirement.
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

static iree_status_t loom_vm_launch_config_set_storage_metadata(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t launch_function,
    uint64_t workgroup_storage_bytes) {
  const uint8_t metadata_attr_index =
      launch_function.vtable->export_metadata_attr_index;
  if (metadata_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch function does not expose export metadata");
  }
  loom_string_id_t metadata_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, loom_vm_launch_config_workgroup_storage_metadata_key(),
      &metadata_key_id));
  const loom_named_attr_update_t update = loom_named_attr_replace(
      metadata_key_id, loom_attr_u64(workgroup_storage_bytes));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  const iree_status_t status = loom_rewriter_replace_attr_dict(
      &rewriter, launch_function.op, metadata_attr_index,
      loom_make_named_attr_update_slice(&update, 1));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_vm_launch_config_finalize_run(loom_pass_t* pass,
                                                 loom_module_t* module) {
  loom_vm_launch_config_program_t* program =
      loom_vm_launch_config_program_from_pass(pass);
  if (program == NULL) return iree_ok_status();
  if (program->state != LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_MATERIALIZED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel launch configurations were not materialized");
  }

  iree_status_t status = iree_ok_status();
  for (loom_vm_launch_config_program_entry_t* entry = program->entries.head;
       entry != NULL && iree_status_is_ok(status); entry = entry->next) {
    loom_func_like_t device_function = {0};
    status = loom_vm_launch_config_resolve_device_function(module, entry,
                                                           &device_function);
    uint64_t workgroup_storage_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_vm_launch_config_workgroup_storage_bytes(
          module, device_function, &workgroup_storage_bytes);
    }
    loom_vm_launch_config_resolved_function_t launch_function = {0};
    if (iree_status_is_ok(status)) {
      status = loom_vm_launch_config_resolve_function(
          module, entry->launch_function_name_id, IREE_SV("launch function"),
          &launch_function);
    }
    if (iree_status_is_ok(status)) {
      status = loom_vm_launch_config_set_storage_metadata(
          pass, module, launch_function.function, workgroup_storage_bytes);
    }
  }
  if (iree_status_is_ok(status)) {
    program->state = LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_FINALIZED;
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    loom_pass_mark_changed(pass);
  }
  return status;
}
