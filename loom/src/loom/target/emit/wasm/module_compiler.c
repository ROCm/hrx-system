// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_compiler.h"

#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function_model.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/arch/wasm/ops/ops.h"
#include "loom/target/arch/wasm/provider.h"
#include "loom/target/registers.h"

typedef struct loom_wasm_local_entry_t {
  // Descriptor-set-local register class ID from allocation.
  uint16_t descriptor_reg_class_id;
  // Class-local target-id assignment base from allocation.
  uint32_t location_base;
  // WebAssembly value type stored in this local.
  loom_wasm_value_type_t value_type;
  // Function-local WebAssembly index assigned by program planning.
  uint32_t local_index;
} loom_wasm_local_entry_t;

typedef struct loom_wasm_program_build_t {
  // Module being planned.
  loom_module_t* module;
  // Mutable function plan storage.
  loom_wasm_function_plan_t* functions;
  // Number of initialized function plans.
  iree_host_size_t function_count;
  // Mutable interned signature storage.
  loom_wasm_function_type_t* types;
  // Number of initialized signatures.
  iree_host_size_t type_count;
  // Mutable symbol-to-function-index table.
  uint32_t* function_indices_by_symbol;
  // Number of exported functions.
  iree_host_size_t export_count;
} loom_wasm_program_build_t;

static iree_string_view_t loom_wasm_program_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static iree_string_view_t loom_wasm_program_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol == NULL) {
    return IREE_SV("<unnamed>");
  }
  const iree_string_view_t name =
      loom_wasm_program_string_or_empty(module, symbol->name_id);
  return iree_string_view_is_empty(name) ? IREE_SV("<unnamed>") : name;
}

static iree_string_view_t loom_wasm_program_function_export_name(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_op_t* function_op) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  if (!loom_func_like_is_exported(function) &&
      (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC) == 0) {
    return iree_string_view_empty();
  }
  const iree_string_view_t export_name = loom_wasm_program_string_or_empty(
      module, loom_func_like_export_symbol(function));
  return iree_string_view_is_empty(export_name)
             ? loom_wasm_program_symbol_name(module, symbol)
             : export_name;
}

static iree_status_t loom_wasm_program_read_value_type(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, loom_value_id_t value_id,
    loom_wasm_value_type_t* out_value_type) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_low_type_is_register(type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u is not declared as a register value",
        (unsigned)value_id);
  }
  if (loom_low_register_type_descriptor_set_stable_id(type) !=
      descriptor_set->stable_id) {
    const iree_string_view_t descriptor_set_key =
        loom_low_descriptor_set_string(descriptor_set,
                                       descriptor_set->key_string_ref);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u does not use descriptor set '%.*s'",
        (unsigned)value_id, (int)descriptor_set_key.size,
        descriptor_set_key.data);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  if (unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm signature value %u declares %u register units",
        (unsigned)value_id, unit_count);
  }
  return loom_wasm_value_type_from_descriptor_register_class(
      loom_low_register_type_class_id(type), out_value_type);
}

static iree_status_t loom_wasm_program_build_value_type_list(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_id_t* value_ids, uint32_t value_count,
    iree_arena_allocator_t* arena, const loom_wasm_value_type_t** out_types) {
  *out_types = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_wasm_value_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*types), (void**)&types));
  for (uint32_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_program_read_value_type(
        module, descriptor_set, value_ids[i], &types[i]));
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_build_function_type(
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_wasm_function_type_t* out_type) {
  *out_type = (loom_wasm_function_type_t){0};
  const loom_func_like_t function =
      loom_func_like_const_cast(allocation->module, allocation->function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm program planning requires a func-like op");
  }

  uint16_t parameter_count = 0;
  const loom_value_id_t* parameters =
      loom_func_like_arg_ids(function, &parameter_count);
  const loom_value_slice_t results =
      loom_low_func_def_results(allocation->function_op);

  iree_status_t status = loom_wasm_program_build_value_type_list(
      allocation->module, allocation->target.descriptor_set, parameters,
      parameter_count, arena, &out_type->parameters);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_program_build_value_type_list(
        allocation->module, allocation->target.descriptor_set, results.values,
        results.count, arena, &out_type->results);
  }
  if (iree_status_is_ok(status)) {
    out_type->parameter_count = parameter_count;
    out_type->result_count = results.count;
  }
  return status;
}

static bool loom_wasm_program_function_types_equal(
    const loom_wasm_function_type_t* lhs,
    const loom_wasm_function_type_t* rhs) {
  if (lhs->parameter_count != rhs->parameter_count ||
      lhs->result_count != rhs->result_count) {
    return false;
  }
  if (lhs->parameter_count != 0 &&
      memcmp(lhs->parameters, rhs->parameters,
             lhs->parameter_count * sizeof(*lhs->parameters)) != 0) {
    return false;
  }
  return lhs->result_count == 0 ||
         memcmp(lhs->results, rhs->results,
                lhs->result_count * sizeof(*lhs->results)) == 0;
}

static iree_status_t loom_wasm_program_intern_function_type(
    loom_wasm_program_build_t* build, const loom_wasm_function_type_t* type,
    uint32_t* out_type_index) {
  for (iree_host_size_t i = 0; i < build->type_count; ++i) {
    if (loom_wasm_program_function_types_equal(&build->types[i], type)) {
      *out_type_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  if (build->type_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm type index exceeds u32");
  }
  *out_type_index = (uint32_t)build->type_count;
  build->types[build->type_count++] = *type;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_build_function_allocation(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_allocation, bool* out_accepted) {
  *out_allocation = (loom_low_allocation_table_t){0};
  *out_accepted = false;
  bool synthesis_admitted = false;
  IREE_RETURN_IF_ERROR(loom_low_diagnostic_admit_allocation_synthesis(
      module, function_op, diagnostic_emitter, &synthesis_admitted));
  if (!synthesis_admitted) {
    return iree_ok_status();
  }
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, function_op,
      /*function_target_facts=*/NULL, descriptor_registry, diagnostic_emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, arena, &model);
  if (iree_status_is_ok(status)) {
    const loom_low_allocation_options_t allocation_options = {
        .emitter = diagnostic_emitter,
    };
    status = loom_low_allocate_function(&model, &allocation_options, arena,
                                        out_allocation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_diagnostics_emit(out_allocation, /*flags=*/0,
                                                  diagnostic_emitter);
  }
  loom_low_function_model_deinitialize(&model);
  if (iree_status_is_ok(status) && out_allocation->error_count != 0) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status) &&
      out_allocation->target.descriptor_set !=
          loom_wasm_core_simd128_descriptor_set()) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm program planning requires descriptor set 'wasm.core.simd128'");
  }
  if (iree_status_is_ok(status) && (out_allocation->spill_count != 0 ||
                                    out_allocation->spill_plan_count != 0)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm program planning requires an unspilled allocation");
  }
  if (iree_status_is_ok(status) &&
      (out_allocation->edge_copy_count != 0 ||
       out_allocation->edge_copy_group_count != 0)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm program planning requires structured allocation without CFG edge "
        "copies");
  }
  if (iree_status_is_ok(status) &&
      out_allocation->packet_move_group_count != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm program planning requires allocation without packet-local moves");
  }
  if (iree_status_is_ok(status)) {
    *out_accepted = true;
  }
  return status;
}

static loom_wasm_local_entry_t* loom_wasm_program_find_local_entry(
    loom_wasm_local_entry_t* entries, iree_host_size_t entry_count,
    uint16_t descriptor_reg_class_id, uint32_t location_base) {
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    loom_wasm_local_entry_t* entry = &entries[i];
    if (entry->descriptor_reg_class_id == descriptor_reg_class_id &&
        entry->location_base == location_base) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_wasm_program_assignment_value_type(
    const loom_low_allocation_assignment_t* assignment,
    loom_wasm_value_type_t* out_value_type) {
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u is not allocated to a target-local id",
        (unsigned)assignment->value_id);
  }
  if (assignment->location_count != 1 || assignment->unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u uses a multi-unit target-id assignment",
        (unsigned)assignment->value_id);
  }
  return loom_wasm_value_type_from_descriptor_register_class(
      assignment->descriptor_reg_class_id, out_value_type);
}

static iree_status_t loom_wasm_program_set_value_local(
    const loom_low_allocation_table_t* allocation,
    uint32_t* local_indices_by_value_ordinal, loom_value_id_t value_id,
    uint32_t local_index) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(allocation->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= allocation->liveness.value_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm allocation value %u is outside the function value domain",
        (unsigned)value_id);
  }
  uint32_t* value_local = &local_indices_by_value_ordinal[value_ordinal];
  if (*value_local != LOOM_WASM_PROGRAM_INDEX_NONE &&
      *value_local != local_index) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u changed local mapping",
                            (unsigned)value_id);
  }
  *value_local = local_index;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_append_local_type(
    loom_wasm_value_type_t* local_types, iree_host_size_t local_capacity,
    iree_host_size_t* local_count, loom_wasm_value_type_t value_type,
    uint32_t* out_local_index) {
  if (*local_count >= local_capacity || *local_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local index exceeds planned capacity");
  }
  *out_local_index = (uint32_t)*local_count;
  local_types[(*local_count)++] = value_type;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_project_function_allocation(
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_wasm_function_plan_t* function) {
  const loom_func_like_t function_like =
      loom_func_like_const_cast(allocation->module, allocation->function_op);
  if (!loom_func_like_isa(function_like)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm planning requires a func-like function");
  }
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function_like, &parameter_count);
  iree_host_size_t local_capacity = 0;
  if (!iree_host_size_checked_add(parameter_count, allocation->assignment_count,
                                  &local_capacity) ||
      local_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local count exceeds u32");
  }

  loom_wasm_value_type_t* local_types = NULL;
  loom_wasm_local_entry_t* local_entries = NULL;
  if (local_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, local_capacity, sizeof(*local_types), (void**)&local_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, local_capacity, sizeof(*local_entries), (void**)&local_entries));
  }

  if (allocation->liveness.value_count >= LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm function value domain exceeds ordinal range");
  }
  uint32_t* local_indices_by_value_ordinal = NULL;
  if (allocation->liveness.value_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, allocation->liveness.value_count,
                                  sizeof(*local_indices_by_value_ordinal),
                                  (void**)&local_indices_by_value_ordinal));
    for (iree_host_size_t i = 0; i < allocation->liveness.value_count; ++i) {
      local_indices_by_value_ordinal[i] = LOOM_WASM_PROGRAM_INDEX_NONE;
    }
  }

  loom_low_allocation_value_scratch_t value_scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch);
  iree_host_size_t local_count = 0;
  iree_host_size_t local_entry_count = 0;
  for (uint32_t i = 0; i < parameter_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t value_id = parameter_ids[i];
    loom_wasm_value_type_t value_type = 0;
    status = loom_wasm_program_read_value_type(
        allocation->module, allocation->target.descriptor_set, value_id,
        &value_type);
    uint32_t local_index = 0;
    if (iree_status_is_ok(status)) {
      status = loom_wasm_program_append_local_type(
          local_types, local_capacity, &local_count, value_type, &local_index);
    }
    const loom_low_allocation_assignment_t* assignment = NULL;
    if (iree_status_is_ok(status)) {
      assignment = loom_low_allocation_try_map_active_value_assignment(
          allocation, value_id, NULL);
    }
    if (iree_status_is_ok(status) && assignment != NULL) {
      loom_wasm_local_entry_t* existing = loom_wasm_program_find_local_entry(
          local_entries, local_entry_count, assignment->descriptor_reg_class_id,
          assignment->location_base);
      if (existing != NULL) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Wasm parameters %u and %u share allocator target id %u",
            existing->local_index, local_index, assignment->location_base);
      } else {
        local_entries[local_entry_count++] = (loom_wasm_local_entry_t){
            .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
            .location_base = assignment->location_base,
            .value_type = value_type,
            .local_index = local_index,
        };
        status = loom_wasm_program_set_value_local(
            allocation, local_indices_by_value_ordinal, value_id, local_index);
      }
    }
  }

  for (iree_host_size_t i = 0;
       i < allocation->assignment_count && iree_status_is_ok(status); ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    loom_wasm_value_type_t value_type = 0;
    status = loom_wasm_program_assignment_value_type(assignment, &value_type);
    loom_wasm_local_entry_t* existing = NULL;
    if (iree_status_is_ok(status)) {
      existing = loom_wasm_program_find_local_entry(
          local_entries, local_entry_count, assignment->descriptor_reg_class_id,
          assignment->location_base);
    }
    uint32_t local_index = LOOM_WASM_PROGRAM_INDEX_NONE;
    if (iree_status_is_ok(status) && existing != NULL) {
      if (existing->value_type != value_type) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "Wasm target id %u changes value type",
                                  assignment->location_base);
      } else {
        local_index = existing->local_index;
      }
    } else if (iree_status_is_ok(status)) {
      status = loom_wasm_program_append_local_type(
          local_types, local_capacity, &local_count, value_type, &local_index);
      if (iree_status_is_ok(status)) {
        local_entries[local_entry_count++] = (loom_wasm_local_entry_t){
            .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
            .location_base = assignment->location_base,
            .value_type = value_type,
            .local_index = local_index,
        };
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_wasm_program_set_value_local(
          allocation, local_indices_by_value_ordinal, assignment->value_id,
          local_index);
    }
  }
  loom_low_allocation_release_value_scratch(&value_scratch);

  if (iree_status_is_ok(status)) {
    function->value_ids = allocation->liveness.value_ids;
    function->local_indices_by_value_ordinal = local_indices_by_value_ordinal;
    function->value_count =
        (loom_value_ordinal_t)allocation->liveness.value_count;
    function->local_types = local_types;
    function->parameter_count = parameter_count;
    function->local_count = (uint32_t)local_count;
  }
  return status;
}

static iree_status_t loom_wasm_program_build_function(
    loom_wasm_program_build_t* build, loom_op_t* function_op,
    loom_wasm_function_plan_t* function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_accepted) {
  *out_accepted = false;
  loom_low_allocation_table_t allocation = {0};
  bool allocation_accepted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_program_build_function_allocation(
      build->module, function_op, descriptor_registry, diagnostic_emitter,
      arena, &allocation, &allocation_accepted));
  if (!allocation_accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_wasm_program_project_function_allocation(
      &allocation, arena, function));
  IREE_RETURN_IF_ERROR(loom_wasm_program_build_function_type(&allocation, arena,
                                                             &function->type));
  IREE_RETURN_IF_ERROR(loom_wasm_program_intern_function_type(
      build, &function->type, &function->type_index));
  *out_accepted = true;
  return iree_ok_status();
}

iree_status_t loom_wasm_program_plan_build(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_accepted, loom_wasm_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_accepted);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_accepted = false;
  *out_plan = (loom_wasm_program_plan_t){0};
  if (module->symbols.count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm program planning requires low functions");
  }

  loom_wasm_program_build_t build = {
      .module = module,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*build.functions),
                                                 (void**)&build.functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*build.types),
                                                 (void**)&build.types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*build.function_indices_by_symbol),
      (void**)&build.function_indices_by_symbol));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    build.function_indices_by_symbol[i] = LOOM_WASM_PROGRAM_INDEX_NONE;
  }

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_op_t* defining_op = symbol->defining_op;
    if (defining_op == NULL) {
      continue;
    }
    if (loom_low_func_decl_isa(defining_op)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm module imports for low.func.decl are not implemented");
    }
    if (loom_low_kernel_def_isa(defining_op)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm program planning for low.kernel.def is not implemented");
    }
    if (!loom_low_func_def_isa(defining_op)) {
      continue;
    }
    if (build.function_count >= LOOM_WASM_PROGRAM_INDEX_NONE ||
        symbol_index >= LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm function index exceeds supported range");
    }

    loom_wasm_function_plan_t* function =
        &build.functions[build.function_count];
    *function = (loom_wasm_function_plan_t){
        .body = loom_low_func_def_body(defining_op),
        .name = loom_wasm_program_symbol_name(module, symbol),
        .export_name =
            loom_wasm_program_function_export_name(module, symbol, defining_op),
        .function_index = (uint32_t)build.function_count,
    };
    build.function_indices_by_symbol[symbol_index] = function->function_index;
    if (!iree_string_view_is_empty(function->export_name)) {
      ++build.export_count;
    }
    bool function_accepted = false;
    IREE_RETURN_IF_ERROR(loom_wasm_program_build_function(
        &build, defining_op, function, descriptor_registry, diagnostic_emitter,
        arena, &function_accepted));
    if (!function_accepted) {
      return iree_ok_status();
    }
    ++build.function_count;
  }

  if (build.function_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm program planning requires at least one "
                            "low.func.def");
  }
  *out_plan = (loom_wasm_program_plan_t){
      .module = module,
      .functions = build.functions,
      .function_count = build.function_count,
      .types = build.types,
      .type_count = build.type_count,
      .function_indices_by_symbol = build.function_indices_by_symbol,
      .symbol_count = module->symbols.count,
      .export_count = build.export_count,
  };
  *out_accepted = true;
  return iree_ok_status();
}

iree_status_t loom_wasm_compile_module_binary(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, bool* out_emitted,
    loom_wasm_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_emitted = false;
  *out_module = (loom_wasm_module_binary_t){0};

  loom_wasm_program_plan_t plan = {0};
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_program_plan_build(module, descriptor_registry,
                                                    diagnostic_emitter, arena,
                                                    &accepted, &plan));
  if (!accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_program_emit_binary(&plan, allocator, out_module));
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_module_emit(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm module artifacts do not produce artifact manifests");
  }

  loom_wasm_module_binary_t module = {0};
  bool module_emitted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_compile_module_binary(
      request->module, request->low_descriptor_registry,
      request->diagnostic_emitter, request->scratch_arena, request->allocator,
      &module_emitted, &module));
  if (!module_emitted) {
    return iree_ok_status();
  }
  iree_byte_span_t contents =
      iree_make_byte_span(module.data, module.data_length);
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &contents, request->allocator, &out_artifact->contents);
  if (iree_status_is_ok(status)) {
    out_artifact->target_artifact_format =
        LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY;
    *out_emitted = true;
  }
  iree_allocator_free(request->allocator, contents.data);
  return status;
}

const loom_target_emitter_t loom_wasm_module_emitter = {
    .name = IREE_SVL("wasm-binary"),
    .public_artifact_format = IREE_SVL("wasm-binary"),
    .default_identifier = IREE_SVL("module.wasm"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .default_pipeline_options =
        {
            .control_flow_lowering =
                LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        },
    .emit = loom_wasm_module_emit,
};

const loom_target_provider_t loom_wasm_module_provider = {
    .emitter_list =
        {
            .values =
                (const loom_target_emitter_t* const[]){
                    &loom_wasm_module_emitter},
            .count = 1,
        },
    .canonical_module_emitter = &loom_wasm_module_emitter,
    .canonical_module_fact_type = &loom_wasm_target_fact_type,
};

static const loom_target_provider_t* const kWasmCompilerProviders[] = {
    &loom_wasm_target_provider,
    &loom_wasm_module_provider,
};

const loom_target_provider_set_t loom_wasm_compiler_provider_set = {
    .providers = kWasmCompilerProviders,
    .provider_count = IREE_ARRAYSIZE(kWasmCompilerProviders),
};
