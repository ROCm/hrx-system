// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_compile.h"

#include <string.h>

static void loom_amdgpu_program_require_callable_registers(
    const loom_amdgpu_program_callable_t* callable,
    loom_amdgpu_program_resource_builder_t* resources) {
  const uint32_t sgpr_count =
      LOOM_AMDGPU_CALL_ABI_SGPR_BASE +
      iree_max(callable->argument_requirements.sgpr_count,
               callable->result_requirements.sgpr_count);
  const uint32_t vgpr_count =
      LOOM_AMDGPU_CALL_ABI_VGPR_BASE +
      iree_max(callable->argument_requirements.vgpr_count,
               callable->result_requirements.vgpr_count);
  loom_amdgpu_program_resource_builder_require_registers(
      sgpr_count, vgpr_count, /*agpr_count=*/0, resources);
}

// Retained calls can overwrite loader-owned coordinates, so entry ABI values
// become one-shot transport sources instead of lifetime constraints. The
// ordinary no-call path keeps the fixed bindings and its existing machine code.
static iree_status_t loom_amdgpu_program_prepare_retained_entry_options(
    const loom_module_t* module,
    const loom_amdgpu_function_compile_options_t* source_options,
    iree_arena_allocator_t* arena,
    loom_amdgpu_function_compile_options_t* out_options) {
  *out_options = *source_options;
  if (source_options->allocation_fixed_value_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t entry_value_count = 0;
  if (!iree_host_size_checked_add(source_options->allocation_entry_value_count,
                                  source_options->allocation_fixed_value_count,
                                  &entry_value_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU retained entry value count overflowed");
  }
  loom_low_allocation_entry_value_t* entry_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_value_count, sizeof(*entry_values), (void**)&entry_values));
  if (source_options->allocation_entry_value_count != 0) {
    memcpy(
        entry_values, source_options->allocation_entry_values,
        source_options->allocation_entry_value_count * sizeof(*entry_values));
  }
  loom_low_allocation_entry_value_t* converted_values =
      entry_values + source_options->allocation_entry_value_count;
  for (iree_host_size_t i = 0; i < source_options->allocation_fixed_value_count;
       ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &source_options->allocation_fixed_values[i];
    const loom_type_t type =
        loom_module_value_type(module, fixed_value->value_id);
    IREE_ASSERT(loom_low_type_is_register(type));
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(type),
                   fixed_value->location_count);
    converted_values[i] = (loom_low_allocation_entry_value_t){
        .value_id = fixed_value->value_id,
        .location_kind = fixed_value->location_kind,
        .location_base = fixed_value->location_base,
    };
  }
  out_options->allocation_fixed_values = NULL;
  out_options->allocation_fixed_value_count = 0;
  out_options->allocation_entry_values = entry_values;
  out_options->allocation_entry_value_count = entry_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_program_compile_retained(
    loom_module_t* module, const loom_amdgpu_program_compile_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    bool* out_compiled, loom_amdgpu_program_compilation_t* out_compilation) {
  const loom_amdgpu_program_artifact_root_t* root = options->artifact_root;
  IREE_ASSERT_GT(root->helper_count, 0);
  IREE_ASSERT_NE(root->call_table, NULL);
  const loom_low_descriptor_set_t* descriptor_set =
      options->entry.resolved_target->descriptor_set;
  IREE_ASSERT_EQ(root->target, options->entry.resolved_target);
  IREE_ASSERT_EQ(root->call_table->descriptor_set, descriptor_set);

  loom_amdgpu_program_resource_builder_t resources = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_resource_builder_initialize(
      descriptor_set->reg_class_count, arena, &resources));
  for (iree_host_size_t i = 0; i < root->helper_count; ++i) {
    const loom_symbol_id_t helper_symbol_id = root->helper_symbols[i];
    IREE_ASSERT_LT(helper_symbol_id, root->call_table->symbol_count);
    const loom_amdgpu_program_callable_t* callable =
        root->call_table->callables_by_symbol[helper_symbol_id];
    IREE_ASSERT_NE(callable, NULL);
    loom_amdgpu_program_resource_builder_require_function(
        &callable->compilation.frame, &callable->compilation.plan, &resources);
    loom_amdgpu_program_resource_builder_require_private_frame(
        callable->compilation.plan.storage_layout.segment_sizes
            .private_segment_fixed_size,
        &resources);
    loom_amdgpu_program_require_callable_registers(callable, &resources);
  }

  const loom_low_allocation_reserved_range_t call_control_reservation = {
      .register_class = IREE_SV("amdgpu.sgpr"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE,
      .location_count = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
  };
  iree_host_size_t entry_reservation_count = 0;
  if (!iree_host_size_checked_add(
          options->entry.allocation_reserved_range_count, 1,
          &entry_reservation_count)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU entry allocation reservation count overflowed");
  }
  loom_low_allocation_reserved_range_t* entry_reservations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_reservation_count, sizeof(*entry_reservations),
      (void**)&entry_reservations));
  if (options->entry.allocation_reserved_range_count != 0) {
    memcpy(entry_reservations, options->entry.allocation_reserved_ranges,
           options->entry.allocation_reserved_range_count *
               sizeof(*entry_reservations));
  }
  entry_reservations[entry_reservation_count - 1] = call_control_reservation;

  loom_amdgpu_function_compile_options_t entry_options = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_prepare_retained_entry_options(
      module, &options->entry, scratch_arena, &entry_options));
  entry_options.call_contracts =
      loom_amdgpu_program_call_table_provider(root->call_table);
  entry_options.call_states =
      loom_amdgpu_program_call_table_state_provider(root->call_table);
  entry_options.allocation_reserved_ranges = entry_reservations;
  entry_options.allocation_reserved_range_count = entry_reservation_count;
  bool entry_compiled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_function_compile(
      module, root->function_op, &entry_options, arena, &entry_compiled,
      &out_compilation->entry));
  if (!entry_compiled) {
    return iree_ok_status();
  }

  loom_amdgpu_storage_layout_t entry_storage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_resource_builder_append_function(
      &out_compilation->entry.frame, &out_compilation->entry.plan, arena,
      &resources, &entry_storage));
  out_compilation->entry.plan.storage_layout = entry_storage;
  loom_amdgpu_program_resource_builder_require_registers(
      LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE +
          LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
      /*vgpr_count=*/0, /*agpr_count=*/0, &resources);
  loom_amdgpu_program_resource_builder_finish(&resources,
                                              &out_compilation->resources);
  out_compilation->artifact_root = root;
  out_compilation->has_retained_calls = true;
  *out_compiled = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_amdgpu_program_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_program_compilation_t* out_compilation) {
  *out_compiled = false;
  *out_compilation = (loom_amdgpu_program_compilation_t){0};
  IREE_ASSERT_NE(options->artifact_root, NULL);
  IREE_ASSERT_EQ(options->artifact_root->function_op, entry_function_op);
  if (options->artifact_root->helper_count == 0) {
    return loom_amdgpu_function_compile(module, entry_function_op,
                                        &options->entry, arena, out_compiled,
                                        &out_compilation->entry);
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_amdgpu_program_compile_retained(
      module, options, arena, &scratch_arena, out_compiled, out_compilation);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
